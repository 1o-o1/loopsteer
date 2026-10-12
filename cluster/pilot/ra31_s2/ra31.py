"""RA-31 Part A: the dose curve of the branch writes at the last prompt token, teacher-forced, Ouro base.
Plan: plan_ra31.md in this folder. Phase 1 holds the per-item engine (arms, sets, identities, write norms) and the
curve cells; the CLI (run, gates, summarize) is Phase 2.

Per item (r = T - 1): the donor patch of r's loop1.loop_end state (clean B) and the RUN 1c random same-norm patch,
each with the post-norm write of one branch at r in loops 2 to 4 interpolated, w' = w_clean + alpha (w - w_clean):
- set "all": the whole write, prod patch edit (lerp) with a = 1 - alpha and target clean (alpha 0 = the RUN 1c hold
  spec, alpha 1 = a 0, x returned untouched);
- sets "top" (top 5 percent of neurons / top 2 heads by the per-item patched-minus-clean input difference summed
  over loops 2 to 4) and "rand" (seeded, equal count): lib/ra31_hooks.Units31, w' = w + (alpha - 1)(c_S - c_S_clean)
  with c_S the exact contribution of the set to the post-norm write (full-vector rms held);
- set "dall": the same decomposed edit on every unit, donor only, alphas 0, 1, 2: the reduction check against "all".
Readout: NLD at the final logits at r.
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lib")
TREE_DEFAULT = os.path.join(HERE, "tree")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import torch  # noqa: E402

import k17_freeze as KF  # noqa: E402
import run1c as R1  # noqa: E402
import ra31_hooks as H31  # noqa: E402

K = KF.K                                   # 4 loops
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
RND_ALPHAS = (0.0, 2.0)                    # the random patch: the two ends only
BRANCHES = ("mlp", "attn")
COMP = {"mlp": "mlp_out", "attn": "attn_out"}
SETS = ("all", "top", "rand")              # "all" = whole write (prod lerp); top / rand = decomposed subset edit
DALL_ALPHAS = (0.0, 2.0)                   # "dall": decomposed edit on every unit, donor only (reduction check;
#                                            the alpha-1 identity of this edit path is carried by top and rand)
SIZE_SETS = ("top1", "top20", "rand1", "rand20")   # RA addition 2026-10-12: MLP, donor, alphas 0 and 2 only
SIZE_ALPHAS = (0.0, 2.0)
FRAC = {"top1": 0.01, "top": 0.05, "top20": 0.20}  # share of the L * I neurons per ranked set
SIZE_FRAC = {"top1": 0.01, "rand1": 0.01, "top20": 0.20, "rand20": 0.20}
GRAD_EQUIV = 3                             # the attribution pass (forward + backward) in forward-equivalents
LOOPS = (2, 3, 4)                          # the write is changed in loops 2 to 4 (calls from F_ans)
TOP_FRAC = 0.05                            # neurons: 5 percent of L * I
TOP_HEADS = 2                              # heads: 2 of L * H
GAP_E = KF.GAP_E                           # 2.0
ID_TOL = 0.005                             # alpha 0 against the RUN 1c hold, mean NLD gap per dataset and branch
FS_TOL = 0.02                              # first-site projection ratio |ratio - alpha| of the whole-write arm
r6, margin = KF.r6, KF.margin


def arm_name(patch, branch, set_, alpha):
    return "%s_%s_%s_a%s" % (patch, branch, set_, ("%.2f" % alpha).rstrip("0").rstrip("."))


def arm_list(drop=()):
    """[(patch, branch, set, alpha)]: donor x 8 alphas and random x 2 alphas on all / top / rand, plus the donor
    "dall" check at alphas 0, 1, 2. 48 + 12 + 6 = 66 arms. drop: "branch:set" groups removed by an RA decision after
    the pilot (plan section 8, prune rule); "all" cannot be dropped."""
    drop = set(drop)
    if any(d.endswith(":all") for d in drop):
        raise ValueError("the whole-write arm cannot be dropped")
    return [a for a in _arm_list() if "%s:%s" % (a[1], a[2]) not in drop]


def _arm_list():
    out = []
    for patch, alphas in (("donor", ALPHAS), ("rnd", RND_ALPHAS)):
        for br in BRANCHES:
            for s in SETS:
                for a in alphas:
                    out.append((patch, br, s, a))
    for br in BRANCHES:
        for a in DALL_ALPHAS:
            out.append(("donor", br, "dall", a))
    for s_ in SIZE_SETS:
        for a in SIZE_ALPHAS:
            out.append(("donor", "mlp", s_, a))
    return out


def seed_of(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:15], 16)


# ------------------------------------------------------------------ sets (CPU)
def n_top_neurons(L, I, frac=TOP_FRAC):
    return int(round(frac * L * I))


def neuron_scores(actP, actC, loops=LOOPS):
    """actP, actC [K, L, I] float32 -> [L, I] = sum over loops of |a_patched - a_clean| at r."""
    return sum((actP[t - 1] - actC[t - 1]).abs() for t in loops)


def head_scores(zP, zC, H, hd, loops=LOOPS):
    """zP, zC [K, L, H * hd] -> [L, H] = sum over loops of the L2 norm of the head-output difference at r."""
    Lz = zP.shape[1]
    return sum((zP[t - 1] - zC[t - 1]).view(Lz, H, hd).norm(dim=-1) for t in loops)


def rank_order(score, n):
    """Flat indices of the n largest scores in rank order; ties broken by the lower flat index (stable sort)."""
    flat = score.reshape(-1).double()
    order = torch.sort(-flat, stable=True).indices
    return [int(i) for i in order[:n]]


def top_units(score, n):
    """The n largest scores as a sorted index list (a prefix of rank_order, so top1 within top within top20)."""
    return sorted(rank_order(score, n))


def rand_units(text, N, n):
    g = torch.Generator(device="cpu").manual_seed(seed_of(text))
    return sorted(int(i) for i in torch.randperm(N, generator=g)[:n])


def item_sets(ds, key, L, I, H, hd, uC, uP):
    """Neuron and head sets of one item and one patch. uC, uP: Units.take of clean A and the patched run.
    Random sets depend on the item only (the same draw for the donor and the random patch)."""
    nn_ = n_top_neurons(L, I)
    n1, n20 = n_top_neurons(L, I, FRAC["top1"]), n_top_neurons(L, I, FRAC["top20"])
    order20 = rank_order(neuron_scores(uP["act"], uC["act"]), n20)
    return {"mlp": {"all": None, "top": sorted(order20[:nn_]),
                    "rand": rand_units("ra31:neurons:%s:%s" % (ds, key), L * I, nn_),
                    "top1": sorted(order20[:n1]), "top20": sorted(order20),
                    "rand1": rand_units("ra31:neurons1:%s:%s" % (ds, key), L * I, n1),
                    "rand20": rand_units("ra31:neurons20:%s:%s" % (ds, key), L * I, n20),
                    "order20": order20},
            "attn": {"all": None,
                     "top": top_units(head_scores(uP["z"], uC["z"], H, hd), TOP_HEADS),
                     "rand": rand_units("ra31:heads:%s:%s" % (ds, key), L * H, TOP_HEADS)}}


def attribution(Rn, grads, uC, uP, order20, r):
    """First-order estimate per neuron of the margin change from holding it at its clean value, on the patched run:
    est(t, l, i) = grad_w(t, l) . (c_i_clean - c_i), grad_w = d margin / d (post-norm MLP write at r) on the patched
    run, c_i = g * W_down[:, i] a_i / rms(u) (each run's own a and rms). Computed as
    (W_down^T (g * grad_w)) * (a_C / rms_C - a_P / rms_P). -> float32 [3 loops, n20] in the order of order20."""
    L, I = Rn.L, Rn.I
    est = torch.zeros(len(LOOPS), L * I)
    for j, t in enumerate(LOOPS):
        for l in range(L):
            layer = Rn.U.layers[l]
            W, nrm = layer.mlp.down_proj.weight, layer.post_attention_layernorm_2
            eps = H31.UH.norm_eps(nrm)
            gw = grads[(t, l)][0, r].float()
            v = W.float().t() @ (nrm.weight.float() * gw)
            aC, aP = uC["act"][t - 1, l].to(v.device), uP["act"][t - 1, l].to(v.device)
            rC = H31.rms(uC["u_mlp"][t - 1, l].to(v.device), eps)
            rP = H31.rms(uP["u_mlp"][t - 1, l].to(v.device), eps)
            est[j, l * I:(l + 1) * I] = (v * (aC / rC - aP / rP)).cpu()
    return est[:, torch.tensor(order20, dtype=torch.long)]


def attr_bytes(n_items, L, I):
    """Raw size of the attribution store: per item the top-20-percent index (int32) and the estimate per loop
    (float16, 3 loops), plus 5 float32 scalars."""
    n20 = n_top_neurons(L, I, FRAC["top20"])
    return n_items * (n20 * (4 + 2 * len(LOOPS)) + 5 * 4)


def full_sets(rec, patch, L, I, H):
    """The four unit sets of a stored record: the top sets as stored, the random sets rebuilt from the seed (only the
    top sets are stored; review N7)."""
    ds, key = rec["dataset"], rec["key"]
    return {"mlp": {"top": rec["sets"][patch]["mlp_top"],
                    "rand": rand_units("ra31:neurons:%s:%s" % (ds, key), L * I, n_top_neurons(L, I)),
                    "rand1": rand_units("ra31:neurons1:%s:%s" % (ds, key), L * I, n_top_neurons(L, I, FRAC["top1"])),
                    "rand20": rand_units("ra31:neurons20:%s:%s" % (ds, key), L * I,
                                         n_top_neurons(L, I, FRAC["top20"]))},
            "attn": {"top": rec["sets"][patch]["attn_top"],
                     "rand": rand_units("ra31:heads:%s:%s" % (ds, key), L * H, TOP_HEADS)}}


def mask_report(sets, L, I, H):
    """Printed for three items per dataset (T2): layer histogram of the neuron sets, the head sets as (layer, head),
    the overlap of top and random."""
    out = {}
    for s in ("top", "rand"):
        hist = [0] * L
        for u in sets["mlp"][s]:
            hist[u // I] += 1
        out["mlp_" + s] = {"n": len(sets["mlp"][s]), "per_layer": hist}
        out["attn_" + s] = [[u // H, u % H] for u in sets["attn"][s]]
    out["mlp_top_rand_overlap"] = len(set(sets["mlp"]["top"]) & set(sets["mlp"]["rand"]))
    return out


def edit_tables(branch, units, uC, Rn, dev):
    """Units31 spec tables for one branch and unit set: {(t, l): (idx, c_S_clean float32 [d])}, t in loops 2 to 4.
    units None = every neuron (head) of every layer (the "dall" reduction check). c_S_clean from clean A's own input
    and full pre-norm vector (Units.take: act / u_mlp, z / o_attn)."""
    L, I, H, hd = Rn.L, Rn.I, Rn.H, Rn.hd
    n = I if branch == "mlp" else H
    per_l = {}
    if units is None:
        for l in range(L):
            per_l[l] = list(range(n))
    else:
        for u in units:
            per_l.setdefault(u // n, []).append(u % n)
    xs, us = (uC["act"], uC["u_mlp"]) if branch == "mlp" else (uC["z"], uC["o_attn"])
    out = {}
    for l, js in per_l.items():
        layer = Rn.U.layers[l]
        if branch == "mlp":
            idx, W, norm = torch.tensor(sorted(js), dtype=torch.long), layer.mlp.down_proj.weight,                 layer.post_attention_layernorm_2
        else:
            idx, W, norm = H31.head_cols(js, hd), layer.self_attn.o_proj.weight, layer.input_layernorm_2
        idx = idx.to(W.device)
        eps = H31.UH.norm_eps(norm)
        for t in LOOPS:
            c = H31.contrib(xs[t - 1, l].to(W.device), W, idx, us[t - 1, l].to(W.device), norm.weight, eps)
            out[(t, l)] = (idx.to(dev), c.to(dev))
    return out


# ------------------------------------------------------------------ forward engine (GPU path, CPU-tested on the fake)
class Runner(object):
    """KF.Ctx plus the RA-31 unit hooks for one forward (the RA-28 Runner pattern)."""

    def __init__(self, X):
        self.X = X
        self.U = H31.Units31(X.b)
        self.dtype = X.W.dtype
        self.L, self.I, self.H, self.hd, self.d = self.U.L, self.U.I, self.U.H, self.U.hd, self.U.d

    def run(self, ids, row, ivs=(), recs=(), spec=None):
        X = self.X
        reg = X.Reg(X.b, K)
        for rc in recs:
            reg.add_recorder(rc)
        for iv in ivs:
            reg.add_intervention(iv)
        reg.install()
        took = {}
        try:
            if spec is not None:
                self.U.install(spec)
            with reg.batch([row], [0]):
                z = X.b.forward(ids, K, [row.key])
            out = reg.take_records()
            if spec is not None and spec.record:
                took = self.U.take(K)
        finally:
            self.U.remove()
            reg.remove()
        X.n_fwd += 1
        return z[0], out, took

    def run_grad(self, ids, row, ivs, spec, r, tA, tB):
        """One forward under enable_grad with zero leaves on the post-norm MLP writes in spec.grad_w; -> (logits
        detached, {(t, l): d margin / d leaf}), margin = z(tB) - z(tA) at r of the final logits."""
        X = self.X
        reg = X.Reg(X.b, K)
        for iv in ivs:
            reg.add_intervention(iv)
        reg.install()
        try:
            self.U.install(spec)
            with reg.batch([row], [0]), torch.enable_grad():    # binding.forward casts the logits outside its
                z = X.b.forward(ids, K, [row.key], grad=True)     # own grad context; keep the cast on the graph
            leaves = dict(self.U.leaves_w)
        finally:
            self.U.remove()
            reg.remove()
        X.n_fwd += 1
        with torch.enable_grad():
            m = z[0, r, tB] - z[0, r, tA]
            keys = sorted(leaves)
            gs = torch.autograd.grad(m, [leaves[k] for k in keys])
        return z[0].detach(), {k: g.detach() for k, g in zip(keys, gs)}


def write_norms(X, key, rr, rC, first_site=None, rE=None):
    """Per branch and loop t in 2..4: sqrt(sum over layers ||w_arm - w_clean||^2) at r (post-norm writes from the
    'frz' recorder). first_site (whole-write arms): the projection ratio ((w_arm - w_c) . D) / ||D||^2 with
    D = w_E0 - w_c at the first call of that branch in loop 2 (equals alpha up to bf16 rounding, since the input of
    that call is the patched run's; review B2)."""
    out = {}
    for br, comp in COMP.items():
        per = []
        for t in LOOPS:
            s2 = 0.0
            for s in X.frz_sites:
                if s.loop == t and s.component == comp:
                    k_ = (key, "frz", s.key())
                    s2 += float((rr[k_].float() - rC[k_].float()).pow(2).sum())
            per.append(r6(s2 ** 0.5))
        out[br] = per
    fs = None
    if first_site is not None:
        s = min((x for x in X.frz_sites if x.loop == 2 and x.component == COMP[first_site]), key=lambda x: x.layer)
        k_ = (key, "frz", s.key())
        dlt = (rE[k_].float() - rC[k_].float()).reshape(-1)
        den = float(dlt.pow(2).sum())
        fs = r6(float((rr[k_].float() - rC[k_].float()).reshape(-1) @ dlt) / den) if den > 0 else None
    return out, fs


@torch.no_grad()
def process_item(Rn, it, ds, arms=None):
    """All rows of plan section 5 for one item. -> (record, identity flags, info flags)."""
    X = Rn.X
    I_, P = X.I, X.Positions
    A, B, src, tA, tB, key = it["A"], it["B"], sorted(it["pos"]), it["tA"], it["tB"], it["key"]
    if len(A) != len(B):                                  # build_items drops these (counted); never reached in a run
        raise ValueError("prompt A and B differ in length: %s" % key)
    T = len(A)
    r = T - 1
    t0, f0 = time.time(), X.n_fwd
    ids_A = torch.tensor([A], device=X.dev)
    ids_B = torch.tensor([B], device=X.dev)
    spans = {"source": tuple(src), "answer": (r,)}
    pos_r = {"kind": "abs", "abs": [r]}

    def row(payload=None):
        return X.RowInfo(key=key, prompt_len=T, spans=dict(spans), payload=payload or {})

    def le_rec():
        return X.Recorder(id="le", sites=tuple(X.s_le), positions=P("all"), what="state", when="post")

    def frz_rec():
        return X.Recorder(id="frz", sites=tuple(X.frz_sites), positions=P("abs", abs=(r,)), what="state",
                          when="post")

    def H_of(recs):
        return {j: recs[(key, "le", X.s_le[j - 1].key())] for j in range(1, K + 1)}

    def spec(record=False, **kw):
        return H31.Spec31(r, record=record, T=T, **kw)

    # base rows
    z_nh = X.nohook(ids_A)[0]
    zA, rA, uC = Rn.run(ids_A, row(), recs=[le_rec(), frz_rec()], spec=spec(record=True))
    HA = H_of(rA)
    zB, rB, _ = Rn.run(ids_B, row(), recs=[le_rec()])
    HB = H_of(rB)
    cA, cB = KF.vals_at(zA, r, tA, tB), KF.vals_at(zB, r, tA, tB)
    D = margin(cB) - margin(cA)
    delta = (HB[1][r] - HA[1][r]).float()
    n_t = float(delta.norm())
    u = R1.rnd_direction(ds, key, delta.shape[0]).to(delta.device)          # the RUN 1c seeds
    rnd_payload = (HA[1][r].float() + n_t * u).to(Rn.dtype)[None]           # built in the model dtype (RA-28)

    def patch_iv(iid, site, pos):
        return I_.from_spec({"id": iid, "op": "patch", "site": site.key(), "a": 1.0, "target": {"kind": "donor"},
                             "positions": pos}, X.g)
    piv = {"donor": patch_iv("ans", X.s_ans, pos_r), "rnd": patch_iv("rnd", X.s_ans, pos_r)}
    ppay = {"donor": {"ans": {X.s_ans.key(): HB[1][r:r + 1]}}, "rnd": {"rnd": {X.s_ans.key(): rnd_payload}}}
    pcf = patch_iv("pcf", X.s_le[K - 1], pos_r)
    sz = patch_iv("sz", X.s_le[K - 1], {"kind": "span", "span": "source"})
    z_pc, _, _ = Rn.run(ids_A, row({"pcf": {X.s_le[K - 1].key(): HB[K][r:r + 1]}}), ivs=[pcf])
    z_sz, _, _ = Rn.run(ids_A, row({"sz": {X.s_le[K - 1].key(): HB[K][src]}}), ivs=[sz])
    zero = I_.from_spec({"id": "ans", "op": "patch", "site": X.s_ans.key(), "a": 0.0, "target": {"kind": "donor"},
                         "positions": pos_r}, X.g)
    z_zero, _, _ = Rn.run(ids_A, row({"ans": {X.s_ans.key(): HB[1][r:r + 1]}}), ivs=[zero])

    def frz_iv(br, alpha):
        comp = COMP[br]
        iv = I_.from_spec(dict(I_.freeze_spec(comp, pos_r, X.F_ans, X.g), id="frz_" + comp, a=1.0 - alpha), X.g)
        return iv, {"frz_" + comp: {s.key(): rA[(key, "frz", s.key())] for s in iv.sites}}
    z_hold = {}
    for br in BRANCHES:                                   # the RUN 1c holds without the patch (freeze_no_edit)
        iv, pay = frz_iv(br, 0.0)
        z_hold[br], _, _ = Rn.run(ids_A, row(pay), ivs=[iv])

    def go(patch, ivs=(), extra=None, recs=(), sp=None):
        pay = dict(ppay[patch])
        pay.update(extra or {})
        return Rn.run(ids_A, row(pay), ivs=[piv[patch]] + list(ivs), recs=recs, spec=sp)
    zE, rE, uP = go("donor", recs=[le_rec(), frz_rec()], sp=spec(record=True))
    HE = H_of(rE)
    zR, rR, uR = go("rnd", recs=[le_rec(), frz_rec()], sp=spec(record=True))
    HR = H_of(rR)
    L, Iu, Hh, hd = Rn.L, Rn.I, Rn.H, Rn.hd
    sets = {"donor": item_sets(ds, key, L, Iu, Hh, hd, uC, uP), "rnd": item_sets(ds, key, L, Iu, Hh, hd, uC, uR)}
    rec = {"key": key, "dataset": ds, "T": T, "pos": src, "tA": tA, "tB": tB, "D": r6(D), "rows": {},
           "clean": cA, "cleanB": cB,
           "sets": {p: {"mlp_top": sets[p]["mlp"]["top"], "attn_top": sets[p]["attn"]["top"]} for p in sets}}
    for nm, z in (("E0", zE), ("rnd", zR), ("pc", z_pc), ("sz", z_sz)):
        v = KF.vals_at(z, r, tA, tB)
        rec["rows"][nm] = {"final": v, "dm": r6(margin(v) - margin(cA))}
    applied = float((HR[1][r] - HA[1][r]).float().norm())
    rec["rows"]["rnd"].update({"target_norm": r6(n_t), "applied_norm": r6(applied),
                               "norm_ratio": r6(applied / n_t) if n_t > 0 else None})
    rec["rows"]["E0"]["wn"], _ = write_norms(X, key, rE, rA)
    rec["rows"]["rnd"]["wn"], _ = write_norms(X, key, rR, rA)

    a1_equal, z_arm, tabs = [], {}, {}
    rE_by = {"donor": rE, "rnd": rR}
    for (patch, br, s, alpha) in (arms if arms is not None else arm_list()):
        if s == "all":
            iv, pay = frz_iv(br, alpha)
            z, rr, _ = go(patch, ivs=[iv], extra=pay, recs=[frz_rec()])
        else:
            if (patch, br, s) not in tabs:
                units = None if s == "dall" else sets[patch][br][s]
                tabs[(patch, br, s)] = edit_tables(br, units, uC, Rn, X.dev)
            tab = tabs[(patch, br, s)]
            sp = spec(alpha=alpha, **({"edit_mlp": tab} if br == "mlp" else {"edit_attn": tab}))
            z, rr, _ = go(patch, recs=[frz_rec()], sp=sp)
        v = KF.vals_at(z, r, tA, tB)
        wn, fs = write_norms(X, key, rr, rA, first_site=br if s == "all" else None, rE=rE_by[patch])
        nm = arm_name(patch, br, s, alpha)
        rec["rows"][nm] = {"final": v, "dm": r6(margin(v) - margin(cA)), "wn": wn}
        if fs is not None:
            rec["rows"][nm]["first_site_ratio"] = fs
        if patch == "donor" and alpha == 1.0:
            a1_equal.append(bool(torch.equal(z, zE)))
        if alpha == 0.0:
            z_arm[(patch, br, s)] = z
    ident = {
        "hook_identity": bool(torch.equal(zA, z_nh) and torch.equal(z_zero, zA)),
        "freeze_no_edit_identity": all(bool(torch.equal(z_hold[br], zA)) for br in BRANCHES),
        "pc_exact": bool(torch.equal(z_pc[r], zB[r])),
        "sz_exact": bool(torch.equal(z_sz[r], zA[r])),
        "donor_loop1_state": bool(torch.equal(HE[1][r], HB[1][r])),
        "rnd_loop1_state": bool(torch.equal(HR[1][r], rnd_payload[0])),
        "alpha1_equals_patched": bool(a1_equal) and all(a1_equal),
        "set_counts": all(len(sets[p]["mlp"][s]) == n_top_neurons(L, Iu) and
                          len(sets[p]["attn"][s]) == TOP_HEADS for p in sets for s in ("top", "rand")) and
        all(len(sets[p]["mlp"][s]) == n_top_neurons(L, Iu, SIZE_FRAC[s]) for p in sets for s in SIZE_SETS),
    }
    ran = set(nm for nm in rec["rows"])
    top20_a0 = arm_name("donor", "mlp", "top20", 0.0)
    if top20_a0 in ran:                       # the attribution store (plan section 8a): one forward + backward
        sp = spec(grad_w=[(t, l) for t in LOOPS for l in range(L)])
        zg, grads = Rn.run_grad(ids_A, row(dict(ppay["donor"])), [piv["donor"]], sp, r, tA, tB)
        order20 = sets["donor"]["mlp"]["order20"]
        est = attribution(Rn, grads, uC, uP, order20, r)
        mE, mT = margin(rec["rows"]["E0"]["final"]), margin(rec["rows"][top20_a0]["final"])
        pred, meas = float(est.sum()), mT - mE
        rec["attr"] = {"pred": r6(pred), "measured": r6(meas), "n20": len(order20),
                       "grad_pass_logits_equal_E0": bool(torch.equal(zg, zE))}
        rec["_attr"] = {"idx": order20, "est": est,
                        "scalars": [float(D), float(margin(cA)), float(mE), float(mT), pred]}
    fs_ok = []
    for nm, rw in rec["rows"].items():
        if "first_site_ratio" in rw and rw["first_site_ratio"] is not None:
            a = float(nm.rsplit("_a", 1)[1])
            fs_ok.append(abs(rw["first_site_ratio"] - a) <= FS_TOL)
    info = {"first_site_scale": bool(fs_ok) and all(fs_ok),
            "sets_act": all(not torch.equal(z_arm[k_], zE) for k_ in z_arm if k_[0] == "donor"),
            "rnd_reaches_final": not bool(torch.equal(zR[r], zA[r]))}
    rec["identity"], rec["info"] = ident, info
    rec["forwards"] = X.n_fwd - f0
    rec["seconds"] = round(time.time() - t0, 3)
    for k_ in ("cluster", "stratum", "half", "sa", "sb", "src"):
        if k_ in it:
            rec[k_] = it[k_]
    return rec, ident, info


# ------------------------------------------------------------------ readouts and curves (CPU)
def nld(rec, row):
    D = rec["D"]
    if D is None or D < GAP_E or row not in rec["rows"]:
        return None
    return rec["rows"][row]["dm"] / D


def run1c_gap(recs, ref, row, ref_row):
    """Per-item NLD gap between an RA-31 row and the stored RUN 1c row (same item key). -> (mean |gap|, max |gap|,
    mean signed gap of the item means, n)."""
    g = []
    for rec in recs:
        rf = ref.get(rec["key"])
        x = nld(rec, row)
        if rf is None or x is None or rf.get("D") is None or rf["D"] < GAP_E:
            continue
        g.append(x - rf[ref_row] / rf["D"])
    if not g:
        return None
    return {"mean_abs": round(sum(abs(v) for v in g) / len(g), 6), "max_abs": round(max(abs(v) for v in g), 6),
            "mean_gap": round(sum(g) / len(g), 6), "n": len(g), "n_nonzero": sum(1 for v in g if v != 0.0)}


def curve(recs, patch, br, s, alphas=ALPHAS, dyadic=False):
    """Rows of one curve: alpha -> KF.cell (mean NLD, 90 percent cluster bootstrap, 10,000 resamples, seed 0)."""
    KO = KF.stats_mod()
    return {a: KF.cell(KO, recs, lambda rc, nm=arm_name(patch, br, s, a): nld(rc, nm), dyadic=dyadic)
            for a in alphas}


# ================================================================== Phase 2: CLI, gates, run, summary
import argparse  # noqa: E402
import signal  # noqa: E402

import results_common as RC  # noqa: E402

PLAN_DEFAULT = os.path.join(HERE, "plan_ra31.md")
DATA_DEFAULT = os.path.join(HERE, "data")
MODELS = {"1.4b": "ouro_1_4b_base", "2.6b": "ouro_2_6b_base",
          "1.4b-think": "ouro_1_4b_think", "2.6b-think": "ouro_2_6b_think"}     # model axis (plan section 14)
REVISION_PINS = {"1.4b-think": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
                 "2.6b-think": "f1edd81e7ac41355db670500ceaf204e0f73af68"}
DATASETS = ("D1", "D2")
DS_NAME = {"D1": "facts", "D2": "addition"}
GATE_N = 3                                 # identity-gate items per dataset, every session, before any scored row
PILOT_N = 25                               # pilot: the first 25 items of each dataset
BUDGET_H = {"1.4b": 1.5, "2.6b": 5.5,      # G5: projected scored card-hours of a full run
            "1.4b-think": 3.0, "2.6b-think": 5.5}   # Thinking: on an A100 at 0.6 x the single card's speed
G3_TOL = 0.005                             # run 1c reproduction: mean |gap| of the alpha-0 whole write
G4_BIAS, G4_NOISE = 0.01, 0.03             # reduction dall - all at alpha 0: |mean signed gap|, mean |gap|
FLAT = 0.1                                 # predictions: 90 percent interval inside [-0.1, 0.1] of E0
SHARE_MIN = 0.1                            # share printed where |mean(NLD_all - NLD_E0)| >= 0.1
NB = KF.NB
GATED = ("hook_identity", "freeze_no_edit_identity", "pc_exact", "sz_exact", "donor_loop1_state", "rnd_loop1_state",
         "alpha1_equals_patched", "set_counts")       # exit 3 on a gate item, exit 4 on a scored item
GATE_ONLY = ("first_site_scale", "sets_act")          # exit 3 on a gate item only
RUN1C_RND = {"D1": 0.5664, "D2": 0.626}               # run 1c random patch under the MLP hold (facts, addition)
RUN1C_N = {"D1": 267, "D2": 263}
# The pilot (all gates passed, out/pilot_1.4b) ran under the plan before the RA addition of 2026-10-12; its gate is
# accepted for the full run under this plan (the new arms pass G1 on the gate items before any scored row).
ACCEPTED_PILOT_PLANS = ("ae5405dbfa5c62a984342c43368becae5b4eadee2f6ccd917d374c5f3c9b7b92",   # 1.4B card pilot
                        "12c089ff89c63645df45d93aa8e8179b910b97723fdc560ddd641f14c77e58b2",   # RA addition
                        "bd16a27fb82cfaba6b5ae65ed0e15130e296a4b14f7f88eb14f903bc43475dac",   # Part B engine
                        "06653e2a67176a56be7b5dffbc37ba44a76ac5f80ca98693da3163602c0cc4f3",   # model axis
                        "3ee4139ade5f7fdbaf2c76c1c451a9652e4aa7d2f70ccaba0c1baae2e5957658",   # 2.6B budget 5.5
                        "2c34fd6d687cb03803fda6d5f265713e4c0ed1b60fa9498284bfbf902ce111b3",   # Part B grid
                        "e1b944e5f87248620abd8577a0b316eb911524572299794125148e02e621a698")   # wording scrub
EXIT_PLAN, EXIT_GATE, EXIT_IDENTITY, EXIT_STOPPED = 2, 3, 4, 5
GRID, CARD, SCRIPT = "ra31", "scaled_writes", "ra31.py"
STOP = {"flag": False}


def die(code, msg):
    print("[ra31] %s" % msg, flush=True)
    raise SystemExit(code)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="ra31")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--hash", action="store_true")
    ap.add_argument("--model", choices=sorted(MODELS), default="1.4b")
    ap.add_argument("--out")
    ap.add_argument("--plan", default=PLAN_DEFAULT)
    ap.add_argument("--plan_sha256")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--tree", default=TREE_DEFAULT)
    ap.add_argument("--pilot", type=int, default=0, help="pilot: the first N items of each dataset")
    ap.add_argument("--pilot_gate", help="full run: the pilot's pilot_gate.json (must hold pass true)")
    ap.add_argument("--drop", default="", help="comma list of branch:set groups dropped by the RA after the pilot")
    ap.add_argument("--mem_fraction", type=float, default=0.12)
    a = ap.parse_args(argv)
    if a.run and not (a.out and a.plan_sha256):
        ap.error("--run needs --out and --plan_sha256")
    if a.run and not a.pilot and not a.pilot_gate:
        ap.error("a full --run needs --pilot_gate")
    if a.summarize and not a.out:
        ap.error("--summarize needs --out")
    a.drop_list = sorted(x for x in a.drop.split(",") if x)
    try:
        arm_list(a.drop_list)
    except ValueError as e:
        ap.error(str(e))
    return a


# ------------------------------------------------------------------ hashes (CPU)
def code_hashes():
    fs = {"ra31.py": os.path.abspath(__file__)}
    for f in sorted(os.listdir(LIB)):
        if f.endswith(".py"):
            fs["lib/" + f] = os.path.join(LIB, f)
    return {k: KF.sha256_file(p) for k, p in fs.items()}


def check_data(data_dir):
    src = json.load(open(os.path.join(data_dir, "sources.json")))
    bad = [f for f, h in src["outputs"].items() if KF.sha256_file(os.path.join(data_dir, f)) != h]
    if bad:
        die(EXIT_PLAN, "data files differ from data/sources.json: %s" % bad)
    return src["outputs"]


def write_hash_line(out, cfg, data, pins):
    d = {"run": "RA-31 Part A", "plan_sha256": cfg["plan_sha256"],
         "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest(),
         "code_sha256": code_hashes(), "data_sha256": data, "pins": pins,
         "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = os.path.join(out, "hash_line.json")
    if os.path.exists(p):
        old = json.load(open(p))
        for k in ("plan_sha256", "config_sha256", "code_sha256", "data_sha256", "pins"):
            if old.get(k) != d[k]:
                die(EXIT_PLAN, "resume with a changed %s; refusing" % k)
        return old
    json.dump(d, open(p, "w"), indent=1, sort_keys=True)
    return d


# ------------------------------------------------------------------ model and items (GPU host; replaced in tests)
def load_model(args):
    """-> (X, cfg fields, pins). Pins of run 1c (transformers, eager, bf16, prod md5); the 1.4B revision is pinned,
    the 2.6B revision is read from the prod pins file and recorded. Exit 2 on a difference."""
    name = MODELS[args.model]
    tree = os.path.abspath(args.tree)
    ad, b = KF.load_model(tree, args.mem_fraction, name)
    X = KF.Ctx(b, tree)
    tok = ad.tok
    X.tok = tok
    X.enc = lambda t, ans=False: tok(t, add_special_tokens=False)["input_ids"]
    import transformers
    rev = getattr(ad.model.config, "_commit_hash", None)
    prod = KF.prod_hashes(tree)
    bad = []
    if args.model == "1.4b" and rev != R1.OURO_REVISION:
        bad.append("revision %s" % rev)
    if args.model in REVISION_PINS and rev != REVISION_PINS[args.model]:
        bad.append("revision %s" % rev)
    if rev is None:
        bad.append("revision unknown")
    if transformers.__version__ != R1.OURO_TRANSFORMERS:
        bad.append("transformers %s" % transformers.__version__)
    if b.attn_implementation() != "eager":
        bad.append("attention %s" % b.attn_implementation())
    if str(ad.model.dtype) != "torch.bfloat16":
        bad.append("dtype %s" % ad.model.dtype)
    if prod != R1.OURO_PROD_MD5:
        bad.append("prod md5 differs from run 1c")
    if bad:
        die(EXIT_PLAN, "pins differ: %s" % json.dumps(bad))
    cfg = {"model_name": name, "repo": ad.repo, "revision": rev, "dtype": str(ad.model.dtype),
           "attn_implementation": "eager", "k": K, "F_ans": X.F_ans,
           "n_layers": int(ad.model.config.num_hidden_layers), "transformers": transformers.__version__}
    return X, cfg, {"revision": rev, "transformers": transformers.__version__, "prod_md5": prod}


def build_items(X, args, ds, n_limit):
    """-> (items, info). 1.4B: the run 1c stored identities, rebuilt exactly (exit 2 otherwise). 2.6B: the same
    selection code on the 2.6B tokenizer and clean filter (facts: every d1_candidates pair; addition: d2_candidates,
    cap 4 per A-side operand value), in candidate order, stopping at n_limit kept items for a pilot (a prefix of the
    full selection). Unequal prompt lengths are dropped and counted (review N3)."""
    info = {"drops": {"length": 0, "clean_A": 0, "clean_B": 0}}
    if args.model == "1.4b":
        items, inf = R1.build_items(X, "ouro", args.data, ds)
        info.update({k: v for k, v in inf.items() if k != "bad"})
        kept = []
        for it in items:
            if len(it["A"]) != len(it["B"]):
                info["drops"]["length"] += 1
                continue
            kept.append(it)
        items = kept[:n_limit] if n_limit else kept
        info.update({"n_run": len(items), "rule": "run 1c identities, rebuilt exactly"})
        return items, info
    import k15_data as KD
    st = {"A_fail": 0, "B_fail": 0}
    know = KF.make_know(X, st)
    out, scanned = [], 0
    if ds == "D1":
        pool, pinfo = KF.build_d1_pool(X.enc, os.path.join(args.data, "items_d1.json"))
        info["candidates"] = pinfo["candidates"]
        for it in pool:
            if n_limit and len(out) >= n_limit:
                break
            scanned += 1
            if len(it["A"]) != len(it["B"]):
                info["drops"]["length"] += 1
            elif know(it):
                out.append(it)
    else:
        cands = KD.d2_candidates(X.enc)
        info["candidates"] = len(cands)
        n_c = {}
        for p in cands:
            if n_limit and len(out) >= n_limit:
                break
            a1, a2 = [int(v) for v in p["a"].split(" + ")]
            c = "x=%d" % (a1 if p["rel"] == "op1" else a2)
            if n_c.get(c, 0) >= KF.D2_CAP:
                continue
            scanned += 1
            if len(p["A"]) != len(p["B"]):
                info["drops"]["length"] += 1
            elif know(p):
                n_c[c] = n_c.get(c, 0) + 1
                out.append({"key": "%s|%s|%s" % (p["rel"], p["a"], p["b"]), "A": p["A"], "B": p["B"],
                            "pos": sorted(p["pos"]), "tA": p["tA"], "tB": p["tB"], "cluster": c,
                            "stratum": p["rel"], "half": p["half"]})
    info["drops"]["clean_A"], info["drops"]["clean_B"] = st["A_fail"], st["B_fail"]
    info.update({"scanned": scanned, "n_run": len(out),
                 "rule": "2.6B: the run 1c selection code on this model's tokenizer and clean filter"})
    return out, info


def load_ref(data_dir):
    return json.load(open(os.path.join(data_dir, "run1c_ref.json")))["datasets"]


# ------------------------------------------------------------------ gates (CPU)
def gate_check(recs):
    fails = []
    for rec in recs:
        for k_ in GATED:
            if not rec["identity"].get(k_):
                fails.append("%s: %s" % (rec["key"], k_))
        for k_ in GATE_ONLY:
            if not rec["info"].get(k_):
                fails.append("%s: %s" % (rec["key"], k_))
    if not recs:
        fails.append("no gate items")
    return {"pass": not fails, "n_items": len(recs), "failures": fails, "exit3_keys": list(GATED + GATE_ONLY)}


def g3_repro(recs_by_ds, ref):
    """Run 1c reproduction (1.4B): per dataset and branch, mean |NLD(all, 0) - NLD(run 1c hold)| <= 0.005; the count
    of items with a nonzero gap (review N4), E0 and the random arm printed beside it (not gated)."""
    res = {"tol": G3_TOL, "datasets": {}, "failures": []}
    for ds, recs in recs_by_ds.items():
        d = {}
        for br in BRANCHES:
            g = run1c_gap(recs, ref.get(ds, {}), arm_name("donor", br, "all", 0.0), "ans_frz_" + br)
            d["hold_" + br] = g
            if g is None or g["mean_abs"] > G3_TOL:
                res["failures"].append("%s %s: %s" % (ds, br, g))
            d["rnd_hold_" + br] = run1c_gap(recs, ref.get(ds, {}), arm_name("rnd", br, "all", 0.0), "rnd_hold_" + br)
        d["E0"] = run1c_gap(recs, ref.get(ds, {}), "E0", "E0")
        res["datasets"][ds] = d
    res["pass"] = not res["failures"] and bool(recs_by_ds)
    return res


def _gaps(recs, a, b):
    out = []
    for rc in recs:
        x, y = nld(rc, a), nld(rc, b)
        if x is not None and y is not None:
            out.append(x - y)
    return out


def g4_reduction(recs_by_ds, drop=()):
    """dall - all at alpha 0: |mean signed gap| <= 0.01 (bias of a code error) and mean |gap| <= 0.03 (bf16 noise
    ceiling), per dataset and branch; max |gap| and the alpha-2 values printed (the noise floor for reading top and
    rand above alpha 1; review B1). A branch whose subset arms are all dropped is not gated."""
    res = {"bias_tol": G4_BIAS, "noise_tol": G4_NOISE, "datasets": {}, "failures": []}
    for ds, recs in recs_by_ds.items():
        d = {}
        for br in BRANCHES:
            if "%s:top" % br in drop and "%s:rand" % br in drop:
                d[br] = "subset arms dropped"
                continue
            row = {}
            for a in DALL_ALPHAS:
                if a == 1.0:
                    continue
                g = _gaps(recs, arm_name("donor", br, "dall", a), arm_name("donor", br, "all", a))
                row["a%g" % a] = None if not g else {
                    "mean_signed": round(sum(g) / len(g), 6), "mean_abs": round(sum(abs(v) for v in g) / len(g), 6),
                    "max_abs": round(max(abs(v) for v in g), 6), "n": len(g)}
            z = row.get("a0")
            if z is None or abs(z["mean_signed"]) > G4_BIAS or z["mean_abs"] > G4_NOISE:
                res["failures"].append("%s %s: %s" % (ds, br, z))
            d[br] = row
        res["datasets"][ds] = d
    res["pass"] = not res["failures"] and bool(recs_by_ds)
    return res


def check_pilot_gate(path, plan_sha, model):
    try:
        g = json.load(open(path))
    except (OSError, ValueError, TypeError):
        return ["pilot gate file missing or unreadable"], None
    fails = []
    if not g.get("pass"):
        fails.append("pilot gate did not pass")
    if g.get("plan_sha256") != plan_sha and g.get("plan_sha256") not in ACCEPTED_PILOT_PLANS:
        fails.append("pilot gate has another plan sha256")
    if g.get("model") != model:
        fails.append("pilot gate is for model %s" % g.get("model"))
    return fails, g


def forwards_equiv(arms):
    """Forward-equivalents per item: 10 base rows, one per arm, and the attribution pass (GRAD_EQUIV) when the
    top20 alpha-0 arm runs."""
    return 10 + len(arms) + (GRAD_EQUIV if ("donor", "mlp", "top20", 0.0) in arms else 0)


def save_attr(out, ds, i, key, a):
    """One item's attribution shard: out/attr_<ds>/<i>.npz (packed per dataset by --summarize)."""
    import numpy as np
    d = os.path.join(out, "attr_%s" % ds)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "%05d.npz" % i), key=np.array(key), idx=np.array(a["idx"], dtype=np.int32),
             est=a["est"].numpy().astype(np.float16), scalars=np.array(a["scalars"], dtype=np.float32))


def pack_attr(out, ds):
    """attr_<ds>.npz: keys [N], idx int32 [N, n20] (rank order; top1 and top5 are its prefixes), est float16
    [N, 3, n20] (loops 2 to 4), scalars float32 [N, 5] = D, margin clean A, margin E0, margin top20 alpha 0, the
    summed estimate. -> (path, summary of the linear check) or (None, None)."""
    import numpy as np
    d = os.path.join(out, "attr_%s" % ds)
    if not os.path.isdir(d):
        return None, None
    fs = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    if not fs:
        return None, None
    zs = [np.load(os.path.join(d, f)) for f in fs]
    keys = np.array([str(z["key"]) for z in zs])
    idx = np.stack([z["idx"] for z in zs])
    est = np.stack([z["est"] for z in zs])
    sc = np.stack([z["scalars"] for z in zs])
    path = os.path.join(out, "attr_%s.npz" % ds)
    np.savez_compressed(path, keys=keys, idx=idx, est=est, scalars=sc)
    sct = torch.from_numpy(sc.astype(np.float64))          # statistics in torch (no numpy BLAS call)
    pred, meas = sct[:, 4], sct[:, 3] - sct[:, 2]
    r_ = None
    if len(fs) > 2 and float(pred.std()) > 0 and float(meas.std()) > 0:
        r_ = float(torch.corrcoef(torch.stack([pred, meas]))[0, 1])
    return path, {"n_items": len(fs), "n20": int(idx.shape[1]), "raw_bytes": int(idx.nbytes + est.nbytes + sc.nbytes),
                  "file_bytes": os.path.getsize(path), "mean_pred": round(float(pred.mean()), 4),
                  "mean_measured": round(float(meas.mean()), 4), "pearson_r": None if r_ is None else round(r_, 4),
                  "mean_abs_err_over_mean_abs_measured": round(float((pred - meas).abs().mean() /
                                                                     max(1e-12, float(meas.abs().mean()))), 4)}


def projected_hours(spi_by_ds, n_by_ds, fwd_pilot, fwd_full):
    """Scored card-hours of a full run from the pilot's seconds per item, scaled by forwards per item (an RA prune
    lowers it), the gate items counted twice."""
    s = sum(spi_by_ds[ds] * (n_by_ds[ds] + GATE_N) for ds in n_by_ds)
    return round(s * fwd_full / float(fwd_pilot) / 3600.0, 3)


def t2_report(X, items):
    """T2 print: tokens at r and at the source span of A and B, equal length."""
    rows = []
    tok = getattr(X, "tok", None)
    dec = (lambda ids: tok.decode(ids)) if tok is not None else (lambda ids: str(ids))
    for it in items:
        r = len(it["A"]) - 1
        rows.append({"key": it["key"], "T_A": len(it["A"]), "T_B": len(it["B"]), "r": r,
                     "A_at_r": dec([it["A"][r]]), "B_at_r": dec([it["B"][r]]),
                     "A_span": dec([it["A"][p] for p in it["pos"]]), "B_span": dec([it["B"][p] for p in it["pos"]]),
                     "equal_length": len(it["A"]) == len(it["B"])})
    return rows


# ------------------------------------------------------------------ run (GPU)
def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def write_hashes(out):
    hs = {}
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if os.path.isfile(p) and (f.startswith("items_") or f.startswith("selection_") or
                                  f in ("config.json", "gate.json", "hash_line.json", "pilot_gate.json",
                                        "pilot_gate_used.json", "end_gates.json", "t2.json")):
            hs[f] = KF.sha256_file(p)
    json.dump(hs, open(os.path.join(out, "hashes.json"), "w"), indent=1, sort_keys=True)


def run(args):
    t_start = time.time()
    out = args.out
    os.makedirs(out, exist_ok=True)
    got = KF.sha256_file(args.plan)
    if got != args.plan_sha256:
        die(EXIT_PLAN, "plan sha256 mismatch: file %s, given %s" % (got, args.plan_sha256))
    data = check_data(args.data)
    pg = None
    if not args.pilot:
        bad, pg = check_pilot_gate(args.pilot_gate, args.plan_sha256, args.model)
        if bad:
            die(EXIT_GATE, "GATE FAILED (pilot gate): %s" % json.dumps(bad))
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))
    X, mcfg, pins = load_model(args)
    Rn = Runner(X)
    arms = arm_list(args.drop_list)
    cfg = {"run": "RA-31 Part A", "model": args.model, "plan_sha256": args.plan_sha256,
           "code_sha256": KF.sha256_file(os.path.abspath(__file__)), "data_sha256": data, "pilot": args.pilot,
           "drop": args.drop_list, "n_arms": len(arms), "alphas": list(ALPHAS), "rnd_alphas": list(RND_ALPHAS),
           "dall_alphas": list(DALL_ALPHAS), "sets": list(SETS), "top_frac": TOP_FRAC, "top_heads": TOP_HEADS,
           "gate_n": GATE_N, "gap_e": GAP_E, "g3_tol": G3_TOL, "g4": [G4_BIAS, G4_NOISE], "fs_tol": FS_TOL,
           "budget_h": BUDGET_H[args.model], "gated": list(GATED), "gate_only": list(GATE_ONLY),
           "torch": torch.__version__,
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    cfg.update(mcfg)
    cfg_path = os.path.join(out, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        for k_ in ("plan_sha256", "code_sha256", "data_sha256", "pilot", "model", "drop", "revision"):
            if old.get(k_) != cfg.get(k_):
                die(EXIT_PLAN, "resume with a changed config field %s" % k_)
        cfg = old
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=1, sort_keys=True)
    hl = write_hash_line(out, cfg, data, pins)
    print("[ra31] hash line %s" % json.dumps({k: hl[k] for k in ("plan_sha256", "config_sha256")}), flush=True)
    prog_path = os.path.join(out, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"sessions": []}
    sess = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "load_seconds": round(time.time() - t_start, 1)}
    prog["sessions"].append(sess)

    def save_prog():
        sess["seconds"] = round(time.time() - t_start, 1)
        sess["forwards"] = X.n_fwd
        sess["peak_alloc_gb"] = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                                 if torch.cuda.is_available() else None)
        json.dump(prog, open(prog_path + ".tmp", "w"), indent=1)
        os.replace(prog_path + ".tmp", prog_path)

    sels = {}
    for ds in DATASETS:
        sel_path = os.path.join(out, "selection_%s.json" % ds)
        if os.path.exists(sel_path):           # resume: the stored selection is used; 1.4B also re-checks the rebuild
            sel = json.load(open(sel_path))
            if args.model == "1.4b":
                items, _ = build_items(X, args, ds, args.pilot)
                if [it["key"] for it in items] != [it["key"] for it in sel["items"]]:
                    die(EXIT_PLAN, "resume with a changed %s item list" % ds)
        else:
            ts, f0 = time.time(), X.n_fwd
            items, info = build_items(X, args, ds, args.pilot)
            info.update({"scan_forwards": X.n_fwd - f0, "scan_seconds": round(time.time() - ts, 1)})
            sel = {"dataset": ds, "info": info, "items": items}
            json.dump(sel, open(sel_path + ".tmp", "w"))
            os.replace(sel_path + ".tmp", sel_path)
        sels[ds] = sel["items"]
        print("[ra31] %s items: %s" % (ds, json.dumps(sel["info"])), flush=True)
    if pg is not None:
        fwd_full = forwards_equiv(arms)
        proj = projected_hours(pg["seconds_per_item"], {ds: len(sels[ds]) for ds in DATASETS},
                               pg["forwards_per_item"], fwd_full)
        json.dump(dict(pg, projected_full_hours=proj, forwards_per_item_full=fwd_full),
                  open(os.path.join(out, "pilot_gate_used.json"), "w"), indent=1)
        if proj > BUDGET_H[args.model]:
            die(EXIT_GATE, "GATE FAILED (NO FIT): projected %.2f card-hours > %.1f" % (proj, BUDGET_H[args.model]))
        print("[ra31] projected full card-hours %.2f (budget %.1f)" % (proj, BUDGET_H[args.model]), flush=True)

    t2 = {ds: t2_report(X, sels[ds][:3]) for ds in DATASETS}
    tg = time.time()
    grecs = []
    for ds in DATASETS:
        for it in sels[ds][:GATE_N]:
            rec, _, _ = process_item(Rn, it, ds, arms)
            rec.pop("_attr", None)
            grecs.append(rec)
    g = gate_check(grecs)
    g["seconds"] = round(time.time() - tg, 1)
    g["records"] = grecs
    json.dump(g, open(os.path.join(out, "gate.json"), "w"), indent=1)
    for ds in DATASETS:
        for row_, rc in zip(t2[ds], [x for x in grecs if x["dataset"] == ds]):
            row_["masks"] = {p: mask_report(full_sets(rc, p, Rn.L, Rn.I, Rn.H), Rn.L, Rn.I, Rn.H)
                             for p in ("donor", "rnd")}
    json.dump(t2, open(os.path.join(out, "t2.json"), "w"), indent=1)
    print("[ra31] T2 %s" % json.dumps({ds: [{k_: v for k_, v in r_.items() if k_ != "masks"} for r_ in t2[ds]]
                                       for ds in DATASETS}), flush=True)
    print("[ra31] identity gate %s (%d items)" % ("PASS" if g["pass"] else "FAIL", g["n_items"]), flush=True)
    save_prog()
    if not g["pass"]:
        die(EXIT_GATE, "GATE FAILED: %s" % json.dumps(g["failures"][:20]))

    for ds in DATASETS:
        rec_path = os.path.join(out, "items_%s.jsonl" % ds)
        done = set(x["key"] for x in load_jsonl(rec_path))
        ta, n_new = time.time(), 0
        with open(rec_path, "a") as fh:
            for it in sels[ds]:
                if STOP["flag"]:
                    break
                if it["key"] in done:
                    continue
                rec, ident, _ = process_item(Rn, it, ds, arms)
                a_ = rec.pop("_attr", None)
                if a_ is not None:
                    save_attr(out, ds, sels[ds].index(it), it["key"], a_)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_new += 1
                bad = [k_ for k_ in GATED if not ident.get(k_)]
                if bad:
                    save_prog()
                    die(EXIT_IDENTITY, "%s item %s identity failed: %s; stopping" % (ds, it["key"], bad))
                if n_new % 20 == 1:
                    print("[ra31] %s %d/%d (%.2fs/item) %.0fs" % (ds, len(done) + n_new, len(sels[ds]),
                                                                (time.time() - ta) / n_new, time.time() - t_start),
                          flush=True)
                    save_prog()
        save_prog()
        print("[ra31] %s done %d new items %.0fs" % (ds, n_new, time.time() - ta), flush=True)
        if STOP["flag"]:
            die(EXIT_STOPPED, "SIGTERM: stopped after the current item (resume with the same command)")
    sess["finished"] = True
    save_prog()
    recs = {ds: load_jsonl(os.path.join(out, "items_%s.jsonl" % ds)) for ds in DATASETS}
    g3 = (g3_repro(recs, load_ref(args.data)) if args.model == "1.4b" else
          {"pass": True, "applied": False, "reason": "no run 1c reference on 2.6B"})
    g4 = g4_reduction(recs, args.drop_list)
    ends = {"plan_sha256": args.plan_sha256, "model": args.model, "g3": g3, "g4": g4}
    if args.pilot:
        spi = {ds: round(sum(r["seconds"] for r in recs[ds]) / max(1, len(recs[ds])), 3) for ds in DATASETS}
        fwd = forwards_equiv(arms)
        ends.update({"pass": bool(g["pass"] and g3["pass"] and g4["pass"]), "pilot_n": args.pilot,
                     "n_items": {ds: len(recs[ds]) for ds in DATASETS}, "seconds_per_item": spi,
                     "forwards_per_item": fwd,
                     "seconds_per_arm_item": {ds: round(spi[ds] / fwd, 4) for ds in DATASETS},
                     "load_seconds": sess["load_seconds"],
                     "drops": {ds: json.load(open(os.path.join(out, "selection_%s.json" % ds)))["info"]["drops"]
                               for ds in DATASETS}})
        if args.model == "1.4b":
            ends["projected_full_hours"] = projected_hours(spi, RUN1C_N, fwd, fwd)
        json.dump(ends, open(os.path.join(out, "pilot_gate.json"), "w"), indent=1)
    else:
        ends["pass"] = bool(g3["pass"] and g4["pass"])
        json.dump(ends, open(os.path.join(out, "end_gates.json"), "w"), indent=1)
    write_hashes(out)
    print("[ra31] G3 run 1c reproduction %s: %s" % ("PASS" if g3["pass"] else "FAIL",
                                                   json.dumps(g3.get("datasets", g3.get("reason")))), flush=True)
    print("[ra31] G4 reduction %s: %s" % ("PASS" if g4["pass"] else "FAIL", json.dumps(g4["datasets"])), flush=True)
    print("[ra31] done %.0fs, %d forwards" % (time.time() - t_start, X.n_fwd), flush=True)
    if not (g3["pass"] and g4["pass"]):
        die(EXIT_GATE, "GATE FAILED (end gates): %s" % json.dumps(g3.get("failures", []) + g4["failures"]))


# ------------------------------------------------------------------ summary (CPU)
def diff_cell(KO, recs, a, b, dyadic=False):
    def f(rc):
        x, y = nld(rc, a), nld(rc, b)
        return None if x is None or y is None else x - y
    return KF.cell(KO, recs, f, dyadic=dyadic)


def share(recs, num_row, den_row, base="E0"):
    """sum(NLD_num - NLD_E0) / sum(NLD_den - NLD_E0) with the paired cluster bootstrap of run 1c (run1c.boot_ratio)."""
    num, den, cl = [], [], []
    for rc in recs:
        x, y, e = nld(rc, num_row), nld(rc, den_row), nld(rc, base)
        ok = None not in (x, y, e)
        num.append(x - e if ok else None)
        den.append(y - e if ok else None)
        cl.append(str(rc["cluster"]))
    return R1.boot_ratio(num, den, cl)


def inside(c, lo=-FLAT, hi=FLAT):
    return c is not None and lo <= c["ci90"][0] and c["ci90"][1] <= hi


def excludes0(c):
    return c is not None and (c["ci90"][0] > 0 or c["ci90"][1] < 0)


def ds_cells(recs, ds, present):
    KO = KF.stats_mod()
    dy = ds == "D1"
    c = {"n_items": len(recs), "n_clusters": len(set(str(r["cluster"]) for r in recs)),
         "n_D_ge_2": sum(1 for r in recs if r["D"] is not None and r["D"] >= GAP_E),
         "identity_counts": {k_: sum(1 for r in recs if r["identity"].get(k_)) for k_ in GATED},
         "info_counts": {k_: sum(1 for r in recs if r["info"].get(k_)) for k_ in GATE_ONLY},
         "base": {nm: KF.cell(KO, recs, lambda r, nm=nm: nld(r, nm), dyadic=dy) for nm in ("E0", "rnd", "pc", "sz")},
         "curves": {}, "write_norm": {}, "derived": {}}
    nr = [r["rows"]["rnd"]["norm_ratio"] for r in recs if r["rows"]["rnd"].get("norm_ratio") is not None]
    c["rnd_norm_ratio"] = {"mean": round(sum(nr) / len(nr), 6) if nr else None, "min": min(nr) if nr else None,
                           "max": max(nr) if nr else None}
    groups = sorted(set((p, br, s) for (p, br, s, a) in present))
    for (p, br, s) in groups:
        al = [a for (p2, b2, s2, a) in present if (p2, b2, s2) == (p, br, s)]
        k_ = "%s|%s|%s" % (p, br, s)
        c["curves"][k_] = {"%g" % a: KF.cell(KO, recs, lambda r, nm=arm_name(p, br, s, a): nld(r, nm), dyadic=dy)
                           for a in al}
        c["write_norm"][k_] = {"%g" % a: [round(sum(r["rows"][arm_name(p, br, s, a)]["wn"][br][j] for r in recs)
                                                / max(1, len(recs)), 4) for j in range(len(LOOPS))] for a in al}
    for br in BRANCHES:
        d = {}

        def has(s_, br=br):
            return ("donor", br, s_) in groups
        if has("all"):
            d["all_minus_E0"] = {"%g" % a: diff_cell(KO, recs, arm_name("donor", br, "all", a), "E0") for a in ALPHAS}
            d["chord_dev"] = {}
            for a in (0.25, 0.5, 0.75):
                def f(rc, a=a, br=br):
                    v = [nld(rc, arm_name("donor", br, "all", x)) for x in (a, 0.0, 1.0)]
                    return None if None in v else v[0] - ((1 - a) * v[1] + a * v[2])
                d["chord_dev"]["%g" % a] = KF.cell(KO, recs, f)
        for s_ in ("top", "rand"):
            if not has(s_):
                continue
            d["share_" + s_] = {}
            for a in ALPHAS:
                if a == 1.0:
                    continue
                den = d["all_minus_E0"]["%g" % a]
                d["share_" + s_]["%g" % a] = (share(recs, arm_name("donor", br, s_, a), arm_name("donor", br, "all", a))
                                             if den is not None and abs(den["mean"]) >= SHARE_MIN else None)
            if has("dall"):
                d["share_%s_over_dall" % s_] = {"%g" % a: share(recs, arm_name("donor", br, s_, a),
                                                               arm_name("donor", br, "dall", a)) for a in (0.0, 2.0)}
        if br == "mlp" and has("top20"):
            d["size_curve"] = {}
            for a in SIZE_ALPHAS:
                row_ = {}
                for s_ in ("top1", "top", "top20", "all"):
                    nm = arm_name("donor", br, s_, a)
                    row_[s_] = {"nld": KF.cell(KO, recs, lambda r, nm=nm: nld(r, nm)),
                                "share": None if s_ == "all" else share(recs, nm, arm_name("donor", br, "all", a))}
                for s_ in ("rand1", "rand", "rand20"):
                    if has(s_):
                        nm = arm_name("donor", br, s_, a)
                        row_[s_] = {"nld": KF.cell(KO, recs, lambda r, nm=nm: nld(r, nm)),
                                    "share": share(recs, nm, arm_name("donor", br, "all", a))}
                d["size_curve"]["%g" % a] = row_
        if has("top") and has("rand"):
            d["top_minus_rand"] = {"%g" % a: diff_cell(KO, recs, arm_name("donor", br, "top", a),
                                                       arm_name("donor", br, "rand", a)) for a in ALPHAS}
        c["derived"][br] = d
    c["predictions"] = predictions(c, ds)
    return c


def shape_reading(devs):
    """Chord deviation at alpha 0.25, 0.5, 0.75 (review N5): "linear" unless at least two 90 percent intervals
    exclude 0; then "threshold" only if all three exclude 0 with one sign, else "not linear"."""
    n_ex = sum(1 for x in devs if excludes0(x))
    if n_ex < 2:
        return "linear"
    signs = set(x["mean"] > 0 for x in devs if x is not None)
    if n_ex == 3 and len(signs) == 1:
        return "threshold"
    return "not linear"


def concentration_reading(sh, tr):
    """Review B3 (a), a correction of the brief's inverted clause: concentrated = share(top, 0) >= 0.5 and the 90
    percent interval of top - rand at alpha 0 excludes 0; spread = share(top, 0) < 0.5."""
    if sh is None:
        return None
    if sh["mean"] >= 0.5:
        return "concentrated" if excludes0(tr) else "neither (share >= 0.5, top - rand includes 0)"
    return "spread"


def predictions(c, ds):
    """Section 8 readings (review B3). The alpha-0 and alpha-1 endpoints are fixed by identity and are not a test."""
    P = {}
    m = c["derived"].get("mlp", {})
    if "chord_dev" in m:
        P["mlp_shape"] = shape_reading([m["chord_dev"][k] for k in ("0.25", "0.5", "0.75")])
        P["mlp_stays_above"] = all(inside(m["all_minus_E0"]["%g" % a]) for a in (1.25, 1.5, 2.0))
    if "share_top" in m:
        sh0 = m["share_top"].get("0")
        P["mlp_concentration"] = concentration_reading(sh0, m.get("top_minus_rand", {}).get("0"))
        P["mlp_share_top_alpha0_signed"] = None if sh0 is None else {"mean": sh0["mean"], "ci90": sh0["ci90"],
                                                                      "sign": "negative" if sh0["mean"] < 0 else
                                                                      "positive"}
    a_ = c["derived"].get("attn", {})
    if "all_minus_E0" in a_:
        P["attn_flat"] = all(inside(a_["all_minus_E0"]["%g" % a]) for a in ALPHAS)
        P["prune_proposal"] = ("drop attn:top and attn:rand (attention whole write flat; the RA decides)"
                               if P["attn_flat"] else "keep every arm")
    if "chord_dev" in a_:
        P["attn_shape"] = shape_reading([a_["chord_dev"][k] for k in ("0.25", "0.5", "0.75")])
    cur = c["curves"].get("rnd|mlp|all", {}).get("0")
    P["rnd_mlp_alpha0"] = {"ours": cur["mean"] if cur else None, "run1c": RUN1C_RND.get(ds)}
    return P


def results_rows(model, cells):
    rows = []
    for ds, c in cells.items():
        pc = c["base"]["pc"]["mean"] if c["base"].get("pc") else None
        for k_, cur in c["curves"].items():
            p, br, s = k_.split("|")
            rk = "rnd|%s|%s" % (br, s)
            for a, st in cur.items():
                if st is None:
                    continue
                rnd = c["curves"].get(rk, {}).get(a) if p == "donor" else None
                rows.append(RC.row(grid=GRID, card=CARD, script=SCRIPT, model=MODELS[model], dataset=DS_NAME[ds],
                                   arm="%s|%s|%s" % (p, br, s), site="r|loop2-4.%s" % COMP[br], loop=1,
                                   strength=float(a), metric="NLD", endpoint="final", item_set="own",
                                   n_items=st["n_defined"], n_clusters=st["n_clusters"], estimate=st["mean"],
                                   ci90_lo=st["ci90"][0], ci90_hi=st["ci90"][1], ci90_method="stored",
                                   ci95_lo=st["ci95"][0], ci95_hi=st["ci95"][1],
                                   random_control_estimate=rnd["mean"] if rnd else None, positive_control_value=pc,
                                   source_key="results.json:cells.%s.curves.%s.%s" % (ds, k_, a)))
    return rows


def summarize(args):
    out = args.out
    cfg = json.load(open(os.path.join(out, "config.json")))
    hs_p = os.path.join(out, "hashes.json")
    hs = json.load(open(hs_p)) if os.path.exists(hs_p) else {}
    hash_ok = bool(hs) and all(KF.sha256_file(os.path.join(out, f)) == h for f, h in hs.items())
    prog = json.load(open(os.path.join(out, "progress.json")))
    gate = json.load(open(os.path.join(out, "gate.json")))
    recs = {ds: load_jsonl(os.path.join(out, "items_%s.jsonl" % ds)) for ds in DATASETS}
    recs = {ds: v for ds, v in recs.items() if v}
    sel = {ds: json.load(open(os.path.join(out, "selection_%s.json" % ds))) for ds in recs}
    present = arm_list(cfg.get("drop", []))
    cells = {ds: ds_cells(v, ds, present) for ds, v in recs.items()}
    g3 = (g3_repro(recs, load_ref(args.data)) if cfg["model"] == "1.4b" else
          {"pass": True, "applied": False, "reason": "no run 1c reference on 2.6B"})
    g4 = g4_reduction(recs, cfg.get("drop", []))
    attr = {}
    for ds in recs:
        _, attr[ds] = pack_attr(out, ds)
    rows = results_rows(cfg["model"], cells)
    RC.write_results(rows, os.path.join(out, "results.csv"))
    gpu_s = sum(s_.get("seconds", 0) for s_ in prog["sessions"])
    ck = {"model": MODELS[cfg["model"]], "plan_sha256": cfg["plan_sha256"], "pilot": cfg["pilot"],
          "drop": cfg["drop"], "gate_pass": gate["pass"],
          "identities_all_items": {k_: all(c["identity_counts"][k_] == c["n_items"] for c in cells.values())
                                   for k_ in GATED},
          "g3_pass": g3["pass"], "g4_pass": g4["pass"],
          "n_items": {ds: c["n_items"] for ds, c in cells.items()},
          "drops": {ds: sel[ds]["info"].get("drops") for ds in sel},
          "complete": {ds: len(recs[ds]) == len(sel[ds]["items"]) for ds in sel},
          "predictions": {ds: c["predictions"] for ds, c in cells.items()},
          "results_csv_rows": len(rows), "gpu_minutes": round(gpu_s / 60.0, 2), "outputs_hashed": hash_ok}
    ck["attribution"] = attr
    res = {"plan_sha256": cfg["plan_sha256"], "config": cfg, "cells": cells, "g3": g3, "g4": g4, "attribution": attr,
           "selection": {ds: sel[ds]["info"] for ds in sel}, "checks": ck}
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    json.dump(ck, open(os.path.join(out, "checks.json"), "w"), indent=1)
    write_summary(out, res)
    print(json.dumps(ck, indent=1))
    return ck


def _f(st):
    return "n/a" if st is None else "%.3f [%.3f, %.3f]" % (st["mean"], st["ci90"][0], st["ci90"][1])


def write_summary(out, res):
    cfg = res["config"]
    L = ["# RA-31 Part A summary (%s, %s)" % (MODELS[cfg["model"]],
                                              "pilot %d" % cfg["pilot"] if cfg["pilot"] else "full"), "",
         "Plan sha256 %s. NLD = dm / D per item (D >= %.1f), mean [90%% percentile A-side cluster bootstrap, %d "
         "resamples, seed 0]. alpha 0 and alpha 1 are fixed by identity (run 1c hold, patched run). Dropped arm "
         "groups: %s." % (res["plan_sha256"], GAP_E, NB, cfg["drop"] or "none"), ""]
    for ds, c in res["cells"].items():
        b = c["base"]
        L += ["## %s (%s): %d items, %d clusters" % (DS_NAME[ds], ds, c["n_items"], c["n_clusters"]), "",
              "Patch alone (E0) %s; random alone %s; positive control %s; structural zero %s; random norm ratio %s." % (
                  _f(b["E0"]), _f(b["rnd"]), _f(b["pc"]), _f(b["sz"]), json.dumps(c["rnd_norm_ratio"])), ""]
        keys = list(c["curves"])
        L += ["| alpha | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for a in ALPHAS:
            L.append("| %g | " % a + " | ".join(_f(c["curves"][k_].get("%g" % a)) for k_ in keys) + " |")
        L += ["", "Write norm at r, own branch, loops 2/3/4 (mean): " +
              json.dumps({k_: v for k_, v in c["write_norm"].items() if k_.startswith("donor")}), ""]
        for br, d in c["derived"].items():
            if "size_curve" in d:
                L.append("- %s size curve (NLD; share of the all-set effect over E0), top1 / top5 / top20 / all and the "
                         "random sets of equal size:" % br)
                for a, row_ in d["size_curve"].items():
                    L.append("  - alpha %s: %s" % (a, "; ".join(
                        "%s %s (share %s)" % (s_, _f(v["nld"]), _f(v["share"])) for s_, v in row_.items())))
            parts = ["%s %s" % (nm, json.dumps({k: _f(v) for k, v in d[nm].items()}))
                     for nm in ("chord_dev", "share_top", "share_rand", "top_minus_rand", "share_top_over_dall",
                                "share_rand_over_dall") if nm in d]
            L.append("- %s: %s" % (br, "; ".join(parts)))
        L += ["", "Predictions: " + json.dumps(c["predictions"]), "",
              "Concentration: the label follows the pre-registered rule (section 8); the signed share(top, 0) is the "
              "finding (a negative share means the top set's write works against the restoration).", ""]
    L += ["## Gates", "", "G3 (run 1c reproduction): " + json.dumps(res["g3"]), "",
          "G4 (dall reduces to all; the alpha-2 gap is the noise floor for top and rand above alpha 1): " +
          json.dumps(res["g4"]), "", "## Attribution store (first-order estimate against the measured top20 alpha-0 "
          "margin change)", "", json.dumps(res.get("attribution")), "",
          "## Checks", "", "```", json.dumps(res["checks"], indent=1), "```"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


def main(argv=None):
    a = parse_args(argv)
    if a.hash:
        print(json.dumps({"plan_sha256": KF.sha256_file(a.plan), "code_sha256": code_hashes(),
                          "data_sha256": json.load(open(os.path.join(a.data, "sources.json")))["outputs"]}, indent=1))
        return 0
    if a.run:
        run(a)
    else:
        summarize(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
