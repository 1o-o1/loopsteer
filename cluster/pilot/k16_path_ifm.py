"""K16 PATH-IFM (brief B-K16-PATH-IFM): pathway and input-refresh tests on the IFM twins, middle pair, 819M, d112.
Plan (frozen definitions, strengths, predictions): notes/k16_path_ifm_plan.md.
Imports k15_smoke, k15_smoke2, k15_pairs, k15_data, k10_ifm; never modifies them. Final readout only (critic C6).

New hooks (installed once per model, instance level):
  H1 r._layer wrapper: current slot, block-output edits, records.
  H2 injection-module wrapper: call counter j (assert == K per forward); edit/record pre(j) (state argument),
     arg(j) (injection argument), post(j) (output).
  H3 CausalSoftmaxAttention wrapper: per-call knockout as a boolean `segments` mask (last query, key set), no-cache
     forward only (critic C11); optional recomputation of last-query attention probabilities (W2).
Arms: Y2 (input protocols, injection models), Y1 (answer edit x knockout 2x2 + controls), Y3 (undo), W2 (window).
Every scored row is one batch-1 forward.

    python k16_path_ifm.py --selftest
    python k16_path_ifm.py --models ifm_p1_ouro,ifm_d112,ifm_p1_mid --out OUT --tag pb   (xllm_venv)
    python k16_path_ifm.py --summarize --out OUT
"""
import argparse
import gc
import glob
import json
import os
import random
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import k15_data as KD  # noqa: E402
import k15_pairs as KP  # noqa: E402
import k15_smoke as S1  # noqa: E402
import k15_smoke2 as S2  # noqa: E402

RUN_ROLE = os.environ.get("RUN_ROLE", "cluster")   # written into records instead of the host name
FULL_POOL = [False]     # cluster copy (--full_pool): lift the per-dataset quotas; every candidate is scanned

ARMS = {"ifm_p1_ouro_inj": ["Y2"], "ifm_p1_mid_inj": ["Y2"], "ifm_p2m_diag": ["Y2"],
        "ifm_p1_ouro": ["Y1", "Y3", "W2"], "ifm_d112": ["Y1", "W2"], "ifm_p1_mid": ["Y3", "W2"],
        "ifm_p2m_r5": ["Y2"], "ifm_p2m_ortho": ["Y2"]}          # cluster fill: the other two 819M designs, as diag
MODEL_ORDER = ["ifm_p1_ouro_inj", "ifm_p1_mid_inj", "ifm_p2m_diag", "ifm_p1_ouro", "ifm_d112", "ifm_p1_mid",
               "ifm_p2m_r5", "ifm_p2m_ortho"]
HUGINN = ("ifm_p1_mid", "ifm_p1_mid_inj", "ifm_p2m_diag", "ifm_p2m_r5", "ifm_p2m_ortho")
DATASETS = ["D1", "D2"]
A_Y2 = [0.5, 1.0]
PV_A = 2.0
D_MIN = 2.0
NBOOT = 1000
N_W2 = 40
SUFFIX = [28, 56, 84, 98, 104]
RND_SEEDS = [0, 1, 2]
KEYSETS = ["SRC"] + ["RND%d" % s for s in RND_SEEDS] + ["NON", "ALL"]
CAP_MIN = 180.0
GUARD_MIN = 150.0
MATTER_DM = 0.1
D45_QUOTA = {"D4": {"T1": 20, "T2": 20}, "D5": {"T2": 40}}       # plan addendum B1


def r5(x):
    return None if x is None else round(float(x), 5)


# ------------------------------------------------------------------ pure helpers (CPU-testable)
def lerp_e(pos, tgt, a, dev):
    """S2.apply_entries lerp entry on batch row 0: positions pos toward tgt [P, H] with weight a."""
    P = len(pos)
    return ("lerp", torch.zeros(P, dtype=torch.long, device=dev), torch.tensor(pos, device=dev),
            tgt.float().to(dev), torch.full((P, 1), float(a), device=dev))


def add_e(pos, delta, dev):
    P = len(pos)
    return ("add", torch.zeros(P, dtype=torch.long, device=dev), torch.tensor(pos, device=dev), delta.float().to(dev))


def ko_mask(T, keys, dev):
    """[1, T, T] bool, True at (query T-1, key in keys): masked_fill(-inf) in the backend-None attention."""
    m = torch.zeros(1, T, T, dtype=torch.bool, device=dev)
    if keys:
        m[0, T - 1, torch.tensor(sorted(keys), device=dev)] = True
    return m


def key_sets(T, src, item_idx):
    """SRC, RND0-2 (same count, not source / first / last), NON (all but source and last), ALL (all but last)."""
    last = T - 1
    src = sorted(set(src))
    pool = [k for k in range(1, last) if k not in src]
    out = {"SRC": src}
    for s in RND_SEEDS:
        rng = random.Random(1000 * s + item_idx)
        out["RND%d" % s] = sorted(rng.sample(pool, min(len(src), len(pool))))
    out["NON"] = [k for k in range(0, last) if k not in src]
    out["ALL"] = list(range(0, last))
    return out


def d1_clusters(items):
    """Cluster id per item = the A-side subject (rel, a), as B-K16-PATH-OURO. Plan amendment A1: connected components
    of the subject graph merged FULL-A's 80 D1 items into 5 clusters (40/19/12/7/2), too few for a bootstrap."""
    return ["%s:%s" % (it["rel"], it["a"]) for it in items]


def cluster_boot(vals, clusters, nb=NBOOT, seed=0):
    """Item mean with a 95% percentile bootstrap over clusters. -> [mean, lo, hi, n_items, n_clusters] or None."""
    pairs = [(float(v), c) for v, c in zip(vals, clusters) if v is not None]
    if not pairs:
        return None
    ids = sorted(set(c for _, c in pairs))
    ix = {c: i for i, c in enumerate(ids)}
    G = len(ids)
    s, n = np.zeros(G), np.zeros(G)
    for v, c in pairs:
        s[ix[c]] += v
        n[ix[c]] += 1
    pt = float(s.sum() / n.sum())
    if G < 2:
        return [round(pt, 4), None, None, len(pairs), G]
    d = np.random.default_rng(seed).integers(0, G, (nb, G))
    bs = s[d].sum(1) / n[d].sum(1)
    return [round(pt, 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4),
            len(pairs), G]


def row_vals(z, tA, tB):
    """Stored per scored row: logit A, logit B, p(B), flip to B, top-1 is A."""
    top = int(z.argmax())
    return [r5(z[tA]), r5(z[tB]), r5(torch.softmax(z.float(), -1)[tB]), int(top == tB), int(top == tA)]


def selftest():
    res = {}
    dev = "cpu"
    o = torch.randn(1, 6, 16).to(torch.bfloat16)
    tg = torch.randn(2, 16)
    e = S2.apply_entries(o, [lerp_e([2, 4], tg, 1.0, dev)])
    res["lerp_a1_exact"] = bool(torch.equal(e[0, [2, 4]], tg.to(o.dtype)) and torch.equal(e[0, [0, 1, 3, 5]], o[0, [0, 1, 3, 5]]))
    e = S2.apply_entries(o, [add_e(list(range(6)), torch.zeros(6, 16), dev)])
    res["zero_add_identity"] = bool(torch.equal(e, o))
    m = ko_mask(6, [1, 3], dev)
    res["mask_cells"] = int(m.sum()) == 2 and bool(m[0, 5, 1]) and bool(m[0, 5, 3])
    # backend-None attention arithmetic: masked keys get probability 0, all-False mask is a no-op
    sc = torch.randn(1, 2, 6, 6)
    base = torch.triu(torch.full((1, 6, 6), float("-inf")), diagonal=1)
    p0 = F.softmax(sc + base.unsqueeze(1), -1)
    p1 = F.softmax(sc + base.masked_fill(m, float("-inf")).unsqueeze(1), -1)
    pf = F.softmax(sc + base.masked_fill(torch.zeros_like(m), float("-inf")).unsqueeze(1), -1)
    res["ko_zero_prob"] = float(p1[0, :, 5, [1, 3]].abs().max())
    res["ko_other_rows_same"] = bool(torch.equal(p1[0, :, :5], p0[0, :, :5]))
    res["false_mask_identity"] = bool(torch.equal(pf, p0))
    ks = key_sets(10, [4, 5], 7)
    res["keysets"] = {k: v for k, v in ks.items()}
    res["keysets_ok"] = (all(len(ks["RND%d" % s]) == 2 and not set(ks["RND%d" % s]) & {0, 4, 5, 9} for s in RND_SEEDS)
                         and ks["NON"] == [0, 1, 2, 3, 6, 7, 8] and ks["ALL"] == list(range(9))
                         and key_sets(10, [4, 5], 7) == ks)
    items = [{"rel": "c", "a": "x", "b": "y"}, {"rel": "c", "a": "y", "b": "z"}, {"rel": "c", "a": "u", "b": "v"},
             {"rel": "k", "a": "x", "b": "w"}]
    cl = d1_clusters(items)
    res["clusters"] = cl
    res["clusters_ok"] = cl == ["c:x", "c:y", "c:u", "k:x"]
    vals = [1.0, 2.0, 3.0, None, 5.0, 6.0]
    cs = ["a", "a", "b", "b", "c", "d"]
    c1 = cluster_boot(vals, cs)
    res["boot"] = c1
    res["boot_ok"] = (abs(c1[0] - 3.4) < 1e-9 and c1[1] <= c1[0] <= c1[2] and c1[3] == 5 and c1[4] == 4
                      and cluster_boot(vals, cs) == c1)
    z = torch.tensor([0.0, 2.0, 1.0])
    res["row_vals"] = row_vals(z, 2, 1)
    res["row_vals_ok"] = res["row_vals"][3] == 1 and res["row_vals"][4] == 0
    ok = all(res[k] for k in ("lerp_a1_exact", "zero_add_identity", "mask_cells", "ko_other_rows_same",
                               "false_mask_identity", "keysets_ok", "clusters_ok", "boot_ok", "row_vals_ok"))
    ok = ok and res["ko_zero_prob"] == 0.0
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


# ------------------------------------------------------------------ hook context
class Ctx(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.slot = None
        self.blk = {}
        self.inj = {}
        self.ko = {}
        self.rec_blk = ()
        self.rec_inj = False
        self.rec_attn = None          # None | "probs" | "check"
        self.out = {}
        self.calls = 0


CTX = Ctx()


def attn_recompute(mod, xq, xk, xv, freqs_cis, segments):
    """Backend-None CausalSoftmaxAttention ops (causal_attention.py), returning last-query probs and the output."""
    from einops import rearrange
    from xllm.modules.causal_attention import repeat_kv
    from xllm.modules.rotary_positional_embedding import apply_rotary_embeddings
    bs, slen, _ = xq.shape
    q = rearrange(xq, 'b l (h d) -> b l h d', h=mod.n_heads)
    k = rearrange(xk, 'b l (h d) -> b l h d', h=mod.n_kv_heads)
    v = rearrange(xv, 'b l (h d) -> b l h d', h=mod.n_kv_heads)
    if mod.rope_head_dim == mod.head_dim:
        q, k = apply_rotary_embeddings(q, k, freqs_cis=freqs_cis, backward=False)
    elif mod.rope_head_dim > 0:
        sp = [mod.rope_head_dim, mod.head_dim - mod.rope_head_dim]
        q_r, q_n = torch.split(q, sp, dim=-1)
        k_r, k_n = torch.split(k, sp, dim=-1)
        q_r, k_r = apply_rotary_embeddings(q_r, k_r, freqs_cis=freqs_cis, backward=False)
        q, k = torch.cat([q_r, q_n], dim=-1), torch.cat([k_r, k_n], dim=-1)
    n_rep = mod.n_heads // mod.n_kv_heads
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    q = rearrange(q, 'b l h d -> b h l d')
    k = rearrange(k, 'b l h d -> b h d l')
    v = rearrange(v, 'b l h d -> b h l d')
    mask = torch.full((bs, slen, slen), float("-inf"), device=q.device)
    mask = torch.triu(mask, diagonal=1).type_as(q)
    if segments is not None:
        mask = mask.masked_fill(segments, value=float("-inf"))
    scores = torch.matmul(q, k) * mod.scale + mask.unsqueeze(1)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q)
    out = rearrange(torch.matmul(probs, v), 'b h l d -> b l (h d)')
    return probs[:, :, -1, :].float(), out


class IM(object):
    """One IFM checkpoint: k10 runner (via k15_smoke.IfmM or k15_pairs.PairM) with hooks H1-H3."""

    def __init__(self, name):
        self.name = name
        if name in HUGINN:
            P = KP.PairM(name)                     # P2m: identity_recurrent_state(example id 0, seed 42)
            self.r, self.m, self.enc, self.dev = P.r, P.m, P.enc, P.dev
            self.init_desc = P.init_desc
        else:
            M = S1.IfmM(name)
            self.r, self.m, self.enc, self.dev = M.r, M.m, M.enc, M.dev
            self.init_desc = "n/a (no hashed state)"
        r, m = self.r, self.m
        self.K = int(r.k)
        self.sched = r.schedule(self.K)
        self.n_slots = len(self.sched)
        self.slot_pt = KP.slot_points(self.sched, self.K)
        self.e = int(m.loop_end_layers)
        self.s0 = int(m.loop_start_layer)
        injm = getattr(m, "loop_input_injection_module", None)
        if injm is None:
            injm = getattr(m, "input_injection", None)
        self.injm = injm
        self.inj_counts = {}
        self.installed = False

    def loop_end_slot(self, j):
        if self.name == "ifm_d112":
            return 28 * j - 1
        return [s for s, pt in self.slot_pt.items() if pt == (j, self.e - 1)][0]

    def ko_check_slot(self):
        """Middle of loop 2 (twin/d112: 42; mid: 26)."""
        e1 = self.loop_end_slot(1)
        Lb = 28 if self.name == "ifm_d112" else self.e - self.s0
        return e1 + 1 + Lb // 2

    @torch.no_grad()
    def raw_logits(self, ids):
        x = torch.tensor([ids], device=self.dev)
        z, _ = self.r.fwd(x, self.K, cache=None, example_ids=torch.zeros(1, dtype=torch.long, device=self.dev))
        return z[0, -1].float()

    def install(self):
        r = self.r
        o_layer = r._layer

        def lay(lid, h, freq, cache, slot, _o=o_layer):
            CTX.slot = slot
            out = _o(lid, h, freq, cache, slot)
            CTX.slot = None
            ed = CTX.blk.get(slot)
            if ed:
                out = S2.apply_entries(out, ed)
            if CTX.rec_blk == "all" or slot in CTX.rec_blk:
                CTX.out[("blk", slot)] = out.detach()
            return out
        r._layer = lay
        if self.injm is not None:
            o_inj = self.injm.forward

            def inj_fwd(state, emb, _o=o_inj):
                CTX.calls += 1
                j = CTX.calls
                ed = CTX.inj.get(("pre", j))
                if ed:
                    state = S2.apply_entries(state, ed)
                ed = CTX.inj.get(("arg", j))
                if ed:
                    emb = S2.apply_entries(emb, ed)
                if CTX.rec_inj:
                    CTX.out[("pre", j)] = state.detach()
                    CTX.out[("arg", j)] = emb.detach()
                out = _o(state, emb)
                ed = CTX.inj.get(("post", j))
                if ed:
                    out = S2.apply_entries(out, ed)
                if CTX.rec_inj:
                    CTX.out[("post", j)] = out.detach()
                return out
            self.injm.forward = inj_fwd
        from xllm.modules.causal_attention import CausalSoftmaxAttention
        n = 0
        for mod in self.m.modules():
            if not isinstance(mod, CausalSoftmaxAttention):
                continue
            assert mod.backend is None, mod.backend
            prev = mod.forward

            def att(xq, xk, xv, g, freqs_cis, segments, fp32_attn_output=False, deterministic=True, cache=None,
                    _p=prev, _m=mod):
                s = CTX.slot
                kom = CTX.ko.get(s) if CTX.ko else None
                if kom is not None:
                    assert cache is None, "knockout only on the no-cache forward"
                    segments = kom if segments is None else (segments | kom)
                rec = None
                if CTX.rec_attn is not None and cache is None:
                    rec = attn_recompute(_m, xq, xk, xv, freqs_cis, segments)
                    CTX.out[("attn", s)] = rec[0]
                res = _p(xq, xk, xv, g, freqs_cis, segments, fp32_attn_output, deterministic, cache)
                if rec is not None and CTX.rec_attn == "check":
                    CTX.out[("attn_chk", s)] = float((res[0].float() - rec[1].float()).abs().max())
                return res
            mod.forward = att
            n += 1
        self.n_attn = n
        self.installed = True

    @torch.no_grad()
    def fwd(self, ids, blk=None, inj=None, ko=None, rec_blk=(), rec_inj=False, rec_attn=None):
        CTX.reset()
        CTX.blk, CTX.inj, CTX.ko = blk or {}, inj or {}, ko or {}
        CTX.rec_blk, CTX.rec_inj, CTX.rec_attn = rec_blk, rec_inj, rec_attn
        x = torch.tensor([ids], device=self.dev)
        try:
            z, _ = self.r.fwd(x, self.K, cache=None, example_ids=torch.zeros(1, dtype=torch.long, device=self.dev))
            out, calls = CTX.out, CTX.calls
        finally:
            CTX.reset()
        if self.injm is not None:
            assert calls == self.K, (calls, self.K)
        self.inj_counts[calls] = self.inj_counts.get(calls, 0) + 1
        return z[0, -1].float(), out


# ------------------------------------------------------------------ checks (first kept item)
@torch.no_grad()
def hook_checks(M, p, z_raw):
    dev = M.dev
    A = p["A"]
    T = len(A)
    src, last = p["pos"], T - 1
    out = {}

    def diff(z):
        return float((z - z_raw).abs().max())
    z0, _ = M.fwd(A)
    out["wrapped_no_edit"] = diff(z0)
    allp = list(range(T))
    H = int(M.m.model_dim)
    zero = torch.zeros(T, H)
    blk = {s: [add_e(allp, zero, dev)] for s in range(M.n_slots)}
    inj = {(k, j): [add_e(allp, zero, dev)] for k in ("pre", "arg", "post") for j in range(1, M.K + 1)} if M.injm is not None else {}
    z1, _ = M.fwd(A, blk=blk, inj=inj)
    out["zero_add_every_point"] = diff(z1)
    fm = ko_mask(T, [], dev)
    z2, _ = M.fwd(A, ko={s: fm for s in range(M.n_slots)})
    out["false_mask_every_slot"] = diff(z2)
    z3, o3 = M.fwd(A, rec_attn="check")
    out["attn_recording_on"] = diff(z3)
    out["attn_recompute_max_abs"] = max(v for k, v in o3.items() if k[0] == "attn_chk")
    out["all_bit_identical"] = all(out[k] == 0.0 for k in ("wrapped_no_edit", "zero_add_every_point",
                                                           "false_mask_every_slot", "attn_recording_on"))
    res = {"hook_identity": out}
    if any(a in ARMS[M.name] for a in ("Y1", "W2")):
        s = M.ko_check_slot()
        km = ko_mask(T, src, dev)
        zc, oc = M.fwd(A, rec_blk="all")
        zk, ok_ = M.fwd(A, ko={s: km}, rec_blk="all", rec_attn="probs")
        pre_last = all(torch.equal(oc[("blk", q)][0, :last], ok_[("blk", q)][0, :last]) for q in range(M.n_slots))
        before = all(torch.equal(oc[("blk", q)][0, last], ok_[("blk", q)][0, last]) for q in range(s))
        at = not torch.equal(oc[("blk", s)][0, last], ok_[("blk", s)][0, last])
        mass = float(ok_[("attn", s)][0][:, src].sum(-1).max())
        res["knockout_placement"] = {"slot": s, "earlier_positions_identical_all_slots": pre_last,
                                     "last_identical_before_slot": before, "last_changed_at_slot": at,
                                     "last_to_source_mass_at_slot_max_head": mass,
                                     "logit_change_max_abs": float((zk - zc).abs().max()),
                                     "ok": bool(pre_last and before and at and mass == 0.0)}
    if M.injm is not None:
        j = 2
        zc, oc = M.fwd(A, rec_inj=True, rec_blk="all")
        _, oB = M.fwd(p["B"], rec_inj=True)
        first_slot = [q for q, (_, _, tag) in enumerate(M.sched) if tag == "loop%d" % j][0]
        tgt = oB[("post", j)][0, src].float()
        _, op = M.fwd(A, inj={("post", j): [lerp_e(src, tgt, 1.0, dev)]}, rec_inj=True, rec_blk="all")
        non = [q for q in range(T) if q not in src]
        r1 = {"before_call_slots_identical": all(torch.equal(oc[("blk", q)], op[("blk", q)]) for q in range(first_slot)),
              "post_nonsource_identical": bool(torch.equal(oc[("post", j)][0, non], op[("post", j)][0, non])),
              "post_source_equals_donor": bool(torch.equal(op[("post", j)][0, src], tgt.to(op[("post", j)].dtype))),
              "pre_identical": bool(torch.equal(oc[("pre", j)], op[("pre", j)]))}
        tga = oB[("arg", j)][0, src].float()
        _, oa = M.fwd(A, inj={("arg", j): [lerp_e(src, tga, 1.0, dev)]}, rec_inj=True, rec_blk="all")
        r2 = {"pre_identical": bool(torch.equal(oc[("pre", j)], oa[("pre", j)])),
              "arg_nonsource_identical": bool(torch.equal(oc[("arg", j)][0, non], oa[("arg", j)][0, non])),
              "arg_source_changed": not torch.equal(oc[("arg", j)][0, src], oa[("arg", j)][0, src]),
              "post_nonsource_identical": bool(torch.equal(oc[("post", j)][0, non], oa[("post", j)][0, non])),
              "post_source_changed": not torch.equal(oc[("post", j)][0, src], oa[("post", j)][0, src]),
              "before_call_slots_identical": all(torch.equal(oc[("blk", q)], oa[("blk", q)]) for q in range(first_slot))}
        res["injection_placement"] = {"call": j, "post_edit": r1, "arg_edit": r2,
                                      "ok": bool(all(r1.values()) and all(r2.values()))}
    return res


# ------------------------------------------------------------------ arms (one item)
@torch.no_grad()
def item_y2(M, p, zA, oA, oB):
    dev, K = M.dev, M.K
    src = p["pos"]
    J = sorted(set([1, 2, K]))
    rows, strength, entry = {}, {}, {}
    zc = {}
    for a in A_Y2:
        for j in J:
            specs = {
                "a": {("arg", j): None},
                "b": {("arg", i): None for i in range(j, K + 1)},
                "post": {("post", j): None},
                "pre": {("pre", j): None}}
            for prot, pts in specs.items():
                inj, s_norm = {}, 0.0
                for pt in pts:
                    tgt = oB[pt][0, src].float()
                    inj[pt] = [lerp_e(src, tgt, a, dev)]
                    s_norm += float((a * (tgt - oA[pt][0, src].float())).norm())
                z, o = M.fwd(p["A"], inj=inj, rec_inj=True)
                key = "%s|%d|%g" % (prot, j, a)
                rows[key] = row_vals(z, p["tA"], p["tB"])
                strength[key] = [len(pts), r5(s_norm)]
                entry[key] = [r5((o[("post", i)][0, src].float() - oA[("post", i)][0, src].float()).norm())
                              for i in range(1, K + 1)]
                zc[key] = z
    same = {("%g" % a): bool(torch.equal(zc["b|%d|%g" % (K, a)], zc["a|%d|%g" % (K, a)])) for a in A_Y2}
    return {"rows": rows, "strength": strength, "entry_delta": entry, "bK_equals_aK": same}


@torch.no_grad()
def para_vec(M, p, slot):
    vs = []
    for ia, ib in p.get("para", []):
        _, ra = M.fwd(ia, rec_blk={slot})
        _, rb = M.fwd(ib, rec_blk={slot})
        vs.append(rb[("blk", slot)][0, -1].float() - ra[("blk", slot)][0, -1].float())
    return torch.stack(vs).mean(0) if vs else None


@torch.no_grad()
def item_y1(M, p, idx, zA, oA, oB):
    dev = M.dev
    A, src = p["A"], p["pos"]
    T = len(A)
    last = T - 1
    e1 = M.loop_end_slot(1)
    W = list(range(e1 + 1, M.n_slots))
    tA, tB = p["tA"], p["tB"]
    ops = {"O1": [lerp_e([last], oB[("blk", e1)][0, [last]], 1.0, dev)]}
    v = para_vec(M, p, e1)
    if v is not None:
        ops["PV"] = [add_e([last], (PV_A * v)[None], dev)]
    ks = key_sets(T, src, idx)
    rows, zs = {}, {}

    def run(name, ids=A, blk=None, ko=None):
        z, _ = M.fwd(ids, blk=blk, ko=ko)
        rows[name] = row_vals(z, tA, tB)
        zs[name] = z
    for op, ent in ops.items():
        run(op, blk={e1: ent})
    for kn, keys in ks.items():
        km = ko_mask(T, keys, dev)
        ko = {s: km for s in W}
        run("KO_" + kn, ko=ko)
        for op, ent in ops.items():
            run("KO_%s+%s" % (kn, op), blk={e1: ent}, ko=ko)
        if kn == "SRC":
            run("B|KO_SRC", ids=p["B"], ko=ko)
            se = {e1: [lerp_e(src, oB[("blk", e1)][0, src], 1.0, dev)]}
            run("SE", blk=se)
            run("SE+KO_SRC", blk=se, ko=ko)
    struct = None
    if max(src) == last - 1:
        struct = bool(torch.equal(zs["SE+KO_SRC"], zs["KO_SRC"]))
    return {"rows": rows, "keysets": ks, "structural_zero": struct, "pv": v is not None,
            "pv_norm": r5(v.norm()) if v is not None else None}


@torch.no_grad()
def item_y3(M, p, oA, oB, rows_have_se=None):
    dev = M.dev
    src = p["pos"]
    e1, e2, e3 = M.loop_end_slot(1), M.loop_end_slot(2), M.loop_end_slot(3)
    se = [lerp_e(src, oB[("blk", e1)][0, src], 1.0, dev)]
    rows = {}
    if rows_have_se is None:
        z, _ = M.fwd(p["A"], blk={e1: se})
        rows["SE"] = row_vals(z, p["tA"], p["tB"])
    for lab, es in (("SE+R2", e2), ("SE+R3", e3)):
        z, _ = M.fwd(p["A"], blk={e1: se, es: [lerp_e(src, oA[("blk", es)][0, src], 1.0, dev)]})
        rows[lab] = row_vals(z, p["tA"], p["tB"])
    return {"rows": rows, "slots": [e1, e2, e3]}


@torch.no_grad()
def item_w2(M, p):
    dev = M.dev
    A, src = p["A"], p["pos"]
    T = len(A)
    z0, o0 = M.fwd(A, rec_attn="probs")
    mass_mean, mass_max = [], []
    for s in range(M.n_slots):
        pr = o0[("attn", s)][0][:, src].sum(-1)          # [heads]
        mass_mean.append(r5(pr.mean()))
        mass_max.append(r5(pr.max()))
    km = ko_mask(T, src, dev)
    single = []
    for s in range(M.n_slots):
        z, _ = M.fwd(A, ko={s: km})
        single.append(row_vals(z, p["tA"], p["tB"]))
    suffix = {}
    for s0 in SUFFIX:
        if s0 >= M.n_slots:
            continue
        z, _ = M.fwd(A, ko={s: km for s in range(s0, M.n_slots)})
        suffix[str(s0)] = row_vals(z, p["tA"], p["tB"])
    return {"mass_mean": mass_mean, "mass_max": mass_max, "single": single, "suffix": suffix,
            "clean": row_vals(z0, p["tA"], p["tB"])}


# ------------------------------------------------------------------ one dataset, one model
def used_minutes(out_dir, tag, t0):
    s = 0.0
    for f in glob.glob(os.path.join(out_dir, "proc_*.json")):
        if os.path.basename(f) == "proc_%s.json" % tag:
            continue
        try:
            s += json.load(open(f))["seconds"]
        except Exception:  # noqa: BLE001
            pass
    return s / 60.0 + (time.time() - t0) / 60.0


def write_proc(out_dir, tag, t0, models, skipped, done):
    json.dump({"models": models, "seconds": round(time.time() - t0, 1), "skipped": skipped, "done": done,
               "host": RUN_ROLE}, open(os.path.join(out_dir, "proc_%s.json" % tag), "w"))


def d45_candidates(ds, data_dir):
    """B-K16-DATA rows (P1 tokenizer file), split test, id order -> k15-style pair dicts. Mapping as
    k16_data.to_pairs (read, not imported: the builder is not on the IFM host)."""
    f = os.path.join(data_dir, ds.lower(), "%s_ifm.jsonl" % ds.lower())
    rows = sorted([json.loads(x) for x in open(f, encoding="utf-8")], key=lambda r: r["id"])
    out = []
    for r in rows:
        if r["split"] != "test":
            continue
        out.append(dict(r, A=r["ids_A"], B=r["ids_B"], pos=r["source_span"], tA=r["answer_first_token_A"],
                        tB=r["answer_first_token_B"], a=r["answer_A"], b=r["answer_B"], rel=r["kind"],
                        half=r["split"], suffix=r["suffix_ids"], para=[(q["ids_A"], q["ids_B"]) for q in r["para"]]))
    return out


def d45_select(cands, know, quota, cap=1):
    """File order; skip when the kind quota is full or the cluster (template) was used `cap` times."""
    n_q, n_c, kept, scanned = {}, {}, [], 0
    for p in cands:
        q, c = p["rel"], p["cluster_id"]
        if q not in quota or n_q.get(q, 0) >= quota[q] or n_c.get(c, 0) >= cap:
            continue
        scanned += 1
        if not know(p):
            continue
        n_q[q] = n_q.get(q, 0) + 1
        n_c[c] = n_c.get(c, 0) + 1
        kept.append(p)
        if all(n_q.get(k, 0) >= v for k, v in quota.items()):
            break
    return kept, scanned


@torch.no_grad()
def run_dataset(M, ds, debug, n_w2, checks_out, data_dir=None):
    t0 = time.time()
    enc_check = None
    if ds == "D1":
        cands, quota = KD.d1_candidates(M.enc), dict(KD.D1_QUOTA)
    elif ds == "D2":
        cands, quota = KD.d2_candidates(M.enc), dict(KD.D2_QUOTA)
    else:
        cands, quota = d45_candidates(ds, data_dir), dict(D45_QUOTA[ds])
        bad = [p["id"] for p in cands if M.enc(p["prompt_A"]) + p["suffix_ids"] != p["ids_A"]
               or M.enc(p["prompt_B"]) + p["suffix_ids"] != p["ids_B"]]
        enc_check = {"rows": len(cands), "mismatch": len(bad), "first_mismatch": bad[:3]}
        print("[k16 %s] %s encoding check %s" % (M.name, ds, enc_check), flush=True)
        if bad:
            return {"items": [], "n": 0, "scanned": 0, "candidates": len(cands), "by_rel": {}, "n_clusters": 0,
                    "quota": quota, "seconds": round(time.time() - t0, 1), "enc_check": enc_check,
                    "skipped": "tokenization mismatch"}
    quota = {k: 10 ** 9 for k in quota} if FULL_POOL[0] else quota
    if debug:
        quota = {k: min(v, 2) for k, v in quota.items()}

    def know(p):
        zA = M.raw_logits(p["A"]) if not M.installed else M.fwd(p["A"])[0]
        if int(zA.argmax()) != p["tA"]:
            return False
        zB = M.raw_logits(p["B"]) if not M.installed else M.fwd(p["B"])[0]
        return int(zB.argmax()) == p["tB"]
    if ds in ("D1", "D2"):
        kept, scanned = KD.select(cands, know, quota)
    else:
        kept, scanned = d45_select(cands, know, quota)
    arms = ARMS[M.name]
    items = []
    if kept and not M.installed:
        z_raw = M.raw_logits(kept[0]["A"])
        M.install()
        checks_out.update(hook_checks(M, kept[0], z_raw))
        print("[k16 %s] hook checks %s" % (M.name, json.dumps(checks_out)), flush=True)
    rec_slots = set()
    if "Y1" in arms or "Y3" in arms:
        rec_slots = {M.loop_end_slot(j) for j in (1, 2, 3)}
    for idx, p in enumerate(kept):
        zA, oA = M.fwd(p["A"], rec_blk=rec_slots, rec_inj=M.injm is not None)
        zB, oB = M.fwd(p["B"], rec_blk=rec_slots, rec_inj=M.injm is not None)
        it = {"idx": idx, "a": p["a"], "b": p["b"], "rel": p["rel"], "half": p["half"], "pos": p["pos"],
              "T": len(p["A"]), "tA": p["tA"], "tB": p["tB"], "clean_A": row_vals(zA, p["tA"], p["tB"]),
              "clean_B": row_vals(zB, p["tA"], p["tB"])}
        if "id" in p:
            it["id"] = p["id"]
        it["D"] = r5((zB[p["tB"]] - zB[p["tA"]]) - (zA[p["tB"]] - zA[p["tA"]]))
        if "Y2" in arms:
            it["Y2"] = item_y2(M, p, zA, oA, oB)
        if "Y1" in arms:
            it["Y1"] = item_y1(M, p, idx, zA, oA, oB)
        if "Y3" in arms:
            it["Y3"] = item_y3(M, p, oA, oB, rows_have_se=True if "Y1" in arms else None)
            if "Y1" in arms:
                it["Y3"]["rows"]["SE"] = it["Y1"]["rows"]["SE"]
        if "W2" in arms and idx < n_w2:
            it["W2"] = item_w2(M, p)
        items.append(it)
        if (idx + 1) % 10 == 0 or idx < 2:
            print("[k16 %s %s] item %d/%d %.0fs" % (M.name, ds, idx + 1, len(kept), time.time() - t0), flush=True)
    if ds == "D1":
        cl = d1_clusters(items)
    elif ds == "D2":
        cl = [it["a"] for it in items]
    else:
        cl = [p["cluster_id"] for p in kept]
    for it, c in zip(items, cl):
        it["cluster"] = c
    by_rel = {}
    for it in items:
        by_rel[it["rel"]] = by_rel.get(it["rel"], 0) + 1
    return {"items": items, "n": len(items), "scanned": scanned, "candidates": len(cands), "by_rel": by_rel,
            "n_clusters": len(set(cl)), "quota": {(k if isinstance(k, str) else "%s:%s" % k): v for k, v in quota.items()},
            "seconds": round(time.time() - t0, 1), "enc_check": enc_check}


def run_model(name, out_dir, debug, n_w2, tag, t_proc0, datasets, heartbeat=None, data_dir=None):
    t0 = time.time()
    M = IM(name)
    load_s = time.time() - t0
    res = {"model": name, "host": RUN_ROLE, "K": M.K, "n_slots": M.n_slots, "arms": ARMS[name],
           "loop_end_slots": {j: M.loop_end_slot(j) for j in (1, 2, 3)},
           "injection_module": None if M.injm is None else type(M.injm).__name__,
           "init": M.init_desc, "dtype": str(next(M.m.parameters()).dtype), "load_seconds": round(load_s, 1),
           "debug": debug, "n_w2": n_w2, "datasets": {}, "checks": {}}
    try:
        res["injection_summary"] = M.r.injection_summary()
    except Exception as ex:  # noqa: BLE001
        res["injection_summary"] = {"error": repr(ex)}
    path = os.path.join(out_dir, "k16_%s.json" % name)

    def dump(partial):
        res["partial"] = partial
        res["inj_counts"] = {str(k): v for k, v in M.inj_counts.items()}
        res["n_attn_wrapped"] = getattr(M, "n_attn", None)
        res["seconds_total"] = round(time.time() - t0, 1)
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    for ds in datasets:
        if used_minutes(out_dir, tag, t_proc0) > GUARD_MIN:
            print("[k16 %s] guard: skip %s" % (name, ds), flush=True)
            res.setdefault("skipped", []).append(ds)
            continue
        res["datasets"][ds] = run_dataset(M, ds, debug, n_w2, res["checks"], data_dir)
        dump(True)
        if heartbeat:
            heartbeat()
        print("[k16 %s] %s done: %d items, %d clusters, %.0fs" % (name, ds, res["datasets"][ds]["n"],
                                                                 res["datasets"][ds]["n_clusters"],
                                                                 res["datasets"][ds]["seconds"]), flush=True)
    dump(False)
    M = None
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ summary
def summarize(out_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "k16_*.json"))):
        d = json.load(open(f))
        res[d["model"]] = d
    order = [m for m in MODEL_ORDER if m in res]
    procs = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "proc_*.json"))}
    gpu_min = round(sum(p["seconds"] for p in procs.values()) / 60.0, 2)

    def V(row, base):
        """row/base value lists -> dict of contrasts (row minus base)."""
        dA, dB = row[0] - base[0], row[1] - base[1]
        return {"dA": dA, "dB": dB, "dm": dB - dA, "pB": row[2], "flip": row[3], "top1A": row[4]}

    def agg(items, getter):
        """getter(item) -> dict or None. Returns CI per field, e (D >= D_MIN) and its excluded count."""
        vals = [(getter(it), it) for it in items]
        vals = [(v, it) for v, it in vals if v is not None]
        if not vals:
            return None
        cl = [it["cluster"] for _, it in vals]
        out = {k: cluster_boot([v[k] for v, _ in vals], cl) for k in ("dA", "dB", "dm", "pB", "flip", "top1A")
               if k in vals[0][0]}
        ee = [(v["dm"] / it["D"]) if (it["D"] is not None and it["D"] >= D_MIN) else None for v, it in vals]
        out["e"] = cluster_boot(ee, cl)
        out["e_excluded"] = sum(1 for x in ee if x is None)
        return out

    def ptv(c, f="e"):
        return None if (c is None or c.get(f) is None) else c[f][0]
    checks = {"hook_identity": {}, "knockout_placement": {}, "injection_placement": {}, "injection_call_counts": {},
              "items": {}, "y2_protocols": {}, "y2_bK_equals_aK": {}, "y1_2x2": {}, "y1_controls": {},
              "y1_positive_control": {}, "y3_undo": {}, "w2_window": {}}
    tables = {}
    for m in order:
        d = res[m]
        ck = d.get("checks", {})
        checks["hook_identity"][m] = ck.get("hook_identity")
        if "knockout_placement" in ck:
            checks["knockout_placement"][m] = ck["knockout_placement"]
        if "injection_placement" in ck:
            checks["injection_placement"][m] = ck["injection_placement"]
        if d.get("injection_module"):
            checks["injection_call_counts"][m] = {"K": d["K"], "calls_per_forward_hist": d.get("inj_counts")}
        checks["items"][m] = {ds: {"n": D["n"], "clusters": D["n_clusters"], "by_rel": D["by_rel"],
                                   "scanned": D["scanned"],
                                   "D_lt_2": sum(1 for it in D["items"] if it["D"] is None or it["D"] < D_MIN)}
                              for ds, D in d["datasets"].items()}
        tables[m] = {}
        for ds, D in d["datasets"].items():
            items = D["items"]
            T = tables[m].setdefault(ds, {})
            if "Y2" in d["arms"] and items:
                y2 = {}
                for key in items[0]["Y2"]["rows"]:
                    c = agg(items, lambda it, k=key: V(it["Y2"]["rows"][k], it["clean_A"]))
                    c["calls"] = items[0]["Y2"]["strength"][key][0]
                    c["delta_norm_sum"] = cluster_boot([it["Y2"]["strength"][key][1] for it in items],
                                                       [it["cluster"] for it in items])
                    c["entry_delta_by_call"] = [round(float(np.mean([it["Y2"]["entry_delta"][key][i] for it in items])), 3)
                                                for i in range(d["K"])]
                    c["e_by_rel"] = {rel: ptv(agg([it for it in items if it["rel"] == rel],
                                                  lambda it, k=key: V(it["Y2"]["rows"][k], it["clean_A"])))
                                     for rel in sorted(set(it["rel"] for it in items))}
                    y2[key] = c
                T["Y2"] = y2
                checks["y2_protocols"].setdefault(m, {})[ds] = {
                    k: {f: c[f] for f in ("dA", "dB", "dm", "flip", "e", "pB", "e_by_rel")} for k, c in y2.items()}
                checks["y2_bK_equals_aK"].setdefault(m, {})[ds] = "%d/%d" % (
                    sum(1 for it in items for v in it["Y2"]["bK_equals_aK"].values() if v),
                    sum(len(it["Y2"]["bK_equals_aK"]) for it in items))
            if "Y1" in d["arms"] and items:
                y1 = {}
                ops = ["O1"] + (["PV"] if all(it["Y1"]["pv"] for it in items) else [])
                for op in ops:
                    y1[op] = {"edit": agg(items, lambda it, o=op: V(it["Y1"]["rows"][o], it["clean_A"]))}
                for kn in KEYSETS:
                    y1["KO_" + kn] = agg(items, lambda it, k=kn: V(it["Y1"]["rows"]["KO_" + k], it["clean_A"]))
                    for op in ops:
                        y1["KO_%s+%s|vsKO" % (kn, op)] = agg(items, lambda it, k=kn, o=op: V(
                            it["Y1"]["rows"]["KO_%s+%s" % (k, o)], it["Y1"]["rows"]["KO_" + k]))
                        y1["KO_%s+%s|interaction" % (kn, op)] = cluster_boot(
                            [V(it["Y1"]["rows"]["KO_%s+%s" % (kn, op)], it["Y1"]["rows"]["KO_" + kn])["dm"]
                             - V(it["Y1"]["rows"][op], it["clean_A"])["dm"] for it in items],
                            [it["cluster"] for it in items])
                        y1["KO_%s+%s|interaction_e" % (kn, op)] = cluster_boot(
                            [((V(it["Y1"]["rows"]["KO_%s+%s" % (kn, op)], it["Y1"]["rows"]["KO_" + kn])["dm"]
                               - V(it["Y1"]["rows"][op], it["clean_A"])["dm"]) / it["D"])
                             if it["D"] is not None and it["D"] >= D_MIN else None for it in items],
                            [it["cluster"] for it in items])
                dko = [V(it["Y1"]["rows"]["B|KO_SRC"], it["Y1"]["rows"]["KO_SRC"])["dm"] for it in items]
                y1["D_KO_SRC"] = cluster_boot(dko, [it["cluster"] for it in items])
                y1["e_KO_over_DKO_O1"] = cluster_boot(
                    [(V(it["Y1"]["rows"]["KO_SRC+O1"], it["Y1"]["rows"]["KO_SRC"])["dm"] / x) if x >= D_MIN else None
                     for it, x in zip(items, dko)], [it["cluster"] for it in items])
                se = agg(items, lambda it: V(it["Y1"]["rows"]["SE"], it["clean_A"]))
                sek = agg(items, lambda it: V(it["Y1"]["rows"]["SE+KO_SRC"], it["Y1"]["rows"]["KO_SRC"]))
                sz = [it["Y1"]["structural_zero"] for it in items if it["Y1"]["structural_zero"] is not None]
                y1["SE"], y1["SE+KO_SRC|vsKO"] = se, sek
                ratio = (sek["dm"][0] / se["dm"][0]) if (se and sek and se["dm"][0]) else None
                y1["by_rel"] = {rel: {"O1": ptv(agg([it for it in items if it["rel"] == rel],
                                                    lambda it: V(it["Y1"]["rows"]["O1"], it["clean_A"]))),
                                      "KO_SRC+O1|vsKO": ptv(agg([it for it in items if it["rel"] == rel],
                                                                lambda it: V(it["Y1"]["rows"]["KO_SRC+O1"],
                                                                             it["Y1"]["rows"]["KO_SRC"]))),
                                      "KO_SRC_top1A": ptv(agg([it for it in items if it["rel"] == rel],
                                                              lambda it: V(it["Y1"]["rows"]["KO_SRC"], it["clean_A"])),
                                                          "top1A"),
                                      "SE+KO_SRC|vsKO": ptv(agg([it for it in items if it["rel"] == rel],
                                                                lambda it: V(it["Y1"]["rows"]["SE+KO_SRC"],
                                                                             it["Y1"]["rows"]["KO_SRC"])))}
                                 for rel in sorted(set(it["rel"] for it in items))}
                T["Y1"] = y1
                checks["y1_2x2"].setdefault(m, {})[ds] = {
                    op: {"edit": {f: y1[op]["edit"][f] for f in ("dA", "dB", "dm", "flip", "e")},
                         "KO_SRC": {f: y1["KO_SRC"][f] for f in ("dA", "dB", "dm", "flip", "top1A")},
                         "edit_given_KO_SRC": {f: y1["KO_SRC+%s|vsKO" % op][f] for f in ("dA", "dB", "dm", "flip", "e")},
                         "interaction_dm": y1["KO_SRC+%s|interaction" % op],
                         "interaction_e": y1["KO_SRC+%s|interaction_e" % op]} for op in ops}
                checks["y1_2x2"][m][ds]["D_KO_SRC"] = y1["D_KO_SRC"]
                checks["y1_controls"].setdefault(m, {})[ds] = {
                    kn: {"KO_dm": y1["KO_" + kn]["dm"], "KO_top1A": y1["KO_" + kn]["top1A"],
                         **{"%s_interaction_e" % op: y1["KO_%s+%s|interaction_e" % (kn, op)] for op in ops},
                         **{"%s_given_KO_e" % op: y1["KO_%s+%s|vsKO" % (kn, op)]["e"] for op in ops}}
                    for kn in KEYSETS if kn != "SRC"}
                checks["y1_positive_control"].setdefault(m, {})[ds] = {
                    "SE_e": se["e"], "SE_dm": se["dm"], "SE_given_KO_SRC_e": sek["e"], "SE_given_KO_SRC_dm": sek["dm"],
                    "dm_ratio_KO_over_noKO": None if ratio is None else round(ratio, 4),
                    "structural_zero_capital": "%d/%d" % (sum(sz), len(sz)),
                    "pass": bool(ratio is not None and ratio <= 0.5 and all(sz))}
            if "Y3" in d["arms"] and items:
                y3 = {lab: agg(items, lambda it, l_=lab: V(it["Y3"]["rows"][l_], it["clean_A"]))
                      for lab in ("SE", "SE+R2", "SE+R3")}
                T["Y3"] = y3
                checks["y3_undo"].setdefault(m, {})[ds] = {
                    lab: {f: c[f] for f in ("dA", "dB", "dm", "flip", "e")} for lab, c in y3.items()}
                checks["y3_undo"][m][ds]["restore_slots"] = items[0]["Y3"]["slots"]
                checks["y3_undo"][m][ds]["e_by_rel"] = {
                    lab: {rel: ptv(agg([it for it in items if it["rel"] == rel],
                                       lambda it, l_=lab: V(it["Y3"]["rows"][l_], it["clean_A"])))
                          for rel in sorted(set(it["rel"] for it in items))} for lab in ("SE", "SE+R2", "SE+R3")}
            w2it = [it for it in items if "W2" in it]
            if "W2" in d["arms"] and w2it:
                ns = d["n_slots"]
                cl = [it["cluster"] for it in w2it]
                single = [cluster_boot([V(it["W2"]["single"][s], it["clean_A"])["dm"] for it in w2it], cl)
                          for s in range(ns)]
                mm = [round(float(np.mean([it["W2"]["mass_mean"][s] for it in w2it])), 4) for s in range(ns)]
                mx = [round(float(np.mean([it["W2"]["mass_max"][s] for it in w2it])), 4) for s in range(ns)]
                matter = [s for s in range(ns) if single[s] and single[s][0] >= MATTER_DM and single[s][1] is not None
                          and single[s][1] > 0]
                suf = {s0: agg(w2it, lambda it, k=s0: V(it["W2"]["suffix"][k], it["clean_A"]))
                       for s0 in w2it[0]["W2"]["suffix"]}
                T["W2"] = {"single": single, "mass_mean": mm, "mass_max": mx, "suffix": suf, "matter": matter}
                imax = int(np.argmax([x[0] if x else -1e9 for x in single]))
                checks["w2_window"].setdefault(m, {})[ds] = {
                    "n_items": len(w2it), "mattering_slots": matter,
                    "dependence_end": max(matter) if matter else None,
                    "max_single_dm": [imax, single[imax]],
                    "suffix_dm": {s0: c["dm"] for s0, c in suf.items()},
                    "suffix_top1A": {s0: c["top1A"] for s0, c in suf.items()},
                    "mass_mean_peak_slot": int(np.argmax(mm)), "mass_mean_peak": max(mm)}
    checks["ci"] = ("95%% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; "
                    "D2: A problem), %d resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters]" % NBOOT)
    checks["e_rule"] = "e = dm / D only where D = m(clean B) - m(clean A) >= %.1f logits" % D_MIN
    checks["downloads"] = []
    checks["gpu_minutes"] = gpu_min
    checks["processes"] = procs
    data_dir = os.path.join(out_dir, "data_status.json")
    checks["d4_d5"] = json.load(open(data_dir)) if os.path.exists(data_dir) else "not run (no data at launch)"
    json.dump({"models": res, "tables": tables, "processes": procs}, open(os.path.join(out_dir, "results.json"), "w"))
    json.dump(checks, open(os.path.join(out_dir, "checks.json"), "w"), indent=1)
    write_summary(out_dir, res, order, tables, checks)
    print(json.dumps({k: checks[k] for k in ("items", "gpu_minutes")}), flush=True)


def write_summary(out_dir, res, order, tables, checks):
    def f2(x):
        return "-" if x is None else "%.2f" % x

    def fci(c):
        if c is None:
            return "-"
        return "%s [%s, %s]" % (f2(c[0]), f2(c[1]), f2(c[2]))

    def cell(c, f="e"):
        if c is None:
            return "-"
        return "%s / %s" % (fci(c.get(f)), f2(c["flip"][0] if c.get("flip") else None))
    L = ["# K16 PATH-IFM summary", "",
         "Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated "
         "baseline; e = dm / D only where D >= %.1f; flip = top-1 is B. CI: %s." % (D_MIN, checks["ci"]), "",
         "GPU minutes: %s. D4/D5: %s." % (checks["gpu_minutes"], checks["d4_d5"]), "",
         "## Hook checks", "", "| model | identity (all bit-identical) | knockout placement ok | injection placement ok | "
         "attn recompute max abs |", "|---|---|---|---|---|"]
    for m in order:
        hi = checks["hook_identity"].get(m) or {}
        L.append("| %s | %s | %s | %s | %s |" % (m, hi.get("all_bit_identical"),
                                               (checks["knockout_placement"].get(m) or {}).get("ok", "n/a"),
                                               (checks["injection_placement"].get(m) or {}).get("ok", "n/a"),
                                               hi.get("attn_recompute_max_abs")))
    L += ["", "## Items", "", "```", json.dumps(checks["items"], indent=1), "```", ""]
    for m in order:
        d = res[m]
        for ds, T in tables[m].items():
            if "Y2" in T:
                K = d["K"]
                J = sorted(set([1, 2, K]))
                L += ["## Y2 input protocols: %s, %s (K=%d), e [CI] / flip; vs clean A" % (m, ds, K), "",
                      "| protocol | a | " + " | ".join("j=%d" % j for j in J) + " |", "|---|---|" + "---|" * len(J)]
                for prot, lab in (("a", "(a) single arg(j)"), ("b", "(b) persistent arg(j..K)"),
                                  ("post", "(c) state post_inj(j)"), ("pre", "(c) state pre_inj(j)")):
                    for a in A_Y2:
                        L.append("| %s | %g | " % (lab, a) + " | ".join(
                            cell(T["Y2"].get("%s|%d|%g" % (prot, j, a))) for j in J) + " |")
                L += ["", "dm (logits) and dz(A) / dz(B) at a=1:", "",
                      "| protocol | " + " | ".join("j=%d" % j for j in J) + " |", "|---|" + "---|" * len(J)]
                for prot in ("a", "b", "post", "pre"):
                    L.append("| %s | " % prot + " | ".join(
                        (lambda c: "-" if c is None else "%s (%s / %s)" % (f2(c["dm"][0]), f2(c["dA"][0]), f2(c["dB"][0])))(
                            T["Y2"].get("%s|%d|1" % (prot, j))) for j in J) + " |")
                L += ["", "Strength at a=1 (edited calls, summed delta norm): " + "; ".join(
                    "%s j=%d: %d, %s" % (prot, j, T["Y2"]["%s|%d|1" % (prot, j)]["calls"],
                                        f2(T["Y2"]["%s|%d|1" % (prot, j)]["delta_norm_sum"][0]))
                    for prot in ("a", "b", "post", "pre") for j in J), "",
                      "Entry delta |post_edit(i) - post_clean(i)| at the source, by call i, a=1: " + "; ".join(
                    "%s j=%d: %s" % (prot, j, T["Y2"]["%s|%d|1" % (prot, j)]["entry_delta_by_call"])
                    for prot in ("a", "b", "post", "pre") for j in J), "",
                      "(b) at j=K equals (a) at j=K bit-exactly: %s" % checks["y2_bK_equals_aK"][m][ds], ""]
            if "Y1" in T:
                y1 = T["Y1"]
                ops = [o for o in ("O1", "PV") if o in y1]
                L += ["## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: %s, %s" % (m, ds), "",
                      "| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |", "|---|---|---|---|---|---|---|---|"]

                def rl(lab, c, base):
                    if c is None:
                        return "| %s | %s | - | - | - | - | - | - |" % (lab, base)
                    return "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                        lab, base, fci(c["dm"]), f2(c["dA"][0]), f2(c["dB"][0]), fci(c["e"]), f2(c["flip"][0]),
                        f2(c["top1A"][0]))
                for op in ops:
                    L.append(rl(op, y1[op]["edit"], "clean"))
                for kn in KEYSETS:
                    L.append(rl("KO_" + kn, y1["KO_" + kn], "clean"))
                    for op in ops:
                        L.append(rl("KO_%s + %s" % (kn, op), y1["KO_%s+%s|vsKO" % (kn, op)], "KO_" + kn))
                L.append(rl("SE (source O1 a=1)", y1["SE"], "clean"))
                L.append(rl("KO_SRC + SE", y1["SE+KO_SRC|vsKO"], "KO_SRC"))
                L += ["", "Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: " + "; ".join(
                    "%s %s %s" % (kn, op, fci(y1["KO_%s+%s|interaction_e" % (kn, op)])) for kn in KEYSETS for op in ops),
                      "", "D_KO_SRC (B minus A under KO_SRC): %s. O1 e normalised by D_KO: %s." % (
                          fci(y1["D_KO_SRC"]), fci(y1["e_KO_over_DKO_O1"])),
                      "", "By relation (e): %s" % json.dumps(y1["by_rel"]),
                      "", "Positive control: %s" % json.dumps(checks["y1_positive_control"][m][ds]), ""]
            if "Y3" in T:
                L += ["## Y3 undo (source O1 a=1 at the loop-1 end, then clean source restored): %s, %s" % (m, ds), "",
                      "| row | dm [CI] | dz(A) | dz(B) | e [CI] | flip |", "|---|---|---|---|---|---|"]
                for lab in ("SE", "SE+R2", "SE+R3"):
                    c = T["Y3"][lab]
                    L.append("| %s | %s | %s | %s | %s | %s |" % (lab, fci(c["dm"]), f2(c["dA"][0]), f2(c["dB"][0]),
                                                                fci(c["e"]), f2(c["flip"][0])))
                L.append("")
            if "W2" in T:
                w = T["W2"]
                L += ["## W2 retrieval window: %s, %s (%d items)" % (m, ds, checks["w2_window"][m][ds]["n_items"]), "",
                      "Mattering slots (mean dm >= %.1f, CI lo > 0): %s" % (MATTER_DM, w["matter"]),
                      "", "Suffix knockout (all calls >= s), dm [CI] / top-1 A: " + "; ".join(
                          "s=%s: %s / %s" % (s0, fci(c["dm"]), f2(c["top1A"][0])) for s0, c in w["suffix"].items()),
                      "", "| slot | mass mean | mass max-head | single-call dm [CI] |", "|---|---|---|---|"]
                for s in range(len(w["single"])):
                    L.append("| %d | %s | %s | %s |" % (s, f2(w["mass_mean"][s]), f2(w["mass_max"][s]), fci(w["single"][s])))
                L.append("")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L))


# ------------------------------------------------------------------ main
def main():
    global GUARD_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--datasets", default="D1,D2")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="p")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--full_pool", action="store_true", help="cluster copy: no quota, every candidate scanned")
    ap.add_argument("--n_w2", type=int, default=N_W2)
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--guard", type=float, default=GUARD_MIN)
    ap.add_argument("--arms", default="", help="cluster extension: comma list of arms (Y1, Y2, Y3, W2) that replaces "
                    "ARMS for every --models entry (records keep the arms they ran)")
    a = ap.parse_args()
    FULL_POOL[0] = a.full_pool
    GUARD_MIN = a.guard
    if a.arms:
        arms = [x for x in a.arms.split(",") if x]
        assert arms and set(arms) <= {"Y1", "Y2", "Y3", "W2"}, arms
        for name in [x for x in a.models.split(",") if x]:
            ARMS[name] = arms
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
        return
    t0 = time.time()
    models = [x for x in a.models.split(",") if x]
    skipped = []
    try:
        for name in models:
            if used_minutes(a.out, a.tag, t0) > GUARD_MIN:
                print("[k16] guard: skip %s" % name, flush=True)
                skipped.append(name)
                continue
            run_model(name, a.out, a.debug, a.n_w2, a.tag, t0, [x for x in a.datasets.split(",") if x],
                      heartbeat=lambda: write_proc(a.out, a.tag, t0, a.models, skipped, False),
                      data_dir=a.data_dir or None)
            write_proc(a.out, a.tag, t0, a.models, skipped, False)
    finally:
        write_proc(a.out, a.tag, t0, a.models, skipped, True)


if __name__ == "__main__":
    main()
