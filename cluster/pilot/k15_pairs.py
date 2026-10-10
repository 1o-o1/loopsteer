"""K15 FULL-B (brief B-K15-FULL-B): matched recipe pairs. Middle-loop injection pair (E1) and depth sampling (E4).
Plan: notes/k15_fullB_plan.md. Builds on k15_smoke.py and k15_smoke2.py (imported, not
modified) and k10_ifm.py (loader and forward).

Models (xllm Huginn class, read through the coda):
  ifm_p1_mid       P1 dense-middle 8/12/8, 8 loops, no injection, state init = prelude output
  ifm_p1_mid_inj   same with diagonal injection of the prelude output before every recurrence
  ifm_p2m_r5       P2 huginn-m 1/2/1, 5 loops, diagonal injection, fixed depth 5 in training
  ifm_p2m_diag     P2 huginn-m fixed-pln5 (PLN-5 depth sampling), diagonal injection
  ifm_p2m_ortho    P2 huginn-m orthoinj (PLN-5), orthogonal injection
Items: D1 capital/city (k13 build_pairs, smoke-2 rule, 80 pairs). D2 arithmetic only for a model with < 30 D1 pairs.
Arms (sites L0 / Lmid / Llast of the recurrent block at loops 1, 2, K; P2 has no separate Lmid):
  S   O2 h += a v (paraphrase vector, smoke-1 rule) and O4 h += a |v| u at the subject positions, a in [0.5, 1, 2]
  C   O1 lerp toward clean B at the subject positions, a = 0.30..0.70 step 0.05, at Llast@1 and at L0@K
Controls: O5 = O1 a=1 at every position at Llast@K (whole loop-K state = clean B, then the coda): expect 1.0.
          O5_last = O1 a=1 at the last position at the final (coda) point: expect 1.0.
Readout = m.output(last coda layer output). P2 initial state = identity_recurrent_state(example id 0, seed 42).

    python k15_pairs.py --selftest
    python k15_pairs.py --models ifm_p1_mid,ifm_p1_mid_inj,ifm_p2m_r5,ifm_p2m_diag,ifm_p2m_ortho --out OUT   (xllm_venv)
    python k15_pairs.py --summarize --out OUT
"""
import argparse
import gc
import glob
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import k15_smoke as S1  # noqa: E402
import k15_smoke2 as S2  # noqa: E402

MODELS = ["ifm_p1_mid", "ifm_p1_mid_inj", "ifm_p2m_r5", "ifm_p2m_diag", "ifm_p2m_ortho"]
CKPT_NAME = {"ifm_p1_mid": "LoopedLM-P1-dense-middle-336b",
             "ifm_p1_mid_inj": "LoopedLM-P1-dense-middle-prelude-injection-336b",
             "ifm_p2m_r5": "LoopedLM-P2-huginn-m-fixed-r5", "ifm_p2m_diag": "LoopedLM-P2-huginn-m-fixed-pln5",
             "ifm_p2m_ortho": "LoopedLM-P2-huginn-m-orthoinj"}
A_S = [0.5, 1.0, 2.0]
A_C = list(S2.A_C)
TAU = S2.TAU
N_HALF = 40
N_D2 = 80
MIN_D1 = 30
CAP_MIN = 120.0
STOP_MIN = 110.0
P2_SEED = 42
N_BOOT = 1000
ARITH_PARA = ["Compute: {} + {} =", "What is {} plus {}? The answer is"]


def fa(a):
    return "%g" % a


def key(arm, a, site):
    return "%s|%s|%s" % (arm, fa(a), site)


# ------------------------------------------------------------------ pure helpers (CPU-testable)
def slot_points(schedule, K):
    """schedule [(slot, layer, tag)] -> {slot: (stage, layer)}; stage 0 prelude, j loop j, K+1 coda."""
    out = {}
    for slot, lid, tag in schedule:
        if tag == "prelude":
            out[slot] = (0, lid)
        elif tag == "coda":
            out[slot] = (K + 1, lid)
        else:
            out[slot] = (int(tag[4:]), lid)
    return out


def make_sites(s0, e, K):
    """Sites of the recurrent block [s0, e): L0, Lmid (only if distinct from L0 and Llast), Llast at loops 1, 2, K."""
    Lb = e - s0
    mid = Lb // 2
    labs = [("L0", s0)]
    if 0 < mid < Lb - 1:
        labs.append(("Lmid", s0 + mid))
    labs.append(("Llast", e - 1))
    return [("%s@%d" % (lab, j), (j, l), j) for j in (1, 2, K) for lab, l in labs], dict(labs)


def first_diff(x, y):
    k = 0
    while k < min(len(x), len(y)) and x[k] == y[k]:
        k += 1
    return k


def boot_idx(n, nb=N_BOOT, seed=0):
    return np.random.default_rng(seed).integers(0, n, (nb, n))


def bc_rows(M):
    """Bimodality coefficient per row of M [R, n] (same formula as k15_smoke2.bimodality)."""
    n = M.shape[1]
    m = M.mean(1, keepdims=True)
    d = M - m
    s2 = (d ** 2).mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (d ** 3).mean(1) / s2 ** 1.5
        k = (d ** 4).mean(1) / s2 ** 2 - 3.0
        G = g * math.sqrt(n * (n - 1)) / (n - 2)
        Kc = ((n + 1) * k + 6.0) * (n - 1) / ((n - 2) * (n - 3))
        out = (G * G + 1.0) / (Kc + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    out[s2 <= 0] = np.nan
    return out


def stat_ci(xs, stat):
    """(point, lo, hi) with a 95% percentile bootstrap over items (N_BOOT resamples, seed 0)."""
    xs = np.array([x for x in xs if x is not None], dtype=np.float64)
    n = len(xs)
    if n == 0:
        return None
    if stat == "mean":
        f = lambda M: M.mean(1)  # noqa: E731
    elif stat == "mid":
        f = lambda M: ((M >= 0.2) & (M <= 0.8)).mean(1)  # noqa: E731
    elif stat == "bc":
        if n < 4:
            return None
        f = bc_rows
    else:
        raise KeyError(stat)
    pt = float(f(xs[None])[0])
    if n < 2:
        return [pt, None, None]
    v = f(xs[boot_idx(n)])
    v = v[~np.isnan(v)]
    return [pt, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def selftest():
    res = {}
    lst = [(i, "prelude") for i in range(8)] + [(8 + i, "loop%d" % j) for j in range(1, 9) for i in range(12)]
    lst += [(20 + i, "coda") for i in range(8)]
    sch = [(s, lid, tag) for s, (lid, tag) in enumerate(lst)]
    sp = slot_points(sch, 8)
    res["mid_slots"] = len(sp)
    res["mid_final"] = list(sp[len(sp) - 1])
    res["mid_loop1_end"] = list(sp[8 + 11])
    sites, labs = make_sites(8, 20, 8)
    res["mid_sites"] = [s[0] + ":" + str(list(s[1])) for s in sites]
    sites2, labs2 = make_sites(1, 3, 5)
    res["p2_sites"] = [s[0] + ":" + str(list(s[1])) for s in sites2]
    res["first_diff"] = [first_diff([222, 989], [222, 991]), first_diff([5], [6]), first_diff([1, 2], [1, 2])]
    g = np.random.default_rng(3)
    xs = list(g.normal(size=80)) + list(g.normal(size=40) + 4)
    res["bc_match_s2"] = abs(float(bc_rows(np.array(xs)[None])[0]) - S2.bimodality(xs))
    c = stat_ci(xs, "mean")
    res["mean_ci_brackets"] = bool(c[1] <= c[0] <= c[2])
    res["mean_ci_repeatable"] = stat_ci(xs, "mean") == c
    res["mid_point"] = stat_ci([0.1, 0.2, 0.5, 0.8, 0.9], "mid")[0]
    # O5: lerp a=1 at every position gives the target exactly; O4 delta norm
    o = torch.randn(3, 5, 16).to(torch.bfloat16)
    tgt = torch.randn(5, 16)
    ed, _ = S2.build_edits([{"kind": "lerp", "pt": (1, 0), "pos": list(range(5)), "tgt": tgt, "a": 1.0}], 1, "cpu")
    e = S2.apply_entries(o, ed[(1, 0)])
    res["o5_allpos_exact"] = bool(torch.equal(e[1], tgt.to(o.dtype)) and torch.equal(e[0], o[0]) and torch.equal(e[2], o[2]))
    v = torch.randn(2, 16)
    u = S1.unit_random((2, 16), 7)
    d = 2.0 * v.norm(dim=-1, keepdim=True) * u
    res["o4_norm_rel_err"] = float(((d.norm(dim=-1) / (2.0 * v.norm(dim=-1))) - 1).abs().max())
    ok = (res["mid_slots"] == 112 and res["mid_final"] == [9, 27] and res["mid_loop1_end"] == [1, 19]
          and len(sites) == 9 and sites[1][1] == (1, 14) and sites[-1][1] == (8, 19)
          and len(sites2) == 6 and [s[1] for s in sites2] == [(1, 1), (1, 2), (2, 1), (2, 2), (5, 1), (5, 2)]
          and res["first_diff"] == [1, 0, 2] and res["bc_match_s2"] < 1e-9 and res["mean_ci_brackets"]
          and res["mean_ci_repeatable"] and abs(res["mid_point"] - 0.6) < 1e-9 and res["o5_allpos_exact"]
          and res["o4_norm_rel_err"] < 1e-5)
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


# ------------------------------------------------------------------ model
class PairM(object):
    def __init__(self, name):
        import k10_ifm
        self.name = name
        r = k10_ifm.make_runner(name)
        self.r, self.m = r, r.m
        assert r.arch == "huginn", r.arch
        assert getattr(self.m, "huginn_hierarchical_state", "none") in ("none", None)
        r.set_depth(r.K_trained)
        self.K = int(r.k)
        m = self.m
        self.s0, self.e, self.nl = int(m.loop_start_layer), int(m.loop_end_layers), int(m.num_layers)
        self.dev = r.dev
        self.slot_pt = slot_points(r.schedule(self.K), self.K)
        self.final = self.slot_pt[len(self.slot_pt) - 1]
        self.sites, self.layers_desc = make_sites(self.s0, self.e, self.K)
        self.loop_ends = [((j, self.e - 1), j) for j in range(1, self.K + 1)]
        self.normal_init = getattr(m, "huginn_state_init", None) == "normal"
        if self.normal_init:
            self.install_init()
        self.init_desc = ("identity_recurrent_state(example id 0 for every row, seed %d)" % P2_SEED if self.normal_init
                          else "native initialize_recurrent_state (huginn_state_init %r)" % m.huginn_state_init)

    def install_init(self):
        m = self.m

        def init(ie, cl, ex, mode, _m=m):
            return _m.identity_recurrent_state(ie, cl, torch.zeros_like(ex), P2_SEED)
        self.r._init_state = init

    def enc(self, t, ans=False):
        ids = self.r.tok.hf(t, add_special_tokens=False)["input_ids"]
        return list(ids) if ans else [self.r.tok.bos_token_id] + list(ids)

    @torch.no_grad()
    def run3(self, rows, edits=None, record=()):
        """Right-padded batch (causal: real positions never see the pads), example ids all 0. {pt: tensor}."""
        r = self.r
        T = max(len(x) for x in rows)
        x = torch.tensor([list(x) + [0] * (T - len(x)) for x in rows], device=self.dev)
        rec, edits = {}, (edits or {})
        orig = r._layer

        def lay(lid, h, freq, cache, slot):
            out = orig(lid, h, freq, cache, slot)
            pt = self.slot_pt[slot]
            if pt in edits:
                out = S2.apply_entries(out, edits[pt])
            if pt in record:
                rec[pt] = out.detach()
            return out
        r._layer = lay
        try:
            logits, _ = r.fwd(x, self.K, cache=None, example_ids=torch.zeros(len(rows), dtype=torch.long, device=self.dev))
        finally:
            r._layer = orig
        self.last_logits = logits
        return rec

    @torch.no_grad()
    def readout(self, h, last):
        hl = h[torch.arange(h.shape[0], device=h.device), torch.tensor(last, device=h.device)]
        return self.m.output(hl[:, None, :], None, None)[:, -1].float()

    def facts(self):
        m, r = self.m, self.r
        inj = getattr(m, "input_injection", None)
        try:
            isum = r.injection_summary()
        except Exception as ex:  # noqa: BLE001
            isum = {"error": repr(ex)}
        return {"checkpoint": CKPT_NAME[self.name], "class": type(m).__name__,
                "prelude_layers": list(range(self.s0)), "recurrent_layers": list(range(self.s0, self.e)),
                "coda_layers": list(range(self.e, self.nl)), "loops": self.K,
                "state_init": getattr(m, "huginn_state_init", None), "init_used": self.init_desc,
                "injection": None if inj is None else type(inj).__name__,
                "injection_summary": isum,
                "sampling_scheme": getattr(m, "sampling_scheme", None), "backprop_depth": getattr(m, "backprop_depth", None),
                "final_point": list(self.final), "sites": [[lab, list(pt)] for lab, pt, _ in self.sites],
                "source": "ckpt/%s/config.json; xllm-loop/xllm/models/looped.py (Huginn 390-595, DepthControlledHuginn "
                          "598-764, InputInjection 69-139, OrthogonalInputInjection 162-179)" % CKPT_NAME[self.name]}


# ------------------------------------------------------------------ items
def d1_candidates(M):
    from k13_edit import build_pairs
    pairs = build_pairs(M.enc)
    for p in pairs:
        p["para"] = [(M.enc(t.format(p["a"])), M.enc(t.format(p["b"]))) for t in S1.PARA[p["rel"]]]
        p["group"] = (p["rel"], p["a"]), (p["rel"], p["b"])
    return pairs


def d2_candidates(M, n=600, seed=0):
    """Minimal arithmetic set (plan D9): '{a} + {b} =' plus the shared answer-token prefix; pairs differ in b."""
    rng = random.Random(seed)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 20000:
        tries += 1
        a, b, b2 = rng.randint(10, 49), rng.randint(10, 49), rng.randint(10, 49)
        if b == b2 or (a, b, b2) in seen:
            continue
        seen.add((a, b, b2))
        A, B = M.enc("%d + %d =" % (a, b)), M.enc("%d + %d =" % (a, b2))
        if len(A) != len(B):
            continue
        diff = [k for k in range(len(A)) if A[k] != B[k]]
        if not diff or diff[-1] - diff[0] + 1 != len(diff):
            continue
        ta, tb = M.enc(" %d" % (a + b), ans=True), M.enc(" %d" % (a + b2), ans=True)
        k = first_diff(ta, tb)
        if k >= min(len(ta), len(tb)):
            continue
        pre = ta[:k]
        para = [(M.enc(t.format(a, b)) + pre, M.enc(t.format(a, b2)) + pre) for t in ARITH_PARA]
        out.append({"A": A + pre, "B": B + pre, "pos": diff, "tA": ta[k], "tB": tb[k], "a": "%d+%d" % (a, b),
                    "b": "%d+%d" % (a, b2), "rel": "arith", "para": para, "group": (("arith", a),), "ans_prefix": pre})
    return out


# ------------------------------------------------------------------ one pair
@torch.no_grad()
def para_vectors(M, p, pts):
    P = len(p["pos"])
    vs, modes = {pt: [] for pt in pts}, []
    for ia, ib in p["para"]:
        ra, rb = M.run3([ia], record=set(pts)), M.run3([ib], record=set(pts))
        sa, sb = S1.subj_span(ia, ib)
        aligned = len(sa) == P and len(sb) == P
        modes.append("aligned" if aligned else "last_token")
        for pt in pts:
            if aligned:
                v = rb[pt][0, sb].float() - ra[pt][0, sa].float()
            else:
                v = (rb[pt][0, sb[-1]].float() - ra[pt][0, sa[-1]].float()).expand(P, -1)
            vs[pt].append(v)
    return {pt: torch.stack(x).mean(0) for pt, x in vs.items()}, modes


@torch.no_grad()
def clean_pass(M, p, rec_pts):
    last = len(p["A"]) - 1
    rA, rB = M.run3([p["A"]], record=rec_pts), M.run3([p["B"]], record=rec_pts)
    zA, zB = M.readout(rA[M.final], [last])[0], M.readout(rB[M.final], [last])[0]
    ok = int(zA.argmax()) == p["tA"] and int(zB.argmax()) == p["tB"]
    return ok, rA, rB, zA, zB


@torch.no_grad()
def run_pair(M, p, rA, rB, zA, zB, seed_base, cells, stats):
    dev, final, K = M.dev, M.final, M.K
    pos, tA, tB = p["pos"], p["tA"], p["tB"]
    last = len(p["A"]) - 1
    lab2pt = dict((lab, pt) for lab, pt, _ in M.sites)
    site_pts = [pt for _, pt, _ in M.sites]
    mB = float(zB[tB] - zB[tA])
    V, modes = para_vectors(M, p, site_pts)
    specs = []
    for si, (lab, pt, _) in enumerate(M.sites):
        v = V[pt]
        u = S1.unit_random(tuple(v.shape), seed_base + si).to(dev)
        for a in A_S:
            specs.append({"arm": "S_O2", "site": lab, "a": a, "kind": "add", "pt": pt, "pos": pos, "delta": a * v})
            specs.append({"arm": "S_O4", "site": lab, "a": a, "kind": "add", "pt": pt, "pos": pos,
                          "delta": a * v.norm(dim=-1, keepdim=True) * u})
    for lab in ("Llast@1", "L0@%d" % K):
        pt = lab2pt[lab]
        for a in A_C:
            specs.append({"arm": "C", "site": lab, "a": a, "kind": "lerp", "pt": pt, "pos": pos,
                          "tgt": rB[pt][0, pos].float()})
    ptK = lab2pt["Llast@%d" % K]
    allpos = list(range(len(p["A"])))
    specs.append({"arm": "O5", "site": "Llast@K_allpos", "a": 1.0, "kind": "lerp", "pt": ptK, "pos": allpos,
                  "tgt": rB[ptK][0].float()})
    specs.append({"arm": "O5_last", "site": "final_lastpos", "a": 1.0, "kind": "lerp", "pt": final, "pos": [last],
                  "tgt": rB[final][0, [last]].float()})
    ed, _ = S2.build_edits(specs, 1, dev)
    n = 1 + len(specs)
    rec = M.run3([p["A"]] * n, ed, {final})
    z = M.readout(rec[final], [last] * n)
    zc = z[0]
    stats["gap"] = max(stats["gap"], float((zc - zA).abs().max()))
    for r_, sp in enumerate(specs):
        zr, base, flag = z[r_ + 1], zc, 0
        stats["rows"] += 1
        t2 = zr.topk(2).values
        if float(t2[0] - t2[1]) < TAU or sp["arm"] in ("O5", "O5_last"):   # controls always scored at batch 1
            e1, _ = S2.build_edits([sp], 0, dev)
            zr = M.readout(M.run3([p["A"]], e1, {final})[final], [last])[0]
            base, flag = zA, 1
            stats["reruns"] += 1
        mA = float(base[tB] - base[tA])
        d = mB - mA
        m_e = float(zr[tB] - zr[tA])
        c = cells.setdefault(key(sp["arm"], sp["a"], sp["site"]),
                             {"effect": [], "flip": [], "dA": [], "dB": [], "rerun": []})
        c["effect"].append((m_e - mA) / d if abs(d) > 1e-3 else None)
        c["flip"].append(int(int(zr.argmax()) == tB))
        c["dA"].append(float(zr[tA] - base[tA]))
        c["dB"].append(float(zr[tB] - base[tB]))
        c["rerun"].append(flag)
    return modes, mB


# ------------------------------------------------------------------ one dataset
@torch.no_grad()
def run_dataset(M, ds, cands, halves, quota_fn, cap_per_group, t0):
    """halves: [(half name, parity or None, n_target)]. Returns dataset dict."""
    rec_pts = set(pt for _, pt, _ in M.sites) | {M.final}
    cells, items = {}, []
    stats = {"gap": 0.0, "rows": 0, "reruns": 0}
    scanned = {}
    for half, parity, n_target in halves:
        quota = quota_fn(n_target)
        n_rel, n_grp, kept = {}, {}, 0
        scanned[half] = 0
        for i, p in enumerate(cands):
            if parity is not None and i % 2 != parity:
                continue
            if n_rel.get(p["rel"], 0) >= quota.get(p["rel"], 0) or any(n_grp.get(g, 0) >= cap_per_group for g in p["group"]):
                continue
            scanned[half] += 1
            ok, rA, rB, zA, zB = clean_pass(M, p, rec_pts)
            if not ok:
                continue
            n_rel[p["rel"]] = n_rel.get(p["rel"], 0) + 1
            for g in p["group"]:
                n_grp[g] = n_grp.get(g, 0) + 1
            kept += 1
            modes, mB = run_pair(M, p, rA, rB, zA, zB, 100003 * (i + 1) + (0 if ds == "D1" else 7), cells, stats)
            items.append({"i": i, "half": half, "a": p["a"], "b": p["b"], "rel": p["rel"], "para_modes": modes,
                          "mB": mB, "mA_single": float(zA[p["tB"]] - zA[p["tA"]])})
            if len(items) == 1:
                stats["first_pair"] = (p, rA, zA)
            print("[k15p %s %s] %s idx %d kept %d/%d %.0fs reruns %d" % (M.name, ds, half, i, kept, n_target,
                                                                       time.time() - t0, stats["reruns"]), flush=True)
            if kept >= n_target:
                break
    return {"cells": cells, "items": items, "scanned": scanned, "stats": stats,
            "pairs_by_rel_half": {"%s:%s" % (it["half"], it["rel"]): sum(1 for x in items if x["half"] == it["half"] and x["rel"] == it["rel"]) for it in items}}


@torch.no_grad()
def scan_all(M, cands, bs=64):
    """Informational: how many candidates the clean model answers (right-padded batches; selection uses batch 1)."""
    ok = 0
    for s in range(0, len(cands), bs):
        ch = cands[s:s + bs]
        rows = [p["A"] for p in ch] + [p["B"] for p in ch]
        z = M.readout(M.run3(rows, record={M.final})[M.final], [len(x) - 1 for x in rows])
        top = z.argmax(-1).tolist()
        n = len(ch)
        ok += sum(1 for k, p in enumerate(ch) if top[k] == p["tA"] and top[n + k] == p["tB"])
    return {"pass": ok, "n": len(cands)}


@torch.no_grad()
def coda_checks(M, first):
    p, rA, zA = first
    last = len(p["A"]) - 1
    out = {"final_point": list(M.final), "final_is_coda": M.final[0] == M.K + 1 and M.e <= M.final[1] < M.nl,
           "coda_layers": M.nl - M.e}
    M.run3([p["A"]], record={M.final})
    out["readout_vs_runner_fwd_max_abs"] = float((M.readout(rA[M.final], [last])[0] - M.last_logits[0, -1].float()).abs().max())
    # native xllm forward vs the runner under the native init (this check only)
    try:
        x = torch.tensor([p["A"]], device=M.dev)
        if M.normal_init:
            del M.r._init_state
        try:
            lr, _ = M.r.fwd(x, M.K, cache=None, init="native")
            nat = M.m(x, False)
            nat = nat[0] if isinstance(nat, tuple) else nat
            out["native_forward_vs_runner_max_abs"] = float((nat[0, -1].float() - lr[0, -1].float()).abs().max())
            out["native_top1_equals_runner"] = int(nat[0, -1].argmax()) == int(lr[0, -1].argmax())
        finally:
            if M.normal_init:
                M.install_init()
    except Exception as ex:  # noqa: BLE001
        out["native_forward_error"] = repr(ex)
    return out


def run_model(name, out_dir, n_half):
    t0 = time.time()
    M = PairM(name)
    load_s = time.time() - t0
    res = {"model": name, "facts": M.facts(), "K": M.K, "load_seconds": round(load_s, 1), "datasets": {}}
    d1 = d1_candidates(M)
    res["d1_scan"] = scan_all(M, d1)
    print("[k15p %s] loaded %.0fs, D1 candidates %d, clean pass (padded scan) %s" % (
        name, load_s, len(d1), res["d1_scan"]), flush=True)
    q = lambda n: {"capital": n // 2, "city": n - n // 2}  # noqa: E731
    D = run_dataset(M, "D1", d1, [("heldout", 1, n_half), ("dev", 0, n_half)], q, 2, t0)
    first = D["stats"].pop("first_pair", None)
    res["datasets"]["D1"] = D
    if len(D["items"]) < MIN_D1:
        d2 = d2_candidates(M)
        res["d2_scan"] = scan_all(M, d2)
        print("[k15p %s] D1 kept %d < %d: D2 arithmetic, candidates %d, scan %s" % (
            name, len(D["items"]), MIN_D1, len(d2), res["d2_scan"]), flush=True)
        D2d = run_dataset(M, "D2", d2, [("all", None, N_D2 if n_half >= N_HALF else n_half)],
                          lambda n: {"arith": n}, 2, t0)
        f2 = D2d["stats"].pop("first_pair", None)
        first = first or f2
        res["datasets"]["D2"] = D2d
    res["coda_checks"] = coda_checks(M, first) if first else {"error": "no pair passed"}
    res["seconds_total"] = round(time.time() - t0, 1)
    res["dtype"] = str(next(M.m.parameters()).dtype)
    json.dump(res, open(os.path.join(out_dir, "k15p_%s.json" % name), "w"))
    print("[k15p %s] done %s total %.0fs" % (name, {k: len(v["items"]) for k, v in res["datasets"].items()},
                                             time.time() - t0), flush=True)
    del M
    gc.collect()
    torch.cuda.empty_cache()


def spent_minutes(out_dir):
    return sum(json.load(open(f))["seconds"] for f in glob.glob(os.path.join(out_dir, "proc_*.json"))) / 60.0


# ------------------------------------------------------------------ summary
def summarize(out_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "k15p_*.json"))):
        d = json.load(open(f))
        res[d["model"]] = d
    order = [m for m in MODELS if m in res]
    procs = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "proc_*.json"))}
    gpu_min = round(sum(p["seconds"] for p in procs.values()) / 60.0, 2)
    skipped = [m for p in procs.values() for m in p.get("skipped", [])]

    def r3(x):
        if x is None:
            return None
        if isinstance(x, list):
            return [r3(y) for y in x]
        return round(x, 3)

    def cel(m, ds, arm, a, site):
        return res[m]["datasets"].get(ds, {}).get("cells", {}).get(key(arm, a, site))

    def ci(m, ds, arm, a, site, f="effect", stat="mean"):
        c = cel(m, ds, arm, a, site)
        return None if c is None else r3(stat_ci(c[f], stat))

    main_ds = {m: ("D1" if len(res[m]["datasets"]["D1"]["items"]) >= MIN_D1 or "D2" not in res[m]["datasets"] else "D2")
               for m in order}
    S_by_loop, C_mid, C_bc, o5, o5l = {}, {}, {}, {}, {}
    for m in order:
        K = res[m]["K"]
        sites = [s[0] for s in res[m]["facts"]["sites"]]
        S_by_loop[m], C_mid[m], C_bc[m] = {}, {}, {}
        for ds in res[m]["datasets"]:
            S_by_loop[m][ds] = {s: {"O2_a1_effect": ci(m, ds, "S_O2", 1.0, s), "O2_a1_flip": ci(m, ds, "S_O2", 1.0, s, "flip"),
                                    "O4_a1_effect": ci(m, ds, "S_O4", 1.0, s), "O4_a2_flip": ci(m, ds, "S_O4", 2.0, s, "flip")}
                                for s in sites}
            for s in ("Llast@1", "L0@%d" % K):
                C_mid[m]["%s|%s" % (ds, s)] = ci(m, ds, "C", 0.5, s, stat="mid")
                C_bc[m]["%s|%s" % (ds, s)] = ci(m, ds, "C", 0.5, s, stat="bc")
        ds = main_ds[m]
        o5[m] = ci(m, ds, "O5", 1.0, "Llast@K_allpos")
        o5l[m] = ci(m, ds, "O5_last", 1.0, "final_lastpos")
    coda_used = {}
    for m in order:
        cc = dict(res[m].get("coda_checks", {}))
        K = res[m]["K"]
        e = ci(m, main_ds[m], "S_O2", 1.0, "Llast@%d" % K)
        cc["S_O2_a1_LlastK_effect"] = e
        cc["coda_readout_used"] = bool(cc.get("final_is_coda") and (cc.get("readout_vs_runner_fwd_max_abs") is not None and cc["readout_vs_runner_fwd_max_abs"] <= 1e-3)
                                       and e is not None and abs(e[0]) > 0)
        coda_used[m] = cc
    checks = {
        "model_facts": {m: {k: res[m]["facts"][k] for k in ("checkpoint", "class", "loops", "prelude_layers",
                                                            "recurrent_layers", "coda_layers", "state_init", "init_used",
                                                            "injection", "sampling_scheme", "backprop_depth", "source")}
                        for m in order},
        "injection_decay": {m: (lambda s: {k: v.get("decay_exp_minus_dt_a") for k, v in s.get("modules", {}).items()})(
            res[m]["facts"]["injection_summary"]) for m in order},
        "pairs_per_model": {m: {ds: len(v["items"]) for ds, v in res[m]["datasets"].items()} for m in order},
        "d1_clean_pass_scan": {m: res[m]["d1_scan"] for m in order},
        "main_dataset": main_ds,
        "coda_readout_used": coda_used,
        "o5_effect_per_model": o5,
        "o5_last_effect_per_model": o5l,
        "S_effect_by_loop": S_by_loop,
        "C_mid_fraction": C_mid,
        "C_bimodality": C_bc,
        "C_mean_effect_a05": {m: {"%s|%s" % (ds, s): ci(m, ds, "C", 0.5, s) for ds in res[m]["datasets"]
                                  for s in ("Llast@1", "L0@%d" % res[m]["K"])} for m in order},
        "rows_rerun_batch1": {m: {ds: "%d/%d" % (v["stats"]["reruns"], v["stats"]["rows"]) for ds, v in res[m]["datasets"].items()}
                              for m in order},
        "clean_batch_vs_single_max_logit_gap": {m: {ds: r3(v["stats"]["gap"]) for ds, v in res[m]["datasets"].items()} for m in order},
        "ci": "95%% percentile bootstrap over items, %d resamples, numpy seed 0; values [point, lo, hi]" % N_BOOT,
        "downloads": [],
        "gpu_minutes": gpu_min,
        "models_skipped": skipped,
    }
    json.dump({"models": res, "processes": procs}, open(os.path.join(out_dir, "results.json"), "w"))
    json.dump(checks, open(os.path.join(out_dir, "checks.json"), "w"), indent=1)

    def f2(x):
        return "-" if x is None else "%.2f" % x

    def fci(c):
        return "-" if c is None else ("%.2f [%s, %s]" % (c[0], f2(c[1]), f2(c[2])))
    L = ["# K15 FULL-B summary (matched recipe pairs)", "",
         "Effect e = (m_edit - m_A)/(m_B - m_A), m = logit tB - logit tA at the CODA readout (m.output of the last coda "
         "layer); flip = top-1 is tB. Rows with top-1 margin < %.1f logit rerun alone at batch 1. CI: %s." % (TAU, checks["ci"]),
         "", "GPU minutes: %s. Skipped: %s." % (gpu_min, ", ".join(skipped) or "none"), "",
         "## Model facts", "", "| model | class | prelude | recurrent | coda | loops | state init | injection | depth sampling (backprop) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        f = res[m]["facts"]
        rng = lambda xs: ("%d-%d" % (xs[0], xs[-1])) if xs else "-"  # noqa: E731
        L.append("| %s | %s | %s | %s | %s | %d | %s | %s | %s (%s) |" % (m, f["class"], rng(f["prelude_layers"]),
                 rng(f["recurrent_layers"]), rng(f["coda_layers"]), f["loops"], f["state_init"], f["injection"],
                 f["sampling_scheme"], f["backprop_depth"]))
    L.append("")
    for m in order:
        d = res[m]
        K = d["K"]
        sites = [s[0] for s in d["facts"]["sites"]]
        L += ["## %s" % m, "",
              "Sites %s. Init: %s. D1 clean pass (padded scan): %s. Coda checks: %s. Injection decay: %s." % (
                  d["facts"]["sites"], d["facts"]["init_used"], d["d1_scan"], coda_used[m],
                  checks["injection_decay"][m]), "",
              "O5 (whole loop-%d state = clean B, then coda) effect %s; O5_last %s." % (K, fci(o5[m]), fci(o5l[m])), ""]
        for ds, D in d["datasets"].items():
            L += ["### %s: %d pairs (%s), scanned %s, reruns %d/%d, clean gap %.3f" % (
                ds, len(D["items"]), D["pairs_by_rel_half"], D["scanned"], D["stats"]["reruns"], D["stats"]["rows"],
                D["stats"]["gap"]), ""]
            for arm, title in (("S_O2", "S, O2 paraphrase vector at the subject positions"), ("S_O4", "S, O4 random direction, same norm")):
                L += ["%s: effect [95%% CI] / flips" % title, "", "| a | " + " | ".join(sites) + " |", "|---" * (len(sites) + 1) + "|"]
                for a in A_S:
                    L.append("| %s | " % fa(a) + " | ".join("%s / %s" % (fci(ci(m, ds, arm, a, s)),
                                                                     (lambda c: f2(c[0]) if c else "-")(ci(m, ds, arm, a, s, "flip")))
                                                           for s in sites) + " |")
                L.append("")
            L += ["S O2 change in logit A / logit B (a=1): " + ", ".join(
                "%s %s/%s" % (s, f2((lambda c: c and sum(c["dA"]) / len(c["dA"]))(cel(m, ds, "S_O2", 1.0, s))),
                              f2((lambda c: c and sum(c["dB"]) / len(c["dB"]))(cel(m, ds, "S_O2", 1.0, s)))) for s in sites), ""]
            for s in ("Llast@1", "L0@%d" % K):
                L += ["C, O1 at %s (subject positions): dose curve" % s, "",
                      "| a | effect [CI] | flips | mid fraction [CI] | BC [CI] |", "|---|---|---|---|---|"]
                for a in A_C:
                    L.append("| %s | %s | %s | %s | %s |" % (fa(a), fci(ci(m, ds, "C", a, s)),
                                                           (lambda c: f2(c[0]) if c else "-")(ci(m, ds, "C", a, s, "flip")),
                                                           fci(ci(m, ds, "C", a, s, stat="mid")), fci(ci(m, ds, "C", a, s, stat="bc"))))
                L.append("")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L))
    print(json.dumps({k: checks[k] for k in ("pairs_per_model", "o5_effect_per_model", "gpu_minutes")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="p")
    ap.add_argument("--n_half", type=int, default=N_HALF)  # debug only
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
        return
    t0 = time.time()
    skipped = []
    try:
        prior = spent_minutes(a.out)
        for name in [x for x in a.models.split(",") if x]:
            used = prior + (time.time() - t0) / 60.0
            if used > STOP_MIN:
                print("[k15p] skip %s: %.1f GPU-min used" % (name, used), flush=True)
                skipped.append(name)
                continue
            run_model(name, a.out, a.n_half)
    finally:
        json.dump({"models": a.models, "seconds": round(time.time() - t0, 1), "skipped": skipped},
                  open(os.path.join(a.out, "proc_%s.json" % a.tag), "w"))


if __name__ == "__main__":
    main()
