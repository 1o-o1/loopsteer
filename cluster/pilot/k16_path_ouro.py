"""K16 pathway tests on Ouro (brief B-K16-PATH-OURO). Plan: loop-signal/notes/k16_path_ouro_plan.md.
Builds on k15_smoke / k15_smoke2 / k15_full / k15_data (imported, never modified).

Ouro is loaded with eager attention: under SDPA the causal mask is None for unpadded inputs, so there is no mask to
edit (critic B5); eager always materializes a 4D additive mask and returns attention probabilities.
Hooks (names by tensor): H1 = model.norm output of loop j (Llast@j, post-norm carried state; edits, restores, records,
readout lm_head(H_j[last])); H2 = layers[L//2] block output (Lmid@j, T3 only); H3 = instance wrapper on every
layers[l].self_attn.forward keyed on (current_ut + 1, layer_idx): mask' = mask + K, K = finfo.min at
[query = last, key in set]; H4 = zero grad leaves added to H_j (Y4 only).
Every scored row runs at batch 1.

Arms per item: Y1 2x2 (edit {off, ans_O1, ans_para, src_O1} at Llast@1 x 10 knockout conditions in loops 2..K),
Y3 undo (src_O1 then clean source restored at Llast@2 or Llast@3), width control (test half), Y4 linear account,
W2 retrieval window (attention mass and single-call knockouts). T3: sentiment CAA at Lmid, per-loop log-odds.

    python k16_path_ouro.py --selftest
    python k16_path_ouro.py --cpu_check --models ouro --out OUT          (CPU, real model: hook checks)
    python k16_path_ouro.py --models ouro --out OUT [--debug]            (GPU)
    python k16_path_ouro.py --summarize --out OUT
"""
import argparse
import gc
import glob
import hashlib
import json
import math
import os
import random
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import k15_data as KD  # noqa: E402
import k15_full as KF  # noqa: E402
import k15_smoke as S1  # noqa: E402
import k15_smoke2 as S2  # noqa: E402

# Hub revisions pinned for the cluster run (values from cluster/weights.json); a local folder path gets None
OURO_REVISION = {"ByteDance/Ouro-1.4B": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
                 "ByteDance/Ouro-2.6B": "1ed04250da1a9936042725d302e81c8fa2ab5abd",
                 "ByteDance/Ouro-1.4B-Thinking": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
                 "ByteDance/Ouro-2.6B-Thinking": "f1edd81e7ac41355db670500ceaf204e0f73af68"}
RUN_ROLE = os.environ.get("RUN_ROLE", "cluster")   # written into records instead of the host name
FULL_POOL = [False]     # cluster copy (--full_pool): lift the per-dataset quotas; every candidate is scanned

REPOS = {"ouro": "ByteDance/Ouro-1.4B", "ouro26": "ByteDance/Ouro-2.6B", "ouro_think": "ByteDance/Ouro-1.4B-Thinking"}
EDITS = ["off", "ans_O1", "ans_para", "src_O1"]
KOS = ["none", "src_all", "src_l2", "src_lo", "src_hi", "rand0", "rand1", "rand2", "nonsrc", "allbut"]
A_PARA = 2.0
GAP_E = 2.0
NBOOT = 1000
W2_MAX = {"ouro": 10 ** 9, "ouro26": 40, "ouro_think": 10 ** 9}
ARMS = {"width", "y4", "w2"}       # optional arms per item (Y1 and Y3 always run); --arms overrides
MIN_ITEMS = 0                       # datasets with fewer kept pairs are scanned only (amendment A1)
MODEL_DIR = {}                      # name -> local snapshot dir (--model_dir), else the hub id from the HF cache
POS_W = [" good", " great", " wonderful", " excellent", " amazing", " fantastic", " beautiful", " happy", " nice",
         " lovely"]
NEG_W = [" bad", " terrible", " awful", " horrible", " poor", " worst", " sad", " ugly", " boring", " disappointing"]
T3_PROMPTS = KD.NEUTRAL150[100:150]
T3_CONDS = [("clean", [], 0.0, "caa"), ("L1", [1], 0.1, "caa"), ("L1", [1], 0.4, "caa"),
            ("all", [1, 2, 3, 4], 0.1, "caa"), ("L1", [1], 0.4, "random"), ("all", [1, 2, 3, 4], 0.1, "random")]


def r5(x):
    return None if x is None else round(float(x), 5)


# ------------------------------------------------------------------ pure helpers (CPU-testable)
def e_lerp(pos, tgt, a, dev):
    P = len(pos)
    return ("lerp", torch.zeros(P, dtype=torch.long, device=dev), torch.tensor(pos, device=dev), tgt.float(),
            torch.full((P, 1), float(a), device=dev))


def e_add(pos, delta, dev):
    P = len(pos)
    return ("add", torch.zeros(P, dtype=torch.long, device=dev), torch.tensor(pos, device=dev), delta.float())


def e_add_all(delta, dev):
    return ("add_all", torch.zeros(1, dtype=torch.long, device=dev), delta.float()[None])


def proj_rescale(D, U):
    """D [P, H] rows projected onto span(U) (U [H, k] orthonormal columns), each rescaled to its own norm."""
    Pj = (D @ U) @ U.T
    return Pj * (D.norm(dim=-1, keepdim=True) / Pj.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def ko_conditions(T, src, idx, L, K=4):
    """Knockout conditions (query = last). Returns ({name: (calls, keys) or None}, {name: keys})."""
    last = T - 1
    later = list(range(2, K + 1))
    c_all = {(j, l) for j in later for l in range(L)}
    c_l2 = {(2, l) for l in range(L)}
    c_lo = {(j, l) for j in later for l in range(L // 2)}
    c_hi = {(j, l) for j in later for l in range(L // 2, L)}
    src = sorted(src)
    out = {"none": None, "src_all": (c_all, src), "src_l2": (c_l2, src), "src_lo": (c_lo, src), "src_hi": (c_hi, src)}
    cand = [k for k in range(1, last) if k not in src]
    for s in range(3):
        if len(cand) >= len(src):
            out["rand%d" % s] = (c_all, sorted(random.Random(1000 * s + idx).sample(cand, len(src))))
    out["nonsrc"] = (c_all, [k for k in range(0, last) if k not in src])
    out["allbut"] = (c_all, list(range(0, last)))
    meta = {k: (None if v is None else v[1]) for k, v in out.items()}
    return out, meta


def vals(z, tA, tB):
    """z [K, V] -> per loop [z(A), z(B), logsumexp, top-1]."""
    lse = torch.logsumexp(z.float(), -1)
    top = z.argmax(-1)
    return [[r5(z[j, tA]), r5(z[j, tB]), r5(lse[j]), int(top[j])] for j in range(z.shape[0])]


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    pr = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pr) < 3:
        return None
    rx, ry = rankdata([a for a, _ in pr]), rankdata([b for _, b in pr])
    n = len(pr)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else round(sxy / (sx * sy), 4)


def boot_cluster(xs, cl, n=NBOOT, seed=0):
    """Mean over items with a 95% percentile bootstrap over clusters (1000 resamples, seed 0). [v, lo, hi, n_items]."""
    groups = {}
    for x, c in zip(xs, cl):
        if x is not None:
            groups.setdefault(c, []).append(float(x))
    if not groups:
        return None
    keys = sorted(groups)
    sums = torch.tensor([sum(groups[k]) for k in keys], dtype=torch.float64)
    cnts = torch.tensor([len(groups[k]) for k in keys], dtype=torch.float64)
    v0 = float(sums.sum() / cnts.sum())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=g)
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    q = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [round(v0, 4), round(float(q[0]), 4), round(float(q[1]), 4), int(cnts.sum())]


def selftest():
    res = {}
    b = boot_cluster([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"])
    res["boot_value_is_item_mean"] = abs(b[0] - 4.0) < 1e-9 and b[3] == 4
    res["boot_deterministic"] = boot_cluster([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"]) == b
    res["spearman"] = [spearman([1, 2, 3, 4], [10, 20, 30, 40]), spearman([1, 2, 3, 4], [4, 3, 2, 1]),
                       spearman([1, 1, 2, 3], [1, 2, 3, 4])]
    kos, meta = ko_conditions(5, [3], 7, 24)
    res["ko_capital"] = (all(set(meta["rand%d" % s]) <= {1, 2} for s in range(3)) and meta["nonsrc"] == [0, 1, 2]
                         and meta["allbut"] == [0, 1, 2, 3] and len(kos["src_all"][0]) == 72
                         and len(kos["src_l2"][0]) == 24 and not (kos["src_lo"][0] & kos["src_hi"][0])
                         and (kos["src_lo"][0] | kos["src_hi"][0]) == kos["src_all"][0]
                         and all(j >= 2 for j, _ in kos["src_all"][0]))
    kos2, meta2 = ko_conditions(9, [0, 1], 3, 24)
    res["ko_city"] = (meta2["nonsrc"] == [2, 3, 4, 5, 6, 7] and all(set(meta2["rand%d" % s]) <= set(range(2, 8))
                                                                    and len(meta2["rand%d" % s]) == 2 for s in range(3)))
    torch.manual_seed(0)
    D = torch.randn(3, 16)
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    res["proj_norm_err"] = float((proj_rescale(D, Q[:, :2]).norm(dim=-1) - D.norm(dim=-1)).abs().max())
    res["proj_full_err"] = float((proj_rescale(D, Q) - D).abs().max())
    o = torch.randn(1, 6, 16).to(torch.bfloat16)
    t = torch.randn(2, 16)
    e = S2.apply_entries(o, [e_lerp([1, 4], t, 1.0, "cpu")])
    res["lerp_a1_exact"] = bool(torch.equal(e[0, [1, 4]], t.to(o.dtype))) and bool(torch.equal(e[0, [0, 2, 3, 5]], o[0, [0, 2, 3, 5]]))
    res["add_zero_identity"] = bool(torch.equal(S2.apply_entries(o, [e_add([2], torch.zeros(1, 16), "cpu")]), o))
    res["add_all_zero_identity"] = bool(torch.equal(S2.apply_entries(o, [e_add_all(torch.zeros(16), "cpu")]), o))
    # additive knockout on an eager-style masked softmax (bf16 mask with finfo.min, float32 softmax)
    T = 6
    sc = torch.randn(1, 2, T, T).to(torch.bfloat16)
    causal = torch.full((T, T), torch.finfo(torch.bfloat16).min).triu(1).to(torch.bfloat16)[None, None]
    Kt = torch.zeros_like(causal)
    Kt[:, :, T - 1, [2, 3]] = torch.finfo(torch.bfloat16).min
    p0 = torch.softmax((sc + causal).float(), -1)
    p1 = torch.softmax((sc + causal + Kt).float(), -1)
    p2 = torch.softmax((sc + (causal + torch.zeros_like(causal))).float(), -1)
    res["toy_knockout_zero"] = bool((p1[0, :, T - 1, [2, 3]] == 0).all()) and bool(torch.equal(p1[0, :, :T - 1], p0[0, :, :T - 1]))
    res["toy_zero_mask_identity"] = bool(torch.equal(p0, p2))
    ok = (res["boot_value_is_item_mean"] and res["boot_deterministic"] and res["spearman"][0] == 1.0
          and res["spearman"][1] == -1.0 and res["spearman"][2] is not None and res["ko_capital"] and res["ko_city"]
          and res["proj_norm_err"] < 1e-4 and res["proj_full_err"] < 1e-4 and res["lerp_a1_exact"]
          and res["add_zero_identity"] and res["add_all_zero_identity"] and res["toy_knockout_zero"]
          and res["toy_zero_mask_identity"])
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


# ------------------------------------------------------------------ model
def load_ouro(repo, dev):
    """k15_smoke.OuroM around an Ouro checkpoint loaded with eager attention on `dev` (k15_full.load_ouro pattern)."""
    import k13_edit
    orig = k13_edit.OuroEditor

    class Ed(orig):
        def __init__(self):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, revision=OURO_REVISION.get(repo))
            self.m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                          revision=OURO_REVISION.get(repo),
                                                          attn_implementation="eager").to(dev).eval()
            self.K = 4
            self.W = self.m.lm_head.weight
    k13_edit.OuroEditor = Ed
    try:
        M = S1.OuroM()
    finally:
        k13_edit.OuroEditor = orig
    M.dev = dev
    M.repo = repo
    for prm in M.m.parameters():
        prm.requires_grad_(False)
    return M


class Runner(object):
    def __init__(self, M):
        self.M, self.m = M, M.m
        self.inner = M.m.model
        self.L = int(M.m.config.num_hidden_layers)
        self.K = int(self.inner.total_ut_steps)
        self.mid = self.L // 2
        self.W = M.W
        self.dev = M.dev
        self.Hd = int(M.m.config.hidden_size)
        self.attn = [ly.self_attn for ly in self.inner.layers[:self.L]]
        self.n_fwd = 0

    def run(self, ids, edits=None, lmid=None, ko=None, rec=False, capture=None, zero_k=False, layers=False,
            grad_leaves=None, wrap=True, mask_log=None):
        T = len(ids)
        last = T - 1
        x = torch.tensor([ids], device=self.dev)
        edits, lmid = edits or {}, lmid or {}
        ko_calls, ko_keys = ko if ko is not None else (set(), [])
        st = {"norm": 0, "mid": 0}
        calls, H, hl, probs, lay, lcnt = {}, {}, {}, {}, {}, {}

        def norm_hook(mod, i, o):
            st["norm"] += 1
            j = st["norm"]
            if grad_leaves is not None:
                o = o + grad_leaves[j - 1]
            if j in edits:
                o = S2.apply_entries(o, edits[j])
            if rec:
                H[j] = o[0].detach().float().clone()
            hl[j] = o[0, last]
            return o

        def mid_hook(mod, i, o):
            st["mid"] += 1
            if st["mid"] in lmid:
                o = S2.apply_entries(o, lmid[st["mid"]])
            return o

        def mk_layer_hook(l):
            def h(mod, i, o):
                c = lcnt.get(l, 0) + 1
                lcnt[l] = c
                lay[(c, l)] = o.detach().clone()
            return h
        hs = [self.inner.norm.register_forward_hook(norm_hook),
              self.inner.layers[self.mid].register_forward_hook(mid_hook)]
        if layers:
            hs += [self.inner.layers[l].register_forward_hook(mk_layer_hook(l)) for l in range(self.L)]
        if wrap:
            for l, att in enumerate(self.attn):
                def fwd(*args, _o=att.forward, _l=l, **kw):
                    key = (int(kw.get("current_ut", 0)) + 1, _l)
                    calls[key] = calls.get(key, 0) + 1
                    am = kw.get("attention_mask")
                    if mask_log is not None:
                        mask_log.append(None if am is None else [list(am.shape), str(am.dtype)])
                    if key in ko_calls:
                        Kt = torch.zeros_like(am)
                        Kt[:, :, last, ko_keys] = torch.finfo(am.dtype).min
                        kw["attention_mask"] = am + Kt
                    elif zero_k:
                        kw["attention_mask"] = am + torch.zeros_like(am)
                    out = _o(*args, **kw)
                    if capture is not None and key in capture:
                        probs[key] = out[1][0, :, last].detach().float().clone()
                    return out
                att.forward = fwd
        try:
            with torch.set_grad_enabled(grad_leaves is not None):
                self.m(input_ids=x, use_cache=False)
                z = torch.stack([F.linear(hl[j], self.W).float() for j in range(1, self.K + 1)])
        finally:
            for h in hs:
                h.remove()
            if wrap:
                for att in self.attn:
                    att.__dict__.pop("forward", None)
        self.n_fwd += 1
        return {"z": z, "H": H, "probs": probs, "layers": lay, "calls": calls, "norm_calls": st["norm"],
                "mid_calls": st["mid"]}

    def grads(self, ids, tA, tB):
        leaves = [torch.zeros(1, len(ids), self.Hd, dtype=self.W.dtype, device=self.dev, requires_grad=True)
                  for _ in range(self.K)]
        out = self.run(ids, grad_leaves=leaves)
        (out["z"][-1, tB] - out["z"][-1, tA]).backward()
        return [lv.grad[0].float() for lv in leaves], out["z"].detach()


# ------------------------------------------------------------------ hook checks (critic B4/B5)
@torch.no_grad()
def model_checks(R, cap, city):
    ids, src, tA, tB = cap["A"], list(cap["pos"]), cap["tA"], cap["tB"]
    T = len(ids)
    last = T - 1
    dev, L, K, mid = R.dev, R.L, R.K, R.mid
    res = {}
    base = R.run(ids, wrap=False)
    w0 = R.run(ids)
    res["call_counts_ok"] = bool(len(w0["calls"]) == K * L and all(v == 1 for v in w0["calls"].values())
                                 and sorted(w0["calls"]) == [(j, l) for j in range(1, K + 1) for l in range(L)]
                                 and w0["norm_calls"] == K and w0["mid_calls"] == K)
    res["wrapper_no_knockout_bit_identical"] = bool(torch.equal(base["z"], w0["z"]))
    res["all_zero_additive_mask_bit_identical"] = bool(torch.equal(base["z"], R.run(ids, zero_k=True)["z"]))
    z0 = R.run(ids, edits={1: [e_add(src, torch.zeros(len(src), R.Hd, device=dev), dev)]},
               lmid={1: [e_add_all(torch.zeros(R.Hd, device=dev), dev)]})["z"]
    res["zero_edit_norm_and_lmid_bit_identical"] = bool(torch.equal(base["z"], z0))
    res["rec_run_bit_identical"] = bool(torch.equal(base["z"], R.run(ids, rec=True)["z"]))
    allc = {(j, l) for j in range(1, K + 1) for l in range(L)}
    res["capture_run_bit_identical"] = bool(torch.equal(base["z"], R.run(ids, capture=allc)["z"]))
    with torch.enable_grad():
        _, zg = R.grads(ids, tA, tB)
    res["grad_leaf_run_bit_identical"] = bool(torch.equal(base["z"], zg))
    res["repeat_run_bit_identical"] = bool(torch.equal(base["z"], R.run(ids, wrap=False)["z"]))
    log = []
    R.run(ids, mask_log=log)
    res["eager_mask"] = {"n_calls": len(log), "none_calls": sum(1 for x in log if x is None),
                         "shape_dtype": log[0] if log else None}
    # knockout placement: single knocked call (2, mid), keys = source
    order = [(j, l) for j in range(1, K + 1) for l in range(L)]
    kc = (2, mid)
    cl = R.run(ids, layers=True)["layers"]
    ko = R.run(ids, ko=({kc}, src), layers=True, capture={kc})
    i0 = order.index(kc)
    P = ko["probs"][kc]
    kp = {"call": list(kc), "keys": src,
          "before_call_all_positions_identical": all(torch.equal(cl[c], ko["layers"][c]) for c in order[:i0]),
          "from_call_positions_0_to_T-2_identical": all(torch.equal(cl[c][:, :last], ko["layers"][c][:, :last])
                                                         for c in order[i0:]),
          "last_position_changed_at_call": not torch.equal(cl[kc][:, last], ko["layers"][kc][:, last]),
          "knocked_probs_exactly_zero": bool((P[:, src] == 0).all()),
          "row_sum_max_abs_err": r5((P.sum(-1) - 1).abs().max()),
          "final_logits_changed": not torch.equal(base["z"], ko["z"])}
    kp["ok"] = bool(kp["before_call_all_positions_identical"] and kp["from_call_positions_0_to_T-2_identical"]
                    and kp["last_position_changed_at_call"] and kp["knocked_probs_exactly_zero"]
                    and kp["row_sum_max_abs_err"] < 0.02)
    res["knockout_placement"] = kp
    # H1 edit placement: add at Llast@1, source position p
    p = src[0]
    g = torch.Generator().manual_seed(0)
    d = torch.randn(1, R.Hd, generator=g).to(dev)
    ed = R.run(ids, edits={1: [e_add([p], d, dev)]}, layers=True)["layers"]
    ep = {"position": p,
          "loop1_all_identical": all(torch.equal(cl[c], ed[c]) for c in order if c[0] == 1),
          "later_positions_before_p_identical": all(torch.equal(cl[c][:, :p], ed[c][:, :p]) for c in order if c[0] > 1),
          "position_p_changed_loop2_layer0": not torch.equal(cl[(2, 0)][:, p], ed[(2, 0)][:, p])}
    ep["ok"] = bool(ep["loop1_all_identical"] and ep["later_positions_before_p_identical"]
                    and ep["position_p_changed_loop2_layer0"])
    res["norm_edit_placement"] = ep
    ld = R.run(ids, lmid={1: [e_add_all(d[0], dev)]}, layers=True)["layers"]
    im = order.index((1, mid))
    lp = {"before_site_identical": all(torch.equal(cl[c], ld[c]) for c in order[:im]),
          "site_all_positions_changed": all(not torch.equal(cl[(1, mid)][0, t], ld[(1, mid)][0, t]) for t in range(T))}
    lp["ok"] = bool(lp["before_site_identical"] and lp["site_all_positions_changed"])
    res["lmid_edit_placement"] = lp
    # city prompt: knockout set that includes position 0 (allbut) stays finite
    kos, _ = ko_conditions(len(city["A"]), city["pos"], 0, L, K)
    zc = R.run(city["A"], ko=kos["allbut"])["z"]
    res["allbut_knockout_finite"] = bool(torch.isfinite(zc).all())
    res["hook_identity_ok"] = bool(res["call_counts_ok"] and res["wrapper_no_knockout_bit_identical"]
                                   and res["all_zero_additive_mask_bit_identical"]
                                   and res["zero_edit_norm_and_lmid_bit_identical"] and res["rec_run_bit_identical"]
                                   and res["capture_run_bit_identical"] and res["grad_leaf_run_bit_identical"]
                                   and res["repeat_run_bit_identical"] and res["eager_mask"]["none_calls"] == 0)
    res["all_ok"] = bool(res["hook_identity_ok"] and kp["ok"] and ep["ok"] and lp["ok"] and res["allbut_knockout_finite"])
    return res


@torch.no_grad()
def sdpa_gap(R, prompts):
    """Clean loop-end logits under eager vs SDPA (FULL-A ran SDPA), and whether SDPA passes mask None."""
    cfg = R.m.config
    prev = cfg._attn_implementation
    ze = [R.run(ids, wrap=False)["z"] for ids in prompts]
    out = {}
    try:
        cfg._attn_implementation = "sdpa"
        log = []
        zs = [R.run(ids, mask_log=log)["z"] for ids in prompts]
        out = {"n": len(prompts), "impl_during": cfg._attn_implementation,
               "max_abs_gap_final": r5(max(float((a[-1] - b[-1]).abs().max()) for a, b in zip(ze, zs))),
               "max_abs_gap_any_loop": r5(max(float((a - b).abs().max()) for a, b in zip(ze, zs))),
               "top1_final_agree": sum(int(a[-1].argmax() == b[-1].argmax()) for a, b in zip(ze, zs)),
               "sdpa_mask_none_calls": sum(1 for x in log if x is None), "sdpa_calls": len(log)}
    except Exception as e:  # noqa: BLE001
        out = {"error": repr(e)}
    try:
        cfg._attn_implementation = prev
    except Exception as e:  # noqa: BLE001
        out["restore_error"] = repr(e)
    out["impl_after"] = cfg._attn_implementation
    out["restored_eager_bit_identical"] = bool(torch.equal(R.run(prompts[0], wrap=False)["z"], ze[0]))
    return out


# ------------------------------------------------------------------ one item
@torch.no_grad()
def process_item(R, p, idx, dirs, do_w2):
    dev = R.dev
    A, tA, tB, src = p["A"], p["tA"], p["tB"], sorted(p["pos"])
    T = len(A)
    last = T - 1
    a_out, b_out = p["_a"], p["_b"]
    HA = [a_out["H"][j] for j in range(1, R.K + 1)]
    HB = [b_out["H"][j] for j in range(1, R.K + 1)]
    it = {"idx": idx, "rel": p["rel"], "a": p["a"], "b": p["b"], "half": p["half"], "split": p["split"],
          "cluster": p["cluster"], "pos": src, "T": T, "tA": tA, "tB": tB,
          "clean": vals(a_out["z"], tA, tB), "cleanB": vals(b_out["z"], tA, tB), "rows": {}}
    _, vlast, modes = KF.para_vectors(R.M, p, [(1, "norm")])
    it["para_modes"] = modes
    dlast = HB[0][last] - HA[0][last]
    edits = {"off": {}, "ans_O1": {1: [e_lerp([last], HB[0][[last]], 1.0, dev)]},
             "src_O1": {1: [e_lerp(src, HB[0][src], 1.0, dev)]}}
    if vlast is not None:
        v = vlast[(1, "norm")]
        edits["ans_para"] = {1: [e_add([last], A_PARA * v[None], dev)]}
        it["para_vec"] = {"norm": r5(v.norm()), "o1_delta_norm": r5(dlast.norm()),
                          "cos_with_o1_delta": r5(F.cosine_similarity(v, dlast, dim=0))}
    kos, it["ko_keys"] = ko_conditions(T, src, idx, R.L, R.K)
    st_none = {}
    for en in EDITS:
        if en not in edits:
            continue
        for kn in KOS:
            if kn not in kos:
                continue
            key = "Y1|%s|%s" % (en, kn)
            if en == "off" and kn == "none":
                it["rows"][key] = it["clean"]
                continue
            o = R.run(A, edits=edits[en], ko=kos[kn], rec=(kn == "none"))
            it["rows"][key] = vals(o["z"], tA, tB)
            if kn == "none":
                st_none[en] = o["H"]
    for jr in (2, 3):
        ed = {1: [e_lerp(src, HB[0][src], 1.0, dev)], jr: [e_lerp(src, HA[jr - 1][src], 1.0, dev)]}
        it["rows"]["Y3|undo%d" % jr] = vals(R.run(A, edits=ed)["z"], tA, tB)
    if "width" in ARMS and p["split"] == "test" and dirs is not None:
        it["rows"]["W|ans_top1"] = vals(R.run(A, edits={1: [e_add([last], proj_rescale(dlast[None], dirs["ans"][:, :1]),
                                                                  dev)]})["z"], tA, tB)
        dsrc = HB[0][src] - HA[0][src]
        for k in (1, 10):
            kk = min(k, dirs["src"].shape[1])
            it["rows"]["W|src_top%d" % k] = vals(R.run(A, edits={1: [e_add(src, proj_rescale(dsrc, dirs["src"][:, :kk]),
                                                                           dev)]})["z"], tA, tB)
    if "y4" in ARMS:
        it["y4"] = y4_account(R, A, tA, tB, src, last, a_out, HA, st_none)
    if do_w2:
        w2_window(R, A, tA, tB, src, a_out, it)
    return it


def y4_account(R, A, tA, tB, src, last, a_out, HA, st_none):
    with torch.enable_grad():
        g, zg = R.grads(A, tA, tB)
    y4 = {"grad_run_bit_identical": bool(torch.equal(zg, a_out["z"]))}
    S = sorted(set(src) | {last})
    for en in ("ans_O1", "ans_para", "src_O1"):
        if en not in st_none:
            continue
        Hs = st_none[en]
        epos = [last] if en.startswith("ans") else src
        d1 = Hs[1] - HA[0]
        r = {"lin": [], "lin_all": [], "n_src": [], "n_last": [], "cos": []}
        for j in range(1, R.K + 1):
            dH = Hs[j] - HA[j - 1]
            r["lin"].append(r5((g[j - 1][S] * dH[S]).sum()))
            r["lin_all"].append(r5((g[j - 1] * dH).sum()))
            r["n_src"].append(r5(dH[src].norm()))
            r["n_last"].append(r5(dH[last].norm()))
            r["cos"].append(r5(F.cosine_similarity(dH[epos].flatten(), d1[epos].flatten(), dim=0)))
        y4[en] = r
    return y4


@torch.no_grad()
def w2_window(R, A, tA, tB, src, a_out, it):
    if True:
        allc = [(j, l) for j in range(1, R.K + 1) for l in range(R.L)]
        cp = R.run(A, capture=set(allc))
        it["w2_capture_bit_identical"] = bool(torch.equal(cp["z"], a_out["z"]))
        it["w2_mass"] = [[r5(cp["probs"][(j, l)][:, src].sum(-1).mean()) for l in range(R.L)] for j in range(1, R.K + 1)]
        it["w2_mass0"] = [[r5(cp["probs"][(j, l)][:, 0].mean()) for l in range(R.L)] for j in range(1, R.K + 1)]
        w = []
        for j in range(1, R.K + 1):
            row = []
            for l in range(R.L):
                z = R.run(A, ko=({(j, l)}, src))["z"][-1]
                row.append([r5(z[tA]), r5(z[tB]), int(z.argmax())])
            w.append(row)
        it["w2_ko"] = w
    return it


# ------------------------------------------------------------------ datasets
DATA_DIR = (os.path.join(os.environ["DATA_ROOT"], "k16_path_ouro") if os.environ.get("DATA_ROOT")
            else os.path.join(HERE, "..", "out", "k16_path_ouro", "data"))   # cluster copy: $DATA_ROOT/k16_path_ouro
EXT_QUOTA = {"D4": {("dev", "T1"): 20, ("test", "T1"): 20, ("dev", "T2"): 20, ("test", "T2"): 20},
             "D5": {("dev", "T2"): 40, ("test", "T2"): 40}}


def ext_candidates(ds, tok_name):
    """B-K16-DATA JSONL rows (data/d4|d5/<ds>_<tok>.jsonl, copied to DATA_DIR) -> k15-style pair dicts.
    a, b carry the cluster id so k15_data.select's cap of 2 applies per cluster (each cluster has <= 3 pairs)."""
    path = os.path.join(DATA_DIR, "%s_%s.jsonl" % (ds.lower(), tok_name))
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    out = []
    for r in rows:
        out.append({"A": r["ids_A"], "B": r["ids_B"], "pos": r["source_span"], "tA": r["answer_first_token_A"],
                    "tB": r["answer_first_token_B"], "a": r["cluster_id"] + ":A", "b": r["cluster_id"] + ":B",
                    "rel": r["kind"], "half": r["split"], "suffix": r["suffix_ids"],
                    "para": [(q["ids_A"], q["ids_B"]) for q in r["para"]], "id": r["id"],
                    "cluster_id": r["cluster_id"], "answers": [r["answer_A"], r["answer_B"]]})
    return out


def candidates(M, ds):
    if ds == "D1":
        return KD.d1_candidates(M.enc), dict(KD.D1_QUOTA)
    if ds == "D2":
        return KD.d2_candidates(M.enc), dict(KD.D2_QUOTA)
    if ds in EXT_QUOTA:
        return ext_candidates(ds, "ouro"), dict(EXT_QUOTA[ds])
    raise KeyError(ds)


def assign_split(p, ds):
    if ds in EXT_QUOTA:
        p["split"], p["cluster"] = p["half"], p["cluster_id"]
    elif ds == "D1":
        p["split"] = "dev" if p["half"] == "dev" else "test"
        p["cluster"] = "%s:%s" % (p["rel"], p["a"])
    else:
        key = str(p["a"])
        p["split"] = "dev" if int(hashlib.sha256(key.encode()).hexdigest()[0], 16) % 2 == 0 else "test"
        p["cluster"] = key


@torch.no_grad()
def dev_dirs(R, kept, kmax=10):
    dev_items = [p for p in kept if p["split"] == "dev"]
    if len(dev_items) < 1:
        return None, {"n_dev": 0}
    Aa = torch.stack([p["_b"]["H"][1][len(p["A"]) - 1] - p["_a"]["H"][1][len(p["A"]) - 1] for p in dev_items])
    Sm = torch.cat([p["_b"]["H"][1][sorted(p["pos"])] - p["_a"]["H"][1][sorted(p["pos"])] for p in dev_items])
    out, info = {}, {"n_dev": len(dev_items), "src_rows": int(Sm.shape[0])}
    for nm, X, k in (("ans", Aa, 1), ("src", Sm, kmax)):
        U, Sv, Vh = torch.linalg.svd(X.float(), full_matrices=False)
        kk = min(k, Vh.shape[0])
        out[nm] = Vh[:kk].T.contiguous()
        e = (Sv ** 2) / (Sv ** 2).sum()
        info[nm] = {"k": kk, "var_top1": r5(e[0]), "var_topk": r5(e[:kk].sum())}
    return out, info


def run_dataset(R, name, ds, debug, deadline, res_ds):
    t0 = time.time()
    cands, quota = candidates(R.M, ds)
    quota = {k: 10 ** 9 for k in quota} if FULL_POOL[0] else quota
    if debug:
        quota = {k: 1 for k in quota}
    stats = {"A_fail": 0, "B_fail": 0}

    def know(p):
        if deadline is not None and time.time() > deadline:
            raise KF.Budget()
        a = R.run(p["A"], rec=True)
        if int(a["z"][-1].argmax()) != p["tA"]:
            stats["A_fail"] += 1
            return False
        b = R.run(p["B"], rec=True)
        if int(b["z"][-1].argmax()) != p["tB"]:
            stats["B_fail"] += 1
            return False
        p["_a"], p["_b"] = a, b
        return True
    truncated = False
    try:
        kept, scanned = KD.select(cands, know, quota)
    except KF.Budget:
        kept, scanned, truncated = [p for p in cands if "_a" in p], None, True
    for p in kept:
        assign_split(p, ds)
    dirs, dinfo = dev_dirs(R, kept)
    res_ds.update({"candidates": len(cands), "scanned": scanned, "pairs": len(kept), "know_stats": stats,
                   "by_rel": {r_: sum(1 for p in kept if p["rel"] == r_) for r_ in sorted(set(p["rel"] for p in kept))},
                   "split_counts": {s: sum(1 for p in kept if p["split"] == s) for s in ("dev", "test")},
                   "width_dirs": dinfo, "truncated_selection": truncated, "items": [],
                   "select_seconds": round(time.time() - t0, 1)})
    print("[k16p %s] %s kept %d scanned %s %.0fs" % (name, ds, len(kept), scanned, time.time() - t0), flush=True)
    run_items = kept
    if len(kept) < MIN_ITEMS:
        res_ds["skipped_lt_min_items"] = MIN_ITEMS
        run_items = []
        print("[k16p %s] %s skipped: %d kept < %d" % (name, ds, len(kept), MIN_ITEMS), flush=True)
    res_ds["arms"] = sorted(ARMS)
    nw2 = 0
    for idx, p in enumerate(run_items):
        if deadline is not None and time.time() > deadline:
            res_ds["truncated_items"] = True
            break
        do_w2 = "w2" in ARMS and nw2 < W2_MAX.get(name, 10 ** 9)
        nw2 += int(do_w2)
        f0, ti = R.n_fwd, time.time()
        it = process_item(R, p, idx, dirs, do_w2)
        it["forwards"], it["seconds"] = R.n_fwd - f0, round(time.time() - ti, 2)
        res_ds["items"].append(it)
        if idx % 5 == 0 or idx == len(kept) - 1:
            print("[k16p %s] %s item %d/%d (%d fwd, %.1fs) total %.0fs" % (
                name, ds, idx + 1, len(kept), it["forwards"], it["seconds"], time.time() - t0), flush=True)
    for p in kept:
        p.pop("_a", None)
        p.pop("_b", None)
    res_ds["seconds"] = round(time.time() - t0, 1)


# ------------------------------------------------------------------ T3
@torch.no_grad()
def run_t3(R, name, prompts):
    t0 = time.time()
    mpts, V, N, Ur, meta = KF.armB_vectors(R.M, name, R.K)
    assert [tuple(x) for x in mpts] == [(s, R.mid) for s in range(1, R.K + 1)], mpts
    pos_ids = [R.M.enc(w, ans=True) for w in POS_W]
    neg_ids = [R.M.enc(w, ans=True) for w in NEG_W]
    assert all(len(x) == 1 for x in pos_ids + neg_ids)
    pos_ids, neg_ids = [x[0] for x in pos_ids], [x[0] for x in neg_ids]
    lo = []
    for pr in prompts:
        ids = R.M.enc(pr)
        row = []
        for sched, loops, a, kind in T3_CONDS:
            lm = {s: [e_add_all(S2.caa_delta(V[s - 1], N[s - 1], a) if kind == "caa" else a * N[s - 1] * Ur[s - 1],
                                R.dev)] for s in loops}
            z = R.run(ids, lmid=lm)["z"]
            row.append([r5(x) for x in (torch.logsumexp(z[:, pos_ids], -1) - torch.logsumexp(z[:, neg_ids], -1)).tolist()])
        lo.append(row)
    meta.update({"conds": [{"sched": s, "loops": lp, "a": a, "vector": k} for s, lp, a, k in T3_CONDS],
                 "prompts": list(prompts), "pos_words": POS_W, "neg_words": NEG_W, "pos_ids": pos_ids,
                 "neg_ids": neg_ids, "lo": lo, "seconds": round(time.time() - t0, 1)})
    print("[k16p %s] T3 %d prompts %.0fs" % (name, len(prompts), time.time() - t0), flush=True)
    return meta


# ------------------------------------------------------------------ one model
def run_model(name, out_dir, dev, debug, budget_min, datasets, t3n, ftag=""):
    t0 = time.time()
    M = load_ouro(MODEL_DIR.get(name, REPOS[name]), dev)
    R = Runner(M)
    load_s = time.time() - t0
    res = {"model": name, "repo": REPOS[name], "host": RUN_ROLE, "device": dev, "debug": debug,
           "L": R.L, "K": R.K, "mid": R.mid, "attn_implementation": M.m.config._attn_implementation,
           "dtype": str(next(M.m.parameters()).dtype), "load_seconds": round(load_s, 1), "datasets": {},
           "budget_min": budget_min, "model_path": os.path.basename(os.path.normpath(MODEL_DIR[name])) if name in MODEL_DIR else REPOS[name], "arms": sorted(ARMS),
           "min_items": MIN_ITEMS}
    path = os.path.join(out_dir, "k16p_%s%s.json" % (name, ftag))

    def dump(partial):
        res["partial"] = partial
        res["seconds_total"] = round(time.time() - t0, 1)
        res["forwards"] = R.n_fwd
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    d1c = KD.d1_candidates(M.enc)
    cap = [p for p in d1c if p["rel"] == "capital"][0]
    city = [p for p in d1c if p["rel"] == "city"][0]
    res["checks"] = model_checks(R, cap, city)
    res["checks"]["eager_vs_sdpa"] = sdpa_gap(R, [p["A"] for p in d1c[:5]])
    dump(True)
    print("[k16p %s] checks all_ok %s eager_vs_sdpa %s %.0fs" % (name, res["checks"]["all_ok"],
                                                                res["checks"]["eager_vs_sdpa"], time.time() - t0), flush=True)
    if not res["checks"]["all_ok"]:
        raise SystemExit("hook checks failed: %s" % json.dumps(res["checks"]))
    deadline = t0 + 60.0 * budget_min if budget_min > 0 else None
    for ds in datasets:
        res["datasets"][ds] = {}
        run_dataset(R, name, ds, debug, deadline, res["datasets"][ds])
        dump(True)
    if t3n > 0:
        res["t3"] = run_t3(R, name, T3_PROMPTS[:t3n])
    dump(False)
    print("[k16p %s] done %.0fs, %d forwards" % (name, time.time() - t0, R.n_fwd), flush=True)
    R = M = None
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


def cpu_check(name, out_dir):
    t0 = time.time()
    M = load_ouro(REPOS[name], "cpu")
    R = Runner(M)
    d1c = KD.d1_candidates(M.enc)
    cap = [p for p in d1c if p["rel"] == "capital"][0]
    city = [p for p in d1c if p["rel"] == "city"][0]
    res = {"model": name, "host": RUN_ROLE, "device": "cpu", "attn_implementation": M.m.config._attn_implementation,
           "prompt": cap["A"], "source": cap["pos"], "checks": model_checks(R, cap, city)}
    res["checks"]["eager_vs_sdpa"] = sdpa_gap(R, [p["A"] for p in d1c[:2]])
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out_dir, "check_cpu_%s.json" % name), "w"), indent=1)
    print(json.dumps(res["checks"]), flush=True)
    return res["checks"]["all_ok"]


# ------------------------------------------------------------------ summary
def met(rv, cv, tA, tB, j=-1):
    zA, zB, lse, top = rv[j]
    czA, czB = cv[j][0], cv[j][1]
    return {"dm": (zB - zA) - (czB - czA), "dzA": zA - czA, "dzB": zB - czB, "pB": math.exp(zB - lse),
            "flip": int(top == tB)}


def Dgap(it):
    return (it["cleanB"][-1][1] - it["cleanB"][-1][0]) - (it["clean"][-1][1] - it["clean"][-1][0])


def rowm(it, key, j=-1):
    rv = it["rows"].get(key)
    return None if rv is None else met(rv, it["clean"], it["tA"], it["tB"], j)


def E1(it, en, kn, j=-1):
    """dm(edit, ko) - dm(off, ko); rand pooled = mean over the available seeds."""
    names = ["rand0", "rand1", "rand2"] if kn == "rand" else [kn]
    xs = []
    for k in names:
        a, b = rowm(it, "Y1|%s|%s" % (en, k), j), rowm(it, "Y1|off|%s" % k, j)
        if a is not None and b is not None:
            xs.append(a["dm"] - b["dm"])
    return sum(xs) / len(xs) if xs else None


def summarize(out_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "k16p_*.json"))):
        d = json.load(open(f))
        if d["model"] in res:      # a second run of the same model (other datasets): merge
            d0 = res[d["model"]]
            d0["datasets"].update(d["datasets"])
            d0.setdefault("extra_runs", []).append({k: d.get(k) for k in ("checks", "seconds_total", "forwards",
                                                                          "load_seconds")})
            if "t3" not in d0 and "t3" in d:
                d0["t3"] = d["t3"]
        else:
            res[d["model"]] = d
    procs = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "proc_*.json"))}
    gpu_min = round(sum(p["seconds"] for p in procs.values() if p.get("device") == "cuda") / 60.0, 2)
    ck = {k: {} for k in ("hook_identity", "knockout_placement", "items", "y1_2x2", "y1_controls", "y1_positive_control",
                          "y3_undo", "width_control", "y4_lin_vs_actual", "w2_window", "t3_per_loop",
                          "eager_vs_sdpa_gap", "e_excluded_small_gap")}
    ck["gpu_minutes"] = gpu_min
    allp = [json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "**", "proc_*.json"), recursive=True)]
    ck["gpu_minutes_including_debug_runs"] = round(sum(p["seconds"] for p in allp if p.get("device") == "cuda") / 60.0, 2)
    ran = set(ds for d in res.values() for ds in d["datasets"])
    ck["datasets_missing"] = {ds: "no B-K16-DATA report or files when the run was launched" for ds in ("D4", "D5")
                              if ds not in ran}
    ck["downloads"] = ("none from the hub. Ouro-1.4B and Ouro-2.6B from the HF cache (pinned revisions); Ouro-1.4B-Thinking "
                       "snapshot 3aaa2224 from the folder given by --model_dir")

    def f2(x):
        if isinstance(x, list):
            x = x[0]
        return "-" if x is None else "%.2f" % x

    def fci(v):
        return "-" if v is None else "%.2f [%.2f, %.2f]" % (v[0], v[1], v[2])

    def C(its, fn):
        return boot_cluster([fn(it) for it in its], [it["cluster"] for it in its])

    L = ["# K16 pathway tests on Ouro (B-K16-PATH-OURO)", "",
         "All scored rows at batch 1, eager attention. dm = change in z(B) - z(A) against the clean A run (logits); "
         "dz(A), dz(B) separately; flip = top-1 is B's answer. Edits at Llast@1 (model.norm output of loop 1). Knockouts "
         "remove attention from the LAST position to the named keys in loops 2-4. E0 = dm(edit, none); E1 = dm(edit, ko) - "
         "dm(off, ko); rescue = E1 - E0. e = dm / D_i only where D_i (clean B minus clean A final margin) >= %.1f. "
         "Cells: mean [95%% bootstrap CI over entity clusters, %d resamples, seed 0]." % (GAP_E, NBOOT), "",
         "GPU minutes (wall time of every GPU process, load included): %s." % gpu_min, ""]
    for m, d in res.items():
        ch = d.get("checks", {})
        ck["hook_identity"][m] = {k: ch.get(k) for k in ("hook_identity_ok", "call_counts_ok", "wrapper_no_knockout_bit_identical",
                                                         "all_zero_additive_mask_bit_identical",
                                                         "zero_edit_norm_and_lmid_bit_identical", "rec_run_bit_identical",
                                                         "capture_run_bit_identical", "grad_leaf_run_bit_identical",
                                                         "repeat_run_bit_identical", "eager_mask", "lmid_edit_placement",
                                                         "norm_edit_placement")}
        cpuf = os.path.join(out_dir, "check_cpu_%s.json" % m)
        if os.path.exists(cpuf):
            ck["hook_identity"][m]["cpu_check_all_ok"] = json.load(open(cpuf))["checks"]["all_ok"]
        ck["knockout_placement"][m] = dict(ch.get("knockout_placement", {}), allbut_finite=ch.get("allbut_knockout_finite"))
        ck["eager_vs_sdpa_gap"][m] = ch.get("eager_vs_sdpa")
        L += ["## %s (%s, L %d, K %d, host %s, %s)" % (m, d["repo"], d["L"], d["K"], d["host"], d["device"]), "",
              "Checks all_ok %s. Eager vs SDPA: %s. Load %.0fs, total %.0fs, %d forwards." % (
                  ch.get("all_ok"), ch.get("eager_vs_sdpa"), d["load_seconds"], d.get("seconds_total", 0), d.get("forwards", 0)), ""]
        for k_ in ("items", "y1_2x2", "y1_controls", "y1_positive_control", "y3_undo", "width_control",
                   "y4_lin_vs_actual", "w2_window", "e_excluded_small_gap"):
            ck[k_].setdefault(m, {})
        for ds, D in d["datasets"].items():
            items = D["items"]
            if not items:
                ck["items"][m][ds] = {"n": 0, "kept_pairs": D.get("pairs"), "scanned": D.get("scanned"),
                                      "know_stats": D.get("know_stats"), "by_rel": D.get("by_rel"),
                                      "skipped_lt_min_items": D.get("skipped_lt_min_items")}
                L += ["### %s / %s: no items run (kept pairs %s, scanned %s, know stats %s, skipped below %s)" % (
                    m, ds, D.get("pairs"), D.get("scanned"), D.get("know_stats"), D.get("skipped_lt_min_items")), ""]
                continue
            nyd = sum(1 for it in items if "w2_ko" in it)
            ck["items"][m][ds] = {"n": len(items), "by_rel": D["by_rel"], "scanned": D["scanned"],
                                  "know_stats": D["know_stats"], "clusters": len(set(it["cluster"] for it in items)),
                                  "split_counts": D["split_counts"], "w2_items": nyd,
                                  "truncated": bool(D.get("truncated_items") or D.get("truncated_selection")),
                                  "para_last_token_fallback": sum(1 for it in items for x in it.get("para_modes", []) if x == "last_token")}
            ck["e_excluded_small_gap"][m][ds] = sum(1 for it in items if Dgap(it) < GAP_E)
            subsets = [("all", items)]
            if ds in ("D1", "D4"):
                subsets += [(r_, [it for it in items if it["rel"] == r_]) for r_ in sorted(set(it["rel"] for it in items))]
            L += ["### %s / %s" % (m, ds), "",
                  "Items %d (%s), clusters %d, scanned %s, know stats %s, split %s, clean gap D [CI] %s; e excluded (D < %.1f): %d."
                  " Paraphrase modes: %s. Width dirs: %s." % (
                      len(items), D["by_rel"], ck["items"][m][ds]["clusters"], D["scanned"], D["know_stats"],
                      D["split_counts"], fci(C(items, Dgap)), GAP_E, ck["e_excluded_small_gap"][m][ds],
                      {x: sum(1 for it in items for y in it.get("para_modes", []) if y == x) for x in ("aligned", "last_token", "empty")},
                      D.get("width_dirs")), ""]
            y12, yc, ypc = {}, {}, {}
            for sn, its in subsets:
                if not its:
                    continue
                y12[sn], yc[sn] = {}, {}
                L += ["#### Y1 final readout, %s (n %d): dm [CI] / flip" % (sn, len(its)), "",
                      "| knockout | " + " | ".join(EDITS) + " |", "|---" * (len(EDITS) + 1) + "|"]
                for kn in KOS:
                    cells = []
                    for en in EDITS:
                        key = "Y1|%s|%s" % (en, kn)
                        if not any(key in it["rows"] for it in its):
                            cells.append("-")
                            continue
                        dmv = C(its, lambda it: (rowm(it, key) or {}).get("dm"))
                        fl = C(its, lambda it: (rowm(it, key) or {}).get("flip"))
                        cells.append("%s / %s" % (fci(dmv), f2(fl)))
                    L.append("| %s | " % kn + " | ".join(cells) + " |")
                L += ["", "#### Y1 E1 and rescue, %s: E1 [CI] / rescue [CI]" % sn, "",
                      "| knockout | ans_O1 | ans_para | src_O1 |", "|---|---|---|---|"]
                for kn in KOS[1:5] + ["rand", "nonsrc", "allbut"]:
                    cells = []
                    for en in ("ans_O1", "ans_para", "src_O1"):
                        e1 = C(its, lambda it: E1(it, en, kn))
                        rs = C(its, lambda it: None if E1(it, en, kn) is None or rowm(it, "Y1|%s|none" % en) is None
                               else E1(it, en, kn) - rowm(it, "Y1|%s|none" % en)["dm"])
                        cells.append("%s / %s" % (fci(e1), fci(rs)))
                        yc[sn].setdefault(en, {})[kn] = {"E1": e1, "rescue": rs}
                    L.append("| %s | " % kn + " | ".join(cells) + " |")
                L += ["", "#### Y1 main cells, %s: final dz(A) / dz(B) / p(B) / flip; dm per loop end 1..K" % sn, "",
                      "| cell | dz(A) | dz(B) | p(B) | flip | dm by loop | flip by loop |", "|---|---|---|---|---|---|---|"]
                for en in EDITS:
                    for kn in ("none", "src_all", "allbut"):
                        key = "Y1|%s|%s" % (en, kn)
                        if not any(key in it["rows"] for it in its) or key == "Y1|off|none":
                            continue
                        fm = {f: C(its, lambda it: (rowm(it, key) or {}).get(f)) for f in ("dzA", "dzB", "pB", "flip", "dm")}
                        traj = [C(its, lambda it: (rowm(it, key, j) or {}).get("dm")) for j in range(d["K"])]
                        trf = [C(its, lambda it: (rowm(it, key, j) or {}).get("flip")) for j in range(d["K"])]
                        L.append("| %s|%s | %s | %s | %s | %s | %s | %s |" % (
                            en, kn, fci(fm["dzA"]), fci(fm["dzB"]), f2(fm["pB"]), f2(fm["flip"]),
                            ", ".join(f2(x) for x in traj), ", ".join(f2(x) for x in trf)))
                        y12[sn]["%s|%s" % (en, kn)] = {"dm": fm["dm"], "dzA": fm["dzA"], "dzB": fm["dzB"], "pB": fm["pB"],
                                                       "flip": fm["flip"], "dm_by_loop": [(x[0] if x else None) for x in traj],
                                                       "e": C(its, lambda it: None if Dgap(it) < GAP_E or rowm(it, key) is None
                                                              else rowm(it, key)["dm"] / Dgap(it))}
                ypc[sn] = {"src_O1_E0": C(its, lambda it: (rowm(it, "Y1|src_O1|none") or {}).get("dm")),
                           "src_O1_e0": C(its, lambda it: None if Dgap(it) < GAP_E or rowm(it, "Y1|src_O1|none") is None
                                          else rowm(it, "Y1|src_O1|none")["dm"] / Dgap(it)),
                           "src_O1_E1_src_all": yc[sn]["src_O1"]["src_all"]["E1"],
                           "src_O1_E1_allbut": yc[sn]["src_O1"]["allbut"]["E1"],
                           "drop_src_all": yc[sn]["src_O1"]["src_all"]["rescue"],
                           "drop_allbut": yc[sn]["src_O1"]["allbut"]["rescue"]}
                L.append("")
            ck["y1_2x2"][m][ds] = y12
            ck["y1_controls"][m][ds] = yc
            ck["y1_positive_control"][m][ds] = ypc
            # Y3
            y3 = {}
            L += ["#### Y3 undo (src_O1 at Llast@1, clean source restored at Llast@2 or Llast@3)", "",
                  "| subset | row | dm by loop | final e [CI] | flip |", "|---|---|---|---|---|"]
            for sn, its in subsets:
                if not its:
                    continue
                y3[sn] = {}
                for key in ("Y1|src_O1|none", "Y3|undo2", "Y3|undo3"):
                    traj = [C(its, lambda it: (rowm(it, key, j) or {}).get("dm")) for j in range(d["K"])]
                    e = C(its, lambda it: None if Dgap(it) < GAP_E or rowm(it, key) is None else rowm(it, key)["dm"] / Dgap(it))
                    fl = C(its, lambda it: (rowm(it, key) or {}).get("flip"))
                    y3[sn][key] = {"e": e, "flip": fl, "dm_by_loop": [(x[0] if x else None) for x in traj]}
                    L.append("| %s | %s | %s | %s | %s |" % (sn, key, ", ".join(f2(x) for x in traj), fci(e), f2(fl)))
            ck["y3_undo"][m][ds] = y3
            L.append("")
            # width
            wt = [it for it in items if it["split"] == "test" and "W|ans_top1" in it["rows"]]
            wc = {}
            if wt:
                L += ["#### Width control (test half, n %d): dm by loop / retention [CI] / final e [CI] / flip" % len(wt), "",
                      "| row | dm by loop | retention | final e | flip |", "|---|---|---|---|---|"]
                for key, j0 in (("Y1|ans_O1|none", 0), ("W|ans_top1", 0), ("Y1|src_O1|none", 1), ("W|src_top1", 1),
                                ("W|src_top10", 1)):
                    def ret(it, key=key, j0=j0):
                        ms = [rowm(it, key, j) for j in range(d["K"])]
                        if any(x is None for x in ms):
                            return None
                        mx = max(abs(x["dm"]) for x in ms[j0:])
                        return None if mx < 1.0 else ms[-1]["dm"] / mx
                    traj = [C(wt, lambda it: (rowm(it, key, j) or {}).get("dm")) for j in range(d["K"])]
                    rt = C(wt, ret)
                    e = C(wt, lambda it: None if Dgap(it) < GAP_E or rowm(it, key) is None else rowm(it, key)["dm"] / Dgap(it))
                    fl = C(wt, lambda it: (rowm(it, key) or {}).get("flip"))
                    wc[key] = {"dm_by_loop": [(x[0] if x else None) for x in traj], "retention": rt, "e": e, "flip": fl}
                    L.append("| %s | %s | %s | %s | %s |" % (key, ", ".join(f2(x) for x in traj), fci(rt), fci(e),
                                                              f2(fl)))
                L.append("")
            ck["width_control"][m][ds] = wc
            # Y4
            y4 = {}
            L += ["#### Y4 linear account: dm_lin(j) = grad_{H_j} m_final . dH_j over source + last (all positions in "
                  "brackets) vs actual final dm", "",
                  "| edit | loop | mean dm_lin [all] | mean actual dm | Spearman lin [all] | abs dH src / last | cos(dH_j, dH_1) |",
                  "|---|---|---|---|---|---|---|"]
            for en in ("ans_O1", "ans_para", "src_O1"):
                its = [it for it in items if en in it.get("y4", {})]
                if not its:
                    continue
                act = [rowm(it, "Y1|%s|none" % en)["dm"] for it in its]
                y4[en] = {}
                for j in range(d["K"]):
                    lin = [it["y4"][en]["lin"][j] for it in its]
                    lina = [it["y4"][en]["lin_all"][j] for it in its]
                    rec = {"mean_lin": r5(sum(lin) / len(lin)), "mean_lin_all": r5(sum(lina) / len(lina)),
                           "mean_actual": r5(sum(act) / len(act)), "spearman": spearman(lin, act),
                           "spearman_all": spearman(lina, act),
                           "n_src": r5(sum(it["y4"][en]["n_src"][j] for it in its) / len(its)),
                           "n_last": r5(sum(it["y4"][en]["n_last"][j] for it in its) / len(its)),
                           "cos": r5(sum(it["y4"][en]["cos"][j] for it in its) / len(its))}
                    y4[en]["loop%d" % (j + 1)] = rec
                    L.append("| %s | %d | %s [%s] | %s | %s [%s] | %s / %s | %s |" % (
                        en, j + 1, f2(rec["mean_lin"]), f2(rec["mean_lin_all"]), f2(rec["mean_actual"]),
                        f2(rec["spearman"]), f2(rec["spearman_all"]), f2(rec["n_src"]), f2(rec["n_last"]), f2(rec["cos"])))
            gi = [it["y4"].get("grad_run_bit_identical", it.get("y4_grad_run_bit_identical")) for it in items if "y4" in it]
            y4["grad_run_bit_identical_all"] = all(gi) if gi else None
            ck["y4_lin_vs_actual"][m][ds] = y4
            L.append("")
            # W2
            w2 = {}
            for sn, its0 in subsets:
                its = [it for it in its0 if "w2_ko" in it]
                if not its:
                    continue
                K_, L_ = d["K"], d["L"]
                mass = [[sum(it["w2_mass"][j][l] for it in its) / len(its) for l in range(L_)] for j in range(K_)]
                mass0 = [[sum(it["w2_mass0"][j][l] for it in its) / len(its) for l in range(L_)] for j in range(K_)]

                def wdm(it, j, l):
                    zA, zB, _ = it["w2_ko"][j][l]
                    return (zB - zA) - (it["clean"][-1][1] - it["clean"][-1][0])
                dm = [[sum(wdm(it, j, l) for it in its) / len(its) for l in range(L_)] for j in range(K_)]
                per_loop = []
                for j in range(K_):
                    sm = C(its, lambda it: sum(wdm(it, j, l) for l in range(L_)))
                    lmax = max(range(L_), key=lambda l: dm[j][l])
                    mmax = max(range(L_), key=lambda l: mass[j][l])
                    per_loop.append({"loop": j + 1, "summed_dm": sm, "max_call_layer": lmax, "max_call_dm": r5(dm[j][lmax]),
                                     "max_mass_layer": mmax, "max_mass": r5(mass[j][mmax]),
                                     "mean_mass": r5(sum(mass[j]) / L_), "mean_mass_pos0": r5(sum(mass0[j]) / L_),
                                     "flip_calls": sum(1 for l in range(L_) for it in its if it["w2_ko"][j][l][2] != it["tA"])})
                w2[sn] = {"n": len(its), "per_loop": per_loop, "mass_grid": [[r5(x) for x in r] for r in mass],
                          "dm_grid": [[r5(x) for x in r] for r in dm]}
                L += ["#### W2 retrieval window, %s (n %d): per loop" % (sn, len(its)), "",
                      "| loop | summed single-call dm [CI] | max call (layer: dm) | max mass (layer: mass) | mean mass src / pos0 | calls flipping top-1 |",
                      "|---|---|---|---|---|---|"]
                for r_ in per_loop:
                    L.append("| %d | %s | %d: %s | %d: %s | %s / %s | %d |" % (
                        r_["loop"], fci(r_["summed_dm"]), r_["max_call_layer"], f2(r_["max_call_dm"]), r_["max_mass_layer"],
                        f2(r_["max_mass"]), f2(r_["mean_mass"]), f2(r_["mean_mass_pos0"]), r_["flip_calls"]))
                L += ["", "Single-call dm grid (rows loops, columns layers 0..%d):" % (L_ - 1), ""]
                for j in range(K_):
                    L.append("- loop %d: %s" % (j + 1, " ".join("%.2f" % x for x in dm[j])))
                L += ["", "Attention mass last->source grid:", ""]
                for j in range(K_):
                    L.append("- loop %d: %s" % (j + 1, " ".join("%.2f" % x for x in mass[j])))
                L.append("")
            ck["w2_window"][m][ds] = w2
        # T3
        if "t3" in d:
            t3 = d["t3"]
            conds = t3["conds"]
            lo = t3["lo"]
            n = len(lo)
            cl_ = list(range(n))
            ck["t3_per_loop"][m] = {}
            L += ["### %s / T3 sentiment CAA at Lmid (layer %d), %d held-out prompts: dLO per loop end [CI]" % (m, d["mid"], n),
                  "", "LO = logsumexp z[POS] - logsumexp z[NEG] at the last prompt position. Clean LO by loop: %s." % (
                      ", ".join(f2(sum(lo[i][0][j] for i in range(n)) / n) for j in range(d["K"]))), "",
                  "| condition | " + " | ".join("loop %d" % (j + 1) for j in range(d["K"])) + " |", "|---" * (d["K"] + 1) + "|"]
            for c, cd in enumerate(conds):
                if c == 0:
                    continue
                cis = [boot_cluster([lo[i][c][j] - lo[i][0][j] for i in range(n)], cl_) for j in range(d["K"])]
                nm = "%s|%s|%g" % (cd["sched"], cd["vector"], cd["a"])
                ck["t3_per_loop"][m][nm] = cis
                L.append("| %s | " % nm + " | ".join(fci(x) for x in cis) + " |")
            L.append("")
    json.dump({"models": res, "processes": procs}, open(os.path.join(out_dir, "results.json"), "w"))
    json.dump(ck, open(os.path.join(out_dir, "checks_detail.json"), "w"), indent=1)
    cc = compact_checks(ck)
    with open(os.path.join(out_dir, "checks.json"), "w") as f:
        f.write("{\n" + ",\n".join('"%s": %s' % (k, json.dumps(v)) for k, v in cc.items()) + "\n}\n")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L))
    print(json.dumps({"gpu_minutes": gpu_min, "items": ck["items"]}), flush=True)


def compact_checks(ck):
    """checks.json: the brief's keys, point values rounded to 2 decimals, [v, lo, hi] for headline CIs."""
    def r(v):
        if v is None:
            return None
        if isinstance(v, list):
            return [None if x is None else round(x, 2) for x in v[:3]]
        return round(v, 2) if isinstance(v, float) else v

    def p(v):
        return None if v is None else round(v[0], 2)

    def subs(d):
        return {("%s/%s" % (ds, sn) if sn != "all" else ds): v for ds, dd in d.items() for sn, v in dd.items()
                if not (ds in ("D1", "D4") and sn == "all")}
    out = {"hook_identity": {m: {"gpu_all_ok": v.get("hook_identity_ok"), "cpu_check_all_ok": v.get("cpu_check_all_ok"),
                                 "norm_edit_placement": (v.get("norm_edit_placement") or {}).get("ok"),
                                 "lmid_edit_placement": (v.get("lmid_edit_placement") or {}).get("ok"),
                                 "eager_mask_none_calls": (v.get("eager_mask") or {}).get("none_calls")}
                             for m, v in ck["hook_identity"].items()},
           "knockout_placement": {m: {"ok": v.get("ok"), "knocked_probs_exactly_zero": v.get("knocked_probs_exactly_zero"),
                                      "row_sum_max_abs_err": v.get("row_sum_max_abs_err"), "allbut_finite": v.get("allbut_finite")}
                                  for m, v in ck["knockout_placement"].items()},
           "items": {m: {ds: (v["n"] if v.get("n") else "skipped: %s kept of %s scanned" % (v.get("kept_pairs"), v.get("scanned")))
                         for ds, v in d.items()} for m, d in ck["items"].items()}}
    y = {}
    for m, d in ck["y1_2x2"].items():
        y[m] = {}
        for k, v in subs(d).items():
            ds, sn = (k.split("/") + ["all"])[:2]
            c = ck["y1_controls"][m][ds][sn]
            y[m][k] = {en: {"E0": r(v.get(en + "|none", {}).get("dm")), "rescue_src_all": r(c[en]["src_all"]["rescue"]),
                            "flip_none": p(v.get(en + "|none", {}).get("flip")),
                            "flip_src_all": p(v.get(en + "|src_all", {}).get("flip"))}
                       for en in ("ans_O1", "ans_para")}
            y[m][k]["off_src_all_dm"] = r(v["off|src_all"]["dm"])
    out["y1_2x2"] = y
    out["y1_controls"] = {m: {k: {en: {kn: p(vv[en][kn]["rescue"]) for kn in ("rand", "nonsrc", "allbut", "src_l2", "src_lo", "src_hi")}
                                  for en in ("ans_O1",)} for k, vv in subs(d).items()} for m, d in ck["y1_controls"].items()}
    out["y1_positive_control"] = {m: {k: {"src_O1_E0": r(v["src_O1_E0"]), "src_O1_E1_src_all": r(v["src_O1_E1_src_all"]),
                                          "src_O1_E1_allbut": p(v["src_O1_E1_allbut"])} for k, v in subs(d).items()}
                                  for m, d in ck["y1_positive_control"].items()}
    out["y3_undo"] = {m: {ds: {"src_O1_e": p(v["all"]["Y1|src_O1|none"]["e"]), "undo2_e": r(v["all"]["Y3|undo2"]["e"]),
                               "undo3_e": r(v["all"]["Y3|undo3"]["e"]), "undo2_flip": p(v["all"]["Y3|undo2"]["flip"])}
                          for ds, v in d.items() if "all" in v} for m, d in ck["y3_undo"].items()}
    out["width_control"] = {m: {ds: {row.replace("Y1|", "").replace("|none", "").replace("W|", ""): p(v["retention"])
                                     for row, v in w.items()} for ds, w in d.items() if w} for m, d in ck["width_control"].items()}
    out["y4_lin_vs_actual"] = {m: {ds: {en: {"lin_loop1": r(v[en]["loop1"]["mean_lin"]), "actual": r(v[en]["loop1"]["mean_actual"]),
                                             "spearman_by_loop": [r(v[en]["loop%d" % j]["spearman"]) for j in range(1, 5)]}
                                        for en in ("ans_O1", "src_O1") if en in v}
                                   for ds, v in d.items() if "ans_O1" in v} for m, d in ck["y4_lin_vs_actual"].items()}
    out["w2_window"] = {m: {k: {"summed_dm_by_loop": [p(x["summed_dm"]) for x in v["per_loop"]],
                                "max_call_by_loop": [[x["max_call_layer"], r(x["max_call_dm"])] for x in v["per_loop"]],
                                "mean_mass_src_by_loop": [r(x["mean_mass"]) for x in v["per_loop"]]}
                            for k, v in subs(d).items()} for m, d in ck["w2_window"].items()}
    out["t3_per_loop"] = {m: {c: [p(x) for x in v] for c, v in d.items()} for m, d in ck["t3_per_loop"].items()}
    out["gpu_minutes"] = ck["gpu_minutes"]
    out["gpu_minutes_including_debug_runs"] = ck.get("gpu_minutes_including_debug_runs")
    out["eager_vs_sdpa_gap"] = {m: (v or {}).get("max_abs_gap_final") for m, v in ck["eager_vs_sdpa_gap"].items()}
    out["e_excluded_small_gap"] = ck["e_excluded_small_gap"]
    out["datasets_missing"] = ck["datasets_missing"]
    out["downloads"] = ck["downloads"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cpu_check", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", default="ouro")
    ap.add_argument("--datasets", default="D1,D2")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="p")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--full_pool", action="store_true", help="cluster copy: no quota, every candidate scanned")
    ap.add_argument("--budget_min", type=float, default=0.0)
    ap.add_argument("--t3n", type=int, default=50)
    ap.add_argument("--ftag", default="")
    ap.add_argument("--arms", default="width,y4,w2")
    ap.add_argument("--min_items", type=int, default=0)
    ap.add_argument("--model_dir", default="")
    a = ap.parse_args()
    FULL_POOL[0] = a.full_pool
    global ARMS, MIN_ITEMS
    ARMS = set(x for x in a.arms.split(",") if x)
    MIN_ITEMS = a.min_items
    if a.model_dir:
        MODEL_DIR[a.models.split(",")[0]] = a.model_dir
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
        return
    dev = "cpu" if a.cpu_check else "cuda"
    t0 = time.time()
    ok = True
    try:
        for name in [x for x in a.models.split(",") if x]:
            if a.cpu_check:
                ok = cpu_check(name, a.out) and ok
            else:
                run_model(name, a.out, dev, a.debug, a.budget_min, [x for x in a.datasets.split(",") if x],
                          5 if a.debug else a.t3n, a.ftag)
    finally:
        json.dump({"models": a.models, "seconds": round(time.time() - t0, 1), "host": RUN_ROLE,
                   "device": dev, "mode": "cpu_check" if a.cpu_check else ("debug" if a.debug else "full"),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                  open(os.path.join(a.out, "proc_%s.json" % a.tag), "w"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
