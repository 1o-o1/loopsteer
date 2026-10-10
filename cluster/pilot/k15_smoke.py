"""K15 smoke (brief B-K15-SMOKE): does any edit operator reach the output on a looped model?

Items: k13 build_pairs, held-out half (odd index), first 40 pairs the clean model answers both A and B.
Sites: layers {0, mid, last} x loops {1, 2, K} of the looped block (Ouro "last" = the model.norm output, the
k13 site); d28 the same layers once; d112 loop j layer l -> layer 28(j-1)+l. Edit positions = k13 subject positions.
Operators (float32 on the live tensor, written back in the model dtype):
  O1 interp  h <- lerp(h, h_B, a)                    a in [0.25, 0.5, 0.75, 1]
  O2 add     h <- h + a v, v = h_B' - h_A' on two paraphrase templates (mean)   a in [0.5, 1, 2, 4]
  O3 O2 then rescale each edited vector to its pre-edit norm
  O4 h <- h + a |v| u, u a unit Gaussian direction per position
  O5 O1 a=1 at the last position, last layer, last loop (positive control)
Metrics per cell: effect e (k13: (m_edit - m_A)/(m_B - m_A), m = logit tB - logit tA at the final readout),
flip (final top-1 == tB), KL(clean || edit) at the last position over the vocab minus {tA, tB} renormalised,
survival |h_edit - h_clean| over edit positions at each loop end >= the edited loop / |applied delta|.

    python k15_smoke.py --selftest
    python k15_smoke.py --models ouro --out OUT                        (Ouro venv)
    python k15_smoke.py --models ifm_p1_ouro,ifm_p1_ouro_inj,ifm_d28,ifm_d112 --out OUT   (xllm_venv)
    python k15_smoke.py --summarize --out OUT
"""
import argparse
import gc
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

A_INTERP = [0.25, 0.5, 0.75, 1.0]
A_ADD = [0.5, 1.0, 2.0, 4.0]
GRID = [("O1", a) for a in A_INTERP] + [(o, a) for o in ("O2", "O3", "O4") for a in A_ADD]
N_PAIRS = 40
CAP_MIN = 60.0
D112_RESERVE_MIN = 12.0
PARA = {"capital": ["The capital city of {} is", "Q: What is the capital of {}? A:"],
        "city": ["The city of {} is located in the country of", "Q: In which country is the city of {}? A:"]}


def fmt_a(a):
    return "%g" % a


def cell_key(op, a, site):
    return "%s|%s|%s" % (op, fmt_a(a), site)


# ------------------------------------------------------------------ operators (pure, CPU-testable)
def op_apply(op, h, a, hB=None, v=None, u=None):
    """h, hB, v, u: float32 [P, H]. Returns the edited [P, H]."""
    if op in ("O1", "O5"):
        return torch.lerp(h, hB, float(a))           # exactly hB at a = 1
    if op == "O2":
        return h + a * v
    if op == "O3":
        n = h + a * v
        return n * (h.norm(dim=-1, keepdim=True) / n.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    if op == "O4":
        return h + a * v.norm(dim=-1, keepdim=True) * u
    raise KeyError(op)


def unit_random(shape, seed):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.randn(shape, generator=g)
    return u / u.norm(dim=-1, keepdim=True)


def subj_span(x, y):
    """Positions between the common prefix and the common suffix of token lists x and y."""
    n = min(len(x), len(y))
    p = 0
    while p < n and x[p] == y[p]:
        p += 1
    s = 0
    while s < n - p and x[-1 - s] == y[-1 - s]:
        s += 1
    return list(range(p, len(x) - s)), list(range(p, len(y) - s))


def kl_off(zc, ze, tA, tB):
    """KL(clean || edit) per row over the vocab without tA, tB. zc [V] or [N, V], ze [N, V]."""
    mask = torch.zeros(ze.shape[-1], dtype=torch.bool, device=ze.device)
    mask[[tA, tB]] = True
    lc = torch.log_softmax(zc.float().masked_fill(mask, float("-inf")), -1)
    le = torch.log_softmax(ze.float().masked_fill(mask, float("-inf")), -1)
    t = torch.where(mask, torch.zeros_like(le), lc.exp() * (lc - le))
    return t.sum(-1)


def selftest():
    torch.manual_seed(0)
    P, H = 5, 64
    h, hB, v = torch.randn(P, H) * 3, torch.randn(P, H) * 3, torch.randn(P, H)
    u = unit_random((P, H), 1)
    res = {}
    res["o1_a1_exact_hB"] = bool(torch.equal(op_apply("O1", h, 1.0, hB=hB), hB))
    res["o1_a05_midpoint_maxerr"] = float((op_apply("O1", h, 0.5, hB=hB) - (h + hB) / 2).abs().max())
    res["o2_maxerr"] = max(float((op_apply("O2", h, a, v=v) - (h + a * v)).abs().max()) for a in A_ADD)
    res["o3_norm_rel_err"] = max(float(((op_apply("O3", h, a, v=v).norm(dim=-1) / h.norm(dim=-1)) - 1).abs().max())
                                 for a in A_ADD)
    res["o4_norm_rel_err"] = max(float((((op_apply("O4", h, a, v=v, u=u) - h).norm(dim=-1)
                                         / (a * v.norm(dim=-1))) - 1).abs().max()) for a in A_ADD)
    res["o4_unit_u"] = float((u.norm(dim=-1) - 1).abs().max())
    res["span"] = subj_span([1, 2, 7, 8, 3], [1, 2, 9, 3])
    z = torch.randn(3, 50)
    res["kl_self_zero"] = float(kl_off(z, z, 4, 9).abs().max())
    res["kl_ignores_answer_tokens"] = float(kl_off(z[0], torch.cat([z[:1, :4], z[:1, 4:5] + 5, z[:1, 5:]], 1), 4, 9).abs().max())
    ok = (res["o1_a1_exact_hB"] and res["o1_a05_midpoint_maxerr"] < 1e-5 and res["o2_maxerr"] == 0.0
          and res["o3_norm_rel_err"] < 1e-5 and res["o4_norm_rel_err"] < 1e-5 and res["o4_unit_u"] < 1e-5
          and res["span"] == ([2, 3], [2]) and res["kl_self_zero"] < 1e-6 and res["kl_ignores_answer_tokens"] < 1e-6)
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


# ------------------------------------------------------------------ models
class OuroM(object):
    """Ouro-1.4B via k13's OuroEditor loader. Points: (loop, 0 | mid | 'norm')."""

    def __init__(self):
        from k13_edit import OuroEditor
        ed = OuroEditor()
        self.enc, self.m, self.W = ed.enc, ed.m, ed.W
        inner = self.m.model
        self.K = int(ed.K)
        self.L = int(self.m.config.num_hidden_layers)
        mid = self.L // 2
        self.mods = {0: inner.layers[0], mid: inner.layers[mid], "norm": inner.norm}
        self.sites = [("%s@%d" % (lab, j), (j, l), j) for j in (1, 2, self.K)
                      for lab, l in (("L0", 0), ("Lmid", mid), ("Llast", "norm"))]
        self.loop_ends = [((j, "norm"), j) for j in range(1, self.K + 1)]
        self.layers_desc = {"L0": "layer 0 output", "Lmid": "layer %d output" % mid,
                            "Llast": "model.norm output after layer %d (carried state, k13 site)" % (self.L - 1)}
        self.dev = "cuda"
        self.readout_desc = ("lm_head(model.norm output of loop K), hooked as in k13; out.logits unused "
                             "(config early_exit_threshold %r)" % (getattr(self.m.config, "early_exit_threshold", None),))

    def order(self, pt):
        return (pt[0], self.L if pt[1] == "norm" else pt[1])

    @torch.no_grad()
    def run(self, ids, B, edit=None, record=()):
        rec, cnt = {}, {}
        self.last_calls = cnt

        def mk(key):
            def hook(mod, i, o):
                c = cnt.get(key, 0) + 1
                cnt[key] = c
                pt = (c, key)
                if edit is not None and pt == edit[0]:
                    o = edit[1](o)
                if pt in record:
                    rec[pt] = o.detach()
                return o
            return hook
        hs = [mod.register_forward_hook(mk(k)) for k, mod in self.mods.items()]
        try:
            self.m(input_ids=torch.tensor([ids] * B, device=self.dev), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return rec

    def readout(self, hlast):
        return torch.nn.functional.linear(hlast, self.W).float()


class IfmM(object):
    """IFM checkpoints via k10_ifm (xllm). Looped: points (loop, layer id). d28/d112: (1, layer id)."""

    def __init__(self, name):
        import k10_ifm
        k10_ifm.NAMES.update({"ifm_d28": "LoopedLM-P1-dense-d28-336b", "ifm_d112": "LoopedLM-P1-dense-d112-336b"})
        try:
            r = k10_ifm.make_runner(name)
        except AttributeError:                       # non-looped twin (k11_layers_ifm.py fallback)
            import xllm.models.transformer as XT
            XT.Transformer.loop_times = 1
            gc.collect()
            torch.cuda.empty_cache()
            r = k10_ifm.make_runner(name)
        if not hasattr(r.m, "loop_end_layers"):
            n_l = len(r.m.layers)
            for k_, v_ in (("loop_start_layer", 0), ("loop_end_layers", n_l), ("num_layers", n_l),
                           ("loop_input_injection_module", None), ("dense_prelude_input_injection_module", None),
                           ("dense_prelude_input_layers", []), ("dense_prelude_source_layer", -1)):
                setattr(r.m, k_, v_)
        self.r, self.m = r, r.m
        self.plain = name in ("ifm_d28", "ifm_d112")
        r.set_depth(1 if self.plain else r.K_trained)
        self.slot_pt = {slot: (int(tag[4:]), lid) for slot, lid, tag in r.schedule(r.k) if tag.startswith("loop")}
        Lb = int(self.m.loop_end_layers) - int(self.m.loop_start_layer)
        if name == "ifm_d112":
            L, mid, K = 28, 14, 4
            self.sites = [("%s@%d" % (lab, j), (1, 28 * (j - 1) + l), j) for j in (1, 2, K)
                          for lab, l in (("L0", 0), ("Lmid", mid), ("Llast", L - 1))]
            self.loop_ends = [((1, 28 * j - 1), j) for j in range(1, K + 1)]
            self.layers_desc = {"map": "loop j layer l -> d112 layer 28(j-1)+l", "L0": 0, "Lmid": mid, "Llast": L - 1}
        elif name == "ifm_d28":
            L, mid = Lb, Lb // 2
            self.sites = [("%s@1" % lab, (1, l), 1) for lab, l in (("L0", 0), ("Lmid", mid), ("Llast", L - 1))]
            self.loop_ends = [((1, L - 1), 1)]
            self.layers_desc = {"L0": 0, "Lmid": mid, "Llast": L - 1}
        else:
            L, mid, K = Lb, Lb // 2, r.k
            s0 = int(self.m.loop_start_layer)
            self.sites = [("%s@%d" % (lab, j), (j, s0 + l), j) for j in (1, 2, K)
                          for lab, l in (("L0", 0), ("Lmid", mid), ("Llast", L - 1))]
            self.loop_ends = [((j, s0 + L - 1), j) for j in range(1, K + 1)]
            self.layers_desc = {"L0": s0, "Lmid": s0 + mid, "Llast": s0 + L - 1}
        self.dev = r.dev
        self.readout_desc = "m.output(state at the last loop-end point), as k13"

    def order(self, pt):
        return pt

    def enc(self, t, ans=False):
        ids = self.r.tok.hf(t, add_special_tokens=False)["input_ids"]
        return ids if ans else [self.r.tok.bos_token_id] + ids

    @torch.no_grad()
    def run(self, ids, B, edit=None, record=()):
        r = self.r
        rec = {}
        orig = r._layer

        def lay(lid, h, freq, cache, slot):
            out = orig(lid, h, freq, cache, slot)
            pt = self.slot_pt.get(slot)
            if edit is not None and pt == edit[0]:
                out = edit[1](out)
            if pt in record:
                rec[pt] = out.detach()
            return out
        r._layer = lay
        try:
            r.fwd(torch.tensor([ids] * B, device=r.dev), r.k, cache=None,
                  example_ids=torch.arange(B, device=r.dev))
        finally:
            r._layer = orig
        return rec

    @torch.no_grad()
    def readout(self, hlast):
        return self.m.output(hlast[:, None, :], None, None)[:, -1].float()


def load_model(name):
    return OuroM() if name == "ouro" else IfmM(name)


# ------------------------------------------------------------------ one model
def para_vectors(M, p, pts):
    """v[pt] float32 [P, H] = mean over the two paraphrases of h_B' - h_A' at the subject positions."""
    P = len(p["pos"])
    vs, modes = {pt: [] for pt in pts}, []
    for t in PARA[p["rel"]]:
        ia, ib = M.enc(t.format(p["a"])), M.enc(t.format(p["b"]))
        sa, sb = subj_span(ia, ib)
        ra, rb = M.run(ia, 1, record=pts), M.run(ib, 1, record=pts)
        aligned = len(sa) == P and len(sb) == P
        modes.append("aligned" if aligned else "last_token")
        for pt in pts:
            if aligned:
                v = rb[pt][0, sb].float() - ra[pt][0, sa].float()
            else:
                v = (rb[pt][0, sb[-1]].float() - ra[pt][0, sa[-1]].float()).expand(P, -1)
            vs[pt].append(v)
    return {pt: torch.stack(vs[pt]).mean(0) for pt in pts}, modes


def make_edit(specs, pos, hB, v, u, deltas):
    def f(o):
        o = o.clone()
        for r_, (op, a) in enumerate(specs):
            if op == "clean":
                continue
            h = o[r_, pos].float()
            new = op_apply(op, h, a, hB=hB, v=v, u=u).to(o.dtype)
            deltas[r_] = float((new.float() - h).norm())
            o[r_, pos] = new
        return o
    return f


def run_model(name, out_dir, t_proc0):
    t0 = time.time()
    M = load_model(name)
    load_s = time.time() - t0
    from k13_edit import build_pairs
    pairs = build_pairs(M.enc)
    site_pts = [pt for _, pt, _ in M.sites]
    le_pts = [pt for pt, _ in M.loop_ends]
    all_pts = set(site_pts) | set(le_pts)
    final = le_pts[-1]
    cells, items = {}, []
    scanned, gap = 0, 0.0
    struct_zero, match_abs, match_rel = [], [], []
    loops_run = None
    quota = {"capital": N_PAIRS // 2, "city": N_PAIRS - N_PAIRS // 2}
    n_rel, n_subj = {"capital": 0, "city": 0}, {}
    for i, p in enumerate(pairs):
        if i % 2 == 0:
            continue
        sa_, sb_ = (p["rel"], p["a"]), (p["rel"], p["b"])
        if n_rel[p["rel"]] >= quota[p["rel"]] or n_subj.get(sa_, 0) >= 2 or n_subj.get(sb_, 0) >= 2:
            continue
        scanned += 1
        tA, tB = p["tA"], p["tB"]
        rA, rB = M.run(p["A"], 1, record=all_pts), M.run(p["B"], 1, record=all_pts)
        if loops_run is None:
            loops_run = {"loop_end_points_hit": sum(1 for q in le_pts if q in rA), "expected": len(le_pts),
                         "module_calls": {str(k): v for k, v in getattr(M, "last_calls", {}).items()}}
            assert loops_run["loop_end_points_hit"] == len(le_pts), loops_run
        zA, zB = M.readout(rA[final][:, -1])[0], M.readout(rB[final][:, -1])[0]
        if int(zA.argmax()) != tA or int(zB.argmax()) != tB:
            continue
        n_rel[p["rel"]] += 1
        n_subj[sa_] = n_subj.get(sa_, 0) + 1
        n_subj[sb_] = n_subj.get(sb_, 0) + 1
        pos = p["pos"]
        mB = float(zB[tB] - zB[tA])
        V, modes = para_vectors(M, p, site_pts)
        item = {"i": i, "a": p["a"], "b": p["b"], "rel": p["rel"], "para_modes": modes,
                "mA_single": float(zA[tB] - zA[tA]), "mB": mB}
        jobs = []
        for si, (lab, pt, j) in enumerate(M.sites):
            hB = rB[pt][0, pos].float()
            v = V[pt]
            u = unit_random(tuple(v.shape), 100003 * i + si).to(v.device)
            specs = [("clean", 0.0)] + GRID
            jobs.append((lab, pt, j, specs, pos, hB, v, u))
        T = len(p["A"])
        jobs.append(("O5", final, M.loop_ends[-1][1], [("clean", 0.0), ("O5", 1.0)], [T - 1],
                     rB[final][0, [T - 1]].float(), None, None))
        for lab, pt, j, specs, epos, hB, v, u in jobs:
            N = len(specs)
            deltas = {}
            rec = M.run(p["A"], N, edit=(pt, make_edit(specs, epos, hB, v, u, deltas)), record=all_pts)
            rec = {q: t.clone() for q, t in rec.items()}
            z = M.readout(rec[final][:, -1])
            gap = max(gap, float((z[0] - zA).abs().max()))
            zc = z[0:1].repeat(N, 1)
            recc = {q: t[0:1].repeat(N, 1, 1) for q, t in rec.items()}
            # rows rerun alone against the batch-1 clean A run: every row at the final point (the structural
            # zero must be exact) and O1 a=1 at every site (state-match control)
            single = [r_ for r_, sp in enumerate(specs) if sp[0] != "clean" and (pt == final or tuple(sp) == ("O1", 1.0))]
            for r_ in single:
                d1 = {}
                rr = M.run(p["A"], 1, edit=(pt, make_edit([specs[r_]], epos, hB, v, u, d1)), record=all_pts)
                for q in rec:
                    rec[q][r_] = rr[q][0]
                    recc[q][r_] = rA[q][0]
                z[r_] = M.readout(rr[final][:, -1])[0]
                zc[r_] = zA
                deltas[r_] = d1[0]
            kls = kl_off(zc, z, tA, tB)
            ends = [(pe, je) for pe, je in M.loop_ends if je >= j]
            later = [q for q in all_pts if M.order(q) > M.order(pt)]
            for r_, (op, a) in enumerate(specs):
                if op == "clean":
                    continue
                site = lab if op != "O5" else "last_pos"
                key = cell_key(op, a, site)
                c = cells.setdefault(key, {"effect": [], "flip": [], "kl": [], "surv": [], "dA": [], "dB": [],
                                           "loop_ends": [je for _, je in ends]})
                mA = float(zc[r_, tB] - zc[r_, tA])
                d = mB - mA
                m_e = float(z[r_, tB] - z[r_, tA])
                e = (m_e - mA) / d if abs(d) > 1e-3 else None
                c["effect"].append(e)
                c["flip"].append(int(int(z[r_].argmax()) == tB))
                c["kl"].append(float(kls[r_]))
                c["dA"].append(float(z[r_, tA] - zc[r_, tA]))
                c["dB"].append(float(z[r_, tB] - zc[r_, tB]))
                dn = deltas.get(r_, 0.0)
                c["surv"].append([float((rec[pe][r_, epos].float() - recc[pe][r_, epos].float()).norm()) / dn
                                  if dn > 0 else None for pe, _ in ends])
                if op != "O5" and pt == final and e is not None:
                    struct_zero.append(abs(e))
                if (op, a) == ("O1", 1.0) and later:
                    dif = [rec[q][r_, epos].float() - rB[q][0, epos].float() for q in later]
                    ma = max(float(x.abs().max()) for x in dif)
                    mr = max(float((x.norm(dim=-1) / rB[q][0, epos].float().norm(dim=-1)).max())
                             for x, q in zip(dif, later))
                    c.setdefault("match_abs", []).append(ma)
                    c.setdefault("match_rel", []).append(mr)
                    match_abs.append(ma)
                    match_rel.append(mr)
        items.append(item)
        print("[k15 %s] pair %d (idx %d) kept %d/%d %.0fs" % (name, scanned, i, len(items), N_PAIRS,
                                                             time.time() - t0), flush=True)
        if len(items) >= N_PAIRS:
            break

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None
    summ = {}
    for key, c in cells.items():
        S = len(c["loop_ends"])
        surv = [mean([s[k] for s in c["surv"]]) for k in range(S)]
        summ[key] = {"effect": mean(c["effect"]), "flip": mean(c["flip"]), "kl": mean(c["kl"]),
                     "survival_by_loop_end": dict(zip([str(x) for x in c["loop_ends"]], surv)),
                     "n": len(c["flip"]), "n_effect": sum(1 for x in c["effect"] if x is not None),
                     "dA": mean(c["dA"]), "dB": mean(c["dB"]),
                     "o1a1_state_match_max_abs": max(c["match_abs"]) if c.get("match_abs") else None,
                     "o1a1_state_match_max_rel": max(c["match_rel"]) if c.get("match_rel") else None,
                     "per_pair_effect": c["effect"], "per_pair_flip": c["flip"],
                     "per_pair_dA": c["dA"], "per_pair_dB": c["dB"]}
    modes = [m_ for it in items for m_ in it["para_modes"]]
    subj = sorted(set("%s:%s" % (it["rel"], x) for it in items for x in (it["a"], it["b"])))
    res = {"model": name, "pairs_total": len(pairs), "heldout_scanned": scanned, "pairs_passed": len(items),
           "pairs_by_relation": n_rel, "n_subjects": len(subj), "subjects": subj, "loops_run": loops_run,
           "readout": M.readout_desc,
           "o1_a1_state_match_maxdiff": {"max_abs": max(match_abs) if match_abs else None,
                                         "max_rel": max(match_rel) if match_rel else None},
           "sites": [[lab, list(pt), j] for lab, pt, j in M.sites], "layers": M.layers_desc,
           "loop_ends": [[list(pt), j] for pt, j in M.loop_ends],
           "para_alignment": {"aligned": modes.count("aligned"), "last_token": modes.count("last_token")},
           "clean_batch_vs_single_max_logit_gap": gap,
           "structural_zero_max_abs_effect": max(struct_zero) if struct_zero else None,
           "dtype": str(next(M.m.parameters()).dtype),
           "cells": summ, "items": items, "load_seconds": round(load_s, 1),
           "seconds_total": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(out_dir, "k15_%s.json" % name), "w"))
    print("[k15 %s] done pairs %d scanned %d load %.0fs total %.0fs gap %.4f" % (
        name, len(items), scanned, load_s, time.time() - t0, gap), flush=True)
    del M
    gc.collect()
    torch.cuda.empty_cache()


def spent_minutes(out_dir):
    s = 0.0
    for f in glob.glob(os.path.join(out_dir, "proc_*.json")):
        s += json.load(open(f))["seconds"]
    return s / 60.0


# ------------------------------------------------------------------ summary
def summarize(out_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "k15_*.json"))):
        d = json.load(open(f))
        res[d["model"]] = d
    procs = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "proc_*.json"))}
    gpu_min = round(sum(p["seconds"] for p in procs.values()) / 60.0, 2)
    skipped = [m for p in procs.values() for m in p.get("skipped", [])]
    json.dump({"models": res, "processes": procs}, open(os.path.join(out_dir, "results.json"), "w"))

    def cell(m, op, a, site, f="flip"):
        c = res.get(m, {}).get("cells", {}).get(cell_key(op, a, site))
        return None if c is None else c[f]

    def r3(x):
        return None if x is None else round(x, 3)
    o4 = [c["flip"] for d in res.values() for k, c in d["cells"].items() if k.startswith("O4|")]
    checks = {
        "pairs_per_model": {m: d["pairs_passed"] for m, d in res.items()},
        "o5_effect_per_model": {m: r3(cell(m, "O5", 1.0, "last_pos", "effect")) for m in res},
        "o4_max_flip_rate": r3(max(o4)) if o4 else None,
        "ouro_o1_loop1_mid_a05_flips": r3(cell("ouro", "O1", 0.5, "Lmid@1")),
        "ouro_o1_loop1_last_a05_flips": r3(cell("ouro", "O1", 0.5, "Llast@1")),
        "ifm_noinj_o1_loop1_a1_flips": r3(cell("ifm_p1_ouro", "O1", 1.0, "Llast@1")),
        "ifm_inj_o1_loop1_a1_flips": r3(cell("ifm_p1_ouro_inj", "O1", 1.0, "Llast@1")),
        "gpu_minutes": gpu_min,
        "structural_zero_max_abs_effect": {m: d["structural_zero_max_abs_effect"] for m, d in res.items()},
        "clean_batch_vs_single_max_logit_gap": {m: r3(d["clean_batch_vs_single_max_logit_gap"]) for m, d in res.items()},
        "para_alignment": {m: d["para_alignment"] for m, d in res.items()},
        "models_skipped": skipped,
        "pairs_by_relation": {m: d.get("pairs_by_relation") for m, d in res.items()},
        "subjects_per_model": {m: d.get("n_subjects") for m, d in res.items()},
        "o1_a1_state_match_maxdiff": {m: d.get("o1_a1_state_match_maxdiff") for m, d in res.items()},
        "loops_run": {m: d.get("loops_run") for m, d in res.items()},
        "readout": {m: d.get("readout") for m, d in res.items()},
    }
    json.dump(checks, open(os.path.join(out_dir, "checks.json"), "w"), indent=1)

    def f2(x):
        return "-" if x is None else "%.2f" % x
    lines = ["# K15 smoke summary", "",
             "Cell = mean effect / flip rate (k13 definitions). Held-out pairs the clean model answers. "
             "Sites: L0 / Lmid / Llast @ loop. O5 = O1 a=1 at the last position, last layer, last loop. "
             "Llast@K (and d28 Llast@1, d112 Llast@4) edits subject positions after the last layer: zero by construction.",
             "", "GPU minutes: %s. Skipped: %s." % (gpu_min, ", ".join(skipped) or "none"), ""]
    for m, d in res.items():
        sites = [s[0] for s in d["sites"]]
        lines += ["## %s (pairs %d of %d held-out scanned, %s, %s subjects; para v: %s)" % (
            m, d["pairs_passed"], d["heldout_scanned"], d.get("pairs_by_relation"), d.get("n_subjects"),
            d["para_alignment"]), "",
            "O1 a=1 state match vs clean B at later sites: %s" % (d.get("o1_a1_state_match_maxdiff"),), "",
            "O5 effect %s, flips %s. Layers: %s" % (f2(cell(m, "O5", 1.0, "last_pos", "effect")),
                                                   f2(cell(m, "O5", 1.0, "last_pos")), d["layers"]), ""]
        for title, fn in (("change in logit A / logit B (edit minus clean)", lambda op, a, s: "%s / %s" % (
                              f2(cell(m, op, a, s, "dA")), f2(cell(m, op, a, s, "dB")))),
                          ("effect / flips", lambda op, a, s: "%s / %s" % (f2(cell(m, op, a, s, "effect")), f2(cell(m, op, a, s)))),
                          ("off-target KL", lambda op, a, s: f2(cell(m, op, a, s, "kl"))),
                          ("survival at the last loop end", lambda op, a, s: f2(
                              (lambda c: None if c is None else list(c.values())[-1])(cell(m, op, a, s, "survival_by_loop_end"))))):
            lines += ["%s:" % title, "", "| op a | " + " | ".join(sites) + " |", "|---" * (len(sites) + 1) + "|"]
            for op, a in GRID:
                lines.append("| %s %s | " % (op, fmt_a(a)) + " | ".join(fn(op, a, s) for s in sites) + " |")
            lines.append("")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines))
    print(json.dumps(checks), flush=True)


def main():
    global N_PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="p")
    ap.add_argument("--n_pairs", type=int, default=N_PAIRS)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
        return
    N_PAIRS = a.n_pairs
    t0 = time.time()
    prior = spent_minutes(a.out)
    skipped = []
    try:
        for name in [x for x in a.models.split(",") if x]:
            used = prior + (time.time() - t0) / 60.0
            if name == "ifm_d112" and used + D112_RESERVE_MIN > CAP_MIN:
                print("[k15] skip ifm_d112: %.1f GPU-min used" % used, flush=True)
                skipped.append(name)
                continue
            run_model(name, a.out, t0)
    finally:
        json.dump({"models": a.models, "seconds": round(time.time() - t0, 1), "skipped": skipped},
                  open(os.path.join(a.out, "proc_%s.json" % a.tag), "w"))


if __name__ == "__main__":
    main()
