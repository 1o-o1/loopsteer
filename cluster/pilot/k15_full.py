"""K15 full A (brief B-K15-FULL-A): the steerability map at scale. Plan: loop-signal/notes/k15_fullA_plan.md.
Builds on k15_smoke.py and k15_smoke2.py (imported, never modified); datasets from k15_data.py.

Datasets: D1 (smoke 2's 80 capital/city pairs), CF (CounterFact azhx/counterfact, 5 relations, up to 80 pairs
each), D2 (two-digit addition, 80 pairs). Sites L0 / Lmid / Llast at loops 1, 2, K (d112 mapped layers).
Arms per pair (one right-padded batch, row 0 clean A; rows with final top-1 margin < TAU rerun alone at batch 1):
  S   O2 (paraphrase vector at the subject/operand positions) and O4 (random, norm a|v|), a in {0.5, 1, 2}
  A   h[last] += a v_last (paraphrase vector at the last position), a in {1, 2, 4}; readout at every loop end
      from the edited loop on: own = delta m at the edited loop's end / D, final = delta m at loop K / D
  C   O1 at Llast@1 (subject positions), a = 0.30..0.70 step 0.05
  O5  O1 a=1 at the last position, final point (positive control)
Arm B: sentiment CAA at Lmid (smoke 2 recipe), 150 neutral prompts, schedules L1 / L2 / LK / all x a in
{0.05, 0.1, 0.2, 0.4}, random control at all loops; matched total strength L1 at 4a vs all at a.

    python k15_full.py --selftest                                    (CPU, fake model)
    python k15_full.py --models ouro --out OUT --cf DATA.json         (Ouro venv)
    python k15_full.py --models ifm_p1_ouro,ifm_p1_ouro_inj,ifm_d112 --out OUT --cf DATA.json   (xllm_venv)
    python k15_full.py --score --out OUT --clf DIR
    python k15_full.py --summarize --out OUT
"""
import argparse
import gc
import glob
import json
import math
import os
import socket
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import k15_data as KD  # noqa: E402
import k15_smoke as S1  # noqa: E402
import k15_smoke2 as S2  # noqa: E402

# Hub revisions pinned for the cluster run (values from cluster/weights.json); a local folder path gets None
OURO_REVISION = {"ByteDance/Ouro-1.4B": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
                 "ByteDance/Ouro-2.6B": "1ed04250da1a9936042725d302e81c8fa2ab5abd",
                 "ByteDance/Ouro-1.4B-Thinking": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
                 "ByteDance/Ouro-2.6B-Thinking": "f1edd81e7ac41355db670500ceaf204e0f73af68"}
RUN_ROLE = os.environ.get("RUN_ROLE", "cluster")   # written into records instead of the host name
FULL_POOL = [False]     # cluster copy (--full_pool): lift the per-dataset quotas; every candidate is scanned
CF_POOL = [""]          # cluster copy (--cf_pool FILE): CF candidates from the shared pool (cluster/pilot/cf_pool.py)

A_S = [0.5, 1.0, 2.0]
A_A = [1.0, 2.0, 4.0]
A_C = list(S2.A_C)
A_B = [0.05, 0.1, 0.2, 0.4]
MATCHED = [(0.05, 0.2), (0.1, 0.4)]          # (all-loops a, single-loop 4a)
TAU = 1.0
THR = 0.25
GAP_MIN = 1.0                                 # logits: own-loop clean gap below this -> arm A item "unreadable"
CLASSES = ("never_read", "rederived", "persisted", "delayed", "unreadable")
GEN_TOKENS = 24
PB = 10
KNOW_BATCH = 96
NBOOT = 1000
MODEL_ORDER = ["ouro", "ouro26", "ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_d112", "ifm_d28"]   # d28: cluster fill
DATASETS = ["D1", "D2", "CF"]
KNOWN_DATASETS = ["D1", "D2", "CF", "D4", "D5"]     # --datasets choices; seed0 by this order (D1, D2, CF as before)
D4_DIR = [""]           # cluster extension (--data_dir): the k16_path_ifm data folder (d4/d4_ifm.jsonl and
#                         d5/d5_ifm.jsonl, IFM tokenizer)


def d45_candidates(ds, enc=None):
    """D4 (code: read and sum) or D5 (two-step) pairs as k16_path_ifm builds them (its rows, paraphrases and template
    cap 1), on the IFM tokenizer file; with enc, every row's prompt + suffix must encode to its ids (k16_path_ifm's
    check)."""
    import k16_path_ifm as K16I
    cands = K16I.d45_candidates(ds, D4_DIR[0])
    if enc is not None:
        bad = [p["id"] for p in cands if enc(p["prompt_A"]) + p["suffix_ids"] != p["ids_A"]
               or enc(p["prompt_B"]) + p["suffix_ids"] != p["ids_B"]]
        if bad:
            raise SystemExit("%s tokenization mismatch on %d of %d rows (first %s)" % (ds, len(bad), len(cands), bad[:3]))
    return cands, dict(K16I.D45_QUOTA[ds]), K16I.d45_select


class Budget(Exception):
    pass


def fa(a):
    return "%g" % a


def ckey(arm, op, a, site):
    return "%s|%s|%s|%s" % (arm, op, fa(a), site)


def r5(x):
    return None if x is None else round(float(x), 5)


# ------------------------------------------------------------------ statistics (pure)
def boot_ci(xs, stat="mean", n=NBOOT, seed=0):
    """95% percentile bootstrap over items. stat: mean | mid | bc. Returns [value, lo, hi] or None."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    t = torch.tensor(xs, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(xs), (n, len(xs)), generator=g)
    s = t[idx]

    def f(v):                                     # v [..., m]
        if stat == "mean":
            return v.mean(-1)
        if stat == "mid":
            return ((v >= 0.2) & (v <= 0.8)).double().mean(-1)
        if stat == "bc":
            m_ = v.shape[-1]
            if m_ < 4:
                return torch.full(v.shape[:-1], float("nan"), dtype=torch.float64)
            mu = v.mean(-1, keepdim=True)
            s2 = ((v - mu) ** 2).mean(-1)
            g1 = ((v - mu) ** 3).mean(-1) / s2.clamp_min(1e-300) ** 1.5
            k = ((v - mu) ** 4).mean(-1) / s2.clamp_min(1e-300) ** 2 - 3.0
            G = g1 * math.sqrt(m_ * (m_ - 1)) / (m_ - 2)
            Kc = ((m_ + 1) * k + 6.0) * (m_ - 1) / ((m_ - 2) * (m_ - 3))
            out = (G * G + 1.0) / (Kc + 3.0 * (m_ - 1) ** 2 / ((m_ - 2) * (m_ - 3)))
            return torch.where(s2 > 0, out, torch.full_like(out, float("nan")))
        raise KeyError(stat)
    v0 = float(f(t))
    bs = f(s)
    bs = bs[~torch.isnan(bs)]
    if v0 != v0:
        return None
    if len(bs) == 0:
        return [round(v0, 4), None, None]
    q = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [round(v0, 4), round(float(q[0]), 4), round(float(q[1]), 4)]


def boot_diff(xs, ys, n=NBOOT, seed=0):
    """Paired bootstrap of mean(xs - ys) over items (None pairs dropped)."""
    d = [x - y for x, y in zip(xs, ys) if x is not None and y is not None]
    return boot_ci(d, "mean", n, seed)


def a_class(own, fin, thr=THR):
    """own: k13 immediate effect at the edited loop (None when the clean gap there is < GAP_MIN logits)."""
    if fin is None:
        return None
    if own is None:
        return "unreadable"
    if own < thr:
        return "never_read" if fin < thr else "delayed"
    return "rederived" if fin < thr else "persisted"


# ------------------------------------------------------------------ models
def load_ouro(repo):
    """k15_smoke.OuroM with another Ouro checkpoint (k13 OuroEditor swapped for the call)."""
    import k13_edit
    orig = k13_edit.OuroEditor

    class Ed(orig):
        def __init__(self):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, revision=OURO_REVISION.get(repo))
            self.m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, revision=OURO_REVISION.get(repo),
                                                          dtype=torch.bfloat16).cuda().eval()
            self.K = 4
            self.W = self.m.lm_head.weight
    k13_edit.OuroEditor = Ed
    try:
        M = S1.OuroM()
    finally:
        k13_edit.OuroEditor = orig
    M.repo = repo
    return M


def load_model(name):
    if name == "ouro26":
        return load_ouro("ByteDance/Ouro-2.6B")
    return S1.load_model(name)


# ------------------------------------------------------------------ per pair
def para_vectors(M, p, site_pts):
    """vsub[pt] [P, H] (smoke-1 rule: aligned or last-token broadcast), vlast[pt] [H]; means over paraphrases."""
    P = len(p["pos"])
    vs, vl, modes = {pt: [] for pt in site_pts}, {pt: [] for pt in site_pts}, []
    for ia, ib in p["para"]:
        sa, sb = S1.subj_span(ia, ib)
        if not sa or not sb:
            modes.append("empty")
            continue
        ra, rb = S2.run2(M, [ia], record=set(site_pts)), S2.run2(M, [ib], record=set(site_pts))
        aligned = len(sa) == P and len(sb) == P
        modes.append("aligned" if aligned else "last_token")
        for pt in site_pts:
            vl[pt].append(rb[pt][0, -1].float() - ra[pt][0, -1].float())
            if aligned:
                vs[pt].append(rb[pt][0, sb].float() - ra[pt][0, sa].float())
            else:
                vs[pt].append((rb[pt][0, sb[-1]].float() - ra[pt][0, sa[-1]].float()).expand(P, -1))
    if not vl[site_pts[0]]:
        return None, None, modes
    return ({pt: torch.stack(x).mean(0) for pt, x in vs.items()},
            {pt: torch.stack(x).mean(0) for pt, x in vl.items()}, modes)


def build_specs(M, p, rB, vsub, vlast, seed, ctx):
    pos, last = p["pos"], len(p["A"]) - 1
    specs = []
    for si, (lab, pt, j) in enumerate(ctx["sites"]):
        v = vsub[pt]
        u = S1.unit_random(tuple(v.shape), seed * 1009 + si).to(v.device)
        for a in A_S:
            specs.append({"arm": "S", "op": "O2", "a": a, "site": lab, "j": j, "kind": "add", "pt": pt, "pos": pos,
                          "delta": a * v})
            specs.append({"arm": "S", "op": "O4", "a": a, "site": lab, "j": j, "kind": "add", "pt": pt, "pos": pos,
                          "delta": a * v.norm(dim=-1, keepdim=True) * u})
        for a in A_A:
            specs.append({"arm": "A", "op": "add", "a": a, "site": lab, "j": j, "kind": "add", "pt": pt,
                          "pos": [last], "delta": (a * vlast[pt])[None]})
    ptc = ctx["lab2pt"]["Llast@1"]
    for a in A_C:
        specs.append({"arm": "C", "op": "O1", "a": a, "site": "Llast@1", "j": 1, "kind": "lerp", "pt": ptc,
                      "pos": pos, "tgt": rB[ptc][0, pos].float()})
    fin = ctx["final"]
    specs.append({"arm": "O5", "op": "O1", "a": 1.0, "site": "last_pos", "j": ctx["K"], "kind": "lerp", "pt": fin,
                  "pos": [last], "tgt": rB[fin][0, [last]].float()})
    return specs


@torch.no_grad()
def process_pair(M, p, rA, rB, seed, ctx, cells, tau=TAU):
    """Runs all arms on one known pair. Appends per-row values to cells. Returns the item record or None."""
    dev = M.dev
    le = ctx["loop_ends"]
    le_pts = [pe for pe, _ in le]
    final = ctx["final"]
    last = len(p["A"]) - 1
    tA, tB = p["tA"], p["tB"]
    vsub, vlast, modes = para_vectors(M, p, ctx["site_pts"])
    if vsub is None:
        return None
    zA = {pe: S2.readout_last(M, rA[pe], [last])[0] for pe in le_pts}
    zB = {pe: S2.readout_last(M, rB[pe], [last])[0] for pe in le_pts}
    mB = float(zB[final][tB] - zB[final][tA])
    specs = build_specs(M, p, rB, vsub, vlast, seed, ctx)
    ed, ij = S2.build_edits(specs, 1, dev)
    n = len(specs)
    rec = S2.run2(M, [p["A"]] * (1 + n), ed, set(le_pts), ij)
    z = {pe: S2.readout_last(M, rec[pe], [last] * (1 + n)) for pe in le_pts}
    hl = {pe: rec[pe][:, last].float() for pe in le_pts}          # last-position state at each loop end (in batch)
    del rec
    gap = float((z[final][0] - zA[final]).abs().max())
    reruns = 0
    for r_, sp in enumerate(specs):
        zr = {pe: z[pe][r_ + 1] for pe in le_pts}
        base = {pe: z[pe][0] for pe in le_pts}
        rr = 0
        if S2.top_margin(zr[final]) < tau:
            e1, i1 = S2.build_edits([sp], 0, dev)
            rec1 = S2.run2(M, [p["A"]], e1, set(le_pts), i1)
            zr = {pe: S2.readout_last(M, rec1[pe], [last])[0] for pe in le_pts}
            base, rr = zA, 1
            reruns += 1
        mA = float(base[final][tB] - base[final][tA])
        D = mB - mA
        ok = abs(D) > 1e-3
        m_e = float(zr[final][tB] - zr[final][tA])
        c = cells.setdefault(ckey(sp["arm"], sp["op"], sp["a"], sp["site"]),
                             {"e": [], "flip": [], "dA": [], "dB": [], "rr": [], "it": []})
        c["e"].append(r5((m_e - mA) / D) if ok else None)
        c["flip"].append(int(int(zr[final].argmax()) == tB))
        c["dA"].append(r5(zr[final][tA] - base[final][tA]))
        c["dB"].append(r5(zr[final][tB] - base[final][tB]))
        c["rr"].append(rr)
        c["it"].append(ctx["n_items"])
        if sp["arm"] == "A":
            traj, dms = [], []
            for pe, je in le:
                if je < sp["j"]:
                    continue
                dm = float(zr[pe][tB] - zr[pe][tA]) - float(base[pe][tB] - base[pe][tA])
                dms.append(dm)
                traj.append(r5(dm / D) if ok else None)
            own_pe = [pe for pe, je in le if je == sp["j"]][0]
            # k13 "immediate": change at the edited loop's end over the clean B-minus-A gap at that same loop end
            gap_own = float(zB[own_pe][tB] - zB[own_pe][tA]) - float(base[own_pe][tB] - base[own_pe][tA])
            c.setdefault("own", []).append(r5(dms[0] / gap_own) if abs(gap_own) >= GAP_MIN else None)
            c.setdefault("own_D", []).append(traj[0])
            c.setdefault("gap_own", []).append(r5(gap_own))
            c.setdefault("fin", []).append(traj[-1])
            c.setdefault("traj", []).append(traj)
            c.setdefault("dm", []).append([r5(x) for x in dms])
            c.setdefault("fown", []).append(int(int(zr[own_pe].argmax()) == tB))
            # state diagnostic (in-batch rows): |h_edit - h_clean| at the last position at each later loop end over
            # |a v_last|, and its cosine with a v_last
            dv = sp["delta"][0].float()
            sv, cv = [], []
            for pe, je in le:
                if je < sp["j"]:
                    continue
                dh = hl[pe][r_ + 1] - hl[pe][0]
                sv.append(r5(dh.norm() / dv.norm().clamp_min(1e-12)))
                cv.append(r5(torch.nn.functional.cosine_similarity(dh, dv, dim=0)))
            c.setdefault("surv", []).append(sv)
            c.setdefault("cosv", []).append(cv)
    D0 = mB - float(zA[final][tB] - zA[final][tA])
    item = {"a": p["a"], "b": p["b"], "rel": p["rel"], "half": p["half"], "modes": modes, "mB": r5(mB),
            "mA_single": r5(zA[final][tB] - zA[final][tA]), "pos_len": len(p["pos"]), "T": len(p["A"]),
            "clean_gap_by_loop": [r5((float(zB[pe][tB] - zB[pe][tA]) - float(zA[pe][tB] - zA[pe][tA])) / D0)
                                  if abs(D0) > 1e-3 else None for pe in le_pts],
            "mA_by_loop": [r5(zA[pe][tB] - zA[pe][tA]) for pe in le_pts],
            "mB_by_loop": [r5(zB[pe][tB] - zB[pe][tA]) for pe in le_pts],
            "reruns": reruns, "rows": n, "batch_gap": r5(gap)}
    if p.get("dataset") in ("d4", "d5"):   # D4, D5 rows (k16_path_ifm data) keep their id: answers repeat across templates
        item["id"] = p["id"]
    return item


@torch.no_grad()
def know_table(M, prompts, final):
    """top-1 at the last position for each unique prompt (right-padded batches)."""
    uniq = sorted(set(tuple(x) for x in prompts), key=lambda t: (len(t), t))
    out = {}
    for i in range(0, len(uniq), KNOW_BATCH):
        rows = [list(t) for t in uniq[i:i + KNOW_BATCH]]
        z = S2.readout_last(M, S2.run2(M, rows, record={final})[final], [len(r) - 1 for r in rows])
        for t, k in zip(uniq[i:i + KNOW_BATCH], z.argmax(-1).tolist()):
            out[t] = int(k)
    return out


def record_pass(cands, table):
    """Record (unique prompt) level clean pass rate from the batched top-1 table: {rel: [known, n], "all": [k, n]}."""
    recs = {}
    for p in cands:
        recs[tuple(p["A"])] = (p["tA"], p["rel"])
        recs[tuple(p["B"])] = (p["tB"], p["rel"])
    out = {}
    for ids, (t, rel) in recs.items():
        for k in (rel, "all"):
            o = out.setdefault(k, [0, 0])
            o[0] += int(table[ids] == t)
            o[1] += 1
    return out


def run_dataset(M, ds, cands, quota, ctx, deadline, use_table, seed0, selector=None):
    final = ctx["final"]
    rec_pts = set(ctx["site_pts"]) | set(pe for pe, _ in ctx["loop_ends"])
    t0 = time.time()
    table = know_table(M, [p["A"] for p in cands] + [p["B"] for p in cands], final) if use_table else None
    t_table = time.time() - t0
    rpass = record_pass(cands, table) if table is not None else None
    if rpass is not None:
        print("[k15f %s] %s record-level clean pass (batched table): %s" % (ctx["name"], ds, rpass), flush=True)
    cells, items = {}, []
    st = {"table_rejects": 0, "single_rejects": 0, "table_pass_single_fail": 0, "no_para": 0}

    def know(p):
        if deadline is not None and time.time() > deadline:
            raise Budget()
        if table is not None and not (table[tuple(p["A"])] == p["tA"] and table[tuple(p["B"])] == p["tB"]):
            st["table_rejects"] += 1
            return False
        rA = S2.run2(M, [p["A"]], record=rec_pts)
        zA = S2.readout_last(M, rA[final], [len(p["A"]) - 1])[0]
        if int(zA.argmax()) != p["tA"]:
            st["single_rejects"] += 1
            st["table_pass_single_fail"] += int(table is not None)
            return False
        rB = S2.run2(M, [p["B"]], record=rec_pts)
        zB = S2.readout_last(M, rB[final], [len(p["B"]) - 1])[0]
        if int(zB.argmax()) != p["tB"]:
            st["single_rejects"] += 1
            st["table_pass_single_fail"] += int(table is not None)
            return False
        ctx["n_items"] = len(items)
        it = process_pair(M, p, rA, rB, seed0 + len(items), ctx, cells)
        if it is None:
            st["no_para"] += 1
            return False
        items.append(it)
        if len(items) % 10 == 0:
            print("[k15f %s] %s kept %d %.0fs" % (ctx["name"], ds, len(items), time.time() - t0), flush=True)
        return True
    truncated = False
    scanned = None
    try:
        if selector is not None:                            # D4, D5: k16_path_ifm's selection (template cap 1)
            kept, scanned = selector(cands, know, quota)
        else:
            kept, scanned = (KD.select(cands, know, quota, cap=10 ** 9) if (CF_POOL[0] and ds == "CF")
                             else KD.select(cands, know, quota))     # shared pool: no subject cap (as production)
    except Budget:
        truncated = True
    by_rel = {}
    for it in items:
        by_rel[it["rel"]] = by_rel.get(it["rel"], 0) + 1
    modes = [m for it in items for m in it["modes"]]
    out = {"pairs": len(items), "scanned": scanned, "candidates": len(cands), "by_rel": by_rel,
           "quota": {("%s:%s" % k if isinstance(k, tuple) else k): v for k, v in quota.items()}, "truncated": truncated, "know_stats": st,
           "para_modes": {m: modes.count(m) for m in sorted(set(modes))},
           "reruns": sum(it["reruns"] for it in items), "rows": sum(it["rows"] for it in items),
           "batch_gap_max": max([it["batch_gap"] for it in items] or [0.0]),
           "seconds": round(time.time() - t0, 1), "table_seconds": round(t_table, 1), "record_pass": rpass,
           "items": items, "cells": cells}
    print("[k15f %s] %s done pairs %d scanned %s reruns %d/%d %.0fs truncated %s" % (
        ctx["name"], ds, len(items), scanned, out["reruns"], out["rows"], time.time() - t0, truncated), flush=True)
    print("[k15f %s] %s paraphrase vectors: last-token fallback %d of %d (modes %s)" % (
        ctx["name"], ds, out["para_modes"].get("last_token", 0), len(modes), out["para_modes"]), flush=True)
    return out


# ------------------------------------------------------------------ arm B
@torch.no_grad()
def armB_vectors(M, name, K):
    """Smoke 2 recipe: v_s, n_s at Lmid of loop s from 'The {noun} was {adj}' pairs (final token)."""
    dev = M.dev
    mpts = S2.mid_pts(M, name, K)
    single = lambda w: len(M.enc(" " + w, ans=True)) == 1  # noqa: E731
    pos_w = [w for w in S2.POS if single(w)][:S2.N_ADJ]
    neg_w = [w for w in S2.NEG if single(w)][:S2.N_ADJ]
    nadj = min(len(pos_w), len(neg_w))
    texts, labels = [], []
    for n_ in S2.NOUNS:
        for k_ in range(nadj):
            texts += ["The %s was %s" % (n_, pos_w[k_]), "The %s was %s" % (n_, neg_w[k_])]
            labels += [1, 0]
    ids = [M.enc(t) for t in texts]
    rec = S2.run2(M, ids, record=set(mpts))
    lastp = torch.tensor([len(x) - 1 for x in ids], device=dev)
    lab_t = torch.tensor(labels, device=dev)
    V, N = [], []
    for pt in mpts:
        h = rec[pt][torch.arange(len(ids), device=dev), lastp].float()
        V.append(h[lab_t == 1].mean(0) - h[lab_t == 0].mean(0))
        N.append(float(h.norm(dim=-1).mean()))
    g = torch.Generator(device="cpu").manual_seed(12345)
    Ur = [(lambda r_: r_ / r_.norm())(torch.randn(V[0].shape[0], generator=g)).to(dev) for _ in mpts]
    meta = {"site_points": [list(pt) for pt in mpts], "adjectives": {"pos": pos_w[:nadj], "neg": neg_w[:nadj]},
            "n_contrast_pairs": len(texts) // 2, "v_norm": [float(v.norm()) for v in V], "resid_norm": N,
            "v_cos_between_loops": {"%d-%d" % (a + 1, b + 1): float(torch.nn.functional.cosine_similarity(V[a], V[b], dim=0))
                                    for a in range(len(V)) for b in range(a + 1, len(V))}}
    return mpts, V, N, Ur, meta


def armB_conds(K):
    scheds = ([("L1", [1]), ("L2", [2]), ("L%d" % K, [K]), ("all", list(range(1, K + 1)))] if K > 1 else [("L1", [1])])
    conds = [("clean", None, 0.0, "caa")]
    for sn, loops in scheds:
        for a in A_B:
            conds.append((sn, loops, a, "caa"))
    for a in A_B:
        conds.append(("all" if K > 1 else "L1", list(range(1, K + 1)), a, "random"))
    return conds


def armB_edits(conds, mpts, V, N, Ur, row_blocks, dev):
    groups = {}
    for c, (sn, loops, a, kind) in enumerate(conds):
        if loops is None or not row_blocks[c]:
            continue
        for s in loops:
            d = S2.caa_delta(V[s - 1], N[s - 1], a) if kind == "caa" else a * N[s - 1] * Ur[s - 1]
            gg = groups.setdefault(mpts[s - 1], ([], []))
            gg[0].extend(row_blocks[c])
            gg[1].append(d[None].expand(len(row_blocks[c]), -1))
    return {pt: [("add_all", torch.tensor(gg[0], device=dev), torch.cat(gg[1]))] for pt, gg in groups.items()}


@torch.no_grad()
def generate_rows(M, prompts_ids, conds, mpts, V, N, Ur, final, eos, gen_tokens=GEN_TOKENS):
    """Greedy generation for every (prompt, condition) row of one batch. Returns rows[pi][c] dicts."""
    dev = M.dev
    C = len(conds)
    P = len(prompts_ids)
    seqs = [list(prompts_ids[pi]) for pi in range(P) for _ in range(C)]
    T0 = [len(s) for s in seqs]
    ed = armB_edits(conds, mpts, V, N, Ur, [[pi * C + c for pi in range(P)] for c in range(C)], dev)
    for _ in range(gen_tokens):
        rr = S2.run2(M, seqs, ed, {final})
        nxt = S2.readout_last(M, rr[final], [len(s) - 1 for s in seqs]).argmax(-1).tolist()
        del rr
        for i_, t in enumerate(nxt):
            seqs[i_].append(int(t))
    conts = []
    for i_, s in enumerate(seqs):
        cont = s[T0[i_]:]
        if eos in cont:
            cont = cont[:cont.index(eos)]
        conts.append(cont)
    # clean-model NLL of each continuation
    h = S2.run2(M, seqs, None, {final})[final]
    rows_i, pos_i, tok_i = [], [], []
    for i_, cont in enumerate(conts):
        for k_, t in enumerate(cont):
            rows_i.append(i_)
            pos_i.append(T0[i_] - 1 + k_)
            tok_i.append(t)
    nll_tok = torch.zeros(len(tok_i), device=dev)
    for c0 in range(0, len(tok_i), 256):
        hi = h[torch.tensor(rows_i[c0:c0 + 256], device=dev), torch.tensor(pos_i[c0:c0 + 256], device=dev)]
        lp = torch.log_softmax(M.readout(hi).float(), -1)
        nll_tok[c0:c0 + 256] = -lp[torch.arange(hi.shape[0], device=dev), torch.tensor(tok_i[c0:c0 + 256], device=dev)]
    del h
    sums, cnts = [0.0] * len(seqs), [0] * len(seqs)
    for k_, i_ in enumerate(rows_i):
        sums[i_] += float(nll_tok[k_])
        cnts[i_] += 1
    out = []
    for pi in range(P):
        row = []
        for c in range(C):
            i_ = pi * C + c
            row.append({"text": S2.decode(M, conts[i_]), "n_tok": len(conts[i_]),
                        "nll": r5(sums[i_] / cnts[i_]) if cnts[i_] else None})
        out.append(row)
    return out


@torch.no_grad()
def run_armB(M, name, K, final, prompts, pb=PB):
    t0 = time.time()
    dev = M.dev
    mpts, V, N, Ur, meta = armB_vectors(M, name, K)
    conds = armB_conds(K)
    C = len(conds)
    eos = S2.eos_id(M)
    gens = []
    for c0 in range(0, len(prompts), pb):
        gens += generate_rows(M, [M.enc(p) for p in prompts[c0:c0 + pb]], conds, mpts, V, N, Ur, final, eos)
        print("[k15f %s] armB %d/%d prompts %.0fs" % (name, len(gens), len(prompts), time.time() - t0), flush=True)
    U = [M.enc(p) for p, _ in S2.UNRELATED]
    U_last = [len(u) - 1 for u in U]
    nU = len(U)
    rows = [u for _ in range(C) for u in U]
    edd = armB_edits(conds, mpts, V, N, Ur, [list(range(c * nU, (c + 1) * nU)) for c in range(C)], dev)
    zd = S2.readout_last(M, S2.run2(M, rows, edd, {final})[final], [x for _ in range(C) for x in U_last])
    ctop = zd[:nU].argmax(-1)
    dmg = []
    for c in range(C):
        blk = zd[c * nU:(c + 1) * nU]
        dmg.append({"agree": int((blk.argmax(-1) == ctop).sum()), "n": nU,
                    "undecided": int(sum(1 for k_ in range(nU) if S2.top_margin(blk[k_]) < TAU)),
                    "per_prompt": (blk.argmax(-1) == ctop).int().tolist()})
    meta.update({"conds": [{"sched": sn, "loops": loops, "a": a, "vector": kind} for sn, loops, a, kind in conds],
                 "prompts": list(prompts), "gens": gens, "damage": dmg, "seconds": round(time.time() - t0, 1),
                 "gen_tokens": GEN_TOKENS, "unrelated": [p for p, _ in S2.UNRELATED]})
    return meta


# ------------------------------------------------------------------ one model
def site_ctx(M, name):
    sites = list(M.sites)
    le = list(M.loop_ends)
    return {"name": name, "sites": sites, "site_pts": [pt for _, pt, _ in sites],
            "lab2pt": dict((lab, pt) for lab, pt, _ in sites), "loop_ends": le, "final": le[-1][0], "K": le[-1][1],
            "n_items": 0}


def run_model(name, out_dir, cf_path, pair_budget_min, debug, n_neutral):
    t0 = time.time()
    M = load_model(name)
    load_s = time.time() - t0
    ctx = site_ctx(M, name)
    deadline = t0 + 60.0 * pair_budget_min if pair_budget_min > 0 else None
    res = {"model": name, "host": RUN_ROLE, "K": ctx["K"],
           "sites": [[lab, list(pt), j] for lab, pt, j in ctx["sites"]], "layers": M.layers_desc,
           "loop_ends": [[list(pe), je] for pe, je in ctx["loop_ends"]], "readout": M.readout_desc,
           "dtype": str(next(M.m.parameters()).dtype), "load_seconds": round(load_s, 1), "datasets": {},
           "debug": debug, "tau": TAU, "thr": THR, "pair_budget_min": pair_budget_min}
    recs = json.load(open(cf_path))["records"]
    path = os.path.join(out_dir, "k15f_%s.json" % name)

    def dump(partial):
        res["partial"] = partial
        res["seconds_total"] = round(time.time() - t0, 1)
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    for ds in DATASETS:
        sel = None
        if ds == "D1":
            cands, quota, use_table, dl = KD.d1_candidates(M.enc), dict(KD.D1_QUOTA), False, None
        elif ds == "D2":
            cands, quota, use_table, dl = KD.d2_candidates(M.enc), dict(KD.D2_QUOTA), True, deadline
        elif ds in ("D4", "D5"):
            if not name.startswith("ifm_"):
                raise SystemExit("%s runs on the IFM tokenizer file only (%s_ifm.jsonl): %s" % (ds, ds.lower(), name))
            cands, quota, sel = d45_candidates(ds, M.enc)
            use_table, dl = True, deadline
        else:
            if CF_POOL[0]:
                import cf_pool as CFP
                cands, cst = CFP.pool_candidates(M.enc, CFP.load_pool(CF_POOL[0]))
            else:
                cands, cst = KD.cf_candidates(M.enc, recs)
            quota, use_table, dl = KD.cf_quota(), True, deadline
        quota = {k: 10 ** 9 for k in quota} if FULL_POOL[0] else quota
        if debug:
            quota = {k: min(v, 2) for k, v in quota.items()}
        out = run_dataset(M, ds, cands, quota, ctx, dl, use_table, seed0=1000 * (KNOWN_DATASETS.index(ds) + 1),
                          selector=sel)
        if ds == "CF":
            out["cand_stats"] = cst
        res["datasets"][ds] = out
        dump(True)
    prompts = KD.NEUTRAL150[:n_neutral]
    res["armB"] = run_armB(M, name, ctx["K"], ctx["final"], prompts)
    dump(False)
    print("[k15f %s] done %.0fs" % (name, time.time() - t0), flush=True)
    del M
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def know_only(name, out_dir, cf_path):
    t0 = time.time()
    M = load_model(name)
    final = site_ctx(M, name)["final"]
    recs = json.load(open(cf_path))["records"]
    res = {"model": name, "host": RUN_ROLE}
    for ds, cands in (("D2", KD.d2_candidates(M.enc)), ("CF", KD.cf_candidates(M.enc, recs)[0])):
        res[ds] = record_pass(cands, know_table(M, [p["A"] for p in cands] + [p["B"] for p in cands], final))
        print("[k15f %s] %s record-level clean pass (batched table): %s" % (name, ds, res[ds]), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out_dir, "know_%s.json" % name), "w"))


# ------------------------------------------------------------------ scoring
def score(out_dir, clf_dir):
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=clf_dir, tokenizer=clf_dir, device=0 if torch.cuda.is_available() else -1)
    out = {"classifier": S2.CLF, "local_dir": clf_dir, "models": {}}
    for f in sorted(glob.glob(os.path.join(out_dir, "k15f_*.json"))):
        if f.endswith("_scores.json"):
            continue
        d = json.load(open(f))
        if "armB" not in d:
            print("[k15f score] %s has no arm B (partial), skipped" % d["model"], flush=True)
            continue
        G = d["armB"]["gens"]
        texts = [(pi, c, g["text"]) for pi, row in enumerate(G) for c, g in enumerate(row)]
        nonempty = [t for t in texts if t[2].strip()]
        preds = clf([t[2] for t in nonempty], batch_size=64, truncation=True, top_k=None)
        sc = {}
        for (pi, c, _), pr in zip(nonempty, preds):
            pp = [x["score"] for x in pr if x["label"].upper().startswith("POS")][0]
            sc["%d|%d" % (pi, c)] = {"pos": int(pp > 0.5), "p_pos": round(float(pp), 5)}
        out["models"][d["model"]] = sc
        print("[k15f score] %s %d texts (%d empty)" % (d["model"], len(texts), len(texts) - len(nonempty)), flush=True)
    json.dump(out, open(os.path.join(out_dir, "k15f_scores.json"), "w"))


# ------------------------------------------------------------------ summary
def summarize(out_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "k15f_*.json"))):
        if f.endswith("_scores.json"):
            continue
        d = json.load(open(f))
        res[d["model"]] = d
    order = [m for m in MODEL_ORDER if m in res]
    sp = os.path.join(out_dir, "k15f_scores.json")
    scores = json.load(open(sp)) if os.path.exists(sp) else {"models": {}}
    procs = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "proc_*.json"))}
    gpu = {}
    for pn, pd in procs.items():
        gpu[pd.get("host", "?")] = gpu.get(pd.get("host", "?"), 0.0) + pd["seconds"] / 60.0
    gpu = {k: round(v, 2) for k, v in gpu.items()}
    gpu_total = round(sum(gpu.values()), 2)

    def cellv(m, ds, k):
        return res[m]["datasets"][ds]["cells"].get(k)

    def ci(m, ds, k, f="e", stat="mean", items=None):
        c = cellv(m, ds, k)
        if c is None:
            return None
        xs = c[f]
        if items is not None:
            xs = [x for x, it in zip(xs, c["it"]) if it in items]
        return boot_ci(xs, stat)

    def fmt(v, flip=None):
        if v is None:
            return "-"
        s = "%.2f [%s, %s]" % (v[0], "-" if v[1] is None else "%.2f" % v[1], "-" if v[2] is None else "%.2f" % v[2])
        return s if flip is None else "%s / %.2f" % (s, flip[0])

    sites_of = {m: [s[0] for s in res[m]["sites"]] for m in order}
    checks = {"pairs_per_model_dataset": {}, "smoke2_repro": {}, "S_loop1_effect": {}, "A_rederived_vs_neverread": {},
              "C_mid_fraction": {}, "C_bimodality": {}, "B_matched_strength": {}, "B_broken_cells": {},
              "downloads": ["azhx/counterfact (HF hub dataset, 2 parquet files, 12.4 MB, laptop HF cache)"],
              "classifier": S2.CLF + " (existing copy, shared-host out/k15_smoke2/clf, no new download)",
              "gpu_minutes": dict(gpu, total=gpu_total), "o5_effect_per_model": {}, "o4_max_flip_rate": {},
              "structural_zero_max_abs_effect": {}, "reruns": {}, "batch_gap_max": {}, "para_modes": {},
              "truncated": {}, "partial": {}, "para_fallback_last_token": {}, "record_pass": {}}
    L = ["# K15 full A summary", "",
         "Effect e = (m_edit - m_A)/(m_B - m_A), m = logit tB - logit tA at the final readout; flip = top-1 is tB. "
         "Cells: mean [95%% bootstrap CI over items, %d resamples, seed 0]. Rows with final top-1 margin < %.1f logit "
         "rerun alone at batch 1. Arm A: own = k13 immediate effect = change in m at the edited loop's end over the "
         "clean m_B - m_A at that same loop end (None, class unreadable, when that clean gap is < %.1f logit); final = "
         "change in m at loop K over m_B - m_A at the final readout. Classes with threshold %.2f: never read (own < t, "
         "final < t), re-derived (own >= t, final < t), persisted (both >= t), delayed (own < t, final >= t)."
         % (NBOOT, TAU, GAP_MIN, THR), "",
         "GPU minutes: %s (total %s)." % (gpu, gpu_total), ""]
    for m in order:
        d = res[m]
        sites = sites_of[m]
        sum_ds = (DATASETS + [x for x in KNOWN_DATASETS if x in d["datasets"] and x not in DATASETS]
                  if d.get("partial") else [x for x in KNOWN_DATASETS if x in d["datasets"]])   # a full run: what ran
        checks["pairs_per_model_dataset"][m] = {ds: (d["datasets"][ds]["pairs"] if ds != "CF" else
                                                     dict(d["datasets"][ds]["by_rel"], total=d["datasets"][ds]["pairs"]))
                                                for ds in sum_ds if ds in d["datasets"]}
        checks["partial"][m] = d.get("partial", False)
        checks["para_fallback_last_token"][m] = {ds: "%d/%d" % (d["datasets"][ds]["para_modes"].get("last_token", 0),
                                                               sum(d["datasets"][ds]["para_modes"].values()))
                                                 for ds in sum_ds if ds in d["datasets"]}
        rp = {ds: d["datasets"][ds].get("record_pass") for ds in ("D2", "CF") if ds in d["datasets"]}
        kf = os.path.join(out_dir, "know_%s.json" % m)
        if os.path.exists(kf):
            kd = json.load(open(kf))
            rp = {ds: (rp.get(ds) or kd.get(ds)) for ds in ("D2", "CF")}
        checks["record_pass"][m] = rp
        checks["S_loop1_effect"][m], checks["A_rederived_vs_neverread"][m] = {}, {}
        checks["C_mid_fraction"][m], checks["C_bimodality"][m] = {}, {}
        checks["o5_effect_per_model"][m], checks["o4_max_flip_rate"][m] = {}, {}
        checks["structural_zero_max_abs_effect"][m], checks["reruns"][m] = {}, {}
        checks["batch_gap_max"][m], checks["para_modes"][m], checks["truncated"][m] = {}, {}, {}
        L += ["## %s" % m, "", "Host %s, K %d, layers %s, readout: %s. Load %.0fs, total %.0fs." % (
            d["host"], d["K"], d["layers"], d["readout"], d["load_seconds"], d.get("seconds_total", 0)), ""]
        Kk = d["K"]
        for ds in sum_ds:
            if ds not in d["datasets"]:
                L += ["### %s / %s: not run (partial file)" % (m, ds), ""]
                continue
            D = d["datasets"][ds]
            if D["pairs"] == 0:
                L += ["### %s / %s: no pairs passed (scanned %s)" % (m, ds, D["scanned"]), ""]
                continue
            o5 = ci(m, ds, ckey("O5", "O1", 1.0, "last_pos"))
            checks["o5_effect_per_model"][m][ds] = o5[0] if o5 else None
            o4 = [sum(c["flip"]) / len(c["flip"]) for k, c in D["cells"].items() if k.startswith("S|O4|")]
            checks["o4_max_flip_rate"][m][ds] = round(max(o4), 3) if o4 else None
            sz = [abs(x) for lab in ("Llast@%d" % Kk,) for op in ("O2", "O4") for a in A_S
                  for x in (cellv(m, ds, ckey("S", op, a, lab)) or {"e": []})["e"] if x is not None]
            checks["structural_zero_max_abs_effect"][m][ds] = round(max(sz), 6) if sz else None
            checks["reruns"][m][ds] = "%d/%d" % (D["reruns"], D["rows"])
            checks["batch_gap_max"][m][ds] = round(D["batch_gap_max"], 3)
            checks["para_modes"][m][ds] = D["para_modes"]
            checks["truncated"][m][ds] = D["truncated"]
            L += ["### %s / %s" % (m, ds), "",
                  "Pairs %d (%s), scanned %s of %d candidates, truncated %s, know stats %s. Paraphrase modes %s. "
                  "Reruns at batch 1: %d of %d rows; max in-batch vs batch-1 clean logit gap %.3f. O5 effect %s." % (
                      D["pairs"], D["by_rel"], D["scanned"], D["candidates"], D["truncated"], D["know_stats"],
                      D["para_modes"], D["reruns"], D["rows"], D["batch_gap_max"], fmt(o5)), ""]
            # S
            L += ["Arm S (source edit at subject/operand positions), effect [CI] / flips:", "",
                  "| op a | " + " | ".join(sites) + " |", "|---" * (len(sites) + 1) + "|"]
            for op in ("O2", "O4"):
                for a in A_S:
                    L.append("| %s %s | " % (op, fa(a)) + " | ".join(
                        fmt(ci(m, ds, ckey("S", op, a, s)), ci(m, ds, ckey("S", op, a, s), "flip")) for s in sites) + " |")
            L += ["", "Arm S change in logit A / logit B (a = 1):", "", "| op | " + " | ".join(sites) + " |",
                  "|---" * (len(sites) + 1) + "|"]
            for op in ("O2", "O4"):
                L.append("| %s | " % op + " | ".join("%.2f / %.2f" % (ci(m, ds, ckey("S", op, 1.0, s), "dA")[0],
                                                                      ci(m, ds, ckey("S", op, 1.0, s), "dB")[0])
                                                    for s in sites) + " |")
            L.append("")
            checks["S_loop1_effect"][m][ds] = {"%s|%s" % (op, s): ci(m, ds, ckey("S", op, 1.0, s))
                                               for op in ("O2", "O4") for s in ("L0@1", "Lmid@1", "Llast@1")}
            if ds == "CF":
                L += ["Arm S by CounterFact relation, O2 a=1, effect [CI] (n):", "",
                      "| relation | n | L0@1 | Lmid@1 | Llast@1 | L0@%d |" % Kk, "|---|---|---|---|---|---|"]
                for rel in KD.CF_RELS:
                    its = set(i for i, it in enumerate(D["items"]) if it["rel"] == rel)
                    if not its:
                        continue
                    L.append("| %s | %d | " % (rel, len(its)) + " | ".join(
                        fmt(ci(m, ds, ckey("S", "O2", 1.0, s), items=its)) for s in ("L0@1", "Lmid@1", "Llast@1", "L0@%d" % Kk)) + " |")
                L.append("")
            # A
            L += ["Arm A (derived edit at the last position): own / final [CI of final] / majority class "
                  "(fractions never read / re-derived / persisted / delayed / unreadable):", "",
                  "| a | " + " | ".join(sites) + " |", "|---" * (len(sites) + 1) + "|"]
            for a in A_A:
                row = []
                for s in sites:
                    c = cellv(m, ds, ckey("A", "add", a, s))
                    own, fin = boot_ci(c["own"]), boot_ci(c["fin"])
                    cls = [a_class(x, y) for x, y in zip(c["own"], c["fin"])]
                    cls = [x for x in cls if x]
                    fr = {k: cls.count(k) / len(cls) for k in CLASSES} if cls else {}
                    maj = max(fr, key=fr.get) if fr else "-"
                    row.append("%s / %s / %s (%s)" % ("-" if own is None else "%.2f" % own[0], fmt(fin), maj,
                                                       "/".join("%.2f" % fr.get(k, 0) for k in CLASSES)))
                    if a == 2.0:
                        checks["A_rederived_vs_neverread"][m].setdefault(ds, {})[s] = {
                            "own": own, "final": fin, "class_fractions": {k: round(v, 3) for k, v in fr.items()},
                            "verdict": maj, "flip_own": round(sum(c["fown"]) / len(c["fown"]), 3),
                            "own_in_final_gap_units": boot_ci(c["own_D"]),
                            "clean_gap_own_logits": boot_ci(c["gap_own"])}
                L.append("| %s | " % fa(a) + " | ".join(row) + " |")
            L += ["", "Arm A per-loop trajectory at a = 2, at each loop end from the edited loop on (state diagnostic from "
                      "the in-batch rows; Ouro loop ends are post-norm, so |dh| there is on another scale):", ""]
            for s in sites:
                c = cellv(m, ds, ckey("A", "add", 2.0, s))
                n_le = max(len(t) for t in c["traj"])
                tr = [boot_ci([t[k] for t in c["traj"] if len(t) > k])[0] for k in range(n_le)]
                sv = [boot_ci([t[k] for t in c["surv"] if len(t) > k])[0] for k in range(n_le)]
                cv = [boot_ci([t[k] for t in c["cosv"] if len(t) > k])[0] for k in range(n_le)]
                L.append("- %s: change in m / D %s; state |dh|/|a v| %s; cos(dh, v) %s; flip at own loop %.2f" % (
                    s, ", ".join("%.2f" % x for x in tr), ", ".join("%.2f" % x for x in sv), ", ".join("%.2f" % x for x in cv),
                    sum(c["fown"]) / len(c["fown"])))
            gl = [boot_ci([it["clean_gap_by_loop"][k] for it in D["items"]]) for k in range(len(d["loop_ends"]))]
            gr = [boot_ci([it["mB_by_loop"][k] - it["mA_by_loop"][k] for it in D["items"]]) for k in range(len(d["loop_ends"]))]
            L += ["", "Clean m_B - m_A at each loop end, logits: %s; as a fraction of the final gap: %s" % (
                ", ".join(fmt(x) for x in gr), ", ".join(fmt(x) for x in gl)), ""]
            # C
            L += ["Arm C (O1 at Llast@1, subject positions):", "",
                  "| a | effect | flips | mid fraction [0.2, 0.8] | BC |", "|---|---|---|---|---|"]
            for a in A_C:
                k = ckey("C", "O1", a, "Llast@1")
                L.append("| %s | %s | %.2f | %s | %s |" % (fa(a), fmt(ci(m, ds, k)), ci(m, ds, k, "flip")[0],
                                                          fmt(ci(m, ds, k, stat="mid")), fmt(ci(m, ds, k, stat="bc"))))
            L.append("")
            kc = ckey("C", "O1", 0.5, "Llast@1")
            checks["C_mid_fraction"][m][ds] = ci(m, ds, kc, stat="mid")
            checks["C_bimodality"][m][ds] = ci(m, ds, kc, stat="bc")
        # arm B
        if "armB" not in d:
            L += ["### %s / arm B: not run (partial file)" % m, ""]
            continue
        B = d["armB"]
        sc = scores["models"].get(m, {})
        nP = len(B["gens"])
        rowsB = []
        for c, cd in enumerate(B["conds"]):
            pos = [sc.get("%d|%d" % (pi, c), {}).get("pos") for pi in range(nP)]
            nll = [B["gens"][pi][c]["nll"] for pi in range(nP)]
            rowsB.append(dict(cd, pos=pos, nll=nll, pos_ci=boot_ci(pos), nll_ci=boot_ci(nll),
                              dmg=B["damage"][c]["agree"] / B["damage"][c]["n"], und=B["damage"][c]["undecided"]))
        L += ["### %s / arm B (sentiment CAA at Lmid, all positions, %d prompts)" % (m, nP), "",
              "Site points %s; %d contrast pairs; |v| %s; residual norm %s." % (
                  B["site_points"], B["n_contrast_pairs"], [round(x, 2) for x in B["v_norm"]],
                  [round(x, 2) for x in B["resid_norm"]]), "",
              "| schedule | vector | a | positive rate [CI] | NLL [CI] | damage agree [undecided] | broken |",
              "|---|---|---|---|---|---|---|"]
        broken = []
        for r_ in rowsB:
            br = r_["dmg"] < 0.8
            if br:
                broken.append("%s|%s|%s" % (r_["sched"], r_["vector"], fa(r_["a"])))
            L.append("| %s | %s | %s | %s | %s | %.2f [%d] | %s |" % (r_["sched"], r_["vector"], fa(r_["a"]), fmt(r_["pos_ci"]),
                                                                     fmt(r_["nll_ci"]), r_["dmg"], r_["und"], "yes" if br else ""))
        checks["B_broken_cells"][m] = broken

        def find(sched, a, vec="caa"):
            for r_ in rowsB:
                if r_["sched"] == sched and abs(r_["a"] - a) < 1e-9 and r_["vector"] == vec:
                    return r_
            return None
        mm = {}
        L += ["", "Matched total strength (positive rate; diff = all minus single, paired bootstrap):", "",
              "| comparison | all-loops | single loop | diff [CI] | damage all / single |", "|---|---|---|---|---|"]
        if Kk > 1:
            for a, a4 in MATCHED:
                for single in ("L1", "L%d" % Kk):
                    ra, rs = find("all", a), find(single, a4)
                    if ra is None or rs is None:
                        continue
                    dif = boot_diff(ra["pos"], rs["pos"])
                    kname = "all@%s vs %s@%s" % (fa(a), single, fa(a4))
                    mm[kname] = {"all": ra["pos_ci"], "single": rs["pos_ci"], "diff": dif,
                                 "damage_all": round(ra["dmg"], 3), "damage_single": round(rs["dmg"], 3)}
                    L.append("| %s | %s | %s | %s | %.2f / %.2f |" % (kname, fmt(ra["pos_ci"]), fmt(rs["pos_ci"]), fmt(dif),
                                                                     ra["dmg"], rs["dmg"]))
        checks["B_matched_strength"][m] = mm
        L += ["", "Example (prompt 0): " + " || ".join("%s %s %s: %r" % (cd["sched"], cd["vector"], fa(cd["a"]), B["gens"][0][c]["text"])
                                                     for c, cd in enumerate(B["conds"]) if c in (0, 3, 7, 11, 15)), ""]
    r1 = res.get("ouro")
    if r1 and r1["datasets"]["D1"]["pairs"]:
        checks["smoke2_repro"] = {"ouro_D1_O2_a1_Lmid@1": ci("ouro", "D1", ckey("S", "O2", 1.0, "Lmid@1")),
                                  "ouro_D1_C_mid_fraction_a05": ci("ouro", "D1", ckey("C", "O1", 0.5, "Llast@1"), stat="mid"),
                                  "ouro_D1_C_bc_a05": ci("ouro", "D1", ckey("C", "O1", 0.5, "Llast@1"), stat="bc"),
                                  "expect": "O2 about 1.0; mid fraction about 0.10 (smoke 2, GB10)"}
    json.dump({"models": {m: {k: v for k, v in res[m].items()} for m in order}, "scores": scores, "processes": procs},
              open(os.path.join(out_dir, "results.json"), "w"))
    json.dump(checks, open(os.path.join(out_dir, "checks.json"), "w"), indent=1)
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L))
    print(json.dumps({k: checks[k] for k in ("pairs_per_model_dataset", "smoke2_repro", "gpu_minutes")}), flush=True)


# ------------------------------------------------------------------ self-test (CPU, fake looped model)
class _FakeTok(object):
    eos_token_id = 1

    def decode(self, ids):
        return " ".join("w%d" % i for i in ids)


class _FakeR(object):
    def __init__(self, V=97, H=24, L=3, K=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(V, H, generator=g)
        self.Wl = [torch.randn(H, H, generator=g) * 0.4 for _ in range(L)]
        self.k, self.L, self.dev, self.tok = K, L, torch.device("cpu"), _FakeTok()

    def _layer(self, lid, h, freq, cache, slot):
        T = h.shape[1]
        c = h.cumsum(1) / torch.arange(1, T + 1, dtype=h.dtype).view(1, T, 1)
        return h + torch.tanh(c @ self.Wl[lid])

    def fwd(self, x, k, cache=None, example_ids=None):
        h = self.E[x]
        slot = 0
        for _ in range(k):
            for lid in range(self.L):
                h = self._layer(lid, h, None, None, slot)
                slot += 1
        return h


class _FakeModel(object):
    def __init__(self, r):
        self.r = r
        self.loop_input_injection_module = None

    def output(self, h, a, b):
        return h @ self.r.E.T

    def parameters(self):
        return iter([self.r.E])


class _FakeM(object):
    def __init__(self):
        self.r = _FakeR()
        self.m = _FakeModel(self.r)
        K, L = self.r.k, self.r.L
        self.slot_pt = {(j - 1) * L + l: (j, l) for j in range(1, K + 1) for l in range(L)}
        self.sites = [("%s@%d" % (lab, j), (j, l), j) for j in (1, 2, K) for lab, l in (("L0", 0), ("Lmid", 1), ("Llast", 2))]
        self.loop_ends = [((j, L - 1), j) for j in range(1, K + 1)]
        self.dev = self.r.dev
        self.layers_desc, self.readout_desc = {}, "fake"

    def enc(self, t, ans=False):
        ids = [2 + zlib.crc32(w.encode()) % 95 for w in t.split(" ") if w]
        return ids if ans else [0] + ids

    def readout(self, hl):
        return (hl @ self.r.E.T).float()


def selftest():
    torch.manual_seed(0)
    res = {}
    # statistics
    xs = [0.0] * 50 + [1.0] * 50
    b = boot_ci(xs)
    res["boot_mean"] = b
    res["boot_deterministic"] = boot_ci(xs) == b
    res["boot_mid"] = boot_ci([0.1, 0.5, 0.5, 0.9], "mid")[0]
    res["boot_bc_two_point"] = boot_ci([0.0] * 40 + [1.0] * 40, "bc")[0]
    res["bc_matches_smoke2"] = abs(boot_ci([0.1, 0.3, 0.2, 0.9, 0.95, 0.5], "bc")[0] -
                                   S2.bimodality([0.1, 0.3, 0.2, 0.9, 0.95, 0.5])) < 1e-3
    res["boot_diff"] = boot_diff([1, 1, 0, 1], [0, 1, 0, 0])[0]
    res["a_class"] = [a_class(0.1, 0.0), a_class(0.6, 0.05), a_class(0.6, 0.5), a_class(0.0, 0.4), a_class(None, 0.1)]
    conds = armB_conds(4)
    res["armB_n_conds"] = len(conds)
    res["matched_in_grid"] = all(any(c[0] == "L1" and abs(c[2] - a4) < 1e-9 for c in conds) and
                                 any(c[0] == "all" and abs(c[2] - a) < 1e-9 and c[3] == "caa" for c in conds)
                                 for a, a4 in MATCHED)
    # fake looped model: the pair pipeline
    M = _FakeM()
    ctx = site_ctx(M, "fake")
    final = ctx["final"]
    A = M.enc("the capital of zork is")
    Bq = M.enc("the capital of blip is")
    zA = S2.readout_last(M, S2.run2(M, [A], record={final})[final], [len(A) - 1])[0]
    zB = S2.readout_last(M, S2.run2(M, [Bq], record={final})[final], [len(Bq) - 1])[0]
    tA = int(zA.argmax())
    dz = zB - zA
    dz[tA] = -1e9
    tB = int(dz.argmax())                         # fake model: B's most promoted token stands in for B's answer
    p = {"A": A, "B": Bq, "pos": S1.subj_span(A, Bq)[0], "tA": tA, "tB": tB, "a": "zork", "b": "blip", "rel": "r",
         "half": "h", "suffix": [],
         "para": [(M.enc("so zork has a capital"), M.enc("so blip has a capital")),
                  (M.enc("about zork the city"), M.enc("about blip the city"))]}
    res["fake_tA_ne_tB"] = tA != tB
    rec_pts = set(ctx["site_pts"]) | set(pe for pe, _ in ctx["loop_ends"])
    rA, rB = S2.run2(M, [A], record=rec_pts), S2.run2(M, [Bq], record=rec_pts)
    outs = {}
    for tau in (-1.0, 1e9):                       # no reruns vs every row rerun at batch 1
        cells = {}
        ctx["n_items"] = 0
        it = process_pair(M, p, rA, rB, 7, ctx, cells, tau=tau)
        outs[tau] = (cells, it)
    c0, c1 = outs[-1.0][0], outs[1e9][0]
    diffs = [abs(x - y) for k in c0 for f in ("e", "dA", "dB") for x, y in zip(c0[k][f], c1[k][f])
             if x is not None and y is not None]
    res["rerun_path_max_diff"] = max(diffs)
    res["rerun_counts"] = [outs[-1.0][1]["reruns"], outs[1e9][1]["reruns"], outs[1e9][1]["rows"]]
    res["rows_per_pair"] = outs[-1.0][1]["rows"]
    res["o5_effect"] = c0[ckey("O5", "O1", 1.0, "last_pos")]["e"][0]
    K = ctx["K"]
    sz = [abs(x) for op in ("O2", "O4") for a in A_S for x in c0[ckey("S", op, a, "Llast@%d" % K)]["e"]]
    res["structural_zero"] = max(sz)
    ak = c0[ckey("A", "add", 2.0, "Llast@%d" % K)]
    res["armA_lastK_own_eq_final"] = (ak["own"][0] is not None and abs(ak["own"][0] - ak["fin"][0]) < 1e-4
                                      and len(ak["traj"][0]) == 1)
    a1 = c0[ckey("A", "add", 2.0, "Llast@1")]
    res["armA_own_k13_check"] = (a1["own"][0] is None or abs(a1["own"][0] - a1["dm"][0][0] / a1["gap_own"][0]) < 1e-3)
    res["armA_loop1_traj_len"] = len(c0[ckey("A", "add", 2.0, "L0@1")]["traj"][0])
    al = c0[ckey("A", "add", 2.0, "Llast@1")]
    res["armA_state_own_llast"] = [al["surv"][0][0], al["cosv"][0][0]]
    res["clean_gap_last_is_1"] = outs[-1.0][1]["clean_gap_by_loop"][-1]
    # O2 at a=1 with an aligned paraphrase: vector equals the paraphrase difference (smoke 1 rule)
    vs, vl, modes = para_vectors(M, p, ctx["site_pts"])
    res["para_modes"] = modes
    # arm B: batched generation equals per-row generation
    mpts = [(s, 1) for s in range(1, K + 1)]
    g = torch.Generator().manual_seed(3)
    V = [torch.randn(24, generator=g) for _ in mpts]
    N = [5.0] * K
    Ur = [torch.randn(24, generator=g) for _ in mpts]
    Ur = [u / u.norm() for u in Ur]
    conds = armB_conds(K)
    prompts = [M.enc("i went to the store and"), M.enc("the weather today")]
    batched = generate_rows(M, prompts, conds, mpts, V, N, Ur, final, 1, gen_tokens=5)
    single = []
    for pr in prompts:
        row = []
        for c in range(len(conds)):
            row.append(generate_rows(M, [pr], [conds[c]], mpts, V, N, Ur, final, 1, gen_tokens=5)[0][0])
        single.append(row)
    res["armB_batched_eq_single_text"] = all(batched[i][c]["text"] == single[i][c]["text"]
                                             for i in range(2) for c in range(len(conds)))
    res["armB_nll_maxdiff"] = max(abs(batched[i][c]["nll"] - single[i][c]["nll"]) for i in range(2) for c in range(len(conds))
                                  if batched[i][c]["nll"] is not None)
    res["armB_steering_changes_text"] = sum(batched[0][c]["text"] != batched[0][0]["text"] for c in range(len(conds)))
    ok = (res["boot_deterministic"] and abs(b[0] - 0.5) < 1e-9 and b[1] < 0.5 < b[2] and abs(res["boot_mid"] - 0.5) < 1e-9
          and res["boot_bc_two_point"] > 0.9 and res["bc_matches_smoke2"] and abs(res["boot_diff"] - 0.5) < 1e-9
          and res["a_class"] == ["never_read", "rederived", "persisted", "delayed", "unreadable"] and res["armB_n_conds"] == 21
          and res["matched_in_grid"] and res["fake_tA_ne_tB"] and res["rerun_path_max_diff"] < 1e-4
          and res["rerun_counts"][1] == res["rerun_counts"][2] and abs(res["o5_effect"] - 1.0) < 1e-4
          and res["structural_zero"] < 1e-5 and res["armA_lastK_own_eq_final"] and res["armA_loop1_traj_len"] == K
          and res["armA_own_k13_check"] and abs(res["armA_state_own_llast"][0] - 1) < 1e-3
          and abs(res["armA_state_own_llast"][1] - 1) < 1e-3
          and abs(res["clean_gap_last_is_1"] - 1.0) < 1e-4 and res["armB_batched_eq_single_text"]
          and res["armB_nll_maxdiff"] < 1e-3 and res["armB_steering_changes_text"] > 0)
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--clf", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cf", default="")
    ap.add_argument("--tag", default="p")
    ap.add_argument("--pair_budget_min", type=float, default=0.0)   # per model; D1 and arm B are never cut
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--full_pool", action="store_true", help="cluster copy: no quota, every candidate scanned")
    ap.add_argument("--cf_pool", default="", help="cluster copy: CF candidates from this shared pool file")
    ap.add_argument("--n_neutral", type=int, default=150)
    ap.add_argument("--know_only", action="store_true")
    ap.add_argument("--datasets", default=",".join(DATASETS), help="cluster extension: comma list of D1, D2, CF, D4, D5")
    ap.add_argument("--data_dir", default="", help="cluster extension: k16_path_ifm data folder (D4, D5 rows)")
    a = ap.parse_args()
    FULL_POOL[0] = a.full_pool
    CF_POOL[0] = a.cf_pool
    D4_DIR[0] = a.data_dir
    dss = [x for x in a.datasets.split(",") if x]
    assert dss and set(dss) <= set(KNOWN_DATASETS) and not ({"D4", "D5"} & set(dss) and not a.data_dir), (dss, a.data_dir)
    DATASETS[:] = dss
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
        return
    t0 = time.time()
    try:
        if a.score:
            score(a.out, a.clf)
            return
        for name in [x for x in a.models.split(",") if x]:
            if a.know_only:
                know_only(name, a.out, a.cf)
            else:
                run_model(name, a.out, a.cf, a.pair_budget_min, a.debug, a.n_neutral)
    finally:
        json.dump({"models": a.models or "score", "seconds": round(time.time() - t0, 1), "host": RUN_ROLE,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                  open(os.path.join(a.out, "proc_%s.json" % a.tag), "w"))


if __name__ == "__main__":
    main()
