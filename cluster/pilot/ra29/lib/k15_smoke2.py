"""K15 smoke 2 (brief B-K15-SMOKE2): source edits vs derived edits, behaviour steering, switch/dial dose curve,
injection-channel edit. Plan: notes/k15_smoke2_plan.md. Builds on k15_smoke.py (imported,
not modified).

Items: smoke-1 rule (k13 build_pairs, quota 20 capital / 20 city, at most 2 per subject, clean model answers A and
B) on the held-out half (the smoke-1 40 pairs), then the same rule on the dev half: 80 pairs.
Arms (sites L0 / Lmid / Llast at loops 1, 2, K as smoke 1; d112 mapped layers):
  A   h[last] += a v_last, v_last = mean over 2 paraphrase templates of h_B'[last] - h_A'[last] at the site.
      a in [0.5, 1, 2, 4]. Damage: same vector at the last position of 20 unrelated fact prompts.
  B   sentiment CAA (mild-rgb v2): v_s at Lmid of loop s from 160 "The {noun} was {adj}" pairs (final token),
      add a * n_s * unit(v_s) at every position at Lmid of loop 1, 2, K, or every loop; a in [0.1, 0.2, 0.4, 0.8];
      random control (all loops, a 0.4, 0.8). 30 neutral prompts, greedy 24 tokens, scored by an SST-2 classifier.
  C   O1 at Llast@1 (subject positions), a = 0.30..0.70 step 0.05, per-item effects.
  D   ifm_p1_ouro_inj: input edit at one loop j (emb at subject positions lerped toward B's), a in [0.5, 1], vs state
      O1 at L0@j.
Controls: O5 (O1 a=1 at the last position, final point), smoke-1 repro row (O2 a=1 Lmid@1 at the subject).
Rows whose final top-1 margin is below TAU are rerun alone at batch 1 against the batch-1 clean run (GB10 numerics).

    python k15_smoke2.py --selftest
    python k15_smoke2.py --models ouro --out OUT                         (Ouro venv)
    python k15_smoke2.py --models ifm_p1_ouro,ifm_p1_ouro_inj,ifm_d112,ifm_d28 --out OUT   (xllm_venv)
    python k15_smoke2.py --score --out OUT --clf DIR                      (Ouro venv)
    python k15_smoke2.py --summarize --out OUT
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

import k15_smoke as S1  # noqa: E402

A_ADD = [0.5, 1.0, 2.0, 4.0]
A_C = [round(0.30 + 0.05 * k, 2) for k in range(9)]
A_D = [0.5, 1.0]
A_B = [0.1, 0.2, 0.4, 0.8]
A_B_RAND = [0.4, 0.8]
N_HALF = 40
CAP_MIN = 90.0
D28_SKIP_AFTER_MIN = 60.0
TAU = 1.0
GEN_TOKENS = 24
CLF = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

UNRELATED = [("The opposite of hot is", "cold"), ("The color of grass is", "green"), ("Two plus two equals", "four"),
             ("The first month of the year is", "January"), ("A week has seven", "days"),
             ("The sun rises in the", "east"), ("The largest planet in the solar system is", "Jupiter"),
             ("Bees make", "honey"), ("The day after Monday is", "Tuesday"), ("Cows give us", "milk"),
             ("Romeo and Juliet was written by William", "Shakespeare"), ("A baby cat is called a", "kitten"),
             ("The opposite of up is", "down"), ("Ice is frozen", "water"), ("The number after nine is", "ten"),
             ("Plants need sunlight and", "water"), ("The color of the sky on a clear day is", "blue"),
             ("Spiders have eight", "legs"), ("The opposite of black is", "white"), ("Fish live in", "water")]
NEUTRAL = ["I went to the store and", "The weather today is", "My neighbor told me that", "Yesterday I watched a movie and",
           "The meeting this morning was", "When I opened the door,", "The food at the restaurant was",
           "Our trip to the city was", "The new phone I bought", "After school, the kids",
           "The book I am reading is", "Last weekend we drove to", "The train arrived at the station and",
           "My brother called me to say", "The hotel room was", "At the office today,", "The concert last night was",
           "I spent the afternoon", "The teacher explained that", "This morning I woke up and",
           "The park near my house is", "We had dinner with friends and", "The package was delivered and",
           "My first day at the new job was", "The flight to Chicago was", "The museum downtown is",
           "I tried the new coffee shop and", "The game on Sunday was", "Our garden this year is",
           "The doctor said that"]
NOUNS = ["movie", "meal", "hotel", "concert", "book", "service", "flight", "game", "lecture", "party",
         "trip", "museum", "show", "phone", "app", "coffee", "teacher", "doctor", "weather", "garden"]
POS = ["wonderful", "great", "amazing", "excellent", "fantastic", "delightful", "lovely", "superb", "brilliant",
       "perfect", "good", "awesome", "incredible", "pleasant", "enjoyable"]
NEG = ["terrible", "awful", "horrible", "disappointing", "dreadful", "bad", "poor", "boring", "miserable",
       "frustrating", "mediocre", "lousy", "painful", "unpleasant", "appalling"]
N_ADJ = 8


def fa(a):
    return "%g" % a


def key(arm, a, site):
    return "%s|%s|%s" % (arm, fa(a), site)


# ------------------------------------------------------------------ pure helpers (CPU-testable)
def apply_entries(o, entries):
    """o [B, T, H]. entries: ("add", rows[n], pos[n], delta[n,H]) | ("lerp", rows, pos, target[n,H], a[n,1])
    | ("add_all", rows[n], delta[n,H]). float32 math, written back in o.dtype."""
    o = o.clone()
    for e in entries:
        if e[0] == "add":
            _, r, p, d = e
            o[r, p] = (o[r, p].float() + d).to(o.dtype)
        elif e[0] == "lerp":
            _, r, p, t, a = e
            o[r, p] = torch.lerp(o[r, p].float(), t, a).to(o.dtype)
        elif e[0] == "add_all":
            _, r, d = e
            o[r] = (o[r].float() + d[:, None, :]).to(o.dtype)
        else:
            raise KeyError(e[0])
    return o


def inj_lerp(emb, entries):
    """entries: (rows[n], pos[n], target[n,H], a[n,1]) applied to the injected input."""
    emb = emb.clone()
    for r, p, t, a in entries:
        emb[r, p] = torch.lerp(emb[r, p].float(), t, a).to(emb.dtype)
    return emb


def build_edits(specs, offset, dev):
    """specs -> ({pt: [entries]}, [(loop j, rows, pos, target, a)]). Spec row r sits at batch row r + offset."""
    groups, inj = {}, []
    for r, sp in enumerate(specs):
        row, P = r + offset, len(sp["pos"])
        if sp["kind"] == "add":
            g = groups.setdefault((sp["pt"], "add"), ([], [], [], []))
            g[0].extend([row] * P), g[1].extend(sp["pos"]), g[2].append(sp["delta"])
        elif sp["kind"] == "lerp":
            g = groups.setdefault((sp["pt"], "lerp"), ([], [], [], []))
            g[0].extend([row] * P), g[1].extend(sp["pos"]), g[2].append(sp["tgt"]), g[3].extend([sp["a"]] * P)
        elif sp["kind"] == "inj":
            inj.append((sp["j"], torch.tensor([row] * P, device=dev), torch.tensor(sp["pos"], device=dev),
                        sp["tgt"], torch.full((P, 1), float(sp["a"]), device=dev)))
        else:
            raise KeyError(sp["kind"])
    edits = {}
    for (pt, kind), g in groups.items():
        r, p = torch.tensor(g[0], device=dev), torch.tensor(g[1], device=dev)
        if kind == "add":
            e = ("add", r, p, torch.cat(g[2]))
        else:
            e = ("lerp", r, p, torch.cat(g[2]), torch.tensor(g[3], device=dev)[:, None])
        edits.setdefault(pt, []).append(e)
    return edits, inj


def bimodality(xs):
    """Sample bimodality coefficient (bias-corrected skewness and excess kurtosis). > 0.555 suggests bimodality."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 4:
        return None
    m = sum(xs) / n
    s2 = sum((x - m) ** 2 for x in xs) / n
    if s2 <= 0:
        return None
    g = (sum((x - m) ** 3 for x in xs) / n) / s2 ** 1.5
    k = (sum((x - m) ** 4 for x in xs) / n) / s2 ** 2 - 3.0
    G = g * math.sqrt(n * (n - 1)) / (n - 2)
    Kc = ((n + 1) * k + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    return (G * G + 1.0) / (Kc + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def mid_fraction(xs, lo=0.2, hi=0.8):
    xs = [x for x in xs if x is not None]
    return (sum(1 for x in xs if lo <= x <= hi) / len(xs)) if xs else None


def caa_delta(v, n, a):
    return a * n * v / v.norm()


def selftest():
    torch.manual_seed(0)
    B, T, H = 4, 6, 32
    o = torch.randn(B, T, H).to(torch.bfloat16)
    res = {}
    d = torch.randn(2, H)
    e = apply_entries(o, [("add", torch.tensor([1, 1]), torch.tensor([2, 5]), d)])
    ref = o.clone()
    ref[1, [2, 5]] = (o[1, [2, 5]].float() + d).to(o.dtype)
    res["add_exact"] = bool(torch.equal(e, ref))
    res["add_untouched"] = bool(torch.equal(e[[0, 2, 3]], o[[0, 2, 3]]))
    t = torch.randn(3, H)
    e = apply_entries(o, [("lerp", torch.tensor([0, 2, 3]), torch.tensor([1, 1, 4]), t, torch.ones(3, 1))])
    res["lerp_a1_exact_target"] = bool(torch.equal(e[[0, 2, 3], [1, 1, 4]], t.to(o.dtype)))
    e = apply_entries(o, [("lerp", torch.tensor([0]), torch.tensor([1]), t[:1], torch.full((1, 1), 0.5))])
    res["lerp_a05_maxerr"] = float((e[0, 1].float() - ((o[0, 1].float() + t[0]) / 2).to(o.dtype).float()).abs().max())
    dd = torch.randn(2, H)
    e = apply_entries(o, [("add_all", torch.tensor([0, 3]), dd)])
    res["add_all_exact"] = bool(torch.equal(e[[0, 3]], (o[[0, 3]].float() + dd[:, None]).to(o.dtype)))
    res["add_all_untouched"] = bool(torch.equal(e[[1, 2]], o[[1, 2]]))
    emb = torch.randn(B, T, H).to(torch.bfloat16)
    tg = torch.randn(2, H)
    e2 = inj_lerp(emb, [(torch.tensor([2, 2]), torch.tensor([1, 2]), tg, torch.ones(2, 1))])
    res["inj_a1_exact"] = bool(torch.equal(e2[2, [1, 2]], tg.to(emb.dtype)))
    res["inj_untouched"] = bool(torch.equal(e2[[0, 1, 3]], emb[[0, 1, 3]]) and torch.equal(e2[2, [0, 3, 4, 5]], emb[2, [0, 3, 4, 5]]))
    specs = [{"kind": "add", "pt": (1, 0), "pos": [3], "delta": torch.ones(1, H)},
             {"kind": "lerp", "pt": (1, 0), "pos": [1, 2], "tgt": torch.zeros(2, H), "a": 1.0},
             {"kind": "inj", "j": 2, "pos": [1], "tgt": torch.zeros(1, H), "a": 0.5}]
    ed, ij = build_edits(specs, 1, "cpu")
    e = apply_entries(o, ed[(1, 0)])
    res["build_edits_rows"] = bool(torch.equal(e[1, 3], (o[1, 3].float() + 1).to(o.dtype)) and
                                   bool((e[2, [1, 2]] == 0).all()) and torch.equal(e[0], o[0]) and
                                   len(ij) == 1 and int(ij[0][1][0]) == 3)
    v = torch.randn(H)
    res["caa_norm_rel_err"] = float((caa_delta(v, 7.0, 0.4).norm() / (0.4 * 7.0) - 1).abs())
    g = torch.Generator().manual_seed(1)
    two = [0.02 * float(torch.randn(1, generator=g)) + (0.0 if i % 2 else 1.0) for i in range(80)]
    nor = [float(x) for x in torch.randn(4000, generator=g)]
    res["bc_two_point"] = bimodality(two)
    res["bc_normal"] = bimodality(nor)
    res["mid_fraction"] = mid_fraction([0.1, 0.2, 0.5, 0.8, 0.9])
    ok = (res["add_exact"] and res["add_untouched"] and res["lerp_a1_exact_target"] and res["lerp_a05_maxerr"] < 0.05
          and res["add_all_exact"] and res["add_all_untouched"] and res["inj_a1_exact"] and res["inj_untouched"]
          and res["build_edits_rows"] and res["caa_norm_rel_err"] < 1e-5 and res["bc_two_point"] > 0.9
          and 0.3 < res["bc_normal"] < 0.37 and abs(res["mid_fraction"] - 0.6) < 1e-9)
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


# ------------------------------------------------------------------ model access
def is_ouro(M):
    return isinstance(M, S1.OuroM)


@torch.no_grad()
def run2(M, rows, edits=None, record=(), inj=None):
    """Right-padded batch of token lists (causal: real positions never see the pads). Returns {pt: tensor}."""
    T = max(len(r) for r in rows)
    x = torch.tensor([r + [0] * (T - len(r)) for r in rows], device=M.dev)
    rec, edits = {}, (edits or {})
    if is_ouro(M):
        assert not inj
        cnt = {}

        def mk(k_):
            def hook(mod, i, o):
                c = cnt.get(k_, 0) + 1
                cnt[k_] = c
                pt = (c, k_)
                if pt in edits:
                    o = apply_entries(o, edits[pt])
                if pt in record:
                    rec[pt] = o.detach()
                return o
            return hook
        hs = [mod.register_forward_hook(mk(k_)) for k_, mod in M.mods.items()]
        try:
            M.m(input_ids=x, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        M.last_calls = cnt
        return rec
    r = M.r
    orig = r._layer

    def lay(lid, h, freq, cache, slot):
        out = orig(lid, h, freq, cache, slot)
        pt = M.slot_pt.get(slot)
        if pt in edits:
            out = apply_entries(out, edits[pt])
        if pt in record:
            rec[pt] = out.detach()
        return out
    r._layer = lay
    injm = getattr(M.m, "loop_input_injection_module", None)
    calls = [0]
    if inj:
        assert injm is not None
        inj_orig = injm.forward

        def inj_fwd(state, emb):
            calls[0] += 1
            es = [e[1:] for e in inj if e[0] == calls[0]]
            return inj_orig(state, inj_lerp(emb, es) if es else emb)
        injm.forward = inj_fwd
    try:
        r.fwd(x, r.k, cache=None, example_ids=torch.arange(len(rows), device=r.dev))
    finally:
        r._layer = orig
        if inj:
            injm.__dict__.pop("forward", None)
    if inj:
        assert calls[0] == r.k, (calls[0], r.k)
    return rec


@torch.no_grad()
def readout_last(M, h, last):
    hl = h[torch.arange(h.shape[0], device=h.device), torch.tensor(last, device=h.device)]
    return M.readout(hl).float()


@torch.no_grad()
def readout_all(M, h):
    if is_ouro(M):
        return torch.nn.functional.linear(h, M.W).float()
    return M.m.output(h, None, None).float()


def decode(M, ids):
    if is_ouro(M):
        return M.enc.__self__.tok.decode(ids, skip_special_tokens=True)
    return M.r.tok.decode(ids)


def eos_id(M):
    return int(M.m.config.eos_token_id) if is_ouro(M) else int(M.r.tok.eos_token_id)


def mid_pts(M, name, K):
    """Lmid point of each loop s = 1..K (d112: layer 28(s-1)+14; d28: layer 14 once)."""
    if is_ouro(M):
        return [(s, M.L // 2) for s in range(1, K + 1)]
    if name == "ifm_d112":
        return [(1, 28 * (s - 1) + 14) for s in range(1, 5)]
    lid = dict((lab, pt) for lab, pt, _ in M.sites)["Lmid@1"][1]
    return [(s, lid) for s in range(1, K + 1)]


def top_margin(z):
    t = z.topk(2).values
    return float(t[0] - t[1])


# ------------------------------------------------------------------ one model
@torch.no_grad()
def run_model(name, out_dir):
    t0 = time.time()
    M = S1.load_model(name)
    load_s = time.time() - t0
    dev = M.dev
    from k13_edit import build_pairs
    pairs = build_pairs(M.enc)
    sites = list(M.sites)
    site_pts = [pt for _, pt, _ in sites]
    final = M.loop_ends[-1][0]
    K = M.loop_ends[-1][1]
    lab2pt = dict((lab, pt) for lab, pt, _ in sites)
    inj_model = name == "ifm_p1_ouro_inj"
    l0 = {}
    if inj_model:
        s0 = lab2pt["L0@1"][1]
        l0 = {j: (j, s0) for j in range(1, K + 1)}
    rec_pts = set(site_pts) | {final} | set(l0.values())
    # unrelated prompts: batch-1 clean and right-padded clean
    U = [M.enc(p) for p, _ in UNRELATED]
    U_last = [len(u) - 1 for u in U]
    U_gold = [M.enc(" " + g, ans=True)[0] for _, g in UNRELATED]
    U_top1 = [int(readout_last(M, run2(M, [u], record={final})[final], [len(u) - 1])[0].argmax()) for u in U]
    zpad = readout_last(M, run2(M, U, record={final})[final], U_last)
    U_pad_agree = sum(int(zpad[k_].argmax()) == U_top1[k_] for k_ in range(len(U)))
    U_gold_ok = sum(int(U_top1[k_] == U_gold[k_]) for k_ in range(len(U)))

    cells, items, dmgA, diagA = {}, [], {}, {}
    gap, reruns, nrows = 0.0, 0, 0
    scanned = {"heldout": 0, "dev": 0}
    loops_run = None
    for half, parity in (("heldout", 1), ("dev", 0)):
        n_rel, n_subj, kept = {"capital": 0, "city": 0}, {}, 0
        quota = {"capital": N_HALF // 2, "city": N_HALF - N_HALF // 2}
        for i, p in enumerate(pairs):
            if i % 2 != parity:
                continue
            sa_, sb_ = (p["rel"], p["a"]), (p["rel"], p["b"])
            if n_rel[p["rel"]] >= quota[p["rel"]] or n_subj.get(sa_, 0) >= 2 or n_subj.get(sb_, 0) >= 2:
                continue
            scanned[half] += 1
            tA, tB, pos = p["tA"], p["tB"], p["pos"]
            rA, rB = run2(M, [p["A"]], record=rec_pts), run2(M, [p["B"]], record=rec_pts)
            if loops_run is None:
                loops_run = {"module_calls": {str(k_): v for k_, v in getattr(M, "last_calls", {}).items()},
                             "final_point_hit": final in rA}
                assert final in rA
            T = len(p["A"])
            last = T - 1
            zA, zB = readout_last(M, rA[final], [last])[0], readout_last(M, rB[final], [last])[0]
            if int(zA.argmax()) != tA or int(zB.argmax()) != tB:
                continue
            n_rel[p["rel"]] += 1
            n_subj[sa_] = n_subj.get(sa_, 0) + 1
            n_subj[sb_] = n_subj.get(sb_, 0) + 1
            kept += 1
            mB = float(zB[tB] - zB[tA])
            # paraphrase vectors: last position (arm A) at every site, subject (smoke-1 rule) at Lmid@1
            pt_r = lab2pt["Lmid@1"]
            vl = {pt: [] for pt in site_pts}
            vs, modes = [], []
            for tm in S1.PARA[p["rel"]]:
                ia, ib = M.enc(tm.format(p["a"])), M.enc(tm.format(p["b"]))
                ra, rb = run2(M, [ia], record=set(site_pts)), run2(M, [ib], record=set(site_pts))
                for pt in site_pts:
                    vl[pt].append(rb[pt][0, -1].float() - ra[pt][0, -1].float())
                sa, sb = S1.subj_span(ia, ib)
                if len(sa) == len(pos) and len(sb) == len(pos):
                    vs.append(rb[pt_r][0, sb].float() - ra[pt_r][0, sa].float())
                    modes.append("aligned")
                else:
                    vs.append((rb[pt_r][0, sb[-1]].float() - ra[pt_r][0, sa[-1]].float()).expand(len(pos), -1))
                    modes.append("last_token")
            vl = {pt: torch.stack(x).mean(0) for pt, x in vl.items()}
            vsub = torch.stack(vs).mean(0)
            for lab, pt, _ in sites:
                dlt = rB[pt][0, last].float() - rA[pt][0, last].float()
                dg = diagA.setdefault(lab, {"cos": [], "norm_ratio": []})
                dg["cos"].append(float(torch.nn.functional.cosine_similarity(vl[pt], dlt, dim=0)))
                dg["norm_ratio"].append(float(vl[pt].norm() / dlt.norm().clamp_min(1e-12)))
            # specs of the main batch
            specs = []
            for lab, pt, _ in sites:
                for a in A_ADD:
                    specs.append({"arm": "A", "site": lab, "a": a, "kind": "add", "pt": pt, "pos": [last],
                                  "delta": (a * vl[pt])[None]})
            specs.append({"arm": "R", "site": "Lmid@1", "a": 1.0, "kind": "add", "pt": pt_r, "pos": pos,
                          "delta": vsub})
            specs.append({"arm": "O5", "site": "last_pos", "a": 1.0, "kind": "lerp", "pt": final, "pos": [last],
                          "tgt": rB[final][0, [last]].float()})
            pt_c = lab2pt["Llast@1"]
            for a in A_C:
                specs.append({"arm": "C", "site": "Llast@1", "a": a, "kind": "lerp", "pt": pt_c, "pos": pos,
                              "tgt": rB[pt_c][0, pos].float()})
            if inj_model:
                embB = M.m.embed(torch.tensor([p["B"]], device=dev))[0, pos].float()
                for j in range(1, K + 1):
                    for a in A_D:
                        specs.append({"arm": "D_input", "site": "inp@%d" % j, "a": a, "kind": "inj", "j": j,
                                      "pos": pos, "tgt": embB})
                        specs.append({"arm": "D_state", "site": "L0@%d" % j, "a": a, "kind": "lerp",
                                      "pt": l0[j], "pos": pos, "tgt": rB[l0[j]][0, pos].float()})
            ed, ij = build_edits(specs, 1, dev)
            rec = run2(M, [p["A"]] * (1 + len(specs)), ed, {final}, ij)
            z = readout_last(M, rec[final], [last] * (1 + len(specs)))
            zc = z[0]
            gap = max(gap, float((zc - zA).abs().max()))
            for r_, sp in enumerate(specs):
                zr, base, rr_flag = z[r_ + 1], zc, 0
                nrows += 1
                if top_margin(zr) < TAU:
                    e1, i1 = build_edits([sp], 0, dev)
                    zr = readout_last(M, run2(M, [p["A"]], e1, {final}, i1)[final], [last])[0]
                    base, rr_flag = zA, 1
                    reruns += 1
                mA = float(base[tB] - base[tA])
                d = mB - mA
                m_e = float(zr[tB] - zr[tA])
                c = cells.setdefault(key(sp["arm"], sp["a"], sp["site"]),
                                     {"effect": [], "flip": [], "dA": [], "dB": [], "rerun": [], "pair": []})
                c["effect"].append((m_e - mA) / d if abs(d) > 1e-3 else None)
                c["flip"].append(int(int(zr.argmax()) == tB))
                c["dA"].append(float(zr[tA] - base[tA]))
                c["dB"].append(float(zr[tB] - base[tB]))
                c["rerun"].append(rr_flag)
                c["pair"].append(len(items))
            # arm A damage on the unrelated prompts
            nU = len(U)
            dspecs, rows = [], list(U)
            for lab, pt, _ in sites:
                for a in A_ADD:
                    dspecs.append((lab, pt, a, len(rows)))
                    rows += U
            dd = {}
            for lab, pt, a, br in dspecs:
                g = dd.setdefault(pt, ([], [], []))
                g[0].extend(range(br, br + nU)), g[1].extend(U_last), g[2].append((a * vl[pt])[None].expand(nU, -1))
            edd = {pt: [("add", torch.tensor(g[0], device=dev), torch.tensor(g[1], device=dev), torch.cat(g[2]))]
                   for pt, g in dd.items()}
            recd = run2(M, rows, edd, {final})
            zd = readout_last(M, recd[final], [len(x) - 1 for x in rows])
            ctop = zd[:nU].argmax(-1)
            for lab, pt, a, br in dspecs:
                blk = zd[br:br + nU]
                ag = int((blk.argmax(-1) == ctop).sum())
                und = int(sum(1 for k_ in range(nU) if top_margin(blk[k_]) < TAU))
                c = dmgA.setdefault(key("A", a, lab), {"agree": 0, "n": 0, "undecided": 0})
                c["agree"] += ag
                c["n"] += nU
                c["undecided"] += und
            items.append({"i": i, "half": half, "a": p["a"], "b": p["b"], "rel": p["rel"], "para_modes": modes,
                          "mB": mB, "mA_single": float(zA[tB] - zA[tA])})
            print("[k15b %s] %s pair idx %d kept %d/%d %.0fs reruns %d" % (name, half, i, kept, N_HALF,
                                                                         time.time() - t0, reruns), flush=True)
            if kept >= N_HALF:
                break

    armB = run_armB(M, name, K, final, U, U_last)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None
    summ = {}
    for k_, c in cells.items():
        summ[k_] = {"effect": mean(c["effect"]), "flip": mean(c["flip"]), "dA": mean(c["dA"]), "dB": mean(c["dB"]),
                    "n": len(c["flip"]), "reruns": sum(c["rerun"]),
                    "per_pair_effect": c["effect"], "per_pair_flip": c["flip"], "per_pair_rerun": c["rerun"]}
    res = {"model": name, "pairs_total": len(pairs), "scanned": scanned, "pairs_passed": len(items),
           "pairs_by_half": {h: sum(1 for it in items if it["half"] == h) for h in ("heldout", "dev")},
           "pairs_by_relation_half": {"%s:%s" % (h, r_): sum(1 for it in items if it["half"] == h and it["rel"] == r_)
                                      for h in ("heldout", "dev") for r_ in ("capital", "city")},
           "n_subjects": len(set("%s:%s" % (it["rel"], x) for it in items for x in (it["a"], it["b"]))),
           "sites": [[lab, list(pt), j] for lab, pt, j in sites], "layers": M.layers_desc, "K": K,
           "loops_run": loops_run, "readout": M.readout_desc, "dtype": str(next(M.m.parameters()).dtype),
           "clean_batch_vs_single_max_logit_gap": gap, "rows_scored": nrows, "rows_rerun_batch1": reruns,
           "tau": TAU, "cells": summ, "armA_damage": dmgA,
           "armA_vector_diag": {lab: {"cos_mean": mean(d["cos"]), "norm_ratio_mean": mean(d["norm_ratio"])}
                                for lab, d in diagA.items()},
           "unrelated": {"prompts": [p for p, _ in UNRELATED], "padded_vs_single_top1_agree": U_pad_agree,
                         "clean_top1_is_gold": U_gold_ok, "n": len(U)},
           "armB": armB, "items": items, "load_seconds": round(load_s, 1), "seconds_total": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(out_dir, "k15b_%s.json" % name), "w"))
    print("[k15b %s] done pairs %d scanned %s gap %.3f reruns %d/%d total %.0fs" % (
        name, len(items), scanned, gap, reruns, nrows, time.time() - t0), flush=True)
    del M
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def run_armB(M, name, K, final, U, U_last):
    t0 = time.time()
    dev = M.dev
    mpts = mid_pts(M, name, K)
    single = lambda w: len(M.enc(" " + w, ans=True)) == 1  # noqa: E731
    pos_w = [w for w in POS if single(w)][:N_ADJ]
    neg_w = [w for w in NEG if single(w)][:N_ADJ]
    nadj = min(len(pos_w), len(neg_w))
    texts, labels = [], []
    for n_ in NOUNS:
        for k_ in range(nadj):
            texts += ["The %s was %s" % (n_, pos_w[k_]), "The %s was %s" % (n_, neg_w[k_])]
            labels += [1, 0]
    ids = [M.enc(t) for t in texts]
    rec = run2(M, ids, record=set(mpts))
    lastp = torch.tensor([len(x) - 1 for x in ids], device=dev)
    lab_t = torch.tensor(labels, device=dev)
    V, N = [], []
    for pt in mpts:
        h = rec[pt][torch.arange(len(ids), device=dev), lastp].float()
        V.append(h[lab_t == 1].mean(0) - h[lab_t == 0].mean(0))
        N.append(float(h.norm(dim=-1).mean()))
    del rec
    g = torch.Generator(device="cpu").manual_seed(12345)
    Ur = [(lambda r_: r_ / r_.norm())(torch.randn(V[0].shape[0], generator=g)).to(dev) for _ in mpts]
    scheds = ([("L1", [1]), ("L2", [2]), ("L4", [K]), ("all", list(range(1, K + 1)))] if K > 1 else [("L1", [1])])
    conds = [("clean", None, 0.0, "caa")]
    for sn, loops in scheds:
        for a in A_B:
            conds.append((sn, loops, a, "caa"))
    for a in A_B_RAND:
        conds.append(("all" if K > 1 else "L1", list(range(1, K + 1)), a, "random"))
    C = len(conds)

    def edits_for(row_blocks):
        """row_blocks[c] = list of batch rows for condition c."""
        groups = {}
        for c, (sn, loops, a, kind) in enumerate(conds):
            if loops is None:
                continue
            for s in loops:
                d = caa_delta(V[s - 1], N[s - 1], a) if kind == "caa" else a * N[s - 1] * Ur[s - 1]
                gg = groups.setdefault(mpts[s - 1], ([], []))
                gg[0].extend(row_blocks[c])
                gg[1].append(d[None].expand(len(row_blocks[c]), -1))
        return {pt: [("add_all", torch.tensor(gg[0], device=dev), torch.cat(gg[1]))] for pt, gg in groups.items()}
    ed_gen = edits_for([[c] for c in range(C)])
    eos = eos_id(M)
    gens = []
    for pi, pr in enumerate(NEUTRAL):
        p_ids = M.enc(pr)
        T0 = len(p_ids)
        seqs = [list(p_ids) for _ in range(C)]
        for _ in range(GEN_TOKENS):
            rr = run2(M, seqs, ed_gen, {final})
            nxt = rr[final][:, -1]
            nxt = M.readout(nxt).float().argmax(-1).tolist()
            for c in range(C):
                seqs[c].append(int(nxt[c]))
        rr = run2(M, seqs, None, {final})
        lp = torch.log_softmax(readout_all(M, rr[final]), -1)
        row = []
        for c in range(C):
            cont = seqs[c][T0:]
            if eos in cont:
                cont = cont[:cont.index(eos)]
            nll = None
            if cont:
                tok = torch.tensor(cont, device=dev)
                nll = float(-lp[c, torch.arange(T0 - 1, T0 - 1 + len(cont), device=dev), tok].mean())
            row.append({"text": decode(M, cont), "n_tok": len(cont), "nll": nll})
        gens.append(row)
        if pi % 10 == 0:
            print("[k15b %s] armB prompt %d/%d %.0fs" % (name, pi + 1, len(NEUTRAL), time.time() - t0),
                  flush=True)
    # damage: every condition on the unrelated prompts
    nU = len(U)
    rows = [u for _ in range(C) for u in U]
    edd = edits_for([list(range(c * nU, (c + 1) * nU)) for c in range(C)])
    zd = readout_last(M, run2(M, rows, edd, {final})[final], [x for _ in range(C) for x in U_last])
    ctop = zd[:nU].argmax(-1)
    dmg = []
    for c in range(C):
        blk = zd[c * nU:(c + 1) * nU]
        dmg.append({"agree": int((blk.argmax(-1) == ctop).sum()), "n": nU,
                    "undecided": int(sum(1 for k_ in range(nU) if top_margin(blk[k_]) < TAU))})
    cos = {"%d-%d" % (a + 1, b + 1): float(torch.nn.functional.cosine_similarity(V[a], V[b], dim=0))
           for a in range(len(V)) for b in range(a + 1, len(V))}
    return {"site_points": [list(pt) for pt in mpts], "adjectives": {"pos": pos_w[:nadj], "neg": neg_w[:nadj]},
            "n_contrast_pairs": len(texts) // 2, "v_norm": [float(v.norm()) for v in V], "resid_norm": N,
            "v_cos_between_loops": cos,
            "conds": [{"sched": sn, "loops": loops, "a": a, "vector": kind} for sn, loops, a, kind in conds],
            "gens": gens, "damage": dmg, "seconds": round(time.time() - t0, 1)}


def spent_minutes(out_dir):
    return sum(json.load(open(f))["seconds"] for f in glob.glob(os.path.join(out_dir, "proc_*.json"))) / 60.0


# ------------------------------------------------------------------ scoring (classifier, .venv)
def score(out_dir, clf_dir):
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=clf_dir, tokenizer=clf_dir, device=0 if torch.cuda.is_available() else -1)
    out = {"classifier": CLF, "local_dir": clf_dir, "models": {}}
    for f in sorted(glob.glob(os.path.join(out_dir, "k15b_*.json"))):
        d = json.load(open(f))
        B = d["armB"]
        texts = [(pi, c, g["text"]) for pi, row in enumerate(B["gens"]) for c, g in enumerate(row)]
        nonempty = [t for t in texts if t[2].strip()]
        preds = clf([t[2] for t in nonempty], batch_size=64, truncation=True, top_k=None)
        sc = {}
        for (pi, c, _), pr in zip(nonempty, preds):
            pp = [x["score"] for x in pr if x["label"].upper().startswith("POS")][0]
            sc["%d|%d" % (pi, c)] = {"pos": int(pp > 0.5), "p_pos": float(pp)}
        out["models"][d["model"]] = sc
        print("[k15b score] %s %d texts (%d empty)" % (d["model"], len(texts), len(texts) - len(nonempty)), flush=True)
    json.dump(out, open(os.path.join(out_dir, "k15b_scores.json"), "w"))


# ------------------------------------------------------------------ summary
def summarize(out_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "k15b_*.json"))):
        if f.endswith("k15b_scores.json"):
            continue
        d = json.load(open(f))
        res[d["model"]] = d
    sp = os.path.join(out_dir, "k15b_scores.json")
    scores = json.load(open(sp)) if os.path.exists(sp) else {"models": {}}
    procs = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "proc_*.json"))}
    gpu_min = round(sum(p["seconds"] for p in procs.values()) / 60.0, 2)
    skipped = [m for p in procs.values() for m in p.get("skipped", [])]

    def r3(x):
        return None if x is None else round(x, 3)

    def cell(m, arm, a, site, f="effect"):
        c = res[m]["cells"].get(key(arm, a, site))
        return None if c is None else c[f]
    order = [m for m in ("ouro", "ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_d112", "ifm_d28") if m in res]
    # arm B tables
    armB = {}
    for m in order:
        B = res[m]["armB"]
        sc = scores["models"].get(m, {})
        rows = []
        for c, cd in enumerate(B["conds"]):
            labs = [sc.get("%d|%d" % (pi, c)) for pi in range(len(B["gens"]))]
            labs_ok = [x for x in labs if x is not None]
            nll = [B["gens"][pi][c]["nll"] for pi in range(len(B["gens"])) if B["gens"][pi][c]["nll"] is not None]
            rows.append(dict(cd, pos_rate=(sum(x["pos"] for x in labs_ok) / len(labs_ok)) if labs_ok else None,
                             p_pos=(sum(x["p_pos"] for x in labs_ok) / len(labs_ok)) if labs_ok else None,
                             n_scored=len(labs_ok), nll=(sum(nll) / len(nll)) if nll else None,
                             damage_agree=B["damage"][c]["agree"] / B["damage"][c]["n"],
                             damage_undecided=B["damage"][c]["undecided"]))
        armB[m] = rows
    # smoke-1 cross-check on the first 40 pairs
    s1dir = os.path.join(out_dir, "..", "k15_smoke")
    cross = {}
    for m in order:
        f1 = os.path.join(s1dir, "k15_%s.json" % m)
        if not os.path.exists(f1):
            cross[m] = "smoke-1 json not found"
            continue
        d1 = json.load(open(f1))
        same = [(it["a"], it["b"]) for it in d1["items"]] == [(it["a"], it["b"]) for it in res[m]["items"][:40]]
        out = {"same_40_pairs": same}
        for (k1, k2) in (("O2|1|Lmid@1", key("R", 1.0, "Lmid@1")), ("O1|0.5|Llast@1", key("C", 0.5, "Llast@1"))):
            e1 = d1["cells"].get(k1, {}).get("per_pair_effect")
            e2 = res[m]["cells"].get(k2, {}).get("per_pair_effect", [])[:40]
            if e1 and same:
                df = [abs(x - y) for x, y in zip(e1, e2) if x is not None and y is not None]
                out[k1] = {"max_abs_diff": r3(max(df)), "smoke1_mean": r3(sum(x for x in e1 if x is not None) / len(df)),
                           "smoke2_mean_first40": r3(sum(y for y in e2 if y is not None) / len(df))}
        cross[m] = out
    checks = {
        "pairs_per_model": {m: res[m]["pairs_passed"] for m in order},
        "pairs_by_relation_half": {m: res[m]["pairs_by_relation_half"] for m in order},
        "armA_loop1_vs_loop4_effect": {m: {"a=1": {"loop1": [r3(cell(m, "A", 1.0, "%s@1" % l)) for l in ("L0", "Lmid", "Llast")],
                                                   "loop4": [r3(cell(m, "A", 1.0, "%s@4" % l)) for l in ("L0", "Lmid", "Llast")]},
                                           "a=4": {"loop1": [r3(cell(m, "A", 4.0, "%s@1" % l)) for l in ("L0", "Lmid", "Llast")],
                                                   "loop4": [r3(cell(m, "A", 4.0, "%s@4" % l)) for l in ("L0", "Lmid", "Llast")]}}
                                       for m in order if m != "ifm_d28"},
        "armB_pos_rate": {m: {"%s|%s|%s" % (r_["sched"], fa(r_["a"]), r_["vector"]): r3(r_["pos_rate"]) for r_ in armB[m]}
                          for m in order},
        "armB_classifier": CLF,
        "armC_mid_fraction_a05": {m: r3(mid_fraction(res[m]["cells"][key("C", 0.5, "Llast@1")]["per_pair_effect"]))
                                  for m in order},
        "armC_bimodality": {m: r3(bimodality(res[m]["cells"][key("C", 0.5, "Llast@1")]["per_pair_effect"]))
                            for m in order},
        "armD_input_edit_flips": ({"inp@%d|a%s" % (j, fa(a)): r3(cell("ifm_p1_ouro_inj", "D_input", a, "inp@%d" % j, "flip"))
                                   for j in range(1, 5) for a in A_D} if "ifm_p1_ouro_inj" in res else None),
        "armD_state_L0_flips": ({"L0@%d|a%s" % (j, fa(a)): r3(cell("ifm_p1_ouro_inj", "D_state", a, "L0@%d" % j, "flip"))
                                 for j in range(1, 5) for a in A_D} if "ifm_p1_ouro_inj" in res else None),
        "smoke1_repro": r3(cell("ouro", "R", 1.0, "Lmid@1")) if "ouro" in res else None,
        "smoke1_repro_first40": (cross.get("ouro", {}).get("O2|1|Lmid@1") if isinstance(cross.get("ouro"), dict) else None),
        "smoke1_cross_check": cross,
        "o5_effect_per_model": {m: r3(cell(m, "O5", 1.0, "last_pos")) for m in order},
        "gpu_minutes": gpu_min,
        "models_skipped": skipped,
        "clean_batch_vs_single_max_logit_gap": {m: r3(res[m]["clean_batch_vs_single_max_logit_gap"]) for m in order},
        "rows_rerun_batch1": {m: "%d/%d" % (res[m]["rows_rerun_batch1"], res[m]["rows_scored"]) for m in order},
        "unrelated_padded_vs_single_agree": {m: "%d/%d" % (res[m]["unrelated"]["padded_vs_single_top1_agree"],
                                                             res[m]["unrelated"]["n"]) for m in order},
        "unrelated_clean_top1_is_gold": {m: "%d/%d" % (res[m]["unrelated"]["clean_top1_is_gold"], res[m]["unrelated"]["n"])
                                         for m in order},
        "armB_scored_texts": {m: sum(r_["n_scored"] for r_ in armB[m]) for m in order},
    }
    json.dump({"models": res, "armB_table": armB, "scores": scores, "processes": procs},
              open(os.path.join(out_dir, "results.json"), "w"))
    json.dump(checks, open(os.path.join(out_dir, "checks.json"), "w"), indent=1)

    def f2(x):
        return "-" if x is None else "%.2f" % x
    L = ["# K15 smoke 2 summary", "",
         "Effect e = (m_edit - m_A)/(m_B - m_A), m = logit tB - logit tA at the final readout; flip = top-1 is tB. "
         "Rows with final top-1 margin < %.1f logit rerun alone at batch 1. Damage = agreement with the clean top-1 "
         "on 20 unrelated fact prompts (1.00 = no damage)." % TAU, "",
         "GPU minutes: %s. Skipped: %s. Classifier (arm B): %s." % (gpu_min, ", ".join(skipped) or "none", CLF), ""]
    for m in order:
        d = res[m]
        sites = [s[0] for s in d["sites"]]
        L += ["## %s" % m, "",
              "Pairs %d (%s), %d subjects, scanned %s. Reruns at batch 1: %d of %d rows. In-batch vs batch-1 clean max "
              "logit gap %.3f. O5 effect %s. Unrelated prompts: padded vs single top-1 agree %d/20, clean top-1 = gold %d/20."
              % (d["pairs_passed"], d["pairs_by_relation_half"], d["n_subjects"], d["scanned"], d["rows_rerun_batch1"],
                 d["rows_scored"], d["clean_batch_vs_single_max_logit_gap"], f2(cell(m, "O5", 1.0, "last_pos")),
                 d["unrelated"]["padded_vs_single_top1_agree"], d["unrelated"]["clean_top1_is_gold"]), "",
              "smoke-1 cross-check (first 40 pairs): %s" % (cross.get(m),), "",
              "Repro row O2 a=1 Lmid@1 at subject positions (80 pairs): effect %s, flips %s." % (
                  f2(cell(m, "R", 1.0, "Lmid@1")), f2(cell(m, "R", 1.0, "Lmid@1", "flip"))), ""]
        L += ["### Arm A (derived edit at the last position), effect / flips", "",
              "| a | " + " | ".join(sites) + " |", "|---" * (len(sites) + 1) + "|"]
        for a in A_ADD:
            L.append("| %s | " % fa(a) + " | ".join("%s / %s" % (f2(cell(m, "A", a, s)), f2(cell(m, "A", a, s, "flip")))
                                                   for s in sites) + " |")
        L += ["", "Arm A change in logit A / logit B:", "", "| a | " + " | ".join(sites) + " |", "|---" * (len(sites) + 1) + "|"]
        for a in A_ADD:
            L.append("| %s | " % fa(a) + " | ".join("%s / %s" % (f2(cell(m, "A", a, s, "dA")), f2(cell(m, "A", a, s, "dB")))
                                                   for s in sites) + " |")
        L += ["", "Arm A damage (agreement on unrelated prompts; undecided rows with margin < %.1f in brackets):" % TAU, "",
              "| a | " + " | ".join(sites) + " |", "|---" * (len(sites) + 1) + "|"]
        for a in A_ADD:
            L.append("| %s | " % fa(a) + " | ".join(
                (lambda c: "-" if c is None else "%.2f [%d]" % (c["agree"] / c["n"], c["undecided"]))(
                    d["armA_damage"].get(key("A", a, s))) for s in sites) + " |")
        L += ["", "Arm A vector vs this prompt's h_B - h_A at the last position: cos / norm ratio: " + ", ".join(
            "%s %s/%s" % (s, f2(d["armA_vector_diag"][s]["cos_mean"]), f2(d["armA_vector_diag"][s]["norm_ratio_mean"]))
            for s in sites), ""]
        B = d["armB"]
        L += ["### Arm B (sentiment CAA at Lmid, all positions)", "",
              "Site points %s; %d contrast pairs; |v| %s; residual norm %s; cos between loops %s." % (
                  B["site_points"], B["n_contrast_pairs"], [round(x, 2) for x in B["v_norm"]],
                  [round(x, 2) for x in B["resid_norm"]], {k_: round(v, 2) for k_, v in B["v_cos_between_loops"].items()}),
              "", "| schedule | vector | a | positive rate | mean P(pos) | NLL | damage agree [undecided] |",
              "|---|---|---|---|---|---|---|"]
        for r_ in armB[m]:
            L.append("| %s | %s | %s | %s | %s | %s | %.2f [%d] |" % (r_["sched"], r_["vector"], fa(r_["a"]),
                                                                    f2(r_["pos_rate"]), f2(r_["p_pos"]), f2(r_["nll"]),
                                                                    r_["damage_agree"], r_["damage_undecided"]))
        L += ["", "Example (prompt 0): " + " || ".join("%s %s: %r" % (cd["sched"], fa(cd["a"]), B["gens"][0][c]["text"])
                                                       for c, cd in enumerate(B["conds"]) if c in (0, 4, 8, 12, 16)), ""]
        L += ["### Arm C (O1 at Llast@1, subject positions)", "", "| a | effect | flips | mid fraction [0.2, 0.8] | BC |",
              "|---|---|---|---|---|"]
        for a in A_C:
            pe = d["cells"][key("C", a, "Llast@1")]["per_pair_effect"]
            L.append("| %s | %s | %s | %s | %s |" % (fa(a), f2(cell(m, "C", a, "Llast@1")), f2(cell(m, "C", a, "Llast@1", "flip")),
                                                   f2(mid_fraction(pe)), f2(bimodality(pe))))
        L.append("")
        if m == "ifm_p1_ouro_inj":
            L += ["### Arm D (injection channel: input edit at one loop vs state O1 at L0 of that loop)", "",
                  "| loop | input a=0.5 | input a=1 | state a=0.5 | state a=1 |", "|---|---|---|---|---|"]
            for j in range(1, 5):
                L.append("| %d | %s | %s | %s | %s |" % (j, *["%s / %s" % (f2(cell(m, arm, a, s)), f2(cell(m, arm, a, s, "flip")))
                                                         for arm, s in (("D_input", "inp@%d" % j), ("D_state", "L0@%d" % j))
                                                         for a in A_D]))
            L.append("")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L))
    print(json.dumps(checks), flush=True)


def main():
    global N_HALF, NEUTRAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--clf", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="p")
    ap.add_argument("--n_half", type=int, default=40)          # debug only
    ap.add_argument("--n_neutral", type=int, default=30)  # debug only
    a = ap.parse_args()
    N_HALF, NEUTRAL = a.n_half, NEUTRAL[:a.n_neutral]
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
        return
    t0 = time.time()
    skipped = []
    try:
        if a.score:
            score(a.out, a.clf)
            return
        prior = spent_minutes(a.out)
        for name in [x for x in a.models.split(",") if x]:
            used = prior + (time.time() - t0) / 60.0
            if (name == "ifm_d28" and used > D28_SKIP_AFTER_MIN) or used > CAP_MIN - 5:
                print("[k15b] skip %s: %.1f GPU-min used" % (name, used), flush=True)
                skipped.append(name)
                continue
            run_model(name, a.out)
    finally:
        json.dump({"models": a.models or "score", "seconds": round(time.time() - t0, 1), "skipped": skipped},
                  open(os.path.join(a.out, "proc_%s.json" % a.tag), "w"))


if __name__ == "__main__":
    main()
