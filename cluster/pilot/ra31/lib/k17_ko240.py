"""B-K17-KO240 wrapper. Plan: loop-signal/notes/k17_ko240_plan.md (frozen before any run).
Imports k16_path_ouro (md5 23f6e13a, asserted at start) and k15_data; never modifies them.

Arms, Ouro-1.4B, eager attention, batch 1:
  repro      replay the pilot D1 selection (k15_data.select, D1_QUOTA), assert it equals the 80 stored pilot items,
             rerun the restricted Y1 2x2 and compare with the stored rows.
  new        every D1 candidate not in the pilot 80, scanned in candidate order with the pilot clean filter; the
             restricted Y1 2x2 on each passing pair. The 160-item set is chosen in the summary by the frozen rule
             (depends on the pass flags and the scan order only).
  undo3all   D2, D4, D5 (pilot selection replayed and asserted against the stored items): src_O1 at Llast@1, then
             every position except last restored to clean A at Llast@3 (audit_path_ouro.md section 5).
Restricted Y1 = k16_path_ouro.process_item with its module lists overridden: edits {off, ans_O1, src_O1} x knockouts
{none, src_all, rand0, rand1, rand2}, no width / y4 / w2 (process_item also runs Y3 undo2/undo3 and the paraphrase
vector, which gives the paraphrase mode). One extra row per D1 item: prompt B under src_all (D_KO, audit P8).

    python k17_ko240.py --selftest
    python k17_ko240.py --out OUT --pilot_dir DIR [--debug] [--arms repro,new,undo3all]
    python k17_ko240.py --summarize --out OUT --pilot_dir DIR [--plan PLAN.md]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import k15_data as KD  # noqa: E402
import k16_path_ouro as KP  # noqa: E402

K16_MD5 = "23f6e13a"
EDITS = ["off", "ans_O1", "src_O1"]
KOS = ["none", "src_all", "rand0", "rand1", "rand2"]
MARGIN = 0.10
ALPHA = 0.05
NB = 10000
N_NEW = 160
N_PILOT = 80
GAP = KP.GAP_E
D1_KEYS = [("heldout", "capital"), ("heldout", "city"), ("dev", "capital"), ("dev", "city")]
ID_KEYS = ("rel", "a", "b", "half", "pos", "T", "tA", "tB")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def restrict():
    KP.EDITS = list(EDITS)
    KP.KOS = list(KOS)
    KP.ARMS = set()


def ident(p):
    return [p["rel"], p["a"], p["b"], p["half"], sorted(p["pos"]), len(p["A"]), p["tA"], p["tB"]]


def ident_stored(it):
    return [it["rel"], it["a"], it["b"], it["half"], it["pos"], it["T"], it["tA"], it["tB"]]


# ------------------------------------------------------------------ GPU part
def make_know(R, stats):
    def know(p):
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
    return know


@torch.no_grad()
def d1_item(R, p, idx):
    KP.assign_split(p, "D1")
    it = KP.process_item(R, p, idx, None, False)
    kos, _ = KP.ko_conditions(len(p["A"]), sorted(p["pos"]), idx, R.L, R.K)
    it["rows"]["B|src_all"] = KP.vals(R.run(p["B"], ko=kos["src_all"])["z"], p["tA"], p["tB"])
    it["pos_len"] = len(p["pos"])
    p.pop("_a", None)
    p.pop("_b", None)
    return it


def row_diff(r1, r2):
    """max abs diff over z(A), z(B), lse at every loop end; top-1 equal at every loop end."""
    d = max(abs(a[k] - b[k]) for a, b in zip(r1, r2) for k in range(3))
    return d, all(a[3] == b[3] for a, b in zip(r1, r2))


@torch.no_grad()
def undo3_all_item(R, p, dev):
    A, src = p["A"], sorted(p["pos"])
    last = len(A) - 1
    a_out, b_out = p["_a"], p["_b"]
    HA = [a_out["H"][j] for j in range(1, R.K + 1)]
    HB = [b_out["H"][j] for j in range(1, R.K + 1)]
    e1 = KP.e_lerp(src, HB[0][src], 1.0, dev)
    s = R.run(A, edits={1: [e1]}, rec=True)
    u3 = R.run(A, edits={1: [e1], 3: [KP.e_lerp(src, HA[2][src], 1.0, dev)]})
    allb = list(range(0, last))
    ua = R.run(A, edits={1: [e1], 3: [KP.e_lerp(allb, HA[2][allb], 1.0, dev)]}, rec=True)
    hf = {"loop3_positions_before_last_equal_clean_A": bool(torch.equal(ua["H"][3][:last], HA[2][:last])),
          "loop3_last_equals_src_O1_run": bool(torch.equal(ua["H"][3][last], s["H"][3][last])),
          "loops1_2_equal_src_O1_run": bool(torch.equal(ua["H"][1], s["H"][1]) and torch.equal(ua["H"][2], s["H"][2])),
          "loop4_last_changed_vs_src_O1": not torch.equal(ua["H"][4][last], s["H"][4][last])}
    hf["ok"] = bool(hf["loop3_positions_before_last_equal_clean_A"] and hf["loop3_last_equals_src_O1_run"]
                    and hf["loops1_2_equal_src_O1_run"])
    tA, tB = p["tA"], p["tB"]
    it = {"rel": p["rel"], "a": p["a"], "b": p["b"], "half": p["half"], "split": p["split"], "cluster": p["cluster"],
          "pos": src, "T": len(A), "tA": tA, "tB": tB, "clean": KP.vals(a_out["z"], tA, tB),
          "cleanB": KP.vals(b_out["z"], tA, tB),
          "rows": {"Y1|src_O1|none": KP.vals(s["z"], tA, tB), "Y3|undo3": KP.vals(u3["z"], tA, tB),
                   "U|undo3_all": KP.vals(ua["z"], tA, tB)},
          "hook_f": hf}
    p.pop("_a", None)
    p.pop("_b", None)
    return it


def run(out, pilot_dir, debug, arms, budget_min):
    t0 = time.time()
    kp_md5 = md5(KP.__file__)
    if not kp_md5.startswith(K16_MD5):
        raise SystemExit("k16_path_ouro.py md5 %s != %s" % (kp_md5, K16_MD5))
    restrict()
    dev = "cuda"
    M = KP.load_ouro(KP.REPOS["ouro"], dev)
    R = KP.Runner(M)
    res = {"model": "ouro", "repo": KP.REPOS["ouro"], "device": dev, "debug": debug, "L": R.L, "K": R.K,
           "attn_implementation": M.m.config._attn_implementation, "dtype": str(next(M.m.parameters()).dtype),
           "k16_path_ouro_md5": kp_md5, "k15_data_md5": md5(KD.__file__), "wrapper_md5": md5(os.path.abspath(__file__)),
           "edits": EDITS, "kos": KOS, "load_seconds": round(time.time() - t0, 1),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "arms": {}}
    path = os.path.join(out, "k17ko_ouro.json")

    def dump(partial):
        res["partial"] = partial
        res["seconds_total"] = round(time.time() - t0, 1)
        res["forwards"] = R.n_fwd
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    d1c = KD.d1_candidates(M.enc)
    cap = [p for p in d1c if p["rel"] == "capital"][0]
    city = [p for p in d1c if p["rel"] == "city"][0]
    res["checks"] = KP.model_checks(R, cap, city)
    dump(True)
    print("[ko240] checks all_ok %s %.0fs" % (res["checks"]["all_ok"], time.time() - t0), flush=True)
    if not res["checks"]["all_ok"]:
        raise SystemExit("hook checks failed: %s" % json.dumps(res["checks"]))
    deadline = t0 + 60.0 * budget_min
    pilot = json.load(open(os.path.join(pilot_dir, "k16p_ouro.json")))
    pilot45 = json.load(open(os.path.join(pilot_dir, "k16p_ouro_d45.json")))
    res["pilot_files_md5"] = {f: md5(os.path.join(pilot_dir, f)) for f in ("k16p_ouro.json", "k16p_ouro_d45.json")}
    p80 = pilot["datasets"]["D1"]["items"]

    if "repro" in arms:
        ta = time.time()
        stats = {"A_fail": 0, "B_fail": 0}
        cands = KD.d1_candidates(M.enc)
        kept, scanned = KD.select(cands, make_know(R, stats), dict(KD.D1_QUOTA))
        same = [ident(p) == ident_stored(it) for p, it in zip(kept, p80)]
        A = {"candidates": len(cands), "scanned": scanned, "kept": len(kept), "know_stats": stats,
             "selection_identical": bool(len(kept) == len(p80) and all(same)), "n_mismatch": same.count(False),
             "items": [], "diffs": {}}
        res["arms"]["repro"] = A
        print("[ko240] repro selection identical %s (kept %d scanned %d)" % (A["selection_identical"], len(kept), scanned),
              flush=True)
        if A["selection_identical"]:
            n = 2 if debug else len(kept)
            for idx, p in enumerate(kept[:n]):
                it = d1_item(R, p, idx)
                st = p80[idx]
                dd = {"clean": row_diff(it["clean"], st["clean"]), "cleanB": row_diff(it["cleanB"], st["cleanB"])}
                for k in it["rows"]:
                    if k in st["rows"]:
                        dd[k] = row_diff(it["rows"][k], st["rows"][k])
                A["diffs"][str(idx)] = dd
                A["items"].append(it)
                if time.time() > deadline:
                    A["truncated"] = True
                    break
        for p in kept:
            p.pop("_a", None)
            p.pop("_b", None)
        A["seconds"] = round(time.time() - ta, 1)
        dump(True)
        print("[ko240] repro done %d items %.0fs" % (len(A["items"]), time.time() - ta), flush=True)

    if "new" in arms:
        ta = time.time()
        stats = {"A_fail": 0, "B_fail": 0}
        cands = KD.d1_candidates(M.enc)
        used = set((it["rel"], it["a"], it["b"]) for it in p80)
        rest = [p for p in cands if (p["rel"], p["a"], p["b"]) not in used]
        know = make_know(R, stats)
        N = {"candidates": len(cands), "pool": len(rest), "pilot_pairs_found_in_pool": len(cands) - len(rest),
             "know_stats": stats, "scan": [], "items": []}
        res["arms"]["new"] = N
        todo = rest[:3] if debug else rest
        for k, p in enumerate(todo):
            if time.time() > deadline:
                N["truncated"] = True
                break
            ok = know(p)
            N["scan"].append({"cand_idx": k, "rel": p["rel"], "a": p["a"], "b": p["b"], "half": p["half"], "pass": ok})
            if ok:
                it = d1_item(R, p, N_PILOT + len(N["items"]))
                it["cand_idx"] = k
                N["items"].append(it)
            if k % 20 == 0:
                print("[ko240] new %d/%d pass %d %.0fs" % (k + 1, len(todo), len(N["items"]), time.time() - ta), flush=True)
        N["seconds"] = round(time.time() - ta, 1)
        dump(True)
        print("[ko240] new done: scanned %d pass %d %.0fs" % (len(N["scan"]), len(N["items"]), time.time() - ta), flush=True)

    if "undo3all" in arms:
        ta = time.time()
        U = {}
        res["arms"]["undo3all"] = U
        for ds in ("D2", "D4", "D5"):
            stats = {"A_fail": 0, "B_fail": 0}
            cands, quota = KP.candidates(R.M, ds)
            kept, scanned = KD.select(cands, make_know(R, stats), quota)
            st = (pilot if ds == "D2" else pilot45)["datasets"][ds]["items"]
            same = [ident(p) == ident_stored(it) for p, it in zip(kept, st)]
            D = {"candidates": len(cands), "scanned": scanned, "kept": len(kept), "know_stats": stats,
                 "selection_identical": bool(len(kept) == len(st) and all(same)), "n_mismatch": same.count(False),
                 "items": [], "diffs": {}}
            if ds in KP.EXT_QUOTA:
                D["data_md5"] = md5(os.path.join(KP.DATA_DIR, "%s_ouro.jsonl" % ds.lower()))
            U[ds] = D
            if D["selection_identical"]:
                n = 2 if debug else len(kept)
                for idx, p in enumerate(kept[:n]):
                    KP.assign_split(p, ds)
                    it = undo3_all_item(R, p, dev)
                    it["idx"] = idx
                    s0 = st[idx]
                    dd = {"clean": row_diff(it["clean"], s0["clean"]), "cleanB": row_diff(it["cleanB"], s0["cleanB"])}
                    for k in ("Y1|src_O1|none", "Y3|undo3"):
                        dd[k] = row_diff(it["rows"][k], s0["rows"][k])
                    D["diffs"][str(idx)] = dd
                    D["items"].append(it)
            for p in kept:
                p.pop("_a", None)
                p.pop("_b", None)
            print("[ko240] undo3all %s identical %s items %d %.0fs" % (ds, D["selection_identical"], len(D["items"]),
                                                                       time.time() - ta), flush=True)
            dump(True)
        res["undo3all_seconds"] = round(time.time() - ta, 1)
    dump(False)
    print("[ko240] done %.0fs, %d forwards" % (time.time() - t0, R.n_fwd), flush=True)


# ------------------------------------------------------------------ statistics (CPU)
def _groups(xs, cl):
    g = {}
    for x, c in zip(xs, cl):
        if x is not None:
            g.setdefault(c, []).append(float(x))
    return g


def boot_dist(xs, cl, n=NB, seed=0):
    """Item mean and its cluster-bootstrap distribution (same estimator as k16_path_ouro.boot_cluster)."""
    g = _groups(xs, cl)
    if not g:
        return None
    keys = sorted(g)
    sums = torch.tensor([sum(g[k]) for k in keys], dtype=torch.float64)
    cnts = torch.tensor([len(g[k]) for k in keys], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=gen)
    return float(sums.sum() / cnts.sum()), sums[idx].sum(1) / cnts[idx].sum(1), int(cnts.sum()), len(keys)


def q(dist, ps):
    return [round(float(x), 4) for x in torch.quantile(dist, torch.tensor(ps, dtype=torch.float64))]


def tost_from(v0, dist, n_items, n_cl, m=MARGIN, alpha=ALPHA):
    B = dist.numel()
    pu = (1 + int((dist >= m).sum())) / (B + 1.0)
    pl = (1 + int((dist <= -m).sum())) / (B + 1.0)
    c90 = q(dist, [alpha, 1 - alpha])
    return {"mean": round(v0, 4), "ci95": q(dist, [0.025, 0.975]), "ci90": c90, "p_upper": round(pu, 5),
            "p_lower": round(pl, 5), "p_tost": round(max(pu, pl), 5),
            "equivalent": bool(max(pu, pl) < alpha and -m < c90[0] and c90[1] < m), "margin": m,
            "n_items": n_items, "n_clusters": n_cl, "n_boot": B}


def tost(xs, cl, m=MARGIN, n=NB, seed=0):
    b = boot_dist(xs, cl, n, seed)
    if b is None:
        return None
    return tost_from(b[0], b[1], b[2], b[3], m)


def tost_twoway(xs, sa, sb, m=MARGIN, n=NB, seed=0):
    """Pigeonhole (two-way subject) bootstrap: each subject drawn with multinomial multiplicity w_s; item weight
    w_a * w_b (audit P9, F8: 18 of 19 capital A-subjects also appear as B)."""
    keep = [i for i, x in enumerate(xs) if x is not None]
    if not keep:
        return None
    subs = sorted(set(sa[i] for i in keep) | set(sb[i] for i in keep))
    ix = {s: k for k, s in enumerate(subs)}
    x = torch.tensor([float(xs[i]) for i in keep], dtype=torch.float64)
    ia = torch.tensor([ix[sa[i]] for i in keep])
    ib = torch.tensor([ix[sb[i]] for i in keep])
    gen = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(subs), (n, len(subs)), generator=gen)
    W = torch.zeros(n, len(subs), dtype=torch.float64)
    W.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    w = W[:, ia] * W[:, ib]
    tot = w.sum(1)
    ok = tot > 0
    dist = (w[ok] * x).sum(1) / tot[ok]
    r = tost_from(float(x.mean()), dist, len(keep), len(subs), m)
    r["n_subjects"] = len(subs)
    r["draws_all_zero_weight"] = int((~ok).sum())
    return r


def ci(xs, cl, n=NB, seed=0):
    b = boot_dist(xs, cl, n, seed)
    if b is None:
        return None
    return [round(b[0], 4)] + q(b[1], [0.025, 0.975]) + [b[2], b[3]]


# ------------------------------------------------------------------ per-item metrics
def fin(it, key):
    r = it["rows"].get(key)
    return None if r is None else KP.met(r, it["clean"], it["tA"], it["tB"])


def metrics(it):
    m = {}
    e0 = fin(it, "Y1|ans_O1|none")
    e1 = KP.E1(it, "ans_O1", "src_all")
    er = KP.E1(it, "ans_O1", "rand")
    D = KP.Dgap(it)
    m["D"] = D
    m["E0"] = None if e0 is None else e0["dm"]
    m["E1"] = e1
    m["rescue"] = None if e0 is None or e1 is None else e1 - e0["dm"]
    m["rescue_rand"] = None if e0 is None or er is None else er - e0["dm"]
    m["rescue_e"] = None if m["rescue"] is None or D < GAP else m["rescue"] / D
    m["flip_none"] = None if e0 is None else e0["flip"]
    f1 = fin(it, "Y1|ans_O1|src_all")
    m["flip_src_all"] = None if f1 is None else f1["flip"]
    s0 = fin(it, "Y1|src_O1|none")
    s1 = KP.E1(it, "src_O1", "src_all")
    m["src_E0"] = None if s0 is None else s0["dm"]
    m["src_E1"] = s1
    m["src_e0"] = None if s0 is None or D < GAP else s0["dm"] / D
    m["src_E1_over_D"] = None if s1 is None or D < GAP else s1 / D
    m["off_src_all_dm"] = (fin(it, "Y1|off|src_all") or {}).get("dm")
    bk, ok = it["rows"].get("B|src_all"), it["rows"].get("Y1|off|src_all")
    if bk is not None and ok is not None:
        dko = (bk[-1][1] - bk[-1][0]) - (ok[-1][1] - ok[-1][0])
        m["D_KO"] = dko
        m["rescue_e_KO"] = None if m["rescue"] is None or dko < GAP else m["rescue"] / dko
    else:
        m["D_KO"] = m["rescue_e_KO"] = None
    return m


def para_mode(it):
    ms = set(it.get("para_modes", []))
    return "aligned" if ms == {"aligned"} else ("last_token" if ms == {"last_token"} else "mixed")


def subj(it, side):
    return "%s:%s" % (it["rel"], it[side])


# ------------------------------------------------------------------ frozen selection rule for the 160
def select_new(scan, n_new=N_NEW):
    """scan: dicts in candidate order with half, rel, a, b, pass. Pilot rule (k15_data.select, cap per (half, rel,
    subject) counting both sides, quota unlimited) at the smallest cap c >= 2 giving >= n_new kept; then drop capital
    items from the end of the scan order down to n_new. Returns (kept cand_idx list, c, kept count at c=2)."""
    def sel(c):
        kept, _ = KD.select([dict(s, half=s["half"]) for s in scan], lambda s: s["pass"],
                            {k: 10 ** 6 for k in D1_KEYS}, cap=c)
        return [s["cand_idx"] for s in kept]
    k2 = sel(2)
    c, kept = 2, k2
    while len(kept) < n_new and c < 50:
        c += 1
        kept = sel(c)
    if len(kept) > n_new:
        rel = {s["cand_idx"]: s["rel"] for s in scan}
        drop = len(kept) - n_new
        out = []
        for ci_ in reversed(kept):
            if drop > 0 and rel[ci_] == "capital":
                drop -= 1
                continue
            out.append(ci_)
        kept = list(reversed(out))
    return kept, c, k2


def selftest():
    res = {"k16_selftest": KP.selftest()}
    r0 = tost([0.0, 0.01, -0.01, 0.0] * 20, [str(i // 2) for i in range(80)])
    r1 = tost([0.2, 0.25, 0.15, 0.2] * 20, [str(i // 2) for i in range(80)])
    res["tost_null_equivalent"] = r0["equivalent"] and r0["p_tost"] < 0.001
    res["tost_shift_not_equivalent"] = (not r1["equivalent"]) and r1["p_tost"] > 0.5
    b = boot_dist([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"], n=1000)
    res["boot_matches_k16"] = KP.boot_cluster([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"])[:3] == \
        [round(b[0], 4)] + [round(float(x), 4) for x in torch.quantile(b[1], torch.tensor([0.025, 0.975], dtype=torch.float64))]
    tw = tost_twoway([0.0, 0.0, 0.0], ["x", "y", "z"], ["y", "z", "x"])
    res["twoway_degenerate_zero"] = tw["mean"] == 0.0 and tw["ci90"] == [0.0, 0.0]
    scan = []
    for i in range(30):
        scan.append({"cand_idx": i, "half": "heldout", "rel": "capital" if i < 24 else "city", "a": "s%d" % (i // 3),
                     "b": "t%d" % (i % 5), "pass": i % 7 != 3})
    kept, c, k2 = select_new(scan, 12)
    res["select_new"] = {"n": len(kept), "c": c, "k2": len(k2)}
    res["select_new_ok"] = len(kept) == 12 and len(set(kept)) == 12 and all(scan[k]["pass"] for k in kept) and \
        kept == sorted(kept)
    res["ok"] = bool(res["k16_selftest"] and res["tost_null_equivalent"] and res["tost_shift_not_equivalent"]
                     and res["boot_matches_k16"] and res["twoway_degenerate_zero"] and res["select_new_ok"])
    print(json.dumps(res))
    return res["ok"]


# ------------------------------------------------------------------ summary
def summarize(out, pilot_dir, plan):
    d = json.load(open(os.path.join(out, "k17ko_ouro.json")))
    pilot = json.load(open(os.path.join(pilot_dir, "k16p_ouro.json")))
    p80 = pilot["datasets"]["D1"]["items"]
    N = d["arms"]["new"]
    newit = {it["cand_idx"]: it for it in N["items"]}
    kept, c, k2 = select_new(N["scan"])
    sets = {"pilot80_stored": p80,
            "new160": [newit[k] for k in kept],
            "pooled240": p80 + [newit[k] for k in kept],
            "new_cap2": [newit[k] for k in k2],
            "pooled_cap2": p80 + [newit[k] for k in k2],
            "new_all": list(N["items"]),
            "pooled_all": p80 + list(N["items"])}
    rp = d["arms"].get("repro", {})
    if rp.get("items"):
        sets["pilot80_rerun"] = rp["items"]
    M = {nm: [metrics(it) for it in its] for nm, its in sets.items()}
    CL = {nm: [it["cluster"] for it in its] for nm, its in sets.items()}

    def cell(nm, key, idx=None):
        ii = range(len(sets[nm])) if idx is None else idx
        return ci([M[nm][i][key] for i in ii], [CL[nm][i] for i in ii])

    ck = {}
    ck["hook_identity"] = {k: d["checks"].get(k) for k in ("all_ok", "hook_identity_ok", "call_counts_ok",
                                                            "wrapper_no_knockout_bit_identical",
                                                            "all_zero_additive_mask_bit_identical",
                                                            "zero_edit_norm_and_lmid_bit_identical")}
    ck["hook_identity"]["knockout_placement_ok"] = d["checks"]["knockout_placement"]["ok"]
    ck["hook_identity"]["norm_edit_placement_ok"] = d["checks"]["norm_edit_placement"]["ok"]
    ck["run"] = {"attn_implementation": d["attn_implementation"], "dtype": d["dtype"], "batch": 1,
                 "k16_path_ouro_md5": d["k16_path_ouro_md5"], "wrapper_md5": d["wrapper_md5"],
                 "cuda_visible_devices": d["cuda_visible_devices"], "forwards": d["forwards"],
                 "seconds_total": d["seconds_total"], "partial": d["partial"], "debug": d["debug"]}
    # pilot reproduction
    if rp:
        mx, top = 0.0, True
        for dd in rp.get("diffs", {}).values():
            for v in dd.values():
                mx = max(mx, v[0])
                top = top and v[1]
        ck["pilot_repro"] = {"selection_identical": rp["selection_identical"], "scanned": rp["scanned"],
                             "items_rerun": len(rp.get("items", [])), "max_abs_logit_diff": round(mx, 5),
                             "top1_identical_all_loops": top}
    # new-pool scan
    ck["new_selection"] = {"pool": N["pool"], "scanned": len(N["scan"]), "pass": len(N["items"]),
                           "know_stats": N["know_stats"], "cap_used": c, "kept_at_cap2": len(k2),
                           "new160_n": len(kept),
                           "new160_by_rel": {r: sum(1 for k in kept if newit[k]["rel"] == r) for r in ("capital", "city")},
                           "truncated": bool(N.get("truncated"))}
    # primary and sensitivity TOST
    T = {}
    for nm in ("pilot80_stored", "pilot80_rerun", "new160", "pooled240", "pooled_cap2", "pooled_all"):
        if nm not in sets:
            continue
        xs = [m["rescue"] for m in M[nm]]
        T[nm] = {"logit": tost(xs, CL[nm]),
                 "e": tost([m["rescue_e"] for m in M[nm]], CL[nm]),
                 "e_KO": tost([m["rescue_e_KO"] for m in M[nm]], CL[nm]) if nm != "pilot80_stored" and
                 any(m["rescue_e_KO"] is not None for m in M[nm]) else None,
                 "logit_twoway": tost_twoway(xs, [subj(it, "a") for it in sets[nm]], [subj(it, "b") for it in sets[nm]]),
                 "n_e_excluded_D_lt_2": sum(1 for m in M[nm] if m["D"] < GAP)}
    ck["tost"] = T
    p = T["pooled240"]
    ck["primary"] = {"cell": "pooled240, ans_O1 rescue under src_all, logits", "mean": p["logit"]["mean"],
                     "ci95": p["logit"]["ci95"], "ci90": p["logit"]["ci90"], "p_tost": p["logit"]["p_tost"],
                     "equivalent": p["logit"]["equivalent"], "n_items": p["logit"]["n_items"],
                     "n_clusters": p["logit"]["n_clusters"],
                     "e_scale": {k: p["e"][k] for k in ("mean", "ci90", "p_tost", "equivalent", "n_items")}}
    # cells, strata, controls
    C = {}
    for nm in ("pilot80_stored", "new160", "pooled240"):
        its = sets[nm]
        strata = {"all": None,
                  "rel=capital": [i for i, it in enumerate(its) if it["rel"] == "capital"],
                  "rel=city": [i for i, it in enumerate(its) if it["rel"] == "city"],
                  "pos_len=1": [i for i, it in enumerate(its) if len(it["pos"]) == 1],
                  "pos_len>=2": [i for i, it in enumerate(its) if len(it["pos"]) >= 2],
                  "para=aligned": [i for i, it in enumerate(its) if para_mode(it) == "aligned"],
                  "para=last_token": [i for i, it in enumerate(its) if para_mode(it) == "last_token"]}
        C[nm] = {}
        for sn, idx in strata.items():
            if idx is not None and not idx:
                continue
            C[nm][sn] = {k: cell(nm, k, idx) for k in ("rescue", "rescue_e", "E0", "E1", "flip_none", "flip_src_all",
                                                         "rescue_rand", "src_E0", "src_e0", "src_E1", "src_E1_over_D",
                                                         "off_src_all_dm", "D", "D_KO", "rescue_e_KO")}
    ck["cells"] = C
    # undo3_all
    U = d["arms"].get("undo3all", {})
    UC = {}
    hf_all = True
    mxu = 0.0
    for ds, D_ in U.items():
        its = D_["items"]
        hf_all = hf_all and all(it["hook_f"]["ok"] for it in its)
        for dd in D_.get("diffs", {}).values():
            for v in dd.values():
                mxu = max(mxu, v[0])
        subsets = [("D2", its)] if ds == "D2" else ([("D4/T1", [it for it in its if it["rel"] == "T1"]),
                                                    ("D4/T2", [it for it in its if it["rel"] == "T2"])]
                                                   if ds == "D4" else [("D5", its)])
        for sn, ss in subsets:
            if not ss:
                continue
            cl = [it["cluster"] for it in ss]

            def e_of(it, key):
                r = fin(it, key)
                return None if r is None or KP.Dgap(it) < GAP else r["dm"] / KP.Dgap(it)
            rec = {"n_items": len(ss), "n_clusters": len(set(cl)), "n_e_excluded": sum(1 for it in ss if KP.Dgap(it) < GAP),
                   "selection_identical": D_["selection_identical"]}
            for key, nm in (("Y1|src_O1|none", "src_O1"), ("Y3|undo3", "undo3"), ("U|undo3_all", "undo3_all")):
                rec[nm] = {"e": ci([e_of(it, key) for it in ss], cl),
                           "flip": ci([(fin(it, key) or {}).get("flip") for it in ss], cl),
                           "dm_by_loop": [(ci([(KP.met(it["rows"][key], it["clean"], it["tA"], it["tB"], j) or {}).get("dm")
                                               for it in ss], cl) or [None])[0] for j in range(4)]}
            rec["relay_share_e"] = ci([None if e_of(it, "Y3|undo3") is None else e_of(it, "Y3|undo3") - e_of(it, "U|undo3_all")
                                       for it in ss], cl)
            UC[sn] = rec
    ck["undo3_all"] = UC
    ck["undo3_all_hook_f_ok"] = hf_all if U else None
    ck["undo3_all_repro_max_abs_logit_diff"] = round(mxu, 5) if U else None
    if "D4/T2" in UC and "D5" in UC and UC["D4/T2"]["undo3_all"]["e"] and UC["D5"]["undo3_all"]["e"]:
        v = {k: UC[k]["undo3_all"]["e"][0] for k in ("D4/T2", "D5")}

        def br(x):
            return "H2 (>= 0.6)" if x >= 0.6 else ("relay (<= 0.15)" if x <= 0.15 else "between")
        per = {k: br(x) for k, x in v.items()}
        ck["undo3_all_verdict"] = {"per_dataset": per, "e": v,
                                   "overall": per["D4/T2"] if per["D4/T2"] == per["D5"] else "mixed: " + json.dumps(per),
                                   "relay_share_e": {k: UC[k]["relay_share_e"] for k in ("D4/T2", "D5")}}
    # gpu minutes (wall time of the GPU processes, load included)
    procs = [json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out)) if f.startswith("proc_")]
    ck["gpu_minutes"] = round(sum(x["seconds"] for x in procs if x.get("device") == "cuda") / 60.0, 2)
    if plan and os.path.exists(plan):
        ck["plan_sha256"] = hashlib.sha256(open(plan, "rb").read()).hexdigest()
    ck["items"] = {nm: {"n_items": len(its), "n_clusters": len(set(CL[nm]))} for nm, its in sets.items()}
    json.dump(ck, open(os.path.join(out, "checks_detail.json"), "w"), indent=1)
    json.dump({"run": d, "sets": {k: [(it.get("cand_idx"), it["rel"], it["a"], it["b"]) for it in v]
                                  for k, v in sets.items()}, "metrics": M}, open(os.path.join(out, "results.json"), "w"))
    cc = compact(ck)
    with open(os.path.join(out, "checks.json"), "w") as fh:
        fh.write("{\n" + ",\n".join('"%s": %s' % (k, json.dumps(v)) for k, v in cc.items()) + "\n}\n")
    write_summary(out, ck)
    print(json.dumps(ck["primary"]), flush=True)
    return ck


def compact(ck):
    """checks.json with the plan's keys; [mean, lo95, hi95] cells, 3 decimals."""
    def r(v):
        return None if v is None else [round(x, 3) for x in v[:3]]

    def t(x):
        if x is None:
            return None
        return {"mean": round(x["mean"], 3), "ci95": [round(v, 3) for v in x["ci95"]],
                "ci90": [round(v, 3) for v in x["ci90"]], "p_tost": x["p_tost"], "equivalent": x["equivalent"],
                "n_items": x["n_items"], "n_clusters": x.get("n_clusters")}
    T = ck["tost"]
    P = ck["cells"]["pooled240"]
    N = ck["cells"]["new160"]
    out = {"hook_identity": ck["hook_identity"], "pilot_repro": ck.get("pilot_repro"),
           "new_selection": ck["new_selection"], "items": ck["items"],
           "ko240_primary": dict(t(T["pooled240"]["logit"]), unit="logits", margin=MARGIN, alpha=ALPHA),
           "ko240_e_scale": dict(t(T["pooled240"]["e"]), unit="e = rescue / D", margin=MARGIN, alpha=ALPHA),
           "new160": {"logit": t(T["new160"]["logit"]), "e": t(T["new160"]["e"])},
           "sensitivity": {"pooled_cap2_logit": t(T["pooled_cap2"]["logit"]), "pooled_all_logit": t(T["pooled_all"]["logit"]),
                           "pooled240_twoway_logit": t(T["pooled240"]["logit_twoway"]),
                           "pooled240_e_KO": t(T["pooled240"]["e_KO"]),
                           "pilot80_stored_logit": t(T["pilot80_stored"]["logit"]),
                           "pilot80_rerun_logit": t((T.get("pilot80_rerun") or {}).get("logit"))},
           "strata": {sn: {"rescue": r(v["rescue"]), "rescue_e": r(v["rescue_e"]), "n_items": v["rescue"][3],
                           "n_clusters": v["rescue"][4]} for sn, v in P.items()},
           "positive_control": {nm: {"src_e0": r(c["all"]["src_e0"]), "src_E1_over_D_src_all": r(c["all"]["src_E1_over_D"]),
                                     "capital_src_E1_over_D_src_all": r((c.get("rel=capital") or {}).get("src_E1_over_D"))}
                                for nm, c in (("pooled240", P), ("new160", N))},
           "rand_control": {"pooled240_rescue_rand": r(P["all"]["rescue_rand"]), "new160_rescue_rand": r(N["all"]["rescue_rand"])},
           "undo3_all": {sn: {"n_items": v["n_items"], "n_clusters": v["n_clusters"], "src_O1_e": r(v["src_O1"]["e"]),
                              "undo3_e": r(v["undo3"]["e"]), "undo3_all_e": r(v["undo3_all"]["e"]),
                              "undo3_all_flip": r(v["undo3_all"]["flip"]), "relay_share_e": r(v["relay_share_e"])}
                         for sn, v in ck["undo3_all"].items()},
           "undo3_all_hook_f": ck["undo3_all_hook_f_ok"],
           "undo3_all_repro_max_abs_logit_diff": ck["undo3_all_repro_max_abs_logit_diff"],
           "undo3_all_verdict": ck.get("undo3_all_verdict")}
    pv = {}
    pl, pe = T["pooled240"]["logit"], T["pooled240"]["e"]
    pv["K1_brief_pooled_within_0.10"] = {"logit": pl["equivalent"], "e": pe["equivalent"]}
    pv["K2_e_scale_p_lt_0.001"] = pe["p_tost"] < 0.001
    pv["K3_logit_tost_passes"] = {"cluster": pl["equivalent"], "twoway": T["pooled240"]["logit_twoway"]["equivalent"]}
    nl = T["new160"]["logit"]
    cap_m = (N.get("rel=capital") or {}).get("rescue")
    city_m = (N.get("rel=city") or {}).get("rescue")
    pv["K4_new160_mean_in_0.03_0.12"] = 0.03 <= nl["mean"] <= 0.12
    pv["K4_capital_gt_city"] = None if cap_m is None or city_m is None else cap_m[0] > city_m[0]
    cp = (N.get("rel=capital") or {}).get("src_E1_over_D")
    pv["K5_positive_control"] = bool(N["all"]["src_e0"][0] >= 0.9 and cp is not None and cp[0] <= 0.02)
    pv["K6_rand_mean_within_0.10"] = abs(P["all"]["rescue_rand"][0]) <= 0.10
    rp = ck.get("pilot_repro") or {}
    pv["K7_pilot_rerun_bit_identical"] = None if not rp else rp["max_abs_logit_diff"] == 0.0
    ek, ee = T["pooled240"]["e_KO"], pe
    pv["K8_e_KO_vs_e"] = None if ek is None else {"e_KO": round(ek["mean"], 4), "e": round(ee["mean"], 4)}
    v = ck.get("undo3_all_verdict")
    pv["U1_undo3_all_branch"] = None if v is None else v["overall"]
    U = ck["undo3_all"]
    pv["U2_D4T1_le_0.15"] = None if not (U.get("D4/T1") or {}).get("undo3_all", {}).get("e") else \
        U["D4/T1"]["undo3_all"]["e"][0] <= 0.15
    pv["U2_D2_le_0.15"] = None if not (U.get("D2") or {}).get("undo3_all", {}).get("e") else \
        U["D2"]["undo3_all"]["e"][0] <= 0.15
    pv["U3_hook_f_and_repro"] = {"hook_f": ck["undo3_all_hook_f_ok"], "repro_diff": ck["undo3_all_repro_max_abs_logit_diff"]}
    out["predictions_vs_outcomes"] = pv
    out["gpu_minutes"] = ck["gpu_minutes"]
    out["plan_sha256"] = ck.get("plan_sha256")
    return out


def write_summary(out, ck):
    def f(v):
        if v is None:
            return "-"
        if isinstance(v, list):
            return "%.3f [%.3f, %.3f]" % (v[0], v[1], v[2])
        return "%.3f" % v

    def ft(t):
        if t is None:
            return "-"
        return "%.3f, 95%% [%.3f, %.3f], 90%% [%.3f, %.3f], p_TOST %.4f, %s, n %d / %s cl" % (
            t["mean"], t["ci95"][0], t["ci95"][1], t["ci90"][0], t["ci90"][1], t["p_tost"],
            "EQUIVALENT" if t["equivalent"] else "not shown", t["n_items"], t.get("n_clusters"))
    L = ["# B-K17-KO240: knockout rescue null on Ouro-1.4B, D1, pooled 240", "",
         "Ouro-1.4B, eager attention, batch 1, k16_path_ouro.py md5 %s unchanged (wrapper k17_ko240.py). Edit ans_O1 = "
         "H_1[last] <- H_1^B[last] at Llast@1; knockout src_all = last->source attention removed in loops 2-4, all "
         "layers. rescue = [dm(ans_O1, src_all) - dm(off, src_all)] - dm(ans_O1, none), final readout, logits. e = "
         "rescue / D (D >= 2). TOST margin %.2f, alpha %.2f, cluster bootstrap over A-side subjects (%d resamples, "
         "seed 0); two-way = pigeonhole bootstrap over subjects on both sides." % (
             ck["run"]["k16_path_ouro_md5"][:8], MARGIN, ALPHA, NB), "",
         "GPU minutes: %s. Forwards: %s." % (ck["gpu_minutes"], ck["run"]["forwards"]), "",
         "## Selection", "", "```", json.dumps(ck["new_selection"]), json.dumps(ck.get("pilot_repro")), "```", "",
         "## TOST (rescue)", "", "| set | logits | e | e per D_KO | logits, two-way |", "|---|---|---|---|---|"]
    for nm, t in ck["tost"].items():
        L.append("| %s | %s | %s | %s | %s |" % (nm, ft(t["logit"]), ft(t["e"]), ft(t["e_KO"]), ft(t["logit_twoway"])))
    L += ["", "## Cells by stratum: mean [95% CI] (items, clusters)", ""]
    for nm, cc in ck["cells"].items():
        L += ["### %s" % nm, "", "| stratum | rescue | rescue e | E0 ans | flip none / src_all | rand rescue | src e0 | "
              "src E1 / D | off src_all dm | D | D_KO |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for sn, v in cc.items():
            L.append("| %s (%s, %s) | %s | %s | %s | %.2f / %.2f | %s | %s | %s | %s | %s | %s |" % (
                sn, v["rescue"][3], v["rescue"][4], f(v["rescue"]), f(v["rescue_e"]), f(v["E0"]), v["flip_none"][0],
                v["flip_src_all"][0], f(v["rescue_rand"]), f(v["src_e0"]), f(v["src_E1_over_D"]), f(v["off_src_all_dm"]),
                f(v["D"]), f(v["D_KO"])))
        L.append("")
    if ck.get("undo3_all"):
        L += ["## undo3_all (src_O1 at Llast@1; every position but last restored to clean A at Llast@3)", "",
              "Hook check (f) all items: %s. Rerun vs stored src_O1 / undo3 rows, max abs logit diff: %s." % (
                  ck["undo3_all_hook_f_ok"], ck["undo3_all_repro_max_abs_logit_diff"]), "",
              "| subset | n (cl) | src_O1 e | undo3 e | undo3_all e | undo3_all flip | relay share e | undo3_all dm by loop |",
              "|---|---|---|---|---|---|---|---|"]
        for sn, r in ck["undo3_all"].items():
            L.append("| %s | %d (%d) | %s | %s | %s | %.2f | %s | %s |" % (
                sn, r["n_items"], r["n_clusters"], f(r["src_O1"]["e"]), f(r["undo3"]["e"]), f(r["undo3_all"]["e"]),
                r["undo3_all"]["flip"][0], f(r["relay_share_e"]),
                ", ".join("-" if x is None else "%.2f" % x for x in r["undo3_all"]["dm_by_loop"])))
        L += ["", "Verdict (frozen rule): %s" % json.dumps(ck.get("undo3_all_verdict")), ""]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--pilot_dir", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--arms", default="repro,new,undo3all")
    ap.add_argument("--budget_min", type=float, default=50.0)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out, a.pilot_dir, a.plan)
        return
    t0 = time.time()
    try:
        run(a.out, a.pilot_dir, a.debug, set(a.arms.split(",")), a.budget_min)
    finally:
        json.dump({"seconds": round(time.time() - t0, 1), "device": "cuda", "mode": "debug" if a.debug else "full",
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "arms": a.arms},
                  open(os.path.join(a.out, "proc_%s.json" % a.tag), "w"))


if __name__ == "__main__":
    main()
