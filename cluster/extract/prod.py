"""Extractor for the production run (prod.steer.runner records + driver/analyze_full.py -> out/summary_full.json).

Rows written, one per (cell, table, arm, endpoint) where the endpoint is a stored {v, ci90, ci95, n_items, n_clusters}:
  <cell>.endpoints.<arm>.<endpoint>                    item_set own (all splits pooled)
  <cell>.endpoints_test.<arm>.<endpoint>               item_set test (X1, X3, X4a)
  <cell>.endpoints_dev.<arm>.<endpoint>                item_set dev (X6)
  <cell>.endpoints_by_stratum.<s>.<arm>.<endpoint>     item_set stratum:<s> (D2, D4, CF)
cell id = <card>__<model>__<dataset>. card: X1 / X1v edit_site_map (grid B), X3 restore, X4a branch_freeze (grid C),
X6 attention_knockout; grid is left empty for X3 and X6 (no grid assignment found for them). model M1 -> ouro,
M3 -> ouro26. dataset D1 facts_capital, CF counterfact, D2 addition, D4 code, VB gsm8k_copy (X1v).
Both intervals are stored by analyze_full.ratio_boot (10000 resamples, seed 0). metric: e for e / e_itemmean /
e_span / ko_e*, otherwise the endpoint name (flip, rho, dep, retention, ...). site / loop / strength from the
records header spec (first patch with a named site: "<span>|<site>", loop from loopN); knockout-only arms take the
boundary from the arm name ("boundary|<layer>"). Random control (X1, X1v): the paired rnd_<arm>_s1 arm, same table
and endpoint. Positive control: <cell>.endpoints.pc_final.e.v.
"""
import json
import os
import re
import sys

from . import common, resum

SCRIPT = "analyze_full.py"
CARD = {"X1": "edit_site_map", "X1v": "edit_site_map", "X3": "restore", "X4a": "branch_freeze",
        "X6": "attention_knockout"}
GRID = {"X1": "B", "X1v": "B", "X4a": "C"}
MODEL = {"M1": "ouro", "M3": "ouro26"}
DATASET = {"D1": "facts_capital", "CF": "counterfact", "D2": "addition", "D4": "code", "VB": "gsm8k_copy"}
RND = {"src_L0_1": "rnd_src_L0_1_s1", "src_Llast_1": "rnd_src_Llast_1_s1", "ans_Llast_1": "rnd_ans_Llast_1_s1"}
LOOP_RE = re.compile(r"loop(\d+)")
BOUND_RE = re.compile(r"_(L0|Lmid|Llast)_(\d+)")
ITEM_RE = re.compile(r'"item": "((?:[^"\\]|\\.)*)"')


def _src(job_dir, src_py):
    return src_py or os.path.join(job_dir, "driver", SCRIPT)


def _summary(job_dir):
    return common.load_json(os.path.join(job_dir, "out", "summary_full.json"))


def _records(job_dir, cell):
    return os.path.join(job_dir, "out", cell, "k4", "records_forward.jsonl")


def _header(job_dir, cell):
    with open(_records(job_dir, cell), "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def _arm_sites(hdr):
    out = {}
    for a in (hdr.get("spec") or {}).get("arms", []):
        site, loop, strength = "", "", ""
        for iv in a.get("interventions", []):
            if isinstance(iv.get("site"), str):
                pos = iv.get("positions") or {}
                site = "%s|%s" % (pos.get("span") or pos.get("kind") or "", iv["site"])
                m = LOOP_RE.search(iv["site"])
                loop = int(m.group(1)) if m else ""
                strength = iv.get("a", "")
                break
        if not site:
            m = BOUND_RE.search(a["name"])
            if m and a["name"].startswith("ko_"):
                site, loop = "boundary|%s" % m.group(1), int(m.group(2))
        out[a["name"]] = (site, loop, strength)
    return out


def _tables(o):
    yield "endpoints", "own", o.get("endpoints") or {}
    if o.get("endpoints_test"):
        yield "endpoints_test", "test", o["endpoints_test"]
    if o.get("endpoints_dev"):
        yield "endpoints_dev", "dev", o["endpoints_dev"]
    for s, t in (o.get("endpoints_by_stratum") or {}).items():
        yield "endpoints_by_stratum.%s" % s, "stratum:%s" % s, t


def _metric(ep):
    return "e" if (ep == "e" or ep.startswith("e_") or ep.startswith("ko_e")) else ep


def extract(job_dir, src_py=None):
    S = _summary(job_dir)
    rows = []
    for cell in sorted(S):
        o = S[cell]
        card, mid, did = cell.split("__")
        sites = _arm_sites(_header(job_dir, cell))
        pc = (((o.get("endpoints") or {}).get("pc_final") or {}).get("e") or {}).get("v")
        for tname, iset, tab in _tables(o):
            for arm, eps in tab.items():
                site, loop, strength = sites.get(arm, ("", "", ""))
                for ep, v in eps.items():
                    if not (isinstance(v, dict) and "ci90" in v):
                        continue
                    rnd = None
                    if card in ("X1", "X1v") and arm in RND:
                        rv = (tab.get(RND[arm]) or {}).get(ep)
                        rnd = rv["v"] if isinstance(rv, dict) else None
                    rows.append(common.row(
                        grid=GRID.get(card, ""), card=CARD[card], script=SCRIPT, model=MODEL.get(mid, mid),
                        dataset=DATASET.get(did, did), arm=arm, site=site, loop=loop, strength=strength,
                        metric=_metric(ep), endpoint=ep, item_set=iset, n_items=v["n_items"],
                        n_clusters=v["n_clusters"], estimate=v["v"], ci90_lo=v["ci90"][0], ci90_hi=v["ci90"][1],
                        ci90_method="stored", ci95_lo=v["ci95"][0], ci95_hi=v["ci95"][1],
                        random_control_estimate=rnd, positive_control_value=pc,
                        source_key="summary_full.%s.%s.%s.%s" % (cell, tname, arm, ep)))
    return rows


def _done_items(path):
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        f.readline()                                   # header
        for line in f:
            m = ITEM_RE.search(line)
            if m:
                out.add(json.loads('"%s"' % m.group(1)))
    return out


def item_sets_by_cell(job_dir):
    """{cell: (planned keys from out/kept/<model>__<ds>.json, item ids with a row in the cell's records)}."""
    S = _summary(job_dir)
    out = {}
    for cell in sorted(S):
        card, mid, did = cell.split("__")
        kp = os.path.join(job_dir, "out", "kept", "%s__%s.json" % (mid, did))
        planned = set(common.load_json(kp)["keys"]) if os.path.exists(kp) else set()
        out[cell] = (planned, _done_items(_records(job_dir, cell)))
    return out


def item_sets(job_dir):
    """{(model, dataset): (planned, done)}; done = items with a row in every cell of that model and dataset
    (item_sets_by_cell gives the per-cell sets)."""
    out = {}
    for cell, (p, d) in item_sets_by_cell(job_dir).items():
        card, mid, did = cell.split("__")
        k = (MODEL.get(mid, mid), DATASET.get(did, did))
        if k in out:
            out[k] = (out[k][0], out[k][1] & d)
        else:
            out[k] = (p, d)
    return out


def _load_analyze(path):
    argv = sys.argv
    sys.argv = [path, "<unused>"]                      # analyze_full reads ROOT = sys.argv[1] at import
    try:
        return resum.load_module(path, "prod_analyze_full_x")
    finally:
        sys.argv = argv


RECOMPUTE_CELLS = ("X1__M1__D1", "X3__M1__D2", "X4a__M3__CF", "X3__M1__D4", "X1v__M1__VB")


def verify(job_dir, src_py=None):
    S = common.load_json(os.path.join(job_dir, "out", "summary_full.json"))      # independent re-read
    rows = extract(job_dir, src_py)
    n = eq = 0
    bad = []

    def cmp(lab, a, b):
        nonlocal n, eq
        n += 1
        if a == b:
            eq += 1
        elif len(bad) < 10:
            bad.append((lab, a, b))

    def _f(x):
        return float(x) if x not in ("", None) else None

    for r in rows:
        k = r["source_key"]
        parts = k.split(".")
        cell, arm, ep = parts[1], parts[-2], parts[-1]
        tab = S[cell]
        for p in parts[2:-2]:
            tab = tab[p]
        v = tab[arm][ep]
        cmp(k + ":estimate", _f(r["estimate"]), float(v["v"]))
        cmp(k + ":ci90", [_f(r["ci90_lo"]), _f(r["ci90_hi"])], [float(x) for x in v["ci90"]])
        cmp(k + ":ci95", [_f(r["ci95_lo"]), _f(r["ci95_hi"])], [float(x) for x in v["ci95"]])
        cmp(k + ":n_items", int(r["n_items"]), int(v["n_items"]))
        cmp(k + ":n_clusters", int(r["n_clusters"]), int(v["n_clusters"]))
    # recompute: one cell per dataset with analyze_full's own endpoints() (ratio_boot / mean_boot), imported
    A = _load_analyze(_src(job_dir, src_py))
    recomputed = []
    for cell in RECOMPUTE_CELLS:
        if cell not in S:
            continue
        allr = A.read_jsonl(_records(job_dir, cell))
        card = cell.split("__")[0]
        got = A.endpoints(card, allr[1:])
        for arm, eps in S[cell]["endpoints"].items():
            for ep, v in eps.items():
                if isinstance(v, dict) and "ci90" in v:
                    cmp("recompute:%s.%s.%s" % (cell, arm, ep), (got.get(arm) or {}).get(ep), v)
        # direct ratio_boot call on one arm (the recon recipe)
        arm = "pc_final" if card != "X1" else "src_Llast_1"
        ok = [x for x in allr[1:] if x.get("arm") == arm and x["D"] is not None and x["D"] >= 2
              and "no_vector" not in x and x.get("sibling") is None]
        rb = A.ratio_boot([(A.acl(x), x["final"]["dm"], x["D"]) for x in ok])
        if "e" in S[cell]["endpoints"][arm]:
            cmp("ratio_boot:%s.%s.e" % (cell, arm), rb, S[cell]["endpoints"][arm]["e"])
        # item-id regex used by item_sets against a full json parse
        cmp("item_ids:%s" % cell, _done_items(_records(job_dir, cell)), {x["item"] for x in allr[1:]})
        recomputed.append(cell)
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "recomputed_cells": recomputed}
