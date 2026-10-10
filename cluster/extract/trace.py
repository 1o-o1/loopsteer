"""Extractor for k17_trace.py output (results.json from --summarize, items_D1.jsonl, items_D2.jsonl).

Rows written (card loop_trace, grid B), arm donor:
  results.datasets.<facts|addition>.regions[<group>][<region>][<site>].e = {mean, ci90, ci95, n_items, n_clusters,
  strata, pc}; site column "<region>|<site label>" (region pre/source/source_last/relay/last, label L0@j, Lmid@j,
  Llast@j), loop = j. group pooled -> item_set own, each stratum -> item_set stratum:<name>.
  random_control_estimate = the sibling .e_rand mean (same items, positions and site; same-norm random direction).
  positive_control_value = the cell's pc (mean e of the exact loop-4 patch at the answer position).
Both intervals are stored (NB = 1000 resamples, seed 0). n_items already counts only items with e defined.
Not extracted: the per-position token grid (datasets.<name>.grid) and the projection tables (no CI).
Strength 1.0 (donor write).
"""
import os

from . import common, resum

SCRIPT = "k17_trace.py"
FILE = {"facts": "D1", "addition": "D2"}
OURO_MODEL = {"ByteDance/Ouro-1.4B": "ouro", "ByteDance/Ouro-2.6B": "ouro26"}


def _src(src_py):
    return src_py or os.path.join(os.environ.get("LOOP_FLOW_SRC", ""), SCRIPT)


def _model(res):
    m = res["config"].get("model") or res["config"].get("repo") or "ByteDance/Ouro-1.4B"
    return OURO_MODEL.get(m, m)


def extract(job_dir, src_py=None):
    res = common.load_json(os.path.join(job_dir, "results.json"))
    model = _model(res)
    rows = []
    for name, d in res["datasets"].items():
        for grp, regs in d["regions"].items():
            iset = "own" if grp == "pooled" else "stratum:%s" % grp
            for region, sites in regs.items():
                for lab, c in sites.items():
                    e = (c or {}).get("e")
                    if not isinstance(e, dict):
                        continue
                    er = c.get("e_rand")
                    rows.append(common.row(
                        grid="B", card="loop_trace", script=SCRIPT, model=model, dataset=name, arm="donor",
                        site="%s|%s" % (region, lab), loop=int(lab.split("@")[1]), strength=1.0, metric="e",
                        endpoint="e", item_set=iset, n_items=e["n_items"], n_clusters=e["n_clusters"],
                        estimate=e["mean"], ci90_lo=e["ci90"][0], ci90_hi=e["ci90"][1], ci90_method="stored",
                        ci95_lo=e["ci95"][0], ci95_hi=e["ci95"][1],
                        random_control_estimate=er["mean"] if isinstance(er, dict) else None,
                        positive_control_value=e.get("pc"),
                        source_key="results.datasets.%s.regions.%s.%s.%s.e" % (name, grp, region, lab)))
    return rows


def selection(job_dir, sel_dir=None):
    """{D: planned keys} from the selection files the run was given (--sel_dir = the branch-freeze full dir),
    accepted only when the sha256 equals config.selection_sha256[D]. Looked up in sel_dir, job_dir, then
    <job_dir>/../../k17_freeze/full."""
    import hashlib
    cfg = common.load_json(os.path.join(job_dir, "config.json"))
    want = cfg.get("selection_sha256") or {}
    dirs = [x for x in (sel_dir, job_dir, os.path.join(job_dir, "..", "..", "k17_freeze", "full")) if x]
    out = {}
    for ds, sha in want.items():
        for x in dirs:
            p = os.path.join(x, "selection_%s.json" % ds)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() == sha:
                        out[ds] = {it["key"] for it in common.load_json(p)["items"]}
                        break
    return out


def item_sets(job_dir, sel_dir=None):
    """{(model, dataset): (planned keys, done keys)}; planned = sha-checked selection (empty set if not found),
    done = keys with a record in items_<D>.jsonl."""
    res = common.load_json(os.path.join(job_dir, "results.json"))
    model = _model(res)
    sel = selection(job_dir, sel_dir)
    out = {}
    for name, ds in FILE.items():
        p = os.path.join(job_dir, "items_%s.jsonl" % ds)
        if os.path.exists(p):
            out[(model, name)] = (sel.get(ds, set()), {r["key"] for r in common.load_jsonl(p)})
    return out


def _f(x):
    return float(x) if x not in ("", None) else None


def verify(job_dir, src_py=None):
    res = common.load_json(os.path.join(job_dir, "results.json"))
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

    for r in rows:
        k = r["source_key"]
        _, _, name, _, grp, region, lab, _ = k.split(".")
        c = res["datasets"][name]["regions"][grp][region][lab]
        s = c["e"]
        cmp(k + ":estimate", _f(r["estimate"]), float(s["mean"]))
        cmp(k + ":ci90", [_f(r["ci90_lo"]), _f(r["ci90_hi"])], [float(x) for x in s["ci90"]])
        cmp(k + ":ci95", [_f(r["ci95_lo"]), _f(r["ci95_hi"])], [float(x) for x in s["ci95"]])
        cmp(k + ":n_items", int(r["n_items"]), int(s["n_items"]))
        cmp(k + ":n_clusters", int(r["n_clusters"]), int(s["n_clusters"]))
        er = c.get("e_rand")
        cmp(k + ":random", _f(r["random_control_estimate"]), float(er["mean"]) if er else None)
        cmp(k + ":pc", _f(r["positive_control_value"]), float(s["pc"]))
    # recompute with the script's own cell() / region_e() / boot() (NB 1000, seed 0), same cluster rule as summarize
    KT = resum.load_module(_src(src_py), "k17trace_x")
    recomputed = []
    for name, ds in FILE.items():
        p = os.path.join(job_dir, "items_%s.jsonl" % ds)
        if not os.path.exists(p) or name not in res["datasets"]:
            continue
        recs = common.load_jsonl(p)
        for r in recs:
            r["cluster_subject"] = r.get("sa", r.get("cluster"))
        cl_key = "cluster_subject" if ds == "D1" else "key"
        for region, lab in (("source_last", "Lmid@1"), ("last", "Llast@2")):
            for col, arm in ((0, "e"), (1, "e_rand")):
                got = KT.cell(recs, (lambda r, lab=lab, rg=region, col=col: KT.region_e(r, lab, rg, col)), cl_key)
                cmp("recompute:%s.pooled.%s.%s.%s" % (name, region, lab, arm), got,
                    res["datasets"][name]["regions"]["pooled"][region][lab][arm])
            recomputed.append("datasets.%s.regions.pooled.%s.%s" % (name, region, lab))
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "recomputed": recomputed}
