"""Extractor for k17_freeze.py output (results.json from --summarize, items_<D>.jsonl, selection_<D>.json).
Also reads k17_freeze_raven.py output (same layout; cells facts/addition, control key pc_final_e, model from
checks.json "model"). No raven artifact exists yet, so the raven path is written from the script and is untested.

Rows written (card branch_freeze, grid C), one per cell and arm:
  results.cells.<cell>.<stat> = {mean, ci90, ci95, n_items, n_defined, n_clusters[, dyadic{ci90, ci95, n_subjects}]}
  stat -> arm (the row name in the items file, not the misleading stat name):
    e0 -> E0, e_attn -> ans_frz_mlp, e_mlp -> ans_frz_attn, src_e -> src, src_frz_attn_e -> src_frz_attn,
    pc_e (raven: pc_final_e) -> pc_final, sz_e -> sz.
  facts e0/e_attn/e_mlp: the primary interval is the dyadic one (ci90_method stored_dyadic, n_clusters =
  n_subjects); the A-side cluster interval is written beside it as endpoint a_side_cluster.
  g5 per read-out loop e: cells.<cell>.g5.<row>[i].e, endpoint g5_loop<j>. The script stores a CI for these
  (same cell() function), so they are written with ci90_method stored.
n_items = n_defined (items with D >= 2 that enter the mean). Positive control = the cell's pc_e mean. No random
control in this script. Strength 1.0 for every arm (full donor write).
"""
import os

from . import common, resum

SCRIPT = "k17_freeze.py"
SCRIPT_RAVEN = "k17_freeze_raven.py"
CELLS = {"facts": ("facts", "own"), "addition": ("addition", "own"), "code_reads": ("code", "stratum:T1"),
         "code_sums": ("code", "stratum:T2"), "code_pooled": ("code", "own"), "twostep": ("twostep", "own")}
CELL_DS = {"facts": "D1", "addition": "D2", "code_reads": "D4", "code_sums": "D4", "code_pooled": "D4", "twostep": "D5"}
CELL_STRATUM = {"code_reads": "T1", "code_sums": "T2"}
ARM = {"e0": "E0", "e_attn": "ans_frz_mlp", "e_mlp": "ans_frz_attn", "src_e": "src", "src_frz_attn_e": "src_frz_attn",
       "pc_e": "pc_final", "pc_final_e": "pc_final", "sz_e": "sz"}
DYADIC_STATS = ("e0", "e_attn", "e_mlp")
ANS_ROWS = ("E0", "ans_frz_mlp", "ans_frz_attn")
OURO_MODEL = {"ByteDance/Ouro-1.4B": "ouro", "ByteDance/Ouro-2.6B": "ouro26", "ByteDance/Ouro-1.4B-Thinking": "ouro_think",
              "ByteDance/Ouro-2.6B-Thinking": "ouro26_think"}


def _src(src_py, raven):
    if src_py:
        return src_py
    return os.path.join(os.environ.get("LOOP_FLOW_SRC", ""), SCRIPT_RAVEN if raven else SCRIPT)


def _load(job_dir):
    res = common.load_json(os.path.join(job_dir, "results.json"))
    ck_path = os.path.join(job_dir, "checks.json")
    ck = common.load_json(ck_path) if os.path.exists(ck_path) else res.get("checks", {})
    raven = "model" in res or "verdict_facts" in ck
    if raven:
        model = ck.get("model") or res.get("model")
        K = 32
    else:
        model = OURO_MODEL.get(res["config"]["model"], res["config"]["model"])
        K = int(res["config"].get("k", 4))
    return res, model, raven, K


def _site(arm, K):
    """(site, loop) from the script's patch sites (docstring of k17_freeze / k17_freeze_raven)."""
    if arm in ANS_ROWS:
        return "r|Llast", 1
    if arm in ("src", "src_frz_attn"):
        return "src|L0", 1
    if arm == "pc_final":
        return "r|Llast", K
    if arm == "sz":
        return "src|Llast", K
    return "", ""


def extract(job_dir, src_py=None):
    res, model, raven, K = _load(job_dir)
    script = SCRIPT_RAVEN if raven else SCRIPT
    rows = []
    for cname, c in res["cells"].items():
        ds, iset = CELLS[cname]
        pc_stat = c.get("pc_final_e") if raven else c.get("pc_e")
        pc = pc_stat["mean"] if pc_stat else None
        base = dict(grid="C", card="branch_freeze", script=script, model=model, dataset=ds, strength=1.0, metric="e",
                    item_set=iset, positive_control_value=pc)
        for stat, arm in ARM.items():
            s = c.get(stat)
            if not isinstance(s, dict):
                continue
            site, loop = _site(arm, K)
            key = "results.cells.%s.%s" % (cname, stat)
            if stat in DYADIC_STATS and "dyadic" in s:
                dy = s["dyadic"]
                rows.append(common.row(arm=arm, site=site, loop=loop, endpoint="e", n_items=s["n_defined"],
                                       n_clusters=dy["n_subjects"], estimate=s["mean"], ci90_lo=dy["ci90"][0],
                                       ci90_hi=dy["ci90"][1], ci90_method="stored_dyadic", ci95_lo=dy["ci95"][0],
                                       ci95_hi=dy["ci95"][1], source_key=key + ".dyadic", **base))
                ep = "a_side_cluster"
            else:
                ep = "e"
            rows.append(common.row(arm=arm, site=site, loop=loop, endpoint=ep, n_items=s["n_defined"],
                                   n_clusters=s["n_clusters"], estimate=s["mean"], ci90_lo=s["ci90"][0],
                                   ci90_hi=s["ci90"][1], ci90_method="stored", ci95_lo=s["ci95"][0],
                                   ci95_hi=s["ci95"][1], source_key=key, **base))
        for arm, per_loop in (c.get("g5") or {}).items():
            site, loop = _site(arm, K)
            for i, g in enumerate(per_loop):
                s = g.get("e")
                if not isinstance(s, dict):
                    continue
                rows.append(common.row(arm=arm, site=site, loop=loop, endpoint="g5_loop%d" % g["loop"],
                                       n_items=s["n_defined"], n_clusters=s["n_clusters"], estimate=s["mean"],
                                       ci90_lo=s["ci90"][0], ci90_hi=s["ci90"][1], ci90_method="stored",
                                       ci95_lo=s["ci95"][0], ci95_hi=s["ci95"][1],
                                       source_key="results.cells.%s.g5.%s.%d.e" % (cname, arm, i), **base))
    return rows


def _items(job_dir, ds):
    p = os.path.join(job_dir, "items_%s.jsonl" % ds)
    return common.load_jsonl(p) if os.path.exists(p) else None


def item_sets(job_dir):
    """{(model, dataset): (planned keys, done keys)}. planned = selection_<D>.json items (the kept list the run was
    given); done = keys with a record in items_<D>.jsonl. Sets collapse the one duplicated D2 key."""
    res, model, raven, K = _load(job_dir)
    out = {}
    name = {"D1": "facts", "D2": "addition", "D4": "code", "D5": "twostep"}
    for ds, dname in name.items():
        sp = os.path.join(job_dir, "selection_%s.json" % ds)
        if not os.path.exists(sp):
            continue
        planned = {it["key"] for it in common.load_json(sp)["items"]}
        recs = _items(job_dir, ds) or []
        out[(model, dname)] = (planned, {r["key"] for r in recs})
    return out


def _walk(d, path):
    cur = d
    for p in path.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur


def _stored(res, key):
    """Stored (estimate, ci90, ci95, n_items, n_clusters) for a source_key, read from results.json."""
    path = key.split(".", 1)[1]
    if path.endswith(".dyadic"):
        s = _walk(res, path[:-len(".dyadic")])
        dy = s["dyadic"]
        return s["mean"], dy["ci90"], dy["ci95"], s["n_defined"], dy["n_subjects"]
    s = _walk(res, path)
    return s["mean"], s["ci90"], s["ci95"], s["n_defined"], s["n_clusters"]


def _f(x):
    return float(x) if x not in ("", None) else None


def verify(job_dir, src_py=None):
    res, model, raven, K = _load(job_dir)
    res_disk = common.load_json(os.path.join(job_dir, "results.json"))      # independent re-read
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
        est, c90, c95, ni, nc = _stored(res_disk, r["source_key"])
        k = r["source_key"]
        cmp(k + ":estimate", _f(r["estimate"]), float(est))
        cmp(k + ":ci90", [_f(r["ci90_lo"]), _f(r["ci90_hi"])], [float(x) for x in c90])
        cmp(k + ":ci95", [_f(r["ci95_lo"]), _f(r["ci95_hi"])], [float(x) for x in c95])
        cmp(k + ":n_items", int(r["n_items"]), int(ni))
        cmp(k + ":n_clusters", int(r["n_clusters"]), int(nc))
    # recompute with the script's own cell() (k17_ko240 boot_dist / q / tost_twoway): e_attn in every cell
    sp = _src(src_py, raven)
    mod = resum.load_module(sp, "k17frz_x")
    base = mod.KF if raven else mod          # raven imports k17_freeze as KF and uses KF.cell with its own e_of
    cell_fn, e_of = base.cell, mod.e_of
    KO = base.stats_mod()
    recomputed = []
    for cname, c in res_disk["cells"].items():
        ds = "D1" if cname == "facts" else ("D2" if cname == "addition" else CELL_DS[cname])
        recs = _items(job_dir, ds)
        if recs is None:
            continue
        st = CELL_STRATUM.get(cname)
        recs = [x for x in recs if st is None or x["stratum"] == st]
        got = cell_fn(KO, recs, e_of("ans_frz_mlp"), dyadic=(ds == "D1"))
        cmp("recompute:%s.e_attn" % cname, got, c["e_attn"])
        recomputed.append("cells.%s.e_attn" % cname)
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "recomputed": recomputed,
            "raven": raven}
