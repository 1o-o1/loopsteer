"""Extractor for k17_control.py output (one dir per model: summary.json, frozen.json, items.json, rows_test.jsonl,
checks_pre.json). Model dirs read: ouro, ouro26, ifm_p1_ouro, ifm_p1_ouro_inj. ouro_o2 is skipped: it holds only a
partial O2 search (search_o2.partial.json), no scored rows.

Rows written (card selective_control, grid D, dataset d3_country = D3 country rows, continent targets), one per
(model, editor, arm, metric):
  summary.cells.<ed>.placements.<arm>.<metric> = [estimate, ci95_lo, ci95_hi]
  arm "<ed>:<arm>", arm in H2 BEST LAST EVERY H2_abs RND (RND = random direction at the H2 site and H2 norms,
  3 seeds pooled per item). BEST_all is dropped (same_as_BEST in every cell, no own rows); RND0-2 single seeds are
  dropped. metric in flip spec ret kl sel para loc loc_same spec_inf to_donor dm_donor e.
The script stores 95 percent only. ci90 is recomputed (ci90_method "recomputed") from rows_test.jsonl with the
script's own Boot (seed 0, NBOOT), item_metrics and ci(bs, 0.90), on the same item order and NaN handling as its
summarize. estimate and ci95 are the stored values. n_items / n_clusters = items (clusters) with a non-NaN value for
that metric (para and loc_same have fewer; the stored cell n is the full test set).
site "<pos>|<layer list>", loop "<loop list>" from frozen.editors.<ed>.<arm>.plc.sites ([loop, layer] or
["inj", loop]); H2_abs and RND use the H2 site. strength = frozen a; H2_abs = mean of the per-item a in its rows;
RND empty. random_control_estimate = the RND estimate of the same editor and metric, on H2 rows only (RND is matched
to the H2 site only). positive_control_value = checks_pre.json pc_final_bit_exact as "true"/"false" (the script has
no numeric positive control).
"""
import os

from . import common, resum

SCRIPT = "k17_control.py"
MODELS = ("ouro", "ouro26", "ifm_p1_ouro", "ifm_p1_ouro_inj")
SKIPPED = {"ouro_o2": "partial O2 search only, no scored rows"}
ARMS = ("H2", "BEST", "LAST", "EVERY", "H2_abs", "RND")
METRIC_KEYS = ("flip", "spec", "ret", "kl", "sel", "para", "loc", "loc_same", "spec_inf", "to_donor", "dm_donor")
RND_SEEDS = (0, 1, 2)


def _src(src_py):
    return src_py or os.path.join(os.environ.get("LOOP_FLOW_SRC", ""), SCRIPT)


def _site(plc):
    sites = plc.get("sites") or []
    layers = ";".join(str(s[1]) if s[0] != "inj" else "inj" for s in sites)
    loops = ";".join(str(s[0]) if s[0] != "inj" else str(s[1]) for s in sites)
    return "%s|%s" % (plc.get("pos", ""), layers), loops


def _model_dirs(job_dir):
    return [(m, os.path.join(job_dir, m)) for m in MODELS if os.path.exists(os.path.join(job_dir, m, "summary.json"))]


def recompute(K, d, level):
    """Per editor: {arm: {metric: (pt, [lo, hi], n_items, n_clusters)}} with the script's Boot and ci at `level`,
    built as summarize builds it (k17_control.summarize)."""
    import numpy as np
    import warnings
    fz = common.load_json(os.path.join(d, "frozen.json"))
    items = common.load_json(os.path.join(d, "items.json"))
    rows = K._rows(d)
    iv = {it["id"]: it for it in items["cont_test"] + items["o2_test"]}
    nboot = 1000 if "debug" in d else K.NBOOT
    out = {}
    for ed in fz["editors"]:
        its = items["o2_test"] if ed == "o2" else items["cont_test"]
        ids = [it["id"] for it in its if "%s|H2|%s" % (ed, it["id"]) in rows]
        if not ids:
            continue
        cl = [iv[i]["cluster"] for i in ids]
        dn = [iv[i]["donor_cluster"] for i in ids] if ed == "o2" else None
        bt = K.Boot(cl, dn, nboot=nboot, seed=0)
        M, a_mean = {}, {}
        for nm in ("H2", "BEST", "LAST", "EVERY", "H2_abs", "RND0", "RND1", "RND2"):
            recs = [rows.get("%s|%s|%s" % (ed, nm, i)) for i in ids]
            if all(r is None for r in recs):
                continue
            ms = [K.item_metrics(r, ed) if r else None for r in recs]
            M[nm] = {k: np.array([m[k] if m else np.nan for m in ms], dtype=float) for k in K.METRICS}
            av = [r.get("a") for r in recs if r and r.get("a") is not None]
            a_mean[nm] = float(np.mean(av)) if av else None
        if all(("RND%d" % s) in M for s in RND_SEEDS):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                M["RND"] = {k: np.nanmean(np.stack([M["RND%d" % s][k] for s in RND_SEEDS]), 0) for k in M["RND0"]}
        res = {}
        for nm in ARMS:
            if nm not in M:
                continue
            arr = M[nm]
            pc = {}
            for k in METRIC_KEYS + ("e",):
                fn = (lambda m: m["dm"] / m["D"]) if k == "e" else (lambda m, k=k: m[k])
                pt, bs = bt.stat(arr, fn)
                with np.errstate(all="ignore"):
                    vals = np.asarray(fn(arr), dtype=float)
                keep = ~np.isnan(vals)
                pc[k] = (pt, K.ci(bs, level), int(keep.sum()), len(set(c for c, f in zip(cl, keep) if f)))
            res[nm] = pc
        out[ed] = {"placements": res, "a_mean": a_mean}
    return out


def extract(job_dir, src_py=None, _rec=None):
    K = resum.load_module(_src(src_py), "k17control_x")
    rows = []
    for model, d in _model_dirs(job_dir):
        S = common.load_json(os.path.join(d, "summary.json"))
        fz = common.load_json(os.path.join(d, "frozen.json"))
        pre = common.load_json(os.path.join(d, "checks_pre.json"))
        pcv = pre.get("pc_final_bit_exact")
        rec = (_rec or {}).get(model) or recompute(K, d, 0.90)
        for ed, cell in S["cells"].items():
            E = fz["editors"].get(ed, {})
            plc_h2 = (E.get("H2") or {}).get("plc") or {}
            for arm in ARMS:
                pl = (cell.get("placements") or {}).get(arm)
                if pl is None:
                    continue
                if arm in ("H2_abs", "RND"):
                    site, loop = _site(plc_h2)
                    strength = rec[ed]["a_mean"].get(arm) if arm == "H2_abs" else None
                else:
                    site, loop = _site(E[arm].get("plc") or {})
                    strength = E[arm].get("a")
                for metric in METRIC_KEYS + ("e",):
                    v = pl.get(metric)
                    if not isinstance(v, list):
                        continue
                    pt, c90, ni, nc = rec[ed]["placements"][arm][metric]
                    rnd = None
                    if arm == "H2":
                        rv = (cell["placements"].get("RND") or {}).get(metric)
                        rnd = rv[0] if isinstance(rv, list) else None
                    rows.append(common.row(
                        grid="D", card="selective_control", script=SCRIPT, model=model, dataset="d3_country",
                        arm="%s:%s" % (ed, arm), site=site, loop=loop, strength=strength, metric=metric,
                        endpoint=metric, item_set="own", n_items=ni, n_clusters=nc, estimate=v[0],
                        ci90_lo=c90[0], ci90_hi=c90[1], ci90_method="recomputed", ci95_lo=v[1], ci95_hi=v[2],
                        random_control_estimate=rnd, positive_control_value=pcv,
                        source_key="%s/summary.cells.%s.placements.%s.%s" % (model, ed, arm, metric)))
    return rows


def item_sets(job_dir):
    """{(model, d3_country): (planned ids = items.json cont_test, done ids = cont_test ids with a row for every
    editor in the summary)}."""
    out = {}
    for model, d in _model_dirs(job_dir):
        items = common.load_json(os.path.join(d, "items.json"))
        S = common.load_json(os.path.join(d, "summary.json"))
        planned = {it["id"] for it in items["cont_test"]}
        have = {}
        for r in common.load_jsonl(os.path.join(d, "rows_test.jsonl")):
            have.setdefault(r["editor"], set()).add(r["item"])
        done = set(planned)
        for ed in S["cells"]:
            done &= have.get(ed, set())
        out[(model, "d3_country")] = (planned, done)
    return out


def _same(a, b, tol=1e-12):
    import math
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def verify(job_dir, src_py=None):
    """(a) every extracted estimate / ci95 / n against the stored summary.json (re-read), exact; (b) the script's
    own bootstrap re-run at 95 percent against the stored point and 95 percent CI for every row (tolerance 1e-12:
    the stored floats came from the same code, BLAS order can move the last bits); (c) the ci90 written equals a
    second independent recompute."""
    K = resum.load_module(_src(src_py), "k17control_v")
    rec90 = {m: recompute(K, d, 0.90) for m, d in _model_dirs(job_dir)}
    rec95 = {m: recompute(K, d, 0.95) for m, d in _model_dirs(job_dir)}
    rows = extract(job_dir, src_py, rec90)
    n = eq = 0
    bad = []
    n_diff = []
    stored = {}

    def cmp(lab, ok, a, b):
        nonlocal n, eq
        n += 1
        if ok:
            eq += 1
        elif len(bad) < 10:
            bad.append((lab, a, b))

    def _f(x):
        return None if x in ("", None) else float(x)

    for r in rows:
        k = r["source_key"]
        model, path = k.split("/", 1)
        _, _, ed, _, arm, metric = path.split(".")
        if model not in stored:
            stored[model] = common.load_json(os.path.join(job_dir, model, "summary.json"))     # independent re-read
        cell = stored[model]["cells"][ed]
        v = cell["placements"][arm][metric]
        cmp(k + ":estimate", _same(_f(r["estimate"]), v[0], 0.0), r["estimate"], v[0])
        cmp(k + ":ci95", _same(_f(r["ci95_lo"]), v[1], 0.0) and _same(_f(r["ci95_hi"]), v[2], 0.0),
            [r["ci95_lo"], r["ci95_hi"]], v[1:])
        pt95, c95, ni, nc = rec95[model][ed]["placements"][arm][metric]
        cmp(k + ":rerun95", _same(pt95, v[0]) and _same(c95[0], v[1]) and _same(c95[1], v[2]), [pt95] + c95, v)
        c90 = rec90[model][ed]["placements"][arm][metric][1]
        cmp(k + ":ci90", _same(_f(r["ci90_lo"]), c90[0], 0.0) and _same(_f(r["ci90_hi"]), c90[1], 0.0),
            [r["ci90_lo"], r["ci90_hi"]], c90)
        cmp(k + ":n_clusters_le_cell", int(r["n_clusters"]) <= int(cell["n_clusters"]), r["n_clusters"],
            cell["n_clusters"])
        if int(r["n_items"]) != int(cell["n_items"]):
            n_diff.append("%s:%s|%s|%s n=%s of %s" % (model, ed, arm, metric, r["n_items"], cell["n_items"]))
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad,
            "n_items_below_cell_n": len(n_diff), "n_items_below_examples": n_diff[:6], "skipped_dirs": SKIPPED}
