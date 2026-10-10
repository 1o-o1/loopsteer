"""Extractor for k15_pairs.py output (FULL-B matched recipe pairs: k15p_<model>.json records + --summarize checks.json).

The script's summarize() computes every cell as r3(stat_ci(xs, stat)) (1000 item resamples, numpy default_rng(0),
95 percent read by two scalar np.percentile calls at 2.5 and 97.5) and stores only some cells in checks.json; the
rest go to summary.md at 2 decimals. The summarize is re-run on a copy of the records with stat_ci wrapped by a
recorder, once at 95 percent and once with the reads at 5 and 95 (resum.ninety plus the scalar percentile patch
below, which resum.ninety does not cover). Cells stored in checks.json are read from it as written; the rest come
from the 95 rerun. Values carry the script's r3 rounding (3 decimals).

Rows (grid B, card edit_site_map):
  S_O2, S_O4  "S_O2|a|site", "S_O4|a|site": subject positions, site "src|<layer>", loop j, metric e.
  C           "C|a|Llast@1", "C|a|L0@K": metric e, mid_fraction (stat mid), bimodality_coef (stat bc).
  O5          "O5|1|Llast@K_allpos": whole loop-K state set to clean B, site "all|Llast", loop K.
  O5_last     "O5_last|1|final_lastpos": last position at the final point, site "last|final".
Random control: S_O2 rows carry the same-site, same-a S_O4 estimate. Positive control: the O5 estimate of the same
model and dataset. n_items = n_clusters = non-missing items (item bootstrap).
"""
import contextlib
import glob
import json
import os
import re

from . import common, resum

SCRIPT = "k15_pairs.py"
DS = {"D1": "facts", "D2": "addition"}
SITE_RE = re.compile(r"^(L0|Lmid|Llast)@(\d+)$")


@contextlib.contextmanager
def ninety_all(hits=None):
    """resum.ninety plus scalar np.percentile(x, 2.5) / (x, 97.5) reads turned into 5 / 95."""
    import numpy as np
    hits = hits if hits is not None else {}
    hits.setdefault("numpy_scalar", 0)
    with resum.ninety(hits):
        inner = np.percentile

        def pct(a, q, *args, **k):
            if isinstance(q, (int, float)) and not isinstance(q, bool):
                if abs(q - 2.5) < 1e-9:
                    hits["numpy_scalar"] += 1
                    q = 5.0
                elif abs(q - 97.5) < 1e-9:
                    hits["numpy_scalar"] += 1
                    q = 95.0
            return inner(a, q, *args, **k)
        np.percentile = pct
        try:
            yield hits
        finally:
            np.percentile = inner


def r3(x):
    if x is None:
        return None
    if isinstance(x, list):
        return [r3(y) for y in x]
    return round(x, 3)


def _subdirs(d):
    return [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]


def run_summarize(src_py, job_dir, ninety):
    mod = resum.load_module(src_py, "k15p_x")
    tmp = resum.copy_records(job_dir, ["k15p_*.json", "proc_*.json"], skip_dirs=_subdirs(job_dir))
    log = {}
    orig = mod.stat_ci

    def rec(xs, stat):
        out = orig(xs, stat)
        log[(stat, tuple(xs))] = out
        return out
    mod.stat_ci = rec
    hits = {}
    with resum.quiet():
        if ninety:
            with ninety_all(hits):
                mod.summarize(tmp)
        else:
            mod.summarize(tmp)
    mod.stat_ci = orig
    return common.load_json(os.path.join(tmp, "checks.json")), log, hits, mod


def _records(job_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(job_dir, "k15p_*.json"))):
        d = common.load_json(f)
        res[d["model"]] = d
    return res


def _site(pos, s):
    m = SITE_RE.match(s)
    return ("%s|%s" % (pos, m.group(1)), int(m.group(2))) if m else ("%s|%s" % (pos, s), "")


def _specs(arm):
    if arm == "C":
        return [("e", "mean"), ("mid_fraction", "mid"), ("bimodality_coef", "bc")]
    return [("e", "mean")]


def _stored_path(ck, m, ds, arm, a, site, stat):
    if arm in ("S_O2", "S_O4") and a == "1" and stat == "mean":
        return ["S_effect_by_loop", m, ds, site, "%s_a1_effect" % arm[2:]]
    if arm == "C" and a == "0.5":
        return [{"mean": "C_mean_effect_a05", "mid": "C_mid_fraction", "bc": "C_bimodality"}[stat], m, "%s|%s" % (ds, site)]
    if arm in ("O5", "O5_last") and ck.get("main_dataset", {}).get(m) == ds:
        return ["o5_effect_per_model" if arm == "O5" else "o5_last_effect_per_model", m]
    return None


def _get(d, path):
    cur = d
    for p in path:
        cur = cur[p]
    return cur


def _value(log, mod, stat, xs, ninety):
    key = (stat, tuple(xs))
    if key in log:
        return r3(log[key]), "summarize"
    if ninety:
        with ninety_all():
            return r3(mod.stat_ci(xs, stat)), "direct"
    return r3(mod.stat_ci(xs, stat)), "direct"


def extract(job_dir, src_py, runs=None):
    if runs is None:
        runs = (run_summarize(src_py, job_dir, False), run_summarize(src_py, job_dir, True))
    (_, log95, _, mod), (_, log90, _, _) = runs
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    rows = []
    for m, d in _records(job_dir).items():
        K = d["K"]
        for ds, D in d["datasets"].items():
            cells = D.get("cells") or {}
            o5c = cells.get("O5|1|Llast@K_allpos")
            o5 = _value(log95, mod, "mean", o5c["effect"], False)[0] if o5c else None
            for k, c in cells.items():
                arm, a, site = k.split("|")
                for metric, stat in _specs(arm):
                    xs = c["effect"]
                    v95, s95 = _value(log95, mod, stat, xs, False)
                    v90, s90 = _value(log90, mod, stat, xs, True)
                    if v95 is None:
                        continue
                    path = _stored_path(ck, m, ds, arm, a, site, stat)
                    if path:
                        v95 = _get(ck, path)
                    if arm == "O5":
                        sname, loop = "all|Llast", K
                    elif arm == "O5_last":
                        sname, loop = "last|final", ""
                    else:
                        sname, loop = _site("src", site)
                    rnd = None
                    if arm == "S_O2":
                        rc = cells.get("S_O4|%s|%s" % (a, site))
                        rnd = _value(log95, mod, "mean", rc["effect"], False)[0] if rc else None
                    n = sum(1 for x in xs if x is not None)
                    skey = ("checks.json:" + json.dumps(path)) if path else (
                        "%s:%s" % ("summary.md" if (s95, s90) == ("summarize", "summarize") else "direct",
                                   json.dumps([m, ds, k, stat])))
                    rows.append(common.row(
                        grid="B", card="edit_site_map", script=SCRIPT, model=m, dataset=DS.get(ds, ds), arm=arm,
                        site=sname, loop=loop, strength=a, metric=metric, endpoint="e", item_set="own", n_items=n,
                        n_clusters=n, estimate=v95[0], ci90_lo=v90[1] if v90 else None,
                        ci90_hi=v90[2] if v90 else None, ci90_method="recomputed", ci95_lo=v95[1], ci95_hi=v95[2],
                        random_control_estimate=rnd[0] if rnd else None, positive_control_value=o5[0] if o5 else None,
                        source_key=skey))
    return rows


def item_sets(job_dir):
    """Planned = the pairs the per-half quota scan selected (it stops at the target or when the candidates run out);
    done = pairs with a record. Both are the recorded items."""
    out = {}
    for m, d in _records(job_dir).items():
        for ds, D in d["datasets"].items():
            ids = {"%s:%s" % (it["half"], it["i"]) for it in D.get("items") or []}
            out[(m, DS.get(ds, ds))] = (set(ids), ids)
    return out


def _sections(md):
    out, model, cur = {}, None, None
    for line in md.split("\n"):
        if line.startswith("## "):
            model, cur = line[3:].strip(), None
            continue
        h = re.match(r"^### (\S+): ", line)
        if h and model:
            cur = (model, h.group(1))
            out[cur] = [line]
        elif cur:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _fci(c):
    f2 = lambda x: "-" if x is None else "%.2f" % x  # noqa: E731
    return "%.2f [%s, %s]" % (c[0], f2(c[1]), f2(c[2]))


def verify(job_dir, src_py):
    """(a) the 95 rerun of summarize == stored checks.json at every stored path a row reads; (b) every row's
    2-decimal rendering is in the stored summary.md (section of its model and dataset; O5 rows in the model header);
    (c) the 90 run gives the same point estimate for every row."""
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    r95, r90 = run_summarize(src_py, job_dir, False), run_summarize(src_py, job_dir, True)
    rows = extract(job_dir, src_py, (r95, r90))
    with open(os.path.join(job_dir, "summary.md"), "r", encoding="utf-8") as f:
        md = f.read()
    sec = _sections(md)
    inv = {v: k for k, v in DS.items()}
    n = eq = 0
    bad = []

    def cmp(lab, a, b, where):
        nonlocal n, eq
        n += 1
        if a == b:
            eq += 1
        elif len(bad) < 10:
            bad.append((lab, where, a, b))
    recs = _records(job_dir)
    for r in rows:
        kind, p = r["source_key"].split(":", 1)
        p = json.loads(p)
        f = lambda x: None if x == "" else float(x)  # noqa: E731
        ext = [f(r["estimate"]), f(r["ci95_lo"]), f(r["ci95_hi"])]
        where = "%s/%s/%s/%s/%s/%s" % (r["model"], r["dataset"], r["arm"], r["site"], r["strength"], r["metric"])
        if kind == "checks.json":
            cmp("rerun95", _get(r95[0], p), _get(ck, p), "/".join(p))
        txt = md if r["arm"].startswith("O5") else sec.get((r["model"], inv[r["dataset"]]), "")
        cmp("summary_md_2dp", _fci(ext) in txt, True, where)
    for m, d in recs.items():
        for ds, D in d["datasets"].items():
            for k, c in (D.get("cells") or {}).items():
                for _, stat in _specs(k.split("|")[0]):
                    key = (stat, tuple(c["effect"]))
                    if r95[1].get(key) is not None:
                        cmp("est90", (r90[1].get(key) or [None])[0], r95[1][key][0], "%s/%s/%s/%s" % (m, ds, k, stat))
    whole = {k: r95[0].get(k) == ck.get(k) for k in ck}
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "quantile_reads_patched": r90[2],
            "direct_fallback_rows": sum(1 for r in rows if r["source_key"].startswith("direct:")),
            "checks_json_keys_equal_on_rerun": sum(whole.values()), "checks_json_keys": len(whole),
            "checks_json_keys_differing": [k for k, v in whole.items() if not v]}
