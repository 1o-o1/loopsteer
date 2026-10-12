"""Extractor for k15_full.py output (FULL-A edit-site map: k15f_<model>.json records + --summarize checks.json).

The script's summarize() computes every cell with boot_ci(xs, stat) (1000 item resamples, torch seed 0, 95 percent
read by torch.quantile) but stores only a few cells in checks.json; the rest go to summary.md at 2 decimals. So the
summarize is re-run on a copy of the records with boot_ci wrapped by a recorder: every (stat, xs) -> [v, lo, hi]
the summarize computed is kept at full precision. Cells stored in checks.json are read from it as written; the
rest come from the rerun. Run once at 95 percent (estimate, ci95) and once inside
resum.ninety() (ci90, "recomputed"). A cell the summarize never computes is computed with the script's own boot_ci
directly (source_key prefix "direct:"; none expected).

Rows (grid B, card edit_site_map, metric e unless named):
  S_O2, S_O4  cells "S|O2|a|site", "S|O4|a|site": source edit at the subject positions, site "src|<layer>", loop j.
  A           cells "A|add|a|site": last-position edit; endpoint "final" (fin) and "own_loop" (own), site "last|<layer>".
  C           cells "C|O1|a|Llast@1": metric e, mid_fraction (stat mid), bimodality_coef (stat bc).
  O5          cell "O5|O1|1|last_pos": positive control row itself, site "last|final".
Random control: S_O2 rows carry the same-site, same-a S_O4 estimate. Positive control: the O5 estimate of the same
model and dataset. Arm B (sentiment) is not extracted. n_items = n_clusters = non-missing items (item bootstrap).
"""
import glob
import json
import os
import re

from . import common, resum

SCRIPT = "k15_full.py"
DS = {"D1": "facts", "D2": "addition", "CF": "counterfact", "D4": "code", "D5": "twostep"}
SITE_RE = re.compile(r"^(L0|Lmid|Llast)@(\d+)$")
ARM = {"S": None, "A": "A", "C": "C", "O5": "O5"}


def _subdirs(d):
    return [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]


def run_summarize(src_py, job_dir, ninety):
    """Re-run summarize on a copy. Returns (checks, log {(stat, xs tuple): [v, lo, hi]}, quantile hits, tmp dir)."""
    mod = resum.load_module(src_py, "k15f_x")
    tmp = resum.copy_records(job_dir, ["k15f_*.json", "proc_*.json", "know_*.json"], skip_dirs=_subdirs(job_dir))
    log = {}
    orig = mod.boot_ci

    def rec(xs, stat="mean", n=mod.NBOOT, seed=0):
        out = orig(xs, stat, n, seed)
        if n == mod.NBOOT and seed == 0:
            log[(stat, tuple(xs))] = out
        return out
    mod.boot_ci = rec
    hits = {}
    with resum.quiet():
        if ninety:
            with resum.ninety(hits):
                mod.summarize(tmp)
        else:
            mod.summarize(tmp)
    mod.boot_ci = orig
    return common.load_json(os.path.join(tmp, "checks.json")), log, hits, mod


def _records(job_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(job_dir, "k15f_*.json"))):
        if f.endswith("_scores.json"):
            continue
        d = common.load_json(f)
        res[d["model"]] = d
    return res


def _site(pos, s):
    m = SITE_RE.match(s)
    return ("%s|%s" % (pos, m.group(1)), int(m.group(2))) if m else ("%s|%s" % (pos, s), "")


def _stored_path(m, ds, arm, op, a, site, stat, field):
    """checks.json path of a cell the summarize stores, else None."""
    if arm == "S" and a == "1" and stat == "mean" and site.endswith("@1"):
        return ["S_loop1_effect", m, ds, "%s|%s" % (op, site)]
    if arm == "A" and a == "2" and stat == "mean":
        return ["A_rederived_vs_neverread", m, ds, site, "own" if field == "own" else "final"]
    if arm == "C" and a == "0.5" and stat in ("mid", "bc"):
        return ["C_mid_fraction" if stat == "mid" else "C_bimodality", m, ds]
    return None


def _specs(arm):
    if arm == "A":
        return [("e", "mean", "fin", "final"), ("e", "mean", "own", "own_loop")]
    if arm == "C":
        return [("e", "mean", "e", "e"), ("mid_fraction", "mid", "e", "e"), ("bimodality_coef", "bc", "e", "e")]
    return [("e", "mean", "e", "e")]


def _value(log, mod, stat, xs, ninety):
    key = (stat, tuple(xs))
    if key in log:
        return log[key], "summarize"
    if ninety:
        with resum.ninety():
            return mod.boot_ci(xs, stat), "direct"
    return mod.boot_ci(xs, stat), "direct"


def extract(job_dir, src_py, runs=None):
    if runs is None:
        runs = (run_summarize(src_py, job_dir, False), run_summarize(src_py, job_dir, True))
    (_, log95, _, mod), (_, log90, _, _) = runs
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    rows = []
    for m, d in _records(job_dir).items():
        for ds, D in d["datasets"].items():
            if not D.get("pairs"):
                continue
            cells = D["cells"]
            o5c = cells.get("O5|O1|1|last_pos")
            o5 = _value(log95, mod, "mean", o5c["e"], False)[0] if o5c else None
            for k, c in cells.items():
                arm, op, a, site = k.split("|")
                if arm not in ARM:
                    continue
                for metric, stat, field, endpoint in _specs(arm):
                    xs = c[field]
                    v95, src95 = _value(log95, mod, stat, xs, False)
                    v90, src90 = _value(log90, mod, stat, xs, True)
                    if v95 is None:
                        continue
                    n = sum(1 for x in xs if x is not None)
                    if arm == "S":
                        sname, loop = _site("src", site)
                        armname = "S_" + op
                    elif arm == "C":
                        sname, loop = _site("src", site)
                        armname = "C"
                    elif arm == "A":
                        sname, loop = _site("last", site)
                        armname = "A"
                    else:
                        sname, loop, armname = "last|final", "", "O5"
                    rnd = None
                    if arm == "S" and op == "O2":
                        rc = cells.get("S|O4|%s|%s" % (a, site))
                        rnd = _value(log95, mod, "mean", rc["e"], False)[0] if rc else None
                    path = _stored_path(m, ds, arm, op, a, site, stat, field)
                    if path:
                        v95 = _get(ck, path)       # stored value read as written (the rerun may differ in the 4th dp)
                    skey = ("checks.json:" + json.dumps(path)) if path else (
                        "%s:%s" % ("summary.md" if (src95, src90) == ("summarize", "summarize") else "direct",
                                   json.dumps([m, ds, k, stat, field])))
                    rows.append(common.row(
                        grid="B", card="edit_site_map", script=SCRIPT, model=m, dataset=DS.get(ds, ds), arm=armname,
                        site=sname, loop=loop, strength=a, metric=metric, endpoint=endpoint, item_set="own",
                        n_items=n, n_clusters=n, estimate=v95[0], ci90_lo=v90[1] if v90 else None,
                        ci90_hi=v90[2] if v90 else None, ci90_method="recomputed", ci95_lo=v95[1], ci95_hi=v95[2],
                        random_control_estimate=rnd[0] if rnd else None,
                        positive_control_value=o5[0] if o5 else None, source_key=skey))
    return rows


def _iid(it):
    return it.get("id") or "%s|%s|%s|%s" % (it.get("half"), it.get("rel"), it.get("a"), it.get("b"))


def item_sets(job_dir):
    """Planned = the pairs the quota scan selected (KD.select keeps scanning until every (half, rel) quota is full,
    the candidates run out, or the pair budget ends: truncated=True, as for CF of ifm_p1_ouro_inj and ifm_d112).
    Done = pairs with a record. Both are the recorded items, so planned == done by construction."""
    out = {}
    for m, d in _records(job_dir).items():
        for ds, D in d["datasets"].items():
            ids = {_iid(it) for it in D.get("items") or []}
            out[(m, DS.get(ds, ds))] = (set(ids), ids)
    return out


def truncated(job_dir):
    return {(m, DS.get(ds, ds)): bool(D.get("truncated")) for m, d in _records(job_dir).items()
            for ds, D in d["datasets"].items()}


def _get(d, path):
    cur = d
    for p in path:
        cur = cur[p]
    return cur


def _sections(md):
    out, cur = {}, None
    for line in md.split("\n"):
        h = re.match(r"^### (\S+) / (\S+)$", line)
        if h:
            cur = (h.group(1), h.group(2))
            out[cur] = []
        elif line.startswith("## "):
            cur = None
        elif cur:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _fmt(v):
    return "%.2f [%s, %s]" % (v[0], "-" if v[1] is None else "%.2f" % v[1], "-" if v[2] is None else "%.2f" % v[2])


def verify(job_dir, src_py):
    """(a) extracted == stored checks.json where the cell is stored; (b) the 95 rerun of summarize == stored
    checks.json at those paths; (c) every row's 2-decimal rendering is in the stored summary.md section of its model
    and dataset; (d) the 90 run gives the same point estimate. n is the count of non-missing items of the record."""
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    r95, r90 = run_summarize(src_py, job_dir, False), run_summarize(src_py, job_dir, True)
    ck95, hits = r95[0], r90[2]
    rows = extract(job_dir, src_py, (r95, r90))
    with open(os.path.join(job_dir, "summary.md"), "r", encoding="utf-8") as f:
        sec = _sections(f.read())
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
    direct = sum(1 for r in rows if r["source_key"].startswith("direct:"))
    for r in rows:
        f = lambda x: None if x == "" else float(x)  # noqa: E731
        ext = [f(r["estimate"]), f(r["ci95_lo"]), f(r["ci95_hi"])]
        kind, p = r["source_key"].split(":", 1)
        p = json.loads(p)
        if kind == "checks.json":
            st = _get(ck, p)
            cmp("extracted", ext, st, "/".join(p))
            cmp("rerun95", _get(ck95, p), st, "/".join(p))
        txt = sec.get((r["model"], inv[r["dataset"]]), "")
        rend = ("%.2f" % ext[0]) if r["endpoint"] == "own_loop" else _fmt(ext)
        cmp("summary_md_2dp", rend in txt, True, "%s/%s/%s/%s/%s/%s" % (r["model"], r["dataset"], r["arm"], r["site"],
                                                                         r["strength"], r["metric"]))
    # (d) 90 run point estimate per row: the 90 log holds the same keys; compare the point
    mod = r95[3]
    recs = _records(job_dir)
    for m, d in recs.items():
        for ds, D in d["datasets"].items():
            for k, c in (D.get("cells") or {}).items():
                arm = k.split("|")[0]
                if arm not in ARM:
                    continue
                for _, stat, field, _ in _specs(arm):
                    key = (stat, tuple(c[field]))
                    if key in r95[1] and r95[1][key] is not None:
                        cmp("est90", (r90[1].get(key) or [None])[0], r95[1][key][0], "%s/%s/%s/%s" % (m, ds, k, stat))
    whole = {k: ck95.get(k) == ck.get(k) for k in ck}
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "quantile_reads_patched": hits,
            "direct_fallback_rows": direct, "checks_json_keys_equal_on_rerun": sum(whole.values()),
            "checks_json_keys": len(whole), "checks_json_keys_differing": [k for k, v in whole.items() if not v],
            "truncated": {"%s|%s" % k: v for k, v in truncated(job_dir).items() if v}, "arm_B": "not extracted"}
