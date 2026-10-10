"""Extractor for k16_path_ouro.py output (k16p_<model><ftag>.json records + --summarize checks_detail.json).

Stored estimates are in checks_detail.json, [v, lo, hi, n_items] at 95 percent (boot_cluster: cluster bootstrap,
1000 resamples, torch seed 0, torch.quantile). The 90 percent interval comes from the script's own summarize re-run
on a copy of the records inside resum.ninety(); in that run boot_cluster is wrapped to append the number of
clusters with a value (n_clusters, which the script does not store per estimate). Subset "all" only (the D1
capital/city and D4 T1/T2 sub-strata are not extracted).

Rows:
  grid B, card edit_site_map:    y1_2x2.<m>.<ds>.all."<edit>|none" for ans_O1, ans_para, src_O1; metrics e, dm.
  grid C, card attention_knockout:
    y1_2x2.<m>.<ds>.all."<edit>|<ko>" for ko src_all, allbut (edit off = knockout only); metrics e, dm.
    y1_controls.<m>.<ds>.all.<edit>.<ko>.{E1, rescue} for ko src_all, src_l2, src_lo, src_hi, rand, nonsrc, allbut.
    w2_window.<m>.<ds>.all.per_loop[j].summed_dm (arm W2, loop j+1).
  grid C, card restore:          y3_undo.<m>.<ds>.all.{"Y1|src_O1|none", "Y3|undo2", "Y3|undo3"}.e (loop 1, 2, 3).
Edits sit at Llast@1: ans_* at the last position (site "last|Llast"), src_O1 at the source span ("src|Llast"),
off has no edit ("last|none"). Knockouts act on the last query position in loops 2-4 (named in arm). Strength:
1 for O1 edits, 2 for ans_para (A_PARA). Random control: the pooled random-key knockout (rand) of the same edit and
metric for the src_* knockouts in y1_controls; no random cell exists for y1_2x2. Positive control:
y1_positive_control.all.src_O1_e0 (metric e) or src_O1_E0 (dm, E1, rescue) of the same model and dataset.
Not extracted: width_control, y4_lin_vs_actual, t3_per_loop (sentiment), sub-strata.
"""
import glob
import json
import os

from . import common, resum

SCRIPT = "k16_path_ouro.py"
DS = {"D1": "facts", "D2": "addition", "D4": "code", "D5": "twostep"}
STRENGTH = {"ans_O1": 1.0, "ans_para": 2.0, "src_O1": 1.0, "off": ""}
SITE = {"ans_O1": "last|Llast", "ans_para": "last|Llast", "src_O1": "src|Llast", "off": "last|none"}
CTRL_KOS = ("src_all", "src_l2", "src_lo", "src_hi", "rand", "nonsrc", "allbut")


def run_summarize(src_py, job_dir, ninety):
    mod = resum.load_module(src_py, "k16o_x")
    tmp = resum.copy_records(job_dir, ["k16p_*.json", "proc_*.json", "check_cpu_*.json"], skip_dirs=("logs",))
    hits = {}
    orig = mod.boot_cluster
    if ninety:
        def bc(xs, cl, *a, **k):
            out = orig(xs, cl, *a, **k)
            if out is not None:
                out = out + [len(set(c for x, c in zip(xs, cl) if x is not None))]
            return out
        mod.boot_cluster = bc
    with resum.quiet():
        if ninety:
            with resum.ninety(hits):
                mod.summarize(tmp)
        else:
            mod.summarize(tmp)
    mod.boot_cluster = orig
    return common.load_json(os.path.join(tmp, "checks_detail.json")), hits


def _get(d, path):
    cur = d
    for p in path:
        cur = cur[p]
    return cur


def _cells(ck):
    """Yield (meta, path) for every extracted cell of the stored checks_detail."""
    for m, dd in (ck.get("y1_2x2") or {}).items():
        for ds, subs in dd.items():
            cells = subs.get("all") or {}
            for cell, f in cells.items():
                en, kn = cell.split("|")
                for met in ("e", "dm"):
                    if f.get(met) is None:
                        continue
                    yield dict(m=m, ds=ds, en=en, kn=kn, metric=met, kind="2x2"), ["y1_2x2", m, ds, "all", cell, met]
    for m, dd in (ck.get("y1_controls") or {}).items():
        for ds, subs in dd.items():
            for en, kos in (subs.get("all") or {}).items():
                for kn in CTRL_KOS:
                    for met in ("E1", "rescue"):
                        if (kos.get(kn) or {}).get(met) is None:
                            continue
                        yield dict(m=m, ds=ds, en=en, kn=kn, metric=met, kind="ctrl"), \
                            ["y1_controls", m, ds, "all", en, kn, met]
    for m, dd in (ck.get("y3_undo") or {}).items():
        for ds, subs in dd.items():
            for key, f in (subs.get("all") or {}).items():
                if f.get("e") is None:
                    continue
                yield dict(m=m, ds=ds, en=key, kn="", metric="e", kind="y3"), ["y3_undo", m, ds, "all", key, "e"]
    for m, dd in (ck.get("w2_window") or {}).items():
        for ds, subs in dd.items():
            for j, pl in enumerate((subs.get("all") or {}).get("per_loop") or []):
                if pl.get("summed_dm") is None:
                    continue
                yield dict(m=m, ds=ds, en="W2", kn="", metric="summed_dm", kind="w2", loop=pl["loop"]), \
                    ["w2_window", m, ds, "all", "per_loop", j, "summed_dm"]


def extract(job_dir, src_py, ck90=None):
    ck = common.load_json(os.path.join(job_dir, "checks_detail.json"))
    if ck90 is None:
        ck90, _ = run_summarize(src_py, job_dir, True)
    rows = []
    for meta, path in _cells(ck):
        v = _get(ck, path)
        v90 = _get(ck90, path)
        m, ds, en, kn, met, kind = meta["m"], meta["ds"], meta["en"], meta["kn"], meta["metric"], meta["kind"]
        pc = (((ck.get("y1_positive_control") or {}).get(m) or {}).get(ds) or {}).get("all") or {}
        pcv = pc.get("src_O1_e0" if met == "e" else "src_O1_E0")
        rnd = None
        if kind == "2x2":
            grid, card = ("B", "edit_site_map") if kn == "none" else ("C", "attention_knockout")
            arm, site, loop, st = (en if kn == "none" else "%s|%s" % (en, kn)), SITE[en], 1, STRENGTH[en]
        elif kind == "ctrl":
            grid, card, arm, site, loop, st = "C", "attention_knockout", "%s|%s" % (en, kn), SITE[en], 1, STRENGTH[en]
            if kn.startswith("src_"):
                r_ = ck["y1_controls"][m][ds]["all"][en].get("rand") or {}
                rnd = (r_.get(met) or [None])[0]
        elif kind == "y3":
            grid, card, site, st = "C", "restore", "src|Llast", 1.0
            arm = {"Y1|src_O1|none": "src_O1", "Y3|undo2": "undo2", "Y3|undo3": "undo3"}[en]
            loop = {"src_O1": 1, "undo2": 2, "undo3": 3}[arm]
            pcv = pc.get("src_O1_e0")
        else:
            grid, card, arm, site, loop, st, pcv = "C", "attention_knockout", "W2", "last|window", meta["loop"], "", None
        rows.append(common.row(
            grid=grid, card=card, script=SCRIPT, model=m, dataset=DS.get(ds, ds), arm=arm, site=site, loop=loop,
            strength=st, metric=met, endpoint="final", item_set="own", n_items=v[3], n_clusters=v90[4],
            estimate=v[0], ci90_lo=v90[1], ci90_hi=v90[2], ci90_method="recomputed", ci95_lo=v[1], ci95_hi=v[2],
            random_control_estimate=rnd, positive_control_value=pcv[0] if pcv else None,
            source_key="checks_detail.json:" + json.dumps(path)))
    return rows


def _ids(items):
    """Item ids. Records do not store the D4/D5 data row id (a/b hold the template id, and up to 3 pairs share a
    template), so a repeated "half|rel|a|b" gets "#k" for its k-th occurrence in selection order (the scan order of
    the candidate file, fixed per run)."""
    seen, out = {}, set()
    for it in items:
        if it.get("id"):
            out.add(it["id"])
            continue
        base = "%s|%s|%s|%s" % (it.get("half"), it.get("rel"), it.get("a"), it.get("b"))
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.add(base if k == 0 else "%s#%d" % (base, k))
    return out


def _merged(job_dir):
    """Datasets per model, merged over k16p_<model><ftag>.json files in sorted order (as summarize does)."""
    res = {}
    for f in sorted(glob.glob(os.path.join(job_dir, "k16p_*.json"))):
        d = common.load_json(f)
        res.setdefault(d["model"], {}).update(d["datasets"])
    return res


def item_sets(job_dir):
    """Planned = the pairs the quota scan selected (the items list; a dataset scanned but skipped below --min_items
    is planned empty, as in public.py); done = items with a record. A truncated scan (budget) is not marked here."""
    out = {}
    for m, dss in _merged(job_dir).items():
        for ds, D in dss.items():
            done = _ids(D.get("items") or [])
            out[(m, DS.get(ds, ds))] = (set() if D.get("skipped_lt_min_items") else set(done), done)
    return out


def verify(job_dir, src_py):
    ck = common.load_json(os.path.join(job_dir, "checks_detail.json"))
    ck95, _ = run_summarize(src_py, job_dir, False)
    ck90, hits = run_summarize(src_py, job_dir, True)
    rows = extract(job_dir, src_py, ck90)
    n = eq = 0
    bad = []
    for r in rows:
        path = json.loads(r["source_key"].split(":", 1)[1])
        st = _get(ck, path)
        ext = [float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"]), int(r["n_items"])]
        r9 = _get(ck90, path)
        for lab, a, b in (("extracted", ext, st), ("rerun95", _get(ck95, path), st), ("est90", r9[0], st[0]),
                          ("n90", r9[3], st[3])):
            n += 1
            if a == b:
                eq += 1
            elif len(bad) < 10:
                bad.append((lab, "/".join(map(str, path)), a, b))
    whole = {k: ck95.get(k) == ck.get(k) for k in ck}
    trunc = {"%s|%s" % (m, ds): True for m, dss in _merged(job_dir).items() for ds, D in dss.items()
             if D.get("truncated_items") or D.get("truncated_selection")}
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "quantile_reads_patched": hits,
            "checks_detail_keys_equal_on_rerun": sum(whole.values()), "checks_detail_keys": len(whole),
            "checks_detail_keys_differing": [k for k, v in whole.items() if not v], "truncated": trunc}
