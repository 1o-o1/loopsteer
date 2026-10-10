"""Extractor for k16_path_ifm.py output (k16_<model>.json records + --summarize results.json / checks.json).

Stored estimates are in results.json tables.<m>.<D> as [mean, lo, hi, n_items, n_clusters] at 95 percent
(cluster_boot: cluster bootstrap, 1000 resamples, numpy default_rng(0), two scalar np.percentile reads at 2.5 and
97.5). The 90 percent interval comes from the script's own summarize re-run on a copy of the records with those
reads at 5 and 95 (k15pairs.ninety_all: resum.ninety plus the scalar percentile patch).

Rows (dataset D1/D2 in the main run dir, D4/D5 in the d45 run dir; each dir is one job):
  grid C, card input_refresh:  Y2."<protocol>|<j>|<a>" (protocol a, b, post, pre; injection call j; strength a),
                               site "src|inj", metrics e, dm.
  grid B, card edit_site_map:  Y1.O1.edit (last position, a 1), Y1.PV.edit (paraphrase vector, a 2), Y1.SE (source
                               span, a 1; the positive control row), site "<last|src>|Llast", loop 1, metrics e, dm.
  grid C, card attention_knockout (last query position, all later slots, key set in arm):
      Y1.KO_<set> (knockout only), Y1."KO_<set>+<op>|vsKO" (edit given knockout, arm "<op>|KO_<set>"): e, dm;
      Y1."KO_<set>+<op>|interaction" / "|interaction_e": metrics interaction_dm / interaction_e;
      Y1."SE+KO_SRC|vsKO" (arm "SE|KO_SRC"): e, dm; Y1.D_KO_SRC (metric D_KO), Y1.e_KO_over_DKO_O1 (metric
      e_over_DKO, arm "O1|KO_SRC");
      W2.single[s] (single-slot window knockout, site "last|slot<s>", metric dm), W2.suffix[s0] (knockout from slot
      s0 on, site "last|from_slot<s0>", metric dm).
  grid C, card restore:        Y3.SE, Y3."SE+R2", Y3."SE+R3" (loop 1, 2, 3), metrics e, dm.
Random control: for KO_SRC rows, the KO_RND0 cell of the same op and metric (seed 0 of the three random key sets;
the script does not pool them). Positive control: Y1.SE e (metric e) or dm (other metrics) for Y1 rows.
Not extracted: by_rel / e_by_rel sub-strata, delta_norm_sum, pB/flip/top1A/dA/dB fields.
"""
import glob
import json
import os

from . import common, resum
from .k15pairs import ninety_all

SCRIPT = "k16_path_ifm.py"
DS = {"D1": "facts", "D2": "addition", "D4": "code", "D5": "twostep"}
KEYSETS = ["SRC", "RND0", "RND1", "RND2", "NON", "ALL"]


def _subdirs(d):
    return [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]


def run_summarize(src_py, job_dir, ninety):
    mod = resum.load_module(src_py, "k16i_x")
    tmp = resum.copy_records(job_dir, ["k16_*.json", "proc_*.json", "data_status.json"], skip_dirs=_subdirs(job_dir))
    hits = {}
    with resum.quiet():
        if ninety:
            with ninety_all(hits):
                mod.summarize(tmp)
        else:
            mod.summarize(tmp)
    return common.load_json(os.path.join(tmp, "results.json"))["tables"], hits


def _get(d, path):
    cur = d
    for p in path:
        cur = cur[p]
    return cur


def _cells(T):
    """Yield (meta, path inside tables) for every extracted value."""
    for m, dss in T.items():
        for ds, t in dss.items():
            base = [m, ds]
            for key, c in (t.get("Y2") or {}).items():
                prot, j, a = key.split("|")
                for met in ("e", "dm"):
                    if c.get(met) is not None:
                        yield dict(grid="C", card="input_refresh", arm=prot, site="src|inj", loop=int(j), st=a,
                                   metric=met), base + ["Y2", key, met]
            y1 = t.get("Y1") or {}
            for op, site, st in (("O1", "last|Llast", 1.0), ("PV", "last|Llast", 2.0)):
                for met in ("e", "dm"):
                    if (y1.get(op) or {}).get("edit", {}).get(met) is not None:
                        yield dict(grid="B", card="edit_site_map", arm=op, site=site, loop=1, st=st, metric=met,
                                   pc=True), base + ["Y1", op, "edit", met]
            for met in ("e", "dm"):
                if (y1.get("SE") or {}).get(met) is not None:
                    yield dict(grid="B", card="edit_site_map", arm="SE", site="src|Llast", loop=1, st=1.0, metric=met,
                               pc=True), base + ["Y1", "SE", met]
            ops = [op for op in ("O1", "PV") if op in y1]
            for kn in KEYSETS:
                ko = "KO_" + kn
                for met in ("e", "dm"):
                    if (y1.get(ko) or {}).get(met) is not None:
                        yield dict(grid="C", card="attention_knockout", arm=ko, site="last|attn", loop="", st="",
                                   metric=met, rnd=[ko, met] if kn == "SRC" else None), base + ["Y1", ko, met]
                for op in ops:
                    k1 = "%s+%s|vsKO" % (ko, op)
                    for met in ("e", "dm"):
                        if (y1.get(k1) or {}).get(met) is not None:
                            yield dict(grid="C", card="attention_knockout", arm="%s|%s" % (op, ko), site="last|Llast",
                                       loop=1, st=1.0 if op == "O1" else 2.0, metric=met, pc=True,
                                       rnd=[k1, met] if kn == "SRC" else None), base + ["Y1", k1, met]
                    for suf, met in (("interaction", "interaction_dm"), ("interaction_e", "interaction_e")):
                        k2 = "%s+%s|%s" % (ko, op, suf)
                        if y1.get(k2) is not None:
                            yield dict(grid="C", card="attention_knockout", arm="%s|%s" % (op, ko), site="last|Llast",
                                       loop=1, st=1.0 if op == "O1" else 2.0, metric=met, pc=True,
                                       rnd=[k2] if kn == "SRC" else None), base + ["Y1", k2]
            for met in ("e", "dm"):
                if (y1.get("SE+KO_SRC|vsKO") or {}).get(met) is not None:
                    yield dict(grid="C", card="attention_knockout", arm="SE|KO_SRC", site="src|Llast", loop=1, st=1.0,
                               metric=met, pc=True), base + ["Y1", "SE+KO_SRC|vsKO", met]
            if y1.get("D_KO_SRC") is not None:
                yield dict(grid="C", card="attention_knockout", arm="B|KO_SRC", site="last|attn", loop="", st="",
                           metric="D_KO"), base + ["Y1", "D_KO_SRC"]
            if y1.get("e_KO_over_DKO_O1") is not None:
                yield dict(grid="C", card="attention_knockout", arm="O1|KO_SRC", site="last|Llast", loop=1, st=1.0,
                           metric="e_over_DKO"), base + ["Y1", "e_KO_over_DKO_O1"]
            for lab, loop in (("SE", 1), ("SE+R2", 2), ("SE+R3", 3)):
                for met in ("e", "dm"):
                    if ((t.get("Y3") or {}).get(lab) or {}).get(met) is not None:
                        yield dict(grid="C", card="restore", arm=lab, site="src|Llast", loop=loop, st=1.0,
                                   metric=met), base + ["Y3", lab, met]
            w2 = t.get("W2") or {}
            for s, v in enumerate(w2.get("single") or []):
                if v is not None:
                    yield dict(grid="C", card="attention_knockout", arm="W2_single", site="last|slot%d" % s, loop="",
                               st="", metric="dm"), base + ["W2", "single", s]
            for s0, c in (w2.get("suffix") or {}).items():
                if (c or {}).get("dm") is not None:
                    yield dict(grid="C", card="attention_knockout", arm="W2_suffix", site="last|from_slot%s" % s0,
                               loop="", st="", metric="dm"), base + ["W2", "suffix", s0, "dm"]


def extract(job_dir, src_py, t90=None):
    T = common.load_json(os.path.join(job_dir, "results.json"))["tables"]
    if t90 is None:
        t90, _ = run_summarize(src_py, job_dir, True)
    rows = []
    for meta, path in _cells(T):
        m, ds = path[0], path[1]
        v, v90 = _get(T, path), _get(t90, path)
        se = (T[m][ds].get("Y1") or {}).get("SE") or {}
        pcv = (se.get("e") if meta["metric"] == "e" else se.get("dm")) if meta.get("pc") else None
        rnd = None
        if meta.get("rnd"):
            rp = [m, ds, "Y1"] + [meta["rnd"][0].replace("KO_SRC", "KO_RND0")] + meta["rnd"][1:]
            try:
                rnd = (_get(T, rp) or [None])[0]
            except KeyError:
                rnd = None
        rows.append(common.row(
            grid=meta["grid"], card=meta["card"], script=SCRIPT, model=m, dataset=DS.get(ds, ds), arm=meta["arm"],
            site=meta["site"], loop=meta["loop"], strength=meta["st"], metric=meta["metric"], endpoint="final",
            item_set="own", n_items=v[3], n_clusters=v[4], estimate=v[0], ci90_lo=v90[1], ci90_hi=v90[2],
            ci90_method="recomputed", ci95_lo=v[1], ci95_hi=v[2], random_control_estimate=rnd,
            positive_control_value=pcv[0] if pcv else None,
            source_key="results.json:" + json.dumps(["tables"] + path)))
    return rows


def item_sets(job_dir):
    """Planned = the pairs the quota scan selected (the items list; a dataset skipped for a tokenization mismatch is
    planned empty); done = items with a record. D4/D5 items carry the data row id; D1/D2 use half|rel|a|b."""
    out = {}
    for f in sorted(glob.glob(os.path.join(job_dir, "k16_*.json"))):
        d = common.load_json(f)
        for ds, D in d["datasets"].items():
            done = {it.get("id") or "%s|%s|%s|%s" % (it.get("half"), it.get("rel"), it.get("a"), it.get("b"))
                    for it in D.get("items") or []}
            out[(d["model"], DS.get(ds, ds))] = (set() if D.get("skipped") else set(done), done)
    return out


def verify(job_dir, src_py):
    T = common.load_json(os.path.join(job_dir, "results.json"))["tables"]
    t95, _ = run_summarize(src_py, job_dir, False)
    t90, hits = run_summarize(src_py, job_dir, True)
    rows = extract(job_dir, src_py, t90)
    n = eq = 0
    bad = []
    for r in rows:
        path = json.loads(r["source_key"].split(":", 1)[1])[1:]
        st = _get(T, path)
        f = lambda x: None if x == "" else float(x)  # noqa: E731
        ext = [f(r["estimate"]), f(r["ci95_lo"]), f(r["ci95_hi"]), int(r["n_items"]), int(r["n_clusters"])]
        r9 = _get(t90, path)
        for lab, a, b in (("extracted", ext, st), ("rerun95", _get(t95, path), st), ("est90", r9[0], st[0]),
                          ("n90", r9[3:5], st[3:5])):
            n += 1
            if a == b:
                eq += 1
            elif len(bad) < 10:
                bad.append((lab, "/".join(map(str, path)), a, b))
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "quantile_reads_patched": hits,
            "tables_equal_on_rerun": t95 == T}
