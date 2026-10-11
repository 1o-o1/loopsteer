"""Shared-set readout of RA-20 item 9: one cell's rows recomputed on a given item set.

The records of one job are copied to a scratch folder, the items outside the set are removed from the copy for
one (model, dataset), the script's own summarize runs on the copy at 95 percent, and the format's extractor reads
the copy as it reads a job (its 90 percent interval comes from its own summarize re-run). Estimator, bootstrap,
seed and cluster unit stay the script's own; only the item list changes. The job's own files are never written.

Formats with a filter (the scripts that run models of a multi-model manifest row): k17_public.py, k17_parcae.py,
k15_full.py, k15_pairs.py, k16_path_ifm.py. Generation (steer_gen.py, steer_gen_math.py, steer_gen_raven.py): the
cell's own result rows with every number recomputed on the scored problems in the set, from the per-item cells
(extract/gen.py recompute: steer_gen's load_cells, outcome and boot_cluster, 95 and 90 percent). For any other script,
rows() raises NotSupported.

canon(script, item id) puts the item ids of different scripts in one space (merge.py intersects them per benchmark):
the dataset tag (D1, D2, D4, D5, CF, VB, LRE) or the half (dev, test, heldout, all, cf) in front and the half at the
end are dropped, so "D1|capital|A|B" (k17_public), "dev|capital|A|B" (k15_full) and "capital|A|B|dev" (k17_freeze)
are one item. k15_pairs ids ("half:i", a position in the scan) have no such key: space() puts them in their own space.
"""
import glob
import json
import os

from . import common, gen, k15full, k15pairs, k16ifm, public, resum


class NotSupported(Exception):
    pass


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _keep_aligned(D, idx, n):
    """Keep positions idx of every list of length n directly under D and under each cell of D['cells']."""
    for k, v in list(D.items()):
        if isinstance(v, list) and len(v) == n:
            D[k] = [v[i] for i in idx]
    for c in (D.get("cells") or {}).values():
        if isinstance(c, dict):
            for k, v in list(c.items()):
                if isinstance(v, list) and len(v) == n:
                    c[k] = [v[i] for i in idx]


def _filter_public(d, model, ds, keep):
    hit = 0
    for f in sorted(glob.glob(os.path.join(d, "res_*.json"))):
        if ".shard" in f:
            os.remove(f)                     # the script's summarize reads merged records only (as public.py does)
            continue
        r = common.load_json(f)
        if r.get("model") != model or public.DS.get(r.get("ds"), r.get("ds")) != ds:
            continue
        r["items"] = [it for it in r.get("items") or [] if str(it["pair_id"]) in keep]
        _dump(r, f)
        hit += 1
    return hit


def _filter_k15(d, model, ds, keep, pattern, dsmap, iid):
    hit = 0
    for f in sorted(glob.glob(os.path.join(d, pattern))):
        if f.endswith("_scores.json"):
            continue
        r = common.load_json(f)
        if r.get("model") != model:
            continue
        for dk, D in r["datasets"].items():
            if dsmap.get(dk, dk) != ds:
                continue
            items = D.get("items") or []
            idx = [i for i, it in enumerate(items) if iid(it) in keep]
            _keep_aligned(D, idx, len(items))
            hit += 1
        _dump(r, f)
    return hit


def _filter_k16ifm(d, model, ds, keep):
    hit = 0
    for f in sorted(glob.glob(os.path.join(d, "k16_*.json"))):
        r = common.load_json(f)
        if r.get("model") != model:
            continue
        for dk, D in r["datasets"].items():
            if k16ifm.DS.get(dk, dk) != ds:
                continue
            items = D.get("items") or []
            idx = [i for i, it in enumerate(items) if _k16_iid(it) in keep]
            _keep_aligned(D, idx, len(items))
            hit += 1
        _dump(r, f)
    return hit


def _k16_iid(it):
    return it.get("id") or "%s|%s|%s|%s" % (it.get("half"), it.get("rel"), it.get("a"), it.get("b"))


def _k15p_iid(it):
    return "%s:%s" % (it["half"], it["i"])


FILTERS = {
    "k17_public.py": (public, _filter_public),
    "k17_parcae.py": (public, _filter_public),
    "k15_full.py": (k15full, lambda d, m, ds, k: _filter_k15(d, m, ds, k, "k15f_*.json", k15full.DS, k15full._iid)),
    "k15_pairs.py": (k15pairs, lambda d, m, ds, k: _filter_k15(d, m, ds, k, "k15p_*.json", k15pairs.DS, _k15p_iid)),
    "k16_path_ifm.py": (k16ifm, _filter_k16ifm),
}


GEN = {"steer_gen.py", "steer_gen_math.py", "steer_gen_raven.py"}
STEER_GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "tree", "prod",
                         "steer_gen.py")       # the bundle's tree copy (load_cells, outcome, boot_cluster)
_TAGS = {"D1", "D2", "D4", "D5", "CF", "VB", "LRE", "dev", "test", "heldout", "all", "cf"}
_HALVES = {"dev", "test", "heldout", "all"}


def supported(script):
    return script in FILTERS or script in GEN


def space(script):
    """Item-id space of a script: k15_pairs ids are positions in its own scan; every other script names the item."""
    return "k15_pairs" if script == "k15_pairs.py" else "common"


def canon(script, iid):
    s = str(iid)
    if script == "k15_pairs.py" or "|" not in s:
        return s
    parts = s.split("|")
    if len(parts) > 1 and parts[0] in _TAGS:
        parts = parts[1:]
    if len(parts) > 2 and parts[-1] in _HALVES:
        parts = parts[:-1]
    return "|".join(parts)


def _gen_rows(job_dir, keep):
    """The cell's own rows (results.csv, item_set own) with the numbers recomputed on scored problems in keep."""
    keep = {int(i) for i in keep}
    r95, _ = gen.recompute(job_dir, STEER_GEN, False, keep=keep)
    r90, _ = gen.recompute(job_dir, STEER_GEN, True, keep=keep)
    rnd = {r: v[4] for r, v in gen.ROWS.items()}
    out = []
    for r in common.read_results(os.path.join(job_dir, "results.csv")):
        if r.get("item_set") != "own" or not r.get("source_key", "").startswith("analysis.json:"):
            continue
        path = json.loads(r["source_key"].split(":", 1)[1])
        row, mname = path[1], path[2]
        v, v90 = (r95.get(row) or {}).get(mname), (r90.get(row) or {}).get(mname)
        if not v or not v90:
            continue
        rv = ((r95.get(rnd.get(row)) or {}).get(mname) or [None])[0] if rnd.get(row) else None
        pc = ((r95.get("r01") or {}).get("gold_B") or [None])[0]
        nr = ((r95.get(row) or {}).get("neither") or [None])[0]
        x = dict(r)
        x.update({k: common.fmt(val) for k, val in (
            ("item_set", "shared"), ("n_items", v[3]), ("n_clusters", v[4]), ("estimate", v[0]), ("ci95_lo", v[1]),
            ("ci95_hi", v[2]), ("ci90_lo", v90[1]), ("ci90_hi", v90[2]), ("ci90_method", "recomputed"),
            ("neither_rate", nr), ("random_control_estimate", rv),
            ("positive_control_value", pc if mname == "gold_B" else None), ("source_key", "shared:" + r["source_key"]))})
        out.append(x)
    return out


def rows(script, src_py, job_dir, model, ds, keep):
    """Result rows of (model, ds) in job_dir recomputed on the items in keep, item_set 'shared'."""
    if script in GEN:
        return _gen_rows(job_dir, keep)
    if script not in FILTERS:
        raise NotSupported(script)
    ext, filt = FILTERS[script]
    keep = set(map(str, keep))
    tmp = resum.copy_records(job_dir, skip_dirs=("logs",))
    if not filt(tmp, model, ds, keep):
        raise KeyError("no records for %s|%s in %s" % (model, ds, job_dir))
    mod = resum.load_module(src_py, "shared_" + os.path.splitext(script)[0])
    with resum.quiet():
        mod.summarize(tmp)
    out = []
    for r in ext.extract(tmp, src_py):
        if r["model"] == model and r["dataset"] == ds:
            r["item_set"] = "shared"
            r["source_key"] = "shared:" + r["source_key"]
            out.append(r)
    return out
