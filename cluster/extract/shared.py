"""Shared-set readout of RA-20 item 9: one cell's rows recomputed on a given item set.

The records of one job are copied to a scratch folder, the items outside the set are removed from the copy for
one (model, dataset), the script's own summarize runs on the copy at 95 percent, and the format's extractor reads
the copy as it reads a job (its 90 percent interval comes from its own summarize re-run). Estimator, bootstrap,
seed and cluster unit stay the script's own; only the item list changes. The job's own files are never written.

Formats with a filter (the scripts that run models of a multi-model manifest row): k17_public.py, k17_parcae.py,
k15_full.py, k15_pairs.py, k16_path_ifm.py. For any other script, rows() raises NotSupported.
"""
import glob
import json
import os

from . import common, k15full, k15pairs, k16ifm, public, resum


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


def supported(script):
    return script in FILTERS


def rows(script, src_py, job_dir, model, ds, keep):
    """Result rows of (model, ds) in job_dir recomputed on the items in keep, item_set 'shared'."""
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
