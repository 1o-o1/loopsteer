"""RA-31 Part B cells (EXT6c) data (CPU, run once before the plan hash): per cell, copies of the unedited reference
job of the cluster round (items.json, r00 chains and cells, an r00 meta extract without host paths) into
data_partb/<cell>/ with data_partb/<cell>/sources_partb.json; the ASDiv pairs and the steer_gen_math wrapper are
copied unchanged from the bundle (tree/data_steer/asdiv/pairs_asdiv.jsonl, steer_gen_math.py). Nothing tokenizes.

    python make_partb_cells.py [--cluster <cluster_local folder>] [--bundle <bundle pilot folder>]
    python make_partb_cells.py --freeze_26b <2.6B Part A pilot folder>    (writes data_partb/partb_sets_2.6b.json)
"""
import argparse
import glob
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = {"svamp_1.4b": "ra20_g2/A/steer_gen/a1_svamp", "asdiv_1.4b": "ra20_tail2/A/steer_gen_math/a1_asdiv",
        "gsm8k_2.6b": "ra20_g2/A/steer_gen/a2_gsm8k", "svamp_2.6b": "ra20_g2/A/steer_gen/a2_svamp",
        "asdiv_2.6b": "ra20_tail2/A/steer_gen_math/a2_asdiv"}
RA30_POOL_26B_ASDIV = "pool_asdiv_ouro_2_6b_base.json"     # read only: the RA-30 builder's copy of the same list


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one(job_dir, out):
    os.makedirs(out, exist_ok=True)
    src = {}

    def only(pat):
        fs = glob.glob(os.path.join(job_dir, pat))
        assert len(fs) == 1, (job_dir, pat, fs)
        return fs[0]
    copies = {"items.json": os.path.join(job_dir, "items.json"), "r00_chains.jsonl": only("r00/chains_*.jsonl"),
              "r00_cells.jsonl": only("r00/cells_*.jsonl"), "r00_meta.json": only("r00/meta_*.json")}
    for dst, p in copies.items():
        rel = os.path.relpath(p, os.path.dirname(os.path.dirname(job_dir))).replace(os.sep, "/")
        src[rel] = sha(p)
        if dst == "r00_meta.json":
            m = json.load(open(p))
            keep = {"model_meta": {"steer_gen": m["model_meta"].get("steer_gen"),
                                   "file_md5": m["model_meta"].get("file_md5"), "name": m["model_meta"].get("name"),
                                   "dtype": m["model_meta"].get("dtype")},
                    "env": {k: m["env"].get(k) for k in ("torch", "transformers", "device")},
                    "model": m.get("model"), "task": m.get("task"), "config_sha256": m.get("config_sha256"),
                    "n_problems": m.get("n_problems"), "horizon": m.get("horizon"),
                    "batch_width_pinned": m.get("batch_width_pinned"), "protocol": m.get("protocol"), "k": m.get("k"),
                    "source_sha256": src[rel]}
            json.dump(keep, open(os.path.join(out, dst), "w"), indent=1, sort_keys=True)
        else:
            shutil.copyfile(p, os.path.join(out, dst))
    sj = json.load(open(os.path.join(job_dir, "r00", "steer_job.json")))
    outs = {f: sha(os.path.join(out, f)) for f in sorted(copies)}
    json.dump({"sources": src, "outputs": outs, "r00_seconds": sj.get("seconds")},
              open(os.path.join(out, "sources_partb.json"), "w"), indent=1, sort_keys=True)
    return outs


def check_ra30_asdiv(job_dir, ra30_data):
    """The 2.6B ASDiv pool equals the RA-30 builder's Table 2 scored list (same idx, same order): -> its sha256."""
    p = os.path.join(ra30_data, RA30_POOL_26B_ASDIV)
    pool = json.load(open(p))
    it = json.load(open(os.path.join(job_dir, "items.json")))
    a = [int(x["idx"]) for x in pool["items"]]
    b = [int(x["idx"]) for x in it["scored"]]
    assert a == b, "the 2.6B ASDiv items.json differs from the RA-30 pool"
    return sha(p)


def build(cluster, bundle, data_root, ra30_data=None):
    res = {}
    for cell, rel in JOBS.items():
        res[cell] = one(os.path.join(cluster, rel), os.path.join(data_root, cell))
    ra30_data = ra30_data or os.path.join(os.path.dirname(HERE), "ra30", "data")
    h = check_ra30_asdiv(os.path.join(cluster, JOBS["asdiv_2.6b"]), ra30_data)
    sp = os.path.join(data_root, "asdiv_2.6b", "sources_partb.json")
    src = json.load(open(sp))
    src["ra30_pool_sha256"] = {"ra30/data/" + RA30_POOL_26B_ASDIV: h, "rule": "scored idx list equal, same order"}
    json.dump(src, open(sp, "w"), indent=1, sort_keys=True)
    res["copies"] = {}
    for ds, fs in (("asdiv", ("pairs_asdiv.jsonl", "MANIFEST.json")), ("mawps", ("pairs_mawps.jsonl", "MANIFEST.json"))):
        dst = os.path.join(HERE, "tree", "data_steer", ds)
        os.makedirs(dst, exist_ok=True)
        for f in fs:
            shutil.copyfile(os.path.join(bundle, "tree", "data_steer", ds, f), os.path.join(dst, f))
        res["copies"]["tree/data_steer/%s/%s" % (ds, fs[0])] = sha(os.path.join(dst, fs[0]))
    shutil.copyfile(os.path.join(bundle, "steer_gen_math.py"), os.path.join(HERE, "steer_gen_math.py"))
    res["copies"]["steer_gen_math.py"] = sha(os.path.join(HERE, "steer_gen_math.py"))
    return res


def freeze_26b(pilot_dir, out_path):
    """The 2.6B top set from a 2.6B base Part A pilot (out/pilot_2.6b): make_partb.freeze_sets with L = 48, the same
    rule (count over donor items, ties by index); written with model ouro_2_6b_base so ra31_gen accepts it."""
    import make_partb as MB
    d = MB.freeze_sets(pilot_dir, L=48, I=5632, H=16)
    d["model"] = "ouro_2_6b_base"
    d["deviation"] = "frozen from the 2.6B Part A pilot; ties by index"
    json.dump(d, open(out_path, "w"), indent=0, sort_keys=True)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(HERE))
    ap.add_argument("--cluster", default=os.path.join(root, "artifacts_spark", "cluster_local"))
    ap.add_argument("--bundle", default=os.path.join(os.path.dirname(root), "loopedcot", "cluster", "pilot"))
    ap.add_argument("--out", default=os.path.join(HERE, "data_partb"))
    ap.add_argument("--ra30_data", default=None)
    ap.add_argument("--freeze_26b", default=None, help="a 2.6B base Part A pilot folder: write partb_sets_2.6b.json")
    a = ap.parse_args()
    if a.freeze_26b:
        print(freeze_26b(a.freeze_26b, os.path.join(a.out, "partb_sets_2.6b.json")))
    else:
        print(json.dumps(build(a.cluster, a.bundle, a.out, a.ra30_data), indent=1))
