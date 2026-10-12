"""RA-31 data (CPU, run once before the plan hash): copies of the RUN 1c Ouro item lists and the per-item RUN 1c
reference values used by the alpha-0 identity, with the sha256 of every source and output in data/sources.json.

    python make_data.py [--run1c ../run1c]
"""
import argparse
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REF_ROWS = ("E0", "ans_frz_mlp", "ans_frz_attn", "rnd", "rnd_hold_mlp", "rnd_hold_attn", "pc_final", "sz")
COPIES = ("items_d1.json", "ouro_D2.json", "ref_values.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(run1c, out):
    os.makedirs(out, exist_ok=True)
    src = {}
    for f in COPIES:
        p = os.path.join(run1c, "data", f)
        shutil.copyfile(p, os.path.join(out, f))
        src["run1c/data/" + f] = sha(p)
    ref = {"rows": list(REF_ROWS), "datasets": {}}
    for ds in ("D1", "D2"):
        p = os.path.join(run1c, "out", "full", "ouro", "items_%s.jsonl" % ds)
        src["run1c/out/full/ouro/items_%s.jsonl" % ds] = sha(p)
        items = {}
        for ln in open(p):
            if not ln.strip():
                continue
            rc = json.loads(ln)
            items[rc["key"]] = dict({"D": rc["D"], "cluster": rc["cluster"]},
                                    **{nm: rc["rows"][nm]["dm"] for nm in REF_ROWS})
        ref["datasets"][ds] = items
    p = os.path.join(out, "run1c_ref.json")
    json.dump(ref, open(p, "w"), indent=0, sort_keys=True)
    outs = {f: sha(os.path.join(out, f)) for f in COPIES + ("run1c_ref.json",)}
    json.dump({"sources": src, "outputs": outs,
               "n_items": {ds: len(v) for ds, v in ref["datasets"].items()}},
              open(os.path.join(out, "sources.json"), "w"), indent=1, sort_keys=True)
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1c", default=os.path.join(os.path.dirname(HERE), "run1c"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    print(json.dumps(build(a.run1c, a.out), indent=1))
