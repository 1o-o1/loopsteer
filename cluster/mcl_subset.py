"""Selection file for the McLeish generation smoke: the first N scored problems, sentinels and shift rows of the
shipped selection (cluster/pilot/data/mcleish_setting/items.json, the 237-problem selection of the shared-host run).

    python cluster/mcl_subset.py IN_ITEMS_JSON OUT_ITEMS_JSON [--n 3]
The output keeps every other field of the selection unchanged; it is written atomically (temp file, then rename).
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    sel = json.load(open(a.src, encoding="utf-8"))
    for k in ("scored", "sentinels", "shift"):
        if isinstance(sel.get(k), list):
            sel[k] = sel[k][:a.n]
    sel["smoke_subset"] = {"n": a.n, "source": os.path.basename(a.src)}
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    tmp = "%s.%d.tmp" % (a.dst, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    os.replace(tmp, a.dst)
    print(json.dumps({k: len(sel.get(k) or []) for k in ("scored", "sentinels", "shift")}))


if __name__ == "__main__":
    main()
