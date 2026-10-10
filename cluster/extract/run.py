"""Write results.csv and items_done.json next to one job's records (RA-20 item 10).

    python cluster/extract/run.py --script k17_public.py --src $REPO/loop-flow/src/k17_public.py --job OUT_DIR

items_done.json maps "<model>|<dataset>" to {"planned": [...], "done": [...]} so merge.py can assert that the
union of done items equals the planned set per job. A script without an extractor exits with code 3 and writes
nothing; merge.py then reports the job's cells as pending.
"""
import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))       # the folder that holds cluster/

FORMAT = {"k17_varbench.py": "varbench", "k17_public.py": "public", "k17_parcae.py": "public", "steer_gen.py": "gen",
          "k17_freeze.py": "freeze", "k17_freeze_raven.py": "freeze", "k17_x4e.py": "x4e", "k17_trace.py": "trace",
          "analyze_full.py": "prod", "k17_control.py": "control", "k15_full.py": "k15full", "k15_pairs.py": "k15pairs",
          "k16_path_ouro.py": "k16ouro", "k16_path_ifm.py": "k16ifm", "k17_freeze_ifm.py": "freeze",
          "steer_gen_raven.py": "genraven", "steer_gen_math.py": "gen"}


def get(script):
    if script not in FORMAT:
        return None
    return importlib.import_module("cluster.extract." + FORMAT[script])


def run(script, src, job):
    from cluster.extract import common
    mod = get(script)
    if mod is None:
        raise SystemExit(3)
    rows = mod.extract(job, src)
    common.write_results(rows, os.path.join(job, "results.csv"))
    sets = mod.item_sets(job)
    json.dump({"%s|%s" % k: {"planned": sorted(map(str, p)), "done": sorted(map(str, d))} for k, (p, d) in sets.items()},
              open(os.path.join(job, "items_done.json"), "w"), indent=0)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    print(json.dumps({"results_rows": run(a.script, a.src, a.job)}))


if __name__ == "__main__":
    main()
