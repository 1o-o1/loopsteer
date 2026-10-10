"""Generation select step at the full pool (RA-20 Amendment 2 item 4; correction of 2026-10-09: every problem that passes
the clean filter is scored).

steer_gen.py select keeps the first n_scored problems, in its seeded shuffle order, that pass both clean filters
(own answer = gold_A on A, = gold_B on B), then places the sentinels (22 in run 1) among the prep items left over,
nearest the prompt-length quantiles of the scored set, and takes the 8 shift rows from the shortest GSM8K eval
prompts outside both sets. Here n_scored is unbounded, so every passing problem is scored and the sentinels come
from the problems that failed a filter. steer_gen.py itself is not modified: this is its own select mode with
--config-json (a file path; the merged config is written next to the outputs as select_cfg_final.json).

    python cluster/steer_select.py --setting a1 --out OUT --settle-dir DIR --config-json CFG.json

The job fails when fewer than --min_scored (40) problems pass. Fewer sentinels than n_sent (when fewer problems
fail a filter than n_sent) or fewer shift rows than n_shift are recorded in select_full_pool.json, not a failure.
A smoke test caps the scored set with --max_scored.
"""
import argparse
import json
import os
import subprocess
import sys


def select(setting, out, settle, cfg, extra, tag):
    path = os.path.join(out, "select_cfg_%s.json" % tag)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
    cmd = [sys.executable, "-m", "prod.steer_gen", "select", "--setting", setting, "--out", out,
           "--settle-dir", settle, "--config-json", path] + extra
    subprocess.check_call(cmd)
    return json.load(open(os.path.join(out, "items.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle-dir", required=True)
    ap.add_argument("--config-json", default="", help="steer_gen config file (JSON object)")
    ap.add_argument("--think-dir", default="")
    ap.add_argument("--min_scored", type=int, default=40, help="fewest scored problems accepted (RA-20 item 1)")
    ap.add_argument("--max_scored", type=int, default=0, help="cap on n_scored (0: none; smoke tests only)")
    a = ap.parse_args()
    extra = ["--think-dir", a.think_dir] if a.think_dir else []
    cfg = json.load(open(a.config_json, encoding="utf-8")) if a.config_json else {}
    os.makedirs(a.out, exist_ok=True)
    n = a.max_scored if a.max_scored > 0 else 100000
    final = select(a.setting, a.out, a.settle_dir, dict(cfg, n_scored=n), extra, "final")
    fails = final.get("filter_fails") or {}
    rec = {"n_scored": len(final["scored"]), "filter_fails": fails,
           "n_sentinels": len(final.get("sentinels") or []), "n_shift": len(final.get("shift") or []),
           "n_sent_config": cfg.get("n_sent"), "n_shift_config": cfg.get("n_shift"), "max_scored": a.max_scored}
    json.dump(rec, open(os.path.join(a.out, "select_full_pool.json"), "w"), indent=1)
    print(json.dumps(rec))
    if rec["n_scored"] < a.min_scored:
        raise SystemExit("only %d problems pass both filters, fewer than %d" % (rec["n_scored"], a.min_scored))


if __name__ == "__main__":
    main()
