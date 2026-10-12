"""Shared pieces of the results extractors (RA-20 item 10).

Every extractor returns a list of dicts with the RESULT_COLUMNS keys and writes results.csv next to the job's
records. CI columns: ci90_lo / ci90_hi hold the 90 percent cluster-bootstrap interval. ci90_method says where it
comes from: "stored" (the script stored a 90 percent interval), "recomputed" (the script stored 95 percent only; the
90 percent interval was recomputed from its per-item rows with its own bootstrap code and seed), or "missing_95_only"
(the script stored 95 percent only and no per-item rows exist; ci90 columns are empty, ci95 columns hold the stored
interval). Columns a script does not produce are left empty, never filled with a guess. metric names the scored
quantity (e for the edit-effect scripts, gold_B for generation, spec/ret/loc/... for selective control); endpoint
names the readout the script chose (e_LO full-span log-odds, e first-token margin); item_set is "own" (the model's
own kept set; the shared-set readout of RA-20 item 9 is computed later by the summarizer, not by an extractor).
neither_rate is filled on generation rows only (grid A): the share of scored problems whose answer is neither gold_A
nor gold_B in the same row, next to the changed-gold rate (metric gold_B) and on every metric row of that row.
"""
import csv
import json
import math
import os

RESULT_COLUMNS = [
    "grid", "card", "script", "model", "dataset", "arm", "site", "loop", "strength", "metric", "endpoint",
    "item_set", "n_items", "n_clusters", "estimate", "neither_rate", "ci90_lo", "ci90_hi", "ci90_method", "ci95_lo", "ci95_hi",
    "random_control_estimate", "positive_control_value", "source_key",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row(**kw):
    out = {c: "" for c in RESULT_COLUMNS}
    for k, v in kw.items():
        if k not in out:
            raise KeyError(f"unknown result column {k}")
        out[k] = fmt(v)
    return out


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return repr(v)
    return str(v)


def write_results(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def read_results(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
