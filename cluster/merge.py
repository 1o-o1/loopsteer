"""RA-20 items 5 and 10: merge every job's results.csv into one results_all.csv per grid, assert per job that the
union of done items equals the planned set, and write the completeness matrix BEFORE any number.

    python cluster/merge.py --out_root OUT --manifest cluster/manifest.yaml --skipped cluster/cells_skipped.json
                            [--joblist cluster/jobs/joblist.tsv] [--dest OUT/merged]

Order of writing (the paper chat reads the matrix first):
  1. completeness.md / completeness.csv: model x dataset x card, each cell done, pending (what is missing),
     skip with its reason, done earlier (not rerun), or not run (scope) with the manifest's reason.
  2. results_all_<grid>.csv: rows of complete cells only. A cell missing any planned item or any arm is pending
     as a whole and contributes no row. Rows of cards that are not run (site selection, selective control) that a
     script writes as a by-product are left out.
  3. shared_sets.csv and the shared-set rows (RA-20 item 9): per manifest table row (manifest.yaml rows; a
     model not listed is its own row), card and dataset, the items every done model of the row keeps (the
     intersection of the planned sets in items_done.json). Each done cell is reported on its own kept set
     (item_set own) and on the shared set (item_set shared: the script's own summarize re-run on the shared
     items, extract/shared.py). Shared set below 40 items: own set only, marked. A row with one done model has
     shared = own. A row with a pending model, or with done models from different scripts, has no shared readout.
  4. grid_<grid>.md: the grid table (model x dataset per card), skip reasons from cells_skipped.json in the
     cells that have no number; each done cell shows its own n and its shared-set mark.
Per job (one output folder): items_done.json from the extractor gives planned and done item ids per
(model, dataset); the assert is done == planned (and planned non-empty unless the scan kept nothing, which makes the
cell a skip for ability). The arm check: every arm the script wrote for one dataset in any job of the run must
have a row with n_items > 0.
"""
import argparse
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cluster import gen_jobs as GJ          # noqa: E402  (manifest loader, routes, output folders)
from cluster.extract import common          # noqa: E402
from cluster.extract import shared as SH    # noqa: E402

MIN_KEPT = [40]      # RA-20 items 1 and 9; --min_kept lowers it for a smoke test only
SCOPE_CARDS = {"site_selection", "selective_control"}   # manifest "Not run on the cluster"
PILOT = os.path.join(HERE, "pilot")
SRC = {"steer_gen.py": os.path.join(HERE, "pilot", "tree", "prod", "steer_gen.py")}     # the bundle's tree copy


def src_of(script, pilot_dir):
    return SRC.get(script) or os.path.join(pilot_dir, script)


def row_groups(m, grid):
    """{model: (row id, [models of the row])} for one grid from manifest.yaml rows; unlisted models are own rows."""
    out = {}
    for r in (m.get("rows") or {}).get(grid) or []:
        for mod in r["models"]:
            out[mod] = (r["id"], list(r["models"]))
    return out


def shared_readout(m, cells, info, out_root, pilot_dir):
    """RA-20 item 9. Returns (shared rows per grid, shared_sets table rows, mark per cell key)."""
    rows_by_grid = collections.defaultdict(list)
    table, mark = [], {}
    groups = collections.OrderedDict()
    for c in cells:
        key = (c["card"], c["model"], c["dataset"])
        if key not in info:
            continue
        rid, rmods = row_groups(m, c["grid"]).get(c["model"], (c["model"], [c["model"]]))
        groups.setdefault((c["grid"], c["card"], c["dataset"], rid), []).append((c, rmods))
    for (grid, card, ds, rid), members in groups.items():
        row_models = members[0][1]
        st = {c["model"]: info[(c["card"], c["model"], c["dataset"])] for c, _ in members}
        done = [mo for mo in st if st[mo]["state"] == "done"]
        pend = [mo for mo in st if st[mo]["state"] == "pending"]
        outside = [mo for mo in row_models if mo not in st or st[mo]["state"] not in ("done", "pending")]
        if not done:
            continue
        note = []
        if outside:
            note.append("not in the shared set (no routed run cell, or skipped): %s" % ",".join(outside))
        scripts = sorted(set(st[mo]["script"] for mo in done))
        n = ""
        if pend:
            state = "pending"
            note.append("models pending: %s" % ",".join(pend))
        elif len(scripts) > 1:
            state = "not comparable"
            note.append("done models come from different scripts: %s" % ",".join(scripts))
        else:
            shared = set.intersection(*[set(st[mo]["planned"]) for mo in done])
            n = len(shared)
            if len(done) == 1:
                state = "shared = own"
            elif n < MIN_KEPT[0]:
                state = "own only (shared n %d < %d)" % (n, MIN_KEPT[0])
            else:
                state = "shared"
        for mo in done:
            i = st[mo]
            ck = (card, mo, ds)
            if state != "shared":
                mark[ck] = state if state != "pending" else "shared set pending"
                continue
            if not SH.supported(i["script"]):
                mark[ck] = "shared readout not built for %s" % i["script"]
                continue
            for r in SH.rows(i["script"], src_of(i["script"], pilot_dir), os.path.join(out_root, i["rel"]),
                             i["mk"], i["dk"], shared):
                if r["card"] in SCOPE_CARDS:
                    continue
                rows_by_grid[grid].append(dict(r, cell_card=card, cell_model=mo, cell_dataset=ds))
            mark[ck] = "shared n %d" % n
        table.append([grid, card, ds, rid, ",".join(row_models), ",".join(done), n, state, "; ".join(note)])
    return rows_by_grid, table, mark


def job_state(out_root, rel, model_key, ds_key):
    """(state, detail, rows) for one routed cell. state: done | pending."""
    d = os.path.join(out_root, rel)
    rpath, ipath = os.path.join(d, "results.csv"), os.path.join(d, "items_done.json")
    if not os.path.isdir(d):
        return "pending", "not started (no output folder)", []
    if not os.path.exists(rpath) or not os.path.exists(ipath):
        return "pending", "no results.csv / items_done.json (job not finished or no extractor for this script)", []
    sets = json.load(open(ipath, encoding="utf-8"))
    keys = [k for k in sets if (ds_key is None or k.endswith("|" + ds_key)) and
            (model_key is None or k.split("|")[0] == model_key)]
    if not keys:
        return "pending", "no item set for %s|%s in items_done.json" % (model_key, ds_key), []
    bad = []
    n_planned = 0
    for k in keys:
        p, dn = set(sets[k]["planned"]), set(sets[k]["done"])
        n_planned += len(p)
        if p != dn:
            bad.append("%s: %d of %d planned items done, %d extra" % (k, len(p & dn), len(p), len(dn - p)))
    if bad:
        return "pending", "; ".join(bad), []
    if n_planned == 0:
        return "skip", "ability (the clean-pass scan kept 0 items)", []
    rows = [r for r in common.read_results(rpath)
            if (model_key in (None, "row") or r["model"] == model_key) and (ds_key is None or r["dataset"] == ds_key)]
    if not rows:
        return "pending", "item sets complete but no result rows", []
    n_max = max(int(r["n_items"] or 0) for r in rows)
    if 0 < n_planned < MIN_KEPT[0] and ds_key is not None:
        return "skip", "ability (the clean-pass scan kept %d items, fewer than %d)" % (n_planned, MIN_KEPT[0]), []
    empty = [r["arm"] for r in rows if not r["n_items"] or int(r["n_items"]) == 0]
    if empty:
        return "pending", "arms without items: %s" % ",".join(sorted(set(empty))[:6]), []
    return "done", "n_planned %d, %d result rows, max n_items %d" % (n_planned, len(rows), n_max), rows


def planned_items(out_root, rel, model_key, ds_key):
    """Union of the planned item ids of one cell (the model's own kept set) from items_done.json."""
    sets = json.load(open(os.path.join(out_root, rel, "items_done.json"), encoding="utf-8"))
    out = set()
    for k, v in sets.items():
        if (ds_key is None or k.endswith("|" + ds_key)) and (model_key is None or k.split("|")[0] == model_key):
            out |= set(v["planned"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.yaml"))
    ap.add_argument("--skipped", default=os.path.join(HERE, "cells_skipped.json"))
    ap.add_argument("--joblist", default=os.path.join(HERE, "jobs", "joblist.tsv"))
    ap.add_argument("--dest", default="")
    ap.add_argument("--pilot_dir", default=PILOT, help="folder of the pilot script copies (shared-set re-summarize)")
    ap.add_argument("--min_kept", type=int, default=40,
                    help="ability floor and shared-set floor (RA-20: 40); a smoke test on 3 items passes 1")
    a = ap.parse_args()
    MIN_KEPT[0] = a.min_kept
    dest = a.dest or os.path.join(a.out_root, "merged")
    os.makedirs(dest, exist_ok=True)
    m, order, last, cells = GJ.load_manifest(a.manifest)
    skips = GJ.load_skips(a.skipped)
    labels = {x["id"]: x["label"] for x in m["models"]}

    # ---- 1. completeness, before any number is read into a table
    comp = []
    rows_by_grid = collections.defaultdict(list)
    union_ok = True
    info = {}
    for c in cells:
        key = (c["card"], c["model"], c["dataset"])
        st, note = c["status"], c["note"]
        sk = skips.get(key)
        if st == "scope":
            word, det = "not run (scope)", note
        elif st == "skip":
            word, det = "skip", note
        elif st == "done":
            word, det = "done earlier (not rerun)", note
        else:
            tgt = GJ.cell_target(*key)
            if tgt is None:
                word, det = ("skip", "%s: %s" % (sk["reason"], sk["why"])) if sk else ("pending", "no pilot script route")
            else:
                u, rel, mk, dk = tgt
                word, det, rows = job_state(a.out_root, rel, mk, dk)
                if word == "pending" and "planned items done" in det:
                    union_ok = False
                info[key] = {"state": word, "rel": rel, "mk": mk, "dk": dk, "script": GJ.route(*key)[0],
                             "planned": planned_items(a.out_root, rel, mk, dk) if word == "done" else set()}
                for r in rows:
                    if r["card"] in SCOPE_CARDS:        # rows a script writes for a card that is not run (scope)
                        continue
                    rows_by_grid[c["grid"]].append(dict(r, cell_card=c["card"], cell_model=c["model"],
                                                        cell_dataset=c["dataset"]))
        comp.append([c["grid"], c["card"], c["model"], c["dataset"], c["status"], word, det])
    with open(os.path.join(dest, "completeness.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["grid", "card", "model", "dataset", "manifest_status", "state", "detail"])
        w.writerows(comp)
    cnt = collections.Counter(x[5] for x in comp)
    md = ["# Completeness matrix (read this before any number)", "",
          "States: done (every planned item and arm has a record), pending (named what is missing; the cell has no "
          "number), skip (reason), done earlier (not rerun on the cluster), not run (scope).", "",
          "Counts: " + ", ".join("%s %d" % kv for kv in sorted(cnt.items())), ""]
    for (grid, card), group in _by(comp, lambda x: (x[0], x[1])):
        dss = list(collections.OrderedDict((x[3], 1) for x in group))
        md += ["## Grid %s, %s" % (grid, card), "", "| model | " + " | ".join(dss) + " |", "|---" * (len(dss) + 1) + "|"]
        for mod, g2 in _by(group, lambda x: x[2]):
            cellmap = {x[3]: x for x in g2}
            md.append("| %s | " % labels.get(mod, mod) + " | ".join(
                _short(cellmap[d]) if d in cellmap else "" for d in dss) + " |")
        md.append("")
    open(os.path.join(dest, "completeness.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---- 2. shared-set readout (RA-20 item 9); its rows go into results_all with item_set shared
    sh_rows, sh_table, sh_mark = shared_readout(m, cells, info, a.out_root, a.pilot_dir)
    for g, rr in sh_rows.items():
        rows_by_grid[g].extend(rr)
    with open(os.path.join(dest, "shared_sets.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["grid", "card", "dataset", "row", "row_models", "done_models", "n_shared", "state", "note"])
        w.writerows(sh_table)

    # ---- 3. results_all per grid (complete cells only; item_set own and shared)
    cols = common.RESULT_COLUMNS + ["cell_card", "cell_model", "cell_dataset"]
    for grid, rows in rows_by_grid.items():
        with open(os.path.join(dest, "results_all_%s.csv" % grid), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    # ---- 4. grid tables: n per complete cell (own set) with its shared-set mark; the state word elsewhere
    cf_path = os.path.join(a.pilot_dir, "data", "cf_pool", "cf_pool.json")
    cf_note = json.load(open(cf_path, encoding="utf-8"))["header"]["note"] if os.path.exists(cf_path) else ""
    for grid in sorted(set(x[0] for x in comp)):
        g = [x for x in comp if x[0] == grid]
        out = ["# Grid %s" % grid, "",
               "Generated by cluster/merge.py from results_all_%s.csv and the completeness matrix. A number appears "
               "only in a complete cell: n items on the model's own kept set (result rows), then the shared-set "
               "mark of the cell's manifest row (shared n, shared = own, or own only when the shared set is below "
               "40). Every estimate with its 90 percent CI is in results_all_%s.csv, item_set own and shared. "
               "Empty cells carry the skip reason or the state." % (grid, grid), ""]
        rows = rows_by_grid.get(grid, [])
        for card, gc in _by(g, lambda x: x[1]):
            dss = list(collections.OrderedDict((x[3], 1) for x in gc))
            out += ["## %s" % card, "", "| model | " + " | ".join(dss) + " |", "|---" * (len(dss) + 1) + "|"]
            for mod, g2 in _by(gc, lambda x: x[2]):
                cm = {x[3]: x for x in g2}
                cellv = []
                for d in dss:
                    if d not in cm:
                        cellv.append("")
                        continue
                    x = cm[d]
                    if x[5] == "done":
                        rr = [r for r in rows if r["cell_card"] == card and r["cell_model"] == mod
                              and r["cell_dataset"] == d and r["item_set"] == "own"]
                        cellv.append("n %d (%d rows); %s" % (max(int(r["n_items"]) for r in rr), len(rr),
                                                             sh_mark.get((card, mod, d), "no shared readout")))
                    else:
                        cellv.append(_short(x))
                out.append("| %s | " % labels.get(mod, mod) + " | ".join(cellv) + " |")
            out.append("")
            if "CF" in dss and cf_note:
                out += ["CF (CounterFact): %s" % cf_note, ""]
        open(os.path.join(dest, "grid_%s.md" % grid), "w", encoding="utf-8").write("\n".join(out) + "\n")
    lvc = local_vs_cluster(a.out_root, dest)
    print(json.dumps({"cells": len(comp), "states": cnt, "results_rows": {g: len(r) for g, r in rows_by_grid.items()},
                      "shared_rows": {g: len(r) for g, r in sh_rows.items()},
                      "shared_states": collections.Counter(x[7] for x in sh_table),
                      "union_equals_planned_all_jobs": union_ok, "local_vs_cluster_rows": lvc, "dest": "merged"}))
    return 0


# ------------------------------------------------------------------ cluster rerun against the local reference
REF_DIR = os.path.join(HERE, "reference_local")
LVC_KEY = ("grid", "card", "script", "model", "dataset", "arm", "site", "loop", "strength", "metric", "endpoint",
           "item_set")
LVC_FLOAT = ("estimate", "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi", "random_control_estimate", "positive_control_value")
LVC_COLS = ["out", "dataset", "status", "rows_local", "rows_cluster", "rows_matched", "rows_only_local",
            "rows_only_cluster", "max_abs_diff", "max_abs_diff_field", "max_abs_diff_estimate", "max_abs_diff_ci",
            "max_abs_diff_random_control", "max_abs_diff_positive_control", "undefined_on_one_side", "n_items_equal",
            "n_clusters_equal"]


def _f(v):
    try:
        return None if v in ("", None) else float(v)
    except ValueError:
        return None


def _rows_by_key(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {tuple(r.get(k) or "" for k in LVC_KEY): r for r in csv.DictReader(f)}


def local_vs_cluster(out_root, dest, ref_dir=REF_DIR):
    """Every cell finished on our cards before the launch (cluster/reference_local/MANIFEST.json) against the
    cluster's results.csv of the same output folder: per (folder, dataset) the largest absolute difference of the
    estimate, the 90% and 95% bounds and the random and positive controls, and whether n_items and n_clusters are equal
    on every matched row. Written to <dest>/local_vs_cluster.csv; reported, never a gate. A folder staged from the pilot
    artifacts (DONE_SRC=<folder>) compares with itself (all zero)."""
    man_p = os.path.join(ref_dir, "MANIFEST.json")
    if not os.path.exists(man_p):
        return 0
    man = json.load(open(man_p, encoding="utf-8"))["cells"]
    out_rows = []
    for rel in sorted(man):
        ref = _rows_by_key(os.path.join(ref_dir, rel, "results.csv"))
        cp = os.path.join(out_root, rel, "results.csv")
        new = _rows_by_key(cp) if os.path.exists(cp) else None
        for ds in sorted({k[4] for k in ref} | ({k[4] for k in new} if new else set())):
            r_k = {k for k in ref if k[4] == ds}
            row = {"out": rel, "dataset": ds, "rows_local": len(r_k)}
            if new is None:
                row.update(status="not run on the cluster", rows_cluster=0)
                out_rows.append(row)
                continue
            n_k = {k for k in new if k[4] == ds}
            both = sorted(r_k & n_k)
            row.update(status="compared", rows_cluster=len(n_k), rows_matched=len(both),
                       rows_only_local=len(r_k - n_k), rows_only_cluster=len(n_k - r_k))
            best, und = {}, 0
            for k in both:
                for c in LVC_FLOAT:
                    x, y = _f(ref[k].get(c)), _f(new[k].get(c))
                    if x is None and y is None:
                        continue
                    if x is None or y is None:
                        und += 1
                        continue
                    best[c] = max(best.get(c, 0.0), abs(x - y))
            mx = max(best.items(), key=lambda kv: kv[1]) if best else ("", 0.0)
            row.update(max_abs_diff=round(mx[1], 6), max_abs_diff_field=mx[0],
                       max_abs_diff_estimate=round(best.get("estimate", 0.0), 6),
                       max_abs_diff_ci=round(max([best.get(c, 0.0) for c in LVC_FLOAT[1:5]]), 6),
                       max_abs_diff_random_control=round(best.get("random_control_estimate", 0.0), 6),
                       max_abs_diff_positive_control=round(best.get("positive_control_value", 0.0), 6),
                       undefined_on_one_side=und,
                       n_items_equal=all(ref[k].get("n_items") == new[k].get("n_items") for k in both),
                       n_clusters_equal=all(ref[k].get("n_clusters") == new[k].get("n_clusters") for k in both))
            out_rows.append(row)
    with open(os.path.join(dest, "local_vs_cluster.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LVC_COLS, lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    return len(out_rows)


def _by(rows, keyf):
    d = collections.OrderedDict()
    for r in rows:
        d.setdefault(keyf(r), []).append(r)
    return d.items()


def _short(x):
    word, det = x[5], x[6]
    if word in ("done earlier (not rerun)",):
        return "done earlier" + (" (%s)" % det if det else "")
    if word == "done":
        return "done"
    if word.startswith("skip") or word.startswith("not run"):
        return "%s: %s" % (word, det.split(";")[0][:90])
    return "pending: %s" % det[:90]


if __name__ == "__main__":
    sys.exit(main())
