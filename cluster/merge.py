"""RA-20 items 5 and 10: merge every job's results.csv into one results_all.csv per grid, assert per job that the
union of done items equals the planned set, and write the completeness matrix BEFORE any number.

    python cluster/merge.py --out_root OUT --manifest cluster/manifest.yaml --skipped cluster/cells_skipped.json
                            [--joblist cluster/jobs/joblist.tsv] [--dest OUT/merged]

Order of writing (the paper chat reads the matrix first):
  1. completeness.md / completeness.csv: model x dataset x card, each cell done, pending (what is missing),
     skip with its reason, done earlier (not rerun), or not run (scope) with the manifest's reason.
  2. results_all_<grid>.csv: rows of complete cells only. A cell missing any planned item or any arm is pending
     as a whole and contributes no row. Rows of cards that are not run (site selection, selective control) that a
     script writes as a by-product are left out. A cell the manifest marks done earlier is read like a run cell
     when a routed job wrote it again (k17_varbench --datasets VBD on the Ouro base models also runs the copy
     pairs) and its folder holds it complete: state done, its rows go in. The dataset column names the cell's own
     dataset: a row labelled with the name of another manifest dataset (k17_public runs the LRE pool as D1 and
     labels it facts) takes the cell's code in lower case (lre). Rows of a folder with a done cell that no cell
     claims go in with empty cell columns (rows_without_cell). No two rows of one results_all file may share the
     key LVC_KEY; a shared key fails the merge (exit 3).
  3. shared_sets.csv and the shared-set rows (B-EXT2 item 0): per benchmark (grid, card, dataset) across every
     model whose cell is done, the items all of them keep (the intersection of the planned sets in items_done.json,
     ids made comparable across scripts by extract/shared.canon; k15_pairs ids form their own space). Each done
     cell is reported on its own kept set (item_set own) and on the shared set (item_set shared: the script's own
     summarize re-run on the shared items, or for generation the per-item cells re-read, extract/shared.py), each
     row with its n. Small sets are kept and marked (own set or shared set below 40). One done model: shared =
     own. Models still pending are named in the note.
  4. grid_<grid>.md: the grid table (model x dataset per card), skip reasons from cells_skipped.json in the
     cells that have no number; each done cell shows its own n and its shared-set mark.
Home directory runs (ALLOW_HOME=1, a cluster with no scratch): before anything is read, every text file of the
output trees the job list names (records, their hash headers, results) has the project root ($PROJECT_ROOT)
rewritten to the token <PROJECT_ROOT>; merged/ gets the same after writing. Never touched: cluster.env, run_all.env,
run_nodes.env (by path; a changed one fails the merge, exit 3), any *.env, the top-level files and _* folders of
OUT_ROOT (run state the scripts read back) and logs/. merged/home_scrub_changed.txt lists every changed file. Then
the merged outputs are scanned for any home path ($HOME, /home/, /Users/, /root/); one hit fails the merge (exit 3,
home_scan.json names the files).
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
    """B-EXT2 item 0. The shared set of a benchmark (grid, card, dataset) is defined across every model whose cell is
    done: the items all of them keep (each script keeps the items the model solves on the clean run). Item ids are
    put in one space per script family (extract/shared.canon); k15_pairs ids are positions in its own scan, so its
    models form their own space (shared.space) with their own intersection. Every done model of a space with two or
    more models gets its rows recomputed on the shared set (item_set shared, n_items = the shared n), however small
    the set (a set below 40 is marked small; an empty one has no rows). One model alone: shared = own. Models still
    pending are named: the set can shrink when they finish (the merge reruns every round). Returns (shared rows per
    grid, shared_sets table rows, mark per cell key)."""
    rows_by_grid = collections.defaultdict(list)
    table, mark = [], {}
    groups = collections.OrderedDict()
    for c in cells:
        key = (c["card"], c["model"], c["dataset"])
        if key in info:
            groups.setdefault((c["grid"], c["card"], c["dataset"]), []).append(c["model"])
    for (grid, card, ds), mods in groups.items():
        st = {mo: info[(card, mo, ds)] for mo in mods}
        done = [mo for mo in mods if st[mo]["state"] == "done"]
        pend = [mo for mo in mods if st[mo]["state"] == "pending"]
        if not done:
            continue
        spaces = collections.OrderedDict()
        for mo in done:
            spaces.setdefault(SH.space(st[mo]["script"]), []).append(mo)
        for sp, members in spaces.items():
            note = []
            if pend:
                note.append("models pending (the set can shrink): %s" % ",".join(pend))
            if len(spaces) > 1:
                note.append("item ids of the %s space are not comparable with the other space%s" % (
                    sp, "s" if len(spaces) > 2 else ""))
            canon = {}
            for mo in members:
                cm = canon.setdefault(mo, collections.defaultdict(set))
                for i in st[mo]["planned"]:
                    cm[SH.canon(st[mo]["script"], i)].add(i)
            shared = set.intersection(*[set(v) for v in canon.values()])
            n = len(shared)
            if len(members) == 1:
                state = "shared = own"
            elif n == 0:
                state = "no common items"
            elif n < MIN_KEPT[0]:
                state = "shared (small: n %d < %d)" % (n, MIN_KEPT[0])
            else:
                state = "shared"
            for mo in members:
                i = st[mo]
                ck = (card, mo, ds)
                if state == "shared = own" or n == 0:
                    mark[ck] = state
                    continue
                if not SH.supported(i["script"]):
                    mark[ck] = "shared n %d; readout not built for %s" % (n, i["script"])
                    continue
                keep = set().union(*[canon[mo][x] for x in shared])
                try:
                    got = SH.rows(i["script"], src_of(i["script"], pilot_dir), os.path.join(out_root, i["rel"]),
                                  i["mk"], i["dk"], keep)
                except Exception as e:                       # noqa: BLE001  (named in the mark; own rows stay)
                    mark[ck] = "shared n %d; readout failed: %s" % (n, repr(e)[:160])
                    continue
                for r in got:
                    if r["card"] in SCOPE_CARDS:
                        continue
                    rows_by_grid[grid].append(cell_row(r, {"card": card, "model": mo, "dataset": ds}))
                mark[ck] = "shared n %d" % n + (" (small)" if n < MIN_KEPT[0] else "")
            table.append([grid, card, ds, sp, ",".join(mods), ",".join(members), n, state, "; ".join(note)])
    return rows_by_grid, table, mark


def takes(r, model_key, ds_key):
    """True when a results.csv row belongs to the cell read with this model key and dataset key."""
    return (model_key in (None, "row") or r["model"] == model_key) and (ds_key is None or r["dataset"] == ds_key)


def cell_row(r, c):
    """One results_all row of manifest cell c (card, model, dataset). The dataset column names the cell's own
    dataset: a label that names another manifest dataset (GJ.DS_NAME, e.g. facts for D1 on the LRE cell) is replaced
    by the cell's dataset code in lower case, so the D1 and LRE rows of one model do not share a key."""
    out = dict(r, cell_card=c["card"], cell_model=c["model"], cell_dataset=c["dataset"])
    if c["dataset"] not in GJ.DS_NAME and r.get("dataset") in GJ.DS_NAME.values():
        out["dataset"] = c["dataset"].lower()
    return out


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
    rows = [r for r in common.read_results(rpath) if takes(r, model_key, ds_key)]
    if not rows:
        return "pending", "item sets complete but no result rows", []
    n_max = max(int(r["n_items"] or 0) for r in rows)
    empty = [r["arm"] for r in rows if not r["n_items"] or int(r["n_items"]) == 0]
    if empty:
        return "pending", "arms without items: %s" % ",".join(sorted(set(empty))[:6]), []
    small = ("small set (n_planned %d < %d), kept; " % (n_planned, MIN_KEPT[0])
             if 0 < n_planned < MIN_KEPT[0] and ds_key is not None else "")
    return "done", "%sn_planned %d, %d result rows, max n_items %d" % (small, n_planned, len(rows), n_max), rows


def planned_items(out_root, rel, model_key, ds_key):
    """Union of the planned item ids of one cell (the model's own kept set) from items_done.json."""
    sets = json.load(open(os.path.join(out_root, rel, "items_done.json"), encoding="utf-8"))
    out = set()
    for k, v in sets.items():
        if (ds_key is None or k.endswith("|" + ds_key)) and (model_key is None or k.split("|")[0] == model_key):
            out |= set(v["planned"])
    return out


HOME_TOKEN = b"<PROJECT_ROOT>"
SKIP_SCRUB = (".env", ".sh", ".sbatch", ".py", ".pt", ".bin", ".safetensors", ".npy", ".npz", ".pkl", ".gz", ".zip")


def home_prefixes():
    """The project root (cluster.env PROJECT_ROOT) as given and resolved, longest first."""
    pr = os.environ.get("PROJECT_ROOT", "").rstrip("/")
    out = {x for x in (pr, os.path.realpath(pr) if pr else "") if x and x != "/"}
    return sorted(out, key=len, reverse=True)


# run state the scripts read back is never scrubbed: the environment files (by path, then any *.env), and every
# top-level file and _* folder of OUT_ROOT (chain markers, job lists, held units, done markers, diagnoses) and logs/
PROTECTED_ENV = ("cluster.env", "run_all.env", "run_nodes.env")


def scrub_roots(out_root, joblist):
    """The folders the scrub rewrites: the output trees the job list names ($OUT_ROOT/<top folder>/...), never a
    _* folder, logs/ or OUT_ROOT itself."""
    tops = set()
    with open(joblist, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rel = r.get("out", "").replace("$OUT_ROOT", "").strip("/")
            top = rel.split("/")[0] if rel else ""
            if top and not top.startswith("_") and top != "logs":
                tops.add(top)
    return [os.path.join(out_root, t) for t in sorted(tops) if os.path.isdir(os.path.join(out_root, t))]


def scrub_home(roots, prefixes, out_root, changed=None):
    """Rewrite every prefix to <PROJECT_ROOT> in each text file under the given folders (line by line through a
    temporary file, then an atomic replace). Never touches the environment files (PROTECTED_ENV, by path, and any
    *.env). Appends each changed file (relative to out_root) to `changed`; returns the number changed."""
    pb = [x.encode() for x in prefixes]
    keep = {os.path.normcase(os.path.realpath(os.path.join(out_root, x))) for x in PROTECTED_ENV}
    n = 0
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                path = os.path.join(dirpath, fn)
                if (fn.endswith(SKIP_SCRUB) or ".scrub." in fn
                        or os.path.normcase(os.path.realpath(path)) in keep):
                    continue
                try:
                    with open(path, "rb") as f:
                        hit = any(x in line for line in f for x in pb)
                except OSError:
                    continue
                if not hit:
                    continue
                tmp = "%s.scrub.%d" % (path, os.getpid())
                with open(path, "rb") as f, open(tmp, "wb") as g:
                    for line in f:
                        for x in pb:
                            line = line.replace(x, HOME_TOKEN)
                        g.write(line)
                os.replace(tmp, path)
                n += 1
                if changed is not None:
                    changed.append(os.path.relpath(path, out_root).replace(os.sep, "/"))
    return n


def _sha(path):
    """sha256 of a file, None when it does not exist."""
    import hashlib
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def home_scan(dest):
    """Files under dest that still hold a home path: $HOME (as given and resolved), /home/, /Users/, /root/."""
    pats = [b"/home/", b"/Users/", b"/root/"]
    for h in (os.environ.get("HOME", ""), os.path.realpath(os.environ.get("HOME", "")) if os.environ.get("HOME") else ""):
        if h and h.rstrip("/") not in ("", "/"):
            pats.append(h.rstrip("/").encode() + b"/")
    hits = []
    for dirpath, _dirs, files in os.walk(dest):
        for fn in files:
            path = os.path.join(dirpath, fn)
            data = open(path, "rb").read()
            found = sorted({x.decode() for x in pats if x in data})
            if found:
                hits.append({"file": os.path.relpath(path, dest), "patterns": found})
    return hits


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
    scrubbed = None
    if os.environ.get("ALLOW_HOME") == "1":        # the run sat in a home directory: scrub before anything is read
        pre = home_prefixes()
        if not pre:
            print("merge: ALLOW_HOME=1 but PROJECT_ROOT is not set (cluster.env); nothing can be scrubbed")
            return 3
        changed = []
        env_before = {x: _sha(os.path.join(a.out_root, x)) for x in PROTECTED_ENV}
        scrubbed = scrub_home(scrub_roots(a.out_root, a.joblist), pre, a.out_root, changed)
        print("merge: running inside a home directory; %d files under the output trees now carry <PROJECT_ROOT>"
              % scrubbed)
    m, order, last, cells = GJ.load_manifest(a.manifest)
    skips = GJ.load_skips(a.skipped)
    labels = {x["id"]: x["label"] for x in m["models"]}

    # ---- 1. completeness, before any number is read into a table
    comp = []
    rows_by_grid = collections.defaultdict(list)
    union_ok = True
    info = {}
    claims = collections.defaultdict(list)      # folder -> (model key, dataset key, grid) of every cell routed to it
    done_rel = set()                            # folders with at least one done cell

    def read_cell(c, key, tgt, by_product=False):
        """Judge one routed cell on its folder; a done cell's rows go into results_all. by_product: a cell the
        manifest marks done earlier, read only when the folder holds it complete (else None, nothing recorded)."""
        u, rel, mk, dk = tgt
        claims[rel].append((mk, dk, c["grid"]))
        word, det, rows = job_state(a.out_root, rel, mk, dk)
        if by_product and word != "done":
            return None
        if word == "done":
            done_rel.add(rel)
        info[key] = {"state": word, "rel": rel, "mk": mk, "dk": dk, "script": GJ.route(*key)[0],
                     "planned": planned_items(a.out_root, rel, mk, dk) if word == "done" else set()}
        for r in rows:
            if r["card"] in SCOPE_CARDS:        # rows a script writes for a card that is not run (scope)
                continue
            rows_by_grid[c["grid"]].append(cell_row(r, c))
        return word, det

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
            tgt = GJ.cell_target(*key)
            got = read_cell(c, key, tgt, by_product=True) if tgt is not None else None
            if got is not None:
                word, det = got[0], "written again by a routed job (manifest: done earlier%s); %s" % (
                    ", " + note if note else "", got[1])
        else:
            tgt = GJ.cell_target(*key)
            if tgt is None:
                word, det = ("skip", "%s: %s" % (sk["reason"], sk["why"])) if sk else ("pending", "no pilot script route")
            else:
                word, det = read_cell(c, key, tgt)
                if word == "pending" and "planned items done" in det:
                    union_ok = False
        comp.append([c["grid"], c["card"], c["model"], c["dataset"], c["status"], word, det])
    # every row of a folder with a done cell reaches results_all: rows no routed cell claims (not a scope card) go in
    # with empty cell columns, in the grid of the folder's first cell
    orphans = collections.Counter()
    for rel in sorted(done_rel):
        for r in common.read_results(os.path.join(a.out_root, rel, "results.csv")):
            if r["card"] in SCOPE_CARDS or any(takes(r, mk, dk) for mk, dk, _g in claims[rel]):
                continue
            rows_by_grid[claims[rel][0][2]].append(dict(r, cell_card="", cell_model="", cell_dataset=""))
            orphans[rel] += 1
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
        w.writerow(["grid", "card", "dataset", "id_space", "models", "done_models", "n_shared", "state", "note"])
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
               "mark of the cell's benchmark (shared n, small when below 40, shared = own, or no common items). "
               "Every estimate with its 90 percent CI is in results_all_%s.csv, item_set own and shared. "
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
    if scrubbed is not None:       # merged files that name OUT_ROOT itself (pending details) get the token too
        scrubbed += scrub_home([dest], home_prefixes(), a.out_root, changed)
        with open(os.path.join(dest, "home_scrub_changed.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(x + "\n" for x in changed))
        moved = [x for x in PROTECTED_ENV if _sha(os.path.join(a.out_root, x)) != env_before[x]]
        if moved:
            print("merge: FAIL, the scrub changed %s" % ", ".join(moved))
            return 3
    dup_keys = {}
    for g, rr in rows_by_grid.items():
        kc = collections.Counter(tuple(r.get(k) or "" for k in LVC_KEY) for r in rr)
        if any(v > 1 for v in kc.values()):
            dup_keys[g] = sum(v - 1 for v in kc.values() if v > 1)
    hits = home_scan(dest)
    with open(os.path.join(dest, "home_scan.json"), "w", encoding="utf-8") as f:
        json.dump({"files_scrubbed": scrubbed, "home_path_hits": hits}, f, indent=1)
    if hits:
        print("merge: FAIL, home paths remain in the merged outputs: %s" % ", ".join(h["file"] for h in hits))
        return 3
    if dup_keys:
        print("merge: FAIL, results_all rows share a key (rows beyond the first, per grid): %s" % json.dumps(dup_keys))
        return 3
    print(json.dumps({"cells": len(comp), "states": cnt, "results_rows": {g: len(r) for g, r in rows_by_grid.items()},
                      "shared_rows": {g: len(r) for g, r in sh_rows.items()},
                      "rows_without_cell": dict(orphans), "duplicate_keys": dup_keys,
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
