"""B-EXT6 merge, one per test: copy the outputs of one RA test into merged/<test>.

    python cluster/merge_ext6.py --test ra29|ra30|ra31 --out_root OUT [--joblist cluster/jobs/joblist.tsv]

The jobs of a test write under $OUT_ROOT/E6/<test>/ (ra29: ouro26/<dataset>/{pilot,full}; ra30:
ouro_2_6b_base/{tokens,pilot,full}/gsm8k; ra31: a_2.6b/{pilot_2.6b,full_2.6b}). The merge replaces
$OUT_ROOT/merged/<test>/ with a copy of that tree (prod_art/ and prod_logs/ left out), then writes
merged/<test>/MERGE.json: per task of the job list (group of the test) its state (done: the done marker exists;
failed: a diagnosis $OUT_ROOT/_diag/<id>.txt exists; not run), and per copied file its size and sha256. The run tree
itself is never changed. Home directory runs (ALLOW_HOME=1): the copy gets the scrub of merge.py (the project root
becomes <PROJECT_ROOT>; merged/<test>/home_scrub_changed.txt lists the files), then the home scan of merge.py; a hit
fails the merge (exit 3, MERGE.json names the files). It reruns every round (the template ignores its done marker).
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cluster import merge as MG             # noqa: E402  (scrub_home, home_prefixes, home_scan)

GROUPS = {"ra29": ("g20_e6ra29", "g24_e6bra29"), "ra30": ("g19_e6ra30", "g26_e6bra30"),
          "ra31": ("g21_e6ra31a", "g22_e6ra31b", "g25_e6bra31t")}           # the B-EXT6 and B-EXT6b groups
SKIP_DIRS = ("prod_art", "prod_logs", "__pycache__")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def task_states(out_root, joblist, test):
    """[(id, stage, datasets, state)] of the test's tasks in the job list."""
    rows = []
    with open(joblist, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["group"] not in GROUPS[test]:
                continue
            if os.path.exists(os.path.join(out_root, "_done", r["id"])):
                st = "done"
            elif os.path.exists(os.path.join(out_root, "_diag", r["id"] + ".txt")):
                st = "failed (see _diag/%s.txt)" % r["id"]
            else:
                st = "not run"
            rows.append({"id": r["id"], "stage": r["stage"], "out": r["out"].replace("$OUT_ROOT/", ""), "state": st})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=sorted(GROUPS))
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--joblist", default=os.path.join(HERE, "jobs", "joblist.tsv"))
    a = ap.parse_args(argv)
    out_root = os.path.abspath(a.out_root)
    src = os.path.join(out_root, "E6", a.test)
    final = os.path.join(out_root, "merged", a.test)
    # built aside, then swapped in: the B-EXT6 and B-EXT6b merges of one test may run at the same time
    dest = os.path.join(out_root, "merged", ".%s.tmp.%d" % (a.test, os.getpid()))
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    files = []
    if os.path.isdir(src):
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*SKIP_DIRS))
    rep = {"test": a.test, "source": "E6/%s" % a.test, "source_exists": os.path.isdir(src),
           "tasks": task_states(out_root, a.joblist, a.test)}
    code = 0
    if os.environ.get("ALLOW_HOME") == "1":
        pre = MG.home_prefixes()
        changed = []
        if pre:
            rep["files_scrubbed"] = MG.scrub_home([dest], pre, dest, changed)
        else:
            rep["files_scrubbed"] = None
            print("merge_ext6: ALLOW_HOME=1 but PROJECT_ROOT is not set (cluster.env); nothing scrubbed")
        with open(os.path.join(dest, "home_scrub_changed.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(c + "\n" for c in changed))
    hits = MG.home_scan(dest)
    rep["home_path_hits"] = hits
    if hits:
        code = 3
        print("merge_ext6: FAIL, home paths remain in merged/%s: %s" % (a.test, ", ".join(h["file"] for h in hits)))
    for root, dirs, fs in os.walk(dest):
        dirs.sort()
        for fn in sorted(fs):
            p = os.path.join(root, fn)
            files.append({"file": os.path.relpath(p, dest).replace(os.sep, "/"), "bytes": os.path.getsize(p),
                          "sha256": sha256(p)})
    rep["files"] = files
    with open(os.path.join(dest, "MERGE.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    old = "%s.old.%d" % (final, os.getpid())
    if os.path.isdir(final):
        os.replace(final, old)
    os.replace(dest, final)
    shutil.rmtree(old, ignore_errors=True)
    n_done = sum(1 for t in rep["tasks"] if t["state"] == "done")
    print("merge_ext6 %s: %d files into merged/%s; tasks done %d of %d" % (a.test, len(files), a.test, n_done,
                                                                         len(rep["tasks"])))
    return code


if __name__ == "__main__":
    sys.exit(main())
