#!/bin/bash
# Stage the done folders before jobs/submit_10_rest.sh. The jobs of cluster/done_local.json ran on our own cards and
# left the job list; the remaining stages of a partly done output folder read their outputs, and the merge reads
# every folder. This script copies each entry's 'pulled' folder (relative to $DONE_SRC) to its 'out' folder under
# $OUT_ROOT with the same layout, checks every copied file against its source by sha256 and every recorded
# results.csv sha256 (results_sha256), writes $OUT_ROOT/_done/<job id> for every entry (ids from jobs/done_local.tsv
# when the entry has none), and last writes $OUT_ROOT/_staged.ok. Any missing folder, copy mismatch or recorded-hash
# mismatch: no markers, no _staged.ok, exit 1. Rerunning is safe (files are copied again and checked again).
# Usage: DONE_SRC=<pilot artifact folder> OUT_ROOT=<output tree> bash jobs/stage_done.sh [DONE_JSON [DONE_TSV]]
#   DONE_JSON default cluster/done_local.json, DONE_TSV default cluster/jobs/done_local.tsv; $PYTHON default python3
set -euo pipefail
: "${DONE_SRC:?set DONE_SRC to the folder that holds the pulled folders}"
: "${OUT_ROOT:?set OUT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DONE_JSON="${1:-$HERE/../done_local.json}"
DONE_TSV="${2:-$HERE/done_local.tsv}"
mkdir -p "$OUT_ROOT"
rm -f "$OUT_ROOT/_staged.ok"
"${PYTHON:-python3}" -B - "$DONE_SRC" "$OUT_ROOT" "$DONE_JSON" "$DONE_TSV" <<'EOF'
import csv, hashlib, json, os, shutil, sys

src_root, out_root, done_json, done_tsv = sys.argv[1:5]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def key(e):
    return (e["script"], e["model"], e["out"], e["stage"], e["shard"])


raw = open(done_json, "rb").read()
jobs = json.loads(raw.decode("utf-8"))["jobs"]
tsv_ids = {}
if os.path.exists(done_tsv):
    with open(done_tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tsv_ids[key(r)] = r["id"]
errors = []
folders = {}                                   # out folder (relative to OUT_ROOT) -> pulled folder (relative to DONE_SRC)
for e in jobs:
    if not e["out"].startswith("$OUT_ROOT/"):
        errors.append("out is not under $OUT_ROOT: %s" % e["out"])
        continue
    rel = e["out"][len("$OUT_ROOT/"):]
    for p in (rel, e["pulled"]):
        if os.path.isabs(p) or ".." in p.replace("\\", "/").split("/"):
            errors.append("path leaves its root: %s" % p)
    if folders.setdefault(rel, e["pulled"]) != e["pulled"]:
        errors.append("%s has two pulled folders: %s, %s" % (e["out"], folders[rel], e["pulled"]))
ids = []
for e in jobs:
    jid = e.get("id") or tsv_ids.get(key(e), "")
    if e.get("id") and tsv_ids and tsv_ids.get(key(e)) not in (None, e["id"]):
        errors.append("%s: id %s in done_local.json, %s in done_local.tsv" % (key(e), e["id"], tsv_ids[key(e)]))
    if not jid:
        errors.append("no job id for %s (not in done_local.json or done_local.tsv)" % (key(e),))
    ids.append(jid)
if errors:
    print("\n".join(errors))
    sys.exit(1)

n_files = 0
for rel, pulled in sorted(folders.items()):
    src = os.path.join(src_root, pulled)
    dst = os.path.join(out_root, rel)
    if not os.path.isdir(src):
        errors.append("pulled folder missing: %s" % src)
        continue
    for root, dirs, files in os.walk(src):
        dirs.sort()
        os.makedirs(os.path.join(dst, os.path.relpath(root, src)), exist_ok=True)     # empty folders too
        for fn in sorted(files):
            s = os.path.join(root, fn)
            d = os.path.join(dst, os.path.relpath(s, src))
            shutil.copy2(s, d)
            n_files += 1
            if sha(d) != sha(s):
                errors.append("copy differs from its source: %s" % d)
n_recorded = 0
for e in jobs:
    if not e.get("results_csv"):
        continue
    rp = e["results_csv"].replace("\\", "/")
    base = e["pulled"].rstrip("/") + "/"
    if not rp.startswith(base):
        errors.append("results_csv %s is not inside its pulled folder %s" % (rp, e["pulled"]))
        continue
    d = os.path.join(out_root, e["out"][len("$OUT_ROOT/"):], rp[len(base):])
    if not os.path.exists(d):
        errors.append("results.csv missing after the copy: %s" % d)
        continue
    h = sha(d)
    n_recorded += 1
    if h != e["results_sha256"]:
        errors.append("%s sha256 %s, recorded %s" % (d, h, e["results_sha256"]))
if errors:
    print("\n".join(errors))
    print("stage_done: FAILED (%d problems); no markers written" % len(errors))
    sys.exit(1)

os.makedirs(os.path.join(out_root, "_done"), exist_ok=True)
for jid in sorted(set(ids)):
    open(os.path.join(out_root, "_done", jid), "a").close()
rep = {"done_local_sha256": hashlib.sha256(raw).hexdigest(), "entries": len(jobs), "folders": len(folders),
       "files_copied": n_files, "recorded_sha256_checked": n_recorded, "markers": len(set(ids))}
with open(os.path.join(out_root, "_staged.ok"), "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rep, sort_keys=True) + "\n")
print("stage_done: OK " + json.dumps(rep, sort_keys=True))
EOF
