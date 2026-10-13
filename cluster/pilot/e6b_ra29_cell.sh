#!/bin/bash
# B-EXT6b: one RA-29 breadth cell on the cluster, the logic of run.sh `cell` (plan 9b799c25) with the output folder
# outside the bundle (run.sh writes into its own folder and refuses a home folder; the cluster runs in one).
#   bash cluster/pilot/e6b_ra29_cell.sh MODEL DATASET OUT PLAN_SHA256
# MODEL ouro14 | ouro26, DATASET mmlu | hellaswag. Steps: the folder's pinned hashes; the pilot (skipped when
# OUT/pilot/debug_gate.json exists); its summary; the 80 rule on the pilot projection (round(n_kept / n_tried x
# n_built) < 80: the full run is not started, exit 0); the full run with that pilot's debug_gate.json; its summary
# and the verdict against the 80 rule. A failing step stops the cell with its exit code (2 to 5, as ra29.py).
set -euo pipefail
M="${1:?model}"; D="${2:?dataset}"; O="${3:?output folder}"; SHA="${4:?plan sha256}"
case "$M" in ouro14|ouro26) ;; *) echo "MODEL must be ouro14 or ouro26" >&2; exit 2;; esac
cd "$(dirname "$0")/ra29"
PY="${PY:-python}"
MEM="--mem_fraction ${E6_MEM_FRACTION:-0.9}"   # ra29.py default 0.5 is a GB10 share; task_pack.sbatch halves it for two lines
mkdir -p "$O"
# push 4: the folder's pinned hashes instead of its CPU tests (5.7 h on the cluster, OSError in inspect.getsource)
(cd ../../../.. && grep -F '  loopedcot/cluster/pilot/ra29/' loopedcot/cluster/jobs/scripts_sha256.txt | sha256sum -c --quiet)
if [ ! -f "$O/pilot/debug_gate.json" ]; then
  "$PY" -u ra29.py --run --model "$M" --dataset "$D" --out "$O/pilot" --plan_sha256 "$SHA" --pilot $MEM
fi
"$PY" -u ra29.py --summarize --out "$O/pilot"
P="$("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1]))["counts"]; print(int(round(c["n_kept"] / max(1, c["n_tried"]) * c["n_built"])))' \
     "$O/pilot/checks.json")"
if [ "$P" -lt 80 ]; then
  echo "[ra29] cell $M $D: projected $P scored items (< 80): full run not started" | tee -a "$O/cell.log"
  exit 0
fi
echo "[ra29] cell $M $D: projected $P scored items (>= 80): full run" | tee -a "$O/cell.log"
"$PY" -u ra29.py --run --model "$M" --dataset "$D" --out "$O/full" --plan_sha256 "$SHA" --debug_gate "$O/pilot/debug_gate.json" $MEM
"$PY" -u ra29.py --summarize --out "$O/full"
N="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["counts"]["n_kept"])' "$O/full/checks.json")"
if [ "$N" -ge 80 ]; then V="meets"; else V="is below"; fi
echo "[ra29] cell $M $D: full run scored $N items ($V the 80 rule)" | tee -a "$O/cell.log"
