#!/bin/bash
# B-EXT6c (push 3): one RA-29 cell on the cluster from pilot/ra29_s3 (plan 0aeb6563), the logic of its run.sh `cell`
# with the output folder outside the bundle (run.sh writes into its own folder and refuses a home folder).
#   bash cluster/pilot/e6c_ra29_cell.sh MODEL DATASET OUT PLAN_SHA256
# MODEL ouro26 | think14 | think26 | mcleish | huginn, DATASET aqua | csqa | arc. Steps: the folder's CPU tests; the pilot
# (skipped when OUT/pilot/debug_gate.json exists); the 80 rule on the pilot projection (under 80: no full run, exit 0);
# the 8.5 h cap (RA29_CAP_H) on the pilot-measured full-run card-hours (over it: no full run, exit 3, as run.sh);
# the full run with that pilot's debug_gate.json; its summary and the verdict. SIGTERM (sbatch --signal=TERM@120 in
# the template) stops the run after the current item with exit 5; the same line resumes.
set -euo pipefail
M="${1:?model}"; D="${2:?dataset}"; O="${3:?output folder}"; SHA="${4:?plan sha256}"
CAP_H="${RA29_CAP_H:-8.5}"
cd "$(dirname "$0")/ra29_s3"
PY="${PY:-python}"
MEM="--mem_fraction ${E6_MEM_FRACTION:-0.9}"
mkdir -p "$O"
"$PY" -u -m unittest discover -s tests
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
H="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["timing"]["full_card_hours_at_this_rate"])' "$O/pilot/checks.json")"
if "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' "$H" "$CAP_H"; then
  echo "[ra29] cell $M $D: pilot-measured full run $H card-hours (cap $CAP_H)" | tee -a "$O/cell.log"
else
  echo "[ra29] cell $M $D: pilot-measured full run $H card-hours exceeds the cap $CAP_H: full run not started" | tee -a "$O/cell.log"
  exit 3
fi
echo "[ra29] cell $M $D: projected $P scored items (>= 80): full run" | tee -a "$O/cell.log"
"$PY" -u ra29.py --run --model "$M" --dataset "$D" --out "$O/full" --plan_sha256 "$SHA" --debug_gate "$O/pilot/debug_gate.json" $MEM
"$PY" -u ra29.py --summarize --out "$O/full"
N="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["counts"]["n_kept"])' "$O/full/checks.json")"
if [ "$N" -ge 80 ]; then V="meets"; else V="is below"; fi
echo "[ra29] cell $M $D: full run scored $N items ($V the 80 rule)" | tee -a "$O/cell.log"
