#!/usr/bin/env bash
# RA-29 driver (plan_ra29.md). One process at a time on one card.
#
#   bash run.sh test                          CPU tests (no GPU, no model) under PY_OURO
#   bash run.sh hash                          plan, code and data sha256 (for the LEDGER line)
#   bash run.sh pilot MODEL DATASET           pilot -> out/MODEL/DATASET/pilot (until 30 kept or 120 tried)
#   bash run.sh pilots                        the three pilots in order: ouro14 aqua, ouro26 aqua, ouro14 csqa
#   bash run.sh full MODEL DATASET            full run -> out/MODEL/DATASET/full (needs that pilot's gate; resumable)
#   bash run.sh aqua26                        2.6B AQuA full run only if 1.4B AQuA scored fewer than 80 (prints which)
#   bash run.sh mains                         main runs in order, stops at the first failure: full ouro14 aqua,
#                                             full ouro14 csqa, pilot (if missing) and full ouro14 arc, aqua26
#   bash run.sh cell MODEL DATASET            one cluster job: the cell's pilot (if missing), the 80 rule on the
#                                             pilot's projected scored count, then the full run (prints which way)
#   bash run.sh summarize MODEL DATASET [pilot|full]   results.csv, results.json, checks.json, summary.md (CPU)
#
# MODEL ouro14 | ouro26, DATASET aqua | csqa | arc | mmlu | hellaswag. PY_OURO = the python of the Ouro venv RA-28 used (RUNSHEET.md).
# Exit codes: 2 plan, pin, item or config mismatch, 3 gate failed or missing pilot gate, 4 identity failed (also on
# a rerun over failed records), 5 stopped by SIGTERM (rerun the same command to resume).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${HOME:-}" ]; then
  case "$HERE/" in "$HOME"/*) echo "refusing: the run folder is inside the home directory; copy it to /var/tmp/ra29" >&2; exit 2;; esac
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
OUT="$HERE/out"
mkdir -p "$OUT/logs"
cd "$HERE"
: "${PY_OURO:?set PY_OURO to the python of the Ouro venv (see RUNSHEET.md)}"
PY="$PY_OURO"
SHA="$(awk '{print $1}' "$HERE/plan_ra29.sha256")"

check_md() {
  case "${1:-}" in ouro14|ouro26) ;; *) echo "MODEL must be ouro14 or ouro26" >&2; exit 2;; esac
  case "${2:-}" in aqua|csqa|arc|mmlu|hellaswag) ;; *) echo "DATASET must be aqua, csqa, arc, mmlu or hellaswag" >&2; exit 2;; esac
}

pilot() {
  check_md "$1" "$2"
  "$PY" ra29.py --run --model "$1" --dataset "$2" --out "$OUT/$1/$2/pilot" --plan_sha256 "$SHA" --pilot \
    2>&1 | tee -a "$OUT/logs/pilot_$1_$2.log"
  return "${PIPESTATUS[0]}"
}

full() {
  check_md "$1" "$2"
  "$PY" ra29.py --run --model "$1" --dataset "$2" --out "$OUT/$1/$2/full" --plan_sha256 "$SHA" \
    --debug_gate "$OUT/$1/$2/pilot/debug_gate.json" 2>&1 | tee -a "$OUT/logs/full_$1_$2.log"
  return "${PIPESTATUS[0]}"
}

aqua26() {
  N="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["counts"]["n_kept"])' \
       "$OUT/ouro14/aqua/full/checks.json")"
  if [ "$N" -lt 80 ]; then
    echo "[ra29] 1.4B AQuA scored $N items after the gap drop (< 80): running 2.6B AQuA" | tee -a "$OUT/logs/mains.log"
    full ouro26 aqua
  else
    echo "[ra29] 1.4B AQuA scored $N items after the gap drop (>= 80): 2.6B AQuA not run" | tee -a "$OUT/logs/mains.log"
  fi
}

cell() {
  check_md "$1" "$2"
  [ -f "$OUT/$1/$2/pilot/debug_gate.json" ] || pilot "$1" "$2"
  P="$("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1]))["counts"]; print(int(round(c["n_kept"] / max(1, c["n_tried"]) * c["n_built"])))' \
       "$OUT/$1/$2/pilot/checks.json")"
  if [ "$P" -lt 80 ]; then
    echo "[ra29] cell $1 $2: projected $P scored items (< 80): full run not started" | tee -a "$OUT/logs/cells.log"
    return 0
  fi
  echo "[ra29] cell $1 $2: projected $P scored items (>= 80): full run" | tee -a "$OUT/logs/cells.log"
  full "$1" "$2"
  N="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["counts"]["n_kept"])' "$OUT/$1/$2/full/checks.json")"
  if [ "$N" -ge 80 ]; then V="meets"; else V="is below"; fi
  echo "[ra29] cell $1 $2: full run scored $N items ($V the 80 rule)" | tee -a "$OUT/logs/cells.log"
}

case "${1:-}" in
  test)
    echo "== tests under $("$PY" -c 'import sys,torch; print(sys.version.split()[0], torch.__version__)')" | tee -a "$OUT/logs/test.log"
    "$PY" -m unittest discover -s tests -v 2>&1 | tee -a "$OUT/logs/test.log"
    exit "${PIPESTATUS[0]}" ;;
  hash)
    "$PY" ra29.py --hash | tee "$OUT/logs/hash.log"
    echo "plan_ra29.sha256: $(cat plan_ra29.sha256)" | tee -a "$OUT/logs/hash.log" ;;
  pilot)
    pilot "${2:-}" "${3:-}" ;;
  pilots)
    pilot ouro14 aqua
    pilot ouro26 aqua
    pilot ouro14 csqa ;;
  full)
    full "${2:-}" "${3:-}" ;;
  aqua26)
    aqua26 ;;
  cell)
    cell "${2:-}" "${3:-}" ;;
  mains)
    full ouro14 aqua
    full ouro14 csqa
    [ -f "$OUT/ouro14/arc/pilot/debug_gate.json" ] || pilot ouro14 arc
    full ouro14 arc
    aqua26
    echo "[ra29] mains done" | tee -a "$OUT/logs/mains.log" ;;
  summarize)
    check_md "${2:-}" "${3:-}"
    W="${4:-full}"
    case "$W" in full|pilot) ;; *) echo "summarize MODEL DATASET pilot|full" >&2; exit 2;; esac
    "$PY" ra29.py --summarize --out "$OUT/$2/$3/$W" 2>&1 | tee "$OUT/logs/summarize_$2_$3_$W.log"
    exit "${PIPESTATUS[0]}" ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
