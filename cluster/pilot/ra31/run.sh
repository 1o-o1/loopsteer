#!/usr/bin/env bash
# RA-31 Part A driver (plan_ra31.md). One process at a time on one card.
#
#   bash run.sh test                          CPU tests (no GPU, no model) under PY_OURO
#   bash run.sh hash                          plan, code and data sha256 (for the LEDGER line)
#   bash run.sh pilot [1.4b|2.6b|1.4b-think|2.6b-think]   pilot: the first 25 items of each dataset, every arm -> out/pilot_<m>
#   bash run.sh full [1.4b|2.6b]              full run -> out/full_<m> (needs a passing pilot; resumable).
#                                             DROP="attn:top,attn:rand" applies an RA prune decision.
#   bash run.sh summarize pilot|full [1.4b|2.6b]   results.csv, checks.json, summary.md (CPU)
#   bash run.sh partb_list                    Part B shards, arms and card-hours (CPU)
#   bash run.sh partb SHARD                   one Part B shard -> out/partb/<arm> (resumable, a DONE.json per arm)
#   bash run.sh partb_summarize               merge every Part B arm under out/partb (CPU)
#
# PY_OURO = the python of the Ouro venv RA-28 used (see RUNSHEET.md). Exit codes: 2 plan, pin, data or config
# mismatch, 3 gate failed or no fit, 4 identity failed on a scored row, 5 stopped by SIGTERM (rerun the command).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${HOME:-}" ]; then
  case "$HERE/" in "$HOME"/*) echo "refusing: the run folder is inside the home directory; copy it to /var/tmp/ra31" >&2; exit 2;; esac
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
OUT="$HERE/out"
mkdir -p "$OUT/logs"
cd "$HERE"
: "${PY_OURO:?set PY_OURO to the python of the Ouro venv (see RUNSHEET.md)}"
PY="$PY_OURO"
SHA="$(awk '{print $1}' "$HERE/plan_ra31.sha256")"
PILOT_N=25

model_arg() {
  case "${1:-1.4b}" in 1.4b|2.6b|1.4b-think|2.6b-think) echo "${1:-1.4b}" ;;
    *) echo "model must be 1.4b, 2.6b, 1.4b-think or 2.6b-think" >&2; exit 2 ;; esac
}

case "${1:-}" in
  test)
    echo "== tests under $("$PY" -c 'import sys,torch; print(sys.version.split()[0], torch.__version__)')" | tee -a "$OUT/logs/test.log"
    "$PY" -m unittest discover -s tests -v 2>&1 | tee -a "$OUT/logs/test.log"
    exit "${PIPESTATUS[0]}" ;;
  hash)
    "$PY" ra31.py --hash | tee "$OUT/logs/hash.log"
    echo "plan_ra31.sha256: $(cat plan_ra31.sha256)" | tee -a "$OUT/logs/hash.log" ;;
  pilot)
    M="$(model_arg "${2:-}")"
    "$PY" ra31.py --run --model "$M" --out "$OUT/pilot_$M" --plan_sha256 "$SHA" --pilot "$PILOT_N" \
      2>&1 | tee -a "$OUT/logs/pilot_$M.log"
    exit "${PIPESTATUS[0]}" ;;
  full)
    M="$(model_arg "${2:-}")"
    "$PY" ra31.py --run --model "$M" --out "$OUT/full_$M" --plan_sha256 "$SHA" \
      --pilot_gate "$OUT/pilot_$M/pilot_gate.json" --drop "${DROP:-}" 2>&1 | tee -a "$OUT/logs/full_$M.log"
    exit "${PIPESTATUS[0]}" ;;
  summarize)
    W="${2:-full}"
    case "$W" in full|pilot) ;; *) echo "summarize pilot|full [1.4b|2.6b]" >&2; exit 2;; esac
    M="$(model_arg "${3:-}")"
    "$PY" ra31.py --summarize --out "$OUT/${W}_$M" 2>&1 | tee "$OUT/logs/summarize_${W}_$M.log"
    exit "${PIPESTATUS[0]}" ;;
  partb_list)
    "$PY" ra31_gen.py list | tee "$OUT/logs/partb_list.log" ;;
  partb)
    : "${2:?partb needs a shard name (bash run.sh partb_list)}"
    "$PY" ra31_gen.py partb --shard "$2" --out "${PARTB_OUT:-$OUT/partb}" 2>&1 | tee -a "$OUT/logs/partb_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  partb_summarize)
    "$PY" ra31_gen.py summarize --out "${PARTB_OUT:-$OUT/partb}" 2>&1 | tee "$OUT/logs/partb_summarize.log"
    exit "${PIPESTATUS[0]}" ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
