#!/usr/bin/env bash
# RA-30 driver (plan_ra30.md). One process at a time on one card. Card order: GSM8K, then ASDiv.
#
#   bash run.sh test                    CPU tests (no GPU, no model) under PY_OURO
#   bash run.sh hash                    plan, code and data sha256 (for the LEDGER line)
#   bash run.sh tokens gsm8k|asdiv      host tokenizer: rebuild every pool prompt, out/tokens (CPU)
#   bash run.sh pilot gsm8k|asdiv       pilot: 25 (GSM8K) or 20 (ASDiv) scored eligible items, gates, rows, G6
#   bash run.sh full gsm8k|asdiv        full run -> out/full/<ds> (resumable; needs an accepted pilot with >= 80
#                                       projected eligible items, else "ASDiv under 80: gate failure", exit 3)
#   bash run.sh summarize pilot|full gsm8k|asdiv   results.csv, results.json, checks.json, summary.md (CPU)
#   bash run.sh mains                   full gsm8k, tokens asdiv, pilot asdiv, full asdiv; stops at the first failure
#   bash run.sh tokens2 gsm8k|asdiv     Ouro-2.6B axis (plan 6b), host tokenizer on the CPU (no card) -> out/ouro_2_6b_base/tokens
#   bash run.sh pilot2 gsm8k|asdiv      Ouro-2.6B pilot, 20 scored items -> out/ouro_2_6b_base/pilot/<ds>
#   bash run.sh full2 gsm8k|asdiv       Ouro-2.6B full run (needs the 2.6B pilot; 80-item gate) -> out/ouro_2_6b_base/full/<ds>
#   bash run.sh summarize2 pilot|full gsm8k|asdiv
#
# PY_OURO = the python of the Ouro venv (transformers 4.56.2; see RUNSHEET.md). Exit codes: 2 plan, pin, data,
# host-token or input mismatch, 3 gate failed, 4 identity failed on a scored row, 5 stopped by SIGTERM (rerun).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${HOME:-}" ]; then
  case "$HERE/" in "$HOME"/*) echo "refusing: the run folder is inside the home directory; copy it to /var/tmp/ra30" >&2; exit 2;; esac
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
OUT="$HERE/out"
mkdir -p "$OUT/logs"
cd "$HERE"
: "${PY_OURO:?set PY_OURO to the python of the Ouro venv (see RUNSHEET.md)}"
PY="$PY_OURO"
SHA="$(awk '{print $1}' "$HERE/plan_ra30.sha256")"
ds_ok() { case "${1:-}" in gsm8k|asdiv) ;; *) echo "dataset must be gsm8k or asdiv" >&2; exit 2;; esac; }

case "${1:-}" in
  test)
    echo "== tests under $("$PY" -c 'import sys,torch,transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__)')" | tee -a "$OUT/logs/test.log"
    PROD_DEV=cpu "$PY" -m unittest discover -s tests -v 2>&1 | tee -a "$OUT/logs/test.log"
    exit "${PIPESTATUS[0]}" ;;
  hash)
    "$PY" ra30.py --hash | tee "$OUT/logs/hash.log"
    echo "plan_ra30.sha256: $(cat plan_ra30.sha256)" | tee -a "$OUT/logs/hash.log" ;;
  tokens)
    ds_ok "${2:-}"
    PROD_DEV=cpu "$PY" ra30.py --tokens --ds "$2" --out "$OUT" 2>&1 | tee "$OUT/logs/tokens_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  pilot|full)
    ds_ok "${2:-}"
    [ -f "$OUT/tokens/$2_check.json" ] || { echo "run 'bash run.sh tokens $2' first" >&2; exit 2; }
    "$PY" ra30.py --run --stage "$1" --ds "$2" --out "$OUT" --plan_sha256 "$SHA" 2>&1 | tee -a "$OUT/logs/$1_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  summarize)
    case "${2:-}" in pilot|full) ;; *) echo "summarize pilot|full gsm8k|asdiv" >&2; exit 2;; esac
    ds_ok "${3:-}"
    "$PY" ra30.py --summarize --out "$OUT/$2/$3" 2>&1 | tee "$OUT/logs/summarize_$2_$3.log"
    exit "${PIPESTATUS[0]}" ;;
  tokens2)
    ds_ok "${2:-}"
    CUDA_VISIBLE_DEVICES="" PROD_DEV=cpu "$PY" ra30.py --tokens --model ouro_2_6b_base --ds "$2" --out "$OUT" 2>&1 | tee "$OUT/logs/tokens2_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  pilot2|full2)
    ds_ok "${2:-}"
    [ -f "$OUT/ouro_2_6b_base/tokens/$2_check.json" ] || { echo "run 'bash run.sh tokens2 $2' first" >&2; exit 2; }
    "$PY" ra30.py --run --model ouro_2_6b_base --stage "${1%2}" --ds "$2" --out "$OUT" --plan_sha256 "$SHA" 2>&1 | tee -a "$OUT/logs/${1}_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  summarize2)
    case "${2:-}" in pilot|full) ;; *) echo "summarize2 pilot|full gsm8k|asdiv" >&2; exit 2;; esac
    ds_ok "${3:-}"
    "$PY" ra30.py --summarize --model ouro_2_6b_base --out "$OUT/ouro_2_6b_base/$2/$3" 2>&1 | tee "$OUT/logs/summarize2_$2_$3.log"
    exit "${PIPESTATUS[0]}" ;;
  mains)
    for st in "full gsm8k" "tokens asdiv" "pilot asdiv" "full asdiv"; do
      echo "== mains: $st" | tee -a "$OUT/logs/mains.log"
      set +e
      bash "$0" $st
      rc=$?
      set -e
      echo "== mains: $st exit $rc" | tee -a "$OUT/logs/mains.log"
      if [ "$rc" -ne 0 ]; then echo "mains stopped at: $st (exit $rc)" | tee -a "$OUT/logs/mains.log"; exit "$rc"; fi
    done ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
