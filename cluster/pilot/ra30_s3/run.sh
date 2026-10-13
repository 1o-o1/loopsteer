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
#   bash run.sh tokens_mcl              McLeish r32 GSM8K (plan 6c), host tokenizer on the CPU (no card)
#   bash run.sh pilot_mcl               McLeish pilot, 20 scored items -> out/mcleish_llama32_r32/pilot/gsm8k
#   bash run.sh full_mcl S [N]          McLeish full run, item shard S of N (default 3; plan 6c rule), needs the pilot
#                                       -> out/mcleish_llama32_r32/full/gsm8k/shard_<S>_of_<N>
#   bash run.sh summarize_mcl pilot|full     (full: merges the shard folders)
#   bash run.sh think_tokens M D          Thinking cells (plan 6e), M = 1p4|2p6, D = gsm8k|asdiv; host tokens (CPU)
#   bash run.sh think_pilot M D screen | rows P N | merge   pilot (GSM8K 30, ASDiv 20 eligible items): the screen
#                                         job, then N parallel rows jobs P = 0..N-1 (afterok), then the CPU merge (G6)
#   bash run.sh think_full M D S N [PHASE] full shard S of N (needs the pilot: plan, ceiling >= 0.5, pool sha256)
#   bash run.sh think_summarize M STAGE D  (full: merges the shard folders)
#                                         the Thinking steps need OUT_ROOT (the pools) and K17_THINK_DIR (checkpoint)
#   bash run.sh mcl_asdiv_tokens|mcl_asdiv_pilot     McLeish ASDiv (plan 6f), pool from OUT_ROOT (EXT5 select)
#   bash run.sh mcl_asdiv_full S N | mcl_asdiv_summarize pilot|full
#   bash run.sh flags pilot|full gsm8k|asdiv    operand flag at summary time, Ouro-1.4B (plan 6d; CPU only)
#   bash run.sh flags2 pilot|full gsm8k|asdiv   the same for Ouro-2.6B
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
  tokens_mcl)
    CUDA_VISIBLE_DEVICES="" PROD_DEV=cpu "$PY" ra30_mcl.py --tokens --out "$OUT" 2>&1 | tee "$OUT/logs/tokens_mcl.log"
    exit "${PIPESTATUS[0]}" ;;
  pilot_mcl)
    [ -f "$OUT/mcleish_llama32_r32/tokens/gsm8k_check.json" ] || { echo "run 'bash run.sh tokens_mcl' first" >&2; exit 2; }
    "$PY" ra30_mcl.py --run --stage pilot --out "$OUT" --plan_sha256 "$SHA" 2>&1 | tee -a "$OUT/logs/pilot_mcl.log"
    exit "${PIPESTATUS[0]}" ;;
  full_mcl)
    NSH="${3:-3}"
    case "${2:-}" in ''|*[!0-9]*) echo "full_mcl S [N]: shard S of N (default 3)" >&2; exit 2;; esac
    case "$NSH" in ''|*[!0-9]*) echo "full_mcl S [N]: N must be a number" >&2; exit 2;; esac
    [ "$2" -lt "$NSH" ] || { echo "shard $2 is not below $NSH" >&2; exit 2; }
    [ -f "$OUT/mcleish_llama32_r32/tokens/gsm8k_check.json" ] || { echo "run 'bash run.sh tokens_mcl' first" >&2; exit 2; }
    "$PY" ra30_mcl.py --run --stage full --shard "$2" --shards "$NSH" --out "$OUT" --plan_sha256 "$SHA" 2>&1       | tee -a "$OUT/logs/full_mcl_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  summarize_mcl)
    case "${2:-}" in pilot|full) ;; *) echo "summarize_mcl pilot|full" >&2; exit 2;; esac
    "$PY" ra30_mcl.py --summarize --out "$OUT/mcleish_llama32_r32/$2/gsm8k" 2>&1 | tee "$OUT/logs/summarize_mcl_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  flags|flags2)
    case "${2:-}" in pilot|full) ;; *) echo "$1 pilot|full gsm8k|asdiv" >&2; exit 2;; esac
    ds_ok "${3:-}"
    D="$OUT/$2/$3"; [ "$1" = flags2 ] && D="$OUT/ouro_2_6b_base/$2/$3"
    CUDA_VISIBLE_DEVICES="" PROD_DEV=cpu "$PY" ra30_flags.py --out "$D" 2>&1 | tee "$OUT/logs/${1}_$2_$3.log"
    exit "${PIPESTATUS[0]}" ;;
  think_tokens|think_pilot|think_full|think_summarize)
    case "${2:-}" in 1p4) M=ouro_1_4b_think;; 2p6) M=ouro_2_6b_think;; *) echo "model 1p4|2p6" >&2; exit 2;; esac
    : "${K17_THINK_DIR:?set K17_THINK_DIR to the Thinking checkpoint folder}"
    case "$1" in
      think_tokens)
        ds_ok "${3:-}"; : "${OUT_ROOT:?set OUT_ROOT (the pools)}"
        CUDA_VISIBLE_DEVICES="" PROD_DEV=cpu "$PY" ra30_think.py --tokens --model "$M" --ds "$3" --pool_root "$OUT_ROOT"           --out "$OUT" 2>&1 | tee "$OUT/logs/think_tokens_${M}_$3.log"
        exit "${PIPESTATUS[0]}" ;;
      think_pilot)
        ds_ok "${3:-}"; : "${OUT_ROOT:?set OUT_ROOT (the pools)}"
        [ -f "$OUT/$M/tokens/$3_check.json" ] || { echo "run think_tokens first" >&2; exit 2; }
        PH="${4:-all}"; EXTRA=""; ENVX=""
        case "$PH" in
          all|screen) ;;
          rows) case "${5:-}${6:-}" in ''|*[!0-9]*) echo "think_pilot M D rows P N" >&2; exit 2;; esac
                [ "$5" -lt "$6" ] || { echo "rows part $5 is not below $6" >&2; exit 2; }
                EXTRA="--rows_part $5 --rows_parts $6" ;;
          merge) ENVX="cpu" ;;
          *) echo "think_pilot phase all|screen|rows P N|merge" >&2; exit 2 ;;
        esac
        if [ "$ENVX" = cpu ]; then export CUDA_VISIBLE_DEVICES="" PROD_DEV=cpu; fi
        "$PY" ra30_think.py --run --model "$M" --ds "$3" --stage pilot --phase "$PH" $EXTRA --pool_root "$OUT_ROOT"           --out "$OUT" --plan_sha256 "$SHA" 2>&1 | tee -a "$OUT/logs/think_pilot_${M}_$3_${PH}${5:+_$5}.log"
        exit "${PIPESTATUS[0]}" ;;
      think_full)
        ds_ok "${3:-}"; : "${OUT_ROOT:?set OUT_ROOT (the pools)}"
        case "${4:-}${5:-}" in ''|*[!0-9]*) echo "think_full M D S N [PHASE]" >&2; exit 2;; esac
        [ "$4" -lt "$5" ] || { echo "shard $4 is not below $5" >&2; exit 2; }
        "$PY" ra30_think.py --run --model "$M" --ds "$3" --stage full --shard "$4" --shards "$5" --phase "${6:-all}"           --pool_root "$OUT_ROOT" --out "$OUT" --plan_sha256 "$SHA" 2>&1 | tee -a "$OUT/logs/think_full_${M}_$3_$4.log"
        exit "${PIPESTATUS[0]}" ;;
      think_summarize)
        case "${3:-}" in pilot|full) ;; *) echo "think_summarize M pilot|full D" >&2; exit 2;; esac
        ds_ok "${4:-}"
        "$PY" ra30_think.py --summarize --out "$OUT/$M/$3/$4" 2>&1 | tee "$OUT/logs/think_summarize_${M}_$3_$4.log"
        exit "${PIPESTATUS[0]}" ;;
    esac ;;
  mcl_asdiv_tokens)
    : "${OUT_ROOT:?set OUT_ROOT (the McLeish ASDiv pool)}"
    CUDA_VISIBLE_DEVICES="" PROD_DEV=cpu "$PY" ra30_mcl.py --tokens --ds asdiv --pool_root "$OUT_ROOT" --out "$OUT" 2>&1       | tee "$OUT/logs/mcl_asdiv_tokens.log"
    exit "${PIPESTATUS[0]}" ;;
  mcl_asdiv_pilot)
    : "${OUT_ROOT:?set OUT_ROOT (the McLeish ASDiv pool)}"
    [ -f "$OUT/mcleish_llama32_r32/tokens/asdiv_check.json" ] || { echo "run mcl_asdiv_tokens first" >&2; exit 2; }
    "$PY" ra30_mcl.py --run --ds asdiv --stage pilot --pool_root "$OUT_ROOT" --out "$OUT" --plan_sha256 "$SHA" 2>&1       | tee -a "$OUT/logs/mcl_asdiv_pilot.log"
    exit "${PIPESTATUS[0]}" ;;
  mcl_asdiv_full)
    : "${OUT_ROOT:?set OUT_ROOT (the McLeish ASDiv pool)}"
    case "${2:-}${3:-}" in ''|*[!0-9]*) echo "mcl_asdiv_full S N" >&2; exit 2;; esac
    [ "$2" -lt "$3" ] || { echo "shard $2 is not below $3" >&2; exit 2; }
    "$PY" ra30_mcl.py --run --ds asdiv --stage full --shard "$2" --shards "$3" --pool_root "$OUT_ROOT" --out "$OUT"       --plan_sha256 "$SHA" 2>&1 | tee -a "$OUT/logs/mcl_asdiv_full_$2.log"
    exit "${PIPESTATUS[0]}" ;;
  mcl_asdiv_summarize)
    case "${2:-}" in pilot|full) ;; *) echo "mcl_asdiv_summarize pilot|full" >&2; exit 2;; esac
    "$PY" ra30_mcl.py --summarize --ds asdiv --out "$OUT/mcleish_llama32_r32/$2/asdiv" 2>&1       | tee "$OUT/logs/mcl_asdiv_summarize_$2.log"
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
  *) sed -n '2,33p' "$0"; exit 2 ;;
esac
