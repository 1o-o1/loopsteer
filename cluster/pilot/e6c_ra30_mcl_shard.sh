#!/bin/bash
# B-EXT6c: one shard of the RA-30 McLeish r32 GSM8K full run (plan 086581bf section 6c), from pilot/ra30_s2.
#   bash cluster/pilot/e6c_ra30_mcl_shard.sh S OUT PLAN_SHA256 NMAX
# The shard count N comes from the McLeish pilot (OUT/mcleish_llama32_r32/pilot/gsm8k/checks.json, field
# shards_rule.shards: 3 if the projected full run is at most 16.2 card-hours, else ceil(projected / 5.4)). The job
# list holds NMAX shard lines (0 .. NMAX-1); a line with S >= N exits 0 without work; N > NMAX stops shard 0 with
# exit 3 (more shards are needed than were submitted).
set -euo pipefail
S="${1:?shard}"; O="${2:?output folder}"; SHA="${3:?plan sha256}"; NMAX="${4:?shard lines}"
cd "$(dirname "$0")/ra30_s2"
PY="${PY:-python}"
C="$O/mcleish_llama32_r32/pilot/gsm8k/checks.json"
[ -f "$C" ] || { echo "no McLeish pilot checks.json ($C): run the pilot first" >&2; exit 2; }
N="$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["shards_rule"]["shards"]))' "$C")"
if [ "$N" -gt "$NMAX" ]; then
  echo "[ra30] the pilot asks for $N shards, $NMAX shard lines were submitted" >&2
  [ "$S" -eq 0 ] && exit 3
  exit 0
fi
if [ "$S" -ge "$N" ]; then echo "[ra30] shard $S of $N: not needed"; exit 0; fi
[ -f "$O/mcleish_llama32_r32/tokens/gsm8k_check.json" ] || { echo "no McLeish tokens step" >&2; exit 2; }
"$PY" -u ra30_mcl.py --run --stage full --shard "$S" --shards "$N" --out "$O" --plan_sha256 "$SHA"
