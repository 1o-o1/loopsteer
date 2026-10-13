#!/bin/bash
# Tonight's round under the PI budget rule (2026-10-12: everything must finish within 10 h of its start; what does
# not fit is not run), on the login node, after the pull:
#   source <OUT_ROOT>/cluster.env && bash loopedcot/cluster/jobs/submit_tonight.sh [--dry-run] [--budget-h N] [--set NAME]
#   (or BUDGET_H=N and SET=NAME in the environment; unset: the planner's default budget and the full cell table)
# 1. Reads the queue (squeue -h -r -u $USER -o "%F|%K|%t|%j|%N|%M") and the free nodes (sinfo -h -N -n $NODELIST -o
#    "%N %t": idle or mix, NODELIST minus the reserve, the last node, minus bad nodes; FREE_NODES=<n> overrides sinfo).
# 2. jobs/plan_tonight.py plans the units EXT5 (submit_fill.sh), EXT6b (submit_ext6b.sh) and EXT6c (submit_ext6c.sh)
#    on the free GPUs next to the queued EXT6 chains (g19 to g23: running for their remaining time, pending after
#    their afterok, one GPU each, never dropped), drops whole cells in the order of its CELLS table until the estimated
#    wall time fits 9 h, refills dropped cells in reverse order while it still fits, and writes
#    $OUT_ROOT/_chain/plan_tonight.tsv and its .summary (the final KEPT and DROPPED lists). Exit 2 when the
#    never-dropped cells alone do not fit: nothing is submitted.
# 3. Each unit with kept lines goes to its own submit script with TIME_LIMIT=10:00:00 (every sbatch passes --time) and
#    the reserve in --exclude (jobs/pack_lib.sh add_excl; added to SLURM_RESTRICT for a script without it). A script
#    with the PLAN_IDS option gets the file of kept ids. A script without it gets the fallback the plan names
#    (submit_fill.sh: FILL_PRIORITIES=<the priorities kept whole>; the others: the whole unit when nothing of it was
#    dropped); what the fallback cannot express is not submitted and is named. The units go in the order EXT5, EXT6b,
#    EXT6c, so a cell that chains behind EXT5 lines (the EXT6c RA-30 McLeish ASDiv cell behind the priority-2 McLeish
#    ASDiv pool) finds their jobs in submitted.tsv; a unit whose kept lines chain behind lines that are not submitted
#    is held back and named. Every pilot-then-full chain goes out in one call with afterok (no second submit).
# 4. Every unit script runs with --dry-run first (it prints its jobs and card-hours), then the final KEPT and DROPPED
#    lists, the wall estimate and the longest chain print again. --dry-run stops there. Otherwise one yes
#    (TONIGHT_YES=yes answers it) submits every unit, each with its own yes variable set.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${REPO:-}" ] || [ -z "${PARTITION:-}" ]; then
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "$HERE/../env/cluster.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi
  done
fi
: "${OUT_ROOT:?source the cluster.env that setup_cluster.sh wrote}" "${PARTITION:?}" "${ACCOUNT:?}" "${LOG_ROOT:?}"
DRY=0
BUDGET_H="${BUDGET_H:-}"   # empty: the planner's default budget
SET="${SET:-}"             # empty: the full cell table; a name: that set of plan_tonight.py SETS (e.g. tonight4)
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1;;
    --budget-h) BUDGET_H="${2:?--budget-h needs hours}"; shift;;
    --set) SET="${2:?--set needs a name}"; shift;;
    *) echo "unknown argument $1 (use --dry-run, --budget-h N, --set NAME)"; exit 2;;
  esac
  shift
done
PLANARG=()
[ -z "$BUDGET_H" ] || PLANARG+=(--budget-h "$BUDGET_H")
[ -z "$SET" ] || PLANARG+=(--set "$SET")
LISTS=$(sed -n 's/^lists //p' "$OUT_ROOT/_chain/jobs.txt" 2> /dev/null)
LISTS="${LISTS:-$HERE/nolocal}"
[ -f "$LISTS/joblist.tsv" ] || { echo "no job list $LISTS/joblist.tsv"; exit 2; }
. "$HERE/pack_lib.sh"      # nodes_setup: USED (NODELIST minus the reserve and bad nodes), RESERVED; exit 2 without NODELIST
nodes_setup
TIME_LIMIT=10:00:00        # the PI rule, whatever the environment says
export TIME_LIMIT
PY="${PYTHON:-$(command -v python3 || command -v python || true)}"
[ -n "$PY" ] || { echo "no python3 on this node"; exit 2; }
mkdir -p "$OUT_ROOT/_chain"
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
TAB=$(printf '\t')

U="${USER:-$(id -un)}"
squeue -h -r -u "$U" -o "%F|%K|%t|%j|%N|%M" > "$TMPD/squeue" 2> "$TMPD/squeue.err" \
  || { echo "squeue failed ($(head -1 "$TMPD/squeue.err")): the EXT6 occupancy is unknown, nothing planned"; exit 2; }
if [ -n "${FREE_NODES:-}" ]; then
  NODEARG=(--free-nodes "$FREE_NODES")
else
  sinfo -h -N -n "$NODELIST" -o "%N %t" > "$TMPD/sinfo" 2> /dev/null && [ -s "$TMPD/sinfo" ] \
    || { echo "sinfo gave no node states for $NODELIST: set FREE_NODES=<n> (free nodes, the reserve not counted)"; exit 2; }
  NODEARG=(--sinfo "$TMPD/sinfo")
fi
PLAN="$OUT_ROOT/_chain/plan_tonight.tsv"
"$PY" "$HERE/plan_tonight.py" --lists "$LISTS" --out-root "$OUT_ROOT" --squeue "$TMPD/squeue" --used "$USED" \
  --reserved "$RESERVED" "${NODEARG[@]}" ${PLANARG[@]+"${PLANARG[@]}"} --plan "$PLAN"
rc=$?
[ "$rc" = 0 ] || { echo "plan_tonight.py exited $rc: nothing submitted"; exit "$rc"; }

# the units to submit: script, yes variable, the environment that selects the kept lines
RUN_S=(); RUN_Y=(); RUN_E=(); RUN_U=()
LOSSY=" "                  # units whose kept lines are not all submitted (a unit that depends on one is held back)
while IFS="$TAB" read -r kind unit script yes n ids fb left needs; do
  [ "$kind" = unit ] && [ "${n:-0}" -gt 0 ] || continue
  for u in ${needs//,/ }; do
    [ "$u" != "-" ] && case "$LOSSY" in *" $u "*) LOSSY="$LOSSY$unit "
      echo "unit $unit: its kept lines chain behind lines of $u that are not submitted: NOT SUBMITTED"; continue 2;; esac
  done
  if [ ! -f "$HERE/$script" ]; then
    echo "unit $unit: no jobs/$script, its $n kept lines are not submitted"; LOSSY="$LOSSY$unit "; continue
  fi
  if grep -q 'PLAN_IDS' "$HERE/$script"; then
    sel="PLAN_IDS=$ids"
  elif [ "$fb" != "-" ]; then
    sel=""; [ "$fb" = ALL ] || sel="$fb"
    msg=""; [ "$left" = "-" ] || { msg="; NOT SUBMITTED: $left"; LOSSY="$LOSSY$unit "; }
    echo "unit $unit: jobs/$script has no PLAN_IDS option; submitted with ${sel:-the whole unit}$msg"
  else
    echo "unit $unit: jobs/$script has no PLAN_IDS option and the plan keeps only part of the unit ($left): NOT SUBMITTED"
    LOSSY="$LOSSY$unit "; continue
  fi
  R2="${SLURM_RESTRICT:-}"
  if ! grep -q 'add_excl' "$HERE/$script"; then           # a script without pack_lib's exclusion: the reserve by hand
    case "$R2" in *--exclude=*) R2=$(echo "$R2" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$RESERVED/");; *) R2="$R2 --exclude=$RESERVED";; esac
  fi
  RUN_U+=("$unit"); RUN_S+=("$script"); RUN_Y+=("$yes"); RUN_E+=("$sel${TAB}$R2")
done < "$PLAN"
[ "${#RUN_S[@]}" -gt 0 ] || { echo "nothing to submit"; exit 0; }

run_unit() {        # $1 index, $2 mode (dry | real)
  local i="$1" sel r2 pre=() args=()
  sel="${RUN_E[$i]%%"$TAB"*}"; r2="${RUN_E[$i]#*"$TAB"}"
  [ -n "$sel" ] && pre+=("$sel")
  if [ "$2" = dry ]; then args=(--dry-run); else pre+=("${RUN_Y[$i]}=yes"); fi
  env -u FILL_PRIORITIES -u PLAN_IDS -u FILL_YES -u EXT6B_YES -u EXT6C_YES TIME_LIMIT="$TIME_LIMIT" SLURM_RESTRICT="$r2" \
      ${pre[@]+"${pre[@]}"} bash "$HERE/${RUN_S[$i]}" ${args[@]+"${args[@]}"}
}

for i in "${!RUN_S[@]}"; do
  sel="${RUN_E[$i]%%"$TAB"*}"
  echo "=== unit ${RUN_U[$i]}: jobs/${RUN_S[$i]} ${sel:-(whole unit)} TIME_LIMIT=$TIME_LIMIT (dry run)"
  run_unit "$i" dry; rc=$?
  [ "$rc" = 0 ] || { echo "unit ${RUN_U[$i]}: the dry run of jobs/${RUN_S[$i]} exited $rc: nothing submitted"; exit 3; }
done
echo "=== tonight's plan (after the drops and the refill)"
cat "$PLAN.summary"
[ "$DRY" = 1 ] && { echo "dry run: nothing submitted"; exit 0; }
ANS="${TONIGHT_YES:-}"
if [ -z "$ANS" ]; then read -r -p "submit the units above (${RUN_U[*]})? type yes: " ANS || ANS=""; fi
[ "$ANS" = yes ] || { echo "not submitted"; exit 1; }
FAIL=0
for i in "${!RUN_S[@]}"; do
  echo "=== unit ${RUN_U[$i]}: jobs/${RUN_S[$i]}"
  run_unit "$i" real || { echo "unit ${RUN_U[$i]}: jobs/${RUN_S[$i]} exited $?"; FAIL=1; }
done
echo "plan: $PLAN; status: bash $HERE/status.sh"
exit $FAIL
