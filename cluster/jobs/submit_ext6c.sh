#!/bin/bash
# The B-EXT6c unit (stage 2; job list jobs/[nolocal/]joblist_ext6c.tsv from cluster/gen_ext6c.py), on the login node:
#   source <OUT_ROOT>/cluster.env && bash loopedcot/cluster/jobs/submit_ext6c.sh [--dry-run]
# It submits the groups of that list (g<N>_e6c...): every run group, then its merges (groups named *merge, afterany on
# every e6c job this script submits or finds in the queue). Nodes, packing and time as jobs/pack_lib.sh (the last
# NODELIST node stays empty, two task lines per GPU job, CARDS = used nodes x 8, --time=$TIME_LIMIT on every job).
# PLAN_IDS=<file> (one task id per line): only those ids are planned and priced (the planner's selection).
# Stage names: *_any waits afterany on the stage before it (a summary of shards); *_alone runs one line per GPU job.
# A task with a done marker is never submitted; a pipeline with a queued line is left alone. A task that depends on a
# task of another unit (for example the flags on g19_e6ra30, or a McLeish ASDiv line on the g18 select): with that
# task's done marker it waits for nothing; with that task's job still in the queue it waits afterok on that job (the
# job id from $OUT_ROOT/_chain/submitted.tsv); otherwise the pipeline is not submitted and named.
# Before it submits, it prints every job with its card-hours (single-card rate), the total, the nodes and the time
# limit, and asks one yes (EXT6C_YES=yes answers it; --dry-run prints and stops).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${REPO:-}" ] || [ -z "${PARTITION:-}" ]; then
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "$HERE/../env/cluster.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi
  done
fi
: "${OUT_ROOT:?source the cluster.env that setup_cluster.sh wrote}" "${PARTITION:?}" "${ACCOUNT:?}" "${LOG_ROOT:?}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
LISTS=$(sed -n 's/^lists //p' "$OUT_ROOT/_chain/jobs.txt" 2> /dev/null)
LISTS="${LISTS:-$HERE/nolocal}"
JL="$LISTS/joblist_ext6c.tsv"
[ -f "$JL" ] || { echo "no job list $JL (run cluster/gen_ext6c.py)"; exit 3; }
PLAN_IDS="${PLAN_IDS:-}"
[ -z "$PLAN_IDS" ] || [ -f "$PLAN_IDS" ] || { echo "PLAN_IDS=$PLAN_IDS: no such file"; exit 2; }
RUNG=$(awk -F'\t' 'NR > 1 && $2 !~ /merge/ && !s[$2]++ {print $2}' "$JL" | paste -sd' ' -)
MERGEG=$(awk -F'\t' 'NR > 1 && $2 ~ /merge/ && !s[$2]++ {print $2}' "$JL" | paste -sd' ' -)
[ -n "$RUNG" ] || { echo "no e6c run group in $JL"; exit 3; }
. "$HERE/pack_lib.sh"      # nodes (RESERVE_NODES), CARDS, packing, TIME_LIMIT
nodes_setup
SUB="$OUT_ROOT/_chain/submitted.tsv"
mkdir -p "$OUT_ROOT/_chain"
touch "$SUB" "$SUB.packed"
TAB=$(printf '\t')
unit_of() { case "$1" in *VENV_OURO*) echo ouro;; *VENV_RAVEN*) echo raven;; *VENV_PARCAE*) echo parcae;; *VENV_IFM*) echo ifm;; *) echo none;; esac; }

# queued lines of every unit ("<stage list> <line>", packed jobs included) and the queued e6c job ids
QUEUED=""; QJ=""
if command -v squeue > /dev/null 2>&1; then
  U="${USER:-$(id -un)}"
  Q=$(squeue -h -r -u "$U" -o "%F %K" 2> /dev/null || true)
  if [ -n "$Q" ]; then
    QUEUED=$(awk 'NR == FNR { q[$1 " " $2] = 1; next }
                  { n = split($3, a, ","); for (i = 1; i <= n; i++) if (($1 " " a[i]) in q) print $2 " " a[i] }' \
             <(echo "$Q") "$SUB" 2> /dev/null | sort -u)
    QUEUED=$(printf '%s\n%s\n' "$QUEUED" "$(queued_packed "$Q")" | grep . | sort -u)
  fi
  for g in $RUNG; do
    for A in $(squeue -h -u "$U" -n "ra20_$g" -o %F 2> /dev/null | sort -u); do
      awk -v a="$A" '$1 == a {f = 1} END {exit !f}' "$SUB" "$SUB.packed" 2> /dev/null \
        || { echo "REFUSED: queued job $A of ra20_$g is not in $SUB (scancel it or wait)"; exit 4; }
      QJ="$QJ:$A"
    done
  done
fi

# line index of every task in its stage list (array task i = line i), every stage list of $LISTS (external deps too)
LINES=$(mktemp)
trap 'rm -f "$LINES"' EXIT
(cd "$LISTS" && for L in g*.s*.tsv; do awk -F'\t' -v l="$L" '{print $1 "\t" l "\t" FNR}' "$L"; done) > "$LINES"
# the plan: one line per (pipeline, stage): group, script, model, out, stage, state, task indices ("-" for none),
# tasks to submit, tasks done, card-seconds, the external dependencies without a done marker ("-" for none)
PLAN=$(awk -F'\t' -v groups="$RUNG" -v done_dir="$OUT_ROOT/_done" -v queued="$QUEUED" -v pids="$PLAN_IDS" '
  BEGIN { n = split(groups, f, " "); for (i = 1; i <= n; i++) want[f[i]] = 1
          m = split(queued, ql, "\n"); for (i = 1; i <= m; i++) if (ql[i] != "") inq[ql[i]] = 1
          if (pids != "") { usep = 1; while ((getline x < pids) > 0) if (x != "") pid[x] = 1 } }
  FILENAME == ARGV[1] { lst[$1] = $2; line[$1] = $3; next }
  FNR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
  { g = $c["group"] }
  !(g in want) { next }
  usep && !($c["id"] in pid) { next }
  { id = $c["id"]; pl = g SUBSEP $c["script"] SUBSEP $c["model"] SUBSEP $c["out"]
    if (!(id in line)) { print "no stage-list line for " id > "/dev/stderr"; bad = 1; next }
    if (!(pl in seen)) { seen[pl] = 1; order[++np] = pl }
    k = pl SUBSEP $c["stage"]
    if (!(k in stseen)) { stseen[k] = 1; stages[pl] = stages[pl] " " $c["stage"] }
    member[id] = pl; deps[id] = $c["depends_on"]; ids[pl] = ids[pl] " " id
    if ((lst[id] " " line[id]) in inq) busy[pl] = busy[pl] " " id
    else if ((getline x < (done_dir "/" id)) >= 0) { close(done_dir "/" id); ndone[k]++ }
    else { todo[k] = todo[k] "," line[id]; secs[k] += $c["est_card_seconds"]; ntodo[k]++ } }
  END {
    for (i = 1; i <= np; i++) {
      pl = order[i]; split(pl, q, SUBSEP); ext = ""
      ni = split(ids[pl], il, " ")
      for (j = 1; j <= ni; j++) { nd = split(deps[il[j]], dl, ",")
        for (d = 1; d <= nd; d++) {
          if (dl[d] == "" || (dl[d] in member)) continue      # this or another e6c pipeline: chained by job ids
          if ((getline x < (done_dir "/" dl[d])) < 0) { if (index(ext " ", " " dl[d] " ") == 0) ext = ext " " dl[d] }
          else close(done_dir "/" dl[d]) } }
      ns = split(stages[pl], sl, " ")
      for (s = 1; s <= ns; s++) { k = pl SUBSEP sl[s]
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%.1f\t%s\n", q[1], q[2], q[3], q[4], sl[s], (busy[pl] != "" ? "queued" : "ok"),
               (todo[k] == "" ? "-" : substr(todo[k], 2)), ntodo[k], ndone[k], secs[k], (ext == "" ? "-" : substr(ext, 2)) } }
    exit bad }' "$LINES" "$JL") || { echo "the e6c job list and its stage lists disagree (run cluster/gen_ext6c.py)"; exit 3; }
[ -n "$PLAN" ] || { echo "nothing planned from $JL${PLAN_IDS:+ (PLAN_IDS=$PLAN_IDS)}"; exit 0; }

qjob() {             # the job id of task $1 while that task is in the queue (from submitted.tsv), else nothing
  local L
  L=$(awk -F'\t' -v id="$1" '$1 == id {print $2 " " $3}' "$LINES" | head -1)
  { [ -n "$L" ] && echo "$QUEUED" | grep -qxF "$L"; } || return 0
  awk -v l="${L% *}" -v n="${L#* }" '$2 == l { m = split($3, a, ","); for (i = 1; i <= m; i++) if (a[i] == n) j = $1 }
                                    END { print j }' "$SUB" "$SUB.packed"
}
# external dependencies: the job of each dependency still in the queue (afterok), else the pipeline is blocked
declare -A EXTJ=() BLOCK=()
while IFS="$TAB" read -r g script model out stage state idxs ntodo ndone secs ext; do
  [ "$ext" != "-" ] || continue
  pl="$g $script $model $out"
  [ -z "${EXTJ[$pl]+x}" ] || continue
  EXTJ[$pl]=""
  for d in $(echo "$ext" | tr ',' ' '); do
    jid=$(qjob "$d")
    if [ -n "$jid" ]; then EXTJ[$pl]="${EXTJ[$pl]}:$jid"; else BLOCK[$pl]="${BLOCK[$pl]:-} $d"; fi
  done
done <<< "$PLAN"

echo "B-EXT6c unit from $JL${PLAN_IDS:+ (PLAN_IDS=$PLAN_IDS)} (one line per job: group, script, model, folder, stage; card-hours at the single-card rate)"
TOT=0
while IFS="$TAB" read -r g script model out stage state idxs ntodo ndone secs ext; do
  pl="$g $script $model $out"; f="${out#\$OUT_ROOT/}"
  line="  $g $script $model $f $stage: "
  if [ "$state" = queued ]; then echo "${line}in the queue, left alone"
  elif [ -n "${BLOCK[$pl]:-}" ]; then echo "${line}NOT SUBMITTED: needs the done marker or the queued job of${BLOCK[$pl]}"
  elif [ "$ntodo" -eq 0 ]; then echo "${line}every task done ($ndone)"
  else
    printf "%s%d task(s), %.2f card-hours%s\n" "$line" "$ntodo" "$(awk -v s="$secs" 'BEGIN {print s / 3600}')" \
      "${EXTJ[$pl]:+ (waits afterok on ${EXTJ[$pl]#:})}"
    TOT=$(awk -v a="$TOT" -v b="$secs" 'BEGIN {print a + b}')
  fi
done <<< "$PLAN"
echo "  $MERGEG: the merges, CPU, about 0.1 card-hours each"
NTASK=$(while IFS="$TAB" read -r g script model out stage state idxs ntodo ndone secs ext; do
          pl="$g $script $model $out"; [ "$state" = ok ] && [ -z "${BLOCK[$pl]:-}" ] && echo "$ntodo"; done <<< "$PLAN" | awk '{n += $1} END {print n + 0}')
printf "total: %d task(s), %.1f card-hours (plus the merges)\n" "$NTASK" "$(awk -v s="$TOT" 'BEGIN {print s / 3600}')"
nodes_say
time_check
pack_report "$(echo "$PLAN" | awk -F'\t' '$6 == "ok" {print $1 "\t" $5 "\t" $7}')"
[ "$DRY" = 1 ] && { echo "dry run: nothing submitted"; exit 0; }
[ "$NTASK" -gt 0 ] || { echo "nothing to submit"; exit 0; }
ANS="${EXT6C_YES:-}"
if [ -z "$ANS" ]; then read -r -p "submit these $NTASK tasks and the merges? type yes: " ANS || ANS=""; fi
[ "$ANS" = yes ] || { echo "not submitted"; exit 1; }

# the restriction: cluster.env plus the bad nodes plus every node not used
BAD=$(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . | paste -sd, -)
R="${SLURM_RESTRICT:-}"
if [ -n "$BAD" ]; then case "$R" in *--exclude=*) R=$(echo "$R" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$BAD/");; *) R="$R --exclude=$BAD";; esac; fi
add_excl
cd "$LISTS" || exit 2
NEW=""; FAIL=0; N=0
prevpl=""; prev=""; skip_pl=""
declare -A JOB_OF=()                 # e6c task id -> its job id(s)
while IFS="$TAB" read -r g script model out stage state idxs ntodo ndone secs ext; do
  pl="$g $script $model $out"
  [ "$pl" = "$prevpl" ] || { prev="${EXTJ[$pl]:-}"; prev="${prev#:}"; prevpl="$pl"; }
  [ "$state" = ok ] && [ -z "${BLOCK[$pl]:-}" ] && [ "$pl" != "$skip_pl" ] || continue
  [ "$idxs" != "-" ] || continue
  # an e6c dependency outside this pipeline (another e6c group) that this run submitted: wait for its jobs too
  dep="$prev"
  for t in $(awk -F'\t' -v l="$g.$stage.tsv" -v ix=",$idxs," '$2 == l && index(ix, "," $3 ",") {print $1}' "$LINES"); do
    for d in $(awk -F'\t' -v id="$t" 'NR > 1 && $1 == id {print $9}' "$JL" | tr ',' ' '); do
      jd="${JOB_OF[$d]:-}"
      if [ -z "$jd" ] && [ ! -f "$OUT_ROOT/_done/$d" ]; then jd=$(qjob "$d"); fi      # an earlier run's job, still queued
      [ -n "$jd" ] && case ":$dep:" in *":$jd:"*) ;; *) dep="${dep:+$dep:}$jd";; esac
    done
  done
  cmd1=$(awk -F'\t' -v i="${idxs%%,*}" 'NR == i {print $2}' "$g.$stage.tsv")
  UNIT=$(unit_of "$cmd1")
  HOLD=""; [ -f "$OUT_ROOT/_held/$UNIT.held" ] && HOLD="--hold"
  if [ "${stage%_any}" != "$stage" ]; then            # a stage named *_any waits afterany (a summary of shards)
    # shellcheck disable=SC2086
    j=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" $R $HOLD --time="$TIME_LIMIT" \
          --output="$LOG_ROOT/ra20_${g}_%A_%a.out" ${dep:+--dependency=afterany:$dep} --array="$idxs%$CARDS" \
          "task_$g.sbatch" "$g.$stage.tsv") && echo "$j $g.$stage.tsv $idxs" >> "$SUB" \
      || { echo "sbatch failed: $pl $stage"; FAIL=1; skip_pl="$pl"; continue; }
  elif [ "${stage%_alone}" != "$stage" ]; then        # a stage named *_alone runs one line per GPU job (no packing)
    j=$(PACK=0 submit_stage "$g" "$stage" "$idxs" "$dep") && [ -n "$j" ] \
      || { echo "sbatch failed: $pl $stage (its later stages are not submitted)"; FAIL=1; skip_pl="$pl"; continue; }
  else
    j=$(submit_stage "$g" "$stage" "$idxs" "$dep") && [ -n "$j" ] \
      || { echo "sbatch failed: $pl $stage (its later stages are not submitted)"; FAIL=1; skip_pl="$pl"; continue; }
  fi
  [ -n "$HOLD" ] && echo "$j" | tr ':' '\n' >> "$OUT_ROOT/_held/$UNIT.jobs"
  for t in $(awk -F'\t' -v l="$g.$stage.tsv" -v ix=",$idxs," '$2 == l && index(ix, "," $3 ",") {print $1}' "$LINES"); do JOB_OF[$t]="$j"; done
  NEW="$NEW:$j"; N=$((N + 1))
  echo "  $g $model $(basename "$out") $stage: job $j, tasks $idxs${dep:+ (afterok $dep)}${HOLD:+ (held, unit $UNIT)}"
  prev="$j"
done <<< "$PLAN"

# the merges, afterany on every new and queued e6c job (never held)
D="$NEW$QJ"
for g in $MERGEG; do
  n=$(grep -c . "$g.s1_merge.tsv")
  cut -f1 "$g.s1_merge.tsv" | while read -r mid; do rm -f "$OUT_ROOT/_done/$mid"; done   # a merge reruns every round
  j=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" $R --time="$TIME_LIMIT" --output="$LOG_ROOT/ra20_${g}_%A_%a.out" \
        ${D:+--dependency=afterany$D} --array="1-$n" "task_$g.sbatch" "$g.s1_merge.tsv") \
    || { echo "sbatch failed: $g"; FAIL=1; continue; }
  echo "$j $g.s1_merge.tsv $(seq -s, 1 "$n")" >> "$SUB"
  echo "  $g: job $j${D:+ (afterany ${D#:})}"
done
echo "submitted: $N run jobs and the merges; status: bash $HERE/status.sh"
exit $FAIL
