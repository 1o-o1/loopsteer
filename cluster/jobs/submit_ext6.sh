#!/bin/bash
# The B-EXT6 unit (RA-29, RA-30, RA-31 Part A overflow on Ouro-2.6B base, RA-31 Part B on Ouro-1.4B base), on the
# login node, after the pull:
#   source <OUT_ROOT>/cluster.env && bash loopedcot/cluster/jobs/submit_ext6.sh [--dry-run]
# It submits only these groups:
#   g19_e6ra30   RA-30 GSM8K: s1_pilot (CPU tests, tokens2, pilot2), then s2_full (full2), afterok
#   g20_e6ra29   RA-29 AQuA and CSQA: per dataset s1_pilot (CPU tests, 2.6B pilot), then s2_full, afterok
#   g21_e6ra31a  RA-31 Part A: s1_pilot (CPU tests, 2.6B pilot), then s2_full, afterok
#   g22_e6ra31b  RA-31 Part B: s1_shard (15 shards, one array task each, CPU tests first), then s2_summarize, afterok
#   g23_e6merge  one merge per test (merge_ext6.py: $OUT_ROOT/E6/<test> -> $OUT_ROOT/merged/<test>), afterany on
#                every job of its test that this script submits or finds in the queue; it reruns every round
# A task with a done marker ($OUT_ROOT/_done/<id>) is never submitted; a pipeline (group, script, model, output
# folder) whose stage is still in the queue (matched through $OUT_ROOT/_chain/submitted.tsv) is left alone. A held
# unit ($OUT_ROOT/_held/<unit>.held) is submitted held again. The job list is the one submit_chain.sh used
# ($OUT_ROOT/_chain/jobs.txt, else jobs/nolocal). Before it submits, it prints every job with its card-hours
# (est_card_seconds of the job list: the single-card rates of the plans, A100 factor 1.0) and the total, and
# asks one yes (EXT6_YES=yes answers it; --dry-run prints and stops).
# One task line per GPU job (PACK=0 by default; packing is not part of this unit). Every sbatch call passes
# --time=$TIME_LIMIT (default 10:00:00), which overrides the template header; the partition's MaxTime is printed when
# sinfo gives it, and a shorter one stops the script (exit 2). A node kept free is excluded by hand through
# SLURM_RESTRICT (cluster.env), for example SLURM_RESTRICT="--exclude=<node>"; the script adds no node logic.
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
RUNG="g19_e6ra30 g20_e6ra29 g21_e6ra31a g22_e6ra31b"
MERGEG="g23_e6merge"
LISTS=$(sed -n 's/^lists //p' "$OUT_ROOT/_chain/jobs.txt" 2> /dev/null)
LISTS="${LISTS:-$HERE/nolocal}"
JL="$LISTS/joblist.tsv"
[ -f "$JL" ] || { echo "no job list $JL"; exit 2; }
for g in $RUNG $MERGEG; do
  awk -F'\t' -v g="$g" '$2 == g {f = 1} END {exit !f}' "$JL" || { echo "no $g task in $JL (regenerate the lists with gen_jobs.py)"; exit 3; }
done
PACK=${PACK:-0}            # one task line per GPU job
CARDS=${CARDS:-48}
EXCL=""                    # no node logic here: SLURM_RESTRICT carries any exclusion
. "$HERE/pack_lib.sh"      # TIME_LIMIT, time_check, submit_stage (PACK=0: every line runs alone, --time on each call)
SUB="$OUT_ROOT/_chain/submitted.tsv"
mkdir -p "$OUT_ROOT/_chain"
touch "$SUB"
TAB=$(printf '\t')
test_of() { case "$1" in g19_e6ra30) echo ra30;; g20_e6ra29) echo ra29;; g21_e6ra31a|g22_e6ra31b) echo ra31;; *) echo none;; esac; }
unit_of() { case "$1" in *VENV_OURO*) echo ouro;; *VENV_RAVEN*) echo raven;; *VENV_PARCAE*) echo parcae;; *VENV_IFM*) echo ifm;; *) echo none;; esac; }

# queued tasks of ours: "<stage list> <line index>", and the queued job ids per test (the merges wait for them)
QUEUED=""
declare -A QJOBS=()
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
      awk -v a="$A" '$1 == a {f = 1} END {exit !f}' "$SUB" 2> /dev/null \
        || { echo "REFUSED: queued job $A of ra20_$g is not in $SUB (scancel it or wait)"; exit 4; }
      t=$(test_of "$g"); QJOBS[$t]="${QJOBS[$t]:-}:$A"
    done
  done
fi

# line index of every task in its stage list (array task i = line i)
LINES=$(mktemp)
trap 'rm -f "$LINES"' EXIT
(cd "$LISTS" && for g in $RUNG $MERGEG; do
   for L in "$g".s*.tsv; do [ -f "$L" ] && awk -F'\t' -v l="$L" '{print $1 "\t" l "\t" FNR}' "$L"; done
 done) > "$LINES"
# the plan: one line per (group, pipeline, stage) of the run groups: state, task indices to submit ("-" for none),
# tasks to submit, tasks done, card-seconds
PLAN=$(awk -F'\t' -v groups="$RUNG" -v done_dir="$OUT_ROOT/_done" -v queued="$QUEUED" '
  BEGIN { n = split(groups, f, " "); for (i = 1; i <= n; i++) want[f[i]] = 1
          m = split(queued, ql, "\n"); for (i = 1; i <= m; i++) if (ql[i] != "") inq[ql[i]] = 1 }
  FILENAME == ARGV[1] { lst[$1] = $2; line[$1] = $3; next }
  FNR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
  { g = $c["group"] }
  !(g in want) { next }
  { id = $c["id"]; pl = g SUBSEP $c["script"] SUBSEP $c["model"] SUBSEP $c["out"]
    if (!(id in line)) { print "no stage-list line for " id > "/dev/stderr"; bad = 1; next }
    if (!(pl in seen)) { seen[pl] = 1; order[++np] = pl }
    k = pl SUBSEP $c["stage"]
    if (!(k in stseen)) { stseen[k] = 1; stages[pl] = stages[pl] " " $c["stage"] }
    if ((lst[id] " " line[id]) in inq) busy[pl] = busy[pl] " " id
    else if ((getline x < (done_dir "/" id)) >= 0) { close(done_dir "/" id); ndone[k]++ }
    else { todo[k] = todo[k] "," line[id]; secs[k] += $c["est_card_seconds"]; ntodo[k]++
           if ($c["shard"] != "") det[k] = det[k] " " $c["shard"] "=" sprintf("%.2f", $c["est_card_seconds"] / 3600) } }
  END {
    for (i = 1; i <= np; i++) {
      pl = order[i]; split(pl, q, SUBSEP)
      state = busy[pl] != "" ? "queued" : "ok"
      ns = split(stages[pl], sl, " ")
      for (s = 1; s <= ns; s++) { k = pl SUBSEP sl[s]
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%.1f\t%s\t%s\n", q[1], q[2], q[3], q[4], sl[s], state,
               (todo[k] == "" ? "-" : substr(todo[k], 2)), ntodo[k], ndone[k], secs[k], (busy[pl] == "" ? "-" : busy[pl]),
               (det[k] == "" ? "-" : substr(det[k], 2)) } }
    exit bad }' "$LINES" "$JL") || { echo "the job list and the stage lists of $LISTS disagree (regenerate them)"; exit 3; }
[ -n "$PLAN" ] || { echo "no B-EXT6 group in $JL (regenerate the lists with gen_jobs.py)"; exit 3; }

echo "B-EXT6 unit from $LISTS (one line per job: group, script, model, folder, stage; card-hours at the single-card rate, A100 factor 1.0)"
echo "$PLAN" | awk -F'\t' '
  { m = split($4, fo, "/"); f = fo[m]; if (fo[m - 1] == "ouro26") f = "ouro26/" f
    line = sprintf("  %s %s %s %s %s: ", $1, $2, $3, f, $5)
    if ($6 == "queued") print line "in the queue, left alone (" $11 ")"
    else if ($8 == 0) print line "every task done (" $9 ")"
    else { printf "%s%d task(s), %.2f card-hours\n", line, $8, $10 / 3600; tot += $10; nt += $8
           if ($12 != "-") { n = split($12, sh, " ")
             for (i = 1; i <= n; i++) { split(sh[i], kv, "="); printf "      shard %s: %s card-hours\n", kv[1], kv[2] } } } }
  END { printf "  %s: one merge per test (ra29, ra30, ra31), CPU, about 0.1 card-hours each\n", "g23_e6merge"
        printf "total: %d task(s), %.1f card-hours (plus the merges)\n", nt, tot / 3600 }'
NTASK=$(echo "$PLAN" | awk -F'\t' '$6 == "ok" {n += $8} END {print n + 0}')
echo "one task line per GPU job; at most $CARDS tasks of one array at once (CARDS); node restriction: ${SLURM_RESTRICT:-none}"
time_check
[ "$DRY" = 1 ] && { echo "dry run: nothing submitted"; exit 0; }
ANS="${EXT6_YES:-}"
if [ -z "$ANS" ]; then read -r -p "submit these $NTASK tasks and the three merges? type yes: " ANS || ANS=""; fi
[ "$ANS" = yes ] || { echo "not submitted"; exit 1; }

# the restriction: cluster.env (with the probe's choice) plus the bad nodes
BAD=$(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . | paste -sd, -)
R="${SLURM_RESTRICT:-}"
if [ -n "$BAD" ]; then case "$R" in *--exclude=*) R=$(echo "$R" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$BAD/");; *) R="$R --exclude=$BAD";; esac; fi
cd "$LISTS" || exit 2
declare -A NJOBS=()
FAIL=0; N=0
prevpl=""; prev=""; skip_pl=""
while IFS="$TAB" read -r g script model out stage state idxs ntodo ndone secs why det; do
  pl="$g $script $model $out"
  [ "$pl" = "$prevpl" ] || { prev=""; prevpl="$pl"; }
  [ "$state" = ok ] && [ "$pl" != "$skip_pl" ] || continue
  [ "$idxs" != "-" ] || continue
  cmd1=$(awk -F'\t' -v i="${idxs%%,*}" 'NR == i {print $2}' "$g.$stage.tsv")
  UNIT=$(unit_of "$cmd1")
  HOLD=""; [ -f "$OUT_ROOT/_held/$UNIT.held" ] && HOLD="--hold"
  # shellcheck disable=SC2086
  j=$(submit_stage "$g" "$stage" "$idxs" "$prev") && [ -n "$j" ] \
    || { echo "sbatch failed: $pl $stage (its later stages are not submitted)"; FAIL=1; skip_pl="$pl"; continue; }
  [ -n "$HOLD" ] && echo "$j" | tr ':' '\n' >> "$OUT_ROOT/_held/$UNIT.jobs"
  t=$(test_of "$g"); NJOBS[$t]="${NJOBS[$t]:-}:$j"; N=$((N + 1))
  echo "  $g $model $(basename "$out") $stage: job $j, tasks $idxs${prev:+ (afterok $prev)}${HOLD:+ (held, unit $UNIT)}"
  prev="$j"
done <<< "$PLAN"

# one merge per test, afterany on the test's new and queued jobs (never held)
while IFS="$TAB" read -r mid mcmd; do
  [ -n "$mid" ] || continue
  t=$(echo "$mcmd" | sed -n 's/.*--test \([a-z0-9]*\).*/\1/p')
  i=$(awk -F'\t' -v id="$mid" '$1 == id {print NR}' "$MERGEG.s1_merge.tsv")
  D="${NJOBS[$t]:-}${QJOBS[$t]:-}"
  j=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" $R --time="$TIME_LIMIT" --output="$LOG_ROOT/ra20_${MERGEG}_%A_%a.out" \
        ${D:+--dependency=afterany$D} --array="$i" "task_$MERGEG.sbatch" "$MERGEG.s1_merge.tsv") \
    || { echo "sbatch failed: merge of $t"; FAIL=1; continue; }
  echo "$j $MERGEG.s1_merge.tsv $i" >> "$SUB"
  echo "  merge $t: job $j${D:+ (afterany ${D#:})}"
done < "$MERGEG.s1_merge.tsv"
echo "submitted: $N run jobs and the merges; status: bash $HERE/status.sh"
exit $FAIL
