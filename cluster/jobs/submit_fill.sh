#!/bin/bash
# The fill round (B-EXT5), on the login node, after the pull:
#   source <OUT_ROOT>/cluster.env && bash loopedcot/cluster/jobs/submit_fill.sh [--dry-run]
# It submits only these groups, in priority order:
#   1 g17_fillmap  edit-site map cells missing on d28 (k15_full, D4 and D5 included), the 819M designs (k16_path_ifm
#                  --arms Y1) and Ouro-2.6B-Thinking (k17_public)
#   2 g18_fillgen  McLeish and Huginn generation on ASDiv and MAWPS (steer_gen_math.py) and Huginn on SVAMP
#                  (steer_gen_raven.py setting hug_svamp)
#   3 g12_ifmv1    the IFM derived v1 rerun: only its tasks that the running chain did not finish
# A task with a done marker ($OUT_ROOT/_done/<id>) is never submitted. A pipeline (group, script, model, output
# folder) with a task still in the Slurm queue is left alone (matched through $OUT_ROOT/_chain/submitted.tsv, which
# every submission appends to). A pipeline whose task depends on a task outside the pipeline without a done marker
# (the xllm build, for the IFM lines) is not submitted and is named. The stages of a pipeline chain with afterok,
# through the templates and stage lists of submit_10_rest.sh (the list submit_chain.sh used:
# $OUT_ROOT/_chain/jobs.txt, else jobs/nolocal). A held unit ($OUT_ROOT/_held/<unit>.held) is submitted held again.
# FILL_PRIORITIES (default 1,2,3) picks the priorities this call submits. Nodes, packing and time: jobs/pack_lib.sh
# (the last NODELIST node stays empty, two task lines per GPU job, --time=$TIME_LIMIT on every job). Before it submits, it prints the tasks and card-hours per priority (est_card_seconds of the job list) and asks one
# yes (FILL_YES=yes answers it; --dry-run prints and stops). After: a pending merge job gets every new job added to
# its afterany dependency; when the merge is not pending, the script names the command that resubmits it.
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
FILL="g17_fillmap:1 g18_fillgen:2 g12_ifmv1:3"
FILL_PRIORITIES=${FILL_PRIORITIES:-1,2,3}       # the priorities this call submits (comma list)
FILL=$(for x in $FILL; do case ",$FILL_PRIORITIES," in *",${x#*:},"*) printf "%s " "$x";; esac; done)
[ -n "$FILL" ] || { echo "FILL_PRIORITIES=$FILL_PRIORITIES names no priority of 1, 2, 3"; exit 2; }
LISTS=$(sed -n 's/^lists //p' "$OUT_ROOT/_chain/jobs.txt" 2> /dev/null)
LISTS="${LISTS:-$HERE/nolocal}"
JL="$LISTS/joblist.tsv"
[ -f "$JL" ] || { echo "no job list $JL"; exit 2; }
PLAN_IDS="${PLAN_IDS:-}"        # one task id per line: only those ids are planned and priced (the planner)
[ -z "$PLAN_IDS" ] || [ -f "$PLAN_IDS" ] || { echo "PLAN_IDS=$PLAN_IDS: no such file"; exit 2; }
. "$HERE/pack_lib.sh"      # nodes (RESERVE_NODES), CARDS, packing, TIME_LIMIT
nodes_setup
SUB="$OUT_ROOT/_chain/submitted.tsv"
mkdir -p "$OUT_ROOT/_chain"
touch "$SUB.packed"
TAB=$(printf '\t')
unit_of() {          # the hold unit (gen_jobs.family_of)
  case "$1" in *steer_gen_raven*) echo mcleish; return;; *steer_gen_math*" mcleish "*) echo mcleish; return;; esac
  case "$1" in *VENV_OURO*steer_gen*|*steer_gen*VENV_OURO*) echo gen; return;; esac
  case "$1" in *VENV_OURO*) echo ouro;; *VENV_RAVEN*) echo raven;; *VENV_PARCAE*) echo parcae;; *VENV_IFM*) echo ifm;; *) echo none;; esac
}

# queued tasks: "<stage list> <line index>" of every array task of ours still in the queue
QUEUED=""
if command -v squeue > /dev/null 2>&1; then
  U="${USER:-$(id -un)}"
  Q=$(squeue -h -r -u "$U" -o "%F %K" 2> /dev/null || true)
  if [ -n "$Q" ]; then
    QUEUED=$(awk 'NR == FNR { q[$1 " " $2] = 1; next }
                  { n = split($3, a, ","); for (i = 1; i <= n; i++) if (($1 " " a[i]) in q) print $2 " " a[i] }' \
             <(echo "$Q") "$SUB" 2> /dev/null | sort -u)
    QUEUED=$(printf '%s\n%s\n' "$QUEUED" "$(queued_packed "$Q")" | grep . | sort -u)
    for g in $(echo "$FILL" | tr ' ' '\n' | cut -d: -f1); do       # a queued fill job the log does not know: refuse
      for A in $(squeue -h -u "$U" -n "ra20_$g" -o %F 2> /dev/null | sort -u); do
        awk -v a="$A" '$1 == a {f = 1} END {exit !f}' "$SUB" "$SUB.packed" 2> /dev/null && continue
        echo "REFUSED: queued job $A of ra20_$g is not in $SUB (scancel it or wait)"; exit 4
      done
    done
  fi
fi

# line index of every task in its stage list (array task i = line i)
LINES=$(mktemp)
trap 'rm -f "$LINES"' EXIT
(cd "$LISTS" && for g in $(echo "$FILL" | tr ' ' '\n' | cut -d: -f1); do
   for L in "$g".s*.tsv; do [ -f "$L" ] && awk -F'\t' -v l="$L" '{print $1 "\t" l "\t" FNR}' "$L"; done
 done) > "$LINES"
# the plan: one line per (priority, group, pipeline, stage): the task line indices to submit ("-" for none)
PLAN=$(awk -F'\t' -v fill="$FILL" -v done_dir="$OUT_ROOT/_done" -v queued="$QUEUED" -v pids="$PLAN_IDS" '
  BEGIN { n = split(fill, f, " "); for (i = 1; i <= n; i++) { split(f[i], p, ":"); pri[p[1]] = p[2] }
          if (pids != "") { usep = 1; while ((getline x < pids) > 0) if (x != "") pid[x] = 1 }
          m = split(queued, ql, "\n"); for (i = 1; i <= m; i++) if (ql[i] != "") inq[ql[i]] = 1 }
  FILENAME == ARGV[1] { lst[$1] = $2; line[$1] = $3; next }
  FNR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
  { g = $c["group"] }
  !(g in pri) { next }
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
      pl = order[i]; split(pl, q, SUBSEP); g = q[1]
      ext = ""
      ni = split(ids[pl], il, " ")
      for (j = 1; j <= ni; j++) {
        nd = split(deps[il[j]], dl, ",")
        for (d = 1; d <= nd; d++) {
          if (dl[d] == "" || member[dl[d]] == pl) continue
          if ((getline x < (done_dir "/" dl[d])) < 0) { if (index(ext " ", " " dl[d] " ") == 0) ext = ext " " dl[d] }
          else close(done_dir "/" dl[d]) } }
      state = busy[pl] != "" ? "queued" : (ext != "" ? "blocked" : "ok")
      why = state == "queued" ? busy[pl] : ext
      ns = split(stages[pl], sl, " ")
      for (s = 1; s <= ns; s++) { k = pl SUBSEP sl[s]
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%.1f\t%s\n", pri[g], g, q[2], q[3], q[4], sl[s], state,
               (todo[k] == "" ? "-" : substr(todo[k], 2)), ntodo[k], ndone[k], secs[k], (why == "" ? "-" : why) } }
    exit bad }' "$LINES" "$JL") || { echo "the job list and the stage lists of $LISTS disagree (regenerate them)"; exit 3; }
[ -n "$PLAN" ] || { echo "no fill group in $JL (regenerate the lists with gen_jobs.py)"; exit 3; }
PLAN=$(echo "$PLAN" | sort -s -n -k1,1)          # priority order; queue order inside a priority

echo "fill round from $LISTS, priorities $FILL_PRIORITIES (priority, group, script, model, folder: tasks to submit, done, card-hours)"
echo "$PLAN" | awk -F'\t' -v sel="$FILL_PRIORITIES" '
  { pl = $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
    if (!(pl in seen)) { seen[pl] = 1; order[++n] = pl; st[pl] = $7; why[pl] = $12 }
    todo[pl] += $9; dn[pl] += $10; if ($7 == "ok") { h[pl] += $11; hp[$1] += $11; tp[$1] += $9 } }
  END {
    for (i = 1; i <= n; i++) { pl = order[i]; split(pl, q, "\t"); m = split(q[5], fo, "/")
      line = sprintf("  P%s %s %s %s %s: ", q[1], q[2], q[3], q[4], fo[m])
      if (st[pl] == "queued") print line "in the queue, left alone (" why[pl] ")"
      else if (st[pl] == "blocked") print line "NOT SUBMITTED: needs the done marker of" why[pl]
      else if (todo[pl] == 0) print line "every task done (" dn[pl] ")"
      else printf "%s%d tasks, %d done, %.1f card-hours\n", line, todo[pl], dn[pl], h[pl] / 3600 }
    tot = 0; ntot = 0
    ns = split(sel, ps, ",")
    for (k = 1; k <= ns; k++) { p = ps[k]; printf "priority %d: %d tasks, %.1f card-hours\n", p, tp[p], hp[p] / 3600; tot += hp[p]; ntot += tp[p] }
    printf "total: %d tasks, %.1f card-hours\n", ntot, tot / 3600 }'
NTASK=$(echo "$PLAN" | awk -F'\t' '$7 == "ok" {n += $9} END {print n + 0}')
[ "$NTASK" -gt 0 ] || { echo "nothing to submit"; exit 0; }
nodes_say
time_check
pack_report "$(echo "$PLAN" | awk -F'\t' '$7 == "ok" {print $2 "\t" $6 "\t" $8}')"
[ "$DRY" = 1 ] && { echo "dry run: nothing submitted"; exit 0; }
ANS="${FILL_YES:-}"
if [ -z "$ANS" ]; then read -r -p "submit these $NTASK tasks? type yes: " ANS || ANS=""; fi
[ "$ANS" = yes ] || { echo "not submitted"; exit 1; }

# the restriction: cluster.env (with the probe's choice) plus the bad nodes
BAD=$(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . | paste -sd, -)
R="${SLURM_RESTRICT:-}"
if [ -n "$BAD" ]; then case "$R" in *--exclude=*) R=$(echo "$R" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$BAD/");; *) R="$R --exclude=$BAD";; esac; fi
add_excl
cd "$LISTS" || exit 2
NEW=""; FAIL=0
prevpl=""; prev=""; skip_pl=""
while IFS="$TAB" read -r pri g script model out stage state idxs ntodo ndone secs why; do
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
  echo "  P$pri $g $model $(basename "$out") $stage: job $j, tasks $idxs${prev:+ (afterok $prev)}${HOLD:+ (held, unit $UNIT)}"
  prev="$j"; NEW="$NEW:$j"
done <<< "$PLAN"
[ -n "$NEW" ] || { echo "nothing submitted"; exit 1; }

# the merge waits for the new jobs too
M=$(awk '$2 == "g9_merge.s1_merge.tsv" {j = $1} END {print j}' "$SUB" 2> /dev/null)
if [ -n "$M" ] && squeue -h -j "$M" -t PD -o %A 2> /dev/null | grep -q .; then
  dep=$(squeue -h -j "$M" -o %E 2> /dev/null | grep -o '[0-9][0-9]*' | sort -u | paste -sd: -)
  D="$dep$NEW"; scontrol update JobId="$M" Dependency="afterany:${D#:}" && echo "merge job $M now also waits for$NEW"
else
  echo "the merge job is not pending: after the fill jobs finish, run: bash $HERE/resubmit_pipeline.sh merge.py all"
fi
exit $FAIL
