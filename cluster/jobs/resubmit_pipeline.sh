#!/bin/bash
# Resubmit one pipeline (one script, model and output folder of the job list) after a fix, on the login node:
#   bash loopedcot/cluster/jobs/resubmit_pipeline.sh <script> <model> [<output folder name>]
#   e.g. bash loopedcot/cluster/jobs/resubmit_pipeline.sh steer_gen.py ouro_2_6b_base a2_gsm8k
# The third argument picks one pipeline when the script and model have several (steer_gen.py runs GSM8K and SVAMP
# per model); without it, more than one match is refused with the list. Only tasks without a done marker
# ($OUT_ROOT/_done/<id>) are submitted, stage after stage with afterok, through the same templates and stage lists as
# submit_10_rest.sh (the list submit_chain.sh used: $OUT_ROOT/_chain/jobs.txt). Refused when any task of the pipeline
# is still in the Slurm queue (matched through $OUT_ROOT/_chain/submitted.tsv, which every submission appends to).
# A held unit ($OUT_ROOT/_held/<unit>.held) is submitted held again. A pending merge job gets the new jobs added to its
# afterany dependency; when the merge is not pending, it is named so it can be resubmitted after this pipeline.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${REPO:-}" ] || [ -z "${PARTITION:-}" ]; then
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "$HERE/../env/cluster.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi
  done
fi
: "${OUT_ROOT:?source the cluster.env that setup_cluster.sh wrote}" "${PARTITION:?}" "${ACCOUNT:?}" "${LOG_ROOT:?}"
SCRIPT="${1:-}"; MODEL="${2:-}"; PICK="${3:-}"
[ -n "$SCRIPT" ] && [ -n "$MODEL" ] || { echo "usage: bash $0 <script> <model> [<output folder name>]"; exit 2; }
LISTS=$(sed -n 's/^lists //p' "$OUT_ROOT/_chain/jobs.txt" 2> /dev/null)
LISTS="${LISTS:-$HERE/nolocal}"
JL="$LISTS/joblist.tsv"
[ -f "$JL" ] || { echo "no job list $JL"; exit 2; }
CARDS=${CARDS:-48}
SUB="$OUT_ROOT/_chain/submitted.tsv"

# the pipelines of the script and model: one per output folder (the folder name filters with the third argument)
mapfile -t OUTS < <(awk -F'\t' -v s="$SCRIPT" -v m="$MODEL" -v p="$PICK" '
  NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
  $c["script"] == s && $c["model"] == m {
    o = $c["out"]; n = split(o, parts, "/")
    if (p == "" || parts[n] == p || o == p) if (!(o in seen)) { seen[o] = 1; print o } }' "$JL")
if [ ${#OUTS[@]} -eq 0 ]; then echo "no pipeline of $SCRIPT $MODEL${PICK:+ $PICK} in $JL"; exit 3; fi
if [ ${#OUTS[@]} -gt 1 ]; then
  echo "$SCRIPT $MODEL has ${#OUTS[@]} pipelines; name one with the third argument:"
  for o in "${OUTS[@]}"; do echo "  bash $0 $SCRIPT $MODEL $(basename "$o")"; done
  exit 3
fi
OUTF="${OUTS[0]}"

# its tasks in queue order: group, stage, id, cmd
mapfile -t ROWS < <(awk -F'\t' -v s="$SCRIPT" -v m="$MODEL" -v o="$OUTF" '
  NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
  $c["script"] == s && $c["model"] == m && $c["out"] == o { print $c["group"] "\t" $c["stage"] "\t" $c["id"] "\t" $c["cmd"] }' "$JL")
GROUP=$(echo "${ROWS[0]}" | cut -f1)
unit_of() {          # the hold unit (gen_jobs.family_of)
  case "$1" in *steer_gen_raven*) echo mcleish; return;; esac
  case "$1" in *VENV_OURO*steer_gen*|*steer_gen*VENV_OURO*) echo gen; return;; esac
  case "$1" in *VENV_OURO*) echo ouro;; *VENV_RAVEN*) echo raven;; *VENV_PARCAE*) echo parcae;; *VENV_IFM*) echo ifm;; *) echo none;; esac
}
UNIT=$(unit_of "$(echo "${ROWS[0]}" | cut -f4-)")
[ "$GROUP" = g9_merge ] && UNIT=none

# refuse while any of its tasks is queued or running. The stage list of a queued array job comes from
# $OUT_ROOT/_chain/submitted.tsv, else from Slurm (scontrol show job: Command; sacct: SubmitLine) for a run submitted
# before that log existed.
if command -v squeue > /dev/null 2>&1; then
  U="${USER:-$(id -un)}"
  Q=$(squeue -h -r -u "$U" -o "%F %K" 2> /dev/null || true)
  KNOWN=$(cat "$SUB" 2> /dev/null || true)
  unknown=""
  for A in $(squeue -h -u "$U" -n "ra20_$GROUP" -o %F 2> /dev/null | sort -u); do
    echo "$KNOWN" | awk -v a="$A" '$1 == a {f = 1} END {exit !f}' && continue
    L=$( { scontrol show job -o "$A" 2> /dev/null | head -1; sacct -n -P -X -j "$A" -o SubmitLine 2> /dev/null; } \
         | grep -o "$GROUP\.s[0-9][a-z0-9_]*\.tsv" | head -1)
    if [ -z "$L" ]; then unknown="$unknown $A"; continue; fi
    KNOWN="$KNOWN
$A $L $(echo "$Q" | awk -v a="$A" '$1 == a {print $2}' | paste -sd, -)"
  done
  [ -z "$unknown" ] || { echo "REFUSED: queued jobs of ra20_$GROUP whose stage list Slurm does not show:$unknown (scancel them or wait)"; exit 4; }
  busy=""; ids=""
  for r in "${ROWS[@]}"; do
    st=$(echo "$r" | cut -f2); id=$(echo "$r" | cut -f3); L="$GROUP.$st.tsv"
    idx=$(grep -n "^$id"$'\t' "$LISTS/$L" | cut -d: -f1)
    for j in $(echo "$KNOWN" | awk -v l="$L" -v x="$idx" '$2 == l { n = split($3, a, ","); for (i = 1; i <= n; i++) if (a[i] == x) print $1 }'); do
      if echo "$Q" | grep -qx "$j $idx"; then busy="$busy $id(job ${j}_$idx)"; ids="$ids ${j}_$idx"; fi
    done
  done
  if [ -n "$busy" ]; then
    echo "REFUSED: tasks of this pipeline are still in the queue:$busy"
    echo "(a stage waiting on a failed one never starts; when that is the case: scancel$ids, then run this again)"
    exit 4
  fi
fi

# the restriction: cluster.env (with the probe's choice) plus the bad nodes
BAD=$(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . | paste -sd, -)
R="${SLURM_RESTRICT:-}"
if [ -n "$BAD" ]; then case "$R" in *--exclude=*) R=$(echo "$R" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$BAD/");; *) R="$R --exclude=$BAD";; esac; fi
HOLD=""; [ -f "$OUT_ROOT/_held/$UNIT.held" ] && HOLD="--hold"
mkdir -p "$OUT_ROOT/_chain"
cd "$LISTS" || exit 2
echo "pipeline: $GROUP $SCRIPT $MODEL $OUTF (unit $UNIT${HOLD:+, held})"
prev=""; NEW=""
for st in $(printf '%s\n' "${ROWS[@]}" | cut -f2 | awk '!seen[$0]++'); do
  L="$GROUP.$st.tsv"; idxs=""; ndone=0
  for id in $(printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v s="$st" '$2 == s {print $3}'); do
    if [ -f "$OUT_ROOT/_done/$id" ]; then ndone=$((ndone + 1)); continue; fi
    idxs="$idxs,$(grep -n "^$id"$'	' "$L" | cut -d: -f1)"
  done
  idxs="${idxs#,}"
  if [ -z "$idxs" ]; then echo "  $st: every task done ($ndone), skipped"; continue; fi
  # shellcheck disable=SC2086
  j=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" $R $HOLD --output="$LOG_ROOT/ra20_${GROUP}_%A_%a.out" \
        ${prev:+--dependency=afterok:$prev} --array="$idxs%$CARDS" "task_$GROUP.sbatch" "$L") || { echo "sbatch failed at $st"; exit 1; }
  echo "$j $L $idxs" >> "$SUB"
  [ -n "$HOLD" ] && echo "$j" >> "$OUT_ROOT/_held/$UNIT.jobs"
  echo "  $st: job $j, tasks $idxs${prev:+ (afterok $prev)}${ndone:+; $ndone done, skipped}"
  prev="$j"; NEW="$NEW:$j"
done
[ -n "$NEW" ] || { echo "nothing to resubmit: every task of the pipeline is done"; exit 0; }

# the merge waits for the new jobs too
M=$(awk '$2 == "g9_merge.s1_merge.tsv" {j = $1} END {print j}' "$SUB" 2> /dev/null)
if [ -n "$M" ] && [ "$GROUP" != g9_merge ] && squeue -h -j "$M" -t PD -o %A 2> /dev/null | grep -q .; then
  dep=$(squeue -h -j "$M" -o %E 2> /dev/null | grep -o '[0-9][0-9]*' | sort -u | paste -sd: -)
  D="$dep$NEW"; scontrol update JobId="$M" Dependency="afterany:${D#:}" && echo "merge job $M now also waits for$NEW"
elif [ "$GROUP" != g9_merge ]; then
  echo "the merge job is not pending: after this pipeline finishes, run: bash $HERE/resubmit_pipeline.sh merge.py all"
fi
