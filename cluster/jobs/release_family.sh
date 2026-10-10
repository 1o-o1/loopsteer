#!/bin/bash
# Release a held unit (ouro, gen, raven, mcleish, parcae or ifm; held when its venv's preflight or its smoke failed):
# after the fix, run on the login node
#   bash loopedcot/cluster/jobs/release_family.sh ifm
# It submits one one-GPU job that reruns the preflight of the unit's venv (env/preflight.sh, PREFLIGHT_FAMILIES; gen
# uses the ouro venv, mcleish the raven venv; for ifm the xllm build too) and the unit's smoke lines
# (env/smoke_cluster.sh, SMOKE_FAMILIES=<unit>). When both pass, the job releases the unit's held jobs
# ($OUT_ROOT/_held/<unit>.jobs, scontrol release), removes $OUT_ROOT/_held/<unit>.held and writes
# $OUT_ROOT/_held/<unit>.released. When one fails, the jobs stay held and the reason is in
# $OUT_ROOT/_diag/release_<unit>.txt. Progress: bash jobs/status.sh.
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
if [ -z "${REPO:-}" ] || [ -z "${PARTITION:-}" ]; then
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "$(dirname "$SELF")/../env/cluster.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi
  done
fi
: "${REPO:?source the cluster.env that setup_cluster.sh wrote}" "${OUT_ROOT:?}"
LC="$REPO/loopedcot"

if [ "${1:-}" = "--run" ]; then            # inside the Slurm job
  f="${2:?family}"
  D="$OUT_ROOT/_diag/release_$f.txt"
  if [ -n "${MODULES:-}" ]; then
    set +u; type module > /dev/null 2>&1 || . /etc/profile > /dev/null 2>&1 || true
    module load $MODULES || { echo "module load $MODULES failed" > "$D"; exit 5; }
    set -u
  fi
  export SETUPTOOLS_USE_DISTUTILS=stdlib
  case "$f" in gen) venv=ouro;; mcleish) venv=raven;; *) venv="$f";; esac
  if ! PREFLIGHT_FAMILIES="$venv" bash "$LC/cluster/env/preflight.sh"; then
    { echo "release $f: preflight failed again; the jobs stay held"; cat "$OUT_ROOT/_preflight.json" 2> /dev/null; } > "$D"
    exit 1
  fi
  if ! SMOKE_FAMILIES="$f" bash "$LC/cluster/env/smoke_cluster.sh"; then
    { echo "release $f: smoke failed; the jobs stay held"; cat "$OUT_ROOT/_smoke_fail_$f.txt" 2> /dev/null; } > "$D"
    exit 1
  fi
  ids=$(grep . "$OUT_ROOT/_held/$f.jobs" | paste -sd, -)
  scontrol release "$ids" || { echo "release $f: scontrol release $ids failed" > "$D"; exit 1; }
  rm -f "$OUT_ROOT/_held/$f.held" "$D"
  date '+%F %T' > "$OUT_ROOT/_held/$f.released"
  echo "family $f released: jobs $ids"
  exit 0
fi

f="${1:-}"
case "$f" in ouro|gen|raven|mcleish|parcae|ifm) ;; *) echo "usage: bash $SELF <ouro|gen|raven|mcleish|parcae|ifm>"; exit 2;; esac
[ -s "$OUT_ROOT/_held/$f.jobs" ] || { echo "family $f has no held jobs ($OUT_ROOT/_held/$f.jobs is empty or missing)"; exit 3; }
RES=$(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . | paste -sd, -)
R="${SLURM_RESTRICT:-}"
if [ -n "$RES" ]; then case "$R" in *--exclude=*) R=$(echo "$R" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$RES/");; *) R="$R --exclude=$RES";; esac; fi
# shellcheck disable=SC2086
j=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" $R --job-name="ra20_release_$f" --gres=gpu:1 \
      --cpus-per-task=8 --mem=96G --time=4:00:00 --output="$LOG_ROOT/ra20_release_${f}_%j.out" \
      --wrap "bash $SELF --run $f") || { echo "sbatch failed"; exit 1; }
echo "release job for family $f: $j (log $LOG_ROOT/ra20_release_${f}_$j.out); progress: bash $LC/cluster/jobs/status.sh"
