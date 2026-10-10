#!/bin/bash
# Cluster smoke, run by hand once after setup_cluster.sh and before the submit scripts:
#   source <OUT_ROOT>/cluster.env && bash loopedcot/cluster/env/smoke_cluster.sh [--fresh]
# On a node without a GPU it re-runs itself through srun on one GPU of the run's node set ($PARTITION, $ACCOUNT,
# $SLURM_RESTRICT from cluster.env).
#
# It runs the task lines of jobs/smoke_g1_copy.tsv (k17_varbench, Ouro-1.4B: window, scan, two row shards, merge,
# summarize), jobs/smoke_g2_gen.tsv (steer_gen a1 GSM8K: checks, prep and screen, select, row r00, analyze) and
# jobs/smoke_fam.tsv (one line per further family: raven k17_public mcl_nr D1, Parcae k17_parcae parcae-370m D1, IFM
# k17_freeze_ifm D1), 3 items each, one line at a time through the real task templates (SLURM_ARRAY_TASK_ID set, OUT_ROOT=$OUT_ROOT/_smoke_run so
# the run's own tree and done markers are never touched). A rerun resumes (finished lines are skipped by their done
# markers); --fresh deletes $OUT_ROOT/_smoke_run first.
#
# Pass rule (both parts):
#   (i) identity, bit-exact: the pinned code hashes match (sha256sum -c), every line exits 0, and the harness checks
#       of cluster/env/smoke_check.py hold (records and results.csv written; zero-edit bit identity, positive control
#       exact, structural zero exact, hook counts, merge union equals planned items, no failed row);
#   (ii) cross-host tolerance: every e value (results.csv estimate, random_control_estimate, positive_control_value of
#       both pipelines) and every margin (logit_B - logit_A of the vb_ouro.json items and rows) within 0.001
#       (absolute) of the recorded smoke (cluster/SMOKE_VALUES.json, bound to SMOKE_HASHES.txt by its source hashes).
# Pass: $OUT_ROOT/_smoke.ok. Failure: $OUT_ROOT/_smoke_fail.txt names the failing check, exit 1; the tolerance diff
# is in $OUT_ROOT/_smoke_tolerance_diff.tsv. Operator override of a tolerance miss only:
#   SMOKE_OVERRIDE_REASON='<text>' bash loopedcot/cluster/env/smoke_cluster.sh
# writes _smoke.ok anyway and records the reason and the full diff in $OUT_ROOT/_smoke_override.txt (also printed).
# Identity failures cannot be overridden.
# Report only, never a gate: the value hashes against SMOKE_HASHES.txt in $OUT_ROOT/_smoke_hashes.json, and, when
# SMOKE_REF names a copy of the recorded smoke tree, the every-field float comparison (cluster/smoke_hash.py compare,
# rel 1e-3 / abs 1e-4) in $OUT_ROOT/_smoke_diff_<pipeline>.tsv.
set -uo pipefail
if [ -z "${REPO:-}" ] || [ -z "${PARTITION:-}" ]; then      # cluster.env not sourced: $OUT_ROOT/cluster.env, else env/
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "$(cd "$(dirname "$0")" && pwd)/cluster.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi
  done
fi
: "${REPO:?source the cluster.env that setup_cluster.sh wrote}" "${OUT_ROOT:?}" "${VENV_OURO:?}"
if [ -n "${MODULES:-}" ]; then
  set +u; type module > /dev/null 2>&1 || . /etc/profile > /dev/null 2>&1 || true
  module load $MODULES || { echo "module load $MODULES failed"; exit 5; }
  set -u
fi
LC="$REPO/loopedcot"; JOBS="$LC/cluster/jobs"; SM="$OUT_ROOT/_smoke_run"
FRESH=0; [ "${1:-}" = "--fresh" ] && FRESH=1

if ! nvidia-smi -L > /dev/null 2>&1; then
  [ -z "${SLURM_JOB_ID:-}" ] && command -v srun > /dev/null 2>&1 || { echo "no GPU here and no srun"; exit 2; }
  echo "no GPU on this node: running through srun on one GPU"
  # shellcheck disable=SC2086
  exec srun --partition="$PARTITION" --account="$ACCOUNT" $SLURM_RESTRICT --gres=gpu:1 --cpus-per-task=8 --mem=96G \
       --time=4:00:00 --job-name=ra20_smoke bash "$0" "$@"
fi

rm -f "$OUT_ROOT/_smoke.ok" "$OUT_ROOT/_smoke_fail.txt" "$OUT_ROOT/_smoke_override.txt" "$OUT_ROOT/_smoke_tolerance_diff.tsv"
fail() {
  printf '%s\n' "$@" > "$OUT_ROOT/_smoke_fail.txt"
  echo "SMOKE FAILED: $1 (details in $OUT_ROOT/_smoke_fail.txt)"
  exit 1
}
[ "$FRESH" = 1 ] && rm -rf "$SM"
mkdir -p "$SM/_logs"
echo "smoke output tree: $SM ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), node $(hostname))"

(cd "$REPO" && sha256sum -c --quiet loopedcot/cluster/jobs/scripts_sha256.txt) || fail "pinned code hashes: sha256sum -c of cluster/jobs/scripts_sha256.txt failed"

for g in g1_copy g2_gen fam; do
  L="$JOBS/smoke_$g.tsv"; T="$JOBS/task_$g.sbatch"
  [ "$g" = fam ] && T="$JOBS/task_g3_map.sbatch"      # any task template runs any line (each line activates its venv)
  n=$(grep -c . "$L")
  for i in $(seq 1 "$n"); do
    id=$(sed -n "${i}p" "$L" | cut -f1)
    echo "$(date +%T) $id"
    OUT_ROOT="$SM" SLURM_ARRAY_TASK_ID="$i" bash "$T" "$L" > "$SM/_logs/$id.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] || fail "stage $id exited $rc" "log: $SM/_logs/$id.log" "$(tail -30 "$SM/_logs/$id.log")"
  done
done

# report only: our hosts' value hashes and, with SMOKE_REF, the every-field float comparison
"$VENV_OURO/bin/python" "$LC/cluster/smoke_hash.py" check "$SM" --hashes "$LC/cluster/SMOKE_HASHES.txt" --prefix out_new/ \
    --present_only > "$OUT_ROOT/_smoke_hashes.json" 2>&1 || true
if [ -n "${SMOKE_REF:-}" ] && [ -d "$SMOKE_REF" ]; then
  for d in B/k17_varbench/ouro A/steer_gen/a1_gsm8k; do
    "$VENV_OURO/bin/python" "$LC/cluster/smoke_hash.py" compare "$SMOKE_REF/$d" "$SM/$d" --diff "$OUT_ROOT/_smoke_diff_$(basename "$d").tsv" || true
  done
fi

# the gate: (i) identity checks, (ii) e and margins within 0.001; smoke_check.py writes _smoke.ok, _smoke_fail.txt
# or (tolerance miss with SMOKE_OVERRIDE_REASON) _smoke_override.txt
"$VENV_OURO/bin/python" "$LC/cluster/env/smoke_check.py" --root "$SM" --lists "$JOBS" --report "$OUT_ROOT/_smoke_report.json" \
    --values "$LC/cluster/SMOKE_VALUES.json" --hashes "$LC/cluster/SMOKE_HASHES.txt" \
    --out-root "$OUT_ROOT" --override-reason "${SMOKE_OVERRIDE_REASON:-}"
rc=$?
case $rc in
  0) if [ -f "$OUT_ROOT/_smoke_override.txt" ]; then
       echo "SMOKE PASSED BY OPERATOR OVERRIDE (tolerance miss): $OUT_ROOT/_smoke.ok, reason and diff in $OUT_ROOT/_smoke_override.txt"
     else
       echo "SMOKE PASSED: $OUT_ROOT/_smoke.ok (report: $OUT_ROOT/_smoke_report.json)"
     fi;;
  3) echo "SMOKE FAILED: cross-host tolerance (details in $OUT_ROOT/_smoke_fail.txt)"; exit 1;;
  *) echo "SMOKE FAILED: identity checks (not overridable; details in $OUT_ROOT/_smoke_fail.txt)"
     [ -f "$OUT_ROOT/_smoke_fail.txt" ] || printf 'smoke_check.py exited %s\n' "$rc" > "$OUT_ROOT/_smoke_fail.txt"
     exit 1;;
esac
