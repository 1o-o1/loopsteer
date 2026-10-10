#!/bin/bash
# The whole launch in one command, after setup_cluster.sh (cluster/README_CLUSTER.md):
#   source <OUT_ROOT>/cluster.env && DONE_SRC=none bash loopedcot/cluster/jobs/submit_chain.sh
# DONE_SRC=none: every job runs on the cluster (the lists of jobs/nolocal/, gen_jobs.py --no-local-done).
# DONE_SRC=<folder>: the pulled folders of cluster/done_local.json are staged from that folder first (jobs/stage_done.sh)
# and the jobs that ran on our cards are skipped (the lists of jobs/).
# Steps, each one waiting for the one before (--kill-on-invalid-dep=yes: a failed step cancels the rest). Every sbatch
# call carries --partition, --account, --output and the node restriction ($SLURM_RESTRICT from cluster.env, plus
# --exclude of every node in $OUT_ROOT/bad_nodes.txt); the template headers are never edited.
#   1. setup_check   CPU job: cluster.env complete (no <PLACEHOLDER>, no folder inside a home directory, SLURM_RESTRICT
#                    and NODELIST set), pinned hashes, the plan files with the *_PLAN_SHA values, the SETTLE files, the
#                    four venvs, HF_HOME, WEIGHTS_ROOT. Writes $OUT_ROOT/_chain/setup_check.ok or .fail.
#   2. probe         one five-minute, one-GPU job per node of NODELIST (after step 1): node=, nvidia-smi -L, a torch CUDA
#                    matmul, RC=. Logs in $OUT_ROOT/_probe/<node>.out. A probe that cannot start within two hours ends
#                    by its deadline and counts as unknown (the node is not excluded).
#   3. probe_collect CPU job after every probe: a node is bad when its RC is not 0 or its log shows "CUDA unknown error"
#                    or "No devices were found"; bad nodes go to $OUT_ROOT/bad_nodes.txt. It keeps the first
#                    $NODES_EXPECTED healthy nodes of NODELIST (then nodes with no verdict, when too few are healthy) in
#                    $OUT_ROOT/run_nodes.txt and excludes every other node of NODELIST ($OUT_ROOT/run_nodes.env, which
#                    cluster.env reads). Then it submits, excluding the bad and the unused nodes: 4. g0 (task_g0_xllm.sbatch), 5. smoke (env/smoke_cluster.sh, after g0), 6. stage_submit
#                    (jobs/stage_done.sh unless DONE_SRC=none, then submit_10_rest.sh of the chosen lists, after smoke).
# Every step writes $OUT_ROOT/_chain/<step>.ok or <step>.fail, loads $MODULES when set, and on any failure writes
# $OUT_ROOT/_diag/chain.<step>.txt (exit code, failing check, host role, GPU, free memory, last 50 log lines). A task
# whose GPU cannot start adds its node to bad_nodes.txt and is requeued once elsewhere (the task template, exit 75).
# Progress: bash jobs/status.sh. A rerun clears the chain markers of the earlier chain; done markers stay.
# Internal use: submit_chain.sh --step setup_check|probe_collect|smoke|stage_submit (inside the Slurm job of that step).
set -uo pipefail
HERE0="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${REPO:-}" ] || [ -z "${PARTITION:-}" ]; then      # cluster.env not sourced: $OUT_ROOT/cluster.env, else env/
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "$HERE0/../env/cluster.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi
  done
fi
: "${REPO:?source the cluster.env that setup_cluster.sh wrote}" "${OUT_ROOT:?source the cluster.env that setup_cluster.sh wrote}"
LC="$REPO/loopedcot"
JOBS="$LC/cluster/jobs"
CH="$OUT_ROOT/_chain"
PR="$OUT_ROOT/_probe"
SELF="$JOBS/submit_chain.sh"
mkdir -p "$CH" "$OUT_ROOT/_diag" "$OUT_ROOT/_tasklog"
if [ "${DONE_SRC:-}" = none ]; then LISTS="$JOBS/nolocal"; else LISTS="$JOBS"; fi
G0=$(cut -f1 "$LISTS/g0_xllm.s1_build.tsv" | head -1)
SIGNATURES="CUDA unknown error\|No devices were found"

load_modules() {
  if [ -n "${MODULES:-}" ]; then
    CHECK="module load $MODULES"
    set +u; type module > /dev/null 2>&1 || . /etc/profile > /dev/null 2>&1 || true
    module load $MODULES || { set -u; exit 5; }
    set -u
  fi
}
home_path() {        # 0 when the path lies inside a home directory (records carry paths; the templates refuse those)
  local r; r="$(readlink -m "$1")/"
  case "$r" in "$(readlink -m "$HOME")"/*|/home/*|/Users/*|/root/*) return 0;; esac
  return 1
}
restrict() {         # SLURM_RESTRICT with every node of bad_nodes.txt added to its --exclude list
  local bad r="${SLURM_RESTRICT:-}"
  bad=$(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . | paste -sd, -)
  if [ -n "$bad" ]; then
    case "$r" in *--exclude=*) r=$(echo "$r" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$bad/");; *) r="$r --exclude=$bad";; esac
  fi
  echo "$r"
}
sb() { echo "--parsable --kill-on-invalid-dep=yes --partition=$PARTITION --account=$ACCOUNT $(restrict)"; }

# ------------------------------------------------------------------ a step inside its Slurm job
if [ "${1:-}" = "--step" ]; then
  STEP="${2:?step name}"
  TL="$OUT_ROOT/_tasklog/chain.$STEP.log"
  : > "$TL"
  exec > >(tee -a "$TL") 2>&1
  CHECK="start"
  diag() {
    sleep 1
    { echo "step: chain.$STEP"
      echo "exit code: $1"
      echo "failing check: $CHECK"
      echo "host role: ${RUN_ROLE:-cluster}"
      echo "slurm: job ${SLURM_JOB_ID:-none} node ${SLURMD_NODENAME:-none}"
      echo "gpu (name, free, total): $(nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2> /dev/null | paste -sd';' - || true)"
      echo "host memory: $(free -g 2> /dev/null | awk '/^Mem:/{print $7 " GiB available of " $2 " GiB"}' || true)"
      echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "--- last 50 lines of $TL"
      tail -n 50 "$TL" 2> /dev/null || true; } > "$OUT_ROOT/_diag/chain.$STEP.txt" 2>&1 || true
  }
  finish() {
    local rc=$1
    if [ "$rc" -eq 0 ]; then
      date '+%Y-%m-%d %H:%M:%S' > "$CH/$STEP.ok"; rm -f "$CH/$STEP.fail" "$OUT_ROOT/_diag/chain.$STEP.txt"
    else
      printf 'exit %s: %s\n' "$rc" "$CHECK" > "$CH/$STEP.fail"; rm -f "$CH/$STEP.ok"; diag "$rc"
    fi
  }
  trap 'rc=$?; finish $rc' EXIT
  trap 'CHECK="$CHECK (killed by SIGTERM: time limit, scancel or node failure)"; exit 143' TERM
  refuse() { CHECK="$1"; echo "REFUSED: $1"; exit 4; }
  load_modules
  case "$STEP" in
    setup_check)
      bad=()
      for v in REPO OUT_ROOT LOG_ROOT DATA_ROOT HF_HOME WEIGHTS_ROOT XLLM_ROOT VENV_OURO VENV_RAVEN VENV_PARCAE VENV_IFM \
               VB_PLAN VB_PLAN_SHA PUB_PLAN PAR_PLAN FRZ_PLAN FRZ_PLAN_SHA FRZO_PLAN FRZO_PLAN_SHA SETTLE PARTITION \
               ACCOUNT NODELIST SLURM_RESTRICT GEN_MEM_FRACTION; do
        val="${!v:-}"
        if [ -z "$val" ]; then bad+=("$v is not set in cluster.env")
        else case "$val" in *"<"*">"*) bad+=("$v is a placeholder: $val");; esac; fi
      done
      home_run=0
      for v in REPO OUT_ROOT LOG_ROOT DATA_ROOT HF_HOME WEIGHTS_ROOT XLLM_ROOT SETTLE VENV_OURO VENV_RAVEN VENV_PARCAE VENV_IFM; do
        if [ -n "${!v:-}" ] && home_path "${!v}"; then
          if [ "${ALLOW_HOME:-0}" = 1 ]; then home_run=1; else bad+=("$v=${!v} is inside a home directory (ALLOW_HOME=1 when the cluster has no scratch)"); fi
        fi
      done
      [ "$home_run" = 0 ] || echo "running inside a home directory; records will be scrubbed at merge"
      (cd "$REPO" && sha256sum -c --quiet loopedcot/cluster/jobs/scripts_sha256.txt) || bad+=("pinned hashes: sha256sum -c failed")
      for p in VB FRZ FRZO; do
        f="$(eval echo "\${${p}_PLAN:-}")"; want="$(eval echo "\${${p}_PLAN_SHA:-}")"
        if [ ! -f "$f" ]; then bad+=("${p}_PLAN file missing: $f")
        elif [ "$(sha256sum "$f" | cut -c1-64)" != "$want" ]; then bad+=("${p}_PLAN sha256 is not ${p}_PLAN_SHA")
        fi
      done
      for f in "${PUB_PLAN:-}" "${PAR_PLAN:-}"; do [ -f "$f" ] || bad+=("plan file missing: $f"); done
      for f in "$LC"/cluster/data/cells_*.jsonl; do
        [ -f "${SETTLE:-}/$(basename "$f")" ] || bad+=("SETTLE file missing: ${SETTLE:-}/$(basename "$f")")
      done
      for v in VENV_OURO VENV_RAVEN VENV_PARCAE VENV_IFM; do
        [ -x "${!v:-}/bin/python" ] || bad+=("$v has no bin/python: ${!v:-}")
      done
      [ -d "${HF_HOME:-}/hub" ] || bad+=("HF_HOME has no hub folder: ${HF_HOME:-}")
      [ -d "${WEIGHTS_ROOT:-}" ] || bad+=("WEIGHTS_ROOT is not a folder: ${WEIGHTS_ROOT:-}")
      if [ ${#bad[@]} -gt 0 ]; then
        CHECK="setup check: ${#bad[@]} problem(s): $(printf '%s; ' "${bad[@]}")"
        printf '%s\n' "${bad[@]}"
        exit 1
      fi
      echo "setup check passed"
      exit 0;;
    probe_collect)
      [ -f "$CH/setup_check.ok" ] || refuse "previous step not passed: no $CH/setup_check.ok"
      CHECK="probe verdicts ($PR)"
      touch "$OUT_ROOT/bad_nodes.txt"
      OK=""; UNKNOWN=""
      for n in $(echo "${NODELIST:-}" | tr ',' ' '); do
        f="$PR/$n.out"
        if [ ! -f "$f" ] || ! grep -q "^RC=" "$f"; then
          echo "unknown $n (the probe did not run; not excluded)"
          UNKNOWN="$UNKNOWN $n"
        elif grep -q "$SIGNATURES" "$f" || ! grep -q "^RC=0" "$f"; then
          echo "BAD     $n: $(grep -m1 -i "error\|No devices\|^RC=" "$f" | cut -c1-120)"
          echo "$n" >> "$OUT_ROOT/bad_nodes.txt"
        else
          echo "ok      $n: $(grep -m1 "^matmul" "$f")"
          OK="$OK $n"
        fi
      done
      sort -u "$OUT_ROOT/bad_nodes.txt" | grep . > "$OUT_ROOT/bad_nodes.tmp"; mv -f "$OUT_ROOT/bad_nodes.tmp" "$OUT_ROOT/bad_nodes.txt"
      echo "bad nodes: $(paste -sd, "$OUT_ROOT/bad_nodes.txt")"
      # the run's nodes: the first NODES_EXPECTED healthy nodes in NODELIST order, then nodes with no verdict (the
      # earlier rule: those are not excluded) when fewer are healthy; every other node of NODELIST is excluded
      CHECK="choice of the run's nodes"
      NE="${NODES_EXPECTED:-6}"; run=(); unused=()
      for n in $OK $UNKNOWN; do [ ${#run[@]} -lt "$NE" ] && run+=("$n"); done
      for n in $(echo "${NODELIST:-}" | tr ',' ' '); do
        case " ${run[*]} " in *" $n "*) ;; *) grep -qx "$n" "$OUT_ROOT/bad_nodes.txt" || unused+=("$n");; esac
      done
      [ ${#run[@]} -gt 0 ] || { CHECK="no usable node: every probe of NODELIST failed"; exit 1; }
      printf '%s\n' "${run[@]}" > "$OUT_ROOT/run_nodes.txt"
      SLURM_RESTRICT="${SLURM_RESTRICT_BASE:-${SLURM_RESTRICT:-}}"
      if [ ${#unused[@]} -gt 0 ]; then
        u=$(printf '%s\n' "${unused[@]}" | paste -sd, -)
        case "$SLURM_RESTRICT" in *--exclude=*) SLURM_RESTRICT=$(echo "$SLURM_RESTRICT" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$u/");;
          *) SLURM_RESTRICT="${SLURM_RESTRICT:+$SLURM_RESTRICT }--exclude=$u";; esac
      fi
      export SLURM_RESTRICT          # every later job inherits it (sbatch exports the environment)
      printf '# written by the probe_collect step of submit_chain.sh; cluster.env reads it\nexport SLURM_RESTRICT="%s"\n' \
        "$SLURM_RESTRICT" > "$OUT_ROOT/run_nodes.env"
      echo "run nodes (${#run[@]} of $NE): $(printf '%s\n' "${run[@]}" | paste -sd, -); not used: $(printf '%s\n' ${unused[@]+"${unused[@]}"} | paste -sd, -)"
      [ ${#run[@]} -ge "$NE" ] || echo "WARNING: only ${#run[@]} usable nodes of the $NE expected; the run continues on them"
      CHECK="sbatch of g0 (preflight and xllm build)"
      # shellcheck disable=SC2046
      J4=$(cd "$LISTS" && sbatch $(sb) --output="$LOG_ROOT/ra20_g0_xllm_%A_%a.out" --array=1 \
             "$JOBS/task_g0_xllm.sbatch" g0_xllm.s1_build.tsv) || exit 1
      echo "4 g0 ($G0)  job $J4"
      CHECK="sbatch of the smoke step"
      # shellcheck disable=SC2046
      J5=$(sbatch $(sb) --dependency=afterok:$J4 --job-name=ra20_chain_smoke --gres=gpu:1 --cpus-per-task=8 --mem=96G \
             --time=4:00:00 --output="$LOG_ROOT/ra20_chain_smoke_%j.out" --wrap "bash $SELF --step smoke") || exit 1
      echo "5 smoke         job $J5 (afterok $J4)"
      CHECK="sbatch of the stage_submit step"
      # shellcheck disable=SC2046
      J6=$(sbatch $(sb) --dependency=afterok:$J5 --job-name=ra20_chain_submit --cpus-per-task=2 --mem=16G --time=2:00:00 \
             --output="$LOG_ROOT/ra20_chain_submit_%j.out" --wrap "bash $SELF --step stage_submit") || exit 1
      echo "6 stage_submit  job $J6 (afterok $J5)"
      printf 'g0 %s\nsmoke %s\nstage_submit %s\n' "$J4" "$J5" "$J6" >> "$CH/jobs.txt"
      exit 0;;
    smoke)
      [ -f "$CH/setup_check.fail" ] && refuse "previous step failed: $CH/setup_check.fail"
      [ -f "$CH/setup_check.ok" ] || refuse "previous step not passed: no $CH/setup_check.ok"
      [ -f "$OUT_ROOT/_diag/$G0.txt" ] && [ ! -f "$OUT_ROOT/_done/$G0" ] && refuse "previous step failed: $OUT_ROOT/_diag/$G0.txt"
      [ -f "$OUT_ROOT/_done/$G0" ] || refuse "previous step not done: no $OUT_ROOT/_done/$G0"
      CHECK="cluster/env/smoke_cluster.sh (its reason is in $OUT_ROOT/_smoke_fail.txt)"
      bash "$LC/cluster/env/smoke_cluster.sh"
      exit $?;;
    stage_submit)
      [ -f "$OUT_ROOT/_smoke_fail.txt" ] && refuse "previous step failed: $OUT_ROOT/_smoke_fail.txt"
      [ -f "$CH/smoke.fail" ] && refuse "previous step failed: $CH/smoke.fail"
      [ -f "$OUT_ROOT/_smoke.ok" ] || refuse "previous step not passed: no $OUT_ROOT/_smoke.ok"
      if [ "${DONE_SRC:-}" != none ]; then
        CHECK="jobs/stage_done.sh (DONE_SRC=${DONE_SRC:-})"
        bash "$JOBS/stage_done.sh" || exit $?
      fi
      CHECK="$LISTS/submit_10_rest.sh (its own diagnosis: $OUT_ROOT/_diag/submit_10_rest.txt)"
      bash "$LISTS/submit_10_rest.sh"
      exit $?;;
    *) CHECK="unknown step $STEP"; exit 2;;
  esac
fi

# ------------------------------------------------------------------ submit the chain (login node)
SN=submit_chain
TL="$OUT_ROOT/_tasklog/$SN.log"
: > "$TL"
exec > >(tee -a "$TL") 2>&1
CHECK="start"
diag() {
  sleep 1
  { echo "step: $SN"; echo "exit code: $1"; echo "failing check: $CHECK"
    echo "host role: ${RUN_ROLE:-cluster} (submit host, no GPU)"
    echo "time: $(date '+%Y-%m-%d %H:%M:%S')"; echo "--- last 50 lines of $TL"; tail -n 50 "$TL"
  } > "$OUT_ROOT/_diag/$SN.txt" 2>&1 || true
}
trap 'rc=$?; if [ $rc -ne 0 ]; then diag $rc; else rm -f "$OUT_ROOT/_diag/$SN.txt"; fi' EXIT
fail() { CHECK="$1"; echo "submit_chain: $1"; exit "${2:-1}"; }
[ -n "${DONE_SRC:-}" ] || fail "set DONE_SRC=none (everything runs on the cluster) or DONE_SRC=<folder with the pulled done folders>" 5
[ "$DONE_SRC" = none ] || [ -d "$DONE_SRC" ] || fail "DONE_SRC=$DONE_SRC is not a folder" 5
[ -f "$LISTS/submit_10_rest.sh" ] || fail "no $LISTS/submit_10_rest.sh (generate it: python cluster/gen_jobs.py, with --no-local-done for DONE_SRC=none)" 5
for v in PARTITION ACCOUNT LOG_ROOT SLURM_RESTRICT NODELIST VENV_OURO; do
  case "${!v:-<>}" in *"<"*">"*) fail "$v is not set or still a placeholder in cluster.env (setup_cluster.sh)" 7;; esac
done
command -v sbatch > /dev/null 2>&1 || fail "no sbatch on this host" 2
if home_path "$OUT_ROOT"; then
  [ "${ALLOW_HOME:-0}" = 1 ] || fail "OUT_ROOT=$OUT_ROOT is inside a home directory (ALLOW_HOME=1 when the cluster has no scratch)" 2
  echo "running inside a home directory; records will be scrubbed at merge"
fi
mkdir -p "$LOG_ROOT" "$PR"
# a new chain chooses its nodes again: back to the restriction that setup wrote
export SLURM_RESTRICT="${SLURM_RESTRICT_BASE:-$SLURM_RESTRICT}"
rm -f "$OUT_ROOT/run_nodes.env" "$OUT_ROOT/run_nodes.txt"
rm -f "$CH"/*.ok "$CH"/*.fail "$OUT_ROOT/_smoke_fail.txt" "$PR"/*.out
rm -f "$OUT_ROOT"/_diag/chain.*.txt
echo "lists: $LISTS; node restriction: $(restrict)"
CHECK="sbatch of step 1 (setup_check)"
# shellcheck disable=SC2046
J1=$(sbatch $(sb) --job-name=ra20_chain_setup --cpus-per-task=2 --mem=8G --time=0:30:00 \
       --output="$LOG_ROOT/ra20_chain_setup_%j.out" --wrap "bash $SELF --step setup_check") || fail "$CHECK"
echo "1 setup_check   job $J1"
RES=$(echo "$SLURM_RESTRICT" | tr ' ' '\n' | grep -- "^--reservation=" || true)
PROBE='echo node=$(hostname -s 2> /dev/null || hostname); nvidia-smi -L; '"$VENV_OURO"'/bin/python -c "import torch; x = torch.randn(1024, 1024, device=\"cuda\"); print(\"matmul\", torch.cuda.get_device_name(0), float((x @ x).abs().sum()))"; rc=$?; echo RC=$rc; exit $rc'
PJ=""
for n in $(echo "$NODELIST" | tr ',' ' '); do
  CHECK="sbatch of the probe on node $n"
  # shellcheck disable=SC2086
  p=$(sbatch --parsable --kill-on-invalid-dep=yes --partition="$PARTITION" --account="$ACCOUNT" $RES --nodelist="$n" \
        --dependency=afterok:$J1 --deadline=now+120minutes --job-name=ra20_probe --gres=gpu:1 --cpus-per-task=2 --mem=8G \
        --time=0:05:00 --output="$PR/$n.out" --wrap "$PROBE") || fail "$CHECK"
  PJ="$PJ:$p"
  echo "2 probe $n  job $p (afterok $J1)"
done
CHECK="sbatch of step 3 (probe_collect)"
# shellcheck disable=SC2046
J3=$(sbatch $(sb) --dependency=afterok:$J1,afterany$PJ --job-name=ra20_chain_probe --cpus-per-task=2 --mem=8G \
       --time=0:30:00 --output="$LOG_ROOT/ra20_chain_probe_%j.out" --wrap "bash $SELF --step probe_collect") || fail "$CHECK"
echo "3 probe_collect job $J3 (after setup_check and every probe; it submits g0, smoke and stage_submit)"
printf 'setup_check %s\nprobes %s\nprobe_collect %s\nlists %s\n' "$J1" "${PJ#:}" "$J3" "$LISTS" > "$CH/jobs.txt"
echo "chain submitted; progress: bash $JOBS/status.sh"
