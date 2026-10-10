#!/bin/bash
# The whole cluster run in one command (cluster/README_CLUSTER.md). Run it on a login node from the project folder:
#   bash repo/loopedcot/cluster/run_all.sh
# It discovers
#   PARTITION  the default partition (the one sinfo marks with *)
#   ACCOUNT    the first association of the user (sacctmgr show assoc, else sshare)
#   project    the current folder (refused inside a home directory); out/, hf/ and ckpt/ go there
#   NODELIST   the GPU nodes of the partition that Slurm reports usable, idle ones first; the probe step of
#              submit_chain.sh keeps the first NODES_EXPECTED (6) healthy ones, so no node list is needed
#   PYTHON     python3 when it has a CUDA build of torch (after module load $MODULES when MODULES is set)
# prints them with the free disk of the project folder, asks one yes, then runs env/setup_cluster.sh with those
# values, sources out/cluster.env, submits jobs/submit_chain.sh with DONE_SRC=none and prints jobs/status.sh.
# A value it cannot discover is asked for by name, one question each, and kept in out/run_all.env for the next run.
# An environment variable of the same name wins over both: PARTITION ACCOUNT NODELIST NODES_EXPECTED MODULES PYTHON
# RESERVATION (for example: ACCOUNT=other bash repo/loopedcot/cluster/run_all.sh).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../loopedcot/cluster
LC="$(dirname "$HERE")"
PROJ="$(pwd -P)"
OUT_ROOT="$PROJ/out"
ANS="$OUT_ROOT/run_all.env"
say() { echo "[run_all] $*"; }
stop() { echo "[run_all] STOP: $1"; exit "${2:-1}"; }

norm() { local p="$1"; command -v cygpath > /dev/null 2>&1 && p=$(cygpath -u "$p"); readlink -m "$p"; }
in_home() {          # the rule of setup_cluster.sh: no run folder inside a home directory
  local r; r="$(norm "$1")/"
  case "$r" in "$(norm "$HOME")"/*|/home/*|/Users/*|/root/*) return 0;; esac
  return 1
}
in_home "$PROJ" && stop "the current folder $PROJ is inside a home directory; cd to a project or scratch folder and run again" 2

# ------------------------------------------------------------------ remembered answers, then discovery
for v in PARTITION ACCOUNT NODELIST NODES_EXPECTED MODULES PYTHON RESERVATION; do
  [ -n "${!v+x}" ] && eval "GIVEN_$v=1"          # set in the environment: wins over the answers file
done
if [ -f "$ANS" ]; then
  while IFS= read -r line; do
    v="${line#export }"; v="${v%%=*}"
    [ -n "$(eval echo "\${GIVEN_$v:-}")" ] || eval "$line"
  done < <(grep '^export [A-Z_]*=' "$ANS")
fi
ask() {              # ask NAME "what it is"; the answer is kept in out/run_all.env
  local name="$1" val
  echo "[run_all] $name could not be discovered: $2"
  printf '%s = ' "$name"
  IFS= read -r val || stop "no answer for $name"
  printf -v "$name" '%s' "$val"
  mkdir -p "$OUT_ROOT"
  { grep -v "^export $name=" "$ANS" 2> /dev/null || true; printf 'export %s=%q\n' "$name" "$val"; } > "$ANS.tmp"
  mv -f "$ANS.tmp" "$ANS"
}
SRC_PARTITION="given"; SRC_ACCOUNT="given"; SRC_NODELIST="given"

if [ -z "${PARTITION:-}" ]; then
  PARTITION=$(sinfo -h -o %P 2> /dev/null | sed -n 's/\*$//p' | head -1); SRC_PARTITION="sinfo default (*)"
  [ -n "$PARTITION" ] || { ask PARTITION "the Slurm partition of the GPU nodes (sinfo shows no default partition)"; SRC_PARTITION="answer"; }
fi
if [ -z "${ACCOUNT:-}" ]; then
  U="${USER:-$(id -un)}"
  ACCOUNT=$(sacctmgr -nP show assoc user="$U" format=account 2> /dev/null | grep -v '^ *$' | head -1 | tr -d ' ')
  SRC_ACCOUNT="sacctmgr (first association)"
  if [ -z "$ACCOUNT" ]; then
    ACCOUNT=$(sshare -h -U -o account 2> /dev/null | awk 'NF {print $1; exit}'); SRC_ACCOUNT="sshare"
  fi
  [ -n "$ACCOUNT" ] || { ask ACCOUNT "the Slurm account to charge (sbatch --account)"; SRC_ACCOUNT="answer"; }
fi
if [ -z "${NODELIST:-}" ]; then
  # GPU nodes (GRES with gpu) of the partition in a usable state; not responding (*) and drained or down nodes are left
  # out, idle nodes come first so the probes start soon
  NODELIST=$(sinfo -h -N -p "$PARTITION" -o '%N|%G|%t' 2> /dev/null | awk -F'|' '
    $2 ~ /gpu/ && $3 !~ /\*$/ {
      s = $3; sub(/[~#!%$@^-]+$/, "", s)
      r = (s == "idle") ? 1 : (s ~ /^mix/) ? 2 : (s ~ /^(alloc|comp|plnd|planned|resv)/) ? 3 : 0
      if (r > 0 && !seen[$1]++) print r "\t" NR "\t" $1 }' | sort -k1,1n -k2,2n | cut -f3 | paste -sd, -)
  SRC_NODELIST="sinfo GPU nodes of $PARTITION"
  [ -n "$NODELIST" ] || { ask NODELIST "the GPU nodes the run may use, comma list (sinfo shows no GPU node in $PARTITION)"; SRC_NODELIST="answer"; }
fi
NODES_EXPECTED="${NODES_EXPECTED:-6}"
RESERVATION="${RESERVATION:-}"

load_modules() {
  [ -n "${MODULES:-}" ] || return 0
  type module > /dev/null 2>&1 || . /etc/profile > /dev/null 2>&1 || true
  # shellcheck disable=SC2086
  module load $MODULES || stop "module load $MODULES failed"
}
torch_ok() { TORCH_INFO=$("${PYTHON:-python3}" -c 'import sys, torch; print(torch.__version__, "CUDA", torch.version.cuda); sys.exit(0 if torch.version.cuda else 1)' 2> /dev/null); }
MODULES="${MODULES:-}"
load_modules
PYTHON="${PYTHON:-python3}"
if ! torch_ok || ! command -v nvcc > /dev/null 2>&1; then
  if [ -z "${GIVEN_MODULES:-}" ] && [ -z "$MODULES" ] && ! grep -q '^export MODULES=' "$ANS" 2> /dev/null; then
    ask MODULES "the environment modules that give a Python with a CUDA build of torch and nvcc (space separated; empty for none)"
    load_modules
  fi
fi
if ! torch_ok; then
  ask PYTHON "the path of a Python interpreter with a CUDA build of torch ($PYTHON has none)"
  torch_ok || stop "$PYTHON has no CUDA build of torch"
fi
command -v "$PYTHON" > /dev/null 2>&1 && PYTHON="$(command -v "$PYTHON")"

# ------------------------------------------------------------------ a chain already in Slurm is not submitted twice
if [ -f "$OUT_ROOT/_chain/jobs.txt" ] && command -v squeue > /dev/null 2>&1; then
  live=$(grep -o '[0-9][0-9]*' "$OUT_ROOT/_chain/jobs.txt" | sort -u | \
         comm -12 - <(squeue -h -u "${USER:-$(id -un)}" -o %i 2> /dev/null | sed 's/_.*//' | sort -u) | paste -sd, -)
  [ -z "$live" ] || stop "jobs of the earlier chain are still in Slurm ($live); watch them with: bash $LC/cluster/jobs/status.sh (or scancel them to start again)" 6
fi

# ------------------------------------------------------------------ what was found, one yes
NN=$(echo "$NODELIST" | tr ',' '\n' | grep -c .)
echo
echo "================ discovered"
printf '%-15s %s\n' "project folder" "$PROJ" \
  "free disk" "$(df -Ph "$PROJ" 2> /dev/null | awk 'NR == 2 {print $4 " free of " $2}') (needs about 80 GB for weights and venvs)" \
  "partition" "$PARTITION ($SRC_PARTITION)" \
  "account" "$ACCOUNT ($SRC_ACCOUNT)" \
  "nodes" "$NN: $NODELIST ($SRC_NODELIST); the probe keeps the first $NODES_EXPECTED healthy ones" \
  "reservation" "${RESERVATION:-none}" \
  "modules" "${MODULES:-none}" \
  "python" "$PYTHON (torch ${TORCH_INFO:-?})" \
  "nvcc" "$(command -v nvcc || echo "none (setup lists what to supply)")" \
  "folders" "out $OUT_ROOT, weights $PROJ/hf and $PROJ/ckpt, repository $(dirname "$LC")"
echo "To change a value, run again with it set, for example: ACCOUNT=<name> bash $HERE/run_all.sh"
printf 'Run setup and submit the chain with these values? Type yes: '
IFS= read -r yes || yes=""
[ "$yes" = yes ] || stop "nothing done (the answer was not yes)" 3

# ------------------------------------------------------------------ setup, chain, status
mkdir -p "$OUT_ROOT"
bash "$HERE/env/setup_cluster.sh" --out-root "$OUT_ROOT" --log-root "$OUT_ROOT/logs" --hf-home "$PROJ/hf" \
  --weights-root "$PROJ/ckpt" --partition "$PARTITION" --account "$ACCOUNT" --nodelist "$NODELIST" \
  --nodes-expected "$NODES_EXPECTED" --reservation "$RESERVATION" --modules "$MODULES" --python "$PYTHON" \
  2>&1 | tee "$OUT_ROOT/run_all_setup.log"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || stop "setup_cluster.sh exited $rc: fix the FAIL lines of its checklist above, then run this script again" "$rc"
grep -q '^nothing left' "$OUT_ROOT/run_all_setup.log" || \
  stop "setup is not complete: supply the SUPPLY lines of its checklist above, then run this script again" 4
# shellcheck disable=SC1091
. "$OUT_ROOT/cluster.env"
DONE_SRC=none bash "$HERE/jobs/submit_chain.sh"
rc=$?
echo
bash "$HERE/jobs/status.sh"
[ "$rc" -eq 0 ] || stop "submit_chain.sh exited $rc; its diagnosis is in the status above" 5
