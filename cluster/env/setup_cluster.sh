#!/bin/bash
# One-time cluster setup for the cluster bundle (loopedcot/cluster). Run it once on a login node from the cloned
# repository; it submits no job. A rerun skips every step that is already done (venv stamps, snapshot folders, the
# xllm clone at its commit, existing folders) and rewrites $OUT_ROOT/cluster.env. It edits no tracked file.
#
#   bash loopedcot/cluster/env/setup_cluster.sh --out-root /proj/x/out --hf-home /proj/x/hf --weights-root /proj/x/ckpt \
#        --partition gpu --account myacct --nodelist n01,n02,n03,n04,n05,n06 [--nodes-expected 6] [--modules "cuda/12.4"] [--dry-run]
#
# Steps
#   1. checks: no path inside a home directory; node python, CPU architecture, torch (the node image), CUDA build,
#      nvcc and driver recorded in $OUT_ROOT/_env.json; the Python minimum of every family (ouro 3.9, raven 3.10,
#      ifm 3.10, parcae 3.11; cluster/env/CLUSTER.md says why).
#   2. $OUT_ROOT and $LOG_ROOT.
#   3. one venv per family from cluster/env/requirements_cluster_<family>.txt, created with --system-site-packages
#      (the loopedcot convention: torch comes from the node image; a pip constraint pins torch to the node version so
#      no package can replace it).
#   4. the xllm-loop clone at the pinned commit (the g0_xllm job builds it on a GPU node).
#   5. every checkpoint of cluster/weights.json at its pinned revision: hub snapshots into $HF_HOME, the IFM folders
#      into $WEIGHTS_ROOT/<folder> (hub local-dir layout; the commit is the first line of
#      .cache/huggingface/download/config.json.metadata). A checkpoint that cannot be downloaded gets its copy command.
#   6. inputs that are not in the repository: the five plan files ($PLAN_DIR) and the SETTLE cell files ($SETTLE).
#   7. the node restriction SLURM_RESTRICT (--exclude of every other node of the partition, or --reservation). The
#      submit scripts pass it with --partition, --account and --output on every sbatch command line; the template
#      headers keep their placeholders and are never edited.
#   8. $OUT_ROOT/cluster.env (outside the repository): every variable the templates and submit scripts use (`source`
#      it before any submit script). Values not given are written as <PLACEHOLDER> and listed in the checklist.
#
# Options (an environment variable of the same name is used when the option is absent):
#   --out-root OUT_ROOT     output tree of the run (required for steps 1-2)
#   --log-root LOG_ROOT     Slurm log folder (default $OUT_ROOT/logs)
#   --hf-home HF_HOME       Hugging Face home; snapshots under $HF_HOME/hub (about 50.6 GB)
#   --weights-root WEIGHTS_ROOT  IFM checkpoint folders (about 27.4 GB)
#   --xllm-root XLLM_ROOT   xllm-loop clone (default $REPO/xllm-loop)
#   --venv-root VENV_ROOT   the four venvs (default $REPO/venvs)
#   --plan-dir PLAN_DIR     the five plan files (default $REPO/loopedcot/cluster/plans)
#   --settle SETTLE         SETTLE cell files of steer_gen (default $REPO/loopedcot/cluster/data)
#   --partition PARTITION --account ACCOUNT
#   --nodelist NODELIST     the nodes the run may use, at least N (comma list or Slurm hostlist); when it names more,
#                           the probe step of submit_chain.sh keeps the first N healthy ones
#   --nodes-expected N      how many nodes NODELIST must name (default 6; CARDS = 8 x N, 48 by default)
#   --modules MODULES       environment modules every job loads first (module load $MODULES; default none)
#   --reservation RESERVATION  optional: a reservation that holds those nodes
#   --python PYTHON         node interpreter whose site-packages hold torch (default python3)
#   --venvs-only            run steps 1-3 only (checks, folders, the family venvs), e.g. to retry a failed pip install
#   --dry-run               no venv creation, pip install, clone or download; every check, path, the _env.json,
#                           the node restriction and the cluster.env write still run
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LC="$(cd "$HERE/../.." && pwd)"                 # .../loopedcot
REPO="$(dirname "$LC")"
XLLM_URL=https://github.com/ifm-ai/xllm-loop.git
XLLM_COMMIT=3af99f493e14fba162625d5dd687abea551a299e
FAMILIES="ouro raven parcae ifm"
declare -A PYMIN=([ouro]=3.9 [raven]=3.10 [parcae]=3.11 [ifm]=3.10)
declare -A WHY=([ouro]="pilot venv Python 3.9" [raven]="McLeish and Huginn modeling files use X | Y unions at import"
                [parcae]="parcae-lm declares Requires-Python >= 3.11"
                [ifm]="xllm-loop uses str | None in a signature at import; fla-core, transformers 5.8.0 need >= 3.10")
DRY=0; VENVS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --hf-home) HF_HOME="$2"; shift 2;;
    --weights-root) WEIGHTS_ROOT="$2"; shift 2;;
    --xllm-root) XLLM_ROOT="$2"; shift 2;;
    --venv-root) VENV_ROOT="$2"; shift 2;;
    --plan-dir) PLAN_DIR="$2"; shift 2;;
    --settle) SETTLE="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --account) ACCOUNT="$2"; shift 2;;
    --nodelist) NODELIST="$2"; shift 2;;
    --reservation) RESERVATION="$2"; shift 2;;
    --nodes-expected) NODES_EXPECTED="$2"; shift 2;;
    --modules) MODULES="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --venvs-only) VENVS_ONLY=1; shift;;
    -h|--help) sed -n '2,45p' "$0"; exit 0;;
    *) echo "unknown option $1 (see --help)"; exit 2;;
  esac
done
OUT_ROOT="${OUT_ROOT:-}"; HF_HOME="${HF_HOME:-}"; WEIGHTS_ROOT="${WEIGHTS_ROOT:-}"
PARTITION="${PARTITION:-}"; ACCOUNT="${ACCOUNT:-}"; NODELIST="${NODELIST:-}"; RESERVATION="${RESERVATION:-}"
NODES_EXPECTED="${NODES_EXPECTED:-6}"; MODULES="${MODULES:-}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT:+$OUT_ROOT/logs}}"
XLLM_ROOT="${XLLM_ROOT:-$REPO/xllm-loop}"
VENV_ROOT="${VENV_ROOT:-$REPO/venvs}"
PLAN_DIR="${PLAN_DIR:-$LC/cluster/plans}"
SETTLE="${SETTLE:-$LC/cluster/data}"
PYTHON="${PYTHON:-python3}"
DATA_ROOT="$LC/cluster/pilot/data"

FAILS=(); TODO=()
say() { echo "[setup] $*"; }
bad() { echo "[setup] FAIL: $*"; FAILS+=("$*"); }
todo() { TODO+=("$*"); }
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
[ "$DRY" = 1 ] && say "DRY RUN: no venv, pip, clone or download"

# ------------------------------------------------------------------ 1. checks
say "repository: $REPO (bundle $LC/cluster)"
norm() { local p="$1"; command -v cygpath > /dev/null 2>&1 && p=$(cygpath -u "$p"); readlink -m "$p"; }   # cygpath: Git bash only
in_home() {     # the template's rule: records carry paths, so no run folder may sit in a home directory
  local r; r="$(norm "$1")/"
  case "$r" in "$(norm "$HOME")"/*|/home/*|/Users/*|/root/*) return 0;; esac
  return 1
}
ALLOW_HOME="${ALLOW_HOME:-0}"; HOME_RUN=0
for v in REPO OUT_ROOT LOG_ROOT HF_HOME WEIGHTS_ROOT XLLM_ROOT VENV_ROOT PLAN_DIR SETTLE; do
  p="${!v:-}"; [ -n "$p" ] || continue
  case "$p" in /*|[A-Za-z]:/*) ;; *) bad "$v=$p is not an absolute path"; continue;; esac
  if in_home "$p"; then
    if [ "$ALLOW_HOME" = 1 ]; then HOME_RUN=1
    else bad "$v=$p is inside a home directory; use a project or scratch folder (or ALLOW_HOME=1 when the cluster has no scratch)"; fi
  fi
done
[ "$HOME_RUN" = 1 ] && echo "running inside a home directory; records will be scrubbed at merge"
if [ ${#FAILS[@]} -gt 0 ]; then echo "[setup] stopped before creating anything: fix the paths above"; exit 2; fi
[ -n "$OUT_ROOT" ] || todo "OUT_ROOT: pass --out-root (output tree; outside any home directory)"
# the project root: the deepest folder that holds the repository and the output tree (the merge job writes it as
# <PROJECT_ROOT> in every record when the run sits in a home directory)
PROJECT_ROOT=""
if [ -n "$OUT_ROOT" ]; then
  PROJECT_ROOT="$(norm "$REPO")"
  until [ "$PROJECT_ROOT" = / ] || [ "${PROJECT_ROOT:0:1}" != / ]; do
    case "$(norm "$OUT_ROOT")/" in "$PROJECT_ROOT"/*) break;; esac
    PROJECT_ROOT=$(dirname "$PROJECT_ROOT")
  done
  case "$PROJECT_ROOT" in /|"") PROJECT_ROOT="";; esac
fi
# ALLOW_HOME=1: a home directory usually has a quota; the space for weights, HF cache, venvs and outputs is checked
# now, before anything is created (estimates: HF cache 50.6 GB, IFM weights 27.4 GB, four venvs and the xllm build
# 15 GB, outputs of all jobs 20 GB; what is already in place counts against the estimate)
if [ "$HOME_RUN" = 1 ] && [ -n "$OUT_ROOT" ]; then
  root="${PROJECT_ROOT:-$OUT_ROOT}"; probe="$root"; while [ ! -d "$probe" ]; do probe=$(dirname "$probe"); done
  used_gb() { [ -n "$1" ] && [ -d "$1" ] && du -sk "$1" 2> /dev/null | awk '{printf "%.1f", $1 / 1048576}' || echo 0; }
  need=$(awk -v a="$(used_gb "$HF_HOME")" -v b="$(used_gb "$WEIGHTS_ROOT")" -v c="$(used_gb "$VENV_ROOT")" -v d="$(used_gb "$OUT_ROOT")"     'function m(x) {return x > 0 ? x : 0} BEGIN {printf "%.1f", m(50.6 - a) + m(27.4 - b) + m(15 - c) + m(20 - d)}')
  free=$(df -Pk "$probe" 2> /dev/null | awk 'NR == 2 {printf "%.1f", $4 / 1048576}')
  qfree=$(quota -w 2> /dev/null | awk '$2 ~ /^[0-9]+\*?$/ {u = $2; sub(/\*/, "", u); l = ($3 > 0) ? $3 : $4; if (l > 0) {f = (l - u) / 1048576; if (best == "" || f < best) best = f}}
    END {if (best != "") printf "%.1f", (best > 0 ? best : 0)}')
  say "home directory space: need about $need GB more (HF cache, IFM weights, venvs, outputs); df free ${free:-?} GB on $probe; quota free ${qfree:-no quota found} GB"
  say "df -h $probe:"; df -h "$probe" 2> /dev/null | sed 's/^/[setup]   /'
  say "quota -s:"; { quota -s 2>&1 || true; } | sed 's/^/[setup]   /'
  avail="$free"; [ -n "$qfree" ] && avail=$(awk -v a="${free:-0}" -v b="$qfree" 'BEGIN {print (b < a ? b : a)}')
  if [ -z "$avail" ] || awk -v n="$need" -v a="$avail" 'BEGIN {exit !(a < n * 1.1)}'; then
    echo "[setup] stopped before creating anything: about $need GB is needed (10 percent margin), ${avail:-unknown} GB is free in the home directory"
    exit 2
  fi
fi
[ -n "$HF_HOME" ] || todo "HF_HOME: pass --hf-home (about 50.6 GB of hub snapshots)"
[ -n "$WEIGHTS_ROOT" ] || todo "WEIGHTS_ROOT: pass --weights-root (IFM checkpoint folders, about 27.4 GB)"
[ -n "$PARTITION" ] || todo "PARTITION: pass --partition (passed on every sbatch command line)"
[ -n "$ACCOUNT" ] || todo "ACCOUNT: pass --account (passed on every sbatch command line)"
if [ -n "$NODELIST" ]; then
  if command -v scontrol > /dev/null 2>&1; then NODES=$(scontrol show hostnames "$NODELIST" | paste -sd, -)
  else NODES=$(echo "$NODELIST" | tr ' ' ','); fi
  NN=$(echo "$NODES" | tr ',' '\n' | grep -c .)
  if [ "$NN" -lt "$NODES_EXPECTED" ]; then bad "NODELIST names $NN nodes ($NODES); the run needs at least $NODES_EXPECTED (--nodes-expected)"
  elif [ "$NN" -gt "$NODES_EXPECTED" ]; then say "NODELIST names $NN nodes: the probe step keeps the first $NODES_EXPECTED healthy ones and excludes the rest"; fi
else
  NODES=""; todo "NODELIST: pass --nodelist with exactly the $NODES_EXPECTED nodes the run may use"
fi

command -v "$PYTHON" > /dev/null 2>&1 || { bad "no interpreter $PYTHON"; PYTHON=""; }
PYV=""; TORCH_V=""; TORCH_CUDA=""
if [ -n "$PYTHON" ]; then
  PYV=$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  TORCH_V=$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null)
  TORCH_CUDA=$("$PYTHON" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
  say "node python $PYV ($(command -v "$PYTHON")), arch $(uname -m), torch ${TORCH_V:-none} cuda ${TORCH_CUDA:-none}"
  [ -n "$TORCH_V" ] || bad "$PYTHON has no torch: the venvs take torch from the node image (--system-site-packages); load the module or pick the interpreter that has it (--python)"
  [ -n "$TORCH_V" ] && [ -z "$TORCH_CUDA" ] && bad "torch $TORCH_V of $PYTHON is not a CUDA build"
  for f in $FAMILIES; do
    if "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info[:2] >= tuple(map(int, '${PYMIN[$f]}'.split('.'))) else 1)"; then :
    else bad "family $f needs Python >= ${PYMIN[$f]} (${WHY[$f]}); node python is $PYV"; fi
  done
fi
NVCC=$(command -v nvcc 2>/dev/null || true)
[ -z "$NVCC" ] && [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ] && NVCC="$CUDA_HOME/bin/nvcc"
NVCC_V=$([ -n "$NVCC" ] && "$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
[ -n "$NVCC" ] || todo "CUDA_HOME: no nvcc on this node; the g0_xllm job needs the nvcc of CUDA ${TORCH_CUDA:-<torch CUDA>} (module load cuda, then set CUDA_HOME in cluster.env)"
CUDA_HOME_V="${CUDA_HOME:-${NVCC:+$(dirname "$(dirname "$NVCC")")}}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"     # empty: the g0_xllm job reads the compute capability on its GPU node

# ------------------------------------------------------------------ 2. folders and the environment record
if [ -n "$OUT_ROOT" ]; then
  mkdir -p "$OUT_ROOT" "$LOG_ROOT" && say "folders: $OUT_ROOT, $LOG_ROOT"
  PY_EXE="${PYTHON:-}" NVCC_EXE="$NVCC" "${PYTHON:-python3}" - "$OUT_ROOT/_env.json" <<'PY'
import json, os, platform, subprocess, sys, time
def sh(c):
    try:
        return subprocess.run(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60).stdout.decode().strip()
    except Exception as e:  # noqa: BLE001
        return "error: %r" % e
py = os.environ.get("PY_EXE") or ""
rec = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "where": "login node (setup_cluster.sh)",
       "arch": platform.machine(), "os": sh("cat /etc/os-release | sed -n 's/^PRETTY_NAME=//p'"),
       "python": sh(py + " -c 'import sys; print(sys.version.split()[0])'") if py else None, "python_exe": sh("command -v " + py) if py else None,
       "torch": sh(py + " -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())'") if py else None,
       "nvcc": sh((os.environ.get("NVCC_EXE") or "nvcc") + " --version | tail -2"),
       "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")}
json.dump(rec, open(sys.argv[1], "w"), indent=1)
print("[setup] environment record:", sys.argv[1])
PY
fi

# ------------------------------------------------------------------ 3. family venvs
CONSTRAINT="$VENV_ROOT/torch_constraint.txt"
for f in $FAMILIES; do
  req="$HERE/requirements_cluster_$f.txt"; V="$VENV_ROOT/$f"
  [ -f "$req" ] || { bad "missing $req"; continue; }
  stamp="$(sha256sum "$req" | cut -c1-64) python=$PYV torch=$TORCH_V"
  if [ -f "$V/.cluster_setup" ] && [ "$(cat "$V/.cluster_setup")" = "$stamp" ]; then say "venv $f: ready ($V)"; continue; fi
  if [ -z "$PYTHON" ] || [ -z "$TORCH_V" ]; then say "venv $f: not built (no node torch, see the checklist)"; continue; fi
  "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info[:2] >= tuple(map(int, '${PYMIN[$f]}'.split('.'))) else 1)" || { say "venv $f: not built (Python < ${PYMIN[$f]}, see the checklist)"; continue; }
  say "venv $f: $V from $(basename "$req")"
  run mkdir -p "$VENV_ROOT"
  [ "$DRY" = 1 ] || echo "torch==$TORCH_V" > "$CONSTRAINT"
  run "$PYTHON" -m venv --system-site-packages "$V"
  # the stamp is written only after pip install returned 0 and the import check passed; a failed step leaves no
  # stamp, so the next run installs again
  [ "$DRY" = 1 ] || rm -f "$V/.cluster_setup"
  run "$V/bin/pip" install --no-cache-dir -c "$CONSTRAINT" -r "$req"
  rc=$?
  if [ "$DRY" = 0 ]; then
    if [ "$rc" -ne 0 ]; then
      bad "venv $f: pip install returned $rc (no stamp written; rerun setup to install again)"
    elif "$V/bin/python" -c "import sys, torch; assert not torch.__file__.startswith(sys.prefix), torch.__file__; print('[setup] venv $f: torch', torch.__version__, 'from', torch.__file__)"; then
      echo "$stamp" > "$V/.cluster_setup"
    else bad "venv $f: torch does not come from the node image (no stamp written)"; fi
  fi
done

if [ "$VENVS_ONLY" = 1 ]; then      # --venvs-only: steps 1-3, then the result of the venv step
  for x in "${FAILS[@]}"; do echo "[setup] FAIL  $x"; done
  [ ${#FAILS[@]} -eq 0 ] || exit 1
  exit 0
fi

# ------------------------------------------------------------------ 4. xllm-loop clone (built by the g0_xllm job)
if [ -d "$XLLM_ROOT/.git" ] && [ "$(git -C "$XLLM_ROOT" rev-parse HEAD 2>/dev/null)" = "$XLLM_COMMIT" ]; then
  say "xllm-loop: $XLLM_ROOT at $XLLM_COMMIT"
else
  say "xllm-loop: clone $XLLM_URL at $XLLM_COMMIT into $XLLM_ROOT"
  [ -d "$XLLM_ROOT/.git" ] || run git clone --quiet "$XLLM_URL" "$XLLM_ROOT"
  run git -C "$XLLM_ROOT" checkout --quiet --detach "$XLLM_COMMIT" || bad "xllm-loop checkout of $XLLM_COMMIT failed"
fi

# ------------------------------------------------------------------ 5. checkpoints of cluster/weights.json
if [ -n "$HF_HOME" ] || [ -n "$WEIGHTS_ROOT" ]; then
  [ -n "$HF_HOME" ] && mkdir -p "$HF_HOME/hub"; [ -n "$WEIGHTS_ROOT" ] && mkdir -p "$WEIGHTS_ROOT"
  WPY="$VENV_ROOT/ouro/bin/python"; [ -x "$WPY" ] || WPY="${PYTHON:-python3}"
  WOUT=$(DRY="$DRY" HF_HOME="$HF_HOME" WEIGHTS_ROOT="$WEIGHTS_ROOT" HF_HUB_OFFLINE=0 "$WPY" - "$LC/cluster/weights.json" <<'PY'
import json, os, shutil, sys
W = json.load(open(sys.argv[1], encoding="utf-8"))
hf, wr, dry = os.environ.get("HF_HOME", ""), os.environ.get("WEIGHTS_ROOT", ""), os.environ.get("DRY") == "1"
ents = [c for c in W["checkpoints"]] + [a for a in W.get("auxiliary", []) if a.get("hub_id")]
def first_line(p):
    try:
        return open(p, encoding="utf-8").readline().strip()
    except OSError:
        return None
def present(e, d):
    if e["location"].startswith("$HF_HOME"):
        blobs = os.path.join(hf, "hub", "models--" + e["hub_id"].replace("/", "--"), "blobs")
        inc = os.path.isdir(blobs) and any(x.endswith(".incomplete") for x in os.listdir(blobs))
        return os.path.isdir(d) and bool(os.listdir(d)) and not inc
    return first_line(os.path.join(d, ".cache", "huggingface", "download", "config.json.metadata")) == e["revision"]
need, rows = {"$HF_HOME": 0, "$WEIGHTS_ROOT": 0}, []
for e in ents:
    loc = e["location"]
    base, root = ("$HF_HOME", hf) if loc.startswith("$HF_HOME") else ("$WEIGHTS_ROOT", wr)
    if not root:
        rows.append(("SKIP", e["key"], "%s not set" % base)); continue
    d = loc.replace(base, root, 1)
    if present(e, d):
        rows.append(("OK", e["key"], "%s at %s" % (e["revision"][:12], d))); continue
    need[base] += int(e.get("size_bytes") or 0)
    if dry:
        rows.append(("WOULD_DOWNLOAD", e["key"], "%s@%s -> %s" % (e["hub_id"], e["revision"][:12], d))); continue
    try:
        from huggingface_hub import snapshot_download
        if base == "$HF_HOME":
            got = snapshot_download(e["hub_id"], revision=e["revision"], cache_dir=os.path.join(hf, "hub"))
        else:
            got = snapshot_download(e["hub_id"], revision=e["revision"], local_dir=d)
        ok = present(e, d) and (base != "$HF_HOME" or os.path.realpath(got) == os.path.realpath(d))
        rows.append(("OK" if ok else "BAD_REVISION", e["key"], "%s@%s -> %s" % (e["hub_id"], e["revision"][:12], got)))
    except Exception as ex:  # noqa: BLE001
        rows.append(("MISSING", e["key"], "%s: %s" % (d, repr(ex)[:160])))
        if base == "$WEIGHTS_ROOT":
            rows.append(("COPY", e["key"], "rsync -a <pilot-host>:<REMOTE_ROOT>/loop_flow/ckpt/%s/ %s/   (or, with access to %s: "
                         "HF_TOKEN=<token> huggingface-cli download %s --revision %s --local-dir %s)"
                         % (e["folder"], d, e["hub_id"], e["hub_id"], e["revision"], d)))
        else:
            rows.append(("COPY", e["key"], "HF_HOME=%s huggingface-cli download %s --revision %s" % (hf, e["hub_id"], e["revision"])))
for root, name in ((hf, "HF_HOME"), (wr, "WEIGHTS_ROOT")):
    if root and os.path.isdir(root):
        free, want = shutil.disk_usage(root).free, need["$" + name]
        rows.append(("DISK" if free > want * 1.05 else "LOW_DISK", name,
                     "%.1f GB to download, %.1f GB free at %s" % (want / 1e9, free / 1e9, root)))
for r in rows:
    print("\t".join(r))
PY
)
  echo "$WOUT" | while IFS=$'\t' read -r st key msg; do say "weights $st $key: $msg"; done
  while IFS=$'\t' read -r st key msg; do
    case "$st" in MISSING|BAD_REVISION|LOW_DISK) bad "checkpoint $key: $msg";; COPY) todo "checkpoint $key: $msg";; esac
  done <<< "$WOUT"
fi

# ------------------------------------------------------------------ 6. plans and SETTLE cell files (not in the repository)
declare -A PLAN=([VB_PLAN]=k17_varbench_plan.md [PUB_PLAN]=k17_public_plan.md [PAR_PLAN]=k17_parcae_plan.md
                 [FRZ_PLAN]=k17_freeze_raven_plan.md [FRZO_PLAN]=k17_freeze_plan.md)
declare -A SHA=()
for v in VB_PLAN PUB_PLAN PAR_PLAN FRZ_PLAN FRZO_PLAN; do
  f="$PLAN_DIR/${PLAN[$v]}"
  if [ -f "$f" ]; then SHA[$v]=$(sha256sum "$f" | cut -c1-64); say "plan $v: $f ${SHA[$v]:0:12}"
  else SHA[$v]="<SHA256 OF $f>"; todo "plan $v: copy loop-signal/notes/${PLAN[$v]} to $f"; fi
done
# steer_gen reads <SETTLE>/<file> in prep, screen and select: the settings and datasets of the task lists, plus the
# a1 GSM8K file of the smoke (prod/steer_gen.py SETTINGS[...]["settle"], cluster/gen_jobs.py SVAMP_SETTLE)
declare -A SETTLE_OF=([a1.gsm8k]=cells_ouro_1_4b_base_gsm8k_natural_k4.jsonl [a2.gsm8k]=cells_ouro_2_6b_base_gsm8k_natural_k4.jsonl
                      [a3.gsm8k]=cells_ouro_1_4b_think_gsm8k_natural2_k4.jsonl [a1.svamp]=cells_ouro_1_4b_base_svamp_natural_k4.jsonl
                      [a2.svamp]=cells_ouro_2_6b_base_svamp_natural_k4.jsonl [a3.svamp]=cells_ouro_1_4b_think_svamp_natural2_k4.jsonl)
NEED_SETTLE=$( (echo "a1 gsm8k"; grep -h -- "--settle-dir" "$LC"/cluster/jobs/g*.tsv 2>/dev/null \
  | sed -n 's/.*K17_DATASET=\([a-z0-9]*\).*--setting \(a[0-9]\).*/\2 \1/p') | sort -u)
while read -r st ds; do
  [ -n "$st" ] || continue
  f="${SETTLE_OF[$st.$ds]:-}"
  [ -n "$f" ] || { bad "no SETTLE file known for setting $st dataset $ds"; continue; }
  if [ -f "$SETTLE/$f" ]; then say "SETTLE $st $ds: $f"
  else
    src=k17_gen; [ "$ds" = svamp ] && src=k17_gen_svamp
    todo "SETTLE $st $ds: rsync -a <pilot-host>:<REMOTE_ROOT>/loop_flow/out/$src/data/settle/$f $SETTLE/"
  fi
done <<< "$NEED_SETTLE"
mkdir -p "$SETTLE" 2>/dev/null || true

# ------------------------------------------------------------------ 7. node restriction (sbatch command lines; templates are not edited)
EXCLUDE=""
if [ -n "$NODES" ] && [ -n "$PARTITION" ] && command -v sinfo > /dev/null 2>&1; then
  ALLN=$(sinfo -h -N -p "$PARTITION" -o %N | sort -u)
  for n in $(echo "$NODES" | tr ',' ' '); do echo "$ALLN" | grep -qx "$n" || bad "node $n is not in partition $PARTITION"; done
  EXCLUDE=$(echo "$ALLN" | grep -vxF -f <(echo "$NODES" | tr ',' '\n') | paste -sd, -)
elif [ -n "$NODES" ]; then
  todo "EXCLUDE: sinfo not available here; rerun setup on a login node with Slurm to write the node restriction"
fi
RESTRICT="${RESERVATION:+--reservation=$RESERVATION }${EXCLUDE:+--exclude=$EXCLUDE}"
if [ -z "$RESTRICT" ] && [ -n "$NODES" ] && [ -n "$PARTITION" ] && command -v sinfo > /dev/null 2>&1; then
  RESTRICT="--partition=$PARTITION"      # the partition holds exactly the run's nodes: nothing to exclude
fi
say "node restriction: ${RESERVATION:+--reservation=$RESERVATION }${EXCLUDE:+--exclude=<$(echo "$EXCLUDE" | tr ',' '\n' | wc -l) other nodes>} (nodes ${NODES:-<NODELIST>}); partition ${PARTITION:-<PARTITION>}, account ${ACCOUNT:-<ACCOUNT>}, logs ${LOG_ROOT:-<LOG_ROOT>}"

# ------------------------------------------------------------------ 8. cluster.env
ph() { if [ -n "${1:-}" ]; then printf '%s' "$1"; else printf '<%s>' "$2"; fi; }
ENVF="${OUT_ROOT:+$OUT_ROOT/cluster.env}"
if [ -z "$ENVF" ]; then
  todo "cluster.env: pass --out-root; the environment file is written to \$OUT_ROOT/cluster.env (outside the repository)"
  ENVF=/dev/null
fi
mkdir -p "$(dirname "$ENVF")" 2> /dev/null || true
cat > "$ENVF" <<EOF
# Written by cluster/env/setup_cluster.sh $(date '+%Y-%m-%d %H:%M'). Source it before the smoke and the submit scripts:
#   source $ENVF
# A value in <ANGLE BRACKETS> was not given to setup: replace it (or rerun setup with the option) before submitting.
export REPO="$REPO"
export OUT_ROOT="$(ph "$OUT_ROOT" OUT_ROOT)"
export LOG_ROOT="$(ph "$LOG_ROOT" LOG_ROOT)"
export DATA_ROOT="$DATA_ROOT"
export HF_HOME="$(ph "$HF_HOME" HF_HOME)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export WEIGHTS_ROOT="$(ph "$WEIGHTS_ROOT" WEIGHTS_ROOT)"
export IFM_CKPT_ROOT="$(ph "$WEIGHTS_ROOT" WEIGHTS_ROOT)"
export XLLM_ROOT="$XLLM_ROOT"
export VENV_OURO="$VENV_ROOT/ouro" VENV_RAVEN="$VENV_ROOT/raven" VENV_PARCAE="$VENV_ROOT/parcae" VENV_IFM="$VENV_ROOT/ifm"
export RUN_ROLE=cluster
export ALLOW_HOME=$ALLOW_HOME                  # 1: the run sits in a home directory (no scratch); the merge job scrubs it
export PROJECT_ROOT="$PROJECT_ROOT"          # written as <PROJECT_ROOT> in every record by the merge job
export VB_PLAN="$PLAN_DIR/${PLAN[VB_PLAN]}" VB_PLAN_SHA="${SHA[VB_PLAN]}"
export PUB_PLAN="$PLAN_DIR/${PLAN[PUB_PLAN]}"
export PAR_PLAN="$PLAN_DIR/${PLAN[PAR_PLAN]}"
export FRZ_PLAN="$PLAN_DIR/${PLAN[FRZ_PLAN]}" FRZ_PLAN_SHA="${SHA[FRZ_PLAN]}"
export FRZO_PLAN="$PLAN_DIR/${PLAN[FRZO_PLAN]}" FRZO_PLAN_SHA="${SHA[FRZO_PLAN]}"
export SETTLE="$SETTLE"
export GEN_MEM_FRACTION=0.9      # steer_gen per-process cap; one task per GPU (needs >= 36 GiB for the 2.6B rows)
export NODES_EXPECTED=$NODES_EXPECTED
export CARDS=$((NODES_EXPECTED * 8))                  # array throttle: $NODES_EXPECTED nodes x 8 GPUs
export MODULES="$MODULES"            # every template and chain step runs: module load \$MODULES (when not empty)
export PARTITION="$(ph "$PARTITION" PARTITION)" ACCOUNT="$(ph "$ACCOUNT" ACCOUNT)"
export NODELIST="$(ph "$NODES" NODELIST)"
export SLURM_RESTRICT="$(ph "$RESTRICT" SLURM_RESTRICT)"      # passed on every sbatch command line
export SLURM_RESTRICT_BASE="$(ph "$RESTRICT" SLURM_RESTRICT)"      # the probe step adds the unused nodes of NODELIST ($OUT_ROOT/run_nodes.env)
export CUDA_HOME="$(ph "$CUDA_HOME_V" CUDA_HOME)"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"      # empty: preflight.sh sets it from nvidia-smi on the GPU node
export MAX_JOBS="\${MAX_JOBS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -f "$(ph "$OUT_ROOT" OUT_ROOT)/run_nodes.env" ]; then . "$(ph "$OUT_ROOT" OUT_ROOT)/run_nodes.env"; fi      # the nodes the probe step chose
EOF
say "environment file: $ENVF"

# ------------------------------------------------------------------ checklist
echo
echo "================ checklist"
n=0
for x in "${FAILS[@]}"; do n=$((n+1)); echo "$n. FAIL  $x"; done
for x in "${TODO[@]}"; do n=$((n+1)); echo "$n. SUPPLY  $x"; done
PH=$([ -f "$ENVF" ] && grep -v '^#' "$ENVF" | grep -o '<[A-Z_][A-Z_ 0-9./a-z-]*>' | sort -u | paste -sd' ' -)
[ -n "$PH" ] && { n=$((n+1)); echo "$n. SUPPLY  placeholders left in $ENVF: $PH"; }
[ $n -eq 0 ] && echo "nothing left: source $ENVF, then DONE_SRC=none bash $LC/cluster/jobs/submit_chain.sh"
echo "next: source $ENVF && DONE_SRC=<staged pilot folder or none> bash $LC/cluster/jobs/submit_chain.sh (cluster/README_CLUSTER.md)"
[ ${#FAILS[@]} -eq 0 ] || exit 1
exit 0
