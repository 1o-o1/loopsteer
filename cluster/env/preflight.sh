#!/bin/bash
# The g0_xllm job (submit_00_xllm.sh runs it alone, before every other job): records the GPU node's environment, checks
# the Ouro, raven and Parcae venvs, builds xllm-loop into the IFM venv (build_xllm.sh, with its import, load and
# forward check of every IFM checkpoint), then checks the IFM venv. Writes $OUT_ROOT/_preflight.json (per-family
# records in $OUT_ROOT/_preflight/) and, per family, $OUT_ROOT/_preflight/<family>.ok or <family>.fail. A failed
# family does not stop the run: the smoke skips its lines, submit_10_rest.sh submits its jobs held, and
# jobs/release_family.sh <family> reruns its preflight and smoke and releases them. Exit 1 only when no family passed.
# PREFLIGHT_FAMILIES="ifm" (release_family.sh) checks only those families, keeps the other records, and exits 1 when
# one of them fails.
#
# Environment: the cluster.env that setup_cluster.sh wrote ($OUT_ROOT/cluster.env) (REPO, OUT_ROOT, HF_HOME, WEIGHTS_ROOT, XLLM_ROOT, VENV_OURO, VENV_RAVEN,
# VENV_PARCAE, VENV_IFM, CUDA_HOME, TORCH_CUDA_ARCH_LIST; the last one is read from nvidia-smi when empty).
set -uo pipefail
: "${REPO:?}" "${OUT_ROOT:?}" "${VENV_OURO:?}" "${VENV_RAVEN:?}" "${VENV_PARCAE:?}" "${VENV_IFM:?}" "${XLLM_ROOT:?}"
HERE="$REPO/loopedcot/cluster/env"
P="$OUT_ROOT/_preflight"
mkdir -p "$P"
rm -f "$OUT_ROOT/_preflight.json"
FAMS="${PREFLIGHT_FAMILIES:-ouro raven parcae ifm}"
for f in $FAMS; do rm -f "$P/$f.ok" "$P/$f.fail" "$P/$f.json"; done
export SETUPTOOLS_USE_DISTUTILS=stdlib PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -n "${CUDA_HOME:-}" ] && [ "${CUDA_HOME#<}" = "$CUDA_HOME" ] && export PATH="$CUDA_HOME/bin:$PATH"
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] || [ "${TORCH_CUDA_ARCH_LIST#<}" != "$TORCH_CUDA_ARCH_LIST" ]; then
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';' -)
fi
export TORCH_CUDA_ARCH_LIST

{
  echo "== node";        hostname; uname -m; cat /etc/os-release 2>/dev/null | sed -n 's/^PRETTY_NAME=//p'
  echo "== python";      python3 --version 2>&1; "$VENV_OURO/bin/python" --version 2>&1
  echo "== torch";       "$VENV_OURO/bin/python" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), torch.__file__)' 2>&1
  echo "== nvcc";        command -v nvcc; nvcc --version 2>&1 | tail -2
  echo "== gpu";         nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>&1
  echo "== TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
} | tee "$P/system.txt"

mkdir -p "$OUT_ROOT/_held"
units_of() { case "$1" in ouro) echo "ouro gen";; raven) echo "raven mcleish";; *) echo "$1";; esac; }
mark() { if [ "$2" = 0 ]; then date '+%F %T' > "$P/$1.ok"; echo "family $1: preflight passed"
         else date '+%F %T' > "$P/$1.fail"; echo "family $1: preflight FAILED (its jobs are held)"
           for u in $(units_of "$1"); do echo "preflight of the $1 venv failed ($P/$1.json)" > "$OUT_ROOT/_held/$u.held"; done
         fi; }
for f in $FAMS; do
  rc=0
  if [ "$f" = ifm ]; then
    echo "== xllm build (log: $P/xllm_build.log)"
    if VENV="$VENV_IFM" bash "$HERE/build_xllm.sh" > "$P/xllm_build.log" 2>&1; then echo "xllm build and check_xllm: ok"
    else echo "xllm build or check_xllm FAILED"; tail -20 "$P/xllm_build.log"; rc=1; fi
  fi
  V="VENV_${f^^}"
  "${!V}/bin/python" "$HERE/preflight.py" --family "$f" --out "$P/$f.json" || rc=1
  mark "$f" "$rc"
done
OKN=0; for f in $FAMS; do [ -f "$P/$f.ok" ] && OKN=$((OKN + 1)); done
if [ -n "${PREFLIGHT_FAMILIES:-}" ]; then RC=$([ "$OKN" -eq "$(echo $FAMS | wc -w)" ] && echo 0 || echo 1)
else RC=$([ "$OKN" -gt 0 ] && echo 0 || echo 1); fi

python3 - "$P" "$OUT_ROOT/_preflight.json" "$RC" <<'PY'
import json, os, sys
p, out, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
rec = {"system": open(os.path.join(p, "system.txt")).read(), "families": {}, "failures": [],
       "families_ok": [f for f in ("ouro", "raven", "parcae", "ifm") if os.path.exists(os.path.join(p, f + ".ok"))],
       "families_held": [f for f in ("ouro", "raven", "parcae", "ifm") if os.path.exists(os.path.join(p, f + ".fail"))]}
for f in ("ouro", "raven", "parcae", "ifm"):
    fp = os.path.join(p, f + ".json")
    if not os.path.exists(fp):
        rec["failures"].append("%s: no record (the check did not run)" % f)
        continue
    r = json.load(open(fp))
    rec["families"][f] = {k: r.get(k) for k in ("ok", "python", "torch", "packages", "config", "xllm", "failures")}
    rec["failures"] += ["%s: %s" % (f, x) for x in r.get("failures", [])]
log = os.path.join(p, "xllm_build.log")
rec["xllm_build_log_tail"] = open(log).read()[-3000:] if os.path.exists(log) else None
if "ifm" in rec["families_held"] and not [x for x in rec["failures"] if x.startswith("ifm:")]:
    rec["failures"].append("ifm: xllm build or check_xllm failed (xllm_build.log)")
rec["ok"] = not rec["failures"] and not rec["families_held"]
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps({"preflight_ok": rec["ok"], "families_ok": rec["families_ok"], "families_held": rec["families_held"],
                  "failures": rec["failures"]}, indent=1))
PY
exit $RC
