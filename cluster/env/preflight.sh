#!/bin/bash
# The g0_xllm job (submit_00_xllm.sh runs it alone, before every other job): records the GPU node's environment, checks
# the Ouro, raven and Parcae venvs, builds xllm-loop into the IFM venv (build_xllm.sh, with its import, load and
# forward check of every IFM checkpoint), then checks the IFM venv. Writes $OUT_ROOT/_preflight.json (per-family
# records in $OUT_ROOT/_preflight/) and exits 1 when any check fails, so the job leaves no done marker and
# submit_10_rest.sh refuses to start.
#
# Environment: the cluster.env that setup_cluster.sh wrote ($OUT_ROOT/cluster.env) (REPO, OUT_ROOT, HF_HOME, WEIGHTS_ROOT, XLLM_ROOT, VENV_OURO, VENV_RAVEN,
# VENV_PARCAE, VENV_IFM, CUDA_HOME, TORCH_CUDA_ARCH_LIST; the last one is read from nvidia-smi when empty).
set -uo pipefail
: "${REPO:?}" "${OUT_ROOT:?}" "${VENV_OURO:?}" "${VENV_RAVEN:?}" "${VENV_PARCAE:?}" "${VENV_IFM:?}" "${XLLM_ROOT:?}"
HERE="$REPO/loopedcot/cluster/env"
P="$OUT_ROOT/_preflight"
mkdir -p "$P"
rm -f "$OUT_ROOT/_preflight.json"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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

RC=0
for f in ouro raven parcae; do
  V="VENV_${f^^}"
  "${!V}/bin/python" "$HERE/preflight.py" --family "$f" --out "$P/$f.json" || RC=1
done
echo "== xllm build (log: $P/xllm_build.log)"
if VENV="$VENV_IFM" bash "$HERE/build_xllm.sh" > "$P/xllm_build.log" 2>&1; then echo "xllm build and check_xllm: ok"
else echo "xllm build or check_xllm FAILED"; tail -20 "$P/xllm_build.log"; RC=1; fi
"$VENV_IFM/bin/python" "$HERE/preflight.py" --family ifm --out "$P/ifm.json" || RC=1

python3 - "$P" "$OUT_ROOT/_preflight.json" "$RC" <<'PY'
import json, os, sys
p, out, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
rec = {"system": open(os.path.join(p, "system.txt")).read(), "families": {}, "failures": []}
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
if rc and not rec["failures"]:
    rec["failures"].append("xllm build or check_xllm failed (xllm_build.log)")
rec["ok"] = not rec["failures"] and rc == 0
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps({"preflight_ok": rec["ok"], "failures": rec["failures"]}, indent=1))
PY
exit $RC
