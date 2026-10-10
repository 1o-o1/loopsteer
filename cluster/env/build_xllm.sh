#!/bin/bash
# Build xllm-loop (the IFM model code) into the IFM venv on the cluster architecture, then check the import and one
# IFM checkpoint load and forward (cluster/env/check_xllm.py). The IFM jobs of the job list wait for this job
# (afterok). A build made on another architecture is not reused.
#
# Environment: VENV (IFM venv, cluster/env/requirements_cluster_ifm.txt), XLLM_ROOT (clone folder), REPO, WEIGHTS_ROOT or
# IFM_CKPT_ROOT (checkpoint folders), CUDA_HOME (nvcc of the CUDA the venv's torch was built with),
# TORCH_CUDA_ARCH_LIST (compute capability of the cluster GPU, for example 9.0), optional MAX_JOBS.
set -euo pipefail
XLLM_URL=https://github.com/ifm-ai/xllm-loop.git
XLLM_COMMIT=3af99f493e14fba162625d5dd687abea551a299e      # the commit of the pilot runs
: "${VENV:?}" "${XLLM_ROOT:?}" "${REPO:?}" "${TORCH_CUDA_ARCH_LIST:?}"
if [ ! -d "$XLLM_ROOT/.git" ]; then
  git clone "$XLLM_URL" "$XLLM_ROOT"
fi
git -C "$XLLM_ROOT" fetch --quiet origin "$XLLM_COMMIT" || true
git -C "$XLLM_ROOT" checkout --quiet --detach "$XLLM_COMMIT"
[ -z "$(git -C "$XLLM_ROOT" status --porcelain --untracked-files=no)" ] || { echo "xllm-loop clone has local changes"; exit 1; }
rm -f "$XLLM_ROOT"/xllm_extension*.so                      # never reuse an extension built elsewhere
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH" MAX_JOBS="${MAX_JOBS:-16}"
cd "$XLLM_ROOT"
"$VENV/bin/pip" install --no-build-isolation --no-deps -e .
cd - > /dev/null
"$VENV/bin/python" "$REPO/loopedcot/cluster/env/check_xllm.py"
