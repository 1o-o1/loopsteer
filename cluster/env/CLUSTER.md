# Target cluster environment

The cluster bundle runs on the same cluster as loopedcot and follows its convention: a venv made with
`python3 -m venv --system-site-packages`, torch from the node image, every other package pinned. The cluster venvs
are built from `requirements_cluster_{ouro,raven,parcae,ifm}.txt` by `setup_cluster.sh`. This file holds the design
notes; `cluster/README_CLUSTER.md` is the operator page (five commands).

## Known

| Fact | Value | Source |
|---|---|---|
| venv form | `python3 -m venv --system-site-packages` | loopedcot `setup_env.sh` lines 3 and 9 |
| torch | from the node image, not pinned | loopedcot `requirements.txt` line 1; `setup_env.sh` line 15 asserts a CUDA build |
| torch build seen | 2.6.0+cu124 | Latent-Loop Transformer `LEDGER.md` line 561 (first GPU check on the cluster) |
| driver | 580.82 | same ledger line |
| GPU memory reported | 98,304 MiB (96 GB); GPU model not recorded | same ledger line |
| transformers | 4.56.2 (Ouro, raven, Parcae venvs) | loopedcot `requirements.txt` line 2 |
| scheduler | Slurm, one GPU per task (`--gres=gpu:1`), partition and account set per site | loopedcot `slurm/gpu_check.sbatch` lines 5 and 13, `slurm/run.sbatch` line 13 |

## Not known

Python version of the node image, CPU architecture, CUDA toolkit (nvcc) version and path, GPU model and compute
capability, whether compute nodes reach the internet, the partition's node names.

## Commands that establish it (any GPU node of the partition)

```
srun -p <partition> -A <account> --gres=gpu:1 --time=0:10:00 bash -c '
  python3 --version; uname -m; cat /etc/os-release | head -2
  python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.__file__)"
  nvcc --version || ls -d /usr/local/cuda* ; module avail cuda 2>&1 | head
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  curl -sI https://huggingface.co | head -1'
sinfo -p <partition> -N -o "%N %G %c %m"
```

`setup_cluster.sh` records the login node's values in `$OUT_ROOT/_env.json`; the preflight job (`preflight.sh`,
first job of `submit_00_xllm.sh`) records the GPU node's values in `$OUT_ROOT/_preflight.json`.

## Python minimum per family

| Family venv | Scripts | Minimum | Why |
|---|---|---|---|
| ouro | k17_varbench (Ouro), k16_path_ouro, k17_freeze, steer_gen, steer_gen_math, merge | 3.9 | pilot venv was 3.9 |
| raven | k17_public, k17_freeze_raven, steer_gen_raven | 3.10 | McLeish and Huginn modeling files use `X \| Y` unions at import |
| ifm | k15_full, k15_pairs, k16_path_ifm, k17_varbench (IFM), k17_freeze_ifm | 3.10 | xllm-loop has `activation: str \| None` in a signature at import; fla-core 0.4.2, transformers 5.8.0, blobfile 3.3.0, nltk 3.9.4 declare >= 3.10 |
| parcae | k17_parcae | 3.11 | parcae-lm 0.1.0 declares Requires-Python >= 3.11 |

All four venvs use the same node interpreter (the one whose site-packages hold torch). If it is older than a family's
minimum, setup stops with the reason; the fix is a node image or module whose python has both torch and the version.

## Package decisions (torch 2.6)

Pins come from the pilot venv of each family with torch, torchvision, triton, `nvidia-*` and `cuda-*` removed (the
node image provides them) and the OS packages of the raven freeze dropped. A pip constraint `torch==<node version>`
keeps any package from replacing the node torch.

| Package | Pilot pin | Cluster pin | Reason |
|---|---|---|---|
| fla-core, flash-linear-attention (ifm) | 0.5.2 | 0.4.2 | 0.5.0 and later declare torch >= 2.7.0. 0.4.2 is the newest release that admits torch 2.6 (Requires-Dist `torch`, no bound; 0.4.0 declared torch >= 2.5); it keeps the two imports xllm makes at load (`fla.modules.conv.triton.causal_conv1d_fwd/_bwd`, `fla.ops.utils.prepare_chunk_indices`) and recommends triton >= 3.2, the triton of torch 2.6 |
| torchvision (ifm, raven) | 0.26.0 | not installed | pins torch 2.11; no script imports it |
| torch (all) | 2.8.0+cu128 / 2.11.0+cu130 | node image | loopedcot convention |
| numpy, pandas, scipy | 2.4.4 / 3.0.2 / 1.17.1 (3.12 venvs), 2.0.2 / 2.3.3 / 1.13.1 (3.9 venv) | same, chosen by Python version | the 3.12 pins need Python >= 3.11, the 3.9 pins have no 3.13 wheel |

No other pinned version declares a torch newer than 2.6: transformers 4.56.2 and 5.8.0 (torch >= 2.4 for 5.8.0),
accelerate 1.10.1 and 1.13.0 (torch >= 2.0), peft 0.17.1 and 0.20.0 (torch >= 1.13), parcae-lm (unbounded torch).
The remote modeling code of Huginn and McLeish uses `torch.nn.attention.flex_attention` (in torch since 2.5).

Believed, not verified (no install was made): that fla-core 0.4.2, transformers 5.8.0 and parcae-lm run on torch 2.6
beyond what their metadata declares, and that xllm-loop compiles with torch 2.6 (its setup.py selects C++17 below
torch 2.10). The preflight job checks every one of these on a GPU node before any array starts.

## Preflight checks (g0_xllm job, alone, before every other job)

1. Node: hostname, `uname -m`, OS, node python, the Ouro venv's python, torch version, CUDA build, cuDNN, torch path,
   nvcc, GPU name, memory, driver, compute capability (`$OUT_ROOT/_preflight/system.txt`).
2. Per family venv (`preflight.py`): Python minimum; torch imports from the node image (not the venv), is a CUDA build,
   sees a GPU; every pinned package at its pinned version; the family's packages and bundle scripts import; one model
   configuration loads (Ouro-1.4B, Huginn: config and remote modeling class; Parcae 370m: ParcaeConfig; IFM no-injection:
   xllm ModelConf and tokenizer path).
3. xllm: `build_xllm.sh` builds the CUDA extension into the IFM venv and `check_xllm.py` loads every IFM checkpoint and
   runs one forward; then the IFM venv check imports xllm and xllm_extension.
4. `$OUT_ROOT/_preflight.json` with `ok` and every failure; exit 1 on any failure, so the job leaves no done marker and
   `submit_10_rest.sh` refuses to start.

## Why `--exclude` and not `--nodelist`

`--nodelist=n1,n2,n3,n4` asks Slurm for a job that holds all four nodes, which a one-GPU array task cannot satisfy.
Setup therefore computes `--exclude=<every other node of the partition>` (read with `sinfo -N -p`), or takes
`--reservation=<name>` (a reservation holding the nodes), and writes it as `SLURM_RESTRICT` into `$OUT_ROOT/cluster.env`.
`submit_chain.sh`, `submit_00_xllm.sh` and `submit_10_rest.sh` pass `--partition`, `--account`, `--output` and
`$SLURM_RESTRICT` on every sbatch command line; the template headers keep placeholders and no script edits a tracked
file. A node added to the partition after setup is not excluded: rerun setup if the partition changed.
`--nodes-expected N` (default 6) sets how many nodes `--nodelist` must name; `CARDS` is 8 per node (48 by default).

## Scripts the task lines run

Groups: the queue groups of `jobs/joblist.tsv` (smoke = `jobs/smoke_*.tsv`). Local imports come from a static scan of
every import, followed recursively. Pilot scripts import their siblings from `cluster/pilot/`; `--tree $LF/tree` puts
`cluster/pilot/tree/` first, so `prod.*` in those scripts is the frozen copy there. The generation lines run
`python -m prod.steer_gen` inside `cluster/pilot/tree/`, and `steer_gen_math.py` and `steer_gen_raven.py` put that tree
first too, so every `prod.*` of the run is the bundle's copy (byte-identical to `loopedcot/prod`; the pair files are in
`cluster/pilot/tree/data_steer/`). Every file below is pinned in `jobs/scripts_sha256.txt`.

| script | groups | local imports | key packages |
|---|---|---|---|
| `cluster/pilot/tree/prod/steer_gen.py` (`-m prod.steer_gen`) | g2_gen, g7_ra23, smoke | prod: common, config, generate, models/ouro, tasks | torch, transformers, numpy, pandas |
| `cluster/pilot/steer_gen_raven.py` | g2_gen | prod: steer_gen, config, generate, models (mcleish, raven), tasks | torch, transformers |
| `cluster/pilot/steer_gen_math.py` | g7_ra23 | steer_gen_raven; prod: steer_gen, tasks | (through prod.steer_gen) |
| `cluster/pilot/gate.py` | g2_gen, g7_ra23, smoke | none | none |
| `cluster/steer_select.py` | g2_gen, smoke | none (runs prod.steer_gen in a subprocess) | none |
| `cluster/pilot/k17_varbench.py` | g1_copy, g6_ifm, smoke | k15_data, k15_smoke, k15_smoke2 | torch, transformers, tokenizers, huggingface_hub, pandas, einops, xllm (IFM) |
| `cluster/vb_slots.py` | g6_ifm, smoke | none | none |
| `cluster/pilot/k17_public.py` | g1_copy, g3_map, g5_huginn, g7_ra23 | k13_edit, k15_data, k16_data, k17_parcae, k17_varbench | torch, transformers, tokenizers, huggingface_hub, pandas |
| `cluster/pilot/k17_parcae.py` | g3_map | k15_data, k16_data, k17_varbench | torch, parcae_lm, tokenizers, huggingface_hub, pandas |
| `cluster/pilot/k15_full.py` | g6_ifm | cf_pool, k13_edit, k15_data, k15_smoke, k15_smoke2 (k10_ifm through them) | torch, transformers |
| `cluster/pilot/k15_pairs.py` | g6_ifm | k10_ifm, k13_edit, k15_smoke, k15_smoke2 | torch, numpy |
| `cluster/pilot/k16_path_ouro.py` | g3_map | k13_edit, k15_data, k15_full, k15_smoke, k15_smoke2 | torch, transformers |
| `cluster/pilot/k16_path_ifm.py` | g6_ifm | k15_data, k15_pairs, k15_smoke, k15_smoke2 | torch, numpy, einops, xllm |
| `cluster/pilot/k17_freeze.py` | g6_ifm, g7_ra23 | k15_data, k16_path_ouro, k17_ko240; tree prod.steer (hooks, interventions, readouts, sites, adapters/ouro) | torch, transformers |
| `cluster/pilot/k17_freeze_ifm.py` | g6_ifm | k10_ifm, k15_data, k15_smoke, k16_path_ouro, k17_freeze | torch (xllm through k10_ifm) |
| `cluster/pilot/k17_freeze_raven.py` | g4_cd | k17_freeze, k17_public; tree prod.models, prod.steer.adapters | torch, transformers, huggingface_hub |
| `cluster/extract/run.py` | every group but g0, g9; smoke | cluster/extract: common, shared, resum and one format module per script | none |
| `cluster/merge.py` | g9_merge | none | none |

Inputs read as data, not imported: `cluster/pilot/data/` (`$DATA_ROOT`), `cluster/pilot/tree/prod/config.yaml` and
`tasks/data/model_revisions.json` (read by the tree's `prod/config.py`), `cluster/pilot/stubs/flash_attn_2_cuda.py`
(appended to `sys.path` by `k10_ifm.py` only when the real module is absent), `cluster/plans/`, `cluster/data/` and
`cluster/pilot/tree/data_steer/` (generation pair files).

## Launch chain (jobs/submit_chain.sh)

| step | Slurm job | refuses when | writes |
|---|---|---|---|
| 1 setup_check | CPU | - | `_chain/setup_check.ok` or `.fail`: cluster.env complete, no placeholder, no folder in a home directory, `SLURM_RESTRICT` set, pins, plan hashes, SETTLE files, venvs, HF_HOME, WEIGHTS_ROOT |
| 2 probe | one five-minute, one-GPU job per node of `NODELIST` (`--nodelist`), afterok 1 | step 1 failed | `_probe/<node>.out`: node=, `nvidia-smi -L`, a torch CUDA matmul, RC= |
| 3 probe_collect | CPU, after 1 and every probe | `_chain/setup_check.ok` missing | `bad_nodes.txt` (RC not 0, "CUDA unknown error", "No devices were found"); submits steps g0, smoke and stage_submit with those nodes excluded |
| g0 | one GPU | `_chain/setup_check.fail` exists | `_done/g0_xllm.s1_build.001` or `_diag/g0_xllm.s1_build.001.txt`; `_preflight.json` |
| smoke | one GPU, afterok g0 | step 1 failed, g0 not done | `_smoke.ok` or `_smoke_fail.txt` (`env/smoke_cluster.sh`) |
| stage_submit | CPU, afterok smoke | `_smoke_fail.txt`, no `_smoke.ok` | `jobs/stage_done.sh` (not with `DONE_SRC=none`), then `submit_10_rest.sh` of the chosen lists |

All paths are under `$OUT_ROOT`. Every sbatch call has `--kill-on-invalid-dep=yes`, so a failed step cancels the rest,
and adds `--exclude` for every node in `bad_nodes.txt` (merged with `SLURM_RESTRICT`; `submit_10_rest.sh` does the same).
A task whose GPU cannot start (`nvidia-smi -L` or its output shows one of the two signatures) adds its node to
`bad_nodes.txt`, writes its diagnosis, exits 75 and requeues itself once (`scontrol update ExcNodeList`, then
`scontrol requeue`: same job id, so the stages waiting on it keep waiting) with that node excluded. A second such
failure is not requeued. `jobs/status.sh` prints the bad nodes and the probe verdicts.
Every task template and chain step loads `$MODULES` when it is set and, on any failure, writes
`$OUT_ROOT/_diag/<task id>.txt` (exit code, failing check, host role, Slurm job and node, GPU name with free and total
memory, host memory, last 50 lines of the task's output in `$OUT_ROOT/_tasklog/`). `jobs/status.sh` counts done,
running, pending and failed tasks per group from these markers, lists the failed ids and prints every diagnosis.

## Smoke gate (env/smoke_cluster.sh, env/smoke_check.py)

Pass needs (i) every in-run identity bit-exact: pinned hashes, every smoke line exits 0, the k17_varbench and steer_gen
harness checks, and for the family lines (raven k17_public mcl_nr D1, Parcae k17_parcae parcae-370m D1, IFM
k17_freeze_ifm D1, `--debug 3`) every hook identity, PC, SZ, PC-site and freeze identity, and for the McLeish
generation line (checks mode and row r00 on 3 problems of the shipped selection) every checks case and row r00 exit 0; and (ii) every e value
(results.csv `estimate`, `random_control_estimate`, `positive_control_value`) and every margin (logit_B - logit_A of
the k17_varbench items and rows) of the two reference pipelines within 0.001 of `cluster/SMOKE_VALUES.json` (bound to
`SMOKE_HASHES.txt`). The family lines have no reference values: their values are reported, not gated.
`SMOKE_OVERRIDE_REASON='<text>'` passes a tolerance miss and records the reason and full diff in
`$OUT_ROOT/_smoke_override.txt`; identity failures cannot be overridden.

## Job lists: jobs/ and jobs/nolocal/

`jobs/` (`python cluster/gen_jobs.py`) leaves out the 74 tasks that ran on our cards (`cluster/done_local.json`); their
outputs are staged from `DONE_SRC` by `jobs/stage_done.sh` (9 folders, 519 files, 742.5 MiB). `jobs/nolocal/`
(`python cluster/gen_jobs.py --no-local-done`) runs them on the cluster under the same ids (`DONE_SRC=none`). The
McLeish generation rows read the 237-problem selection of the earlier McLeish screen, which ships in
`cluster/pilot/data/mcleish_setting/items.json`; each row line copies it into its output folder unless a staged copy
is there. `cluster/reference_local/` holds the results.csv of every cell finished on our cards (18 cells, sha256 in
`MANIFEST.json`); the merge job compares each cluster rerun with it in `$OUT_ROOT/merged/local_vs_cluster.csv`
(largest absolute difference of estimate, bounds and controls, n_items and n_clusters equality; reported, not gated),
and `jobs/status.sh` prints that file.
Staging transfer (from the lab host that holds the pilot artifact folder; placeholders in angle brackets):

```
python3 -c "import json; print('\n'.join(sorted({e['pulled'] for e in json.load(open('<PROJECT_DIR>/repo/loopedcot/cluster/done_local.json'))['jobs'] if e.get('pulled')})))" > <OUT_ROOT>/pulled.txt
rsync -a -r --files-from=<OUT_ROOT>/pulled.txt <lab-user>@<lab-host>:<pilot-artifact-folder>/ <OUT_ROOT>/done_src/
DONE_SRC=<OUT_ROOT>/done_src bash <PROJECT_DIR>/repo/loopedcot/cluster/jobs/submit_chain.sh
```

## Run window and size

Launches until 2026-10-12 11:59; jobs may run until 2026-10-13 07:00 local time. The run uses at most the nodes of
`--nodelist` (6 by default, 48 GPUs, one task per GPU); the rest of the cluster stays free.
