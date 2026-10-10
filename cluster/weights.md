# Weights list for the cluster run (RA-20 Amendment 2 item 3)

Read on 2026-10-08 from the HF hub caches of both pilot hosts, from the IFM checkpoint folders on the shared
machine, and from `prod/tasks/data/model_revisions.json`. Machine-readable copy: `cluster/weights.json`.

Placeholders: `$HF_HOME` is the Hugging Face home on the cluster (snapshots under `$HF_HOME/hub`). `$WEIGHTS_ROOT`
is the folder that holds the IFM checkpoint folders copied to the cluster. Sizes are bytes on disk of the snapshot
(symlinks followed) or of the folder without its `.cache` subfolder.

## Hub checkpoints (load from `$HF_HOME/hub`)

| Manifest model | Hub id | Revision (commit) | Revision source | Size (bytes) |
|---|---|---|---|---|
| Ouro 1.4B base | ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | model_revisions.json, equal to refs/main on both hosts | 2,875,038,874 |
| Ouro 2.6B base | ByteDance/Ouro-2.6B | 1ed04250da1a9936042725d302e81c8fa2ab5abd | model_revisions.json, equal to refs/main on both hosts | 5,341,718,984 |
| Ouro 1.4B Thinking | ByteDance/Ouro-1.4B-Thinking | 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d | model_revisions.json, equal to refs/main (shared machine) | 2,875,359,612 |
| Ouro 2.6B Thinking | ByteDance/Ouro-2.6B-Thinking | f1edd81e7ac41355db670500ceaf204e0f73af68 | model_revisions.json, equal to refs/main (shared machine) | 5,342,034,960 |
| McLeish r32 | smcleish/Recurrent-Llama-3.2-train-recurrence-32 | a5f6f126e9d9302445346844f1d7d29d111a06eb | model_revisions.json, equal to refs/main on both hosts | 5,558,274,592 |
| McLeish non-recurrent twin | smcleish/Llama-3.2-non-recurrent-posttrained | 004e96ace360c1a37c08f4555227136e845087b3 | refs/main on both hosts, equal to k17_public.py CKPT | 6,011,221,474 |
| Huginn K32 | tomg-group-umd/huginn-0125 | bb6621b65e90b6a4b9b29ef88dc83866d450470c | model_revisions.json, equal to refs/main on both hosts | 15,651,519,639 |
| Parcae 370m | SandyResearch/parcae-370m | 439284464ee4999bd1f762da7d044613a4828efe | only snapshot folder (no refs/main), equal to k17_parcae.py SNAP | 1,553,103,203 |
| Parcae 1.3b | SandyResearch/parcae-1.3b | 517f8e6add927e3d73e15beec13a5b500537b9cc | only snapshot folder (no refs/main), equal to k17_parcae.py SNAP | 5,355,492,006 |

Path pattern: `$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>`. Subtotal 50,563,763,344 bytes.

Auxiliary hub file: the Parcae tokenizer `SandyResearch/parcae-tokenizer` at 6247b5d0592876b73660f34c2dd16c9db9e9045c
(2,326,980 bytes; tokenizer.json md5 d1979ceeb54b76acadfc6884f90ea029, as k17_parcae.py checks).

The two Thinking checkpoints are not in the fast-host HF cache. `steer_gen.py` loads Ouro 1.4B Thinking from a local
copy named by the environment variable `K17_THINK_DIR`.

## IFM checkpoints (copy to `$WEIGHTS_ROOT`)

Format: xllm-loop native folder (`config.json`, `model*.safetensors`, `model.safetensors.index.json`,
`artifact_manifest.json`, `tokenizer/`, licence files). Each folder holds huggingface_hub download metadata
(`.cache/huggingface/download/*.metadata`); the first line of every metadata file in a folder gives the same commit,
and that commit equals refs/main of the matching (weights-less) HF cache entry `models--IFM--<folder>`.

| Manifest model | Registry name (k10_ifm) | Folder under `$WEIGHTS_ROOT` | Hub id | Hub commit | Size (bytes) |
|---|---|---|---|---|---|
| IFM no-injection | ifm_p1_ouro | LoopedLM-P1-dense-ouro-336b | IFM/LoopedLM-P1-dense-ouro-336b | e87db19b5e9d5bc188ca2230e0299f260ba579ee | 3,014,160,340 |
| IFM injection twin | ifm_p1_ouro_inj | LoopedLM-P1-dense-ouro-raw-injection-336b | IFM/LoopedLM-P1-dense-ouro-raw-injection-336b | a218b790436c34985202a8ccb80c20ec42775982 | 3,018,885,801 |
| IFM middle pair, no injection | ifm_p1_mid | LoopedLM-P1-dense-middle-336b | IFM/LoopedLM-P1-dense-middle-336b | 38a1464ce82cbb68e3d088fc2b87e5b2c5427482 | 3,014,160,708 |
| IFM middle pair, prelude injection | ifm_p1_mid_inj | LoopedLM-P1-dense-middle-prelude-injection-336b | IFM/LoopedLM-P1-dense-middle-prelude-injection-336b | dfaf77f443c9887cf5fae13bbbd88848df8d274e | 3,018,886,119 |
| 819M design, fixed-r5 | ifm_p2m_r5 | LoopedLM-P2-huginn-m-fixed-r5 | IFM/LoopedLM-P2-huginn-m-fixed-r5 | eeed5e5f525608fa7c523bd191e8508f6cc50102 | 1,644,018,933 |
| 819M design, fixed-pln5 | ifm_p2m_diag | LoopedLM-P2-huginn-m-fixed-pln5 | IFM/LoopedLM-P2-huginn-m-fixed-pln5 | e783483e7f8f5c279277c2955a5cd1f936ead99b | 1,644,019,051 |
| 819M design, orthoinj | ifm_p2m_ortho | LoopedLM-P2-huginn-m-orthoinj | IFM/LoopedLM-P2-huginn-m-orthoinj | 910e79f865107d3c6b834e4071ea990e593e97cc | 1,644,019,110 |
| d28 | ifm_d28 (added by k15_smoke.IfmM) | LoopedLM-P1-dense-d28-336b | IFM/LoopedLM-P1-dense-d28-336b | 0ae1ce772b7126a31de22f1daab6ee12aac8ae99 | 3,014,160,336 |
| d112 | ifm_d112 (added by k15_smoke.IfmM) | LoopedLM-P1-dense-d112-336b | IFM/LoopedLM-P1-dense-d112-336b | 6792a315287b01127b9b6d509a66cdd6bb06f939 | 7,374,789,238 |

Subtotal 27,387,099,636 bytes. The commits match the RA-13 runbook table (`prod/steer/RUNBOOK.md` section 5).

IFM code: xllm-loop at commit 3af99f493e14fba162625d5dd687abea551a299e (https://github.com/ifm-ai/xllm-loop.git).
`k10_ifm.py` reads the folders from its constant `CKPT` and the code from its constant `XLLM`; both are fixed paths
under the pilot user's home today, so the cluster run must point them at `$WEIGHTS_ROOT` and the xllm clone.

## Totals

18 checkpoints, 18 revisions known, 77,950,862,980 bytes (about 78 GB) plus the Parcae tokenizer.

## Not needed for the manifest

Present on the shared machine, not in the manifest: IFM/LoopedLM-P1-dense-huginn-336b, IFM/LoopedLM-P1-moe-ouro-336b,
IFM/LoopedLM-P2-huginn-s-fixed-r5, IFM/LoopedLM-P2-huginn-s-fixed-pln5, SandyResearch/parcae-140m and parcae-770m.

## Loader notes for the cluster

1. Ouro models load through `from_pretrained(<hub id>)` with the HF cache; set `HF_HOME` and `HF_HUB_OFFLINE=1`.
   `k16_path_ouro.load_ouro` (also used by k17_x4e, k17_trace, k17_reread) passes no revision, so it loads refs/main;
   copy the cache with refs/main equal to the revision above.
2. `k17_public.py`, `k17_parcae.py` and `k17_freeze_raven.py` do not read `HF_HOME`: they build the snapshot path
   from a fixed user-home cache path. On the cluster the cache must be at that path or the constant must change.
3. Parcae snapshots have no refs/main; `k17_parcae.py` addresses them by snapshot folder, so copy the folder names
   unchanged.
