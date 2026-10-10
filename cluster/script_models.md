# Pilot scripts: models and datasets they load today (RA-20 Amendment 2, manifest "Script per cell")

Read on 2026-10-08 from the laptop copies (`loop-flow/src/`, `loopedcot/prod/steer_gen.py`, branch steer). The venv
names below are those of the pilot hosts ("fast-host venv", "shared .venv", "IFM venv", "Parcae venv"); their package
lists are not shipped. The cluster venvs are `cluster/env/requirements_cluster_<family>.txt`.
Venv per script comes from the script's docstring, the run scripts on the hosts and the versions in the records.
Dataset codes: D1 facts (capital, city), CF CounterFact, D2 addition, D3 countries (same-subject attributes),
D4 code reads and sums, D5 two-step math, VB / VarBench GSM8K copy, VBD GSM8K derived.

## 1. Script to models

| Script | Models it loads today (registry name) | Venv | Datasets it accepts | Selection flags |
|---|---|---|---|---|
| loopedcot/prod/steer_gen.py | ouro_1_4b_base, ouro_2_6b_base (HF cache, pinned revision); ouro_1_4b_think (local copy named by `K17_THINK_DIR` or `--think-dir`) | fast-host venv (Python 3.9) | arm a: GSM8K VarBench pairs (default) or SVAMP pairs with `K17_DATASET=svamp` and a `--config-json` that sets `dataset: svamp`; arm b: GSM8K, MATH-500 (LoopCD settings, loop-2 teacher-forced test) | `--setting a1/a2/a3/b1/b2/c`, `--row`, `--config-json`, `--n` |
| loop-flow/src/k17_varbench.py | ouro, ouro_think, ouro26, ouro26_think (HF cache, no revision pin, `--model_dir(s)` optional); ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 (k15_smoke.IfmM, k10_ifm) | Ouro: fast-host venv; IFM: IFM venv | VarBench copy (always), plus any of VBD, D1, D2, D4 (`--datasets`, default VBD,D2,D1,D4); `--pool s200/problems/pairs` | `--models`, `--datasets`, `--phase W/M/all`, `--pool`; no `--shard` yet |
| loop-flow/src/k16_path_ouro.py | ouro, ouro26, ouro_think (HF cache, eager, no revision pin) | fast-host venv | D1, D2, D4, D5 (`--datasets`); T3 sentiment on 50 NEUTRAL150 prompts (`--t3n`) | `--models`, `--datasets`, `--arms` |
| loop-flow/src/k16_path_ifm.py | ifm_p1_ouro (arms Y1, Y3, W2), ifm_d112 (Y1, W2), ifm_p1_mid (Y3, W2), ifm_p1_ouro_inj, ifm_p1_mid_inj, ifm_p2m_diag (Y2 only) | IFM venv | D1, D2, D4, D5 (`--datasets`; D4 and D5 from `--data_dir`) | `--models`, `--datasets` |
| loop-flow/src/k17_public.py | group mcl_nr = mcleish (K 32) + llama_nr; group huginn = huginn (K 32, plus K 16 and K 8 runners); group ouro_think | shared .venv (also ran on the fast host) | mcl_nr and huginn: D1, CF, D2, D4, D5, VB; ouro_think: D1, D2, D4 only (other names are skipped silently) | `--model`, `--datasets`, `--stage scan/rows/merge`, `--shard k/n`, `--targets` (caps; full pool needs large values) |
| loop-flow/src/k17_parcae.py | parcae-140m, parcae-370m, parcae-770m, parcae-1.3b | Parcae venv | D1, D2, D4, D5, VB | `--model`, `--datasets`, `--targets` |
| loop-flow/src/k17_freeze.py | ouro_1_4b_base only (prod EagerOuroAdapter from `--tree`) | fast-host venv | D1, D2, D4, D5 | `--datasets`; no `--model` |
| loop-flow/src/k17_freeze_raven.py | huginn (huginn_0125), mcleish (mcleish_llama32_r32), through prod adapters from `--tree` | shared .venv | D1, D2 (k17_public candidate files) | `--model`, `--datasets` |
| loop-flow/src/k17_x4e.py | ByteDance/Ouro-1.4B only (fixed REPO) | fast-host venv | D1, D2 (the freeze run's selection files) | no `--model`, no `--datasets` |
| loop-flow/src/k17_trace.py | ByteDance/Ouro-1.4B only (fixed REPO) | fast-host venv or shared .venv | D1, D2 (freeze selection files, sha256 pinned) | no `--model` |
| loop-flow/src/k17_reread.py | ByteDance/Ouro-1.4B only (fixed REPO) | fast-host venv or shared .venv | sentiment on k15_data.NEUTRAL150 (150 prompts); classifier distilbert SST-2 (`--clf`) | no `--model` |
| loop-flow/src/k17_control.py | ouro, ouro26 (HF cache), ifm_p1_ouro, ifm_p1_ouro_inj (k10_ifm) | Ouro: fast-host venv; IFM: IFM venv | D3 countries (`--data D3.jsonl`) | `--model`, `--phases`, `--editors` |

The manifest's "the RA-19 path" for generation on McLeish and Huginn does not exist in the repo or on either host.

## 2. What the scripts cover in the manifest

1. Grid A generation: Ouro 1.4B base and 2.6B base on GSM8K and SVAMP; Ouro 1.4B Thinking on GSM8K and SVAMP only
   through `--config-json` (setting a3 runs 7 of the run-1 rows, 30 scored problems, and a GSM8K SETTLE cell file;
   a SETTLE Thinking SVAMP cell file was not checked). The loop-2 contrast on Ouro 1.4B: setting b2, mode tf.
   Everything else in grid A is a time skip.
2. Grid B GSM8K copy and derived: k17_varbench on the four Ouro checkpoints, IFM no-injection, injection twin and
   d112; the copy (VB) only through k17_public (McLeish, twin, Huginn) and k17_parcae (Parcae).
3. Grid B edit-site map: Ouro 1.4B base, 2.6B base and 1.4B Thinking on D1, D2, D4, D5 (k16_path_ouro); IFM
   no-injection, injection twin, middle pair, 819M fixed-pln5 and d112 on D1, D2, D4, D5 (k16_path_ifm); McLeish,
   twin and Huginn on all six datasets (k17_public); Parcae on D1 and VB (k17_parcae). No listed script runs CF on
   Ouro Thinking, IFM or Parcae.
4. C and D: branch-freeze on Ouro 1.4B base (D1, D2, D4, D5), Huginn and McLeish (D1, D2); carried-state and trace
   on Ouro 1.4B base (D1, D2); re-read on Ouro 1.4B base (NEUTRAL150, not the new 300-prompt set); control on Ouro
   1.4B, 2.6B, IFM no-injection and injection twin (D3).

Every other manifest cell is in `cluster/cells_skipped.json` with reason "time" and the script that needs a change.

## 3. Host-specific strings that break on the cluster (not edited)

Status 2026-10-09: the job list runs the copies in cluster/pilot/, where every line below is patched (paths from
XLLM_ROOT, IFM_CKPT_ROOT or WEIGHTS_ROOT, HF_HOME, DATA_ROOT; RUN_ROLE instead of the host name; pinned Ouro
revisions). The table describes the laptop originals in loop-flow/src, which are not edited.

Only code lines are listed; docstrings and comments that name hosts are left out.

| File | Line | What | Effect on the cluster |
|---|---|---|---|
| loop-flow/src/k10_ifm.py (imported by k17_varbench, k16_path_ifm, k17_control through k15_smoke.IfmM) | 23 | `XLLM`: fixed path to the xllm-loop clone under the pilot user's home | IFM import fails unless the clone is at that path |
| loop-flow/src/k10_ifm.py | 24 | `CKPT`: fixed path to the IFM checkpoint folders under the pilot user's home | every IFM load fails; must point at `$WEIGHTS_ROOT` |
| loop-flow/src/k17_varbench.py | 99 (used by 100 and 106) | `DS`: Windows laptop path to loopedcot/data_steer (pairs file, IFM tokenizer) | build steps (`--build_items`, `--build_vbd`) fail; runs read the built `--data` folder |
| loop-flow/src/k17_varbench.py | 101-102 | `GSM_DIR`: user-home HF cache path of the GSM8K dataset snapshot | build steps fail |
| loop-flow/src/k17_varbench.py | 103 | `PROMPT_FILE`: Windows laptop path to prod/tasks/data/prompt_gsm8k.txt | build steps fail |
| loop-flow/src/k17_varbench.py | 104-105 | `TOK_OURO`: user-home HF cache path of the Ouro-1.4B tokenizer | build steps fail |
| loop-flow/src/k17_varbench.py | 900-901, 903 | selftest input holds a literal host name, a user name and a user home path | no break; a host string sits in the code that is copied into the output tree (RA-20 item 6) |
| loop-flow/src/k16_path_ouro.py | 685, 730, 1155 | `socket.gethostname()` written to the result and proc files | no break; host name enters records (RA-20 item 6) |
| loop-flow/src/k16_path_ouro.py | 791-792 | checks text names both pilot hosts and the Thinking copy folder | no break; host names enter checks.json |
| loop-flow/src/k16_path_ifm.py | 564, 683 | `socket.gethostname()` written to proc and result files | no break; host name enters records |
| loop-flow/src/k17_public.py | 305 (used at 410) | `HUB`: user-home HF cache path; ignores `HF_HOME` | McLeish, twin, Huginn and Ouro Thinking loads fail unless the cache is at that path |
| loop-flow/src/k17_public.py | 399-404 | `LAPTOP_TOK`: user-home HF cache tokenizer paths | `--build_items` fails; runs read the built `--data` folder |
| loop-flow/src/k17_parcae.py | 68 (used at 75 and 1019) | `HUB`: user-home HF cache path; ignores `HF_HOME` | Parcae model and default tokenizer (`--tok`) loads fail unless the cache is at that path |
| loop-flow/src/k17_parcae.py | 2173 | venv path under the pilot project root written to checks.json | no break; a home-relative path enters records |
| loop-flow/src/k17_freeze_raven.py | 309 | user-home HF cache path for the raven modeling-file hash; `os.listdir` on it | run fails with FileNotFoundError unless the cache is at that path |

No host-specific string in code: steer_gen.py (needs `--settle-dir`, `--out` and `K17_THINK_DIR` for Thinking),
k17_freeze.py (needs `--tree`), k17_x4e.py, k17_trace.py, k17_reread.py, k17_control.py (its IFM path goes through
k10_ifm lines 23-24).

## 4. Other findings for the array generator

1. Script copies differ: k17_varbench.py differs between the laptop and both hosts; k17_control.py differs on the fast
   host; k17_x4e.py differs on the shared machine (md5 compare, 2026-10-08). The cluster copy must be chosen and hashed.
2. `k16_path_ouro.load_ouro` (also used by k17_x4e, k17_trace, k17_reread) and k17_varbench's Ouro loader pass no
   revision: they load whatever refs/main says in the HF cache.
3. k16_path_ouro and k16_path_ifm run the pathway arms (Y1 2x2 at Llast@1 with knockouts, Y3 undo, W2 window, Y2
   input protocols), not a donor patch at every site of loops 1 to K at strengths 0.25 and 1. On the injection twin,
   the middle injection model and 819M fixed-pln5, k16_path_ifm runs Y2 only. The manifest assigns these scripts to
   the edit-site map; the RA chat should confirm that their arms are the cells it wants.
4. k17_public `--targets` values are caps. The full pool needs caps at or above the pool size on every dataset.
