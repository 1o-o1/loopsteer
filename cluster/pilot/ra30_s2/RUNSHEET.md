# RA-30 run sheet (for the runner)

RA-30 = edit the intermediate value in the chain, with the neuron arm set, Ouro-1.4B base, GSM8K then ASDiv
(plan_ra30.md). Start only when the card is free. One card, one process at a time. Do not edit any file in the
folder. If a step fails, stop and report; do not rerun with changes.

## 1. Copy the folder to the GPU host

Copy `loop-flow/src/ra30/` (without `out/` and `__pycache__/`) to `/var/tmp/ra30` on the GPU host. Never put it in a
home folder: run.sh refuses a folder inside the home directory.

    rsync -a --exclude out --exclude __pycache__ loop-flow/src/ra30/ <gpu-host>:/var/tmp/ra30/
    cd /var/tmp/ra30 && sha256sum plan_ra30.md ra30.py run.sh lib/arms.py lib/chain.py lib/ra30_data.py && cat plan_ra30.sha256

Compare the sums with the laptop copy, and the plan sum with `plan_ra30.sha256`, before you continue.

## 2. Python

The Ouro venv of RUN 1c and RA-28 (torch 2.11.0+cu130, transformers 4.56.2). Set the absolute path, then check:

    export PY_OURO=<absolute path of .venv>/bin/python
    "$PY_OURO" -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

Expected: True and transformers 4.56.2 (the run stops with exit 2 on another version). Ouro-1.4B loads from the
Hugging Face cache of the host at revision 574fa66. run.sh sets HF_HUB_OFFLINE=1, so nothing is downloaded. If RUN
1c set HF_HOME, export the same value. Every token id is built on this host's tokenizer (step 3); never copy
token-id files from the laptop.

## 3. Before each GPU step

    nvidia-smi --query-compute-apps=pid,used_memory --format=csv; free -g

Launch only when no other process uses the card and at least 30 GB are free.

## 4. Commands, in order

All in /var/tmp/ra30 with PY_OURO exported. GSM8K first; ASDiv only after the RA keeps it.

| step | command | expected time | pass condition |
|---|---|---|---|
| 1 | `bash run.sh test` | about 1 min, CPU | ends `OK (skipped=7)` (54 tests; 6 float32-only tests in the bf16 class and the laptop-chain tokenizer test are skipped) |
| 2 | `bash run.sh hash` | seconds | plan_sha256 equals `plan_ra30.sha256`; send it for the LEDGER before step 3 |
| 3 | `bash run.sh tokens gsm8k` | under 1 min, CPU | exit 0; `out/tokens/gsm8k_check.json` field `pass` true |
| 4 | (done 2026-10-12: the GSM8K pilot under plan 7f96eb63 passed every gate; the full run accepts it) | | |
| 5 | `bash run.sh hash` again after copying the re-frozen plan (section 1) | seconds | plan_sha256 equals `plan_ra30.sha256` |
| 6 | `nohup bash run.sh mains > out/logs/nohup_mains.txt 2>&1 &` | about 5 h | runs full gsm8k (about 2.1 h), tokens asdiv, pilot asdiv (about 40 min, 20 items), full asdiv (about 2.3 h) and stops at the first failure; `out/logs/mains.log` names the step and its exit |

Ouro-2.6B (overflow, only after the 1.4B mains end and only when the RA sends it; plan section 6b), GSM8K:

| step | command | expected time | pass condition |
|---|---|---|---|
| 2a | `bash run.sh tokens2 gsm8k` | under 1 min, CPU only (the step hides the card) | exit 0; `out/ouro_2_6b_base/tokens/gsm8k_check.json` field `pass` true |
| 2b | `nohup bash run.sh pilot2 gsm8k > out/logs/nohup_pilot2_gsm8k.txt 2>&1 &` | about 35 min | `gates PASS`, ends `[ra30] done`, exit 0; report as for a pilot |
| 2c | after the RA keeps it: `nohup bash run.sh full2 gsm8k > out/logs/nohup_full2_gsm8k.txt 2>&1 &` | about 3.0 h | ends `[ra30] done`, exit 0 |

ASDiv on 2.6B (breadth cell, same arms): the same three steps with `asdiv` in place of `gsm8k`: `bash run.sh tokens2
asdiv` (CPU only), pilot2 asdiv (about 55 min), full2 asdiv (about 3.2 h; prints `ASDiv under 80: gate failure` and
exits 3 if its pilot projects fewer than 80 eligible items). Outputs go to out/ouro_2_6b_base/; the 1.4B folders are
not touched.

McLeish r32 GSM8K (PI breadth cell, plan section 6c; only when the RA sends it):

| step | command | expected time | pass condition |
|---|---|---|---|
| 3a | `bash run.sh tokens_mcl` | under 1 min, CPU only (the step hides the card) | exit 0; `out/mcleish_llama32_r32/tokens/gsm8k_check.json` field `pass` true |
| 3b | `nohup bash run.sh pilot_mcl > out/logs/nohup_pilot_mcl.txt 2>&1 &` | about 2.2 h | `gates PASS`, ends `[ra30] done`, exit 0 (exit 3 with `G6 failed` if the text-edit ceiling is under 0.5) |
| 3c | after the RA keeps it: `nohup bash run.sh full_mcl > out/logs/nohup_full_mcl.txt 2>&1 &` | about 7.9 h | ends `[ra30] done`; exit 3 with `GSM8K under 80: gate failure` if the pilot projects fewer than 80 eligible items |

The McLeish venv is the same .venv (transformers 4.56.2). Report as for an Ouro pilot, plus the three McLeish
predictions in `summary.md`.

If the ASDiv pilot projects fewer than 80 eligible items, `full asdiv` prints `ASDiv under 80: gate failure` and
exits 3: mains stops there; that is a reported result, not an error to fix. Each step can also be run alone in the
same order (`bash run.sh full gsm8k`, `bash run.sh tokens asdiv`, `bash run.sh pilot asdiv`, `bash run.sh full
asdiv`). The full GSM8K log prints `neuron non-vacuity nc_full ...` and `... ns_full ...` lines after the gates.

The full run screens every pool item again in its own folder (out/full/<ds>; the pilot folder is not reused) and
reads the pilot's horizon-cap decision from out/pilot/<ds>/checks.json. Stop a run only with
`pkill -TERM -f 'ra30.py --run'`. It stops after the current item with exit code 5. Run the same command again to
resume; it skips the items already screened or written (the gates run again first, on 20 items in the pilot and 5
in the full run).

## 5. Exit codes

0 done. 2 plan sha256 mismatch, a pin difference (transformers, Ouro revision or modeling md5, dtype), a data file
that differs from data/sources.json, more than 1 percent of prompts not rebuilding on the host tokenizer, a missing
tokens step, or an input that changed on resume. 3 gate failed: G1 to G5 before any scored row (`gates.json`, items
with `pass` false), or G6 in the pilot (`g6.json`: the text-edit ceiling below 0.5; the summary is still written).
4 an identity failed on a scored row. 5 stopped by SIGTERM (resumable). On 2, 3 or 4: stop and report; do not
rerun.

## 6. What to report

After each pilot (out/pilot/<ds>/):
1. Exit codes, the plan sha256 of step 2, `out/tokens/<ds>_check.json`.
2. `gates.json`: `pass`, `G2_identical_share`, both `identity_*` blocks, `norm_ratio_applied_pass_share`; for the
   first 3 items the `G4` block (the tokens at P).
3. `summary.md` in full (the arm table with base first; CA of cv1, kv1 and text side by side; P3, P4, P5; neuron
   shares and the t5 - r5 gap).
4. `checks.json` in full: dropped counts with reasons, eligible rate and projected eligible count (the 80-item
   gate), seconds per item and per arm and item, `projected_full_card_hours`, the horizon cap decision, G2b.
5. `g6.json` and `examples.json` (5 chains: clean, text, cv1, kv1).

After the full run: exit code, `hash_line.json`, `gates.json` (`pass`), `checks.json`, `summary.md`, sha256 of
`results.csv` and `rows.jsonl`, the card-hours used (sum of `sec_item` and `sec_clean`), any line with `FAIL`.

Then pull the outputs back to the laptop:

    rsync -a <gpu-host>:/var/tmp/ra30/out/ loop-flow/artifacts_spark/ra30/
