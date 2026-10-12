# RA-29 run sheet (for the runner)

RA-29 = MCQ late edit on Ouro-1.4B and Ouro-2.6B base: last-token and option-block donor patches at each loop and inside loop 4,
teacher-forced, letter-margin readout (plan_ra29.md). Start only when the card is free. One card, one process at a
time. Do not edit any file in the folder. If a step fails, stop and report; do not rerun with changes.

## 1. Copy the folder to the GPU host

Copy `loop-flow/src/ra29/` (without `out/` and `__pycache__/`) to `/var/tmp/ra29` on the GPU host. Never put it in a
home folder: run.sh refuses a folder inside the home directory.

    rsync -a --exclude out --exclude __pycache__ loop-flow/src/ra29/ <gpu-host>:/var/tmp/ra29/
    cd /var/tmp/ra29 && sha256sum plan_ra29.md ra29.py run.sh data/items_aqua.json data/items_csqa.json && cat plan_ra29.sha256

Compare the sums with the laptop copy, and the plan sum with `plan_ra29.sha256`, before you continue.

## 2. Python

The Ouro venv RA-28 and RUN 1c used: the `.venv` folder next to the project's `loop_flow` folder on the GPU host
(torch 2.11.0+cu130, transformers 4.56.2). Set the absolute path, then check:

    export PY_OURO=<absolute path of .venv>/bin/python
    "$PY_OURO" -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

Expected: True and transformers 4.56.2 (the run stops with exit 2 on another version). Ouro-1.4B (revision 574fa66c)
and Ouro-2.6B (revision 1ed04250) load from the Hugging Face cache of the host. run.sh sets HF_HUB_OFFLINE=1, so a
missing revision fails at load and nothing is downloaded. If RA-28 set HF_HOME, export the same value.

## 3. Before each GPU step

    nvidia-smi --query-compute-apps=pid,used_memory --format=csv; free -g

Launch only when no other process uses the card and at least 30 GB are free.

## 4. Commands, in order

All in /var/tmp/ra29 with PY_OURO exported. MODEL is ouro14 or ouro26; DATASET is aqua, csqa or arc.

The pilots of steps 3 and 4 ran under plan 960f88cd and passed (RA decision: RA-29 kept on three datasets). Keep
`out/` on the host when you copy the new folder (section 1 excludes `out`, so the pilots stay). This plan adds two
arms; the full runs accept those pilots, and the new arms pass the identity gate at the start of every run.

| step | command | expected time | pass condition |
|---|---|---|---|
| 1 | `bash run.sh test` | under 2 min, CPU | ends `OK` (29 tests; the T2 tokenizer tests may report `skipped` if the tokenizer is not cached under this python's HF_HOME) |
| 2 | `bash run.sh hash` | seconds | plan_sha256 equals `plan_ra29.sha256`; send it for the LEDGER before step 5 |
| 3 | `bash run.sh pilots` | done | (ran under plan 960f88cd; do not rerun) |
| 4 | report the pilots, wait for the RA decision | done | RA-29 kept |
| 5 | `nohup bash run.sh mains > out/logs/nohup_mains.txt 2>&1 &` | about 2.2 card-hours | each run logs `[ra29] identity gate PASS` then ends `[ra29] done`; the aqua26 line says which way it went; the log ends `[ra29] mains done`; exit 0 |
| 6 | `bash run.sh summarize MODEL DATASET full` for each full run | seconds each, CPU | exit 0; rewrites results.csv, results.json, checks.json, summary.md |

`mains` runs, in order, stopping at the first failure:
1. `full ouro14 aqua` (about 0.15 card-hours);
2. `full ouro14 csqa` (about 0.95 card-hours);
3. `pilot ouro14 arc` if out/ouro14/arc/pilot/debug_gate.json is missing (about 3 minutes), then `full ouro14 arc`
   (about 0.7 card-hours, an estimate until the ARC pilot measures it);
4. `aqua26`: reads the scored count of step 1 (checks.json counts.n_kept, after the gap drop) and runs
   `full ouro26 aqua` (about 0.4 card-hours) only if it is below 80. It prints which way it went.
A failed step stops `mains`; after a SIGTERM (exit 5) run `bash run.sh mains` again: finished runs resume with no
new items, and the rest continue. Single steps: `bash run.sh full MODEL DATASET`, `bash run.sh aqua26`.

Breadth cells (cluster unit, one job per cell, not on the single card): `bash run.sh cell ouro14 mmlu`,
`cell ouro26 mmlu`, `cell ouro14 hellaswag`, `cell ouro26 hellaswag`. Each job runs the cell's pilot (if its
debug_gate.json is missing), projects the scored count from the pilot, starts the full run only if it is 80 or
more, and prints the result to out/logs/cells.log. Projected card-hours: 1.2, 2.75, 2.0 and 4.75 at the single
card's rate, about 0.4, 0.9, 0.65 and 1.6 on an A100 [belief, factor 3]. Each cell writes only out/MODEL/DATASET/.

Each pilot stops at 30 scored items or 120 tried items. A full run needs its own pilot's `debug_gate.json` (exit 3
otherwise).

Stop a run only with `pkill -TERM -f 'ra29.py --run'`. It stops after the current item with exit code 5. Run the
same command again to resume; it skips the items already written.

## 5. Exit codes

0 done. 2 plan sha256 mismatch, a pin difference (Ouro revision, transformers version, dtype, attention, prod md5), a
data file that differs from data/sources.json, a token-id mismatch of the cue, the letters or any item (checked
before the first item), or a changed input on resume. 3 the identity gate at the start of a run failed (`gate.json` field `failures`;
nothing scored was written), the pilot gate failed (`debug_gate.json`), or the pilot gate is missing for a full run. 4 an identity failed (the log names the item); a rerun of the same command exits 4 again.
5 stopped by SIGTERM (resumable). On 2, 3 or 4: stop and report; do not rerun.

## 6. What to report

After step 3, per pilot (out/MODEL/DATASET/pilot/):
1. Exit code, and `debug_gate.json` in full (pass, counts with drops by reason, timing, norm_ratio_min_max).
2. `summary.md` in full (the table with lt_L1 first, NLD by B's gold letter, the predictions, timing).
3. `checks.json` fields `counts`, `gates`, `identity_failures`, `timing`, `block_maps`.
4. The seconds per arm and item, and `full_card_hours_at_this_rate` (both in `checks.json` field `timing`).

After steps 5 and 6: the exit code of `mains`, the aqua26 line from out/logs/mains.log, `gate.json`, `checks.json`
and `summary.md` (with the letter-split table) of each full run, sha256 of `results.csv`, `results.json`,
`checks.json` and `records.jsonl`, the ARC pilot's `debug_gate.json`, and any line with `identity failed` or
`GATE FAILED`.

Then pull the outputs back to the laptop:

    rsync -a <gpu-host>:/var/tmp/ra29/out/ loop-flow/artifacts_spark/ra29/
