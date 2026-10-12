# RA-31 run sheet (for the runner)

RA-31 Part A = the dose curve of the branch writes at the last prompt token of Ouro base, teacher-forced
(plan_ra31.md). Start only when the card is free. One card, one process at a time. Do not edit any file in the
folder. If a step fails, stop and report; do not rerun with changes.

## 1. Copy the folder to the GPU host

Copy `loop-flow/src/ra31/` (without `out/` and `__pycache__/`) to `/var/tmp/ra31` on the GPU host. Never put it in a
home folder: run.sh refuses a folder inside the home directory.

    rsync -a --exclude out --exclude __pycache__ loop-flow/src/ra31/ <gpu-host>:/var/tmp/ra31/
    cd /var/tmp/ra31 && sha256sum plan_ra31.md ra31.py run.sh lib/ra31_hooks.py && cat plan_ra31.sha256

Compare the sums with the laptop copy, and the plan sum with `plan_ra31.sha256`, before you continue.

## 2. Python

The Ouro venv RA-28 used (the one RUN 1c used: torch 2.11.0+cu130, transformers 4.56.2). Set the absolute path:

    export PY_OURO=<absolute path of .venv>/bin/python
    "$PY_OURO" -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

Expected: True and transformers 4.56.2 (the run stops with exit 2 on another version). Ouro-1.4B loads from the
host's Hugging Face cache at revision 574fa66c. run.sh sets HF_HUB_OFFLINE=1, so a missing revision fails at load and
nothing is downloaded. If RA-28 set HF_HOME, export the same value.

## 3. Before each GPU step

    nvidia-smi --query-compute-apps=pid,used_memory --format=csv; free -g

Launch only when no other process uses the card and at least 30 GB are free.

## 4. Commands, in order

All in /var/tmp/ra31 with PY_OURO exported. Steps 5 and 6 run only after the RA's keep decision on the pilot report.

| step | command | expected time | pass condition |
|---|---|---|---|
| 1 | `bash run.sh test` | 1 to 3 min, CPU | ends `OK` (31 tests) |
| 2 | `bash run.sh hash` | seconds | plan_sha256 equals `plan_ra31.sha256`; send it for the LEDGER before step 3 |
| 3 | `bash run.sh pilot 1.4b` | about 8 min | `[ra31] identity gate PASS`, `G3 ... PASS`, `G4 ... PASS`, `[ra31] done`, exit 0 |
| 4 | `bash run.sh summarize pilot 1.4b` | under 1 min, CPU | exit 0; then send the pilot report (section 6) and wait |
| 5 | `nohup bash run.sh full 1.4b > out/logs/nohup_full_1.4b.txt 2>&1 &` (with `DROP=...` only if the RA says so) | about 1.3 h | log ends `[ra31] done`, exit 0 |
| 6 | `bash run.sh summarize full 1.4b` | 1 to 3 min, CPU | exit 0; writes results.csv, checks.json, summary.md |

2.6B (only on the RA's instruction, after the 1.4B main runs): the same steps 3 to 6 with `2.6b`. It needs
Ouro-2.6B base in the host's cache; if the load fails, report it and stop.

Expected card time: the pilot is done (7.94 s per item at 76 forwards). The full 1.4B run has 83 forwards per item
plus one backward, about 8.9 s per item and about 1.32 card-hours for 530 items. The full run starts only if the
pilot's rate projects to at most 1.5 card-hours (5.5 on 2.6B); otherwise it exits 3 with `NO FIT`. The pilot gate
of out/pilot_1.4b (made under the previous plan sha256, ae5405db...) is accepted by the code; do not rerun the
pilot.

Stop a full run only with `pkill -TERM -f 'ra31.py --run'`. It stops after the current item with exit code 5. Run the
same command again to resume. It skips the items already written, and the identity gate runs again first.

## 5. Exit codes

0 done. 2 plan sha256 mismatch, a pin difference (Ouro revision, transformers version, dtype, attention, prod md5), a
data file that differs from data/sources.json, an item that does not rebuild, or a changed input on resume. 3 gate
failed: the identity gate before any scored row (`gate.json` fields `failures`, `exit3_keys`), the end gates G3 (run 1c
reproduction) or G4 (reduction) (`pilot_gate.json` or `end_gates.json`), a missing or failed pilot gate, or `NO FIT`.
4 an identity failed on a scored row (the log names the item). 5 stopped by SIGTERM (resumable). On 2, 3 or 4: stop
and report; do not rerun.

## 6. What to report

After steps 3 and 4 (the pilot report):
1. Exit codes and the plan sha256 of step 2.
2. `out/pilot_1.4b/pilot_gate.json` in full: gates G3 and G4, dropped counts with reasons, seconds per item and per
   arm and item, `projected_full_hours`.
3. `out/pilot_1.4b/gate.json` fields `pass`, `failures`, `seconds`; `out/pilot_1.4b/t2.json` in full.
4. `out/pilot_1.4b/summary.md` (the alpha tables, the derived readings, the predictions and the prune proposal).

After steps 5 and 6:
1. Exit codes of steps 5 and 6.
2. `out/full_1.4b/hash_line.json` (plan and config sha256), `gate.json` fields `pass` and `seconds`,
   `end_gates.json` in full, `pilot_gate_used.json`.
3. `out/full_1.4b/checks.json` in full and `summary.md`.
4. sha256 of `results.csv`, `checks.json` and every `items_*.jsonl` and `selection_*.json` (also in `hashes.json`).
   Also the sha256 and size of `attr_D1.npz` and `attr_D2.npz` (about 70 MB each) and `checks.json` field
   `attribution`.
5. Card-minutes used (`checks.json` field `gpu_minutes`, `progress.json` sessions).
6. Any line with `identity failed` or `GATE FAILED`.

Then pull the outputs back to the laptop:

    rsync -a <gpu-host>:/var/tmp/ra31/out/ loop-flow/artifacts_spark/ra31/

## 7. Part B (cluster unit; the packager wraps these commands)

    bash run.sh partb_list                 # 15 shards, arms and card-hours (CPU)
    bash run.sh partb <shard>              # one shard; PARTB_OUT sets the output root (default out/partb)
    bash run.sh partb_summarize            # merge every arm under the output root (CPU)

Each arm writes <root>/<arm>/DONE.json; a requeued shard skips done arms. Exit 2: a pin or data mismatch (steer_gen
md5, r00 meta version, data_partb hashes); exit 3: the alpha-1 identity (ident against r00_rerun) failed; any other
nonzero code is the G.main exit of an arm. Shard ident_allpos runs r00_rerun before ident, so run it as one job.
Copy the folder without out/ (out/ holds local logs and the Part A pilot outputs).

## 8. Model axis (cluster; plan section 14)

Per model, the pilot first, then the full run on its own pilot gate:

    bash run.sh pilot 1.4b-think && bash run.sh summarize pilot 1.4b-think
    bash run.sh full 1.4b-think && bash run.sh summarize full 1.4b-think
    (the same with 2.6b-think)

The models load from the host's Hugging Face cache at the pinned revisions (HF_HUB_OFFLINE=1); a missing revision
exits 2 at load. Items are selected inside the pilot and full jobs; nothing is built in advance.
