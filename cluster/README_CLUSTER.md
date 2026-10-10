# Running the cluster bundle (operator page)

Run five commands in order, on a login node, from a project or scratch folder (never a home directory). Text in
`<ANGLE BRACKETS>` is yours to fill in. If a command fails, stop and send back the output of
`bash <PROJECT_DIR>/repo/loopedcot/cluster/jobs/status.sh`, which also prints every file under `<OUT_ROOT>/_diag/`. For command 2,
also send its closing checklist. Design notes are in `<PROJECT_DIR>/repo/loopedcot/cluster/env/CLUSTER.md`.

1. Clone the repository:

   ```
   git clone <REPOSITORY_URL> <PROJECT_DIR>/repo/loopedcot
   ```

   Success: git prints `Cloning into '<PROJECT_DIR>/repo/loopedcot'...` and `ls <PROJECT_DIR>/repo/loopedcot/cluster` lists
   `README_CLUSTER.md`.

2. Set up once (submits no job; a rerun skips what is done):

   ```
   bash <PROJECT_DIR>/repo/loopedcot/cluster/env/setup_cluster.sh --out-root <OUT_ROOT> --log-root <OUT_ROOT>/logs --hf-home <HF_HOME> --weights-root <WEIGHTS_ROOT> --xllm-root <PROJECT_DIR>/repo/xllm-loop --venv-root <PROJECT_DIR>/repo/venvs --plan-dir <PROJECT_DIR>/repo/loopedcot/cluster/plans --settle <PROJECT_DIR>/repo/loopedcot/cluster/data --partition <PARTITION> --account <ACCOUNT> --nodelist <NODE1,NODE2,NODE3,NODE4,NODE5,NODE6> --nodes-expected 6 --reservation "<RESERVATION>" --modules "<MODULES>" --python <PYTHON_WITH_TORCH>
   ```

   Write `--reservation ""` when there is no reservation and `--modules ""` when no module is needed.
   Success: the checklist ends with `nothing left: source <OUT_ROOT>/cluster.env, then DONE_SRC=none bash
   <PROJECT_DIR>/repo/loopedcot/cluster/jobs/submit_chain.sh` and the exit code is 0.

3. Load the environment (in every new shell):

   ```
   source <OUT_ROOT>/cluster.env
   ```

   Success: it prints nothing, and `echo $SLURM_RESTRICT` prints `--exclude=...`, `--reservation=...` or `--partition=...`.

4. Submit the whole run as one chain (setup check, a GPU probe of every node, then preflight and xllm build, then smoke,
   then every array; nodes that fail the probe are excluded):

   ```
   DONE_SRC=none bash <PROJECT_DIR>/repo/loopedcot/cluster/jobs/submit_chain.sh
   ```

   Success: it prints `1 setup_check job <id>`, one `2 probe <node> job <id>` line per node, `3 probe_collect job <id>`,
   then `chain submitted; progress: bash <PROJECT_DIR>/repo/loopedcot/cluster/jobs/status.sh`.

5. Watch progress (repeat as often as you like):

   ```
   bash <PROJECT_DIR>/repo/loopedcot/cluster/jobs/status.sh
   ```

   Success: the `failed` column is 0 in every row, the last line is `diagnosis files: 0`, and at the end `done` equals
   `jobs` in the `total` row. A failed chain step shows up under `chain markers` as `<step>.fail`, and its diagnosis
   is printed below the table.
