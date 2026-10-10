# Running the cluster bundle (operator page)

Three commands, on a login node, from your project or scratch folder (never a home directory). Text in
`<ANGLE BRACKETS>` is yours to fill in. Design notes are in `repo/loopedcot/cluster/env/CLUSTER.md`.

1. Clone the repository into the project folder:

   ```
   cd <PROJECT_DIR>
   git clone <REPOSITORY_URL> repo/loopedcot
   ```

   Success: `ls repo/loopedcot/cluster` lists `README_CLUSTER.md` and `run_all.sh`.

2. Set up and submit the whole run:

   ```
   bash repo/loopedcot/cluster/run_all.sh
   ```

   It finds the default partition, your first Slurm account, the GPU nodes of the partition and a Python with a CUDA
   build of torch. It prints them with the free disk of the project folder and asks once: type `yes`. Then it sets up
   (four venvs, about 80 GB of weights in `<PROJECT_DIR>/hf` and `<PROJECT_DIR>/ckpt`, no job), submits the chain and
   prints the status. The chain probes the GPU of every listed node and keeps the first six healthy nodes. The run
   never uses more.
   - A value it cannot find is asked for by name (for example `ACCOUNT = `). The answer is kept in
     `<PROJECT_DIR>/out/run_all.env`, so a rerun does not ask again.
   - To change a value, set it on the command line, for example
     `ACCOUNT=<ACCOUNT> MODULES="<MODULES>" bash repo/loopedcot/cluster/run_all.sh`.
   - When it stops at the setup checklist, do what the `FAIL` or `SUPPLY` lines say and run the same command again. A
     rerun skips what is done.

   Success: it prints `1 setup_check job <id>`, one `2 probe <node> job <id>` line per node,
   `3 probe_collect job <id>`, `chain submitted`, and then the status table.

3. Watch progress (as often as you like):

   ```
   bash repo/loopedcot/cluster/jobs/status.sh
   ```

   Success: the `failed` column is 0 in every row, the last line is `diagnosis files: 0`, and at the end `done` equals
   `jobs` in the `total` row.

## What to send back

- When command 2 stops: its full output, or `<PROJECT_DIR>/out/run_all_setup.log` when it stopped at the setup
  checklist.
- When a job fails, or once a day: the full output of command 3. It includes the chain markers, the bad nodes, the
  run's nodes, every probe verdict and every diagnosis file under `<PROJECT_DIR>/out/_diag/`.
- At the end: `<PROJECT_DIR>/out/merged/` (the merged results and `local_vs_cluster.csv`).
