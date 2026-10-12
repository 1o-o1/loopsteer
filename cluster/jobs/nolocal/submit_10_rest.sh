#!/bin/bash
# Step 2 of 2: every other job in queue order, after the xllm build-and-import job finished OK. One pipeline
# = one (group, script, model, output folder); its stages chain with afterok, so a failing config blocks only
# its own pipeline (RA-20 item 4). Pipelines and groups queue at once; the order below is the queue order for
# a throttled queue (item 8). Each sbatch call names the task indices of one stage list. The merge job waits
# for every pipeline (afterany). Usage: source <OUT_ROOT>/cluster.env && bash jobs/submit_10_rest.sh
# (jobs/submit_chain.sh runs it as its last step.) It stops without $OUT_ROOT/_smoke.ok, without the preflight
# done marker, and when cluster.env lacks PARTITION, ACCOUNT, LOG_ROOT or the node restriction SLURM_RESTRICT.
# These lists come from gen_jobs.py --no-local-done: every job runs on the cluster, nothing is staged.
set -euo pipefail
cd "$(dirname "$0")"
CARDS=${CARDS:-48}
# cluster.env: the one already sourced, else $OUT_ROOT/cluster.env, else the first-push location
if [ -z "${PARTITION:-}" ] || [ -z "${REPO:-}" ]; then
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "../../env/cluster.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi
  done
fi
: "${OUT_ROOT:?set OUT_ROOT (source the cluster.env that setup_cluster.sh wrote)}"
SN=$(basename "$0" .sh); mkdir -p "$OUT_ROOT/_diag" "$OUT_ROOT/_tasklog"; TL="$OUT_ROOT/_tasklog/$SN.log"; : > "$TL"
exec > >(tee -a "$TL") 2>&1
CHECK="start"
diag() { sleep 1; { echo "step: $SN"; echo "exit code: $1"; echo "failing check: $CHECK"; echo "host role: ${RUN_ROLE:-cluster} (submit host, no GPU)"; echo "host memory: $(free -g 2> /dev/null | awk '/^Mem:/{print $7 " GiB available of " $2 " GiB"}' || true)"; echo "time: $(date '+%Y-%m-%d %H:%M:%S')"; echo "--- last 50 lines of $TL"; tail -n 50 "$TL"; } > "$OUT_ROOT/_diag/$SN.txt" 2>&1 || true; }
trap 'rc=$?; if [ $rc -ne 0 ]; then diag $rc; else rm -f "$OUT_ROOT/_diag/$SN.txt"; fi' EXIT
CHECK="cluster.env values PARTITION ACCOUNT LOG_ROOT SLURM_RESTRICT (the node restriction)"
for v in PARTITION ACCOUNT LOG_ROOT SLURM_RESTRICT; do
  case "${!v:-<>}" in *"<"*">"*) echo "$v is not set or still a placeholder in cluster.env (setup_cluster.sh)"; exit 7;; esac
done
# nodes found bad (submit_chain.sh probes, or a task that could not start its GPU) join the exclude list
BAD=$( { sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null || true; } | { grep . || true; } | paste -sd, -)
RESTRICT="$SLURM_RESTRICT"
if [ -n "$BAD" ]; then case "$RESTRICT" in *--exclude=*) RESTRICT=$(echo "$RESTRICT" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$BAD/");; *) RESTRICT="$RESTRICT --exclude=$BAD";; esac; fi
SB="--partition=$PARTITION --account=$ACCOUNT $RESTRICT"
echo "node restriction: $RESTRICT (nodes ${NODELIST:-not recorded}; bad nodes: ${BAD:-none})"
CHECK="smoke marker $OUT_ROOT/_smoke.ok"
if [ ! -f "$OUT_ROOT/_smoke.ok" ]; then
  echo "no $OUT_ROOT/_smoke.ok: run cluster/env/smoke_cluster.sh first (it must pass)"; exit 6
fi
CHECK="preflight and xllm build done marker $OUT_ROOT/_done/g0_xllm.s1_build.001"
if [ ! -f "$OUT_ROOT/_done/g0_xllm.s1_build.001" ]; then
  echo "the xllm build-and-import job g0_xllm.s1_build.001 has not finished OK (no $OUT_ROOT/_done/g0_xllm.s1_build.001): run submit_00_xllm.sh and wait"; exit 3
fi
# held units: a unit (ouro, gen, raven, mcleish, parcae, ifm) whose preflight or smoke failed
# ($OUT_ROOT/_held/<unit>.held, written by env/preflight.sh and env/smoke_cluster.sh) is submitted held
# (sbatch --hold); its job ids go to $OUT_ROOT/_held/<unit>.jobs, and jobs/release_family.sh <unit> reruns its
# preflight and smoke and releases them. The merge job waits for them (afterany) and is never held.
mkdir -p "$OUT_ROOT/_held"
H() { if [ -f "$OUT_ROOT/_held/$1.held" ]; then echo "--hold"; fi; }
K() { if [ -f "$OUT_ROOT/_held/$1.held" ]; then echo "$2" >> "$OUT_ROOT/_held/$1.jobs"; fi; }
# every submission: Slurm job id, stage list, task indices (jobs/resubmit_pipeline.sh matches the queue to tasks)
mkdir -p "$OUT_ROOT/_chain"
S() { echo "$1 $2 $3" >> "$OUT_ROOT/_chain/submitted.tsv"; }
for f in ouro gen raven mcleish parcae ifm; do
  rm -f "$OUT_ROOT/_held/$f.jobs" "$OUT_ROOT/_held/$f.released"
  if [ -f "$OUT_ROOT/_held/$f.held" ]; then echo "family $f held ($(head -1 "$OUT_ROOT/_held/$f.held")); its jobs are submitted held (release after the fix: bash jobs/release_family.sh $f)"; fi
done
CHECK="sbatch of the stage arrays (the last printed line is the call that failed)"
ALL=""
echo "group g1_copy: 49 jobs, 47.5 card-hours (GSM8K copy and derived)"
J1=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=1%$CARDS task_g1_copy.sbatch g1_copy.s2_window.tsv)   # k17_varbench.py ouro_1_4b_base s2_window
K ouro $J1
S $J1 g1_copy.s2_window.tsv 1
ALL="$ALL:$J1"
J2=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J1 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s3_scan.tsv)   # k17_varbench.py ouro_1_4b_base s3_scan
K ouro $J2
S $J2 g1_copy.s3_scan.tsv 1
ALL="$ALL:$J2"
J3=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J2 --array=1,2,3,4%$CARDS task_g1_copy.sbatch g1_copy.s4_rows.tsv)   # k17_varbench.py ouro_1_4b_base s4_rows
K ouro $J3
S $J3 g1_copy.s4_rows.tsv 1,2,3,4
ALL="$ALL:$J3"
J4=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J3 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s5_merge.tsv)   # k17_varbench.py ouro_1_4b_base s5_merge
K ouro $J4
S $J4 g1_copy.s5_merge.tsv 1
ALL="$ALL:$J4"
J5=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J4 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s6_summarize.tsv)   # k17_varbench.py ouro_1_4b_base s6_summarize
K ouro $J5
S $J5 g1_copy.s6_summarize.tsv 1
ALL="$ALL:$J5"
J6=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=2%$CARDS task_g1_copy.sbatch g1_copy.s2_window.tsv)   # k17_varbench.py ouro_2_6b_base s2_window
K ouro $J6
S $J6 g1_copy.s2_window.tsv 2
ALL="$ALL:$J6"
J7=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J6 --array=2%$CARDS task_g1_copy.sbatch g1_copy.s3_scan.tsv)   # k17_varbench.py ouro_2_6b_base s3_scan
K ouro $J7
S $J7 g1_copy.s3_scan.tsv 2
ALL="$ALL:$J7"
J8=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J7 --array=5,6,7,8,9,10,11%$CARDS task_g1_copy.sbatch g1_copy.s4_rows.tsv)   # k17_varbench.py ouro_2_6b_base s4_rows
K ouro $J8
S $J8 g1_copy.s4_rows.tsv 5,6,7,8,9,10,11
ALL="$ALL:$J8"
J9=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J8 --array=2%$CARDS task_g1_copy.sbatch g1_copy.s5_merge.tsv)   # k17_varbench.py ouro_2_6b_base s5_merge
K ouro $J9
S $J9 g1_copy.s5_merge.tsv 2
ALL="$ALL:$J9"
J10=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J9 --array=2%$CARDS task_g1_copy.sbatch g1_copy.s6_summarize.tsv)   # k17_varbench.py ouro_2_6b_base s6_summarize
K ouro $J10
S $J10 g1_copy.s6_summarize.tsv 2
ALL="$ALL:$J10"
J11=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=3%$CARDS task_g1_copy.sbatch g1_copy.s2_window.tsv)   # k17_varbench.py ouro_1_4b_think s2_window
K ouro $J11
S $J11 g1_copy.s2_window.tsv 3
ALL="$ALL:$J11"
J12=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J11 --array=3%$CARDS task_g1_copy.sbatch g1_copy.s3_scan.tsv)   # k17_varbench.py ouro_1_4b_think s3_scan
K ouro $J12
S $J12 g1_copy.s3_scan.tsv 3
ALL="$ALL:$J12"
J13=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J12 --array=12,13,14,15%$CARDS task_g1_copy.sbatch g1_copy.s4_rows.tsv)   # k17_varbench.py ouro_1_4b_think s4_rows
K ouro $J13
S $J13 g1_copy.s4_rows.tsv 12,13,14,15
ALL="$ALL:$J13"
J14=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J13 --array=3%$CARDS task_g1_copy.sbatch g1_copy.s5_merge.tsv)   # k17_varbench.py ouro_1_4b_think s5_merge
K ouro $J14
S $J14 g1_copy.s5_merge.tsv 3
ALL="$ALL:$J14"
J15=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J14 --array=3%$CARDS task_g1_copy.sbatch g1_copy.s6_summarize.tsv)   # k17_varbench.py ouro_1_4b_think s6_summarize
K ouro $J15
S $J15 g1_copy.s6_summarize.tsv 3
ALL="$ALL:$J15"
J16=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=4%$CARDS task_g1_copy.sbatch g1_copy.s2_window.tsv)   # k17_varbench.py ouro_2_6b_think s2_window
K ouro $J16
S $J16 g1_copy.s2_window.tsv 4
ALL="$ALL:$J16"
J17=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J16 --array=4%$CARDS task_g1_copy.sbatch g1_copy.s3_scan.tsv)   # k17_varbench.py ouro_2_6b_think s3_scan
K ouro $J17
S $J17 g1_copy.s3_scan.tsv 4
ALL="$ALL:$J17"
J18=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J17 --array=16,17,18,19,20,21,22%$CARDS task_g1_copy.sbatch g1_copy.s4_rows.tsv)   # k17_varbench.py ouro_2_6b_think s4_rows
K ouro $J18
S $J18 g1_copy.s4_rows.tsv 16,17,18,19,20,21,22
ALL="$ALL:$J18"
J19=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J18 --array=4%$CARDS task_g1_copy.sbatch g1_copy.s5_merge.tsv)   # k17_varbench.py ouro_2_6b_think s5_merge
K ouro $J19
S $J19 g1_copy.s5_merge.tsv 4
ALL="$ALL:$J19"
J20=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J19 --array=4%$CARDS task_g1_copy.sbatch g1_copy.s6_summarize.tsv)   # k17_varbench.py ouro_2_6b_think s6_summarize
K ouro $J20
S $J20 g1_copy.s6_summarize.tsv 4
ALL="$ALL:$J20"
J21=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=1%$CARDS task_g1_copy.sbatch g1_copy.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J21
S $J21 g1_copy.s1_scan.tsv 1
ALL="$ALL:$J21"
J22=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J21 --array=1,2,3,4,5,6,7,8%$CARDS task_g1_copy.sbatch g1_copy.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J22
S $J22 g1_copy.s2_rows.tsv 1,2,3,4,5,6,7,8
ALL="$ALL:$J22"
J23=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J22 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J23
S $J23 g1_copy.s3_merge.tsv 1
ALL="$ALL:$J23"
J24=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J23 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J24
S $J24 g1_copy.s4_summarize.tsv 1
ALL="$ALL:$J24"
echo "group g2_gen: 188 jobs, 72.4 card-hours (generation, GSM8K then SVAMP, McLeish GSM8K rows)"
J25=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=1%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_base s1_checks
K gen $J25
S $J25 g2_gen.s1_checks.tsv 1
ALL="$ALL:$J25"
J26=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J25 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_base s2_prep_screen
K gen $J26
S $J26 g2_gen.s2_prep_screen.tsv 1
ALL="$ALL:$J26"
J27=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J26 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_base s3_select
K gen $J27
S $J27 g2_gen.s3_select.tsv 1
ALL="$ALL:$J27"
J28=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J27 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_base s4_row
K gen $J28
S $J28 g2_gen.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
ALL="$ALL:$J28"
J29=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J28 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s5_anschain_presence.tsv)   # steer_gen.py ouro_1_4b_base s5_anschain_presence
K gen $J29
S $J29 g2_gen.s5_anschain_presence.tsv 1
ALL="$ALL:$J29"
J30=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J29 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_base s6_analyze
K gen $J30
S $J30 g2_gen.s6_analyze.tsv 1
ALL="$ALL:$J30"
J31=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=2%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_2_6b_base s1_checks
K gen $J31
S $J31 g2_gen.s1_checks.tsv 2
ALL="$ALL:$J31"
J32=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J31 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_base s2_prep_screen
K gen $J32
S $J32 g2_gen.s2_prep_screen.tsv 2
ALL="$ALL:$J32"
J33=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J32 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_2_6b_base s3_select
K gen $J33
S $J33 g2_gen.s3_select.tsv 2
ALL="$ALL:$J33"
J34=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J33 --array=24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_2_6b_base s4_row
K gen $J34
S $J34 g2_gen.s4_row.tsv 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46
ALL="$ALL:$J34"
J35=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J34 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s5_anschain_presence.tsv)   # steer_gen.py ouro_2_6b_base s5_anschain_presence
K gen $J35
S $J35 g2_gen.s5_anschain_presence.tsv 2
ALL="$ALL:$J35"
J36=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J35 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_base s6_analyze
K gen $J36
S $J36 g2_gen.s6_analyze.tsv 2
ALL="$ALL:$J36"
J37=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=3%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_think s1_checks
K gen $J37
S $J37 g2_gen.s1_checks.tsv 3
ALL="$ALL:$J37"
J38=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J37 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_think s2_prep_screen
K gen $J38
S $J38 g2_gen.s2_prep_screen.tsv 3
ALL="$ALL:$J38"
J39=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J38 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_think s3_select
K gen $J39
S $J39 g2_gen.s3_select.tsv 3
ALL="$ALL:$J39"
J40=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J39 --array=47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_think s4_row
K gen $J40
S $J40 g2_gen.s4_row.tsv 47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69
ALL="$ALL:$J40"
J41=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J40 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s5_anschain_presence.tsv)   # steer_gen.py ouro_1_4b_think s5_anschain_presence
K gen $J41
S $J41 g2_gen.s5_anschain_presence.tsv 3
ALL="$ALL:$J41"
J42=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J41 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_think s6_analyze
K gen $J42
S $J42 g2_gen.s6_analyze.tsv 3
ALL="$ALL:$J42"
J43=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=4%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_base s1_checks
K gen $J43
S $J43 g2_gen.s1_checks.tsv 4
ALL="$ALL:$J43"
J44=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J43 --array=4%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_base s2_prep_screen
K gen $J44
S $J44 g2_gen.s2_prep_screen.tsv 4
ALL="$ALL:$J44"
J45=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J44 --array=4%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_base s3_select
K gen $J45
S $J45 g2_gen.s3_select.tsv 4
ALL="$ALL:$J45"
J46=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J45 --array=70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_base s4_row
K gen $J46
S $J46 g2_gen.s4_row.tsv 70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92
ALL="$ALL:$J46"
J47=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J46 --array=4%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_base s6_analyze
K gen $J47
S $J47 g2_gen.s6_analyze.tsv 4
ALL="$ALL:$J47"
J48=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=5%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_2_6b_base s1_checks
K gen $J48
S $J48 g2_gen.s1_checks.tsv 5
ALL="$ALL:$J48"
J49=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J48 --array=5%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_base s2_prep_screen
K gen $J49
S $J49 g2_gen.s2_prep_screen.tsv 5
ALL="$ALL:$J49"
J50=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J49 --array=5%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_2_6b_base s3_select
K gen $J50
S $J50 g2_gen.s3_select.tsv 5
ALL="$ALL:$J50"
J51=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J50 --array=93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_2_6b_base s4_row
K gen $J51
S $J51 g2_gen.s4_row.tsv 93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115
ALL="$ALL:$J51"
J52=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J51 --array=5%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_base s6_analyze
K gen $J52
S $J52 g2_gen.s6_analyze.tsv 5
ALL="$ALL:$J52"
J53=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=6%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_think s1_checks
K gen $J53
S $J53 g2_gen.s1_checks.tsv 6
ALL="$ALL:$J53"
J54=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J53 --array=6%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_think s2_prep_screen
K gen $J54
S $J54 g2_gen.s2_prep_screen.tsv 6
ALL="$ALL:$J54"
J55=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J54 --array=6%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_think s3_select
K gen $J55
S $J55 g2_gen.s3_select.tsv 6
ALL="$ALL:$J55"
J56=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J55 --array=116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_think s4_row
K gen $J56
S $J56 g2_gen.s4_row.tsv 116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138
ALL="$ALL:$J56"
J57=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J56 --array=6%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_think s6_analyze
K gen $J57
S $J57 g2_gen.s6_analyze.tsv 6
ALL="$ALL:$J57"
J58=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen_raven.py mcleish_llama32_r32 s4_row
K mcleish $J58
S $J58 g2_gen.s4_row.tsv 139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160
ALL="$ALL:$J58"
J59=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J58 --array=7%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen_raven.py mcleish_llama32_r32 s6_analyze
K mcleish $J59
S $J59 g2_gen.s6_analyze.tsv 7
ALL="$ALL:$J59"
echo "group g3_map: 85 jobs, 106.6 card-hours (edit-site map at full pools)"
J60=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=1%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_base s1_run
K ouro $J60
S $J60 g3_map.s1_run.tsv 1
ALL="$ALL:$J60"
J61=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=2%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_base s1_run
K ouro $J61
S $J61 g3_map.s1_run.tsv 2
ALL="$ALL:$J61"
J62=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=3%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J62
S $J62 g3_map.s1_run.tsv 3
ALL="$ALL:$J62"
J63=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=1%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J63
S $J63 g3_map.s1_scan.tsv 1
ALL="$ALL:$J63"
J64=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J63 --array=1,2,3,4,5,6,7,8,9,10,11%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J64
S $J64 g3_map.s2_rows.tsv 1,2,3,4,5,6,7,8,9,10,11
ALL="$ALL:$J64"
J65=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J64 --array=1%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J65
S $J65 g3_map.s3_merge.tsv 1
ALL="$ALL:$J65"
J66=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J65 --array=1%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J66
S $J66 g3_map.s4_summarize.tsv 1
ALL="$ALL:$J66"
J67=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=2%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J67
S $J67 g3_map.s1_scan.tsv 2
ALL="$ALL:$J67"
J68=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J67 --array=12,13,14,15,16,17,18,19,20,21,22,23,24%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J68
S $J68 g3_map.s2_rows.tsv 12,13,14,15,16,17,18,19,20,21,22,23,24
ALL="$ALL:$J68"
J69=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J68 --array=2%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J69
S $J69 g3_map.s3_merge.tsv 2
ALL="$ALL:$J69"
J70=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J69 --array=2%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J70
S $J70 g3_map.s4_summarize.tsv 2
ALL="$ALL:$J70"
J71=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=3%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J71
S $J71 g3_map.s1_scan.tsv 3
ALL="$ALL:$J71"
J72=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J71 --array=25,26,27,28,29,30,31,32,33,34%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J72
S $J72 g3_map.s2_rows.tsv 25,26,27,28,29,30,31,32,33,34
ALL="$ALL:$J72"
J73=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J72 --array=3%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J73
S $J73 g3_map.s3_merge.tsv 3
ALL="$ALL:$J73"
J74=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J73 --array=3%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J74
S $J74 g3_map.s4_summarize.tsv 3
ALL="$ALL:$J74"
J75=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=4%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J75
S $J75 g3_map.s1_scan.tsv 4
ALL="$ALL:$J75"
J76=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J75 --array=35,36,37,38,39,40,41,42,43,44%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J76
S $J76 g3_map.s2_rows.tsv 35,36,37,38,39,40,41,42,43,44
ALL="$ALL:$J76"
J77=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J76 --array=4%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J77
S $J77 g3_map.s3_merge.tsv 4
ALL="$ALL:$J77"
J78=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J77 --array=4%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J78
S $J78 g3_map.s4_summarize.tsv 4
ALL="$ALL:$J78"
J79=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=5%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J79
S $J79 g3_map.s1_scan.tsv 5
ALL="$ALL:$J79"
J80=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J79 --array=45,46,47,48,49%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J80
S $J80 g3_map.s2_rows.tsv 45,46,47,48,49
ALL="$ALL:$J80"
J81=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J80 --array=5%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J81
S $J81 g3_map.s3_merge.tsv 5
ALL="$ALL:$J81"
J82=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J81 --array=5%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J82
S $J82 g3_map.s4_summarize.tsv 5
ALL="$ALL:$J82"
J83=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=6%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J83
S $J83 g3_map.s1_scan.tsv 6
ALL="$ALL:$J83"
J84=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J83 --array=50%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J84
S $J84 g3_map.s2_rows.tsv 50
ALL="$ALL:$J84"
J85=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J84 --array=6%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J85
S $J85 g3_map.s3_merge.tsv 6
ALL="$ALL:$J85"
J86=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J85 --array=6%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J86
S $J86 g3_map.s4_summarize.tsv 6
ALL="$ALL:$J86"
J87=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=7%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J87
S $J87 g3_map.s1_scan.tsv 7
ALL="$ALL:$J87"
J88=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J87 --array=51,52%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J88
S $J88 g3_map.s2_rows.tsv 51,52
ALL="$ALL:$J88"
J89=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J88 --array=7%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J89
S $J89 g3_map.s3_merge.tsv 7
ALL="$ALL:$J89"
J90=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J89 --array=7%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J90
S $J90 g3_map.s4_summarize.tsv 7
ALL="$ALL:$J90"
J91=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=8%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J91
S $J91 g3_map.s1_scan.tsv 8
ALL="$ALL:$J91"
J92=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J91 --array=53,54,55,56%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J92
S $J92 g3_map.s2_rows.tsv 53,54,55,56
ALL="$ALL:$J92"
J93=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J92 --array=8%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J93
S $J93 g3_map.s3_merge.tsv 8
ALL="$ALL:$J93"
J94=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J93 --array=8%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J94
S $J94 g3_map.s4_summarize.tsv 8
ALL="$ALL:$J94"
J95=$(sbatch --parsable $SB $(H parcae) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=4%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k17_parcae.py parcae_370m s1_run
K parcae $J95
S $J95 g3_map.s1_run.tsv 4
ALL="$ALL:$J95"
J96=$(sbatch --parsable $SB $(H parcae) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=5%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k17_parcae.py parcae_1_3b s1_run
K parcae $J96
S $J96 g3_map.s1_run.tsv 5
ALL="$ALL:$J96"
echo "group g4_cd: 4 jobs, 3.5 card-hours (C and D scripts (raven branch-freeze))"
J97=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=1%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py mcleish_llama32_r32 s1_run
K raven $J97
S $J97 g4_cd.s1_run.tsv 1
ALL="$ALL:$J97"
J98=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=2%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py mcleish_llama32_r32 s1_run
K raven $J98
S $J98 g4_cd.s1_run.tsv 2
ALL="$ALL:$J98"
J99=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=3%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py huginn_0125 s1_run
K raven $J99
S $J99 g4_cd.s1_run.tsv 3
ALL="$ALL:$J99"
J100=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=4%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py huginn_0125 s1_run
K raven $J100
S $J100 g4_cd.s1_run.tsv 4
ALL="$ALL:$J100"
echo "group g6_ifm: 51 jobs, 54.5 card-hours (IFM wave incl. the RA-23 IFM branch-freeze)"
J101=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro s2_window
K ifm $J101
S $J101 g6_ifm.s2_window.tsv 1
ALL="$ALL:$J101"
J102=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J101 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro s3_scan
K ifm $J102
S $J102 g6_ifm.s3_scan.tsv 1
ALL="$ALL:$J102"
J103=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J102 --array=1,2,3,4,5%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro s4_rows
K ifm $J103
S $J103 g6_ifm.s4_rows.tsv 1,2,3,4,5
ALL="$ALL:$J103"
J104=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J103 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro s5_merge
K ifm $J104
S $J104 g6_ifm.s5_merge.tsv 1
ALL="$ALL:$J104"
J105=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J104 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro s6_summarize
K ifm $J105
S $J105 g6_ifm.s6_summarize.tsv 1
ALL="$ALL:$J105"
J106=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro_inj s2_window
K ifm $J106
S $J106 g6_ifm.s2_window.tsv 2
ALL="$ALL:$J106"
J107=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J106 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro_inj s3_scan
K ifm $J107
S $J107 g6_ifm.s3_scan.tsv 2
ALL="$ALL:$J107"
J108=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J107 --array=6,7,8,9,10%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro_inj s4_rows
K ifm $J108
S $J108 g6_ifm.s4_rows.tsv 6,7,8,9,10
ALL="$ALL:$J108"
J109=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J108 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro_inj s5_merge
K ifm $J109
S $J109 g6_ifm.s5_merge.tsv 2
ALL="$ALL:$J109"
J110=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J109 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro_inj s6_summarize
K ifm $J110
S $J110 g6_ifm.s6_summarize.tsv 2
ALL="$ALL:$J110"
J111=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_d112 s2_window
K ifm $J111
S $J111 g6_ifm.s2_window.tsv 3
ALL="$ALL:$J111"
J112=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J111 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_d112 s3_scan
K ifm $J112
S $J112 g6_ifm.s3_scan.tsv 3
ALL="$ALL:$J112"
J113=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J112 --array=11,12,13,14,15%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_d112 s4_rows
K ifm $J113
S $J113 g6_ifm.s4_rows.tsv 11,12,13,14,15
ALL="$ALL:$J113"
J114=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J113 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_d112 s5_merge
K ifm $J114
S $J114 g6_ifm.s5_merge.tsv 3
ALL="$ALL:$J114"
J115=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J114 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_d112 s6_summarize
K ifm $J115
S $J115 g6_ifm.s6_summarize.tsv 3
ALL="$ALL:$J115"
J116=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_full.py ifm_p1_ouro s1_run
K ifm $J116
S $J116 g6_ifm.s1_run.tsv 1
ALL="$ALL:$J116"
J117=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_full.py ifm_p1_ouro_inj s1_run
K ifm $J117
S $J117 g6_ifm.s1_run.tsv 2
ALL="$ALL:$J117"
J118=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_full.py ifm_d112 s1_run
K ifm $J118
S $J118 g6_ifm.s1_run.tsv 3
ALL="$ALL:$J118"
J119=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=4%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p1_mid s1_run
K ifm $J119
S $J119 g6_ifm.s1_run.tsv 4
ALL="$ALL:$J119"
J120=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=5%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p1_mid_inj s1_run
K ifm $J120
S $J120 g6_ifm.s1_run.tsv 5
ALL="$ALL:$J120"
J121=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=6%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_r5 s1_run
K ifm $J121
S $J121 g6_ifm.s1_run.tsv 6
ALL="$ALL:$J121"
J122=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=7%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_diag s1_run
K ifm $J122
S $J122 g6_ifm.s1_run.tsv 7
ALL="$ALL:$J122"
J123=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=8%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_ortho s1_run
K ifm $J123
S $J123 g6_ifm.s1_run.tsv 8
ALL="$ALL:$J123"
J124=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=9%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro s1_run
K ifm $J124
S $J124 g6_ifm.s1_run.tsv 9
ALL="$ALL:$J124"
J125=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=10%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro s1_run
K ifm $J125
S $J125 g6_ifm.s1_run.tsv 10
ALL="$ALL:$J125"
J126=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=11%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J126
S $J126 g6_ifm.s1_run.tsv 11
ALL="$ALL:$J126"
J127=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=12%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J127
S $J127 g6_ifm.s1_run.tsv 12
ALL="$ALL:$J127"
J128=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=13%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J128
S $J128 g6_ifm.s1_run.tsv 13
ALL="$ALL:$J128"
J129=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=14%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J129
S $J129 g6_ifm.s1_run.tsv 14
ALL="$ALL:$J129"
J130=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=15%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J130
S $J130 g6_ifm.s1_run.tsv 15
ALL="$ALL:$J130"
J131=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=16%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J131
S $J131 g6_ifm.s1_run.tsv 16
ALL="$ALL:$J131"
J132=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=17%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J132
S $J132 g6_ifm.s1_run.tsv 17
ALL="$ALL:$J132"
J133=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=18%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J133
S $J133 g6_ifm.s1_run.tsv 18
ALL="$ALL:$J133"
J134=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=19%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_diag s1_run
K ifm $J134
S $J134 g6_ifm.s1_run.tsv 19
ALL="$ALL:$J134"
J135=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=20%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_d112 s1_run
K ifm $J135
S $J135 g6_ifm.s1_run.tsv 20
ALL="$ALL:$J135"
J136=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=21%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_d112 s1_run
K ifm $J136
S $J136 g6_ifm.s1_run.tsv 21
ALL="$ALL:$J136"
J137=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=22%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J137
S $J137 g6_ifm.s1_run.tsv 22
ALL="$ALL:$J137"
J138=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=23%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J138
S $J138 g6_ifm.s1_run.tsv 23
ALL="$ALL:$J138"
J139=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=24%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J139
S $J139 g6_ifm.s1_run.tsv 24
ALL="$ALL:$J139"
echo "group g12_ifmv1: 27 jobs, 23.4 card-hours (extension item 5: IFM k17_varbench rerun with the VBD V1 cells (--vbd_v1))"
J140=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --array=1%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro s2_window
K ifm $J140
S $J140 g12_ifmv1.s2_window.tsv 1
ALL="$ALL:$J140"
J141=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J140 --array=1%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro s3_scan
K ifm $J141
S $J141 g12_ifmv1.s3_scan.tsv 1
ALL="$ALL:$J141"
J142=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J141 --array=1,2,3,4,5%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro s4_rows
K ifm $J142
S $J142 g12_ifmv1.s4_rows.tsv 1,2,3,4,5
ALL="$ALL:$J142"
J143=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J142 --array=1%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro s5_merge
K ifm $J143
S $J143 g12_ifmv1.s5_merge.tsv 1
ALL="$ALL:$J143"
J144=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J143 --array=1%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro s6_summarize
K ifm $J144
S $J144 g12_ifmv1.s6_summarize.tsv 1
ALL="$ALL:$J144"
J145=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --array=2%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro_inj s2_window
K ifm $J145
S $J145 g12_ifmv1.s2_window.tsv 2
ALL="$ALL:$J145"
J146=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J145 --array=2%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro_inj s3_scan
K ifm $J146
S $J146 g12_ifmv1.s3_scan.tsv 2
ALL="$ALL:$J146"
J147=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J146 --array=6,7,8,9,10%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro_inj s4_rows
K ifm $J147
S $J147 g12_ifmv1.s4_rows.tsv 6,7,8,9,10
ALL="$ALL:$J147"
J148=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J147 --array=2%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro_inj s5_merge
K ifm $J148
S $J148 g12_ifmv1.s5_merge.tsv 2
ALL="$ALL:$J148"
J149=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J148 --array=2%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro_inj s6_summarize
K ifm $J149
S $J149 g12_ifmv1.s6_summarize.tsv 2
ALL="$ALL:$J149"
J150=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --array=3%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s2_window.tsv)   # k17_varbench.py ifm_d112 s2_window
K ifm $J150
S $J150 g12_ifmv1.s2_window.tsv 3
ALL="$ALL:$J150"
J151=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J150 --array=3%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s3_scan.tsv)   # k17_varbench.py ifm_d112 s3_scan
K ifm $J151
S $J151 g12_ifmv1.s3_scan.tsv 3
ALL="$ALL:$J151"
J152=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J151 --array=11,12,13,14,15%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s4_rows.tsv)   # k17_varbench.py ifm_d112 s4_rows
K ifm $J152
S $J152 g12_ifmv1.s4_rows.tsv 11,12,13,14,15
ALL="$ALL:$J152"
J153=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J152 --array=3%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s5_merge.tsv)   # k17_varbench.py ifm_d112 s5_merge
K ifm $J153
S $J153 g12_ifmv1.s5_merge.tsv 3
ALL="$ALL:$J153"
J154=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g12_ifmv1_%A_%a.out --dependency=afterok:$J153 --array=3%$CARDS task_g12_ifmv1.sbatch g12_ifmv1.s6_summarize.tsv)   # k17_varbench.py ifm_d112 s6_summarize
K ifm $J154
S $J154 g12_ifmv1.s6_summarize.tsv 3
ALL="$ALL:$J154"
echo "group g14_ifmy1: 7 jobs, 12.8 card-hours (extension: the Y1 arm (edit-site rows) on seven IFM cells)"
J155=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=1%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J155
S $J155 g14_ifmy1.s1_run.tsv 1
ALL="$ALL:$J155"
J156=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=2%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J156
S $J156 g14_ifmy1.s1_run.tsv 2
ALL="$ALL:$J156"
J157=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=3%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J157
S $J157 g14_ifmy1.s1_run.tsv 3
ALL="$ALL:$J157"
J158=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=4%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J158
S $J158 g14_ifmy1.s1_run.tsv 4
ALL="$ALL:$J158"
J159=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=5%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J159
S $J159 g14_ifmy1.s1_run.tsv 5
ALL="$ALL:$J159"
J160=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=6%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J160
S $J160 g14_ifmy1.s1_run.tsv 6
ALL="$ALL:$J160"
J161=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=7%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_diag s1_run
K ifm $J161
S $J161 g14_ifmy1.s1_run.tsv 7
ALL="$ALL:$J161"
echo "group g7_ra23: 175 jobs, 112.1 card-hours (RA-23: Ouro Thinking branch-freeze, LRE facts)"
J162=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J162
S $J162 g7_ra23.s1_run.tsv 1
ALL="$ALL:$J162"
J163=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J163
S $J163 g7_ra23.s1_run.tsv 2
ALL="$ALL:$J163"
J164=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J164
S $J164 g7_ra23.s1_run.tsv 3
ALL="$ALL:$J164"
J165=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J165
S $J165 g7_ra23.s1_run.tsv 4
ALL="$ALL:$J165"
J166=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J166
S $J166 g7_ra23.s1_run.tsv 5
ALL="$ALL:$J166"
J167=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J167
S $J167 g7_ra23.s1_run.tsv 6
ALL="$ALL:$J167"
J168=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J168
S $J168 g7_ra23.s1_scan.tsv 1
ALL="$ALL:$J168"
J169=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J168 --array=1,2,3,4,5,6%$CARDS task_g7_ra23.sbatch g7_ra23.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J169
S $J169 g7_ra23.s2_rows.tsv 1,2,3,4,5,6
ALL="$ALL:$J169"
J170=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J169 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J170
S $J170 g7_ra23.s3_merge.tsv 1
ALL="$ALL:$J170"
J171=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J170 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J171
S $J171 g7_ra23.s4_summarize.tsv 1
ALL="$ALL:$J171"
J172=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J172
S $J172 g7_ra23.s1_scan.tsv 2
ALL="$ALL:$J172"
J173=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J172 --array=7%$CARDS task_g7_ra23.sbatch g7_ra23.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J173
S $J173 g7_ra23.s2_rows.tsv 7
ALL="$ALL:$J173"
J174=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J173 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J174
S $J174 g7_ra23.s3_merge.tsv 2
ALL="$ALL:$J174"
J175=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J174 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J175
S $J175 g7_ra23.s4_summarize.tsv 2
ALL="$ALL:$J175"
J176=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_base s1_checks
K gen $J176
S $J176 g7_ra23.s1_checks.tsv 1
ALL="$ALL:$J176"
J177=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J176 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_base s2_prep_screen
K gen $J177
S $J177 g7_ra23.s2_prep_screen.tsv 1
ALL="$ALL:$J177"
J178=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J177 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_base s3_select
K gen $J178
S $J178 g7_ra23.s3_select.tsv 1
ALL="$ALL:$J178"
J179=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J178 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_base s4_row
K gen $J179
S $J179 g7_ra23.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
ALL="$ALL:$J179"
J180=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J179 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_base s6_analyze
K gen $J180
S $J180 g7_ra23.s6_analyze.tsv 1
ALL="$ALL:$J180"
J181=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_base s1_checks
K gen $J181
S $J181 g7_ra23.s1_checks.tsv 2
ALL="$ALL:$J181"
J182=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J181 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_base s2_prep_screen
K gen $J182
S $J182 g7_ra23.s2_prep_screen.tsv 2
ALL="$ALL:$J182"
J183=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J182 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_base s3_select
K gen $J183
S $J183 g7_ra23.s3_select.tsv 2
ALL="$ALL:$J183"
J184=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J183 --array=23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_base s4_row
K gen $J184
S $J184 g7_ra23.s4_row.tsv 23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
ALL="$ALL:$J184"
J185=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J184 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_base s6_analyze
K gen $J185
S $J185 g7_ra23.s6_analyze.tsv 2
ALL="$ALL:$J185"
J186=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_think s1_checks
K gen $J186
S $J186 g7_ra23.s1_checks.tsv 3
ALL="$ALL:$J186"
J187=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J186 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_think s2_prep_screen
K gen $J187
S $J187 g7_ra23.s2_prep_screen.tsv 3
ALL="$ALL:$J187"
J188=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J187 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_think s3_select
K gen $J188
S $J188 g7_ra23.s3_select.tsv 3
ALL="$ALL:$J188"
J189=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J188 --array=45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_think s4_row
K gen $J189
S $J189 g7_ra23.s4_row.tsv 45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66
ALL="$ALL:$J189"
J190=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J189 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_think s6_analyze
K gen $J190
S $J190 g7_ra23.s6_analyze.tsv 3
ALL="$ALL:$J190"
J191=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_base s1_checks
K gen $J191
S $J191 g7_ra23.s1_checks.tsv 4
ALL="$ALL:$J191"
J192=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J191 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_base s2_prep_screen
K gen $J192
S $J192 g7_ra23.s2_prep_screen.tsv 4
ALL="$ALL:$J192"
J193=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J192 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_base s3_select
K gen $J193
S $J193 g7_ra23.s3_select.tsv 4
ALL="$ALL:$J193"
J194=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J193 --array=67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_base s4_row
K gen $J194
S $J194 g7_ra23.s4_row.tsv 67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88
ALL="$ALL:$J194"
J195=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J194 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_base s6_analyze
K gen $J195
S $J195 g7_ra23.s6_analyze.tsv 4
ALL="$ALL:$J195"
J196=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_base s1_checks
K gen $J196
S $J196 g7_ra23.s1_checks.tsv 5
ALL="$ALL:$J196"
J197=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J196 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_base s2_prep_screen
K gen $J197
S $J197 g7_ra23.s2_prep_screen.tsv 5
ALL="$ALL:$J197"
J198=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J197 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_base s3_select
K gen $J198
S $J198 g7_ra23.s3_select.tsv 5
ALL="$ALL:$J198"
J199=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J198 --array=89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_base s4_row
K gen $J199
S $J199 g7_ra23.s4_row.tsv 89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110
ALL="$ALL:$J199"
J200=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J199 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_base s6_analyze
K gen $J200
S $J200 g7_ra23.s6_analyze.tsv 5
ALL="$ALL:$J200"
J201=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_think s1_checks
K gen $J201
S $J201 g7_ra23.s1_checks.tsv 6
ALL="$ALL:$J201"
J202=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J201 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_think s2_prep_screen
K gen $J202
S $J202 g7_ra23.s2_prep_screen.tsv 6
ALL="$ALL:$J202"
J203=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J202 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_think s3_select
K gen $J203
S $J203 g7_ra23.s3_select.tsv 6
ALL="$ALL:$J203"
J204=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J203 --array=111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_think s4_row
K gen $J204
S $J204 g7_ra23.s4_row.tsv 111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132
ALL="$ALL:$J204"
J205=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J204 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_think s6_analyze
K gen $J205
S $J205 g7_ra23.s6_analyze.tsv 6
ALL="$ALL:$J205"
echo "group g11_gen26t: 107 jobs, 236.2 card-hours (extension item 1: generation on Ouro-2.6B-Thinking (setting a4; GSM8K, SVAMP, ASDiv, MAWPS))"
J206=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen.py ouro_2_6b_think s1_checks
K gen $J206
S $J206 g11_gen26t.s1_checks.tsv 1
ALL="$ALL:$J206"
J207=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J206 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_think s2_prep_screen
K gen $J207
S $J207 g11_gen26t.s2_prep_screen.tsv 1
ALL="$ALL:$J207"
J208=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J207 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen.py ouro_2_6b_think s3_select
K gen $J208
S $J208 g11_gen26t.s3_select.tsv 1
ALL="$ALL:$J208"
J209=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J208 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen.py ouro_2_6b_think s4_row
K gen $J209
S $J209 g11_gen26t.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
ALL="$ALL:$J209"
J210=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J209 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s5_anschain_presence.tsv)   # steer_gen.py ouro_2_6b_think s5_anschain_presence
K gen $J210
S $J210 g11_gen26t.s5_anschain_presence.tsv 1
ALL="$ALL:$J210"
J211=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J210 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_think s6_analyze
K gen $J211
S $J211 g11_gen26t.s6_analyze.tsv 1
ALL="$ALL:$J211"
J212=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen.py ouro_2_6b_think s1_checks
K gen $J212
S $J212 g11_gen26t.s1_checks.tsv 2
ALL="$ALL:$J212"
J213=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J212 --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_think s2_prep_screen
K gen $J213
S $J213 g11_gen26t.s2_prep_screen.tsv 2
ALL="$ALL:$J213"
J214=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J213 --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen.py ouro_2_6b_think s3_select
K gen $J214
S $J214 g11_gen26t.s3_select.tsv 2
ALL="$ALL:$J214"
J215=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J214 --array=24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen.py ouro_2_6b_think s4_row
K gen $J215
S $J215 g11_gen26t.s4_row.tsv 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46
ALL="$ALL:$J215"
J216=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J215 --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_think s6_analyze
K gen $J216
S $J216 g11_gen26t.s6_analyze.tsv 2
ALL="$ALL:$J216"
J217=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_think s1_checks
K gen $J217
S $J217 g11_gen26t.s1_checks.tsv 3
ALL="$ALL:$J217"
J218=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J217 --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_think s2_prep_screen
K gen $J218
S $J218 g11_gen26t.s2_prep_screen.tsv 3
ALL="$ALL:$J218"
J219=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J218 --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_think s3_select
K gen $J219
S $J219 g11_gen26t.s3_select.tsv 3
ALL="$ALL:$J219"
J220=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J219 --array=47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_think s4_row
K gen $J220
S $J220 g11_gen26t.s4_row.tsv 47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68
ALL="$ALL:$J220"
J221=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J220 --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_think s6_analyze
K gen $J221
S $J221 g11_gen26t.s6_analyze.tsv 3
ALL="$ALL:$J221"
J222=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_think s1_checks
K gen $J222
S $J222 g11_gen26t.s1_checks.tsv 4
ALL="$ALL:$J222"
J223=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J222 --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_think s2_prep_screen
K gen $J223
S $J223 g11_gen26t.s2_prep_screen.tsv 4
ALL="$ALL:$J223"
J224=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J223 --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_think s3_select
K gen $J224
S $J224 g11_gen26t.s3_select.tsv 4
ALL="$ALL:$J224"
J225=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J224 --array=69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_think s4_row
K gen $J225
S $J225 g11_gen26t.s4_row.tsv 69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90
ALL="$ALL:$J225"
J226=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J225 --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_think s6_analyze
K gen $J226
S $J226 g11_gen26t.s6_analyze.tsv 4
ALL="$ALL:$J226"
echo "group g13_paththink: 7 jobs, 50.2 card-hours (extension: restore, read window and attention block on the Ouro Thinking models)"
J227=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=1%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J227
S $J227 g13_paththink.s1_run.tsv 1
ALL="$ALL:$J227"
J228=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=2%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J228
S $J228 g13_paththink.s1_run.tsv 2
ALL="$ALL:$J228"
J229=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=3%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J229
S $J229 g13_paththink.s1_run.tsv 3
ALL="$ALL:$J229"
J230=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=4%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J230
S $J230 g13_paththink.s1_run.tsv 4
ALL="$ALL:$J230"
J231=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=5%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J231
S $J231 g13_paththink.s1_run.tsv 5
ALL="$ALL:$J231"
J232=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=6%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J232
S $J232 g13_paththink.s1_run.tsv 6
ALL="$ALL:$J232"
J233=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=7%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J233
S $J233 g13_paththink.s1_run.tsv 7
ALL="$ALL:$J233"
echo "group g15_hgen: 26 jobs, 25.3 card-hours (extension item 4: Huginn GSM8K generation)"
J234=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g15_hgen_%A_%a.out --array=1,2%$CARDS task_g15_hgen.sbatch g15_hgen.s2_screen.tsv)   # steer_gen_raven.py huginn_0125 s2_screen
K mcleish $J234
S $J234 g15_hgen.s2_screen.tsv 1,2
ALL="$ALL:$J234"
J235=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g15_hgen_%A_%a.out --dependency=afterok:$J234 --array=1%$CARDS task_g15_hgen.sbatch g15_hgen.s3_select.tsv)   # steer_gen_raven.py huginn_0125 s3_select
K mcleish $J235
S $J235 g15_hgen.s3_select.tsv 1
ALL="$ALL:$J235"
J236=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g15_hgen_%A_%a.out --dependency=afterok:$J235 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22%$CARDS task_g15_hgen.sbatch g15_hgen.s4_row.tsv)   # steer_gen_raven.py huginn_0125 s4_row
K mcleish $J236
S $J236 g15_hgen.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
ALL="$ALL:$J236"
J237=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g15_hgen_%A_%a.out --dependency=afterok:$J236 --array=1%$CARDS task_g15_hgen.sbatch g15_hgen.s6_analyze.tsv)   # steer_gen_raven.py huginn_0125 s6_analyze
K mcleish $J237
S $J237 g15_hgen.s6_analyze.tsv 1
ALL="$ALL:$J237"
echo "group g16_ifmgrid: 2 jobs, 2.7 card-hours (Figure 2 grid: k15_full rerun of ifm_p1_ouro (D1, D2, CF))"
J238=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g16_ifmgrid_%A_%a.out --array=1%$CARDS task_g16_ifmgrid.sbatch g16_ifmgrid.s1_run.tsv)   # k15_full.py ifm_p1_ouro s1_run
K ifm $J238
S $J238 g16_ifmgrid.s1_run.tsv 1
ALL="$ALL:$J238"
J239=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g16_ifmgrid_%A_%a.out --array=2%$CARDS task_g16_ifmgrid.sbatch g16_ifmgrid.s1_run.tsv)   # k15_full.py ifm_p1_ouro s1_run
K ifm $J239
S $J239 g16_ifmgrid.s1_run.tsv 2
ALL="$ALL:$J239"
echo "group g17_fillmap: 31 jobs, 32.9 card-hours (B-EXT5 fill priority 1: edit-site map on d28, the 819M designs, Ouro-2.6B-Thinking)"
J240=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=1%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k15_full.py ifm_d28 s1_run
K ifm $J240
S $J240 g17_fillmap.s1_run.tsv 1
ALL="$ALL:$J240"
J241=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=1%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_scan.tsv)   # k17_public.py ouro26_think s1_scan
K raven $J241
S $J241 g17_fillmap.s1_scan.tsv 1
ALL="$ALL:$J241"
J242=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J241 --array=1,2%$CARDS task_g17_fillmap.sbatch g17_fillmap.s2_rows.tsv)   # k17_public.py ouro26_think s2_rows
K raven $J242
S $J242 g17_fillmap.s2_rows.tsv 1,2
ALL="$ALL:$J242"
J243=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J242 --array=1%$CARDS task_g17_fillmap.sbatch g17_fillmap.s3_merge.tsv)   # k17_public.py ouro26_think s3_merge
K raven $J243
S $J243 g17_fillmap.s3_merge.tsv 1
ALL="$ALL:$J243"
J244=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J243 --array=1%$CARDS task_g17_fillmap.sbatch g17_fillmap.s4_summarize.tsv)   # k17_public.py ouro26_think s4_summarize
K raven $J244
S $J244 g17_fillmap.s4_summarize.tsv 1
ALL="$ALL:$J244"
J245=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=2%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_scan.tsv)   # k17_public.py ouro26_think s1_scan
K raven $J245
S $J245 g17_fillmap.s1_scan.tsv 2
ALL="$ALL:$J245"
J246=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J245 --array=3,4,5%$CARDS task_g17_fillmap.sbatch g17_fillmap.s2_rows.tsv)   # k17_public.py ouro26_think s2_rows
K raven $J246
S $J246 g17_fillmap.s2_rows.tsv 3,4,5
ALL="$ALL:$J246"
J247=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J246 --array=2%$CARDS task_g17_fillmap.sbatch g17_fillmap.s3_merge.tsv)   # k17_public.py ouro26_think s3_merge
K raven $J247
S $J247 g17_fillmap.s3_merge.tsv 2
ALL="$ALL:$J247"
J248=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J247 --array=2%$CARDS task_g17_fillmap.sbatch g17_fillmap.s4_summarize.tsv)   # k17_public.py ouro26_think s4_summarize
K raven $J248
S $J248 g17_fillmap.s4_summarize.tsv 2
ALL="$ALL:$J248"
J249=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=3%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_scan.tsv)   # k17_public.py ouro26_think s1_scan
K raven $J249
S $J249 g17_fillmap.s1_scan.tsv 3
ALL="$ALL:$J249"
J250=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J249 --array=6,7,8,9,10,11,12,13%$CARDS task_g17_fillmap.sbatch g17_fillmap.s2_rows.tsv)   # k17_public.py ouro26_think s2_rows
K raven $J250
S $J250 g17_fillmap.s2_rows.tsv 6,7,8,9,10,11,12,13
ALL="$ALL:$J250"
J251=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J250 --array=3%$CARDS task_g17_fillmap.sbatch g17_fillmap.s3_merge.tsv)   # k17_public.py ouro26_think s3_merge
K raven $J251
S $J251 g17_fillmap.s3_merge.tsv 3
ALL="$ALL:$J251"
J252=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --dependency=afterok:$J251 --array=3%$CARDS task_g17_fillmap.sbatch g17_fillmap.s4_summarize.tsv)   # k17_public.py ouro26_think s4_summarize
K raven $J252
S $J252 g17_fillmap.s4_summarize.tsv 3
ALL="$ALL:$J252"
J253=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=2%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_r5 s1_run
K ifm $J253
S $J253 g17_fillmap.s1_run.tsv 2
ALL="$ALL:$J253"
J254=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=3%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_r5 s1_run
K ifm $J254
S $J254 g17_fillmap.s1_run.tsv 3
ALL="$ALL:$J254"
J255=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=4%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_r5 s1_run
K ifm $J255
S $J255 g17_fillmap.s1_run.tsv 4
ALL="$ALL:$J255"
J256=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=5%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_diag s1_run
K ifm $J256
S $J256 g17_fillmap.s1_run.tsv 5
ALL="$ALL:$J256"
J257=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=6%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_diag s1_run
K ifm $J257
S $J257 g17_fillmap.s1_run.tsv 6
ALL="$ALL:$J257"
J258=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=7%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_ortho s1_run
K ifm $J258
S $J258 g17_fillmap.s1_run.tsv 7
ALL="$ALL:$J258"
J259=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=8%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_ortho s1_run
K ifm $J259
S $J259 g17_fillmap.s1_run.tsv 8
ALL="$ALL:$J259"
J260=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g17_fillmap_%A_%a.out --array=9%$CARDS task_g17_fillmap.sbatch g17_fillmap.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_ortho s1_run
K ifm $J260
S $J260 g17_fillmap.s1_run.tsv 9
ALL="$ALL:$J260"
echo "group g18_fillgen: 135 jobs, 133.0 card-hours (B-EXT5 fill priority 2: McLeish and Huginn generation on SVAMP, ASDiv, MAWPS)"
J261=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --array=1%$CARDS task_g18_fillgen.sbatch g18_fillgen.s1_prep.tsv)   # steer_gen_raven.py huginn_0125 s1_prep
K mcleish $J261
S $J261 g18_fillgen.s1_prep.tsv 1
ALL="$ALL:$J261"
J262=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J261 --array=1,2%$CARDS task_g18_fillgen.sbatch g18_fillgen.s2_screen.tsv)   # steer_gen_raven.py huginn_0125 s2_screen
K mcleish $J262
S $J262 g18_fillgen.s2_screen.tsv 1,2
ALL="$ALL:$J262"
J263=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J262 --array=1%$CARDS task_g18_fillgen.sbatch g18_fillgen.s3_select.tsv)   # steer_gen_raven.py huginn_0125 s3_select
K mcleish $J263
S $J263 g18_fillgen.s3_select.tsv 1
ALL="$ALL:$J263"
J264=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J263 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22%$CARDS task_g18_fillgen.sbatch g18_fillgen.s4_row.tsv)   # steer_gen_raven.py huginn_0125 s4_row
K mcleish $J264
S $J264 g18_fillgen.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
ALL="$ALL:$J264"
J265=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J264 --array=1%$CARDS task_g18_fillgen.sbatch g18_fillgen.s6_analyze.tsv)   # steer_gen_raven.py huginn_0125 s6_analyze
K mcleish $J265
S $J265 g18_fillgen.s6_analyze.tsv 1
ALL="$ALL:$J265"
J266=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --array=2%$CARDS task_g18_fillgen.sbatch g18_fillgen.s1_prep.tsv)   # steer_gen_math.py mcleish_llama32_r32 s1_prep
K mcleish $J266
S $J266 g18_fillgen.s1_prep.tsv 2
ALL="$ALL:$J266"
J267=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J266 --array=3,4%$CARDS task_g18_fillgen.sbatch g18_fillgen.s2_screen.tsv)   # steer_gen_math.py mcleish_llama32_r32 s2_screen
K mcleish $J267
S $J267 g18_fillgen.s2_screen.tsv 3,4
ALL="$ALL:$J267"
J268=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J267 --array=2%$CARDS task_g18_fillgen.sbatch g18_fillgen.s3_select.tsv)   # steer_gen_math.py mcleish_llama32_r32 s3_select
K mcleish $J268
S $J268 g18_fillgen.s3_select.tsv 2
ALL="$ALL:$J268"
J269=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J268 --array=23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44%$CARDS task_g18_fillgen.sbatch g18_fillgen.s4_row.tsv)   # steer_gen_math.py mcleish_llama32_r32 s4_row
K mcleish $J269
S $J269 g18_fillgen.s4_row.tsv 23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
ALL="$ALL:$J269"
J270=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J269 --array=2%$CARDS task_g18_fillgen.sbatch g18_fillgen.s6_analyze.tsv)   # steer_gen_math.py mcleish_llama32_r32 s6_analyze
K mcleish $J270
S $J270 g18_fillgen.s6_analyze.tsv 2
ALL="$ALL:$J270"
J271=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --array=3%$CARDS task_g18_fillgen.sbatch g18_fillgen.s1_prep.tsv)   # steer_gen_math.py huginn_0125 s1_prep
K mcleish $J271
S $J271 g18_fillgen.s1_prep.tsv 3
ALL="$ALL:$J271"
J272=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J271 --array=5,6%$CARDS task_g18_fillgen.sbatch g18_fillgen.s2_screen.tsv)   # steer_gen_math.py huginn_0125 s2_screen
K mcleish $J272
S $J272 g18_fillgen.s2_screen.tsv 5,6
ALL="$ALL:$J272"
J273=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J272 --array=3%$CARDS task_g18_fillgen.sbatch g18_fillgen.s3_select.tsv)   # steer_gen_math.py huginn_0125 s3_select
K mcleish $J273
S $J273 g18_fillgen.s3_select.tsv 3
ALL="$ALL:$J273"
J274=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J273 --array=45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66%$CARDS task_g18_fillgen.sbatch g18_fillgen.s4_row.tsv)   # steer_gen_math.py huginn_0125 s4_row
K mcleish $J274
S $J274 g18_fillgen.s4_row.tsv 45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66
ALL="$ALL:$J274"
J275=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J274 --array=3%$CARDS task_g18_fillgen.sbatch g18_fillgen.s6_analyze.tsv)   # steer_gen_math.py huginn_0125 s6_analyze
K mcleish $J275
S $J275 g18_fillgen.s6_analyze.tsv 3
ALL="$ALL:$J275"
J276=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --array=4%$CARDS task_g18_fillgen.sbatch g18_fillgen.s1_prep.tsv)   # steer_gen_math.py mcleish_llama32_r32 s1_prep
K mcleish $J276
S $J276 g18_fillgen.s1_prep.tsv 4
ALL="$ALL:$J276"
J277=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J276 --array=7,8%$CARDS task_g18_fillgen.sbatch g18_fillgen.s2_screen.tsv)   # steer_gen_math.py mcleish_llama32_r32 s2_screen
K mcleish $J277
S $J277 g18_fillgen.s2_screen.tsv 7,8
ALL="$ALL:$J277"
J278=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J277 --array=4%$CARDS task_g18_fillgen.sbatch g18_fillgen.s3_select.tsv)   # steer_gen_math.py mcleish_llama32_r32 s3_select
K mcleish $J278
S $J278 g18_fillgen.s3_select.tsv 4
ALL="$ALL:$J278"
J279=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J278 --array=67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88%$CARDS task_g18_fillgen.sbatch g18_fillgen.s4_row.tsv)   # steer_gen_math.py mcleish_llama32_r32 s4_row
K mcleish $J279
S $J279 g18_fillgen.s4_row.tsv 67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88
ALL="$ALL:$J279"
J280=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J279 --array=4%$CARDS task_g18_fillgen.sbatch g18_fillgen.s6_analyze.tsv)   # steer_gen_math.py mcleish_llama32_r32 s6_analyze
K mcleish $J280
S $J280 g18_fillgen.s6_analyze.tsv 4
ALL="$ALL:$J280"
J281=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --array=5%$CARDS task_g18_fillgen.sbatch g18_fillgen.s1_prep.tsv)   # steer_gen_math.py huginn_0125 s1_prep
K mcleish $J281
S $J281 g18_fillgen.s1_prep.tsv 5
ALL="$ALL:$J281"
J282=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J281 --array=9,10%$CARDS task_g18_fillgen.sbatch g18_fillgen.s2_screen.tsv)   # steer_gen_math.py huginn_0125 s2_screen
K mcleish $J282
S $J282 g18_fillgen.s2_screen.tsv 9,10
ALL="$ALL:$J282"
J283=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J282 --array=5%$CARDS task_g18_fillgen.sbatch g18_fillgen.s3_select.tsv)   # steer_gen_math.py huginn_0125 s3_select
K mcleish $J283
S $J283 g18_fillgen.s3_select.tsv 5
ALL="$ALL:$J283"
J284=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J283 --array=89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110%$CARDS task_g18_fillgen.sbatch g18_fillgen.s4_row.tsv)   # steer_gen_math.py huginn_0125 s4_row
K mcleish $J284
S $J284 g18_fillgen.s4_row.tsv 89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110
ALL="$ALL:$J284"
J285=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g18_fillgen_%A_%a.out --dependency=afterok:$J284 --array=5%$CARDS task_g18_fillgen.sbatch g18_fillgen.s6_analyze.tsv)   # steer_gen_math.py huginn_0125 s6_analyze
K mcleish $J285
S $J285 g18_fillgen.s6_analyze.tsv 5
ALL="$ALL:$J285"
echo "group g19_e6ra30: 2 jobs, 3.6 card-hours (B-EXT6: RA-30 chain-value edit on Ouro-2.6B base, GSM8K (pilot2, then full2))"
J286=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g19_e6ra30_%A_%a.out --array=1%$CARDS task_g19_e6ra30.sbatch g19_e6ra30.s1_pilot.tsv)   # ra30.py ouro_2_6b_base s1_pilot
K ouro $J286
S $J286 g19_e6ra30.s1_pilot.tsv 1
ALL="$ALL:$J286"
J287=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g19_e6ra30_%A_%a.out --dependency=afterok:$J286 --array=1%$CARDS task_g19_e6ra30.sbatch g19_e6ra30.s2_full.tsv)   # ra30.py ouro_2_6b_base s2_full
K ouro $J287
S $J287 g19_e6ra30.s2_full.tsv 1
ALL="$ALL:$J287"
echo "group g20_e6ra29: 4 jobs, 2.7 card-hours (B-EXT6: RA-29 MCQ late edit on Ouro-2.6B base, AQuA and CSQA (pilot, then full))"
J288=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g20_e6ra29_%A_%a.out --array=1%$CARDS task_g20_e6ra29.sbatch g20_e6ra29.s1_pilot.tsv)   # ra29.py ouro_2_6b_base s1_pilot
K ouro $J288
S $J288 g20_e6ra29.s1_pilot.tsv 1
ALL="$ALL:$J288"
J289=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g20_e6ra29_%A_%a.out --dependency=afterok:$J288 --array=1%$CARDS task_g20_e6ra29.sbatch g20_e6ra29.s2_full.tsv)   # ra29.py ouro_2_6b_base s2_full
K ouro $J289
S $J289 g20_e6ra29.s2_full.tsv 1
ALL="$ALL:$J289"
J290=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g20_e6ra29_%A_%a.out --array=2%$CARDS task_g20_e6ra29.sbatch g20_e6ra29.s1_pilot.tsv)   # ra29.py ouro_2_6b_base s1_pilot
K ouro $J290
S $J290 g20_e6ra29.s1_pilot.tsv 2
ALL="$ALL:$J290"
J291=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g20_e6ra29_%A_%a.out --dependency=afterok:$J290 --array=2%$CARDS task_g20_e6ra29.sbatch g20_e6ra29.s2_full.tsv)   # ra29.py ouro_2_6b_base s2_full
K ouro $J291
S $J291 g20_e6ra29.s2_full.tsv 2
ALL="$ALL:$J291"
echo "group g21_e6ra31a: 2 jobs, 3.3 card-hours (B-EXT6: RA-31 Part A scaled writes on Ouro-2.6B base (pilot, then full))"
J292=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g21_e6ra31a_%A_%a.out --array=1%$CARDS task_g21_e6ra31a.sbatch g21_e6ra31a.s1_pilot.tsv)   # ra31.py ouro_2_6b_base s1_pilot
K ouro $J292
S $J292 g21_e6ra31a.s1_pilot.tsv 1
ALL="$ALL:$J292"
J293=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g21_e6ra31a_%A_%a.out --dependency=afterok:$J292 --array=1%$CARDS task_g21_e6ra31a.sbatch g21_e6ra31a.s2_full.tsv)   # ra31.py ouro_2_6b_base s2_full
K ouro $J293
S $J293 g21_e6ra31a.s2_full.tsv 1
ALL="$ALL:$J293"
echo "group g22_e6ra31b: 16 jobs, 20.1 card-hours (g22_e6ra31b)"
J294=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g22_e6ra31b_%A_%a.out --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15%$CARDS task_g22_e6ra31b.sbatch g22_e6ra31b.s1_shard.tsv)   # ra31_gen.py ouro_1_4b_base s1_shard
K ouro $J294
S $J294 g22_e6ra31b.s1_shard.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
ALL="$ALL:$J294"
J295=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g22_e6ra31b_%A_%a.out --dependency=afterok:$J294 --array=1%$CARDS task_g22_e6ra31b.sbatch g22_e6ra31b.s2_summarize.tsv)   # ra31_gen.py ouro_1_4b_base s2_summarize
K ouro $J295
S $J295 g22_e6ra31b.s2_summarize.tsv 1
ALL="$ALL:$J295"
echo "group g23_e6merge: 3 jobs, 0.2 card-hours (B-EXT6: copy each test's outputs into merged/ra29, ra30, ra31)"
J296=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g23_e6merge_%A_%a.out --dependency=afterany:$J288:$J289:$J290:$J291 --array=1%$CARDS task_g23_e6merge.sbatch g23_e6merge.s1_merge.tsv)   # merge_ext6.py all s1_merge
K none $J296
S $J296 g23_e6merge.s1_merge.tsv 1
ALL="$ALL:$J296"
J297=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g23_e6merge_%A_%a.out --dependency=afterany:$J286:$J287 --array=2%$CARDS task_g23_e6merge.sbatch g23_e6merge.s1_merge.tsv)   # merge_ext6.py all s1_merge
K none $J297
S $J297 g23_e6merge.s1_merge.tsv 2
ALL="$ALL:$J297"
J298=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g23_e6merge_%A_%a.out --dependency=afterany:$J292:$J293:$J294:$J295 --array=3%$CARDS task_g23_e6merge.sbatch g23_e6merge.s1_merge.tsv)   # merge_ext6.py all s1_merge
K none $J298
S $J298 g23_e6merge.s1_merge.tsv 3
ALL="$ALL:$J298"
echo "group g24_e6bra29: 4 jobs, 10.7 card-hours (B-EXT6b priority 1: RA-29 MMLU and HellaSwag on Ouro-1.4B and 2.6B base (one job per cell))"
J299=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g24_e6bra29_%A_%a.out --array=1%$CARDS task_g24_e6bra29.sbatch g24_e6bra29.s1_cell.tsv)   # ra29.py ouro_1_4b_base s1_cell
K ouro $J299
S $J299 g24_e6bra29.s1_cell.tsv 1
ALL="$ALL:$J299"
J300=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g24_e6bra29_%A_%a.out --array=2%$CARDS task_g24_e6bra29.sbatch g24_e6bra29.s1_cell.tsv)   # ra29.py ouro_2_6b_base s1_cell
K ouro $J300
S $J300 g24_e6bra29.s1_cell.tsv 2
ALL="$ALL:$J300"
J301=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g24_e6bra29_%A_%a.out --array=3%$CARDS task_g24_e6bra29.sbatch g24_e6bra29.s1_cell.tsv)   # ra29.py ouro_1_4b_base s1_cell
K ouro $J301
S $J301 g24_e6bra29.s1_cell.tsv 3
ALL="$ALL:$J301"
J302=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g24_e6bra29_%A_%a.out --array=4%$CARDS task_g24_e6bra29.sbatch g24_e6bra29.s1_cell.tsv)   # ra29.py ouro_2_6b_base s1_cell
K ouro $J302
S $J302 g24_e6bra29.s1_cell.tsv 4
ALL="$ALL:$J302"
echo "group g25_e6bra31t: 4 jobs, 4.5 card-hours (B-EXT6b priority 2: RA-31 Part A on Ouro-1.4B and 2.6B Thinking (pilot, then full))"
J303=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g25_e6bra31t_%A_%a.out --array=1%$CARDS task_g25_e6bra31t.sbatch g25_e6bra31t.s1_pilot.tsv)   # ra31.py ouro_1_4b_think s1_pilot
K ouro $J303
S $J303 g25_e6bra31t.s1_pilot.tsv 1
ALL="$ALL:$J303"
J304=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g25_e6bra31t_%A_%a.out --dependency=afterok:$J303 --array=1%$CARDS task_g25_e6bra31t.sbatch g25_e6bra31t.s2_full.tsv)   # ra31.py ouro_1_4b_think s2_full
K ouro $J304
S $J304 g25_e6bra31t.s2_full.tsv 1
ALL="$ALL:$J304"
J305=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g25_e6bra31t_%A_%a.out --array=2%$CARDS task_g25_e6bra31t.sbatch g25_e6bra31t.s1_pilot.tsv)   # ra31.py ouro_2_6b_think s1_pilot
K ouro $J305
S $J305 g25_e6bra31t.s1_pilot.tsv 2
ALL="$ALL:$J305"
J306=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g25_e6bra31t_%A_%a.out --dependency=afterok:$J305 --array=2%$CARDS task_g25_e6bra31t.sbatch g25_e6bra31t.s2_full.tsv)   # ra31.py ouro_2_6b_think s2_full
K ouro $J306
S $J306 g25_e6bra31t.s2_full.tsv 2
ALL="$ALL:$J306"
echo "group g26_e6bra30: 2 jobs, 4.1 card-hours (B-EXT6b priority 3: RA-30 ASDiv on Ouro-2.6B base (pilot2, then full2))"
J307=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g26_e6bra30_%A_%a.out --array=1%$CARDS task_g26_e6bra30.sbatch g26_e6bra30.s1_pilot.tsv)   # ra30.py ouro_2_6b_base s1_pilot
K ouro $J307
S $J307 g26_e6bra30.s1_pilot.tsv 1
ALL="$ALL:$J307"
J308=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g26_e6bra30_%A_%a.out --dependency=afterok:$J307 --array=1%$CARDS task_g26_e6bra30.sbatch g26_e6bra30.s2_full.tsv)   # ra30.py ouro_2_6b_base s2_full
K ouro $J308
S $J308 g26_e6bra30.s2_full.tsv 1
ALL="$ALL:$J308"
echo "group g28_e6bmerge: 3 jobs, 0.2 card-hours (B-EXT6b: copy each test's outputs into merged/ra29, ra30, ra31)"
J309=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g28_e6bmerge_%A_%a.out --dependency=afterany:$J299:$J300:$J301:$J302 --array=1%$CARDS task_g28_e6bmerge.sbatch g28_e6bmerge.s1_merge.tsv)   # merge_ext6.py all s1_merge
K none $J309
S $J309 g28_e6bmerge.s1_merge.tsv 1
ALL="$ALL:$J309"
J310=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g28_e6bmerge_%A_%a.out --dependency=afterany:$J307:$J308 --array=2%$CARDS task_g28_e6bmerge.sbatch g28_e6bmerge.s1_merge.tsv)   # merge_ext6.py all s1_merge
K none $J310
S $J310 g28_e6bmerge.s1_merge.tsv 2
ALL="$ALL:$J310"
J311=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g28_e6bmerge_%A_%a.out --dependency=afterany:$J303:$J304:$J305:$J306 --array=3%$CARDS task_g28_e6bmerge.sbatch g28_e6bmerge.s1_merge.tsv)   # merge_ext6.py all s1_merge
K none $J311
S $J311 g28_e6bmerge.s1_merge.tsv 3
ALL="$ALL:$J311"
echo "group g5_huginn: 88 jobs, 135.1 card-hours (Huginn incl. the RA-23 LRE cell)"
J312=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J312
S $J312 g5_huginn.s1_scan.tsv 1
ALL="$ALL:$J312"
J313=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J312 --array=1,2%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J313
S $J313 g5_huginn.s2_rows.tsv 1,2
ALL="$ALL:$J313"
J314=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J313 --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J314
S $J314 g5_huginn.s3_merge.tsv 1
ALL="$ALL:$J314"
J315=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J314 --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J315
S $J315 g5_huginn.s4_summarize.tsv 1
ALL="$ALL:$J315"
J316=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J316
S $J316 g5_huginn.s1_scan.tsv 2
ALL="$ALL:$J316"
J317=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J316 --array=3,4,5,6,7,8,9,10,11,12,13,14,15,16%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J317
S $J317 g5_huginn.s2_rows.tsv 3,4,5,6,7,8,9,10,11,12,13,14,15,16
ALL="$ALL:$J317"
J318=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J317 --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J318
S $J318 g5_huginn.s3_merge.tsv 2
ALL="$ALL:$J318"
J319=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J318 --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J319
S $J319 g5_huginn.s4_summarize.tsv 2
ALL="$ALL:$J319"
J320=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J320
S $J320 g5_huginn.s1_scan.tsv 3
ALL="$ALL:$J320"
J321=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J320 --array=17,18,19,20,21,22,23,24%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J321
S $J321 g5_huginn.s2_rows.tsv 17,18,19,20,21,22,23,24
ALL="$ALL:$J321"
J322=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J321 --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J322
S $J322 g5_huginn.s3_merge.tsv 3
ALL="$ALL:$J322"
J323=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J322 --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J323
S $J323 g5_huginn.s4_summarize.tsv 3
ALL="$ALL:$J323"
J324=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J324
S $J324 g5_huginn.s1_scan.tsv 4
ALL="$ALL:$J324"
J325=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J324 --array=25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J325
S $J325 g5_huginn.s2_rows.tsv 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
ALL="$ALL:$J325"
J326=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J325 --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J326
S $J326 g5_huginn.s3_merge.tsv 4
ALL="$ALL:$J326"
J327=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J326 --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J327
S $J327 g5_huginn.s4_summarize.tsv 4
ALL="$ALL:$J327"
J328=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J328
S $J328 g5_huginn.s1_scan.tsv 5
ALL="$ALL:$J328"
J329=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J328 --array=41,42,43,44,45%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J329
S $J329 g5_huginn.s2_rows.tsv 41,42,43,44,45
ALL="$ALL:$J329"
J330=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J329 --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J330
S $J330 g5_huginn.s3_merge.tsv 5
ALL="$ALL:$J330"
J331=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J330 --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J331
S $J331 g5_huginn.s4_summarize.tsv 5
ALL="$ALL:$J331"
J332=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J332
S $J332 g5_huginn.s1_scan.tsv 6
ALL="$ALL:$J332"
J333=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J332 --array=46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J333
S $J333 g5_huginn.s2_rows.tsv 46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
ALL="$ALL:$J333"
J334=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J333 --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J334
S $J334 g5_huginn.s3_merge.tsv 6
ALL="$ALL:$J334"
J335=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J334 --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J335
S $J335 g5_huginn.s4_summarize.tsv 6
ALL="$ALL:$J335"
J336=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J336
S $J336 g5_huginn.s1_scan.tsv 7
ALL="$ALL:$J336"
J337=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J336 --array=62,63,64,65,66,67%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J337
S $J337 g5_huginn.s2_rows.tsv 62,63,64,65,66,67
ALL="$ALL:$J337"
J338=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J337 --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J338
S $J338 g5_huginn.s3_merge.tsv 7
ALL="$ALL:$J338"
J339=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J338 --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J339
S $J339 g5_huginn.s4_summarize.tsv 7
ALL="$ALL:$J339"
echo "group g9_merge: 1 jobs, 0.2 card-hours (merge and grid tables)"
J340=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g9_merge_%A_%a.out --dependency=afterany${ALL} --array=1%$CARDS task_g9_merge.sbatch g9_merge.s1_merge.tsv)   # merge.py all s1_merge
K none $J340
S $J340 g9_merge.s1_merge.tsv 1
echo "submitted: 1023 jobs, 1097.9 card-hours (the xllm build excluded)"
