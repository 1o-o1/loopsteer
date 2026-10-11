#!/bin/bash
# Step 2 of 2: every other job in queue order, after the xllm build-and-import job finished OK. One pipeline
# = one (group, script, model, output folder); its stages chain with afterok, so a failing config blocks only
# its own pipeline (RA-20 item 4). Pipelines and groups queue at once; the order below is the queue order for
# a throttled queue (item 8). Each sbatch call names the task indices of one stage list. The merge job waits
# for every pipeline (afterany). Usage: source <OUT_ROOT>/cluster.env && bash jobs/submit_10_rest.sh
# (jobs/submit_chain.sh runs it as its last step.) It stops without $OUT_ROOT/_smoke.ok, without the preflight
# done marker, and when cluster.env lacks PARTITION, ACCOUNT, LOG_ROOT or the node restriction SLURM_RESTRICT.
# Run jobs/stage_done.sh first: it copies the done folders (done_local.json) into $OUT_ROOT and leaves
# $OUT_ROOT/_staged.ok; without that marker this script stops.
set -euo pipefail
cd "$(dirname "$0")"
CARDS=${CARDS:-48}
# cluster.env: the one already sourced, else $OUT_ROOT/cluster.env, else the first-push location
if [ -z "${PARTITION:-}" ] || [ -z "${REPO:-}" ]; then
  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "../env/cluster.env"; do
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
CHECK="staged done folders $OUT_ROOT/_staged.ok"
if [ ! -f "$OUT_ROOT/_staged.ok" ]; then
  echo "the done folders are not staged (no $OUT_ROOT/_staged.ok): run DONE_SRC=<pilot artifact folder> OUT_ROOT=$OUT_ROOT bash jobs/stage_done.sh"; exit 5
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
echo "group g1_copy: 20 jobs, 22.9 card-hours (GSM8K copy and derived)"
J1=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=1,2,3,4,5,6,7%$CARDS task_g1_copy.sbatch g1_copy.s4_rows.tsv)   # k17_varbench.py ouro_2_6b_think s4_rows
K ouro $J1
S $J1 g1_copy.s4_rows.tsv 1,2,3,4,5,6,7
ALL="$ALL:$J1"
J2=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J1 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s5_merge.tsv)   # k17_varbench.py ouro_2_6b_think s5_merge
K ouro $J2
S $J2 g1_copy.s5_merge.tsv 1
ALL="$ALL:$J2"
J3=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J2 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s6_summarize.tsv)   # k17_varbench.py ouro_2_6b_think s6_summarize
K ouro $J3
S $J3 g1_copy.s6_summarize.tsv 1
ALL="$ALL:$J3"
J4=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=1%$CARDS task_g1_copy.sbatch g1_copy.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J4
S $J4 g1_copy.s1_scan.tsv 1
ALL="$ALL:$J4"
J5=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J4 --array=1,2,3,4,5,6,7,8%$CARDS task_g1_copy.sbatch g1_copy.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J5
S $J5 g1_copy.s2_rows.tsv 1,2,3,4,5,6,7,8
ALL="$ALL:$J5"
J6=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J5 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J6
S $J6 g1_copy.s3_merge.tsv 1
ALL="$ALL:$J6"
J7=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J6 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J7
S $J7 g1_copy.s4_summarize.tsv 1
ALL="$ALL:$J7"
echo "group g2_gen: 144 jobs, 62.9 card-hours (generation, GSM8K then SVAMP, McLeish GSM8K rows)"
J8=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_2_6b_base s4_row
K gen $J8
S $J8 g2_gen.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
ALL="$ALL:$J8"
J9=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J8 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s5_anschain_presence.tsv)   # steer_gen.py ouro_2_6b_base s5_anschain_presence
K gen $J9
S $J9 g2_gen.s5_anschain_presence.tsv 1
ALL="$ALL:$J9"
J10=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J9 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_base s6_analyze
K gen $J10
S $J10 g2_gen.s6_analyze.tsv 1
ALL="$ALL:$J10"
J11=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=1%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_think s1_checks
K gen $J11
S $J11 g2_gen.s1_checks.tsv 1
ALL="$ALL:$J11"
J12=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J11 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_think s2_prep_screen
K gen $J12
S $J12 g2_gen.s2_prep_screen.tsv 1
ALL="$ALL:$J12"
J13=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J12 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_think s3_select
K gen $J13
S $J13 g2_gen.s3_select.tsv 1
ALL="$ALL:$J13"
J14=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J13 --array=20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_think s4_row
K gen $J14
S $J14 g2_gen.s4_row.tsv 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42
ALL="$ALL:$J14"
J15=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J14 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s5_anschain_presence.tsv)   # steer_gen.py ouro_1_4b_think s5_anschain_presence
K gen $J15
S $J15 g2_gen.s5_anschain_presence.tsv 2
ALL="$ALL:$J15"
J16=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J15 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_think s6_analyze
K gen $J16
S $J16 g2_gen.s6_analyze.tsv 2
ALL="$ALL:$J16"
J17=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=2%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_base s1_checks
K gen $J17
S $J17 g2_gen.s1_checks.tsv 2
ALL="$ALL:$J17"
J18=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J17 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_base s2_prep_screen
K gen $J18
S $J18 g2_gen.s2_prep_screen.tsv 2
ALL="$ALL:$J18"
J19=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J18 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_base s3_select
K gen $J19
S $J19 g2_gen.s3_select.tsv 2
ALL="$ALL:$J19"
J20=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J19 --array=43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_base s4_row
K gen $J20
S $J20 g2_gen.s4_row.tsv 43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65
ALL="$ALL:$J20"
J21=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J20 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_base s6_analyze
K gen $J21
S $J21 g2_gen.s6_analyze.tsv 3
ALL="$ALL:$J21"
J22=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=3%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_2_6b_base s1_checks
K gen $J22
S $J22 g2_gen.s1_checks.tsv 3
ALL="$ALL:$J22"
J23=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J22 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_base s2_prep_screen
K gen $J23
S $J23 g2_gen.s2_prep_screen.tsv 3
ALL="$ALL:$J23"
J24=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J23 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_2_6b_base s3_select
K gen $J24
S $J24 g2_gen.s3_select.tsv 3
ALL="$ALL:$J24"
J25=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J24 --array=66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_2_6b_base s4_row
K gen $J25
S $J25 g2_gen.s4_row.tsv 66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88
ALL="$ALL:$J25"
J26=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J25 --array=4%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_base s6_analyze
K gen $J26
S $J26 g2_gen.s6_analyze.tsv 4
ALL="$ALL:$J26"
J27=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_think s4_row
K gen $J27
S $J27 g2_gen.s4_row.tsv 89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107
ALL="$ALL:$J27"
J28=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J27 --array=5%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_think s6_analyze
K gen $J28
S $J28 g2_gen.s6_analyze.tsv 5
ALL="$ALL:$J28"
J29=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen_raven.py mcleish_llama32_r32 s4_row
K mcleish $J29
S $J29 g2_gen.s4_row.tsv 108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
ALL="$ALL:$J29"
J30=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J29 --array=6%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen_raven.py mcleish_llama32_r32 s6_analyze
K mcleish $J30
S $J30 g2_gen.s6_analyze.tsv 6
ALL="$ALL:$J30"
echo "group g3_map: 85 jobs, 106.6 card-hours (edit-site map at full pools)"
J31=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=1%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_base s1_run
K ouro $J31
S $J31 g3_map.s1_run.tsv 1
ALL="$ALL:$J31"
J32=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=2%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_base s1_run
K ouro $J32
S $J32 g3_map.s1_run.tsv 2
ALL="$ALL:$J32"
J33=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=3%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J33
S $J33 g3_map.s1_run.tsv 3
ALL="$ALL:$J33"
J34=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=1%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J34
S $J34 g3_map.s1_scan.tsv 1
ALL="$ALL:$J34"
J35=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J34 --array=1,2,3,4,5,6,7,8,9,10,11%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J35
S $J35 g3_map.s2_rows.tsv 1,2,3,4,5,6,7,8,9,10,11
ALL="$ALL:$J35"
J36=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J35 --array=1%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J36
S $J36 g3_map.s3_merge.tsv 1
ALL="$ALL:$J36"
J37=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J36 --array=1%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J37
S $J37 g3_map.s4_summarize.tsv 1
ALL="$ALL:$J37"
J38=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=2%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J38
S $J38 g3_map.s1_scan.tsv 2
ALL="$ALL:$J38"
J39=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J38 --array=12,13,14,15,16,17,18,19,20,21,22,23,24%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J39
S $J39 g3_map.s2_rows.tsv 12,13,14,15,16,17,18,19,20,21,22,23,24
ALL="$ALL:$J39"
J40=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J39 --array=2%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J40
S $J40 g3_map.s3_merge.tsv 2
ALL="$ALL:$J40"
J41=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J40 --array=2%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J41
S $J41 g3_map.s4_summarize.tsv 2
ALL="$ALL:$J41"
J42=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=3%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J42
S $J42 g3_map.s1_scan.tsv 3
ALL="$ALL:$J42"
J43=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J42 --array=25,26,27,28,29,30,31,32,33,34%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J43
S $J43 g3_map.s2_rows.tsv 25,26,27,28,29,30,31,32,33,34
ALL="$ALL:$J43"
J44=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J43 --array=3%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J44
S $J44 g3_map.s3_merge.tsv 3
ALL="$ALL:$J44"
J45=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J44 --array=3%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J45
S $J45 g3_map.s4_summarize.tsv 3
ALL="$ALL:$J45"
J46=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=4%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J46
S $J46 g3_map.s1_scan.tsv 4
ALL="$ALL:$J46"
J47=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J46 --array=35,36,37,38,39,40,41,42,43,44%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J47
S $J47 g3_map.s2_rows.tsv 35,36,37,38,39,40,41,42,43,44
ALL="$ALL:$J47"
J48=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J47 --array=4%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J48
S $J48 g3_map.s3_merge.tsv 4
ALL="$ALL:$J48"
J49=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J48 --array=4%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J49
S $J49 g3_map.s4_summarize.tsv 4
ALL="$ALL:$J49"
J50=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=5%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J50
S $J50 g3_map.s1_scan.tsv 5
ALL="$ALL:$J50"
J51=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J50 --array=45,46,47,48,49%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J51
S $J51 g3_map.s2_rows.tsv 45,46,47,48,49
ALL="$ALL:$J51"
J52=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J51 --array=5%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J52
S $J52 g3_map.s3_merge.tsv 5
ALL="$ALL:$J52"
J53=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J52 --array=5%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J53
S $J53 g3_map.s4_summarize.tsv 5
ALL="$ALL:$J53"
J54=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=6%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J54
S $J54 g3_map.s1_scan.tsv 6
ALL="$ALL:$J54"
J55=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J54 --array=50%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J55
S $J55 g3_map.s2_rows.tsv 50
ALL="$ALL:$J55"
J56=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J55 --array=6%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J56
S $J56 g3_map.s3_merge.tsv 6
ALL="$ALL:$J56"
J57=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J56 --array=6%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J57
S $J57 g3_map.s4_summarize.tsv 6
ALL="$ALL:$J57"
J58=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=7%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J58
S $J58 g3_map.s1_scan.tsv 7
ALL="$ALL:$J58"
J59=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J58 --array=51,52%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J59
S $J59 g3_map.s2_rows.tsv 51,52
ALL="$ALL:$J59"
J60=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J59 --array=7%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J60
S $J60 g3_map.s3_merge.tsv 7
ALL="$ALL:$J60"
J61=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J60 --array=7%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J61
S $J61 g3_map.s4_summarize.tsv 7
ALL="$ALL:$J61"
J62=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=8%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J62
S $J62 g3_map.s1_scan.tsv 8
ALL="$ALL:$J62"
J63=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J62 --array=53,54,55,56%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J63
S $J63 g3_map.s2_rows.tsv 53,54,55,56
ALL="$ALL:$J63"
J64=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J63 --array=8%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J64
S $J64 g3_map.s3_merge.tsv 8
ALL="$ALL:$J64"
J65=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J64 --array=8%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J65
S $J65 g3_map.s4_summarize.tsv 8
ALL="$ALL:$J65"
J66=$(sbatch --parsable $SB $(H parcae) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=4%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k17_parcae.py parcae_370m s1_run
K parcae $J66
S $J66 g3_map.s1_run.tsv 4
ALL="$ALL:$J66"
J67=$(sbatch --parsable $SB $(H parcae) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=5%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k17_parcae.py parcae_1_3b s1_run
K parcae $J67
S $J67 g3_map.s1_run.tsv 5
ALL="$ALL:$J67"
echo "group g4_cd: 4 jobs, 3.5 card-hours (C and D scripts (raven branch-freeze))"
J68=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=1%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py mcleish_llama32_r32 s1_run
K raven $J68
S $J68 g4_cd.s1_run.tsv 1
ALL="$ALL:$J68"
J69=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=2%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py mcleish_llama32_r32 s1_run
K raven $J69
S $J69 g4_cd.s1_run.tsv 2
ALL="$ALL:$J69"
J70=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=3%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py huginn_0125 s1_run
K raven $J70
S $J70 g4_cd.s1_run.tsv 3
ALL="$ALL:$J70"
J71=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=4%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py huginn_0125 s1_run
K raven $J71
S $J71 g4_cd.s1_run.tsv 4
ALL="$ALL:$J71"
echo "group g6_ifm: 50 jobs, 52.9 card-hours (IFM wave incl. the RA-23 IFM branch-freeze)"
J72=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro s2_window
K ifm $J72
S $J72 g6_ifm.s2_window.tsv 1
ALL="$ALL:$J72"
J73=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J72 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro s3_scan
K ifm $J73
S $J73 g6_ifm.s3_scan.tsv 1
ALL="$ALL:$J73"
J74=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J73 --array=1,2,3,4,5%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro s4_rows
K ifm $J74
S $J74 g6_ifm.s4_rows.tsv 1,2,3,4,5
ALL="$ALL:$J74"
J75=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J74 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro s5_merge
K ifm $J75
S $J75 g6_ifm.s5_merge.tsv 1
ALL="$ALL:$J75"
J76=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J75 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro s6_summarize
K ifm $J76
S $J76 g6_ifm.s6_summarize.tsv 1
ALL="$ALL:$J76"
J77=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro_inj s2_window
K ifm $J77
S $J77 g6_ifm.s2_window.tsv 2
ALL="$ALL:$J77"
J78=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J77 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro_inj s3_scan
K ifm $J78
S $J78 g6_ifm.s3_scan.tsv 2
ALL="$ALL:$J78"
J79=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J78 --array=6,7,8,9,10%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro_inj s4_rows
K ifm $J79
S $J79 g6_ifm.s4_rows.tsv 6,7,8,9,10
ALL="$ALL:$J79"
J80=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J79 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro_inj s5_merge
K ifm $J80
S $J80 g6_ifm.s5_merge.tsv 2
ALL="$ALL:$J80"
J81=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J80 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro_inj s6_summarize
K ifm $J81
S $J81 g6_ifm.s6_summarize.tsv 2
ALL="$ALL:$J81"
J82=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_d112 s2_window
K ifm $J82
S $J82 g6_ifm.s2_window.tsv 3
ALL="$ALL:$J82"
J83=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J82 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_d112 s3_scan
K ifm $J83
S $J83 g6_ifm.s3_scan.tsv 3
ALL="$ALL:$J83"
J84=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J83 --array=11,12,13,14,15%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_d112 s4_rows
K ifm $J84
S $J84 g6_ifm.s4_rows.tsv 11,12,13,14,15
ALL="$ALL:$J84"
J85=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J84 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_d112 s5_merge
K ifm $J85
S $J85 g6_ifm.s5_merge.tsv 3
ALL="$ALL:$J85"
J86=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J85 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_d112 s6_summarize
K ifm $J86
S $J86 g6_ifm.s6_summarize.tsv 3
ALL="$ALL:$J86"
J87=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_full.py ifm_p1_ouro_inj s1_run
K ifm $J87
S $J87 g6_ifm.s1_run.tsv 1
ALL="$ALL:$J87"
J88=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_full.py ifm_d112 s1_run
K ifm $J88
S $J88 g6_ifm.s1_run.tsv 2
ALL="$ALL:$J88"
J89=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p1_mid s1_run
K ifm $J89
S $J89 g6_ifm.s1_run.tsv 3
ALL="$ALL:$J89"
J90=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=4%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p1_mid_inj s1_run
K ifm $J90
S $J90 g6_ifm.s1_run.tsv 4
ALL="$ALL:$J90"
J91=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=5%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_r5 s1_run
K ifm $J91
S $J91 g6_ifm.s1_run.tsv 5
ALL="$ALL:$J91"
J92=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=6%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_diag s1_run
K ifm $J92
S $J92 g6_ifm.s1_run.tsv 6
ALL="$ALL:$J92"
J93=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=7%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_ortho s1_run
K ifm $J93
S $J93 g6_ifm.s1_run.tsv 7
ALL="$ALL:$J93"
J94=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=8%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro s1_run
K ifm $J94
S $J94 g6_ifm.s1_run.tsv 8
ALL="$ALL:$J94"
J95=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=9%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro s1_run
K ifm $J95
S $J95 g6_ifm.s1_run.tsv 9
ALL="$ALL:$J95"
J96=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=10%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J96
S $J96 g6_ifm.s1_run.tsv 10
ALL="$ALL:$J96"
J97=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=11%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J97
S $J97 g6_ifm.s1_run.tsv 11
ALL="$ALL:$J97"
J98=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=12%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J98
S $J98 g6_ifm.s1_run.tsv 12
ALL="$ALL:$J98"
J99=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=13%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J99
S $J99 g6_ifm.s1_run.tsv 13
ALL="$ALL:$J99"
J100=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=14%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J100
S $J100 g6_ifm.s1_run.tsv 14
ALL="$ALL:$J100"
J101=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=15%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J101
S $J101 g6_ifm.s1_run.tsv 15
ALL="$ALL:$J101"
J102=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=16%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J102
S $J102 g6_ifm.s1_run.tsv 16
ALL="$ALL:$J102"
J103=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=17%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J103
S $J103 g6_ifm.s1_run.tsv 17
ALL="$ALL:$J103"
J104=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=18%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_diag s1_run
K ifm $J104
S $J104 g6_ifm.s1_run.tsv 18
ALL="$ALL:$J104"
J105=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=19%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_d112 s1_run
K ifm $J105
S $J105 g6_ifm.s1_run.tsv 19
ALL="$ALL:$J105"
J106=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=20%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_d112 s1_run
K ifm $J106
S $J106 g6_ifm.s1_run.tsv 20
ALL="$ALL:$J106"
J107=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=21%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J107
S $J107 g6_ifm.s1_run.tsv 21
ALL="$ALL:$J107"
J108=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=22%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J108
S $J108 g6_ifm.s1_run.tsv 22
ALL="$ALL:$J108"
J109=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=23%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J109
S $J109 g6_ifm.s1_run.tsv 23
ALL="$ALL:$J109"
echo "group g14_ifmy1: 7 jobs, 12.8 card-hours (extension: the Y1 arm (edit-site rows) on seven IFM cells)"
J110=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=1%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J110
S $J110 g14_ifmy1.s1_run.tsv 1
ALL="$ALL:$J110"
J111=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=2%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J111
S $J111 g14_ifmy1.s1_run.tsv 2
ALL="$ALL:$J111"
J112=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=3%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J112
S $J112 g14_ifmy1.s1_run.tsv 3
ALL="$ALL:$J112"
J113=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=4%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J113
S $J113 g14_ifmy1.s1_run.tsv 4
ALL="$ALL:$J113"
J114=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=5%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J114
S $J114 g14_ifmy1.s1_run.tsv 5
ALL="$ALL:$J114"
J115=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=6%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J115
S $J115 g14_ifmy1.s1_run.tsv 6
ALL="$ALL:$J115"
J116=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g14_ifmy1_%A_%a.out --array=7%$CARDS task_g14_ifmy1.sbatch g14_ifmy1.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_diag s1_run
K ifm $J116
S $J116 g14_ifmy1.s1_run.tsv 7
ALL="$ALL:$J116"
echo "group g7_ra23: 175 jobs, 112.1 card-hours (RA-23: Ouro Thinking branch-freeze, LRE facts)"
J117=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J117
S $J117 g7_ra23.s1_run.tsv 1
ALL="$ALL:$J117"
J118=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J118
S $J118 g7_ra23.s1_run.tsv 2
ALL="$ALL:$J118"
J119=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J119
S $J119 g7_ra23.s1_run.tsv 3
ALL="$ALL:$J119"
J120=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J120
S $J120 g7_ra23.s1_run.tsv 4
ALL="$ALL:$J120"
J121=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J121
S $J121 g7_ra23.s1_run.tsv 5
ALL="$ALL:$J121"
J122=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J122
S $J122 g7_ra23.s1_run.tsv 6
ALL="$ALL:$J122"
J123=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J123
S $J123 g7_ra23.s1_scan.tsv 1
ALL="$ALL:$J123"
J124=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J123 --array=1,2,3,4,5,6%$CARDS task_g7_ra23.sbatch g7_ra23.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J124
S $J124 g7_ra23.s2_rows.tsv 1,2,3,4,5,6
ALL="$ALL:$J124"
J125=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J124 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J125
S $J125 g7_ra23.s3_merge.tsv 1
ALL="$ALL:$J125"
J126=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J125 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J126
S $J126 g7_ra23.s4_summarize.tsv 1
ALL="$ALL:$J126"
J127=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J127
S $J127 g7_ra23.s1_scan.tsv 2
ALL="$ALL:$J127"
J128=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J127 --array=7%$CARDS task_g7_ra23.sbatch g7_ra23.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J128
S $J128 g7_ra23.s2_rows.tsv 7
ALL="$ALL:$J128"
J129=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J128 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J129
S $J129 g7_ra23.s3_merge.tsv 2
ALL="$ALL:$J129"
J130=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J129 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J130
S $J130 g7_ra23.s4_summarize.tsv 2
ALL="$ALL:$J130"
J131=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_base s1_checks
K gen $J131
S $J131 g7_ra23.s1_checks.tsv 1
ALL="$ALL:$J131"
J132=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J131 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_base s2_prep_screen
K gen $J132
S $J132 g7_ra23.s2_prep_screen.tsv 1
ALL="$ALL:$J132"
J133=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J132 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_base s3_select
K gen $J133
S $J133 g7_ra23.s3_select.tsv 1
ALL="$ALL:$J133"
J134=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J133 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_base s4_row
K gen $J134
S $J134 g7_ra23.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
ALL="$ALL:$J134"
J135=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J134 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_base s6_analyze
K gen $J135
S $J135 g7_ra23.s6_analyze.tsv 1
ALL="$ALL:$J135"
J136=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_base s1_checks
K gen $J136
S $J136 g7_ra23.s1_checks.tsv 2
ALL="$ALL:$J136"
J137=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J136 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_base s2_prep_screen
K gen $J137
S $J137 g7_ra23.s2_prep_screen.tsv 2
ALL="$ALL:$J137"
J138=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J137 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_base s3_select
K gen $J138
S $J138 g7_ra23.s3_select.tsv 2
ALL="$ALL:$J138"
J139=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J138 --array=23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_base s4_row
K gen $J139
S $J139 g7_ra23.s4_row.tsv 23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
ALL="$ALL:$J139"
J140=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J139 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_base s6_analyze
K gen $J140
S $J140 g7_ra23.s6_analyze.tsv 2
ALL="$ALL:$J140"
J141=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_think s1_checks
K gen $J141
S $J141 g7_ra23.s1_checks.tsv 3
ALL="$ALL:$J141"
J142=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J141 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_think s2_prep_screen
K gen $J142
S $J142 g7_ra23.s2_prep_screen.tsv 3
ALL="$ALL:$J142"
J143=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J142 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_think s3_select
K gen $J143
S $J143 g7_ra23.s3_select.tsv 3
ALL="$ALL:$J143"
J144=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J143 --array=45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_think s4_row
K gen $J144
S $J144 g7_ra23.s4_row.tsv 45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66
ALL="$ALL:$J144"
J145=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J144 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_think s6_analyze
K gen $J145
S $J145 g7_ra23.s6_analyze.tsv 3
ALL="$ALL:$J145"
J146=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_base s1_checks
K gen $J146
S $J146 g7_ra23.s1_checks.tsv 4
ALL="$ALL:$J146"
J147=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J146 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_base s2_prep_screen
K gen $J147
S $J147 g7_ra23.s2_prep_screen.tsv 4
ALL="$ALL:$J147"
J148=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J147 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_base s3_select
K gen $J148
S $J148 g7_ra23.s3_select.tsv 4
ALL="$ALL:$J148"
J149=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J148 --array=67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_base s4_row
K gen $J149
S $J149 g7_ra23.s4_row.tsv 67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88
ALL="$ALL:$J149"
J150=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J149 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_base s6_analyze
K gen $J150
S $J150 g7_ra23.s6_analyze.tsv 4
ALL="$ALL:$J150"
J151=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_base s1_checks
K gen $J151
S $J151 g7_ra23.s1_checks.tsv 5
ALL="$ALL:$J151"
J152=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J151 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_base s2_prep_screen
K gen $J152
S $J152 g7_ra23.s2_prep_screen.tsv 5
ALL="$ALL:$J152"
J153=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J152 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_base s3_select
K gen $J153
S $J153 g7_ra23.s3_select.tsv 5
ALL="$ALL:$J153"
J154=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J153 --array=89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_base s4_row
K gen $J154
S $J154 g7_ra23.s4_row.tsv 89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110
ALL="$ALL:$J154"
J155=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J154 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_base s6_analyze
K gen $J155
S $J155 g7_ra23.s6_analyze.tsv 5
ALL="$ALL:$J155"
J156=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_think s1_checks
K gen $J156
S $J156 g7_ra23.s1_checks.tsv 6
ALL="$ALL:$J156"
J157=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J156 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_think s2_prep_screen
K gen $J157
S $J157 g7_ra23.s2_prep_screen.tsv 6
ALL="$ALL:$J157"
J158=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J157 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_think s3_select
K gen $J158
S $J158 g7_ra23.s3_select.tsv 6
ALL="$ALL:$J158"
J159=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J158 --array=111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_think s4_row
K gen $J159
S $J159 g7_ra23.s4_row.tsv 111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132
ALL="$ALL:$J159"
J160=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J159 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_think s6_analyze
K gen $J160
S $J160 g7_ra23.s6_analyze.tsv 6
ALL="$ALL:$J160"
echo "group g11_gen26t: 107 jobs, 236.2 card-hours (extension item 1: generation on Ouro-2.6B-Thinking (setting a4; GSM8K, SVAMP, ASDiv, MAWPS))"
J161=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen.py ouro_2_6b_think s1_checks
K gen $J161
S $J161 g11_gen26t.s1_checks.tsv 1
ALL="$ALL:$J161"
J162=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J161 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_think s2_prep_screen
K gen $J162
S $J162 g11_gen26t.s2_prep_screen.tsv 1
ALL="$ALL:$J162"
J163=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J162 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen.py ouro_2_6b_think s3_select
K gen $J163
S $J163 g11_gen26t.s3_select.tsv 1
ALL="$ALL:$J163"
J164=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J163 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen.py ouro_2_6b_think s4_row
K gen $J164
S $J164 g11_gen26t.s4_row.tsv 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
ALL="$ALL:$J164"
J165=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J164 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s5_anschain_presence.tsv)   # steer_gen.py ouro_2_6b_think s5_anschain_presence
K gen $J165
S $J165 g11_gen26t.s5_anschain_presence.tsv 1
ALL="$ALL:$J165"
J166=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J165 --array=1%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_think s6_analyze
K gen $J166
S $J166 g11_gen26t.s6_analyze.tsv 1
ALL="$ALL:$J166"
J167=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen.py ouro_2_6b_think s1_checks
K gen $J167
S $J167 g11_gen26t.s1_checks.tsv 2
ALL="$ALL:$J167"
J168=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J167 --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_think s2_prep_screen
K gen $J168
S $J168 g11_gen26t.s2_prep_screen.tsv 2
ALL="$ALL:$J168"
J169=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J168 --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen.py ouro_2_6b_think s3_select
K gen $J169
S $J169 g11_gen26t.s3_select.tsv 2
ALL="$ALL:$J169"
J170=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J169 --array=24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen.py ouro_2_6b_think s4_row
K gen $J170
S $J170 g11_gen26t.s4_row.tsv 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46
ALL="$ALL:$J170"
J171=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J170 --array=2%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_think s6_analyze
K gen $J171
S $J171 g11_gen26t.s6_analyze.tsv 2
ALL="$ALL:$J171"
J172=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_think s1_checks
K gen $J172
S $J172 g11_gen26t.s1_checks.tsv 3
ALL="$ALL:$J172"
J173=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J172 --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_think s2_prep_screen
K gen $J173
S $J173 g11_gen26t.s2_prep_screen.tsv 3
ALL="$ALL:$J173"
J174=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J173 --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_think s3_select
K gen $J174
S $J174 g11_gen26t.s3_select.tsv 3
ALL="$ALL:$J174"
J175=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J174 --array=47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_think s4_row
K gen $J175
S $J175 g11_gen26t.s4_row.tsv 47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68
ALL="$ALL:$J175"
J176=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J175 --array=3%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_think s6_analyze
K gen $J176
S $J176 g11_gen26t.s6_analyze.tsv 3
ALL="$ALL:$J176"
J177=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_think s1_checks
K gen $J177
S $J177 g11_gen26t.s1_checks.tsv 4
ALL="$ALL:$J177"
J178=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J177 --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_think s2_prep_screen
K gen $J178
S $J178 g11_gen26t.s2_prep_screen.tsv 4
ALL="$ALL:$J178"
J179=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J178 --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_think s3_select
K gen $J179
S $J179 g11_gen26t.s3_select.tsv 4
ALL="$ALL:$J179"
J180=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J179 --array=69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90%$CARDS task_g11_gen26t.sbatch g11_gen26t.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_think s4_row
K gen $J180
S $J180 g11_gen26t.s4_row.tsv 69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90
ALL="$ALL:$J180"
J181=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g11_gen26t_%A_%a.out --dependency=afterok:$J180 --array=4%$CARDS task_g11_gen26t.sbatch g11_gen26t.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_think s6_analyze
K gen $J181
S $J181 g11_gen26t.s6_analyze.tsv 4
ALL="$ALL:$J181"
echo "group g13_paththink: 7 jobs, 50.2 card-hours (extension: restore, read window and attention block on the Ouro Thinking models)"
J182=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=1%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J182
S $J182 g13_paththink.s1_run.tsv 1
ALL="$ALL:$J182"
J183=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=2%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J183
S $J183 g13_paththink.s1_run.tsv 2
ALL="$ALL:$J183"
J184=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=3%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J184
S $J184 g13_paththink.s1_run.tsv 3
ALL="$ALL:$J184"
J185=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=4%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J185
S $J185 g13_paththink.s1_run.tsv 4
ALL="$ALL:$J185"
J186=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=5%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J186
S $J186 g13_paththink.s1_run.tsv 5
ALL="$ALL:$J186"
J187=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=6%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J187
S $J187 g13_paththink.s1_run.tsv 6
ALL="$ALL:$J187"
J188=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g13_paththink_%A_%a.out --array=7%$CARDS task_g13_paththink.sbatch g13_paththink.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_think s1_run
K ouro $J188
S $J188 g13_paththink.s1_run.tsv 7
ALL="$ALL:$J188"
echo "group g5_huginn: 88 jobs, 135.1 card-hours (Huginn incl. the RA-23 LRE cell)"
J189=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J189
S $J189 g5_huginn.s1_scan.tsv 1
ALL="$ALL:$J189"
J190=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J189 --array=1,2%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J190
S $J190 g5_huginn.s2_rows.tsv 1,2
ALL="$ALL:$J190"
J191=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J190 --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J191
S $J191 g5_huginn.s3_merge.tsv 1
ALL="$ALL:$J191"
J192=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J191 --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J192
S $J192 g5_huginn.s4_summarize.tsv 1
ALL="$ALL:$J192"
J193=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J193
S $J193 g5_huginn.s1_scan.tsv 2
ALL="$ALL:$J193"
J194=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J193 --array=3,4,5,6,7,8,9,10,11,12,13,14,15,16%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J194
S $J194 g5_huginn.s2_rows.tsv 3,4,5,6,7,8,9,10,11,12,13,14,15,16
ALL="$ALL:$J194"
J195=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J194 --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J195
S $J195 g5_huginn.s3_merge.tsv 2
ALL="$ALL:$J195"
J196=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J195 --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J196
S $J196 g5_huginn.s4_summarize.tsv 2
ALL="$ALL:$J196"
J197=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J197
S $J197 g5_huginn.s1_scan.tsv 3
ALL="$ALL:$J197"
J198=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J197 --array=17,18,19,20,21,22,23,24%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J198
S $J198 g5_huginn.s2_rows.tsv 17,18,19,20,21,22,23,24
ALL="$ALL:$J198"
J199=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J198 --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J199
S $J199 g5_huginn.s3_merge.tsv 3
ALL="$ALL:$J199"
J200=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J199 --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J200
S $J200 g5_huginn.s4_summarize.tsv 3
ALL="$ALL:$J200"
J201=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J201
S $J201 g5_huginn.s1_scan.tsv 4
ALL="$ALL:$J201"
J202=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J201 --array=25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J202
S $J202 g5_huginn.s2_rows.tsv 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
ALL="$ALL:$J202"
J203=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J202 --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J203
S $J203 g5_huginn.s3_merge.tsv 4
ALL="$ALL:$J203"
J204=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J203 --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J204
S $J204 g5_huginn.s4_summarize.tsv 4
ALL="$ALL:$J204"
J205=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J205
S $J205 g5_huginn.s1_scan.tsv 5
ALL="$ALL:$J205"
J206=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J205 --array=41,42,43,44,45%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J206
S $J206 g5_huginn.s2_rows.tsv 41,42,43,44,45
ALL="$ALL:$J206"
J207=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J206 --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J207
S $J207 g5_huginn.s3_merge.tsv 5
ALL="$ALL:$J207"
J208=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J207 --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J208
S $J208 g5_huginn.s4_summarize.tsv 5
ALL="$ALL:$J208"
J209=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J209
S $J209 g5_huginn.s1_scan.tsv 6
ALL="$ALL:$J209"
J210=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J209 --array=46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J210
S $J210 g5_huginn.s2_rows.tsv 46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
ALL="$ALL:$J210"
J211=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J210 --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J211
S $J211 g5_huginn.s3_merge.tsv 6
ALL="$ALL:$J211"
J212=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J211 --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J212
S $J212 g5_huginn.s4_summarize.tsv 6
ALL="$ALL:$J212"
J213=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J213
S $J213 g5_huginn.s1_scan.tsv 7
ALL="$ALL:$J213"
J214=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J213 --array=62,63,64,65,66,67%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J214
S $J214 g5_huginn.s2_rows.tsv 62,63,64,65,66,67
ALL="$ALL:$J214"
J215=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J214 --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J215
S $J215 g5_huginn.s3_merge.tsv 7
ALL="$ALL:$J215"
J216=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J215 --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J216
S $J216 g5_huginn.s4_summarize.tsv 7
ALL="$ALL:$J216"
echo "group g9_merge: 1 jobs, 0.2 card-hours (merge and grid tables)"
J217=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g9_merge_%A_%a.out --dependency=afterany${ALL} --array=1%$CARDS task_g9_merge.sbatch g9_merge.s1_merge.tsv)   # merge.py all s1_merge
K none $J217
S $J217 g9_merge.s1_merge.tsv 1
echo "submitted: 688 jobs, 795.5 card-hours (the xllm build excluded)"
