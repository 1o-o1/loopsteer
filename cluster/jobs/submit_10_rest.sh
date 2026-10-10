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
for f in ouro gen raven mcleish parcae ifm; do
  rm -f "$OUT_ROOT/_held/$f.jobs" "$OUT_ROOT/_held/$f.released"
  if [ -f "$OUT_ROOT/_held/$f.held" ]; then echo "family $f held ($(head -1 "$OUT_ROOT/_held/$f.held")); its jobs are submitted held (release after the fix: bash jobs/release_family.sh $f)"; fi
done
CHECK="sbatch of the stage arrays (the last printed line is the call that failed)"
ALL=""
echo "group g1_copy: 20 jobs, 22.9 card-hours (GSM8K copy and derived)"
J1=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=1,2,3,4,5,6,7%$CARDS task_g1_copy.sbatch g1_copy.s4_rows.tsv)   # k17_varbench.py ouro_2_6b_think s4_rows
K ouro $J1
ALL="$ALL:$J1"
J2=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J1 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s5_merge.tsv)   # k17_varbench.py ouro_2_6b_think s5_merge
K ouro $J2
ALL="$ALL:$J2"
J3=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J2 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s6_summarize.tsv)   # k17_varbench.py ouro_2_6b_think s6_summarize
K ouro $J3
ALL="$ALL:$J3"
J4=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --array=1%$CARDS task_g1_copy.sbatch g1_copy.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J4
ALL="$ALL:$J4"
J5=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J4 --array=1,2,3,4,5,6,7,8%$CARDS task_g1_copy.sbatch g1_copy.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J5
ALL="$ALL:$J5"
J6=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J5 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J6
ALL="$ALL:$J6"
J7=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g1_copy_%A_%a.out --dependency=afterok:$J6 --array=1%$CARDS task_g1_copy.sbatch g1_copy.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J7
ALL="$ALL:$J7"
echo "group g2_gen: 144 jobs, 62.9 card-hours (generation, GSM8K then SVAMP, McLeish GSM8K rows)"
J8=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_2_6b_base s4_row
K gen $J8
ALL="$ALL:$J8"
J9=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J8 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s5_anschain_presence.tsv)   # steer_gen.py ouro_2_6b_base s5_anschain_presence
K gen $J9
ALL="$ALL:$J9"
J10=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J9 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_base s6_analyze
K gen $J10
ALL="$ALL:$J10"
J11=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=1%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_think s1_checks
K gen $J11
ALL="$ALL:$J11"
J12=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J11 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_think s2_prep_screen
K gen $J12
ALL="$ALL:$J12"
J13=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J12 --array=1%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_think s3_select
K gen $J13
ALL="$ALL:$J13"
J14=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J13 --array=20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_think s4_row
K gen $J14
ALL="$ALL:$J14"
J15=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J14 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s5_anschain_presence.tsv)   # steer_gen.py ouro_1_4b_think s5_anschain_presence
K gen $J15
ALL="$ALL:$J15"
J16=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J15 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_think s6_analyze
K gen $J16
ALL="$ALL:$J16"
J17=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=2%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_1_4b_base s1_checks
K gen $J17
ALL="$ALL:$J17"
J18=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J17 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_1_4b_base s2_prep_screen
K gen $J18
ALL="$ALL:$J18"
J19=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J18 --array=2%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_1_4b_base s3_select
K gen $J19
ALL="$ALL:$J19"
J20=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J19 --array=43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_base s4_row
K gen $J20
ALL="$ALL:$J20"
J21=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J20 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_base s6_analyze
K gen $J21
ALL="$ALL:$J21"
J22=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=3%$CARDS task_g2_gen.sbatch g2_gen.s1_checks.tsv)   # steer_gen.py ouro_2_6b_base s1_checks
K gen $J22
ALL="$ALL:$J22"
J23=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J22 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s2_prep_screen.tsv)   # steer_gen.py ouro_2_6b_base s2_prep_screen
K gen $J23
ALL="$ALL:$J23"
J24=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J23 --array=3%$CARDS task_g2_gen.sbatch g2_gen.s3_select.tsv)   # steer_gen.py ouro_2_6b_base s3_select
K gen $J24
ALL="$ALL:$J24"
J25=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J24 --array=66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_2_6b_base s4_row
K gen $J25
ALL="$ALL:$J25"
J26=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J25 --array=4%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_2_6b_base s6_analyze
K gen $J26
ALL="$ALL:$J26"
J27=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen.py ouro_1_4b_think s4_row
K gen $J27
ALL="$ALL:$J27"
J28=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J27 --array=5%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen.py ouro_1_4b_think s6_analyze
K gen $J28
ALL="$ALL:$J28"
J29=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --array=108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127%$CARDS task_g2_gen.sbatch g2_gen.s4_row.tsv)   # steer_gen_raven.py mcleish_llama32_r32 s4_row
K mcleish $J29
ALL="$ALL:$J29"
J30=$(sbatch --parsable $SB $(H mcleish) --output=$LOG_ROOT/ra20_g2_gen_%A_%a.out --dependency=afterok:$J29 --array=6%$CARDS task_g2_gen.sbatch g2_gen.s6_analyze.tsv)   # steer_gen_raven.py mcleish_llama32_r32 s6_analyze
K mcleish $J30
ALL="$ALL:$J30"
echo "group g3_map: 85 jobs, 106.6 card-hours (edit-site map at full pools)"
J31=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=1%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_base s1_run
K ouro $J31
ALL="$ALL:$J31"
J32=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=2%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_2_6b_base s1_run
K ouro $J32
ALL="$ALL:$J32"
J33=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=3%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k16_path_ouro.py ouro_1_4b_think s1_run
K ouro $J33
ALL="$ALL:$J33"
J34=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=1%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J34
ALL="$ALL:$J34"
J35=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J34 --array=1,2,3,4,5,6,7,8,9,10,11%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J35
ALL="$ALL:$J35"
J36=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J35 --array=1%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J36
ALL="$ALL:$J36"
J37=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J36 --array=1%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J37
ALL="$ALL:$J37"
J38=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=2%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J38
ALL="$ALL:$J38"
J39=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J38 --array=12,13,14,15,16,17,18,19,20,21,22,23,24%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J39
ALL="$ALL:$J39"
J40=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J39 --array=2%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J40
ALL="$ALL:$J40"
J41=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J40 --array=2%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J41
ALL="$ALL:$J41"
J42=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=3%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J42
ALL="$ALL:$J42"
J43=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J42 --array=25,26,27,28,29,30,31,32,33,34%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J43
ALL="$ALL:$J43"
J44=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J43 --array=3%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J44
ALL="$ALL:$J44"
J45=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J44 --array=3%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J45
ALL="$ALL:$J45"
J46=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=4%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J46
ALL="$ALL:$J46"
J47=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J46 --array=35,36,37,38,39,40,41,42,43,44%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J47
ALL="$ALL:$J47"
J48=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J47 --array=4%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J48
ALL="$ALL:$J48"
J49=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J48 --array=4%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J49
ALL="$ALL:$J49"
J50=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=5%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J50
ALL="$ALL:$J50"
J51=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J50 --array=45,46,47,48,49%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J51
ALL="$ALL:$J51"
J52=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J51 --array=5%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J52
ALL="$ALL:$J52"
J53=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J52 --array=5%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J53
ALL="$ALL:$J53"
J54=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=6%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J54
ALL="$ALL:$J54"
J55=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J54 --array=50%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J55
ALL="$ALL:$J55"
J56=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J55 --array=6%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J56
ALL="$ALL:$J56"
J57=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J56 --array=6%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J57
ALL="$ALL:$J57"
J58=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=7%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J58
ALL="$ALL:$J58"
J59=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J58 --array=51,52%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J59
ALL="$ALL:$J59"
J60=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J59 --array=7%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J60
ALL="$ALL:$J60"
J61=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J60 --array=7%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J61
ALL="$ALL:$J61"
J62=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=8%$CARDS task_g3_map.sbatch g3_map.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J62
ALL="$ALL:$J62"
J63=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J62 --array=53,54,55,56%$CARDS task_g3_map.sbatch g3_map.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J63
ALL="$ALL:$J63"
J64=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J63 --array=8%$CARDS task_g3_map.sbatch g3_map.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J64
ALL="$ALL:$J64"
J65=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --dependency=afterok:$J64 --array=8%$CARDS task_g3_map.sbatch g3_map.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J65
ALL="$ALL:$J65"
J66=$(sbatch --parsable $SB $(H parcae) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=4%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k17_parcae.py parcae_370m s1_run
K parcae $J66
ALL="$ALL:$J66"
J67=$(sbatch --parsable $SB $(H parcae) --output=$LOG_ROOT/ra20_g3_map_%A_%a.out --array=5%$CARDS task_g3_map.sbatch g3_map.s1_run.tsv)   # k17_parcae.py parcae_1_3b s1_run
K parcae $J67
ALL="$ALL:$J67"
echo "group g4_cd: 4 jobs, 3.5 card-hours (C and D scripts (raven branch-freeze))"
J68=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=1%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py mcleish_llama32_r32 s1_run
K raven $J68
ALL="$ALL:$J68"
J69=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=2%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py mcleish_llama32_r32 s1_run
K raven $J69
ALL="$ALL:$J69"
J70=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=3%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py huginn_0125 s1_run
K raven $J70
ALL="$ALL:$J70"
J71=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g4_cd_%A_%a.out --array=4%$CARDS task_g4_cd.sbatch g4_cd.s1_run.tsv)   # k17_freeze_raven.py huginn_0125 s1_run
K raven $J71
ALL="$ALL:$J71"
echo "group g6_ifm: 50 jobs, 52.9 card-hours (IFM wave incl. the RA-23 IFM branch-freeze)"
J72=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro s2_window
K ifm $J72
ALL="$ALL:$J72"
J73=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J72 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro s3_scan
K ifm $J73
ALL="$ALL:$J73"
J74=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J73 --array=1,2,3,4,5%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro s4_rows
K ifm $J74
ALL="$ALL:$J74"
J75=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J74 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro s5_merge
K ifm $J75
ALL="$ALL:$J75"
J76=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J75 --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro s6_summarize
K ifm $J76
ALL="$ALL:$J76"
J77=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_p1_ouro_inj s2_window
K ifm $J77
ALL="$ALL:$J77"
J78=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J77 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_p1_ouro_inj s3_scan
K ifm $J78
ALL="$ALL:$J78"
J79=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J78 --array=6,7,8,9,10%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_p1_ouro_inj s4_rows
K ifm $J79
ALL="$ALL:$J79"
J80=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J79 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_p1_ouro_inj s5_merge
K ifm $J80
ALL="$ALL:$J80"
J81=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J80 --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_p1_ouro_inj s6_summarize
K ifm $J81
ALL="$ALL:$J81"
J82=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s2_window.tsv)   # k17_varbench.py ifm_d112 s2_window
K ifm $J82
ALL="$ALL:$J82"
J83=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J82 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s3_scan.tsv)   # k17_varbench.py ifm_d112 s3_scan
K ifm $J83
ALL="$ALL:$J83"
J84=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J83 --array=11,12,13,14,15%$CARDS task_g6_ifm.sbatch g6_ifm.s4_rows.tsv)   # k17_varbench.py ifm_d112 s4_rows
K ifm $J84
ALL="$ALL:$J84"
J85=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J84 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s5_merge.tsv)   # k17_varbench.py ifm_d112 s5_merge
K ifm $J85
ALL="$ALL:$J85"
J86=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --dependency=afterok:$J85 --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s6_summarize.tsv)   # k17_varbench.py ifm_d112 s6_summarize
K ifm $J86
ALL="$ALL:$J86"
J87=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=1%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_full.py ifm_p1_ouro_inj s1_run
K ifm $J87
ALL="$ALL:$J87"
J88=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=2%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_full.py ifm_d112 s1_run
K ifm $J88
ALL="$ALL:$J88"
J89=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=3%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p1_mid s1_run
K ifm $J89
ALL="$ALL:$J89"
J90=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=4%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p1_mid_inj s1_run
K ifm $J90
ALL="$ALL:$J90"
J91=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=5%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_r5 s1_run
K ifm $J91
ALL="$ALL:$J91"
J92=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=6%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_diag s1_run
K ifm $J92
ALL="$ALL:$J92"
J93=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=7%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k15_pairs.py ifm_p2m_ortho s1_run
K ifm $J93
ALL="$ALL:$J93"
J94=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=8%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro s1_run
K ifm $J94
ALL="$ALL:$J94"
J95=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=9%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro s1_run
K ifm $J95
ALL="$ALL:$J95"
J96=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=10%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J96
ALL="$ALL:$J96"
J97=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=11%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_ouro_inj s1_run
K ifm $J97
ALL="$ALL:$J97"
J98=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=12%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J98
ALL="$ALL:$J98"
J99=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=13%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J99
ALL="$ALL:$J99"
J100=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=14%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid s1_run
K ifm $J100
ALL="$ALL:$J100"
J101=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=15%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J101
ALL="$ALL:$J101"
J102=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=16%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J102
ALL="$ALL:$J102"
J103=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=17%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p1_mid_inj s1_run
K ifm $J103
ALL="$ALL:$J103"
J104=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=18%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_p2m_diag s1_run
K ifm $J104
ALL="$ALL:$J104"
J105=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=19%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_d112 s1_run
K ifm $J105
ALL="$ALL:$J105"
J106=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=20%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k16_path_ifm.py ifm_d112 s1_run
K ifm $J106
ALL="$ALL:$J106"
J107=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=21%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J107
ALL="$ALL:$J107"
J108=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=22%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J108
ALL="$ALL:$J108"
J109=$(sbatch --parsable $SB $(H ifm) --output=$LOG_ROOT/ra20_g6_ifm_%A_%a.out --array=23%$CARDS task_g6_ifm.sbatch g6_ifm.s1_run.tsv)   # k17_freeze_ifm.py ifm_p1_ouro s1_run
K ifm $J109
ALL="$ALL:$J109"
echo "group g7_ra23: 175 jobs, 112.1 card-hours (RA-23: Ouro Thinking branch-freeze, LRE facts)"
J110=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J110
ALL="$ALL:$J110"
J111=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J111
ALL="$ALL:$J111"
J112=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_1_4b_think s1_run
K ouro $J112
ALL="$ALL:$J112"
J113=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J113
ALL="$ALL:$J113"
J114=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J114
ALL="$ALL:$J114"
J115=$(sbatch --parsable $SB $(H ouro) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s1_run.tsv)   # k17_freeze.py ouro_2_6b_think s1_run
K ouro $J115
ALL="$ALL:$J115"
J116=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_scan.tsv)   # k17_public.py mcl_nr s1_scan
K raven $J116
ALL="$ALL:$J116"
J117=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J116 --array=1,2,3,4,5,6%$CARDS task_g7_ra23.sbatch g7_ra23.s2_rows.tsv)   # k17_public.py mcl_nr s2_rows
K raven $J117
ALL="$ALL:$J117"
J118=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J117 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s3_merge.tsv)   # k17_public.py mcl_nr s3_merge
K raven $J118
ALL="$ALL:$J118"
J119=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J118 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s4_summarize.tsv)   # k17_public.py mcl_nr s4_summarize
K raven $J119
ALL="$ALL:$J119"
J120=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_scan.tsv)   # k17_public.py ouro_think s1_scan
K raven $J120
ALL="$ALL:$J120"
J121=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J120 --array=7%$CARDS task_g7_ra23.sbatch g7_ra23.s2_rows.tsv)   # k17_public.py ouro_think s2_rows
K raven $J121
ALL="$ALL:$J121"
J122=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J121 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s3_merge.tsv)   # k17_public.py ouro_think s3_merge
K raven $J122
ALL="$ALL:$J122"
J123=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J122 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s4_summarize.tsv)   # k17_public.py ouro_think s4_summarize
K raven $J123
ALL="$ALL:$J123"
J124=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_base s1_checks
K gen $J124
ALL="$ALL:$J124"
J125=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J124 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_base s2_prep_screen
K gen $J125
ALL="$ALL:$J125"
J126=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J125 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_base s3_select
K gen $J126
ALL="$ALL:$J126"
J127=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J126 --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_base s4_row
K gen $J127
ALL="$ALL:$J127"
J128=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J127 --array=1%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_base s6_analyze
K gen $J128
ALL="$ALL:$J128"
J129=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_base s1_checks
K gen $J129
ALL="$ALL:$J129"
J130=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J129 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_base s2_prep_screen
K gen $J130
ALL="$ALL:$J130"
J131=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J130 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_base s3_select
K gen $J131
ALL="$ALL:$J131"
J132=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J131 --array=23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_base s4_row
K gen $J132
ALL="$ALL:$J132"
J133=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J132 --array=2%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_base s6_analyze
K gen $J133
ALL="$ALL:$J133"
J134=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_think s1_checks
K gen $J134
ALL="$ALL:$J134"
J135=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J134 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_think s2_prep_screen
K gen $J135
ALL="$ALL:$J135"
J136=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J135 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_think s3_select
K gen $J136
ALL="$ALL:$J136"
J137=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J136 --array=45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_think s4_row
K gen $J137
ALL="$ALL:$J137"
J138=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J137 --array=3%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_think s6_analyze
K gen $J138
ALL="$ALL:$J138"
J139=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_base s1_checks
K gen $J139
ALL="$ALL:$J139"
J140=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J139 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_base s2_prep_screen
K gen $J140
ALL="$ALL:$J140"
J141=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J140 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_base s3_select
K gen $J141
ALL="$ALL:$J141"
J142=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J141 --array=67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_base s4_row
K gen $J142
ALL="$ALL:$J142"
J143=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J142 --array=4%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_base s6_analyze
K gen $J143
ALL="$ALL:$J143"
J144=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_2_6b_base s1_checks
K gen $J144
ALL="$ALL:$J144"
J145=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J144 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_2_6b_base s2_prep_screen
K gen $J145
ALL="$ALL:$J145"
J146=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J145 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_2_6b_base s3_select
K gen $J146
ALL="$ALL:$J146"
J147=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J146 --array=89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_2_6b_base s4_row
K gen $J147
ALL="$ALL:$J147"
J148=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J147 --array=5%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_2_6b_base s6_analyze
K gen $J148
ALL="$ALL:$J148"
J149=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s1_checks.tsv)   # steer_gen_math.py ouro_1_4b_think s1_checks
K gen $J149
ALL="$ALL:$J149"
J150=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J149 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s2_prep_screen.tsv)   # steer_gen_math.py ouro_1_4b_think s2_prep_screen
K gen $J150
ALL="$ALL:$J150"
J151=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J150 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s3_select.tsv)   # steer_gen_math.py ouro_1_4b_think s3_select
K gen $J151
ALL="$ALL:$J151"
J152=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J151 --array=111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132%$CARDS task_g7_ra23.sbatch g7_ra23.s4_row.tsv)   # steer_gen_math.py ouro_1_4b_think s4_row
K gen $J152
ALL="$ALL:$J152"
J153=$(sbatch --parsable $SB $(H gen) --output=$LOG_ROOT/ra20_g7_ra23_%A_%a.out --dependency=afterok:$J152 --array=6%$CARDS task_g7_ra23.sbatch g7_ra23.s6_analyze.tsv)   # steer_gen_math.py ouro_1_4b_think s6_analyze
K gen $J153
ALL="$ALL:$J153"
echo "group g5_huginn: 88 jobs, 135.1 card-hours (Huginn incl. the RA-23 LRE cell)"
J154=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J154
ALL="$ALL:$J154"
J155=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J154 --array=1,2%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J155
ALL="$ALL:$J155"
J156=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J155 --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J156
ALL="$ALL:$J156"
J157=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J156 --array=1%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J157
ALL="$ALL:$J157"
J158=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J158
ALL="$ALL:$J158"
J159=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J158 --array=3,4,5,6,7,8,9,10,11,12,13,14,15,16%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J159
ALL="$ALL:$J159"
J160=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J159 --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J160
ALL="$ALL:$J160"
J161=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J160 --array=2%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J161
ALL="$ALL:$J161"
J162=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J162
ALL="$ALL:$J162"
J163=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J162 --array=17,18,19,20,21,22,23,24%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J163
ALL="$ALL:$J163"
J164=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J163 --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J164
ALL="$ALL:$J164"
J165=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J164 --array=3%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J165
ALL="$ALL:$J165"
J166=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J166
ALL="$ALL:$J166"
J167=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J166 --array=25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J167
ALL="$ALL:$J167"
J168=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J167 --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J168
ALL="$ALL:$J168"
J169=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J168 --array=4%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J169
ALL="$ALL:$J169"
J170=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J170
ALL="$ALL:$J170"
J171=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J170 --array=41,42,43,44,45%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J171
ALL="$ALL:$J171"
J172=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J171 --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J172
ALL="$ALL:$J172"
J173=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J172 --array=5%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J173
ALL="$ALL:$J173"
J174=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J174
ALL="$ALL:$J174"
J175=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J174 --array=46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J175
ALL="$ALL:$J175"
J176=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J175 --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J176
ALL="$ALL:$J176"
J177=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J176 --array=6%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J177
ALL="$ALL:$J177"
J178=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s1_scan.tsv)   # k17_public.py huginn s1_scan
K raven $J178
ALL="$ALL:$J178"
J179=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J178 --array=62,63,64,65,66,67%$CARDS task_g5_huginn.sbatch g5_huginn.s2_rows.tsv)   # k17_public.py huginn s2_rows
K raven $J179
ALL="$ALL:$J179"
J180=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J179 --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s3_merge.tsv)   # k17_public.py huginn s3_merge
K raven $J180
ALL="$ALL:$J180"
J181=$(sbatch --parsable $SB $(H raven) --output=$LOG_ROOT/ra20_g5_huginn_%A_%a.out --dependency=afterok:$J180 --array=7%$CARDS task_g5_huginn.sbatch g5_huginn.s4_summarize.tsv)   # k17_public.py huginn s4_summarize
K raven $J181
ALL="$ALL:$J181"
echo "group g9_merge: 1 jobs, 0.2 card-hours (merge and grid tables)"
J182=$(sbatch --parsable $SB $(H none) --output=$LOG_ROOT/ra20_g9_merge_%A_%a.out --dependency=afterany${ALL} --array=1%$CARDS task_g9_merge.sbatch g9_merge.s1_merge.tsv)   # merge.py all s1_merge
K none $J182
echo "submitted: 567 jobs, 496.3 card-hours (the xllm build excluded)"
