#!/bin/bash
# Progress of the cluster run, from the markers the templates write (no Slurm call needed):
#   source <OUT_ROOT>/cluster.env && bash loopedcot/cluster/jobs/status.sh
# The job list is the one submit_chain.sh used ($OUT_ROOT/_chain/jobs.txt: jobs/ or jobs/nolocal/).
# Per queue group of jobs/joblist.tsv:
#   done     $OUT_ROOT/_done/<id> exists
#   failed   $OUT_ROOT/_diag/<id>.txt exists and no done marker
#   running  $OUT_ROOT/_tasklog/<id>.log exists (the task started), no done marker, no diagnosis
#   pending  none of these (not started)
# With squeue on PATH it also prints the RUNNING and PENDING Slurm counts per job name, and names started tasks of a
# group that has no running Slurm task (lost: killed without a diagnosis, e.g. out of host memory; read the Slurm
# log). Then it lists the failed task ids, the chain markers, and prints every diagnosis file in $OUT_ROOT/_diag.
# Exit 0 always (a report); 1 only when OUT_ROOT or the job list is missing.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${OUT_ROOT:-}" ] && [ -f "$HERE/../env/cluster.env" ]; then . "$HERE/../env/cluster.env"; fi
: "${OUT_ROOT:?source the cluster.env that setup_cluster.sh wrote}"
LD=$(sed -n 's/^lists //p' "$OUT_ROOT/_chain/jobs.txt" 2> /dev/null)
JL="${LD:-$HERE}/joblist.tsv"
echo "job list: $JL"
[ -f "$JL" ] || { echo "no job list $JL"; exit 1; }
SQ=""
if command -v squeue > /dev/null 2>&1; then
  SQ=$(squeue -h -r -u "${USER:-$(id -un)}" -o "%j %T" 2> /dev/null || true)
fi
DONE_L=$(ls "$OUT_ROOT/_done" 2> /dev/null || true)
DIAG_L=$(ls "$OUT_ROOT/_diag" 2> /dev/null | sed -n "s/\.txt$//p" || true)
LOG_L=$(ls "$OUT_ROOT/_tasklog" 2> /dev/null | sed -n "s/\.log$//p" || true)
SQ="$SQ" DONE_L="$DONE_L" DIAG_L="$DIAG_L" LOG_L="$LOG_L" awk -F'\t' '
  function load(name, arr,   n, i, L) { n = split(ENVIRON[name], L, "\n"); for (i = 1; i <= n; i++) if (L[i] != "") arr[L[i]] = 1 }
  BEGIN { load("DONE_L", D); load("DIAG_L", X); load("LOG_L", G)
          n = split(ENVIRON["SQ"], L, "\n"); for (i = 1; i <= n; i++) { split(L[i], f, " "); if (f[2] == "RUNNING") r[f[1]]++; else if (f[2] == "PENDING") p[f[1]]++ } }
  NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
  {
    id = $col["id"]; g = $col["group"]
    if (!(g in seen)) { seen[g] = 1; order[++ng] = g }
    tot[g]++
    if (id in D) done[g]++
    else if (id in X) { fail[g]++; failed[++nf] = id }
    else if (id in G) { run[g]++; started[g] = started[g] " " id }
    else pend[g]++
  }
  END {
    printf "%-12s %6s %6s %8s %8s %7s %9s %9s\n", "group", "jobs", "done", "running", "pending", "failed", "slurm_R", "slurm_PD"
    for (i = 1; i <= ng; i++) {
      g = order[i]; j = "ra20_" g
      printf "%-12s %6d %6d %8d %8d %7d %9s %9s\n", g, tot[g], done[g], run[g], pend[g], fail[g], (ENVIRON["SQ"] == "" ? "-" : r[j] + 0), (ENVIRON["SQ"] == "" ? "-" : p[j] + 0)
      T["jobs"] += tot[g]; T["done"] += done[g]; T["run"] += run[g]; T["pend"] += pend[g]; T["fail"] += fail[g]
      if (ENVIRON["SQ"] != "" && run[g] > 0 && r[j] == 0) lost = lost "\n  " g ":" started[g]
    }
    printf "%-12s %6d %6d %8d %8d %7d\n", "total", T["jobs"], T["done"], T["run"], T["pend"], T["fail"]
    printf "\nfailed task ids (%d):\n", nf
    for (i = 1; i <= nf; i++) printf "  %s\n", failed[i]
    if (lost != "") printf "\nstarted but no running Slurm task in the group (lost without a diagnosis):%s\n", lost
  }' "$JL"
echo
echo "chain markers ($OUT_ROOT/_chain):"
ls "$OUT_ROOT/_chain" 2> /dev/null | sed 's/^/  /' || true
for m in _smoke.ok _smoke_fail.txt _smoke_override.txt _staged.ok; do [ -e "$OUT_ROOT/$m" ] && echo "  $m"; done
echo
echo "bad nodes ($OUT_ROOT/bad_nodes.txt, excluded from every later submission): $(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . | paste -sd, - || true)"
echo "run nodes ($OUT_ROOT/run_nodes.txt, chosen by the probe step): $(grep . "$OUT_ROOT/run_nodes.txt" 2> /dev/null | paste -sd, - || true)"
for f in "$OUT_ROOT"/_probe/*.out; do
  [ -e "$f" ] || continue
  echo "  probe $(basename "$f" .out): $(grep -m1 "^RC=" "$f" || echo "no RC yet") $(grep -m1 -i "CUDA unknown error\|No devices were found" "$f" || true)"
done
echo
if [ -f "$OUT_ROOT/merged/local_vs_cluster.csv" ]; then      # written by the merge job; reported, not gated
  echo "cluster rerun against the local reference ($OUT_ROOT/merged/local_vs_cluster.csv):"
  cut -d, -f1-3,6,9,10,16,17 "$OUT_ROOT/merged/local_vs_cluster.csv" | sed 's/^/  /'
  echo
fi
n=0
for f in "$OUT_ROOT"/_diag/*.txt; do
  [ -e "$f" ] || continue
  n=$((n + 1))
  echo "================ diagnosis $(basename "$f" .txt)"
  cat "$f"
done
echo "diagnosis files: $n"
exit 0
