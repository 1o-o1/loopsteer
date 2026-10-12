# Node reservation and two-lines-per-GPU packing for jobs/submit_fill.sh, jobs/submit_ext6.sh and jobs/submit_ext6b.sh
# (sourced, not run). PI rules of 2026-10-12:
# - Nodes: every node of NODELIST except the last RESERVE_NODES (default 1), which go to --exclude and stay empty;
#   with $OUT_ROOT/run_nodes.txt (the probe's healthy nodes) the used nodes are its intersection with them; nodes of
#   $OUT_ROOT/bad_nodes.txt are not used. No NODELIST: exit 2.
# - CARDS (the --array throttle, one GPU job each) defaults to used nodes x 8; two task lines share one GPU job, so
#   up to CARDS x 2 task lines run at once.
# - Packing: two task lines of one stage go into one one-GPU job (jobs/task_pack.sbatch: two background processes,
#   each through the group's own template with its own done marker and diagnosis; exit 75 requeues the job once and
#   the line already done is skipped on the rerun). A line runs alone (one line per GPU job, the group's template) when
#   its time at 0.6 x the card's speed (est_card_seconds / 0.6) exceeds PACK_MAX_H (default 9 h, 1 h under the 10 h
#   limit), or when it is a Huginn or McLeish generation line (steer_gen_raven,
#   steer_gen_math: K = 32 recurrence at batch 16, memory not verified twice on one card). An odd count leaves one
#   single-line packed job. PACK=0 runs every line alone.
# - Memory: task_pack.sbatch halves every share-of-the-card setting for a two-line job (E6_MEM_FRACTION 0.9 -> 0.45,
#   GEN_MEM_FRACTION, PROD_MEM_FRACTION 0.85 -> 0.425).
# - Time: every sbatch call passes --time=$TIME_LIMIT (default 10:00:00), which overrides the template header; the
#   partition's MaxTime (sinfo -h -o %l -p $PARTITION) is printed and a shorter one stops the script (exit 2).
RESERVE_NODES=${RESERVE_NODES:-1}
PACK=${PACK:-1}
PACK_MAX_H=${PACK_MAX_H:-9}
TIME_LIMIT=${TIME_LIMIT:-10:00:00}

to_s() {             # Slurm time (D-HH:MM:SS, D-HH, HH:MM:SS, MM:SS, MM, UNLIMITED) -> seconds; empty when unknown
  case "$1" in UNLIMITED|unlimited|infinite|INFINITE) echo 999999999; return;; esac
  echo "$1" | awk '/^[0-9]+(-[0-9]+)?(:[0-9]+)?(:[0-9]+)?$/ {
    d = 0; t = $0; if (index(t, "-")) { d = substr(t, 1, index(t, "-") - 1); t = substr(t, index(t, "-") + 1) }
    n = split(t, a, ":")
    if (index($0, "-")) s = a[1] * 3600 + (n > 1 ? a[2] * 60 : 0) + (n > 2 ? a[3] : 0)
    else if (n == 3) s = a[1] * 3600 + a[2] * 60 + a[3]; else if (n == 2) s = a[1] * 60 + a[2]; else s = a[1] * 60
    print d * 86400 + s }'
}

time_check() {       # the partition's MaxTime against TIME_LIMIT: print it; exit 2 when it is shorter
  local mx ms ts
  ts=$(to_s "$TIME_LIMIT")
  [ -n "$ts" ] || { echo "TIME_LIMIT=$TIME_LIMIT is not a Slurm time"; exit 2; }
  mx=$(sinfo -h -o %l -p "$PARTITION" 2> /dev/null | grep . | sort -u | head -1)
  ms=$(to_s "$mx")
  if [ -z "$ms" ]; then
    echo "time: --time=$TIME_LIMIT on every job; partition $PARTITION MaxTime unknown (sinfo gave '${mx:-nothing}')"; return 0
  fi
  echo "time: --time=$TIME_LIMIT on every job; partition $PARTITION MaxTime $mx"
  if [ "$ms" -lt "$ts" ]; then
    echo "the MaxTime $mx of partition $PARTITION is below TIME_LIMIT=$TIME_LIMIT: set TIME_LIMIT=$mx or use a longer partition"; exit 2
  fi
}
PACK_SPEED=0.6

nodes_setup() {      # -> USED, RESERVED, EXCL (comma lists), CSRC, CARDS
  local all cand bad
  if [ -z "${NODELIST:-}" ]; then
    echo "no node list: NODELIST (cluster.env) is empty, so the reserved node cannot be held empty (set NODELIST=<nodes>)"
    exit 2
  fi
  all=$( { scontrol show hostnames "$NODELIST" 2> /dev/null | grep . || echo "$NODELIST" | tr ',' '\n'; } | grep . )
  RESERVED=$(echo "$all" | tail -n "$RESERVE_NODES" | paste -sd, -)
  cand=$(echo "$all" | head -n "-$RESERVE_NODES" 2> /dev/null || true)
  [ "$RESERVE_NODES" -gt 0 ] || cand="$all"
  CSRC="NODELIST minus the last $RESERVE_NODES"
  if [ -s "$OUT_ROOT/run_nodes.txt" ]; then
    cand=$(echo "$cand" | grep -xF -f <(grep . "$OUT_ROOT/run_nodes.txt") || true)
    CSRC="$CSRC, within $OUT_ROOT/run_nodes.txt"
  fi
  bad=$(sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null | grep . || true)
  USED=$(echo "$cand" | grep . | grep -vxF -f <(echo "${bad:-__no_bad_node__}") | paste -sd, -)
  [ -n "$USED" ] || { echo "no usable node: $CSRC leaves none (bad nodes: ${bad:-none})"; exit 2; }
  EXCL=$(echo "$all" | grep -vxF -f <(echo "$USED" | tr ',' '\n') | paste -sd, -)
  CARDS=${CARDS:-$(( $(echo "$USED" | tr ',' '\n' | grep -c .) * 8 ))}
}

nodes_say() {
  echo "nodes: used $USED ($CSRC); reserved and empty: $RESERVED; excluded: ${EXCL:-none}"
  echo "concurrency: at most $CARDS GPU jobs of one array at once (CARDS), two task lines per packed job: up to $((CARDS * 2)) task lines at once"
}

add_excl() {         # R plus --exclude of every node not used
  if [ -n "$EXCL" ]; then
    case "$R" in *--exclude=*) R=$(echo "$R" | sed "s/--exclude=\([^ ]*\)/--exclude=\1,$EXCL/");; *) R="$R --exclude=$EXCL";; esac
  fi
}

pack_split() {       # $1 stage list, $2 comma list of line indices -> "ALONE <idxs>" and "PACK <idxs>" lines (- for none)
  awk -F'\t' -v idxs="$2" -v lst="$1" -v pack="$PACK" -v maxs="$PACK_MAX_H" -v sp="$PACK_SPEED" '
    BEGIN { n = split(idxs, a, ","); for (i = 1; i <= n; i++) want[a[i]] = 1 }
    FILENAME == lst { if (FNR in want) { id[FNR] = $1; cmd[FNR] = $2 }; next }
    FNR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    { est[$c["id"]] = $c["est_card_seconds"] }
    END { al = ""; pk = ""
      for (i = 1; i <= n; i++) { k = a[i]
        lone = (pack != 1) || (est[id[k]] / sp > maxs * 3600) || (cmd[k] ~ /steer_gen_raven|steer_gen_math/)
        if (lone) al = al "," k; else pk = pk "," k }
      print "ALONE " (al == "" ? "-" : substr(al, 2)); print "PACK " (pk == "" ? "-" : substr(pk, 2)) }' "$1" "$JL"
}

pack_report() {      # $1 PLAN-like lines "group<TAB>stage<TAB>idxs": print the lines that run alone, per group
  local g st ix al pk
  while IFS="$TAB" read -r g st ix; do
    [ "$ix" != "-" ] && [ -n "$ix" ] || continue
    al=$(cd "$LISTS" && pack_split "$g.$st.tsv" "$ix" | sed -n 's/^ALONE //p')
    pk=$(cd "$LISTS" && pack_split "$g.$st.tsv" "$ix" | sed -n 's/^PACK //p')
    if [ "$al" != "-" ]; then
      echo "  run alone (one line per GPU job): $g.$st lines $al$( [ "$pk" != "-" ] && echo "; packed two per GPU: lines $pk")"
    fi
  done <<< "$1"
}

submit_stage() {     # $1 group, $2 stage, $3 idxs, $4 dependency job ids (a:b) or "" -> echo job ids (a:b); globals R HOLD
  local g="$1" st="$2" ix="$3" dep="$4" al pk j ids="" pf n
  al=$(pack_split "$g.$st.tsv" "$ix" | sed -n 's/^ALONE //p')
  pk=$(pack_split "$g.$st.tsv" "$ix" | sed -n 's/^PACK //p')
  if [ "$al" != "-" ]; then
    # shellcheck disable=SC2086
    j=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" $R $HOLD --time="$TIME_LIMIT" \
          --output="$LOG_ROOT/ra20_${g}_%A_%a.out" ${dep:+--dependency=afterok:$dep} --array="$al%$CARDS" \
          "task_$g.sbatch" "$g.$st.tsv") || return 1
    echo "$j $g.$st.tsv $al" >> "$SUB"
    ids="$j"
  fi
  if [ "$pk" != "-" ]; then
    mkdir -p "$OUT_ROOT/_chain/pack"
    pf=$(mktemp "$OUT_ROOT/_chain/pack/$g.$st.XXXXXXXX") || return 1
    echo "$pk" | tr ',' '\n' | paste -d' ' - - | sed 's/ $//' > "$pf"
    n=$(grep -c . "$pf")
    # shellcheck disable=SC2086
    j=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" $R $HOLD --job-name="ra20_$g" --time="$TIME_LIMIT" \
          --output="$LOG_ROOT/ra20_${g}_pack_%A_%a.out" ${dep:+--dependency=afterok:$dep} --array="1-$n%$CARDS" \
          "task_pack.sbatch" "$g.$st.tsv" "$pf" "$g") || return 1
    echo "$j PACK $pf $g.$st.tsv" >> "$OUT_ROOT/_chain/packed.tsv"
    echo "$j $g.$st.tsv $pk" >> "$SUB.packed"
    ids="${ids:+$ids:}$j"
  fi
  echo "$ids"
}

queued_packed() {    # "<stage list> <line>" of every line of a packed job still in the queue ($1: squeue "%F %K" lines)
  [ -s "$OUT_ROOT/_chain/packed.tsv" ] || return 0
  awk 'NR == FNR { q[$1 " " $2] = 1; next }
       { while ((getline ln < $3) > 0) { k++; if (($1 " " k) in q) { m = split(ln, a, " "); for (i = 1; i <= m; i++) print $4 " " a[i] } }
         close($3); k = 0 }' <(echo "$1") "$OUT_ROOT/_chain/packed.tsv" 2> /dev/null
}
