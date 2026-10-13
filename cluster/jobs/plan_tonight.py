#!/usr/bin/env python3
"""Budget planner for tonight's cluster round (PI rule of 2026-10-12: everything must finish within 10 h of its start;
what does not fit is not run). jobs/submit_tonight.sh calls it; by hand, for a dry plan:

    python3 jobs/plan_tonight.py --lists jobs/nolocal --out-root $OUT_ROOT --free-nodes 5 --squeue q.txt --plan p.tsv

Inputs
- The job lists (<lists>/joblist.tsv, plus joblist_ext6c.tsv when present) and their stage lists (<group>.<stage>.tsv, array task i = line i). Planned units:
  EXT5 (g17_fillmap priority 1, g18_fillgen priority 2, g12_ifmv1 priority 3), EXT6b (g24, g25, g26 and the g28
  merges), EXT6c (every group named g<N>_e6c..., or EXT6C_GROUPS). A group absent from the list is skipped silently.
  Card-hours per line come from est_card_seconds (single-card rate, A100 factor 1.0).
- Done markers ($OUT_ROOT/_done/<id>): a done line is not planned. A pipeline (group, script, model, output folder)
  with a line still in the queue is left alone, like the per-unit submit scripts do.
- Dependencies of a line outside its pipeline: its depends_on plus the lines its cell's "needs" field matches (the
  jobs its pool comes from). A done dependency is met; a queued one (an EXT6 line in the queue) is an afterok edge on
  that queued chain; a dependency on a planned line is an afterok edge in the schedule (the stage chains behind it);
  any other dependency (the xllm build, an EXT6 line neither done nor queued) blocks the pipeline, which is then not
  planned.
- The queue (--squeue: squeue -h -r -u $USER -o "%F|%K|%t|%j|%N|%M") with $OUT_ROOT/_chain/submitted.tsv,
  submitted.tsv.packed and packed.tsv. EXT6 (g19 to g23) is never dropped and not planned: it is running. Its queued
  array tasks, and those of a planned unit's queued pipelines, are already-started chains inside the schedule: one GPU
  per array task (EXT6 unpacked), a running task for its est minus its elapsed time (%M), a pending task for its full
  est after the queued earlier stages of its pipeline (afterok). They go ahead of the plan's jobs and are never
  dropped. Merges are CPU jobs (no GPU). With --sinfo, a running task on a node that is not free is not on our GPUs.
- Nodes: --free-nodes N (FREE_NODES), else --sinfo (sinfo -h -N -n $NODELIST -o "%N %t") over --used (NODELIST
  minus the reserve, the last node, which is always excluded; minus bad nodes): nodes in state idle or mix. 8 GPUs each.

Packing and time (jobs/pack_lib.sh): two task lines of one pipeline stage share one GPU job, each at 0.6 x the
single-card rate; a line runs alone (one GPU, rate 1.0) when est / 0.6 exceeds PACK_MAX_H (9 h), when it is a Huginn or
McLeish generation line (steer_gen_raven, steer_gen_math), or with PACK=0. Every chain (pilot then full, and every
multi-stage pipeline) is submitted in one go with afterok: stage k+1 waits for every job of stage k, no second submit.
The wall time is estimated by list scheduling on the free GPUs, highest remaining chain first. Each unit's merges
(merge_ext6.py lines) are CPU jobs: no GPU, not in the schedule.

Drops: while the estimate exceeds BUDGET_H (9 h, 1 h under the 10 h job limit), whole cells are dropped in the step
order of CELLS. After each drop, a kept cell with a dependency on a dropped line is dropped with it ("dropped: depends
on <job>"). Then the refill pass: dropped cells come back in reverse drop order when the estimate stays at or under
the budget (a cell whose dependency is still dropped waits), so only what fits runs and no GPU idles for want of a
cell. Cells of step "never" are kept; when they alone exceed the budget the planner prints it and exits 2. A never
cell that depends on a droppable line, and a planned line that matches no cell, are errors (exit 3).
The final KEPT and DROPPED lists (each cell with its card-hours; for a drop its reason, the order step or the
dependency), the wall estimate and the longest chain are printed and written to <plan>.summary, which
submit_tonight.sh prints again before its yes and at the end of its dry run.

Output (--plan): a TSV the submit script reads. "unit" rows: unit, submit script, yes variable, kept lines, the file of
kept ids (PLAN_IDS), the fallback selection when the script has no PLAN_IDS option (EXT5: FILL_PRIORITIES=<the whole
priorities kept>; EXT6b, EXT6c: ALL when nothing of the unit was dropped; "-" for none), what the fallback leaves
out, and the units whose kept lines this unit depends on. "line" rows: unit, group, stage list, line index, id. "drop"
rows: unit, cell, card-hours, reason.
Sets (--set NAME, SET=NAME in submit_tonight.sh): a named cell table from SETS replaces CELLS; planned lines that
match no cell of the set are left out (not planned, not dropped), so the full plan stays the default.
Exit codes: 0 plan written; 2 the never-dropped set does not fit; 3 input problem.
"""
import argparse
import csv
import heapq
import os
import re
import sys

BUDGET_H = 9.0
GPUS_PER_NODE = 8
PACK_SPEED = 0.6
ALONE_CMD = r"steer_gen_raven|steer_gen_math|e6c_ra29_cell"   # these lines never share a card (pack_lib.sh; the
                                                               # RA-29 cells: submit_ext6c.sh stage *_alone)
FREE_STATES = ("idle", "mix", "mixed")
DONE_STATES = ("CD", "F", "CA", "TO", "NF", "PR", "OOM", "BF", "DL")

# (unit, group-name regex, submit script, its yes variable); submitted in this order (a unit's dependencies first)
UNITS = [
    ("EXT5", r"^(g17_fillmap|g18_fillgen|g12_ifmv1)$", "submit_fill.sh", "FILL_YES"),
    # g26_e6bra30 (RA-30 ASDiv 2.6B of EXT6b) stays absent: push 3 runs that cell fresh in EXT6c (g38, the waiver copy)
    ("EXT6b", r"^(g24_e6bra29|g25_e6bra31t|g28_e6bmerge)$", "submit_ext6b.sh", "EXT6B_YES"),
    ("EXT6c", r"^g\d+_e6c", "submit_ext6c.sh", "EXT6C_YES"),
]
FILL_PRIORITY = {"g17_fillmap": 1, "g18_fillgen": 2, "g12_ifmv1": 3}
EXT6_RUNNING = ("g19_e6ra30", "g20_e6ra29", "g21_e6ra31a", "g22_e6ra31b", "g23_e6merge")
MERGE_SCRIPT = r"^merge_ext6\.py$"
JOB_LISTS = ("joblist.tsv", "joblist_ext6c.tsv")

# The cell table (PI drop order and RA decisions of 2026-10-12). A matcher is {job-list column: regex}; the key "any"
# searches datasets, out and cmd together. A line belongs to the first cell it matches. "step" is "never" (kept) or
# the drop step; cells are dropped in table order. "needs" matches the lines the cell's pool comes from (planned lines
# of any unit): they are afterok predecessors, and when one of them is dropped the cell is dropped with it.
MCLEISH_ASDIV_POOL = {"group": r"^g18_fillgen$", "model": r"(?i)mcleish", "datasets": r"(?i)asdiv",
                      "stage": r"^s(1_prep|2_screen|3_select)$"}
CELLS = [
    # step, label, unit, matcher, needs
    ("never", "EXT5 priority 2 McLeish ASDiv pool (prep, screen, select)", "EXT5", MCLEISH_ASDIV_POOL, None),
    ("never", "EXT6c RA-30 McLeish ASDiv", "EXT6c", {"script": r"^ra30", "model": r"(?i)mcleish", "any": r"(?i)asdiv"},
     MCLEISH_ASDIV_POOL),
    ("never", "EXT6c RA-30 Thinking GSM8K pilot", "EXT6c",
     {"script": r"^ra30", "model": r"(?i)think", "datasets": r"(?i)gsm8k"}, None),
    ("never", "EXT6c RA-30 flags on EXT6 GSM8K", "EXT6c", {"script": r"^ra30_flags", "datasets": r"(?i)gsm8k"}, None),
    ("never", "EXT6c McLeish", "EXT6c", {"model": r"(?i)mcleish"}, None),
    ("never", "EXT6c Huginn", "EXT6c", {"model": r"(?i)huginn"}, None),
    ("1", "EXT5 priority 3", "EXT5", {"group": r"^g12_ifmv1$"}, None),
    ("2", "EXT5 priority 2 MAWPS", "EXT5", {"group": r"^g18_fillgen$", "datasets": r"(?i)mawps"}, None),
    ("2", "EXT5 priority 2 ASDiv", "EXT5", {"group": r"^g18_fillgen$", "datasets": r"(?i)asdiv"}, None),
    ("2", "EXT5 priority 2 SVAMP", "EXT5", {"group": r"^g18_fillgen$", "datasets": r"(?i)svamp"}, None),
    ("3", "EXT6b RA-31 Thinking Part A", "EXT6b", {"group": r"^g25_e6bra31t$"}, None),
    ("3", "EXT6c RA-29 Thinking", "EXT6c", {"script": r"^ra29", "model": r"(?i)think"}, None),
    ("3", "EXT6c RA-30 Thinking ASDiv pilot", "EXT6c", {"script": r"^ra30", "model": r"(?i)think", "datasets": r"(?i)asdiv"},
     None),
    ("4", "EXT6b RA-30 ASDiv 2.6B", "EXT6b", {"group": r"^g26_e6bra30$"}, None),
    ("4", "EXT6c RA-30 ASDiv 2.6B", "EXT6c", {"group": r"^g\d+_e6cra30d$"}, None),
    ("4", "EXT6c RA-30 flags on EXT6b ASDiv", "EXT6c", {"script": r"^ra30_flags", "datasets": r"(?i)asdiv"}, None),
    ("5", "EXT6b RA-29 HellaSwag 2.6B", "EXT6b",
     {"group": r"^g24_e6bra29$", "model": r"(?i)2[._]6b", "datasets": r"(?i)hellaswag"}, None),
    ("5", "EXT6b RA-29 HellaSwag 1.4B", "EXT6b",
     {"group": r"^g24_e6bra29$", "model": r"(?i)1[._]4b", "datasets": r"(?i)hellaswag"}, None),
    ("5", "EXT6b RA-29 MMLU 2.6B", "EXT6b", {"group": r"^g24_e6bra29$", "model": r"(?i)2[._]6b", "datasets": r"(?i)mmlu"}, None),
    ("5", "EXT6b RA-29 MMLU 1.4B", "EXT6b", {"group": r"^g24_e6bra29$", "model": r"(?i)1[._]4b", "datasets": r"(?i)mmlu"}, None),
    ("5", "EXT6c RA-29 Ouro 2.6B CSQA and ARC", "EXT6c", {"script": r"^ra29", "model": r"(?i)ouro_2_6b_base"}, None),
    ("6", "EXT6c 2.6B Part B", "EXT6c", {"script": r"^ra31_gen", "model": r"(?i)2[._]6b"}, None),
    ("7", "EXT6c 1.4B Part B", "EXT6c", {"script": r"^ra31_gen", "model": r"(?i)1[._]4b"}, None),
    ("8", "EXT5 priority 1", "EXT5", {"group": r"^g17_fillmap$"}, None),
]


def _ra29_cells(step, label, group, pairs):
    return [(step, "%s %s %s" % (label, m, ds), "EXT6c" if group.startswith("g35") else "EXT6b",
             {"group": r"^%s$" % group, "model": r"^%s$" % m, "datasets": r"^%s$" % ds}, None) for m, ds in pairs]


# Named cell tables (--set). Cells are dropped in table order, so the lowest priority comes first.
# tonight4 (PI, push 4): 1 RA-29 Thinking multiple choice (both sizes), 2 RA-29 MMLU and HellaSwag on Ouro 1.4B and
# 2.6B, 3 RA-29 McLeish and Huginn multiple choice; nothing else.
SETS = {
    "tonight4":
        _ra29_cells("3", "RA-29", "g35_e6cra29", [("huginn_0125", "csqa"), ("huginn_0125", "aqua"),
                                                   ("mcleish_llama32_r32", "csqa"), ("mcleish_llama32_r32", "arc"),
                                                   ("mcleish_llama32_r32", "aqua")])
        + _ra29_cells("2", "RA-29", "g24_e6bra29", [("ouro_2_6b_base", "hellaswag"), ("ouro_2_6b_base", "mmlu"),
                                                    ("ouro_1_4b_base", "hellaswag"), ("ouro_1_4b_base", "mmlu")])
        + _ra29_cells("1", "RA-29", "g35_e6cra29", [("ouro_2_6b_think", "csqa"), ("ouro_2_6b_think", "arc"),
                                                    ("ouro_2_6b_think", "aqua"), ("ouro_1_4b_think", "csqa"),
                                                    ("ouro_1_4b_think", "arc"), ("ouro_1_4b_think", "aqua")]),
}


class PlanError(Exception):
    pass


def matches(line, m):
    for k, rx in m.items():
        v = " ".join(line.get(x, "") or "" for x in ("datasets", "out", "cmd")) if k == "any" else (line.get(k, "") or "")
        if not re.search(rx, v):
            return False
    return True


def unit_of(group, ext6c_groups=None):
    for unit, rx, _s, _y in UNITS:
        if unit == "EXT6c" and ext6c_groups is not None:
            if group in ext6c_groups:
                return unit
            continue
        if re.search(rx, group):
            return unit
    return None


def cell_of(line):
    """The first CELLS row (step, label, unit, matcher, needs) a line matches, or None."""
    for c in CELLS:
        if line["unit"] == c[2] and matches(line, c[3]):
            return c
    return None


def load(lists, ext6c_groups=None):
    """Every line of the planned units and of EXT6 (dicts with the job-list columns plus unit (None for EXT6), list,
    idx, est, merge, deps) in job-list order."""
    path = os.path.join(lists, "joblist.tsv")
    if not os.path.isfile(path):
        raise PlanError("no job list %s" % path)
    rows, seen = [], set()
    for name in JOB_LISTS:        # joblist.tsv, then the EXT6c list that gen_ext6c.py writes beside it
        q = os.path.join(lists, name)
        if not os.path.isfile(q):
            continue
        with open(q, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
    lines = []
    for r in rows:
        u = unit_of(r["group"], ext6c_groups)
        if u is None and r["group"] not in EXT6_RUNNING:
            continue
        r = dict(r)
        r["unit"] = u
        r["est"] = float(r["est_card_seconds"] or 0)
        r["merge"] = bool(re.search(MERGE_SCRIPT, r["script"]))
        r["deps"] = [d for d in (r.get("depends_on") or "").split(",") if d]
        lines.append(r)
    groups = {r["group"] for r in lines}
    where = {}
    for name in sorted(os.listdir(lists)):
        m = re.match(r"^(g\d+_\w+)\.(s\d+_\w+)\.tsv$", name)
        if not (m and m.group(1) in groups):
            continue
        with open(os.path.join(lists, name), "rb") as f:
            raw = f.read().decode("utf-8").split("\n")
        for i, ln in enumerate(raw, 1):
            if ln:
                where[ln.split("\t", 1)[0]] = (name, i)
    for r in lines:
        if r["id"] not in where:
            raise PlanError("no stage-list line for %s (regenerate the lists with gen_jobs.py)" % r["id"])
        r["list"], r["idx"] = where[r["id"]]
    return lines


def alone(line, pack, pack_max_h):
    return pack != 1 or line["est"] / PACK_SPEED > pack_max_h * 3600 or bool(re.search(ALONE_CMD, line.get("cmd", "") or ""))


def to_s(t):
    """Seconds of a Slurm time (D-HH:MM:SS, HH:MM:SS, MM:SS); 0 when it does not parse (N/A, INVALID)."""
    try:
        d = 0
        if "-" in t:
            d, t = t.split("-", 1)
        s = 0
        for p in t.split(":"):
            s = s * 60 + int(p)
        return int(d) * 86400 + s
    except ValueError:
        return 0


def read_queue(path):
    """[(job id, array index, state, job name, node, elapsed seconds)] of
    squeue -o "%F|%K|%t|%j|%N|%M", finished tasks left out."""
    out = []
    if not path:
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            p = [x.strip() for x in ln.rstrip("\r\n").split("|")]
            if len(p) < 4 or not p[0]:
                continue
            p += [""] * (6 - len(p))
            out.append((p[0], p[1], p[2], p[3], p[4], to_s(p[5]) if p[5] else 0))
    return [q for q in out if q[2] not in DONE_STATES]


def read_chain(out_root, queue):
    """({job id: group} of every submission we know, {(stage list, line)} of lines whose array task is queued,
    {(job id, array index): (stage list, [lines], packed)} of every array task we can name)."""
    chain = os.path.join(out_root, "_chain")
    qset = {(q[0], q[1]) for q in queue}
    job_group, queued, rowmap = {}, set(), {}
    for name in ("submitted.tsv", "submitted.tsv.packed"):
        p = os.path.join(chain, name)
        if not os.path.isfile(p):
            continue
        for ln in open(p, encoding="utf-8", errors="replace"):
            f = ln.split()
            if len(f) < 2:
                continue
            job_group[f[0]] = f[1].split(".")[0]
            if name == "submitted.tsv" and len(f) > 2:
                for k in f[2].split(","):
                    if k.isdigit():
                        rowmap[(f[0], k)] = (f[1], [int(k)], False)
                        if (f[0], k) in qset:
                            queued.add((f[1], int(k)))
    p = os.path.join(chain, "packed.tsv")
    if os.path.isfile(p):
        for ln in open(p, encoding="utf-8", errors="replace"):
            f = ln.split()
            if len(f) < 4:
                continue
            job_group[f[0]] = f[3].split(".")[0]
            try:
                rows = open(f[2], encoding="utf-8").read().splitlines()
            except OSError:
                continue
            for k, row in enumerate(rows, 1):
                idx = [int(x) for x in row.split() if x.isdigit()]
                rowmap[(f[0], str(k))] = (f[3], idx, True)
                if (f[0], str(k)) in qset:
                    for x in idx:
                        queued.add((f[3], x))
    return job_group, queued, rowmap


def fixed_jobs(queue, rowmap, job_group, byloc, planned_groups, free_set, hold_s):
    """The queued array tasks of EXT6 and of the planned units as already-started, never-dropped chains: one GPU job
    per array task (EXT6 is unpacked: one line; a packed task: its two lines at 0.6), a running task for its est
    minus its elapsed time (at least 6 min), a pending task for its full est after the queued earlier stages of its
    pipeline (afterok). Merges are CPU jobs (no GPU). A running task on a node that is not free is not on our GPUs.
    A task we cannot name holds one GPU for hold_s. -> (jobs, counts)."""
    n = {k: 0 for k in ("ext6_run", "ext6_pend", "plan_run", "plan_pend", "merge", "offnode", "unknown", "other")}
    ent = []
    for jid, k, st, name, node, el in queue:
        m = rowmap.get((jid, k))
        g = m[0].split(".")[0] if m else (job_group.get(jid) or (name[5:] if name.startswith("ra20_") else ""))
        if g not in EXT6_RUNNING and g not in planned_groups:
            n["other"] += 1
            continue
        lines = [byloc[(m[0], i)] for i in m[1] if (m[0], i) in byloc] if m else []
        if (lines and all(r["merge"] for r in lines)) or (not lines and "merge" in g):
            n["merge"] += 1
            continue
        running = st != "PD"
        if running and free_set is not None and node not in free_set:
            n["offnode"] += 1
            continue
        n[("ext6" if g in EXT6_RUNNING else "plan") + ("_run" if running else "_pend")] += 1
        if not lines:
            n["unknown"] += 1
            ent.append({"dur": float(hold_s), "running": True, "pipe": ("?", jid, k), "stage": 0, "ids": [],
                        "label": "queued %s_%s (not in submitted.tsv)" % (jid, k)})
            continue
        r0 = lines[0]
        est = max(r["est"] for r in lines) / (PACK_SPEED if m[2] else 1.0)
        ent.append({"dur": max(est - el, 360.0) if running else est, "running": running, "ids": [r["id"] for r in lines],
                    "pipe": (r0["group"], r0["script"], r0["model"], r0["out"]),
                    "stage": int(re.match(r"s(\d+)", r0["stage"]).group(1)) if re.match(r"s(\d+)", r0["stage"]) else 0,
                    "label": "%s %s %s (queued)" % (r0["group"], r0["model"], r0["out"].rstrip("/").split("/")[-1])})
    jobs = []
    for e in ent:
        preds = [] if e["running"] else [i for i, o in enumerate(ent) if o["pipe"] == e["pipe"] and o["stage"] < e["stage"]]
        jobs.append({"dur": e["dur"], "preds": preds, "label": e["label"], "fixed": True, "running": e["running"],
                     "ids": e["ids"]})
    return jobs, n


def free_nodes(sinfo_path, used):
    states = {}
    for ln in open(sinfo_path, encoding="utf-8", errors="replace"):
        f = ln.split()
        if len(f) >= 2:
            states.setdefault(f[0], set()).add(f[1].lower())
    return [n for n in used if states.get(n) and all(s in FREE_STATES for s in states[n])]


class Plan(object):
    def __init__(self, lines, out_root, pack, pack_max_h, queued, others=()):
        self.lines = lines
        self.pipes, self.order = {}, []
        for r in lines:
            if r["merge"]:
                continue
            pl = (r["group"], r["script"], r["model"], r["out"])
            if pl not in self.pipes:
                self.pipes[pl] = {"stages": [], "lines": {}}
                self.order.append(pl)
            p = self.pipes[pl]
            if r["stage"] not in p["lines"]:
                p["stages"].append(r["stage"])
                p["lines"][r["stage"]] = []
            p["lines"][r["stage"]].append(r)
            r["pipe"] = pl
            r["alone"] = alone(r, pack, pack_max_h)
            c = cell_of(r)
            r["cell"], r["step"] = (c[1], c[0]) if c else (None, None)
        runl = [r for r in lines if not r["merge"]]
        done_dir = os.path.join(out_root, "_done")
        self.done = {r["id"] for r in lines if os.path.exists(os.path.join(done_dir, r["id"]))}
        qids = {r["id"] for r in list(runl) + list(others) if (r["list"], r["idx"]) in queued}
        # dependencies outside the pipeline: depends_on plus the lines the cell's needs field matches
        needs = {}
        for c in CELLS:
            if c[4]:
                needs[c[1]] = [r["id"] for r in runl if matches(r, c[4])]
        for r in runl:
            own = {x["id"] for s in self.pipes[r["pipe"]]["stages"] for x in self.pipes[r["pipe"]]["lines"][s]}
            xd = [d for d in r["deps"] + needs.get(r["cell"], []) if d not in own]
            xd = [d for i, d in enumerate(xd) if d not in xd[:i] and d not in self.done
                  and not os.path.exists(os.path.join(done_dir, d))]
            r["xdeps"] = [d for d in xd if d not in qids]
            r["qdeps"] = [d for d in xd if d in qids]          # queued: afterok on the queued chain
        self.state = {}
        for pl in self.order:
            rs = [x for s in self.pipes[pl]["stages"] for x in self.pipes[pl]["lines"][s]]
            self.state[pl] = ("queued", "") if any(x["id"] in qids for x in rs) else ("ok", "")
        planned = {r["id"]: r for r in runl}
        changed = True
        while changed:             # a dependency that is neither done, queued nor to run blocks the pipeline
            changed = False
            for pl in self.order:
                if self.state[pl][0] != "ok":
                    continue
                rs = [x for s in self.pipes[pl]["stages"] for x in self.pipes[pl]["lines"][s] if x["id"] not in self.done]
                ext = []
                for x in rs:
                    for d in x["xdeps"]:
                        if (d not in planned or self.state[planned[d]["pipe"]][0] == "blocked") and d not in ext:
                            ext.append(d)
                if ext:
                    self.state[pl] = ("blocked", " ".join(ext))
                    changed = True
        self.todo = [r for r in runl if r["id"] not in self.done and self.state[r["pipe"]][0] == "ok"]
        self.merges = [r for r in lines if r["merge"]]

    def jobs(self, kept):
        """GPU jobs of a kept id set: {dur, preds, ids, label}; preds are the previous stage of the pipeline and the
        jobs of the kept lines a line depends on outside its pipeline. Merges (CPU) are not jobs here."""
        jobs, of_id = [], {}

        def add(dur, preds, rs, label):
            jobs.append({"dur": dur, "preds": list(preds), "ids": [r["id"] for r in rs], "label": label, "lines": rs})
            for r in rs:
                of_id[r["id"]] = len(jobs) - 1
            return len(jobs) - 1

        for pl in self.order:
            prev = []
            label = "%s %s %s" % (pl[0], pl[2], pl[3].rstrip("/").split("/")[-1])
            for st in self.pipes[pl]["stages"]:
                rs = [r for r in self.pipes[pl]["lines"][st] if r["id"] in kept]
                if not rs:
                    continue
                cur = [add(r["est"], prev, [r], label) for r in rs if r["alone"]]
                pk = [r for r in rs if not r["alone"]]
                for i in range(0, len(pk), 2):
                    pair = pk[i:i + 2]
                    cur.append(add(max(r["est"] for r in pair) / PACK_SPEED, prev, pair, label))
                prev = cur
        for i, j in enumerate(jobs):
            for r in j["lines"]:
                for d in r["xdeps"]:
                    if d in of_id and of_id[d] != i and of_id[d] not in j["preds"]:
                        j["preds"].append(of_id[d])
        return jobs                    # the units' merges are CPU jobs: no GPU, not in the schedule


def schedule(jobs, gpus):
    """(plan wall seconds, plan longest chain seconds, its label, end of the fixed jobs): list scheduling on `gpus`
    GPUs. Fixed jobs (queued chains, "fixed": True) go first, running ones before pending ones, as Slurm serves the
    older submission first; then the plan's jobs, highest remaining chain first. The wall and the longest chain are
    those of the plan's jobs."""
    if not jobs:
        return 0.0, 0.0, "-", 0.0
    n = len(jobs)
    succ = [[] for _ in range(n)]
    npred = [len(set(j["preds"])) for j in jobs]
    for i, j in enumerate(jobs):
        for p in set(j["preds"]):
            succ[p].append(i)
    topo, left = [], list(npred)
    stack = [i for i in range(n) if left[i] == 0]
    while stack:
        i = stack.pop()
        topo.append(i)
        for s in succ[i]:
            left[s] -= 1
            if left[s] == 0:
                stack.append(s)
    if len(topo) != n:
        raise PlanError("the dependencies form a cycle")
    bl = [0.0] * n
    for i in reversed(topo):
        bl[i] = jobs[i]["dur"] + max([bl[s] for s in succ[i]] or [0.0])
    mine = [i for i in range(n) if not jobs[i].get("fixed")]
    if mine:
        top = max(mine, key=lambda i: (bl[i], -i))
        crit, crit_label = bl[top], jobs[top]["label"]
    else:
        crit, crit_label = 0.0, "-"
    if gpus <= 0:
        return (float("inf") if mine else 0.0), crit, crit_label, float("inf")

    def key(i):
        j = jobs[i]
        return (0 if j.get("fixed") and j.get("running") else 1 if j.get("fixed") else 2, -bl[i], i)

    ready = [key(i) for i in range(n) if npred[i] == 0]
    heapq.heapify(ready)
    running, t, free, wall, fend = [], 0.0, gpus, 0.0, 0.0
    while ready or running:
        while ready and free > 0:
            i = heapq.heappop(ready)[2]
            heapq.heappush(running, (t + jobs[i]["dur"], i))
            free -= 1
        t = running[0][0]
        while running and running[0][0] <= t:
            end, i = heapq.heappop(running)
            free += 1
            if jobs[i].get("fixed"):
                fend = max(fend, end)
            else:
                wall = max(wall, end)
            for s in succ[i]:
                npred[s] -= 1
                if npred[s] == 0:
                    heapq.heappush(ready, key(s))
    return wall, crit, crit_label, fend


def with_fixed(fixed, pj):
    """The fixed jobs followed by the plan's jobs, the plan's predecessor indices shifted; a plan job whose line
    depends on a queued line (qdeps) waits for the fixed job that runs it (afterok)."""
    k = len(fixed)
    at = {i: n for n, f in enumerate(fixed) for i in f.get("ids", [])}
    out = list(fixed)
    for j in pj:
        q = sorted({at[d] for r in j.get("lines", []) for d in r.get("qdeps", []) if d in at})
        out.append(dict(j, preds=[p + k for p in j["preds"]] + q))
    return out


def fmt_h(s):
    return "inf (no GPU free)" if s == float("inf") else "%.1f h" % (s / 3600.0)


def ch(rs):
    return sum(r["est"] for r in rs) / 3600.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lists", required=True, help="folder of joblist.tsv and the stage lists")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--squeue", default="", help='squeue -h -r -u $USER -o "%%F|%%K|%%t|%%j|%%N|%%M" output')
    ap.add_argument("--sinfo", default="", help='sinfo -h -N -n $NODELIST -o "%%N %%t" output')
    ap.add_argument("--used", default="", help="usable nodes (NODELIST minus the reserve and bad nodes), comma list")
    ap.add_argument("--reserved", default="", help="the reserved node(s), printed")
    ap.add_argument("--free-nodes", type=int, default=None, help="override: free nodes (the reserve already excluded)")
    ap.add_argument("--budget-h", type=float, default=BUDGET_H)
    ap.add_argument("--plan", default="", help="plan file to write (default $OUT_ROOT/_chain/plan_tonight.tsv)")
    ap.add_argument("--set", default="", help="a named cell table of SETS instead of CELLS (%s)" % ", ".join(sorted(SETS)))
    a = ap.parse_args(argv)
    global CELLS
    if a.set:
        if a.set not in SETS:
            print("plan_tonight: no set %s (sets: %s)" % (a.set, ", ".join(sorted(SETS))))
            return 3
        CELLS = SETS[a.set]
    pack = int(os.environ.get("PACK", "1") or 1)
    pack_max_h = float(os.environ.get("PACK_MAX_H", "9") or 9)
    e6c = os.environ.get("EXT6C_GROUPS", "").replace(",", " ").split()
    budget = a.budget_h * 3600

    try:
        lines = load(a.lists, e6c or None)
        queue = read_queue(a.squeue)
        job_group, queued, rowmap = read_chain(a.out_root, queue)
        plan = Plan([r for r in lines if r["unit"]], a.out_root, pack, pack_max_h, queued,
                    [r for r in lines if not r["unit"]])
    except PlanError as e:
        print("plan_tonight: %s" % e)
        return 3
    if a.set:
        out_set = [r for r in plan.todo if r["cell"] is None]
        plan.todo = [r for r in plan.todo if r["cell"] is not None]
        print("set %s: %d planned lines, %d lines outside the set left out" % (a.set, len(plan.todo), len(out_set)))
    bad = [r["id"] for r in plan.todo if r["cell"] is None]
    if bad:
        print("plan_tonight: %d planned line(s) match no cell of CELLS (add them to the table at the top of "
              "jobs/plan_tonight.py): %s" % (len(bad), " ".join(bad[:10])))
        return 3
    todo_by_id = {r["id"]: r for r in plan.todo}
    bad = ["%s needs %s (%s)" % (r["cell"], d, todo_by_id[d]["cell"]) for r in plan.todo if r["step"] == "never"
           for d in r["xdeps"] if d in todo_by_id and todo_by_id[d]["step"] != "never"]
    if bad:
        print("plan_tonight: a never-dropped cell depends on droppable lines (make them never-dropped in CELLS): %s"
              % "; ".join(sorted(set(bad))[:10]))
        return 3

    # nodes, GPUs and the queued chains (EXT6 and queued lines of the planned units) inside the schedule
    if a.free_nodes is not None:
        nfree, free_set, nsrc = a.free_nodes, None, "FREE_NODES"
    elif a.sinfo:
        used = [n for n in a.used.split(",") if n]
        fl = free_nodes(a.sinfo, used)
        nfree, free_set, nsrc = len(fl), set(fl), "idle or mix of %s" % (a.used or "none")
    else:
        print("plan_tonight: give --free-nodes (FREE_NODES) or --sinfo")
        return 3
    gpus = nfree * GPUS_PER_NODE
    planned_groups = {r["group"] for r in lines if r["unit"]}
    byloc = {(r["list"], r["idx"]): r for r in lines}
    fixed, nq = fixed_jobs(queue, rowmap, job_group, byloc, planned_groups, free_set, budget)

    def est(kept):
        return schedule(with_fixed(fixed, plan.jobs(kept)), gpus)

    print("tonight's plan from %s: budget %.1f h of wall time (every job --time=10:00:00)" % (a.lists, a.budget_h))
    print("nodes: %d free (%s), reserved and excluded: %s; %d GPUs, %d process slots (two per GPU, each at %.1f x the "
          "single-card rate)" % (nfree, nsrc, a.reserved or "-", gpus, 2 * gpus, PACK_SPEED))
    print("queued chains in the schedule, never dropped, one GPU per array task, ahead of the plan: EXT6 %d running, %d "
          "pending; planned units %d running, %d pending; merges (CPU, no GPU): %d; running on nodes that are not free: "
          "%d; other jobs of ours, not counted: %d" % (nq["ext6_run"], nq["ext6_pend"], nq["plan_run"], nq["plan_pend"],
                                                       nq["merge"], nq["offnode"], nq["other"]))
    if nq["unknown"]:
        print("  %d queued task(s) not in submitted.tsv: each holds one GPU for the whole %.1f h" % (nq["unknown"], a.budget_h))
    if fixed:
        print("  the queued chains alone end at about %s" % fmt_h(schedule(fixed, gpus)[3]))
    for pl in plan.order:
        st, why = plan.state[pl]
        name = "%s %s %s" % (pl[0], pl[2], pl[3].rstrip("/").split("/")[-1])
        if st == "queued":
            print("  %s: in the queue, left alone (in the schedule above)" % name)
        elif st == "blocked":
            print("  %s: NOT PLANNED: needs the done marker of %s" % (name, why))

    # cells: (step, label, unit, todo lines) in table order
    cells = [(c[0], c[1], c[2], [r for r in plan.todo if r["cell"] == c[1]]) for c in CELLS]
    cells = [c for c in cells if c[3]]
    by_label = {c[1]: c for c in cells}
    print("cells to run (lines, card-hours at the single-card rate):")
    for step, label, unit, rs in cells:
        print("  %s %s: %d lines, %.1f card-h" % ("never" if step == "never" else "(%s)" % step, label, len(rs), ch(rs)))

    never = {r["id"] for c in cells if c[0] == "never" for r in c[3]}
    w_never = est(never)[0]
    if never and w_never > budget:
        print("the never-dropped cells alone need %s on %d GPUs next to the queued chains, over the %.1f h budget: "
              "nothing is planned" % (fmt_h(w_never), gpus, a.budget_h))
        return 2

    kept = {r["id"] for r in plan.todo}
    dropped = []               # (step, label, unit, lines, reason), in drop order

    def cascade():
        """Drop every kept cell with a dependency on a line that is not kept; -> printed lines."""
        msgs, again = [], True
        while again:
            again = False
            for r in plan.todo:
                if r["id"] not in kept:
                    continue
                d = next((d for d in r["xdeps"] if d in todo_by_id and d not in kept), None)
                if d is None:
                    continue
                c = by_label[r["cell"]]
                kept.difference_update(x["id"] for x in c[3])
                dropped.append((c[0], c[1], c[2], c[3], "depends on %s" % d))
                msgs.append("drop %s: %.1f card-h, dropped: depends on %s" % (c[1], ch(c[3]), d))
                again = True
        return msgs

    wall = est(kept)[0]
    print("everything: %.1f card-h, estimated wall %s" % (ch(plan.todo), fmt_h(wall)))
    for step, label, unit, rs in cells:
        if wall <= budget:
            break
        ids = {r["id"] for r in rs} & kept
        if step == "never" or not ids:
            continue
        kept -= ids
        dropped.append((step, label, unit, rs, "order step %s" % step))
        msgs = cascade()
        wall = est(kept)[0]
        print("drop (%s) %s: %.1f card-h; estimated wall now %s" % (step, label, ch(rs), fmt_h(wall)))
        for m in msgs:
            print("  " + m)
    if wall > budget:          # cannot happen: the never-dropped set fits
        print("still over the budget after every drop")
        return 2

    # refill: dropped cells back in reverse drop order while the estimate stays within the budget; a cell whose
    # dependency is still dropped waits; passes repeat until nothing more fits
    again = True
    while again:
        again = False
        for c in list(reversed(dropped)):
            ids = {r["id"] for r in c[3]}
            trial = kept | ids
            if any(x in todo_by_id and x not in trial for r in c[3] for x in r["xdeps"]):
                continue
            w = est(trial)[0]
            if w <= budget:
                kept = trial
                dropped.remove(c)
                print("refill (%s) %s: %.1f card-h; estimated wall now %s" % (c[0], c[1], ch(c[3]), fmt_h(w)))
                again = True

    keptl = [r for r in plan.todo if r["id"] in kept]
    jobs = plan.jobs(kept)
    wall, crit, crit_label, fend = est(kept)
    dl = {d[1] for d in dropped}
    keptc = [c for c in cells if c[1] not in dl]
    summary = ["KEPT (%d cells, %.1f card-h):" % (len(keptc), ch(keptl))]
    summary += ["  %s %s: %.1f card-h" % ("never" if c[0] == "never" else "(%s)" % c[0], c[1], ch(c[3]))
                for c in keptc] or ["  nothing"]
    summary.append("DROPPED (%d cells, %.1f card-h):" % (len(dropped), sum(ch(d[3]) for d in dropped)))
    summary += ["  (%s) %s: %.1f card-h, reason: %s" % (d[0], d[1], ch(d[3]), d[4]) for d in dropped] or ["  nothing"]
    nmerge = len([m for m in plan.merges if m["unit"] in {r["unit"] for r in keptl}])
    summary.append("estimated wall %s (budget %.1f h), longest chain %s (%s); %d GPU jobs plus %d merges (CPU); "
                   "queued chains end at about %s" % (fmt_h(wall), a.budget_h, fmt_h(crit), crit_label, len(jobs),
                                                      nmerge, fmt_h(fend) if fixed else "0.0 h"))
    print("\n".join(summary))

    # the plan file, and the summary the submit script prints again before its yes
    path = a.plan or os.path.join(a.out_root, "_chain", "plan_tonight.tsv")
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)
    out = ["# plan_tonight\tfree_nodes=%d\tgpus=%d\twall_h=%.2f\tcard_h=%.1f" % (nfree, gpus, wall / 3600.0, ch(keptl))]
    for unit, _rx, script, yes in UNITS:
        ul = [r for r in keptl if r["unit"] == unit]
        ids_file = "%s.%s.ids" % (path, unit)
        with open(ids_file, "w", newline="\n") as f:
            f.write("".join(r["id"] + "\n" for r in ul))
        cut = [c for c in dropped if c[2] == unit]
        if not ul:
            fb, left = "-", "-"
        elif unit == "EXT5":
            pri = sorted({FILL_PRIORITY[r["group"]] for r in ul})
            part = sorted({FILL_PRIORITY[r["group"]] for c in cut for r in c[3]})
            whole = [p for p in pri if p not in part]
            fb = "FILL_PRIORITIES=%s" % ",".join(map(str, whole)) if whole else "-"
            left = ",".join("priority %d (partly dropped)" % p for p in pri if p in part) or "-"
        else:
            fb, left = ("ALL", "-") if not cut else ("-", "the whole unit (partly dropped)")
            if a.set:
                fb, left = "-", "the whole unit (set %s)" % a.set
        needs = sorted({todo_by_id[x]["unit"] for r in ul for x in r["xdeps"] if x in kept and todo_by_id[x]["unit"] != unit})
        out.append("unit\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s" % (unit, script, yes, len(ul), ids_file, fb, left,
                                                             ",".join(needs) or "-"))
    for r in keptl:
        out.append("line\t%s\t%s\t%s\t%d\t%s" % (r["unit"], r["group"], r["list"], r["idx"], r["id"]))
    for step, label, unit, rs, why in dropped:
        out.append("drop\t%s\t%s\t%.2f\t%s" % (unit, label, ch(rs), why))
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    with open(path + ".summary", "w", newline="\n") as f:
        f.write("\n".join(summary) + "\n")
    print("plan: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
