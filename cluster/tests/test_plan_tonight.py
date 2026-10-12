"""Tonight's budget planner (jobs/plan_tonight.py) and its submit script (jobs/submit_tonight.sh), with fake Slurm.

    cd loopedcot; python -B -m pytest cluster/tests/test_plan_tonight.py -q -p no:cacheprovider

1. List scheduling: the longest remaining chain goes first; a chain's stages run in order; queued chains (fixed jobs)
   go ahead of the plan and the wall is the plan's.
2. FREE_NODES=5 and 2 over the real job list (jobs/nolocal): the estimate fits 9 h; the drops follow the step order
   of CELLS and stop at the first fit; the refill brings dropped cells back in reverse order while they fit; kept plus
   dropped is every line to run; the final KEPT and DROPPED lists print with card-hours and reasons.
3. EXT6 (g19 to g23) array tasks in the queue are already-started chains in the schedule: a running task for its est
   minus its elapsed time, a pending one after its afterok, one GPU each; merges hold no GPU; a running task on a node
   that is not free is not on our GPUs; a task missing from submitted.tsv holds a GPU for the whole budget; jobs of
   other groups and finished tasks are not counted.
4. Never-dropped cells: EXT6c McLeish and Huginn, and the EXT5 priority-2 McLeish ASDiv pool (prep, screen, select),
   are kept when everything else goes; when they alone do not fit, the planner exits 2 and writes no plan, and
   submit_tonight.sh submits nothing.
5. Dependencies: the EXT6c RA-30 McLeish ASDiv cell chains (afterok) behind the protected pool; a kept cell whose
   pool comes from a dropped line is dropped with it ("dropped: depends on <job>"); a never cell that needs a
   droppable line is an error; pilot-then-full chains run in order and are kept or dropped whole.
6. Groups absent from the job list are skipped silently; a planned line that matches no cell is an error (exit 3);
   done lines are not planned; a pipeline with a queued line is left alone and its tasks are queued chains.
7. submit_tonight.sh end to end: the reserve (the last NODELIST node) is in --exclude of every sbatch call and no used
   node is; --time=10:00:00 on every call; every full stage goes out in the same run with afterok on its pilot;
   --dry-run submits nothing; the final KEPT and DROPPED lists print after the unit dry runs and before the yes;
   submit_fill.sh gets FILL_PRIORITIES of the whole priorities kept when it has no PLAN_IDS.
The tests read the tree; they write only to temporary folders.
"""
import csv
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.dirname(HERE)
LC = os.path.dirname(CL)
TOP = os.path.dirname(LC)
JOBS = os.path.join(CL, "jobs")
NOLOCAL = os.path.join(JOBS, "nolocal")
sys.dont_write_bytecode = True
sys.path.insert(0, JOBS)
import plan_tonight as P  # noqa: E402

PARTITION, ACCOUNT = "gpuq", "acct1"
SIX = ["n01", "n02", "n03", "n04", "n05", "n06"]
COLS = ["id", "group", "stage", "script", "model", "datasets", "shard", "out", "depends_on", "est_card_seconds",
        "rate_source", "cmd"]
DROP_LABELS = [c[1] for c in P.CELLS if c[0] != "never"]
POOL = "EXT5 priority 2 McLeish ASDiv pool (prep, screen, select)"


def _bash():
    """bash for the shell-script tests; on Windows only Git's bash (the WSL launcher cannot read these paths)."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    d = os.path.dirname(git) if git else ""
    for _ in range(3):
        for c in (os.path.join(d, "bin", "bash.exe"), os.path.join(d, "usr", "bin", "bash.exe")):
            if d and os.path.exists(c):
                return c
        d = os.path.dirname(d)
    return None


needs_bash = pytest.mark.skipif(_bash() is None, reason="no bash")


# ------------------------------------------------------------------ helpers
def _out_root(tmp, done=("g0_xllm.s1_build.001",)):
    out = tmp / "out"
    (out / "_done").mkdir(parents=True, exist_ok=True)
    (out / "_chain").mkdir(parents=True, exist_ok=True)
    for d in done:
        (out / "_done" / d).write_text("")
    return out


def _plan(tmp, capsys, lists=NOLOCAL, free=None, queue="", sinfo=None, used=",".join(SIX[:5]), out=None):
    """Run plan_tonight.main in process: -> (exit code, printed text, plan rows by kind)."""
    out = out or _out_root(tmp)
    q = tmp / "q.txt"
    q.write_text(queue)
    plan = tmp / "plan.tsv"
    if plan.exists():
        plan.unlink()
    args = ["--lists", str(lists), "--out-root", str(out), "--squeue", str(q), "--plan", str(plan),
            "--used", used, "--reserved", "n06"]
    if free is not None:
        args += ["--free-nodes", str(free)]
    if sinfo is not None:
        (tmp / "sinfo.txt").write_text(sinfo)
        args += ["--sinfo", str(tmp / "sinfo.txt")]
    old = dict(os.environ)
    for k in ("PACK", "PACK_MAX_H", "EXT6C_GROUPS"):
        os.environ.pop(k, None)
    try:
        rc = P.main(args)
    finally:
        os.environ.clear()
        os.environ.update(old)
    text = capsys.readouterr().out
    rows = {}
    if plan.exists():
        for ln in open(str(plan)).read().splitlines():
            if not ln.startswith("#"):
                f = ln.split("\t")
                rows.setdefault(f[0], []).append(f[1:])
    return rc, text, rows


def _row(i, group, stage, script, model, ds, est_h, deps="", out=None, cmd=None):
    return {"id": "%s.%s.%03d" % (group, stage, i), "group": group, "stage": stage, "script": script, "model": model,
            "datasets": ds, "shard": "", "out": out or "$OUT_ROOT/X/%s/%s_%s" % (group, model, ds),
            "depends_on": deps, "est_card_seconds": str(int(est_h * 3600)), "rate_source": "test",
            "cmd": cmd or "python -u %s --model %s" % (script, model)}


def _lists(tmp, rows, name="lists", ext6c_rows=()):
    """A job list (and joblist_ext6c.tsv for ext6c_rows) and the stage lists (array task i = line i of
    <group>.<stage>.tsv)."""
    d = tmp / name
    d.mkdir(exist_ok=True)
    for fname, rs in (("joblist.tsv", rows), ("joblist_ext6c.tsv", ext6c_rows)):
        if fname != "joblist.tsv" and not rs:
            continue
        with open(str(d / fname), "w", newline="\n") as f:
            w = csv.DictWriter(f, COLS, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
            w.writeheader()
            for r in rs:
                w.writerow(r)
    rows = list(rows) + list(ext6c_rows)
    per = {}
    for r in rows:
        per.setdefault("%s.%s.tsv" % (r["group"], r["stage"]), []).append("%s\t%s" % (r["id"], r["cmd"]))
    for k, v in per.items():
        (d / k).write_text("\n".join(v) + "\n")
    return d


def _todo(lists, out_root):
    return P.Plan([r for r in P.load(str(lists)) if r["unit"]], str(out_root), 1, 9.0, set())


def _todo_cells(lists, out_root):
    """{cell label: ids} of the lines to run, as the planner classifies them."""
    cells = {}
    for r in _todo(lists, out_root).todo:
        cells.setdefault(r["cell"], set()).add(r["id"])
    return cells


def _check_drops(text, rows, cells):
    """The budget drops are the first droppable cells of CELLS that have lines, in order; no drop after the estimate
    fit; every refill was dropped before and leaves the estimate within 9 h; the final estimate fits 9 h; the DROPPED
    list is the drops minus the refills; kept plus dropped is every line to run; no never cell is dropped."""
    probs = []
    order = [lab for lab in DROP_LABELS if lab in cells]
    drops = re.findall(r"^drop \(\w+\) (.+?): [\d.]+ card-h; estimated wall now ([\d.]+|inf)", text, re.M)
    cascaded = re.findall(r"^  drop (.+?): [\d.]+ card-h, dropped: depends on \S+$", text, re.M)
    refills = re.findall(r"^refill \(\w+\) (.+?): [\d.]+ card-h; estimated wall now ([\d.]+) h$", text, re.M)
    budget = [d[0] for d in drops]
    if [x for x in order if x not in cascaded][:len(budget)] != budget:
        probs.append("drops %s are not the head of the drop order %s" % (budget, order))
    walls = [float(w) if w != "inf" else 1e9 for w in re.findall(r"^everything: [\d.]+ card-h, estimated wall ([\d.]+|inf)", text, re.M)]
    walls += [float(d[1]) if d[1] != "inf" else 1e9 for d in drops]
    if any(w <= 9.0 for w in walls[:-1]):
        probs.append("a drop happened after the estimate already fit: %s" % walls)
    if any(r[0] not in budget + cascaded or float(r[1]) > 9.0 for r in refills):
        probs.append("a refill was not dropped before or does not fit: %s" % refills)
    final = re.findall(r"^estimated wall ([\d.]+) h \(budget 9\.0 h\), longest chain ([\d.]+) h", text, re.M)
    if not final or float(final[-1][0]) > 9.0 + 1e-9 or float(final[-1][1]) > float(final[-1][0]) + 1e-9:
        probs.append("the final estimate %s does not fit 9 h (or the chain exceeds the wall)" % final)
    gone = [x for x in budget + cascaded if x not in [r[0] for r in refills]]
    listed = re.findall(r"^  \(\w+\) (.+?): [\d.]+ card-h, reason: ", text, re.M)
    if sorted(listed) != sorted(gone):
        probs.append("the DROPPED list %s is not the drops minus the refills %s" % (listed, gone))
    kept = {r[4] for r in rows.get("line", [])}
    dropped = set().union(*[cells[d] for d in gone]) if gone else set()
    every = set().union(*cells.values()) if cells else set()
    if kept & dropped or kept | dropped != every:
        probs.append("kept %d + dropped %d != %d lines to run" % (len(kept), len(dropped), len(every)))
    if any(c[0] == "never" and c[1] in gone for c in P.CELLS):
        probs.append("a never-dropped cell was dropped: %s" % gone)
    return probs


# ------------------------------------------------------------------ 1. the scheduler
def test_schedule_longest_chain_first():
    """Two GPUs; chain A (5 h) then B (3 h), and C, D (4 h each). Longest chain first: A and C at 0, D at 4, B at 5;
    wall 8 h. Shortest first would give 4 + 5 + 3 = 12 h on the chain."""
    h = 3600.0
    jobs = [{"dur": 4 * h, "preds": [], "label": "C"}, {"dur": 4 * h, "preds": [], "label": "D"},
            {"dur": 5 * h, "preds": [], "label": "A"}, {"dur": 3 * h, "preds": [2], "label": "A"}]
    wall, crit, label, fend = P.schedule(jobs, 2)
    assert (wall, crit, label, fend) == (8 * h, 8 * h, "A", 0.0)
    assert P.schedule(jobs, 1)[0] == 16 * h
    assert P.schedule(jobs, 0)[0] == float("inf") and P.schedule([], 0)[0] == 0.0
    back = [{"dur": 3 * h, "preds": [1], "label": "B"}, {"dur": 5 * h, "preds": [], "label": "A"}]   # pred after
    assert P.schedule(back, 4)[:2] == (8 * h, 8 * h)


def test_queued_chains_go_first_and_the_wall_is_the_plans():
    """A running queued task (5 h left) and its pending full (2 h, afterok) are served before a plan job of 3 h. One
    GPU: queued 0-5 and 5-7, plan 7-10 (wall 10 h). Two GPUs: the plan job runs 0-3 beside the queued chain."""
    h = 3600.0
    fixed = [{"dur": 5 * h, "preds": [], "label": "e6 pilot", "fixed": True, "running": True},
             {"dur": 2 * h, "preds": [0], "label": "e6 full", "fixed": True, "running": False}]
    plan = [{"dur": 3 * h, "preds": [], "label": "mine"}]
    assert P.schedule(P.with_fixed(fixed, plan), 1) == (10 * h, 3 * h, "mine", 7 * h)
    assert P.schedule(P.with_fixed(fixed, plan), 2) == (3 * h, 3 * h, "mine", 7 * h)
    assert P.schedule(fixed, 2) == (0.0, 0.0, "-", 7 * h)


def test_cell_table_order_and_the_pool_dependency_field():
    """The PI order of 2026-10-12 and the RA decision: steps never, 1..8 in table order; the RA-30 McLeish ASDiv cell
    carries the pool as its needs field; the pool is never dropped."""
    steps = [c[0] for c in P.CELLS]
    assert steps == sorted(steps, key=lambda s: -1 if s == "never" else int(s)), steps
    assert DROP_LABELS == [
        "EXT5 priority 3", "EXT5 priority 2 MAWPS", "EXT5 priority 2 ASDiv", "EXT5 priority 2 SVAMP",
        "EXT6b RA-31 Thinking Part A", "EXT6c RA-29 Thinking", "EXT6c RA-30 Thinking", "EXT6b RA-30 ASDiv 2.6B",
        "EXT6c RA-30 flags on EXT6b ASDiv", "EXT6b RA-29 HellaSwag 2.6B", "EXT6b RA-29 HellaSwag 1.4B", "EXT6b RA-29 MMLU 2.6B", "EXT6b RA-29 MMLU 1.4B",
        "EXT6c 2.6B Part B", "EXT6c 1.4B Part B", "EXT5 priority 1"]
    cell = {c[1]: c for c in P.CELLS}
    assert cell["EXT6c RA-30 McLeish ASDiv"][4] is P.MCLEISH_ASDIV_POOL and cell[POOL][0] == "never"
    assert {"EXT6c McLeish", "EXT6c Huginn"} <= {c[1] for c in P.CELLS if c[0] == "never"}


# ------------------------------------------------------------------ 2. FREE_NODES 5 and 2 on the real job list
@pytest.mark.parametrize("free", [5, 2])
def test_free_nodes_keep_and_drop_sequence(tmp_path, capsys, free):
    out = _out_root(tmp_path)
    rc, text, rows = _plan(tmp_path, capsys, free=free, out=out)
    assert rc == 0, text
    assert "%d GPUs, %d process slots" % (free * 8, free * 16) in text, text
    cells = _todo_cells(NOLOCAL, out)
    probs = _check_drops(text, rows, cells)
    assert not probs, "\n".join(probs) + "\n" + text
    kept = {r[4] for r in rows["line"]}
    assert cells[POOL] <= kept and len(cells[POOL]) == 4, sorted(cells[POOL])
    drops = re.findall(r"^drop \((\w+)\) (.+?):", text, re.M)
    if free == 2:     # 16 GPUs: at least P3, P2 MAWPS and ASDiv go (the ASDiv pool stays)
        assert drops[:3] == [("1", "EXT5 priority 3"), ("2", "EXT5 priority 2 MAWPS"), ("2", "EXT5 priority 2 ASDiv")], text
        if not os.path.exists(os.path.join(NOLOCAL, "joblist_ext6c.tsv")):     # the stage-2 list: P3 comes back
            assert re.findall(r"^refill \((\w+)\) (.+?):", text, re.M) == [("1", "EXT5 priority 3")], text
    elif not os.path.exists(os.path.join(NOLOCAL, "joblist_ext6c.tsv")):
        assert "DROPPED (0 cells, 0.0 card-h):" in text, text
    for u in rows["unit"]:
        n = int(u[3])
        assert n == len([r for r in rows["line"] if r[0] == u[0]]) == len(open(u[4]).read().split()), u


# ------------------------------------------------------------------ 3. EXT6 occupancy
def _ext6_rows():
    """EXT6 lines (g19 pilot 1 h then full 2 h, the g23 merge) and one planned line (g17, 1 h)."""
    o = "$OUT_ROOT/E6/ra30"
    return [_row(1, "g19_e6ra30", "s1_pilot", "ra30.py", "ouro_2_6b_base", "ra30", 1.0, out=o),
            _row(1, "g19_e6ra30", "s2_full", "ra30.py", "ouro_2_6b_base", "ra30", 2.0, out=o, deps="g19_e6ra30.s1_pilot.001"),
            _row(1, "g23_e6merge", "s1_merge", "merge_ext6.py", "all", "ra30", 0.1, out="$OUT_ROOT/merged/ra30",
                 deps="g19_e6ra30.s1_pilot.001,g19_e6ra30.s2_full.001"),
            _row(1, "g17_fillmap", "s1_scan", "k17_public.py", "ouro26_think", "D1", 1.0, out="$OUT_ROOT/B/a")]


QUEUE = "\n".join([
    "501|1|R|ra20_g19_e6ra30|n01|30:00",      # EXT6 pilot running for 30 min: 0.5 h left
    "502|1|PD|ra20_g19_e6ra30||0:00",         # EXT6 full pending (afterok): 2 h after the pilot
    "503|1|PD|ra20_g23_e6merge||0:00",        # merge: CPU, no GPU
    "504|1|R|ra20_g22_e6ra31b|n03|1:00:00",   # running on a node that is not free: not on our GPUs (with sinfo)
    "505|1|PD|ra20_g20_e6ra29||0:00",         # EXT6 not in submitted.tsv: holds one GPU for the whole budget
    "506|1|PD|ra20_g9_merge||0:00",           # not EXT6, not a planned unit: not counted
    "507|1|CD|ra20_g21_e6ra31a|n01|2:00:00",  # finished: not counted
]) + "\n"
SINFO = "n01 idle\nn02 mix\nn03 alloc\nn04 drain\nn05 idle\nn06 idle\nn02 mix\n"
SUBMITTED = "501 g19_e6ra30.s1_pilot.tsv 1\n502 g19_e6ra30.s2_full.tsv 1\n503 g23_e6merge.s1_merge.tsv 1\n"


def test_ext6_chains_in_the_schedule(tmp_path, capsys):
    lists = _lists(tmp_path, _ext6_rows())
    out = _out_root(tmp_path)
    (out / "_chain" / "submitted.tsv").write_text(SUBMITTED)
    rc, text, rows = _plan(tmp_path, capsys, lists=lists, queue=QUEUE, sinfo=SINFO, out=out)
    assert rc == 0, text
    # free: n01, n02, n05 (n06 is the reserve, n03 alloc, n04 drain): 24 GPUs, nothing subtracted up front
    assert "nodes: 3 free" in text and "24 GPUs, 48 process slots" in text, text
    assert ("EXT6 1 running, 2 pending; planned units 0 running, 0 pending; merges (CPU, no GPU): 1; running on nodes "
            "that are not free: 1; other jobs of ours, not counted: 1") in text, text
    assert "1 queued task(s) not in submitted.tsv: each holds one GPU for the whole 9.0 h" in text, text
    assert "the queued chains alone end at about 9.0 h" in text and [r[4] for r in rows["line"]] == ["g17_fillmap.s1_scan.001"]
    # the chains: pilot 0.5 h left, full 2 h after it (afterok), the unknown task 9 h; no merge job
    lines = P.load(str(lists))
    q = P.read_queue(str(tmp_path / "q.txt"))
    jg, _qd, rowmap = P.read_chain(str(out), q)
    byloc = {(r["list"], r["idx"]): r for r in lines}
    fixed, n = P.fixed_jobs(q, rowmap, jg, byloc, {"g17_fillmap"}, {"n01", "n02", "n05"}, 9 * 3600)
    assert sorted(round(j["dur"] / 3600, 2) for j in fixed) == [0.5, 2.0, 9.0], fixed
    full = [j for j in fixed if abs(j["dur"] - 7200) < 1][0]
    assert [fixed[i]["dur"] for i in full["preds"]] == [1800.0] and not full["running"], fixed
    known = [j for j in fixed if "not in submitted" not in j["label"]]
    assert P.schedule([dict(j, preds=[known.index(fixed[i]) for i in j["preds"]]) for j in known], 1)[3] == 2.5 * 3600
    # FREE_NODES: no node states, so the task on n03 counts as ours too
    rc, text, _ = _plan(tmp_path, capsys, lists=lists, queue=QUEUE, free=3, out=out)
    assert rc == 0 and "EXT6 2 running, 2 pending" in text and "running on nodes that are not free: 0" in text, text


# ------------------------------------------------------------------ 4, 5. never-dropped cells and dependencies
def _synthetic(big_h=8.0, mc_full_h=2.0):
    """Every cell kind: 30 priority-1 lines of 8 h (alone: 32 h on 8 GPUs), the McLeish ASDiv pool and its rows, a
    MAWPS row, EXT6b cells, and EXT6c: RA-30 McLeish ASDiv (pilot, full; needs the pool), RA-29 McLeish and Huginn,
    RA-30 Thinking (pilot, full), Part B 2.6B and 1.4B."""
    R = []
    for i in range(1, 31):
        R.append(_row(i, "g17_fillmap", "s1_run", "k15_full.py", "ifm_d28", "D%d" % i, big_h, out="$OUT_ROOT/B/m%d" % i))
    po = "$OUT_ROOT/A/steer_gen_math/mcleish_asdiv"
    R.append(_row(1, "g18_fillgen", "s1_prep", "steer_gen_math.py", "mcleish_llama32_r32", "ASDIV_gen", 0.1, out=po))
    R.append(_row(1, "g18_fillgen", "s2_screen", "steer_gen_math.py", "mcleish_llama32_r32", "ASDIV_gen", 1.0, out=po,
                  deps="g18_fillgen.s1_prep.001"))
    R.append(_row(1, "g18_fillgen", "s3_select", "steer_gen_math.py", "mcleish_llama32_r32", "ASDIV_gen", 0.1, out=po,
                  deps="g18_fillgen.s2_screen.001"))
    R.append(_row(1, "g18_fillgen", "s4_row", "steer_gen_math.py", "mcleish_llama32_r32", "ASDIV_gen", 1.0, out=po,
                  deps="g18_fillgen.s3_select.001"))
    R.append(_row(2, "g18_fillgen", "s1_prep", "steer_gen_math.py", "huginn_0125", "MAWPS_gen", 0.5))
    R.append(_row(1, "g24_e6bra29", "s1_cell", "ra29.py", "ouro_2_6b_base", "hellaswag", 4.0))
    R.append(_row(1, "g25_e6bra31t", "s1_pilot", "ra31.py", "ouro_1_4b_think", "ra31", 1.0))
    mo = "$OUT_ROOT/E6/ra30/mcleish_asdiv"
    R.append(_row(1, "g29_e6cra30", "s1_pilot", "ra30.py", "mcleish_llama32_r32", "asdiv", 0.5, out=mo))
    R.append(_row(1, "g29_e6cra30", "s2_full", "ra30.py", "mcleish_llama32_r32", "asdiv", mc_full_h, out=mo,
                  deps="g29_e6cra30.s1_pilot.001"))
    to = "$OUT_ROOT/E6/ra30/think26"
    R.append(_row(2, "g29_e6cra30", "s1_pilot", "ra30.py", "ouro_2_6b_think", "gsm8k", 1.0, out=to))
    R.append(_row(2, "g29_e6cra30", "s2_full", "ra30.py", "ouro_2_6b_think", "gsm8k", 3.0, out=to,
                  deps="g29_e6cra30.s1_pilot.002"))
    R.append(_row(1, "g30_e6cra29", "s1_cell", "ra29.py", "mcleish_llama32_r32", "aqua", 1.0))
    R.append(_row(2, "g30_e6cra29", "s1_cell", "ra29.py", "huginn_0125", "csqa", 1.0))
    R.append(_row(1, "g31_e6cra31b", "s1_shard", "ra31_gen.py", "ouro_2_6b_base", "ra31", 3.0, out="$OUT_ROOT/E6/b26"))
    R.append(_row(2, "g31_e6cra31b", "s1_shard", "ra31_gen.py", "ouro_1_4b_base", "ra31", 3.0, out="$OUT_ROOT/E6/b14"))
    return R


def test_never_dropped_cells_kept_refill_and_final_lists(tmp_path, capsys):
    """One node: the 30 priority-1 lines (8 h each, alone) force every drop down to step 8; the never cells stay; the
    refill then brings back every other cell in reverse drop order (priority 1 still does not fit). The final lists:
    KEPT holds every cell but priority 1 with card-hours, DROPPED holds priority 1 with its card-hours and order step,
    then the wall estimate and the longest chain; the same block is in <plan>.summary."""
    lists = _lists(tmp_path, _synthetic())
    out = _out_root(tmp_path)
    rc, text, rows = _plan(tmp_path, capsys, lists=lists, free=1, out=out)
    assert rc == 0, text
    cells = _todo_cells(lists, out)
    assert set(cells) == {POOL, "EXT6c RA-30 McLeish ASDiv", "EXT6c McLeish", "EXT6c Huginn", "EXT5 priority 2 ASDiv",
                          "EXT5 priority 2 MAWPS", "EXT6b RA-31 Thinking Part A", "EXT6c RA-30 Thinking",
                          "EXT6b RA-29 HellaSwag 2.6B", "EXT6c 2.6B Part B", "EXT6c 1.4B Part B", "EXT5 priority 1"}, cells
    probs = _check_drops(text, rows, cells)
    assert not probs, "\n".join(probs) + "\n" + text
    drops = re.findall(r"^drop \(\w+\) (.+?):", text, re.M)
    assert drops == [lab for lab in DROP_LABELS if lab in cells], drops                # every droppable cell went
    refills = re.findall(r"^refill \(\w+\) (.+?):", text, re.M)
    assert refills == list(reversed(drops[:-1])), refills                                # all but P1, reverse order
    kept = {r[4] for r in rows["line"]}
    assert kept == set().union(*cells.values()) - cells["EXT5 priority 1"], text
    k = text.index("KEPT (11 cells, ")
    d = text.index("DROPPED (1 cells, 240.0 card-h):")
    assert k < d < text.index("estimated wall ", d), text
    assert "  (8) EXT5 priority 1: 240.0 card-h, reason: order step 8" in text[d:], text[d:]
    for c in cells:
        if c != "EXT5 priority 1":
            assert re.search(r"^  (never|\(\d\)) %s: [\d.]+ card-h$" % re.escape(c), text[k:d], re.M), (c, text[k:d])
    assert open(str(tmp_path / "plan.tsv.summary")).read() == text[k:text.index("\nplan: ")] + "\n"


def test_ra30_mcleish_asdiv_chains_behind_the_pool_and_chains_stay_whole(tmp_path):
    """The RA-30 McLeish ASDiv pilot waits (afterok) for the pool's select job; its full waits for its pilot; the
    Thinking RA-30 pilot and full are one chain (one cell, scheduled in order, submitted in one go)."""
    lists = _lists(tmp_path, _synthetic())
    plan = _todo(lists, _out_root(tmp_path))
    kept = {r["id"] for r in plan.todo}
    jobs = plan.jobs(kept)
    of = {i: n for n, j in enumerate(jobs) for i in j["ids"]}
    assert of["g18_fillgen.s3_select.001"] in jobs[of["g29_e6cra30.s1_pilot.001"]]["preds"]
    assert of["g29_e6cra30.s1_pilot.001"] in jobs[of["g29_e6cra30.s2_full.001"]]["preds"]
    assert of["g29_e6cra30.s1_pilot.002"] in jobs[of["g29_e6cra30.s2_full.002"]]["preds"]
    by = {r["id"]: r for r in plan.todo}
    assert by["g29_e6cra30.s1_pilot.002"]["cell"] == by["g29_e6cra30.s2_full.002"]["cell"] == "EXT6c RA-30 Thinking"
    # the pool (prep 0.1 h, screen 1 h, select 0.1 h: generation lines, alone at rate 1.0), then pilot 0.5 h and
    # full 2 h packed at 0.6: on 8 GPUs the wall is the chain, 1.2 + 2.5 / 0.6 h
    sub = plan.jobs({"g18_fillgen.s1_prep.001", "g18_fillgen.s2_screen.001", "g18_fillgen.s3_select.001",
                     "g29_e6cra30.s1_pilot.001", "g29_e6cra30.s2_full.001"})
    wall, crit, _l, _f = P.schedule(sub, 8)
    assert abs(wall - (1.2 + 2.5 / 0.6) * 3600) < 2 and abs(crit - wall) < 2, (wall / 3600, crit / 3600)


def test_a_cell_whose_pool_is_dropped_goes_with_it(tmp_path, capsys, monkeypatch):
    """Give the 1.4B Part B cell a needs field on the MAWPS row: when MAWPS is dropped for the budget, Part B 1.4B is
    dropped with it and printed as 'dropped: depends on <job>'."""
    cells = [c if c[1] != "EXT6c 1.4B Part B" else c[:4] + ({"group": r"^g18_fillgen$", "datasets": r"(?i)mawps"},)
             for c in P.CELLS]
    monkeypatch.setattr(P, "CELLS", cells)
    R = [r for r in _synthetic() if r["group"] in ("g18_fillgen", "g31_e6cra31b")]
    R[-3]["est_card_seconds"] = str(20 * 3600)                    # the MAWPS line: 20 h, must go
    rc, text, rows = _plan(tmp_path, capsys, lists=_lists(tmp_path, R), free=1)
    assert rc == 0, text
    assert re.search(r"^drop \(2\) EXT5 priority 2 MAWPS: ", text, re.M), text
    assert re.search(r"^  drop EXT6c 1\.4B Part B: [\d.]+ card-h, dropped: depends on g18_fillgen\.s1_prep\.002$", text, re.M), text
    kept = {r[4] for r in rows["line"]}
    assert "g31_e6cra31b.s1_shard.002" not in kept and "g31_e6cra31b.s1_shard.001" in kept, kept
    assert ["EXT6c", "EXT6c 1.4B Part B", "3.00", "depends on g18_fillgen.s1_prep.002"] in rows["drop"], rows["drop"]
    assert "  (7) EXT6c 1.4B Part B: 3.0 card-h, reason: depends on g18_fillgen.s1_prep.002" in text, text
    assert not re.search(r"^refill \(7\)", text, re.M), text          # its pool is still dropped: no refill


def test_a_never_cell_that_needs_a_droppable_line_is_an_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(P, "CELLS", [c for c in P.CELLS if c[1] != POOL])
    rc, text, rows = _plan(tmp_path, capsys, lists=_lists(tmp_path, _synthetic()), free=5)
    assert rc == 3 and "never-dropped cell depends on droppable lines" in text and not rows, text


def test_exit_2_when_the_never_dropped_cells_alone_do_not_fit(tmp_path, capsys):
    """RA-30 McLeish ASDiv full of 4.5 h: pool 1.2 h + (pilot 0.5 h + full 4.5 h) / 0.6 = 9.5 h, over 9 h."""
    lists = _lists(tmp_path, _synthetic(mc_full_h=4.5))
    rc, text, plan = _plan(tmp_path, capsys, lists=lists, free=6)
    assert rc == 2 and "never-dropped cells alone need" in text and not plan, text


@needs_bash
def test_submit_tonight_submits_nothing_on_exit_2(tmp_path):
    lists = _lists(tmp_path, _synthetic(mc_full_h=4.5))
    out = _out_root(tmp_path)
    (out / "_chain" / "jobs.txt").write_text("lists %s\n" % lists.as_posix())
    run = _tonight(tmp_path, {"FREE_NODES": "6", "TONIGHT_YES": "yes"}, out=out)
    assert run["rc"] == 2 and not run["calls"] and "nothing submitted" in run["out"], run["out"][-2000:]


# ------------------------------------------------------------------ 6. absent groups, unclassified, done, queued
def test_absent_groups_are_skipped_silently(tmp_path, capsys):
    rows = [r for r in _synthetic() if r["group"] in ("g17_fillmap", "g24_e6bra29")][-3:]
    rc, text, plan = _plan(tmp_path, capsys, lists=_lists(tmp_path, rows), free=5)
    assert rc == 0 and "DROPPED (0 cells, 0.0 card-h):" in text, text
    for word in ("EXT6c", "priority 2", "g18", "g25", "g26", "Thinking", "McLeish"):
        assert word not in text, (word, text)
    assert [u[0] for u in plan["unit"] if int(u[3])] == ["EXT5", "EXT6b"], plan["unit"]


def test_a_line_without_a_cell_is_an_error(tmp_path, capsys):
    rows = [_row(1, "g31_e6cra30", "s1_pilot", "ra30.py", "ouro_1_4b_base", "gsm8k", 1.0)]
    rc, text, _ = _plan(tmp_path, capsys, lists=_lists(tmp_path, rows), free=5)
    assert rc == 3 and "match no cell" in text and "g31_e6cra30.s1_pilot.001" in text, text


def test_done_lines_and_queued_pipelines_are_not_planned(tmp_path, capsys):
    rows = [_row(1, "g17_fillmap", "s1_scan", "k17_public.py", "ouro26_think", "D1", 1.0, out="$OUT_ROOT/B/a"),
            _row(1, "g17_fillmap", "s2_rows", "k17_public.py", "ouro26_think", "D1", 2.0, out="$OUT_ROOT/B/a",
                 deps="g17_fillmap.s1_scan.001"),
            _row(2, "g17_fillmap", "s1_scan", "k17_public.py", "ouro26_think", "D2", 1.0, out="$OUT_ROOT/B/b"),
            _row(2, "g17_fillmap", "s2_rows", "k17_public.py", "ouro26_think", "D2", 2.0, out="$OUT_ROOT/B/b",
                 deps="g17_fillmap.s1_scan.002"),
            _row(3, "g17_fillmap", "s1_scan", "k17_public.py", "ouro26_think", "D4", 1.0, out="$OUT_ROOT/B/c",
                 deps="g0_xllm.s1_build.001")]
    lists = _lists(tmp_path, rows)
    out = _out_root(tmp_path, done=("g17_fillmap.s1_scan.001",))                  # no xllm marker: pipeline c blocked
    (out / "_chain" / "submitted.tsv").write_text("201 g17_fillmap.s1_scan.tsv 2\n")
    rc, text, plan = _plan(tmp_path, capsys, lists=lists, free=1, queue="201|2|R|ra20_g17_fillmap|n01\n", out=out)
    assert rc == 0, text
    assert {r[4] for r in plan["line"]} == {"g17_fillmap.s2_rows.001"}, plan["line"]
    assert "in the queue, left alone" in text and "NOT PLANNED: needs the done marker of g0_xllm.s1_build.001" in text, text
    assert "planned units 1 running, 0 pending" in text and "the queued chains alone end at about 1.0 h" in text, text


# ------------------------------------------------------------------ 7. submit_tonight.sh end to end
def _fakes(tmp):
    """Fake sbatch (logs the submit folder and every argument, one call per line, prints 1001, 1002, ...), squeue
    (FAKE_SQUEUE for the planner's format, empty otherwise), sinfo (FAKE_SINFO for -N) and scontrol (show hostnames)."""
    fake = tmp / "bin"
    fake.mkdir(exist_ok=True)
    log = (tmp / "sbatch.log").as_posix()
    bodies = {
        "sbatch": ('d=$(pwd -W 2> /dev/null || pwd)\nprintf "%%s\\037" "$d" "$@" >> "%s"\nprintf "\\n" >> "%s"\n'
                   'n=$(grep -c "" "%s")\necho $((1000 + n))\n' % (log, log, log)),
        "squeue": 'case "$*" in *"%F|%K|%t|%j|%N"*) [ -n "${FAKE_SQUEUE:-}" ] && cat "$FAKE_SQUEUE";; esac\nexit 0',
        "sinfo": 'case "$*" in *-N*) [ -n "${FAKE_SINFO:-}" ] && cat "$FAKE_SINFO";; esac\nexit 0',
        "scontrol": 'case "$1 $2" in "show hostnames") echo "$3" | tr "," "\\n";; esac\nexit 0',
    }
    for name, body in bodies.items():
        (fake / name).write_bytes(("#!/bin/bash\n%s\n" % body).encode())
        os.chmod(str(fake / name), 0o755)
    open(log, "w").close()
    return fake, log


NO_VALUE = {"--parsable", "--hold", "--requeue", "--no-requeue", "--exclusive", "--wait", "--test-only", "-H"}


def _parse_sbatch(fields):
    opts, pos, i = {}, [], 0
    while i < len(fields):
        a = fields[i]
        if pos or not a.startswith("-"):
            pos.append(a)
        elif "=" in a and a.startswith("--"):
            k, v = a.split("=", 1)
            opts.setdefault(k, []).append(v)
        elif a in NO_VALUE:
            opts.setdefault(a, []).append("")
        else:
            opts.setdefault(a, []).append(fields[i + 1] if i + 1 < len(fields) else "")
            i += 1
        i += 1
    return opts, pos


def _tonight(tmp, extra, args=(), out=None):
    fake, log = _fakes(tmp)
    out = out or _out_root(tmp)
    (tmp / "sinfo.txt").write_text("".join("%s idle\n" % n for n in SIX))
    env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(),
               VENV_OURO="/proj/venvs/ouro", PARTITION=PARTITION, ACCOUNT=ACCOUNT, NODELIST=",".join(SIX),
               SLURM_RESTRICT="", FAKE_SINFO=(tmp / "sinfo.txt").as_posix(), PYTHON=sys.executable,
               PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for k in ("MODULES", "ALLOW_HOME", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "EXT6_YES", "FILL_YES", "EXT6B_YES",
              "EXT6C_YES", "TONIGHT_YES", "FILL_PRIORITIES", "PLAN_IDS", "FREE_NODES", "CARDS", "PACK", "TIME_LIMIT",
              "EXT6C_GROUPS", "FAKE_SQUEUE"):
        env.pop(k, None)
    env["TIME_LIMIT"] = "6:00:00"          # submit_tonight.sh forces 10:00:00 whatever the environment says
    env.update(extra)
    r = subprocess.run([_bash(), os.path.join(JOBS, "submit_tonight.sh").replace("\\", "/")] + list(args), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    calls = []
    for k, ln in enumerate(open(log, "rb").read().decode("utf-8").split("\n")[:-1], 1):
        f = ln.split("\x1f")
        opts, pos = _parse_sbatch(f[1:-1] if f and f[-1] == "" else f[1:])
        calls.append({"id": 1000 + k, "cwd": f[0], "opts": opts, "pos": pos})
    return {"rc": r.returncode, "out": r.stdout.decode("utf-8", "replace"), "calls": calls}


def _call_problems(run):
    """Every call: --exclude holds the reserve n06 and no used node, --time=10:00:00. Every call on a later stage
    (s2_full) of a pilot chain has afterok on a call of the same run (one submit, no second round)."""
    probs = [] if run["rc"] == 0 else ["submit_tonight.sh exited %d" % run["rc"]]
    if not run["calls"]:
        probs.append("no sbatch call")
    ids = set()
    for c in run["calls"]:
        exc = set(",".join(c["opts"].get("--exclude", [])).split(",")) - {""}
        if exc & set(SIX) != {"n06"}:
            probs.append("sbatch call %d excludes %s of the six (want n06 only)" % (c["id"], sorted(exc & set(SIX))))
        if c["opts"].get("--time", [""])[-1] != "10:00:00":
            probs.append("sbatch call %d --time=%s, not 10:00:00" % (c["id"], c["opts"].get("--time")))
        lst = c["pos"][1] if len(c["pos"]) > 1 else ""
        if ".s2_full." in lst:
            dep = c["opts"].get("--dependency", [""])[-1]
            m = re.match(r"^afterok:(\d+(?::\d+)*)$", dep)
            if not (m and set(int(x) for x in m.group(1).split(":")) <= ids):
                probs.append("sbatch call %d (%s) --dependency=%s: not afterok on this run's pilot" % (c["id"], lst, dep))
        ids.add(c["id"])
    return probs


def _groups(run):
    return {os.path.basename(c["pos"][1]).split(".")[0] for c in run["calls"] if len(c["pos"]) > 1}


@needs_bash
def test_submit_tonight_reserve_excluded_time_and_chains_in_one_go(tmp_path):
    """sinfo: all six idle -> five free (n06 reserved); empty queue; everything fits: both units go out whole, the
    pilot-then-full chains in the same run with afterok."""
    run = _tonight(tmp_path, {"TONIGHT_YES": "yes"})
    probs = _call_problems(run)
    assert not probs, "\n".join(probs) + "\n" + run["out"][-3000:]
    assert "nodes: 5 free (idle or mix of n01,n02,n03,n04,n05)" in run["out"], run["out"][:3000]
    groups = _groups(run)
    assert {"g17_fillmap", "g18_fillgen", "g12_ifmv1", "g24_e6bra29", "g25_e6bra31t"} <= groups, groups
    assert "g26_e6bra30" not in groups, groups          # absent in push 2 (RA decision; push 3 reruns it fresh)
    assert not groups & {"g19_e6ra30", "g20_e6ra29", "g21_e6ra31a", "g22_e6ra31b", "g23_e6merge"}, groups
    assert any(".s2_full." in (c["pos"][1] if len(c["pos"]) > 1 else "") for c in run["calls"]), "no full stage"


@needs_bash
def test_submit_tonight_dry_run_submits_nothing(tmp_path):
    run = _tonight(tmp_path, {"FREE_NODES": "5"}, args=("--dry-run",))
    assert run["rc"] == 0 and not run["calls"] and "dry run: nothing submitted" in run["out"], run["out"][-3000:]
    assert "=== unit EXT5: jobs/submit_fill.sh" in run["out"] and "=== unit EXT6b: jobs/submit_ext6b.sh" in run["out"]


@needs_bash
def test_submit_tonight_partial_units(tmp_path):
    """Two free nodes and 4 EXT6 tasks missing from submitted.tsv (each holds a GPU for 9 h): P3 and the P2 MAWPS and
    ASDiv cells go and none fits back; P2 keeps the
    McLeish ASDiv pool and SVAMP. Without PLAN_IDS, submit_fill.sh gets FILL_PRIORITIES=1 (the whole priority kept)
    and names priority 2 as not submitted; EXT6b, kept whole, goes out whole."""
    q = tmp_path / "queue.txt"
    q.write_text("".join("30%d|1|PD|ra20_g22_e6ra31b|\n" % i for i in range(4)))
    run = _tonight(tmp_path, {"FREE_NODES": "2", "TONIGHT_YES": "yes", "FAKE_SQUEUE": q.as_posix()})
    probs = _call_problems(run)
    assert not probs, "\n".join(probs) + "\n" + run["out"][-3000:]
    assert "EXT6 0 running, 4 pending" in run["out"] and "4 queued task(s) not in submitted.tsv" in run["out"], run["out"][:3000]
    if "PLAN_IDS" not in open(os.path.join(JOBS, "submit_fill.sh")).read():
        assert _groups(run) == {"g17_fillmap", "g24_e6bra29", "g25_e6bra31t", "g28_e6bmerge"}, _groups(run)
        assert "submitted with FILL_PRIORITIES=1; NOT SUBMITTED: priority 2 (partly dropped)" in run["out"], run["out"]


@needs_bash
def test_submit_tonight_prints_the_final_lists_in_the_dry_run_and_before_the_yes(tmp_path):
    """--dry-run: the KEPT and DROPPED lists, the wall estimate and the longest chain print after the unit dry runs and
    before 'dry run: nothing submitted'. A real run prints them before its first submission."""
    run = _tonight(tmp_path, {"FREE_NODES": "2"}, args=("--dry-run",))
    o = run["out"]
    assert run["rc"] == 0 and not run["calls"], o[-3000:]
    i = o.index("=== tonight's plan (after the drops and the refill)")
    assert o.rindex("(dry run)") < i < o.index("KEPT (", i) < o.index("DROPPED (", i) \
        < o.index("estimated wall ", i) < o.rindex("dry run: nothing submitted") and o.rstrip().endswith("dry run: nothing submitted"), o[-3000:]
    assert re.search(r"^  \(2\) EXT5 priority 2 MAWPS: [\d.]+ card-h, reason: order step 2$", o[i:], re.M), o[i:]
    assert re.search(r"^estimated wall [\d.]+ h \(budget 9\.0 h\), longest chain [\d.]+ h \(", o[i:], re.M), o[i:]
    real = tmp_path / "real"
    real.mkdir()
    run = _tonight(real, {"FREE_NODES": "5", "TONIGHT_YES": "yes"})
    o = run["out"]
    assert run["rc"] == 0 and run["calls"], o[-3000:]
    i = o.index("=== tonight's plan (after the drops and the refill)")
    assert o.index("KEPT (", i) < o.index("DROPPED (", i) < o.index("=== unit EXT5: jobs/submit_fill.sh\n"), o[-4000:]



def test_ext6c_list_queued_ext6_dependencies_and_the_flag_cells(tmp_path, capsys):
    """joblist_ext6c.tsv is read beside joblist.tsv. The GSM8K flags depend on the EXT6 full (queued): they are planned
    (never dropped) and wait for that queued chain (afterok), so they end after it. The ASDiv flags depend on the
    EXT6b ASDiv full; when the budget drops that cell (20 h, alone), the flags go with it ('depends on'). With no queue
    and no done marker, the GSM8K flags are blocked by the EXT6 dependency."""
    o19, o26 = "$OUT_ROOT/E6/ra30", "$OUT_ROOT/E6/ra30b"
    main = [_row(1, "g19_e6ra30", "s1_pilot", "ra30.py", "ouro_2_6b_base", "ra30", 1.0, out=o19),
            _row(1, "g19_e6ra30", "s2_full", "ra30.py", "ouro_2_6b_base", "ra30", 3.0, out=o19, deps="g19_e6ra30.s1_pilot.001"),
            _row(1, "g26_e6bra30", "s1_pilot", "ra30.py", "ouro_2_6b_base", "asdiv", 0.5, out=o26),
            _row(1, "g26_e6bra30", "s2_full", "ra30.py", "ouro_2_6b_base", "asdiv", 20.0, out=o26, deps="g26_e6bra30.s1_pilot.001")]
    e6c = [_row(1, "g30_e6cra30f", "s1_flags", "ra30_flags.py", "ouro_2_6b_base", "gsm8k", 0.1, out="$OUT_ROOT/f/g",
                deps="g19_e6ra30.s2_full.001"),
           _row(2, "g30_e6cra30f", "s1_flags", "ra30_flags.py", "ouro_2_6b_base", "asdiv", 0.1, out="$OUT_ROOT/f/a",
                deps="g26_e6bra30.s2_full.001")]
    lists = _lists(tmp_path, main, ext6c_rows=e6c)
    out = _out_root(tmp_path)
    (out / "_chain" / "submitted.tsv").write_text("601 g19_e6ra30.s1_pilot.tsv 1\n602 g19_e6ra30.s2_full.tsv 1\n")
    q = "601|1|R|ra20_g19_e6ra30|n01|0:00\n602|1|PD|ra20_g19_e6ra30||0:00\n"
    rc, text, rows = _plan(tmp_path, capsys, lists=lists, free=1, queue=q, out=out)
    assert rc == 0, text
    assert "  never EXT6c RA-30 flags on EXT6 GSM8K: 1 lines, 0.1 card-h" in text, text
    # g26 (EXT6b RA-30 ASDiv 2.6B) is absent in push 2: never planned or dropped; the ASDiv flags that need it are held
    assert "EXT6b RA-30 ASDiv 2.6B" not in text, text
    assert "NOT PLANNED: needs the done marker of g26_e6bra30.s2_full.001" in text, text
    assert {r[4] for r in rows["line"]} == {"g30_e6cra30f.s1_flags.001"}, rows["line"]
    # the flags wait for the queued EXT6 chain: pilot 1 h, full 3 h (unpacked), then the flags (0.1 h packed at 0.6)
    assert re.search(r"^estimated wall 4\.2 h \(budget 9\.0 h\)", text, re.M), text
    rc, text, rows = _plan(tmp_path, capsys, lists=lists, free=1, out=_out_root(tmp_path / "empty"))
    assert rc == 0 and "NOT PLANNED: needs the done marker of g19_e6ra30.s2_full.001" in text, text
