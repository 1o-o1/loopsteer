"""Dry run of the whole cluster chain with fake Slurm over the real job lists (jobs/nolocal/), plus the static checks a
bundle must pass before it is handed over (PLAN.md, SLURM CONTEXT: "Gate before any handover").

    cd loopedcot; python -B -m pytest cluster/tests/test_slurm_dryrun.py -q -p no:cacheprovider

1. submit_chain.sh (top level, --step probe_collect, --step stage_submit) with fake sbatch, squeue, scontrol and
   nvidia-smi: every sbatch call is well formed, every task of the job list is submitted once, every afterok covers
   the task's depends_on, the merge waits afterany for every other job.
2. Every task line of jobs/nolocal/*.tsv and jobs/smoke_*.tsv: bundle paths exist, scripts are pinned, configs exist,
   activated venvs are exported by the cluster.env that setup_cluster.sh --dry-run writes.
3. Every model key of the job lists and every checkpoint path a task line names maps to cluster/weights.json.
4. bash -n on every *.sh and *.sbatch, py_compile (into a temporary folder) on every *.py under cluster/.
5. Every pin of jobs/scripts_sha256.txt matches; every file a task runs (and the bundle modules it imports) is pinned.
6. Every *.sh, *.sbatch and *.tsv under cluster/ has LF line endings.
7. B-EXT6: jobs/submit_ext6.sh with fake Slurm submits only its groups (done markers skipped, afterok inside each
   pipeline, one merge per test afterany on its jobs, card-hours per job and in total, nothing without a yes); the
   templates hold a wall limit of at most 6 h; every task line names the frozen plan sha256 of its folder copy;
   merge_ext6.py copies and scrubs into merged/<test> and leaves the run tree alone.
Every failure names the file, the line or the job id. The tests read the tree; they write only to temporary folders.
"""
import ast
import csv
import glob
import hashlib
import json
import os
import py_compile
import re
import shlex
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
PINS = os.path.join(JOBS, "scripts_sha256.txt")
sys.dont_write_bytecode = True

# the values the chain runs with; the run's node list has seven nodes, n07 fails its probe, n08 is excluded by setup
PARTITION, ACCOUNT = "gpuq", "acct1"
NODES = ["n01", "n02", "n03", "n04", "n05", "n06", "n07"]
BASE_EXCLUDE = {"n08"}
BAD_NODES = {"n07"}
SLURM_ENV = {"PARTITION": PARTITION, "ACCOUNT": ACCOUNT, "SLURM_RESTRICT": "--exclude=n08", "NODELIST": ",".join(NODES),
             "NODES_EXPECTED": "6"}
LIST_RE = re.compile(r"^(g\d+_\w+\.s\d+_\w+|smoke_\w+)\.tsv$")
NOT_CHECKPOINT = {("build_xllm.sh", "xllm"), ("merge.py", "all"), ("merge_ext6.py", "all")}   # no model loaded
STDLIB_MODULES = {"unittest"}        # python -m <stdlib module> in a task line (the B-EXT6 folder tests)
E6_GROUPS = ("g19_e6ra30", "g20_e6ra29", "g21_e6ra31a", "g22_e6ra31b")
E6_MERGE = "g23_e6merge"             # one merge per test, afterany on that test's jobs
E6B_GROUPS = ("g24_e6bra29", "g25_e6bra31t", "g26_e6bra30")       # B-EXT6b, jobs/submit_ext6b.sh
E6B_MERGE = "g28_e6bmerge"
# (script, run groups, merge group, yes variable, run tasks expected, task given a done marker, Part B shards)
EXT6_UNITS = [("submit_ext6.sh", E6_GROUPS, E6_MERGE, "EXT6_YES", 8 + 16, "g20_e6ra29.s1_pilot.002", 15),
              ("submit_ext6b.sh", E6B_GROUPS, E6B_MERGE, "EXT6B_YES", 4 + 4 + 2, "g24_e6bra29.s1_cell.002", 0)]


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


def _rel(p):
    return os.path.relpath(p, TOP).replace("\\", "/")


def _report(problems, what):
    assert not problems, "%d %s:\n%s" % (len(problems), what, "\n".join(problems[:120]))


# ------------------------------------------------------------------ task lists and their lines
def _task_lists():
    """(path, display name) of every task list: the stage lists and smoke lists of jobs/nolocal/, jobs/smoke_*.tsv."""
    out = [p for p in sorted(glob.glob(os.path.join(NOLOCAL, "*.tsv"))) if LIST_RE.match(os.path.basename(p))]
    out += sorted(glob.glob(os.path.join(JOBS, "smoke_*.tsv")))
    return out


def _lines(path):
    """(line number, id, command) of a task list (the template runs line SLURM_ARRAY_TASK_ID: id TAB command)."""
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    return [(i, ln.split("\t", 1)[0], ln.split("\t", 1)[1] if "\t" in ln else "") for i, ln in enumerate(raw, 1)]


def _joblist(d):
    with open(os.path.join(d, "joblist.tsv"), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _pins():
    """{repo-relative path: (sha256, line number)} of jobs/scripts_sha256.txt and the malformed lines."""
    pins, bad = {}, []
    with open(PINS, "rb") as f:
        text = f.read().decode("utf-8")
    for i, ln in enumerate(text.split("\n"), 1):
        if not ln.strip():
            continue
        m = re.match(r"^([0-9a-f]{64})  (\S.*)$", ln)
        if not m:
            bad.append("jobs/scripts_sha256.txt:%d: not 'sha256  path' (sha256sum -c format): %r" % (i, ln[:100]))
            continue
        if m.group(2) in pins:
            bad.append("jobs/scripts_sha256.txt:%d: %s pinned twice" % (i, m.group(2)))
        pins[m.group(2)] = (m.group(1), i)
    return pins, bad


def _expand(tok):
    """A bundle path of a task line on this machine: $REPO -> the repository top, $DATA_ROOT -> its default
    (cluster/pilot/data, the template's default); None for any other variable."""
    tok = re.sub(r"\$\{(\w+)\}", r"$\1", tok)
    if tok.startswith("$REPO/"):
        return os.path.normpath(os.path.join(TOP, tok[len("$REPO/"):]))
    if tok.startswith("$DATA_ROOT/"):
        return os.path.normpath(os.path.join(CL, "pilot", "data", tok[len("$DATA_ROOT/"):]))
    return None


def _parse_line(cmd):
    """What a task line runs: [(kind, path or module, cwd)] with kind python / module / shell, the venv variables
    it activates, and an error text when the line does not split as a shell command."""
    runs, venvs = [], []
    try:
        toks = shlex.split(cmd.replace("$$", "PID"), posix=True)
    except ValueError as e:
        return runs, venvs, "shell split: %s" % e
    cwd = None
    seps = {"&&", "||", ";", "|", "{", "}", "(", ")"}
    i = 0
    while i < len(toks):
        t = toks[i].rstrip(";")
        nxt = toks[i + 1].rstrip(";") if i + 1 < len(toks) else ""
        if t == "cd" and nxt:
            cwd = _expand(nxt) or nxt
        elif t in ("source", ".") and nxt:
            m = re.match(r"^\$\{?(VENV_\w+?)\}?/bin/activate$", nxt)
            if m:
                venvs.append(m.group(1))
        elif t in ("python", "python3") or re.match(r"^\$\{?VENV_\w+\}?/bin/python3?$", t):
            j = i + 1
            while j < len(toks) and toks[j] in ("-u", "-B", "-O", "-I", "-E", "-s"):
                j += 1
            if j < len(toks) and toks[j] == "-m" and j + 1 < len(toks):
                runs.append(("module", toks[j + 1].rstrip(";"), cwd))
            elif j < len(toks) and toks[j] != "-c" and toks[j] not in seps:
                runs.append(("python", toks[j].rstrip(";"), cwd))
        elif t in ("bash", "sh") and nxt and not nxt.startswith("-"):
            runs.append(("shell", nxt, cwd))
        i += 1
    return runs, venvs, None


def _resolve(kind, target, cwd):
    """The bundle file a run names, or (None, why)."""
    if kind == "module":
        if target in STDLIB_MODULES:
            return None, None
        if not cwd or not os.path.isabs(cwd):
            return None, "python -m %s without a known cd folder" % target
        base = os.path.join(cwd, *target.split("."))
        for c in (base + ".py", os.path.join(base, "__main__.py")):
            if os.path.isfile(c):
                return os.path.normpath(c), None
        return None, "python -m %s: no %s.py under %s" % (target, base, _rel(cwd))
    p = _expand(target)
    if p is None and not target.startswith("$") and cwd and os.path.isabs(cwd):
        p = os.path.normpath(os.path.join(cwd, target))
    if p is None:
        return None, "%s %s: not a bundle path ($REPO/... or a path relative to a bundle cd)" % (kind, target)
    if not os.path.isfile(p):
        return None, "%s %s: file missing (%s)" % (kind, target, _rel(p))
    return p, None


def _py_imports(path, cwd):
    """Bundle files a python file imports (the package __init__ files included): resolved against its own folder,
    the task's cd folder and loopedcot/ (cluster.*); only files inside cluster/ count."""
    try:
        with open(path, "rb") as f:
            tree = ast.parse(f.read(), filename=path)
    except SyntaxError:
        return set()
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [(a.name, 0, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [((node.module or ""), node.level, [a.name for a in node.names])]
    bases = [os.path.dirname(path)] + ([cwd] if cwd and os.path.isabs(cwd) else []) + [LC]
    found = set()
    for mod, level, subs in names:
        if level:
            base = os.path.dirname(path)
            for _ in range(level - 1):
                base = os.path.dirname(base)
            cands = [(base, mod.split(".") if mod else [])]
        else:
            cands = [(b, mod.split(".")) for b in bases if b != LC or mod.split(".")[0] == "cluster"]
        for b, parts in cands:
            hit = []
            d = b
            for k, part in enumerate(parts):
                d = os.path.join(d, part)
                if os.path.isfile(os.path.join(d, "__init__.py")):
                    hit.append(os.path.join(d, "__init__.py"))
                if k == len(parts) - 1 and os.path.isfile(d + ".py"):
                    hit.append(d + ".py")
            if subs and os.path.isdir(d):
                for s in subs:
                    for c in (os.path.join(d, s + ".py"), os.path.join(d, s, "__init__.py")):
                        if os.path.isfile(c):
                            hit.append(c)
            if not parts and subs:
                for s in subs:
                    if os.path.isfile(os.path.join(b, s + ".py")):
                        hit.append(os.path.join(b, s + ".py"))
            hit = [os.path.normpath(h) for h in hit if os.path.normpath(h).startswith(os.path.normpath(CL) + os.sep)]
            if hit:
                found |= set(hit)
                break
    return found


def _shell_refs(path):
    """Bundle scripts a shell script runs: $HERE/<x>, $REPO/loopedcot/<x>, $LC/cluster/<x> (*.py, *.sh)."""
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", "replace")
    out = set()
    for m in re.finditer(r"\$\{?(HERE|REPO|LC)\}?/([\w./-]+\.(?:py|sh))\b", text):
        base = {"HERE": os.path.dirname(path), "REPO": TOP, "LC": LC}[m.group(1)]
        p = os.path.normpath(os.path.join(base, m.group(2)))
        if os.path.isfile(p) and p.startswith(os.path.normpath(CL) + os.sep):
            out.add(p)
    return out


def _run_closure():
    """{bundle file: first task id that runs it}: the scripts every task line runs, the bundle modules they import and
    the bundle scripts the shell scripts run, followed to the end."""
    seen = {}
    todo = []
    for path in _task_lists():
        for n, jid, cmd in _lines(path):
            runs, _, _ = _parse_line(cmd)
            for kind, target, cwd in runs:
                p, _why = _resolve(kind, target, cwd)
                if p and p not in seen:
                    seen[p] = jid
                    todo.append((p, cwd, jid))
    while todo:
        p, cwd, jid = todo.pop()
        nxt = _py_imports(p, cwd) if p.endswith(".py") else _shell_refs(p) if p.endswith(".sh") else set()
        for q in nxt:
            if q not in seen:
                seen[q] = jid
                todo.append((q, cwd if q.endswith(".py") else None, jid))
    return seen


# ------------------------------------------------------------------ setup_cluster.sh --dry-run: the cluster.env
@pytest.fixture(scope="module")
def cluster_env(tmp_path_factory):
    """The cluster.env that setup_cluster.sh --dry-run writes for a six-node run (a copy of env/, weights.json and the
    job templates in a temporary repository layout, fake sinfo and scontrol)."""
    if _bash() is None:
        pytest.skip("no bash")
    tmp = tmp_path_factory.mktemp("setup")
    lc = tmp / "proj" / "repo" / "loopedcot" / "cluster"
    (lc / "jobs").mkdir(parents=True)
    shutil.copytree(os.path.join(CL, "env"), str(lc / "env"), ignore=shutil.ignore_patterns("cluster.env", "__pycache__"))
    shutil.copy(os.path.join(CL, "weights.json"), str(lc / "weights.json"))
    for f in glob.glob(os.path.join(JOBS, "task_*.sbatch")) + glob.glob(os.path.join(JOBS, "g2_gen.*.tsv")):
        shutil.copy(f, str(lc / "jobs"))
    fake = tmp / "bin"
    fake.mkdir()
    for name, body in (("sinfo", 'printf "n01\\nn02\\nn03\\nn04\\nn05\\nn06\\n"'), ("scontrol", 'echo "$3" | tr "," "\\n"')):
        (fake / name).write_bytes(("#!/bin/bash\n%s\n" % body).encode())
        os.chmod(str(fake / name), 0o755)
    home = tmp / "home"
    home.mkdir()
    proj = (tmp / "proj").as_posix()
    env = dict(os.environ, HOME=home.as_posix(), PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for k in ("OUT_ROOT", "HF_HOME", "WEIGHTS_ROOT", "PARTITION", "ACCOUNT", "NODELIST", "TORCH_CUDA_ARCH_LIST", "ALLOW_HOME"):
        env.pop(k, None)
    cmd = [_bash(), (lc / "env" / "setup_cluster.sh").as_posix(), "--dry-run", "--python", sys.executable.replace("\\", "/"),
           "--hf-home", proj + "/hf", "--weights-root", proj + "/ckpt", "--partition", PARTITION, "--account", ACCOUNT,
           "--out-root", proj + "/out", "--nodelist", "n01,n02,n03,n04,n05,n06"]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p = os.path.join(proj, "out", "cluster.env")
    assert os.path.exists(p), "setup_cluster.sh --dry-run wrote no cluster.env (exit %d):\n%s" % (
        r.returncode, r.stdout.decode("utf-8", "replace")[-2500:])
    with open(p, encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------ 1. the chain with fake Slurm
def _fakes(tmp):
    """Fake sbatch (logs the submit folder and every argument, one call per line, prints job ids 1001, 1002, ...),
    squeue (an empty queue), scontrol (show hostnames) and nvidia-smi (one card)."""
    fake = tmp / "bin"
    fake.mkdir(exist_ok=True)
    log = (tmp / "sbatch.log").as_posix()
    bodies = {
        "sbatch": ('d=$(pwd -W 2> /dev/null || pwd)\nprintf "%%s\\037" "$d" "$@" >> "%s"\nprintf "\\n" >> "%s"\n'
                   'n=$(grep -c "" "%s")\necho $((1000 + n))\n' % (log, log, log)),
        "squeue": "exit 0",
        "scontrol": 'case "$1 $2" in "show hostnames") echo "$3" | tr "," "\\n";; esac\nexit 0',
        "nvidia-smi": 'case "$*" in *-L*) echo "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-0)";; '
                      '*) echo "NVIDIA A100-SXM4-80GB, 81000 MiB, 81920 MiB";; esac',
    }
    for name, body in bodies.items():
        (fake / name).write_bytes(("#!/bin/bash\n%s\n" % body).encode())
        os.chmod(str(fake / name), 0o755)
    open(log, "w").close()
    return fake, log


NO_VALUE = {"--parsable", "--hold", "--requeue", "--no-requeue", "--exclusive", "--wait", "--test-only", "-H"}


def _parse_sbatch(fields):
    """(options {name: [values]}, positional arguments) of one sbatch call, as sbatch reads them."""
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


def _array(spec):
    """Task indices of an --array value (1,2,5-7%48), or None when it does not parse."""
    spec = spec.split("%", 1)
    if len(spec) == 2 and not re.match(r"^[1-9]\d*$", spec[1]):
        return None
    idx = []
    for part in spec[0].split(","):
        m = re.match(r"^(\d+)(?:-(\d+)(?::(\d+))?)?$", part)
        if not m:
            return None
        a, b, s = int(m.group(1)), int(m.group(2) or m.group(1)), int(m.group(3) or 1)
        idx += list(range(a, b + 1, s))
    return idx


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """submit_chain.sh with DONE_SRC=none: the top level, then --step probe_collect (n01..n06 healthy, n07 fails its
    probe), then --step stage_submit (with _smoke.ok and the g0 done marker; SLURM_RESTRICT from run_nodes.env, as the
    probe_collect job exports it). Returns the runs and the sbatch calls of each."""
    if _bash() is None:
        pytest.skip("no bash")
    tmp = tmp_path_factory.mktemp("chain")
    fake, log = _fakes(tmp)
    out = tmp / "out"
    out.mkdir()
    env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(),
               DONE_SRC="none", VENV_OURO="/proj/venvs/ouro", CARDS="48",
               PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""), **SLURM_ENV)
    for k in ("MODULES", "SLURM_RESTRICT_BASE", "ALLOW_HOME", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        env.pop(k, None)
    chain_sh = os.path.join(JOBS, "submit_chain.sh").replace("\\", "/")
    runs = []

    def run(name, args, env_):
        before = len(open(log, "rb").read().split(b"\n")) - 1
        r = subprocess.run([_bash(), chain_sh] + args, env=env_, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = open(log, "rb").read().decode("utf-8").split("\n")[:-1]
        calls = []
        for k, ln in enumerate(lines[before:], before + 1):
            f = ln.split("\x1f")
            opts, pos = _parse_sbatch(f[1:-1] if f and f[-1] == "" else f[1:])
            calls.append({"id": 1000 + k, "cwd": f[0], "args": f[1:-1], "opts": opts, "pos": pos})
        runs.append({"name": name, "rc": r.returncode, "out": r.stdout.decode("utf-8", "replace"), "calls": calls,
                     "first": 1000 + before + 1})
        return r

    run("submit_chain.sh", [], env)
    (out / "_chain" / "setup_check.ok").write_text("ok\n")
    pr = out / "_probe"
    pr.mkdir(exist_ok=True)
    for n in NODES:
        if n in BAD_NODES:
            (pr / ("%s.out" % n)).write_text("node=%s\nNo devices were found\nRC=1\n" % n)
        else:
            (pr / ("%s.out" % n)).write_text("node=%s\nGPU 0: card\nmatmul card 1.0\nRC=0\n" % n)
    run("--step probe_collect", ["--step", "probe_collect"], env)
    g0 = _lines(os.path.join(NOLOCAL, "g0_xllm.s1_build.tsv"))[0][1]
    (out / "_done").mkdir(exist_ok=True)
    (out / "_done" / g0).write_text("")
    (out / "_smoke.ok").write_text("ok\n")
    (out / "_chain" / "smoke.ok").write_text("ok\n")
    env_s = dict(env)
    rn = out / "run_nodes.env"
    if rn.exists():
        m = re.search(r'export SLURM_RESTRICT="([^"]*)"', rn.read_text())
        if m:
            env_s["SLURM_RESTRICT"] = m.group(1)
    run("--step stage_submit", ["--step", "stage_submit"], env_s)
    return {"runs": runs, "out": out, "env": env}


def _check_call(c, run, env, issued):
    """Problems of one sbatch call (partition, account, restriction, output, parsable, array, dependency, files)."""
    p = []
    o, pos = c["opts"], c["pos"]
    tag = "%s: sbatch call %d (job %d%s)" % (run["name"], c["id"] - run["first"] + 1, c["id"],
                                              ", " + " ".join(pos[:2]) if pos else ", --wrap")
    if "--parsable" not in o:
        p.append("%s: no --parsable (the job id is read from its output)" % tag)
    if o.get("--partition", [""])[-1] != PARTITION or len(set(o.get("--partition", []))) > 1:
        p.append("%s: --partition is %s, not %s" % (tag, o.get("--partition"), PARTITION))
    if o.get("--account", [""])[-1] != ACCOUNT:
        p.append("%s: --account is %s, not %s" % (tag, o.get("--account"), ACCOUNT))
    outp = o.get("--output", [""])[-1]
    log_root, out_root = env["LOG_ROOT"], env["OUT_ROOT"]
    if not outp:
        p.append("%s: no --output" % tag)
    elif not (outp.startswith(log_root + "/") or outp.startswith(out_root + "/_probe/")):
        p.append("%s: --output %s is not under LOG_ROOT or OUT_ROOT/_probe" % (tag, outp))
    if "--nodelist" in o:
        if o["--nodelist"][-1] not in NODES:
            p.append("%s: --nodelist=%s is not a node of NODELIST" % (tag, o["--nodelist"][-1]))
    else:
        exc = set(",".join(o.get("--exclude", [])).split(",")) - {""}
        want = BASE_EXCLUDE | (BAD_NODES if run["name"] != "submit_chain.sh" else set())
        if not want <= exc:
            p.append("%s: --exclude=%s lacks the node restriction %s" % (tag, ",".join(sorted(exc)), ",".join(sorted(want - exc))))
    wrap = set(o.get("--wrap", []))
    for a in c["args"]:
        if a not in wrap and a.split("=", 1)[-1] not in wrap and re.search(r"<[A-Z_]+>|\$", a):
            p.append("%s: argument %r keeps a placeholder or an unexpanded variable" % (tag, a[:120]))
    for dep in o.get("--dependency", []):
        for clause in dep.split(","):
            parts = clause.split(":")
            if parts[0] not in ("afterok", "afterany", "afternotok", "after", "aftercorr") or len(parts) < 2 \
                    or not all(re.match(r"^\d+$", x) for x in parts[1:]):
                p.append("%s: malformed --dependency clause %r" % (tag, clause))
                continue
            for x in parts[1:]:
                if not (run["first"] <= int(x) < c["id"]) or int(x) not in issued:
                    p.append("%s: --dependency %s:%s names a job not issued earlier in this run" % (tag, parts[0], x))
    if "--wrap" in o:
        w = o["--wrap"][-1]
        m = re.match(r"^bash (\S+) --step (\w+)$", w)
        if m and not os.path.isfile(m.group(1)):
            p.append("%s: --wrap runs %s, which does not exist" % (tag, m.group(1)))
        elif not m and not w.startswith("echo node="):
            p.append("%s: unknown --wrap command %r" % (tag, w[:120]))
        return p
    if len(pos) < 2:
        p.append("%s: needs a template and a task list, has %s" % (tag, pos))
        return p
    tpl = os.path.join(c["cwd"], pos[0])
    lst = os.path.join(c["cwd"], pos[1])
    if not pos[0].endswith(".sbatch") or not os.path.isfile(tpl):
        p.append("%s: template %s missing (submit folder %s)" % (tag, pos[0], c["cwd"]))
    if not os.path.isfile(lst):
        p.append("%s: task list %s missing (submit folder %s)" % (tag, pos[1], c["cwd"]))
        return p
    n = len(_lines(lst))
    arr = o.get("--array", [""])[-1]
    idx = _array(arr) if arr else None
    if not idx:
        p.append("%s: --array %r missing or malformed" % (tag, arr))
    else:
        out_of = [i for i in idx if not 1 <= i <= n]
        if out_of:
            p.append("%s: --array index %s outside %s (%d lines)" % (tag, out_of[:5], pos[1], n))
    c["tasks"] = [(lst, i) for i in (idx or []) if 1 <= i <= n]
    return p


@needs_bash
def test_chain_dry_run_submits_every_task_well_formed(chain):
    runs = chain["runs"]
    probs = []
    for run in runs:
        if run["rc"] != 0:
            probs.append("%s exited %d:\n%s" % (run["name"], run["rc"], run["out"][-2500:]))
        issued = set()
        for c in run["calls"]:
            probs += _check_call(c, run, chain["env"], issued)
            issued.add(c["id"])
    top, pc, ss = runs
    # the top level: setup_check, one probe per node, probe_collect after both
    if len(top["calls"]) != 2 + len(NODES):
        probs.append("submit_chain.sh: %d sbatch calls, expected %d (setup_check, %d probes, probe_collect)"
                     % (len(top["calls"]), 2 + len(NODES), len(NODES)))
    elif "--step probe_collect" not in top["calls"][-1]["opts"].get("--wrap", [""])[-1] or \
            set(re.findall(r"\d+", top["calls"][-1]["opts"].get("--dependency", [""])[-1])) != \
            {str(c["id"]) for c in top["calls"][:-1]}:
        probs.append("submit_chain.sh: probe_collect does not wait for setup_check and every probe")
    if not (pc["calls"] and pc["calls"][0]["pos"] and pc["calls"][0]["pos"][0].endswith("task_g0_xllm.sbatch")):
        probs.append("--step probe_collect: the first call is not the g0 template")
    # stage_submit: every task of the job list once, afterok covers depends_on, the merge waits afterany for all
    rows = {r["id"]: r for r in _joblist(NOLOCAL)}
    job_tasks, seen = {}, {}
    for c in ss["calls"]:
        ids = []
        for lst, i in c.get("tasks", []):
            ln = _lines(lst)[i - 1]
            ids.append(ln[1])
            seen.setdefault(ln[1], []).append(c["id"])
        job_tasks[c["id"]] = ids
    g0 = {k for k, r in rows.items() if r["group"] == "g0_xllm"}
    for k in sorted(set(rows) - g0):
        if k not in seen:
            probs.append("--step stage_submit: task %s (nolocal/joblist.tsv) is never submitted" % k)
        elif len(seen[k]) > 1:
            probs.append("--step stage_submit: task %s submitted %d times (jobs %s)" % (k, len(seen[k]), seen[k]))
    for k in sorted(set(seen) - set(rows)):
        probs.append("--step stage_submit: task %s submitted but not in nolocal/joblist.tsv" % k)
    merge = [c for c in ss["calls"] if c["pos"] and os.path.basename(c["pos"][0]) == "task_g9_merge.sbatch"]
    for c in ss["calls"]:
        deps = c["opts"].get("--dependency", [""])[-1]
        kind = "afterany:" if c["pos"] and os.path.basename(c["pos"][0]) in ("task_%s.sbatch" % E6_MERGE,
                                                                             "task_%s.sbatch" % E6B_MERGE) else "afterok:"
        ok_ids = {int(x) for cl in deps.split(",") if cl.startswith(kind) for x in cl.split(":")[1:]}
        covered = {t for j in ok_ids for t in job_tasks.get(j, [])}
        for t in job_tasks[c["id"]]:
            need = {d for d in rows.get(t, {}).get("depends_on", "").split(",") if d and d not in g0}
            if need - covered:
                probs.append("--step stage_submit: job %d task %s depends_on %s, not in its afterok jobs %s"
                             % (c["id"], t, sorted(need - covered)[:3], sorted(ok_ids)))
    if len(merge) != 1:
        probs.append("--step stage_submit: %d merge (task_g9_merge.sbatch) calls, expected 1" % len(merge))
    else:
        m = merge[0]
        deps = m["opts"].get("--dependency", [""])[-1]
        anyids = {int(x) for cl in deps.split(",") if cl.startswith("afterany:") for x in cl.split(":")[1:]}
        others = {c["id"] for c in ss["calls"] if c is not m}
        if others - anyids:
            probs.append("--step stage_submit: the merge (job %d) does not wait afterany for jobs %s"
                         % (m["id"], sorted(others - anyids)[:10]))
        if "--hold" in m["opts"]:
            probs.append("--step stage_submit: the merge is submitted --hold")
    _report(probs, "chain problems")


# ------------------------------------------------------------------ 2. task lines
def test_task_lines_paths_scripts_configs_and_venvs(cluster_env):
    pins, _ = _pins()
    exported = set(re.findall(r"(?:^export |^| )([A-Z][A-Z0-9_]*)=", cluster_env, re.M))
    probs = []
    lists = _task_lists()
    if not lists:
        probs.append("no task lists in jobs/nolocal/ or jobs/smoke_*.tsv")
    for path in lists:
        name = _rel(path)
        for n, jid, cmd in _lines(path):
            where = "%s:%d (%s)" % (name, n, jid)
            if not jid or not cmd:
                probs.append("%s: not 'id TAB command'" % where)
                continue
            for m in re.finditer(r"\$\{?REPO\}?/(loopedcot/cluster/[^\s;&|()<>'\"]+)", cmd):
                rel = m.group(1).rstrip("}.,")
                if not os.path.exists(os.path.join(TOP, rel)):
                    probs.append("%s: names %s, missing from the bundle" % (where, rel))
            for m in re.finditer(r"\$\{?DATA_ROOT\}?/([^\s;&|()<>'\"]+)", cmd):
                rel = m.group(1).rstrip("}.,")
                if not os.path.exists(os.path.join(CL, "pilot", "data", rel)):
                    probs.append("%s: names $DATA_ROOT/%s, missing from cluster/pilot/data (the DATA_ROOT default)" % (where, rel))
            for m in re.finditer(r"--config-json[= ](\S+)", cmd):
                p = _expand(m.group(1))
                if p is None or not os.path.isfile(p):
                    probs.append("%s: --config-json %s does not exist" % (where, m.group(1)))
            runs, venvs, err = _parse_line(cmd)
            if err:
                probs.append("%s: %s" % (where, err))
            for kind, target, cwd in runs:
                p, why = _resolve(kind, target, cwd)
                if why:
                    probs.append("%s: %s" % (where, why))
                elif p and p.endswith(".py") and _rel(p) not in pins:
                    probs.append("%s: runs %s, not in jobs/scripts_sha256.txt" % (where, _rel(p)))
            for v in venvs:
                if v not in exported:
                    probs.append("%s: activates $%s, not exported by cluster.env (setup_cluster.sh --dry-run)" % (where, v))
    _report(probs, "task-line problems")


# ------------------------------------------------------------------ 3. checkpoints
def test_every_checkpoint_maps_to_weights_json():
    with open(os.path.join(CL, "weights.json"), encoding="utf-8") as f:
        w = json.load(f)
    ck = {c["key"]: c for c in w["checkpoints"]}
    locs = {c.get("location", "") for c in w["checkpoints"] + w.get("auxiliary", [])}
    sys.path.insert(0, LC)
    try:
        from cluster import gen_jobs as GJ          # the k17_public run groups (mcl_nr = McLeish + Llama non-recurrent)
        groups = {}
        for k, g in GJ.PUB_GROUP.items():
            groups.setdefault(g, set()).add(k)
    except Exception as e:                           # noqa: BLE001
        pytest.fail("cluster/gen_jobs.py does not import (it holds the model-key mapping PUB_GROUP): %r" % e)
    probs = []
    for d in (NOLOCAL, JOBS):
        for r in _joblist(d):
            m, script = r["model"], r["script"]
            if (script, m) in NOT_CHECKPOINT:
                continue
            stem = os.path.splitext(script)[0]
            keys = {m} & set(ck) or groups.get(m, set()) or \
                {k for k, c in ck.items() if c.get("registry_names", {}).get(stem) == m}
            if not keys:
                probs.append("%s/joblist.tsv %s: model %r (%s) maps to no checkpoint of cluster/weights.json"
                             % (_rel(d), r["id"], m, script))
            for k in keys - set(ck):
                probs.append("%s/joblist.tsv %s: model %r maps to %s, not in cluster/weights.json" % (_rel(d), r["id"], m, k))
    for path in _task_lists():
        for n, jid, cmd in _lines(path):
            for m in re.finditer(r"\$\{?(HF_HOME|WEIGHTS_ROOT|IFM_CKPT_ROOT)\}?/([^\s;&|()<>'\"]+)", cmd):
                var, rest = m.group(1), m.group(2).rstrip("}.,")
                if var == "HF_HOME" and not rest.startswith("hub/models--"):
                    continue
                if var == "HF_HOME":
                    rest = "/".join(rest.split("/")[:4])                      # hub/models--X/snapshots/<rev>
                else:
                    rest = rest.split("/")[0]
                    var = "WEIGHTS_ROOT"
                if "$%s/%s" % (var, rest) not in locs:
                    probs.append("%s:%d (%s): checkpoint $%s/%s is no location in cluster/weights.json"
                                 % (_rel(path), n, jid, m.group(1), rest))
    _report(probs, "checkpoint problems")


# ------------------------------------------------------------------ 4. syntax
def test_shell_and_python_syntax(tmp_path):
    probs = []
    shells = [p for p in sorted(glob.glob(os.path.join(CL, "**", "*"), recursive=True))
              if p.endswith((".sh", ".sbatch")) and os.path.isfile(p)]
    if _bash():
        loop = 'for f in "$@"; do e=$(bash -n "$f" 2>&1) || printf "%s\\t%s\\n" "$f" "$(echo "$e" | head -3 | tr "\\n" " ")"; done'
        r = subprocess.run([_bash(), "-c", loop, "x"] + [p.replace("\\", "/") for p in shells],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for ln in r.stdout.decode("utf-8", "replace").splitlines():
            if "\t" in ln:
                f, e = ln.split("\t", 1)
                probs.append("bash -n %s: %s" % (_rel(f), e.strip()))
    pys = [p for p in sorted(glob.glob(os.path.join(CL, "**", "*.py"), recursive=True))
           if "__pycache__" not in p and os.path.isfile(p)]
    for p in pys:
        rel = _rel(p)
        try:
            py_compile.compile(p, cfile=os.path.join(str(tmp_path), rel.replace("/", "_") + "c"), doraise=True)
        except py_compile.PyCompileError as e:
            probs.append("py_compile %s: %s" % (rel, str(e.msg).strip().splitlines()[-1]))
            continue
        try:                                          # the cluster's interpreter is Python 3.11
            with open(p, "rb") as f:
                ast.parse(f.read(), filename=p, feature_version=(3, 11))
        except SyntaxError as e:
            probs.append("%s:%s: not Python 3.11 syntax: %s" % (rel, e.lineno, e.msg))
    if not shells or not pys:
        probs.append("found %d shell files and %d python files under cluster/" % (len(shells), len(pys)))
    _report(probs, "syntax problems")


# ------------------------------------------------------------------ 5. pins
def test_pins_match_and_every_run_file_is_pinned():
    pins, probs = _pins()
    with open(PINS, "rb") as f:
        if b"\r" in f.read():
            probs.append("jobs/scripts_sha256.txt has CR line endings (sha256sum -c reads the CR into the path)")
    for rel, (sha, n) in sorted(pins.items()):
        p = os.path.join(TOP, rel)
        if not os.path.isfile(p):
            probs.append("jobs/scripts_sha256.txt:%d: %s missing" % (n, rel))
            continue
        with open(p, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != sha:
            probs.append("jobs/scripts_sha256.txt:%d: %s sha256 %s, pinned %s" % (n, rel, got[:12], sha[:12]))
    for p, jid in sorted(_run_closure().items()):
        if _rel(p) not in pins:
            probs.append("%s runs (directly or by import) %s, not in jobs/scripts_sha256.txt" % (jid, _rel(p)))
    _report(probs, "pin problems")


# ------------------------------------------------------------------ 6. line endings
def test_shell_templates_and_lists_are_lf():
    probs = []
    files = [p for p in sorted(glob.glob(os.path.join(CL, "**", "*"), recursive=True))
             if p.endswith((".sh", ".sbatch", ".tsv")) and os.path.isfile(p)]
    for p in files:
        with open(p, "rb") as f:
            data = f.read()
        if b"\r" in data:
            n = data[:data.index(b"\r")].count(b"\n") + 1
            probs.append("%s:%d: CR line ending (%d CR in the file)" % (_rel(p), n, data.count(b"\r")))
    _report(probs, "files that are not LF")


# ------------------------------------------------------------------ 7. the B-EXT6 unit: jobs/submit_ext6.sh
def _ext6(tmp, extra_env=None, args=(), script="submit_ext6.sh"):
    """submit_ext6.sh (or submit_ext6b.sh) with fake Slurm over jobs/nolocal: -> (run dict as the chain fixture builds it, sbatch calls)."""
    fake, log = _fakes(tmp)
    out = tmp / "out"
    (out / "_done").mkdir(parents=True, exist_ok=True)
    (out / "bad_nodes.txt").write_text("n07\n")
    env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(),
               VENV_OURO="/proj/venvs/ouro", CARDS="48", PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""),
               **SLURM_ENV)
    for k in ("MODULES", "ALLOW_HOME", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "EXT6_YES", "FILL_YES", "MAX_NODES",
              "FILL_PRIORITIES", "EXT6B_YES"):
        env.pop(k, None)
    env.update(extra_env or {})
    for k in [k for k, v in env.items() if v is None]:
        env.pop(k)
    r = subprocess.run([_bash(), os.path.join(JOBS, script).replace("\\", "/")] + list(args), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    calls = []
    for k, ln in enumerate(open(log, "rb").read().decode("utf-8").split("\n")[:-1], 1):
        f = ln.split("\x1f")
        opts, pos = _parse_sbatch(f[1:-1] if f and f[-1] == "" else f[1:])
        calls.append({"id": 1000 + k, "cwd": f[0], "args": f[1:-1], "opts": opts, "pos": pos})
    run = {"name": script, "rc": r.returncode, "out": r.stdout.decode("utf-8", "replace"), "calls": calls,
           "first": 1001}
    return run, env


def _call_lines(c):
    """(stage list, line) of every task line an sbatch call runs: a packed call (task_pack.sbatch <list> <pair file>
    <group>) runs the lines its pair file names for its array indices; any other call runs its array indices."""
    if c["pos"] and c["pos"][0] == "task_pack.sbatch":
        rows = [ln.split() for ln in open(c["pos"][2]).read().splitlines()]
        lst = os.path.join(c["cwd"], c["pos"][1])
        return [(lst, int(x)) for _l, i in c.get("tasks", []) for x in rows[i - 1]]
    return c.get("tasks", [])


@needs_bash
@pytest.mark.parametrize("unit", EXT6_UNITS, ids=[u[0] for u in EXT6_UNITS])
def test_submit_ext6_submits_only_its_groups_with_chains_and_merges(tmp_path, unit):
    script, groups, merge_group, yes, n_run, done, n_shards = unit
    rows = {r["id"]: r for r in _joblist(NOLOCAL)}
    e6 = {k: r for k, r in rows.items() if r["group"] in groups}
    merges = {k: r for k, r in rows.items() if r["group"] == merge_group}
    probs = []
    if len(e6) != n_run or len(merges) != 3:
        probs.append("nolocal/joblist.tsv: %d %s run tasks (expected %d) and %d merges (expected 3)"
                     % (len(e6), script, n_run, len(merges)))
    (tmp_path / "out" / "_done").mkdir(parents=True)
    (tmp_path / "out" / "_done" / done).write_text("")
    run, env = _ext6(tmp_path, {yes: "yes"}, script=script)
    if run["rc"] != 0:
        probs.append(script + " exited %d:\n%s" % (run["rc"], run["out"][-2500:]))
    issued, job_tasks, seen = set(), {}, {}
    for c in run["calls"]:
        probs += _check_call(c, run, env, issued)
        if script == "submit_ext6.sh" and ((c["pos"] and c["pos"][0] == "task_pack.sbatch")
                                           or c["opts"].get("--time", [""])[-1] != "10:00:00"):
            probs.append("submit_ext6.sh: call %d is packed or lacks --time=10:00:00: %s" % (c["id"], c["args"][:12]))
        issued.add(c["id"])
        ids = [_lines(lst)[i - 1][1] for lst, i in _call_lines(c)]
        job_tasks[c["id"]] = ids
        for t in ids:
            seen.setdefault(t, []).append(c["id"])
    for k in sorted(set(e6) - {done}):
        if len(seen.get(k, [])) != 1:
            probs.append(script + ": task %s submitted %d times" % (k, len(seen.get(k, []))))
    if done in seen:
        probs.append(script + ": %s has a done marker but was submitted" % done)
    for t in sorted(set(seen) - set(e6) - set(merges)):
        probs.append(script + ": submitted %s, not a B-EXT6 task" % t)
    job_of = {t: j[0] for t, j in seen.items()}
    for c in run["calls"]:
        deps = c["opts"].get("--dependency", [""])[-1]
        ok_ids = {int(x) for cl in deps.split(",") if cl.startswith("afterok:") for x in cl.split(":")[1:]}
        any_ids = {int(x) for cl in deps.split(",") if cl.startswith("afterany:") for x in cl.split(":")[1:]}
        for t in job_tasks[c["id"]]:
            need = {job_of[d] for d in rows[t]["depends_on"].split(",") if d and d in job_of}
            if t in merges:
                if need - any_ids:
                    probs.append(script + ": merge %s does not wait afterany for jobs %s" % (t, sorted(need - any_ids)))
                if "--hold" in c["opts"]:
                    probs.append(script + ": merge %s submitted --hold" % t)
            elif need - ok_ids:
                probs.append(script + ": %s does not wait afterok for jobs %s" % (t, sorted(need - ok_ids)))
    for t in merges:
        if len(seen.get(t, [])) != 1:
            probs.append(script + ": merge %s submitted %d times" % (t, len(seen.get(t, []))))
    per_job = [ln for ln in run["out"].splitlines() if re.search(r"^  g\d+_e6\S+ .*: \d+ task\(s\), [\d.]+ card-hours$", ln)]
    shards = [ln for ln in run["out"].splitlines() if re.search(r"^      shard \w+: [\d.]+ card-hours$", ln)]
    stages = {(r["group"], r["out"], r["stage"]) for k, r in e6.items() if k != done}
    if len(per_job) != len(stages) or len(shards) != n_shards or not re.search(r"^total: \d+ task\(s\), [\d.]+ card-hours", run["out"], re.M):
        probs.append(script + ": card-hours not printed per job (%d lines) and in total:\n%s" % (len(per_job), run["out"][-2000:]))
    # --dry-run and a missing yes submit nothing
    for name, extra, args, rc in (("dry", {}, ["--dry-run"], 0), ("no", {yes: "no"}, [], 1)):
        sub = tmp_path / name
        sub.mkdir()
        r2, _ = _ext6(sub, extra, args, script=script)
        if r2["calls"] or r2["rc"] != rc:
            probs.append(script + " %s: exit %d (expected %d), %d sbatch calls (expected 0)" % (name, r2["rc"], rc, len(r2["calls"])))
    _report(probs, script + " problems")


def test_ext6_wall_limits_and_frozen_plan_shas():
    probs = []
    for d in (JOBS, NOLOCAL):
        for g in E6_GROUPS + (E6_MERGE,) + E6B_GROUPS + (E6B_MERGE,):
            p = os.path.join(d, "task_%s.sbatch" % g)
            if not os.path.isfile(p):
                probs.append("%s missing" % _rel(p))
                continue
            m = re.search(r"^#SBATCH --time=(\d+):(\d\d):(\d\d)$", open(p, encoding="utf-8").read(), re.M)
            if not m or int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) > 10 * 3600:
                probs.append("%s: wall limit %s, over 10 h" % (_rel(p), m.group(0) if m else "missing"))
    shas = {}
    for t in ("ra29", "ra30", "ra31"):
        pd = os.path.join(CL, "pilot", t)
        s = open(os.path.join(pd, "plan_%s.sha256" % t), encoding="utf-8").read().split()[0]
        if hashlib.sha256(open(os.path.join(pd, "plan_%s.md" % t), "rb").read()).hexdigest() != s:
            probs.append("cluster/pilot/%s: plan_%s.md does not match plan_%s.sha256" % (t, t, t))
        shas[t] = s
    for path in glob.glob(os.path.join(NOLOCAL, "g*_e6*.tsv")):
        for n, jid, cmd in _lines(path):
            for t, s in shas.items():
                if "/pilot/%s " % t in cmd + " " and "--plan_sha256" in cmd and "--plan_sha256 %s" % s not in cmd:
                    probs.append("%s:%d (%s): --plan_sha256 is not the frozen %s plan %s" % (_rel(path), n, jid, t, s[:8]))
            if "e6b_ra29_cell.sh" in cmd and not cmd.endswith(" " + shas["ra29"]):
                probs.append("%s:%d (%s): the RA-29 cell is not given the frozen ra29 plan %s" % (_rel(path), n, jid, shas["ra29"][:8]))
    _report(probs, "B-EXT6 template and plan problems")


def test_merge_ext6_copies_scrubs_and_keeps_the_run_tree(tmp_path):
    out = tmp_path / "out"
    cfg = out / "E6" / "ra30" / "ouro_2_6b_base" / "pilot" / "gsm8k" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{"out": "/proj/run/out/E6/ra30"}\n')
    (out / "E6" / "ra30" / "prod_logs").mkdir()
    (out / "E6" / "ra30" / "prod_logs" / "l.txt").write_text("x\n")
    (out / "_done").mkdir()
    (out / "_done" / "g19_e6ra30.s1_pilot.001").write_text("")
    env = dict(os.environ, ALLOW_HOME="1", PROJECT_ROOT="/proj/run")
    r = subprocess.run([sys.executable, "-B", os.path.join(CL, "merge_ext6.py"), "--test", "ra30", "--out_root", str(out),
                        "--joblist", os.path.join(NOLOCAL, "joblist.tsv")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    probs = []
    if r.returncode != 0:
        probs.append("merge_ext6.py exited %d: %s" % (r.returncode, r.stdout.decode("utf-8", "replace")[-1500:]))
    m = out / "merged" / "ra30"
    got = (m / "ouro_2_6b_base" / "pilot" / "gsm8k" / "config.json")
    if not got.exists() or "<PROJECT_ROOT>/out/E6/ra30" not in got.read_text():
        probs.append("merged/ra30 config.json missing or not scrubbed")
    if "/proj/run/" not in cfg.read_text():
        probs.append("the run tree was changed by the merge")
    if (m / "prod_logs").exists():
        probs.append("prod_logs/ copied into merged/ra30")
    rep = json.load(open(str(m / "MERGE.json"), encoding="utf-8")) if (m / "MERGE.json").exists() else {}
    states = {t["id"]: t["state"] for t in rep.get("tasks", [])}
    if states != {"g19_e6ra30.s1_pilot.001": "done", "g19_e6ra30.s2_full.001": "not run",
                  "g26_e6bra30.s1_pilot.001": "not run", "g26_e6bra30.s2_full.001": "not run"}:
        probs.append("MERGE.json task states %s" % states)
    left = [x for x in os.listdir(str(out / "merged")) if x != "ra30"]
    if left:
        probs.append("merged/ keeps the work folders %s" % left)
    _report(probs, "merge_ext6 problems")


def test_ext6_partb_shards_are_the_folder_shards():
    lines = _lines(os.path.join(NOLOCAL, "g22_e6ra31b.s1_shard.tsv"))
    names = [re.search(r"--shard (\w+)", c).group(1) for _, _, c in lines]
    r = subprocess.run([sys.executable, "-B", os.path.join(CL, "pilot", "ra31", "ra31_gen.py"), "list"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        pytest.skip("ra31_gen.py list does not run here (torch): %s" % r.stdout.decode("utf-8", "replace")[-300:])
    listed = re.findall(r"^(\w+)\s+\d+ arms", r.stdout.decode("utf-8", "replace"), re.M)
    listed = [x for x in listed if x != "total"]
    assert sorted(names) == sorted(listed) and len(names) == 15, (names, listed)


# ------------------------------------------------------------------ 8. nodes, packing, time (PI rules 2026-10-12)
CAP_SCRIPTS = ["submit_fill.sh"]        # stage 2 (submit_ext6.sh runs one line per GPU, no node logic)
CAP_SCRIPTS = CAP_SCRIPTS + (["submit_ext6b.sh"] if os.path.exists(os.path.join(JOBS, "submit_ext6b.sh")) else [])
CAP_YES = {"submit_ext6.sh": "EXT6_YES", "submit_fill.sh": "FILL_YES", "submit_ext6b.sh": "EXT6B_YES"}
SIX = ["n01", "n02", "n03", "n04", "n05", "n06"]


def _cap_run(tmp, script, extra):
    (tmp / "out" / "_done").mkdir(parents=True, exist_ok=True)
    (tmp / "out" / "_done" / "g0_xllm.s1_build.001").write_text("")      # the fill's IFM lines wait for the xllm build
    env = {CAP_YES[script]: "yes", "CARDS": None}
    env.update(extra)
    return _ext6(tmp, env, script=script)


@needs_bash
@pytest.mark.parametrize("script", CAP_SCRIPTS)
def test_submit_scripts_reserve_the_last_node_pack_and_time(tmp_path, script):
    """A six-node NODELIST: every sbatch call excludes exactly the last node among the six (n06) and never a used one;
    the arrays are throttled to used nodes x 8 = 40 GPU jobs; every call passes --time=10:00:00; every packed job
    carries two lines except at most one single line per packed call (an odd count); every task line is submitted
    once; the summary names the used and reserved nodes and the concurrent task lines (CARDS x 2)."""
    run, _ = _cap_run(tmp_path, script, {"NODELIST": ",".join(SIX)})
    probs = [] if run["rc"] == 0 else ["%s exited %d:\n%s" % (script, run["rc"], run["out"][-2000:])]
    if not run["calls"]:
        probs.append("%s: no sbatch call" % script)
    if "used n01,n02,n03,n04,n05" not in run["out"] or "reserved and empty: n06" not in run["out"] \
            or "up to 80 task lines at once" not in run["out"]:
        probs.append("%s: the summary does not name the used and reserved nodes and 80 task lines:\n%s" % (script, run["out"][-1500:]))
    n_pack = 0
    for c in run["calls"]:
        exc = set(",".join(c["opts"].get("--exclude", [])).split(",")) - {""}
        if exc & set(SIX) != {"n06"}:
            probs.append("%s: sbatch call %d excludes %s of the six (want n06 only)" % (script, c["id"], sorted(exc & set(SIX))))
        if c["opts"].get("--time", [""])[-1] != "10:00:00":
            probs.append("%s: sbatch call %d --time=%s, not 10:00:00" % (script, c["id"], c["opts"].get("--time")))
        arr = c["opts"].get("--array", [""])[-1]
        if not (c["pos"] and "merge" in c["pos"][0]) and not arr.endswith("%40"):
            probs.append("%s: sbatch call %d --array=%s, not throttled to %%40" % (script, c["id"], arr))
        if c["pos"] and c["pos"][0] == "task_pack.sbatch":
            n_pack += 1
            rows = [ln.split() for ln in open(c["pos"][2]).read().splitlines()]
            if any(len(r) != 2 for r in rows[:-1]) or len(rows[-1]) not in (1, 2):
                probs.append("%s: packed call %d pair rows %s (two lines each, one single at most, last)" % (script, c["id"], rows[:5]))
    if not n_pack:
        probs.append("%s: no packed job" % script)
    _report(probs, "node, packing and time problems")


@needs_bash
@pytest.mark.parametrize("script", CAP_SCRIPTS)
def test_submit_scripts_refuse_without_a_node_list(tmp_path, script):
    run, _ = _cap_run(tmp_path, script, {"NODELIST": ""})
    assert run["rc"] == 2 and not run["calls"] and "no node list" in run["out"], (run["rc"], run["out"][-1500:])


@needs_bash
def test_submit_fill_priorities_select_groups(tmp_path):
    """FILL_PRIORITIES=1: only the priority-1 group (g17_fillmap) is submitted and only priority 1 is priced."""
    run, _ = _cap_run(tmp_path, "submit_fill.sh", {"FILL_PRIORITIES": "1", "NODELIST": ",".join(SIX)})
    groups = {os.path.basename(c["pos"][1]).split(".")[0] for c in run["calls"] if len(c["pos"]) > 1}
    assert run["rc"] == 0 and groups == {"g17_fillmap"}, (run["rc"], groups, run["out"][-2000:])
    assert re.search(r"^priority 1: \d+ tasks", run["out"], re.M) and "priority 2:" not in run["out"] \
        and "priority 3:" not in run["out"] and re.search(r"^total: \d+ tasks", run["out"], re.M), run["out"][-2000:]


def _pack_env(tmp, cmd1, restart=0):
    """task_pack.sbatch of group g20_e6ra29 on a two-line list (ids of that group), fake nvidia-smi and scontrol."""
    out = tmp / "out"
    fake = tmp / "bin"
    fake.mkdir(exist_ok=True)
    slog = (tmp / "scontrol.log").as_posix()
    (fake / "nvidia-smi").write_bytes(b'#!/bin/bash\necho "GPU 0: card"\n')
    (fake / "scontrol").write_bytes(('#!/bin/bash\necho "$*" >> "%s"\n' % slog).encode())
    for f in ("nvidia-smi", "scontrol"):
        os.chmod(str(fake / f), 0o755)
    o = out.as_posix()
    lst = tmp / "list.tsv"
    lst.write_bytes(("g20_e6ra29.s1_pilot.001\t%s\n"
                     "g20_e6ra29.s1_pilot.002\techo $E6_MEM_FRACTION $GEN_MEM_FRACTION $PROD_MEM_FRACTION > %s/f2; echo x >> %s/c2\n"
                     % (cmd1, o, o)).encode())
    pairs = tmp / "pairs.tsv"
    pairs.write_bytes(b"1 2\n")
    (tmp / "home").mkdir(exist_ok=True)
    env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=o, HOME=(tmp / "home").as_posix(), SLURM_SUBMIT_DIR=NOLOCAL.replace("\\", "/"),
               SLURM_ARRAY_TASK_ID="1", SLURM_ARRAY_JOB_ID="500", SLURM_JOB_ID="501", SLURMD_NODENAME="n03",
               SLURM_RESTRICT="--exclude=n06", SLURM_RESTART_COUNT=str(restart),
               PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for v in ("HF_HOME", "SETTLE", "WEIGHTS_ROOT", "IFM_CKPT_ROOT", "XLLM_ROOT", "DATA_ROOT", "MODULES", "E6_MEM_FRACTION",
              "GEN_MEM_FRACTION", "PROD_MEM_FRACTION", "ALLOW_HOME"):
        env.pop(v, None)
    r = subprocess.run([_bash(), os.path.join(NOLOCAL, "task_pack.sbatch").replace("\\", "/"), lst.as_posix(), pairs.as_posix(),
                        "g20_e6ra29"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    calls = open(slog).read().splitlines() if os.path.exists(slog) else []
    return r, out, calls


@needs_bash
def test_task_pack_halves_fractions_and_requeue_skips_the_done_line(tmp_path):
    """Two lines in one packed job: both run with every share-of-the-card setting halved and leave their done markers.
    A line whose GPU does not start (exit 75) requeues the job once with the node excluded and writes the packed
    diagnosis naming both lines; the other line's done marker stands, and the rerun runs only the line not done."""
    ok = "echo $E6_MEM_FRACTION $GEN_MEM_FRACTION $PROD_MEM_FRACTION > %s/f1; echo x >> %s/c1"
    a = tmp_path / "a"
    a.mkdir()
    r, out, calls = _pack_env(a, ok % ((a / "out").as_posix(), (a / "out").as_posix()))
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    for k in ("1", "2"):
        assert (out / ("f" + k)).read_text().split() == ["0.45", "0.45", "0.425"], so[-1500:]
        assert (out / "_done" / ("g20_e6ra29.s1_pilot.00" + k)).exists()
    b = tmp_path / "b"
    b.mkdir()
    r, out, calls = _pack_env(b, 'echo "RuntimeError: CUDA unknown error"; exit 1')
    so = r.stdout.decode()
    assert r.returncode == 75, so[-2000:]
    assert calls == ["update JobId=500_1 ExcNodeList=n03,n06", "requeue 500_1"], calls
    dg = (out / "_diag" / "pack_500_1.txt").read_text()
    assert "g20_e6ra29.s1_pilot.001: exit 75" in dg and "g20_e6ra29.s1_pilot.002: exit 0" in dg, dg
    assert (out / "_done" / "g20_e6ra29.s1_pilot.002").exists() and not (out / "_done" / "g20_e6ra29.s1_pilot.001").exists()
    o = out.as_posix()
    r, out, calls2 = _pack_env(b, ok % (o, o), restart=1)                # the requeued run, the GPU now starts
    so = r.stdout.decode()
    assert r.returncode == 0 and (out / "c2").read_text().split() == ["x"] and (out / "c1").read_text().split() == ["x"], so[-2000:]
    assert "done already: g20_e6ra29.s1_pilot.002" in so and (out / "_done" / "g20_e6ra29.s1_pilot.001").exists()
