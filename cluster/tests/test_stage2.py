"""Stage 2 (push 2) checks: the pushed EXT6 pins stay byte-identical; jobs/submit_ext6c.sh (the e6c job list of
cluster/gen_ext6c.py) with fake Slurm; PLAN_IDS in submit_fill.sh, submit_ext6b.sh and submit_ext6c.sh; the McLeish
shard wrapper. The tests read the tree and write only to temporary folders.

    cd loopedcot; python -B -m pytest cluster/tests/test_stage2.py -q -p no:cacheprovider
"""
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.dirname(HERE)
TOP = os.path.dirname(os.path.dirname(CL))
JOBS = os.path.join(CL, "jobs")
NOLOCAL = os.path.join(JOBS, "nolocal")
SIX = "n01,n02,n03,n04,n05,n06"


def _bash():
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


def test_pushed_ext6_pins_stay_byte_identical():
    """Every line of jobs/pins_ext6_pushed.txt (the pins of the pushed EXT6 files) is a line of jobs/scripts_sha256.txt,
    and its file still has that sha256, so a requeued EXT6 task passes its pin check after push 2."""
    pins = open(os.path.join(JOBS, "scripts_sha256.txt"), "rb").read().decode("utf-8").splitlines()
    frozen = open(os.path.join(JOBS, "pins_ext6_pushed.txt"), "rb").read().decode("utf-8").splitlines()
    assert len(frozen) >= 200
    missing = [ln for ln in frozen if ln not in set(pins)]
    assert not missing, missing[:5]
    for ln in frozen:
        h, p = ln.split("  ", 1)
        assert hashlib.sha256(open(os.path.join(TOP, p), "rb").read()).hexdigest() == h, p


def _fakes(tmp, queue=""):
    fake = tmp / "bin"
    fake.mkdir(parents=True, exist_ok=True)
    log = (tmp / "sbatch.log").as_posix()
    (fake / "sbatch").write_bytes(('#!/bin/bash\necho "$*" >> "%s"\nn=$(grep -c . "%s")\necho $((1000 + n))\n' % (log, log)).encode())
    (fake / "squeue").write_bytes(('#!/bin/bash\ncase "$*" in *"%%F %%K"*) printf "%s";; esac\nexit 0\n' % queue).encode())
    (fake / "scontrol").write_bytes(b'#!/bin/bash\ncase "$1 $2" in "show hostnames") echo "$3" | tr "," "\\n";; esac\nexit 0\n')
    for n in ("sbatch", "squeue", "scontrol"):
        os.chmod(str(fake / n), 0o755)
    return fake, log


def _run(tmp, script, extra=None, args=(), queue="", sub=""):
    fake, log = _fakes(tmp, queue)
    out = tmp / "out"
    (out / "_done").mkdir(parents=True, exist_ok=True)
    (out / "_chain").mkdir(exist_ok=True)
    (out / "_done" / "g0_xllm.s1_build.001").write_text("")
    if sub:
        (out / "_chain" / "submitted.tsv").write_text(sub)
    env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(),
               PARTITION="gpuq", ACCOUNT="acct1", SLURM_RESTRICT="--exclude=n08", NODELIST=SIX, USER="op",
               PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for k in ("CARDS", "PLAN_IDS", "PACK", "EXT6_YES", "EXT6B_YES", "EXT6C_YES", "FILL_YES", "FILL_PRIORITIES", "MODULES"):
        env.pop(k, None)
    env.update(extra or {})
    r = subprocess.run([_bash(), os.path.join(JOBS, script).replace("\\", "/")] + list(args), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    calls = open(log).read().splitlines() if os.path.exists(log) else []
    return r.returncode, r.stdout.decode("utf-8", "replace"), calls, out


def _rows(name):
    with open(os.path.join(NOLOCAL, name), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _line(lst, tid):
    return [ln.split("\t")[0] for ln in open(os.path.join(NOLOCAL, lst), encoding="utf-8")].index(tid) + 1


@needs_bash
def test_submit_ext6c_chains_blocks_and_waits_on_other_units(tmp_path):
    """No marker of the other units: the McLeish pipeline is submitted (pilot, then the shard lines afterok, then the
    summary afterok), the four flags lines are named and not submitted, the merge waits afterany for every new job;
    every call excludes the reserved node and passes --time=10:00:00. With the g19 pilot done and the g19 full run
    in the queue (job 7001), the GSM8K pilot flags line waits for nothing and the GSM8K full flags line waits afterok
    on 7001."""
    rows = _rows("joblist_ext6c.tsv")
    assert {r["group"] for r in rows} == {"g29_e6cra30m", "g30_e6cra30f", "g32_e6cmerge", "g33_e6cra31b14"}   # 2.6B Part B: push 3
    rc, so, calls, out = _run(tmp_path / "a", "submit_ext6c.sh", {"EXT6C_YES": "yes"})
    assert rc == 0, so[-2500:]
    assert so.count("NOT SUBMITTED") == 4, so[-2500:]                    # the flags (no g19 or g26 marker)
    assert all("--time=10:00:00" in c and "--exclude=n08,n06" in c for c in calls), calls
    run = [c for c in calls if "g29_e6cra30m" in c]
    assert len(run) == 3 and "--dependency" not in run[0], run
    assert "--dependency=afterok:1001" in run[1] and "--dependency=afterok:1002" in run[2], run
    assert not [c for c in calls if "g30_e6cra30f" in c]
    # 1.4B Part B: per cell its 4 independent shards (no dependency), then cell_summarize afterany on their jobs
    pb = [(1001 + k, c) for k, c in enumerate(calls) if "g33_e6cra31b14" in c]
    shard_jobs = [j for j, c in pb if "s1_shard" in c]
    summ = [c for j, c in pb if "s2_summarize_any" in c]
    assert len(summ) == 3 and shard_jobs and all("--dependency" not in c for j, c in pb if "s1_shard" in c), pb
    assert all(re.search(r"--dependency=afterany:\d+", c) for c in summ), summ
    assert sorted(int(x) for c in summ for x in c.split("--dependency=afterany:")[1].split()[0].split(":")) == sorted(shard_jobs)
    merge = [c for c in calls if "g32_e6cmerge" in c]
    others = [str(1001 + k) for k, c in enumerate(calls) if "g32_e6cmerge" not in c]
    assert len(merge) == 1 and merge[0].split("--dependency=afterany:")[1].split()[0].split(":") == others, merge
    # the other unit: g19 pilot done, g19 full queued as job 7001
    full_line = _line("g19_e6ra30.s2_full.tsv", "g19_e6ra30.s2_full.001")
    b = tmp_path / "b"
    b.mkdir()
    (b / "out" / "_done").mkdir(parents=True)
    (b / "out" / "_done" / "g19_e6ra30.s1_pilot.001").write_text("")
    rc, so, calls, out = _run(b, "submit_ext6c.sh", {"EXT6C_YES": "yes"}, queue="7001 %d\n" % full_line,
                              sub="7001 g19_e6ra30.s2_full.tsv %d\n" % full_line)
    assert rc == 0, so[-2500:]
    fl = [c for c in calls if "g30_e6cra30f" in c]
    assert len(fl) == 2 and "--dependency" not in fl[0] and "--dependency=afterok:7001" in fl[1], fl
    assert so.count("NOT SUBMITTED") == 2, so[-2500:]


@needs_bash
@pytest.mark.parametrize("script,listname,yes", [("submit_ext6c.sh", "joblist_ext6c.tsv", "EXT6C_YES"),
                                                 ("submit_ext6b.sh", "joblist.tsv", "EXT6B_YES"),
                                                 ("submit_fill.sh", "joblist.tsv", "FILL_YES")])
def test_plan_ids_select_only_the_listed_tasks(tmp_path, script, listname, yes):
    """PLAN_IDS=<file>: only the listed task ids are planned, priced and submitted; a missing file stops with exit 2."""
    groups = {"submit_ext6c.sh": ("g29_e6cra30m",), "submit_ext6b.sh": ("g24_e6bra29",), "submit_fill.sh": ("g17_fillmap",)}[script]
    ids = [r["id"] for r in _rows(listname) if r["group"] in groups and r["stage"].startswith("s1")][:2]
    f = tmp_path / "ids.txt"
    f.write_text("\n".join(ids) + "\n")
    rc, so, calls, out = _run(tmp_path / "a", script, {yes: "yes", "PLAN_IDS": f.as_posix()})
    assert rc == 0, so[-2500:]
    m = re.search(r"^total: (\d+) task", so, re.M)
    assert m and int(m.group(1)) == len(ids), so[-2000:]
    sub = set()
    for c in calls:
        lst = re.search(r" (g\d+_\w+\.s\d+_\w+\.tsv)", c)
        pf = re.search(r"task_pack\.sbatch \S+ (\S+) ", c + " ")
        if pf:
            lines = [int(x) for ln in open(pf.group(1)).read().split("\n") if ln for x in ln.split()]
        else:
            lines = [int(x) for x in re.search(r"--array=(\S+)", c).group(1).split("%")[0].split(",") if "-" not in x] if "--array=" in c else []
        if lst and "merge" not in lst.group(1):
            ls = [ln.split("\t")[0] for ln in open(os.path.join(NOLOCAL, lst.group(1)), encoding="utf-8")]
            sub |= {ls[i - 1] for i in lines}
    assert sub == set(ids), (sorted(sub), ids)
    rc, so, calls, out = _run(tmp_path / "b", script, {yes: "yes", "PLAN_IDS": (tmp_path / "none.txt").as_posix()})
    assert rc == 2 and not calls, so[-1500:]


@needs_bash
def test_mcleish_shard_wrapper_reads_the_pilot_shard_count(tmp_path):
    """e6c_ra30_mcl_shard.sh: a shard line at or above the pilot's N exits 0 without work; N above the submitted
    lines stops shard 0 with exit 3; no pilot checks.json: exit 2."""
    o = tmp_path / "o"
    c = o / "mcleish_llama32_r32" / "pilot" / "gsm8k"
    sh = os.path.join(CL, "pilot", "e6c_ra30_mcl_shard.sh").replace("\\", "/")
    env = dict(os.environ, PY=sys.executable.replace("\\", "/"))
    r = subprocess.run([_bash(), sh, "4", o.as_posix(), "x" * 64, "6"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 2
    c.mkdir(parents=True)
    (c / "checks.json").write_text('{"shards_rule": {"projected_full_card_hours": 12.0, "shards": 3}}')
    r = subprocess.run([_bash(), sh, "4", o.as_posix(), "x" * 64, "6"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0 and b"not needed" in r.stdout
    (c / "checks.json").write_text('{"shards_rule": {"projected_full_card_hours": 40.0, "shards": 8}}')
    r = subprocess.run([_bash(), sh, "0", o.as_posix(), "x" * 64, "6"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 3
    r = subprocess.run([_bash(), sh, "2", o.as_posix(), "x" * 64, "6"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0


def test_ext6c_lines_name_the_frozen_ra30_s2_plan():
    s = open(os.path.join(CL, "pilot", "ra30_s2", "plan_ra30.sha256"), encoding="utf-8").read().split()[0]
    assert hashlib.sha256(open(os.path.join(CL, "pilot", "ra30_s2", "plan_ra30.md"), "rb").read()).hexdigest() == s
    for r in _rows("joblist_ext6c.tsv"):
        if r["group"] == "g29_e6cra30m" and r["stage"] in ("s1_pilot", "s2_shard"):
            assert s in r["cmd"], r["id"]


# the gate's forbidden strings, built from pieces so that this file does not match itself
_W = ["cl" + "aude", "op" + "us", "son" + "net", "hai" + "ku", "anthr" + "opic"]
# hosts, the user (his name is also his home folder: /home/<user>, C:\Users\<user>) and his mail; the generic
# /home/user0-style paths of the scrub-test fixtures are not a home of anyone
_HOST = ["dg" + "x1", "rtx" + "6000", "sp" + "ark[0-9]", "utd" + "allas", "cs0" + "97715", "sas" + "11", "sab" + "bir",
         "@g" + "mail"]
SCAN_EXT = (".py", ".sh", ".md", ".txt", ".tsv", ".sbatch", ".yaml", ".cfg", ".toml")
# not scanned: dataset JSON and JSONL (CounterFact and MMLU use the words as proper nouns). One known hit is allowed,
# keyed to its file's sha256: line 22 of the frozen k17_varbench plan copy (k17_varbench checks the plan's sha256 at
# launch, so the copy is not edited in push 2). Any other hit, or that file with another sha256, fails.
SCAN_SKIP_DIRS = ("__pycache__", ".pytest_cache")
SCAN_SKIP = ()
SCAN_ALLOW = {("cluster/plans/k17_varbench_plan.md", 22): "10beefdc3992c9d637d7b2add0f2963c1cafd5357efe9fd58e4bfffd612a9f1f",
              # k17_varbench.py checks its own sha256 against its plan at launch and phase M against phase W: not edited
              ("cluster/pilot/k17_varbench.py", 169): "43808060ac3948b6c3c0cd91f0edc06bacfe4254f982ece05c0253dca9f2c0d1",
              # the pushed EXT6 RA-31 copy (plan 3ee4139a; its jobs exit 2 on a plan mismatch): not edited
              ("cluster/pilot/ra31/plan_ra31.md", 169): "3ee4139ade5f7fdbaf2c76c1c451a9652e4aa7d2f70ccaba0c1baae2e5957658"}
_WARN = ["cri" + "tic", "ag" + "ent"]           # reported as warnings, not failures


def test_no_host_user_home_mail_or_assistant_strings():
    """No host, user, home or e-mail string and no assistant or model-vendor word (whole word, any case) in any
    .py, .sh, .md, .txt, .tsv, .sbatch, .yaml, .cfg or .toml file under cluster/."""
    pat = re.compile(r"\b(%s)\b|%s" % ("|".join(_W), "|".join(_HOST)), re.I)
    warn = re.compile(r"\b(%s)\b" % "|".join(_WARN), re.I)
    hits, allowed, warned = [], [], []
    for root, dirs, files in os.walk(CL):
        dirs[:] = [d for d in dirs if d not in SCAN_SKIP_DIRS]
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, os.path.dirname(CL))
            if not f.endswith(SCAN_EXT) or rel.startswith(SCAN_SKIP):
                continue
            for n, ln in enumerate(open(p, "rb").read().decode("utf-8", "replace").splitlines(), 1):
                m, mw = pat.search(ln), warn.search(ln)
                if not (m or mw):
                    continue
                key = (rel.replace(os.sep, "/"), n)
                if key in SCAN_ALLOW and hashlib.sha256(open(p, "rb").read()).hexdigest() == SCAN_ALLOW[key]:
                    allowed.append("%s:%d: %r" % (key[0], n, (m or mw).group(0)))
                elif m:
                    hits.append("%s:%d: %r" % (key[0], n, m.group(0)))
                else:
                    warned.append("%s:%d: %r" % (key[0], n, mw.group(0)))
    print("scan: %d failures, %d allow-listed (%s), %d warnings" % (len(hits), len(allowed), "; ".join(allowed), len(warned)))
    for w in warned:
        print("warning: " + w)
    assert not hits, hits[:20]
    assert len(allowed) == len(SCAN_ALLOW), allowed           # every allowed line still exists, its file still pinned
