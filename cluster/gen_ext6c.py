"""B-EXT6c job list (stage 2, push 2): new groups in new files only; the pushed EXT6 lists stay as they are.

    python cluster/gen_ext6c.py            # writes jobs/joblist_ext6c.tsv, jobs/nolocal/joblist_ext6c.tsv, the stage
                                           # lists and templates of the e6c groups, and appends their pins

Groups (job list jobs/[nolocal/]joblist_ext6c.tsv, submitted by jobs/submit_ext6c.sh):
  g29_e6cra30m  RA-30 McLeish r32 GSM8K (plan 086581bf, pilot/ra30_s2): s1_pilot (CPU tests, tokens_mcl, pilot_mcl,
                summary), s2_shard (NMAX shard lines, each reads N from the pilot: cluster/pilot/e6c_ra30_mcl_shard.sh),
                s3_summarize (summarize_mcl full)
  g30_e6cra30f  RA-30 operand flags (ra30_flags.py, CPU, plan 6d) on the cluster's Ouro-2.6B outputs: pilot and full
                of GSM8K (after g19_e6ra30) and of ASDiv (after g26_e6bra30)
  g32_e6cmerge  merge_ext6.py --test ra30 after the e6c jobs (afterany), so merged/ra30 holds the McLeish rows and flags;
                submit_ext6c.sh removes its done marker before each submission, so it reruns every round
Card-hours: the single-card rate (the builder's A100 figures at 0.6 x the card, times 0.6).
Pins: jobs/scripts_sha256.txt keeps every existing line byte-identical; the files of these groups are appended (a
file already pinned with another sha stops the generator: the pushed EXT6 pins never change).
"""
import csv
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cluster import gen_jobs as GJ          # noqa: E402  (the task template, read only)

JOBS = os.path.join(HERE, "jobs")
TOP = os.path.dirname(os.path.dirname(HERE))
PY, LF = GJ.PY, GJ.LF
COLS = ["id", "group", "stage", "script", "model", "datasets", "shard", "out", "depends_on", "est_card_seconds",
        "rate_source", "cmd"]
MCL_NMAX = 6          # shard lines submitted; the pilot's rule asks 3, or ceil(projected / 5.4) up to 6 (32.4 card-h)
TIME_H = 10
MERGE_GROUP = "g32_e6cmerge"
PARTB = True          # RA-31 Part B grid: the 1.4B cells (push 2); the 2.6B cells are push 3


def sha_of(test_dir, test="ra30"):
    d = os.path.join(HERE, "pilot", test_dir)
    s = open(os.path.join(d, "plan_%s.sha256" % test), encoding="utf-8").read().split()[0]
    if hashlib.sha256(open(os.path.join(d, "plan_%s.md" % test), "rb").read()).hexdigest() != s:
        raise SystemExit("cluster/pilot/%s: plan_%s.md does not match plan_%s.sha256" % (test_dir, test, test))
    return s


def partb_cells():
    """[(cell, [(shard, single-card hours)])] from `ra31_gen.py cell_list` of pilot/ra31_s2 (the builder's table)."""
    import re
    import subprocess
    d = os.path.join(HERE, "pilot", "ra31_s2")
    out = subprocess.check_output([sys.executable, "-B", "ra31_gen.py", "cell_list"], cwd=d).decode("utf-8")
    cells = {}
    for ln in out.splitlines():
        m = re.match(r"^((\w+?_\d\.\db)_(all|lc|top|rand))\s+\d+ arms(?: \+ gate)?\s+([\d.]+) h single card", ln)
        if m:
            cells.setdefault(m.group(2), []).append((m.group(1), float(m.group(4))))
    order = ["svamp_1.4b", "asdiv_1.4b", "mawps_1.4b"]           # the 1.4B cells (push 2); 2.6B: push 3
    if any(c not in cells or sorted(x for x, _h in cells[c]) != sorted(c + "_" + k for k in ("all", "lc", "top", "rand"))
           for c in order):
        raise SystemExit("ra31_gen.py cell_list: unexpected 1.4B cells %s" % {c: cells.get(c) for c in order})
    return [(c, cells[c]) for c in order]


class L(object):
    def __init__(self):
        self.rows = []

    def add(self, group, stage, script, model, datasets, shard, out, deps, est, src, cmd):
        jid = "%s.%s.%03d" % (group, stage, sum(1 for r in self.rows if r["group"] == group and r["stage"] == stage) + 1)
        self.rows.append({"id": jid, "group": group, "stage": stage, "script": script, "model": model,
                          "datasets": ",".join(datasets), "shard": shard, "out": out, "depends_on": ",".join(deps),
                          "est_card_seconds": int(round(est)), "rate_source": src,
                          "cmd": "source $VENV_OURO/bin/activate && " + cmd})
        return jid


def build():
    J = L()
    # ---- RA-30 McLeish r32 GSM8K (plan 086581bf section 6c)
    s = sha_of("ra30_s2")
    o = "$OUT_ROOT/E6/ra30"
    w = "cd %s/ra30_s2 && PROD_ART=%s/prod_art_mcl PROD_LOGS=%s/prod_logs_mcl " % (LF, o, o)
    m = "mcleish_llama32_r32"
    p = J.add("g29_e6cra30m", "s1_pilot", "ra30_mcl.py", m, ["gsm8k"], "", o, [], 3.7 * 0.6 * 3600 + 120,
              "RA-30 builder: pilot about 3.7 A100-h at 0.6 x the card = 2.2 h at the single-card rate, plus tests",
              "%sPROD_DEV=cpu %s -m unittest discover -s tests && %sCUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s ra30_mcl.py --tokens --out %s"
              " && %s%s ra30_mcl.py --run --stage pilot --out %s --plan_sha256 %s"
              " && %s%s ra30_mcl.py --summarize --out %s/%s/pilot/gsm8k" % (w, PY, w, PY, o, w, PY, o, s, w, PY, o, m))
    shards = [J.add("g29_e6cra30m", "s2_shard", "ra30_mcl.py", m, ["gsm8k"], str(k), o, [p],
                    4.6 * 0.6 * 3600 if k < 3 else 60,
                    "RA-30 builder: each full shard about 4.6 A100-h at 0.6 x = 2.8 h at the single-card rate"
                    + ("" if k < 3 else "; shard line beyond the default 3: exits 0 unless the pilot asks for it"),
                    "cd %s && PROD_ART=%s/prod_art_mcl PROD_LOGS=%s/prod_logs_mcl bash e6c_ra30_mcl_shard.sh %d %s %s %d"
                    % (LF, o, o, k, o, s, MCL_NMAX)) for k in range(MCL_NMAX)]
    summ = J.add("g29_e6cra30m", "s3_summarize", "ra30_mcl.py", m, ["gsm8k"], "", o, shards, 300, "CPU",
                 "%s%s ra30_mcl.py --summarize --out %s/%s/full/gsm8k" % (w, PY, o, m))
    # ---- RA-30 operand flags on the Ouro-2.6B outputs (plan 6d, CPU; flags2 of run.sh)
    flags = []
    for ds, grp, pil, full in (("gsm8k", "g19_e6ra30", "g19_e6ra30.s1_pilot.001", "g19_e6ra30.s2_full.001"),
                               ("asdiv", "g26_e6bra30", "g26_e6bra30.s1_pilot.001", "g26_e6bra30.s2_full.001")):
        for st, dep in (("pilot", pil), ("full", full)):
            flags.append(J.add("g30_e6cra30f", "s1_flags", "ra30_flags.py", "ouro_2_6b_base", [ds], st,
                               "%s/ouro_2_6b_base/%s/%s" % (o, st, ds), [dep],
                               300, "CPU (flags2 %s %s)" % (st, ds),
                               "cd %s/ra30_s2 && CUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s ra30_flags.py --out %s/ouro_2_6b_base/%s/%s"
                               % (LF, PY, o, st, ds)))
    # ---- merge: merged/ra30 again after the e6c jobs
    J.add(MERGE_GROUP, "s1_merge", "merge_ext6.py", "all", ["ra30"], "ra30", "$OUT_ROOT/merged/ra30",
          [p] + shards + [summ] + flags, 300, "CPU",
          "%s $REPO/loopedcot/cluster/merge_ext6.py --test ra30 --out_root $OUT_ROOT" % PY)
    # ---- RA-31 Part B grid, 1.4B (push 2; plan 36eb4fea section 15, pilot/ra31_s2; RUNSHEET section 9): 12 independent
    # shards (svamp, asdiv, mawps x all, lc, top, rand; no order among them), each with the folder's CPU tests first;
    # then per cell cell_summarize (stage s2_summarize_any: afterany on its 4 shards). The MAWPS cells read the round's
    # pool (--pool_root $OUT_ROOT). The 2.6B cells (behind freeze26) are push 3.
    s31 = sha_of("ra31_s2", "ra31")
    o31 = "$OUT_ROOT/E6/ra31"
    w31 = "cd %s/ra31_s2 && " % LF
    ids31 = []
    for cell, shs in partb_cells():
        if not cell.endswith("1.4b"):
            continue
        ds = cell.split("_")[0]
        oc = "%s/partb_%s" % (o31, cell)
        env = "PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs " % (oc, oc)
        extra = " --pool_root $OUT_ROOT" if ds == "mawps" else ""
        sh_ids = [J.add("g33_e6cra31b14", "s1_shard", "ra31_gen.py", "ouro_1_4b_base", [ds, "partb"], sh, oc, [],
                        h * 3600 + 180, "ra31_gen.py cell_list (plan %s): %.2f h at the single-card rate, plus the CPU tests"
                        % (s31[:8], h),
                        "%s%s -m unittest discover -s tests && %s%s%s ra31_gen.py cell --cell %s --shard %s --out %s%s"
                        " --mem_fraction $GEN_MEM_FRACTION" % (w31, PY, w31, env, PY, cell, sh, oc, extra))
                  for sh, h in shs]
        sm = J.add("g33_e6cra31b14", "s2_summarize_any", "ra31_gen.py", "ouro_1_4b_base", [ds, "partb"], "", oc, sh_ids,
                   300, "CPU (cell_summarize, afterany on the 4 shards)",
                   "%s%s%s ra31_gen.py cell_summarize --cell %s --out %s%s" % (w31, env, PY, cell, oc, extra))
        ids31 += sh_ids + [sm]
    J.add(MERGE_GROUP, "s1_merge", "merge_ext6.py", "all", ["ra31"], "ra31", "$OUT_ROOT/merged/ra31", ids31, 300, "CPU",
          "%s $REPO/loopedcot/cluster/merge_ext6.py --test ra31 --out_root $OUT_ROOT" % PY)
    return J


def write(J, out_dir):
    import glob
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "g*_e6c*.s*.tsv")) + glob.glob(os.path.join(out_dir, "task_g*_e6c*.sbatch")):
        os.remove(f)                     # the e6c lists and templates of an earlier generation
    with open(os.path.join(out_dir, "joblist_ext6c.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in J.rows:
            w.writerow(r)
    stages = {}
    for r in J.rows:
        stages.setdefault((r["group"], r["stage"]), []).append(r)
    for (g, s), rs in stages.items():
        with open(os.path.join(out_dir, "%s.%s.tsv" % (g, s)), "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write("%s\t%s\n" % (r["id"], r["cmd"]))
    for g in sorted({r["group"] for r in J.rows}):
        txt = GJ.SBATCH.format(group=g, time="%d:00:00" % TIME_H)
        open(os.path.join(out_dir, "task_%s.sbatch" % g), "w", encoding="utf-8", newline="\n").write(txt)


PIN_FILES = ["loopedcot/cluster/gen_ext6c.py", "loopedcot/cluster/jobs/submit_ext6c.sh",
             "loopedcot/cluster/pilot/e6c_ra30_mcl_shard.sh", "loopedcot/cluster/jobs/plan_tonight.py",
             "loopedcot/cluster/jobs/submit_tonight.sh"]


def pin():
    """jobs/scripts_sha256.txt: every line of a pushed EXT6 file (jobs/pins_ext6_pushed.txt) stays byte-identical (a
    requeued EXT6 task checks it; a changed EXT6 file stops the generator); every other existing line is rehashed in
    place (the stage-2 files: submit_fill.sh, submit_ext6b.sh, ...); the e6c files are appended."""
    path = os.path.join(JOBS, "scripts_sha256.txt")
    frozen = {}
    for ln in open(os.path.join(JOBS, "pins_ext6_pushed.txt"), "rb").read().decode("utf-8").splitlines():
        h, p = ln.split("  ", 1)
        frozen[p] = ln
    out, have, changed = [], set(), 0
    for ln in open(path, "rb").read().decode("utf-8").splitlines():
        h, p = ln.split("  ", 1)
        if p not in frozen and not os.path.exists(os.path.join(TOP, p)):
            changed += 1                 # a stage-2 file that left the tree (an intermediate copy): its pin goes too
            continue
        have.add(p)
        cur = hashlib.sha256(open(os.path.join(TOP, p), "rb").read()).hexdigest()
        if p in frozen:
            if ln != frozen[p] or cur != h:
                raise SystemExit("pushed EXT6 pin changed or its file changed: %s" % p)
            out.append(ln)
        else:
            changed += cur != h
            out.append("%s  %s" % (cur, p))
    want = list(PIN_FILES)
    for fold in ("ra30_s2", "ra31_s2") if PARTB else ("ra30_s2",):
        for root, dirs, files in os.walk(os.path.join(HERE, "pilot", fold)):
            dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", ".pytest_cache"))
            for f in sorted(files):
                if not f.endswith(".pyc"):
                    want.append(os.path.relpath(os.path.join(root, f), TOP).replace(os.sep, "/"))
    add = 0
    for p in sorted(set(want) - have):
        out.append("%s  %s" % (hashlib.sha256(open(os.path.join(TOP, p), "rb").read()).hexdigest(), p))
        add += 1
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(x + "\n" for x in out))
    return add, changed


def main():
    J = build()
    for d in (JOBS, os.path.join(JOBS, "nolocal")):
        write(J, d)
    n, c = pin()
    print("e6c: %d task lines, %d groups, %.1f card-hours; %d pins appended, %d stage-2 pins rehashed" % (
        len(J.rows), len({r["group"] for r in J.rows}), sum(r["est_card_seconds"] for r in J.rows) / 3600.0, n, c))


if __name__ == "__main__":
    main()
