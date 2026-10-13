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
        m = re.match(r"^((\w+?_\d\.\db)_(?:all|lc|top|rand)\w*)\s+\d+ arms(?: \+ gate)?\s+([\d.]+) h single card", ln)
        if m:
            cells.setdefault(m.group(2), []).append((m.group(1), float(m.group(3))))
    order = ["svamp_1.4b", "asdiv_1.4b", "mawps_1.4b", "gsm8k_2.6b", "svamp_2.6b", "asdiv_2.6b", "mawps_2.6b"]
    for c in order:
        kinds = {x[len(c) + 1:].split("_")[0] for x, _h in cells.get(c, [])}
        if kinds != {"all", "lc", "top", "rand"}:
            raise SystemExit("ra31_gen.py cell_list: unexpected shards of %s: %s" % (c, cells.get(c)))
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
    for ds, grp, pil, full in (("gsm8k", "g19_e6ra30", "g19_e6ra30.s1_pilot.001", "g19_e6ra30.s2_full.001"),):
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
            continue                     # the 2.6B cells: push 3, below
        ds = cell.split("_")[0]
        oc = "%s/partb_%s" % (o31, cell)
        env = "PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs " % (oc, oc)
        extra = " --pool_root $OUT_ROOT" if ds == "mawps" else ""
        sh_ids = [J.add("g33_e6cra31b14", "s1_shard", "ra31_gen.py", "ouro_1_4b_base", [ds, "partb"], sh, oc, [],
                        h * 3600 + 180, "ra31_gen.py cell_list (plan %s): %.2f h at the single-card rate, plus the CPU tests"
                        % (s31[:8], h),
                        "%s && %s%s%s ra31_gen.py cell --cell %s --shard %s --out %s%s"
                        " --mem_fraction $GEN_MEM_FRACTION" % (GJ.unit_pins("ra31_s2"), w31, env, PY, cell, sh, oc, extra))
                  for sh, h in shs]
        sm = J.add("g33_e6cra31b14", "s2_summarize_any", "ra31_gen.py", "ouro_1_4b_base", [ds, "partb"], "", oc, sh_ids,
                   300, "CPU (cell_summarize, afterany on the 4 shards)",
                   "%s%s%s ra31_gen.py cell_summarize --cell %s --out %s%s" % (w31, env, PY, cell, oc, extra))
        ids31 += sh_ids + [sm]
    J.add(MERGE_GROUP, "s1_merge", "merge_ext6.py", "all", ["ra31"], "ra31", "$OUT_ROOT/merged/ra31", ids31, 300, "CPU",
          "%s $REPO/loopedcot/cluster/merge_ext6.py --test ra31 --out_root $OUT_ROOT" % PY)
    # ======================================================================== push 3
    # ---- RA-31 Part B grid, 2.6B (plan 36eb4fea): freeze26 (CPU) from the EXT6 2.6B Part A pilot, then 22 independent
    # shards (gsm8k split by items) with --top_sets, each afterok on freeze26 and on that pilot; cell_summarize afterany
    sets = o31 + "/partb26_sets.json"
    frz = J.add("g34_e6cra31b26", "s0_freeze", "ra31_gen.py", "ouro_2_6b_base", ["partb"], "freeze26", o31 + "/partb26_freeze",
                ["g21_e6ra31a.s1_pilot.001"], 300, "CPU (freeze26)",
                "%sCUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s ra31_gen.py freeze26 --pilot %s/a_2.6b/pilot_2.6b --out %s"
                % (w31, PY, o31, sets))
    ids31b = [frz]
    for cell, shs in partb_cells():
        if not cell.endswith("2.6b"):
            continue
        ds = cell.split("_")[0]
        oc = "%s/partb_%s" % (o31, cell)
        env = "PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs " % (oc, oc)
        extra = " --top_sets %s" % sets + (" --pool_root $OUT_ROOT" if ds == "mawps" else "")
        sh_ids = [J.add("g34_e6cra31b26", "s1_shard", "ra31_gen.py", "ouro_2_6b_base", [ds, "partb"], sh, oc,
                        [frz, "g21_e6ra31a.s1_pilot.001"], h * 3600 + 180,
                        "ra31_gen.py cell_list (plan %s): %.2f h at the single-card rate, plus the CPU tests" % (s31[:8], h),
                        "%s && %s%s%s ra31_gen.py cell --cell %s --shard %s --out %s%s"
                        " --mem_fraction $GEN_MEM_FRACTION" % (GJ.unit_pins("ra31_s2"), w31, env, PY, cell, sh, oc, extra))
                  for sh, h in shs]
        ids31b += sh_ids + [J.add("g34_e6cra31b26", "s2_summarize_any", "ra31_gen.py", "ouro_2_6b_base", [ds, "partb"], "", oc,
                                  sh_ids, 300, "CPU (cell_summarize, afterany on the shards)",
                                  "%s%s%s ra31_gen.py cell_summarize --cell %s --out %s%s" % (w31, env, PY, cell, oc, extra))]

    # ---- RA-29 cells (plan 0aeb6563, pilot/ra29_s3; plan sections 13 and 14), one job each, run alone (one line per GPU,
    # stage *_alone) with sbatch --signal=TERM@120 in the template: cluster/pilot/e6c_ra29_cell.sh (tests, pilot, the 80
    # rule, the 8.5 h cap, full). 2.6B AQuA ran on the card. Card-hours: the plan's single-card totals (pilot + full).
    s29 = sha_of("ra29_s3", "ra29")
    ids29 = []
    for m, key, ds, h in (("ouro26", "ouro_2_6b_base", "csqa", 2.36), ("ouro26", "ouro_2_6b_base", "arc", 1.46),
                          ("think14", "ouro_1_4b_think", "aqua", 0.06), ("think14", "ouro_1_4b_think", "csqa", 0.98),
                          ("think14", "ouro_1_4b_think", "arc", 0.64), ("think26", "ouro_2_6b_think", "aqua", 0.45),
                          ("think26", "ouro_2_6b_think", "csqa", 2.35), ("think26", "ouro_2_6b_think", "arc", 1.53),
                          ("mcleish", "mcleish_llama32_r32", "aqua", 0.11), ("mcleish", "mcleish_llama32_r32", "csqa", 1.62),
                          ("mcleish", "mcleish_llama32_r32", "arc", 1.01), ("huginn", "huginn_0125", "aqua", 0.36),
                          ("huginn", "huginn_0125", "csqa", 6.99)):
        o29 = "$OUT_ROOT/E6/ra29/%s/%s" % (m, ds)
        ids29.append(J.add("g35_e6cra29", "s1_cell_alone", "ra29.py", key, [ds], m, o29, [], h * 3600 + 180,
                           "plan_ra29 sections 13-14: %.2f h at the single-card rate (pilot + full), plus the CPU tests" % h,
                           "cd %s && PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs bash e6c_ra29_cell.sh %s %s %s %s"
                           % (LF, o29, o29, m, ds, o29, s29)))

    # ---- RA-30 (plan ef4f9e7b, pilot/ra30_s3): the Thinking pilots (plan 6e; tokens CPU, screen, rows P of N, merge
    # CPU; the merge exits 3 under the ceiling 0.5, a reported result), McLeish ASDiv (plan 6f; tokens afterok on the
    # McLeish ASDiv select of EXT5, pilot, 3 full shards, summarize), the fresh Ouro-2.6B ASDiv cell (the 80 gate waived;
    # g26 of EXT6b never ran) and the operand flags with summary notes (run.sh flags2) on every Ouro-2.6B output.
    s30 = sha_of("ra30_s3", "ra30")
    o = "$OUT_ROOT/E6/ra30"
    w3 = "cd %s/ra30_s3 && " % LF
    ids30 = []
    THINK = {"1p4": ("ouro_1_4b_think", "$HF_HOME/hub/models--ByteDance--Ouro-1.4B-Thinking/snapshots/"
                                        "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"),
             "2p6": ("ouro_2_6b_think", "$HF_HOME/hub/models--ByteDance--Ouro-2.6B-Thinking/snapshots/"
                                        "f1edd81e7ac41355db670500ceaf204e0f73af68")}
    # (model, dataset, screen h, [rows h per part]) at 0.6 x the card in the builder's table; x 0.6 for the single card
    for M, ds, scr, rows in (("1p4", "gsm8k", 2.8, [4.0, 4.0]), ("2p6", "gsm8k", 3.7, [3.5, 3.5, 3.5]),
                             ("1p4", "asdiv", 3.4, [3.1]), ("2p6", "asdiv", 4.5, [2.0, 2.0])):
        key, tdir = THINK[M]
        ot = "%s/%s/pilot/%s" % (o, M, ds)
        env = "K17_THINK_DIR=%s PROD_ART=%s/prod_art_think PROD_LOGS=%s/prod_logs_think " % (tdir, o, o)
        base = "%s ra30_think.py --run --model %s --ds %s --stage pilot --pool_root $OUT_ROOT --out %s --plan_sha256 %s" % (
            PY, M, ds, o, s30)
        t = J.add("g36_e6cra30t", "s1_tokens", "ra30_think.py", key, [ds], M, ot, [], 300, "CPU (tests, think_tokens)",
                  "%sPROD_DEV=cpu %s -m unittest discover -s tests && %s%sCUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s ra30_think.py"
                  " --tokens --model %s --ds %s --pool_root $OUT_ROOT --out %s" % (w3, PY, w3, env, PY, M, ds, o))
        sc = J.add("g36_e6cra30t", "s2_screen", "ra30_think.py", key, [ds], M, ot, [t], scr * 0.6 * 3600,
                   "RA-30 builder: screen %.1f A100-h at 0.6 x = %.2f h at the single-card rate" % (scr, scr * 0.6),
                   "%s%s%s --phase screen" % (w3, env, base))
        rw = [J.add("g36_e6cra30t", "s3_rows", "ra30_think.py", key, [ds], "%s.%d" % (M, k), ot, [sc], h * 0.6 * 3600,
                    "RA-30 builder: rows part %d of %d, %.1f A100-h at 0.6 x = %.2f h at the single-card rate"
                    % (k, len(rows), h, h * 0.6),
                    "%s%s%s --phase rows --rows_part %d --rows_parts %d" % (w3, env, base, k, len(rows)))
              for k, h in enumerate(rows)]
        mg = J.add("g36_e6cra30t", "s4_merge", "ra30_think.py", key, [ds], M, ot, rw, 300,
                   "CPU (pilot merge, G6; exit 3 under the ceiling 0.5 is a reported result)",
                   "%s%sCUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s --phase merge" % (w3, env, base))
        ids30 += [t, sc] + rw + [mg]
    # McLeish ASDiv (pool: EXT5 g18 McLeish ASDiv select)
    m = "mcleish_llama32_r32"
    om = "%s/%s/pilot/asdiv" % (o, m)
    wm = "%sPROD_ART=%s/prod_art_mcl PROD_LOGS=%s/prod_logs_mcl " % (w3, o, o)
    t = J.add("g37_e6cra30a", "s1_tokens", "ra30_mcl.py", m, ["asdiv"], "", om, ["g18_fillgen.s3_select.002"], 300,
              "CPU (tests, mcl_asdiv_tokens)",
              "%sPROD_DEV=cpu %s -m unittest discover -s tests && %sCUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s ra30_mcl.py --tokens"
              " --ds asdiv --pool_root $OUT_ROOT --out %s" % (w3, PY, wm, PY, o))
    pl = J.add("g37_e6cra30a", "s2_pilot", "ra30_mcl.py", m, ["asdiv"], "", om, [t], 4.6 * 0.6 * 3600,
               "RA-30 builder: pilot 4.6 A100-h at 0.6 x = 2.76 h at the single-card rate",
               "%s%s ra30_mcl.py --run --ds asdiv --stage pilot --pool_root $OUT_ROOT --out %s --plan_sha256 %s"
               " && %s%s ra30_mcl.py --summarize --ds asdiv --out %s/%s/pilot/asdiv" % (wm, PY, o, s30, wm, PY, o, m))
    fs = [J.add("g37_e6cra30a", "s3_full", "ra30_mcl.py", m, ["asdiv"], str(k), om, [pl], 1.8 * 0.6 * 3600,
                "RA-30 builder: full shard 1.8 A100-h at 0.6 x = 1.08 h at the single-card rate",
                "%s%s ra30_mcl.py --run --ds asdiv --stage full --shard %d --shards 3 --pool_root $OUT_ROOT --out %s"
                " --plan_sha256 %s" % (wm, PY, k, o, s30)) for k in range(3)]
    sm = J.add("g37_e6cra30a", "s4_summarize", "ra30_mcl.py", m, ["asdiv"], "", om, fs, 300, "CPU (mcl_asdiv_summarize full)",
               "%s%s ra30_mcl.py --summarize --ds asdiv --out %s/%s/full/asdiv" % (wm, PY, o, m))
    ids30 += [t, pl] + fs + [sm]
    # the fresh Ouro-2.6B ASDiv cell (pilot2 then full2; the 80 gate waived in this plan)
    w = "%sPROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs " % (w3, o, o)
    m26 = "--model ouro_2_6b_base"
    a1 = J.add("g38_e6cra30d", "s1_pilot", "ra30.py", "ouro_2_6b_base", ["asdiv"], "", o + "/ouro_2_6b_base/pilot/asdiv", [],
               0.9 * 3600 + 120, "plan_ra30 6b: ASDiv pilot2 about 0.9 card-h, plus tests and tokens2",
               "%sPROD_DEV=cpu %s -m unittest discover -s tests && %sCUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s ra30.py --tokens %s"
               " --ds asdiv --out %s && %s%s ra30.py --run %s --stage pilot --ds asdiv --out %s --plan_sha256 %s"
               " && %s%s ra30.py --summarize %s --out %s/ouro_2_6b_base/pilot/asdiv"
               % (w, PY, w, PY, m26, o, w, PY, m26, o, s30, w, PY, m26, o))
    a2 = J.add("g38_e6cra30d", "s2_full", "ra30.py", "ouro_2_6b_base", ["asdiv"], "", o + "/ouro_2_6b_base/pilot/asdiv", [a1],
               3.2 * 3600, "plan_ra30 6b: ASDiv full2 about 3.2 card-h (1,059 x 8 s + about 92 x 32 s + gates)",
               "%s%s ra30.py --run %s --stage full --ds asdiv --out %s --plan_sha256 %s"
               " && %s%s ra30.py --summarize %s --out %s/ouro_2_6b_base/full/asdiv" % (w, PY, m26, o, s30, w, PY, m26, o))
    ids30 += [a1, a2]
    # operand flags with the summary notes (run.sh flags2 of ra30_s3), CPU, after each Ouro-2.6B output
    for ds, st, dep in (("gsm8k", "pilot", "g19_e6ra30.s1_pilot.001"), ("gsm8k", "full", "g19_e6ra30.s2_full.001"),
                        ("asdiv", "pilot", a1), ("asdiv", "full", a2)):
        ids30.append(J.add("g39_e6cra30n", "s1_flags", "ra30_flags.py", "ouro_2_6b_base", [ds], st,
                           "%s/ouro_2_6b_base/%s/%s" % (o, st, ds), [dep], 300, "CPU (flags2 %s %s with summary notes)" % (st, ds),
                           "%sCUDA_VISIBLE_DEVICES= PROD_DEV=cpu %s ra30_flags.py --out %s/ouro_2_6b_base/%s/%s"
                           % (w3, PY, o, st, ds)))

    # ---- the push-3 merges (afterany on the push-3 jobs of each test)
    for test, ids in (("ra29", ids29), ("ra30", ids30), ("ra31", ids31b)):
        J.add(MERGE_GROUP, "s1_merge", "merge_ext6.py", "all", [test], test, "$OUT_ROOT/merged/%s" % test, ids, 300, "CPU",
              "%s $REPO/loopedcot/cluster/merge_ext6.py --test %s --out_root $OUT_ROOT" % (PY, test))
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
        txt = GJ.with_hfmod(g, GJ.SBATCH.format(group=g, time="%d:00:00" % TIME_H))
        if g == "g35_e6cra29":            # RA-29 cells: SIGTERM 120 s before the limit (the run resumes after an item)
            txt = txt.replace("#SBATCH --requeue\n", "#SBATCH --requeue\n#SBATCH --signal=TERM@120\n", 1)
        open(os.path.join(out_dir, "task_%s.sbatch" % g), "w", encoding="utf-8", newline="\n").write(txt)


PIN_FILES = ["loopedcot/cluster/gen_ext6c.py", "loopedcot/cluster/jobs/submit_ext6c.sh",
             "loopedcot/cluster/pilot/e6c_ra30_mcl_shard.sh", "loopedcot/cluster/jobs/plan_tonight.py",
             "loopedcot/cluster/jobs/submit_tonight.sh", "loopedcot/cluster/pilot/e6c_ra29_cell.sh"]


def pin():
    """jobs/scripts_sha256.txt: every line of a pushed EXT6 file (jobs/pins_ext6_pushed.txt) stays byte-identical (a
    requeued EXT6 task checks it; a changed EXT6 file stops the generator); every other existing line is rehashed in
    place (the stage-2 files: submit_fill.sh, submit_ext6b.sh, ...); the e6c files are appended."""
    path = os.path.join(JOBS, "scripts_sha256.txt")
    frozen = {}
    # push 1 (EXT6) and push 2 (stage 2, minus the login-node scripts push 3 changes: plan_tonight.py, submit_ext6c.sh,
    # gen_ext6c.py; no task runs them) stay byte-identical, so a requeued task of an earlier push passes its pin check
    for name in ("pins_ext6_pushed.txt", "pins_stage2_pushed.txt"):
        for ln in open(os.path.join(JOBS, name), "rb").read().decode("utf-8").splitlines():
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
    for fold in ("ra30_s2", "ra31_s2", "ra30_s3", "ra29_s3") if PARTB else ("ra30_s2",):
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
