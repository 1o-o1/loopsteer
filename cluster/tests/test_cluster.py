"""CPU tests of the cluster bundle: job list, skip list, hashes, pilot copies, extractors and the shared-set readout.

    python -B -m pytest cluster/tests -q -p no:cacheprovider

Tests marked by `needs_artifacts` read the pilot artifacts named by $PILOT_ARTIFACTS (the run outputs the paper's
numbers came from) and are skipped without it. They copy what they need into a temporary folder; the artifacts are
never written.
"""
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.dirname(HERE)
LC = os.path.dirname(CL)
TOP = os.path.dirname(LC)
PILOT = os.path.join(CL, "pilot")
JOBS = os.path.join(CL, "jobs")
sys.path.insert(0, LC)
sys.dont_write_bytecode = True

from cluster import gen_jobs as GJ          # noqa: E402
from cluster import merge as MG             # noqa: E402
from cluster.extract import run as EXR      # noqa: E402
from cluster.extract import shared as SH    # noqa: E402

ART = os.environ.get("PILOT_ARTIFACTS", "")
needs_artifacts = pytest.mark.skipif(not (ART and os.path.isdir(ART)), reason="PILOT_ARTIFACTS not set")
NOT_RUN = ("k17_control", "k17_vbrank", "k17_cgcheck", "k17_closers", "k17_x4e", "k17_trace", "k17_reread")


def joblist():
    with open(os.path.join(JOBS, "joblist.tsv"), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ------------------------------------------------------------------ job list and skip list
def test_every_cell_has_a_word_and_no_pending():
    with open(os.path.join(JOBS, "cells.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and not [r for r in rows if r["word"] == "pending"]
    with open(os.path.join(JOBS, "skip_conflicts.tsv"), encoding="utf-8") as f:
        assert len(f.read().strip().splitlines()) == 1          # header only


def test_not_run_table_has_no_job():
    m, order, last, cells = GJ.load_manifest(os.path.join(CL, "manifest.yaml"))
    scope = [c for c in cells if c["status"] == "scope"]
    assert len(scope) == 50
    for c in scope:
        assert GJ.route(c["card"], c["model"], c["dataset"]) is None, c
    for r in joblist():
        cmd = r["cmd"]
        assert not any(x in cmd for x in NOT_RUN), r["id"]
        assert not re.search(r"--setting (b1|b2|c)\b", cmd), r["id"]          # LoopCD and the loop-2 contrast
        assert "huginn16" not in cmd and "huginn8" not in cmd
        if "k17_public.py --run" in cmd:
            assert "--no_reduced" in cmd                                        # no Huginn K16 / K8 rows
        if r["script"] == "steer_gen.py":
            assert r["model"] in ("ouro_1_4b_base", "ouro_2_6b_base", "ouro_1_4b_think", "ouro_2_6b_think")
        if r["script"] == "steer_gen_math.py":            # B-EXT5 fill: McLeish and Huginn through its mcleish mode
            assert (r["model"] in ("ouro_1_4b_base", "ouro_2_6b_base", "ouro_1_4b_think", "ouro_2_6b_think")
                    or (r["model"] in GJ.GENR and r["group"] == GJ.FILL_GEN_GROUP and " mcleish " in cmd)) \
                and "screen_A_from_settle" not in cmd
        if r["script"] == "steer_gen_raven.py":            # McLeish: default setting; Huginn (item 4): --setting hug
            assert (r["model"], "--setting hug" in cmd) in (("mcleish_llama32_r32", False), ("huginn_0125", True))


def test_job_paths_and_configs():
    for r in joblist():
        for p in re.findall(r"\$REPO/(loopedcot/\S+?\.(?:py|json))\b", r["cmd"]):
            assert os.path.exists(os.path.join(TOP, p)), (r["id"], p)
        assert "loop-flow" not in r["cmd"] and "WEIGHTS_ROOT/ouro" not in r["cmd"]
    cfgs = sorted(glob.glob(os.path.join(JOBS, "gen_cfg", "*.json")))
    assert len(cfgs) == 17                       # 16 job configs + smoke_a1_gsm8k.json (cluster/env/smoke_cluster.sh)
    for p in cfgs:
        d = json.load(open(p, encoding="utf-8"))
        if "svamp" in os.path.basename(p):
            assert d["dataset"] == "svamp" and d["settle"].startswith("cells_") and "$" not in d["settle"]


def test_full_pools_ifm_wave_and_generation_rules():
    rows = joblist()
    build = [r for r in rows if r["group"] == "g0_xllm"]
    assert len(build) == 1 and build[0]["cmd"].endswith("cluster/env/preflight.sh")
    assert "build_xllm.sh" in open(os.path.join(CL, "env", "preflight.sh"), encoding="utf-8").read()
    for r in rows:
        cmd = r["cmd"]
        if r["model"].startswith("ifm_"):
            assert r["group"] in ("g6_ifm", "g12_ifmv1", "g14_ifmy1", "g16_ifmgrid", "g17_fillmap"), r["id"]   # IFM waves
            if r["stage"] in ("s1_run", "s2_window"):
                assert build[0]["id"] in r["depends_on"].split(","), r["id"]
        if r["script"] in ("k15_full.py", "k16_path_ifm.py", "k16_path_ouro.py") and "--models" in cmd:
            assert "--full_pool" in cmd, r["id"]
        if r["script"] == "k15_pairs.py":
            assert "--n_half 100000" in cmd
        for t in re.findall(r"--targets (\S+)", cmd):
            assert all(int(x.split("=")[1]) >= 100000 for x in t.split(",")), r["id"]
        if r["script"] == "k17_varbench.py" and "--phase M" in cmd:
            assert "plan_M_" in cmd and "--phase all" not in cmd
        if "steer_select.py" in cmd:
            assert "--floor" not in cmd and "--max_scored" not in cmd
    assert not os.path.exists(os.path.join(JOBS, "submit_all.sh"))
    s0 = open(os.path.join(JOBS, "submit_00_xllm.sh"), encoding="utf-8").read()
    s1 = open(os.path.join(JOBS, "submit_10_rest.sh"), encoding="utf-8").read()
    assert sum(1 for ln in s0.splitlines() if ln.startswith("sbatch ")) == 1 and "g0_xllm.s1_build.tsv" in s0
    assert "_done/%s" % build[0]["id"] in s1 and "exit 3" in s1 and "g0_xllm" not in s1.split("exit 3\nfi\n", 1)[1]
    from cluster.extract import common
    assert "neither_rate" in common.RESULT_COLUMNS


def test_gradient_skip_cf_pool_venvs_and_home_guard():
    rows = joblist()
    for r in rows:
        cmd = r["cmd"]
        if r["group"] == "g0_xllm":
            assert cmd == "bash $REPO/loopedcot/cluster/env/preflight.sh"
            continue
        assert cmd.startswith("source $VENV_"), r["id"]
        want = {"k17_public.py": "RAVEN", "k17_freeze_raven.py": "RAVEN", "k17_parcae.py": "PARCAE",
                "k15_full.py": "IFM", "k15_pairs.py": "IFM", "k16_path_ifm.py": "IFM"}.get(r["script"])
        if r["script"] == "k17_varbench.py":
            want = "IFM" if r["model"].startswith("ifm_") else "OURO"
        if want:
            assert cmd.startswith("source $VENV_%s/" % want), r["id"]
        if r["script"] in ("k17_varbench.py", "k17_public.py", "k17_parcae.py") and (" --run " in cmd or "--phase" in cmd):
            assert "--no_gradient_rank" in cmd, r["id"]
        assert "--fd_check" not in cmd and "--bf16_check" not in cmd
        if r["script"] == "k15_full.py":
            assert "--cf_pool $DATA_ROOT/cf_pool/cf_pool.json" in cmd
    tpl = open(os.path.join(JOBS, "task_g1_copy.sbatch"), encoding="utf-8").read()
    assert "inside a home directory" in tpl and "source \"$VENV/bin/activate\"" not in tpl


def test_cf_pool_file_and_candidates():
    sys.path.insert(0, PILOT)
    import cf_pool as CFP
    pool = CFP.load_pool(os.path.join(PILOT, "data", "cf_pool", "cf_pool.json"))
    assert pool["header"]["n_pairs"] == len(pool["pairs"]) == 3741
    assert pool["header"]["builder_stats"]["candidates"] == 9125 and pool["header"]["builder_stats"]["pos0_removed"] == 5384
    assert all(v["in_pool"] == v["kept_keys"] for v in pool["header"]["checks_against_production"].values())
    assert "Ouro tokenizer" in pool["header"]["note"]

    def enc(text, ans=False):                     # one token per character: answers of one char only are single
        return [ord(c) for c in text]
    c, st = CFP.pool_candidates(enc, pool)
    assert sum(v["pool_pairs"] for v in st.values()) == 3741
    assert all(set(x) >= {"A", "B", "pos", "tA", "tB", "a", "b", "rel", "half", "para"} for x in c)
    for fam in ("llama", "huginn"):
        d = json.load(open(os.path.join(PILOT, "data", "public", "cands_%s_CF.json" % fam), encoding="utf-8"))
        assert d["stats"]["source"].startswith("cf_pool.json") and all(x["sel"]["subjects"] == [] for x in d["cands"])
        assert {x["pool_key"] for x in d["cands"]} <= {q["key"] for q in pool["pairs"]}


def test_smoke_hash_grad_acceptance(tmp_path):
    from cluster import smoke_hash as SHH
    a, b = tmp_path / "off", tmp_path / "on"
    a.mkdir()
    b.mkdir()
    it = {"pair_id": "p", "rows": {"O1|ans|Llast@1|1": {"f": [1.0]}, "O1|ans|2.12|1": {"f": [0.5]}}, "G": {"2.12": 0.1},
          "t_fwd": 1.0}
    (a / "res_m_D1.json").write_text(json.dumps({"items": [it]}), encoding="utf-8")
    it2 = {"pair_id": "p", "rows": {"O1|ans|Llast@1|1": {"f": [1.0]}}, "t_fwd": 2.0}
    (b / "res_m_D1.json").write_text(json.dumps({"items": [it2]}), encoding="utf-8")
    assert SHH.grad(str(a), str(b)) == 0
    it2["rows"]["O1|ans|Llast@1|1"]["f"] = [1.5]
    (b / "res_m_D1.json").write_text(json.dumps({"items": [it2]}), encoding="utf-8")
    assert SHH.grad(str(a), str(b)) == 1


def test_vb_slots_keeps_the_frozen_plan_hash(tmp_path):
    from cluster import vb_slots
    sys.path.insert(0, PILOT)
    import k17_varbench as VB
    plan = tmp_path / "plan.md"
    plan.write_text("# t\n## 12. x\nA\nLaunch code sha256: `%s`\n## 13. Run log\n- earlier\n\n## 14. Review\nR\n" % ("0" * 64),
                    encoding="utf-8")
    out = tmp_path / "o"
    out.mkdir()
    (out / "window_ouro.json").write_text(json.dumps({"selected": ["Lmid@2"], "n_items": 3}), encoding="utf-8")
    subprocess.check_call([sys.executable, "-B", os.path.join(CL, "vb_slots.py"), "--plan", str(plan), "--out", str(out),
                           "--model", "ouro"], stdout=subprocess.DEVNULL)
    a, b = VB.plan_hashes(str(plan)), VB.plan_hashes(str(out / "plan_M_ouro.md"))
    assert a["plan_frozen_sha256"] == b["plan_frozen_sha256"] and a["plan_sha256"] != b["plan_sha256"]
    txt = (out / "plan_M_ouro.md").read_text(encoding="utf-8")
    assert txt.index("phase W slots selected") < txt.index("## 14.")


def test_skip_list_changes():
    d = json.load(open(os.path.join(CL, "cells_skipped.json"), encoding="utf-8"))
    assert d["counts"]["total"] == len(d["cells"])
    keys = {(c["card"], c["model"], c["dataset"]) for c in d["cells"]}
    for m, ds in (("ifm_p1_ouro", "CF"), ("ifm_p1_ouro_inj", "CF"), ("ifm_d112", "CF"), ("ifm_p2m_r5", "D1"),
                  ("ifm_p2m_ortho", "D1")):
        assert ("edit-site map", m, ds) not in keys
        assert GJ.route("edit-site map", m, ds) is not None
    for k in (("loops against depth", "mcleish_llama32_r32", "D1"), ("loops against depth", "mcleish_llama32_r32", "D2"),
              ("generation placement", "mcleish_llama32_r32", "SVAMP_gen")):
        assert k in keys
    fill = [("ifm_d28", d_) for d_ in ("D1", "CF", "D2", "D4", "D5")] + [("ouro_2_6b_think", d_) for d_ in ("D1", "D2", "D4", "D5")] \
        + [(m, d_) for m in ("ifm_p2m_r5", "ifm_p2m_ortho") for d_ in ("D2", "D4", "D5")] + [("ifm_p2m_diag", "D4"), ("ifm_p2m_diag", "D5")]
    for m, ds in fill:                                  # B-EXT5 fill: routed, off the skip list, logged in changes
        assert ("edit-site map", m, ds) not in keys and GJ.route("edit-site map", m, ds) is not None
    assert ("generation placement", "huginn_0125", "SVAMP_gen") not in keys
    assert len(d["changes"]) == 26 + len(fill) + 2 and all(c["date"] == "2026-10-12" for c in d["changes"][26:])


def test_scripts_sha256_matches_files():
    lines = open(os.path.join(JOBS, "scripts_sha256.txt"), encoding="utf-8").read().splitlines()
    assert len(lines) > 40
    for ln in lines:
        h, rel = ln.split("  ", 1)
        assert hashlib.sha256(open(os.path.join(TOP, rel), "rb").read()).hexdigest() == h, rel


# ------------------------------------------------------------------ pilot copies
def test_pinned_revisions_equal_weights_json():
    w = {c["hub_id"]: c["revision"] for c in json.load(open(os.path.join(CL, "weights.json")))["checkpoints"]}
    n = 0
    for f in glob.glob(os.path.join(PILOT, "*.py")):
        s = open(f, encoding="utf-8").read()
        for repo, rev in re.findall(r'"([\w.-]+/[\w.-]+)": "([0-9a-f]{40})"', s):
            if repo in w:
                assert w[repo] == rev, (f, repo)
                n += 1
        for name, rev in re.findall(r'"models--SandyResearch--(parcae-[\w.]+)/snapshots/([0-9a-f]{40})"', s):
            if "SandyResearch/" + name in w:
                assert w["SandyResearch/" + name] == rev
                n += 1
        for repo, rev in re.findall(r'\("([\w.-]+/[\w.-]+)", "([0-9a-f]{40})"\)', s):
            if repo in w:
                assert w[repo] == rev, (f, repo)
                n += 1
    assert n >= 15


def test_pilot_environment_paths(tmp_path):
    code = ("import sys; sys.path.insert(0, %r); import k10_ifm, k15_full, k16_path_ouro, k17_public; "
            "print(k10_ifm.XLLM, k10_ifm.CKPT, k15_full.RUN_ROLE, k16_path_ouro.DATA_DIR, k17_public.HUB)" % PILOT)
    env = dict(os.environ, XLLM_ROOT="/x/xllm", WEIGHTS_ROOT="/w", DATA_ROOT="/d", HF_HOME=str(tmp_path / "hf"),
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("IFM_CKPT_ROOT", None)
    env.pop("RUN_ROLE", None)
    env.pop("HF_HUB_CACHE", None)
    out = subprocess.check_output([sys.executable, "-B", "-c", code], env=env, cwd=str(tmp_path)).decode().split()
    assert out[0] == "/x/xllm" and out[1] == "/w" and out[2] == "cluster"
    assert out[3].replace("\\", "/") == "/d/k16_path_ouro"
    assert out[4].replace("\\", "/").startswith(str(tmp_path / "hf").replace("\\", "/"))


# ------------------------------------------------------------------ extractors on the pilot artifacts
EXTRACT_CASES = [("k17_varbench.py", "k17_varbench/full"), ("k17_public.py", "k17_public/full"),
                 ("k17_parcae.py", "k17_parcae/full_final"), ("steer_gen.py", "k17_gen/a1"),
                 ("k15_full.py", "k15_fullA"), ("k15_pairs.py", "k15_fullB"), ("k16_path_ouro.py", "k16_path_ouro"),
                 ("k16_path_ifm.py", "k16_path_ifm")]


@needs_artifacts
@pytest.mark.parametrize("script,job", EXTRACT_CASES)
def test_extractor_reproduces_stored_values(script, job):
    src = MG.src_of(script, PILOT)
    v = EXR.get(script).verify(os.path.join(ART, job), src)
    assert v["rows"] > 0 and v["compared"] > 0
    bad = v["mismatches"]
    # the one known case: a value stored by the pilot run differs from a re-run of its own summarize in the 4th dp
    for lab, path, a, b in bad:
        assert lab == "rerun95" and max(abs(x - y) for x, y in zip(a, b)) <= 1e-4, (lab, path, a, b)
    assert v["compared"] - v["equal"] == len(bad) <= 1


# ------------------------------------------------------------------ shared-set readout
SHARED_CASES = [("k15_pairs.py", "k15_fullB", "ifm_p1_mid", "facts"), ("k16_path_ifm.py", "k16_path_ifm", "ifm_p1_mid", "addition"),
                ("k17_parcae.py", "k17_parcae/full_final", "parcae-370m", "facts"),
                ("k15_full.py", "k15_fullA", "ifm_d112", "facts")]
VALS = ["estimate", "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi", "n_items", "n_clusters", "random_control_estimate",
        "positive_control_value"]


def _key(r):
    return (r["card"], r["arm"], r["site"], r["loop"], r["strength"], r["metric"], r["endpoint"])


@needs_artifacts
@pytest.mark.parametrize("script,job,model,ds", SHARED_CASES)
def test_shared_rows_on_own_set_equal_a_fresh_summarize(script, job, model, ds, tmp_path):
    src = MG.src_of(script, PILOT)
    ext = EXR.get(script)
    jd = os.path.join(ART, job)
    own = ext.item_sets(jd)[(model, ds)][0]
    fresh = str(tmp_path / "fresh")
    shutil.copytree(jd, fresh, ignore=shutil.ignore_patterns("logs"))
    from cluster.extract import resum
    with resum.quiet():
        resum.load_module(src, "fresh_x").summarize(fresh)
    base = {_key(r): [r[f] for f in VALS] for r in ext.extract(fresh, src) if r["model"] == model and r["dataset"] == ds}
    sh = {_key(r): [r[f] for f in VALS] for r in SH.rows(script, src, jd, model, ds, own)}
    assert base and sh == base
    sub = sorted(own)[:len(own) - 7]
    rows = SH.rows(script, src, jd, model, ds, sub)
    assert rows and max(int(r["n_items"]) for r in rows) == len(sub)
    assert all(r["item_set"] == "shared" for r in rows)


def test_shared_readout_states(monkeypatch):
    """Row logic without artifacts: shared, own only below 40, one-model row, pending, different scripts."""
    m = {"rows": {"B": [{"id": "pair", "models": ["a", "b"]}, {"id": "trio", "models": ["c", "d", "e"]},
                        {"id": "mix", "models": ["f", "g"]}]}}
    cells = [{"grid": "B", "card": "edit-site map", "model": x, "dataset": "D1"} for x in "abcdefgh"]
    big = set(map(str, range(60)))
    info = {
        ("edit-site map", "a", "D1"): dict(state="done", script="k15_pairs.py", planned=big, rel="ra", mk="a", dk="facts"),
        ("edit-site map", "b", "D1"): dict(state="done", script="k15_pairs.py", planned=set(map(str, range(10, 80))),
                                           rel="rb", mk="b", dk="facts"),
        ("edit-site map", "c", "D1"): dict(state="done", script="k15_pairs.py", planned=big, rel="rc", mk="c", dk="facts"),
        ("edit-site map", "d", "D1"): dict(state="done", script="k15_pairs.py", planned=set(map(str, range(30))),
                                           rel="rd", mk="d", dk="facts"),
        ("edit-site map", "e", "D1"): dict(state="skip", script="k15_pairs.py", planned=set(), rel="re", mk="e", dk="facts"),
        ("edit-site map", "f", "D1"): dict(state="done", script="k15_pairs.py", planned=big, rel="rf", mk="f", dk="facts"),
        ("edit-site map", "g", "D1"): dict(state="done", script="k16_path_ifm.py", planned=big, rel="rg", mk="g", dk="facts"),
        ("edit-site map", "h", "D1"): dict(state="done", script="k15_pairs.py", planned=big, rel="rh", mk="h", dk="facts"),
    }
    calls = []

    def fake_rows(script, src, job, model, ds, keep):
        calls.append((model, len(keep)))
        return [{"card": "edit_site_map", "model": model, "n_items": str(len(keep)), "item_set": "shared"}]
    monkeypatch.setattr(SH, "rows", fake_rows)
    # B-EXT2 item 0: one shared set per benchmark across every done model (manifest rows no longer group); the
    # k15_pairs ids form their own space; small sets are kept and marked
    rows, table, mark = MG.shared_readout(m, cells, info, "OUT", PILOT)
    st = {t[3]: t for t in table}
    assert set(st) == {"k15_pairs", "common"}
    k15 = st["k15_pairs"]
    assert k15[5].split(",") == ["a", "b", "c", "d", "f", "h"] and k15[6] == 20 and k15[7] == "shared (small: n 20 < 40)"
    assert st["common"][5] == "g" and st["common"][7] == "shared = own"
    assert sorted(calls) == [(x, 20) for x in "abcdfh"] and len(rows["B"]) == 6
    assert mark[("edit-site map", "a", "D1")] == "shared n 20 (small)" and mark[("edit-site map", "g", "D1")] == "shared = own"
    info[("edit-site map", "b", "D1")]["state"] = "pending"
    calls.clear()
    rows, table, mark = MG.shared_readout(m, cells, info, "OUT", PILOT)
    k15 = {t[3]: t for t in table}["k15_pairs"]
    assert k15[6] == 30 and "models pending (the set can shrink): b" in k15[8]
    assert sorted(calls) == [(x, 30) for x in "acdfh"]


@needs_artifacts
def test_merge_end_to_end_on_artifact_copies(tmp_path):
    """Fake output tree from artifact copies (middle pair and 819M designs via k15_pairs, Parcae via k17_parcae):
    the script's summarize, the extractor, then merge.py. Checks the completeness matrix, the shared sets, the
    shared rows in results_all_B.csv and the marks in grid_B.md."""
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    jobs = []
    for m in ("ifm_p1_mid", "ifm_p1_mid_inj", "ifm_p2m_r5", "ifm_p2m_diag", "ifm_p2m_ortho"):
        d = out / "B" / "k15_pairs" / m
        d.mkdir(parents=True)
        shutil.copy(os.path.join(ART, "k15_fullB", "k15p_%s.json" % m), str(d))
        jobs.append(("k15_pairs.py", d))
    for m in ("parcae-370m", "parcae-1.3b"):
        d = out / "B" / "k17_parcae" / m
        d.mkdir(parents=True)
        for f in glob.glob(os.path.join(ART, "k17_parcae", "full_final", "res_%s_D1.json" % m)) + \
                glob.glob(os.path.join(ART, "k17_parcae", "full_final", "res_%s_VB.json" % m)) + \
                glob.glob(os.path.join(ART, "k17_parcae", "full_final", "meta_%s.json" % m)):
            shutil.copy(f, str(d))
        jobs.append(("k17_parcae.py", d))
    for script, d in jobs:
        src = os.path.join(PILOT, script)
        subprocess.check_call([sys.executable, "-B", src, "--summarize", "--out", str(d)], env=env,
                              stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-B", os.path.join(CL, "extract", "run.py"), "--script", script,
                               "--src", src, "--job", str(d)], env=env, stdout=subprocess.DEVNULL)
    res = json.loads(subprocess.check_output([sys.executable, "-B", os.path.join(CL, "merge.py"), "--out_root", str(out)],
                                             env=env).decode().strip().splitlines()[-1])
    dest = out / "merged"
    with open(str(dest / "completeness.csv"), encoding="utf-8") as f:
        comp = {(r["card"], r["model"], r["dataset"]): r["state"] for r in csv.DictReader(f)}
    assert comp[("edit-site map", "ifm_p1_mid", "D1")] == "done"
    assert comp[("edit-site map", "parcae_370m", "D1")] == "done"
    with open(str(dest / "shared_sets.csv"), encoding="utf-8") as f:
        sh = {(r["card"], r["dataset"], r["id_space"]): r for r in csv.DictReader(f)}
    # B-EXT2 item 0: one shared set per benchmark across every done model; the k15_pairs ids form their own space
    k15m = ("ifm_p1_mid", "ifm_p1_mid_inj", "ifm_p2m_r5", "ifm_p2m_diag", "ifm_p2m_ortho")
    mid = sh[("edit-site map", "D1", "k15_pairs")]
    assert set(mid["done_models"].split(",")) == set(k15m)
    sets = [EXR.get("k15_pairs.py").item_sets(str(out / "B" / "k15_pairs" / m))[(m, "facts")][0] for m in k15m]
    n_mid = len(set.intersection(*[set(map(str, x)) for x in sets]))
    assert int(mid["n_shared"]) == n_mid
    with open(str(dest / "results_all_B.csv"), encoding="utf-8") as f:
        rr = list(csv.DictReader(f))
    shared_mid = [r for r in rr if r["item_set"] == "shared" and r["cell_model"] == "ifm_p1_mid"]
    if n_mid:
        assert mid["state"].startswith("shared") and shared_mid          # small sets are kept (marked)
        assert max(int(r["n_items"]) for r in shared_mid) <= n_mid
    else:
        assert mid["state"] == "no common items" and not shared_mid
    par = sh[("edit-site map", "D1", "common")]
    assert set(par["done_models"].split(",")) == {"parcae_370m", "parcae_1_3b"}
    grid = open(str(dest / "grid_B.md"), encoding="utf-8").read()
    assert "CF (CounterFact): CounterFact pool of the cluster run" in grid
    if n_mid:
        assert "shared n %d" % n_mid in grid
    assert res["shared_rows"].get("B", 0) == len([r for r in rr if r["item_set"] == "shared"])


# ------------------------------------------------------------------ RA-23 additions (2026-10-09)
def test_ra23_routes_done_registry_and_submit_order():
    assert GJ.route("branch-freeze", "ouro_1_4b_think", "D1") == ("k17_freeze.py", "frzo")
    assert GJ.route("branch-freeze", "ouro_2_6b_think", "D4") == ("k17_freeze.py", "frzo")
    assert GJ.route("branch-freeze", "ouro_2_6b_think", "D5") is None
    assert GJ.route("branch-freeze", "ifm_p1_ouro", "D2") == ("k17_freeze_ifm.py", "frzi")
    assert GJ.route("generation placement", "mcleish_llama32_r32", "GSM8K_gen") == ("steer_gen_raven.py", "genr")
    for m in ("mcleish_llama32_r32", "llama_nr", "huginn_0125", "ouro_1_4b_think"):
        assert GJ.route("edit-site map", m, "LRE") == ("k17_public.py", "publre")
    assert GJ.route("edit-site map", "ifm_p1_ouro", "LRE") is None
    rows = joblist()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    with open(os.path.join(JOBS, "done_local.tsv"), encoding="utf-8") as f:
        done = list(csv.DictReader(f, delimiter="\t"))
    dids = {d["id"] for d in done}
    assert {d["shard"] for d in done if d["script"] == "steer_gen_raven.py"} == {"r00", "r01"}
    assert not dids & set(ids)
    assert not [r for r in rows if dids & set(r["depends_on"].split(","))]
    for f in glob.glob(os.path.join(JOBS, "g*.tsv")):
        txt = open(f, encoding="utf-8").read()
        assert not [d for d in dids if d + "\t" in txt], f
    mcl = [r for r in rows if r["script"] == "steer_gen_raven.py" and r["stage"] == "s4_row"
           and r["model"] == "mcleish_llama32_r32"]
    assert sorted(r["shard"] for r in mcl) == sorted(x for x in GJ.GENR_ROWS if x not in ("r00", "r01"))
    assert all("test -f $OUT_ROOT/A/steer_gen_raven/mcleish/items.json" in r["cmd"] for r in mcl)
    for r in rows:
        if r["script"] in ("k17_freeze.py", "k17_freeze_ifm.py"):
            assert "$FRZO_PLAN_SHA" in r["cmd"] and "--summarize" in r["cmd"]
            assert ("--full_pool" in r["cmd"]) == (r["script"] == "k17_freeze.py")
        if "_LRE" in r["out"]:
            assert r["datasets"] == "D1" and ("$DATA_ROOT/public_lre" in r["cmd"] or r["stage"] == "s4_summarize")
            assert (r["group"] == "g5_huginn") == (r["model"] == "huginn")
    s1 = open(os.path.join(JOBS, "submit_10_rest.sh"), encoding="utf-8").read()
    said = re.findall(r'echo "group (\w+): (\d+) jobs, ([\d.]+) card-hours', s1)
    order = [g for g, _, _ in said]
    assert order == [g for g in GJ.GROUP_ORDER if g in order] and order[-2:] == ["g5_huginn", "g9_merge"]
    ch = json.load(open(os.path.join(JOBS, "card_hours.json"), encoding="utf-8"))["groups"]
    for g, n, h in said:
        assert int(n) == ch[g]["jobs"] and abs(float(h) - ch[g]["card_hours"]) <= 0.06, g
    s0 = open(os.path.join(JOBS, "submit_00_xllm.sh"), encoding="utf-8").read()
    assert 'echo "group g0_xllm: 1 jobs' in s0


def test_lre_pool_rules():
    d = os.path.join(PILOT, "data", "public_lre")
    meta = json.load(open(os.path.join(d, "build_meta.json"), encoding="utf-8"))
    lre = os.path.join(LC, "data_steer", "lre")
    for name, h in meta["inputs"].items():
        assert hashlib.sha256(open(os.path.join(lre, name), "rb").read()).hexdigest() == h
    keys = {}
    for suf in ("ouro", "huginn", "mcleish"):
        for x in open(os.path.join(lre, "pairs_lre_%s.jsonl" % suf), encoding="utf-8"):
            q = json.loads(x)
            keys[(suf, q["key"])] = q
    for fam, suf in GJ_LRE_FAMS:
        f = os.path.join(d, "cands_%s_D1.json" % fam)
        assert hashlib.sha256(open(f, "rb").read()).hexdigest() == meta["outputs"]["cands_%s_D1.json" % fam]["sha256"]
        cands = json.load(open(f, encoding="utf-8"))["cands"]
        per_rel, per_subj = {}, {}
        for c in cands:
            q = keys[(suf, c["pair_id"][4:])]
            assert not q["answer_in_prompt"] and c["ds"] == "D1" and c["split"] == q["half"]
            assert c["A"][0] != 128000 and len(c["A"]) == len(c["B"]) and c["tA"] != c["tB"]
            if fam != "llama":
                assert 0 not in c["src"]                                 # the runner's position-0 rule
            per_rel[c["rel"]] = per_rel.get(c["rel"], 0) + 1
            for s in c["sel"]["subjects"]:
                per_subj[s] = per_subj.get(s, 0) + 1
        assert max(per_rel.values()) <= meta["cap_per_relation"] and max(per_subj.values()) <= 2


GJ_LRE_FAMS = (("ouro", "ouro"), ("huginn", "huginn"), ("llama", "mcleish"))


def test_steer_gen_math_data_layer(tmp_path):
    """steer_gen_math installs the ASDiv / MAWPS data layer in a fresh process (steer_gen's globals are patched there,
    never in this test process): every pool problem kept, task rows = the A side, a registered SVAMP-format task."""
    code = ("import sys, json; sys.path.insert(0, %r); import steer_gen_math as M\n"
            "out = {}\n"
            "for ds in ('asdiv', 'mawps'):\n"
            "    S = M.install(ds, %r)\n"
            "    pool, n = S.core_pool()\n"
            "    from prod.tasks import rows, task_cfg\n"
            "    rw = {r['idx']: r for r in rows(ds)}\n"
            "    ok = all(rw[int(p['idx'])]['input'] == p['question_A'] and rw[int(p['idx'])]['target'] == str(p['gold_A']) for p in pool)\n"
            "    out[ds] = [len(pool), n, ok, task_cfg(ds)['prompt_file'], S.TASK_A, S.DATASET, S.prep_a.__name__]\n"
            "print(json.dumps(out))" % (PILOT, str(tmp_path)))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code], env=env, cwd=str(tmp_path)).decode().strip().splitlines()[-1])
    assert out["asdiv"][:3] == [1114, 1114, True] and out["mawps"][:3] == [2082, 2082, True]
    assert out["asdiv"][3] == "prompt_svamp.txt" and out["asdiv"][4:6] == ["asdiv", "asdiv"]
    import prod.steer_gen as SG
    assert SG.TASK_A in ("gsm8k", "svamp")                      # this process untouched


# ------------------------------------------------------------------ setting.json writer, done-folder staging
def test_write_json_per_process_temp(tmp_path, monkeypatch):
    """steer_gen.write_json renames a per-process temp file (row tasks that start together all write setting.json;
    a shared '<path>.tmp' let one task rename another's temp away); the bytes are unchanged; steer_gen_math no
    longer replaces the writer."""
    import prod.steer_gen as SG
    seen = []
    real = os.replace
    monkeypatch.setattr(SG.os, "replace", lambda a, b: (seen.append(a), real(a, b))[1])
    p = str(tmp_path / "setting.json")
    obj = {"b": [1, 2.5], "a": {"z": None, "y": "x"}, "t": tmp_path}
    SG.write_json(p, obj)
    assert seen == ["%s.%d.tmp" % (p, os.getpid())]
    assert os.listdir(str(tmp_path)) == ["setting.json"]
    q = str(tmp_path / "old.json")               # the earlier writer (fixed '<path>.tmp'): same bytes
    with open(q + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    real(q + ".tmp", q)
    assert open(p, "rb").read() == open(q, "rb").read()
    src = open(os.path.join(PILOT, "steer_gen_math.py"), encoding="utf-8").read()
    assert "write_json_pid" not in src and "S.write_json =" not in src


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


def _stage_fixture(root):
    src = os.path.join(root, "src")
    for rel, data in (("pa/x.json", b'{"a": 1}\n'), ("pa/sub/y.txt", b"y\n"), ("pa/results.csv", b"k,v\n1,2\n"),
                      ("pb/z.jsonl", b"{}\n")):
        os.makedirs(os.path.dirname(os.path.join(src, rel)), exist_ok=True)
        open(os.path.join(src, rel), "wb").write(data)
    os.makedirs(os.path.join(src, "pa", "empty"))
    base = {"evidence": [], "ledger": "", "shard": ""}
    jobs = [dict(base, id="g9_t.s1_run.001", script="a.py", model="m", stage="s1_run", out="$OUT_ROOT/A/a/m",
                 pulled="pa", results_csv="pa/results.csv",
                 results_sha256=hashlib.sha256(b"k,v\n1,2\n").hexdigest()),
            dict(base, id="", script="a.py", model="m", stage="s2_row", shard="r00", out="$OUT_ROOT/A/a/m", pulled="pa"),
            dict(base, id="g9_t.s1_run.002", script="b.py", model="m", stage="s1_run", out="$OUT_ROOT/B/b/m", pulled="pb")]
    dj = os.path.join(root, "done_local.json")
    json.dump({"note": "test", "jobs": jobs}, open(dj, "w", encoding="utf-8"))
    dt = os.path.join(root, "done_local.tsv")
    with open(dt, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "group", "stage", "script", "model", "shard", "out", "copy_from_pilot_artifacts", "ledger"])
        for j, jid in zip(jobs, ("g9_t.s1_run.001", "g9_t.s2_row.005", "g9_t.s1_run.002")):
            w.writerow([jid, "g9_t", j["stage"], j["script"], j["model"], j["shard"], j["out"], j["pulled"], ""])
    return src, dj, dt


@needs_bash
def test_stage_done_copies_checks_and_marks(tmp_path):
    root = tmp_path.as_posix()
    src, dj, dt = _stage_fixture(root)
    out = root + "/out"
    env = dict(os.environ, DONE_SRC=src, OUT_ROOT=out, PYTHON=sys.executable.replace("\\", "/"))
    cmd = [_bash(), os.path.join(JOBS, "stage_done.sh").replace("\\", "/"), dj, dt]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, r.stdout.decode()
    for rel, dst in (("pa/x.json", "A/a/m/x.json"), ("pa/sub/y.txt", "A/a/m/sub/y.txt"),
                     ("pa/results.csv", "A/a/m/results.csv"), ("pb/z.jsonl", "B/b/m/z.jsonl")):
        assert open(os.path.join(out, dst), "rb").read() == open(os.path.join(src, rel), "rb").read()
    assert os.path.isdir(os.path.join(out, "A/a/m/empty"))
    assert sorted(os.listdir(os.path.join(out, "_done"))) == ["g9_t.s1_run.001", "g9_t.s1_run.002", "g9_t.s2_row.005"]
    ok = json.load(open(os.path.join(out, "_staged.ok")))
    assert ok["entries"] == 3 and ok["folders"] == 2 and ok["files_copied"] == 4 and ok["recorded_sha256_checked"] == 1
    assert ok["done_local_sha256"] == hashlib.sha256(open(dj, "rb").read()).hexdigest()
    # a source that no longer has the recorded sha256: exit 1, no _staged.ok, no new markers
    open(os.path.join(src, "pa/results.csv"), "wb").write(b"k,v\n1,3\n")
    out2 = root + "/out2"
    r = subprocess.run(cmd, env=dict(env, OUT_ROOT=out2), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1 and b"recorded" in r.stdout
    assert not os.path.exists(os.path.join(out2, "_staged.ok")) and not os.path.exists(os.path.join(out2, "_done"))
    # a rerun into the staged tree with the bad source removes the old _staged.ok
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1 and not os.path.exists(os.path.join(out, "_staged.ok"))
    # a missing pulled folder fails too
    open(os.path.join(src, "pa/results.csv"), "wb").write(b"k,v\n1,2\n")
    shutil.rmtree(os.path.join(src, "pb"))
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1 and b"pulled folder missing" in r.stdout and not os.path.exists(os.path.join(out, "_staged.ok"))


def _fake_sbatch(tmp_path, rc=0):
    """A fake sbatch on PATH: logs its arguments, prints a job id (1001, 1002, ...)."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    log = (tmp_path / "sbatch.log").as_posix()
    body = ('#!/bin/bash\necho "$*" >> "%s"\nn=$(grep -c . "%s")\necho $((1000 + n))\nexit %d\n' % (log, log, rc))
    (fake / "sbatch").write_bytes(body.encode())
    os.chmod(str(fake / "sbatch"), 0o755)
    return fake, log


SLURM_ENV = {"PARTITION": "gpuq", "ACCOUNT": "acct1", "SLURM_RESTRICT": "--exclude=n05,n06", "NODELIST": "n01,n02,n03,n04"}


@needs_bash
def test_submit_10_rest_needs_staged_marker(tmp_path):
    """submit_10_rest.sh refuses (exit 7) without the cluster.env values, (6) without _smoke.ok, (5) without
    _staged.ok, then passes --partition, --account, --output and $SLURM_RESTRICT on every sbatch command line. The
    template headers keep their placeholders (nothing edits them)."""
    build = [r["id"] for r in joblist() if r["group"] == "g0_xllm"]
    out = tmp_path / "out"
    os.makedirs(str(out / "_done"))
    for b_ in build:
        open(str(out / "_done" / b_), "w").close()
    fake, log = _fake_sbatch(tmp_path)
    open(log, "w").close()
    jobs = tmp_path / "jobs"
    shutil.copytree(JOBS, str(jobs))
    before = {t.name: t.read_bytes() for t in jobs.glob("task_*.sbatch")}
    assert all(b"#SBATCH --partition=<PARTITION>" in v for v in before.values())
    env = dict(os.environ, OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(),
               PATH=str(fake) + os.pathsep + os.environ.get("PATH", ""))
    for k in list(SLURM_ENV) + ["REPO"]:
        env.pop(k, None)
    cmd = [_bash(), (jobs / "submit_10_rest.sh").as_posix()]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 7 and b"PARTITION" in r.stdout and not open(log).read()
    assert "exit code: 7" in open(str(out / "_diag" / "submit_10_rest.txt")).read()
    env.update(SLURM_ENV, REPO=TOP.replace("\\", "/"))
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 6 and b"_smoke.ok" in r.stdout and not open(log).read()
    open(str(out / "_smoke.ok"), "w").close()
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 5 and b"_staged.ok" in r.stdout and not open(log).read()
    assert b"node restriction: --exclude=n05,n06" in r.stdout
    open(str(out / "_staged.ok"), "w").close()
    (out / "bad_nodes.txt").write_text("n03\n")
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    calls = open(log).read().splitlines()
    n_calls = sum(1 for ln in (jobs / "submit_10_rest.sh").read_text(encoding="utf-8").splitlines() if "=$(sbatch " in ln)
    assert r.returncode == 0 and len(calls) == n_calls > 50, r.stdout.decode()[-1500:]
    for c in calls:
        assert "--partition=gpuq --account=acct1 --exclude=n05,n06,n03" in c and "--output=%s/logs/ra20_" % out.as_posix() in c, c
        assert " task_g" in c and "array_" not in c
    assert not (out / "_diag" / "submit_10_rest.txt").exists()
    assert before == {t.name: t.read_bytes() for t in jobs.glob("task_*.sbatch")}


_JOB_CODE = {            # run from cluster/pilot/tree, as the job-list lines are (the bundle's own prod tree)
    "steer_gen": "from prod import steer_gen as S; assert 'tree' in S.__file__; S.job_a = lambda *a, **k: {'exit': %(c)d}; "
                 "sys.exit(S.main(['run', '--setting', 'a1', '--row', 'r00', '--out', '%(o)s']))",
    "steer_gen_raven": "sys.path.insert(0, '..'); import steer_gen_raven as R; assert 'tree' in R.S.__file__; "
                       "R.job = lambda *a, **k: {'exit': %(c)d}; sys.exit(R.main(['run', '--row', 'r00', '--out', '%(o)s']))",
    "steer_gen_math": "sys.path.insert(0, '..'); import steer_gen_math as M; S = M.install('asdiv', '%(o)s'); "
                      "assert 'tree' in S.__file__; "
                      "S.job_a = lambda *a, **k: {'exit': %(c)d}; sys.exit(M.main(['--dataset', 'asdiv', '--rows-dir', "
                      "'%(o)s', 'ouro', 'run', '--setting', 'a1', '--row', 'r00', '--out', '%(o)s', '--config-json', "
                      "'../../jobs/gen_cfg/a1_asdiv.json']))"}


@needs_bash
@pytest.mark.parametrize("entry", sorted(_JOB_CODE))
def test_failed_generation_row_writes_no_done_marker(tmp_path, entry):
    """The array template under the real `run` entry points: a row whose steer_job.json records exit 3 (incomplete
    rows) makes the task exit non-zero and leaves no _done marker; the same row with exit 0 writes the marker."""
    py = sys.executable.replace("\\", "/")
    out = (tmp_path / "out").as_posix()
    os.makedirs(str(tmp_path / "home"))
    for code in (3, 0):
        job_out = "%s/A/x/%s_%d" % (out, entry, code)
        cmd = 'cd $REPO/loopedcot/cluster/pilot/tree && "%s" -B -c "import sys; %s"' % (py, _JOB_CODE[entry] % {"c": code, "o": job_out})
        jid = "g2_gen.s4_row.%03d" % (code + 1)
        lst = tmp_path / ("list_%d.tsv" % code)
        open(str(lst), "w", newline="\n").write("%s\t%s\n" % (jid, cmd))
        env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=out, HOME=(tmp_path / "home").as_posix(),
                   SLURM_ARRAY_TASK_ID="1")
        for v in ("HF_HOME", "SETTLE", "WEIGHTS_ROOT", "IFM_CKPT_ROOT", "XLLM_ROOT", "DATA_ROOT"):
            env.pop(v, None)
        r = subprocess.run([_bash(), os.path.join(JOBS, "task_g2_gen.sbatch").replace("\\", "/"), lst.as_posix()],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert r.returncode == code, r.stdout.decode()[-2000:]
        assert os.path.exists(os.path.join(out, "_done", jid)) == (code == 0), r.stdout.decode()[-2000:]
        dg = os.path.join(out, "_diag", jid + ".txt")
        assert os.path.exists(dg) == (code != 0)
        if code:
            t = open(dg).read()
            for want in ("exit code: 3", "failing check: task command", "host role: cluster", "gpu (name, free, total):",
                         "host memory:", "--- last 50 lines of"):
                assert want in t, (want, t)


# ------------------------------------------------------------------ cluster environment (setup, preflight, smoke)
def test_smoke_lists_use_the_job_list_commands():
    """The smoke task lines are the job list's own command strings with the item counts set to 3: Ouro-1.4B VarBench
    against the Ouro 2.6B Thinking pipeline, steer_gen a1 GSM8K against the a1 SVAMP pipeline."""
    rows = joblist()
    sm = GJ.smoke_lists()
    vb_real = [r for r in rows if r["script"] == "k17_varbench.py" and r["model"] == "ouro_2_6b_think"]
    vb_smoke = dict((jid.split(".")[2], cmd) for jid, cmd in sm["g1_copy"])
    for r in vb_real:
        want = (r["cmd"].replace("ouro26_think", "ouro").replace(" --model_dirs ouro=" + GJ.THINK26, "")
                .replace("--n_vb 100000 --vbd_n ouro=100000", "--n_vb 3 --vbd_n ouro=3"))
        want = re.sub(r"--shard \d+/\d+", "--shard 0/2", want)
        assert want == vb_smoke[r["stage"]] or want.replace("0/2", "1/2") in [c for _, c in sm["g1_copy"]], r["id"]
    gen_real = [r for r in rows if r["script"] == "steer_gen.py" and r["model"] == "ouro_1_4b_base"
                and r["out"].endswith("a1_svamp") and r["shard"] in ("", "r00") and r["stage"] != "s5_anschain_presence"]
    gen_smoke = dict((jid.split(".")[2], cmd) for jid, cmd in sm["g2_gen"])
    assert sorted(gen_smoke) == sorted(r["stage"] for r in gen_real)
    for r in gen_real:
        want = (r["cmd"].replace("a1_svamp.json", "smoke_a1_gsm8k.json").replace("a1_svamp", "a1_gsm8k")
                .replace("K17_DATASET=svamp", "K17_DATASET=gsm8k").replace(" --n 4 ", " --n 3 "))
        if r["stage"] == "s3_select":
            want += " --min_scored 3 --max_scored 3"
        assert want == gen_smoke[r["stage"]], r["id"]
    for g in ("g1_copy", "g2_gen", "fam"):
        lines = open(os.path.join(JOBS, "smoke_%s.tsv" % g), encoding="utf-8").read().splitlines()
        assert lines == ["%s\t%s" % x for x in sm[g]]
    # family lines: the job-list command of that family with the dataset and item count of the smoke
    fam = dict((jid.split(".")[2], cmd) for jid, cmd in sm["fam"])
    assert sorted(fam) == ["f_ifm", "f_mcleish", "f_parcae", "f_raven"]
    assert "steer_gen_raven.py checks" in fam["f_mcleish"] and "--row r00" in fam["f_mcleish"] and "--n 3" in fam["f_mcleish"]
    for key, script, venv in (("f_raven", "k17_public.py", "RAVEN"), ("f_parcae", "k17_parcae.py", "PARCAE"),
                              ("f_ifm", "k17_freeze_ifm.py", "IFM")):
        c = fam[key]
        assert c.startswith("source $VENV_%s/bin/activate && " % venv) and "/cluster/pilot/%s " % script in c
        assert "--targets D1=3" in c or "--debug 3" in c
        assert any(r["script"] == script for r in rows)
    assert json.load(open(os.path.join(JOBS, "gen_cfg", "smoke_a1_gsm8k.json"))) == GJ.SMOKE_CFG


def _smoke_tree(root, pc_exact=3, union=True, gate=True, exit_code=0):
    vb, gen = os.path.join(root, "B/k17_varbench/ouro"), os.path.join(root, "A/steer_gen/a1_gsm8k")
    for d in (vb, os.path.join(gen, "chk"), os.path.join(gen, "r00"), os.path.join(root, "_done")):
        os.makedirs(d, exist_ok=True)
    k = {"hook_identity": {"ouro": {"identity_ok": True, "call_counts_ok": True, "zero_add_every_site_bit_identical": True,
                                    "wrapper_bit_identical": True, "repeat_bit_identical": True, "all_ok": True}},
         "pc_final": {"ouro": {"exact": pc_exact, "n": 3}}, "structural_zero": {"ouro": {"exact": 6, "n": 6}},
         "edit_placement": {"ouro": {"ok": True}}, "causality": {"ouro": {"rows_ok": 66, "rows": 66}},
         "items": {"ouro": {"kept": 3}}}
    json.dump(k, open(os.path.join(vb, "checks.json"), "w"))
    json.dump({ds: {"planned": 3, "merged": 3 if union else 2, "union_equals_planned": union, "duplicates": 0}
               for ds in ("varbench", "vbd")}, open(os.path.join(vb, "merge_ouro.json"), "w"))
    json.dump({}, open(os.path.join(vb, "vb_ouro.json"), "w"))
    json.dump({g: {"pass": gate} for g in ("gate_zero_edit", "gate_placement", "gate_cache_equiv", "gate_identities")},
              open(os.path.join(gen, "chk", "checks_a1.json"), "w"))
    c = json.load(open(os.path.join(gen, "chk", "checks_a1.json")))
    c["gate_zero_edit"].update({"inert_hooks_bit_identical": True, "prefill_bit_identical": True})
    json.dump(c, open(os.path.join(gen, "chk", "checks_a1.json"), "w"))
    json.dump({"scored": [{"idx": 1}]}, open(os.path.join(gen, "items.json"), "w"))
    json.dump({"exit": exit_code}, open(os.path.join(gen, "r00", "steer_job.json"), "w"))
    open(os.path.join(gen, "r00", "cells_x.jsonl"), "w").write("{}\n")
    json.dump({}, open(os.path.join(gen, "analysis.json"), "w"))
    for d in (vb, gen):
        open(os.path.join(d, "results.csv"), "w").write("a,b\n1,2\n")
    for rel, m in (("B/k17_public/mcl_nr_D1", "mcl_nr"), ("B/k17_parcae/parcae-370m", "parcae-370m")):
        d = os.path.join(root, rel)
        os.makedirs(d, exist_ok=True)
        json.dump({"hook_identity": {m: {"all_ok": True, "b_zero_edit_identity": True}},
                   "pc_final": {m + "|D1": {"all_exact": pc_exact == 3}}, "sz": {m + "|D1": {"all_exact": True}},
                   "pc_site": {m + "|D1": {"all_exact": True}}}, open(os.path.join(d, "checks.json"), "w"))
        open(os.path.join(d, "results.csv"), "w").write("dataset,arm,estimate\nD1,O1,0.5\n")
    d = os.path.join(root, "C/k17_freeze_ifm/ifm_p1_ouro_D1")
    os.makedirs(d, exist_ok=True)
    json.dump({f: True for f in ("zero_edit_bit_identical", "freeze_no_edit_bit_identical",
                                 "source_attn_frozen_e_zero_bit_exact", "pc_exact_all", "sz_exact_all",
                                 "readout_identity_all")}, open(os.path.join(d, "checks.json"), "w"))
    open(os.path.join(d, "results.csv"), "w").write("dataset,arm,estimate\nD1,E0,0.0\n")
    d = os.path.join(root, "A/steer_gen_raven/mcleish_smoke")
    os.makedirs(os.path.join(d, "chk"), exist_ok=True)
    os.makedirs(os.path.join(d, "r00"), exist_ok=True)
    json.dump({"pass": gate, "cases": {"zero_edit_eq_A": {"pass": gate}, "source1_differs_A": {"note": "reported"}}},
              open(os.path.join(d, "chk", "checks_mcl.json"), "w"))
    json.dump({"exit": exit_code}, open(os.path.join(d, "r00", "steer_job.json"), "w"))
    open(os.path.join(d, "r00", "cells_x.jsonl"), "w").write("{}\n")
    for g in ("g1_copy", "g2_gen", "fam"):
        for ln in open(os.path.join(JOBS, "smoke_%s.tsv" % g), encoding="utf-8"):
            open(os.path.join(root, "_done", ln.split("\t")[0]), "w").close()


@pytest.mark.parametrize("case,ok", [({}, True), ({"pc_exact": 2}, False), ({"union": False}, False),
                                     ({"gate": False}, False), ({"exit_code": 3}, False)])
def test_smoke_check_correctness_rule(tmp_path, case, ok):
    root = str(tmp_path / "smoke")
    _smoke_tree(root, **case)
    r = subprocess.run([sys.executable, "-B", os.path.join(CL, "env", "smoke_check.py"), "--root", root, "--lists", JOBS,
                        "--report", str(tmp_path / "rep.json")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert (r.returncode == 0) == ok, r.stdout.decode()
    rep = json.load(open(str(tmp_path / "rep.json")))
    assert rep["ok"] == ok and (ok or rep["failed"])
    os.remove(os.path.join(root, "_done", sorted(os.listdir(os.path.join(root, "_done")))[0]))
    r = subprocess.run([sys.executable, "-B", os.path.join(CL, "env", "smoke_check.py"), "--root", root, "--lists", JOBS],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1                      # a stage without its done marker fails the smoke


def test_smoke_hash_compare_tolerance(tmp_path):
    sys.path.insert(0, CL)
    import smoke_hash as SHH
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "x").mkdir(parents=True)
    ref = {"e": [1.0, 2.0, 100.0], "small": 1e-6, "n": 3, "tok": [5, 6], "seconds": 1.0}
    json.dump(ref, open(str(a / "x" / "r.json"), "w"))
    near = {"e": [1.0 + 5e-4, 2.0 * (1 + 9e-4), 100.05], "small": 9e-5, "n": 3, "tok": [5, 6], "seconds": 9.0}
    json.dump(near, open(str(b / "x" / "r.json"), "w"))
    rep = SHH.compare(str(a), str(b))
    assert rep["within_tolerance"] and rep["n_out_of_tolerance"] == 0 and rep["n_float_compared"] == 4
    far = dict(near, e=[1.01, 2.0, 100.0], n=4, tok=[5, 7])
    json.dump(far, open(str(b / "x" / "r.json"), "w"))
    diff = tmp_path / "diff.tsv"
    rep = SHH.compare(str(a), str(b), out_path=str(diff))
    assert not rep["within_tolerance"] and rep["n_out_of_tolerance"] == 1 and rep["n_exact_differs"] == 2
    lines = diff.read_text().splitlines()
    assert lines[0].split("\t") == ["file", "field", "expected", "got", "abs_diff", "kind"]
    assert any(ln.startswith("x/r.json\t/e[0]\t1.0\t1.01\t0.01\tout_of_tolerance") for ln in lines)


def _gate_files(root, shift=0.0):
    """The gated records of a smoke tree: both results.csv files and vb_ouro.json (shift moves one margin)."""
    sys.path.insert(0, CL)
    import smoke_hash as SHH
    vb, gen = os.path.join(root, SHH.VB_DIR), os.path.join(root, SHH.GEN_DIR)
    cols = list(SHH.ID_COLS) + list(SHH.E_COLS)
    for d, rows in ((vb, [["B", "edit_site_map", "k17_varbench.py", "ouro", "gsm8k_copy", "O1", "src|L0", "1", "1", "e",
                           "e_LO", "own", "k1", "0.9426", "-0.0015", "1.0"],
                          ["B", "edit_site_map", "k17_varbench.py", "ouro", "vbd", "O1", "ans|Llast", "4", "1", "e", "e",
                           "own", "k2", "0.5", "", "1.0"]]),
                    (gen, [["A", "generation placement", "steer_gen.py", "ouro_1_4b_base", "gsm8k", "r00:clean_A", "none",
                            "", "", "gold_B", "parsed_answer", "own", "k3", "0.0", "", ""]])):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(cols)
            w.writerows(rows)
    it = {"clean_A": [10.0, 8.0, 10.0, 1], "clean_B": [8.0, 10.5, 10.5, 2], "clean_A_loops": [[9.0, 8.5, 9.0, 1]],
          "rows": {"O1|src|L0@1|1": {"f": [8.2, 10.3 + shift, 10.3, 2], "L": [[9.0, 9.1, 9.1, 2]]}}}
    json.dump({"varbench": {"items": {"p1": it}}, "vbd": {"items": {}}}, open(os.path.join(vb, "vb_ouro.json"), "w"))


def _gate_reference(tmp_path):
    """A recorded smoke (identity ok, gated records), its SMOKE_HASHES-style file and the SMOKE_VALUES file."""
    sys.path.insert(0, CL)
    import smoke_hash as SHH
    ref = str(tmp_path / "ref")
    _smoke_tree(ref)
    _gate_files(ref)
    hashes, values = str(tmp_path / "H.txt"), str(tmp_path / "V.json")
    SHH.write(ref, hashes)
    SHH.write_values(ref, hashes, "", values)
    return ref, hashes, values


def test_smoke_values_file_is_bound_to_smoke_hashes():
    """cluster/SMOKE_VALUES.json: every source value hash is the SMOKE_HASHES.txt line of that file (out_new/)."""
    sys.path.insert(0, CL)
    import smoke_hash as SHH
    v = json.load(open(os.path.join(CL, "SMOKE_VALUES.json"), encoding="utf-8"))
    rec = SHH.hash_lines(os.path.join(CL, "SMOKE_HASHES.txt"))
    assert sorted(v["sources"]) == sorted(SHH.GATE_SOURCES) and v["tol_abs"] == 1e-3 and v["n_values"] == len(v["values"])
    for s, x in v["sources"].items():
        assert x["hashes_path"] == "out_new/" + s and rec[x["hashes_path"]] == x["value_sha256"]
    kinds = set(k.split("|")[0] + "|" + k.split("|")[1] for k in v["values"])
    assert kinds == {"e|k17_varbench", "e|steer_gen", "margin|k17_varbench"}


@pytest.mark.parametrize("case", ["pass", "tolerance_fail", "override", "identity_fail"])
def test_smoke_gate_identity_tolerance_override(tmp_path, case):
    """smoke_check.py --values --out-root: pass (|d| <= 0.001) writes _smoke.ok; a margin 0.002 off exits 3 with
    _smoke_fail.txt and no _smoke.ok; the same with an override reason writes _smoke.ok and _smoke_override.txt
    (reason and full diff, also printed); an identity failure exits 1 even with an override reason."""
    ref, hashes, values = _gate_reference(tmp_path)
    new, out = str(tmp_path / "new"), str(tmp_path / "out")
    os.makedirs(out)
    _smoke_tree(new, pc_exact=2 if case == "identity_fail" else 3)
    _gate_files(new, shift={"pass": 0.0009}.get(case, 0.002))
    cmd = [sys.executable, "-B", os.path.join(CL, "env", "smoke_check.py"), "--root", new, "--lists", JOBS,
           "--report", os.path.join(out, "rep.json"), "--values", values, "--hashes", hashes, "--out-root", out]
    if case in ("override", "identity_fail"):
        cmd += ["--override-reason", "known cross-GPU drift, checked by hand"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    ok, ovr, fail = (os.path.exists(os.path.join(out, f)) for f in ("_smoke.ok", "_smoke_override.txt", "_smoke_fail.txt"))
    key = "margin|k17_varbench|varbench|p1|rows/O1|src|L0@1|1/f"
    rep = json.load(open(os.path.join(out, "rep.json")))
    if case == "pass":
        assert r.returncode == 0 and ok and not ovr and not fail, so
        assert rep["identity_ok"] and rep["tolerance_ok"] and rep["tolerance"]["n_fail"] == 0
        assert json.load(open(os.path.join(out, "_smoke.ok")))["override"] is None
    elif case == "tolerance_fail":
        assert r.returncode == 3 and not ok and not ovr and fail, so
        assert rep["identity_ok"] and not rep["tolerance_ok"] and rep["tolerance"]["counts"]["out_of_tolerance"] == 1
        assert "SMOKE_OVERRIDE_REASON" in open(os.path.join(out, "_smoke_fail.txt")).read()
        assert key in open(os.path.join(out, "_smoke_tolerance_diff.tsv")).read()
    elif case == "override":
        assert r.returncode == 0 and ok and ovr and not fail, so
        txt = open(os.path.join(out, "_smoke_override.txt")).read()
        assert "reason: known cross-GPU drift, checked by hand" in txt and key in txt and "out_of_tolerance" in txt
        assert "reason: known cross-GPU drift" in so and key in so                       # printed too
        assert json.load(open(os.path.join(out, "_smoke.ok")))["override"]["reason"].startswith("known cross-GPU")
    else:
        assert r.returncode == 1 and not ok and not ovr and fail, so
        assert "cannot override" in open(os.path.join(out, "_smoke_fail.txt")).read()
    # the gate also fails when SMOKE_VALUES.json no longer matches the hash file it was taken from
    if case == "pass":
        txt = open(hashes).read().replace(json.load(open(values))["sources"]["B/k17_varbench/ouro/results.csv"]
                                          ["value_sha256"], "0" * 64)
        open(hashes, "w").write(txt)
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert r.returncode == 3 and "binding" in open(os.path.join(out, "_smoke_fail.txt")).read()


def _fake_smoke_repo(tmp_path, prepared):
    """A repository with the real smoke_cluster.sh, smoke_check.py and smoke_hash.py; the smoke lists and templates
    are stand-ins (the first line copies a prepared smoke tree into $OUT_ROOT, the template marks the line done)."""
    repo = tmp_path / "repo"
    cl = repo / "loopedcot" / "cluster"
    (cl / "env").mkdir(parents=True)
    (cl / "jobs").mkdir()
    for f in ("env/smoke_cluster.sh", "env/smoke_check.py", "smoke_hash.py"):
        shutil.copy(os.path.join(CL, f), str(cl / f))
    tpl = ('#!/bin/bash\nset -euo pipefail\nLIST="$1"\nCMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST" | cut -f2-)\n'
           'ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST" | cut -f1)\nmkdir -p "$OUT_ROOT/_done"\neval "$CMD"\n'
           'touch "$OUT_ROOT/_done/$ID"\n')
    (cl / "jobs" / "task_g3_map.sbatch").write_bytes(tpl.encode())          # the template of the family list
    for g in ("g1_copy", "g2_gen", "fam"):
        (cl / "jobs" / ("task_%s.sbatch" % g)).write_bytes(tpl.encode())
        ids = [ln.split("\t")[0] for ln in open(os.path.join(JOBS, "smoke_%s.tsv" % g), encoding="utf-8") if ln.strip()]
        lines = ["%s\t%s" % (i, 'cp -r "%s/." "$OUT_ROOT/"' % prepared if (g, n) == ("g1_copy", 0) else "true")
                 for n, i in enumerate(ids)]
        (cl / "jobs" / ("smoke_%s.tsv" % g)).write_bytes(("\n".join(lines) + "\n").encode())
    h = hashlib.sha256(open(str(cl / "env" / "smoke_check.py"), "rb").read()).hexdigest()
    (cl / "jobs" / "scripts_sha256.txt").write_bytes(("%s  loopedcot/cluster/env/smoke_check.py\n" % h).encode())
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_bytes(b'#!/bin/bash\necho "GPU 0: test card, 1 MiB"\n')
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_bytes(('#!/bin/bash\nexec "%s" -B "$@"\n' % sys.executable.replace("\\", "/")).encode())
    for p in (bindir / "nvidia-smi", venv / "python"):
        os.chmod(str(p), 0o755)
    return repo, cl, bindir, venv.parent


@needs_bash
@pytest.mark.parametrize("case", ["pass", "tolerance_fail", "override", "identity_fail", "stage_fail"])
def test_smoke_cluster_sh_gate_and_override(tmp_path, case):
    """smoke_cluster.sh end to end with stand-in task lines, per family: a tolerance miss holds the family whose values
    miss (SMOKE_OVERRIDE_REASON passes it instead, recorded), an identity failure holds its family (no override), a
    failing smoke line holds its family and the others go on; _smoke.ok lists the passed and held families."""
    ref, hashes, values = _gate_reference(tmp_path)
    prepared = (tmp_path / "prepared").as_posix()
    _smoke_tree(prepared, pc_exact=2 if case == "identity_fail" else 3)
    _gate_files(prepared, shift={"pass": 0.0}.get(case, 0.002))
    shutil.rmtree(os.path.join(prepared, "_done"))
    repo, cl, bindir, venv = _fake_smoke_repo(tmp_path, prepared)
    if case == "stage_fail":          # the Parcae line fails (the line names its venv, as the real lines do)
        fl = (cl / "jobs" / "smoke_fam.tsv").read_text().splitlines()
        fl = [ln.split("\t")[0] + "\techo VENV_PARCAE; false" if "f_parcae" in ln else ln for ln in fl]
        (cl / "jobs" / "smoke_fam.tsv").write_bytes(("\n".join(fl) + "\n").encode())
    shutil.copy(hashes, str(cl / "SMOKE_HASHES.txt"))
    shutil.copy(values, str(cl / "SMOKE_VALUES.json"))
    out = (tmp_path / "out").as_posix()
    os.makedirs(out)
    env = dict(os.environ, REPO=repo.as_posix(), OUT_ROOT=out, VENV_OURO=venv.as_posix(), PARTITION="gpuq",
               PATH=bindir.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    env.pop("MODULES", None)
    env.pop("SMOKE_OVERRIDE_REASON", None)
    env.pop("SMOKE_REF", None)
    if case in ("override", "identity_fail"):
        env["SMOKE_OVERRIDE_REASON"] = "operator: drift reviewed"
    r = subprocess.run([_bash(), (cl / "env" / "smoke_cluster.sh").as_posix()], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    ok, ovr = os.path.exists(out + "/_smoke.ok"), os.path.exists(out + "/_smoke_override.txt")
    held = sorted(f[:-5] for f in os.listdir(out + "/_held") if f.endswith(".held")) if os.path.isdir(out + "/_held") else []
    rec = json.load(open(out + "/_smoke.ok")) if ok else {}
    if case == "pass":
        assert r.returncode == 0 and ok and not ovr and not held and "SMOKE PASSED:" in so, so[-3000:]
    elif case == "tolerance_fail":
        assert r.returncode == 0 and ok and held == ["ouro"] and rec["units_held"] == ["ouro"], so[-3000:]
        assert "cross-host tolerance" in open(out + "/_held/ouro.held").read() and "parcae" in rec["units_passed"]
    elif case == "override":
        assert r.returncode == 0 and ok and ovr and not held and "OPERATOR OVERRIDE" in so and "operator: drift reviewed" in so, so[-3000:]
    elif case == "identity_fail":
        assert r.returncode == 0 and ok and ovr and "ouro" in held, so[-3000:]
        assert "identity check failed: varbench pc_final exact" in open(out + "/_held/ouro.held").read()
        assert "ouro" not in rec["units_passed"]
    else:
        assert r.returncode == 0 and ok and held == ["ouro", "parcae"] or held == ["parcae"], (held, so[-3000:])
        assert "smoke stage smoke.fam.f_parcae.001 exited 1" in open(out + "/_held/parcae.held").read()
        assert "family parcae held, the others go on" in so and "smoke.fam.f_ifm.001" in so
        assert "parcae" not in rec["units_passed"] and "ifm" in rec["units_passed"]


def _setup_tree(tmp_path):
    lc = tmp_path / "proj" / "repo" / "loopedcot" / "cluster"
    (lc / "jobs").mkdir(parents=True)
    shutil.copytree(os.path.join(CL, "env"), str(lc / "env"), ignore=shutil.ignore_patterns("cluster.env", "__pycache__"))
    shutil.copy(os.path.join(CL, "weights.json"), str(lc / "weights.json"))
    for f in glob.glob(os.path.join(JOBS, "task_*.sbatch")) + glob.glob(os.path.join(JOBS, "g2_gen.*.tsv")):
        shutil.copy(f, str(lc / "jobs"))
    fake = tmp_path / "bin"
    fake.mkdir()
    for name, body in (("sinfo", 'printf "n01\\nn02\\nn03\\nn04\\nn05\\nn06\\nn05\\n"'),
                       ("scontrol", 'echo "$3" | tr "," "\\n"')):
        (fake / name).write_bytes(("#!/bin/bash\n%s\n" % body).encode())
        os.chmod(str(fake / name), 0o755)
    return lc, fake


@needs_bash
def test_setup_cluster_dry_run_headers_env_and_home_refusal(tmp_path):
    lc, fake = _setup_tree(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    proj = (tmp_path / "proj").as_posix()
    env = dict(os.environ, HOME=home.as_posix(), PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for k in ("OUT_ROOT", "HF_HOME", "WEIGHTS_ROOT", "PARTITION", "ACCOUNT", "NODELIST", "TORCH_CUDA_ARCH_LIST"):
        env.pop(k, None)
    py = sys.executable.replace("\\", "/")
    setup = (lc / "env" / "setup_cluster.sh").as_posix()
    base = [_bash(), setup, "--dry-run", "--python", py, "--hf-home", proj + "/hf", "--weights-root", proj + "/ckpt",
            "--partition", "gpuq", "--account", "acct1", "--nodes-expected", "4"]
    # the defaults: 6 nodes, 48 cards; a partition that holds exactly the run's nodes needs no --exclude
    r = subprocess.run(base[:-2] + ["--out-root", proj + "/out6", "--nodelist", "n01,n02,n03,n04,n05,n06"], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    e6 = open(proj + "/out6/cluster.env", encoding="utf-8").read()
    assert "NODES_EXPECTED=6" in e6 and "CARDS=48" in e6 and 'SLURM_RESTRICT="--partition=gpuq"' in e6, r.stdout.decode()[-1500:]
    # a folder inside the home directory: refused before anything is created
    r = subprocess.run(base + ["--out-root", home.as_posix() + "/out", "--nodelist", "n01,n02,n03,n04"], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 2 and b"inside a home directory" in r.stdout and not (home / "out").exists()
    # three nodes: refused at the checklist
    r = subprocess.run(base + ["--out-root", proj + "/out", "--nodelist", "n01,n02,n03"], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1 and b"needs at least 4 (--nodes-expected)" in r.stdout
    r = subprocess.run(base + ["--out-root", proj + "/out", "--nodelist", "n01,n02,n03,n04"], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode()
    assert "WOULD_DOWNLOAD ouro_1_4b_base" in out and "WOULD_DOWNLOAD ifm_p1_ouro" in out and "[dry-run]" in out
    assert os.path.isdir(proj + "/out/logs") and os.path.exists(proj + "/out/_env.json")
    # setup edits no tracked file: the templates keep their placeholders, cluster.env goes to $OUT_ROOT
    for t in (lc / "jobs").glob("task_*.sbatch"):
        assert t.read_bytes() == open(os.path.join(JOBS, t.name), "rb").read()
    assert not (lc / "env" / "cluster.env").exists()
    e = open(proj + "/out/cluster.env", encoding="utf-8").read()
    for v in ("REPO", "OUT_ROOT", "LOG_ROOT", "DATA_ROOT", "HF_HOME", "HF_HUB_OFFLINE", "WEIGHTS_ROOT", "IFM_CKPT_ROOT",
              "XLLM_ROOT", "VENV_OURO", "VENV_RAVEN", "VENV_PARCAE", "VENV_IFM", "RUN_ROLE", "VB_PLAN", "VB_PLAN_SHA",
              "PUB_PLAN", "PAR_PLAN", "FRZ_PLAN", "FRZ_PLAN_SHA", "FRZO_PLAN", "FRZO_PLAN_SHA", "SETTLE",
              "GEN_MEM_FRACTION", "CARDS", "PARTITION", "ACCOUNT", "NODELIST", "SLURM_RESTRICT", "MODULES",
              "NODES_EXPECTED"):
        assert re.search(r"(^export | )%s=" % v, e, re.M), v
    assert 'NODELIST="n01,n02,n03,n04"' in e and 'SLURM_RESTRICT="--exclude=n05,n06"' in e and "CARDS=32" in e
    assert 'MODULES=""' in e and "NODES_EXPECTED=4" in e
    # every variable the generated templates and submit scripts read from the environment is exported by cluster.env
    names = set()
    for f in glob.glob(os.path.join(JOBS, "task_*.sbatch")) + glob.glob(os.path.join(JOBS, "g*.tsv")) + \
            [os.path.join(JOBS, "submit_00_xllm.sh"), os.path.join(JOBS, "submit_10_rest.sh")] + \
            glob.glob(os.path.join(JOBS, "smoke_*.tsv")):
        names |= set(re.findall(r"\$\{?([A-Z][A-Z0-9_]+)", open(f, encoding="utf-8").read()))
    internal = {"SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID", "SLURMD_NODENAME", "HOME", "LIST", "CMD",
                "ID", "J", "ALL", "CARDS", "OUT_ROOT", "SN", "TL", "CHECK", "SB", "BAD", "GPUS", "NODE", "RESTRICT",
                "SLURM_RESTART_COUNT", "HOME_RUN"}
    names -= internal | {n for n in names if re.match(r"^J\d+$", n)}
    exported = set(re.findall(r"\b([A-Z][A-Z0-9_]+)=", e))
    assert not names - exported, sorted(names - exported)
    # --nodes-expected replaces the fixed four; --modules lands in cluster.env
    r = subprocess.run(base + ["--out-root", proj + "/out3", "--nodelist", "n01,n02,n03", "--nodes-expected", "3",
                               "--modules", "cuda/12.4 gcc/12"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert b"needs at least" not in r.stdout, r.stdout.decode()[-2000:]
    e3 = open(proj + "/out3/cluster.env", encoding="utf-8").read()
    assert 'MODULES="cuda/12.4 gcc/12"' in e3 and "NODES_EXPECTED=3" in e3 and "CARDS=24" in e3
    assert 'SLURM_RESTRICT="--exclude=n04,n05,n06"' in e3
    # more nodes than needed: accepted, the probe step keeps the first healthy ones; cluster.env keeps the base
    # restriction and reads the probe's choice ($OUT_ROOT/run_nodes.env) when it exists
    r = subprocess.run(base + ["--out-root", proj + "/out4", "--nodelist", "n01,n02,n03,n04,n05,n06"], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert "keeps the first 4 healthy ones" in so and "needs at least" not in so, so[-2000:]
    e4 = open(proj + "/out4/cluster.env", encoding="utf-8").read()
    assert 'SLURM_RESTRICT_BASE="--partition=gpuq"' in e4 and "/out4/run_nodes.env" in e4 and "CARDS=32" in e4
    with open(proj + "/out4/run_nodes.env", "w") as f:
        f.write('export SLURM_RESTRICT="--partition=gpuq --exclude=n05,n06"\n')
    got = subprocess.run([_bash(), "-c", '. %s/out4/cluster.env && echo "$SLURM_RESTRICT|$SLURM_RESTRICT_BASE"' % proj],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert got.returncode == 0 and got.stdout.decode().strip() == "--partition=gpuq --exclude=n05,n06|--partition=gpuq"


# ------------------------------------------------------------------ launch chain, status, no-local-done lists
def test_no_local_done_lists_keep_ids():
    """gen_jobs.py --no-local-done (jobs/nolocal/): every job of jobs/joblist.tsv with the same id and command, plus
    the done_local jobs under the ids jobs/done_local.tsv gives them; no staging gate in its submit_10_rest.sh."""
    nl_dir = os.path.join(JOBS, "nolocal")
    with open(os.path.join(nl_dir, "joblist.tsv"), encoding="utf-8") as f:
        nl = {r["id"]: r["cmd"] for r in csv.DictReader(f, delimiter="\t")}
    main = {r["id"]: r["cmd"] for r in joblist()}
    with open(os.path.join(JOBS, "done_local.tsv"), encoding="utf-8") as f:
        done = [r["id"] for r in csv.DictReader(f, delimiter="\t")]
    assert all(nl.get(k) == v for k, v in main.items())
    assert set(nl) - set(main) == set(done) and len(done) > 0
    assert not os.path.exists(os.path.join(nl_dir, "held_without_staging.tsv"))
    mcl = [c for c in nl.values() if "steer_gen_raven.py run" in c and "--setting hug" not in c]
    assert len(mcl) == 22 and all("/mcleish_setting/items.json" in c and "mv -f" in c for c in mcl)
    s1 = open(os.path.join(nl_dir, "submit_10_rest.sh"), encoding="utf-8").read()
    assert "_staged.ok" not in s1 and "task_g1_copy.sbatch" in s1
    assert not os.path.exists(os.path.join(nl_dir, "scripts_sha256.txt"))


def _chain_env(tmp_path, done_src="none"):
    fake, log = _fake_sbatch(tmp_path)
    open(log, "w").close()
    out = tmp_path / "out"
    out.mkdir()
    env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(),
               DONE_SRC=done_src, VENV_OURO="/proj/venvs/ouro", PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""),
               **SLURM_ENV)
    env.pop("MODULES", None)
    return env, out, log


@needs_bash
def test_submit_chain_order_dependencies_and_refusals(tmp_path):
    """submit_chain.sh: four sbatch calls, each afterok on the one before, every call with partition, account and
    the node restriction; DONE_SRC=none uses jobs/nolocal/. The steps refuse on the previous step's failure marker
    and write _chain/<step>.fail and _diag/chain.<step>.txt."""
    env, out, log = _chain_env(tmp_path)
    chain = os.path.join(JOBS, "submit_chain.sh").replace("\\", "/")
    r = subprocess.run([_bash(), chain], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    calls = open(log).read().splitlines()
    nodes = SLURM_ENV["NODELIST"].split(",")
    assert len(calls) == 2 + len(nodes), calls
    assert "--step setup_check" in calls[0] and "afterok" not in calls[0]
    assert "--partition=gpuq --account=acct1 --exclude=n05,n06" in calls[0] and "--kill-on-invalid-dep=yes" in calls[0]
    for n, c in zip(nodes, calls[1:-1]):
        assert "--nodelist=%s" % n in c and "--dependency=afterok:1001" in c and "--time=0:05:00" in c and "--gres=gpu:1" in c
        assert "nvidia-smi -L" in c and "torch" in c and "RC=" in c and "/_probe/%s.out" % n in c
    ids = ":".join(str(1002 + i) for i in range(len(nodes)))
    assert "--dependency=afterok:1001,afterany:%s" % ids in calls[-1] and "--step probe_collect" in calls[-1]
    jt = open(str(out / "_chain" / "jobs.txt")).read()
    assert jt.rstrip().endswith("/cluster/jobs/nolocal")
    # probe_collect: good, RC non-zero, each signature, no verdict; then g0, smoke, stage_submit without the bad nodes
    (out / "_chain" / "setup_check.ok").write_text("ok\n")
    pr = out / "_probe"
    pr.mkdir(exist_ok=True)
    (pr / "n01.out").write_text("node=n01\nGPU 0: card\nmatmul card 1.0\nRC=0\n")
    (pr / "n02.out").write_text("node=n02\nGPU 0: card\nTraceback\nRC=1\n")
    (pr / "n03.out").write_text("node=n03\nRuntimeError: CUDA unknown error - this may be due to ...\nRC=0\n")
    (pr / "n04.out").write_text("node=n04\nNo devices were found\nRC=1\n")
    env_p = dict(env, NODELIST="n01,n02,n03,n04,n07")
    open(log, "w").close()
    r = subprocess.run([_bash(), chain, "--step", "probe_collect"], env=env_p, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    assert (out / "bad_nodes.txt").read_text().split() == ["n02", "n03", "n04"]
    assert "unknown n07" in so and "ok      n01" in so
    calls = open(log).read().splitlines()
    assert len(calls) == 3 and all("--exclude=n05,n06,n02,n03,n04" in c for c in calls), calls
    assert "task_g0_xllm.sbatch" in calls[0] and "afterok:1001" in calls[1] and "--step smoke" in calls[1]
    assert "afterok:1002" in calls[2] and "--step stage_submit" in calls[2]
    st = subprocess.run([_bash(), os.path.join(JOBS, "status.sh").replace("\\", "/")], env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT).stdout.decode()
    assert "excluded from every later submission): n02,n03,n04" in st and "probe n04: RC=1 No devices were found" in st
    # DONE_SRC missing: refused before any sbatch call, with a diagnosis
    (tmp_path / "b").mkdir()
    env2, out2, log2 = _chain_env(tmp_path / "b", done_src="")
    r = subprocess.run([_bash(), chain], env=env2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 5 and not open(log2).read() and (out2 / "_diag" / "submit_chain.txt").exists()
    # steps: smoke refuses on setup_check.fail; stage_submit refuses on _smoke_fail.txt
    (out / "_chain" / "setup_check.fail").write_text("exit 1: x\n")
    r = subprocess.run([_bash(), chain, "--step", "smoke"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 4 and b"REFUSED: previous step failed" in r.stdout
    assert (out / "_chain" / "smoke.fail").exists()
    d = (out / "_diag" / "chain.smoke.txt").read_text()
    assert "exit code: 4" in d and "failing check: previous step failed" in d and "--- last 50 lines" in d
    (out / "_smoke_fail.txt").write_text("identity\n")
    r = subprocess.run([_bash(), chain, "--step", "stage_submit"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 4 and (out / "_chain" / "stage_submit.fail").exists()
    # setup_check: an incomplete cluster.env fails with every problem named
    env3 = dict(env, VENV_OURO="<VENV_OURO>")
    r = subprocess.run([_bash(), chain, "--step", "setup_check"], env=env3, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1 and (out / "_chain" / "setup_check.fail").exists()
    assert "VENV_OURO is a placeholder" in (out / "_diag" / "chain.setup_check.txt").read_text()


@needs_bash
def test_status_counts_failed_ids_and_diagnoses(tmp_path):
    """status.sh: per group done / running / pending / failed from the markers, the failed ids, every diagnosis."""
    out = tmp_path / "out"
    for d in ("_done", "_diag", "_tasklog"):
        (out / d).mkdir(parents=True)
    rows = joblist()
    g1 = [r["id"] for r in rows if r["group"] == "g1_copy"]
    (out / "_done" / g1[0]).write_text("")
    (out / "_diag" / (g1[1] + ".txt")).write_text("exit code: 3\nfailing check: task command\n")
    (out / "_tasklog" / (g1[1] + ".log")).write_text("x\n")
    (out / "_tasklog" / (g1[2] + ".log")).write_text("x\n")
    (out / "_diag" / "chain.smoke.txt").write_text("exit code: 4\n")
    env = dict(os.environ, OUT_ROOT=out.as_posix())
    r = subprocess.run([_bash(), os.path.join(JOBS, "status.sh").replace("\\", "/")], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so
    line = [ln for ln in so.splitlines() if ln.startswith("g1_copy ")][0].split()
    assert line[1:6] == [str(len(g1)), "1", "1", str(len(g1) - 3), "1"], line
    assert "failed task ids (1):" in so and "  " + g1[1] in so
    assert "================ diagnosis %s" % g1[1] in so and "================ diagnosis chain.smoke" in so
    assert "diagnosis files: 2" in so


def test_local_vs_cluster_report(tmp_path):
    """merge.local_vs_cluster: per (folder, dataset) the largest absolute difference of estimate, bounds and controls,
    n_items / n_clusters equality, rows only on one side, and 'not run' for a folder without a cluster results.csv."""
    ref, out, dest = tmp_path / "ref", tmp_path / "out", tmp_path / "merged"
    cols = list(MG.LVC_KEY) + ["n_items", "n_clusters"] + list(MG.LVC_FLOAT)

    def write(path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def row(arm, est, n=10, lo=0.1, rc=""):
        d = {k: "x" for k in MG.LVC_KEY}
        d.update(dataset="D1", arm=arm, n_items=n, n_clusters=5, estimate=est, ci90_lo=lo, ci90_hi=0.9, ci95_lo=0.05,
                 ci95_hi=0.95, random_control_estimate=rc, positive_control_value=1.0)
        return d
    write(str(ref / "B/s/m1/results.csv"), [row("a", 0.5), row("b", 0.2), row("c", 0.3)])
    write(str(ref / "B/s/m2/results.csv"), [row("a", 0.5)])
    json.dump({"cells": {"B/s/m1": {}, "B/s/m2": {}}}, open(str(ref / "MANIFEST.json"), "w"))
    write(str(out / "B/s/m1/results.csv"), [row("a", 0.5004), row("b", 0.2, n=9, lo=0.13, rc=0.01), row("d", 0.1)])
    os.makedirs(str(dest))
    n = MG.local_vs_cluster(str(out), str(dest), ref_dir=str(ref))
    with open(str(dest / "local_vs_cluster.csv"), encoding="utf-8") as f:
        rows = {r["out"]: r for r in csv.DictReader(f)}
    assert n == 2 and rows["B/s/m2"]["status"] == "not run on the cluster"
    r = rows["B/s/m1"]
    assert r["status"] == "compared" and r["rows_matched"] == "2" and r["rows_only_local"] == "1" and r["rows_only_cluster"] == "1"
    assert abs(float(r["max_abs_diff"]) - 0.03) < 1e-9 and r["max_abs_diff_field"] == "ci90_lo"
    assert abs(float(r["max_abs_diff_estimate"]) - 0.0004) < 1e-9 and r["undefined_on_one_side"] == "1"
    assert r["n_items_equal"] == "False" and r["n_clusters_equal"] == "True"
    # status.sh prints the report when the merge has written it
    for d in ("_done", "_diag", "_tasklog"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "merged").mkdir()
    shutil.copy(str(dest / "local_vs_cluster.csv"), str(out / "merged" / "local_vs_cluster.csv"))
    if _bash():
        p = subprocess.run([_bash(), os.path.join(JOBS, "status.sh").replace("\\", "/")], env=dict(os.environ, OUT_ROOT=out.as_posix()),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert b"cluster rerun against the local reference" in p.stdout and b"B/s/m1,D1,compared" in p.stdout


def test_reference_local_manifest():
    """cluster/reference_local: every file has the sha256 its manifest records; every folder is a job-list output."""
    man = json.load(open(os.path.join(CL, "reference_local", "MANIFEST.json"), encoding="utf-8"))["cells"]
    outs = {r["out"].replace("$OUT_ROOT/", "") for r in joblist()}
    with open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8") as f:
        outs |= {r["out"].replace("$OUT_ROOT/", "") for r in csv.DictReader(f, delimiter="\t")}
    assert len(man) >= 5
    for rel, e in man.items():
        p = os.path.join(CL, "reference_local", rel, "results.csv")
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == e["results_sha256"] and rel in outs, rel



@needs_bash
@needs_bash
def test_probe_collect_keeps_first_healthy_nodes(tmp_path):
    """probe_collect with more nodes than NODES_EXPECTED: the first healthy nodes in NODELIST order run, nodes with no
    verdict fill up only when too few are healthy, every other node is excluded from the base restriction (a stale
    SLURM_RESTRICT is ignored), run_nodes.env and run_nodes.txt record the choice and status.sh prints it."""
    env, out, log = _chain_env(tmp_path)
    chain = os.path.join(JOBS, "submit_chain.sh").replace("\\", "/")
    (out / "_chain").mkdir()
    (out / "_chain" / "setup_check.ok").write_text("ok\n")
    pr = out / "_probe"
    pr.mkdir()
    (pr / "n01.out").write_text("node=n01\nTraceback\nRC=1\n")
    for n in ("n02", "n04", "n07", "n08"):
        (pr / ("%s.out" % n)).write_text("node=%s\nGPU 0: card\nmatmul card 1.0\nRC=0\n" % n)
    env_p = dict(env, NODELIST="n01,n02,n03,n04,n07,n08", NODES_EXPECTED="2", SLURM_RESTRICT_BASE="--exclude=n05,n06",
                 SLURM_RESTRICT="--exclude=n05,n06,stale")
    r = subprocess.run([_bash(), chain, "--step", "probe_collect"], env=env_p, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    assert (out / "run_nodes.txt").read_text().split() == ["n02", "n04"]
    assert "run nodes (2 of 2): n02,n04; not used: n03,n07,n08" in so
    assert 'export SLURM_RESTRICT="--exclude=n05,n06,n03,n07,n08"' in (out / "run_nodes.env").read_text()
    calls = open(log).read().splitlines()
    assert len(calls) == 3 and all("--exclude=n05,n06,n03,n07,n08,n01" in c and "stale" not in c for c in calls), calls
    st = subprocess.run([_bash(), os.path.join(JOBS, "status.sh").replace("\\", "/")], env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT).stdout.decode()
    assert "chosen by the probe step): n02,n04" in st
    # too few healthy nodes: the node with no verdict fills up, and the run continues with a warning
    open(log, "w").close()
    r = subprocess.run([_bash(), chain, "--step", "probe_collect"], env=dict(env_p, NODES_EXPECTED="6"),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0 and "run nodes (5 of 6): n02,n04,n07,n08,n03; not used: " in so and "WARNING: only 5" in so, so[-1500:]
    # a new chain goes back to the base restriction and forgets the earlier choice
    open(log, "w").close()
    r = subprocess.run([_bash(), chain], env=env_p, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0 and not (out / "run_nodes.env").exists() and not (out / "run_nodes.txt").exists()
    first = open(log).read().splitlines()[0]
    assert "--exclude=n05,n06" in first and "stale" not in first, first


def _run_all_tree(tmp_path, sinfo_parts, sinfo_nodes, sacct, sshare="", squeue=""):
    """run_all.sh in a copy of the bundle layout, with stub setup, chain and status scripts and fake Slurm tools."""
    proj = tmp_path / "proj"
    cl = proj / "repo" / "loopedcot" / "cluster"
    (cl / "env").mkdir(parents=True)
    (cl / "jobs").mkdir()
    shutil.copy(os.path.join(CL, "run_all.sh"), str(cl / "run_all.sh"))
    log = (tmp_path / "calls.log").as_posix()
    stubs = {
        "env/setup_cluster.sh": 'echo "setup $*" >> "%s"\nmkdir -p "$2"\necho "export CHAIN_ENV=1" > "$2/cluster.env"\n'
                                'echo "nothing left: source $2/cluster.env"\n' % log,
        "jobs/submit_chain.sh": 'echo "chain DONE_SRC=$DONE_SRC CHAIN_ENV=$CHAIN_ENV" >> "%s"\n' % log,
        "jobs/status.sh": 'echo "STATUS PRINTED"\n'}
    for rel, body in stubs.items():
        (cl / rel).write_bytes(("#!/bin/bash\n" + body).encode())
    fake = tmp_path / "bin"
    fake.mkdir()
    tools = {"sinfo": 'case "$*" in *-N*) printf "%s";; *) printf "%s";; esac' % (sinfo_nodes, sinfo_parts),
             "sacctmgr": 'printf "%s"' % sacct, "sshare": 'printf "%s"' % sshare, "squeue": 'printf "%s"' % squeue,
             "nvcc": "echo release 12.4", "fakepy": 'echo "2.6.0 CUDA 12.4"'}
    for name, body in tools.items():
        (fake / name).write_bytes(("#!/bin/bash\n%s\n" % body).encode())
        os.chmod(str(fake / name), 0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=home.as_posix(), USER="op", PYTHON=(fake / "fakepy").as_posix(),
               PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for k in ("PARTITION", "ACCOUNT", "NODELIST", "NODES_EXPECTED", "MODULES", "RESERVATION", "OUT_ROOT"):
        env.pop(k, None)
    return proj, cl, env, log, home


def _run_all(cl, cwd, env, answers):
    return subprocess.run([_bash(), (cl / "run_all.sh").as_posix()], cwd=str(cwd), env=env, input=answers.encode(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@needs_bash
def test_run_all_discovers_asks_and_submits(tmp_path):
    """run_all.sh: the default partition (*), the first association, the usable GPU nodes of the partition (idle
    first; not responding, drained, down and CPU nodes left out); one yes, then setup with those values, the chain with
    DONE_SRC=none after sourcing cluster.env, and status.sh. Refusals: no yes, a home folder, a live earlier chain."""
    nodes = ("g03|gpu:8|mix\\ng01|gpu:8|idle\\nc01|(null)|idle\\ng02|gpu:8|drain\\ng04|gpu:8|idle*\\n"
             "g05|gpu:8(S:0-1)|alloc\\ng01|gpu:8|idle\\ng06|gpu:8|down\\n")
    proj, cl, env, log, home = _run_all_tree(tmp_path, "debug\\ngpu*\\n", nodes, "acctA\\nacctB\\n")
    r = _run_all(cl, proj, env, "no\n")
    assert r.returncode == 3 and not os.path.exists(log), r.stdout.decode()[-1500:]
    r = _run_all(cl, proj, env, "yes\n")
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    assert "could not be discovered" not in so and "free disk" in so and "STATUS PRINTED" in so
    assert "gpu (sinfo default (*))" in so and "acctA (sacctmgr (first association))" in so
    calls = open(log).read().splitlines()
    assert len(calls) == 2 and calls[1] == "chain DONE_SRC=none CHAIN_ENV=1", calls
    setup = calls[0]
    assert "--partition gpu --account acctA --nodelist g01,g03,g05 --nodes-expected 6" in setup, setup
    assert "/proj/out --log-root" in setup and "/proj/hf --weights-root" in setup and "/proj/ckpt" in setup, setup
    assert "--python " in setup and "fakepy" in setup and "--modules  --python" in setup.replace('""', "")
    # a chain of the earlier run still in Slurm: not submitted twice
    proj2, cl2, env2, log2, _ = _run_all_tree(tmp_path / "b", "gpu*\\n", nodes, "acctA\\n", squeue="5003\\n77_4\\n")
    (proj2 / "out" / "_chain").mkdir(parents=True)
    (proj2 / "out" / "_chain" / "jobs.txt").write_text("setup_check 5001\nprobes 5002:5003\n")
    r = _run_all(cl2, proj2, env2, "yes\n")
    assert r.returncode == 6 and b"still in Slurm (5003)" in r.stdout and not os.path.exists(log2), r.stdout.decode()[-1500:]
    # the current folder inside the home directory: refused before any question
    (home / "p").mkdir()
    r = _run_all(cl, home / "p", env, "yes\n")
    assert r.returncode == 2 and b"inside a home directory" in r.stdout


@needs_bash
def test_run_all_asks_by_name_and_remembers(tmp_path):
    """Nothing discoverable (no default partition, no GPU node, no association): one question per value, by name;
    the answers go to out/run_all.env and the next run asks nothing. An environment variable wins over an answer."""
    proj, cl, env, log, _ = _run_all_tree(tmp_path, "debug\\ngpu\\n", "", "", sshare="")
    r = _run_all(cl, proj, env, "gpuq\nacct9\nn1,n2,n3,n4,n5,n6,n7\nyes\n")
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    for name in ("PARTITION", "ACCOUNT", "NODELIST"):
        assert so.count("%s could not be discovered" % name) == 1, name
    assert "MODULES could not be discovered" not in so
    ans = (proj / "out" / "run_all.env").read_text()
    assert "export PARTITION=gpuq" in ans and "export ACCOUNT=acct9" in ans and "export NODELIST=" in ans, ans
    assert "--partition gpuq --account acct9 --nodelist n1,n2,n3,n4,n5,n6,n7 " in open(log).read()
    open(log, "w").close()
    r = _run_all(cl, proj, dict(env, ACCOUNT="acctX"), "yes\n")
    so = r.stdout.decode()
    assert r.returncode == 0 and "could not be discovered" not in so, so[-2000:]
    assert "--partition gpuq --account acctX --nodelist n1,n2,n3,n4,n5,n6,n7 " in open(log).read()
    # the sshare fallback when sacctmgr has no association
    proj3, cl3, env3, log3, _ = _run_all_tree(tmp_path / "c", "gpu*\\n", "g1|gpu:8|idle\\n", "", sshare="acctS  op\\n")
    r = _run_all(cl3, proj3, env3, "yes\n")
    assert r.returncode == 0 and "--account acctS " in open(log3).read(), r.stdout.decode()[-1500:]


def test_merge_scrub_and_home_scan_units(tmp_path):
    """scrub_home over scrub_roots: the project root becomes <PROJECT_ROOT> in the text files of the output trees the
    job list names, and each changed file is listed. Never touched: the environment files (cluster.env, run_all.env,
    run_nodes.env, by path, even inside an output tree), top-level files, _* folders (chain markers, job lists) and
    logs/. home_scan names each merged file that still holds a home path."""
    proj = tmp_path / "home" / "proj"
    out = proj / "out"
    d = out / "B" / "job"
    d.mkdir(parents=True)
    root = str(proj)
    (d / "records.jsonl").write_text('{"argv": "%s/repo/x.py", "v": 1}\n{"v": 2}\n' % root.replace("\\", "\\\\"))
    (d / "results.csv").write_text("a,b\n%s/out,2\n" % root)
    (d / "clean.csv").write_text("a\n1\n")
    (d / "copy.env").write_text("X=%s\n" % root)
    keep = {}
    for rel in ("cluster.env", "run_all.env", "run_nodes.env", "_env.json", "_chain/jobs.txt", "_held/ifm.held",
                "logs/ra20_g2_gen_1_1.out", "_tasklog/x.log"):
        f = out / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('export OUT_ROOT="%s/out"\nlists %s/repo/loopedcot/cluster/jobs/nolocal\n' % (root, root))
        keep[rel] = f.read_bytes()
    jl = tmp_path / "joblist.tsv"
    jl.write_text("id\tgroup\tout\na\tg1\t$OUT_ROOT/B/job\nb\tg0\t$OUT_ROOT/_xllm\nc\tg9\t$OUT_ROOT\n")
    roots = MG.scrub_roots(str(out), str(jl))
    assert roots == [os.path.join(str(out), "B")]
    changed = []
    n = MG.scrub_home(roots, [root, root.replace("\\", "\\\\")], str(out), changed)
    assert n == 2 and sorted(changed) == ["B/job/records.jsonl", "B/job/results.csv"]
    assert (d / "results.csv").read_text() == "a,b\n<PROJECT_ROOT>/out,2\n"
    assert json.loads((d / "records.jsonl").read_text().splitlines()[0])["argv"] == "<PROJECT_ROOT>/repo/x.py"
    assert root in (d / "copy.env").read_text()
    for rel, b in keep.items():
        assert (out / rel).read_bytes() == b, rel
    # an environment file inside a scrubbed tree is protected by its path too
    (out / "B" / "cluster.env").write_text("X=%s\n" % root)
    assert MG.scrub_home([str(out)], [root], str(out / "B"), []) >= 0
    assert root in (out / "B" / "cluster.env").read_text()
    dest = tmp_path / "merged"
    dest.mkdir()
    (dest / "ok.csv").write_text("x\n<PROJECT_ROOT>/out\n")
    (dest / "bad.md").write_text("see /home/someone/out\n")
    assert MG.home_scan(str(dest)) == [{"file": "bad.md", "patterns": ["/home/"]}]


@needs_artifacts
def test_merge_home_run_scrubs_and_fails_on_leftovers(tmp_path):
    """ALLOW_HOME=1: merge.py rewrites the project root in the job folders before reading them, writes the merged
    outputs with no home path (home_scan.json empty, exit 0); a merged file with another home path fails it (exit 3).
    Without PROJECT_ROOT it refuses (exit 3)."""
    home = tmp_path / "home"
    proj = home / "proj"
    out = proj / "out"
    for m in ("parcae-370m", "parcae-1.3b"):
        d = out / "B" / "k17_parcae" / m
        d.mkdir(parents=True)
        for f in glob.glob(os.path.join(ART, "k17_parcae", "full_final", "res_%s_D1.json" % m)) + \
                glob.glob(os.path.join(ART, "k17_parcae", "full_final", "meta_%s.json" % m)):
            shutil.copy(f, str(d))
        src = os.path.join(PILOT, "k17_parcae.py")
        subprocess.check_call([sys.executable, "-B", src, "--summarize", "--out", str(d)], stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-B", os.path.join(CL, "extract", "run.py"), "--script", "k17_parcae.py",
                               "--src", src, "--job", str(d)], stdout=subprocess.DEVNULL)
        (d / "task.log").write_text("python %s/repo/loopedcot/cluster/pilot/k17_parcae.py --out %s\n" % (proj, d))
    keep = {}
    for rel in ("cluster.env", "run_all.env", "run_nodes.env", "_chain/jobs.txt"):
        f = out / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('export OUT_ROOT="%s"\nlists %s/repo/loopedcot/cluster/jobs/nolocal\n' % (out, proj))
        keep[rel] = f.read_bytes()
    merge = [sys.executable, "-B", os.path.join(CL, "merge.py"), "--out_root", str(out)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", ALLOW_HOME="1", HOME=str(home))
    env.pop("PROJECT_ROOT", None)
    r = subprocess.run(merge, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 3 and b"PROJECT_ROOT is not set" in r.stdout
    env["PROJECT_ROOT"] = str(proj)
    r = subprocess.run(merge, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0 and "running inside a home directory" in so, so[-2000:]
    log = (out / "B" / "k17_parcae" / "parcae-370m" / "task.log").read_text()
    assert str(proj) not in log and log.startswith("python <PROJECT_ROOT>/repo/loopedcot/")
    scan = json.loads((out / "merged" / "home_scan.json").read_text())
    assert scan["home_path_hits"] == [] and scan["files_scrubbed"] >= 2
    for rel, b in keep.items():
        assert (out / rel).read_bytes() == b, rel
    changed = (out / "merged" / "home_scrub_changed.txt").read_text().split()
    assert "B/k17_parcae/parcae-370m/task.log" in changed and not [x for x in changed if x.endswith(".env") or x.startswith("_")]
    (out / "merged" / "stray.md").write_text("left over: /root/x\n")
    r = subprocess.run(merge, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 3 and b"home paths remain in the merged outputs: stray.md" in r.stdout, r.stdout.decode()[-1500:]


@needs_bash
def test_allow_home_setup_space_check_and_template_guard(tmp_path):
    """ALLOW_HOME=1: setup accepts a project in the home directory, prints the message, df and quota, and stops when
    the space is short (df or quota); without it the refusal stays. The array template guard: refused by default,
    the message with ALLOW_HOME=1."""
    lc, fake = _setup_tree(tmp_path)
    proj = (tmp_path / "proj").as_posix()
    py = sys.executable.replace("\\", "/")
    setup = (lc / "env" / "setup_cluster.sh").as_posix()

    def tools(df_free_kb, quota_lines):
        (fake / "df").write_bytes(("#!/bin/bash\necho 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
                                   "echo 'fs 900000000 1 %d 1%% /h'\n" % df_free_kb).encode())
        (fake / "quota").write_bytes(("#!/bin/bash\necho 'Disk quotas for user op'\necho 'Filesystem blocks quota limit "
                                      "grace files quota limit grace'\n%s\n" % quota_lines).encode())
        for n in ("df", "quota"):
            os.chmod(str(fake / n), 0o755)

    env = dict(os.environ, HOME=proj, PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for k in ("OUT_ROOT", "HF_HOME", "WEIGHTS_ROOT", "PARTITION", "ACCOUNT", "NODELIST", "ALLOW_HOME", "PROJECT_ROOT"):
        env.pop(k, None)
    cmd = [_bash(), "-c", 'PATH="$(cygpath -u "$0" 2> /dev/null || echo "$0"):$PATH"; exec bash "$@"', fake.as_posix(), setup, "--dry-run", "--python", py,
           "--out-root", proj + "/out", "--hf-home", proj + "/hf",
           "--weights-root", proj + "/ckpt", "--partition", "gpuq", "--account", "acct1", "--nodelist",
           "n01,n02,n03,n04,n05,n06"]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 2 and b"ALLOW_HOME=1 when the cluster has no scratch" in r.stdout
    tools(500 * 1048576, "echo 'fs 1000 0 0 0 1 0 0 0'")          # 500 GB free, no quota limit
    r = subprocess.run(cmd, env=dict(env, ALLOW_HOME="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert "running inside a home directory; records will be scrubbed at merge" in so, so[-2000:]
    assert "need about 113.0 GB more" in so and "quota free no quota found" in so and "quota -s:" in so
    assert "stopped before creating anything" not in so and "<PROJECT_ROOT>" not in so, so[-2000:]
    e = open(proj + "/out/cluster.env", encoding="utf-8").read()
    assert "export ALLOW_HOME=1" in e and re.search(r'export PROJECT_ROOT="[^"<]+/proj"', e), e
    # the run folders inside the clone (run_all.sh started from cluster/): the project root widens to the folder that
    # holds the repository and every run folder, and is never left as a placeholder
    c = lc.as_posix()
    r = subprocess.run(cmd[:8] + ["--out-root", c + "/out", "--hf-home", c + "/hf", "--weights-root", c + "/ckpt"] + cmd[14:]
                       + ["--project-root", c], env=dict(env, ALLOW_HOME="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert "<PROJECT_ROOT>" not in so and "stopped before creating anything" not in so, so[-2000:]
    e = open(c + "/out/cluster.env", encoding="utf-8").read()
    assert re.search(r'export PROJECT_ROOT="[^"<]+/proj/repo"', e), e
    tools(50 * 1048576, "echo 'fs 1000 0 0 0 1 0 0 0'")           # 50 GB free: short
    r = subprocess.run(cmd, env=dict(env, ALLOW_HOME="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 2 and b"stopped before creating anything: about 113.0 GB is needed" in r.stdout, r.stdout.decode()[-1500:]
    tools(500 * 1048576, "echo 'fs 10485760 20971520 20971520 0 1 0 0 0'")   # quota: 10 GB of 20 GB used
    r = subprocess.run(cmd, env=dict(env, ALLOW_HOME="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 2 and b"10.0 GB is free in the home directory" in r.stdout, r.stdout.decode()[-1500:]
    # the array template guard, run on its own
    t = open(os.path.join(JOBS, "task_g2_gen.sbatch"), encoding="utf-8").read()
    i = t.index("for v in REPO OUT_ROOT DATA_ROOT HF_HOME")
    j = t.index("records will be scrubbed at merge", i)
    guard = t[i:t.index("\n", j)] + "\necho passed\n"
    if os.name == "nt":          # Git bash: the template compares POSIX paths, as on the cluster
        guard = 'HOME=$(cygpath -u "$HOME"); OUT_ROOT=$(cygpath -u "$OUT_ROOT")\n' + guard
    genv = dict(env, OUT_ROOT=proj + "/out", TL=(tmp_path / "tl.log").as_posix())
    r = subprocess.run([_bash(), "-c", guard], env=genv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 2 and b"inside a home directory" in r.stdout and b"passed" not in r.stdout
    r = subprocess.run([_bash(), "-c", guard], env=dict(genv, ALLOW_HOME="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0 and b"running inside a home directory; records will be scrubbed at merge\npassed" in r.stdout


@needs_bash
def test_run_all_allow_home(tmp_path):
    """run_all.sh in a home folder: refused, then accepted with ALLOW_HOME=1 (message, ALLOW_HOME=1 reaches setup)."""
    proj, cl, env, log, home = _run_all_tree(tmp_path, "gpu*\\n", "g1|gpu:8|idle\\n", "acctA\\n")
    env = dict(env, HOME=proj.as_posix())
    env.pop("ALLOW_HOME", None)
    r = _run_all(cl, proj, env, "yes\n")
    assert r.returncode == 2 and b"ALLOW_HOME=1 bash" in r.stdout and not os.path.exists(log)
    (cl / "env" / "setup_cluster.sh").write_bytes(
        ('#!/bin/bash\necho "setup ALLOW_HOME=$ALLOW_HOME $*" >> "%s"\nmkdir -p "$2"\n: > "$2/cluster.env"\n'
         'echo "nothing left"\n' % log).encode())
    r = _run_all(cl, proj, dict(env, ALLOW_HOME="1"), "yes\n")
    so = r.stdout.decode()
    assert r.returncode == 0 and "running inside a home directory; records will be scrubbed at merge" in so, so[-1500:]
    assert open(log).read().startswith("setup ALLOW_HOME=1 ")


@needs_bash
def test_setup_picks_python311_and_pins_torch_without_local_label(tmp_path):
    """setup without --python: python3 (no torch, as a 3.9 node default would be) is passed over for python3.11; every
    venv prints that interpreter; the pip constraint is torch==2.8.0, not torch==2.8.0+cu128; stamps follow pip rc 0."""
    lc, fake = _setup_tree(tmp_path)
    log = (tmp_path / "pip.log").as_posix()
    venv_pip = '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = -c ] && cat "$2" >> "%s"; shift; done\nexit 0\n' % log
    py311 = ('#!/bin/bash\ncase "$*" in\n'
             '  *"-m venv"*) V="${@: -1}"; mkdir -p "$V/bin"; printf %%s \'%s\' > "$V/bin/pip"; '
             'printf \'#!/bin/bash\\necho torch from the node image\\n\' > "$V/bin/python"; chmod +x "$V/bin/pip" "$V/bin/python";;\n'
             '  *"sys.version_info[:3]"*) echo 3.11.9;;\n'
             '  *"torch.__version__)"*) echo 2.8.0+cu128;;\n'
             '  *"torch.version.cuda or"*) echo 12.8;;\n'
             '  *) exit 0;;\nesac\n') % venv_pip.replace("'", "'\\''")
    (fake / "python3.11").write_bytes(py311.encode())
    (fake / "python3").write_bytes(b"#!/bin/bash\nexit 1\n")
    for n in ("python3.11", "python3"):
        os.chmod(str(fake / n), 0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=home.as_posix())
    for k in ("OUT_ROOT", "HF_HOME", "WEIGHTS_ROOT", "PARTITION", "ACCOUNT", "NODELIST", "PYTHON", "ALLOW_HOME"):
        env.pop(k, None)
    proj = (tmp_path / "proj").as_posix()
    cmd = [_bash(), "-c", 'PATH="$(cygpath -u "$0" 2> /dev/null || echo "$0"):$PATH"; exec bash "$@"', fake.as_posix(),
           (lc / "env" / "setup_cluster.sh").as_posix(), "--venvs-only", "--out-root", proj + "/out"]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2500:]
    for f in ("ouro", "raven", "parcae", "ifm"):
        assert re.search(r"venv %s: .* interpreter \S*python3\.11 \(Python 3\.11\.9\)" % f, so), (f, so[-2500:])
        assert os.path.exists(os.path.join(proj, "repo", "venvs", f, ".cluster_setup"))
    assert open(log).read().split() == ["torch==2.8.0"] * 4
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.stdout.decode().count("ready (") == 4 and r.stdout.count(b", Python 3.11.9)") == 4


@pytest.mark.skipif(shutil.which("git") is None or not os.path.isdir(os.path.join(LC, ".git")), reason="no git checkout")
def test_gitignore_keeps_run_folders_out_of_the_clone():
    """run_all.sh started from cluster/ puts out/, hf/ and ckpt/ there: cluster/.gitignore ignores them, and no bundle
    file is ignored."""
    for f in ("cluster/out/cluster.env", "cluster/out/run_all.env", "cluster/hf/hub/x", "cluster/ckpt/x/config.json"):
        assert subprocess.run(["git", "-C", LC, "check-ignore", "-q", f]).returncode == 0, f
    tracked = subprocess.check_output(["git", "-C", LC, "ls-files", "cluster"]).decode().split()
    r = subprocess.run(["git", "-C", LC, "check-ignore", "--no-index"] + tracked, stdout=subprocess.PIPE)
    assert r.returncode == 1 and not r.stdout.strip(), r.stdout.decode()[:500]


@needs_bash
def test_run_all_picks_python311(tmp_path):
    """run_all.sh with no PYTHON: python3 does not qualify, python3.11 does; it is passed to setup and printed."""
    proj, cl, env, log, _ = _run_all_tree(tmp_path, "gpu*\\n", "g1|gpu:8|idle\\n", "acctA\\n")
    fake = tmp_path / "bin"
    (fake / "python3").write_bytes(b"#!/bin/bash\nexit 1\n")
    (fake / "python3.11").write_bytes(b'#!/bin/bash\necho "Python 3.11.9, torch 2.8.0+cu128 CUDA 12.8"\n')
    for n in ("python3", "python3.11"):
        os.chmod(str(fake / n), 0o755)
    env = dict(env)
    env.pop("PYTHON", None)
    r = subprocess.run([_bash(), "-c", 'PATH="$(cygpath -u "$0" 2> /dev/null || echo "$0"):$PATH"; exec bash "$@"',
                        fake.as_posix(), (cl / "run_all.sh").as_posix()], cwd=str(proj), env=env, input=b"yes\n",
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0 and "could not be discovered" not in so, so[-2000:]
    assert re.search(r"python3\.11 \(Python 3\.11\.9, torch 2\.8\.0\+cu128 CUDA 12\.8\); all four venvs use it", so), so[-2000:]
    assert re.search(r"--python \S*python3\.11\b", open(log).read())
    assert re.search(r"--project-root \S*/proj\b", open(log).read())


def _posix_path_cmd(fake, *args):
    """bash with the fake folder first on PATH (Git's bash launcher puts /usr/bin first otherwise)."""
    return [_bash(), "-c", 'PATH="$(cygpath -u "$0" 2> /dev/null || echo "$0"):$PATH"; exec bash "$@"', fake.as_posix()] + list(args)


@needs_bash
def test_preflight_per_family_markers_and_subset_rerun(tmp_path):
    """preflight.sh: a failed IFM build marks only ifm as failed (ifm.fail, families_held ["ifm"]) and exits 0, so the
    chain goes on; PREFLIGHT_FAMILIES=ifm reruns IFM alone and keeps the other markers; no family passing exits 1."""
    top = tmp_path / "top"
    env_dir = top / "loopedcot" / "cluster" / "env"
    shutil.copytree(os.path.join(CL, "env"), str(env_dir), ignore=shutil.ignore_patterns("cluster.env", "__pycache__"))
    flag = tmp_path / "build_ok"
    (env_dir / "build_xllm.sh").write_bytes(('#!/bin/bash\n[ -f "%s" ]\n' % flag.as_posix()).encode())
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python3").write_bytes(('#!/bin/bash\nexec "%s" "$@"\n' % sys.executable.replace("\\", "/")).encode())
    (fake / "nvidia-smi").write_bytes(b"#!/bin/bash\necho 8.0\n")
    fail_all = tmp_path / "fail_all"
    venvs = {}
    for f in ("ouro", "raven", "parcae", "ifm"):
        b = tmp_path / "venvs" / f / "bin"
        b.mkdir(parents=True)
        (b / "python").write_bytes((
            '#!/bin/bash\ncase "$*" in *preflight.py*) out="${@: -1}"; [ -f "%s" ] && exit 1;'
            ' echo \'{"ok": true, "failures": []}\' > "$out"; exit 0;; *) echo "Python 3.11.9";; esac\n'
            % fail_all.as_posix()).encode())
        venvs["VENV_" + f.upper()] = (tmp_path / "venvs" / f).as_posix()
    for p in list(fake.iterdir()) + [tmp_path / "venvs" / f / "bin" / "python" for f in ("ouro", "raven", "parcae", "ifm")]:
        os.chmod(str(p), 0o755)
    out = tmp_path / "out"
    env = dict(os.environ, REPO=top.as_posix(), OUT_ROOT=out.as_posix(), XLLM_ROOT=(tmp_path / "x").as_posix(),
               TORCH_CUDA_ARCH_LIST="8.0", **venvs)
    env.pop("PREFLIGHT_FAMILIES", None)
    pre = (env_dir / "preflight.sh").as_posix()
    r = subprocess.run(_posix_path_cmd(fake, pre), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    P = out / "_preflight"
    assert r.returncode == 0, so[-2000:]
    assert sorted(x.name for x in P.glob("*.ok")) == ["ouro.ok", "parcae.ok", "raven.ok"] and (P / "ifm.fail").exists()
    rec = json.loads((out / "_preflight.json").read_text())
    assert rec["families_held"] == ["ifm"] and rec["families_ok"] == ["ouro", "raven", "parcae"] and rec["ok"] is False
    assert "family ifm: preflight FAILED" in so
    assert "preflight of the ifm venv failed" in (out / "_held" / "ifm.held").read_text()
    assert not (out / "_held" / "ouro.held").exists()
    flag.write_text("1")
    r = subprocess.run(_posix_path_cmd(fake, pre), env=dict(env, PREFLIGHT_FAMILIES="ifm"), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    assert r.returncode == 0, r.stdout.decode()[-1500:]
    assert (P / "ifm.ok").exists() and not (P / "ifm.fail").exists() and (P / "ouro.ok").exists()
    assert json.loads((out / "_preflight.json").read_text())["families_held"] == []
    fail_all.write_text("1")
    r = subprocess.run(_posix_path_cmd(fake, pre), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1 and not list(P.glob("*.ok"))


@needs_bash
def test_submit_10_rest_holds_failed_family_and_status_shows_it(tmp_path):
    """With _preflight/ifm.fail: every IFM stage is submitted with --hold and recorded in _held/ifm.jobs, no other
    stage is held, and status.sh names the held family and the release command."""
    nl = os.path.join(JOBS, "nolocal")
    build = "g0_xllm.s1_build.001"
    out = tmp_path / "out"
    for d in ("_done", "_preflight"):
        (out / d).mkdir(parents=True)
    (out / "_done" / build).write_text("ok\n")
    (out / "_smoke.ok").write_text("ok\n")
    (out / "_preflight" / "ifm.fail").write_text("x\n")
    (out / "_held").mkdir()
    (out / "_held" / "ifm.held").write_text("preflight of the ifm venv failed (x)\n")
    fake, log = _fake_sbatch(tmp_path)
    env = dict(os.environ, OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(), REPO=TOP.replace("\\", "/"),
               PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""), CARDS="48", **SLURM_ENV)
    r = subprocess.run([_bash(), os.path.join(nl, "submit_10_rest.sh").replace("\\", "/")], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    assert "family ifm held (preflight of the ifm venv failed (x)); its jobs are submitted held" in so
    calls = open(log).read().splitlines()
    held = [c for c in calls if "--hold" in c]
    ifm_lists = {ln.split("\t")[0] for f in glob.glob(os.path.join(nl, "g*.tsv")) for ln in open(f, encoding="utf-8")
                 if "VENV_IFM" in ln}
    assert held and all(any(("task_%s.sbatch" % g) in c for g in GJ.GROUP_ORDER) for c in held)
    n_ifm_calls = sum(1 for ln in open(os.path.join(nl, "submit_10_rest.sh"), encoding="utf-8") if ln.startswith("K ifm "))
    assert len(held) == n_ifm_calls and len(ifm_lists) > 0
    ids = (out / "_held" / "ifm.jobs").read_text().split()
    assert len(ids) == n_ifm_calls and not (out / "_held" / "ouro.jobs").exists()
    st = subprocess.run([_bash(), os.path.join(JOBS, "status.sh").replace("\\", "/")], env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT).stdout.decode()
    assert "HELD family ifm: preflight of the ifm venv failed (x); %d jobs held" % n_ifm_calls in st and "release_family.sh ifm" in st
    merge = [c for c in calls if "task_g9_merge.sbatch" in c]
    assert len(merge) == 1 and "--hold" not in merge[0]


def test_g3_decide_cluster_measured_rate():
    """Gate G3 (cache equivalence): n - n/32 with every disagreement at a near tie passes as before; with
    --g3-measured-rate a miss of the agreement count alone passes as Deviation D5m with the measured rate recorded
    (pass_strict False); a disagreement at a recompute margin >= 1 logit fails either way."""
    sys.path.insert(0, os.path.join(CL, "pilot", "tree"))
    try:
        from prod import steer_gen as SG
    finally:
        sys.path.pop(0)

    def kinds(agree, n, bad=0):
        return {"clean": {"argmax_agree": n, "n": n, "max_abs_dz": 1.0, "max_abs_dz_top5": 1.0, "disagree_at_margin_ge_1": 0},
                "post@1": {"argmax_agree": agree, "n": n, "max_abs_dz": 10.0, "max_abs_dz_top5": 3.0,
                           "disagree_at_margin_ge_1": bad}}
    g = SG.g3_decide(kinds(92, 96), 1.0, 1.0)
    assert g["pass"] is False and "deviation" not in g          # the smoke on the cluster: 92 of 96, rule 93
    g = SG.g3_decide(kinds(92, 96), 1.0, 1.0, measured_rate_ok=True)
    assert g["pass"] is True and g["pass_strict"] is False and g["deviation"]["id"] == "D5m"
    assert g["deviation"]["measured_agreement"]["post@1"] == {"agree": 92, "n": 96, "rate": 0.9583}
    assert g["deviation"]["kinds_below_n_minus_n_over_32"] == ["post@1"]
    g = SG.g3_decide(kinds(372, 384), 1.0, 1.0, measured_rate_ok=True)          # 12 items: 384 - 12
    assert g["pass"] is True and g["pass_strict"] is True and "deviation" not in g
    g = SG.g3_decide(kinds(380, 384, bad=1), 1.0, 1.0, measured_rate_ok=True)
    assert g["pass"] is False and "deviation" not in g
    for f in ("g2_gen.s1_checks.tsv", "nolocal/g2_gen.s1_checks.tsv", "smoke_g2_gen.tsv"):
        lines = [ln for ln in open(os.path.join(JOBS, f), encoding="utf-8") if "steer_gen checks" in ln]
        assert lines and all("--n 12 --g3-items 12 --g3-measured-rate" in ln for ln in lines), f


def test_smoke_check_families_leave_out_held_checks(tmp_path):
    """smoke_check.py --families raven: no Ouro, Parcae or IFM check and no tolerance gate; the raven checks run."""
    root = tmp_path / "sm"
    root.mkdir()
    rep = tmp_path / "rep.json"
    subprocess.run([sys.executable, "-B", os.path.join(CL, "env", "smoke_check.py"), "--root", str(root), "--lists", JOBS,
                    "--report", str(rep), "--values", os.path.join(CL, "SMOKE_VALUES.json"),
                    "--hashes", os.path.join(CL, "SMOKE_HASHES.txt"), "--families", "raven"],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    r = json.loads(rep.read_text())
    names = [c["check"] for c in r["checks"]]
    assert r["families_checked"] == ["raven"] and "tolerance" not in r
    assert not [n for n in names if n.startswith(("varbench", "steer_gen", "parcae", "ifm", "mcleish"))], names
    assert any(n.startswith("stage smoke.fam.f_raven") for n in names)
    assert not [n for n in names if "g1_copy" in n or "g2_gen" in n or "f_ifm" in n or "f_parcae" in n or "f_mcleish" in n]


@needs_bash
def test_release_family_submits_one_gpu_job(tmp_path):
    """release_family.sh: a family without held jobs is refused; otherwise one one-GPU job runs its --run step."""
    fake, log = _fake_sbatch(tmp_path)
    out = tmp_path / "out"
    (out / "_held").mkdir(parents=True)
    env = dict(os.environ, OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(), REPO=TOP.replace("\\", "/"),
               PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""), **SLURM_ENV)
    rel = os.path.join(JOBS, "release_family.sh").replace("\\", "/")
    r = subprocess.run([_bash(), rel, "ifm"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 3 and not os.path.exists(log)
    r = subprocess.run([_bash(), rel, "nope"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 2
    (out / "_held" / "ifm.jobs").write_text("2001\n2002\n")
    (out / "bad_nodes.txt").write_text("n03\n")
    r = subprocess.run([_bash(), rel, "ifm"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, r.stdout.decode()[-1500:]
    c = open(log).read().splitlines()
    assert len(c) == 1 and "--gres=gpu:1" in c[0] and "--exclude=n05,n06,n03" in c[0] and "--run ifm" in c[0]


def _settle_of(line):
    """The SETTLE file a steer_gen prep or select line reads: its config's settle, else the setting's."""
    sys.path.insert(0, os.path.join(CL, "pilot", "tree"))
    try:
        from prod import steer_gen as SG
    finally:
        sys.path.pop(0)
    st = re.search(r"--setting (\S+)", line).group(1)
    if "steer_gen_raven.py" in line:                   # its settings (hug_svamp, B-EXT5) register at import
        sys.path.insert(0, PILOT)
        try:
            import steer_gen_raven  # noqa: F401
        finally:
            sys.path.pop(0)
    cj = re.search(r"--config-json \S*/cluster/jobs/(gen_cfg/\S+\.json)", line)
    cfg = json.load(open(os.path.join(JOBS, cj.group(1)), encoding="utf-8")) if cj else {}
    return cfg.get("settle") or SG.SETTINGS[st]["settle"]


def test_every_settle_file_the_task_lists_name_is_shipped():
    """Every SETTLE file a task line reads (steer_gen prep, screen and select, in jobs/ and jobs/nolocal/) is in
    cluster/data/, and setup_check checks all of them (it checks every cluster/data/cells_*.jsonl)."""
    need = {}
    for f in glob.glob(os.path.join(JOBS, "g*.tsv")) + glob.glob(os.path.join(JOBS, "nolocal", "g*.tsv")):
        for ln in open(f, encoding="utf-8"):
            if "--settle-dir" in ln:
                need.setdefault(_settle_of(ln), set()).add(ln.split("\t")[0])
    have = {os.path.basename(p) for p in glob.glob(os.path.join(CL, "data", "cells_*.jsonl"))}
    assert len(need) == 9 and not set(need) - have, sorted(set(need) - have)
    for name in ("cells_ouro_2_6b_base_gsm8k_natural_k4.jsonl", "cells_ouro_1_4b_think_svamp_natural2_k4.jsonl",
                 "cells_huginn_0125_svamp_natural_k32.jsonl"):
        assert name in need and name in have


def _resubmit_env(tmp_path, queue="", merge_pending=True, named="", command="", with_sub=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake, log = _fake_sbatch(tmp_path)
    (fake / "squeue").write_bytes((
        '#!/bin/bash\ncase "$*" in\n'
        '  *"%%F %%K"*) printf "%s";;\n'
        '  *"-n ra20_"*) printf "%s";;\n'
        '  *"-t PD"*) %s;;\n'
        '  *"%%E"*) echo "afterany:5001_*(unfulfilled),afterany:5002(unfulfilled)";;\n'
        '  *) ;;\nesac\n' % (queue, named, 'echo 9999' if merge_pending else 'true')).encode())
    (fake / "scontrol").write_bytes(('#!/bin/bash\nif [ "$1" = show ]; then printf "%s"; exit 0; fi\n'
                                     'echo "scontrol $*" >> "%s"\n' % (command, log)).encode())
    for n in ("squeue", "scontrol"):
        os.chmod(str(fake / n), 0o755)
    out = tmp_path / "out"
    for d in ("_done", "_chain"):
        (out / d).mkdir(parents=True)
    nl = os.path.join(JOBS, "nolocal").replace("\\", "/")
    (out / "_chain" / "jobs.txt").write_text("setup_check 1\nlists %s\n" % nl)
    rows = [r for r in csv.DictReader(open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8"), delimiter="\t")
            if r["script"] == "steer_gen.py" and r["model"] == "ouro_2_6b_base" and r["out"].endswith("/a2_gsm8k")]
    sub = ["9999 g9_merge.s1_merge.tsv 1"]
    for r in rows:
        L = "%s.%s.tsv" % (r["group"], r["stage"])
        idx = [ln.split("\t")[0] for ln in open(os.path.join(JOBS, "nolocal", L), encoding="utf-8")].index(r["id"]) + 1
        sub.append("700%d %s %d" % (len(sub), L, idx))
    if with_sub:
        (out / "_chain" / "submitted.tsv").write_text("\n".join(sub) + "\n")
    env = dict(os.environ, OUT_ROOT=out.as_posix(), LOG_ROOT=(out / "logs").as_posix(), REPO=TOP.replace("\\", "/"),
               USER="op", PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""), CARDS="48", **SLURM_ENV)
    return env, out, log, rows


@needs_bash
def test_resubmit_pipeline_skips_done_chains_stages_and_refuses_queued(tmp_path):
    """resubmit_pipeline.sh: an ambiguous script and model is refused with the choices; a queued task of the pipeline
    refuses it; otherwise the stages without done markers are submitted with afterok from one to the next, only their
    undone task indices, through the same template and stage lists, and the pending merge waits for them too."""
    rs = os.path.join(JOBS, "resubmit_pipeline.sh").replace("\\", "/")
    env, out, log, rows = _resubmit_env(tmp_path)
    r = subprocess.run([_bash(), rs, "steer_gen.py", "ouro_2_6b_base"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 3 and b"a2_gsm8k" in r.stdout and b"a2_svamp" in r.stdout and not os.path.exists(log)
    for r_ in rows:
        if r_["stage"] in ("s1_checks",):
            (out / "_done" / r_["id"]).write_text("ok\n")
    rows4 = [r_ for r_ in rows if r_["stage"] == "s4_row"]
    for r_ in rows4[:5]:
        (out / "_done" / r_["id"]).write_text("ok\n")
    r = subprocess.run([_bash(), rs, "steer_gen.py", "ouro_2_6b_base", "a2_gsm8k"], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    calls = open(log).read().splitlines()
    sb = [c for c in calls if not c.startswith("scontrol")]
    stages = list(dict.fromkeys(r_["stage"] for r_ in rows))
    assert len(sb) == len(stages) - 1 and "g2_gen.s2_prep_screen.tsv" in sb[0] and "--dependency" not in sb[0], sb
    for k, c in enumerate(sb[1:]):
        assert "--dependency=afterok:%d" % (1001 + k) in c, c
    s4 = [c for c in sb if "g2_gen.s4_row.tsv" in c][0]
    idx4 = [ln.split("\t")[0] for ln in open(os.path.join(JOBS, "nolocal", "g2_gen.s4_row.tsv"), encoding="utf-8")]
    want = ",".join(str(idx4.index(r_["id"]) + 1) for r_ in rows4[5:])
    assert "--array=%s%%48" % want in s4 and "task_g2_gen.sbatch" in s4 and "--exclude=n05,n06" in s4
    sc = [c for c in calls if c.startswith("scontrol")]
    assert sc and "JobId=9999" in sc[0] and "Dependency=afterany:5001:5002:1001:" in sc[0], sc
    assert len(open(out / "_chain" / "submitted.tsv").read().splitlines()) == 1 + len(rows) + len(sb)
    # a queued task of the pipeline: refused
    one = rows4[7]
    L = "g2_gen.s4_row.tsv"
    jid = [ln.split()[0] for ln in open(out / "_chain" / "submitted.tsv") if ln.split()[1] == L and ln.split()[2] == str(idx4.index(one["id"]) + 1)][0]
    env2, out2, log2, _ = _resubmit_env(tmp_path / "b", queue="%s %d\\n" % (jid, idx4.index(one["id"]) + 1))
    r = subprocess.run([_bash(), rs, "steer_gen.py", "ouro_2_6b_base", "a2_gsm8k"], env=env2, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    assert r.returncode == 4 and one["id"].encode() in r.stdout and not os.path.exists(log2), r.stdout.decode()[-1500:]
    # a run submitted before the log existed: the stage list comes from Slurm's job record; unknown jobs refuse
    pos = idx4.index(one["id"]) + 1
    env3, _o3, log3, _ = _resubmit_env(tmp_path / "c", queue="8001 %d\\n8002 3\\n" % pos, named="8001\\n8002\\n",
                                       command="JobId=8001 Command=/x/task_g2_gen.sbatch g2_gen.s4_row.tsv\\n",
                                       with_sub=False)
    r = subprocess.run([_bash(), rs, "steer_gen.py", "ouro_2_6b_base", "a2_gsm8k"], env=env3, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 4 and ("scancel 8001_%d" % pos) in so and not os.path.exists(log3), so[-1500:]
    env4, _o4, log4, _ = _resubmit_env(tmp_path / "d", queue="8001 %d\\n" % pos, named="8001\\n", command="",
                                       with_sub=False)
    r = subprocess.run([_bash(), rs, "steer_gen.py", "ouro_2_6b_base", "a2_gsm8k"], env=env4, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    assert r.returncode == 4 and b"whose stage list Slurm does not show: 8001" in r.stdout and not os.path.exists(log4)


def test_steer_gen_raven_imports_cluster_extract_from_the_bundle(tmp_path):
    """The McLeish analyze step imports cluster.extract (results.csv): in an isolated interpreter started outside the
    repository (as a cluster task is), the module's own path setup finds the bundle's cluster/extract, and prod stays
    the bundle's tree copy."""
    code = ("import runpy, sys\n"
            "runpy.run_path(sys.argv[1], run_name='probe')\n"
            "from cluster.extract import common, gen\n"
            "import prod.steer_gen as S\n"
            "print(common.__file__); print(gen.__file__); print(S.__file__)\n")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code, os.path.join(PILOT, "steer_gen_raven.py")],
                       cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode()
    assert r.returncode == 0, out[-2000:]
    paths = [os.path.normcase(os.path.abspath(x)) for x in out.strip().splitlines()[-3:]]
    assert paths[0] == os.path.normcase(os.path.join(CL, "extract", "common.py"))
    assert paths[1] == os.path.normcase(os.path.join(CL, "extract", "gen.py"))
    assert paths[2] == os.path.normcase(os.path.join(PILOT, "tree", "prod", "steer_gen.py"))


@needs_bash
def test_resubmit_pipeline_reruns_only_the_analyze_stage(tmp_path):
    """The McLeish pipeline with every row task done and the analyze task failed: resubmit_pipeline.sh submits only
    the analyze task (its index in g2_gen.s6_analyze.tsv), with no dependency, and the pending merge waits for it."""
    rs = os.path.join(JOBS, "resubmit_pipeline.sh").replace("\\", "/")
    env, out, log, _rows = _resubmit_env(tmp_path)
    rows = [r for r in csv.DictReader(open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8"), delimiter="\t")
            if r["script"] == "steer_gen_raven.py" and r["model"] == "mcleish_llama32_r32"]
    assert {r["stage"] for r in rows} == {"s4_row", "s6_analyze"} and len(rows) == 23
    for r in rows:
        if r["stage"] != "s6_analyze":
            (out / "_done" / r["id"]).write_text("ok\n")
    an = [r for r in rows if r["stage"] == "s6_analyze"][0]
    idx = [ln.split("\t")[0] for ln in open(os.path.join(JOBS, "nolocal", "g2_gen.s6_analyze.tsv"), encoding="utf-8")].index(an["id"]) + 1
    r = subprocess.run([_bash(), rs, "steer_gen_raven.py", an["model"]], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-2000:]
    calls = open(log).read().splitlines()
    sb = [c for c in calls if not c.startswith("scontrol")]
    assert len(sb) == 1 and "g2_gen.s6_analyze.tsv" in sb[0] and "--array=%d%%48" % idx in sb[0], sb
    assert "--dependency" not in sb[0] and "task_g2_gen.sbatch" in sb[0]
    assert "s4_row: every task done (22), skipped" in so
    assert any("JobId=9999" in c and c.endswith(":1001") for c in calls if c.startswith("scontrol")), calls


@pytest.mark.parametrize("case", ["nvidia_smi_signature", "output_signature", "second_failure", "plain_failure"])
def test_task_bad_gpu_node_requeues_once(tmp_path, case):
    """A task whose GPU cannot start ("No devices were found" from nvidia-smi -L, or "CUDA unknown error" in its output)
    adds its node to bad_nodes.txt, writes its diagnosis, exits 75 and requeues itself once with the node excluded
    (scontrol update ExcNodeList, scontrol requeue); after a requeue (SLURM_RESTART_COUNT 1) it does not requeue
    again; a failure without either signature keeps its own exit code and marks no node."""
    out = (tmp_path / "out").as_posix()
    fake = tmp_path / "bin"
    fake.mkdir()
    slog = (tmp_path / "scontrol.log").as_posix()
    gpu = "No devices were found" if case in ("nvidia_smi_signature", "second_failure") else "GPU 0: card"
    (fake / "nvidia-smi").write_bytes(('#!/bin/bash\necho "%s"\n' % gpu).encode())
    (fake / "scontrol").write_bytes(('#!/bin/bash\necho "$*" >> "%s"\n' % slog).encode())
    for f in ("nvidia-smi", "scontrol"):
        os.chmod(str(fake / f), 0o755)
    cmd = {"output_signature": 'echo "RuntimeError: CUDA unknown error"; exit 1', "plain_failure": "exit 3"}.get(case, "true")
    lst = tmp_path / "list.tsv"
    lst.write_bytes(("g2_gen.s4_row.001\t%s\n" % cmd).encode())
    os.makedirs(str(tmp_path / "home"))
    env = dict(os.environ, REPO=TOP.replace("\\", "/"), OUT_ROOT=out, HOME=(tmp_path / "home").as_posix(),
               SLURM_ARRAY_TASK_ID="1", SLURM_ARRAY_JOB_ID="500", SLURM_JOB_ID="501", SLURMD_NODENAME="n03",
               SLURM_RESTRICT="--exclude=n05,n06", PATH=fake.as_posix() + os.pathsep + os.environ.get("PATH", ""))
    for v in ("HF_HOME", "SETTLE", "WEIGHTS_ROOT", "IFM_CKPT_ROOT", "XLLM_ROOT", "DATA_ROOT", "MODULES"):
        env.pop(v, None)
    if case == "second_failure":
        env["SLURM_RESTART_COUNT"] = "1"
    r = subprocess.run([_bash(), os.path.join(JOBS, "task_g2_gen.sbatch").replace("\\", "/"), lst.as_posix()],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    dg = open(os.path.join(out, "_diag", "g2_gen.s4_row.001.txt")).read()
    calls = open(slog).read().splitlines() if os.path.exists(slog) else []
    if case == "plain_failure":
        assert r.returncode == 3 and not os.path.exists(os.path.join(out, "bad_nodes.txt")) and not calls, so[-1500:]
        return
    assert r.returncode == 75, so[-1500:]
    assert open(os.path.join(out, "bad_nodes.txt")).read().split() == ["n03"]
    assert "GPU start on node n03" in dg and not os.path.exists(os.path.join(out, "_done", "g2_gen.s4_row.001"))
    if case == "second_failure":
        assert not calls and "not requeued" in dg
    else:
        assert calls == ["update JobId=500_1 ExcNodeList=n03,n05,n06", "requeue 500_1"], calls
        assert "requeued once with --exclude=n03,n05,n06" in dg


# ------------------------------------------------------------------ cluster extension (2026-10-11)
def _base_lists():
    """Stage lists and templates of the groups the running cluster has queued: the extension must not touch them."""
    return {"g0_xllm", "g1_copy", "g2_gen", "g3_map", "g4_cd", "g5_huginn", "g6_ifm", "g7_ra23", "g9_merge"}


def test_ext1_ouro_2_6b_think_generation_pipelines():
    """Item 1: setting a4 (Ouro-2.6B-Thinking) on GSM8K, SVAMP (steer_gen) and ASDiv, MAWPS (steer_gen_math) in its
    own group g11_gen26t: four pipelines, hold unit gen, the 2.6B Thinking hub folder, SETTLE files shipped."""
    sg = open(os.path.join(PILOT, "tree", "prod", "steer_gen.py"), "rb").read().decode("utf-8")
    assert '"ouro_2_6b_think": {"layers": 48, "modeling_md5_prefix": "4135d08e", "local_env": "K17_THINK_DIR"}' in sg
    assert '"a4": {"arm": "a", "model": "ouro_2_6b_think"' in sg
    for name in ("cells_ouro_2_6b_think_gsm8k_natural2_k4.jsonl", "cells_ouro_2_6b_think_svamp_natural2_k4.jsonl"):
        b = open(os.path.join(CL, "data", name), "rb").read()
        assert b"/home/" not in b and json.loads(b.split(b"\n", 1)[0])["tag"] == name[6:-6]
    for lists in (JOBS, os.path.join(JOBS, "nolocal")):
        with open(os.path.join(lists, "joblist.tsv"), encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f, delimiter="\t") if r["model"] == "ouro_2_6b_think"
                    and r["script"] in ("steer_gen.py", "steer_gen_math.py")]
        pipes = {(r["group"], r["script"], r["out"].rsplit("/", 1)[1]) for r in rows}
        assert pipes == {("g11_gen26t", "steer_gen.py", "a4_gsm8k"), ("g11_gen26t", "steer_gen.py", "a4_svamp"),
                         ("g11_gen26t", "steer_gen_math.py", "a4_asdiv"), ("g11_gen26t", "steer_gen_math.py", "a4_mawps")}
        assert len(rows) == 28 + 27 + 26 + 26
        for r in rows:
            assert GJ.family_of(r["cmd"]) == "gen"
            assert "--setting a4" in r["cmd"] and GJ.THINK26 in r["cmd"] and GJ.THINK14 not in r["cmd"]
            assert "jobs/gen_cfg/a4_" in r["cmd"]
        for g in _base_lists():
            assert not [r for r in rows if r["group"] == g]
        assert os.path.exists(os.path.join(lists, "task_g11_gen26t.sbatch"))
    assert GJ.GROUP_ORDER.index("g11_gen26t") < GJ.GROUP_ORDER.index("g5_huginn")


def test_ext_items_1_3_lines_routes_and_arms_flag(tmp_path, monkeypatch):
    """Items 1 and 3: seven k16_path_ouro lines on the Thinking models (g13_paththink, unit ouro), seven k16_path_ifm
    --arms Y1 lines (g14_ifmy1, unit ifm, after the xllm build); the seven edit-site map cells read the _y1 folders
    and their first-run folders stay read through their own card; --arms left out keeps every model's arms."""
    import copy
    import importlib.util
    with open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    t = [r for r in rows if r["group"] == "g13_paththink"]
    y = [r for r in rows if r["group"] == "g14_ifmy1"]
    assert len(t) == 7 and len(y) == 7
    assert all(GJ.family_of(r["cmd"]) == "ouro" and "--arms w2 --min_items 40" in r["cmd"] for r in t)
    assert {r["out"].rsplit("/", 1)[1] for r in t} == {"ouro_think_D1", "ouro_think_D2", "ouro_think_D4", "ouro26_think_D1",
                                                       "ouro26_think_D2", "ouro26_think_D4", "ouro26_think_D5"}
    assert all(GJ.THINK26 in r["cmd"] for r in t if "ouro26_think" in r["out"])
    assert all(GJ.family_of(r["cmd"]) == "ifm" and "--arms Y1" in r["cmd"] and r["out"].endswith("_y1")
               and r["depends_on"] == "g0_xllm.s1_build.001" for r in y)
    for m, ds in GJ.Y1_CELLS:
        assert GJ.cell_target("edit-site map", m, ds)[1] == "B/k16_path_ifm/%s_%s_y1" % (m, ds)
        assert GJ.cell_target(GJ.FIRST_RUN_CARD, m, ds)[1] == "B/k16_path_ifm/%s_%s" % (m, ds)
    assert GJ.cell_target(GJ.FIRST_RUN_CARD, "ifm_p1_ouro", "D4")[1] == "B/k16_path_ifm/ifm_p1_ouro_D4"
    w = json.load(open(os.path.join(CL, "weights.json"), encoding="utf-8"))
    assert [c for c in w["checkpoints"] if c["key"] == "ouro_2_6b_think"][0]["registry_names"]["k16_path_ouro"] == "ouro26_think"
    assert '"ouro26_think": "ByteDance/Ouro-2.6B-Thinking"' in open(os.path.join(PILOT, "k16_path_ouro.py"), encoding="utf-8").read()
    spec = importlib.util.spec_from_file_location("k16i_flag", os.path.join(PILOT, "k16_path_ifm.py"))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(PILOT)
    spec.loader.exec_module(mod)
    base = copy.deepcopy(mod.ARMS)
    calls = []
    monkeypatch.setattr(mod, "run_model", lambda name, *a, **k: calls.append((name, list(mod.ARMS[name]))))
    monkeypatch.setattr(mod, "write_proc", lambda *a, **k: None)
    for extra, want in (([], base["ifm_p1_mid"]), (["--arms", "Y1"], ["Y1"])):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["k16_path_ifm.py", "--models", "ifm_p1_mid", "--datasets", "D2", "--out",
                                          str(tmp_path / ("o%d" % len(extra))), "--guard", "100000000", "--full_pool"] + extra)
        mod.main()
        assert calls == [("ifm_p1_mid", want)]
        if not extra:
            assert mod.ARMS == base


def test_ext_item5_ifm_vbd_v1_rerun():
    """Item 5: the IFM k17_varbench pipelines rerun with --vbd_v1 in group g12_ifmv1 and folders <key>_v1 (window and
    scan included); the first g6_ifm pipelines keep their lines; the merge reads _v1; the plan names the new code."""
    with open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="	") if r["script"] == "k17_varbench.py"]
    v1 = [r for r in rows if r["group"] == "g12_ifmv1"]
    old = [r for r in rows if r["group"] == "g6_ifm"]
    assert {r["model"] for r in v1} == GJ.VB_V1 == {r["model"] for r in old}
    assert len(v1) == len(old) == 27 and {r["stage"] for r in v1} == {r["stage"] for r in old}
    for r in v1:
        k = GJ.VB_KEY[r["model"]]
        assert r["out"] == "$OUT_ROOT/B/k17_varbench/%s_v1" % k
        assert ("--vbd_v1 %s=1" % k in r["cmd"]) == (r["stage"] != "s6_summarize")      # summarize reads records only
        assert GJ.family_of(r["cmd"]) == "ifm"
    assert not [r for r in old if "--vbd_v1" in r["cmd"] or r["out"].endswith("_v1")]
    assert not [r for r in rows if r["model"] not in GJ.VB_V1 and "--vbd_v1" in r["cmd"]]
    for m in GJ.VB_V1:
        assert GJ.cell_target("GSM8K copy and derived", m, "VBD")[1].endswith("_v1")
    sys.path.insert(0, PILOT)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kvb_item5", os.path.join(PILOT, "k17_varbench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ph = mod.plan_hashes(os.path.join(CL, "plans", "k17_varbench_plan.md"))
    assert ph["plan_code_sha256"] == mod.code_hashes(os.path.join(PILOT, "k17_varbench.py"))["code_sha256"]
    assert mod.VBD_V1 == {"ouro": True, "ouro_think": True, "ouro26": True, "ouro26_think": True,
                          "ifm_p1_ouro": False, "ifm_p1_ouro_inj": False, "ifm_d112": False}     # defaults unchanged


def test_ext_item4_huginn_generation_pipeline():
    """Item 4: Huginn GSM8K generation through steer_gen_raven.py --setting hug in group g15_hgen (unit mcleish):
    two screens, select, the 22 rows, analyze, on the shipped prep; the McLeish pipeline is unchanged."""
    with open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="	") if r["group"] == "g15_hgen"]
    assert [r["stage"] for r in rows] == ["s2_screen"] * 2 + ["s3_select"] + ["s4_row"] * 22 + ["s6_analyze"]
    assert sorted(r["shard"] for r in rows if r["stage"] == "s4_row") == sorted(GJ.GENR_ROWS)
    for r in rows:
        assert r["model"] == "huginn_0125" and r["out"] == "$OUT_ROOT/A/steer_gen_raven/huginn"
        assert GJ.family_of(r["cmd"]) == "mcleish" and "--setting hug" in r["cmd"]
        assert "$DATA_ROOT/huginn_setting/prep.json" in r["cmd"]
    assert os.path.exists(os.path.join(PILOT, "data", "huginn_setting", "prep.json"))
    assert GJ.cell_target("generation placement", "huginn_0125", "GSM8K_gen")[1] == "A/steer_gen_raven/huginn"
    assert GJ.cell_target("generation placement", "mcleish_llama32_r32", "GSM8K_gen")[1] == GJ.GENR_DIR


def test_k15_full_d4_option_keeps_defaults_and_loads_d4(monkeypatch):
    """Figure 2 code panel: k15_full --datasets keeps D1, D2, CF as the default (the existing lines), the arm
    strengths and seeds of D1, D2, CF are unchanged, D4 rows load from the k16_path_ifm data, and
    g16_ifmgrid.s1_run.002 runs D4 into its own folder that the edit-site map code cell of ifm_p1_ouro reads."""
    import importlib.util
    monkeypatch.syspath_prepend(PILOT)
    spec = importlib.util.spec_from_file_location("k15f_d4", os.path.join(PILOT, "k15_full.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DATASETS == ["D1", "D2", "CF"] and mod.KNOWN_DATASETS[:3] == mod.DATASETS
    assert (mod.A_S, mod.A_A) == ([0.5, 1.0, 2.0], [1.0, 2.0, 4.0])
    assert [1000 * (mod.KNOWN_DATASETS.index(d) + 1) for d in mod.DATASETS] == [1000, 2000, 3000]
    src = open(os.path.join(PILOT, "k15_full.py"), encoding="utf-8").read()
    assert '{"arm": "S", "op": "O2", "a": a, "site": lab' in src and '{"arm": "S", "op": "O4", "a": a, "site": lab' in src
    monkeypatch.setattr(mod, "D4_DIR", [os.path.join(PILOT, "data", "k16_path_ifm")])
    cands, quota, sel = mod.d45_candidates("D4")
    assert len(cands) == 806 and quota == {"T1": 20, "T2": 20} and sel.__name__ == "d45_select"
    assert all(p["half"] == "test" and p["pos"] and p["para"] and p["id"].startswith("d4-") for p in cands)
    cands, quota, sel = mod.d45_candidates("D5")
    assert len(cands) == 594 and quota == {"T2": 40} and sel.__name__ == "d45_select"
    assert all(p["half"] == "test" and p["pos"] and p["para"] and p["id"].startswith("d5-") for p in cands)
    with open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8") as f:
        g = [r for r in csv.DictReader(f, delimiter="	") if r["group"] == "g16_ifmgrid"]
    assert [r["id"] for r in g] == ["g16_ifmgrid.s1_run.001", "g16_ifmgrid.s1_run.002"]
    assert "--datasets" not in g[0]["cmd"] and g[0]["out"].endswith("/ifm_p1_ouro_grid")
    assert "--datasets D4 --data_dir $DATA_ROOT/k16_path_ifm" in g[1]["cmd"] and g[1]["out"].endswith("_grid_D4")
    assert all(GJ.family_of(r["cmd"]) == "ifm" and r["depends_on"] == "g0_xllm.s1_build.001" for r in g)
    assert GJ.cell_target("edit-site map", "ifm_p1_ouro", "D4")[1] == "B/k15_full/ifm_p1_ouro_grid_D4"
    assert GJ.cell_target(GJ.FIRST_RUN_CARD, "ifm_p1_ouro", "D4")[1] == "B/k16_path_ifm/ifm_p1_ouro_D4"
    from cluster.extract import k15full
    assert k15full.DS["D4"] == "code"


def test_k15_full_datasets_flag_main(monkeypatch, tmp_path):
    """k15_full main(): no --datasets keeps D1, D2, CF (the existing lines); --datasets D4 needs --data_dir; an unknown
    name is refused."""
    import importlib.util
    monkeypatch.syspath_prepend(PILOT)
    spec = importlib.util.spec_from_file_location("k15f_main", os.path.join(PILOT, "k15_full.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "selftest", lambda: True)
    for extra, want in (([], ["D1", "D2", "CF"]), (["--datasets", "D4", "--data_dir", "x"], ["D4"]),
                        (["--datasets", "D1,CF"], ["D1", "CF"])):
        monkeypatch.setattr(sys, "argv", ["k15_full.py", "--selftest", "--out", str(tmp_path)] + extra)
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert e.value.code == 0 and mod.DATASETS == want, extra
    for extra in (["--datasets", "D4"], ["--datasets", "D1,D5"], ["--datasets", "D3"], ["--datasets", ""]):
        monkeypatch.setattr(sys, "argv", ["k15_full.py", "--selftest", "--out", str(tmp_path)] + extra)
        with pytest.raises(AssertionError):
            mod.main()
    mod.DATASETS[:] = ["D1", "D2", "CF"]
    assert mod.KNOWN_DATASETS == ["D1", "D2", "CF", "D4", "D5"] and mod.MODEL_ORDER[:5] == [
        "ouro", "ouro26", "ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_d112"] and mod.MODEL_ORDER[5:] == ["ifm_d28"]


# ------------------------------------------------------------------ B-EXT5: the fill round
def _nl_rows(group=None):
    with open(os.path.join(JOBS, "nolocal", "joblist.tsv"), encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if group is None or r["group"] == group]


def test_fill_groups_lines_routes_and_units():
    """B-EXT5 fill: g17_fillmap holds d28 (k15_full D1, D2, CF, D4, D5 in one process), the Y1 arm on eight 819M cells
    and the k17_public ouro_think run on Ouro-2.6B-Thinking (D1, D2, D4); Ouro-2.6B-Thinking D5 reads the existing
    pathway folder (no job); g18_fillgen holds five generation pipelines (prep, screen A and B, select, 22 rows,
    analyze) on the raven venv, unit mcleish; McLeish SVAMP stays a skip; the fill priorities match submit_fill.sh."""
    rows = _nl_rows()
    g17 = [r for r in rows if r["group"] == "g17_fillmap"]
    d28 = [r for r in g17 if r["model"] == "ifm_d28"]
    assert len(d28) == 1 and "--datasets D1,D2,CF,D4,D5 --data_dir $DATA_ROOT/k16_path_ifm" in d28[0]["cmd"]
    assert d28[0]["out"] == "$OUT_ROOT/B/k15_full/ifm_d28" and GJ.family_of(d28[0]["cmd"]) == "ifm"
    for ds in ("D1", "D2", "CF", "D4", "D5"):
        assert GJ.cell_target("edit-site map", "ifm_d28", ds)[1] == "B/k15_full/ifm_d28"
    y1 = [r for r in g17 if r["script"] == "k16_path_ifm.py"]
    assert {(r["model"], r["datasets"]) for r in y1} == GJ.Y1_FILL and len(y1) == 8
    for r in y1:
        assert "--arms Y1" in r["cmd"] and r["out"].endswith("_%s_y1" % r["datasets"]) and GJ.family_of(r["cmd"]) == "ifm"
        assert r["depends_on"] == "g0_xllm.s1_build.001"
        assert GJ.cell_target("edit-site map", r["model"], r["datasets"])[1] == r["out"][len("$OUT_ROOT/"):]
    pub = [r for r in g17 if r["script"] == "k17_public.py"]
    old = [r for r in rows if r["script"] == "k17_public.py" and r["model"] == "ouro_think" and "ouro_think_D" in r["out"]]
    assert sorted({r["datasets"] for r in pub}) == ["D1", "D2", "D4"]
    noshard = lambda c: re.sub(r" --shard \d+/\d+", "", c)          # noqa: E731
    for r in pub:                                       # the ouro_think command with the model key swapped
        twin = [o for o in old if o["stage"] == r["stage"] and o["datasets"] == r["datasets"]][0]
        assert noshard(r["cmd"]).replace("ouro26_think", "ouro_think") == noshard(twin["cmd"]), r["id"]
        assert "--model ouro26_think" in r["cmd"] or r["stage"] == "s4_summarize"
        assert GJ.family_of(r["cmd"]) == "raven"
    assert GJ.cell_target("edit-site map", "ouro_2_6b_think", "D4")[1] == "B/k17_public/ouro26_think_D4"
    assert GJ.cell_target("edit-site map", "ouro_2_6b_think", "D5")[1] == "C/k16_path_ouro/ouro26_think_D5"
    assert not [r for r in rows if r["out"].endswith("ouro26_think_D5") and r["group"] != "g13_paththink"]
    g18 = [r for r in rows if r["group"] == "g18_fillgen"]
    pipes = {}
    for r in g18:
        pipes.setdefault((r["model"], r["datasets"], r["out"]), []).append(r)
    assert {k[:2] for k in pipes} == {("huginn_0125", "SVAMP_gen"), ("mcleish_llama32_r32", "ASDIV_gen"),
                                      ("huginn_0125", "ASDIV_gen"), ("mcleish_llama32_r32", "MAWPS_gen"),
                                      ("huginn_0125", "MAWPS_gen")}
    for (m, ds, out), rr in pipes.items():
        assert [r["stage"] for r in rr] == ["s1_prep", "s2_screen", "s2_screen", "s3_select"] + ["s4_row"] * 22 + ["s6_analyze"]
        assert sorted(r["shard"] for r in rr if r["stage"] == "s4_row") == sorted(GJ.GENR_ROWS)
        assert out[len("$OUT_ROOT/"):] == GJ.cell_target("generation placement", m, ds)[1]
        for r in rr:
            c = r["cmd"]
            assert c.startswith("source $VENV_RAVEN/bin/activate && ") and GJ.family_of(c) == "mcleish"
            if ds == "SVAMP_gen":
                assert "K17_DATASET=svamp" in c and "steer_gen_raven.py" in c and "--setting hug_svamp" in c
            else:
                assert "steer_gen_math.py --dataset %s --rows-dir %s mcleish " % (GJ.GENM_DS[ds], out) in c
                assert ("--setting %s " % GJ.GENR_SET[m]) in c
        scr = [r for r in rr if r["stage"] == "s2_screen"]
        sel = [r for r in rr if r["stage"] == "s3_select"][0]
        if ds == "SVAMP_gen":
            assert "--settle-dir $SETTLE" in rr[0]["cmd"] and all("--hi" not in r["cmd"] for r in scr)
        else:
            n = GJ.GENR_SCREEN[m]
            assert all("--hi %d " % n in r["cmd"] for r in scr) and "--n-screened %d" % n in sel["cmd"]
        assert "--script steer_gen_raven.py --src $REPO/loopedcot/cluster/pilot/steer_gen_raven.py" in rr[-1]["cmd"]
    assert GJ.route("generation placement", "mcleish_llama32_r32", "SVAMP_gen") is None
    sf = open(os.path.join(JOBS, "submit_fill.sh"), encoding="utf-8").read()
    assert 'FILL="%s"' % " ".join("%s:%d" % kv for kv in sorted(GJ.FILL_PRIORITY.items(), key=lambda kv: kv[1])) in sf
    assert "loopedcot/cluster/jobs/submit_fill.sh" in GJ.handover_files()


def test_fill_scripts_new_settings_and_models():
    """k17_public knows Ouro-2.6B-Thinking (revision of weights.json, the Ouro tokenizer family, the ouro_think targets,
    48 layers); steer_gen_raven setting hug_svamp is hug on the SVAMP pairs with Huginn's SETTLE k32 SVAMP file
    (shipped); results rows name the setting's dataset; k16_path_ifm lists the 819M designs."""
    import importlib.util
    w = {c["key"]: c for c in json.load(open(os.path.join(CL, "weights.json"), encoding="utf-8"))["checkpoints"]}
    src = open(os.path.join(PILOT, "k17_public.py"), encoding="utf-8").read()
    assert '"ouro26_think": ("ByteDance/Ouro-2.6B-Thinking", "%s")' % w["ouro_2_6b_think"]["revision"] in src
    assert w["ouro_2_6b_think"]["registry_names"]["k17_public"] == "ouro26_think"
    code = ("import sys, json; sys.path.insert(0, %r); import k17_public as P; "
            "print(json.dumps([P.TOKFAM['ouro26_think'], P.TARGET['ouro26_think'] == P.TARGET['ouro_think'], "
            "P.arch_of('ouro26_think'), P.GROUP_FAM['ouro26_think'], sorted(P.OURO_GROUPS)]))" % PILOT)
    out = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).decode().strip().splitlines()[-1])
    assert out == ["ouro", True, ["ouro", 4, 48], "ouro", ["ouro26_think", "ouro_think"]]
    code = ("import sys, json; sys.path.insert(0, %r); import steer_gen_raven as R; "
            "c = R.S.SETTINGS['hug_svamp']; h = R.S.SETTINGS['hug']; "
            "an = {'setting': 'hug_svamp', 'rows': {'r00': {'gold_A': [1.0, 1.0, 1.0, 3, 3]}}}; "
            "rr = R.results_rows(an); rg = R.results_rows(dict(an, setting='hug')); "
            "print(json.dumps([c['dataset'], c['settle'], {k: v for k, v in c.items() if k not in ('dataset', 'settle')} == "
            "{k: v for k, v in h.items() if k != 'settle'}, R.SPEC['hug_svamp'] == R.SPEC['hug'], "
            "sorted({r['dataset'] for r in rr}), sorted({r['dataset'] for r in rg}), rr[0]['model']]))" % PILOT)
    out = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).decode().strip().splitlines()[-1])
    assert out == ["svamp", "cells_huginn_0125_svamp_natural_k32.jsonl", True, True, ["svamp"], ["gsm8k"], "huginn_0125"]
    assert os.path.exists(os.path.join(CL, "data", "cells_huginn_0125_svamp_natural_k32.jsonl"))
    spec = importlib.util.spec_from_file_location("k16i_fill", os.path.join(PILOT, "k16_path_ifm.py"))
    assert all(m in open(spec.origin, encoding="utf-8").read() for m in ('"ifm_p2m_r5": ["Y2"]', '"ifm_p2m_ortho": ["Y2"]'))


def test_steer_gen_math_raven_mode_installs_both_settings(tmp_path):
    """steer_gen_math.py mcleish mode: the dataset goes into both raven settings (mcl, hug) with no SETTLE file; prep
    reads the pool span of the active setting's tokenizer (huginn for --setting hug)."""
    code = ("import sys, json; sys.path.insert(0, %r); import steer_gen_math as M; S = M.install('asdiv', %r); "
            "R = M.install_mcleish(S); R.ACTIVE[0] = 'hug'; "
            "print(json.dumps([[R.S.SETTINGS[s].get('dataset'), R.S.SETTINGS[s].get('settle')] for s in ('mcl', 'hug')] + "
            "[R.spec()['pool_tok'], R.prep.__module__ == M.__name__ or R.prep.__qualname__]))" % (PILOT, str(tmp_path)))
    out = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code], cwd=str(tmp_path),
                                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).decode().strip().splitlines()[-1])
    assert out[:2] == [["asdiv", None], ["asdiv", None]] and out[2] == "huginn"
    src = open(os.path.join(PILOT, "steer_gen_math.py"), encoding="utf-8").read()
    assert 'ps = pr["tok"][R.spec()["pool_tok"]]' in src and 'pr["tok"]["mcleish"]' not in src


def _fill_env(tmp_path, queue="", named="", merge_pending=True):
    env, out, log, _ = _resubmit_env(tmp_path, queue=queue, named=named, merge_pending=merge_pending)
    (out / "_done" / "g0_xllm.s1_build.001").write_text("ok\n")
    return env, out, log


@needs_bash
def test_submit_fill_priorities_done_queued_and_merge(tmp_path):
    """submit_fill.sh: --dry-run prints every fill pipeline and card-hours per priority and submits nothing; without
    the xllm build marker the IFM pipelines are named and not submitted; a "no" submits nothing; with yes it submits
    only the fill groups, never a task with a done marker, leaves a pipeline with a queued task alone, chains the
    stages of a pipeline with afterok, in priority order, and the pending merge waits for every new job."""
    sf = os.path.join(JOBS, "submit_fill.sh").replace("\\", "/")
    rows = _nl_rows()
    fill = [r for r in rows if r["group"] in GJ.FILL_PRIORITY]
    hours = {p: sum(float(r["est_card_seconds"]) for r in fill if GJ.FILL_PRIORITY[r["group"]] == p) / 3600 for p in (1, 2, 3)}
    env, out, log = _fill_env(tmp_path / "a")
    r = subprocess.run([_bash(), sf, "--dry-run"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0 and "dry run: nothing submitted" in so and not os.path.exists(log), so[-2000:]
    for p in (1, 2, 3):
        n = sum(1 for r_ in fill if GJ.FILL_PRIORITY[r_["group"]] == p)
        assert "priority %d: %d tasks, %.1f card-hours" % (p, n, hours[p]) in so, so[-1500:]
    # no xllm build marker: the IFM pipelines are named, the others planned
    env, out, log = _fill_env(tmp_path / "b")
    os.remove(str(out / "_done" / "g0_xllm.s1_build.001"))
    r = subprocess.run([_bash(), sf, "--dry-run"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0 and so.count("NOT SUBMITTED: needs the done marker of g0_xllm.s1_build.001") == 9 + 3, so[-3000:]
    # a no: nothing
    env, out, log = _fill_env(tmp_path / "c")
    r = subprocess.run([_bash(), sf], env=env, input=b"no\n", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 1 and b"not submitted" in r.stdout and not os.path.exists(log)
    # the real round: g12 done but one pipeline, one g18 pipeline half done, one g17 task queued
    g12 = [r_ for r_ in fill if r_["group"] == "g12_ifmv1"]
    left = [r_ for r_ in g12 if r_["model"] == "ifm_d112"]
    g18 = [r_ for r_ in fill if r_["group"] == "g18_fillgen" and r_["out"].endswith("/mcleish_asdiv")]
    q = [r_ for r_ in fill if r_["group"] == "g17_fillmap" and r_["model"] == "ifm_p2m_r5" and r_["datasets"] == "D4"][0]
    qi = [ln.split("\t")[0] for ln in open(os.path.join(JOBS, "nolocal", "g17_fillmap.s1_run.tsv"), encoding="utf-8")].index(q["id"]) + 1
    env, out, log = _fill_env(tmp_path / "d", queue="8001 %d\n" % qi)
    with open(out / "_chain" / "submitted.tsv", "a") as f:
        f.write("8001 g17_fillmap.s1_run.tsv %d\n" % qi)
    for r_ in [r_ for r_ in g12 if r_ not in left] + [r_ for r_ in g18 if r_["stage"] in ("s1_prep", "s2_screen")] \
            + [r_ for r_ in g18 if r_["stage"] == "s4_row"][:3]:
        (out / "_done" / r_["id"]).write_text("ok\n")
    env["FILL_YES"] = "yes"
    r = subprocess.run([_bash(), sf], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    so = r.stdout.decode()
    assert r.returncode == 0, so[-3000:]
    assert "in the queue, left alone ( %s)" % q["id"] in so
    calls = open(log).read().splitlines()
    sb = [c for c in calls if not c.startswith("scontrol")]
    assert all(any("task_%s.sbatch %s." % (g, g) in c for g in GJ.FILL_PRIORITY) for c in sb), sb
    pri = [GJ.FILL_PRIORITY[c.split()[-1].split(".")[0]] for c in sb]
    assert pri == sorted(pri) and pri[0] == 1 and pri[-1] == 3        # submitted in priority order
    lists = {}
    for c in sb:
        L = c.split()[-1]
        idx = c.split("--array=")[1].split("%")[0].split(",")
        lists.setdefault(L, []).extend(int(i) for i in idx)
    sub_ids = set()
    for L, idx in lists.items():
        ids = [ln.split("\t")[0] for ln in open(os.path.join(JOBS, "nolocal", L), encoding="utf-8")]
        sub_ids |= {ids[i - 1] for i in idx}
    want = {r_["id"] for r_ in fill if not (out / "_done" / r_["id"]).exists() and r_["id"] != q["id"]}
    assert sub_ids == want, (sorted(sub_ids - want)[:5], sorted(want - sub_ids)[:5])
    def call(L, rid):                                    # the sbatch call that holds task rid of stage list L
        ids = [ln.split("\t")[0] for ln in open(os.path.join(JOBS, "nolocal", L), encoding="utf-8")]
        k = [n for n, c in enumerate(sb) if c.split()[-1] == L and
             str(ids.index(rid) + 1) in c.split("--array=")[1].split("%")[0].split(",")]
        assert len(k) == 1, (L, rid)
        return k[0], sb[k[0]]
    ks, cs = call("g18_fillgen.s3_select.tsv", [r_ for r_ in g18 if r_["stage"] == "s3_select"][0]["id"])
    assert "--dependency" not in cs                     # prep and both screens done: the select waits for nothing
    kr, cr = call("g18_fillgen.s4_row.tsv", [r_ for r_ in g18 if r_["stage"] == "s4_row"][5]["id"])
    assert "--dependency=afterok:%d" % (1001 + ks) in cr and len(cr.split("--array=")[1].split("%")[0].split(",")) >= 19
    ka, ca = call("g18_fillgen.s6_analyze.tsv", [r_ for r_ in g18 if r_["stage"] == "s6_analyze"][0]["id"])
    assert "--dependency=afterok:%d" % (1001 + kr) in ca
    for r_ in left:
        call("%s.%s.tsv" % (r_["group"], r_["stage"]), r_["id"])
    sc = [c for c in calls if c.startswith("scontrol")]
    assert len(sc) == 1 and "JobId=9999" in sc[0]
    assert sc[0].split("Dependency=afterany:")[1] == "5001:5002:" + ":".join(str(1001 + k) for k in range(len(sb)))
