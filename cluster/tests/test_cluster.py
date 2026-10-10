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
            assert r["model"] in ("ouro_1_4b_base", "ouro_2_6b_base", "ouro_1_4b_think")
        if r["script"] == "steer_gen_math.py":
            assert r["model"] in ("ouro_1_4b_base", "ouro_2_6b_base", "ouro_1_4b_think") and "screen_A_from_settle" not in cmd
        if r["script"] == "steer_gen_raven.py":
            assert r["model"] == "mcleish_llama32_r32" and "--setting" not in cmd


def test_job_paths_and_configs():
    for r in joblist():
        for p in re.findall(r"\$REPO/(loopedcot/\S+?\.(?:py|json))\b", r["cmd"]):
            assert os.path.exists(os.path.join(TOP, p)), (r["id"], p)
        assert "loop-flow" not in r["cmd"] and "WEIGHTS_ROOT/ouro" not in r["cmd"]
    cfgs = sorted(glob.glob(os.path.join(JOBS, "gen_cfg", "*.json")))
    assert len(cfgs) == 13                       # 12 job configs + smoke_a1_gsm8k.json (cluster/env/smoke_cluster.sh)
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
            assert r["group"] == "g6_ifm", r["id"]
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
    for ds in ("D1", "D2", "CF"):
        assert ("edit-site map", "ifm_d28", ds) in keys
    for k in (("edit-site map", "ifm_p2m_diag", "D4"), ("edit-site map", "ifm_p2m_diag", "D5"),
              ("loops against depth", "mcleish_llama32_r32", "D1"), ("loops against depth", "mcleish_llama32_r32", "D2")):
        assert k in keys
    assert len(d["changes"]) == 23


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
    rows, table, mark = MG.shared_readout(m, cells, info, "OUT", PILOT)
    st = {t[3]: t[7] for t in table}
    assert st["pair"] == "shared" and mark[("edit-site map", "a", "D1")] == "shared n 50"
    assert st["trio"].startswith("own only (shared n 30") and mark[("edit-site map", "c", "D1")].startswith("own only")
    assert st["mix"] == "not comparable" and st["h"] == "shared = own"
    assert sorted(calls) == [("a", 50), ("b", 50)] and len(rows["B"]) == 2
    info[("edit-site map", "b", "D1")]["state"] = "pending"
    rows, table, mark = MG.shared_readout(m, cells, info, "OUT", PILOT)
    assert {t[3]: t[7] for t in table}["pair"] == "pending"


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
        sh = {(r["card"], r["dataset"], r["row"]): r for r in csv.DictReader(f)}
    mid = sh[("edit-site map", "D1", "ifm_mid_pair")]
    assert mid["done_models"] == "ifm_p1_mid,ifm_p1_mid_inj"
    sets = [EXR.get("k15_pairs.py").item_sets(str(out / "B" / "k15_pairs" / m))[(m, "facts")][0]
            for m in ("ifm_p1_mid", "ifm_p1_mid_inj")]
    n_mid = len(sets[0] & sets[1])
    assert int(mid["n_shared"]) == n_mid
    with open(str(dest / "results_all_B.csv"), encoding="utf-8") as f:
        rr = list(csv.DictReader(f))
    shared_mid = [r for r in rr if r["item_set"] == "shared" and r["cell_model"] == "ifm_p1_mid"]
    if n_mid >= 40:
        assert mid["state"] == "shared" and shared_mid
        assert max(int(r["n_items"]) for r in shared_mid) == n_mid
    else:
        assert mid["state"].startswith("own only") and not shared_mid
    par = sh[("edit-site map", "D1", "parcae")]
    assert par["done_models"] == "parcae_370m,parcae_1_3b"
    grid = open(str(dest / "grid_B.md"), encoding="utf-8").read()
    assert "CF (CounterFact): CounterFact pool of the cluster run" in grid
    assert ("shared n %d" % n_mid in grid) if n_mid >= 40 else ("own only (shared n %d" % n_mid in grid)
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
    mcl = [r for r in rows if r["script"] == "steer_gen_raven.py" and r["stage"] == "s4_row"]
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
@pytest.mark.parametrize("case", ["pass", "tolerance_fail", "override", "identity_fail"])
def test_smoke_cluster_sh_gate_and_override(tmp_path, case):
    """smoke_cluster.sh end to end with stand-in task lines: SMOKE_OVERRIDE_REASON passes a tolerance miss only."""
    ref, hashes, values = _gate_reference(tmp_path)
    prepared = (tmp_path / "prepared").as_posix()
    _smoke_tree(prepared, pc_exact=2 if case == "identity_fail" else 3)
    _gate_files(prepared, shift={"pass": 0.0}.get(case, 0.002))
    shutil.rmtree(os.path.join(prepared, "_done"))
    repo, cl, bindir, venv = _fake_smoke_repo(tmp_path, prepared)
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
    if case == "pass":
        assert r.returncode == 0 and ok and not ovr and "SMOKE PASSED:" in so, so[-3000:]
    elif case == "tolerance_fail":
        assert r.returncode == 1 and not ok and "cross-host tolerance" in so, so[-3000:]
    elif case == "override":
        assert r.returncode == 0 and ok and ovr and "OPERATOR OVERRIDE" in so and "operator: drift reviewed" in so, so[-3000:]
    else:
        assert r.returncode == 1 and not ok and not ovr and "not overridable" in so, so[-3000:]


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
    assert r.returncode == 1 and b"must name exactly 4 (--nodes-expected)" in r.stdout
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
                "SLURM_RESTART_COUNT"}
    names -= internal | {n for n in names if re.match(r"^J\d+$", n)}
    exported = set(re.findall(r"\b([A-Z][A-Z0-9_]+)=", e))
    assert not names - exported, sorted(names - exported)
    # --nodes-expected replaces the fixed four; --modules lands in cluster.env
    r = subprocess.run(base + ["--out-root", proj + "/out3", "--nodelist", "n01,n02,n03", "--nodes-expected", "3",
                               "--modules", "cuda/12.4 gcc/12"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert b"must name exactly" not in r.stdout, r.stdout.decode()[-2000:]
    e3 = open(proj + "/out3/cluster.env", encoding="utf-8").read()
    assert 'MODULES="cuda/12.4 gcc/12"' in e3 and "NODES_EXPECTED=3" in e3 and "CARDS=24" in e3
    assert 'SLURM_RESTRICT="--exclude=n04,n05,n06"' in e3


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
    mcl = [c for c in nl.values() if "steer_gen_raven.py run" in c]
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
