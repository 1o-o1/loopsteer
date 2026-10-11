"""merge.py keeps every result row: the Ouro base copy rows of k17_varbench (a cell the manifest marks done earlier,
written again by the VBD job) and the LRE rows of k17_public (labelled facts, the D1 name) reach results_all, and no
two rows of one results_all file share the key merge.LVC_KEY.

    python -B -m pytest cluster/tests/test_merge_rows.py -q -p no:cacheprovider

The artifact test builds a fake output tree from the local cluster runs under $PILOT_ARTIFACTS/cluster_local
(results.csv and items_done.json of each job); the artifacts are never written.
"""
import collections
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.dirname(HERE)
LC = os.path.dirname(CL)
sys.path.insert(0, LC)
sys.dont_write_bytecode = True

from cluster import merge as MG             # noqa: E402
from cluster.extract import common          # noqa: E402

ART = os.environ.get("PILOT_ARTIFACTS", "")
needs_artifacts = pytest.mark.skipif(not (ART and os.path.isdir(ART)), reason="PILOT_ARTIFACTS not set")

# output folder -> the cluster_local run that holds it
LOCAL_RUNS = {"B/k17_varbench/ouro": "g1_copy", "B/k17_varbench/ouro26": "g1_copy",
              "B/k17_varbench/ouro_think": "ra20_c0t", "B/k17_varbench/ouro26_think": "ra20_c0t",
              "B/k17_public/ouro_think_D1": "ra20_c0t", "B/k17_public/ouro_think_D2": "ra20_c0t",
              "B/k17_public/ouro_think_D4": "ra20_c0t", "B/k17_public/ouro_think_LRE": "ra20_tail2",
              "B/k15_full/ifm_p1_ouro": "g3_map/ifm_p1_ouro", "A/steer_gen/a1_gsm8k": "ra20_g2",
              "A/steer_gen/a1_svamp": "ra20_g2", "A/steer_gen/a2_gsm8k": "ra20_g2",
              "A/steer_gen_math/a1_asdiv": "ra20_tail2", "C/k17_freeze/ouro_think_D1": "ra20_tail2",
              "C/k17_freeze/ouro_think_D2": "ra20_tail2", "C/k17_freeze/ouro_think_D4": "ra20_tail2",
              "C/k17_freeze/ouro26_think_D1": "ra20_tail2", "C/k17_freeze/ouro26_think_D2": "ra20_tail2"}


def _merge(out, *extra):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("ALLOW_HOME", None)
    p = subprocess.run([sys.executable, "-B", os.path.join(CL, "merge.py"), "--out_root", str(out)] + list(extra),
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode().strip().splitlines()


def _results_all(out):
    by_grid = {}
    for p in glob.glob(os.path.join(str(out), "merged", "results_all_*.csv")):
        with open(p, encoding="utf-8") as f:
            by_grid[os.path.basename(p)[12:-4]] = list(csv.DictReader(f))
    return by_grid


def _completeness(out):
    with open(os.path.join(str(out), "merged", "completeness.csv"), encoding="utf-8") as f:
        return {(r["card"], r["model"], r["dataset"]): r for r in csv.DictReader(f)}


def _no_shared_key(by_grid):
    for g, rr in by_grid.items():
        kc = collections.Counter(tuple(r[k] for k in MG.LVC_KEY) for r in rr)
        assert not [k for k, v in kc.items() if v > 1], g


# ------------------------------------------------------------------ helpers, no artifacts
def test_cell_row_dataset_label():
    r = {"model": "ouro_think", "dataset": "facts"}
    assert MG.cell_row(r, {"card": "edit-site map", "model": "ouro_1_4b_think", "dataset": "LRE"})["dataset"] == "lre"
    assert MG.cell_row(r, {"card": "edit-site map", "model": "ouro_1_4b_think", "dataset": "D1"})["dataset"] == "facts"
    g = MG.cell_row({"dataset": "gsm8k"}, {"card": "generation placement", "model": "m", "dataset": "GSM8K_gen"})
    assert g["dataset"] == "gsm8k" and g["cell_dataset"] == "GSM8K_gen"
    assert r["dataset"] == "facts"          # the source row is not changed


def _fake_job(root, rel, model, sets, rows):
    """A job folder with items_done.json ({dataset: n items}) and results.csv (rows: (card, dataset, arm))."""
    d = os.path.join(str(root), rel)
    os.makedirs(d)
    json.dump({"%s|%s" % (model, ds): {"planned": [str(i) for i in range(n)], "done": [str(i) for i in range(n)]}
               for ds, n in sets.items()}, open(os.path.join(d, "items_done.json"), "w"))
    common.write_results([common.row(grid="B", card=card, script=script, model=model, dataset=ds, arm=arm,
                                     site="src|L0", loop=1, strength=1, metric="e", endpoint="e", item_set="own",
                                     n_items=3, n_clusters=3, estimate=0.5) for script, card, ds, arm in rows],
                         os.path.join(d, "results.csv"))


def test_merge_keeps_by_product_lre_and_unclaimed_rows(tmp_path):
    """Synthetic tree: the Ouro base VB rows (manifest done earlier) reach results_all, the D1 and LRE rows of
    Ouro-1.4B-Thinking keep distinct keys, an unclaimed row goes in with empty cell columns."""
    out = tmp_path / "out"
    vb = "k17_varbench.py"
    _fake_job(out, "B/k17_varbench/ouro", "ouro", {"gsm8k_copy": 3, "gsm8k_derived": 3},
              [(vb, "edit_site_map", "gsm8k_copy", a) for a in ("O1", "O2")] +
              [(vb, "edit_site_map", "gsm8k_derived", a) for a in ("O1", "O2")] +
              [(vb, "edit_site_map", "facts_vb", "O1"), (vb, "site_selection", "facts_vb", "latest")])
    pub = "k17_public.py"
    _fake_job(out, "B/k17_public/ouro_think_D1", "ouro_think", {"facts": 3}, [(pub, "edit_site_map", "facts", "O1")])
    _fake_job(out, "B/k17_public/ouro_think_LRE", "ouro_think", {"facts": 3}, [(pub, "edit_site_map", "facts", "O1")])
    rc, lines = _merge(out, "--min_kept", "1")
    assert rc == 0, lines
    res = json.loads(lines[-1])
    assert res["duplicate_keys"] == {} and res["rows_without_cell"] == {"B/k17_varbench/ouro": 1}
    comp = _completeness(out)
    vbc = comp[("GSM8K copy and derived", "ouro_1_4b_base", "VB")]
    assert vbc["manifest_status"] == "done" and vbc["state"] == "done" and "done earlier" in vbc["detail"]
    assert comp[("GSM8K copy and derived", "ouro_2_6b_base", "VB")]["state"] == "done earlier (not rerun)"
    rr = _results_all(out)["B"]
    _no_shared_key({"B": rr})
    cnt = collections.Counter((r["model"], r["dataset"], r["cell_dataset"]) for r in rr)
    assert cnt[("ouro", "gsm8k_copy", "VB")] == 2 and cnt[("ouro", "gsm8k_derived", "VBD")] == 2
    assert cnt[("ouro", "facts_vb", "")] == 1 and not [r for r in rr if r["card"] == "site_selection"]
    assert cnt[("ouro_think", "facts", "D1")] == 1 and cnt[("ouro_think", "lre", "LRE")] == 1


# ------------------------------------------------------------------ the local cluster runs
@needs_artifacts
def test_merge_keeps_every_row_of_the_local_cluster_runs(tmp_path):
    out = tmp_path / "out"
    n_in = collections.Counter()
    for rel, run in LOCAL_RUNS.items():
        src = os.path.join(ART, "cluster_local", run, rel)
        d = out / rel
        d.mkdir(parents=True)
        for fn in ("results.csv", "items_done.json"):
            shutil.copy(os.path.join(src, fn), str(d))
        for r in common.read_results(os.path.join(src, "results.csv")):
            if r["card"] not in MG.SCOPE_CARDS:
                n_in[(r["script"], r["model"])] += 1
    rc, lines = _merge(out)
    assert rc == 0, lines
    res = json.loads(lines[-1])
    assert res["duplicate_keys"] == {} and res["rows_without_cell"] == {}
    by_grid = _results_all(out)
    _no_shared_key(by_grid)
    own = [r for rr in by_grid.values() for r in rr if r["item_set"] != "shared"]
    assert collections.Counter((r["script"], r["model"]) for r in own) == n_in
    comp = _completeness(out)
    for m, k in (("ouro_1_4b_base", "ouro"), ("ouro_2_6b_base", "ouro26")):
        assert comp[("GSM8K copy and derived", m, "VB")]["state"] == "done"
        copy = [r for r in own if r["model"] == k and r["dataset"] == "gsm8k_copy"]
        assert len(copy) == 52 and {r["cell_dataset"] for r in copy} == {"VB"}
    ot = collections.Counter((r["dataset"], r["cell_dataset"]) for r in own
                             if r["script"] == "k17_public.py" and r["model"] == "ouro_think")
    assert ot[("facts", "D1")] == 66 and ot[("lre", "LRE")] == 66


# ------------------------------------------------------------------ B-EXT2 item 0
def test_canonical_item_ids_across_scripts():
    from cluster.extract import shared as SH
    facts = [("k17_public.py", "D1|capital|Austria|Norway"), ("k15_full.py", "dev|capital|Austria|Norway"),
             ("k17_freeze.py", "capital|Austria|Norway|heldout"), ("k16_path_ifm.py", "heldout|capital|Austria|Norway")]
    assert {SH.canon(s_, i) for s_, i in facts} == {"capital|Austria|Norway"}
    assert SH.canon("k17_public.py", "D2|op1|10 + 28|40 + 28") == SH.canon("k15_full.py", "all|op1|10 + 28|40 + 28")
    assert SH.canon("k17_public.py", "D4|d4-T1-t0000-p0") == SH.canon("k17_freeze.py", "d4-T1-t0000-p0")
    assert SH.canon("k17_varbench.py", "gsm8k-1004-x") == SH.canon("k17_public.py", "gsm8k-1004-x")
    assert SH.canon("steer_gen.py", 1004) == SH.canon("steer_gen_raven.py", "1004") == "1004"
    assert SH.space("k15_pairs.py") != SH.space("k15_full.py") == SH.space("steer_gen.py")


def test_small_sets_are_kept_and_shared_per_benchmark(tmp_path):
    """Default floor 40: a 3-item cell is done (small set, marked) and its rows reach results_all; two Thinking models
    of one benchmark get one shared set (ids made comparable)."""
    out = tmp_path / "out"
    pub = "k17_public.py"
    _fake_job(out, "B/k17_public/ouro_think_D1", "ouro_think", {"facts": 3}, [(pub, "edit_site_map", "facts", "O1")])
    rc, lines = _merge(out)
    assert rc == 0, lines
    c = _completeness(out)[("edit-site map", "ouro_1_4b_think", "D1")]
    assert c["state"] == "done" and "small set" in c["detail"]
    rr = _results_all(out)["B"]
    assert [r["n_items"] for r in rr if r["cell_dataset"] == "D1"] == ["3"]
    with open(os.path.join(str(out), "merged", "shared_sets.csv"), encoding="utf-8") as f:
        sh = list(csv.DictReader(f))
    assert [(r["card"], r["dataset"], r["state"]) for r in sh] == [("edit-site map", "D1", "shared = own")]


def test_merge_template_reruns_every_round():
    for lists in ("", "nolocal"):
        d = os.path.join(CL, "jobs", lists)
        t = open(os.path.join(d, "task_g9_merge.sbatch"), encoding="utf-8").read()
        assert 'echo "done already: $ID"; exit 0' not in t and "running it again" in t
        assert 'touch "$OUT_ROOT/_done/$ID"' in t
        for g in glob.glob(os.path.join(d, "task_g*.sbatch")):
            if not g.endswith("task_g9_merge.sbatch"):
                assert 'echo "done already: $ID"; exit 0' in open(g, encoding="utf-8").read(), g


GEN_LOCAL = {"A/steer_gen/a1_gsm8k": "ra20_g2", "A/steer_gen/a2_gsm8k": "ra20_g2"}


def _copy_gen(src, d):
    os.makedirs(d)
    for fn in ("results.csv", "items_done.json", "items.json", "setting.json", "analysis.json"):
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), d)
    for rd in glob.glob(os.path.join(src, "r[0-9][0-9]")):
        os.makedirs(os.path.join(d, os.path.basename(rd)))
        for f in glob.glob(os.path.join(rd, "cells_*.jsonl")):
            shutil.copy(f, os.path.join(d, os.path.basename(rd)))


@needs_artifacts
def test_generation_shared_rows(tmp_path):
    """GSM8K generation on Ouro-1.4B and Ouro-2.6B base (local cluster runs): the shared set is the problems both
    score; recomputing a cell's rows on its own full set reproduces its stored rows; the shared rows carry n."""
    from cluster.extract import shared as SH
    out = tmp_path / "out"
    for rel, run in GEN_LOCAL.items():
        _copy_gen(os.path.join(ART, "cluster_local", run, rel), str(out / rel))
    d1 = str(out / "A/steer_gen/a1_gsm8k")
    own = {r["source_key"]: r for r in common.read_results(os.path.join(d1, "results.csv"))}
    full = {int(it["idx"]) for it in json.load(open(os.path.join(d1, "items.json")))["scored"]}
    re_own = SH.rows("steer_gen.py", "", d1, "row", None, full)
    assert re_own and len(re_own) == len(own)
    for r in re_own:
        o = own[r["source_key"][len("shared:"):]]
        for k in ("estimate", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "n_items", "n_clusters"):
            assert r[k] == o[k], (r["source_key"], k)
    rc, lines = _merge(out)
    assert rc == 0, lines
    with open(os.path.join(str(out), "merged", "shared_sets.csv"), encoding="utf-8") as f:
        sh = {(r["card"], r["dataset"]): r for r in csv.DictReader(f)}
    g = sh[("generation placement", "GSM8K_gen")]
    sets = [{int(it["idx"]) for it in json.load(open(os.path.join(str(out), rel, "items.json")))["scored"]}
            for rel in GEN_LOCAL]
    n = len(sets[0] & sets[1])
    assert g["done_models"] == "ouro_1_4b_base,ouro_2_6b_base" and int(g["n_shared"]) == n and g["state"] == "shared"
    rr = _results_all(out)["A"]
    _no_shared_key({"A": rr})
    shr = [r for r in rr if r["item_set"] == "shared"]
    assert {r["cell_model"] for r in shr} == {"ouro_1_4b_base", "ouro_2_6b_base"}
    assert {r["n_items"] for r in shr if r["arm"].startswith("r00:")} == {str(n)}
    assert len(shr) == len([r for r in rr if r["item_set"] == "own"])


def test_k16_path_ouro_and_k17_public_rows_of_one_cell_keep_distinct_keys(tmp_path):
    """Item 1: k16_path_ouro writes edit_site_map rows for Ouro-1.4B-Thinking facts, which k17_public also writes for
    its edit-site map cell. Both reach results_all (the script is in the key); no shared key."""
    out = tmp_path / "out"
    _fake_job(out, "B/k17_public/ouro_think_D1", "ouro_think", {"facts": 3},
              [("k17_public.py", "edit_site_map", "facts", "O1")])
    _fake_job(out, "C/k16_path_ouro/ouro_think_D1", "ouro_think", {"facts": 3},
              [("k16_path_ouro.py", "edit_site_map", "facts", "O1"), ("k16_path_ouro.py", "source_restore", "facts", "Y3")])
    rc, lines = _merge(out, "--min_kept", "1")
    assert rc == 0, lines
    rr = [r for g in _results_all(out).values() for r in g]
    _no_shared_key({"all": rr})
    got = collections.Counter((r["script"], r["card"], r["cell_card"]) for r in rr if r["item_set"] == "own")
    assert got[("k17_public.py", "edit_site_map", "edit-site map")] == 1
    assert got[("k16_path_ouro.py", "edit_site_map", "pathway checks (restore, read window, attention block)")] == 1
    assert got[("k16_path_ouro.py", "source_restore", "pathway checks (restore, read window, attention block)")] == 1
