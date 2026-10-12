"""CPU tests of the RA-31 Part B cells (EXT6c, plan section 15): the cell table, rows and card-hours (every shard under
9 h at 0.6 x); the cell data, pins and the 2.6B ASDiv cross-check with RA-30; the 2.6B sets (required, from the
freeze26 file only, a 1.4B file refused), freeze26 and the dep26 submit helper; the ASDiv and MAWPS tasks through the
bundle's steer_gen_math (in a subprocess, since it repoints steer_gen's globals); one cell end to end on the fake (four
shards, identity, summary), a runtime-pool cell, the exits; the GSM8K 1.4B Part B table unchanged.

    cd ra31 && python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (TESTS, os.path.join(HERE, "tree"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PROD_DEV"] = "cpu"

import torch  # noqa: E402

import ra31_gen as G  # noqa: E402
import test_partb as TP  # noqa: E402
import prod.steer_gen as SG  # noqa: E402


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue().strip()


class TestCellTable(unittest.TestCase):
    def test_arms_rows_hours(self):
        rows = {"svamp_1.4b": (244, 252), "asdiv_1.4b": (395, 417), "gsm8k_2.6b": (395, 417),
                "svamp_2.6b": (248, 252), "asdiv_2.6b": (395, 417)}
        for cell in G.CELLS:
            g = G.cell_groups(cell)
            self.assertEqual(sorted(g), ["all", "lc", "rand", "top"])
            names = [a["name"] for v in g.values() for a in v]
            self.assertEqual(len(names), 22)
            self.assertEqual(len(set(names)), 22)
            self.assertIn("r00_rerun", [a["name"] for a in g["lc"]])
            self.assertEqual(sorted(n for n in names if n.startswith("allpos")), ["allpos_attn_a0", "allpos_mlp_a0"])
            sh = G.cell_shards(cell)
            self.assertEqual(sorted(set(v["group"] for v in sh.values())), ["all", "lc", "rand", "top"])
            for nm, v in sh.items():
                self.assertLessEqual(G.cell_hours(cell, nm)[1], G.MAX_SHARD_PACKED_H, nm)
                self.assertEqual(len([x for x in sh.values() if x["group"] == v["group"]]), v["n_parts"])
            if cell in rows:
                sc, ids, plens, _ = G.cell_rows(cell)
                self.assertEqual((len(sc), len(ids)), rows[cell], cell)
                self.assertTrue(all(plens[i] for i in sc))
                G.check_data(G.cell_data(cell))
                self.assertEqual(G.check_pins(data_dir=G.cell_data(cell))[1], [])
                parts = {}
                for v in sh.values():                                          # the parts of a group cover
                    parts.setdefault(v["group"], []).extend(ids[v["part"]::v["n_parts"]])   # every row once
                for grp, xs in parts.items():
                    self.assertEqual(sorted(xs), sorted(ids), (cell, grp))
            else:
                with self.assertRaises(SystemExit) as cm:                      # MAWPS: the pool comes at run time
                    G.cell_rows(cell)
                self.assertEqual(cm.exception.code, 2)
        self.assertGreater(len(G.cell_shards("gsm8k_2.6b")), 4)                # 2.6B GSM8K is split by items
        a = sorted(set(e["alpha"] for x in G.cell_groups("svamp_1.4b")["rand"] for e in x["edits"] if x["k"] == 4))
        self.assertEqual(a, [0.5, 0.75, 1.25, 1.5])
        self.assertEqual([x["name"] for x in G.gate_arms()], ["gate_rerun", "gate_ident"])

    def test_asdiv_26b_pool_equals_ra30(self):
        src = json.load(open(os.path.join(G.cell_data("asdiv_2.6b"), "sources_partb.json")))
        self.assertIn("ra30_pool_sha256", src)

    def test_gsm8k_part_b_unchanged(self):
        S = G.shards()
        self.assertEqual(len([a for v in S.values() for a in v]), 54)
        self.assertEqual(round(sum(G.shard_hours(v) for v in S.values()), 2), 32.53)

    def test_26b_sets_required_and_file_rules(self):
        with self.assertRaises(SystemExit) as cm:
            G.cell_top_sets("gsm8k_2.6b")
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit) as cm:                              # a 1.4B set is refused
            G.cell_top_sets("gsm8k_2.6b", os.path.join(G.DATA_B, "partb_sets_1.4b.json"))
        self.assertEqual(cm.exception.code, 2)
        d = tempfile.mkdtemp(prefix="ra31_c_")
        self.addCleanup(shutil.rmtree, d, True)
        p = os.path.join(d, "s26.json")
        json.dump({"model": "ouro_2_6b_base", "mlp": {"top": list(range(G.N_TOP_26B)),
                                                      "rand": list(range(1, G.N_TOP_26B + 1))}}, open(p, "w"))
        s = G.cell_top_sets("gsm8k_2.6b", p)
        self.assertEqual((len(s["mlp"]["top"]), len(s["mlp"]["rand"])), (G.N_TOP_26B, G.N_TOP_26B))

    def test_freeze26_and_dep26(self):
        d = tempfile.mkdtemp(prefix="ra31_f_")
        self.addCleanup(shutil.rmtree, d, True)
        root = os.path.join(d, "root")
        pilot = os.path.join(root, G.PILOT_26B)
        os.makedirs(pilot)
        with self.assertRaises(SystemExit) as cm:                              # neither submitted nor done
            quiet(G.dep26, root)
        self.assertEqual(cm.exception.code, 2)
        os.makedirs(os.path.join(root, "_chain"))
        with open(os.path.join(root, "_chain", "submitted.tsv"), "w") as f:
            f.write("111 g20_e6ra29.s1_pilot.tsv 1\n4242 g21_e6ra31a.s1_pilot.tsv 1\n")
        self.assertEqual(quiet(G.dep26, root), "afterok:4242")
        out = os.path.join(root, G.SETS_26B)
        with self.assertRaises(SystemExit) as cm:                              # the freeze needs a passing gate
            quiet(G.freeze26, pilot, out)
        self.assertEqual(cm.exception.code, 2)
        g = torch.Generator().manual_seed(0)
        for ds in ("D1", "D2"):
            with open(os.path.join(pilot, "items_%s.jsonl" % ds), "w") as f:
                for i in range(3):
                    top = sorted(torch.randperm(48 * 5632, generator=g)[:G.N_TOP_26B].tolist())
                    f.write(json.dumps({"sets": {"donor": {"mlp_top": top, "attn_top": [1, 2]}}}) + "\n")
        json.dump({"pass": True}, open(os.path.join(pilot, "pilot_gate.json"), "w"))
        self.assertEqual(quiet(G.dep26, root), "done")
        quiet(G.freeze26, pilot, out)
        s = G.cell_top_sets("svamp_2.6b", out)
        self.assertEqual(len(s["mlp"]["top"]), G.N_TOP_26B)
        self.assertEqual(quiet(G.dep26, root), "done set")


class TestMathTasks(unittest.TestCase):
    def _sub(self, cell, task, need_ids):
        d = tempfile.mkdtemp(prefix="ra31_a_")
        self.addCleanup(shutil.rmtree, d, True)
        code = ("import sys, json; sys.path[:0] = [%r, %r, %r]; import os; os.environ['PROD_DEV'] = 'cpu'\n"
                "import ra31_gen as G\n"
                "t = G.setup_task(%r, %r)\n"
                "from prod.tasks import rows\n"
                "ids = set(int(r['idx']) for r in rows(%r))\n"
                "need = G.cell_rows(%r)[1] if %r else []\n"
                "print(json.dumps({'task': t, 'n_rows': len(ids), 'missing': [i for i in need if i not in ids]}))\n"
                % (HERE, os.path.join(HERE, "tree"), os.path.join(HERE, "lib"), cell, d, task, cell, need_ids))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_asdiv(self):
        for cell in ("asdiv_1.4b", "asdiv_2.6b"):
            out = self._sub(cell, "asdiv", True)
            self.assertEqual((out["task"], out["missing"]), ("asdiv", []))
            self.assertGreater(out["n_rows"], 1000)

    def test_mawps(self):
        out = self._sub("mawps_1.4b", "mawps", False)
        self.assertEqual(out["task"], "mawps")
        self.assertGreater(out["n_rows"], 1900)


class TestCellEndToEnd(unittest.TestCase):
    """run_cell on the fake: data root and decode replaced; everything else as on the cluster."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ra31_e_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rig = TP.Rig(self)
        self.rows = TP.prompts(6, seed=8)
        root = self.root = os.path.join(self.tmp, "data")
        d = os.path.join(root, "svamp_1.4b")
        os.makedirs(d)
        ids = sorted(int(k) for k in self.rows)
        json.dump({"scored": [{"idx": i, "plen": len(self.rows[str(i)])} for i in ids[:5]],
                   "sentinels": [{"idx": ids[5]}]}, open(os.path.join(d, "items.json"), "w"))
        base = self.rig.decode(SG.SteerHooks(4), self.rows)
        with open(os.path.join(d, "r00_chains.jsonl"), "w") as f:
            for k_, v in sorted(base.items()):
                f.write(json.dumps({"idx": int(k_), "chain_ids": v}) + "\n")
        with open(os.path.join(d, "r00_cells.jsonl"), "w") as f:
            for k_, v in sorted(base.items()):
                f.write(json.dumps({"row_idx": int(k_), "trace_answer": str(v[-1]), "trace_correct": v[-1] % 2 == 0,
                                    "n_trace": len(v)}) + "\n")
        json.dump({"model_meta": {"steer_gen": G.STEER_GEN_VERSION}}, open(os.path.join(d, "r00_meta.json"), "w"))
        json.dump({"sources": {}, "outputs": {f: G.sha256_file(os.path.join(d, f)) for f in os.listdir(d)}},
                  open(os.path.join(d, "sources_partb.json"), "w"))
        sets = TP.fake_sets()
        json.dump({"mlp": dict(sets["mlp"], info={"n_tied_at_cut": 3}), "attn": sets["attn"], "rule": "r"},
                  open(os.path.join(root, "partb_sets_1.4b.json"), "w"))

    def infos(self, tok, ids, task, bad=False):
        return {str(i): {"plen": len(self.rows[str(i)]) + (1 if bad else 0), "a_ids": self.rows[str(i)], "edits": [],
                         "lcdl": None} for i in ids}

    def gen(self, a, hooks, ids, out_dir, mem):
        os.makedirs(out_dir, exist_ok=True)
        rows = {str(i): self.rows[str(i)] for i in ids}
        keep = dict(hooks.rows)
        got = self.rig.decode(hooks, rows, k=a["k"])
        hooks.rows = keep
        with open(os.path.join(out_dir, "chains_fake.jsonl"), "w") as f:
            for k_, v in sorted(got.items()):
                f.write(json.dumps({"idx": int(k_), "chain_ids": v}) + "\n")
        with open(os.path.join(out_dir, "cells_fake.jsonl"), "w") as f:
            for k_, v in sorted(got.items()):
                f.write(json.dumps({"row_idx": int(k_), "trace_answer": str(v[-1]), "trace_correct": v[-1] % 2 == 0,
                                    "n_trace": len(v)}) + "\n")
        return {"exit": 0}

    def run_cell(self, shard, out, **kw):
        return G.run_cell("svamp_1.4b", shard, out, gen=kw.pop("gen", self.gen), tok_fn=lambda: None,
                          data_root=self.root, infos_fn=kw.pop("infos_fn", self.infos), **kw)

    def test_every_shard_and_summary(self):
        out = os.path.join(self.tmp, "out")
        for nm in G.cell_shards("svamp_1.4b", self.root):                    # independent: any order
            self.run_cell(nm, out)
        for nm in ("svamp_1.4b_all", "svamp_1.4b_lc"):
            idn = json.load(open(os.path.join(out, nm, "gate_ident", "DONE.json")))
            self.assertTrue(idn["identity"]["pass"])
        self.assertEqual(json.load(open(os.path.join(out, "svamp_1.4b_lc", "gate_ident", "DONE.json")))
                         ["identity"]["reference"], "r00_rerun")              # lc reuses its full rerun
        ck = G.summarize_cell("svamp_1.4b", out, data_root=self.root)
        self.assertEqual((ck["n_arms"], ck["n_done"], ck["csv_rows"]), (22, 22, 22))
        self.assertTrue(ck["identity_pass"])
        self.assertEqual(ck["rerun_vs_stored_r00"]["n_identical"], 6)
        self.assertEqual(ck["set_ties_at_cut"]["n_tied_at_cut"], 3)
        self.assertTrue(os.path.exists(os.path.join(out, "summary_partb_svamp_1.4b.md")))

    def test_item_parts_merge_and_duplicates_stop(self):
        saved = (G.MAX_SHARD_PACKED_H, G.MAX_PARTS)
        G.MAX_SHARD_PACKED_H, G.MAX_PARTS = 0.0, 2
        self.addCleanup(lambda: (setattr(G, "MAX_SHARD_PACKED_H", saved[0]), setattr(G, "MAX_PARTS", saved[1])))
        sh = G.cell_shards("svamp_1.4b", self.root)
        self.assertEqual(len(sh), 8)
        out = os.path.join(self.tmp, "parts")
        for nm in sh:
            self.run_cell(nm, out)
        ck = G.summarize_cell("svamp_1.4b", out, data_root=self.root)
        self.assertEqual((ck["n_done"], ck["csv_rows"]), (22, 22))
        a = os.path.join(out, "svamp_1.4b_top_p1of2", "k4_mlp_top_a0.5")
        b = os.path.join(out, "svamp_1.4b_top_p2of2", "k4_mlp_top_a0.5")
        shutil.copyfile(os.path.join(a, "cells_fake.jsonl"), os.path.join(b, "cells_fake.jsonl"))
        with self.assertRaises(SystemExit) as cm:
            G.summarize_cell("svamp_1.4b", out, data_root=self.root)
        self.assertEqual(cm.exception.code, 2)

    def test_prompt_length_mismatch_exit_2(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_cell("svamp_1.4b_all", os.path.join(self.tmp, "o2"),
                          infos_fn=lambda tok, ids, task: self.infos(tok, ids, task, bad=True))
        self.assertEqual(cm.exception.code, 2)

    def test_wrong_shard_for_cell_exit_2(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_cell("gsm8k_2.6b_top", os.path.join(self.tmp, "o3"))
        self.assertEqual(cm.exception.code, 2)

    def test_runtime_pool_cell(self):
        """mawps_1.4b: items.json (and r00 chains when present) read from <pool_root>/<pool_rel>, hashed; -> 2 without."""
        pr = os.path.join(self.tmp, "OUT_ROOT")
        pd = os.path.join(pr, G.CELLS["mawps_1.4b"]["pool_rel"])
        os.makedirs(os.path.join(pd, "r00"))
        shutil.copyfile(os.path.join(self.root, "svamp_1.4b", "items.json"), os.path.join(pd, "items.json"))
        shutil.copyfile(os.path.join(self.root, "svamp_1.4b", "r00_chains.jsonl"),
                        os.path.join(pd, "r00", "chains_x.jsonl"))
        saved = G.setup_task
        G.setup_task = lambda cell, rows_dir, here=None: "mawps"
        self.addCleanup(setattr, G, "setup_task", saved)
        out = os.path.join(self.tmp, "om")
        with self.assertRaises(SystemExit) as cm:
            G.run_cell("mawps_1.4b", "mawps_1.4b_all", out, gen=self.gen, tok_fn=lambda: None, data_root=self.root,
                       infos_fn=self.infos)
        self.assertEqual(cm.exception.code, 2)
        G.run_cell("mawps_1.4b", "mawps_1.4b_all", out, gen=self.gen, tok_fn=lambda: None, data_root=self.root,
                   infos_fn=self.infos, pool_root=pr)
        head = json.load(open(os.path.join(out, "mawps_1.4b_all", "shard.json")))
        self.assertEqual(head["pool_items_sha256"], G.sha256_file(os.path.join(pd, "items.json")))
        self.assertTrue(json.load(open(os.path.join(out, "mawps_1.4b_all", "gate_ident", "DONE.json")))
                        ["identity"]["pass"])
        G.run_cell("mawps_1.4b", "mawps_1.4b_lc", out, gen=self.gen, tok_fn=lambda: None, data_root=self.root,
                   infos_fn=self.infos, pool_root=pr)
        ck = G.summarize_cell("mawps_1.4b", out, data_root=self.root, pool_root=pr)
        self.assertIn("deviation_pool", ck)
        self.assertEqual(ck["rerun_vs_stored_r00"]["n_identical"], 6)


class TestHygiene(unittest.TestCase):
    def test_shell_files_have_lf_only(self):
        for f in sorted(os.listdir(HERE)):
            if f.endswith(".sh"):
                with open(os.path.join(HERE, f), "rb") as fh:
                    self.assertEqual(fh.read().count(bytes([13])), 0, f)

    def test_no_host_user_home_strings(self):
        import re
        words = ["sas" + "11", "sab" + "bir", "/ho" + "me/", "/us" + "ers/", "dg" + "x1", "rtx" + "6000",
                 "/var/" + "tmp/ra20"]
        pat = re.compile("(?i)(" + "|".join(re.escape(w) for w in words) + ")")
        fs = ["ra31_gen.py", "make_partb_cells.py", "steer_gen_math.py", "tests/test_partb_cells.py"]
        for cell in G.CELLS:
            if G.CELLS[cell]["pool"] == "data":
                fs += ["data_partb/%s/%s" % (cell, f) for f in ("r00_meta.json", "sources_partb.json")]
        for f in fs:
            with open(os.path.join(HERE, f), encoding="utf-8") as fh:
                self.assertIsNone(pat.search(fh.read()), f)


if __name__ == "__main__":
    unittest.main()
