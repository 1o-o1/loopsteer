"""RA-30 driver tests on a wide-vocabulary CPU fake of Ouro (printable ASCII tokenizer, so a GSM8K-style chain
round-trips): screen, rows of every arm, per-item keys and clearing (review B2), the prompt-B length rule (B3), the
gates on the card path (G1, G4, G5 against a fresh donor, the B1 identities), the run loop (gate failure stops
before any row; a passing run writes rows and resumes without repeating them) and the summary files.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "lib"), os.path.join(ROOT, "tree"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PROD_DEV"] = "cpu"

import torch  # noqa: E402

import prod_fakes as PF  # noqa: E402
if PF.ROOT in sys.path and PF.ROOT not in (HERE, ROOT):
    sys.path.remove(PF.ROOT)
import arms as AR  # noqa: E402
import ra30 as R  # noqa: E402
import ra30_data as RD  # noqa: E402
from prod import steer_gen as S  # noqa: E402
from prod.models.ouro import patch_universal_cache  # noqa: E402

K = 4
CHARS = [chr(i) for i in range(32, 127)] + ["\n"]
Q = ("Tom has 90 apples. He puts them in 6 bags and then buys 25 more apples. How many apples does he have "
     "in one bag plus the new ones?")
CHAIN = "He puts 90/6 = <<90/6=15>>15 apples in each bag.\nSo he has 15+25 = <<15+25=40>>40 apples.\n#### 40"


class WideTok(PF.CharTok):
    vocab_size = 128

    def __init__(self):
        self.id_of = {c: i + 3 for i, c in enumerate(CHARS)}
        self.piece = {i: c for c, i in self.id_of.items()}
        self.piece.update({0: "", 1: "", 2: ""})
        self.special = {self.pad_token: 0, self.eos_token: 1, self.bos_token: 2}


def wide_ctx(dtype=torch.float32):
    cfg = PF.FakeOuroConfig(2, "sdpa", K)
    cfg.vocab_size = 128
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m = PF.OuroForCausalLM(cfg)
        PF._perturb_norms(m)
    m = m.eval().to(dtype)
    steer = S.SteerHooks(K)
    ad = S.SteeredOuroAdapter("ouro_1_4b_base", steer=steer, kv=True)
    ad.model, ad.tok = m, WideTok()
    patch_universal_cache(m)
    ad.set_depth(K)
    steer.install(m, kv=True)
    steer.record_fn = ad.record_donor
    hook = AR.NeuronHook(steer, layer=1, loop=1).install(m)
    return R.Ctx(ad, steer, hook, ad.tok, "gsm8k")


def short_prompt(tok, ds, q):
    return "Question: " + q + "\nAnswer:"


def item(idx, name="Tom"):
    tok = WideTok()
    qa = Q.replace("Tom", name)
    qb = qa.replace("90 apples", "60 apples")
    pa = short_prompt(tok, "gsm8k", qa)
    i = pa.index("90")
    return {"idx": idx, "value_A": 90, "value_B": 60, "gold_A": "40", "gold_B": "35", "span": [i, i + 2],
            "plen": len(pa), "var_name": "x", "question_A": qa, "question_B": qb}


def clean_fn(ctx, prompt):
    return ctx.tok(CHAIN)["input_ids"] + [ctx.tok.eos_token_id]


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = (RD.prompt_text, R.HORIZON)
        RD.prompt_text = short_prompt
        R.HORIZON = 48

    @classmethod
    def tearDownClass(cls):
        RD.prompt_text, R.HORIZON = cls.saved

    def setUp(self):
        self.ctx = wide_ctx()

    def test_screen_and_plen_rule(self):
        sc = R.screen_item(self.ctx, item(1), clean_fn)
        self.assertIsNone(sc["drop"], sc)
        self.assertEqual((sc["derived_A"], sc["derived_B"]), ("15", "10"))
        self.assertEqual(sc["P"][1], item(1)["plen"] + sc["prefix_len"])
        bad = dict(item(1), question_B=item(1)["question_B"] + " ")       # review B3: prompt B longer
        self.assertEqual(R.screen_item(self.ctx, bad, clean_fn)["drop"], "plen_mismatch")
        bad = dict(item(1), plen=item(1)["plen"] + 1)
        self.assertEqual(R.screen_item(self.ctx, bad, clean_fn)["drop"], "plen_mismatch")

    def test_run_item_rows_keys_and_clearing(self):
        it1, it2 = item(1), item(2, "Ann")
        s1 = R.screen_item(self.ctx, it1, clean_fn)
        s2 = R.screen_item(self.ctx, it2, clean_fn)
        g1 = R.geom_of(it1, s1, self.ctx.tok, "gsm8k")
        dsets = R.dataset_sets(self.ctx, [g1])
        rows, meta = R.run_item(self.ctx, it1, s1, dsets, False)
        self.assertEqual(sorted(r["arm"] for r in rows), sorted(R.ACTIVE_ARMS))
        self.assertEqual(len([a for a in R.ACTIVE_ARMS if AR.ARM[a]["batch"] == "prefix"]), 9)
        self.assertEqual(len([a for a in R.ACTIVE_ARMS if AR.ARM[a]["batch"] == "scratch"]), 3)
        self.assertTrue(all(k.startswith("gsm8k:1|") for k in self.ctx.steer.rows))
        d1 = self.ctx.steer.donors[AR.key_of("gsm8k", 1, "cv1")]
        self.assertIn((0, 0, "k"), d1["kv"])                               # the shared prefix donor holds K/V
        self.assertIs(d1, self.ctx.steer.donors[AR.key_of("gsm8k", 1, "kv1")])
        self.assertIn("rnd1", meta["norm_ratio_prerounding"])
        R.clear_item(self.ctx)
        for x in (self.ctx.steer.rows, self.ctx.steer.donors, self.ctx.steer.stats, self.ctx.hook.specs):
            self.assertEqual(len(x), 0)
        rows2, _m = R.run_item(self.ctx, it2, s2, dsets, True)              # split item: no ds5 arm
        self.assertNotIn("nc_ds5", [r["arm"] for r in rows2])
        self.assertTrue(all(k.startswith("gsm8k:2|") for k in self.ctx.steer.rows))
        fl = next(r for r in rows if r["arm"] == "floor")
        self.assertEqual(fl["chain_ids"][:fl["n_prefix"]], s1["prefix_A"])
        tx = next(r for r in rows if r["arm"] == "text")
        self.assertEqual(tx["chain_ids"][:tx["n_prefix"]], s1["prefix_B"])

    def test_host_tokens(self):
        tmp = tempfile.mkdtemp()
        saved = RD.load_pool
        pool = {"n": 2, "items": [item(1), dict(item(2, "Ann"), plen=5)]}
        RD.load_pool = lambda ds: pool
        try:
            chk = R.host_tokens("gsm8k", tmp, tok=WideTok())
            self.assertEqual((chk["n"], chk["n_mismatch"], chk["mismatch_idx"]), (2, 1, [2]))
            rows = R.read_jsonl(os.path.join(tmp, "tokens", "gsm8k.jsonl"))
            self.assertEqual(len(rows[0]["prompt_A"]), item(1)["plen"])
        except SystemExit as e:
            self.assertEqual(e.code, R.EXIT_PLAN)                            # 1 of 2 is over 1 percent
        finally:
            RD.load_pool = saved
            shutil.rmtree(tmp)

    def test_neuron_arms_on_still_build(self):
        it = item(1)
        sc = R.screen_item(self.ctx, it, clean_fn)
        saved = R.ACTIVE_ARMS
        R.ACTIVE_ARMS = list(AR.ARM_NAMES)
        try:
            g = R.geom_of(it, sc, self.ctx.tok, "gsm8k")
            rows, meta = R.run_item(self.ctx, it, sc, R.dataset_sets(self.ctx, [g]), False)
        finally:
            R.ACTIVE_ARMS = saved
        self.assertEqual(sorted(r["arm"] for r in rows), sorted(AR.ARM_NAMES))
        self.assertTrue(all(r["neuron_applied"] > 0 for r in rows if r["arm"] in ("nc_full", "ns_full")))

    def test_neuron_nonvacuity(self):
        it = item(1)
        sc = R.screen_item(self.ctx, it, clean_fn)
        nv = R.neuron_nonvacuity(self.ctx, it, sc)
        for a in ("nc_full", "ns_full"):
            v = nv[a]
            self.assertTrue(v["a_equals_donor"])
            self.assertGreater(v["delta_a_norm"], 0)
            self.assertIn(v["verdict"], ("inert hook", "no effect of the last-layer MLP part"))
            self.assertEqual(v["verdict"] == "inert hook", v["logits_equal"] or v["delta_a_norm"] == 0)
        self.assertEqual(len(self.ctx.steer.rows) + len(self.ctx.hook.specs), 0)

    def test_pilot_of_record(self):
        tmp = tempfile.mkdtemp()
        d = os.path.join(tmp, "pilot", "asdiv")
        os.makedirs(d)
        try:
            with self.assertRaises(SystemExit) as e:
                R.pilot_of_record(tmp, "asdiv", None)
            self.assertEqual(e.exception.code, R.EXIT_PLAN)                 # no pilot
            R.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": "other"})
            R.write_json(os.path.join(d, "checks.json"), {"projected_eligible": 120,
                                                          "horizon_cap_rule": {"use_cap_in_full": True}})
            with self.assertRaises(SystemExit) as e:
                R.pilot_of_record(tmp, "asdiv", "current")
            self.assertEqual(e.exception.code, R.EXIT_PLAN)                 # plan not accepted
            self.assertTrue(R.pilot_of_record(tmp, "asdiv", "other")["use_cap"])   # the current plan's own pilot
            R.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": R.ACCEPTED_PILOT_PLANS[0]})
            self.assertEqual(R.pilot_of_record(tmp, "asdiv", "current")["projected_eligible"], 120)
            R.write_json(os.path.join(d, "checks.json"), {"projected_eligible": 79})
            with self.assertRaises(SystemExit) as e:
                R.pilot_of_record(tmp, "asdiv", "current")
            self.assertEqual(e.exception.code, R.EXIT_GATE)                 # "ASDiv under 80: gate failure"
        finally:
            shutil.rmtree(tmp)

    def test_full_stage_runs_nonvacuity_and_keeps_idx11(self):
        tmp = tempfile.mkdtemp()
        d = os.path.join(tmp, "pilot", "gsm8k")
        os.makedirs(d)
        R.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": R.ACCEPTED_PILOT_PLANS[0]})
        R.write_json(os.path.join(d, "checks.json"), {"projected_eligible": 154,
                                                      "horizon_cap_rule": {"use_cap_in_full": True}})
        pool = {"n": 3, "items": [item(i, n) for i, n in zip((11, 2, 3), ("Tom", "Ann", "Bob"))]}
        saved = (RD.load_pool, RD.load_split, R.check_data, R.gates_item)
        RD.load_pool = lambda ds: pool
        RD.load_split = lambda ds: {"order": [3, 11, 2]}
        R.check_data = lambda: {"files": {}}
        R.gates_item = lambda ctx, it, sc: {"pass": True, "G2": {"identical": True}, "kv_equal_cvP_cv1": True,
                                            "kv_equal_kv4_cv3": True,
                                            "tok_identical_cvP_cv1": {"identical": True, "pass": True},
                                            "tok_identical_kv4_cv3": {"identical": True, "pass": True},
                                            "norm_ratio_applied_pass": True, "idx": it["idx"], "floor_alone": []}
        try:
            a = argparse.Namespace(ds="gsm8k", stage="full", out=tmp, plan_sha256="", no_plan_check=True)
            self.assertEqual(R.run(a, ctx=self.ctx, clean_fn=clean_fn), 0)
            out = os.path.join(tmp, "full", "gsm8k")
            with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertTrue(cfg["use_cap"])
            self.assertFalse(cfg["run_neuron_arms"])
            self.assertEqual(cfg["pilot_plan_sha256"], R.ACCEPTED_PILOT_PLANS[0])
            with open(os.path.join(out, "gates.json"), encoding="utf-8") as f:
                self.assertIn("nc_full", json.load(f)["neuron_nonvacuity"])
            with open(os.path.join(out, "summary.md"), encoding="utf-8") as f:
                self.assertIn("Neuron hook check", f.read())
            with open(os.path.join(out, "examples.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)[0]["idx"], 11)
            self.assertEqual(len({r["idx"] for r in R.read_jsonl(os.path.join(out, "rows.jsonl"))}), 3)
        finally:
            (RD.load_pool, RD.load_split, R.check_data, R.gates_item) = saved
            shutil.rmtree(tmp)

    def test_model_axis_paths_and_data(self):
        """Plan 6b: the 2.6B axis reads its own pool, order and sources file and writes under out/<model>/; the
        1.4B paths, files and pilot sizes are those of plan ed722548."""
        self.assertEqual(R.MODEL, "ouro_1_4b_base")
        self.assertEqual(R.model_root("OUT"), "OUT")
        self.assertEqual(R.n_scored_pilot("gsm8k"), 25)
        self.assertEqual(RD.sources_name(), "sources.json")
        self.assertEqual(RD.order_key("gsm8k"), "ra30:split:gsm8k")
        self.assertIn(R.ACCEPTED_PILOT_PLANS[0][:8], ("7f96eb63",))
        self.assertIn("ed722548db0644b604f194c889d0fe1b649a4e9e02cf27ed211ea667debfd10e", R.ACCEPTED_PILOT_PLANS)
        R.set_model("ouro_2_6b_base")
        try:
            self.assertEqual(R.model_root("OUT"), os.path.join("OUT", "ouro_2_6b_base"))
            self.assertEqual(R.n_scored_pilot("gsm8k"), 20)
            pool = R.pool_of("gsm8k")
            self.assertEqual((pool["n"], pool["model"]), (425, "ouro_2_6b_base"))
            order = R.order_of("gsm8k")["order"]
            self.assertEqual(order, RD.draw_order([it["idx"] for it in pool["items"]], "gsm8k", "ouro_2_6b_base"))
            self.assertEqual(sorted(order), sorted(it["idx"] for it in pool["items"]))
            src = R.check_data()                                            # the 2.6B sources file, hashes checked
            self.assertEqual(sorted(src["files"]), ["pool_asdiv_ouro_2_6b_base.json", "pool_gsm8k_ouro_2_6b_base.json",
                                                    "split_asdiv_ouro_2_6b_base.json",
                                                    "split_gsm8k_ouro_2_6b_base.json"])
            self.assertEqual(src["pool_n"], {"gsm8k": 425, "asdiv": 1059})
            self.assertEqual(R.n_scored_pilot("asdiv"), 20)
            pa = R.pool_of("asdiv")
            self.assertEqual(R.order_of("asdiv")["order"],
                             RD.draw_order([it["idx"] for it in pa["items"]], "asdiv", "ouro_2_6b_base"))
        finally:
            R.set_model("ouro_1_4b_base")
        src = R.check_data()
        self.assertEqual(sorted(src["files"]), ["pool_asdiv.json", "pool_gsm8k.json", "prescreen_asdiv.json",
                                                "prescreen_gsm8k.json", "split_asdiv.json", "split_gsm8k.json"])
        self.assertEqual(R.main(["--hash", "--model", "ouro_2_6b_base", "--ds", "asdiv"]), 0)
        R.set_model("ouro_1_4b_base")
        self.assertEqual(R.n_scored_pilot("asdiv"), 20)
        self.assertEqual(R.n_scored_pilot("gsm8k"), 25)

    def test_model_axis_full_run_paths(self):
        tmp = tempfile.mkdtemp()
        d = os.path.join(tmp, "ouro_2_6b_base", "pilot", "gsm8k")
        os.makedirs(d)
        R.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": R.ACCEPTED_PILOT_PLANS[1]})
        R.write_json(os.path.join(d, "checks.json"), {"projected_eligible": 150,
                                                      "horizon_cap_rule": {"use_cap_in_full": False}})
        pool = {"n": 2, "items": [item(i, n) for i, n in zip((5, 6), ("Tom", "Ann"))]}
        saved = (RD.load_pool, RD.load_split, R.check_data, R.gates_item)
        RD.load_pool = lambda ds, model=None: pool
        RD.load_split = lambda ds, model=None: {"order": [6, 5]}
        R.check_data = lambda: {"files": {}}
        R.gates_item = lambda ctx, it, sc: {"pass": True, "G2": {"identical": True}, "kv_equal_cvP_cv1": True,
                                            "kv_equal_kv4_cv3": True,
                                            "tok_identical_cvP_cv1": {"identical": True, "pass": True},
                                            "tok_identical_kv4_cv3": {"identical": True, "pass": True},
                                            "norm_ratio_applied_pass": True, "idx": it["idx"], "floor_alone": []}
        R.set_model("ouro_2_6b_base")
        try:
            a = argparse.Namespace(ds="gsm8k", stage="full", out=tmp, plan_sha256="", no_plan_check=True)
            self.assertEqual(R.run(a, ctx=self.ctx, clean_fn=clean_fn), 0)
            out = os.path.join(tmp, "ouro_2_6b_base", "full", "gsm8k")
            with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["model"], "ouro_2_6b_base")
            self.assertFalse(cfg["use_cap"])
            self.assertFalse(os.path.exists(os.path.join(tmp, "full")))     # nothing in the 1.4B paths
            with open(os.path.join(out, "checks.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["n_pool"], 2)
        finally:
            R.set_model("ouro_1_4b_base")
            (RD.load_pool, RD.load_split, R.check_data, R.gates_item) = saved
            shutil.rmtree(tmp)

    def test_cap_new(self):
        self.assertEqual(R.cap_new(20, 100, False), R.HORIZON - 20)
        self.assertEqual(R.cap_new(20, 30, True), min(R.HORIZON - 20, 2 * 10 + 64))
        self.assertEqual(R.cap_new(0, 5, True), min(R.HORIZON, 74))

    def test_gates_item(self):
        it = item(1)
        sc = R.screen_item(self.ctx, it, clean_fn)
        g = R.gates_item(self.ctx, it, sc)
        self.assertTrue(g["G1"])
        self.assertTrue(g["G4"]["equal_len"] and g["G4"]["P_ends_prefix"])
        self.assertEqual("".join(g["G4"]["tokens_A"]), "15")
        self.assertTrue(g["G5_patch_lands"])
        self.assertTrue(g["G5_neuron_full"])
        self.assertTrue(g["kv_equal_cvP_cv1"] and g["kv_equal_kv4_cv3"])
        self.assertTrue(g["tok_identical_cvP_cv1"]["identical"])
        self.assertIn("rnd1", g["norm_ratio_applied"])
        self.assertEqual(len(self.ctx.steer.rows), 0)
        self.assertIn("G2", g)                       # the injected chain is not the fake's own: G2 may fail here

    def _run(self, gates_pass):
        tmp = tempfile.mkdtemp()
        pool = {"n": 4, "items": [item(i, n) for i, n in zip((1, 2, 3, 4), ("Tom", "Ann", "Bob", "Eve"))]}
        saved = (RD.load_pool, RD.load_split, R.check_data, dict(RD.SPLIT_N), R.N_SCORED_PILOT, dict(R.N_GATE),
                 R.gates_item)
        RD.load_pool = lambda ds: pool
        RD.load_split = lambda ds: {"order": [3, 1, 4, 2]}
        R.check_data = lambda: {"files": {}}
        RD.SPLIT_N["pilot"], R.N_SCORED_PILOT, R.N_GATE["pilot"] = 1, {"gsm8k": 2}, 1
        real = R.gates_item

        def gi(ctx, it, sc):
            g = real(ctx, it, sc)
            g["pass"] = bool(gates_pass)
            return g
        R.gates_item = gi
        try:
            a = argparse.Namespace(ds="gsm8k", stage="pilot", out=tmp, plan_sha256="", no_plan_check=True)
            try:
                R.run(a, ctx=self.ctx, clean_fn=clean_fn)
                code = 0
            except SystemExit as e:
                code = e.code
            return tmp, code
        finally:
            (RD.load_pool, RD.load_split, R.check_data, sp, R.N_SCORED_PILOT, ng, R.gates_item) = saved
            RD.SPLIT_N.update(sp)
            R.N_GATE.update(ng)

    def test_run_gate_failure_writes_no_row(self):
        tmp, code = self._run(gates_pass=False)
        try:
            out = os.path.join(tmp, "pilot", "gsm8k")
            with open(os.path.join(out, "gates.json"), encoding="utf-8") as f:
                gp = json.load(f)
            self.assertFalse(gp["pass"])
            self.assertEqual(code, R.EXIT_GATE)
            self.assertFalse(os.path.exists(os.path.join(out, "rows.jsonl")))
            self.assertEqual(len(R.read_jsonl(os.path.join(out, "screen.jsonl"))), 2)
        finally:
            shutil.rmtree(tmp)

    def test_run_writes_rows_summary_and_resumes(self):
        tmp, code = self._run(gates_pass=True)
        try:
            out = os.path.join(tmp, "pilot", "gsm8k")
            rows = R.read_jsonl(os.path.join(out, "rows.jsonl"))
            self.assertEqual(len({r["idx"] for r in rows}), 2)              # neuron arms off: no split
            self.assertIn(code, (0, R.EXIT_GATE))                           # G6 may fail on the fake (no ceiling)
            for f in ("results.json", "results.csv", "checks.json", "summary.md", "examples.json", "sets.json",
                      "hash_line.json"):
                self.assertTrue(os.path.exists(os.path.join(out, f)), f)
            with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
                res = json.load(f)
            self.assertEqual(res["arms"]["base"]["n"], 2)
            self.assertNotIn("nc_full", res["arms"])
            self.assertIn("kv1", res["predictions"]["side_by_side_CA"])
            n0 = len(rows)
            tmp2 = tmp
            saved = R.run_item
            R.run_item = lambda *a, **k: (_ for _ in ()).throw(AssertionError("item repeated"))
            try:
                _t, code2 = self._run_in(tmp2)
            finally:
                R.run_item = saved
            self.assertEqual(len(R.read_jsonl(os.path.join(out, "rows.jsonl"))), n0)
        finally:
            shutil.rmtree(tmp)

    def _run_in(self, tmp):
        pool = {"n": 4, "items": [item(i, n) for i, n in zip((1, 2, 3, 4), ("Tom", "Ann", "Bob", "Eve"))]}
        saved = (RD.load_pool, RD.load_split, R.check_data, dict(RD.SPLIT_N), R.N_SCORED_PILOT, dict(R.N_GATE),
                 R.gates_item)
        RD.load_pool = lambda ds: pool
        RD.load_split = lambda ds: {"order": [3, 1, 4, 2]}
        R.check_data = lambda: {"files": {}}
        RD.SPLIT_N["pilot"], R.N_SCORED_PILOT, R.N_GATE["pilot"] = 1, {"gsm8k": 2}, 1
        R.gates_item = lambda ctx, it, sc: {"pass": True, "G2": {"identical": True}, "kv_equal_cvP_cv1": True,
                                            "kv_equal_kv4_cv3": True, "tok_identical_cvP_cv1": {"identical": True,
                                                                                                  "pass": True},
                                            "tok_identical_kv4_cv3": {"identical": True, "pass": True},
                                            "norm_ratio_applied_pass": True, "idx": it["idx"], "floor_alone": []}
        try:
            a = argparse.Namespace(ds="gsm8k", stage="pilot", out=tmp, plan_sha256="", no_plan_check=True)
            try:
                R.run(a, ctx=self.ctx, clean_fn=lambda c, p: (_ for _ in ()).throw(AssertionError("rescreen")))
                code = 0
            except SystemExit as e:
                code = e.code
            return tmp, code
        finally:
            (RD.load_pool, RD.load_split, R.check_data, sp, R.N_SCORED_PILOT, ng, R.gates_item) = saved
            RD.SPLIT_N.update(sp)
            R.N_GATE.update(ng)


if __name__ == "__main__":
    unittest.main()
