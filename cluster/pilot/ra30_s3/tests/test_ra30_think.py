"""RA-30 Thinking cells (plan 6e): gold-intermediate matching (GSM8K solutions, ASDiv equation trees), the chat
geometry, the screen, rows, gates, phases, shards, the pilot record (ceiling gate and pool sha256 pin) and the
summary merge, on the wide-vocabulary CPU fake of Ouro."""
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

import gold as GD  # noqa: E402
import ra30 as R0  # noqa: E402
import ra30_think as T  # noqa: E402
from test_ra30_driver import WideTok, Q, wide_ctx  # noqa: E402

SOL = "He puts 90/6 = <<90/6=15>>15 apples in each bag.\nSo 15+25 = <<15+25=40>>40.\n#### 40"
THINK = ("Okay, let's see. Tom has 90 apples in 6 bags, so each bag has 90 / 6 = 15 apples. "
         "Then 15 plus 25 is 40.\n#### 40")


def item(idx, name="Tom"):
    qa = Q.replace("Tom", name)
    qb = qa.replace("90 apples", "60 apples")
    return {"idx": idx, "value_A": 90, "value_B": 60, "gold_A": "40", "gold_B": "35", "var_name": "x",
            "question_A": qa, "question_B": qb, "char_span_A": [qa.index("90"), qa.index("90") + 2],
            "char_span_B": [qb.index("60"), qb.index("60") + 2], "pool_has_chat_geometry": False}


class ThinkTok(WideTok):
    """WideTok with an id for the closing think tag (prod's chat branch of build_prompts asks for it)."""

    def convert_tokens_to_ids(self, t):
        return 2 if t == "</think>" else super(ThinkTok, self).convert_tokens_to_ids(t)


def think_ctx():
    c = wide_ctx()
    c.ad.tok = ThinkTok()
    return T.Ctx(c.ad, c.steer, c.ad.tok, "gsm8k", "ouro_1_4b_think")


def clean_fn(ctx, prompt):
    return ctx.tok(THINK)["input_ids"] + [ctx.tok.eos_token_id]


class GoldTests(unittest.TestCase):
    def test_gsm8k_pairs(self):
        pairs, why = GD.gsm8k_pairs(SOL, 90, 60, "40", "35")
        self.assertIsNone(why)
        self.assertEqual(pairs, [(15.0, 10.0), (40.0, 35.0)])
        self.assertEqual(GD.gsm8k_pairs(SOL, 90, 60, "40", "36")[1], "gold_inconsistent")
        self.assertEqual(GD.gsm8k_pairs(SOL, 90, 60, "41", "35")[1], "gold_inconsistent")
        self.assertEqual(GD.gsm8k_pairs("", 90, 60, "40", "35")[1], "no_gold_solution")
        self.assertEqual(GD.gsm8k_pairs("<<90/6=16>>16\n#### 16", 90, 60, "16", "10")[1], "gold_parse")

    def test_asdiv_pairs(self):
        pairs, why = GD.asdiv_pairs("(7.0*(12.0-5.0))", "( 7.0 * ( 37.0 - 5.0 ) )", "49", "224")
        self.assertIsNone(why)
        self.assertEqual(pairs, [(7.0, 32.0), (49.0, 224.0)])
        self.assertEqual(GD.candidates(pairs, "paint 12 rooms, 7 hours each, 5 done", "49"), [])   # 7 is a prompt number
        p2, _ = GD.asdiv_pairs("(3.0*(12.0-5.0))", "( 3.0 * ( 37.0 - 5.0 ) )", "21", "96")
        self.assertEqual(GD.candidates(p2, "3 rooms, 12 and 5", "21"), [(7.0, 32.0)])
        self.assertEqual(GD.asdiv_pairs("21-17", "( 21.0 - 16.0 )", "4", "5")[0], [(4.0, 5.0)])  # one step: root only

    def test_match_rules(self):
        c = GD.candidates(GD.gsm8k_pairs(SOL, 90, 60, "40", "35")[0], Q, "40")
        self.assertEqual(c, [(15.0, 10.0)])
        m = GD.match(THINK, c)
        self.assertIsNone(m["drop"])
        self.assertEqual((m["derived_A"], m["derived_B"]), ("15", "10"))
        self.assertEqual(THINK[m["char_span"][0]:m["char_span"][1]], "15")
        self.assertEqual(GD.match("no numbers here", c)["drop"], "value_not_in_chain")
        self.assertEqual(GD.match("x", [])["drop"], "no_changed_intermediate")
        self.assertEqual(GD.match("is 15 now", [(15.0, 9.0)])["drop"], "token_count")
        self.assertEqual(GD.match("is 15 now", [(15.0, 10.5)])["drop"], "derived_B_format")
        mc = GD.match("so 12,000 dollars", [(12000.0, 15000.0)])
        self.assertEqual((mc["derived_A"], mc["derived_B"]), ("12,000", "15,000"))
        self.assertEqual(GD.match("so 12,0000 x", [(120000.0, 15.0)])["drop"], "value_format")
        self.assertEqual(GD.carry_prose("so 10 + 25 = 35", "15", "10"), "B")
        self.assertEqual(GD.carry_prose("so 15 + 25 = 40", "15", "10"), "A")
        self.assertEqual(GD.carry_prose("so 3 + 25", "15", "10"), "none")


class ThinkDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = (T.HORIZON, dict(T._GOLD), R0.HORIZON)
        T.HORIZON = 200
        T._GOLD["g"] = {"7": SOL, "8": SOL, "9": SOL, "10": SOL}

    @classmethod
    def tearDownClass(cls):
        T.HORIZON, R0.HORIZON = cls.saved[0], cls.saved[2]
        T._GOLD.clear()
        T._GOLD.update(cls.saved[1])

    def setUp(self):
        self.ctx = think_ctx()

    def test_geometry_screen_rows_gates(self):
        it = item(7)
        g = T.geometry(self.ctx.tok, "gsm8k", it)
        self.assertIsNotNone(g)
        self.assertEqual(self.ctx.tok.decode(g["prompt_A"][g["span"][0]:g["span"][1]]), "90")
        self.assertTrue(self.ctx.tok.decode(g["prompt_A"]).startswith("user: "))      # the chat template
        sc = T.screen_item(self.ctx, it, clean_fn)
        self.assertIsNone(sc["drop"], sc)
        self.assertEqual((sc["derived_A"], sc["derived_B"]), ("15", "10"))
        rows, meta = T.run_item(self.ctx, it, sc)
        self.assertEqual(sorted(r["arm"] for r in rows), sorted(T.ACTIVE_ARMS))
        rows2, _m = T.run_item(self.ctx, it, sc, max_batch=2)
        T.clear_item(self.ctx)
        self.assertEqual(len(rows2), len(rows))
        gt = T.gates_item(self.ctx, it, sc)
        self.assertTrue(gt["G1"] and gt["G5_patch_lands"] and gt["G4"]["P_ends_prefix"])
        self.assertEqual("".join(gt["G4"]["tokens_A"]), "15")
        bad = dict(it, pool_has_chat_geometry=True, span=[0, 1], plen=3)
        self.assertEqual(T.screen_item(self.ctx, bad, clean_fn)["drop"], "plen_mismatch")

    def _args(self, tmp, stage, phase="all", shard=0, shards=1, part=0, parts=1):
        return argparse.Namespace(model="ouro_1_4b_think", ds="gsm8k", stage=stage, phase=phase, shard=shard,
                                  rows_part=part, rows_parts=parts,
                                  shards=shards, horizon=200, pool_root=None, pool_local=False, think_dir=None,
                                  out=tmp, plan_sha256="", no_plan_check=True)

    def test_pilot_full_pin_gate_and_shards(self):
        tmp = tempfile.mkdtemp()
        items = [item(i, n) for i, n in zip((7, 8, 9, 10), ("Tom", "Ann", "Bob", "Eve"))]
        rec = {"path": "x", "sha256": "abc", "n": 4, "local": True}
        saved = (T.N_SCORED_PILOT, dict(T.N_GATE), T.MIN_ELIGIBLE_FULL)
        T.N_SCORED_PILOT, T.N_GATE["pilot"], T.N_GATE["full"], T.MIN_ELIGIBLE_FULL = {"gsm8k": 2}, 1, 1, 1
        try:
            self.assertEqual(T.run(self._args(tmp, "pilot", "screen"), ctx=self.ctx, clean_fn=clean_fn, items=items,
                                   pool_rec=rec), 0)
            pdir = os.path.join(tmp, "ouro_1_4b_think", "pilot", "gsm8k")
            self.assertFalse(os.path.exists(os.path.join(pdir, "rows.jsonl")))
            try:
                code = T.run(self._args(tmp, "pilot", "rows"), ctx=self.ctx, clean_fn=clean_fn, items=items,
                             pool_rec=rec)
            except SystemExit as e:
                code = e.code
            self.assertIn(code, (0, 3))                                    # the fake has no ceiling
            with open(os.path.join(pdir, "g6.json"), encoding="utf-8") as f:
                g6 = json.load(f)
            full = lambda sh, r: T.run(self._args(tmp, "full", shard=sh, shards=2), ctx=self.ctx, clean_fn=clean_fn,
                                       items=items, pool_rec=r)
            with self.assertRaises(SystemExit) as e:
                full(0, rec)
            self.assertEqual(e.exception.code, 2)                          # the pilot's plan (None) is not accepted
            R0.write_json(os.path.join(pdir, "hash_line.json"), {"plan_sha256": R0.ACCEPTED_PILOT_PLANS[0]})
            if not g6["pass"]:
                with self.assertRaises(SystemExit) as e:
                    full(0, rec)
                self.assertEqual(e.exception.code, 3)                      # pilot ceiling under 0.5
                R0.write_json(os.path.join(pdir, "g6.json"), dict(g6, **{"pass": True}))
            with self.assertRaises(SystemExit) as e:
                full(0, dict(rec, sha256="other"))
            self.assertEqual(e.exception.code, 2)                          # pool pin
            self.assertEqual(full(0, rec), 0)
            self.assertEqual(full(1, rec), 0)
            fdir = os.path.join(tmp, "ouro_1_4b_think", "full", "gsm8k")
            res = T.summarize(fdir)
            self.assertEqual(res["n_items"], 4)
            with open(os.path.join(fdir, "checks.json"), encoding="utf-8") as f:
                chk = json.load(f)
            self.assertEqual(chk["shards_merged"], ["shard_0_of_2", "shard_1_of_2"])
            self.assertEqual(chk["n_pool"], 4)
        finally:
            (T.N_SCORED_PILOT, ng, T.MIN_ELIGIBLE_FULL) = saved
            T.N_GATE.update(ng)
            shutil.rmtree(tmp)

    def test_pilot_rows_parts_and_merge(self):
        tmp = tempfile.mkdtemp()
        items = [item(i, n) for i, n in zip((7, 8, 9, 10), ("Tom", "Ann", "Bob", "Eve"))]
        rec = {"path": "x", "sha256": "abc", "n": 4, "local": True}
        saved = (T.N_SCORED_PILOT, dict(T.N_GATE))
        T.N_SCORED_PILOT, T.N_GATE["pilot"] = {"gsm8k": 3}, 2
        try:
            run = lambda **kw: T.run(self._args(tmp, "pilot", **kw), ctx=self.ctx, clean_fn=clean_fn, items=items,
                                     pool_rec=rec)
            self.assertEqual(run(phase="screen"), 0)
            pdir = os.path.join(tmp, "ouro_1_4b_think", "pilot", "gsm8k")
            with self.assertRaises(SystemExit) as e:
                run(phase="merge")                                         # no parts yet
            self.assertEqual(e.exception.code, 2)
            self.assertEqual(run(phase="rows", part=0, parts=2), 0)
            with self.assertRaises(SystemExit) as e:
                run(phase="merge")                                         # part 1 missing
            self.assertEqual(e.exception.code, 2)
            self.assertEqual(run(phase="rows", part=1, parts=2), 0)
            i0 = {r["idx"] for r in R0.read_jsonl(os.path.join(pdir, "rows_part_0_of_2", "rows.jsonl"))}
            i1 = {r["idx"] for r in R0.read_jsonl(os.path.join(pdir, "rows_part_1_of_2", "rows.jsonl"))}
            self.assertEqual((len(i0), len(i1), i0 & i1), (2, 1, set()))
            try:
                code = run(phase="merge")
            except SystemExit as e:
                code = e.code
            self.assertIn(code, (0, 3))
            self.assertTrue(os.path.exists(os.path.join(pdir, "g6.json")))
            with open(os.path.join(pdir, "checks.json"), encoding="utf-8") as f:
                chk = json.load(f)
            self.assertEqual(sorted(chk["thinking_focus"]), ["CA_cv1", "CA_kv1", "CA_span1", "CA_text"])
            self.assertIn("text", chk["rewrite_back"])
            with open(os.path.join(pdir, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertNotIn("Identity cvP = cv1", md)
            self.assertIn("cvP equals cv1 token for token on", md)
            with open(os.path.join(pdir, "gates.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["n"], 2)                     # one gate item per part
        finally:
            (T.N_SCORED_PILOT, ng) = saved
            T.N_GATE.update(ng)
            shutil.rmtree(tmp)

    def test_rewrite_back(self):
        self.assertTrue(T.rewrite_back(". Wait, 90/6 is 15, not 10. #### 40", "15"))
        self.assertFalse(T.rewrite_back(" apples. Then 10 + 25 = 35.\n#### 35", "15"))
        self.assertFalse(T.rewrite_back(" ok\n#### 15", "15"))              # after the answer marker: not counted

    def test_pool_missing_exit_2(self):
        with self.assertRaises(SystemExit) as e:
            T.load_pool("ouro_1_4b_think", "gsm8k", pool_root=tempfile.gettempdir())
        self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
