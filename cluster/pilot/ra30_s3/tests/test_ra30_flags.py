"""RA-30 summary-time operand flag (plan 6d): re-parse of the stored clean chains, the flag, CA with and without the
flagged items, and the stop on a re-parse that does not reproduce the stored derived value."""
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

import ra30_data as RD  # noqa: E402
import ra30_flags as F  # noqa: E402
from test_ra30_driver import WideTok, Q, CHAIN  # noqa: E402

Q2 = "Ann has 4 pens and keeps half, then buys 9 more pens. How many?"
CH2 = "She has 4 / 2 = <<4/2=2>>2 per bag and 2 + 9 = <<2+9=11>>11\n#### 11"


class FlagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tok = WideTok()
        self.pool = {"n": 2, "items": [
            {"idx": 1, "question_A": Q, "question_B": Q, "value_A": 90, "value_B": 60, "gold_A": "40",
             "gold_B": "35"},
            {"idx": 2, "question_A": Q2, "question_B": Q2, "value_A": 4, "value_B": 8, "gold_A": "11",
             "gold_B": "13"}]}
        self.saved = RD.load_pool
        RD.load_pool = lambda ds, model=None: self.pool
        with open(os.path.join(self.tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "ouro_1_4b_base", "ds": "gsm8k"}, f)
        scr = [{"idx": 1, "drop": None, "clean_ids": self.tok(CHAIN)["input_ids"], "derived_A": "15", "e1": "90/6"},
               {"idx": 2, "drop": None, "clean_ids": self.tok(CH2)["input_ids"], "derived_A": "2", "e1": "4/2"}]
        rows = [{"idx": i, "arm": a, "outcome": o} for i, a, o in
                ((1, "text", "B"), (1, "cv1", "B"), (2, "text", "B"), (2, "cv1", "A"))]
        for name, rr in (("screen.jsonl", scr), ("rows.jsonl", rows)):
            with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as f:
                for r in rr:
                    f.write(json.dumps(r) + "\n")

    def tearDown(self):
        RD.load_pool = self.saved
        shutil.rmtree(self.tmp)

    def test_flag_and_ca(self):
        res = F.flags(self.tmp, tok=self.tok)
        self.assertEqual((res["flagged"], res["reparse_mismatch"]), ([2], []))
        self.assertEqual(res["CA"]["cv1"]["all"][0], 0.5)
        self.assertEqual(res["CA"]["cv1"]["without_flagged"][0], 1.0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "summary_operand.md")))

    def test_notes_wording(self):
        rows = []
        for i, same, ca_p, ca_1 in ((1, True, "B", "B"), (2, False, "B", "A"), (3, True, "A", "A")):
            rows += [{"idx": i, "arm": "cv1", "outcome": ca_1, "chain_ids": [1, 2], "carry": "B"},
                     {"idx": i, "arm": "cvP", "outcome": ca_p, "chain_ids": [1, 2] if same else [1, 3], "carry": "B"},
                     {"idx": i, "arm": "floor", "outcome": "A", "chain_ids": [1], "carry": "B" if i == 3 else "A"}]
        scr = [{"idx": i, "derived_B_in_prompt": i == 3} for i in (1, 2, 3)]
        gates = {"items": [{"idx": 1, "tok_identical_kv4_cv3": {"identical": True}},
                           {"idx": 2, "tok_identical_kv4_cv3": {"identical": False}}]}
        for name, rr in (("rows.jsonl", rows), ("screen.jsonl", scr)):
            with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as f:
                for r in rr:
                    f.write(json.dumps(r) + "\n")
        with open(os.path.join(self.tmp, "gates.json"), "w", encoding="utf-8") as f:
            json.dump(gates, f)
        res = F.notes(self.tmp)
        self.assertEqual((res["cvP_cv1"]["token_identical"], res["cvP_cv1"]["n"]), (2, 3))
        self.assertEqual(res["kv4_cv3"], {"token_identical": 1, "gate_items": 2})
        self.assertEqual(res["carried_value_floor"]["items_derived_B_in_prompt_A"], 1)
        with open(os.path.join(self.tmp, "summary_notes.md"), encoding="utf-8") as f:
            txt = f.read()
        self.assertIn("cvP equals cv1 token for token on 2 of 3 chains", txt)
        self.assertIn("kv4 ran on the 2 gate items only", txt)
        self.assertNotIn("identity cvP", txt.lower().replace("not an identity", ""))

    def test_reparse_mismatch_stops(self):
        with open(os.path.join(self.tmp, "screen.jsonl"), encoding="utf-8") as f:
            scr = [json.loads(x) for x in f]
        scr[0]["e1"] = "91/6"
        with open(os.path.join(self.tmp, "screen.jsonl"), "w", encoding="utf-8") as f:
            for r in scr:
                f.write(json.dumps(r) + "\n")
        with self.assertRaises(SystemExit) as e:
            F.flags(self.tmp, tok=self.tok)
        self.assertEqual(e.exception.code, 4)


if __name__ == "__main__":
    unittest.main()
