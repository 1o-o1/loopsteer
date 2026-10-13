"""RA-30 McLeish r32 tests (plan 6c) on the prod CPU fake of McLeish (prelude 1, core 2, coda 1, K = 4 here):
T1 cache semantics of post@1, kv1 and the persistent arms in the raven static cache, x0 per problem, the screen,
the rows of every arm with shared donors, the gates, the run loop and the summary with the pre-registered
thresholds. The fake's K is 4; the driver's K is set to it for these tests and restored."""
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
import ra30_data as RD  # noqa: E402
import ra30_mcl as M  # noqa: E402
import steer_gen_raven as SGR  # noqa: E402
from prod.models.raven import install_attention_patch  # noqa: E402
from test_ra30_driver import WideTok, Q, CHAIN  # noqa: E402

KF = 4


def fake_ctx(vocab=128, seed=0):
    mod = PF._raven_module("mcleish")
    cfg = PF.RavenConfig("mcleish")
    cfg.vocab_size = cfg.padded_vocab_size = vocab
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = mod.RavenForCausalLM(cfg, "mcleish")
        PF._perturb_norms(m)
    m = m.eval()
    SGR.ACTIVE[0] = "mcl"
    steer = SGR.RavenSteerHooks(KF)
    ad = M.MclAdapter(steer=steer, kv=True, t_max=512)
    ad.model, ad.tok = m, WideTok()
    ad._pad_id, ad._stop_ids = 0, [1]
    install_attention_patch(m)
    ad.max_positions = None
    ad.dtype = torch.float32
    ad.set_depth(KF)
    ad.after_load()
    return M.Ctx(ad, steer, ad.tok)


def short_prompt(tok, ds, q):
    return "Question: " + q + "\nAnswer:"


def item(idx, name="Tom"):
    qa = Q.replace("Tom", name)
    qb = qa.replace("90 apples", "60 apples")
    pa = short_prompt(None, "gsm8k", qa)
    i = pa.index("90") + 1                                   # McLeish spans count the BOS
    return {"idx": idx, "value_A": 90, "value_B": 60, "gold_A": "40", "gold_B": "35", "span": [i, i + 2],
            "plen": len(pa) + 1, "var_name": "x", "question_A": qa, "question_B": qb}


def clean_fn(ctx, idx, prompt):
    return ctx.tok(CHAIN)["input_ids"] + [ctx.tok.eos_token_id]


class McLeishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = (M.K, M.HORIZON, RD.prompt_text)
        M.K, M.HORIZON, RD.prompt_text = KF, 40, short_prompt

    @classmethod
    def tearDownClass(cls):
        M.K, M.HORIZON, RD.prompt_text = cls.saved

    def setUp(self):
        self.ctx = fake_ctx()
        self.it = item(7)
        self.sc = M.screen_item(self.ctx, self.it, clean_fn)
        self.assertIsNone(self.sc["drop"], self.sc)
        self.g = M.geom_of(self.ctx, self.it, self.sc)

    def prefill_cache(self, arm):
        """One decode step of prod's raven decode (prefill + first token) with an external cache: the K/V entries
        of every block call, [steps, B, heads, T, hd]."""
        st, ad = self.ctx.steer, self.ctx.ad
        idx = self.it["idx"]
        M.clear_item(self.ctx)
        k = M.key_of(idx, arm)
        st.rows[str(k)] = M.row_info(arm, self.g, idx)
        if st.needs_donor(str(k)):
            dk = str(M.key_of(idx, "kvP"))
            st.rows[dk] = M.row_info("kvP", self.g, idx)
            st.donors[str(k)] = ad.record_donor(dk)
        seq = st.rows[str(k)]["a_ids"]
        T = len(seq)
        cache = ad.make_cache(KF, T + 3, 1)
        ad.decode([seq], 1, KF, [], row_ids=[k], cache=cache)
        return cache.key_cache[:, :, :, :T].clone(), cache.value_cache[:, :, :, :T].clone(), T

    def test_x0_is_per_problem(self):
        ad = self.ctx.ad
        t1, t2 = ad.table(M.key_of(1095, "cv1")), ad.table(M.key_of(1095, "screen"))
        self.assertTrue(torch.equal(t1, t2))
        self.assertTrue(torch.equal(t1, SGR.x0_table(ad.model, 1095, 512).float()))
        self.assertFalse(torch.equal(t1, ad.table(M.key_of(1096, "cv1"))))

    def test_T1_cache_semantics(self):
        """Raven static cache: one entry per block call (prelude 0..P-1, core P + C (j-1) + l, coda the last Cd)."""
        P_, C_, Cd = 1, 2, 1
        p0, p1 = self.g["P"]
        k0, v0, T = self.prefill_cache("floor")
        core = lambda j: [P_ + C_ * (j - 1) + l for l in range(C_)]
        coda = [k0.shape[0] - 1]
        # cv1: post@1 at P is loop 2's state input -> loops 2..K and the coda change at P, never prelude or loop 1
        k1, v1, _ = self.prefill_cache("cv1")
        for s in [0] + core(1):
            self.assertTrue(torch.equal(k0[s], k1[s]), s)
        for j in range(2, KF + 1):
            for s in core(j):
                self.assertFalse(torch.equal(k0[s, :, :, p0:p1], k1[s, :, :, p0:p1]), (j, s))
        self.assertFalse(torch.equal(k0[coda[0], :, :, p0:p1], k1[coda[0], :, :, p0:p1]))
        self.assertTrue(torch.equal(k0[:, :, :, :p0], k1[:, :, :, :p0]))
        # kv1: only the loop-1 entries at P, equal to the donor sequence's own loop-1 entries; earlier positions fixed
        kk, vk, _ = self.prefill_cache("kv1")
        kt, vt, _ = self.prefill_cache("text")
        for s in core(1):
            self.assertTrue(torch.allclose(kk[s, :, :, p0:p1], kt[s, :, :, p0:p1], atol=1e-5))
            self.assertTrue(torch.allclose(vk[s, :, :, p0:p1], vt[s, :, :, p0:p1], atol=1e-5))
        self.assertTrue(torch.equal(k0[0], kk[0]))
        self.assertTrue(torch.equal(k0[:, :, :, :p0], kk[:, :, :, :p0]))
        # kvP (loop 0 = the prelude, then every core loop): those entries at P equal the donor sequence's
        kp, vp, _ = self.prefill_cache("kvP")
        for s in [0] + [x for j in range(1, KF + 1) for x in core(j)]:
            self.assertTrue(torch.allclose(kp[s, :, :, p0:p1], kt[s, :, :, p0:p1], atol=1e-5), s)
        self.assertTrue(torch.equal(k0[:, :, :, :p0], kp[:, :, :, :p0]))
        # eP (review B2): e at P = the donor's; the core and coda entries at P equal the donor's, the prelude stays A's
        ke, ve, _ = self.prefill_cache("eP")
        for s in [x for j in range(1, KF + 1) for x in core(j)] + coda:
            self.assertTrue(torch.allclose(ke[s, :, :, p0:p1], kt[s, :, :, p0:p1], atol=1e-5), s)
            self.assertTrue(torch.allclose(ve[s, :, :, p0:p1], vt[s, :, :, p0:p1], atol=1e-5), s)
        self.assertTrue(torch.equal(ke[0], k0[0]))
        # cvP: e is re-injected at every loop from A's own tokens, so cvP is not the text edit at the cache level
        kc, _vc, _ = self.prefill_cache("cvP")
        self.assertFalse(all(torch.allclose(kc[s], kt[s], atol=1e-5) for j in range(2, KF + 1) for s in core(j)))

    def test_rows_gates_and_clearing(self):
        rows, meta = M.run_item(self.ctx, self.it, self.sc)
        self.assertEqual([r["arm"] for r in rows],
                         [a for a in M.ARM_NAMES if M.ARM[a]["batch"] == "prefix"]
                         + [a for a in M.ARM_NAMES if M.ARM[a]["batch"] == "scratch"])
        st = self.ctx.steer
        d = st.donors[str(M.key_of(7, "cv1"))]
        self.assertIs(d, st.donors[str(M.key_of(7, "kvP"))])
        self.assertIn((1, 0, "k"), d["kv"])
        self.assertIn("rnd1", meta["norm_ratio_prerounding"])
        fl = next(r for r in rows if r["arm"] == "floor")
        self.assertEqual(fl["chain_ids"][:fl["n_prefix"]], self.sc["prefix_A"])
        M.clear_item(self.ctx)
        self.assertEqual(len(st.rows) + len(st.donors) + len(st.stats), 0)
        g = M.gates_item(self.ctx, self.it, self.sc, all_loops=True)
        self.assertTrue(g["G1"])
        self.assertTrue(g["G5_all_loops"])
        self.assertTrue(g["G5_patch_lands"])
        self.assertTrue(g["G4"]["equal_len"] and g["G4"]["P_ends_prefix"])
        self.assertEqual("".join(g["G4"]["tokens_A"]), "15")
        self.assertIsNotNone(st.model)                                     # hooks reinstalled after G1

    def test_value_is_operand_drop(self):
        q = "Ann has 4 pens and keeps half, then buys 9 more pens. How many?"
        it = dict(item(8), question_A=q, question_B=q.replace("4 pens", "8 pens"), value_A=4, value_B=8,
                  gold_A="11", gold_B="13")
        pa = short_prompt(None, "gsm8k", q)
        i = pa.index("4 pens") + 1
        it.update(span=[i, i + 1], plen=len(pa) + 1)
        ch = "She has 4 / 2 = <<4/2=2>>2 per bag and 2 + 9 = <<2+9=11>>11\n#### 11"
        sc = M.screen_item(self.ctx, it, lambda c, i_, p: c.tok(ch)["input_ids"] + [1])
        self.assertEqual(sc["drop"], "value_is_operand")

    def test_thresholds(self):
        self.assertTrue(M.verdict_about0([0.03, 0.0, 0.08, 20]))
        self.assertFalse(M.verdict_about0([0.03, 0.0, 0.12, 20]))
        self.assertFalse(M.verdict_about0([0.06, 0.0, 0.09, 20]))
        self.assertEqual((M.ABOUT0, M.MOST), ({"point": 0.05, "hi90": 0.10}, {"ratio": 0.67, "lo90": 0.5}))

    def test_run_pilot_and_summary(self):
        tmp = tempfile.mkdtemp()
        pool = {"n": 3, "items": [item(i, n) for i, n in zip((5, 6, 9), ("Tom", "Ann", "Bob"))]}
        saved = (RD.load_pool, RD.load_split, M.check_data, M.N_SCORED_PILOT, dict(M.N_GATE))
        RD.load_pool = lambda ds, model=None: pool
        RD.load_split = lambda ds, model=None: {"order": [9, 5, 6]}
        M.check_data = lambda: {"files": {}}
        M.N_SCORED_PILOT, M.N_GATE["pilot"] = 2, 1
        try:
            a = argparse.Namespace(stage="pilot", out=tmp, plan_sha256="", no_plan_check=True)
            try:
                code = M.run(a, ctx=self.ctx, clean_fn=clean_fn)
            except SystemExit as e:
                code = e.code
            out = os.path.join(tmp, M.MODEL, "pilot", "gsm8k")
            with open(os.path.join(out, "gates.json"), encoding="utf-8") as f:
                gp = json.load(f)
            if gp["pass"]:
                self.assertIn(code, (0, 3))                                 # G6 can fail on the fake
                rows = M.R0.read_jsonl(os.path.join(out, "rows.jsonl"))
                self.assertEqual(len({r["idx"] for r in rows}), 2)
                for f in ("results.json", "results.csv", "checks.json", "summary.md", "examples.json"):
                    self.assertTrue(os.path.exists(os.path.join(out, f)), f)
                with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
                    res = json.load(f)
                self.assertIn("cvP_most", res["predictions"])
                self.assertIn("kv1_about0", res["predictions"])
            else:
                self.assertEqual(code, 3)
                self.assertFalse(os.path.exists(os.path.join(out, "rows.jsonl")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "pilot")))       # nothing in the Ouro paths
        finally:
            (RD.load_pool, RD.load_split, M.check_data, M.N_SCORED_PILOT, ng) = saved
            M.N_GATE.update(ng)
            shutil.rmtree(tmp)

    def test_pilot_of_record_waiver(self):
        tmp = tempfile.mkdtemp()
        d = os.path.join(tmp, "pilot", "gsm8k")
        os.makedirs(d)
        try:
            M.R0.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": "cur"})
            M.R0.write_json(os.path.join(d, "checks.json"), {"projected_eligible": 25,
                                                             "horizon_cap_rule": {"use_cap_in_full": True}})
            self.assertEqual(M.pilot_of_record(tmp, "cur")["projected_eligible"], 25)   # under 80: waived (D1)
            M.R0.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": M.MCL_ACCEPTED_PILOT_PLANS[-1]})
            self.assertEqual(M.pilot_of_record(tmp, "later")["projected_eligible"], 25)  # the push-2 pilot
            M.R0.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": "cur"})
            M.R0.write_json(os.path.join(d, "checks.json"), {"projected_eligible": 19})
            with self.assertRaises(SystemExit) as e:
                M.pilot_of_record(tmp, "cur")
            self.assertEqual(e.exception.code, M.R0.EXIT_GATE)
            with self.assertRaises(SystemExit) as e:
                M.pilot_of_record(tmp, "other")
            self.assertEqual(e.exception.code, M.R0.EXIT_PLAN)
        finally:
            shutil.rmtree(tmp)

    def test_full_shards_and_merge(self):
        tmp = tempfile.mkdtemp()
        root = os.path.join(tmp, M.MODEL)
        pd = os.path.join(root, "pilot", "gsm8k")
        os.makedirs(pd)
        M.R0.write_json(os.path.join(pd, "hash_line.json"), {"plan_sha256": M.R0.ACCEPTED_PILOT_PLANS[0]})
        M.R0.write_json(os.path.join(pd, "checks.json"), {"projected_eligible": 30,
                                                          "horizon_cap_rule": {"use_cap_in_full": False}})
        pool = {"n": 4, "items": [item(i, n) for i, n in zip((5, 6, 9, 11), ("Tom", "Ann", "Bob", "Eve"))]}
        saved = (RD.load_pool, RD.load_split, M.check_data, M.gates_item, dict(M.N_GATE))
        RD.load_pool = lambda ds, model=None: pool
        RD.load_split = lambda ds, model=None: {"order": [9, 5, 6, 11]}
        M.check_data = lambda: {"files": {}}
        M.gates_item = lambda ctx, it, sc, **kw: {"pass": True, "G2": {"identical": True}, "idx": it["idx"],
                                                  "floor_alone": []}
        try:
            for sh in (0, 1):
                a = argparse.Namespace(stage="full", out=tmp, plan_sha256="", no_plan_check=True, shard=sh, shards=2)
                self.assertEqual(M.run(a, ctx=self.ctx, clean_fn=clean_fn), 0)
            full = os.path.join(root, "full", "gsm8k")
            s0 = {r["idx"] for r in M.R0.read_jsonl(os.path.join(full, "shard_0_of_2", "screen.jsonl"))}
            s1 = {r["idx"] for r in M.R0.read_jsonl(os.path.join(full, "shard_1_of_2", "screen.jsonl"))}
            self.assertEqual((s0, s1), ({9, 6}, {5, 11}))
            res = M.summarize(full)
            self.assertEqual(res["n_items"], 4)
            self.assertEqual(res["checks"]["shards_merged"], ["shard_0_of_2", "shard_1_of_2"])
            self.assertTrue(res["checks"]["gate_80"].startswith("waived"))
            self.assertIn("eP_over_text", res["predictions"])
            self.assertIn("cvP_over_eP", res["predictions"])
            self.assertEqual((M.shard_count(10.0), M.shard_count(16.2), M.shard_count(20.0)), (3, 3, 4))
            with open(os.path.join(full, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertIn("Breadth cell", md)
            self.assertIn("cvP equals cv1 token for token on", md)
            self.assertIn("kv4 was not run in this cell.", md)
            with open(os.path.join(full, "shard_0_of_2", "config.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["gate_80"], "waived (RA D1, breadth cell)")
            M.R0.append_jsonl(os.path.join(full, "shard_1_of_2", "screen.jsonl"), {"idx": 9, "drop": "x"})
            with self.assertRaises(SystemExit):
                M.summarize(full)                                            # an item in two shards
            with self.assertRaises(SystemExit):
                M.run(argparse.Namespace(stage="pilot", out=tmp, plan_sha256="", no_plan_check=True, shard=0,
                                         shards=2), ctx=self.ctx, clean_fn=clean_fn)
        finally:
            (RD.load_pool, RD.load_split, M.check_data, M.gates_item, ng) = saved
            M.N_GATE.update(ng)
            shutil.rmtree(tmp)

    def test_asdiv_pool_at_run_time_and_pin(self):
        tmp = tempfile.mkdtemp()
        try:
            M.set_ds("asdiv")
            with self.assertRaises(SystemExit) as e:
                M.get_pool(tmp)
            self.assertEqual(e.exception.code, 2)                          # the EXT5 select output is missing
            prs = RD.read_jsonl(RD.pairs_path("asdiv"))[:3]
            p = os.path.join(tmp, M.POOL_RT["asdiv"])
            os.makedirs(os.path.dirname(p))
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"scored": [{"idx": r["idx"], "var_name": r["var_name"], "value_A": r["value_A"],
                                       "value_B": r["value_B"], "gold_A": r["gold_A"], "gold_B": r["gold_B"],
                                       "span": [1, 2], "plen": 9} for r in prs]}, f)
            pool, rec = M.get_pool(tmp)
            self.assertEqual((pool["n"], rec["n"], rec["path"]), (3, 3, M.POOL_RT["asdiv"]))
            self.assertEqual(pool["items"][0]["question_A"], prs[0]["question_A"])
            self.assertEqual(rec["sha256"], RD.sha256_file(p))
            d = os.path.join(tmp, "pilot", "asdiv")
            os.makedirs(d)
            M.R0.write_json(os.path.join(d, "hash_line.json"), {"plan_sha256": "cur"})
            M.R0.write_json(os.path.join(d, "checks.json"), {"projected_eligible": 30})
            M.R0.write_json(os.path.join(d, "pool_record.json"), dict(rec, sha256="old"))
            with self.assertRaises(SystemExit) as e:
                M.pilot_of_record(tmp, "cur", rec)
            self.assertEqual(e.exception.code, 2)                          # sha256 pin
            M.R0.write_json(os.path.join(d, "pool_record.json"), rec)
            self.assertEqual(M.pilot_of_record(tmp, "cur", rec)["projected_eligible"], 30)
            self.assertEqual(M.check_data()["files"], {"pairs_asdiv.jsonl": RD.sha256_file(RD.pairs_path("asdiv"))})
        finally:
            M.set_ds("gsm8k")
            shutil.rmtree(tmp)

    def test_data_and_pool(self):
        pool = RD.load_pool("gsm8k", M.MODEL)
        self.assertEqual(pool["n"], 237)
        self.assertEqual(RD.load_split("gsm8k", M.MODEL)["order"],
                         RD.draw_order([it["idx"] for it in pool["items"]], "gsm8k", M.MODEL))
        src = M.check_data()
        self.assertEqual(sorted(src["files"]), ["pool_gsm8k_mcleish_llama32_r32.json",
                                                "split_gsm8k_mcleish_llama32_r32.json"])


if __name__ == "__main__":
    unittest.main()
