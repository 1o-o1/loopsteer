"""RA-30 CPU tests (plan_ra30.md section 10): the chain logic and every drop rule, the token geometry (CharTok and,
when cached, the Ouro tokenizer), the data files, and on the prod CPU fake of Ouro (float32, 2 layers, k = 4) the
technical proofs T1 (prefix pass + decode reproduces the clean chain; cache semantics of a loop-j patch), T4
(donor from prompt A + prefix_B), T5 (random same-norm), the neuron hook identities, the neuron sets, batching of
arms, the arm table, and no host, user or home string in the folder.
"""
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "lib"), os.path.join(ROOT, "tree")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PROD_DEV"] = "cpu"
os.environ.setdefault("K17_DATASET", "gsm8k")

import torch  # noqa: E402

import prod_fakes as PF  # noqa: E402
if PF.ROOT in sys.path and PF.ROOT not in (HERE, ROOT):
    sys.path.remove(PF.ROOT)
import chain as CH  # noqa: E402
import arms as AR  # noqa: E402
import ra30_data as RD  # noqa: E402
from prod import steer_gen as S  # noqa: E402
from prod.models.ouro import patch_universal_cache  # noqa: E402

K = 4
Q = ("Tom has 90 apples. He puts them in 6 bags and then buys 25 more apples. How many apples does he have "
     "in one bag plus the new ones?")
CHAIN = ("He puts 90/6 = <<90/6=15>>15 apples in each bag.\nSo he has 15+25 = <<15+25=40>>40 apples.\n#### 40")


def fake_adapter(L=2, seed=0, dtype=torch.float32):
    fake = PF.make_fake_ouro("sdpa", L=L, total_ut_steps=K, seed=seed).to(dtype)
    steer = S.SteerHooks(K)
    ad = S.SteeredOuroAdapter("ouro_1_4b_base", steer=steer, kv=True)
    ad.model, ad.tok = fake, PF.CharTok()
    patch_universal_cache(fake)
    ad.set_depth(K)
    steer.install(fake, kv=True)
    steer.record_fn = ad.record_donor
    return ad, steer


def rand_ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [int(x) for x in torch.randint(3, PF.VOCAB, (n,), generator=g)]


def geom(prompt_A, prompt_B, prefix_A, prefix_B, P, span):
    return {"prompt_A": prompt_A, "prompt_B": prompt_B, "prefix_A": prefix_A, "prefix_B": prefix_B, "P": P,
            "span": span}


def prefill(ad, steer, seq, keys):
    """The prefill of prod's OuroAdapter.decode for one row, keeping the StaticUTCache."""
    T = len(seq)
    steer.ensure_donors(keys)
    cache = ad._cache_factory(1, T + 1)
    steer.batch_keys = keys
    try:
        out = ad.model(input_ids=torch.tensor([seq]), attention_mask=torch.ones(1, T, dtype=torch.long),
                       position_ids=torch.arange(T)[None], past_key_values=cache, use_cache=True,
                       cache_position=torch.arange(T), logits_to_keep=1)
    finally:
        steer.batch_keys = None
    return out.logits[:, -1].clone(), [c[:, :, :T].clone() for c in cache.key_cache], cache


class ChainTests(unittest.TestCase):
    def test_safe_eval(self):
        self.assertEqual(CH.safe_eval("90/6"), 15.0)
        self.assertEqual(CH.safe_eval("2 x 3 + 1"), 7.0)
        self.assertEqual(CH.safe_eval("$1,200-200"), 1000.0)
        self.assertEqual(CH.safe_eval("-(4-6)*2"), 4.0)
        self.assertIsNone(CH.safe_eval("__import__('os')"))
        self.assertIsNone(CH.safe_eval("5/0"))
        self.assertEqual(CH.fmt_num(37.5), "37.5")
        self.assertEqual(CH.fmt_num(40.0), "40")

    def test_analyze_kept(self):
        r = CH.analyze(CHAIN, Q, 90, 60, "40", "35")
        self.assertIsNone(r["drop"], r)
        self.assertEqual((r["derived_A"], r["derived_B"], r["answer_edit"]), ("15", "10", "35"))
        self.assertEqual(CHAIN[r["char_span"][0]:r["char_span"][1]], "15")
        self.assertTrue(CHAIN.index("=15") + 1 == r["char_span"][0])       # first occurrence: the result slot

    def test_drop_reasons(self):
        cases = [
            ("no equation here\n#### 40", Q, 90, 60, "40", "35", "no_equation"),
            ("90/6 = <<90/6=15>>15\n#### 15", Q, 90, 60, "15", "10", "one_step"),
            ("<<90/6=16>>16 <<16+25=41>>41\n#### 41", Q, 90, 60, "41", "35", "chain_arithmetic_error"),
            (CHAIN, Q, 90, 60, "41", "35", "last_result_not_answer"),
            ("<<90/15=6>>6 <<6+25=31>>31\n#### 31", Q, 90, 60, "31", "29", "no_intermediate"),
            (CHAIN, Q.replace("25 more", "90 more"), 90, 60, "40", "35", "ambiguous_value"),
            ("<<25*2=50>>50 <<50+90=140>>140\n#### 140", Q, 90, 60, "140", "110", "no_changed_step"),
            ("<<90/6=15>>15 <<15+25=40>>40\n#### 40", Q, 90, 61, "40", "35", "derived_B_format"),
            ("<<90/6=15>>15 <<15+25=40>>40\n#### 40", Q, 90, 12, "40", "27", "token_count"),
            (CHAIN, Q, 90, 60, "40", "36", "inconsistent"),
            ("<<90/6=15>>fifteen <<15+25=40>>40\n#### 40", Q, 90, 60, "40", "35", None),
            ("<<90/0.06=1,500>>1,500 <<1500+25=1525>>1525\n#### 1525", Q.replace(" 6 ", " 0.06 "), 90, 60,
             "1525", "1025", "value_format"),
            ("<<90/6=15>>15 <<15+25=40>>40 <<40*2=80>>80\n#### 80", Q.replace("25 more", "25 more and 40 cents"),
             90, 60, "80", "70", "ambiguous_operand"),
        ]
        for ch, q, va, vb, ga, gb, want in cases:
            r = CH.analyze(ch, q, va, vb, ga, gb)
            self.assertEqual(r["drop"], want, (ch, r))
        self.assertEqual(set(CH.DROP_REASONS) >= {c[-1] for c in cases if c[-1]}, True)

    def test_value_earlier(self):
        """Review B4: an earlier written copy of derived_A, or derived_A equal to the previous result."""
        ch = "About 15 apples. <<25*2=50>>50 then 90/6 = <<90/6=15>>15 and <<15+50=65>>65\n#### 65"
        self.assertEqual(CH.analyze(ch, Q, 90, 60, "65", "60")["drop"], "value_earlier")
        self.assertIsNone(CH.analyze(ch.replace("About 15", "About 17"), Q, 90, 60, "65", "60")["drop"])
        q = "Tom has 90 apples and 3 boxes. He puts them in 6 bags and buys 25 more. How many?"
        ch2 = "<<3*5=15>>15 then <<90/6=15>>15 and <<15+25=40>>40\n#### 40"
        self.assertEqual(CH.analyze(ch2, q, 90, 60, "40", "35")["drop"], "value_earlier")
        self.assertIsNone(CH.analyze(CHAIN, Q, 90, 60, "40", "35")["drop"])
        self.assertIn("value_earlier", CH.DROP_REASONS)
        self.assertIn("plen_mismatch", CH.DROP_REASONS)

    def test_n8_n9_fields(self):
        r = CH.analyze(CHAIN, Q, 90, 60, "40", "35")
        self.assertEqual((r["n_lead_same"], r["gold_B_in_prompt_or_prefix"], r["derived_B_in_prompt"]),
                         (1, False, False))
        r = CH.analyze(CHAIN, Q.replace("25 more", "25 more, 35 pears"), 90, 60, "40", "35")
        self.assertTrue(r["gold_B_in_prompt_or_prefix"])

    def test_rule_of_record_and_r1_information_mode(self):
        ch = "<<25*2=50>>50 <<90/6=15>>15 <<15+50=65>>65\n#### 65"
        self.assertEqual(CH.analyze(ch, Q, 90, 60, "65", "60", mode="r1")["drop"], "e1_independent")
        r = CH.analyze(ch, Q, 90, 60, "65", "60")
        self.assertIsNone(r["drop"], r)
        self.assertEqual((r["i0"], r["derived_A"], r["derived_B"], r["answer_edit"]), (1, "15", "10", "60"))
        # the final equation is never the derived value; an unchanged step is skipped
        self.assertEqual(CH.analyze("<<25*2=50>>50 <<50+90=140>>140\n#### 140", Q, 90, 60, "140", "110")["drop"],
                         "no_changed_step")
        r = CH.analyze("<<90*0=0>>0 <<90/6=15>>15 <<15+25=40>>40\n#### 40", Q, 90, 60, "40", "35")
        self.assertEqual((r["drop"], r["i0"]), (None, 1))

    def test_readouts(self):
        self.assertEqual(CH.outcome("35", "40", "35"), "B")
        self.assertEqual(CH.outcome("40.0", "40", "35"), "A")
        self.assertEqual(CH.outcome(None, "40", "35"), "neither")
        self.assertEqual(CH.outcome("7", "40", "35"), "neither")
        self.assertEqual(CH.carry(">>10 apples.\nSo 10+25 = <<10+25=35>>35\n#### 35", "15", "10"), "B")
        self.assertEqual(CH.carry(">>10 apples.\nSo 15+25 = <<15+25=40>>40\n#### 40", "15", "10"), "A")
        self.assertEqual(CH.carry(">>10 apples.\nSo 3+25 = <<3+25=28>>28\n#### 28", "15", "10"), "none")
        self.assertEqual(CH.carry("#### 10 <<10+1=11>>", "15", "10"), "none")     # after the marker: ignored


class GeometryTests(unittest.TestCase):
    def test_chartok_geometry(self):
        tok = PF.CharTok()
        r = CH.analyze(CHAIN, Q, 90, 60, "40", "35")
        prompt = tok("Question: x\nAnswer:")["input_ids"]
        ids = tok(CHAIN)["input_ids"]
        g = CH.token_geometry(tok, prompt, ids, r)
        self.assertIsNone(g["drop"])
        p0, p1 = g["P"]
        seqA, seqB = prompt + g["prefix_A"], prompt + g["prefix_B"]
        self.assertEqual(len(seqA), len(seqB))
        self.assertEqual(p1, len(seqA))                                    # the prefix ends on the value
        self.assertEqual(tok.decode(seqA[p0:p1]), "15")
        self.assertEqual(tok.decode(seqB[p0:p1]), "10")
        self.assertEqual(seqA[:p0], seqB[:p0])

    def test_token_boundary_and_count(self):
        class Merge(PF.CharTok):
            def __call__(self, text, add_special_tokens=False, **kw):
                return {"input_ids": [self.id_of.get(c, 3) for c in text.replace("10", "1")]}
        tok = PF.CharTok()
        r = CH.analyze(CHAIN, Q, 90, 60, "40", "35")
        g = CH.token_geometry(Merge(), [5, 6], tok(CHAIN)["input_ids"], r)
        self.assertEqual(g["drop"], "token_count_ids")

    def test_ouro_tokenizer_real_chains(self):
        """T2 on the laptop: the derived-value tokens of the prescreen's first items, recomputed from the Table 2
        single-card chains with the cached Ouro tokenizer (skipped when the tokenizer or the chains are absent)."""
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import make_data as MD
            tok = MD.load_tok()
        except Exception as e:                                             # noqa: BLE001
            self.skipTest("Ouro tokenizer not cached: %s" % type(e).__name__)
        ids = tok("<<90/6=15>>15", add_special_tokens=False)["input_ids"]
        self.assertEqual([tok.decode([i]) for i in ids][-2:], ["1", "5"])   # one token per digit
        n = 0
        for ds in RD.DATASETS:
            pre = os.path.join(RD.DATA, "prescreen_%s.json" % ds)
            src = os.path.join(os.path.dirname(os.path.dirname(ROOT)), MD.SOURCES[ds]["chains"])
            if not (os.path.exists(pre) and os.path.exists(src)):
                continue
            with open(pre, encoding="utf-8") as f:
                shown = json.load(f)["shown"]
            chains = {int(r["idx"]): r for r in RD.read_jsonl(src)}
            pool = {int(it["idx"]): it for it in RD.load_pool(ds)["items"]}
            for s in shown:
                it, cr = pool[s["idx"]], chains[s["idx"]]
                _c, txt, own = RD.cut_parse(tok, ds, cr["chain_ids"])
                r = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"])
                pids = RD.prompt_ids(tok, ds, it["question_A"])
                g = CH.token_geometry(tok, pids, cr["chain_ids"], r)
                self.assertIsNone(g["drop"])
                self.assertEqual("".join(g["tokens_A"]), g["derived_A"])
                self.assertEqual("".join(g["tokens_B"]), g["derived_B"])
                self.assertEqual(len(pids) + len(g["prefix_A"]), len(pids) + len(g["prefix_B"]))
                self.assertEqual(g["P"][1], len(pids) + len(g["prefix_A"]))
                n += 1
        if n == 0:
            self.skipTest("no prescreen file")


class DataTests(unittest.TestCase):
    def test_pools_splits_sources(self):
        with open(os.path.join(RD.DATA, "sources.json"), encoding="utf-8") as f:
            src = json.load(f)
        for f, h in src["files"].items():
            self.assertEqual(RD.sha256_file(os.path.join(RD.DATA, f)), h, f)
        for ds in RD.DATASETS:
            pool = RD.load_pool(ds)
            ids = [int(it["idx"]) for it in pool["items"]]
            self.assertEqual(len(ids), len(set(ids)))
            sp = RD.load_split(ds)
            self.assertEqual(sp["order"], RD.draw_order(ids, ds))
            self.assertEqual(sorted(sp["order"]), sorted(ids))
            self.assertEqual(sp["n_split"], {"main": 50, "pilot": 10})
            for it in pool["items"][:5]:
                self.assertNotEqual(it["question_A"], it["question_B"])
        self.assertEqual(src["pool_n"], {"gsm8k": 395, "asdiv": 1036})


class FakeModelTests(unittest.TestCase):
    DTYPE = torch.float32

    def setUp(self):
        self.ad, self.steer = fake_adapter(dtype=self.DTYPE)
        self.p = rand_ids(24, 1)
        self.clean, _ = self.ad.decode([self.p], 30, K, [], row_ids=None)
        self.clean = self.clean[0]
        self.n_pre = 10
        self.preA = self.clean[:self.n_pre]
        self.preB = self.preA[:-2] + [(self.preA[-2] % 50) + 5, (self.preA[-1] % 50) + 7]
        T = len(self.p) + self.n_pre
        self.P = [T - 2, T]
        self.g = geom(self.p, self.p[:3] + [9] + self.p[4:], self.preA, self.preB, self.P, [3, 4])

    def rows(self, names, key_of=lambda a: "it|" + a):
        for a in names:
            self.steer.rows[key_of(a)] = AR.row_info(a, self.g, "fake:0")
        return [key_of(a) for a in names]

    def test_T1_prefix_pass_and_decode_reproduce_clean_chain(self):
        for seed in (1, 2, 3):
            p = rand_ids(20 + seed, seed)
            clean = self.ad.decode([p], 30, K, [], row_ids=None)[0][0]
            for n_pre in (5, 12):
                g = geom(p, p, clean[:n_pre], clean[:n_pre], [len(p) + n_pre - 1, len(p) + n_pre], [0, 1])
                self.steer.rows["floor"] = AR.row_info("floor", g, "fake:%d" % seed)
                cont = self.ad.decode([p + clean[:n_pre]], 30 - n_pre, K, [], row_ids=["floor"])[0][0]
                self.assertEqual(cont, clean[n_pre:], (seed, n_pre))

    def test_idle_hooks_bit_exact(self):
        keys = self.rows(["floor"])
        seq = self.p + self.preA
        hook = AR.NeuronHook(self.steer, layer=1).install(self.ad.model)
        z1, kv1, _ = prefill(self.ad, self.steer, seq, keys)
        hook.remove()
        self.steer.remove()
        z0, kv0, _ = prefill(self.ad, self.steer, seq, None)
        self.assertTrue(torch.equal(z0, z1))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(kv0, kv1)))

    def test_T1_cache_semantics_of_a_loop_j_patch(self):
        L = self.ad.model.config.num_hidden_layers
        seq = self.p + self.preA
        T, p0 = len(seq), self.P[0]
        self.assertEqual(T, self.P[1])                                     # the prefix ends on the value
        z0, kv0, _ = prefill(self.ad, self.steer, seq, self.rows(["floor"]))
        for j in (1, 2, 3):
            zj, kvj, _ = prefill(self.ad, self.steer, seq, self.rows(["cv%d" % j]))
            for ut in range(K):
                for li in range(L):
                    a, b = kv0[ut * L + li], kvj[ut * L + li]
                    self.assertTrue(torch.equal(a[:, :, :p0], b[:, :, :p0]))   # earlier positions never change
                    if ut < j:
                        self.assertTrue(torch.equal(a, b), (j, ut, li))     # loops 1..j keep their K/V at P
                    else:
                        self.assertFalse(torch.equal(a[:, :, p0:], b[:, :, p0:]), (j, ut, li))
            self.assertFalse(torch.equal(z0, zj))
        # a post patch at the end of loop 4 writes no K/V entry: with P on the last position it moves only the
        # next-token read-out, with P before it nothing at all (a structural zero for later tokens)
        for P, moves in (([p0, T], True), ([p0 - 3, p0 - 1], False)):
            key = "it|post4_%d" % P[0]
            self.steer.rows[key] = dict(AR.row_info("cv1", dict(self.g, P=P), "fake:0"),
                                        edits=[{"op": "patch", "site": "post", "loop": 4, "pos": P, "a": 1.0}])
            z4, kv4, _ = prefill(self.ad, self.steer, seq, [key])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(kv0, kv4)))
            self.assertEqual(not torch.equal(z0, z4), moves)

    def test_T1_loop4_kv_arm_writes_what_later_tokens_read(self):
        L = self.ad.model.config.num_hidden_layers
        seq = self.p + self.preA
        p0, p1 = self.P
        _z0, kv0, c0 = prefill(self.ad, self.steer, seq, self.rows(["floor"]))
        _zk, kvk, ck = prefill(self.ad, self.steer, seq, self.rows(["kv4"]))
        _zd, _kvd, cd = prefill(self.ad, self.steer, self.p + self.preB, None)    # the donor sequence, clean
        for ut in range(K):
            for li in range(L):
                i = ut * L + li
                if ut < K - 1:
                    self.assertTrue(torch.equal(kv0[i], kvk[i]))
                    continue
                self.assertTrue(torch.equal(kv0[i][:, :, :p0], kvk[i][:, :, :p0]))
                self.assertFalse(torch.equal(kv0[i][:, :, p0:p1], kvk[i][:, :, p0:p1]))
                self.assertTrue(torch.allclose(ck.key_cache[i][:, :, p0:p1], cd.key_cache[i][:, :, p0:p1], atol=1e-6))
                self.assertTrue(torch.allclose(ck.value_cache[i][:, :, p0:p1], cd.value_cache[i][:, :, p0:p1],
                                               atol=1e-6))
        # later tokens read it: the continuation of the kv4 row differs from the floor's for this fake
        keys = ["it|floor", "it|kv4"]
        outs = self.ad.decode([seq, seq], 8, K, [], row_ids=keys)[0]
        self.assertEqual(outs[0], self.clean[self.n_pre:self.n_pre + 8])

    def test_T4_donor_is_prompt_A_plus_prefix_B(self):
        keys = self.rows(["cv1", "cv3"])
        self.steer.ensure_donors(keys)
        d = self.steer.donors[keys[0]]
        self.steer.rec = {"cols": slice(self.P[0], self.P[1]), "want": {"post"}, "post": {}, "pre": {}, "kv": {},
                          "emb": None, "logits": []}
        with torch.no_grad():
            self.ad.model(input_ids=torch.tensor([self.p + self.preB]), use_cache=False, logits_to_keep=1)
        ref = self.steer.rec
        self.steer.rec = None
        for j in range(1, K + 1):
            self.assertTrue(torch.equal(d["post"][j], ref["post"][j][0]))
        # the patched run's loop-j end state at P equals the donor's, bit-exact (lerp at a = 1 returns the end)
        for key, j in zip(keys, (1, 3)):
            self.steer.rec = {"cols": slice(self.P[0], self.P[1]), "want": {"post"}, "post": {}, "pre": {},
                              "kv": {}, "emb": None, "logits": []}
            prefill(self.ad, self.steer, self.p + self.preA, [key])
            got = self.steer.rec["post"][j][0]
            self.steer.rec = None
            self.assertTrue(torch.equal(got, self.steer.donors[key]["post"][j]))

    def test_T5_random_same_norm(self):
        keys = self.rows(["cv1", "rnd1"])
        self.steer.stats.clear()
        self.steer.prompt_counted.clear()
        for k in keys:
            prefill(self.ad, self.steer, self.p + self.preA, [k])
        dn = self.steer.stats[(keys[0], "post@1")][0]
        rn = self.steer.stats[(keys[1], "post@1")][0]
        self.assertAlmostEqual(rn / dn, 1.0, places=5)
        u1 = S.unit_dirs("fake:0", "post@1", list(range(*self.P)), 0, 16)
        u2 = S.unit_dirs("fake:0", "post@1", list(range(*self.P)), 0, 16)
        u3 = S.unit_dirs("fake:1", "post@1", list(range(*self.P)), 0, 16)
        self.assertTrue(torch.equal(u1, u2))
        self.assertFalse(torch.equal(u1, u3))
        self.assertEqual(self.steer.rows[keys[1]]["edits"][0]["rkey"], "fake:0")

    def test_neuron_hook(self):
        m = self.ad.model
        L = m.config.num_hidden_layers
        hook = AR.NeuronHook(self.steer, layer=L - 1).install(m)
        lo, hi = self.P
        aA = AR.record_acts(m, self.steer, hook, self.p + self.preA, lo, hi)
        aD = AR.record_acts(m, self.steer, hook, self.p + self.preB, lo, hi)
        self.assertEqual(aA.dtype, next(m.parameters()).dtype)
        self.assertFalse(torch.equal(aA, aD))
        seen = {}

        def spy(module, args):
            if self.steer.f is not None and self.steer.f.get("ut") == 0:
                seen["a"] = args[0][0, lo:hi].clone()
        h2 = m.model.layers[L - 1].mlp.down_proj.register_forward_pre_hook(spy)
        keys = self.rows(["nc_full"])
        z_idle, _kv, _ = prefill(self.ad, self.steer, self.p + self.preA, keys)
        a_idle = seen["a"]
        hook.specs[keys[0]] = {"lo": lo, "hi": hi, "idx": None, "donor": aD}
        z_full, _kv, _ = prefill(self.ad, self.steer, self.p + self.preA, keys)
        self.assertTrue(torch.equal(seen["a"], aD))                        # full set: the donor values exactly
        self.assertFalse(torch.equal(z_idle, z_full))
        hook.specs[keys[0]] = {"lo": lo, "hi": hi, "idx": torch.tensor([], dtype=torch.long), "donor": aD}
        z_none, _kv, _ = prefill(self.ad, self.steer, self.p + self.preA, keys)
        self.assertTrue(torch.equal(z_none, z_idle))                       # empty set: bit-exact no-op
        idx = torch.tensor([0, 5, 7])
        hook.specs[keys[0]] = {"lo": lo, "hi": hi, "idx": idx, "donor": aD}
        prefill(self.ad, self.steer, self.p + self.preA, keys)
        rest = [i for i in range(aA.shape[-1]) if i not in (0, 5, 7)]
        self.assertTrue(torch.equal(seen["a"][:, idx], aD[:, idx]))
        self.assertTrue(torch.equal(seen["a"][:, rest], a_idle[:, rest]))
        self.assertTrue(torch.equal(a_idle, aA))                           # live a equals the recorded a (fake)
        h2.remove()
        hook.remove()

    def test_batch_of_arms_matches_single_rows(self):
        """Review N3: the full prefix batch and the full scratch batch, neuron rows included."""
        m = self.ad.model
        hook = AR.NeuronHook(self.steer, layer=m.config.num_hidden_layers - 1).install(m)
        keys = self.rows(AR.ARM_NAMES)
        for setting, pre in (("chain", "nc_"), ("span", "ns_")):
            if setting == "chain":
                lo, hi = self.P
                sA, sD = self.p + self.preA, self.p + self.preB
            else:
                lo, hi = self.g["span"]
                sA, sD = self.g["prompt_A"], self.g["prompt_B"]
            aA = AR.record_acts(m, self.steer, hook, sA, lo, hi)
            aD = AR.record_acts(m, self.steer, hook, sD, lo, hi)
            sets = AR.neuron_sets(aA, aD, [0, 1], "fake", "gsm8k", setting, "fake:0")
            for st in AR.NEURON_SETS:
                idx = None if sets[st] is None else torch.tensor(sets[st], dtype=torch.long)
                hook.specs["it|" + pre + st] = {"lo": lo, "hi": hi, "idx": idx, "donor": aD}
        for batch in ("prefix", "scratch"):
            bk = [k for k in keys if AR.ARM[k.split("|")[1]]["batch"] == batch]
            seqs = [self.steer.rows[k]["a_ids"] for k in bk]
            self.assertEqual(len({len(x) for x in seqs}), 1)              # equal lengths: no padding
            out = self.ad.decode(seqs, 15, K, [], row_ids=bk)[0]
            for k, x, o in zip(bk, seqs, out):
                one = self.ad.decode([x], 15, K, [], row_ids=[k])[0][0]
                self.assertEqual(one, o, k)
            if batch == "prefix":
                self.assertEqual(out[bk.index("it|floor")], self.clean[self.n_pre:self.n_pre + 15])
        self.assertTrue(all(hook.n_applied.get("it|" + a, 0) > 0 for a in ("nc_full", "ns_full", "nc_t20")))
        hook.remove()

    def test_B1_identities_at_the_cache_level(self):
        """Review B1: cvP = cv1 and kv4 = cv3 (K and V of every loop, layer and position after the prefix pass)."""
        seq = self.p + self.preA
        T = len(seq)
        for a, b in AR.IDENTITIES:
            ka, kb = self.rows([a, b])
            _za, _kva, ca = prefill(self.ad, self.steer, seq, [ka])
            _zb, _kvb, cb = prefill(self.ad, self.steer, seq, [kb])
            for i in range(len(ca.key_cache)):
                self.assertTrue(torch.equal(ca.key_cache[i][:, :, :T], cb.key_cache[i][:, :, :T]), (a, b, i))
                self.assertTrue(torch.equal(ca.value_cache[i][:, :, :T], cb.value_cache[i][:, :, :T]), (a, b, i))

    def test_kv1_writes_the_loop1_kv_of_the_donor(self):
        L = self.ad.model.config.num_hidden_layers
        seq = self.p + self.preA
        p0, p1 = self.P
        _z0, kv0, _c0 = prefill(self.ad, self.steer, seq, self.rows(["floor"]))
        _zk, kvk, ck = prefill(self.ad, self.steer, seq, self.rows(["kv1"]))
        _zd, _kd, cd = prefill(self.ad, self.steer, self.p + self.preB, None)
        for li in range(L):
            self.assertTrue(torch.allclose(ck.key_cache[li][:, :, p0:p1], cd.key_cache[li][:, :, p0:p1], atol=1e-6))
            self.assertTrue(torch.allclose(ck.value_cache[li][:, :, p0:p1], cd.value_cache[li][:, :, p0:p1],
                                           atol=1e-6))
            self.assertTrue(torch.equal(kv0[li][:, :, :p0], kvk[li][:, :, :p0]))
        self.assertEqual(AR.row_info("kv1", self.g, "x")["edits"],
                         [{"op": "kv", "loops": [1], "pos": [p0, p1], "a": 1.0}])


class FakeModelBF16Tests(FakeModelTests):
    """Review N6: T4, the neuron hook, the idle hooks and the B1 identities on the fake cast to bfloat16."""
    DTYPE = torch.bfloat16

    def _f32_only(self):
        self.skipTest("float32 only")
    test_T1_prefix_pass_and_decode_reproduce_clean_chain = _f32_only
    test_T1_cache_semantics_of_a_loop_j_patch = _f32_only
    test_T1_loop4_kv_arm_writes_what_later_tokens_read = _f32_only
    test_T5_random_same_norm = _f32_only
    test_batch_of_arms_matches_single_rows = _f32_only
    test_kv1_writes_the_loop1_kv_of_the_donor = _f32_only


class ArmAndSetTests(unittest.TestCase):
    def test_arm_table(self):
        self.assertEqual(AR.ARM_NAMES[0], "base")
        self.assertEqual(len(AR.ARM_NAMES), len(set(AR.ARM_NAMES)))
        self.assertEqual(len([a for a in AR.ARM_NAMES if AR.ARM[a]["batch"] == "prefix"]), 15)
        self.assertEqual(len([a for a in AR.ARM_NAMES if AR.ARM[a]["batch"] == "scratch"]), 9)
        self.assertEqual(AR.IDENTITIES, (("cvP", "cv1"), ("kv4", "cv3")))
        g = geom([5] * 20, [5] * 3 + [6] + [5] * 16, [7] * 6, [7] * 5 + [8], [25, 26], [3, 4])
        for a in AR.ARM_NAMES:
            r = AR.row_info(a, g, "x:1")
            if AR.ARM[a]["batch"] == "prefix":
                self.assertEqual(len(r["a_ids"]), 26)
                self.assertEqual(r["donor_range"], (25, 26))
            else:
                self.assertEqual(len(r["a_ids"]), 20)
                self.assertEqual(r["donor_range"], (3, 20))
        self.assertEqual(AR.row_info("cv2", g, "x:1")["edits"], [{"op": "patch", "site": "post", "loop": 2,
                                                                  "pos": [25, 26], "a": 1.0}])
        self.assertEqual(AR.row_info("text", g, "x:1")["a_ids"], [5] * 20 + [7] * 5 + [8])
        self.assertEqual(AR.row_info("base", g, "x:1")["edits"][0]["pos"], "last_prompt")
        self.assertEqual(AR.key_of("gsm8k", 7, "cv1"), "gsm8k:7|cv1")
        self.assertNotIn("kv4", AR.ARM_NAMES)
        self.assertIn("kv1", AR.PREFIX_ARMS)
        bad = dict(g, prompt_B=[5] * 21)                                   # review B3
        with self.assertRaises(ValueError):
            AR.row_info("base", bad, "x:1")
        bad = dict(g, prompt_B=[6] * 20)
        with self.assertRaises(ValueError):
            AR.row_info("span1", bad, "x:1")

    def test_neuron_sets(self):
        I = 5632
        self.assertEqual([AR.k_of(AR.FRAC[s], I) for s in ("t1", "t5", "t20")], [56, 282, 1126])
        g = torch.Generator().manual_seed(0)
        aA = torch.randn(2, I, generator=g).to(torch.bfloat16)
        aD = torch.randn(2, I, generator=g).to(torch.bfloat16)
        ds = AR.top_set(AR.dataset_scores([(aA, aD)]), 282)
        s1 = AR.neuron_sets(aA, aD, ds, "ouro_1_4b_base", "gsm8k", "chain", "gsm8k:1")
        s2 = AR.neuron_sets(aA, aD, None, "ouro_1_4b_base", "gsm8k", "chain", "gsm8k:2")
        self.assertIsNone(s1["full"])
        self.assertEqual([len(s1[s]) for s in ("t1", "t5", "t20", "r5", "ds5")], [56, 282, 1126, 282, 282])
        self.assertNotIn("ds5", s2)                                        # split items: no dataset-level arm
        self.assertEqual(s1["t5"], s2["t5"])
        self.assertNotEqual(s1["r5"], s2["r5"])
        self.assertEqual(s1["r5"], AR.neuron_sets(aA, aD, ds, "ouro_1_4b_base", "gsm8k", "chain", "gsm8k:1")["r5"])
        self.assertTrue(set(s1["t1"]) <= set(s1["t5"]) <= set(s1["t20"]))
        self.assertEqual(AR.top_set(torch.tensor([1.0, 3.0, 3.0, 0.0]), 2), [1, 2])

    def test_no_host_user_home_strings(self):
        bad = re.compile("(" + "|".join(["/" + "home/", "/" + "Users/", r"C:\\\\" + "Users", "sas" + "11", "dgx" + "1",
                                         "rtx" + "6000", r"/var/tmp/[a-z]+/out"]) + ")", re.I)
        hits = []
        for d, _sub, files in os.walk(ROOT):
            if "__pycache__" in d or os.path.join("tree", "data_steer") in d or os.sep + "out" in d:
                continue
            for f in files:
                if not f.endswith((".py", ".md", ".json", ".sh", ".txt", ".yaml")):
                    continue
                p = os.path.join(d, f)
                if f == "test_ra30.py":
                    continue
                with open(p, encoding="utf-8", errors="replace") as fh:
                    if bad.search(fh.read()):
                        hits.append(os.path.relpath(p, ROOT))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
