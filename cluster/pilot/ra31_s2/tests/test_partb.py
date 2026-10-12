"""CPU tests of RA-31 Part B (no GPU, no checkpoint, no download): the write hooks inside the bundle's steer_gen
SteerHooks on the prod CPU fake of Ouro under prod's own cached batch decode (left padding): the alpha-1 identity of
every op bit-exact, the cache semantics (T1: the cached greedy chain equals a no-cache teacher-forced recompute with
the same hooks), the scale and subset ops, all-position against from-last positions, the loop-2 arm; the shard table
and card-hours; the frozen sets; the pins; one shard end to end on the fake (run_shard with the decode replaced by
prod's decode over the fake) and the merge.

    cd ra31 && python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(HERE, "lib")
TREE = os.path.join(HERE, "tree")
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (TESTS, TREE, LIB, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PROD_DEV"] = "cpu"

import torch  # noqa: E402

import ra31_gen as G  # noqa: E402
import prod_fakes as PF  # noqa: E402

if PF.ROOT not in (HERE, LIB, TREE, TESTS) and PF.ROOT in sys.path:
    sys.path.remove(PF.ROOT)

import prod.steer_gen as SG  # noqa: E402

L_F, I_F, H_F, HD_F = 4, 32, 2, 8


def fake_sets():
    g = torch.Generator().manual_seed(3)
    return {"mlp": {"top": sorted(torch.randperm(L_F * I_F, generator=g)[:6].tolist()),
                    "rand": sorted(torch.randperm(L_F * I_F, generator=g)[:6].tolist())},
            "attn": {"top": [1, 6], "rand": [2, 5]}}


def prompts(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for i in range(n):
        T = 6 + (i % 3) * 2                                           # unequal lengths: left padding in a batch
        out[str(100 + i)] = [int(x) for x in torch.randint(3, 60, (T,), generator=g)]
    return out


class Rig(object):
    """One fake Ouro (float32) and a decode with any hooks, through prod's OuroAdapter.decode (cached, left-padded)."""

    def __init__(self, test, k=4):
        self.fake = PF.make_fake_ouro("eager", L=L_F, total_ut_steps=4)
        self.k = k
        self.test = test

    def decode(self, hooks, rows, k=None, n_new=6, width=3):
        k = k or self.k
        holder = {}

        def cls(name):
            holder["ad"] = SG.SteeredOuroAdapter(name, steer=hooks)
            return holder["ad"]
        ad = PF.real_ouro_adapter_over(self.fake, k, cls=cls)
        hooks.install(self.fake)
        hooks.rows = {key: {"plen": len(ids), "a_ids": ids, "edits": [], "lcdl": None} for key, ids in rows.items()}
        keys = sorted(rows)
        out = {}
        try:
            for i in range(0, len(keys), width):
                kk = keys[i:i + width]
                gen = ad.decode([rows[x] for x in kk], n_new, k, [], row_ids=kk)[0]
                for x, gx in zip(kk, gen):
                    out[x] = [int(t) for t in gx]
        finally:
            hooks.remove()
        return out


def arm(edits, k=4):
    return {"name": "x", "k": k, "edits": edits}


def E(br, op, a, s="all", loops=(2, 3, 4), pos="from_last"):
    return {"branch": br, "op": op, "alpha": a, "set": s, "loops": list(loops), "positions": pos}


class TestHooks(unittest.TestCase):
    def setUp(self):
        self.rig = Rig(self)
        self.rows = prompts()
        self.base = self.rig.decode(SG.SteerHooks(4), self.rows)

    def test_alpha1_identity_every_op(self):
        edits = [E(b, "scale", 1.0) for b in ("mlp", "attn")] + \
                [E(b, "subset", 1.0, s) for b in ("mlp", "attn") for s in ("top", "rand")]
        h = G.make_hooks(4, edits, fake_sets())
        got = self.rig.decode(h, self.rows)
        self.assertEqual(got, self.base)
        self.assertGreater(h.n_edit_calls, 0)

    def test_scale_and_subset_change_the_chain(self):
        for e in (E("mlp", "scale", 0.0), E("attn", "scale", 2.0), E("mlp", "subset", 0.0, "top")):
            got = self.rig.decode(G.make_hooks(4, [e], fake_sets()), self.rows)
            self.assertNotEqual(got, self.base, e)

    def test_subset_all_reduces_to_scale(self):
        """The decomposition over every neuron equals the whole write up to float32 rounding (same chains)."""
        sets = fake_sets()
        sets["mlp"]["top"] = list(range(L_F * I_F))
        a = self.rig.decode(G.make_hooks(4, [E("mlp", "scale", 0.5)], sets), self.rows)
        b = self.rig.decode(G.make_hooks(4, [E("mlp", "subset", 0.5, "top")], sets), self.rows)
        self.assertEqual(a, b)

    def test_cache_semantics_T1(self):
        """The cached greedy chain under an edit equals the argmax of a no-cache forward of prompt + chain with the
        same hooks (positions >= plen - 1 edited in both): decoding uses the edited cache unchanged."""
        e = E("mlp", "subset", 0.3, "top")
        h = G.make_hooks(4, [e], fake_sets())
        got = self.rig.decode(h, self.rows, width=1)
        h2 = G.make_hooks(4, [e], fake_sets())
        h2.install(self.rig.fake)
        h2.rows = {key: {"plen": len(ids), "a_ids": ids, "edits": [], "lcdl": None} for key, ids in self.rows.items()}
        try:
            for key, ids in self.rows.items():
                seq = ids + got[key][:-1]
                h2.batch_keys = [key]
                with torch.no_grad():
                    z = self.rig.fake(input_ids=torch.tensor([seq]), use_cache=False).logits[0]
                h2.batch_keys = None
                pred = z[len(ids) - 1:].argmax(-1).tolist()
                self.assertEqual(pred, got[key], key)
        finally:
            h2.remove()

    def logits_under(self, hooks, key):
        """No-cache logits of one prompt with the hooks installed (the prefill of the cached decode)."""
        hooks.install(self.rig.fake)
        ids = self.rows[key]
        hooks.rows = {key: {"plen": len(ids), "a_ids": ids, "edits": [], "lcdl": None}}
        self.rig.fake.model.total_ut_steps, self.rig.fake.early_exit_step = 4, 3
        try:
            hooks.batch_keys = [key]
            with torch.no_grad():
                return self.rig.fake(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
        finally:
            hooks.batch_keys = None
            hooks.remove()

    def test_all_positions_differs_from_last(self):
        key = sorted(self.rows)[0]
        T = len(self.rows[key])
        za = self.logits_under(G.make_hooks(4, [E("attn", "scale", 0.0, pos="all")], fake_sets()), key)
        zb = self.logits_under(G.make_hooks(4, [E("attn", "scale", 0.0)], fake_sets()), key)
        z0 = self.logits_under(SG.SteerHooks(4), key)
        self.assertTrue(torch.equal(zb[:T - 1], z0[:T - 1]))          # from-last leaves earlier positions alone
        self.assertFalse(torch.equal(za[:T - 1], z0[:T - 1]))         # all-position edits them
        self.assertTrue(torch.equal(za[T - 1], zb[T - 1]))           # r's attention write is zero in both, so r
        ma = self.logits_under(G.make_hooks(4, [E("mlp", "scale", 0.0, pos="all")], fake_sets()), key)   # reads
        mb = self.logits_under(G.make_hooks(4, [E("mlp", "scale", 0.0)], fake_sets()), key)   # nothing earlier;
        self.assertFalse(torch.equal(ma[T - 1], mb[T - 1]))          # the MLP arm reaches r through attention

    def test_loop2_arm_at_k2(self):
        rig2 = Rig(self, k=2)
        base2 = rig2.decode(SG.SteerHooks(2), self.rows, k=2)
        same = rig2.decode(G.make_hooks(2, [E("mlp", "scale", 1.0, loops=(2,))], fake_sets()), self.rows, k=2)
        self.assertEqual(same, base2)
        h = G.make_hooks(2, [E("mlp", "scale", 0.0, loops=(2,))], fake_sets())
        self.assertNotEqual(rig2.decode(h, self.rows, k=2), base2)


class TestTable(unittest.TestCase):
    def test_shards(self):
        S = G.shards()
        names = [a["name"] for v in S.values() for a in v]
        self.assertEqual(len(names), 54)                     # 53 planned arms + r00_rerun
        self.assertEqual(len(set(names)), 54)
        self.assertEqual(sum(1 for n in names if n.startswith("k4_")), 42)
        self.assertEqual(sum(1 for n in names if n.startswith("k2_")), 8)
        self.assertEqual(sorted(n for n in names if n.startswith("allpos")), ["allpos_attn_a0", "allpos_mlp_a0"])
        self.assertTrue(all(G.shard_hours(v) < 6.0 for v in S.values()))
        self.assertFalse(any(a["k"] == 4 and any(e["alpha"] == 1.0 for e in a["edits"]) and a["name"] != "ident"
                             for v in S.values() for a in v))

    def test_frozen_sets_and_pins(self):
        d = json.load(open(os.path.join(G.DATA_B, "partb_sets_1.4b.json")))
        self.assertEqual(len(d["mlp"]["top"]), 6758)
        self.assertEqual(len(d["mlp"]["rand"]), 6758)
        self.assertEqual(len(d["attn"]["top"]), 2)
        self.assertEqual(d["n_items"], 50)
        G.check_data()
        pins, bad = G.check_pins()
        self.assertEqual(bad, [], pins)
        scored, ids = G.row_ids()
        self.assertEqual((len(scored), len(ids)), (395, 417))


class TestSummaryTies(unittest.TestCase):
    def test_summary_prints_the_tie_count(self):
        d = tempfile.mkdtemp(prefix="ra31_s_")
        self.addCleanup(shutil.rmtree, d, True)
        ck = G.summarize(d)
        self.assertEqual(ck["n_done"], 0)
        self.assertEqual(ck["set_ties_at_cut"]["mlp"]["n_tied_at_cut"], 803)
        self.assertIn("803", open(os.path.join(d, "summary_partb.md")).read())


class TestShardEndToEnd(unittest.TestCase):
    """run_shard on the fake: data dir and decode replaced, everything else (hooks, done markers, identity against
    r00_rerun, summarize) as on the cluster."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ra31_b_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rig = Rig(self)
        self.rows = prompts(6, seed=4)
        d = self.data = os.path.join(self.tmp, "data")
        os.makedirs(d)
        ids = sorted(int(k) for k in self.rows)
        json.dump({"scored": [{"idx": i} for i in ids[:5]], "sentinels": [{"idx": ids[5]}]},
                  open(os.path.join(d, "items.json"), "w"))
        base = self.rig.decode(SG.SteerHooks(4), self.rows)
        with open(os.path.join(d, "r00_chains.jsonl"), "w") as f:
            for k_, v in sorted(base.items()):
                f.write(json.dumps({"idx": int(k_), "chain_ids": v}) + "\n")
        with open(os.path.join(d, "r00_cells.jsonl"), "w") as f:
            for k_, v in sorted(base.items()):
                f.write(json.dumps({"row_idx": int(k_), "trace_answer": str(v[-1]), "trace_correct": v[-1] % 2 == 0,
                                    "n_trace": len(v)}) + "\n")
        json.dump({"model_meta": {"steer_gen": G.STEER_GEN_VERSION, "file_md5": {}}},
                  open(os.path.join(d, "r00_meta.json"), "w"))
        sets = fake_sets()
        json.dump({"mlp": sets["mlp"], "attn": sets["attn"], "n_items": 0},
                  open(os.path.join(d, "partb_sets_1.4b.json"), "w"))
        outs = {f: G.sha256_file(os.path.join(d, f)) for f in os.listdir(d)}
        json.dump({"sources": {}, "outputs": outs}, open(os.path.join(d, "sources_partb.json"), "w"))
        self.saved = G.row_infos
        G.row_infos = lambda tok, ids: {str(i): {"plen": len(self.rows[str(i)]), "a_ids": self.rows[str(i)],
                                                 "edits": [], "lcdl": None} for i in ids}
        self.addCleanup(setattr, G, "row_infos", self.saved)

    def gen(self, a, hooks, ids, out_dir, mem):
        """Prod's cached decode over the fake in place of G.main; writes the chains and cells G.main would."""
        os.makedirs(out_dir, exist_ok=True)
        rows = {str(i): self.rows[str(i)] for i in ids}
        holder_rows = dict(hooks.rows)
        got = self.rig.decode(hooks, rows, k=a["k"])
        hooks.rows = holder_rows
        with open(os.path.join(out_dir, "chains_fake.jsonl"), "w") as f:
            for k_, v in sorted(got.items()):
                f.write(json.dumps({"idx": int(k_), "chain_ids": v}) + "\n")
        with open(os.path.join(out_dir, "cells_fake.jsonl"), "w") as f:
            for k_, v in sorted(got.items()):
                f.write(json.dumps({"row_idx": int(k_), "trace_answer": str(v[-1]), "trace_correct": v[-1] % 2 == 0,
                                    "n_trace": len(v)}) + "\n")
        return {"exit": 0}

    def test_ident_allpos_shard_and_merge(self):
        out = os.path.join(self.tmp, "out")
        G.run_shard("ident_allpos", out, gen=self.gen, tok_fn=lambda: None, data_dir=self.data)
        for nm in ("r00_rerun", "ident", "allpos_mlp_a0", "allpos_attn_a0"):
            self.assertTrue(os.path.exists(os.path.join(out, nm, "DONE.json")), nm)
        idn = json.load(open(os.path.join(out, "ident", "DONE.json")))
        self.assertTrue(idn["identity"]["pass"])
        self.assertTrue(idn["vs_stored_r00"]["pass"])
        G.run_shard("ident_allpos", out, gen=self.gen, tok_fn=lambda: None, data_dir=self.data)   # resume: skips
        G.run_shard("mlp_top_hi", out, gen=self.gen, tok_fn=lambda: None, data_dir=self.data)
        ck = G.summarize(out, data_dir=self.data)
        self.assertEqual(ck["n_done"], 7)
        self.assertTrue(ck["identity_pass"])
        self.assertEqual(ck["csv_rows"], 7)
        t = json.load(open(os.path.join(out, "results_partb.json")))["arms"]
        self.assertGreater(t["k4_mlp_top_a2"]["cS_norm"]["cS_norm_mlp_top"], 0)
        self.assertEqual(t["k4_attn_all_a0"]["status"], "not run")

    def test_identity_failure_exit_3(self):
        def bad(a, hooks, ids, out_dir, mem):
            r = self.gen(a, hooks, ids, out_dir, mem)
            if a["name"] == "ident":
                p = os.path.join(out_dir, "chains_fake.jsonl")
                lines = open(p).read().splitlines()
                row = json.loads(lines[0])
                row["chain_ids"][0] = (row["chain_ids"][0] + 1) % 60
                lines[0] = json.dumps(row)
                open(p, "w").write("\n".join(lines) + "\n")
            return r
        with self.assertRaises(SystemExit) as cm:
            G.run_shard("ident_allpos", os.path.join(self.tmp, "o2"), gen=bad, tok_fn=lambda: None, data_dir=self.data)
        self.assertEqual(cm.exception.code, 3)

    def test_pin_mismatch_exit_2(self):
        json.dump({"model_meta": {"steer_gen": "other"}}, open(os.path.join(self.data, "r00_meta.json"), "w"))
        with self.assertRaises(SystemExit) as cm:
            G.run_shard("ident_allpos", os.path.join(self.tmp, "o3"), gen=self.gen, tok_fn=lambda: None,
                        data_dir=self.data)
        self.assertEqual(cm.exception.code, 2)


class TestHygiene(unittest.TestCase):
    def test_no_host_user_home_strings(self):
        import re
        words = ["sas" + "11", "sab" + "bir", "/ho" + "me/", "/us" + "ers/", "dg" + "x1", "rtx" + "6000", "/var/" + "tmp/ra20"]
        pat = re.compile("(?i)(" + "|".join(re.escape(w) for w in words) + ")")
        for f in ("ra31_gen.py", "make_partb.py", "tests/test_partb.py", "data_partb/partb_sets_1.4b.json",
                  "data_partb/sources_partb.json"):
            with open(os.path.join(HERE, f), encoding="utf-8") as fh:
                self.assertIsNone(pat.search(fh.read()), f)


if __name__ == "__main__":
    unittest.main()
