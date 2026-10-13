"""CPU tests of RA-29 (no GPU, no checkpoint, no download).

Covered: the item files (hashes, counts, letter balance, transposition, block and answer positions, T2 on the cached
tokenizer when it is present); every arm and identity of process_item on the prod CPU fake of Ouro (T1 hook and
zero identities, patch written, nothing before the patch changes, causal prefix, positive control exactly 1.000,
lt_L4 = pc, block at loop 4 a structural zero, T5 random same norm); the screen and gap drops; the statistics and
outputs; run() pilot -> full -> summarize with the loader replaced by the fake; the exit codes; no host, user or
home string in the folder.

    cd ra29 && python -m unittest discover -s tests -v
"""
import csv
import hashlib
import json
import os
import random
import re
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

import ra29 as R  # noqa: E402
import make_data as MD  # noqa: E402
import k17_freeze as KF  # noqa: E402
import prod_fakes as PF  # noqa: E402

if PF.ROOT not in (HERE, LIB, TREE, TESTS) and PF.ROOT in sys.path:
    sys.path.remove(PF.ROOT)

from prod.steer.adapters import bind  # noqa: E402
from prod.steer.adapters.ouro import OuroBinding  # noqa: E402
from prod.steer.adapters.raven import RavenBinding  # noqa: E402

DATA = os.path.join(HERE, "data")
EXPECT_BUILT = {"aqua": 250, "csqa": 1215, "arc": 1172, "mmlu": 1999, "hellaswag": 2000}
EXPECT_DUP = {"aqua": 4, "csqa": 6, "arc": 0, "mmlu": 1, "hellaswag": 0}
FOUR = ("arc", "mmlu", "hellaswag")
KNOWN_LOCAL_DRIFT = ("aqua:37",)   # the item of the first host rebuild (see test_T2_ids_reproduce_the_host_build)


def non_ascii(it, h):
    return any(ord(ch) > 127 for ch in h + it["input_A"] + it["input_B"])


def tmpdir(test):
    d = tempfile.mkdtemp(prefix="ra29_t_")
    test.addCleanup(shutil.rmtree, d, True)
    return d


class Fake(object):
    """KF.Ctx over the prod CPU fake of Ouro (as RUN 1c's tests), kept open for one test."""

    def __init__(self, test, L=8):
        fake = PF.make_fake_ouro("eager", L=L, total_ut_steps=KF.K)
        ad = PF.real_ouro_adapter_over(fake, KF.K)
        cm = OuroBinding.trust_for_tests(PF.OuroDecoderLayer)
        cm.__enter__()
        test.addCleanup(cm.__exit__, None, None, None)
        self.b = bind(ad)
        self.X = KF.Ctx(self.b, TREE)


def fake_item(X, seed, T=24, b0=6, b1=14, tA=None, tB=None):
    g = random.Random(seed)
    A = [g.randrange(3, 64) for _ in range(T)]
    B = list(A)
    for p in range(b0 + 2, b1):
        B[p] = g.randrange(3, 64)
    zA = X.nohook(torch.tensor([A]))[0, -1]
    zB = X.nohook(torch.tensor([B]))[0, -1]
    if tA is None:
        tA, tB = int(zA.argmax()), int(zB.argmax())
    return {"key": "t:%d" % seed, "A": A, "B": B, "r": T - 1, "block": list(range(b0, b1)), "tA": tA, "tB": tB,
            "lA": "a", "lB": "b", "letter_ids": sorted({tA, tB})}


def fake_items(X, n, start=0):
    out, s = [], start
    while len(out) < n:
        it = fake_item(X, s)
        s += 1
        if it["tA"] != it["tB"]:
            out.append(it)
    return out


# ------------------------------------------------------------------ item files (real data, CPU)
class TestData(unittest.TestCase):
    def setUp(self):
        self.src = json.load(open(os.path.join(DATA, "sources.json")))

    def test_files_match_sources(self):
        for f, h in self.src["outputs_sha256"].items():
            self.assertEqual(KF.sha256_file(os.path.join(DATA, f)), h, f)

    def test_counts_drops_and_letter_balance(self):
        for ds in R.DATASETS:
            meta, head, built = R.load_items(DATA, ds)
            self.assertEqual(len(built), EXPECT_BUILT[ds], ds)
            self.assertEqual(meta["drops"].get("duplicate_gold_text", 0), EXPECT_DUP[ds], ds)
            self.assertEqual(set(meta["drops"]) - {"duplicate_gold_text"}, set(), ds)
            c = {}
            for it in built:
                self.assertNotEqual(it["lA"], it["lB"])
                c[it["lB"]] = c.get(it["lB"], 0) + 1
            main = [c.get(l, 0) for l in "abcde"[:4 if ds in FOUR else 5]]
            self.assertLessEqual(max(main) - min(main), 1, (ds, c))

    def test_transposition_and_positions(self):
        for ds in R.DATASETS:
            meta, head, built = R.load_items(DATA, ds)
            tail = len(meta["suffix_ids"]) + len(meta["cue_ids"])
            for it in built:
                parse = MD.fmt_fns(MD.TASK_FORMAT[ds])[0]
                qa, ta = parse(it["input_A"], it["n_opt"])
                qb, tb = parse(it["input_B"], it["n_opt"])
                self.assertEqual(qa, qb)
                ia, ib = "abcdef".index(it["lA"]), "abcdef".index(it["lB"])
                sw = list(ta)
                sw[ia], sw[ib] = ta[ib], ta[ia]
                self.assertEqual(sw, tb)
                self.assertEqual(sorted(ta), sorted(tb))
                self.assertEqual(it["r"], it["T"] - 1)
                b0, b1 = it["block"]
                self.assertTrue(0 < b0 < b1 <= it["T"] - tail - 1, it["key"])
                self.assertEqual(it["n_prompt_tokens"], it["grid_n_prompt_tokens"])

    def test_raven_item_files(self):
        want = {("aqua", "huginn"): 250, ("csqa", "huginn"): 1215, ("aqua", "mcleish"): 250, ("csqa", "mcleish"): 1215,
                ("arc", "mcleish"): 1172}
        for tag in ("think14", "think26"):
            want.update({("aqua", tag): 250, ("csqa", tag): 1215, ("arc", tag): 1172})
        cue = {"huginn": [318], "mcleish": [320], "think14": [365], "think26": [365]}
        for (ds, tag), n in want.items():
            meta, head, built = R.load_items(DATA, ds, tag)
            self.assertEqual(len(built), n, (ds, tag))
            self.assertEqual(meta["tok"], tag)
            self.assertEqual(meta["bos_id"], 128000 if tag == "mcleish" else None)
            self.assertEqual(meta["cue_ids"], cue[tag])
            if tag.startswith("think"):
                self.assertTrue(meta["chat_template"])
                self.assertEqual(meta["head_file"], "prefix_%s__%s.txt" % (ds, tag))
                self.assertEqual(meta["suffix_pre_text"], "</think>")
            for m in R.MODELS.values():
                if m.get("items_tag") == tag:
                    break
            else:
                self.fail("no model reads items tag %s" % tag)
            c = {}
            for it in built:
                self.assertNotEqual(it["lA"], it["lB"])
                c[it["lB"]] = c.get(it["lB"], 0) + 1
                self.assertEqual(it["n_prompt_tokens"], it["grid_n_prompt_tokens"])
            main = [c.get(l, 0) for l in "abcde"[:4 if ds in FOUR else 5]]
            self.assertLessEqual(max(main) - min(main), 1, (ds, tag, c))

    def test_T2_raven_ids_reproduce_the_host_build(self):
        """Huginn, McLeish and Ouro-Thinking item ids under each model's own tokenizer (skipped where it is not
        cached); off the pin only items with a non-ASCII character may differ."""
        import transformers
        dss = {"huginn": ("aqua", "csqa"), "mcleish": ("aqua", "csqa", "arc"), "think14": ("aqua", "csqa", "arc"),
               "think26": ("aqua", "csqa", "arc")}
        for tag in dss:
            V = MD.TOK_VARIANTS[tag]
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(V["repo"], revision=V["revision"], local_files_only=True,
                                                    trust_remote_code=bool(V.get("trust")))
            except Exception:                                          # noqa: BLE001
                continue
            enc = lambda t, tok=tok: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
            bad = []
            for ds in dss[tag]:
                meta, head, built = R.load_items(DATA, ds, tag)
                R.check_tail_ids(enc, meta)
                bos = [meta["bos_id"]] if meta.get("bos_id") is not None else []
                tail = list(meta["suffix_ids"]) + list(meta["cue_ids"])
                for it in built:
                    A = bos + enc(head + it["input_A"] + meta["a_prefix"]) + tail
                    if R.ids_sha(A) != it["ids_A_sha256"] and not non_ascii(it, head):
                        bad.append(it["key"])
                    elif R.ids_sha(A) != it["ids_A_sha256"] and transformers.__version__ == R.TRANSFORMERS_PIN:
                        bad.append(it["key"])
            print("[T2] %s under transformers %s: items differing from the host build %s"
                  % (tag, transformers.__version__, bad[:5]))
            self.assertEqual(bad, [])

    def test_order_is_seeded_permutation(self):
        for ds in R.DATASETS:
            d = json.load(open(os.path.join(DATA, "items_%s.json" % ds)))
            self.assertEqual(sorted(it["order"] for it in d["items"]), list(range(len(d["items"]))))

    def tok_or_skip(self):
        try:
            import transformers
            from transformers import AutoTokenizer
            rp, rv = MD.REPOS["ouro_1_4b_base"]
            tok = AutoTokenizer.from_pretrained(rp, revision=rv, trust_remote_code=True, local_files_only=True)
        except Exception as e:                                         # noqa: BLE001
            self.skipTest("tokenizer not cached here: %s" % type(e).__name__)
        return tok, transformers.__version__

    def test_T2_ids_reproduce_the_host_build(self):
        """The item files were built on the GPU host under the pin (transformers 4.56.2). Under the pin every item's
        ids must reproduce; under another version (the laptop's 5.7) only items with a non-ASCII character may differ
        (the two tokenizer versions split such characters differently; aqua:37 is one of them), and every ASCII-only
        item must reproduce."""
        tok, ver = self.tok_or_skip()
        enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
        bad = []
        for ds in R.DATASETS:
            meta, head, built = R.load_items(DATA, ds)
            tail = list(meta["suffix_ids"]) + list(meta["cue_ids"])
            self.assertEqual(enc(meta["suffix_text"]), meta["suffix_ids"])
            self.assertEqual(enc(" ("), meta["cue_ids"])
            labels = meta.get("labels", {})
            for l in "abcd":
                self.assertEqual(enc(labels.get(l, l)), [meta["letter_ids"][l]])
            for it in built:
                h = head[it["head"]] if isinstance(head, dict) else head
                A = enc(h + it["input_A"] + meta["a_prefix"]) + tail
                B = enc(h + it["input_B"] + meta["a_prefix"]) + tail
                if (len(A), len(B)) != (it["T"], it["T"]) or R.ids_sha(A) != it["ids_A_sha256"] or                         R.ids_sha(B) != it["ids_B_sha256"]:
                    bad.append((it["key"], non_ascii(it, h)))
        print("[T2] transformers %s: %d items differ from the host build, %d of them ASCII-only; first: %s"
              % (ver, len(bad), sum(1 for _, na in bad if not na), [k for k, _ in bad[:5]]))
        if ver == R.TRANSFORMERS_PIN:
            self.assertEqual(bad, [])
        else:
            self.assertEqual([k for k, na in bad if not na], [])
            self.assertIn(KNOWN_LOCAL_DRIFT[0], [k for k, _ in bad])

    def test_T2_tokens_on_cached_tokenizer(self):
        tok, ver = self.tok_or_skip()
        enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
        for ds in R.DATASETS:
            meta, head, built = R.load_items(DATA, ds)
            hh = (lambda x: head[x["head"]]) if isinstance(head, dict) else (lambda x: head)
            for it0 in [x for x in built if not non_ascii(x, hh(x))][:3]:
                it = R.materialize(enc, meta, head, it0)
                b = it["block"]
                self.assertEqual(len(it["A"]), len(it["B"]))
                self.assertEqual(it["A"][:b[0]], it["B"][:b[0]])
                self.assertEqual(it["A"][b[-1] + 1:], it["B"][b[-1] + 1:])
                toks = tok.convert_ids_to_tokens(it["A"][b[0]:b[-1] + 1])
                if "labels" not in meta:                       # inline layout: the block starts at ' ('
                    self.assertEqual(toks[0], tok.convert_ids_to_tokens(enc(" ("))[0])
                self.assertEqual(tok.decode(it["A"][it["r"]:]), " (")
                self.assertEqual(tok.decode(it["A"][b[-1] + 1:b[-1] + 1 + len(enc(meta["a_prefix"]))]),
                                 meta["a_prefix"])
                print("[T2] %s block %s A %s | B %s" % (it["key"], [b[0], b[-1] + 1],
                                                       ascii(tok.decode(it["A"][b[0]:b[-1] + 1])),
                                                       ascii(tok.decode(it["B"][b[0]:b[-1] + 1]))))


# ------------------------------------------------------------------ one item on the fake
class TestItem(unittest.TestCase):
    def test_all_arms_and_identities(self):
        F = Fake(self)
        for it in fake_items(F.X, 3):
            rec = R.process_item(F.X, it, "ouro14", "t", gap_min=1e-6)
            self.assertIsNone(rec["drop"])
            self.assertEqual(R.identity_failures(rec), [], rec["identity"])
            self.assertEqual(rec["forwards"], 3 + 1 + len(R.ARMS))
            self.assertEqual(rec["arms"]["pc"]["nld"], 1.0)
            self.assertEqual(rec["arms"]["lt_L4"]["nld"], 1.0)
            self.assertEqual(rec["forwards"], 21)
            for a in ("lt_L4first", "lt_L4q1", "lt_L4mid", "lt_L4q3"):
                self.assertIsNotNone(rec["arms"][a]["nld"])
            self.assertEqual(rec["arms"]["blk_L4"]["nld"], 0.0)
            self.assertEqual(rec["arms"]["pc"]["writes"], "B")
            for a in ("rnd_lt_L1", "rnd_lt_L4", "rnd_blk_L1"):
                self.assertAlmostEqual(rec["arms"][a]["norm_ratio"], 1.0, places=4)
            self.assertEqual(rec["arms"]["swap"]["writes"], "B")
            self.assertEqual(rec["arms"]["clean_A"]["writes"], "A")
            self.assertEqual(R.ARM_NAMES[0], "lt_L1")

    def test_identity_detects_a_dropped_patch(self):
        F = Fake(self)
        run0 = F.X.run

        def run_drop(ids, row, ivs=(), recs=()):
            return run0(ids, row, ivs=[iv for iv in ivs if iv.id != "lt_L2_L2"], recs=recs)
        F.X.run = run_drop
        rec = R.process_item(F.X, fake_items(F.X, 1)[0], "ouro14", "t", gap_min=1e-6)
        self.assertIn("patch_written", R.identity_failures(rec))

    def test_identity_detects_a_dropped_mid_loop_patch(self):
        F = Fake(self)
        run0 = F.X.run

        def run_drop(ids, row, ivs=(), recs=()):
            return run0(ids, row, ivs=[iv for iv in ivs if iv.id not in ("lt_L4mid_L4", "lt_L4q3_L4")], recs=recs)
        F.X.run = run_drop
        rec = R.process_item(F.X, fake_items(F.X, 1)[0], "ouro14", "t", gap_min=1e-6)
        self.assertIn("patch_written", R.identity_failures(rec))

    def test_identity_detects_a_write_at_an_earlier_loop4_site(self):
        """T1 addition: a q3 patch that also wrote the q1 site (earlier in loop 4) fails before_patch_unchanged."""
        F = Fake(self)
        s = R.mid_sites(F.X.g)
        run0 = F.X.run

        def run_extra(ids, row, ivs=(), recs=()):
            ivs = list(ivs)
            if any(iv.id == "lt_L4q3_L4" for iv in ivs):
                pay = row.payload["lt_L4q3_L4"][s["Lq3"].key()]
                row.payload["extra"] = {s["Lq1"].key(): pay}
                ivs.append(F.X.I.from_spec({"id": "extra", "op": "patch", "site": s["Lq1"].key(), "a": 1.0,
                                            "target": {"kind": "donor"},
                                            "positions": {"kind": "abs", "abs": [len(ids[0]) - 1]}}, F.X.g))
            return run0(ids, row, ivs=ivs, recs=recs)
        F.X.run = run_extra
        rec = R.process_item(F.X, fake_items(F.X, 1)[0], "ouro14", "t", gap_min=1e-6)
        self.assertIn("before_patch_unchanged", R.identity_failures(rec))

    def test_mid_loop_sites_are_the_k15_map_sites(self):
        from prod.steer.sites import matrix_site
        F = Fake(self)
        g = F.X.g
        s0, sm = matrix_site(g, "L0@4"), matrix_site(g, "Lmid@4")
        self.assertEqual((s0.loop, s0.layer), (4, 0))
        self.assertEqual((sm.loop, sm.layer), (4, g.n_core // 2))
        ms = R.mid_sites(g)
        self.assertEqual(ms["L0"], s0)
        self.assertEqual(ms["Lmid"], sm)
        self.assertEqual([(ms[l].loop, ms[l].layer) for l in R.MID_LABELS],
                         [(4, 0), (4, g.n_core // 4), (4, g.n_core // 2), (4, 3 * g.n_core // 4)])
        self.assertEqual(R.mid_layers(24), {"L0": 0, "Lq1": 6, "Lmid": 12, "Lq3": 18})
        self.assertEqual(R.mid_layers(48), {"L0": 0, "Lq1": 12, "Lmid": 24, "Lq3": 36})
        self.assertEqual([a[0] for a in R.ARMS if a[4] != "end"], ["lt_L4first", "lt_L4q1", "lt_L4mid", "lt_L4q3"])
        self.assertEqual(len(R.ARMS), 17)

    def test_screen_and_gap_drops(self):
        F = Fake(self)
        it = fake_items(F.X, 1)[0]
        bad = dict(it, tA=(it["tA"] + 1) % 64 or 3)
        rec = R.process_item(F.X, bad, "ouro14", "t")
        self.assertEqual(rec["drop"], "screen_A")
        self.assertEqual(rec["forwards"], 3)
        self.assertEqual(R.identity_failures(rec), [])
        rec = R.process_item(F.X, it, "ouro14", "t", gap_min=1e9)
        self.assertEqual(rec["drop"], "gap_below_2")

    def test_random_rows_seeded_unit(self):
        u1 = R.rnd_rows("ouro14", "aqua", "k", "rnd_blk_L1", 5, 32)
        u2 = R.rnd_rows("ouro14", "aqua", "k", "rnd_blk_L1", 5, 32)
        self.assertTrue(torch.equal(u1, u2))
        self.assertTrue(torch.allclose(u1.norm(dim=1), torch.ones(5)))
        self.assertFalse(torch.equal(u1, R.rnd_rows("ouro14", "aqua", "k", "rnd_lt_L1", 5, 32)))

    def test_random_payload_built_in_model_dtype(self):
        """RA-28 lesson: with a bf16 dtype the written payload is the bf16-rounded vector, and the identity holds."""
        F = Fake(self)
        it = fake_items(F.X, 1)[0]
        rec = R.process_item(F.X, it, "ouro14", "t", gap_min=1e-6, dtype=torch.bfloat16)
        self.assertTrue(rec["identity"]["patch_written"], rec["identity"])
        self.assertAlmostEqual(rec["arms"]["rnd_lt_L1"]["norm_ratio"], 1.0, places=1)


# ------------------------------------------------------------------ statistics and outputs
def srec(i, D=4.0, nld=None):
    arms = {a: {"m": 0.0, "dm": 0.0, "nld": (nld or {}).get(a, 0.1), "top1": 5, "writes": "A", "norm_ratio": 1.0}
            for a in R.ARM_NAMES}
    arms["swap"] = {"m": 1.0, "dm": D, "nld": 1.0, "top1": 6, "writes": "B"}
    arms["clean_A"] = {"m": 0.0, "dm": 0.0, "nld": 0.0, "top1": 5, "writes": "A"}
    return {"key": "k%d" % i, "model": "ouro14", "dataset": "t", "lA": "a", "lB": "abcde"[i % 5], "drop": None,
            "D": D, "arms": arms, "seconds": 2.0, "seconds_arm": {a: 0.1 for a in R.ARM_NAMES},
            "identity": {k: True for k in R.GATED}}


class TestStats(unittest.TestCase):
    def test_outputs(self):
        d = tmpdir(self)
        recs = [srec(i, nld={"blk_L1": 0.9}) for i in range(10)] + [dict(srec(99), drop="screen_A")]
        res, chk = R.write_outputs(d, recs, 50, {"model": "ouro14", "dataset": "t"})
        self.assertAlmostEqual(res["blk_L1"]["mean"], 0.9, places=4)
        self.assertEqual(res["swap"]["mean"], 1.0)
        self.assertEqual(chk["counts"]["n_kept"], 10)
        self.assertEqual(chk["counts"]["drops"], {"screen_A": 1})
        self.assertAlmostEqual(chk["timing"]["full_card_hours_at_this_rate"], 2.0 * 50 / 3600, places=3)
        rows = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
        self.assertEqual(len(rows), 10 * (len(R.ARM_NAMES) + 2))
        self.assertEqual(rows[0]["arm"], "lt_L1")
        rj = json.load(open(os.path.join(d, "results.json")))
        self.assertEqual(sorted(rj["nld_by_gold_letter_B"]), list("abcde"))
        self.assertTrue(rj["predictions"]["P2_blk_L1_ge_0.8"])
        self.assertFalse(rj["predictions"]["P1_holds"])
        s = open(os.path.join(d, "summary.md")).read()
        self.assertLess(s.index("| lt_L1 |"), s.index("| blk_L1 |"))
        self.assertIn("| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |", s)


# ------------------------------------------------------------------ run() end to end on the fake
def fake_enc(t):
    return [3 + (ord(c) % 60) for c in t]


def make_fake_data(d, n=8, ds="aqua", tag=None, bos=None):
    head, a_prefix = "Q: ", "\nA:"
    enc_p = (lambda t: [bos] + fake_enc(t)) if bos is not None else fake_enc
    tail = fake_enc("\nThe answer is") + fake_enc(" (")
    g = random.Random(0)
    counts, items = {}, []
    for i in range(n):
        texts = ["%s%s" % (chr(97 + g.randrange(26)), chr(97 + g.randrange(26))) + str(j) for j in range(5)]
        row = {"idx": i, "input": MD.build_input("q%d?" % i, texts), "target": "ABCDE"[i % 5],
               "options": list("ABCDE")}
        it = MD.build_item(ds, row, head, a_prefix, tail, enc_p, None, counts)
        it["order"] = i
        items.append(it)
    meta = {"task": ds, "n_rows": n, "n_built": n, "drops": {}, "a_prefix": a_prefix,
            "suffix_text": "\nThe answer is", "suffix_ids": fake_enc("\nThe answer is"), "cue_ids": fake_enc(" ("),
            "letter_ids": {l: fake_enc(l)[0] for l in "abcdef"},
            "head_sha256": hashlib.sha256(head.encode("utf-8")).hexdigest()}
    if bos is not None:
        meta["bos_id"] = bos
    os.makedirs(d, exist_ok=True)
    iname = R.items_file(ds, tag)
    json.dump({"meta": meta, "items": items}, open(os.path.join(d, iname), "w"))
    open(os.path.join(d, "prefix_%s.txt" % ds), "w", newline="").write(head)
    json.dump({"outputs_sha256": {f: KF.sha256_file(os.path.join(d, f))
                                  for f in (iname, "prefix_%s.txt" % ds)}},
              open(os.path.join(d, "sources.json"), "w"))
    return items


class TestE2E(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir(self)
        self.data = os.path.join(self.d, "data")
        make_fake_data(self.data, n=16)
        self.plan = os.path.join(self.d, "plan.md")
        open(self.plan, "w").write("plan\n")
        self.sha = KF.sha256_file(self.plan)
        self.F = Fake(self)

    def loader(self, model, tree, mem):
        return self.F.X, fake_enc, {"fake": True}

    def proc(self, X, it, model, ds):
        """Fake-model seam: B's block tokens are redrawn (seeded by the key) so the toy model's top-1 moves, and the
        letters are the toy model's own top-1 tokens."""
        g = random.Random(it["key"])
        B = list(it["A"])
        for p in it["block"]:
            B[p] = g.randrange(3, 64)
        it = dict(it, B=B)
        zA = X.nohook(torch.tensor([it["A"]]))[0, -1]
        zB = X.nohook(torch.tensor([it["B"]]))[0, -1]
        it = dict(it, tA=int(zA.argmax()), tB=int(zB.argmax()))
        it["letter_ids"] = sorted({it["tA"], it["tB"]})
        return R.process_item(X, it, model, ds, gap_min=1e-6)

    def args(self, *extra):
        return R.parse_args(["--run", "--model", "ouro14", "--dataset", "aqua", "--plan", self.plan,
                             "--plan_sha256", self.sha, "--data", self.data] + list(extra))

    def test_pilot_full_summarize(self):
        pil = os.path.join(self.d, "pilot")
        R.run(self.args("--out", pil, "--pilot"), loader=self.loader, process=self.proc)
        g = json.load(open(os.path.join(pil, "debug_gate.json")))
        self.assertTrue(g["pass"], g)
        for f in ("results.csv", "results.json", "checks.json", "summary.md", "config.json", "records.jsonl"):
            self.assertTrue(os.path.exists(os.path.join(pil, f)), f)
        chk = json.load(open(os.path.join(pil, "checks.json")))
        self.assertEqual(len(chk["block_maps"]), 3)
        full = os.path.join(self.d, "full")
        R.run(self.args("--out", full, "--debug_gate", os.path.join(pil, "debug_gate.json")), loader=self.loader,
              process=self.proc)
        n1 = len(R.load_jsonl(os.path.join(full, "records.jsonl")))
        R.run(self.args("--out", full, "--debug_gate", os.path.join(pil, "debug_gate.json")), loader=self.loader,
              process=self.proc)                                        # resume: nothing new
        self.assertEqual(len(R.load_jsonl(os.path.join(full, "records.jsonl"))), n1)
        a = [json.loads(x) for x in open(os.path.join(pil, "records.jsonl"))]
        b = {r["key"]: r for r in R.load_jsonl(os.path.join(full, "records.jsonl"))}
        for r in a:                                                     # pilot rows reproduce in the full run
            self.assertEqual(r["arms"], b[r["key"]]["arms"])
        R.summarize(R.parse_args(["--summarize", "--out", full, "--data", self.data]))

    def test_plan_mismatch_exit_2(self):
        with self.assertRaises(SystemExit) as cm:
            R.run(R.parse_args(["--run", "--model", "ouro14", "--dataset", "aqua", "--plan", self.plan,
                                "--plan_sha256", "0" * 64, "--data", self.data, "--out",
                                os.path.join(self.d, "o"), "--pilot"]), loader=self.loader, process=self.proc)
        self.assertEqual(cm.exception.code, 2)

    def test_full_without_pilot_gate_exit_3(self):
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", os.path.join(self.d, "o"), "--debug_gate", os.path.join(self.d, "none.json")),
                  loader=self.loader, process=self.proc)
        self.assertEqual(cm.exception.code, 3)

    def test_tokenizer_drift_exit_2(self):
        def loader(model, tree, mem):
            return self.F.X, (lambda t: fake_enc(t)[:-1]), {}
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", os.path.join(self.d, "o"), "--pilot"), loader=loader, process=self.proc)
        self.assertEqual(cm.exception.code, 2)

    def test_tail_ids_mismatch_exit_2_before_any_item(self):
        def loader(model, tree, mem):
            return self.F.X, (lambda t: fake_enc(t) if t != " (" else [7]), {}
        out = os.path.join(self.d, "o")
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", out, "--pilot"), loader=loader, process=self.proc)
        self.assertEqual(cm.exception.code, 2)
        self.assertEqual(R.load_jsonl(os.path.join(out, "records.jsonl")), [])

    def bad_after_gate(self):
        """A process that fails pc_exact on scored items after the gate's 3 and the first 2 scored rows."""
        n = {"kept": 0}

        def proc_bad(X, it, model, ds):
            rec = self.proc(X, it, model, ds)
            if rec["drop"] is None:
                n["kept"] += 1
                if n["kept"] > R.GATE_N + 2:
                    rec["identity"]["pc_exact"] = False
            return rec
        return proc_bad

    def test_rerun_after_exit_4_exits_4_again(self):
        proc_bad = self.bad_after_gate()
        out = os.path.join(self.d, "o")
        for p in (proc_bad, self.proc):
            with self.assertRaises(SystemExit) as cm:
                R.run(self.args("--out", out, "--pilot"), loader=self.loader, process=p)
            self.assertEqual(cm.exception.code, 4)

    def test_full_run_exits_4_on_failed_records(self):
        pil = os.path.join(self.d, "pilot")
        R.run(self.args("--out", pil, "--pilot"), loader=self.loader, process=self.proc)
        recs = R.load_jsonl(os.path.join(pil, "records.jsonl"))
        bad = [r for r in recs if r["drop"] is None][0]
        bad["identity"]["pc_exact"] = False
        self.assertIn("pc_exact", R.identity_failures(bad))
        full = os.path.join(self.d, "full")
        os.makedirs(full)
        with open(os.path.join(full, "records.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", full, "--debug_gate", os.path.join(pil, "debug_gate.json")),
                  loader=self.loader, process=self.proc)
        self.assertEqual(cm.exception.code, 4)

    def test_trailing_partial_line_is_skipped_and_repaired(self):
        pil = os.path.join(self.d, "pilot")
        R.run(self.args("--out", pil, "--pilot"), loader=self.loader, process=self.proc)
        rp = os.path.join(pil, "records.jsonl")
        n = len(R.load_jsonl(rp))
        with open(rp, "a") as f:
            f.write('{"key": "cut')
        self.assertEqual(len(R.load_jsonl(rp, repair=True)), n)
        self.assertEqual(len(open(rp).read().splitlines()), n)

    def test_identity_failure_exit_4(self):
        out = os.path.join(self.d, "o")
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", out, "--pilot"), loader=self.loader, process=self.bad_after_gate())
        self.assertEqual(cm.exception.code, 4)
        self.assertTrue(json.load(open(os.path.join(out, "gate.json")))["pass"])

    def test_identity_gate_failure_exit_3_before_any_scored_row(self):
        def proc(X, it, model, ds):
            rec = self.proc(X, it, model, ds)
            if rec["drop"] is None:
                rec["identity"]["lt4_equals_pc"] = False
            return rec
        out = os.path.join(self.d, "o")
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", out, "--pilot"), loader=self.loader, process=proc)
        self.assertEqual(cm.exception.code, 3)
        self.assertFalse(os.path.exists(os.path.join(out, "records.jsonl")))

    def test_full_run_accepts_the_pilot_of_plan_960f88cd(self):
        pil = os.path.join(self.d, "pilot")
        R.run(self.args("--out", pil, "--pilot"), loader=self.loader, process=self.proc)
        gp = os.path.join(pil, "debug_gate.json")
        g = json.load(open(gp))
        g["plan_sha256"] = R.ACCEPTED_PILOT_PLANS[0]
        json.dump(g, open(gp, "w"))
        R.run(self.args("--out", os.path.join(self.d, "full"), "--debug_gate", gp), loader=self.loader,
              process=self.proc)
        g["plan_sha256"] = "1" * 64
        json.dump(g, open(gp, "w"))
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", os.path.join(self.d, "full2"), "--debug_gate", gp), loader=self.loader,
                  process=self.proc)
        self.assertEqual(cm.exception.code, 3)


# ------------------------------------------------------------------ folder hygiene
# ------------------------------------------------------------------ raven family (Huginn, McLeish) on the CPU fake
class RavenFake(object):
    """The prod CPU fake of the raven forward (huginn or mcleish variant) with a 4-layer core and 128 positions, bound
    and wrapped by ra29.raven_ctx (RA-25's lib/k17_freeze_raven.Ctx), K = 32."""

    def __init__(self, test, variant="huginn"):
        old = PF.RavenConfig

        class C4(old):
            def __init__(self, v):
                old.__init__(self, v)
                self.n_layers_in_recurrent_block = 4
                self.block_size = 128
        PF.RavenConfig = C4
        try:
            fake = PF.make_fake_raven(variant)
        finally:
            PF.RavenConfig = old
        ad = PF.real_raven_adapter_over(fake, R.K_RAVEN)
        ad.max_positions = 128 if variant == "huginn" else None
        cm = RavenBinding.trust_for_tests(type(fake.transformer.prelude[0]), label=variant + ": fake")
        cm.__enter__()
        test.addCleanup(cm.__exit__, None, None, None)
        cpu = PF.raven_cpu()
        cpu.__enter__()
        test.addCleanup(cpu.__exit__, None, None, None)
        b = bind(ad)
        b.set_init_state("fixed_table", 128)
        self.X = R.raven_ctx(ad, b, TREE)


def raven_items(X, n, T=24, b0=6, b1=14):
    out, s = [], 0
    while len(out) < n and s < 200:
        g = random.Random(s)
        A = [g.randrange(3, 64) for _ in range(T)]
        B = list(A)
        for p in range(b0, b1):
            B[p] = g.randrange(3, 64)
        key = "t:%d" % s
        zA = X.nohook(torch.tensor([A]), key)[0, -1]
        zB = X.nohook(torch.tensor([B]), key)[0, -1]
        s += 1
        if int(zA.argmax()) != int(zB.argmax()):
            out.append({"key": key, "A": A, "B": B, "r": T - 1, "block": list(range(b0, b1)),
                        "tA": int(zA.argmax()), "tB": int(zB.argmax()), "lA": "a", "lB": "b",
                        "letter_ids": sorted({int(zA.argmax()), int(zB.argmax())})})
    return out


class TestRaven(unittest.TestCase):
    def test_arms_and_identities_both_variants(self):
        for variant in ("huginn", "mcleish"):
            F = RavenFake(self, variant)
            items = raven_items(F.X, 2)
            self.assertEqual(len(items), 2, variant)
            for it in items:
                rec = R.process_item(F.X, it, variant, "t", gap_min=1e-6)
                self.assertIsNone(rec["drop"])
                self.assertEqual(rec["family"], "raven")
                self.assertEqual(R.identity_failures(rec), [], (variant, rec["identity"]))
                self.assertEqual(rec["forwards"], 3 + 1 + len(R.ARMS_RAVEN))
                self.assertEqual(rec["forwards"], 21)
                self.assertEqual(rec["arms"]["pc"]["nld"], 1.0)
                self.assertEqual(rec["arms"]["sz_fin_blk"]["nld"], 0.0)
                self.assertTrue(rec["identity"]["blk_e_block_equals_B"])
                self.assertIn("blk_L32", rec["arms"])
                # contraction ratios (review B2(b)): the block holds B's loop-32 state exactly under blk_e,
                # blk_persist and blk_L32; the coda patch leaves every loop state as in A
                for a in ("blk_e", "blk_persist", "blk_L32"):
                    self.assertEqual(rec["arms"][a]["ratio_blk32"], 1.0, a)
                self.assertEqual(rec["arms"]["pc"]["ratio_blk32"], 0.0)
                self.assertEqual(rec["arms"]["pc"]["ratio_r32"], 0.0)
                for a in R.ARMS_RAVEN:
                    self.assertIn("ratio_r32", rec["arms"][a[0]])
                for a in ("rnd_lt_L1", "rnd_lt_L32", "rnd_blk_L1"):
                    self.assertAlmostEqual(rec["arms"][a]["norm_ratio"], 1.0, places=4)
                self.assertNotIn("clean_loop_margin_A", rec)

    def test_blk_e_identity_detects_a_dropped_prelude_patch(self):
        F = RavenFake(self, "huginn")
        it = raven_items(F.X, 1)[0]
        run0 = F.X.run

        def run_drop(ids, row, ivs=(), recs=()):
            return run0(ids, row, ivs=[iv for iv in ivs if iv.id != "blk_e_L1"], recs=recs)
        F.X.run = run_drop
        rec = R.process_item(F.X, it, "huginn", "t", gap_min=1e-6)
        bad = R.identity_failures(rec)
        self.assertIn("blk_e_block_equals_B", bad)
        self.assertIn("patch_written", bad)
        self.assertEqual([a[0] for a in R.ARMS_RAVEN].index("blk_L32"),
                         [a[0] for a in R.ARMS_RAVEN].index("blk_L31") + 1)

    def test_end_of_loop_32_is_not_the_readout_input(self):
        """The coda reads the block positions, so the loop-32 end patch at r is a result, not the positive control."""
        F = RavenFake(self, "huginn")
        it = raven_items(F.X, 1)[0]
        rec = R.process_item(F.X, it, "huginn", "t", gap_min=1e-6)
        self.assertNotEqual(rec["arms"]["lt_L32"]["nld"], 1.0)

    def test_sites_are_the_k15_map_sites(self):
        from prod.steer.sites import matrix_site
        F = RavenFake(self, "huginn")
        ms = R.mid_sites(F.X.g, "raven")
        self.assertEqual(ms["L0"], matrix_site(F.X.g, "L0@32"))
        self.assertEqual(ms["Lmid"], matrix_site(F.X.g, "Lmid@32"))
        self.assertEqual((ms["Lmid"].loop, ms["Lmid"].layer), (32, F.X.g.n_core // 2))
        self.assertEqual([a[0] for a in R.ARMS_RAVEN][0], "lt_L1")
        self.assertEqual(R.ARMS_RAVEN[[a[0] for a in R.ARMS_RAVEN].index("blk_persist")][2], tuple(range(1, 33)))

    def test_dropped_patch_and_earlier_site_write_are_detected(self):
        F = RavenFake(self, "mcleish")
        it = raven_items(F.X, 1)[0]
        run0 = F.X.run

        def run_drop(ids, row, ivs=(), recs=()):
            return run0(ids, row, ivs=[iv for iv in ivs if iv.id != "blk_persist_L17"], recs=recs)
        F.X.run = run_drop
        rec = R.process_item(F.X, it, "mcleish", "t", gap_min=1e-6)
        self.assertIn("patch_written", R.identity_failures(rec))
        s = R.mid_sites(F.X.g, "raven")

        def run_extra(ids, row, ivs=(), recs=()):
            ivs = list(ivs)
            if any(iv.id == "lt_L32mid_L32" for iv in ivs):
                pay = row.payload["lt_L32mid_L32"][s["Lmid"].key()]
                row.payload["extra"] = {s["L0"].key(): pay}
                ivs.append(F.X.I.from_spec({"id": "extra", "op": "patch", "site": s["L0"].key(), "a": 1.0,
                                            "target": {"kind": "donor"},
                                            "positions": {"kind": "abs", "abs": [len(ids[0]) - 1]}}, F.X.g))
            return run0(ids, row, ivs=ivs, recs=recs)
        F.X.run = run_extra
        rec = R.process_item(F.X, it, "mcleish", "t", gap_min=1e-6)
        self.assertIn("before_patch_unchanged", R.identity_failures(rec))

    def test_raven_predictions_and_summary(self):
        def make(n):
            recs = []
            for i in range(n):
                arms = {a[0]: {"m": 0.0, "dm": 0.0, "nld": 0.02, "top1": 5, "writes": "A", "norm_ratio": 1.0,
                               "ratio_blk32": 0.01, "ratio_r32": 0.02} for a in R.ARMS_RAVEN}
                for a, v in (("blk_persist", 0.7), ("blk_L32", 0.6), ("blk_e", 0.95), ("lt_L32", 0.8), ("pc", 1.0)):
                    arms[a]["nld"] = v
                arms["swap"] = {"m": 1.0, "dm": 4.0, "nld": 1.0, "top1": 6, "writes": "B"}
                arms["clean_A"] = {"m": 0.0, "dm": 0.0, "nld": 0.0, "top1": 5, "writes": "A"}
                recs.append({"key": "k%d" % i, "model": "huginn", "dataset": "t", "lA": "a", "lB": "abcde"[i % 5],
                             "drop": None, "family": "raven", "D": 4.0, "arms": arms, "seconds": 9.0,
                             "seconds_arm": {a[0]: 0.5 for a in R.ARMS_RAVEN},
                             "identity": {k: True for k in R.GATED_RAVEN}})
            return recs
        d0 = tmpdir(self)
        R.write_outputs(d0, make(10), 100, {"model": "huginn", "dataset": "t"})
        pv0 = json.load(open(os.path.join(d0, "results.json")))["predictions"]
        self.assertEqual(list(pv0), ["not_scored"])                       # review N1: under 80 scored items
        d = tmpdir(self)
        recs = make(80)
        res, chk = R.write_outputs(d, recs, 100, {"model": "huginn", "dataset": "t"})
        pv = json.load(open(os.path.join(d, "results.json")))["predictions"]
        self.assertTrue(pv["PR1_holds"] and pv["PR2_holds"], pv)
        self.assertTrue(pv["PR1_single_loop_block_ci_inside_0.1"])
        self.assertAlmostEqual(pv["blk_persist_minus_blk_L32"]["mean"], 0.1, places=4)
        self.assertEqual(pv["contraction_ratio_blk32"]["blk_L1"], 0.01)
        rows = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
        self.assertEqual(len(rows), 80 * (len(R.ARMS_RAVEN) + 2))
        s = open(os.path.join(d, "summary.md")).read()
        self.assertIn("| B's letter | n | lt_L1 | blk_L1 | blk_persist | lt_L31 | lt_L32mid | lt_L32 |", s)
        self.assertLess(s.index("| lt_L1 |"), s.index("| blk_L1 |"))
        self.assertIn("refresh-model case", s)


class TestRavenE2E(unittest.TestCase):
    def test_pilot_full_on_the_fake_with_bos(self):
        d = tmpdir(self)
        data = os.path.join(d, "data")
        make_fake_data(data, n=32, tag="mcleish", bos=2)
        plan = os.path.join(d, "plan.md")
        open(plan, "w").write("plan\n")
        sha = KF.sha256_file(plan)
        F = RavenFake(self, "mcleish")

        def loader(model, tree, mem):
            return F.X, fake_enc, {"fake": True}

        def proc(X, it, model, ds):
            self.assertEqual(it["A"][0], 2)                        # the family's BOS leads every prompt
            g = random.Random(it["key"])                           # toy-model seam: random tokens after the BOS
            A = [2] + [g.randrange(3, 64) for _ in it["A"][1:]]
            B = list(A)
            for p_ in it["block"]:
                B[p_] = g.randrange(3, 64)
            it = dict(it, A=A, B=B)
            zA = X.nohook(torch.tensor([A]), it["key"])[0, -1]
            zB = X.nohook(torch.tensor([B]), it["key"])[0, -1]
            it.update(tA=int(zA.argmax()), tB=int(zB.argmax()))
            it["letter_ids"] = sorted({it["tA"], it["tB"]})
            return R.process_item(X, it, model, ds, gap_min=1e-6)

        def args(*extra):
            return R.parse_args(["--run", "--model", "mcleish", "--dataset", "aqua", "--plan", plan,
                                 "--plan_sha256", sha, "--data", data] + list(extra))
        pil = os.path.join(d, "pilot")
        R.run(args("--out", pil, "--pilot"), loader=loader, process=proc)
        self.assertTrue(json.load(open(os.path.join(pil, "debug_gate.json")))["pass"])
        R.run(args("--out", os.path.join(d, "full"), "--debug_gate", os.path.join(pil, "debug_gate.json")),
              loader=loader, process=proc)
        chk = json.load(open(os.path.join(d, "full", "checks.json")))
        self.assertEqual(chk["identity_failures"], {})
        cfg = json.load(open(os.path.join(d, "full", "config.json")))
        self.assertEqual(cfg["arms"][0], "lt_L1")
        self.assertIn("blk_persist", cfg["arms"])


class TestFolder(unittest.TestCase):
    def test_plan_hash_file_matches(self):
        line = open(os.path.join(HERE, "plan_ra29.sha256")).read().split()
        self.assertEqual(line[1], "plan_ra29.md")
        self.assertEqual(line[0], KF.sha256_file(R.PLAN_DEFAULT))

    def test_p1_text_in_plan(self):
        txt = " ".join(open(R.PLAN_DEFAULT, encoding="utf-8").read().split())
        self.assertIn("the last-token patch lands before the last loop ends: at the end of loop 3 or the middle of "
                      "loop 4 it gives NLD at least 0.5, where facts give at most 0.17 at the same sites, and at the "
                      "end of loop 1 it gives below 0.2", txt)

    def test_no_host_user_home_strings(self):
        words = ["sas" + "11", "C:" + r"\\" + "Users", "C:" + "/Users", "/ho" + "me/", "/Us" + "ers/",
                 "E:" + r"\\" + "Research", "E:" + "/Research", "rtx" + "6000", "dg" + "x1"]
        pat = re.compile("|".join(words), re.IGNORECASE)
        bad = []
        for root, dirs, files in os.walk(HERE):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "out")]
            for f in files:
                if f.endswith((".py", ".md", ".json", ".txt", ".sh")):
                    p = os.path.join(root, f)
                    if pat.search(open(p, encoding="utf-8", errors="ignore").read()):
                        bad.append(os.path.relpath(p, HERE))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
