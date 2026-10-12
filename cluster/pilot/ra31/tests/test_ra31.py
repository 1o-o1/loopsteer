"""CPU tests of RA-31 Part A (no GPU, no checkpoint, no download).

Covered: the write decomposition (contributions of a partition sum to the post-norm write); the edit arithmetic on
bf16 (alpha 1 returns w bit-exact); the unit hooks idle bit-exact and their guards; the set builders (scores over
loops 2 to 4 only, stable top-k, seeded random sets of equal count); the edit tables; process_item on the prod CPU
fake of Ouro in float32 and in bf16: every identity of plan section 6 (hook and zero patch, hold without patch,
positive control, structural zero, donor and random loop-1 states, alpha 1 equals the patched run on every set and
branch, set counts), the first-site scale of the whole-write arm, alpha 0 of the whole-write arm equals the prod
hold rows of k17_freeze, the decomposed edit on every unit reduces to the whole write (and a wrong clean value breaks
it); the RUN 1c reference data and its means; the curve cells; no host, user or home string.

    cd ra31 && python -m unittest discover -s tests -v
"""
import json
import os
import re
import sys
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

import ra31 as R  # noqa: E402
import ra31_hooks as H31  # noqa: E402
import k17_freeze as KF  # noqa: E402
import prod_fakes as PF  # noqa: E402

if PF.ROOT not in (HERE, LIB, TREE, TESTS) and PF.ROOT in sys.path:
    sys.path.remove(PF.ROOT)

from prod.steer.adapters import bind  # noqa: E402
from prod.steer.adapters.ouro import OuroBinding  # noqa: E402


class Fake(object):
    """KF.Ctx and the RA-31 Runner over the prod CPU fake of Ouro (L layers, d 16, 2 heads of 8, I 32)."""

    def __init__(self, test, L=4, dtype=torch.float32):
        fake = PF.make_fake_ouro("eager", L=L, total_ut_steps=KF.K)
        fake = fake.to(dtype)
        ad = PF.real_ouro_adapter_over(fake, KF.K)
        cm = OuroBinding.trust_for_tests(PF.OuroDecoderLayer)
        cm.__enter__()
        test.addCleanup(cm.__exit__, None, None, None)
        self.b = bind(ad)
        self.X = KF.Ctx(self.b, TREE)
        self.Rn = R.Runner(self.X)


def fake_items(X, n, seed=0, ds="D1"):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        A = [int(x) for x in torch.randint(3, 60, (9,), generator=g)]
        B = list(A)
        B[2] = (B[2] + 7) % 57 + 3
        B[3] = (B[3] + 11) % 57 + 3
        zA = X.nohook(torch.tensor([A]))[0, -1]
        zB = X.nohook(torch.tensor([B]))[0, -1]
        d = zB - zA
        out.append({"key": "%s|k%d" % (ds, i), "A": A, "B": B, "pos": [2, 3], "tA": int(d.argmin()),
                    "tB": int(d.argmax()), "cluster": "c%d" % (i // 2), "sa": "s%d" % (i // 2),
                    "sb": "s%d" % (i // 2 + 7)})
    return out


# ------------------------------------------------------------------ arithmetic and sets (synthetic)
class TestArithmetic(unittest.TestCase):
    def test_decomposition_sums_to_the_write(self):
        """c_S over a partition of the units sums to the post-norm write g * u / rms(u) (float64 reference)."""
        g = torch.Generator().manual_seed(0)
        W = torch.randn(16, 40, generator=g, dtype=torch.float64)
        x = torch.randn(40, generator=g, dtype=torch.float64)
        gw = torch.randn(16, generator=g, dtype=torch.float64)
        u = W @ x
        w = gw * u / torch.sqrt(u.pow(2).mean() + 1e-6)
        parts = [torch.arange(0, 7), torch.arange(7, 30), torch.arange(30, 40)]
        tot = sum(H31.contrib(x, W, p, u, gw, 1e-6).double() for p in parts)
        self.assertLess(float((tot - w).abs().max()), 1e-5)

    def test_edit_row_alpha1_exact_bf16(self):
        g = torch.Generator().manual_seed(1)
        for _ in range(20):
            scale = torch.exp(torch.randn(64, generator=g) * 6)
            w = (torch.randn(64, generator=g) * scale).to(torch.bfloat16)
            c1, c0 = torch.randn(64, generator=g) * scale, torch.randn(64, generator=g) * scale
            self.assertTrue(torch.equal(H31.edit_row(w, c1, c0, 1.0), w))
            want = (w.float() - (c1 - c0)).to(torch.bfloat16)
            self.assertTrue(torch.equal(H31.edit_row(w, c1, c0, 0.0), want))

    def test_head_cols(self):
        self.assertEqual(H31.head_cols([3, 1], 4).tolist(), [4, 5, 6, 7, 12, 13, 14, 15])


class TestSets(unittest.TestCase):
    def test_scores_use_loops_2_to_4_only(self):
        P = torch.zeros(4, 3, 10)
        C = torch.zeros(4, 3, 10)
        P[0, 0, 0] = 100.0                         # loop 1 difference: ignored
        P[1, 1, 2] = 1.0
        P[3, 1, 2] = -2.0
        s = R.neuron_scores(P, C)
        self.assertEqual(float(s[0, 0]), 0.0)
        self.assertEqual(float(s[1, 2]), 3.0)
        zP = torch.zeros(4, 3, 8)
        zP[2, 2, 4:8] = 1.0                        # head 1 of layer 2 (hd 4), loop 3
        hs = R.head_scores(zP, torch.zeros(4, 3, 8), 2, 4)
        self.assertEqual(tuple(hs.shape), (3, 2))
        self.assertAlmostEqual(float(hs[2, 1]), 2.0)

    def test_top_units_stable_ties(self):
        s = torch.tensor([[1.0, 3.0, 3.0], [3.0, 0.0, 2.0]])
        self.assertEqual(R.top_units(s, 2), [1, 2])
        self.assertEqual(R.top_units(s, 3), [1, 2, 3])

    def test_random_sets_seeded_equal_count(self):
        a = R.rand_units("ra31:neurons:D1:k0", 1000, 50)
        self.assertEqual(a, R.rand_units("ra31:neurons:D1:k0", 1000, 50))
        self.assertNotEqual(a, R.rand_units("ra31:neurons:D1:k1", 1000, 50))
        self.assertEqual(len(set(a)), 50)
        self.assertEqual(R.n_top_neurons(24, 5632), 6758)
        self.assertEqual(R.n_top_neurons(48, 5632), 13517)

    def test_arm_list(self):
        arms = R.arm_list()
        self.assertEqual(len(arms), 72)
        self.assertEqual(sum(1 for a in arms if a[0] == "donor"), 60)
        self.assertEqual(sorted(a for a in arms if a[2] in R.SIZE_SETS),
                         sorted(("donor", "mlp", s_, a) for s_ in R.SIZE_SETS for a in (0.0, 2.0)))
        self.assertEqual(R.forwards_equiv(arms), 10 + 72 + R.GRAD_EQUIV)
        self.assertEqual(len(set(R.arm_name(*a) for a in arms)), 72)
        self.assertEqual(R.arm_name("donor", "mlp", "top", 0.25), "donor_mlp_top_a0.25")
        self.assertEqual(R.arm_name("rnd", "attn", "all", 2.0), "rnd_attn_all_a2")


# ------------------------------------------------------------------ hooks and items on the prod fake
class TestFake(unittest.TestCase):
    def test_units_idle_bit_exact_and_guards(self):
        F = Fake(self)
        X, Rn = F.X, F.Rn
        it = fake_items(X, 1)[0]
        ids = torch.tensor([it["A"]])
        row = X.RowInfo(key="k", prompt_len=9, spans={"answer": (8,)}, payload={})
        z0, _, _ = Rn.run(ids, row)
        z1, _, took = Rn.run(ids, row, spec=H31.Spec31(8, record=True, T=9))
        self.assertTrue(torch.equal(z0, z1))
        self.assertEqual(tuple(took["act"].shape), (4, 4, 32))
        self.assertEqual(tuple(took["z"].shape), (4, 4, 16))
        with self.assertRaises(RuntimeError):
            Rn.run(ids, row, spec=H31.Spec31(8, record=True, T=7))
        with self.assertRaises(ValueError):
            Rn.U.install(H31.Spec31(8, alpha=0.5).__class__.__bases__[0](8, scale_act={(2, 0): torch.tensor([0])}))
        Rn.U.remove()

    def _check_item(self, F, it, ds):
        rec, ident, info = R.process_item(F.Rn, it, ds)
        self.assertTrue(all(ident.values()), ident)
        self.assertTrue(info["first_site_scale"], {k: v.get("first_site_ratio") for k, v in rec["rows"].items()
                                                   if "first_site_ratio" in v})
        self.assertTrue(info["sets_act"])
        self.assertEqual(rec["forwards"], 10 + 72 + 1)                  # + the attribution pass
        self.assertTrue(rec["attr"]["grad_pass_logits_equal_E0"])
        self.assertAlmostEqual(rec["rows"]["rnd"]["norm_ratio"], 1.0, places=2)
        return rec

    def test_process_item_float32(self):
        F = Fake(self)
        for it in fake_items(F.X, 3):
            rec = self._check_item(F, it, "D1")
            self.assertAlmostEqual(rec["rows"]["pc"]["dm"] / rec["D"], 1.0, places=5)
            self.assertEqual(rec["rows"]["sz"]["dm"], 0.0)
            for br in R.BRANCHES:
                for a in (0.0, 2.0):                   # the decomposed edit on every unit reduces to the whole write
                    w = rec["rows"][R.arm_name("donor", br, "all", a)]["dm"]
                    d = rec["rows"][R.arm_name("donor", br, "dall", a)]["dm"]
                    self.assertLess(abs(w - d) / rec["D"], 1e-3, (br, a, w, d))
                wn = [rec["rows"][R.arm_name("donor", br, "all", a)]["wn"][br][0] for a in (0.0, 1.0)]
                self.assertEqual(wn[0], 0.0)           # alpha 0: loop-2 write of that branch is the clean write
                self.assertGreater(wn[1], 0.0)

    def test_process_item_bf16(self):
        F = Fake(self, dtype=torch.bfloat16)
        self.assertEqual(F.Rn.dtype, torch.bfloat16)
        for it in fake_items(F.X, 2, seed=3, ds="D2"):
            rec = self._check_item(F, it, "D2")
            for br in R.BRANCHES:
                w = rec["rows"][R.arm_name("donor", br, "all", 0.0)]["dm"]
                d = rec["rows"][R.arm_name("donor", br, "dall", 0.0)]["dm"]
                self.assertLess(abs(w - d) / rec["D"], 0.05, (br, w, d))

    def test_all_alpha0_equals_prod_hold(self):
        """alpha 0 of the whole-write arm is the RUN 1c hold: the same rows as k17_freeze.process_item, bit-equal."""
        F = Fake(self, dtype=torch.bfloat16)
        it = fake_items(F.X, 1, seed=5)[0]
        arms = [("donor", br, "all", 0.0) for br in R.BRANCHES]
        rec, _, _ = R.process_item(F.Rn, it, "D1", arms=arms)
        kf, _ = KF.process_item(F.X, it["A"], it["B"], it["pos"], it["tA"], it["tB"], it["key"])
        self.assertEqual(rec["rows"]["donor_mlp_all_a0"]["final"], kf["rows"]["ans_frz_mlp"]["final"])
        self.assertEqual(rec["rows"]["donor_attn_all_a0"]["final"], kf["rows"]["ans_frz_attn"]["final"])
        self.assertEqual(rec["rows"]["E0"]["final"], kf["rows"]["E0"]["final"])

    def test_reduction_detects_a_wrong_clean_value(self):
        """A table built from a perturbed clean record must break the alpha-0 reduction of dall to all."""
        F = Fake(self)
        it = fake_items(F.X, 1, seed=7)[0]
        real = R.edit_tables
        try:
            def bad(br, units, uC, *a):
                uC2 = {k: v.clone() for k, v in uC.items()}
                uC2["act"] += 0.5
                return real(br, units, uC2, *a)
            R.edit_tables = bad
            rec, _, _ = R.process_item(F.Rn, it, "D1", arms=[("donor", "mlp", "all", 0.0),
                                                             ("donor", "mlp", "dall", 0.0)])
        finally:
            R.edit_tables = real
        self.assertNotAlmostEqual(rec["rows"]["donor_mlp_all_a0"]["dm"], rec["rows"]["donor_mlp_dall_a0"]["dm"],
                                  places=3)

    def test_sets_and_masks(self):
        F = Fake(self)
        it = fake_items(F.X, 1, seed=11)[0]
        rec, _, _ = R.process_item(F.Rn, it, "D1", arms=[("donor", "mlp", "top", 0.0)])
        self.assertEqual(sorted(rec["sets"]["donor"]), ["attn_top", "mlp_top"])     # top sets only (N7)
        S = {p: R.full_sets(rec, p, 4, 32, 2) for p in ("donor", "rnd")}
        self.assertEqual(S["donor"]["mlp"]["rand"], S["rnd"]["mlp"]["rand"])
        self.assertEqual(len(S["donor"]["mlp"]["rand"]), R.n_top_neurons(4, 32))
        rep = R.mask_report(S["donor"], 4, 32, 2)
        self.assertEqual(sum(rep["mlp_top"]["per_layer"]), R.n_top_neurons(4, 32))
        self.assertEqual(len(rep["attn_top"]), 2)
        uC = {"z": torch.randn(4, 4, 16), "o_attn": torch.randn(4, 4, 16)}
        tab = R.edit_tables("attn", None, uC, F.Rn, "cpu")
        self.assertEqual(sorted(tab), [(t, l) for t in (2, 3, 4) for l in range(4)])
        self.assertEqual(tab[(2, 0)][0].tolist(), list(range(16)))
        self.assertEqual(tab[(2, 0)][1].dtype, torch.float32)
        tab2 = R.edit_tables("attn", [5], uC, F.Rn, "cpu")          # unit 5 = layer 2, head 1
        self.assertEqual(sorted(tab2), [(t, 2) for t in (2, 3, 4)])
        self.assertEqual(tab2[(3, 2)][0].tolist(), list(range(8, 16)))


# ------------------------------------------------------------------ data, curves, hygiene
class TestTopAllEqualsDall(unittest.TestCase):
    def test_top_with_every_unit_equals_dall_bit_exact(self):
        """Review N1: the subset index map (u // n, u % n) with every unit listed gives the dall tables and the same
        logits, bit-exact, for both branches (bf16 fake)."""
        F = Fake(self, dtype=torch.bfloat16)
        Rn, X = F.Rn, F.X
        it = fake_items(X, 1, seed=13)[0]
        A, T = it["A"], len(it["A"])
        r = T - 1
        ids = torch.tensor([A])
        row = X.RowInfo(key="k", prompt_len=T, spans={"answer": (r,)}, payload={})
        _, _, uC = Rn.run(ids, row, spec=H31.Spec31(r, record=True, T=T))
        for br, n in (("mlp", Rn.L * Rn.I), ("attn", Rn.L * Rn.H)):
            t_all = R.edit_tables(br, None, uC, Rn, "cpu")
            t_lst = R.edit_tables(br, list(range(n)), uC, Rn, "cpu")
            self.assertEqual(sorted(t_all), sorted(t_lst))
            for k in t_all:
                self.assertTrue(torch.equal(t_all[k][0], t_lst[k][0]) and torch.equal(t_all[k][1], t_lst[k][1]))
            kw = "edit_mlp" if br == "mlp" else "edit_attn"
            za, _, _ = Rn.run(ids, row, spec=H31.Spec31(r, T=T, alpha=0.5, **{kw: t_all}))
            zl, _, _ = Rn.run(ids, row, spec=H31.Spec31(r, T=T, alpha=0.5, **{kw: t_lst}))
            self.assertTrue(torch.equal(za, zl))


class TestGates(unittest.TestCase):
    def _recs(self, gaps, base=1.0, D=4.0):
        return [{"key": "k%d" % i, "D": D, "cluster": "c%d" % (i // 2), "identity": {}, "info": {},
                 "rows": dict({R.arm_name("donor", br, "all", a): {"dm": D * base} for br in R.BRANCHES
                               for a in R.DALL_ALPHAS},
                              **{R.arm_name("donor", br, "dall", a): {"dm": D * (base + g)} for br in R.BRANCHES
                                 for a in R.DALL_ALPHAS})} for i, g in enumerate(gaps)]

    def test_g4_bias_and_noise(self):
        ok = R.g4_reduction({"D1": self._recs([0.02, -0.02] * 5)})
        self.assertTrue(ok["pass"], ok)
        self.assertAlmostEqual(ok["datasets"]["D1"]["mlp"]["a0"]["max_abs"], 0.02)
        self.assertIn("a2", ok["datasets"]["D1"]["mlp"])
        bias = R.g4_reduction({"D1": self._recs([0.02] * 10)})
        self.assertFalse(bias["pass"])
        noise = R.g4_reduction({"D1": self._recs([0.05, -0.05] * 5)})
        self.assertFalse(noise["pass"])
        dropped = R.g4_reduction({"D1": self._recs([0.05] * 4)}, drop=["attn:rand", "attn:top", "mlp:rand",
                                                                        "mlp:top"])
        self.assertTrue(dropped["pass"])

    def test_g3_counts_nonzero_gaps(self):
        recs = [{"key": "k%d" % i, "D": 4.0, "cluster": "c", "rows": {
            R.arm_name(p, br, "all", 0.0): {"dm": 4.0 + (0.004 if i == 0 else 0.0)} for p in ("donor", "rnd")
            for br in R.BRANCHES}} for i in range(4)]
        for rc in recs:
            rc["rows"]["E0"] = {"dm": 0.0}
        ref = {"D1": {"k%d" % i: {"D": 4.0, "ans_frz_mlp": 4.0, "ans_frz_attn": 4.0, "rnd_hold_mlp": 4.0,
                                  "rnd_hold_attn": 4.0, "E0": 0.0} for i in range(4)}}
        g = R.g3_repro({"D1": recs}, ref)
        self.assertTrue(g["pass"])
        self.assertEqual(g["datasets"]["D1"]["hold_mlp"]["n_nonzero"], 1)
        self.assertEqual(g["datasets"]["D1"]["E0"]["n_nonzero"], 0)
        ref["D1"]["k1"]["ans_frz_mlp"] = 3.0
        self.assertFalse(R.g3_repro({"D1": recs}, ref)["pass"])

    def test_gate_check_keys(self):
        good = {"key": "a", "identity": {k: True for k in R.GATED}, "info": {k: True for k in R.GATE_ONLY}}
        self.assertTrue(R.gate_check([good])["pass"])
        for k in R.GATED + R.GATE_ONLY:
            bad = json.loads(json.dumps(good))
            (bad["identity"] if k in R.GATED else bad["info"])[k] = False
            self.assertFalse(R.gate_check([bad])["pass"], k)

    def test_readings(self):
        c = lambda m, lo, hi: {"mean": m, "ci90": [lo, hi]}  # noqa: E731
        self.assertEqual(R.shape_reading([c(0.1, 0.05, 0.2), c(0.2, 0.1, 0.3), c(0.1, 0.02, 0.2)]), "threshold")
        self.assertEqual(R.shape_reading([c(0.1, 0.05, 0.2), c(-0.2, -0.3, -0.1), c(0, -0.1, 0.1)]), "not linear")
        self.assertEqual(R.shape_reading([c(0.1, 0.05, 0.2), c(0, -0.1, 0.1), c(0, -0.1, 0.1)]), "linear")
        self.assertEqual(R.concentration_reading(c(0.7, 0.6, 0.8), c(0.3, 0.1, 0.5)), "concentrated")
        self.assertTrue(R.concentration_reading(c(0.7, 0.6, 0.8), c(0.1, -0.1, 0.3)).startswith("neither"))
        self.assertEqual(R.concentration_reading(c(0.3, 0.2, 0.4), c(0.1, 0.0, 0.2)), "spread")
        self.assertTrue(R.inside(c(0.0, -0.05, 0.09)) and not R.inside(c(0.0, -0.05, 0.11)))

    def test_projection_and_drop_args(self):
        self.assertAlmostEqual(R.projected_hours({"D1": 7.2, "D2": 7.2}, {"D1": 267, "D2": 263}, 76, 76),
                               round(7.2 * 536 / 3600, 3))
        self.assertLess(R.projected_hours({"D1": 7.2}, {"D1": 100}, 76, 52), R.projected_hours({"D1": 7.2},
                                                                                             {"D1": 100}, 76, 76))
        a = R.parse_args(["--summarize", "--out", "x", "--drop", "attn:top,attn:rand"])
        self.assertEqual(len(R.arm_list(a.drop_list)), 72 - 2 * (8 + 2))     # donor 8 alphas, random 2, per set
        with self.assertRaises(SystemExit):
            R.parse_args(["--summarize", "--out", "x", "--drop", "mlp:all"])
        with self.assertRaises(SystemExit):
            R.parse_args(["--run", "--out", "x", "--plan_sha256", "y"])          # full without a pilot gate


class TestRunPath(unittest.TestCase):
    """run() pilot -> full -> resume -> summarize on the prod fake, the loaders replaced; the exits."""

    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="ra31_t_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.plan = os.path.join(self.tmp, "plan.md")
        with open(self.plan, "w") as f:
            f.write("test plan\n")
        self.sha = KF.sha256_file(self.plan)
        self.F = Fake(self)
        saved = (R.load_model, R.build_items, R.GAP_E)
        self.items = {ds: fake_items(self.F.X, 5, seed=20 + i, ds=ds) for i, ds in enumerate(R.DATASETS)}

        def restore():
            R.load_model, R.build_items, R.GAP_E = saved
        self.addCleanup(restore)
        R.GAP_E = 1e-6
        R.load_model = lambda args: (self.F.X, {"model_name": "fake", "revision": "fake"}, {"revision": "fake"})

        def items(X, args, ds, n):
            its = self.items[ds]
            return (its[:n] if n else its), {"drops": {"length": 0, "clean_A": 0, "clean_B": 0}, "n_run": len(its)}
        R.build_items = items

    def args(self, *extra):
        return R.parse_args(["--run", "--model", "2.6b", "--plan", self.plan, "--plan_sha256", self.sha] + list(extra))

    def test_pilot_full_resume_summarize(self):
        pil = os.path.join(self.tmp, "pilot")
        R.run(self.args("--out", pil, "--pilot", "4"))
        pg = json.load(open(os.path.join(pil, "pilot_gate.json")))
        self.assertTrue(pg["pass"], pg)
        self.assertEqual(pg["forwards_per_item"], 85)
        self.assertEqual(len(os.listdir(os.path.join(pil, "attr_D1"))), 4)
        self.assertTrue(os.path.exists(os.path.join(pil, "t2.json")))
        t2 = json.load(open(os.path.join(pil, "t2.json")))
        self.assertEqual(len(t2["D1"]), 3)
        self.assertIn("masks", t2["D1"][0])
        full = os.path.join(self.tmp, "full")
        R.run(self.args("--out", full, "--pilot_gate", os.path.join(pil, "pilot_gate.json"),
                        "--drop", "attn:top,attn:rand"))
        n1 = sum(1 for _ in open(os.path.join(full, "items_D1.jsonl")))
        self.assertEqual(n1, 5)
        R.run(self.args("--out", full, "--pilot_gate", os.path.join(pil, "pilot_gate.json"),
                        "--drop", "attn:top,attn:rand"))                         # resume: nothing new
        self.assertEqual(sum(1 for _ in open(os.path.join(full, "items_D1.jsonl"))), 5)
        ck = R.summarize(R.parse_args(["--summarize", "--out", full]))
        self.assertEqual(ck["attribution"]["D1"]["n_items"], 5)
        self.assertEqual(ck["attribution"]["D1"]["raw_bytes"], R.attr_bytes(5, 4, 32))
        self.assertTrue(os.path.exists(os.path.join(full, "attr_D2.npz")))
        self.assertIn("size curve", open(os.path.join(full, "summary.md")).read())
        self.assertIn("mlp_share_top_alpha0_signed", ck["predictions"]["D1"])
        self.assertTrue(ck["gate_pass"] and ck["g4_pass"] and ck["outputs_hashed"])
        self.assertTrue(all(ck["identities_all_items"].values()))
        self.assertIn("mlp_shape", ck["predictions"]["D1"])
        rows = list(open(os.path.join(full, "results.csv")))
        self.assertEqual(len(rows) - 1, ck["results_csv_rows"])
        with self.assertRaises(SystemExit) as cm:                                 # changed drop on resume
            R.run(self.args("--out", full, "--pilot_gate", os.path.join(pil, "pilot_gate.json")))
        self.assertEqual(cm.exception.code, 2)

    def test_thinking_pilot_runs_without_g3(self):
        out = os.path.join(self.tmp, "think")
        a = R.parse_args(["--run", "--model", "1.4b-think", "--plan", self.plan, "--plan_sha256", self.sha,
                          "--out", out, "--pilot", "3"])
        R.run(a)
        pg = json.load(open(os.path.join(out, "pilot_gate.json")))
        self.assertTrue(pg["pass"])
        self.assertEqual(pg["model"], "1.4b-think")
        self.assertFalse(pg["g3"]["applied"])

    def test_plan_mismatch_exit_2(self):
        a = R.parse_args(["--run", "--model", "2.6b", "--plan", self.plan, "--plan_sha256", "0" * 64,
                          "--out", os.path.join(self.tmp, "x"), "--pilot", "2"])
        with self.assertRaises(SystemExit) as cm:
            R.run(a)
        self.assertEqual(cm.exception.code, 2)

    def test_identity_gate_failure_exit_3_no_scored_row(self):
        real = R.process_item

        def broken(*a, **k):
            rec, ident, info = real(*a, **k)
            ident["pc_exact"] = False
            rec["identity"] = ident
            return rec, ident, info
        R.process_item = broken
        self.addCleanup(setattr, R, "process_item", real)
        out = os.path.join(self.tmp, "g")
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", out, "--pilot", "3"))
        self.assertEqual(cm.exception.code, 3)
        self.assertFalse(os.path.exists(os.path.join(out, "items_D1.jsonl")))

    def test_full_without_passing_pilot_gate_exit_3(self):
        p = os.path.join(self.tmp, "pg.json")
        json.dump({"pass": False, "plan_sha256": self.sha, "model": "2.6b"}, open(p, "w"))
        with self.assertRaises(SystemExit) as cm:
            R.run(self.args("--out", os.path.join(self.tmp, "f"), "--pilot_gate", p))
        self.assertEqual(cm.exception.code, 3)


class TestModelAxis(unittest.TestCase):
    def test_thinking_models_pins_and_budgets(self):
        revs = json.load(open(os.path.join(TREE, "prod", "tasks", "data", "model_revisions.json")))["models"]
        for m, name in (("1.4b-think", "ouro_1_4b_think"), ("2.6b-think", "ouro_2_6b_think")):
            self.assertEqual(R.MODELS[m], name)
            self.assertEqual(R.REVISION_PINS[m], revs[name]["revision"])
            self.assertIn(m, R.BUDGET_H)
            self.assertEqual(R.parse_args(["--summarize", "--out", "x", "--model", m]).model, m)
        self.assertEqual((R.MODELS["1.4b"], R.BUDGET_H["1.4b"], R.MODELS["2.6b"], R.BUDGET_H["2.6b"]),
                         ("ouro_1_4b_base", 1.5, "ouro_2_6b_base", 5.5))           # 2.6B budget 5.5


class TestLengthDrop(unittest.TestCase):
    def test_unequal_length_is_a_counted_drop(self):
        """Review N3 on the 1.4B path: run1c.build_items replaced by a stub returning one unequal pair."""
        saved = R.R1.build_items
        self.addCleanup(setattr, R.R1, "build_items", saved)
        its = [{"key": "a", "A": [1, 2], "B": [1, 3]}, {"key": "b", "A": [1, 2], "B": [1, 2, 3]},
               {"key": "c", "A": [4, 2], "B": [5, 2]}]
        R.R1.build_items = lambda X, m, d, ds: (list(its), {"n_matched": 3, "bad": []})
        a = R.parse_args(["--summarize", "--out", "x"])
        items, info = R.build_items(None, a, "D1", 0)
        self.assertEqual([it["key"] for it in items], ["a", "c"])
        self.assertEqual(info["drops"]["length"], 1)
        items, _ = R.build_items(None, a, "D1", 1)
        self.assertEqual([it["key"] for it in items], ["a"])


class TestSizeSetsAndAttribution(unittest.TestCase):
    def test_ranked_sets_nest_and_count(self):
        F = Fake(self)
        it = fake_items(F.X, 1, seed=17)[0]
        rec, ident, _ = R.process_item(F.Rn, it, "D1", arms=[("donor", "mlp", "top20", 0.0)])
        self.assertTrue(ident["set_counts"])
        a = rec["attr"]
        self.assertEqual(a["n20"], R.n_top_neurons(4, 32, 0.20))
        self.assertEqual(sorted(rec["_attr"]["idx"][:R.n_top_neurons(4, 32)]), rec["sets"]["donor"]["mlp_top"])
        S = R.full_sets(rec, "donor", 4, 32, 2)
        self.assertEqual(len(S["mlp"]["rand1"]), R.n_top_neurons(4, 32, 0.01))
        self.assertEqual(len(S["mlp"]["rand20"]), R.n_top_neurons(4, 32, 0.20))
        self.assertNotEqual(S["mlp"]["rand20"][:6], S["mlp"]["rand"])
        self.assertEqual(tuple(rec["_attr"]["est"].shape), (3, a["n20"]))

    def test_attribution_is_the_first_order_margin_change(self):
        """Prompt B plays the patched run and clean A the reference: holding the top-20-percent set at A's values
        with alpha = 1 - eps changes the margin by eps * sum(est) to first order (float32 fake)."""
        F = Fake(self)
        Rn, X = F.Rn, F.X
        it = fake_items(X, 1, seed=19)[0]
        T = len(it["A"])
        r, tA, tB = T - 1, it["tA"], it["tB"]
        row = X.RowInfo(key="k", prompt_len=T, spans={"answer": (r,)}, payload={})
        _, _, uA = Rn.run(torch.tensor([it["A"]]), row, spec=H31.Spec31(r, record=True, T=T))
        zB, _, uB = Rn.run(torch.tensor([it["B"]]), row, spec=H31.Spec31(r, record=True, T=T))
        order20 = R.rank_order(R.neuron_scores(uB["act"], uA["act"]), R.n_top_neurons(4, 32, 0.2))
        sp = H31.Spec31(r, T=T, grad_w=[(t, l) for t in R.LOOPS for l in range(4)])
        zg, grads = Rn.run_grad(torch.tensor([it["B"]]), row, [], sp, r, tA, tB)
        self.assertTrue(torch.equal(zg, zB))
        pred = float(R.attribution(Rn, grads, uA, uB, order20, r).sum())
        tab = R.edit_tables("mlp", sorted(order20), uA, Rn, "cpu")
        mB = float(zB[r, tB] - zB[r, tA])
        est = R.attribution(Rn, grads, uA, uB, order20, r)
        for u in order20[:3]:                    # single neurons: the finite difference matches closely
            e1 = float(R.attribution(Rn, grads, uA, uB, [u], r).sum())
            t1 = R.edit_tables("mlp", [u], uA, Rn, "cpu")
            z, _, _ = Rn.run(torch.tensor([it["B"]]), row, spec=H31.Spec31(r, T=T, alpha=1 - 1e-3, edit_mlp=t1))
            d = (float(z[r, tB] - z[r, tA]) - mB) / 1e-3
            self.assertLess(abs(d - e1), 0.02 * abs(e1) + 2e-3, (u, d, e1))
        eps = 1e-4                               # the whole set: first order only at small eps (the terms cancel)
        z, _, _ = Rn.run(torch.tensor([it["B"]]), row, spec=H31.Spec31(r, T=T, alpha=1 - eps, edit_mlp=tab))
        d = (float(z[r, tB] - z[r, tA]) - mB) / eps
        self.assertLess(abs(d - pred), 0.02 * float(est.abs().sum()), (d, pred))

    def test_store_roundtrip_and_size_at_530(self):
        import numpy as np
        import shutil
        import tempfile
        self.assertEqual(R.attr_bytes(530, 24, 5632), 530 * (27034 * 10 + 20))          # 143,290,800 bytes
        self.assertLess(R.attr_bytes(530, 24, 5632) / 2 ** 20, 140)                     # 136.7 MiB raw per run
        F = Fake(self, dtype=torch.bfloat16)
        d = tempfile.mkdtemp(prefix="ra31_a_")
        self.addCleanup(shutil.rmtree, d, True)
        its = fake_items(F.X, 3, seed=23)
        for i, it in enumerate(its):
            rec, _, _ = R.process_item(F.Rn, it, "D1", arms=[("donor", "mlp", "top20", 0.0)])
            R.save_attr(d, "D1", i, it["key"], rec["_attr"])
        path, info = R.pack_attr(d, "D1")
        z = np.load(path)
        self.assertEqual(list(z["keys"]), [it["key"] for it in its])
        self.assertEqual(z["est"].dtype, np.float16)
        self.assertEqual(z["idx"].dtype, np.int32)
        self.assertEqual(z["est"].shape, (3, 3, R.n_top_neurons(4, 32, 0.2)))
        self.assertEqual(info["raw_bytes"], R.attr_bytes(3, 4, 32))
        est = torch.from_numpy(z["est"].astype(np.float32)).double().sum(dim=(1, 2))
        self.assertTrue(torch.allclose(torch.from_numpy(z["scalars"][:, 4]).double(), est, rtol=2e-2, atol=1e-2))

    def test_accepted_pilot_plan(self):
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix="ra31_p_")
        self.addCleanup(shutil.rmtree, d, True)
        p = os.path.join(d, "pg.json")
        json.dump({"pass": True, "plan_sha256": R.ACCEPTED_PILOT_PLANS[0], "model": "1.4b"}, open(p, "w"))
        self.assertEqual(R.check_pilot_gate(p, "f" * 64, "1.4b")[0], [])
        json.dump({"pass": True, "plan_sha256": "e" * 64, "model": "1.4b"}, open(p, "w"))
        self.assertTrue(R.check_pilot_gate(p, "f" * 64, "1.4b")[0])


class TestData(unittest.TestCase):
    def test_sources_and_run1c_means(self):
        src = json.load(open(os.path.join(HERE, "data", "sources.json")))
        import make_data as MD
        for f, h in src["outputs"].items():
            self.assertEqual(MD.sha(os.path.join(HERE, "data", f)), h)
        ref = json.load(open(os.path.join(HERE, "data", "run1c_ref.json")))["datasets"]
        self.assertEqual({k: len(v) for k, v in ref.items()}, {"D1": 267, "D2": 263})

        def mean(ds, row):
            v = [x[row] / x["D"] for x in ref[ds].values() if x["D"] >= 2.0]
            return sum(v) / len(v)
        self.assertAlmostEqual(mean("D1", "ans_frz_mlp"), 1.2773, places=3)
        self.assertAlmostEqual(mean("D1", "rnd_hold_mlp"), 0.5664, places=3)
        self.assertAlmostEqual(mean("D2", "ans_frz_mlp"), 1.4132, places=3)
        self.assertAlmostEqual(mean("D2", "rnd_hold_mlp"), 0.626, places=3)

    def test_run1c_gap_and_curve(self):
        recs = [{"key": "k%d" % i, "D": 4.0, "cluster": "c%d" % (i // 2),
                 "rows": {R.arm_name("donor", "mlp", "all", a): {"dm": 4.0 * (1 - a) + 0.004 * (i % 2)}
                          for a in R.ALPHAS}} for i in range(20)]
        ref = {"k%d" % i: {"D": 4.0, "ans_frz_mlp": 4.0} for i in range(20)}
        g = R.run1c_gap(recs, ref, "donor_mlp_all_a0", "ans_frz_mlp")
        self.assertEqual(g["n"], 20)
        self.assertAlmostEqual(g["max_abs"], 0.001)
        c = R.curve(recs, "donor", "mlp", "all")
        self.assertAlmostEqual(c[0.0]["mean"], 1.0005, places=4)
        self.assertAlmostEqual(c[2.0]["mean"], -0.9995, places=4)
        self.assertEqual(c[1.0]["n_clusters"], 10)


class TestHygiene(unittest.TestCase):
    def test_no_host_user_home_strings(self):
        words = ["sas" + "11", "sab" + "bir", "/ho" + "me/", "/us" + "ers/", "dg" + "x1", "rtx" + "6000"]
        pat = re.compile("(?i)(" + "|".join(re.escape(w) for w in words) + ")")
        for f in ("ra31.py", "make_data.py", "lib/ra31_hooks.py", "tests/test_ra31.py", "plan_ra31.md", "run.sh",
                  "RUNSHEET.md"):
            p = os.path.join(HERE, f)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    self.assertIsNone(pat.search(fh.read()), f)


if __name__ == "__main__":
    unittest.main()
