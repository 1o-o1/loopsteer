"""RUN 1c (LEDGER 2026-10-11, queued after RUN 1): branch freeze at matched depth on IFM d112, and a random same-norm
patch under the holds on Ouro-1.4B. Plan: plan_run1c.md in this folder (sha256 checked before anything runs).

d112 (ifm_d112, 112 layers, no loop): donor patch (clean B) of the last prompt token r at the output of layer 28 of 112
(0-based layer 27, the d112 layer that k15_smoke maps to the loop-1 end of the looped IFM model). Holds: the MLP write or
the attention write of r at its clean-A value in every later layer (28 to 111). Hooks: lib/k17_freeze_ifm.IfmCtx over
lib/k15_smoke.IfmM (unchanged copies; state edits at runner._layer outputs, writes at attention.wo and nffn.fc2).

Ouro-1.4B base: lib/k17_freeze.process_item (the freeze pilot's 13 rows and identities, unchanged) plus a random
same-norm patch of r at the end of loop 1, alone, with the MLP write held (loops 2 to 4) and with the attention write
held. Hooks: the production package copy in tree/prod (the freeze-plan md5 pins), built through
interventions.from_spec / freeze_spec as k17_freeze does. This file adds no hook.

    python run1c.py --hash
    python run1c.py --run --model d112|ouro --out OUT --plan_sha256 SHA --debug 10
    python run1c.py --run --model d112|ouro --out OUT --plan_sha256 SHA --debug_gate DEBUG_OUT/debug_gate.json
    python run1c.py --summarize --out OUT

Exit codes: 2 plan, pin, item or config mismatch; 3 gate failed (identity gate before any scored row, d112 sign
gate, missing debug gate, projected time over budget); 4 identity failed on a scored row; 5 stopped by SIGTERM (resumable).
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import torch  # noqa: E402

import k17_freeze as KF  # noqa: E402
import results_common as RC  # noqa: E402

PLAN_DEFAULT = os.path.join(HERE, "plan_run1c.md")
DATA_DEFAULT = os.path.join(HERE, "data")
TREE_DEFAULT = os.path.join(HERE, "tree")
MODEL_KEYS = ("d112", "ouro")
DATASETS = ("D1", "D2")
DS_NAME = {"D1": "facts", "D2": "addition"}
D112_NAME = "ifm_d112"
D112_CKPT = "LoopedLM-P1-dense-d112-336b"
D112_LAYERS = 112
PATCH_LAYER = 27               # 0-based: the output of layer 28 of 112 (= the loop-1 end of the looped IFM model)
HOLD_FROM = 28                 # the holds act in every later layer, 28 to 111 (0-based)
OURO_REPO = "ByteDance/Ouro-1.4B"
OURO_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"     # freeze plan section 4
OURO_TRANSFORMERS = "4.56.2"
OURO_PROD_MD5 = {                                               # the freeze pilot's config.json prod_md5
    "prod/config.py": "41843c88a6ca7a86d502db3c166d7f53",
    "prod/models/ouro.py": "747a09deee4cd38619ed6bf36787313d",
    "prod/steer/adapters/__init__.py": "fe77df8bab1299ff8f25310c7dcefd10",
    "prod/steer/adapters/ouro.py": "f9b7869c7463817c2e452f2c35f9a83a",
    "prod/steer/hooks.py": "943ff4105e0ca7c2e436bc8caddcbf13",
    "prod/steer/interventions.py": "574fca55456f444d1864efe0629a45b6",
    "prod/steer/readouts.py": "94659233915e6240aa118642aa5c12be",
    "prod/steer/sites.py": "7c0234a9a1f3dc1344bd7b4b7df0ac1b"}
ITEMS_D1_SHA = "637cf5574d3425d45af6d4fb5051a30b0f69247a931f913117e9c3d2df1998ad"
N_EXPECTED = {"d112": {"D1": 80, "D2": 80}, "ouro": {"D1": 267, "D2": 263}}
GATE_N = 3                     # identity-gate items per dataset, before any scored row, every session
DEBUG_N = 10                   # debug step: the first 10 items of each dataset
SIGN_MIN = 0.2                 # sign gate (d112 only): positive control - patch alone >= 0.2 per dataset
SIGN_GATE_MODELS = ("d112",)   # Ouro keeps the freeze pilot's own gates (identities, positive control, zero)
REF_TOL = 0.15                 # comparison with earlier values of the patch alone: |diff| <= 0.15 (recorded only)
R_MIN = 0.5                    # R is printed only where the patch alone has mean NLD >= 0.5
FULL_BUDGET_MIN = 45.0        # a full run starts only if its projected scored-row minutes fit this budget
GAP_E = KF.GAP_E               # 2.0
NB = KF.NB                     # 10,000 resamples, seed 0
GRID, CARD, SCRIPT = "run1c", "matched_depth_freeze", "run1c.py"
EXIT_PLAN, EXIT_GATE, EXIT_IDENTITY, EXIT_STOPPED = 2, 3, 4, 5
GATED = {"d112": ("hook_identity", "readout_identity", "hold_identity", "patch_state_is_donor",
                  "source_attn_held_identity", "source_edit_reaches_r", "pc_exact", "sz_exact"),
         "ouro": ("hook_identity", "readout_identity", "freeze_no_edit_identity", "source_attn_frozen_identity",
                  "source_edit_reaches_r", "pc_exact", "sz_exact", "ans_loop1_state_is_donor", "rerun_identity",
                  "rnd_loop1_rows_equal")}
GATE_ONLY = ("holds_act",)     # non-vacuity of the holds: checked on the gate items before any scored row
# canonical condition -> record row name, per model
ROWS = {"d112": {"patch": "patch", "patch_hold_mlp": "patch_hold_mlp", "patch_hold_attn": "patch_hold_attn",
                 "pc": "pc_final", "sz": "sz", "hold_mlp": "hold_mlp", "hold_attn": "hold_attn"},
        "ouro": {"patch": "E0", "patch_hold_mlp": "ans_frz_mlp", "patch_hold_attn": "ans_frz_attn",
                 "pc": "pc_final", "sz": "sz", "hold_mlp": "frz_mlp", "hold_attn": "frz_attn",
                 "rnd": "rnd", "rnd_hold_mlp": "rnd_hold_mlp", "rnd_hold_attn": "rnd_hold_attn"}}
STOP = {"flag": False}
r6, margin = KF.r6, KF.margin


def die(code, msg):
    print("[run1c] %s" % msg, flush=True)
    raise SystemExit(code)


# ------------------------------------------------------------------ arguments (CPU)
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="run1c")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--hash", action="store_true", help="print the plan sha256 and the code sha256 (no model)")
    ap.add_argument("--model", choices=MODEL_KEYS)
    ap.add_argument("--out")
    ap.add_argument("--plan", default=PLAN_DEFAULT)
    ap.add_argument("--plan_sha256")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--tree", default=TREE_DEFAULT)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--debug_gate", help="full run: the debug step's debug_gate.json (must hold pass true)")
    ap.add_argument("--mem_fraction", type=float, default=0.12)
    a = ap.parse_args(argv)
    if a.debug < 0:
        ap.error("--debug must be >= 0")
    if a.run and not (a.model and a.out and a.plan_sha256):
        ap.error("--run needs --model, --out and --plan_sha256")
    if a.run and not a.debug and not a.debug_gate:
        ap.error("a full --run needs --debug_gate")
    if a.summarize and not a.out:
        ap.error("--summarize needs --out")
    return a


# ------------------------------------------------------------------ hashes (CPU)
def check_plan(plan, plan_sha256):
    got = KF.sha256_file(plan)
    if got != plan_sha256:
        die(EXIT_PLAN, "plan sha256 mismatch: file %s, given %s" % (got, plan_sha256))
    return got


def code_hashes():
    fs = {"run1c.py": os.path.abspath(__file__)}
    for f in sorted(os.listdir(LIB)):
        if f.endswith(".py"):
            fs["lib/" + f] = os.path.join(LIB, f)
    return {k: KF.sha256_file(p) for k, p in fs.items()}


DATA_FILES = {"d112": ("d112_D1.json", "d112_D2.json", "ref_values.json"),
              "ouro": ("items_d1.json", "ouro_D2.json", "ref_values.json")}


def data_hashes(data_dir, model):
    return {f: KF.sha256_file(os.path.join(data_dir, f)) for f in DATA_FILES[model]}


HASH_KEYS = ("plan_sha256", "code_sha256", "data_sha256", "pins", "config_sha256")


def write_hash_line(out, plan_sha, cfg, data, pins):
    """One hash line before the first scored row (no host, user or path). On resume the same line must come out."""
    d = {"run": "RUN 1c", "plan_sha256": plan_sha,
         "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest(),
         "code_sha256": code_hashes(), "data_sha256": data, "pins": pins,
         "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = os.path.join(out, "hash_line.json")
    if os.path.exists(p):
        old = json.load(open(p))
        for k in HASH_KEYS:
            if old.get(k) != d[k]:
                die(EXIT_PLAN, "resume with a changed %s; refusing" % k)
        return old
    json.dump(d, open(p, "w"), indent=1, sort_keys=True)
    return d


# ------------------------------------------------------------------ items (CPU-testable)
def match_items(cands, rows, keyfn, ds):
    """Rebuilt prompts for the stored identities. cands: k15_data candidates on this model's tokenizer; rows: stored
    identity rows. A row matches one candidate on (rel, a, b, half); pos, T, tA and tB (and A, B where stored) must be
    equal. Returns (items, bad)."""
    by = {}
    for p in cands:
        by.setdefault((p["rel"], p["a"], p["b"], p["half"]), []).append(p)
    items, bad = [], []
    for rw in rows:
        k = (rw["rel"], rw["a"], rw["b"], rw["half"])
        ps = by.get(k, [])
        if len(ps) != 1:
            bad.append({"key": list(k), "n_match": len(ps)})
            continue
        p = ps[0]
        ok = (sorted(p["pos"]) == sorted(rw["pos"]) and len(p["A"]) == rw.get("T", len(p["A"])) and
              p["tA"] == rw["tA"] and p["tB"] == rw["tB"] and
              ("A" not in rw or (p["A"] == rw["A"] and p["B"] == rw["B"])))
        if not ok:
            bad.append({"key": list(k), "identity_mismatch": True})
            continue
        it = {"key": rw.get("key") or keyfn(rw), "A": p["A"], "B": p["B"], "pos": sorted(p["pos"]), "tA": p["tA"],
              "tB": p["tB"], "cluster": rw["cluster"], "stratum": rw["rel"], "half": rw["half"]}
        for f in ("sa", "sb", "src", "idx", "stored_D"):
            if f in rw:
                it[f] = rw[f]
        if ds == "D1" and "sa" not in it:
            it["sa"], it["sb"] = "%s:%s" % (rw["rel"], rw["a"]), "%s:%s" % (rw["rel"], rw["b"])
        items.append(it)
    keys = [it["key"] for it in items]
    if len(set(keys)) != len(keys):
        bad.append({"duplicate_keys": len(keys) - len(set(keys))})
    return items, bad


def key_d1(rw):
    return "%s|%s|%s|%s" % (rw["rel"], rw["a"], rw["b"], rw["half"])


def key_d2(rw):
    return "%s|%s|%s" % (rw["rel"], rw["a"], rw["b"])


def ouro_d2_rows(data_dir):
    """data/ouro_D2.json -> identity rows with rel, a, b, half parsed from the stored pair key."""
    rows = []
    for it in json.load(open(os.path.join(data_dir, "ouro_D2.json")))["items"]:
        rel, a, b = it["pair"].split("|")
        rows.append(dict(it, rel=rel, a=a, b=b))
    return rows


def build_items(X, model, data_dir, ds):
    """-> (items, info). Prompts are rebuilt with the unchanged k15_data builders on this model's tokenizer and must
    match the stored identities exactly; any mismatch stops the run (exit 2)."""
    import k15_data as KD
    if model == "ouro" and ds == "D1":
        items, info = KF.build_d1(X.enc, os.path.join(data_dir, "items_d1.json"))   # the pilot's own D1 path
        bad = info.get("bad", [])
    elif model == "ouro":
        items, bad = match_items(KD.d2_candidates(X.enc), ouro_d2_rows(data_dir), key_d2, ds)
        info = {"n_rows": N_EXPECTED[model][ds], "n_matched": len(items), "bad": bad}
    else:
        rows = json.load(open(os.path.join(data_dir, "d112_%s.json" % ds)))["items"]
        cands = KD.d1_candidates(X.enc) if ds == "D1" else KD.d2_candidates(X.enc)
        items, bad = match_items(cands, rows, key_d1 if ds == "D1" else key_d2, ds)
        info = {"n_rows": len(rows), "n_matched": len(items), "bad": bad, "candidates": len(cands)}
    if bad or len(items) != N_EXPECTED[model][ds]:
        die(EXIT_PLAN, "%s %s items do not rebuild: %d matched of %d, bad %s" % (
            model, ds, len(items), N_EXPECTED[model][ds], json.dumps(bad[:5])))
    info["rule"] = "stored identities, prompts rebuilt by k15_data on this tokenizer, exact match"
    return items, info


# ------------------------------------------------------------------ d112 rows (GPU path, CPU-testable)
def _eq(a, b):
    return bool(torch.equal(a, b))


def check_d112_ctx(X):
    """The matched-depth site must exist: one pass of 112 layers, the k15_smoke d112 map putting the loop-1 end at
    layer 27, and a write hook on every layer."""
    bad = []
    if X.K != 1 or X.L != D112_LAYERS or X.s0 != 0 or X.e0 != D112_LAYERS:
        bad.append("not one pass of %d layers (K %s, L %s)" % (D112_LAYERS, X.K, X.L))
    le = getattr(X.M, "loop_ends", None)
    if not le or le[0][0] != (1, PATCH_LAYER):
        bad.append("k15_smoke d112 map does not put the loop-1 end at layer %d: %s" % (PATCH_LAYER, le and le[0]))
    if len(X.handles) != 2 * D112_LAYERS:
        bad.append("write hooks %d, expected %d" % (len(X.handles), 2 * D112_LAYERS))
    if (1, PATCH_LAYER) not in X.slot_pt.values():
        bad.append("no state hook at layer %d" % PATCH_LAYER)
    return bad


@torch.no_grad()
def process_d112(X, it):
    """Rows of plan section 5 (d112), 13 batch-1 forwards. -> (record, identity flags, info flags)."""
    import k17_freeze_ifm as KFI
    A, B, src, tA, tB, key = it["A"], it["B"], sorted(it["pos"]), it["tA"], it["tB"], it["key"]
    T = len(A)
    r = T - 1
    last = X.e0 - 1
    SITE, FIN, SRC0 = ("state", 1, PATCH_LAYER), ("state", 1, last), ("state", 1, X.s0)
    hold_calls = [(1, l) for l in range(HOLD_FROM, X.e0)]
    src_calls = [(1, l) for l in range(X.s0 + 1, X.e0)]
    t0, f0 = time.time(), X.n_fwd
    z_nh = X.nohook(A)[0, -1]
    zA, rA = X.run(A, want=[SITE, FIN] + [(k_,) + c for k_ in ("attn", "mlp") for c in src_calls])
    zB, rB = X.run(B, want=[SITE, FIN, SRC0])
    HA_s, HB_s, HA_f, HB_f = rA[SITE][0], rB[SITE][0], rA[FIN][0], rB[FIN][0]

    def hold(kind, calls):
        return {(kind,) + c: KFI.copy_at([r], rA[(kind,) + c][0, r:r + 1]) for c in calls}
    patch = {SITE: KFI.copy_at([r], HB_s[r:r + 1])}
    hm, ha, ha_src = hold("mlp", hold_calls), hold("attn", hold_calls), hold("attn", src_calls)
    srcp = {SRC0: KFI.copy_at(src, rB[SRC0][0, src])}
    pcf = {FIN: KFI.copy_at([r], HB_f[r:r + 1])}
    sz = {FIN: KFI.copy_at(src, HB_f[src])}
    z_zero, _ = X.run(A, ops={SITE: (lambda o: o)})               # strength 0: the tensor passes unchanged
    z_P, rP = X.run(A, ops=patch, want=[SITE])
    z_Pm, rPm = X.run(A, ops={**patch, **hm}, want=[SITE])
    z_Pa, rPa = X.run(A, ops={**patch, **ha}, want=[SITE])
    z_hm, _ = X.run(A, ops=hm)
    z_ha, _ = X.run(A, ops=ha)
    z_src, _ = X.run(A, ops=srcp)
    z_sf, rSf = X.run(A, ops={**srcp, **ha_src}, want=[FIN])
    z_pc, _ = X.run(A, ops=pcf)
    z_sz, _ = X.run(A, ops=sz)
    ident = {
        "hook_identity": _eq(zA, z_nh) and _eq(z_zero, zA),
        "readout_identity": _eq(X.readout(HA_f, r), zA) and _eq(X.readout(HB_f, r), zB),
        "hold_identity": _eq(z_hm, zA) and _eq(z_ha, zA),
        "patch_state_is_donor": all(_eq(rr[SITE][0][r], HB_s[r]) for rr in (rP, rPm, rPa)),
        "source_attn_held_identity": _eq(z_sf, zA) and _eq(rSf[FIN][0][r], HA_f[r]),
        "source_edit_reaches_r": not _eq(z_src, zA),
        "pc_exact": _eq(z_pc, zB),
        "sz_exact": _eq(z_sz, zA),
    }
    info = {"holds_act": (not _eq(z_Pm, z_P)) and (not _eq(z_Pa, z_P)),
            "hold_mlp_acts": not _eq(z_Pm, z_P), "hold_attn_acts": not _eq(z_Pa, z_P),
            "patch_reaches_final": not _eq(z_P, zA)}
    cA, cB = KFI.vals(zA, tA, tB), KFI.vals(zB, tA, tB)
    rec = {"key": key, "T": T, "pos": src, "tA": tA, "tB": tB, "clean_final": cA, "cleanB_final": cB,
           "D": r6(margin(cB) - margin(cA)), "rows": {}}
    for nm, z in (("patch", z_P), ("patch_hold_mlp", z_Pm), ("patch_hold_attn", z_Pa), ("hold_mlp", z_hm),
                  ("hold_attn", z_ha), ("zero", z_zero), ("src", z_src), ("src_hold_attn", z_sf),
                  ("pc_final", z_pc), ("sz", z_sz)):
        v = KFI.vals(z, tA, tB)
        rec["rows"][nm] = {"final": v, "dm": r6(margin(v) - margin(cA))}
    tn = float((HB_s[r] - HA_s[r]).float().norm())
    rec["site_target_norm"] = r6(tn)
    rec["identity"], rec["info"] = ident, info
    rec["forwards"] = X.n_fwd - f0
    rec["seconds"] = round(time.time() - t0, 3)
    for k_ in ("cluster", "sa", "sb", "stratum", "half", "idx", "stored_D"):
        if k_ in it:
            rec[k_] = it[k_]
    return rec, ident, info


# ------------------------------------------------------------------ Ouro rows (GPU path, CPU-testable)
def rnd_direction(ds, key, d):
    """Unit Gaussian direction, float32, seeded by the item: torch CPU generator, seed = first 15 hex digits of
    sha256('run1c:<ds>:<key>')."""
    s = int(hashlib.sha256(("run1c:%s:%s" % (ds, key)).encode("utf-8")).hexdigest()[:15], 16)
    g = torch.Generator(device="cpu").manual_seed(s)
    u = torch.randn(d, generator=g, dtype=torch.float32)
    return u / u.norm()


@torch.no_grad()
def process_ouro(X, it, ds):
    """lib/k17_freeze.process_item (13 rows, unchanged) plus 5 forwards: clean A and clean B again with recorders,
    then the random same-norm patch of r at the end of loop 1 alone, with the MLP write held and with the attention
    write held (loops 2 to 4, freeze_spec from F = first_changed(loop1.loop_end), as the pilot's holds).
    -> (record, identity flags, info flags)."""
    A, B, src, tA, tB, key = it["A"], it["B"], sorted(it["pos"]), it["tA"], it["tB"], it["key"]
    rec, ident = KF.process_item(X, A, B, src, tA, tB, key)
    I, P, K = X.I, X.Positions, KF.K
    T = len(A)
    r = T - 1
    t0, f0 = time.time(), X.n_fwd
    ids_A = torch.tensor([A], device=X.dev)
    ids_B = torch.tensor([B], device=X.dev)
    spans = {"source": tuple(src), "answer": (r,)}
    at_r = P("abs", abs=(r,))
    pos_r = {"kind": "abs", "abs": [r]}

    def row(payload=None):
        return X.RowInfo(key=key, prompt_len=T, spans=dict(spans), payload=payload or {})

    def le_rec():
        return X.Recorder(id="le", sites=tuple(X.s_le), positions=P("all"), what="state", when="post")

    def H_of(recs):
        return {j: recs[(key, "le", X.s_le[j - 1].key())] for j in range(1, K + 1)}
    frz_rec = X.Recorder(id="frz", sites=tuple(X.frz_sites), positions=at_r, what="state", when="post")
    zA, rA = X.run(ids_A, row(), recs=[le_rec(), frz_rec])
    zA = zA[0]
    HA = H_of(rA)
    zB, rB = X.run(ids_B, row(), recs=[le_rec()])
    HB = H_of(rB)
    delta = (HB[1][r] - HA[1][r]).float()
    n_t = float(delta.norm())
    u = rnd_direction(ds, key, delta.shape[0]).to(delta.device)
    payload = (HA[1][r].float() + n_t * u).to(HA[1].dtype)[None]
    rnd = I.from_spec({"id": "rnd", "op": "patch", "site": X.s_ans.key(), "a": 1.0, "target": {"kind": "donor"},
                       "positions": pos_r}, X.g)

    def freeze(comp):
        iv = I.from_spec(dict(I.freeze_spec(comp, pos_r, X.F_ans, X.g), id="frz_" + comp), X.g)
        return iv, {s.key(): rA[(key, "frz", s.key())] for s in iv.sites}
    fm, pay_fm = freeze("mlp_out")
    fa, pay_fa = freeze("attn_out")

    def edited(ivs, extra):
        p = {"rnd": {X.s_ans.key(): payload}}
        p.update(extra)
        z, rr = X.run(ids_A, row(p), ivs=ivs, recs=[le_rec()])
        return z[0], H_of(rr)
    z_R, H_R = edited([rnd], {})
    z_Rm, H_Rm = edited([rnd, fm], {"frz_mlp_out": pay_fm})
    z_Ra, H_Ra = edited([rnd, fa], {"frz_attn_out": pay_fa})
    cA = KF.vals_at(zA, r, tA, tB)
    ident["rerun_identity"] = cA == rec["clean"][K - 1]
    ident["rnd_loop1_rows_equal"] = _eq(H_Rm[1], H_R[1]) and _eq(H_Ra[1], H_R[1])
    info = {"holds_act": (rec["rows"]["ans_frz_mlp"]["final"] != rec["rows"]["E0"]["final"] and
                          rec["rows"]["ans_frz_attn"]["final"] != rec["rows"]["E0"]["final"] and
                          not _eq(z_Rm, z_R) and not _eq(z_Ra, z_R)),
            "rnd_reaches_final": not _eq(z_R[r], zA[r])}
    applied = float((H_R[1][r] - HA[1][r]).float().norm())
    for nm, z in (("rnd", z_R), ("rnd_hold_mlp", z_Rm), ("rnd_hold_attn", z_Ra)):
        v = KF.vals_at(z, r, tA, tB)
        rec["rows"][nm] = {"final": v, "dm": r6(margin(v) - margin(cA))}
    rec["rows"]["rnd"].update({"target_norm": r6(n_t), "applied_norm": r6(applied),
                               "norm_ratio": r6(applied / n_t) if n_t > 0 else None})
    rec["identity"], rec["info"] = ident, info
    rec["forwards"] += X.n_fwd - f0
    rec["seconds"] = round(rec["seconds"] + time.time() - t0, 3)
    for k_ in ("cluster", "stratum", "half", "sa", "sb", "src"):
        if k_ in it:
            rec[k_] = it[k_]
    return rec, ident, info


def process(X, model, it, ds):
    return process_d112(X, it) if model == "d112" else process_ouro(X, it, ds)


# ------------------------------------------------------------------ per-item metrics and gates (CPU)
def nld(rec, row):
    D = rec["D"]
    if D is None or D < GAP_E:
        return None
    return rec["rows"][row]["dm"] / D


def gate_check(recs, model):
    """Identity gate before any scored row: every gated identity bit-exact and the holds act on every gate item."""
    fails = []
    for rec in recs:
        for k_ in GATED[model]:
            if not rec["identity"].get(k_):
                fails.append("%s: %s" % (rec["key"], k_))
        for k_ in GATE_ONLY:
            if not rec["info"].get(k_):
                fails.append("%s: %s" % (rec["key"], k_))
    if not recs:
        fails.append("no gate items")
    return {"pass": not fails, "n_items": len(recs), "failures": fails,
            "rule": "identities %s bit-exact and %s on every gate item" % (",".join(GATED[model]), ",".join(GATE_ONLY))}


def identity_failures(ident, model):
    return [k_ for k_ in GATED[model] if not ident.get(k_)]


def sign_gate(recs_by_ds, model, sign_min=SIGN_MIN):
    """Per dataset: mean over items of NLD(positive control) - NLD(patch alone), items with both defined, >= 0.2."""
    rp, rpc = ROWS[model]["patch"], ROWS[model]["pc"]
    res = {"sign_min": sign_min, "rule": "mean(pc - patch alone) >= %.2f on every cell" % sign_min,
           "datasets": {}, "failures": []}
    for ds, recs in recs_by_ds.items():
        gaps = [nld(r, rpc) - nld(r, rp) for r in recs if nld(r, rpc) is not None and nld(r, rp) is not None]
        gap = sum(gaps) / len(gaps) if gaps else None
        ok = gap is not None and gap >= sign_min
        res["datasets"][ds] = {"pc_minus_patch": None if gap is None else round(gap, 6), "n_defined": len(gaps),
                               "n_items": len(recs), "pass": ok}
        if not ok:
            res["failures"].append("%s: pc - patch = %s < %.2f" % (ds, None if gap is None else round(gap, 6),
                                                                  sign_min))
    if not recs_by_ds:
        res["failures"].append("no items")
    res["pass"] = not res["failures"]
    return res


def check_debug_gate(path, plan_sha, model):
    try:
        g = json.load(open(path))
    except (OSError, ValueError, TypeError):
        return ["debug gate file missing or unreadable"], None
    fails = []
    if not g.get("pass"):
        fails.append("debug gate did not pass")
    if g.get("plan_sha256") != plan_sha:
        fails.append("debug gate has another plan sha256")
    if g.get("model") != model:
        fails.append("debug gate is for model %s" % g.get("model"))
    return fails, g


def projected_minutes(spi, n_items):
    """Scored-row minutes of a full run from the debug step's seconds per item (gate items counted twice)."""
    return round(sum(spi[ds] * (n_items[ds] + GATE_N) for ds in n_items) / 60.0, 2)


# ------------------------------------------------------------------ model loading (GPU)
def load_ctx(model, args):
    """-> (X, cfg fields, pins). Stops with exit 2 when a pin or the matched-depth site is missing."""
    if model == "d112":
        import k10_ifm
        import k17_freeze_ifm as KFI
        import k15_smoke as S1
        X = KFI.IfmCtx(D112_NAME)
        bad = check_d112_ctx(X)
        if k10_ifm.NAMES.get(D112_NAME) != D112_CKPT:
            bad.append("checkpoint name %s" % k10_ifm.NAMES.get(D112_NAME))
        if bad:
            die(EXIT_PLAN, "d112 site check failed: %s" % json.dumps(bad))
        cfg = {"checkpoint": D112_CKPT, "attn_backend": X.r.attn_backend, "arch": X.r.arch,
               "n_layers": X.L, "k": X.K, "dtype": str(next(X.m.parameters()).dtype),
               "patch_site": "state at the output of layer %d (0-based; layer %d of %d), position r" % (
                   PATCH_LAYER, PATCH_LAYER + 1, D112_LAYERS),
               "holds": "attention.wo or nffn.fc2 output at position r, layers %d to %d" % (HOLD_FROM, X.e0 - 1),
               "pc_site": "state at the output of layer %d, position r" % (X.e0 - 1),
               "d112_map": X.M.layers_desc, "k15_smoke_md5": KF.md5_file(S1.__file__),
               "k10_ifm_md5": KF.md5_file(k10_ifm.__file__)}
        pins = {"checkpoint": D112_CKPT, "patch_layer": PATCH_LAYER, "hold_from": HOLD_FROM}
        return X, cfg, pins
    ad, b = KF.load_model(os.path.abspath(args.tree), args.mem_fraction, "ouro_1_4b_base")
    X = KF.Ctx(b, os.path.abspath(args.tree))
    tok = ad.tok
    X.enc = lambda t, ans=False: tok(t, add_special_tokens=False)["input_ids"]
    import transformers
    rev = getattr(ad.model.config, "_commit_hash", None)
    prod = KF.prod_hashes(os.path.abspath(args.tree))
    bad = []
    if rev != OURO_REVISION:
        bad.append("revision %s" % rev)
    if transformers.__version__ != OURO_TRANSFORMERS:
        bad.append("transformers %s" % transformers.__version__)
    if b.attn_implementation() != "eager":
        bad.append("attention %s" % b.attn_implementation())
    if str(ad.model.dtype) != "torch.bfloat16":
        bad.append("dtype %s" % ad.model.dtype)
    if prod != OURO_PROD_MD5:
        bad.append("prod md5 differs from the freeze pilot")
    if bad:
        die(EXIT_PLAN, "Ouro pins differ from the freeze plan: %s" % json.dumps(bad))
    cfg = {"repo": ad.repo, "revision": rev, "dtype": str(ad.model.dtype), "attn_implementation": "eager",
           "k": KF.K, "F_ans": X.F_ans, "F_src": X.F_src, "n_freeze_sites": len(X.frz_sites),
           "transformers": transformers.__version__,
           "patch_site": "loop1.loop_end, position r", "holds": "mlp_out or attn_out at r, loops 2 to 4",
           "pc_site": "loop4.loop_end, position r",
           "random_patch": "r's loop-1 end state + ||H_B - H_A|| u, u a unit Gaussian direction seeded per item"}
    pins = {"revision": rev, "transformers": transformers.__version__, "prod_md5": prod}
    return X, cfg, pins


# ------------------------------------------------------------------ run (GPU)
def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def write_hashes(out):
    hs = {}
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if os.path.isfile(p) and (f.startswith("items_") or f.startswith("selection_") or
                                  f in ("config.json", "gate.json", "hash_line.json", "debug_gate.json",
                                        "sign_gate.json", "debug_gate_used.json")):
            hs[f] = KF.sha256_file(p)
    json.dump(hs, open(os.path.join(out, "hashes.json"), "w"), indent=1, sort_keys=True)


def run(args):
    t_start = time.time()
    out = args.out
    os.makedirs(out, exist_ok=True)
    plan_sha = check_plan(args.plan, args.plan_sha256)
    model = args.model
    if KF.sha256_file(os.path.join(args.data, "items_d1.json")) != ITEMS_D1_SHA:
        die(EXIT_PLAN, "data/items_d1.json is not the PI's file")
    dg = None
    if not args.debug:
        bad, dg = check_debug_gate(args.debug_gate, plan_sha, model)
        if bad:
            die(EXIT_GATE, "GATE FAILED (debug gate): %s" % json.dumps(bad))
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))
    X, mcfg, pins = load_ctx(model, args)
    data = data_hashes(args.data, model)
    cfg = {"run": "RUN 1c", "model_key": model, "plan_sha256": plan_sha, "plan_file": os.path.basename(args.plan),
           "code_sha256": KF.sha256_file(os.path.abspath(__file__)), "data_sha256": data, "batch": 1,
           "gap_e": GAP_E, "datasets": ",".join(DATASETS), "debug": args.debug, "gate_n": GATE_N,
           "sign_min": SIGN_MIN, "ref_tol": REF_TOL, "r_min": R_MIN, "full_budget_min": FULL_BUDGET_MIN,
           "gated_identities": list(GATED[model]), "gate_only": list(GATE_ONLY), "torch": torch.__version__,
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    cfg.update(mcfg)
    cfg_path = os.path.join(out, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        for k_ in ("plan_sha256", "code_sha256", "data_sha256", "debug", "model_key", "checkpoint", "revision"):
            if old.get(k_) != cfg.get(k_):
                die(EXIT_PLAN, "resume with a changed config field %s" % k_)
        cfg = old
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=1, sort_keys=True)
    hl = write_hash_line(out, plan_sha, cfg, data, pins)
    print("[run1c] hash line %s" % json.dumps({k: hl[k] for k in ("plan_sha256", "config_sha256")}), flush=True)
    prog_path = os.path.join(out, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"sessions": []}
    sess = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "load_seconds": round(time.time() - t_start, 1)}
    prog["sessions"].append(sess)

    def save_prog():
        sess["seconds"] = round(time.time() - t_start, 1)
        sess["forwards"] = X.n_fwd
        sess["peak_alloc_gb"] = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                                 if torch.cuda.is_available() else None)
        json.dump(prog, open(prog_path + ".tmp", "w"), indent=1)
        os.replace(prog_path + ".tmp", prog_path)

    sels = {}
    for ds in DATASETS:
        sel_path = os.path.join(out, "selection_%s.json" % ds)
        items, info = build_items(X, model, args.data, ds)
        if args.debug:
            items = items[:args.debug]
        info["n_run"] = len(items)
        if os.path.exists(sel_path):
            old = json.load(open(sel_path))
            if [it["key"] for it in old["items"]] != [it["key"] for it in items]:
                die(EXIT_PLAN, "resume with a changed %s item list" % ds)
        json.dump({"dataset": ds, "info": info, "items": items}, open(sel_path + ".tmp", "w"))
        os.replace(sel_path + ".tmp", sel_path)
        sels[ds] = items
        print("[run1c] %s items: %s" % (ds, json.dumps({k_: v for k_, v in info.items() if k_ != "bad"})), flush=True)
    if dg is not None:
        proj = projected_minutes(dg["seconds_per_item"], {ds: len(sels[ds]) for ds in DATASETS})
        json.dump(dict(dg, projected_full_minutes=proj), open(os.path.join(out, "debug_gate_used.json"), "w"),
                  indent=1)
        if proj > FULL_BUDGET_MIN:
            die(EXIT_GATE, "GATE FAILED (NO FIT): projected %.1f scored-row minutes > %.0f" % (proj, FULL_BUDGET_MIN))
        print("[run1c] projected scored-row minutes %.1f (budget %.0f)" % (proj, FULL_BUDGET_MIN), flush=True)

    tg = time.time()
    grecs = []
    for ds in DATASETS:
        for it in sels[ds][:GATE_N]:
            rec, _, _ = process(X, model, it, ds)
            rec["dataset"] = ds
            grecs.append(rec)
    g = gate_check(grecs, model)
    g["seconds"] = round(time.time() - tg, 1)
    g["records"] = grecs
    json.dump(g, open(os.path.join(out, "gate.json"), "w"), indent=1)
    print("[run1c] gate %s (%d items)" % ("PASS" if g["pass"] else "FAIL", g["n_items"]), flush=True)
    save_prog()
    if not g["pass"]:
        die(EXIT_GATE, "GATE FAILED: %s" % json.dumps(g["failures"][:20]))

    for ds in DATASETS:
        rec_path = os.path.join(out, "items_%s.jsonl" % ds)
        done = set()
        for x in load_jsonl(rec_path):
            done.add(x["key"])
        ta, n_new = time.time(), 0
        with open(rec_path, "a") as fh:
            for it in sels[ds]:
                if STOP["flag"]:
                    break
                if it["key"] in done:
                    continue
                rec, ident, _ = process(X, model, it, ds)
                rec["dataset"] = ds
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_new += 1
                bad = identity_failures(ident, model)
                if bad:
                    save_prog()
                    die(EXIT_IDENTITY, "%s item %s identity failed: %s; stopping" % (ds, it["key"], bad))
                if n_new % 20 == 1:
                    print("[run1c] %s %d/%d (%.2fs/item) %.0fs" % (ds, len(done) + n_new, len(sels[ds]),
                                                                  (time.time() - ta) / n_new, time.time() - t_start),
                          flush=True)
                    save_prog()
        save_prog()
        print("[run1c] %s done %d new items %.0fs" % (ds, n_new, time.time() - ta), flush=True)
        if STOP["flag"]:
            die(EXIT_STOPPED, "SIGTERM: stopped after the current item (resume with the same command)")
    sess["finished"] = True
    save_prog()
    recs = {ds: load_jsonl(os.path.join(out, "items_%s.jsonl" % ds)) for ds in DATASETS}
    sg = None
    if model in SIGN_GATE_MODELS:                 # d112 only (PI correction 2026-10-11); Ouro keeps the pilot gates
        sg = sign_gate(recs, model)
        sg.update({"plan_sha256": plan_sha, "model": model, "debug_n": args.debug})
        json.dump(sg, open(os.path.join(out, "sign_gate.json"), "w"), indent=1)
    if args.debug:
        dbg = {"pass": bool(g["pass"] and (sg is None or sg["pass"])), "plan_sha256": plan_sha, "model": model,
               "debug_n": args.debug, "sign": sg["datasets"] if sg else None,
               "failures": sg["failures"] if sg else [],
               "n_items": {ds: len(recs[ds]) for ds in DATASETS},
               "seconds_per_item": {ds: round(sum(r["seconds"] for r in recs[ds]) / max(1, len(recs[ds])), 3)
                                    for ds in DATASETS},
               "load_seconds": sess["load_seconds"]}
        json.dump(dbg, open(os.path.join(out, "debug_gate.json"), "w"), indent=1)
    write_hashes(out)
    if sg is not None:
        print("[run1c] sign gate %s: %s" % ("PASS" if sg["pass"] else "FAIL", json.dumps(sg["datasets"])), flush=True)
    print("[run1c] done %.0fs, %d forwards" % (time.time() - t_start, X.n_fwd), flush=True)
    if sg is not None and not sg["pass"]:
        die(EXIT_GATE, "GATE FAILED (sign gate): %s" % json.dumps(sg["failures"]))


# ------------------------------------------------------------------ summary, results.csv, checks (CPU)
def boot_ratio(num, den, cl, n=NB, seed=0):
    """Ratio of item sums sum(num)/sum(den) with its paired cluster bootstrap (RUN 1's estimator, unchanged)."""
    keep = [i for i in range(len(num)) if num[i] is not None and den[i] is not None]
    if not keep:
        return None
    g = {}
    for i in keep:
        s = g.setdefault(cl[i], [0.0, 0.0])
        s[0] += num[i]
        s[1] += den[i]
    keys = sorted(g, key=str)
    sn = torch.tensor([g[k][0] for k in keys], dtype=torch.float64)
    sd = torch.tensor([g[k][1] for k in keys], dtype=torch.float64)
    if float(sd.sum()) == 0.0:
        return None
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=gen)
    dn, dd = sn[idx].sum(1), sd[idx].sum(1)
    ok = dd != 0
    dist = dn[ok] / dd[ok]
    q = lambda ps: [round(float(x), 4) for x in torch.quantile(dist, torch.tensor(ps, dtype=torch.float64))]
    return {"mean": round(float(sn.sum() / sd.sum()), 4), "ci90": q([0.05, 0.95]), "ci95": q([0.025, 0.975]),
            "n_items": len(num), "n_defined": len(keep), "n_clusters": len(keys)}


def conditions(model):
    return [c for c in ("patch", "patch_hold_mlp", "patch_hold_attn", "rnd", "rnd_hold_mlp", "rnd_hold_attn", "pc",
                        "sz") if c in ROWS[model]]


def cell_stats(recs, ds, model):
    KO = KF.stats_mod()
    dy = ds == "D1"
    rows = ROWS[model]
    c = {"n_items": len(recs), "n_clusters": len(set(str(r["cluster"]) for r in recs)),
         "n_D_ge_2": sum(1 for r in recs if r["D"] is not None and r["D"] >= GAP_E),
         "identity_counts": {k_: sum(1 for r in recs if r["identity"].get(k_)) for k_ in GATED[model]},
         "info_counts": {k_: sum(1 for r in recs if r["info"].get(k_)) for k_ in sorted(recs[0]["info"])} if recs else {},
         "strata": {s: sum(1 for r in recs if r["stratum"] == s) for s in sorted(set(str(r["stratum"]) for r in recs))},
         "nld": {}, "R": {}}
    for cond in conditions(model):
        c["nld"][cond] = KF.cell(KO, recs, lambda r, row=rows[cond]: nld(r, row), dyadic=dy)
    for arm in ("patch", "rnd"):
        if arm not in rows:
            continue
        st = c["nld"][arm]
        c["%s_ge_r_min" % arm] = st is not None and st["mean"] >= R_MIN
        if c["%s_ge_r_min" % arm]:
            den = [nld(r, rows[arm]) for r in recs]
            cl = [str(r["cluster"]) for r in recs]
            for h in ("hold_mlp", "hold_attn"):
                c["R"]["%s_%s" % (arm, h)] = boot_ratio([nld(r, rows["%s_%s" % (arm, h)]) for r in recs], den, cl)
    if model == "ouro":
        nr = [r["rows"]["rnd"]["norm_ratio"] for r in recs if r["rows"]["rnd"].get("norm_ratio") is not None]
        c["rnd_applied_norm_ratio"] = {"mean": round(sum(nr) / len(nr), 6) if nr else None,
                                       "min": min(nr) if nr else None, "max": max(nr) if nr else None}
    okD = [r for r in recs if r["D"] is not None and r["D"] >= GAP_E]
    c["mean_D"] = round(sum(r["D"] for r in okD) / max(1, len(okD)), 4)
    c["bounds"] = "A-side cluster 90%" + (" (two-way subject bound beside it)" if dy else "")
    return c


def ref_compare(cells, model, ref):
    out = {"tol": REF_TOL, "gated": False, "rows": {}}
    for ds, c in cells.items():
        for cond in ("patch", "patch_hold_mlp", "patch_hold_attn"):
            rv = ref.get("%s|%s|%s" % (model, ds, cond))
            st = c["nld"].get(cond)
            if rv is None or st is None:
                continue
            diff = st["mean"] - rv["estimate"]
            out["rows"]["%s|%s" % (ds, cond)] = {"ours": st["mean"], "ref": rv["estimate"], "diff": round(diff, 4),
                                                 "within_tol": abs(diff) <= REF_TOL, "ref_source": rv["source"]}
    return out


SITE_STR = {"d112": {"patch": "r|L%d.resid_post" % PATCH_LAYER, "hold": "r|L%d-L%d.%%s" % (HOLD_FROM, D112_LAYERS - 1),
                     "pc": "r|L%d.resid_post" % (D112_LAYERS - 1), "loop": 1},
            "ouro": {"patch": "r|loop1.loop_end", "hold": "r|loop2-4.%s", "pc": "r|loop4.loop_end", "loop": 1}}
MODEL_NAME = {"d112": D112_NAME, "ouro": "ouro_1_4b_base"}


def results_rows(model, cells):
    """RA-20 item 10 rows (results_common columns), one per (dataset, condition, metric). ci90: A-side cluster
    bootstrap (stored). random_control_estimate: the random arm's NLD under the same hold (Ouro only).
    positive_control_value: the cell's positive-control NLD."""
    rows = []
    S = SITE_STR[model]
    for ds, c in cells.items():
        pc = c["nld"]["pc"]["mean"] if c["nld"].get("pc") else None

        def add(arm, site, metric, st, key, rnd=None):
            if st is None:
                return
            rows.append(RC.row(grid=GRID, card=CARD, script=SCRIPT, model=MODEL_NAME[model], dataset=DS_NAME[ds],
                               arm=arm, site=site, loop=S["loop"], strength=1.0, metric=metric, endpoint="final",
                               item_set="own", n_items=st.get("n_defined", st.get("n_items")),
                               n_clusters=st["n_clusters"], estimate=st["mean"], ci90_lo=st["ci90"][0],
                               ci90_hi=st["ci90"][1], ci90_method="stored", ci95_lo=st["ci95"][0],
                               ci95_hi=st["ci95"][1], random_control_estimate=rnd, positive_control_value=pc,
                               source_key="results.json:cells.%s.%s" % (ds, key)))

        def rmean(cond):
            st = c["nld"].get(cond)
            return st["mean"] if st else None
        for arm in ("patch", "rnd"):
            if arm not in ROWS[model]:
                continue
            add(arm, S["patch"], "NLD", c["nld"][arm], "nld." + arm, rmean("rnd") if arm == "patch" else None)
            for h, comp in (("hold_mlp", "mlp_out"), ("hold_attn", "attn_out")):
                cond = "%s_%s" % (arm, h)
                site = "%s+hold %s" % (S["patch"], S["hold"] % comp)
                add("%s+%s" % (arm, h), site, "NLD", c["nld"][cond], "nld." + cond,
                    rmean("rnd_" + h) if arm == "patch" else None)
                if c.get("%s_ge_r_min" % arm):
                    add("%s+%s" % (arm, h), site, "R", c["R"].get(cond), "R." + cond)
        add("positive_control", S["pc"], "NLD", c["nld"]["pc"], "nld.pc")
        add("structural_zero", S["pc"] + " (source span)", "NLD", c["nld"]["sz"], "nld.sz")
    return rows


def summarize(args):
    out = args.out
    cfg = json.load(open(os.path.join(out, "config.json")))
    model = cfg["model_key"]
    hs_p = os.path.join(out, "hashes.json")
    hs = json.load(open(hs_p)) if os.path.exists(hs_p) else {}
    hash_ok = bool(hs) and all(KF.sha256_file(os.path.join(out, f)) == h for f, h in hs.items())
    prog = json.load(open(os.path.join(out, "progress.json")))
    gate = json.load(open(os.path.join(out, "gate.json")))
    hl = json.load(open(os.path.join(out, "hash_line.json")))
    data, sel = {}, {}
    for ds in DATASETS:
        p = os.path.join(out, "items_%s.jsonl" % ds)
        if os.path.exists(p):
            data[ds] = load_jsonl(p)
            sel[ds] = json.load(open(os.path.join(out, "selection_%s.json" % ds)))
    cells = {ds: cell_stats(recs, ds, model) for ds, recs in data.items() if recs}
    complete = {ds: len(data.get(ds, [])) == len(sel[ds]["items"]) for ds in sel}
    ref = json.load(open(os.path.join(HERE, "data", "ref_values.json")))["values"]
    rows = results_rows(model, cells)
    RC.write_results(rows, os.path.join(out, "results.csv"))
    ident_all = {k_: all(c["identity_counts"][k_] == c["n_items"] for c in cells.values()) for k_ in GATED[model]}
    sg = sign_gate(data, model) if model in SIGN_GATE_MODELS else None
    gpu_s = sum(s.get("seconds", 0) for s in prog["sessions"])
    ck = {"model": MODEL_NAME[model], "plan_sha256": cfg["plan_sha256"], "config_sha256_hash_line": hl["config_sha256"],
          "gate_pass": gate["pass"], "identities_all_items": ident_all,
          "all_identities_bit_exact": all(ident_all.values()) if cells else False,
          "sign_gate": ({"pass": sg["pass"], "datasets": sg["datasets"]} if sg else
                        "not applied (Ouro keeps the freeze pilot's gates)"),
          "n_items": {ds: c["n_items"] for ds, c in cells.items()},
          "n_clusters": {ds: c["n_clusters"] for ds, c in cells.items()}, "complete": complete,
          "debug": cfg["debug"],
          "nld_mean": {ds: {cond: (st["mean"] if st else None) for cond, st in c["nld"].items()}
                       for ds, c in cells.items()},
          "R_printed": {ds: c.get("patch_ge_r_min", False) for ds, c in cells.items()},
          "ref_comparison": ref_compare(cells, model, ref),
          "results_csv_rows": len(rows), "gpu_minutes": round(gpu_s / 60.0, 2), "outputs_hashed": hash_ok}
    if model == "ouro":
        ck["rnd_applied_norm_ratio"] = {ds: c["rnd_applied_norm_ratio"] for ds, c in cells.items()}
    res = {"plan_sha256": cfg["plan_sha256"], "model": MODEL_NAME[model], "cells": cells,
           "selection": {ds: sel[ds]["info"] for ds in sel}, "complete": complete, "checks": ck}
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    json.dump(ck, open(os.path.join(out, "checks.json"), "w"), indent=1)
    write_summary(out, cfg, model, cells, ck)
    print(json.dumps(ck, indent=1))
    return ck


def write_summary(out, cfg, model, cells, ck):
    fmt = KF.fmt
    L = ["# RUN 1c summary (%s)" % MODEL_NAME[model], "",
         "Plan sha256 %s. NLD = dm / D per item (clean gap D >= %.1f), mean over items [90%% percentile A-side cluster "
         "bootstrap, %d resamples, seed 0] (items with NLD defined / clusters). R = NLD under a hold / NLD of the "
         "patch alone, printed only where the patch alone is at least %.1f; otherwise NLD under each hold." % (
             cfg["plan_sha256"], GAP_E, NB, R_MIN), "",
         "| dataset | n items | clusters | patch alone | MLP write held | attention write held | positive control | "
         "structural zero |" + (" random alone | random, MLP held | random, attention held |" if model == "ouro" else ""),
         "|---|---|---|---|---|---|---|---|" + ("---|---|---|" if model == "ouro" else "")]
    for ds, c in cells.items():
        dy = ds == "D1"
        held = []
        for h in ("hold_mlp", "hold_attn"):
            st = c["R"].get("patch_" + h) if c.get("patch_ge_r_min") else None
            held.append(("R " + fmt(st)) if st is not None else ("NLD " + fmt(c["nld"]["patch_" + h], dy)))
        cols = [fmt(c["nld"]["patch"], dy)] + held + [fmt(c["nld"]["pc"]), fmt(c["nld"]["sz"])]
        if model == "ouro":
            cols += [fmt(c["nld"][x], dy) for x in ("rnd", "rnd_hold_mlp", "rnd_hold_attn")]
        L.append("| %s (%s) | %d | %d | %s |" % (DS_NAME[ds], ds, c["n_items"], c["n_clusters"], " | ".join(cols)))
    L += ["", "## Identities, info flags and strata", ""]
    for ds, c in cells.items():
        L.append("- %s: identity counts %s; info counts %s; strata %s; mean D %.3f (%d items with D >= 2)" % (
            ds, json.dumps(c["identity_counts"]), json.dumps(c["info_counts"]), json.dumps(c["strata"]),
            c["mean_D"], c["n_D_ge_2"]))
    L += ["", "## Checks", "", "```", json.dumps(ck, indent=1), "```"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


def main(argv=None):
    a = parse_args(argv)
    if a.hash:
        d = {"plan_sha256": KF.sha256_file(a.plan), "code_sha256": code_hashes(),
             "data_sha256": {m: data_hashes(a.data, m) for m in MODEL_KEYS}}
        print(json.dumps(d, indent=1))
        return 0
    if a.run:
        run(a)
    else:
        summarize(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
