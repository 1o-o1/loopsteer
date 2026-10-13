"""RA-29 (2026-10-12): MCQ late edit, an out-of-sample test of the read-schedule rule on Ouro-1.4B and 2.6B base.
Plan: plan_ra29.md in this folder (its sha256 is checked before anything runs).

Built on the edit-site map harness as RUN 1c runs it: lib/k17_freeze (Ctx over the production hooks in tree/prod,
load_model, prod_hashes, margin), lib/k17_ko240.boot_dist (cluster bootstrap), the production patch op
(interventions.from_spec op "patch", target donor) and RUN 1c's random same-norm recipe. This file adds no hook:
it builds the items (data/items_<ds>.json from make_data.py), the payloads, the arms and the letter-margin readout.

    python ra29.py --hash
    python ra29.py --run --model ouro14|ouro26 --dataset aqua|csqa|arc --out OUT --plan_sha256 SHA --pilot
    python ra29.py --run --model ... --dataset ... --out OUT --plan_sha256 SHA --debug_gate PILOT_OUT/debug_gate.json
    python ra29.py --summarize --out OUT

Exit codes: 2 plan, pin, item or config mismatch; 3 gate failed or missing pilot gate; 4 identity failed on a scored
item; 5 stopped by SIGTERM (rerun the same command to resume).
"""
import argparse
import csv
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

PLAN_DEFAULT = os.path.join(HERE, "plan_ra29.md")
DATA_DEFAULT = os.path.join(HERE, "data")
TREE_DEFAULT = os.path.join(HERE, "tree")
K = KF.K                                   # 4 loops
GAP_E = KF.GAP_E                           # 2.0 logits
NB = KF.NB                                 # 10,000 resamples, seed 0
PILOT_KEEP, PILOT_TRY = 30, 120            # pilot: until 30 items pass the screen and the gap, at most 120 tried
SCREEN_MIN = 80                            # a dataset is kept for the full run only with >= 80 scored items
MODELS = {"ouro14": {"name": "ouro_1_4b_base", "revision": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"},
          "ouro26": {"name": "ouro_2_6b_base", "revision": "1ed04250da1a9936042725d302e81c8fa2ab5abd"},
          # raven family (prelude, recurrent core, coda; K = 32 recurrences), items built with their own tokenizer
          "huginn": {"name": "huginn_0125", "revision": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
                     "family": "raven", "items_tag": "huginn"},
          "mcleish": {"name": "mcleish_llama32_r32", "revision": "a5f6f126e9d9302445346844f1d7d29d111a06eb",
                      "family": "raven", "items_tag": "mcleish"},
          # Ouro Thinking: the Ouro family and arms; items in the grid's own Thinking prompt, built per model
          "think14": {"name": "ouro_1_4b_think", "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
                      "items_tag": "think14"},
          "think26": {"name": "ouro_2_6b_think", "revision": "f1edd81e7ac41355db670500ceaf204e0f73af68",
                      "items_tag": "think26"}}
K_RAVEN = 32
DATASETS = ("aqua", "csqa", "arc", "mmlu", "hellaswag")
TRANSFORMERS_PIN = "4.56.2"
PROD_MD5 = {                                                    # RUN 1c / freeze pilot pins, same tree copy
    "prod/config.py": "41843c88a6ca7a86d502db3c166d7f53",
    "prod/models/ouro.py": "747a09deee4cd38619ed6bf36787313d",
    "prod/steer/adapters/__init__.py": "fe77df8bab1299ff8f25310c7dcefd10",
    "prod/steer/adapters/ouro.py": "f9b7869c7463817c2e452f2c35f9a83a",
    "prod/steer/hooks.py": "943ff4105e0ca7c2e436bc8caddcbf13",
    "prod/steer/interventions.py": "574fca55456f444d1864efe0629a45b6",
    "prod/steer/readouts.py": "94659233915e6240aa118642aa5c12be",
    "prod/steer/sites.py": "7c0234a9a1f3dc1344bd7b4b7df0ac1b"}
EXIT_PLAN, EXIT_GATE, EXIT_IDENTITY, EXIT_STOPPED = 2, 3, 4, 5
# The three pilots (all gates passed: ouro14 aqua, ouro26 aqua, ouro14 csqa) ran under plan 960f88cd, before the
# loop-4 quarter arms were added; their gates are accepted for the full runs under this plan. The new arms pass the
# identity gate on the gate items of every session before any scored row (run, GATE_N).
ACCEPTED_PILOT_PLANS = ("960f88cddbe06767c9333a1ab1a29142bbcce5ef62f1b55d63d1d28b41861451",
                        "9b799c25bf9a1dffa701d8c2e391092ca00cecd3d3ce9c1aae472645671c5d7a",
                        "99994c6ef0201f225d8c45cc8f6e9f0bfcf3bb22597459817b740f5becf6de01")
GATE_N, GATE_TRY = 3, 60                   # identity gate: the first 3 scored items in order (at most 60 tried)
STOP = {"flag": False}

# arm -> (kind, loops, positions, site); site "end" = loop<j>.loop_end; "L0", "Lq1", "Lmid", "Lq3" = the layer
# outputs (RESID_POST) of loop j's layer 0, n_core // 4, n_core // 2 and 3 * n_core // 4, built the way the k15 map's
# sites.matrix_site builds L0@j and Lmid@j (canonical(Site("loop", j, layer, RESID_POST))). Default baseline first.
ARMS = [
    ("lt_L1", "donor", (1,), "r", "end"),
    ("blk_L1", "donor", (1,), "block", "end"),
    ("lt_L2", "donor", (2,), "r", "end"),
    ("lt_L3", "donor", (3,), "r", "end"),
    ("lt_L4first", "donor", (4,), "r", "L0"),
    ("lt_L4q1", "donor", (4,), "r", "Lq1"),
    ("lt_L4mid", "donor", (4,), "r", "Lmid"),
    ("lt_L4q3", "donor", (4,), "r", "Lq3"),
    ("lt_L4", "donor", (4,), "r", "end"),
    ("blk_L2", "donor", (2,), "block", "end"),
    ("blk_L3", "donor", (3,), "block", "end"),
    ("blk_L4", "donor", (4,), "block", "end"),
    ("persist_lt_L1to4", "donor", (1, 2, 3, 4), "r", "end"),
    ("rnd_lt_L1", "random", (1,), "r", "end"),
    ("rnd_lt_L4", "random", (4,), "r", "end"),
    ("rnd_blk_L1", "random", (1,), "block", "end"),
    ("pc", "donor", (4,), "r", "end"),
]
MID_LABELS = ("L0", "Lq1", "Lmid", "Lq3")                      # in layer order inside loop 4


def mid_layers(n_core):
    return {"L0": 0, "Lq1": n_core // 4, "Lmid": n_core // 2, "Lq3": 3 * n_core // 4}


def mid_sites(g, fam="ouro"):
    """Loop-4 layer-output sites; L0 and Lmid are checked against the k15 map's sites.matrix_site. Raven: the k15
    map's own L0@32 and Lmid@32."""
    from prod.steer.sites import Component, Site, canonical, matrix_site
    if fam == "raven":
        return {lab: matrix_site(g, "%s@%d" % (lab, K_RAVEN)) for lab in FAM["raven"]["mid"]}
    lay = mid_layers(g.n_core)
    if len(set(lay.values())) != len(lay) or lay["Lq3"] >= g.n_core - 1:
        raise SystemExit("loop-4 quarter sites need n_core >= 8 distinct layers (n_core %d)" % g.n_core)
    out = {lab: canonical(g, Site("loop", K, lay[lab], Component.RESID_POST)) for lab in MID_LABELS}
    assert out["L0"] == matrix_site(g, "L0@%d" % K) and out["Lmid"] == matrix_site(g, "Lmid@%d" % K)
    return out
ARM_NAMES = [a[0] for a in ARMS]
FREE_DECODE_ARMS = ("lt_L4", "lt_L1", "blk_L1", "swap")
GATED = ("hook_identity", "zero_identity", "pc_exact", "lt4_equals_pc", "persist14_equals_lt4", "blk4_structural_zero",
         "patch_written", "before_patch_unchanged", "causal_prefix_unchanged")

# Raven family (Huginn-0125, McLeish r32), K = 32. The final logits at r are the coda's output over every position, so
# the end of loop 32 at r is not the readout input: the positive control is the coda's last layer output at r
# ("fin", the input of the final norm; lib/k17_freeze_raven.Ctx.s_fin, as RA-25), and the end of loop 32 is a result
# (the k15 map's Llast@32). L0 and Lmid are the k15 map's L0@32 and Lmid@32 (sites.matrix_site).
ARMS_RAVEN = [
    ("lt_L1", "donor", (1,), "r", "end"),
    ("blk_L1", "donor", (1,), "block", "end"),
    ("blk_L2", "donor", (2,), "block", "end"),
    ("blk_L3", "donor", (3,), "block", "end"),
    ("blk_L31", "donor", (31,), "block", "end"),
    ("blk_L32", "donor", (32,), "block", "end"),                  # review B1: the coda's direct read of the block
    ("blk_persist", "donor", tuple(range(1, K_RAVEN + 1)), "block", "end"),
    ("blk_e", "donor", (1,), "block", "pe"),                       # review B2(a): B's prelude output at the block
    ("lt_L31", "donor", (31,), "r", "end"),
    ("lt_L32first", "donor", (32,), "r", "L0"),
    ("lt_L32mid", "donor", (32,), "r", "Lmid"),
    ("lt_L32", "donor", (32,), "r", "end"),
    ("rnd_lt_L1", "random", (1,), "r", "end"),
    ("rnd_lt_L32", "random", (32,), "r", "end"),
    ("rnd_blk_L1", "random", (1,), "block", "end"),
    ("pc", "donor", (32,), "r", "fin"),
    ("sz_fin_blk", "donor", (32,), "block", "fin"),
]
GATED_RAVEN = ("hook_identity", "zero_identity", "pc_exact", "sz_fin_exact", "blk_e_block_equals_B", "patch_written",
               "before_patch_unchanged",
               "causal_prefix_unchanged")
FAM = {"ouro": {"K": K, "arms": ARMS, "mid": MID_LABELS, "gated": GATED,
                "letter_arms": ("lt_L1", "blk_L1", "lt_L3", "lt_L4q1", "lt_L4mid", "lt_L4q3"),
                "rnd_arms": ("rnd_lt_L1", "rnd_lt_L4", "rnd_blk_L1")},
       "raven": {"K": K_RAVEN, "arms": ARMS_RAVEN, "mid": ("L0", "Lmid"), "gated": GATED_RAVEN,
                 "letter_arms": ("lt_L1", "blk_L1", "blk_persist", "lt_L31", "lt_L32mid", "lt_L32"),
                 "rnd_arms": ("rnd_lt_L1", "rnd_lt_L32", "rnd_blk_L1")}}
# Raven predictions (pre-registered before the hash; plan section 1): about 0 = |mean NLD| <= 0.1
ABOUT0, LANDS, REACH_PC = 0.1, 0.5, 0.9
RAVEN_PR_ARMS = ("blk_L1", "blk_L2", "blk_L3", "blk_L31", "blk_L32", "blk_persist", "blk_e", "lt_L1", "lt_L31", "lt_L32")


def fam_of(x):
    """Family of a ctx (attribute family, default ouro) or of a record (field family, default ouro)."""
    if isinstance(x, dict):
        return x.get("family", "ouro")
    return getattr(x, "family", "ouro")


def arm_names(recs):
    fams = {fam_of(r) for r in recs} or {"ouro"}
    assert len(fams) == 1, fams
    return [a[0] for a in FAM[fams.pop()]["arms"]]


def die(code, msg):
    print("[ra29] %s" % msg, flush=True)
    raise SystemExit(code)


def r6(x):
    return None if x is None else round(float(x), 6)


# ------------------------------------------------------------------ items (CPU)
def prefix_files(meta, ds):
    """Exemplar-head file names: one per dataset, or one per exemplar key (MMLU: prefix_mmlu__<key>.txt)."""
    if "heads" in meta:
        return {k: "prefix_%s__%s.txt" % (ds, k) for k in meta["heads"]}
    return {None: meta.get("head_file", "prefix_%s.txt" % ds)}


def items_file(ds, tag=None):
    return "items_%s.json" % ds if not tag else "items_%s__%s.json" % (ds, tag)


def load_items(data_dir, ds, tag=None):
    """-> (meta, head, built). head is the exemplar text, or a dict key -> text when items name their own head.
    tag: a tokenizer family other than Ouro (items_<ds>__<tag>.json)."""
    d = json.load(open(os.path.join(data_dir, items_file(ds, tag)), encoding="utf-8"))
    meta = d["meta"]
    heads = {}
    for k, fn in prefix_files(meta, ds).items():
        heads[k] = open(os.path.join(data_dir, fn), encoding="utf-8", newline="").read()
        want = meta["heads"][k] if k is not None else meta["head_sha256"]
        if hashlib.sha256(heads[k].encode("utf-8")).hexdigest() != want:
            die(EXIT_PLAN, "%s does not match its items file" % fn)
    built = sorted((it for it in d["items"] if not it["drop"]), key=lambda it: it["order"])
    return meta, (heads[None] if None in heads else heads), built


def ids_sha(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode("utf-8")).hexdigest()


def materialize(enc, meta, head, it):
    """Token-level item; exit 2 when the ids differ from the ones make_data.py stored (tokenizer drift)."""
    tail = list(meta["suffix_ids"]) + list(meta["cue_ids"])
    if isinstance(head, dict):
        head = head[it["head"]]
    bos = [meta["bos_id"]] if meta.get("bos_id") is not None else []
    A = bos + enc(head + it["input_A"] + meta["a_prefix"]) + tail
    B = bos + enc(head + it["input_B"] + meta["a_prefix"]) + tail
    if ids_sha(A) != it["ids_A_sha256"] or ids_sha(B) != it["ids_B_sha256"] or len(A) != it["T"]:
        die(EXIT_PLAN, "item %s: token ids differ from data/items (tokenizer drift)" % it["key"])
    b0, b1 = it["block"]
    return {"key": it["key"], "A": A, "B": B, "r": it["r"], "block": list(range(b0, b1)),
            "tA": meta["letter_ids"][it["lA"]], "tB": meta["letter_ids"][it["lB"]], "lA": it["lA"], "lB": it["lB"],
            "letter_ids": sorted(meta["letter_ids"][l] for l in "abcde"[:it["n_opt"]])}


def rnd_rows(model, ds, key, arm, n, d):
    """n unit Gaussian rows, float32, one draw per item and arm: seed = first 15 hex digits of
    sha256('ra29:<model>:<ds>:<key>:<arm>') (RUN 1c's recipe with the arm in the key)."""
    s = int(hashlib.sha256(("ra29:%s:%s:%s:%s" % (model, ds, key, arm)).encode("utf-8")).hexdigest()[:15], 16)
    g = torch.Generator(device="cpu").manual_seed(s)
    u = torch.randn(n, d, generator=g, dtype=torch.float32)
    return u / u.norm(dim=1, keepdim=True)


def margin_at(z, r, tA, tB):
    return float(z[r, tB]) - float(z[r, tA])


def writes(tok, it):
    if tok == it["tB"]:
        return "B"
    if tok == it["tA"]:
        return "A"
    return "other_letter" if tok in it["letter_ids"] else "non_letter"


# ------------------------------------------------------------------ one item (GPU; CPU fake in the tests)
@torch.no_grad()
def process_item(X, it, model, ds, gap_min=GAP_E, dtype=None):
    """Screen, then every arm of plan section 5 on one item. -> record (with 'drop' set when screened out)."""
    I, P = X.I, X.Positions
    fam = fam_of(X)
    F = FAM[fam]
    Kf, MIDS = F["K"], F["mid"]
    dtype = dtype or getattr(X, "dtype", None) or X.W.dtype
    t0, f0 = time.time(), X.n_fwd
    A, B, r, blk, tA, tB, key = it["A"], it["B"], it["r"], it["block"], it["tA"], it["tB"], it["key"]
    T = len(A)
    pre = blk[0] - 1                                        # the token before the block (causality check)
    S = sorted(set(blk) | {r, pre})
    ix = {p: i for i, p in enumerate(S)}
    POS = {"r": [r], "block": list(blk)}
    ids_A = torch.tensor([A], device=X.dev)
    ids_B = torch.tensor([B], device=X.dev)
    rec = {"key": key, "model": model, "dataset": ds, "lA": it["lA"], "lB": it["lB"], "T": T, "r": r,
           "block": [blk[0], blk[-1] + 1], "drop": None, "arms": {}, "seconds_arm": {}}
    if fam != "ouro":
        rec["family"] = fam

    def row(payload=None):
        return X.RowInfo(key=key, prompt_len=T, spans={"answer": (r,)}, payload=payload or {})

    def le_rec():
        return X.Recorder(id="le", sites=tuple(X.s_le), positions=P("abs", abs=tuple(S)), what="state", when="post")

    def H_of(rr):
        return {j: rr[(key, "le", X.s_le[j - 1].key())] for j in range(1, Kf + 1)}

    s_mid = mid_sites(X.g, fam)

    def mid_rec():
        return X.Recorder(id="mid", sites=tuple(s_mid[l] for l in MIDS), positions=P("abs", abs=(r,)),
                          what="state", when="post")

    def fin_rec():                                       # raven: the coda's last layer output (the readout input)
        return X.Recorder(id="fin", sites=(X.s_fin,), positions=P("abs", abs=tuple(S)), what="state", when="post")

    s_pe = I.expand("input.prelude_out", {"kind": "once"}, X.g)[0] if fam == "raven" else None

    def pe_rec():                                        # raven: the prelude output (review B2(a))
        return X.Recorder(id="pe", sites=(s_pe,), positions=P("abs", abs=tuple(S)), what="state", when="post")

    def recs_all():
        return [le_rec(), mid_rec()] + ([fin_rec(), pe_rec()] if fam == "raven" else [])

    def M_of(rr):
        out = {lab: rr[(key, "mid", s_mid[lab].key())] for lab in MIDS}
        if fam == "raven":
            out["fin"] = rr[(key, "fin", X.s_fin.key())]
            out["pe"] = rr[(key, "pe", s_pe.key())]
        return out

    def site_of(label, j):
        if label == "end":
            return X.s_le[j - 1]
        if label == "pe":
            return s_pe
        return X.s_fin if label == "fin" else s_mid[label]

    z_nh = (X.nohook(ids_A, key) if fam == "raven" else X.nohook(ids_A))[0]
    zA, rA = X.run(ids_A, row(), recs=recs_all())
    zA = zA[0]
    HA, MA = H_of(rA), M_of(rA)
    rec["screen_A_top1"] = int(zA[r].argmax())
    if rec["screen_A_top1"] != tA:
        rec["drop"] = "screen_A"
    zB, rB = X.run(ids_B, row(), recs=recs_all())
    zB = zB[0]
    HB, MB = H_of(rB), M_of(rB)
    rec["screen_B_top1"] = int(zB[r].argmax())
    mA, mB = margin_at(zA, r, tA, tB), margin_at(zB, r, tA, tB)
    D = mB - mA
    rec.update({"mA": r6(mA), "mB": r6(mB), "D": r6(D)})
    if fam == "ouro":                                  # raven: the coda reads every position, no one-row read-out
        rec["clean_loop_margin_A"] = [r6(margin_at(X.readouts({j: HA[j][ix[r]:ix[r] + 1]})[j], 0, tA, tB))
                                      for j in range(1, K + 1)]
        rec["clean_loop_margin_B"] = [r6(margin_at(X.readouts({j: HB[j][ix[r]:ix[r] + 1]})[j], 0, tA, tB))
                                      for j in range(1, K + 1)]
    if rec["drop"] is None and rec["screen_B_top1"] != tB:
        rec["drop"] = "screen_B"
    if rec["drop"] is None and D < gap_min:
        rec["drop"] = "gap_below_2"
    rec["identity"] = {"hook_identity": bool(torch.equal(zA, z_nh))}
    if rec["drop"]:
        rec["forwards"], rec["seconds"] = X.n_fwd - f0, round(time.time() - t0, 3)
        return rec

    sel = {name: [ix[p] for p in POS[name]] for name in POS}

    def payload_for(name, kind, j, pos, label):
        if label in ("fin", "pe"):
            return MB[label][sel[pos]]                      # donor only (raven readout input, prelude output)
        if label != "end":
            return MB[label]                                # donor only; positions = r
        if kind == "donor":
            return HB[j][sel[pos]]
        dA = HA[j][sel[pos]]
        n = (HB[j][sel[pos]] - dA).norm(dim=1, keepdim=True)
        u = rnd_rows(model, ds, key, name, len(sel[pos]), dA.shape[1]).to(dA.device)
        return (dA + n * u).to(dtype).float()             # built in the model dtype (RA-28 lesson)

    def arm_run(name, kind, loops, pos, label, a=1.0):
        ivs, pay = [], {}
        for j in loops:
            iid = "%s_L%d" % (name, j)
            sk = site_of(label, j).key()
            ivs.append(I.from_spec({"id": iid, "op": "patch", "site": sk, "a": a, "target": {"kind": "donor"},
                                    "positions": {"kind": "abs", "abs": POS[pos]}}, X.g))
            pay[iid] = {sk: payload_for(name, kind, j, pos, label)}
        ta = time.time()
        z, rr = X.run(ids_A, row(pay), ivs=ivs, recs=recs_all())
        return z[0], H_of(rr), M_of(rr), pay, ivs, time.time() - ta

    z0, H0, _, _, _, _ = arm_run("zero", "donor", (1,), "r", "end", a=0.0)
    ok = {"zero_identity": bool(torch.equal(z0, zA)), "patch_written": True, "before_patch_unchanged": True,
          "causal_prefix_unchanged": True}
    Z, HS = {}, {}
    for name, kind, loops, pos, label in F["arms"]:
        z, H, Mz, pay, ivs, sec = arm_run(name, kind, loops, pos, label)
        Z[name] = z
        HS[name] = H
        P_idx = sel[pos]
        for j, iv in zip(loops, ivs):
            got = H[j][P_idx] if label == "end" else (Mz[label][P_idx] if label in ("fin", "pe") else Mz[label])
            if not torch.equal(got, pay[iv.id][site_of(label, j).key()]):
                ok["patch_written"] = False
        if loops[0] == Kf and pos == "r":                    # last-loop sites that run before this write are unchanged
            earlier = MIDS if label in ("end", "fin") else MIDS[:MIDS.index(label)]
            if not all(torch.equal(Mz[l], MA[l]) for l in earlier):
                ok["before_patch_unchanged"] = False
        if label == "fin" and not all(torch.equal(H[i], HA[i]) for i in range(1, Kf + 1)):
            ok["before_patch_unchanged"] = False             # the coda write leaves every loop state unchanged
        j0 = loops[0]
        others = [i for i in range(len(S)) if i not in P_idx]
        for i in range(1, j0):
            if not torch.equal(H[i], HA[i]):
                ok["before_patch_unchanged"] = False
        if label != "pe" and others and not torch.equal(H[j0][others], HA[j0][others]):
            ok["before_patch_unchanged"] = False         # (pe: r and the later positions change from loop 1 on)
        if not all(torch.equal(H[i][ix[pre]], HA[i][ix[pre]]) for i in range(1, Kf + 1)):
            ok["causal_prefix_unchanged"] = False
        m = margin_at(z, r, tA, tB)
        top = int(z[r].argmax())
        a = {"m": r6(m), "dm": r6(m - mA), "nld": r6((m - mA) / D), "top1": top, "writes": writes(top, it)}
        if kind == "random":
            dl = torch.stack([pay[iv.id][X.s_le[j - 1].key()] for j, iv in zip(loops, ivs)]) - \
                torch.stack([HA[j][P_idx] for j in loops])
            tg = torch.stack([HB[j][P_idx] - HA[j][P_idx] for j in loops])
            tn = float(tg.norm())
            a.update({"applied_norm": r6(dl.norm()), "target_norm": r6(tn),
                      "norm_ratio": r6(float(dl.norm()) / tn) if tn > 0 else None})
        if fam == "raven":                               # review B2(b): how much of B's loop-32 state the arm holds
            for nm_, idx in (("ratio_blk32", sel["block"]), ("ratio_r32", sel["r"])):
                den = float((HB[Kf][idx] - HA[Kf][idx]).norm())
                a[nm_] = r6(float((H[Kf][idx] - HA[Kf][idx]).norm()) / den) if den > 0 else None
        rec["arms"][name] = a
        rec["seconds_arm"][name] = round(sec, 4)
    top_B = int(zB[r].argmax())
    rec["arms"]["swap"] = {"m": r6(mB), "dm": r6(D), "nld": 1.0, "top1": top_B, "writes": writes(top_B, it)}
    rec["arms"]["clean_A"] = {"m": r6(mA), "dm": 0.0, "nld": 0.0, "top1": int(zA[r].argmax()),
                              "writes": writes(int(zA[r].argmax()), it)}
    ok["pc_exact"] = bool(torch.equal(Z["pc"][r], zB[r])) and rec["arms"]["pc"]["nld"] == 1.0
    if fam == "raven":
        ok["sz_fin_exact"] = bool(torch.equal(Z["sz_fin_blk"][r], zA[r]))
        ok["blk_e_block_equals_B"] = all(torch.equal(HS["blk_e"][j][sel["block"]], HB[j][sel["block"]])
                                         for j in range(1, Kf + 1))
    else:
        ok.update({"lt4_equals_pc": bool(torch.equal(Z["lt_L4"], Z["pc"])),
                   "persist14_equals_lt4": bool(torch.equal(Z["persist_lt_L1to4"], Z["lt_L4"])),
                   "blk4_structural_zero": bool(torch.equal(Z["blk_L4"][r], zA[r]))})
    rec["identity"].update(ok)
    rec["forwards"], rec["seconds"] = X.n_fwd - f0, round(time.time() - t0, 3)
    return rec


def identity_failures(rec):
    return [k for k in FAM[fam_of(rec)]["gated"] if not rec["identity"].get(k)] if rec["drop"] is None else \
        ([] if rec["identity"].get("hook_identity") else ["hook_identity"])


# ------------------------------------------------------------------ statistics (CPU)
def boot(xs, keys):
    import k17_ko240 as KO
    b = KO.boot_dist(xs, keys, NB, 0)
    if b is None:
        return None
    return {"mean": round(b[0], 4), "ci90": KO.q(b[1], [0.05, 0.95]), "n_items": b[2]}


def cells(recs):
    kept = [r for r in recs if r["drop"] is None]
    keys = [r["key"] for r in kept]
    out = {}
    names = arm_names(recs)
    fam = fam_of(recs[0]) if recs else "ouro"
    for arm in names + ["swap", "clean_A"]:
        c = boot([r["arms"][arm]["nld"] for r in kept], keys) if kept else None
        if c is not None:
            c["writes_B"] = round(sum(r["arms"][arm]["writes"] == "B" for r in kept) / len(kept), 4)
            if arm.startswith("rnd"):
                nr = [r["arms"][arm]["norm_ratio"] for r in kept if r["arms"][arm].get("norm_ratio") is not None]
                c["norm_ratio_min_max"] = [min(nr), max(nr)] if nr else None
            for rk in ("ratio_blk32", "ratio_r32"):                # raven contraction ratios (review B2(b))
                xs = [r["arms"][arm][rk] for r in kept if r["arms"][arm].get(rk) is not None]
                if xs:
                    c[rk + "_mean"] = round(sum(xs) / len(xs), 4)
        out[arm] = c
    if fam == "raven" and kept:                                    # review B1: persistence beyond the coda read
        out["blk_persist_minus_blk_L32"] = boot([r["arms"]["blk_persist"]["nld"] - r["arms"]["blk_L32"]["nld"]
                                                 for r in kept], keys)
    by_letter = {}
    for l in "abcde":
        sub = [r for r in kept if r["lB"] == l]
        if sub:
            by_letter[l] = {arm: round(sum(r["arms"][arm]["nld"] for r in sub) / len(sub), 4)
                            for arm in FAM[fam]["letter_arms"]}
            by_letter[l]["n"] = len(sub)
    return out, by_letter


def counts(recs, n_built):
    drops = {}
    for r in recs:
        if r["drop"]:
            drops[r["drop"]] = drops.get(r["drop"], 0) + 1
    kept = sum(1 for r in recs if r["drop"] is None)
    return {"n_built": n_built, "n_tried": len(recs), "n_kept": kept, "drops": drops,
            "lB_kept": {l: sum(1 for r in recs if r["drop"] is None and r["lB"] == l) for l in "abcde"}}


def timing(recs, n_built):
    kept = [r for r in recs if r["drop"] is None]
    sec_try = sum(r["seconds"] for r in recs) / max(1, len(recs))
    arm_sec = {a: round(sum(r["seconds_arm"][a] for r in kept) / max(1, len(kept)), 4) for a in arm_names(recs)} \
        if kept else {}
    return {"seconds_per_tried_item": round(sec_try, 3),
            "seconds_per_kept_item": round(sum(r["seconds"] for r in kept) / max(1, len(kept)), 3),
            "seconds_per_arm_and_item": arm_sec,
            "full_card_hours_at_this_rate": round(sec_try * n_built / 3600.0, 3)}


RESULT_FIELDS = ["model", "dataset", "key", "lA", "lB", "arm", "kind", "loops", "positions", "site", "D", "m", "dm", "nld",
                 "top1", "writes", "norm_ratio", "seconds"]


def result_rows(recs):
    rows = []
    for r in recs:
        if r["drop"]:
            continue
        spec = {a[0]: a for a in FAM[fam_of(r)]["arms"]}
        for arm in [a[0] for a in FAM[fam_of(r)]["arms"]] + ["swap", "clean_A"]:
            a = r["arms"][arm]
            s = spec.get(arm)
            rows.append({"model": r["model"], "dataset": r["dataset"], "key": r["key"], "lA": r["lA"], "lB": r["lB"],
                         "arm": arm, "kind": s[1] if s else "clean", "loops": "-".join(map(str, s[2])) if s else "",
                         "positions": s[3] if s else "", "site": s[4] if s else "", "D": r["D"], "m": a["m"], "dm": a["dm"], "nld": a["nld"],
                         "top1": a["top1"], "writes": a["writes"], "norm_ratio": a.get("norm_ratio", ""),
                         "seconds": r["seconds_arm"].get(arm, "")})
    return rows


def write_outputs(out, recs, n_built, meta_info):
    res, by_letter = cells(recs)
    ct = counts(recs, n_built)
    tm = timing(recs, n_built)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for row_ in result_rows(recs):
            w.writerow(row_)
    pv = {}
    fam = fam_of(recs[0]) if recs else "ouro"
    n_kept = sum(1 for r in recs if r["drop"] is None)
    if fam == "raven" and n_kept < SCREEN_MIN:                      # review N1
        pv = {"not_scored": "fewer than %d scored items (%d)" % (SCREEN_MIN, n_kept)}
    elif fam == "raven" and all(res.get(a) for a in RAVEN_PR_ARMS):
        def about0(arms):
            return all(abs(res[a]["mean"]) <= ABOUT0 for a in arms)

        def ci_in(arms):                                            # review N2: the 90 percent CI inside +-0.1
            return all(-ABOUT0 <= res[a]["ci90"][0] and res[a]["ci90"][1] <= ABOUT0 for a in arms)
        single = ("blk_L1", "blk_L2", "blk_L3", "blk_L31")
        pv = {"PR1_single_loop_block_about_0": about0(single), "PR1_single_loop_block_ci_inside_0.1": ci_in(single),
              "PR1_blk_L32_ge_0.5": res["blk_L32"]["mean"] >= LANDS,
              "PR1_blk_persist_ge_0.5": res["blk_persist"]["mean"] >= LANDS,
              "PR1_blk_e_ge_0.9": res["blk_e"]["mean"] >= REACH_PC,
              "PR2_lt_L1_and_L31_about_0": about0(("lt_L1", "lt_L31")),
              "PR2_lt_L1_and_L31_ci_inside_0.1": ci_in(("lt_L1", "lt_L31")),
              "PR2_lt_L32_end_ge_0.5": res["lt_L32"]["mean"] >= LANDS,
              "contraction_ratio_blk32": {a: res[a].get("ratio_blk32_mean") for a in single + ("blk_L32", "blk_persist",
                                                                                              "blk_e")},
              "contraction_ratio_r32": {a: res[a].get("ratio_r32_mean") for a in ("lt_L1", "lt_L31", "lt_L32")},
              "blk_persist_minus_blk_L32": res.get("blk_persist_minus_blk_L32"),
              "note": "about 0 = |mean NLD| <= %.1f; lands = mean NLD >= %.1f; reaches the positive control = mean "
                      "NLD >= %.1f; pc (coda output at r) is the positive-control identity; the raven rows are read "
                      "as the refresh-model case of the placement rule, not as a commit-point measurement"
                      % (ABOUT0, LANDS, REACH_PC)}
        pv["PR1_holds"] = (pv["PR1_single_loop_block_about_0"] and pv["PR1_blk_L32_ge_0.5"] and
                           pv["PR1_blk_persist_ge_0.5"] and pv["PR1_blk_e_ge_0.9"])
        pv["PR2_holds"] = pv["PR2_lt_L1_and_L31_about_0"] and pv["PR2_lt_L32_end_ge_0.5"]
    elif all(res.get(a) for a in ("lt_L1", "lt_L3", "lt_L4mid", "blk_L1")):
        late = max(res["lt_L3"]["mean"], res["lt_L4mid"]["mean"])          # P1 as frozen (lt_L4q1 / q3 locate)
        pv = {"P1_lt_L3_or_L4mid_ge_0.5": late >= 0.5, "P1_lt_L1_below_0.2": res["lt_L1"]["mean"] < 0.2,
              "P1_holds": late >= 0.5 and res["lt_L1"]["mean"] < 0.2,
              "P2_blk_L1_ge_0.8": res["blk_L1"]["mean"] >= 0.8,
              "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, "
                      "not a result. Facts reference at the same sites (k15 map): at most 0.17."}
    json.dump({"cells": res, "nld_by_gold_letter_B": by_letter, "predictions": pv, "n_boot": NB, "seed": 0,
               "ci": "90 percent cluster bootstrap over items (each item its own cluster)"},
              open(os.path.join(out, "results.json"), "w"), indent=1)
    idf = {}
    for r in recs:
        for k in identity_failures(r):
            idf[k] = idf.get(k, 0) + 1
    chk = dict(meta_info, counts=ct, timing=tm, identity_failures=idf,
               gates={"pc_exact_all": all(r["identity"].get("pc_exact") for r in recs if r["drop"] is None),
                      "hook_identity_all": all(r["identity"].get("hook_identity") for r in recs),
                      "kept_ge_80": ct["n_kept"] >= SCREEN_MIN})
    json.dump(chk, open(os.path.join(out, "checks.json"), "w"), indent=1)
    write_summary(out, res, by_letter, ct, tm, pv, dict(meta_info, family=fam, arm_names=arm_names(recs)))
    return res, chk


def write_summary(out, res, by_letter, ct, tm, pv, meta_info):
    L = ["# RA-29 %s %s" % (meta_info.get("model"), meta_info.get("dataset")), "",
         "Items: %d built, %d tried, %d kept; drops %s; new gold letter of the kept items %s." % (
             ct["n_built"], ct["n_tried"], ct["n_kept"], json.dumps(ct["drops"]), json.dumps(ct["lB_kept"])), "",
         "| arm | NLD mean | 90% CI | writes B's letter |", "|---|---|---|---|"]
    fam = meta_info.get("family", "ouro")
    for arm in meta_info.get("arm_names", ARM_NAMES) + ["swap", "clean_A"]:
        c = res.get(arm)
        if c:
            L.append("| %s | %.3f | [%.3f, %.3f] | %.3f |" % (arm, c["mean"], c["ci90"][0], c["ci90"][1], c["writes_B"]))
    nr = {a: res[a].get("norm_ratio_min_max") for a in FAM[fam]["rnd_arms"] if res.get(a)}
    if fam == "raven":
        note = ("Default baseline first (lt_L1: last-token donor patch at the end of loop 1). pc writes the coda's "
                "output at r (the readout input) and equals the swap at r by construction; sz_fin_blk (the same write "
                "at the block) is a structural zero. lt_L32 (end of loop 32) is a result: the coda still reads the "
                "block positions. blk_e writes B's prelude output at the block (the block then runs as in B in every "
                "loop); blk_L32 is the coda's direct read of the block. The raven rows are read as the refresh-model "
                "case of the placement rule, not as a commit-point measurement.")
    else:
        note = ("Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the "
                "readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every "
                "position by causality at r (r is the last position), so it carries no separate information.")
    L += ["", note, "",
          "Random controls, norm ratio min and max over items: %s" % json.dumps(nr), "",
          "Mean NLD by the gold letter of prompt B (check of the letter bias):", ""]
    la = [a for a in FAM[fam]["letter_arms"] if any(a in v for v in by_letter.values())]
    L += ["| B's letter | n | " + " | ".join(la) + " |", "|---|---|" + "---|" * len(la)]
    for l in sorted(by_letter):
        L.append("| %s | %d | " % (l, by_letter[l]["n"]) + " | ".join("%.3f" % by_letter[l][a] for a in la) + " |")
    L += ["", "Predictions: %s" % json.dumps(pv), "", "Timing: %s" % json.dumps(tm)]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ model (GPU)
RAVEN_MD5 = {                                                   # the RA-25 tree's raven files (same tree copy)
    "prod/models/raven.py": "7af33a1665a573a6ecf2cb34c871efe2",
    "prod/models/huginn.py": "682e838692d02fa4270f3d3730bf5f44",
    "prod/models/mcleish.py": "0b3391bbe641176bef2c99ecf3b091be",
    "prod/steer/adapters/raven.py": "61e125bc3f11e897fd382d0670bd2e3d"}


def raven_ctx(ad, b, tree):
    """lib/k17_freeze_raven.Ctx (RA-25's raven harness: s_le, s_fin, x0 table, nohook) with the Ouro-style run()."""
    import k17_freeze_raven as KFR
    KFR.TREE[0] = tree

    class RavenX(KFR.Ctx):
        family = "raven"

        def run(self, ids, row, ivs=(), recs=()):
            z, out, _ = KFR.Ctx.run(self, ids, row, ivs=ivs, recs=recs)
            return z, out

    return RavenX(ad, b)


def load_ctx_raven(model, tree, mem_fraction):
    import transformers
    spec = MODELS[model]
    if tree not in sys.path:
        sys.path.insert(0, tree)
    from prod import models as PM
    from prod.steer.adapters import bind
    ad = PM.get(spec["name"])
    ad.load()
    ad.set_depth(K_RAVEN)
    b = bind(ad)
    X = raven_ctx(ad, b, tree)
    tok = ad.tok
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    rev = getattr(ad.model.config, "_commit_hash", None)
    files = [f for f in list(PROD_MD5) + list(RAVEN_MD5) if "ouro" not in f]
    prod = {f: KF.md5_file(os.path.join(tree, f)) for f in files}
    want = dict(list(PROD_MD5.items()) + list(RAVEN_MD5.items()))
    bad = []
    if rev != spec["revision"]:
        bad.append("revision %s" % rev)
    if transformers.__version__ != TRANSFORMERS_PIN:
        bad.append("transformers %s" % transformers.__version__)
    if any(prod[f] != want[f] for f in files):
        bad.append("prod md5 differs from the RA-25 tree")
    if bad:
        die(EXIT_PLAN, "pins differ: %s" % json.dumps(bad))
    X.tok = tok
    return X, enc, {"repo_name": spec["name"], "revision": rev, "transformers": transformers.__version__,
                    "prod_md5": prod, "dtype": str(X.dtype).replace("torch.", ""), "k": K_RAVEN,
                    "x0_policy": "fixed_table", "site_final_input": X.s_fin.key()}


def load_ctx(model, tree, mem_fraction):
    import transformers
    spec = MODELS[model]
    if spec.get("family") == "raven":
        return load_ctx_raven(model, tree, mem_fraction)
    ad, b = KF.load_model(tree, mem_fraction, spec["name"])
    X = KF.Ctx(b, tree)
    tok = ad.tok
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    rev = getattr(ad.model.config, "_commit_hash", None)
    prod = KF.prod_hashes(tree)
    bad = []
    if rev != spec["revision"]:
        bad.append("revision %s" % rev)
    if transformers.__version__ != TRANSFORMERS_PIN:
        bad.append("transformers %s" % transformers.__version__)
    if b.attn_implementation() != "eager":
        bad.append("attention %s" % b.attn_implementation())
    if str(ad.model.dtype) != "torch.bfloat16":
        bad.append("dtype %s" % ad.model.dtype)
    if prod != PROD_MD5:
        bad.append("prod md5 differs from the RUN 1c tree")
    if bad:
        die(EXIT_PLAN, "pins differ: %s" % json.dumps(bad))
    X.tok = tok
    return X, enc, {"repo_name": spec["name"], "revision": rev, "transformers": transformers.__version__,
                    "prod_md5": prod, "dtype": "bfloat16", "attn": "eager", "k": K}


def block_map(X, it, n_show=None):
    tok = getattr(X, "tok", None)
    conv = (lambda ids: tok.convert_ids_to_tokens(ids)) if tok is not None else (lambda ids: [str(i) for i in ids])
    b = it["block"]
    return {"key": it["key"], "block": [b[0], b[-1] + 1], "r": it["r"],
            "A_tokens": conv([it["A"][p] for p in b]), "B_tokens": conv([it["B"][p] for p in b]),
            "r_token_A": conv([it["A"][it["r"]]])[0], "pre_token": conv([it["A"][b[0] - 1]])[0]}


# ------------------------------------------------------------------ run
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="ra29")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--hash", action="store_true")
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--out")
    ap.add_argument("--plan", default=PLAN_DEFAULT)
    ap.add_argument("--plan_sha256")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--tree", default=TREE_DEFAULT)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--debug_gate")
    ap.add_argument("--mem_fraction", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.run and not (a.model and a.dataset and a.out and a.plan_sha256):
        ap.error("--run needs --model, --dataset, --out and --plan_sha256")
    if a.run and not a.pilot and not a.debug_gate:
        ap.error("a full --run needs --debug_gate (the pilot's debug_gate.json)")
    if a.summarize and not a.out:
        ap.error("--summarize needs --out")
    return a


def code_hashes():
    fs = {"ra29.py": os.path.abspath(__file__)}
    for f in sorted(os.listdir(LIB)):
        if f.endswith(".py"):
            fs["lib/" + f] = os.path.join(LIB, f)
    return {k: KF.sha256_file(p) for k, p in fs.items()}


def load_jsonl(p, repair=False):
    """Records of a jsonl file. A trailing line that does not parse (a write cut by SIGKILL) is skipped; with
    repair=True the file is rewritten without it, so the next append starts on a clean line."""
    if not os.path.exists(p):
        return []
    with open(p) as f:
        lines = [x for x in f if x.strip()]
    out = []
    for i, x in enumerate(lines):
        try:
            out.append(json.loads(x))
        except ValueError:
            if i != len(lines) - 1:
                raise
            print("[ra29] skipped a trailing line of %s that does not parse" % os.path.basename(p), flush=True)
            if repair:
                with open(p + ".tmp", "w") as g:
                    for r in out:
                        g.write(json.dumps(r) + "\n")
                os.replace(p + ".tmp", p)
    return out


def check_tail_ids(enc, meta):
    """B3: the cue, the " (" token and the letters re-encoded under this host's tokenizer; exit 2 on a mismatch."""
    bad = []
    pre = list(enc(meta["suffix_pre_text"])) if meta.get("suffix_pre_text") else []   # Thinking: </think> first
    if pre + list(enc(meta["suffix_text"])) != list(meta["suffix_ids"]):
        bad.append("suffix")
    if list(enc(" (")) != list(meta["cue_ids"]):
        bad.append("cue")
    labels = meta.get("labels", {})
    for l in "abcde":
        if list(enc(labels.get(l, l))) != [meta["letter_ids"][l]]:
            bad.append("letter %s" % labels.get(l, l))
    if bad:
        die(EXIT_PLAN, "token ids differ from data/items on this host: %s" % bad)


def run(args, loader=None, process=None):
    """loader and process are test seams (CPU fakes); the card run passes neither."""
    t_start = time.time()
    out = args.out
    os.makedirs(out, exist_ok=True)
    plan_sha = KF.sha256_file(args.plan)
    if plan_sha != args.plan_sha256:
        die(EXIT_PLAN, "plan sha256 mismatch: file %s, given %s" % (plan_sha, args.plan_sha256))
    if not args.pilot:
        try:
            g = json.load(open(args.debug_gate))
        except (OSError, ValueError, TypeError):
            die(EXIT_GATE, "GATE FAILED: pilot debug_gate.json missing or unreadable")
        if not g.get("pass") or g.get("plan_sha256") not in (plan_sha,) + ACCEPTED_PILOT_PLANS or g.get("model") != args.model or \
                g.get("dataset") != args.dataset:
            die(EXIT_GATE, "GATE FAILED: pilot gate did not pass for this plan, model and dataset")
    tag = MODELS[args.model].get("items_tag")
    meta, head, built = load_items(args.data, args.dataset, tag)
    data_sha = {f: KF.sha256_file(os.path.join(args.data, f))
                for f in [items_file(args.dataset, tag)] + sorted(prefix_files(meta, args.dataset).values())}
    src = json.load(open(os.path.join(args.data, "sources.json")))
    for f, h in data_sha.items():
        if src["outputs_sha256"].get(f) != h:
            die(EXIT_PLAN, "data/%s differs from data/sources.json" % f)
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))
    X, enc, pins = (loader or load_ctx)(args.model, os.path.abspath(args.tree), args.mem_fraction)
    check_tail_ids(enc, meta)
    mats = [materialize(enc, meta, head, it0) for it0 in built]       # B3: every item before the loop (exit 2)
    if fam_of(X) != MODELS[args.model].get("family", "ouro"):
        die(EXIT_PLAN, "context family %s does not match model %s" % (fam_of(X), args.model))
    if fam_of(X) == "raven":                          # one fixed x0 table (RA-25) covering the longest prompt
        X.b.set_init_state("fixed_table", max(len(m["A"]) for m in mats) + 1)
    cfg = {"run": "RA-29", "model": args.model, "dataset": args.dataset, "pilot": bool(args.pilot),
           "plan_sha256": plan_sha, "code_sha256": code_hashes(), "data_sha256": data_sha, "pins": pins,
           "gap_e": GAP_E, "n_boot": NB, "arms": [x[0] for x in FAM[MODELS[args.model].get("family", "ouro")]["arms"]],
           "gated": list(FAM[MODELS[args.model].get("family", "ouro")]["gated"]), "torch": torch.__version__}
    cp = os.path.join(out, "config.json")
    if os.path.exists(cp):
        old = json.load(open(cp))
        for k in ("plan_sha256", "code_sha256", "data_sha256", "pins", "model", "dataset", "pilot"):
            if old.get(k) != cfg.get(k):
                die(EXIT_PLAN, "resume with a changed config field %s" % k)
    else:
        json.dump(cfg, open(cp, "w"), indent=1, sort_keys=True)
    grecs = []
    for it in mats:                                                   # identity gate, every session
        if sum(1 for r in grecs if r["drop"] is None) >= GATE_N or len(grecs) >= GATE_TRY:
            break
        grecs.append((process or process_item)(X, it, args.model, args.dataset))
    gfail = {r["key"]: identity_failures(r) for r in grecs if identity_failures(r)}
    gate = {"pass": not gfail and any(r["drop"] is None for r in grecs), "n_tried": len(grecs),
            "n_scored": sum(1 for r in grecs if r["drop"] is None), "failures": gfail, "arms": arm_names(grecs)}
    json.dump(gate, open(os.path.join(out, "gate.json"), "w"), indent=1)
    print("[ra29] identity gate %s (%d scored of %d tried)" % ("PASS" if gate["pass"] else "FAIL", gate["n_scored"],
                                                               gate["n_tried"]), flush=True)
    if not gate["pass"]:
        die(EXIT_GATE, "GATE FAILED (identity gate before any scored row): %s" % json.dumps(gfail))
    rec_path = os.path.join(out, "records.jsonl")
    recs = load_jsonl(rec_path, repair=True)
    bad = [r["key"] for r in recs if identity_failures(r)]
    if bad:
        die(EXIT_IDENTITY, "records hold identity failures: %s" % bad[:5])
    done = {r["key"] for r in recs}
    maps = []
    with open(rec_path, "a") as fh:
        for it in mats:
            if STOP["flag"]:
                break
            kept = sum(1 for r in recs if r["drop"] is None)
            if args.pilot and (kept >= PILOT_KEEP or len(recs) >= PILOT_TRY):
                break
            if it["key"] in done:
                continue
            if len(maps) < 3:
                maps.append(block_map(X, it))
            rec = (process or process_item)(X, it, args.model, args.dataset)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            recs.append(rec)
            bad = identity_failures(rec)
            if bad:
                die(EXIT_IDENTITY, "item %s identity failed: %s; stopping" % (rec["key"], bad))
            if len(recs) % 20 == 1:
                print("[ra29] %s %s %d tried, %d kept, %.0fs" % (args.model, args.dataset, len(recs),
                                                                sum(1 for r in recs if r["drop"] is None),
                                                                time.time() - t_start), flush=True)
    if STOP["flag"]:
        die(EXIT_STOPPED, "SIGTERM: stopped after the current item (resume with the same command)")
    info = {"model": args.model, "dataset": args.dataset, "pilot": bool(args.pilot), "plan_sha256": plan_sha,
            "block_maps": maps, "meta_drops_before_screen": meta["drops"], "n_rows": meta["n_rows"]}
    res, chk = write_outputs(out, recs, len(built), info)
    if not args.pilot and chk["identity_failures"]:
        die(EXIT_IDENTITY, "identity failures in the records: %s" % json.dumps(chk["identity_failures"]))
    if args.pilot:
        ok = chk["gates"]["pc_exact_all"] and chk["gates"]["hook_identity_all"] and not chk["identity_failures"] \
            and chk["counts"]["n_kept"] > 0
        json.dump({"pass": bool(ok), "plan_sha256": plan_sha, "model": args.model, "dataset": args.dataset,
                   "timing": chk["timing"], "counts": chk["counts"],
                   "norm_ratio_min_max": {a: (res.get(a) or {}).get("norm_ratio_min_max")
                                          for a in FAM[MODELS[args.model].get("family", "ouro")]["rnd_arms"]}},
                  open(os.path.join(out, "debug_gate.json"), "w"), indent=1)
        if not ok:
            die(EXIT_GATE, "GATE FAILED: %s" % json.dumps(chk["gates"]))
    print("[ra29] done: %s" % json.dumps(chk["counts"]), flush=True)
    return res, chk


def summarize(args):
    recs = load_jsonl(os.path.join(args.out, "records.jsonl"))
    cfg = json.load(open(os.path.join(args.out, "config.json")))
    meta, _, built = load_items(args.data, cfg["dataset"], MODELS[cfg["model"]].get("items_tag"))
    return write_outputs(args.out, recs, len(built), {"model": cfg["model"], "dataset": cfg["dataset"],
                                                      "pilot": cfg["pilot"], "plan_sha256": cfg["plan_sha256"]})


def main(argv=None):
    a = parse_args(argv)
    if a.hash:
        print(json.dumps({"plan_sha256": KF.sha256_file(a.plan) if os.path.exists(a.plan) else None,
                          "code_sha256": code_hashes()}, indent=1))
        return
    if a.summarize:
        summarize(a)
        return
    run(a)


if __name__ == "__main__":
    main()
