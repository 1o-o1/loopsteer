"""RA-30 on McLeish r32 (plan_ra30.md section 6c): the chain-value edit on a recurrent-depth model, GSM8K.

The Ouro driver ra30.py is not changed; this file reuses its helpers (plan check, pilot of record, cap rule,
bootstrap, files) and the chain logic of lib/chain.py. Hooks, x0 policy and decode come from the cluster bundle's
McLeish generation path, lib/steer_gen_raven.py (an unchanged copy): RavenSteerHooks (sites post@j = the output of
the last core block at loop j, kv@j = the K and V parts of attn.Wqkv at loop j, before rotary) and
SteeredMcLeishAdapter (prod's raven decode with the fixed per-item x0 table). One subclass here: the x0 table is
drawn from the problem id (key // KEY_MULT), so every arm row of an item, its donors and its clean screen read the
same x0 as the Table 2 McLeish run (seed 1000003 * problem id).

    python ra30_mcl.py --tokens --out OUT                       (host tokenizer, CPU)
    python ra30_mcl.py --run --stage pilot|full --out OUT --plan_sha256 SHA
    python ra30_mcl.py --summarize --out OUT/mcleish_llama32_r32/<stage>/gsm8k
"""
import argparse
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "lib"), os.path.join(HERE, "tree")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("K17_DATASET", "gsm8k")

import torch  # noqa: E402

import chain as CH  # noqa: E402
import ra30 as R0  # noqa: E402
import ra30_data as RD  # noqa: E402
import steer_gen_raven as SGR  # noqa: E402

MODEL = SGR.MODEL                      # mcleish_llama32_r32, revision a5f6f126
K = SGR.K                              # 32 recurrences (trained depth)
HORIZON = 512
DS = "gsm8k"
TASK = "gsm8k"
# plan 6f: McLeish ASDiv reads its pool on the cluster at run time (the EXT5 priority-2 select output of job
# g18_fillgen.s3_select.002); the pilot records its sha256 and the full shards stop when it differs (exit 2)
POOL_RT = {"asdiv": "A/steer_gen_math/mcleish_asdiv/items.json"}
DS_NAME = {"gsm8k": "GSM8K", "asdiv": "ASDiv"}


def set_ds(ds):
    global DS, TASK
    if ds not in ("gsm8k", "asdiv"):
        R0.die(R0.EXIT_PLAN, "McLeish dataset %s" % ds)
    DS, TASK = ds, RD.BASE_TASK[ds]


def get_pool(pool_root=None):
    """({"n", "items"}, record): GSM8K from data/pool_gsm8k_<model>.json; ASDiv from the cluster's select output,
    joined to the pairs file for the question texts."""
    if DS == "gsm8k":
        pool = RD.load_pool(DS, MODEL)
        p = os.path.join(RD.DATA, "pool_%s_%s.json" % (DS, MODEL))
        return pool, {"path": "data/" + os.path.basename(p), "sha256": RD.sha256_file(p), "n": pool["n"]}
    if not pool_root:
        R0.die(R0.EXIT_PLAN, "--pool_root (the cluster's OUT_ROOT) is required for McLeish %s" % DS)
    p = os.path.join(pool_root, POOL_RT[DS])
    if not os.path.exists(p):
        R0.die(R0.EXIT_PLAN, "the McLeish %s pool %s is missing (it comes from g18_fillgen.s3_select.002)"
               % (DS, POOL_RT[DS]))
    with open(p, encoding="utf-8") as f:
        scored = json.load(f)["scored"]
    pairs = {(int(r["idx"]), r["var_name"]): r for r in RD.read_jsonl(RD.pairs_path(DS))}
    items = []
    for it in scored:
        pr = pairs.get((int(it["idx"]), it["var_name"]))
        if pr is None:
            R0.die(R0.EXIT_PLAN, "pool item %s has no pair" % it["idx"])
        items.append(dict(it, question_A=pr["question_A"], question_B=pr["question_B"]))
    return {"n": len(items), "items": items}, {"path": POOL_RT[DS], "sha256": RD.sha256_file(p), "n": len(items)}
KEY_MULT = 100                         # row key = problem id * 100 + arm number
N_SCORED_PILOT = 20
N_GATE = {"pilot": 10, "full": 3}
LOOPS = list(range(1, K + 1))
ABOUT0 = {"point": 0.05, "hi90": 0.10}         # "about 0": CA <= 0.05 and its 90% upper bound <= 0.10
MOST = {"ratio": 0.67, "lo90": 0.5}             # "most": CA(cvP)/CA(text) >= 0.67 with its 90% lower bound >= 0.5
# RA D1 (plan 6c): the 80-item gate is waived for this breadth cell; the full run needs n >= 20 projected eligible
MIN_ELIGIBLE_FULL = 20
# RA D2: the full run is sharded by items (shard s of n takes the order positions s, s + n, ...). Review N6 rule:
# 3 shards if the pilot's projected_full_card_hours <= 16.2 (3 x 9 h x 0.6), else ceil(projected / 5.4)
SHARDS_FULL = 3


def shard_count(projected_card_hours):
    import math
    if projected_card_hours is None or projected_card_hours <= 3 * 9 * 0.6:
        return SHARDS_FULL
    return int(math.ceil(projected_card_hours / (9 * 0.6)))

# name, batch, sequence, donor, edits (pos "P" = derived-value positions), role
ARMS = [
    ("base", "scratch", "A", "B", [("patch", [1], "last_prompt")], "default baseline: last prompt token, post@1"),
    ("text", "prefix", "A+prefB", None, [], "text edit (ceiling)"),
    ("floor", "prefix", "A+prefA", None, [], "unedited (floor)"),
    ("cv1", "prefix", "A+prefA", "A+prefB", [("patch", [1], "P")], "chain-value patch, end of loop 1"),
    ("kv1", "prefix", "A+prefA", "A+prefB", [("kv", [1], "P")], "loop-1 key-value patch at P"),
    ("rnd1", "prefix", "A+prefA", "A+prefB", [("rand", [1], "P")], "random same-norm at P, loop 1"),
    ("cvP", "prefix", "A+prefA", "A+prefB", [("patch", LOOPS, "P")], "persistent chain-value patch, loops 1..32"),
    ("kvP", "prefix", "A+prefA", "A+prefB", [("kv", [0] + LOOPS, "P")],
     "persistent key-value patch, prelude (loop 0) and loops 1..32"),
    ("eP", "prefix", "A+prefA", "A+prefB", [("patch_e", [0], "P")],
     "input-injection patch: prelude output e at P = donor's (interpretation control)"),
    ("span1", "scratch", "A", "B", [("patch", [1], "span")], "span patch, end of loop 1 (reference)"),
    ("clean", "scratch", "A", None, [], "clean A in the scratch batch (in-batch reference)"),
]
ARM_NAMES = [a[0] for a in ARMS]
ARM = {a[0]: {"batch": a[1], "seq": a[2], "donor": a[3], "edits": a[4], "role": a[5]} for a in ARMS}
ARM_NO = {"screen": 0}
ARM_NO.update({a: i + 1 for i, a in enumerate(ARM_NAMES)})
DONOR_ROW = {"prefix": "kvP", "scratch": "base"}     # kvP's donor record holds every loop's K/V and post states


def cap_new(n_pre, n_clean, use_cap):
    """ra30.cap_new on this driver's horizon (plan section 10 rule)."""
    full = HORIZON - n_pre
    return min(full, 2 * (int(n_clean) - n_pre) + 64) if use_cap else full


def key_of(idx, arm):
    return int(idx) * KEY_MULT + ARM_NO[arm]


class MclAdapter(SGR.SteeredMcLeishAdapter):
    """steer_gen_raven's adapter; the x0 table of a row key is the table of its problem id."""

    def table(self, key):
        base = int(key) // KEY_MULT
        cache = self.__dict__.setdefault("_ra30_tables", {})
        t = cache.get(base)
        if t is None:
            t = SGR.x0_table(self.model, base, self.t_max).to(self.dtype)
            cache[base] = t
        return t


class Ctx(object):
    def __init__(self, ad, steer, tok):
        from prod.tasks import task_cfg
        self.ad, self.steer, self.tok, self.model = ad, steer, tok, ad.model
        self.stops = list(task_cfg(TASK)["stops"])
        self.eos = sorted(set([int(tok.eos_token_id)]) | set(int(x) for x in ad.stop_ids()))


def load_ctx():
    import transformers
    if transformers.__version__ != R0.TRANSFORMERS_PIN:
        R0.die(R0.EXIT_PLAN, "transformers %s, pinned %s" % (transformers.__version__, R0.TRANSFORMERS_PIN))
    SGR.ACTIVE[0] = "mcl"
    steer = SGR.RavenSteerHooks(K)
    ad = MclAdapter(steer=steer, kv=True)
    ad.load()
    ad.set_depth(K)
    return Ctx(ad, steer, ad.tok)


def clear_item(ctx):
    st = ctx.steer
    st.rows.clear()
    st.donors.clear()
    st.stats.clear()
    st.prompt_counted.clear()
    ctx.ad.__dict__.setdefault("_ra30_tables", {}).clear()


def prompt_ids(tok, question):
    """G.main's McLeish encoding of the GSM8K prompt: BOS + the plain ids (steer_gen_raven.enc)."""
    return SGR.enc(tok, RD.prompt_text(tok, DS, question))


def cut_parse(ctx, chain_ids):
    from prod.models.ouro import strip_tail
    from prod.tasks import make_find_cut, parse_own
    ids = strip_tail([list(chain_ids)], ctx.eos)[0]
    c, _mk = make_find_cut(ctx.tok, ctx.stops, ctx.eos, chat_template=False, eos_cut=True)(ids)
    txt = ctx.tok.decode(ids[:c], clean_up_tokenization_spaces=False)
    return ids, c, txt, parse_own(txt, TASK)


def clean_decode(ctx, idx, prompt):
    gen, _t = ctx.ad.decode([list(prompt)], HORIZON, K, ctx.eos, stop_strings=ctx.stops,
                            row_ids=[key_of(idx, "screen")])
    return list(gen[0])


def screen_item(ctx, it, clean_fn=None):
    tok = ctx.tok
    pA, pB = prompt_ids(tok, it["question_A"]), prompt_ids(tok, it["question_B"])
    s0, s1 = int(it["span"][0]), int(it["span"][1])
    rec = {"idx": int(it["idx"]), "plen": len(pA)}
    if not (len(pA) == len(pB) == int(it["plen"])) or pA[:s0] != pB[:s0] or pA[s1:] != pB[s1:] \
            or pA[s0:s1] == pB[s0:s1]:
        rec.update(drop="plen_mismatch", clean_ids=None)
        return rec
    t0 = time.time()
    raw = (clean_fn or clean_decode)(ctx, int(it["idx"]), pA)
    rec["sec_clean"] = round(time.time() - t0, 3)
    ids, cut, txt, own = cut_parse(ctx, raw)
    rec.update(clean_ids=ids, cut=cut, own=own, n_clean=len(ids))
    if CH.outcome(own, it["gold_A"], it["gold_B"]) != "A":
        rec["drop"] = "clean_wrong"
        return rec
    a = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"])
    r1 = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"], mode="r1")
    rec["r1_kept_info"] = r1["drop"] is None
    if a["drop"] is None and any(CH.feq(v, CH.numval(a["derived_A"])) for _s, _e, v in CH.operands(a["e1"])):
        a = dict(a, drop="value_is_operand")      # plan 6c: the first copy could be the operand ("4 / 2 = 2")
    if a["drop"] is None:
        a = CH.token_geometry(tok, pA, ids, a)
    rec["drop"] = a["drop"]
    if a["drop"] is None:
        keep = ("derived_A", "derived_B", "answer_edit", "answer_full_subst", "P", "prefix_A", "prefix_B",
                "prefix_len", "tokens_A", "tokens_B", "canonical_B", "e1", "i0", "m", "derived_B_in_prompt",
                "gold_B_in_prompt_or_prefix", "n_lead_same")
        rec.update({k: a[k] for k in keep})
    return rec


def geom_of(ctx, it, sc):
    return {"prompt_A": prompt_ids(ctx.tok, it["question_A"]), "prompt_B": prompt_ids(ctx.tok, it["question_B"]),
            "prefix_A": sc["prefix_A"], "prefix_B": sc["prefix_B"], "P": sc["P"], "span": it["span"],
            "clean_ids": sc["clean_ids"]}


def row_info(arm, g, idx):
    a = ARM[arm]
    pa, pb = list(g["prompt_A"]), list(g["prompt_B"])
    seqs = {"A": pa, "B": pb, "A+prefA": pa + list(g["prefix_A"]), "A+prefB": pa + list(g["prefix_B"])}
    seq = seqs[a["seq"]]
    s0, s1 = int(g["span"][0]), int(g["span"][1])
    if a["batch"] == "scratch" and (len(pb) != len(pa) or pa[:s0] != pb[:s0] or pa[s1:] != pb[s1:]):
        raise ValueError("prompt B does not match prompt A outside the span")
    rng = (int(g["P"][0]), int(g["P"][1])) if a["batch"] == "prefix" else (s0, len(pa))
    edits = []
    for op, loops, pos in a["edits"]:
        p = [int(g["P"][0]), int(g["P"][1])] if pos == "P" else pos
        if op == "patch":
            edits.append({"op": "patch", "site": "post", "loop": list(loops), "pos": p, "a": 1.0})
        elif op == "patch_e":
            edits.append({"op": "patch", "site": "pre_out", "loop": list(loops), "pos": p, "a": 1.0})
        elif op == "kv":
            edits.append({"op": "kv", "loops": list(loops), "pos": p, "a": 1.0})
        else:
            edits.append({"op": "rand", "site": "post", "loop": list(loops), "pos": p, "seed": 0, "ref": "patch",
                          "rkey": "%s:%d" % (DS, int(idx))})
    return {"plen": len(pa), "span": (s0, s1), "a_ids": seq,
            "b_ids": seqs[a["donor"]] if a["donor"] else list(seq), "edits": edits, "lcdl": None,
            "donor_range": rng, "t0": rng[0]}


def build_rows(ctx, it, g, arms=None):
    idx = int(it["idx"])
    names = list(arms or ARM_NAMES)
    keys = {}
    for a in names:
        k = key_of(idx, a)
        ctx.steer.rows[str(k)] = row_info(a, g, idx)
        keys[a] = k
    for batch, drow in DONOR_ROW.items():
        need = [a for a in names if ARM[a]["batch"] == batch and ctx.steer.needs_donor(str(keys[a]))]
        if not need:
            continue
        dk = str(key_of(idx, drow))
        if dk not in ctx.steer.rows:
            ctx.steer.rows[dk] = row_info(drow, g, idx)
        d = ctx.ad.record_donor(dk)
        for a in need:
            ctx.steer.donors[str(keys[a])] = d
    return keys


def readout(ctx, it, sc, arm, chain_ids, n_prefix):
    ids, cut, txt, own = cut_parse(ctx, chain_ids)
    cont = ctx.tok.decode(ids[n_prefix:cut], clean_up_tokenization_spaces=False) if cut > n_prefix else ""
    mk = txt.rfind(CH.OWN_MARKER)
    marker_tok = None
    if mk >= 0:
        ends = CH.char_offsets(ctx.tok, ids[:cut])
        marker_tok = next((i for i, e in enumerate(ends) if e > mk), None)
    return {"arm": arm, "chain_ids": ids, "cut": cut, "text": txt, "own": own,
            "outcome": CH.outcome(own, it["gold_A"], it["gold_B"]),
            "carry": CH.carry(cont, sc["derived_A"], sc["derived_B"]), "n_chain": len(ids),
            "hit_horizon": len(ids) >= HORIZON, "marker_tok": marker_tok, "n_prefix": n_prefix}


def run_item(ctx, it, sc, use_cap=False):
    g = geom_of(ctx, it, sc)
    t0 = time.time()
    keys = build_rows(ctx, it, g)
    out = []
    plen = len(g["prompt_A"])
    for batch, n_pre in (("prefix", len(g["prefix_A"])), ("scratch", 0)):
        names = [a for a in ARM_NAMES if ARM[a]["batch"] == batch]
        rk = [keys[a] for a in names]
        seqs = [ctx.steer.rows[str(k)]["a_ids"] for k in rk]
        if len({len(s) for s in seqs}) != 1:
            raise RuntimeError("rows of one batch differ in length")
        tb = time.time()
        gen, _t = ctx.ad.decode(seqs, cap_new(n_pre, sc["n_clean"], use_cap), K, ctx.eos,
                                stop_strings=ctx.stops, row_ids=rk)
        sec = time.time() - tb
        for a, k, gg in zip(names, rk, gen):
            pre = list(ctx.steer.rows[str(k)]["a_ids"][plen:])
            r = readout(ctx, it, sc, a, pre + list(gg), n_pre)
            r["sec_batch"] = round(sec, 3)
            r["batch_rows"] = len(names)
            out.append(r)
    ref = {r["arm"]: r["chain_ids"] for r in out}
    for r in out:
        r["div_from_ref"] = R0.first_div(r["chain_ids"], ref["floor" if ARM[r["arm"]]["batch"] == "prefix" else "clean"])
    stats = {"%s|%s" % kk: v for kk, v in ctx.steer.stats.items()}
    a_ = stats.get("%d|post@1" % keys["rnd1"])
    b_ = stats.get("%d|post@1" % keys["cv1"])
    meta = {"idx": int(it["idx"]), "sec_item": round(time.time() - t0, 3),
            "norm_ratio_prerounding": {"rnd1": round(a_[0] / b_[0], 6)} if (a_ and b_ and b_[0] > 0) else {}}
    return out, meta


# ====================================================================== gates
@torch.no_grad()
def fwd_last(ctx, seq, key):
    return ctx.ad.forward_rows([list(seq)], [str(key)])[0].float().cpu()


def tie_check(ctx, seq_prefix, key, a, b):
    d = R0.first_div(a, b)
    if d is None:
        return {"identical": True, "pass": True}
    if d >= min(len(a), len(b)):
        return {"identical": False, "div": d, "pass": False, "why": "length"}
    z = fwd_last(ctx, list(seq_prefix) + list(a[:d]), key)
    t = z.topk(2).values
    m = float(t[0] - t[1])
    return {"identical": False, "div": d, "margin": round(m, 4), "pass": m < R0.TIE}


def new_rec(lo, hi, want):
    return {"cols": slice(int(lo), int(hi)), "want": set(want), "post": {}, "pre": {}, "kv": {}, "emb": None,
            "logits": []}


@torch.no_grad()
def gates_item(ctx, it, sc, all_loops=False):
    """G1 hook identity (idle hooks against removed hooks), G2 the floor alone against the clean chain (tie rule),
    G3 teacher-forced prefix argmax, G4 positions, G5 cv1 and kv1 land on a fresh donor record."""
    from prod.models.ouro import strip_tail
    tok, idx = ctx.tok, int(it["idx"])
    g = geom_of(ctx, it, sc)
    seqA, seqB = g["prompt_A"] + g["prefix_A"], g["prompt_A"] + g["prefix_B"]
    lo, hi = g["P"]
    plen, npre = len(g["prompt_A"]), len(g["prefix_A"])
    res = {"idx": idx}
    for a in ("floor", "cv1", "kv1", "kvP", "cvP", "eP"):
        ctx.steer.rows[str(key_of(idx, a))] = row_info(a, g, idx)
    kf = key_of(idx, "floor")
    z1 = fwd_last(ctx, seqA, kf)
    ctx.steer.remove()
    try:
        z0 = fwd_last(ctx, seqA, kf)
    finally:
        ctx.steer.install(ctx.model, kv=True)
    res["G1"] = bool(torch.equal(z0, z1))
    flo = ctx.ad.decode([seqA], HORIZON - npre, K, ctx.eos, stop_strings=ctx.stops, row_ids=[kf])[0][0]
    flo = strip_tail([list(flo)], ctx.eos)[0]
    res["G2"] = tie_check(ctx, seqA, kf, list(sc["clean_ids"])[npre:], flo)
    res["floor_alone"] = flo
    z = ctx.ad.forward_rows([seqA], [str(kf)], want_logits="all")[0, plen - 1:plen + npre - 1].float()
    am = z.argmax(-1).tolist()
    bad = [i for i, (x, y) in enumerate(zip(am, g["prefix_A"])) if x != y]
    if bad:
        top = z[bad[0]].topk(2).values
        res["G3"] = {"n_bad": len(bad), "first": bad[0], "margin": round(float(top[0] - top[1]), 4),
                     "pass": float(top[0] - top[1]) < R0.TIE}
    else:
        res["G3"] = {"n_bad": 0, "pass": True}
    res["G4"] = {"P": [lo, hi], "tokens_A": [tok.decode([i]) for i in seqA[lo:hi]],
                 "tokens_B": [tok.decode([i]) for i in seqB[lo:hi]], "equal_len": len(seqA) == len(seqB),
                 "P_ends_prefix": hi == plen + npre}
    fresh = ctx.ad.record_donor(str(key_of(idx, "kvP")))
    for a in ("cv1", "kv1", "kvP", "cvP", "eP"):
        ctx.steer.donors[str(key_of(idx, a))] = fresh
    ok = True
    ctx.steer.rec = new_rec(lo, hi, {"post"})
    try:
        ctx.ad.forward_rows([seqA], [str(key_of(idx, "cv1"))])
        ok &= bool(torch.equal(ctx.steer.rec["post"][1][0], fresh["post"][1]))
    finally:
        ctx.steer.rec = None
    ctx.steer.rec = new_rec(lo, hi, {"kv"})
    try:
        ctx.ad.forward_rows([seqA], [str(key_of(idx, "kv1"))])
        n_core = len(ctx.model.transformer.core_block)
        for li in range(n_core):
            for w in ("k", "v"):
                ok &= bool(torch.equal(ctx.steer.rec["kv"][(1, li, w)][0], fresh["kv"][(1, li, w)]))
    finally:
        ctx.steer.rec = None
    if all_loops:                        # review N2: every loop of cvP, kvP (0..32) and eP lands at P
        n_core = len(ctx.model.transformer.core_block)
        n_pre = len(ctx.model.transformer.prelude)
        ctx.steer.rec = new_rec(lo, hi, {"post"})
        try:
            ctx.ad.forward_rows([seqA], [str(key_of(idx, "cvP"))])
            for j in range(1, K + 1):
                ok &= bool(torch.equal(ctx.steer.rec["post"][j][0], fresh["post"][j]))
        finally:
            ctx.steer.rec = None
        ctx.steer.rec = new_rec(lo, hi, {"kv"})
        try:
            ctx.ad.forward_rows([seqA], [str(key_of(idx, "kvP"))])
            for j in range(0, K + 1):
                for li in range(n_pre if j == 0 else n_core):
                    for w in ("k", "v"):
                        ok &= bool(torch.equal(ctx.steer.rec["kv"][(j, li, w)][0], fresh["kv"][(j, li, w)]))
        finally:
            ctx.steer.rec = None
        ctx.steer.rec = new_rec(lo, hi, {"post"})
        try:
            ctx.ad.forward_rows([seqA], [str(key_of(idx, "eP"))])
            ok &= bool(torch.equal(ctx.steer.rec["post"][0][0], fresh["post"][0]))
        finally:
            ctx.steer.rec = None
        res["G5_all_loops"] = True
    res["G5_patch_lands"] = bool(ok)
    res["pass"] = bool(res["G1"] and res["G2"]["pass"] and res["G3"]["pass"] and res["G4"]["equal_len"]
                       and res["G4"]["P_ends_prefix"] and res["G5_patch_lands"])
    clear_item(ctx)
    return res


# ====================================================================== run
def model_root(out_root):
    return os.path.join(out_root, MODEL)


def check_data():
    if DS != "gsm8k":                     # ASDiv: the pairs file (sources_think.json); the pool is pinned at run time
        with open(os.path.join(RD.DATA, "sources_think.json"), encoding="utf-8") as f:
            st = json.load(f)
        if RD.sha256_file(RD.pairs_path(DS)) != st["pairs"][DS]["sha256"]:
            R0.die(R0.EXIT_PLAN, "the %s pairs file differs from sources_think.json" % DS)
        return {"files": {"pairs_%s.jsonl" % DS: st["pairs"][DS]["sha256"]}}
    with open(os.path.join(RD.DATA, RD.sources_name(MODEL)), encoding="utf-8") as f:
        src = json.load(f)
    for fn, h in src["files"].items():
        if RD.sha256_file(os.path.join(RD.DATA, fn)) != h:
            R0.die(R0.EXIT_PLAN, "data file %s differs from %s" % (fn, RD.sources_name(MODEL)))
    return src


def code_hashes():
    fs = ["ra30_mcl.py", "ra30.py", "lib/chain.py", "lib/ra30_data.py", "lib/steer_gen_raven.py",
          "tree/prod/steer_gen.py", "tree/prod/models/raven.py", "tree/prod/models/mcleish.py",
          "tree/prod/tasks/__init__.py"]
    return {f: RD.sha256_file(os.path.join(HERE, f)) for f in fs}


# the McLeish GSM8K pilot of push 2 runs under plan 086581bf; a full run of a later plan accepts it
MCL_ACCEPTED_PILOT_PLANS = R0.ACCEPTED_PILOT_PLANS + ("086581bf0b2c109436b4c09dd47e8f1e6c2d168dd28ef0c431cc684df8081ba8",)


def pilot_of_record(out_root, plan_sha, pool_rec=None):
    """The McLeish full run reads its pilot: plan accepted, n >= MIN_ELIGIBLE_FULL projected eligible (RA D1: the
    80-item gate is waived for this cell), the horizon cap decision."""
    d = os.path.join(out_root, "pilot", DS)
    pc, ph = os.path.join(d, "checks.json"), os.path.join(d, "hash_line.json")
    if not (os.path.exists(pc) and os.path.exists(ph)):
        R0.die(R0.EXIT_PLAN, "no McLeish pilot in %s; run the pilot first" % d)
    with open(pc, encoding="utf-8") as f:
        chk = json.load(f)
    with open(ph, encoding="utf-8") as f:
        pplan = json.load(f).get("plan_sha256")
    if pplan not in MCL_ACCEPTED_PILOT_PLANS + ((plan_sha,) if plan_sha else ()):
        R0.die(R0.EXIT_PLAN, "the McLeish pilot ran under plan %s, which is not accepted" % pplan)
    pr = os.path.join(d, "pool_record.json")
    if pool_rec is not None and os.path.exists(pr):
        with open(pr, encoding="utf-8") as f:
            if json.load(f).get("sha256") != pool_rec["sha256"]:
                R0.die(R0.EXIT_PLAN, "the pool file differs from the one the pilot read (sha256 pin)")
    elif pool_rec is not None and DS != "gsm8k":
        R0.die(R0.EXIT_PLAN, "the McLeish %s pilot has no pool record" % DS)
    n = chk.get("projected_eligible")
    if n is None or n < MIN_ELIGIBLE_FULL:
        R0.die(R0.EXIT_GATE, "McLeish %s under %d: gate failure (projected eligible %s)" % (DS_NAME[DS],
                                                                                           MIN_ELIGIBLE_FULL, n))
    return {"use_cap": bool((chk.get("horizon_cap_rule") or {}).get("use_cap_in_full")), "plan_sha256": pplan,
            "projected_eligible": n}


def shard_dir(out, shard, shards):
    return out if int(shards) <= 1 else os.path.join(out, "shard_%d_of_%d" % (int(shard), int(shards)))


def run(args, ctx=None, clean_fn=None):
    set_ds(getattr(args, "ds", "gsm8k"))
    stage = args.stage
    shard, shards = int(getattr(args, "shard", 0)), int(getattr(args, "shards", 1))
    if stage == "pilot" and shards != 1:
        R0.die(R0.EXIT_PLAN, "the pilot is not sharded")
    if not 0 <= shard < shards:
        R0.die(R0.EXIT_PLAN, "shard %d of %d" % (shard, shards))
    out = shard_dir(os.path.join(model_root(args.out), stage, DS), shard, shards)
    os.makedirs(out, exist_ok=True)
    plan_sha = R0.check_plan(args.plan_sha256) if not args.no_plan_check else None
    src = check_data()
    pool, pool_rec = get_pool(getattr(args, "pool_root", None))
    if DS == "gsm8k":
        order = RD.load_split(DS, MODEL)["order"][shard::shards]
    else:
        order = RD.draw_order([it["idx"] for it in pool["items"]], DS, MODEL)[shard::shards]
    by = {int(it["idx"]): it for it in pool["items"]}
    n_target = N_SCORED_PILOT if stage == "pilot" else None
    use_cap, pilot_plan = False, None
    if stage == "full":
        pilot = pilot_of_record(model_root(args.out), plan_sha, pool_rec)
        use_cap, pilot_plan = pilot["use_cap"], pilot["plan_sha256"]
    cfg = {"ds": DS, "stage": stage, "model": MODEL, "k": K, "horizon": HORIZON, "use_cap": use_cap,
           "pilot_plan_sha256": pilot_plan, "n_scored_pilot": n_target, "arms": ARM_NAMES, "tie": R0.TIE,
           "key_mult": KEY_MULT, "about0": ABOUT0, "most": MOST, "shard": shard, "shards": shards,
           "min_eligible_full": MIN_ELIGIBLE_FULL, "gate_80": "waived (RA D1, breadth cell)"}
    if DS != "gsm8k":
        cfg["pool"] = pool_rec
    hl = os.path.join(out, "hash_line.json")
    hline = {"plan_sha256": plan_sha, "config_sha256": R0.sha256_text(json.dumps(cfg, sort_keys=True)),
             "code_sha256": code_hashes(), "data_sha256": src["files"]}
    if os.path.exists(hl):
        with open(hl, encoding="utf-8") as f:
            old = json.load(f)
        if {k: old.get(k) for k in hline} != hline:
            R0.die(R0.EXIT_PLAN, "inputs changed since the first row of %s (hash_line.json)" % out)
    else:
        R0.write_json(hl, hline)
        R0.write_json(os.path.join(out, "config.json"), cfg)
        if DS != "gsm8k":
            R0.write_json(os.path.join(out, "pool_record.json"), pool_rec)
    signal.signal(signal.SIGTERM, R0.on_term)
    if ctx is None:
        ctx = load_ctx()
    sp = os.path.join(out, "screen.jsonl")
    screened = {int(r["idx"]): r for r in R0.read_jsonl(sp)}
    elig = [i for i in order if i in screened and screened[i]["drop"] is None]
    for i in order:
        if n_target is not None and len(elig) >= n_target:
            break
        if i in screened:
            continue
        if R0.STOP["flag"]:
            R0.die(R0.EXIT_STOPPED, "stopped during the screen")
        r = screen_item(ctx, by[i], clean_fn)
        clear_item(ctx)
        R0.append_jsonl(sp, r)
        screened[i] = r
        if r["drop"] is None:
            elig.append(i)
    if n_target is not None:
        elig = elig[:n_target]
    R0.log("%s mcleish %s: scanned %d, eligible %d" % (stage, DS, len(screened), len(elig)))
    gates = [gates_item(ctx, by[i], screened[i], all_loops=(k == 0)) for k, i in enumerate(elig[:N_GATE[stage]])]
    gp = {"items": gates, "n": len(gates), "pass": all(g["pass"] for g in gates),
          "G2_identical_share": (sum(1 for g in gates if g["G2"]["identical"]) / len(gates)) if gates else None}
    R0.write_json(os.path.join(out, "gates_%d.json" % int(time.time())), gp)
    R0.write_json(os.path.join(out, "gates.json"), gp)
    if not gp["pass"]:
        R0.die(R0.EXIT_GATE, "gate failed (gates.json)")
    R0.log("gates PASS on %d items; G2 identical share %s" % (len(gates), gp["G2_identical_share"]))
    rp = os.path.join(out, "rows.jsonl")
    done = {int(r["idx"]) for r in R0.read_jsonl(os.path.join(out, "items_done.jsonl"))}
    floor_alone = {g["idx"]: g["floor_alone"] for g in gates}
    for i in elig:
        if i in done:
            continue
        if R0.STOP["flag"]:
            R0.die(R0.EXIT_STOPPED, "stopped after item %d" % i)
        rows, meta = run_item(ctx, by[i], screened[i], use_cap)
        clear_item(ctx)
        if i in floor_alone:
            fr = next(r for r in rows if r["arm"] == "floor")
            seqA = prompt_ids(ctx.tok, by[i]["question_A"]) + list(screened[i]["prefix_A"])
            meta["G2b"] = tie_check(ctx, seqA, key_of(i, "floor"), floor_alone[i], fr["chain_ids"][fr["n_prefix"]:])
            clear_item(ctx)
        for r in rows:
            r["idx"] = i
            R0.append_jsonl(rp, r)
        R0.append_jsonl(os.path.join(out, "items_done.jsonl"), meta)
    if stage == "pilot":
        rows = R0.read_jsonl(rp)
        txt = [r for r in rows if r["arm"] == "text"]
        ca = sum(1 for r in txt if r["outcome"] == "B") / max(1, len(txt))
        R0.write_json(os.path.join(out, "g6.json"), {"ca_text": ca, "n": len(txt), "pass": ca >= R0.CEILING_MIN})
        if ca < R0.CEILING_MIN:
            summarize(out)
            R0.die(R0.EXIT_GATE, "G6 failed: text-edit ceiling %.3f < %.2f" % (ca, R0.CEILING_MIN))
    summarize(out)
    R0.log("done")
    return 0


# ====================================================================== summary
def verdict_about0(v):
    return None if v is None else bool(v[0] <= ABOUT0["point"] and v[2] <= ABOUT0["hi90"])


def merged(out, name):
    """A file of the run folder plus the same file of every shard folder in it (RA D2 merge); an item seen twice
    is an error."""
    parts = [out] + sorted(os.path.join(out, d) for d in (os.listdir(out) if os.path.isdir(out) else [])
                           if d.startswith("shard_") and os.path.isdir(os.path.join(out, d)))
    rows = []
    for p in parts:
        rows += R0.read_jsonl(os.path.join(p, name))
    return rows, [os.path.basename(p) for p in parts[1:]]


def summarize(out):
    rows, shards = merged(out, "rows.jsonl")
    scr, _s = merged(out, "screen.jsonl")
    metas, _s = merged(out, "items_done.jsonl")
    seen = {}
    for r in scr:
        if int(r["idx"]) in seen:
            R0.die(R0.EXIT_PLAN, "item %s screened in two shards" % r["idx"])
        seen[int(r["idx"])] = 1
    by = {}
    for r in rows:
        by.setdefault(int(r["idx"]), {})[r["arm"]] = r
    items = sorted(by)
    scrd = {int(r["idx"]): r for r in scr}
    res = {"n_items": len(items), "arms": {}}
    for a in ARM_NAMES:
        its = [i for i in items if a in by[i]]
        if not its:
            continue
        v = {"n": len(its)}
        for name, f in (("CA", lambda r: r["outcome"] == "B"), ("A", lambda r: r["outcome"] == "A"),
                        ("break", lambda r: r["outcome"] == "neither"), ("carry_B", lambda r: r["carry"] == "B"),
                        ("horizon", lambda r: r["hit_horizon"]), ("no_parse", lambda r: r["own"] is None)):
            v[name] = R0.boot_mean([1.0 if f(by[i][a]) else 0.0 for i in its])
        v["mean_chain"] = round(sum(by[i][a]["n_chain"] for i in its) / len(its), 1)
        res["arms"][a] = v

    def ca(a, its):
        return [1.0 if by[i][a]["outcome"] == "B" else 0.0 for i in its]
    both = lambda *a: [i for i in items if all(x in by[i] for x in a)]
    pr = {"thresholds": {"about0": ABOUT0, "most": MOST}}
    for a in ("cv1", "kv1"):
        v = res["arms"].get(a, {}).get("CA")
        pr["%s_about0" % a] = {"CA": v, "holds": verdict_about0(v)}
    it2 = both("cvP", "text")
    rr = R0.boot_ratio(ca("cvP", it2), ca("text", it2))
    pr["cvP_most"] = {"ratio": rr, "holds": None if rr is None else bool(rr[0] >= MOST["ratio"] and rr[1] >= MOST["lo90"])}
    pr["kvP_over_text"] = R0.boot_ratio(ca("kvP", both("kvP", "text")), ca("text", both("kvP", "text")))
    pr["eP_over_text"] = R0.boot_ratio(ca("eP", both("eP", "text")), ca("text", both("eP", "text")))
    pr["cvP_over_eP"] = R0.boot_ratio(ca("cvP", both("cvP", "eP")), ca("eP", both("cvP", "eP")))
    pr["base_CA"] = res["arms"].get("base", {}).get("CA")
    drops = {}
    for r in scr:
        k = r["drop"] or "eligible"
        drops[k] = drops.get(k, 0) + 1
    tot, late = 0, 0
    for i in items:
        nc = scrd.get(i, {}).get("n_clean")
        if nc is None:
            continue
        for _a, r in by[i].items():
            if r["own"] is None or r["marker_tok"] is None:
                continue
            tot += 1
            if r["marker_tok"] - r["n_prefix"] > cap_new(r["n_prefix"], nc, True):
                late += 1
    sec_items = [m["sec_item"] for m in metas]
    sec_clean = [r["sec_clean"] for r in scr if r.get("sec_clean") is not None]
    rate = drops.get("eligible", 0) / max(1, len(scr))
    n_pool = None
    for cand in [out] + [os.path.join(out, d) for d in shards]:
        prp = os.path.join(cand, "pool_record.json")
        if os.path.exists(prp):
            with open(prp, encoding="utf-8") as f:
                n_pool = json.load(f).get("n")
            break
    if n_pool is None:
        try:
            n_pool = RD.load_pool(DS, MODEL)["n"]
        except (OSError, KeyError):
            n_pool = None
    proj = None
    if n_pool and sec_items and sec_clean:
        proj = round((n_pool * sum(sec_clean) / len(sec_clean) + n_pool * rate * sum(sec_items) / len(sec_items))
                     / 3600.0, 2)
    checks = {"drops": drops, "eligible_rate": round(rate, 4), "n_pool": n_pool,
              "projected_eligible": round(rate * n_pool) if n_pool else None,
              "gate_80": "waived (RA D1: the full run needs n >= %d projected eligible)" % MIN_ELIGIBLE_FULL,
              "gate_n20": (rate * n_pool >= MIN_ELIGIBLE_FULL) if n_pool else None,
              "sec_per_item_mean": round(sum(sec_items) / len(sec_items), 2) if sec_items else None,
              "sec_per_arm_item": round(sum(sec_items) / len(sec_items) / len(ARM_NAMES), 3) if sec_items else None,
              "sec_clean_mean": round(sum(sec_clean) / len(sec_clean), 2) if sec_clean else None,
              "projected_full_card_hours": proj,
              "horizon_cap_rule": {"rows_with_answer": tot, "rows_marker_after_cap": late,
                                   "share": (late / tot) if tot else None,
                                   "use_cap_in_full": bool(tot) and late / tot < 0.01},
              "norm_ratio_prerounding": [m.get("norm_ratio_prerounding") for m in metas[:10]],
              "G2b": [m.get("G2b") for m in metas if m.get("G2b")],
              "r1_rule_kept_info": sum(1 for r in scr if r.get("r1_kept_info"))}
    checks["shards_merged"] = shards
    checks["shards_rule"] = {"projected_full_card_hours": proj, "shards": shard_count(proj),
                             "rule": "3 if <= 16.2 card-hours, else ceil(projected / 5.4)"}
    checks["breadth_cell"] = "McLeish r32: the 80-item gate is waived (RA D1); n is printed with every interval"
    res["predictions"] = pr
    res["checks"] = checks
    R0.write_json(os.path.join(out, "results.json"), res)
    R0.write_json(os.path.join(out, "checks.json"), checks)
    import csv
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "arm", "outcome", "carry", "own", "n_chain", "hit_horizon", "div_from_ref"])
        for i in items:
            for a in ARM_NAMES:
                if a in by[i]:
                    r = by[i][a]
                    w.writerow([i, a, r["outcome"], r["carry"], r["own"], r["n_chain"], r["hit_horizon"],
                                r.get("div_from_ref")])
    ex = [{"idx": i, "derived_A": scrd.get(i, {}).get("derived_A"), "derived_B": scrd.get(i, {}).get("derived_B"),
           "chains": {a: by[i][a]["text"] for a in ("clean", "text", "cv1", "cvP", "kvP", "eP") if a in by[i]}}
          for i in items[:(5 if "pilot" in out else 20)]]
    R0.write_json(os.path.join(out, "examples.json"), ex)
    lines = ["# RA-30 McLeish r32 %s" % os.path.basename(os.path.dirname(out)), "",
             "Breadth cell (RA D1: the 80-item gate is waived). Items n = %d%s. Pool: the steer_gen_raven McLeish "
             "%s scored list (%s problems). K = 32." % (len(items), (", merged from %d shards" % len(shards)) if shards
                                                       else "", DS_NAME[DS], n_pool), "",
             "| arm | n | CA | 90% CI | A | break | carry B |", "|---|---|---|---|---|---|---|"]
    for a in ARM_NAMES:
        v = res["arms"].get(a)
        if v:
            c = v["CA"]
            lines.append("| %s | %d | %.3f | [%.3f, %.3f] | %.3f | %.3f | %.3f |" % (
                a, v["n"], c[0], c[1], c[2], v["A"][0], v["break"][0], v["carry_B"][0]))
    lines += ["", "Predictions (about 0: CA <= %.2f and 90%% upper bound <= %.2f; most: CA(cvP)/CA(text) >= %.2f "
              "and 90%% lower bound >= %.2f):" % (ABOUT0["point"], ABOUT0["hi90"], MOST["ratio"], MOST["lo90"]),
              json.dumps(pr), "", "Checks: %s" % json.dumps(checks, sort_keys=True)]
    if DS == "asdiv":
        lines += ["", R0.ASDIV_NOTE.replace("plan 6g: the gate is waived", "plan 6g and RA D1: the gate is waived")]
    import ra30_flags                         # audit wording limits (W2, W10): never an identity line
    nt = ra30_flags.notes(out, rows=rows, scr=scr)
    with open(os.path.join(out, "summary_notes.md"), encoding="utf-8") as f:
        lines += ["", "## Notes (audit wording limits)", ""] + [x for x in f.read().split("\n")[2:] if x]
    res["notes"] = nt
    R0.write_json(os.path.join(out, "results.json"), res)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return res


def host_tokens(out, tok=None, pool_root=None):
    """Host tokenizer step (CPU): every pool prompt rebuilt with McLeish's encoding; out/<model>/tokens."""
    import transformers
    if tok is None:
        if transformers.__version__ != R0.TRANSFORMERS_PIN:
            R0.die(R0.EXIT_PLAN, "transformers %s, pinned %s" % (transformers.__version__, R0.TRANSFORMERS_PIN))
        SGR.ACTIVE[0] = "mcl"
        tok = SGR.load_tokenizer()
    d = os.path.join(model_root(out), "tokens")
    os.makedirs(d, exist_ok=True)
    pool, _rec = get_pool(pool_root)
    bad, rows = [], []
    for it in pool["items"]:
        pA, pB = prompt_ids(tok, it["question_A"]), prompt_ids(tok, it["question_B"])
        s0, s1 = int(it["span"][0]), int(it["span"][1])
        ok = (len(pA) == len(pB) == int(it["plen"]) and pA[:s0] == pB[:s0] and pA[s1:] == pB[s1:]
              and pA[s0:s1] != pB[s0:s1])
        rows.append({"idx": int(it["idx"]), "prompt_A": pA, "prompt_B": pB, "ok": ok})
        if not ok:
            bad.append(int(it["idx"]))
    with open(os.path.join(d, "%s.jsonl" % DS), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    chk = {"dataset": DS, "model": MODEL, "n": len(rows), "n_mismatch": len(bad), "mismatch_idx": bad,
           "transformers": transformers.__version__, "pass": len(bad) <= 0.01 * len(rows)}
    R0.write_json(os.path.join(d, "%s_check.json" % DS), chk)
    R0.log("host tokens mcleish %s: %d items, %d mismatches" % (DS, len(rows), len(bad)))
    if not chk["pass"]:
        R0.die(R0.EXIT_PLAN, "more than 1 percent of McLeish prompts do not rebuild on the host tokenizer")
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--plan_sha256", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--no_plan_check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ds", choices=("gsm8k", "asdiv"), default="gsm8k")
    ap.add_argument("--pool_root", default=os.environ.get("OUT_ROOT"))
    a = ap.parse_args(argv)
    set_ds(a.ds)
    if a.tokens:
        host_tokens(a.out, pool_root=a.pool_root)
        return 0
    if a.summarize:
        summarize(a.out)
        return 0
    if a.run:
        return run(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
