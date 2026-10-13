"""RA-30 (2026-10-12): edit the intermediate value in the chain, with the neuron arm set, Ouro-1.4B base.
Plan: plan_ra30.md in this folder (its sha256 is in plan_ra30.sha256 and is checked before anything runs).

    python ra30.py --hash
    python ra30.py --tokens --ds gsm8k --out OUT                                      (host tokenizer, CPU)
    python ra30.py --run --stage pilot --ds gsm8k --out OUT --plan_sha256 SHA
    python ra30.py --run --stage full  --ds gsm8k --out OUT --plan_sha256 SHA      (resumable)
    python ra30.py --summarize --out OUT/<stage>/<ds>

A run: S screen (every scanned pool item: the clean greedy chain of prompt A, batch 1, then the CPU screen and the
token geometry on the host tokenizer) -> split (the first n_split eligible items in the stored order) -> X
precompute (neuron activations of the split, the dataset-level sets) -> G gates on the gate items (G1 to G5, before
any scored row) -> R arm rows per item (one 15-row prefix batch and one 9-row scratch batch) -> G6 (pilot) ->
summary. Exit codes: 2 plan, pin, data or input mismatch; 3 gate failed; 4 identity failed on a scored row; 5
stopped by SIGTERM (rerun the same command to resume).
"""
import argparse
import hashlib
import json
import math
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

import arms as AR  # noqa: E402
import chain as CH  # noqa: E402
import ra30_data as RD  # noqa: E402

PLAN = os.path.join(HERE, "plan_ra30.md")
SHA_FILE = os.path.join(HERE, "plan_ra30.sha256")
MODEL = RD.DEFAULT_MODEL           # set by --model (plan section 6b); the 1.4B paths and files never change
TRANSFORMERS_PIN = "4.56.2"
K = RD.K
HORIZON = RD.HORIZON
N_SCORED_PILOT = {"gsm8k": 25, "asdiv": 20}
# RA decision after the GSM8K pilot (plan section 6a): the 12 neuron arms gave CA 0.000 and are not run in the
# mains; the code stays. With them off there is no precompute split, so every eligible item is scored.
RUN_NEURON_ARMS = False
ACTIVE_ARMS = [a for a in AR.ARM_NAMES if RUN_NEURON_ARMS or not a.startswith(("nc_", "ns_"))]
# The GSM8K pilot of record ran under plan 7f96eb63 (all gates passed); its checks.json (eligible count, horizon cap
# decision) is accepted by the full run of the current plan.
ACCEPTED_PILOT_PLANS = ("7f96eb63b0b503136fc376c37c123bdd8fa006f9e02e36482caf7dc23290b7bf",
                        "ed722548db0644b604f194c889d0fe1b649a4e9e02cf27ed211ea667debfd10e",
                        "352bd818e1b69a338e157877eb2ced3c39a73bccc72f84c123b9e266b1215d8e")
N_SCORED_PILOT_MODEL = {"ouro_2_6b_base": {"gsm8k": 20, "asdiv": 20}}   # the 2.6B axis: 20 scored pilot items


def set_model(name):
    global MODEL
    if name not in RD.MODELS:
        die(EXIT_PLAN, "unknown model %s" % name)
    MODEL = name


def model_root(out_root):
    """out/ for 1.4B (the paths of record), out/<model>/ for every other model."""
    return out_root if MODEL == RD.DEFAULT_MODEL else os.path.join(out_root, MODEL)


def pool_of(ds):
    return RD.load_pool(ds) if MODEL == RD.DEFAULT_MODEL else RD.load_pool(ds, MODEL)


def order_of(ds):
    return RD.load_split(ds) if MODEL == RD.DEFAULT_MODEL else RD.load_split(ds, MODEL)


def n_scored_pilot(ds):
    return N_SCORED_PILOT_MODEL.get(MODEL, N_SCORED_PILOT)[ds]
EXAMPLE_IDX = {"gsm8k": (11,)}       # chains kept in every summary's examples.json (the paper's example box)
DS_NAME = {"gsm8k": "GSM8K", "asdiv": "ASDiv"}
GATE_MIN_ELIGIBLE = 80
# plan 6g (RA): the 80-item gate is waived for ASDiv (1.4B and 2.6B); the full run needs n >= 20; GSM8K keeps 80
GATE_MIN_BY_DS = {"asdiv": 20}
ASDIV_NOTE = "ASDiv: mostly one-step problems; under the 80 rule (plan 6g: the gate is waived, n >= 20)."
N_GATE = {"pilot": 20, "full": 5}
TIE = 1.0                         # G2 / G3 tie rule: a divergence passes only at a top-2 margin below 1 logit
NB = 10000
CEILING_MIN = 0.5
EXIT_PLAN, EXIT_GATE, EXIT_IDENTITY, EXIT_STOPPED = 2, 3, 4, 5
STOP = {"flag": False}


def die(code, msg):
    print("[ra30] %s" % msg, flush=True)
    raise SystemExit(code)


def log(msg):
    print("[ra30] %s" % msg, flush=True)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    return RD.read_jsonl(path) if os.path.exists(path) else []


# ====================================================================== context
class Ctx(object):
    """The adapter, the hooks and the tokenizer of one process. `ad` is a prod SteeredOuroAdapter (kv=True)."""

    def __init__(self, ad, steer, hook, tok, ds):
        self.ad, self.steer, self.hook, self.tok, self.ds = ad, steer, hook, tok, ds
        self.model = ad.model
        self.stops, self.eos = RD.stops_and_eos(tok, ds)


def load_ctx(ds):
    """Ouro-1.4B base through prod's SteeredOuroAdapter (pins: revision, modeling md5, transformers)."""
    import transformers
    if transformers.__version__ != TRANSFORMERS_PIN:
        die(EXIT_PLAN, "transformers %s, pinned %s" % (transformers.__version__, TRANSFORMERS_PIN))
    from prod import steer_gen as S
    steer = S.SteerHooks(K)
    ad = S.SteeredOuroAdapter(MODEL, steer=steer, kv=True)
    ad.load()
    ad.set_depth(K)
    L = int(ad.model.config.num_hidden_layers)
    hook = AR.NeuronHook(steer, layer=L - 1, loop=1).install(ad.model)
    if next(ad.model.parameters()).dtype != torch.bfloat16:
        die(EXIT_PLAN, "model dtype is not bf16")
    return Ctx(ad, steer, hook, ad.tok, ds)


def clear_item(ctx):
    """Review B2: nothing of an item survives into the next one."""
    st = ctx.steer
    st.rows.clear()
    st.donors.clear()
    st.stats.clear()
    st.prompt_counted.clear()
    ctx.hook.specs.clear()
    ctx.hook.n_applied.clear()


# ====================================================================== screen
def clean_decode(ctx, prompt):
    ctx.steer.rows.clear()
    gen, _t = ctx.ad.decode([list(prompt)], HORIZON, K, ctx.eos, stop_strings=ctx.stops, row_ids=None)
    return list(gen[0])


def screen_item(ctx, it, clean_fn=None):
    """One pool item: regenerate the clean chain, parse, screen, token geometry (host tokenizer). Returns the
    screen record (drop None = eligible)."""
    tok, ds = ctx.tok, ctx.ds
    pA = RD.prompt_ids(tok, ds, it["question_A"])
    pB = RD.prompt_ids(tok, ds, it["question_B"])
    s0, s1 = int(it["span"][0]), int(it["span"][1])
    rec = {"idx": int(it["idx"]), "plen": len(pA)}
    if not (len(pA) == len(pB) == int(it["plen"])) or pA[:s0] != pB[:s0] or pA[s1:] != pB[s1:] \
            or pA[s0:s1] == pB[s0:s1]:
        rec.update(drop="plen_mismatch", clean_ids=None)                   # review B3
        return rec
    t0 = time.time()
    ids = (clean_fn or clean_decode)(ctx, pA)
    rec["sec_clean"] = round(time.time() - t0, 3)
    from prod.models.ouro import strip_tail
    ids = strip_tail([ids], ctx.eos)[0]
    cut, txt, own = RD.cut_parse(tok, ds, ids)
    rec.update(clean_ids=ids, cut=cut, own=own, n_clean=len(ids))
    if CH.outcome(own, it["gold_A"], it["gold_B"]) != "A":
        rec["drop"] = "clean_wrong"
        return rec
    a = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"])
    r1 = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"], mode="r1")
    rec["r1_kept_info"] = r1["drop"] is None
    if a["drop"] is None:
        a = CH.token_geometry(tok, pA, ids, a)
    rec["drop"] = a["drop"]
    if a["drop"] is None:
        keep = ("derived_A", "derived_B", "answer_edit", "answer_full_subst", "P", "prefix_A", "prefix_B",
                "prefix_len", "tokens_A", "tokens_B", "canonical_B", "e1", "i0", "m", "derived_B_in_prompt",
                "gold_B_in_prompt_or_prefix", "n_lead_same")
        rec.update({k: a[k] for k in keep})
    return rec


def geom_of(it, sc, tok, ds):
    return {"prompt_A": RD.prompt_ids(tok, ds, it["question_A"]), "prompt_B": RD.prompt_ids(tok, ds, it["question_B"]),
            "prefix_A": sc["prefix_A"], "prefix_B": sc["prefix_B"], "P": sc["P"], "span": it["span"],
            "clean_ids": sc["clean_ids"]}


# ====================================================================== neuron precompute
def acts_for(ctx, g, setting):
    """(a_A, a_D) at the edited positions: chain value (A + prefix_A vs A + prefix_B at P) or span (prompt A vs B)."""
    if setting == "chain":
        lo, hi = g["P"]
        seqA, seqD = g["prompt_A"] + g["prefix_A"], g["prompt_A"] + g["prefix_B"]
    else:
        lo, hi = int(g["span"][0]), int(g["span"][1])
        seqA, seqD = g["prompt_A"], g["prompt_B"]
    return (AR.record_acts(ctx.model, ctx.steer, ctx.hook, seqA, lo, hi),
            AR.record_acts(ctx.model, ctx.steer, ctx.hook, seqD, lo, hi))


def dataset_sets(ctx, geoms):
    I = int(ctx.model.config.intermediate_size)
    out = {}
    for setting in ("chain", "span"):
        pairs = [acts_for(ctx, g, setting) for g in geoms]
        out[setting] = AR.top_set(AR.dataset_scores(pairs), AR.k_of(AR.FRAC["ds5"], I)) if pairs else None
    return out


# ====================================================================== rows of one item
def build_rows(ctx, it, g, ds_sets, in_split, arms=None):
    """steer.rows, shared donors (recorded once per batch from the donor row, K/V included) and neuron specs.
    Returns {arm: key} of the arms built."""
    ds, idx = ctx.ds, int(it["idx"])
    item_key = "%s:%d" % (ds, idx)
    names = list(arms or AR.ARM_NAMES)
    keys = {}
    for a in names:
        k = AR.key_of(ds, idx, a)
        ctx.steer.rows[k] = AR.row_info(a, g, item_key)
        keys[a] = k
    for batch, drow in AR.DONOR_ROW.items():
        need = [a for a in names if AR.ARM[a]["batch"] == batch and ctx.steer.needs_donor(keys[a])]
        if not need:
            continue
        dk = AR.key_of(ds, idx, drow)
        if dk not in ctx.steer.rows:
            ctx.steer.rows[dk] = AR.row_info(drow, g, item_key)
        d = ctx.ad.record_donor(dk)
        for a in need:
            ctx.steer.donors[keys[a]] = d
    sets = {}
    for setting, pre in (("chain", "nc_"), ("span", "ns_")):
        nn = [a for a in names if a.startswith(pre)]
        if not nn:
            continue
        aA, aD = acts_for(ctx, g, setting)
        ds_set = None if in_split else (ds_sets or {}).get(setting)
        sets[setting] = AR.neuron_sets(aA, aD, ds_set, MODEL, ds, setting, item_key)
        lo, hi = AR.neuron_range(nn[0], g)
        for a in nn:
            s = a[len(pre):]
            if s not in sets[setting]:
                ctx.steer.rows.pop(keys.pop(a))
                continue
            idx_t = None if sets[setting][s] is None else torch.tensor(sets[setting][s], dtype=torch.long)
            ctx.hook.specs[keys[a]] = {"lo": lo, "hi": hi, "idx": idx_t, "donor": aD}
    return keys, sets


def readout(ctx, it, sc, arm, chain_ids, n_prefix):
    tok = ctx.tok
    from prod.models.ouro import strip_tail
    ids = strip_tail([list(chain_ids)], ctx.eos)[0]
    cut, txt, own = RD.cut_parse(tok, ctx.ds, ids)
    cont = tok.decode(ids[n_prefix:cut], clean_up_tokenization_spaces=False) if cut > n_prefix else ""
    mk = txt.rfind(CH.OWN_MARKER)
    marker_tok = None
    if mk >= 0:
        ends = CH.char_offsets(tok, ids[:cut])
        marker_tok = next((i for i, e in enumerate(ends) if e > mk), None)
    return {"arm": arm, "chain_ids": ids, "cut": cut, "text": txt, "own": own,
            "outcome": CH.outcome(own, it["gold_A"], it["gold_B"]),
            "carry": CH.carry(cont, sc["derived_A"], sc["derived_B"]), "n_chain": len(ids),
            "hit_horizon": len(ids) >= HORIZON, "marker_tok": marker_tok, "n_prefix": n_prefix}


def decode_batch(ctx, keys, n_new):
    seqs = [ctx.steer.rows[k]["a_ids"] for k in keys]
    if len({len(s) for s in seqs}) != 1:
        raise RuntimeError("rows of one batch differ in length")
    gen, _t = ctx.ad.decode(seqs, n_new, K, ctx.eos, stop_strings=ctx.stops, row_ids=keys)
    return gen


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def cap_new(n_pre, n_clean, use_cap):
    """New tokens for a batch: the horizon, or under the pre-registered cap rule (plan section 10)
    min(512 - prefix, 2 x the clean chain's continuation after the prefix + 64)."""
    full = HORIZON - n_pre
    return min(full, 2 * (int(n_clean) - n_pre) + 64) if use_cap else full


def run_item(ctx, it, sc, ds_sets, in_split, use_cap=False):
    """All arms of one item: the prefix batch, then the scratch batch. Returns the rows (one per arm)."""
    g = geom_of(it, sc, ctx.tok, ctx.ds)
    t0 = time.time()
    keys, sets = build_rows(ctx, it, g, ds_sets, in_split, arms=ACTIVE_ARMS)
    t1 = time.time()
    out = []
    plen = len(g["prompt_A"])
    for batch, n_pre in (("prefix", len(g["prefix_A"])), ("scratch", 0)):
        names = [a for a in AR.ARM_NAMES if AR.ARM[a]["batch"] == batch and a in keys]
        bk = [keys[a] for a in names]
        tb = time.time()
        gen = decode_batch(ctx, bk, cap_new(n_pre, sc["n_clean"], use_cap))
        sec = time.time() - tb
        for a, k, gg in zip(names, bk, gen):
            pre = list(ctx.steer.rows[k]["a_ids"][plen:])
            r = readout(ctx, it, sc, a, pre + list(gg), n_pre)
            r["sec_batch"] = round(sec, 3)
            r["batch_rows"] = len(names)
            out.append(r)
    ref = {r["arm"]: r["chain_ids"] for r in out}
    for r in out:
        base = ref["floor"] if AR.ARM[r["arm"]]["batch"] == "prefix" else ref["clean"]
        r["div_from_ref"] = first_div(r["chain_ids"], base)
    stats = {"%s|%s" % kk: v for kk, v in ctx.steer.stats.items()}
    for r in out:
        k = keys[r["arm"]]
        st = {s.split("|", 2)[-1]: v for s, v in stats.items() if s.startswith(k + "|")}
        r["edit_stats"] = st                                   # [sum |delta|, sum |delta|/|h|, n] (pre-rounding)
        r["neuron_applied"] = ctx.hook.n_applied.get(k, 0)
    nr = {}
    for rn, cv, site in (("rnd1", "cv1", "post@1"), ("rnd3", "cv3", "post@3")):
        a = stats.get("%s|%s" % (keys.get(rn), site))
        b = stats.get("%s|%s" % (keys.get(cv), site))
        if a and b and b[0] > 0:
            nr[rn] = round(a[0] / b[0], 6)
    meta = {"idx": int(it["idx"]), "in_split": bool(in_split), "sec_setup": round(t1 - t0, 3),
            "sec_item": round(time.time() - t0, 3), "norm_ratio_prerounding": nr,
            "sets": {s: {k: v for k, v in d.items() if k != "full"} for s, d in sets.items()}}
    return out, meta


# ====================================================================== gates
@torch.no_grad()
def prefill(ctx, seq, keys, rec=None):
    """The prefill of prod's OuroAdapter.decode for one row (StaticUTCache kept), optional steer recorder."""
    st = ctx.steer
    T = len(seq)
    dev = next(ctx.model.parameters()).device
    st.f = None
    st.ensure_donors(keys)
    cache = ctx.ad._cache_factory(1, T + 1)
    st.batch_keys, st.rec = keys, rec
    try:
        out = ctx.model(input_ids=torch.tensor([list(seq)], device=dev),
                        attention_mask=torch.ones(1, T, dtype=torch.long, device=dev),
                        position_ids=torch.arange(T, device=dev)[None], past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(T, device=dev), logits_to_keep=1)
    finally:
        st.batch_keys, st.rec = None, None
    return out.logits[:, -1].float().cpu(), cache


def new_rec(lo, hi, want):
    return {"cols": slice(int(lo), int(hi)), "want": set(want), "post": {}, "pre": {}, "kv": {}, "emb": None,
            "logits": []}


@torch.no_grad()
def margin_at(ctx, seq):
    """Top-2 logit margin of the next token after seq (one no-cache forward)."""
    dev = next(ctx.model.parameters()).device
    z = ctx.model(input_ids=torch.tensor([list(seq)], device=dev), use_cache=False, logits_to_keep=1).logits[0, -1]
    t = z.float().topk(2).values
    return float(t[0] - t[1])


def tie_check(ctx, seq_prefix, a, b):
    """Token sequences a (reference) and b continue seq_prefix. Identical, or the first divergence sits at a top-2
    margin below TIE on the reference path."""
    d = first_div(a, b)
    if d is None:
        return {"identical": True, "pass": True}
    if d >= min(len(a), len(b)):
        return {"identical": False, "div": d, "pass": False, "why": "length"}
    m = margin_at(ctx, list(seq_prefix) + list(a[:d]))
    return {"identical": False, "div": d, "margin": round(m, 4), "pass": m < TIE}


@torch.no_grad()
def gates_item(ctx, it, sc):
    """G1 to G5 for one gate item (review B1, B2, N5). Returns a dict with `pass`."""
    tok, ds, idx = ctx.tok, ctx.ds, int(it["idx"])
    g = geom_of(it, sc, tok, ds)
    item_key = "%s:%d" % (ds, idx)
    seqA = g["prompt_A"] + g["prefix_A"]
    seqB = g["prompt_A"] + g["prefix_B"]
    lo, hi = g["P"]
    plen, npre = len(g["prompt_A"]), len(g["prefix_A"])
    res = {"idx": idx}
    names = ["floor", "cv1", "cv2", "cv3", "cvP", "kv1", "kv4", "rnd1", "rnd3", "nc_full"]
    for a in names:
        ctx.steer.rows[AR.key_of(ds, idx, a)] = AR.row_info(a, g, item_key)
    K_ = {a: AR.key_of(ds, idx, a) for a in names}
    # G1 hook identity: a registered idle row against no row keys
    z0, c0 = prefill(ctx, seqA, None)
    z1, c1 = prefill(ctx, seqA, [K_["floor"]])
    L = int(ctx.model.config.num_hidden_layers)
    res["G1"] = bool(torch.equal(z0, z1)) and all(torch.equal(c0.key_cache[i][:, :, :len(seqA)],
                                                               c1.key_cache[i][:, :, :len(seqA)])
                                                   for i in range(L * K))
    # G2 floor alone (batch 1) against the clean chain after the prefix
    clean = list(sc["clean_ids"])
    from prod.models.ouro import strip_tail
    flo = ctx.ad.decode([seqA], HORIZON - npre, K, ctx.eos, stop_strings=ctx.stops, row_ids=[K_["floor"]])[0][0]
    flo = strip_tail([list(flo)], ctx.eos)[0]
    res["G2"] = tie_check(ctx, seqA, clean[npre:], list(flo))
    res["floor_alone"] = list(flo)
    # G3 teacher-forced prefix: argmax at every prefix position equals the next prefix token
    dev = next(ctx.model.parameters()).device
    z = ctx.model(input_ids=torch.tensor([seqA], device=dev), use_cache=False,
                  logits_to_keep=npre + 1).logits[0, :-1].float()
    am = z.argmax(-1).tolist()
    bad = [i for i, (x, y) in enumerate(zip(am, g["prefix_A"])) if x != y]
    if bad:
        top = z[bad[0]].topk(2).values
        res["G3"] = {"n_bad": len(bad), "first": bad[0], "margin": round(float(top[0] - top[1]), 4),
                     "pass": float(top[0] - top[1]) < TIE}
    else:
        res["G3"] = {"n_bad": 0, "pass": True}
    # G4 positions
    res["G4"] = {"P": [lo, hi], "tokens_A": [tok.decode([i]) for i in seqA[lo:hi]],
                 "tokens_B": [tok.decode([i]) for i in seqB[lo:hi]], "equal_len": len(seqA) == len(seqB),
                 "P_ends_prefix": hi == plen + npre}
    # G5 edits land, against a FRESH donor record of A + prefix_B (review B2)
    fresh = ctx.ad.record_donor(K_["kv1"])
    for a in ("cv1", "cv2", "cv3", "cvP", "kv1", "kv4", "rnd1", "rnd3"):
        ctx.steer.donors[K_[a]] = fresh
    ok = True
    recs = {}
    for a in ("floor", "cv1", "cv2", "cv3", "rnd1", "rnd3"):
        recs[a] = new_rec(lo, hi, {"post"})
        prefill(ctx, seqA, [K_[a]], rec=recs[a])
    for j in (1, 2, 3):
        ok &= bool(torch.equal(recs["cv%d" % j]["post"][j][0], fresh["post"][j]))
    rk = new_rec(lo, hi, {"kv"})
    prefill(ctx, seqA, [K_["kv1"]], rec=rk)
    for li in range(L):
        for w in ("k", "v"):
            ok &= bool(torch.equal(rk["kv"][(0, li, w)][0], fresh["kv"][(0, li, w)]))
    res["G5_patch_lands"] = bool(ok)
    # N5: norm ratio on the applied bf16 delta (pooled over P), tolerance 1e-2
    nr = {}
    for rn, cv, j in (("rnd1", "cv1", 1), ("rnd3", "cv3", 3)):
        dr = (recs[rn]["post"][j][0].float() - recs["floor"]["post"][j][0].float()).norm()
        dc = (recs[cv]["post"][j][0].float() - recs["floor"]["post"][j][0].float()).norm()
        nr[rn] = round(float(dr / dc), 5) if float(dc) > 0 else None
    res["norm_ratio_applied"] = nr
    res["norm_ratio_applied_pass"] = all(v is not None and abs(v - 1) <= 1e-2 for v in nr.values())
    # neuron full set: a at P equals the donor's
    aA, aD = acts_for(ctx, g, "chain")
    ctx.hook.specs[K_["nc_full"]] = {"lo": lo, "hi": hi, "idx": None, "donor": aD}
    seen = {}

    def spy(module, args):
        f = ctx.steer.f
        if f is not None and f.get("ut") == 0 and f.get("keys") is not None:
            seen["a"] = args[0][0, lo:hi].detach().clone()
    h = ctx.model.model.layers[L - 1].mlp.down_proj.register_forward_pre_hook(spy)
    try:
        prefill(ctx, seqA, [K_["nc_full"]])
    finally:
        h.remove()
    res["G5_neuron_full"] = bool(torch.equal(seen.get("a"), aD)) if "a" in seen else False
    # B1 identities at the cache level on the card: cvP = cv1, kv4 = cv3 (K/V of every loop, layer, position)
    T = len(seqA)
    for a, b in AR.IDENTITIES:
        _za, ca = prefill(ctx, seqA, [K_[a]])
        _zb, cb = prefill(ctx, seqA, [K_[b]])
        res["kv_equal_%s_%s" % (a, b)] = all(torch.equal(ca.key_cache[i][:, :, :T], cb.key_cache[i][:, :, :T])
                                             and torch.equal(ca.value_cache[i][:, :, :T], cb.value_cache[i][:, :, :T])
                                             for i in range(L * K))
    # identity rows decoded alone: token-identical share is printed (kv4 lives only here)
    ida = ["cv1", "cvP", "cv3", "kv4"]                    # one 4-row batch (equal lengths, no padding)
    gen = ctx.ad.decode([seqA] * len(ida), HORIZON - npre, K, ctx.eos, stop_strings=ctx.stops,
                        row_ids=[K_[a] for a in ida])[0]
    dec = {a: strip_tail([list(x)], ctx.eos)[0] for a, x in zip(ida, gen)}
    for a, b in AR.IDENTITIES:
        res["tok_identical_%s_%s" % (a, b)] = tie_check(ctx, seqA, list(dec[b]), list(dec[a]))
    res["pass"] = bool(res["G1"] and res["G2"]["pass"] and res["G3"]["pass"] and res["G4"]["equal_len"]
                       and res["G4"]["P_ends_prefix"] and res["G5_patch_lands"] and res["G5_neuron_full"])
    clear_item(ctx)
    return res


# ====================================================================== the run
def check_plan(sha):
    with open(SHA_FILE, encoding="utf-8") as f:
        want = f.read().split()[0]
    got = RD.sha256_file(PLAN)
    if sha != want or got != want:
        die(EXIT_PLAN, "plan sha256 mismatch (file %s, plan_ra30.sha256 %s, argument %s)" % (got, want, sha))
    return got


def check_data():
    with open(os.path.join(RD.DATA, RD.sources_name(MODEL)), encoding="utf-8") as f:
        src = json.load(f)
    for fn, h in src["files"].items():
        if RD.sha256_file(os.path.join(RD.DATA, fn)) != h:
            die(EXIT_PLAN, "data file %s differs from sources.json" % fn)
    return src


def code_hashes():
    fs = ["ra30.py", "lib/arms.py", "lib/chain.py", "lib/ra30_data.py", "lib/ra28_score.py",
          "tree/prod/steer_gen.py", "tree/prod/models/ouro.py", "tree/prod/tasks/__init__.py"]
    return {f: RD.sha256_file(os.path.join(HERE, f)) for f in fs}


def on_term(_s, _f):
    STOP["flag"] = True


def pilot_of_record(out_root, ds, plan_sha):
    """The full run reads its dataset's pilot: plan accepted (the current plan or one in ACCEPTED_PILOT_PLANS),
    the 80-item gate on the projected eligible count, and the horizon cap decision."""
    d = os.path.join(out_root, "pilot", ds)
    pc, ph = os.path.join(d, "checks.json"), os.path.join(d, "hash_line.json")
    if not (os.path.exists(pc) and os.path.exists(ph)):
        die(EXIT_PLAN, "no pilot of %s in %s; run the pilot first" % (ds, d))
    with open(pc, encoding="utf-8") as f:
        chk = json.load(f)
    with open(ph, encoding="utf-8") as f:
        pplan = json.load(f).get("plan_sha256")
    if pplan not in ACCEPTED_PILOT_PLANS + ((plan_sha,) if plan_sha else ()):
        die(EXIT_PLAN, "the pilot of %s ran under plan %s, which is not accepted" % (ds, pplan))
    n = chk.get("projected_eligible")
    n_min = GATE_MIN_BY_DS.get(ds, GATE_MIN_ELIGIBLE)
    if n is None or n < n_min:
        die(EXIT_GATE, "%s under %d: gate failure (projected eligible %s)" % (DS_NAME[ds], n_min, n))
    return {"use_cap": bool((chk.get("horizon_cap_rule") or {}).get("use_cap_in_full")), "plan_sha256": pplan,
            "projected_eligible": n}


@torch.no_grad()
def neuron_nonvacuity(ctx, it, sc):
    """RA check after the pilot (neuron arms CA 0.000): on one gate item, nc_full and ns_full applied once. Prints the
    norm of the applied change of the down_proj input a (and of its MLP output down_proj(a)) at the edited positions
    and whether the next-token logits differ from the idle pass. Verdict: 'inert hook' if the change is 0 or the
    logits are bit-equal, else 'no effect of the last-layer MLP part'."""
    tok, ds, idx = ctx.tok, ctx.ds, int(it["idx"])
    g = geom_of(it, sc, tok, ds)
    item_key = "%s:%d" % (ds, idx)
    L = int(ctx.model.config.num_hidden_layers)
    down = ctx.model.model.layers[L - 1].mlp.down_proj
    out = {}
    for arm, setting in (("nc_full", "chain"), ("ns_full", "span")):
        if setting == "chain":
            seq, (lo, hi) = g["prompt_A"] + g["prefix_A"], g["P"]
        else:
            seq, (lo, hi) = g["prompt_A"], (int(g["span"][0]), int(g["span"][1]))
        aA, aD = acts_for(ctx, g, setting)
        key = AR.key_of(ds, idx, arm)
        ctx.steer.rows[key] = AR.row_info(arm, g, item_key)
        seen = {}

        def spy(module, args):
            f = ctx.steer.f
            if f is not None and f.get("ut") == 0 and f.get("keys") is not None:
                seen["a"] = args[0][0, lo:hi].detach().clone()
        h = down.register_forward_pre_hook(spy)
        try:
            z0, _c = prefill(ctx, seq, [key])
            a0 = seen.pop("a")
            ctx.hook.specs[key] = {"lo": lo, "hi": hi, "idx": None, "donor": aD}
            z1, _c = prefill(ctx, seq, [key])
            a1 = seen.pop("a")
        finally:
            h.remove()
        da = (a1.float() - a0.float())
        du = down(a1).float() - down(a0).float()
        eq = bool(torch.equal(z0, z1))
        rec = {"positions": [lo, hi], "delta_a_norm": round(float(da.norm()), 6),
               "delta_mlp_out_norm": round(float(du.norm()), 6), "logits_equal": eq,
               "logits_max_abs_diff": round(float((z1 - z0).abs().max()), 6),
               "a_equals_donor": bool(torch.equal(a1, aD.to(a1.device)))}
        rec["verdict"] = ("inert hook" if (rec["delta_a_norm"] == 0 or eq)
                          else "no effect of the last-layer MLP part")
        out[arm] = rec
        log("neuron non-vacuity %s: |delta a| %.4g, |delta MLP out| %.4g, logits equal %s, max |dz| %.4g -> %s"
            % (arm, rec["delta_a_norm"], rec["delta_mlp_out_norm"], eq, rec["logits_max_abs_diff"], rec["verdict"]))
    clear_item(ctx)
    return out


def run(args, ctx=None, clean_fn=None):
    ds, stage = args.ds, args.stage
    out = os.path.join(model_root(args.out), stage, ds)
    os.makedirs(out, exist_ok=True)
    plan_sha = check_plan(args.plan_sha256) if not args.no_plan_check else None
    src = check_data()
    pool = pool_of(ds)
    order = order_of(ds)["order"]
    by = {int(it["idx"]): it for it in pool["items"]}
    n_split = RD.SPLIT_N[stage] if RUN_NEURON_ARMS else 0
    n_target = (n_split + n_scored_pilot(ds)) if stage == "pilot" else None
    use_cap, pilot_plan = False, None
    if stage == "full":
        pilot = pilot_of_record(model_root(args.out), ds, plan_sha)
        use_cap, pilot_plan = pilot["use_cap"], pilot["plan_sha256"]
    cfg = {"ds": ds, "stage": stage, "model": MODEL, "k": K, "horizon": HORIZON, "use_cap": use_cap,
           "n_split": n_split, "run_neuron_arms": RUN_NEURON_ARMS, "pilot_plan_sha256": pilot_plan,
           "n_scored_pilot": n_scored_pilot(ds) if stage == "pilot" else None, "arms": ACTIVE_ARMS,
           "identity_arms": [a[0] for a in AR.IDENTITY_ARMS], "tie": TIE}
    hl = os.path.join(out, "hash_line.json")
    hline = {"plan_sha256": plan_sha, "config_sha256": sha256_text(json.dumps(cfg, sort_keys=True)),
             "code_sha256": code_hashes(), "data_sha256": src["files"]}
    if os.path.exists(hl):
        with open(hl, encoding="utf-8") as f:
            old = json.load(f)
        if {k: old.get(k) for k in hline} != hline:
            die(EXIT_PLAN, "inputs changed since the first row of %s (hash_line.json)" % out)
    else:
        write_json(hl, hline)
        write_json(os.path.join(out, "config.json"), cfg)
    signal.signal(signal.SIGTERM, on_term)
    if ctx is None:
        ctx = load_ctx(ds)
    # ---- S: screen in the stored order (resumable)
    sp = os.path.join(out, "screen.jsonl")
    screened = {int(r["idx"]): r for r in read_jsonl(sp)}
    elig = [i for i in order if i in screened and screened[i]["drop"] is None]
    for i in order:
        if n_target is not None and len(elig) >= n_target:
            break
        if i in screened:
            continue
        if STOP["flag"]:
            die(EXIT_STOPPED, "stopped during the screen")
        r = screen_item(ctx, by[i], clean_fn)
        append_jsonl(sp, r)
        screened[i] = r
        if r["drop"] is None:
            elig.append(i)
    if n_target is not None:
        elig = elig[:n_target]
    split, scored = elig[:n_split], elig[n_split:]
    log("%s %s: scanned %d, eligible %d, split %d, scored %d" % (stage, ds, len(screened), len(elig), len(split),
                                                                 len(scored)))
    # ---- X: dataset-level neuron sets from the split
    setp = os.path.join(out, "sets.json")
    if os.path.exists(setp):
        with open(setp, encoding="utf-8") as f:
            dsets = json.load(f)
    else:
        geoms = [geom_of(by[i], screened[i], ctx.tok, ds) for i in split]
        ds5 = (dataset_sets(ctx, geoms) if (RUN_NEURON_ARMS and split and len(split) == n_split)
               else {"chain": None, "span": None})
        dsets = {"split_ids": split, "ds5": ds5}
        write_json(setp, dsets)
    # ---- G: gates on the first gate items of the scored list, every session, before any scored row
    gate_ids = scored[:N_GATE[stage]]
    gates = [gates_item(ctx, by[i], screened[i]) for i in gate_ids]
    nonvac = neuron_nonvacuity(ctx, by[gate_ids[0]], screened[gate_ids[0]]) if (stage == "full" and gate_ids) else None
    gp = {"items": gates, "n": len(gates), "pass": all(g["pass"] for g in gates),
          "G2_identical_share": (sum(1 for g in gates if g["G2"]["identical"]) / len(gates)) if gates else None}
    for a, b in AR.IDENTITIES:
        gp["identity_%s_%s" % (a, b)] = {
            "kv_equal_share": (sum(1 for g in gates if g["kv_equal_%s_%s" % (a, b)]) / len(gates)) if gates else None,
            "token_identical_share": (sum(1 for g in gates if g["tok_identical_%s_%s" % (a, b)]["identical"])
                                      / len(gates)) if gates else None,
            "pass_tie_rule": all(g["tok_identical_%s_%s" % (a, b)]["pass"] for g in gates)}
    gp["neuron_nonvacuity"] = nonvac
    gp["norm_ratio_applied_pass_share"] = (sum(1 for g in gates if g["norm_ratio_applied_pass"]) / len(gates)
                                           if gates else None)
    write_json(os.path.join(out, "gates_%d.json" % int(time.time())), gp)
    write_json(os.path.join(out, "gates.json"), gp)
    if not gp["pass"]:
        die(EXIT_GATE, "gate failed (gates.json)")
    log("gates PASS on %d items; G2 identical share %s" % (len(gates), gp["G2_identical_share"]))
    # ---- R: arms per item (resumable)
    rp = os.path.join(out, "rows.jsonl")
    done = {int(r["idx"]) for r in read_jsonl(os.path.join(out, "items_done.jsonl"))}
    floor_alone = {g["idx"]: g["floor_alone"] for g in gates}
    for i in split + scored:
        if i in done:
            continue
        if STOP["flag"]:
            die(EXIT_STOPPED, "stopped after item %d" % i)
        rows, meta = run_item(ctx, by[i], screened[i], dsets["ds5"], i in split, use_cap)
        clear_item(ctx)
        fr = next(r for r in rows if r["arm"] == "floor")
        if i in floor_alone:                                     # G2b: the floor in the batch against alone
            npre = fr["n_prefix"]
            seqA = RD.prompt_ids(ctx.tok, ds, by[i]["question_A"]) + list(screened[i]["prefix_A"])
            meta["G2b"] = tie_check(ctx, seqA, floor_alone[i], fr["chain_ids"][npre:])
        for r in rows:
            r["idx"] = i
            append_jsonl(rp, r)
        append_jsonl(os.path.join(out, "items_done.jsonl"), meta)
    # ---- G6 (pilot): the text-edit ceiling
    if stage == "pilot":
        rows = read_jsonl(rp)
        txt = [r for r in rows if r["arm"] == "text" and r["idx"] in set(scored)]
        ca = sum(1 for r in txt if r["outcome"] == "B") / max(1, len(txt))
        write_json(os.path.join(out, "g6.json"), {"ca_text": ca, "n": len(txt), "pass": ca >= CEILING_MIN})
        if ca < CEILING_MIN:
            log("G6 FAIL: text-edit ceiling %.3f < %.2f on %d items" % (ca, CEILING_MIN, len(txt)))
            summarize(out)
            die(EXIT_GATE, "G6 failed")
    summarize(out)
    log("done")
    return 0


# ====================================================================== summary
def boot_mean(xs, n=NB, seed=0):
    if not xs:
        return None
    t = torch.tensor(xs, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(xs), (n, len(xs)), generator=g)
    bs = t[idx].mean(1)
    q = torch.quantile(bs, torch.tensor([0.05, 0.95], dtype=torch.float64))
    return [round(float(t.mean()), 4), round(float(q[0]), 4), round(float(q[1]), 4), len(xs)]


def boot_ratio(num, den, n=NB, seed=0):
    if not num or sum(den) == 0:
        return None
    a = torch.tensor(num, dtype=torch.float64)
    b = torch.tensor(den, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(num), (n, len(num)), generator=g)
    db = b[idx].sum(1)
    ok = db > 0
    bs = a[idx].sum(1)[ok] / db[ok]
    q = torch.quantile(bs, torch.tensor([0.05, 0.95], dtype=torch.float64))
    return [round(float(a.sum() / b.sum()), 4), round(float(q[0]), 4), round(float(q[1]), 4), len(num)]


def boot_diff(x, y, n=NB, seed=0):
    return boot_mean([a - b for a, b in zip(x, y)], n, seed)


def summarize(out):
    rows = read_jsonl(os.path.join(out, "rows.jsonl"))
    scr = read_jsonl(os.path.join(out, "screen.jsonl"))
    metas = read_jsonl(os.path.join(out, "items_done.jsonl"))
    split = set()
    if os.path.exists(os.path.join(out, "sets.json")):
        with open(os.path.join(out, "sets.json"), encoding="utf-8") as f:
            split = set(json.load(f)["split_ids"])
    by = {}
    for r in rows:
        by.setdefault(int(r["idx"]), {})[r["arm"]] = r
    items = sorted(by)
    scrd = {int(r["idx"]): r for r in scr}
    sub_A = [i for i in items if by[i].get("floor", {}).get("outcome") == "A"
             and by[i].get("clean", {}).get("outcome") == "A"]                 # review N4
    no_leak = [i for i in items if not scrd.get(i, {}).get("gold_B_in_prompt_or_prefix")]   # review N8
    carry_ok = [i for i in items if not scrd.get(i, {}).get("derived_B_in_prompt")]
    res = {"n_items": len(items), "arms": {}}
    for a in AR.ARM_NAMES:
        its = [i for i in items if a in by[i]]
        if not its:
            continue
        v = {"n": len(its)}
        for name, f in (("CA", lambda r: r["outcome"] == "B"), ("A", lambda r: r["outcome"] == "A"),
                        ("break", lambda r: r["outcome"] == "neither"), ("carry_B", lambda r: r["carry"] == "B"),
                        ("carry_A", lambda r: r["carry"] == "A"), ("horizon", lambda r: r["hit_horizon"]),
                        ("no_parse", lambda r: r["own"] is None)):
            v[name] = boot_mean([1.0 if f(by[i][a]) else 0.0 for i in its])
        v["CA_floorA_cleanA"] = boot_mean([1.0 if by[i][a]["outcome"] == "B" else 0.0 for i in its if i in sub_A])
        v["CA_no_goldB_coincidence"] = boot_mean([1.0 if by[i][a]["outcome"] == "B" else 0.0
                                                  for i in its if i in no_leak])
        v["carry_B_derivedB_not_in_prompt"] = boot_mean([1.0 if by[i][a]["carry"] == "B" else 0.0
                                                         for i in its if i in carry_ok])
        v["mean_chain"] = round(sum(by[i][a]["n_chain"] for i in its) / len(its), 1)
        res["arms"][a] = v

    def ca(a, its):
        return [1.0 if by[i][a]["outcome"] == "B" else 0.0 for i in its]
    both = lambda *a: [i for i in items if all(x in by[i] for x in a)]
    pr = {}
    it3 = both("cv1", "text")
    pr["P3_ratio_cv1_text"] = boot_ratio(ca("cv1", it3), ca("text", it3))
    pr["P3_ratio_kv1_text"] = boot_ratio(ca("kv1", both("kv1", "text")), ca("text", both("kv1", "text")))
    pr["side_by_side_CA"] = {a: res["arms"].get(a, {}).get("CA") for a in ("cv1", "kv1", "text")}
    it4 = both("cv1", "cv2", "cv3")
    pr["P4_cv1_minus_cv2"] = boot_diff(ca("cv1", it4), ca("cv2", it4))
    pr["P4_cv2_minus_cv3"] = boot_diff(ca("cv2", it4), ca("cv3", it4))
    pr["P5_base_CA"] = res["arms"].get("base", {}).get("CA")
    pr["identity_cvP_cv1_token_share"] = (sum(1 for i in both("cvP", "cv1")
                                              if by[i]["cvP"]["chain_ids"] == by[i]["cv1"]["chain_ids"])
                                          / max(1, len(both("cvP", "cv1"))))
    neur = {}
    for pre, full in (("nc_", "cv1"), ("ns_", "span1")):
        for s in AR.NEURON_SETS:
            a = pre + s
            its = both(a, full)
            if its:
                neur["share_" + a] = boot_ratio(ca(a, its), ca(full, its))
        its = both(pre + "t5", pre + "r5")
        if its:
            neur["gap_%st5_r5" % pre] = boot_diff(ca(pre + "t5", its), ca(pre + "r5", its))
    # review N2: the horizon cap rule
    caprule = {}
    lens = {i: scrd[i]["n_clean"] for i in items if i in scrd and scrd[i].get("n_clean") is not None}
    tot, late = 0, 0
    for i in items:
        if i not in lens:
            continue
        for a, r in by[i].items():
            if r["own"] is None or r["marker_tok"] is None:
                continue
            tot += 1
            cap = cap_new(r["n_prefix"], lens[i], True)
            if r["marker_tok"] - r["n_prefix"] > cap:
                late += 1
    caprule = {"rows_with_answer": tot, "rows_marker_after_cap": late,
               "share": (late / tot) if tot else None,
               "use_cap_in_full": bool(tot) and late / tot < 0.01,
               "rule": "cap = min(512 - prefix, 2 x clean continuation after the prefix + 64) if under 1% of "
                       "parsed rows pass it"}
    drops = {}
    for r in scr:
        k = r["drop"] or "eligible"
        drops[k] = drops.get(k, 0) + 1
    sec_items = [m["sec_item"] for m in metas]
    sec_clean = [r["sec_clean"] for r in scr if r.get("sec_clean") is not None]
    n_arms = len(AR.ARM_NAMES)
    elig_rate = drops.get("eligible", 0) / max(1, len(scr))
    proj = None
    ds = os.path.basename(out)
    try:
        cpath = os.path.join(out, "config.json")
        mdl = RD.DEFAULT_MODEL
        if os.path.exists(cpath):
            with open(cpath, encoding="utf-8") as f:
                mdl = json.load(f).get("model", RD.DEFAULT_MODEL)
        n_pool = (RD.load_pool(ds) if mdl == RD.DEFAULT_MODEL else RD.load_pool(ds, mdl))["n"]
        if sec_items and sec_clean:
            proj = round((n_pool * (sum(sec_clean) / len(sec_clean))
                          + n_pool * elig_rate * (sum(sec_items) / len(sec_items))) / 3600.0, 2)
    except (OSError, KeyError):
        n_pool = None
    checks = {"drops": drops, "eligible_rate": round(elig_rate, 4), "n_pool": n_pool,
              "projected_eligible": round(elig_rate * n_pool) if n_pool else None,
              "gate_80": (elig_rate * n_pool >= 80) if n_pool else None,
              "split_n": len(split), "sec_per_item_mean": round(sum(sec_items) / len(sec_items), 2) if sec_items else None,
              "sec_per_arm_item": round(sum(sec_items) / len(sec_items) / n_arms, 3) if sec_items else None,
              "sec_clean_mean": round(sum(sec_clean) / len(sec_clean), 2) if sec_clean else None,
              "projected_full_card_hours": proj, "horizon_cap_rule": caprule,
              "norm_ratio_prerounding": [m.get("norm_ratio_prerounding") for m in metas[:10]],
              "G2b": [m.get("G2b") for m in metas if m.get("G2b")],
              "n_lead_same_hist": _hist([scrd[i].get("n_lead_same") for i in items if i in scrd]),
              "r1_rule_kept_info": sum(1 for r in scr if r.get("r1_kept_info"))}
    res["predictions"] = pr
    res["neuron"] = neur
    res["checks"] = checks
    write_json(os.path.join(out, "results.json"), res)
    write_json(os.path.join(out, "checks.json"), checks)
    import csv
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "arm", "outcome", "carry", "own", "n_chain", "hit_horizon", "div_from_ref", "in_split"])
        for i in items:
            for a in AR.ARM_NAMES:
                if a in by[i]:
                    r = by[i][a]
                    w.writerow([i, a, r["outcome"], r["carry"], r["own"], r["n_chain"], r["hit_horizon"],
                                r.get("div_from_ref"), i in split])
    ex = []
    ds_ = os.path.basename(out)
    keep = [i for i in EXAMPLE_IDX.get(ds_, ()) if i in by]
    for i in keep + [i for i in items if i not in keep][:(5 if "pilot" in out else 20)]:
        ex.append({"idx": i, "derived_A": scrd.get(i, {}).get("derived_A"), "derived_B": scrd.get(i, {}).get("derived_B"),
                   "chains": {a: by[i][a]["text"] for a in ("clean", "text", "cv1", "kv1") if a in by[i]}})
    write_json(os.path.join(out, "examples.json"), ex)
    lines = ["# RA-30 %s" % out.replace(os.sep, "/").split("/out/")[-1], "",
             "Items %d (split %d). Pool: Table 2 single-card a1 scored list. Neuron arms carry only the last "
             "layer's MLP part of the loop-1 state difference." % (len(items), len(split)), "",
             "| arm | n | CA | 90% CI | A | break | carry B | CA (floor and clean end in A) |", "|---|---|---|---|---|---|---|---|"]
    for a in AR.ARM_NAMES:
        v = res["arms"].get(a)
        if not v:
            continue
        c = v["CA"]
        lines.append("| %s | %d | %.3f | [%.3f, %.3f] | %.3f | %.3f | %.3f | %s |" % (
            a, v["n"], c[0], c[1], c[2], v["A"][0], v["break"][0], v["carry_B"][0],
            "%.3f" % v["CA_floorA_cleanA"][0] if v["CA_floorA_cleanA"] else "-"))
    lines += ["", "CA side by side (cv1, kv1, text): %s" % json.dumps(pr["side_by_side_CA"]),
              "P3 CA(cv1)/CA(text): %s; CA(kv1)/CA(text): %s" % (pr["P3_ratio_cv1_text"], pr["P3_ratio_kv1_text"]),
              "P4 cv1-cv2: %s; cv2-cv3: %s" % (pr["P4_cv1_minus_cv2"], pr["P4_cv2_minus_cv3"]),
              "P5 base CA: %s" % (pr["P5_base_CA"],),
              "Identity cvP = cv1 token-identical share: %.3f" % pr["identity_cvP_cv1_token_share"],
              "Neuron: %s" % json.dumps(neur)]
    gpath = os.path.join(out, "gates.json")
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as f:
            nv = json.load(f).get("neuron_nonvacuity")
        if nv:
            lines.append("Neuron hook check (neuron arms not run in the mains; their pilot CA 0.000 is not a result): "
                         + "; ".join("%s: %s (|delta a| %s, logits equal %s, max |dz| %s)"
                                     % (a, v["verdict"], v["delta_a_norm"], v["logits_equal"],
                                        v["logits_max_abs_diff"]) for a, v in sorted(nv.items())))
    lines += ["", "Checks: %s" % json.dumps(checks, sort_keys=True)]
    if ds == "asdiv":
        lines += ["", ASDIV_NOTE + " n = %d items; every interval is printed with its n." % len(items)]
        checks["note_asdiv"] = ASDIV_NOTE
        write_json(os.path.join(out, "checks.json"), checks)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return res


def _hist(xs):
    h = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return h


def host_tokens(ds, out, tok=None):
    """Host tokenizer step (CPU, before the pilot; RA-29 lesson): every token id the run uses is built on the host
    tokenizer at run time. This step rebuilds prompt A and prompt B of every pool item on the host tokenizer and
    writes out/tokens/<ds>.jsonl (ids) and <ds>_check.json (plen and span agreement with the pool, the tokenizer
    and transformers versions). More than 1 percent mismatches stops with exit 2."""
    import transformers
    if tok is None:
        if transformers.__version__ != TRANSFORMERS_PIN:
            die(EXIT_PLAN, "transformers %s, pinned %s" % (transformers.__version__, TRANSFORMERS_PIN))
        from prod import steer_gen as S
        tok = S.load_tokenizer(MODEL)
    d = os.path.join(model_root(out), "tokens")
    os.makedirs(d, exist_ok=True)
    pool = pool_of(ds)
    bad, rows = [], []
    for it in pool["items"]:
        pA = RD.prompt_ids(tok, ds, it["question_A"])
        pB = RD.prompt_ids(tok, ds, it["question_B"])
        s0, s1 = int(it["span"][0]), int(it["span"][1])
        ok = (len(pA) == len(pB) == int(it["plen"]) and pA[:s0] == pB[:s0] and pA[s1:] == pB[s1:]
              and pA[s0:s1] != pB[s0:s1])
        rows.append({"idx": int(it["idx"]), "prompt_A": pA, "prompt_B": pB, "ok": ok})
        if not ok:
            bad.append(int(it["idx"]))
    with open(os.path.join(d, "%s.jsonl" % ds), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    chk = {"dataset": ds, "n": len(rows), "n_mismatch": len(bad), "mismatch_idx": bad,
           "transformers": transformers.__version__, "tokenizer_class": type(tok).__name__,
           "pass": len(bad) <= 0.01 * len(rows)}
    write_json(os.path.join(d, "%s_check.json" % ds), chk)
    log("host tokens %s: %d items, %d mismatches (dropped as plen_mismatch in the screen)" % (ds, len(rows), len(bad)))
    if not chk["pass"]:
        die(EXIT_PLAN, "more than 1 percent of %s prompts do not rebuild on the host tokenizer" % ds)
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--ds", choices=RD.DATASETS, default="gsm8k")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--plan_sha256", default="")
    ap.add_argument("--no_plan_check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", choices=RD.MODELS, default=RD.DEFAULT_MODEL)
    a = ap.parse_args(argv)
    set_model(a.model)
    if a.ds not in N_SCORED_PILOT_MODEL.get(MODEL, N_SCORED_PILOT):
        die(EXIT_PLAN, "the %s axis has no dataset %s" % (MODEL, a.ds))
    if a.hash:
        print(json.dumps({"plan_sha256": RD.sha256_file(PLAN), "code_sha256": code_hashes(),
                          "data_sha256": check_data()["files"]}, indent=1, sort_keys=True))
        return 0
    if a.tokens:
        host_tokens(a.ds, a.out)
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
