"""RA-30 on the Ouro Thinking models (plan_ra30.md section 6e): Ouro-1.4B-Thinking and Ouro-2.6B-Thinking, GSM8K and
ASDiv, with gold-intermediate matching (lib/gold.py) instead of equation parsing.

Same decode path and arms as the Ouro base runs (SteeredOuroAdapter with kv=True, lib/arms.py rows, the 12 arms
without the neuron arms), the steer_gen Thinking setting (chat template with the exemplars in the user turn, the
think tag not a stop, the task's stop strings, horizon 4096), and the Table 2 parser on the natural cut
(parse_own reading after the think tag). ra30.py, lib/chain.py, lib/arms.py and lib/ra30_data.py are unchanged.

Pools are the Table 2 Thinking scored lists on the cluster, read at run time from --pool_root ($OUT_ROOT):
A/steer_gen/a3_gsm8k, a4_gsm8k and A/steer_gen_math/a3_asdiv, a4_asdiv (items.json). The pilot records the sha256
of the file it read; the full shards stop with exit 2 when the file is missing or differs. --pool_local (smoke only)
reads data/pool_<ds>.json and computes the chat geometry itself.

    python ra30_think.py --tokens --model M --ds D --pool_root ROOT --out OUT                 (CPU)
    python ra30_think.py --run --model M --ds D --stage pilot|full [--shard S --shards N] [--phase screen|rows]
        --pool_root ROOT --out OUT --plan_sha256 SHA
    python ra30_think.py --summarize --out OUT/<model>/<stage>/<ds>
"""
import argparse
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
import gold as GD  # noqa: E402
import ra30 as R0  # noqa: E402
import ra30_data as RD  # noqa: E402

MODELS = {"ouro_1_4b_think": {"setting": "a3", "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"},
          "ouro_2_6b_think": {"setting": "a4", "revision": "f1edd81e7ac41355db670500ceaf204e0f73af68"}}
POOL_REL = {("ouro_1_4b_think", "gsm8k"): "A/steer_gen/a3_gsm8k/items.json",
            ("ouro_2_6b_think", "gsm8k"): "A/steer_gen/a4_gsm8k/items.json",
            ("ouro_1_4b_think", "asdiv"): "A/steer_gen_math/a3_asdiv/items.json",
            ("ouro_2_6b_think", "asdiv"): "A/steer_gen_math/a4_asdiv/items.json"}
K = 4
HORIZON = 4096                       # steer_gen a3 / a4
N_SCORED_PILOT = {"gsm8k": 30, "asdiv": 20}      # RA: GSM8K pilots scan until 30 eligible (or the pool ends)
N_GATE = {"pilot": 10, "full": 3}
MIN_ELIGIBLE_FULL = 20
ACTIVE_ARMS = list(R0.ACTIVE_ARMS)   # the 12 rows of the Ouro mains (no neuron arm)
ACCEPTED_PILOT_PLANS = R0.ACCEPTED_PILOT_PLANS


class Ctx(object):
    def __init__(self, ad, steer, tok, ds, model):
        from prod.tasks import task_cfg
        self.ad, self.steer, self.tok, self.ds, self.model_name = ad, steer, tok, ds, model
        self.model = ad.model
        self.hook = None
        self.task = RD.BASE_TASK[ds]
        self.stops = list(task_cfg(self.task)["stops"])
        self.eos = [int(tok.eos_token_id)]


def load_ctx(model, ds, think_dir, mem_fraction=None):
    import transformers
    if transformers.__version__ != R0.TRANSFORMERS_PIN:
        R0.die(R0.EXIT_PLAN, "transformers %s, pinned %s" % (transformers.__version__, R0.TRANSFORMERS_PIN))
    if not think_dir or not os.path.isdir(think_dir):
        R0.die(R0.EXIT_PLAN, "the Thinking checkpoint folder is missing (--think_dir or K17_THINK_DIR)")
    from prod import steer_gen as S
    steer = S.SteerHooks(K)
    kw = {} if mem_fraction is None else {"mem_fraction": float(mem_fraction)}
    ad = S.SteeredOuroAdapter(model, steer=steer, kv=True, think_dir=think_dir, **kw)
    ad.load()
    ad.set_depth(K)
    if next(ad.model.parameters()).dtype != torch.bfloat16:
        R0.die(R0.EXIT_PLAN, "model dtype is not bf16")
    return Ctx(ad, steer, ad.tok, ds, model)


def clear_item(ctx):
    for x in (ctx.steer.rows, ctx.steer.donors, ctx.steer.stats):
        x.clear()
    ctx.steer.prompt_counted.clear()


# ====================================================================== pool and geometry
def pairs_by(ds):
    return {(int(r["idx"]), r["var_name"]): r for r in RD.read_jsonl(RD.pairs_path(ds))}


def load_pool(model, ds, pool_root=None, pool_local=False):
    """Items with question texts and the pair fields; the source path and its sha256."""
    if pool_local:
        pool = RD.load_pool(ds)
        items, path, chat_geom = pool["items"], os.path.join(RD.DATA, "pool_%s.json" % ds), False
    else:
        if not pool_root:
            R0.die(R0.EXIT_PLAN, "--pool_root is required (the cluster's OUT_ROOT)")
        path = os.path.join(pool_root, POOL_REL[(model, ds)])
        if not os.path.exists(path):
            R0.die(R0.EXIT_PLAN, "the Thinking pool %s is missing" % POOL_REL[(model, ds)])
        with open(path, encoding="utf-8") as f:
            items = json.load(f)["scored"]
        chat_geom = True
    pr = pairs_by(ds)
    out = []
    for it in items:
        p = pr.get((int(it["idx"]), it["var_name"]))
        if p is None:
            R0.die(R0.EXIT_PLAN, "pool item %s has no pair" % it["idx"])
        d = dict(it, question_A=p["question_A"], question_B=p["question_B"], char_span_A=p["char_span_A"],
                 char_span_B=p["char_span_B"])
        if ds == "asdiv":
            d.update(equation_A=p["equation_A"], equation_B=p["equation_B"])
        d["pool_has_chat_geometry"] = chat_geom
        out.append(d)
    rel = POOL_REL.get((model, ds)) if not pool_local else "data/pool_%s.json" % ds
    return out, {"path": rel, "sha256": RD.sha256_file(path), "n": len(out), "local": bool(pool_local)}


def chat_prompts(tok, ds, it):
    from prod.tasks import build_prompts, task_cfg
    c = task_cfg(RD.BASE_TASK[ds])
    rows = [{"input": it["question_A"], "target": "", "options": [], "gold_text": "", "idx": 0},
            {"input": it["question_B"], "target": "", "options": [], "gold_text": "", "idx": 0}]
    pa, pb = build_prompts(tok, RD.BASE_TASK[ds], rows, chat_template=True)[0]
    return pa, pb, c["q_prefix"]


def geometry(tok, ds, it):
    """Prompt ids of A and B under the chat template and the token span of the changed number (character offsets
    through the tokenizer's offset mapping), with the checks of steer_gen.item_geometry."""
    pa, pb, qp = chat_prompts(tok, ds, it)
    out = {}
    for side, p, q, cs in (("A", pa, it["question_A"], it["char_span_A"]), ("B", pb, it["question_B"], it["char_span_B"])):
        o = p.rfind(qp + q)
        if o < 0:
            return None
        o += len(qp)
        enc = tok(p, add_special_tokens=False, return_offsets_mapping=True)
        sel = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > o + cs[0] and a < o + cs[1] and b > a]
        out[side] = (enc["input_ids"], (sel[0], sel[-1] + 1) if sel else None)
    (ia, sa), (ib, sb) = out["A"], out["B"]
    if sa is None or sa != sb or len(ia) != len(ib) or ia[:sa[0]] != ib[:sa[0]] or ia[sa[1]:] != ib[sa[1]:] \
            or ia[sa[0]:sa[1]] == ib[sa[0]:sa[1]]:
        return None
    if it.get("pool_has_chat_geometry") and (list(sa) != list(it["span"]) or len(ia) != int(it["plen"])):
        return None
    return {"prompt_A": ia, "prompt_B": ib, "span": list(sa)}


def cut_parse(ctx, chain_ids):
    from prod.models.ouro import strip_tail
    from prod.tasks import make_find_cut, parse_own
    ids = strip_tail([list(chain_ids)], ctx.eos)[0]
    c, _mk = make_find_cut(ctx.tok, ctx.stops, ctx.eos, chat_template=True, eos_cut=True,
                           think_tag_is_stop=False)(ids)
    txt = ctx.tok.decode(ids[:c], clean_up_tokenization_spaces=False)
    return ids, c, txt, parse_own(txt, ctx.task, think_tag_is_stop=False)


_GOLD = {}


def gold_pairs(ds, it):
    if ds == "gsm8k":
        if "g" not in _GOLD:
            with open(os.path.join(RD.DATA, "gold_gsm8k.json"), encoding="utf-8") as f:
                _GOLD["g"] = json.load(f)["answers"]
        return GD.gsm8k_pairs(_GOLD["g"].get(str(int(it["idx"]))), it["value_A"], it["value_B"], it["gold_A"],
                              it["gold_B"])
    return GD.asdiv_pairs(it["equation_A"], it["equation_B"], it["gold_A"], it["gold_B"])


def clean_decode(ctx, prompt):
    gen, _t = ctx.ad.decode([list(prompt)], HORIZON, K, ctx.eos, stop_strings=ctx.stops, row_ids=None)
    return list(gen[0])


def screen_item(ctx, it, clean_fn=None):
    rec = {"idx": int(it["idx"])}
    g = geometry(ctx.tok, ctx.ds, it)
    if g is None:
        rec.update(drop="plen_mismatch", clean_ids=None)
        return rec
    pA = g["prompt_A"]
    rec.update(plen=len(pA), span=g["span"])
    t0 = time.time()
    raw = (clean_fn or clean_decode)(ctx, pA)
    rec["sec_clean"] = round(time.time() - t0, 3)
    ids, cut, txt, own = cut_parse(ctx, raw)
    rec.update(clean_ids=ids, cut=cut, own=own, n_clean=len(ids))
    if CH.outcome(own, it["gold_A"], it["gold_B"]) != "A":
        rec["drop"] = "clean_wrong"
        return rec
    pairs, why = gold_pairs(ctx.ds, it)
    if why:
        rec["drop"] = why
        return rec
    cands = GD.candidates(pairs, it["question_A"], it["gold_A"])
    m = GD.match(txt, cands)
    if m["drop"] is None:
        m = CH.token_geometry(ctx.tok, pA, ids, m)
    rec["drop"] = m["drop"]
    rec["n_candidates"] = len(cands)
    if m["drop"] is None:
        rec.update({k: m[k] for k in ("derived_A", "derived_B", "P", "prefix_A", "prefix_B", "prefix_len",
                                      "tokens_A", "tokens_B", "canonical_B", "value_A", "value_B")})
        rec["derived_B_in_prompt"] = any(CH.feq(m["value_B"], q) for q in CH.prompt_numbers(it["question_A"])
                                         if q is not None)
    return rec


def geom_of(ctx, it, sc):
    g = geometry(ctx.tok, ctx.ds, it)
    return dict(g, prefix_A=sc["prefix_A"], prefix_B=sc["prefix_B"], P=sc["P"], clean_ids=sc["clean_ids"])


# ====================================================================== rows
def cap_new(n_pre, n_clean, use_cap):
    full = HORIZON - n_pre
    return min(full, 2 * (int(n_clean) - n_pre) + 64) if use_cap else full


def rewrite_back(cont, derived_A):
    """Plan 6e: after P the chain writes value_A again (the original value of the edited step) before its answer
    marker. Read on the continuation after the prefix."""
    a = CH.numval(derived_A)
    return any(CH.feq(CH.numval(m.group(0)), a) for m in CH.NUM_RE.finditer(CH.body(cont or "")))


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
            "carry": GD.carry_prose(cont, sc["derived_A"], sc["derived_B"]),
            "rewrite_back": rewrite_back(cont, sc["derived_A"]), "n_chain": len(ids),
            "hit_horizon": len(ids) >= HORIZON, "marker_tok": marker_tok, "n_prefix": n_prefix}


def run_item(ctx, it, sc, use_cap=False, max_batch=None):
    """All arms of one item; max_batch (smoke on a small card only) splits a batch into sub-batches."""
    g = geom_of(ctx, it, sc)
    t0 = time.time()
    keys, _sets = R0.build_rows(ctx, it, g, None, False, arms=ACTIVE_ARMS)
    out = []
    plen = len(g["prompt_A"])
    for batch, n_pre in (("prefix", len(g["prefix_A"])), ("scratch", 0)):
        names = [a for a in ACTIVE_ARMS if AR.ARM[a]["batch"] == batch]
        bk = [keys[a] for a in names]
        seqs = [ctx.steer.rows[k]["a_ids"] for k in bk]
        if len({len(s) for s in seqs}) != 1:
            raise RuntimeError("rows of one batch differ in length")
        tb = time.time()
        step = int(max_batch or len(bk))
        gen = []
        for b0 in range(0, len(bk), step):
            g_, _t = ctx.ad.decode(seqs[b0:b0 + step], cap_new(n_pre, sc["n_clean"], use_cap), K, ctx.eos,
                                   stop_strings=ctx.stops, row_ids=bk[b0:b0 + step])
            gen += list(g_)
        sec = time.time() - tb
        for a, k, gg in zip(names, bk, gen):
            r = readout(ctx, it, sc, a, list(ctx.steer.rows[k]["a_ids"][plen:]) + list(gg), n_pre)
            r["sec_batch"] = round(sec, 3)
            r["batch_rows"] = len(names)
            out.append(r)
    ref = {r["arm"]: r["chain_ids"] for r in out}
    for r in out:
        r["div_from_ref"] = R0.first_div(r["chain_ids"], ref["floor" if AR.ARM[r["arm"]]["batch"] == "prefix" else "clean"])
    return out, {"idx": int(it["idx"]), "sec_item": round(time.time() - t0, 3)}


@torch.no_grad()
def gates_item(ctx, it, sc):
    """G1 idle row against no row keys (prefill logits and K/V), G2 the floor alone against the clean chain (tie
    rule), G3 the teacher-forced prefix, G4 positions, G5 post@1..3 and kv1 land on a fresh donor record."""
    from prod.models.ouro import strip_tail
    tok, ds, idx = ctx.tok, ctx.ds, int(it["idx"])
    g = geom_of(ctx, it, sc)
    item_key = "%s:%d" % (ds, idx)
    seqA, seqB = g["prompt_A"] + g["prefix_A"], g["prompt_A"] + g["prefix_B"]
    lo, hi = g["P"]
    plen, npre = len(g["prompt_A"]), len(g["prefix_A"])
    res = {"idx": idx}
    names = ["floor", "cv1", "cv2", "cv3", "kv1"]
    KK = {a: AR.key_of(ds, idx, a) for a in names}
    for a in names:
        ctx.steer.rows[KK[a]] = AR.row_info(a, g, item_key)
    L = int(ctx.model.config.num_hidden_layers)
    z0, c0 = R0.prefill(ctx, seqA, None)
    z1, c1 = R0.prefill(ctx, seqA, [KK["floor"]])
    res["G1"] = bool(torch.equal(z0, z1)) and all(torch.equal(c0.key_cache[i][:, :, :len(seqA)],
                                                               c1.key_cache[i][:, :, :len(seqA)]) for i in range(L * K))
    flo = ctx.ad.decode([seqA], HORIZON - npre, K, ctx.eos, stop_strings=ctx.stops, row_ids=[KK["floor"]])[0][0]
    flo = strip_tail([list(flo)], ctx.eos)[0]
    res["G2"] = R0.tie_check(ctx, seqA, list(sc["clean_ids"])[npre:], flo)
    res["floor_alone_len"] = len(flo)
    dev = next(ctx.model.parameters()).device
    z = ctx.model(input_ids=torch.tensor([seqA], device=dev), use_cache=False,
                  logits_to_keep=npre + 1).logits[0, :-1].float()
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
    fresh = ctx.ad.record_donor(KK["kv1"])
    for a in ("cv1", "cv2", "cv3", "kv1"):
        ctx.steer.donors[KK[a]] = fresh
    ok = True
    for j in (1, 2, 3):
        rec = R0.new_rec(lo, hi, {"post"})
        R0.prefill(ctx, seqA, [KK["cv%d" % j]], rec=rec)
        ok &= bool(torch.equal(rec["post"][j][0], fresh["post"][j]))
    rk = R0.new_rec(lo, hi, {"kv"})
    R0.prefill(ctx, seqA, [KK["kv1"]], rec=rk)
    for li in range(L):
        for w in ("k", "v"):
            ok &= bool(torch.equal(rk["kv"][(0, li, w)][0], fresh["kv"][(0, li, w)]))
    res["G5_patch_lands"] = bool(ok)
    res["pass"] = bool(res["G1"] and res["G2"]["pass"] and res["G3"]["pass"] and res["G4"]["equal_len"]
                       and res["G4"]["P_ends_prefix"] and res["G5_patch_lands"])
    clear_item(ctx)
    return res


# ====================================================================== run
def stage_dir(out_root, model, stage, ds, shard=0, shards=1):
    d = os.path.join(out_root, model, stage, ds)
    return d if int(shards) <= 1 else os.path.join(d, "shard_%d_of_%d" % (int(shard), int(shards)))


def check_data():
    with open(os.path.join(RD.DATA, "sources_think.json"), encoding="utf-8") as f:
        src = json.load(f)
    for fn, h in src["files"].items():
        if RD.sha256_file(os.path.join(RD.DATA, fn)) != h:
            R0.die(R0.EXIT_PLAN, "data file %s differs from sources_think.json" % fn)
    for ds, p in src["pairs"].items():
        if RD.sha256_file(RD.pairs_path(ds)) != p["sha256"]:
            R0.die(R0.EXIT_PLAN, "pairs file of %s differs from sources_think.json" % ds)
    return src


def code_hashes():
    fs = ["ra30_think.py", "ra30.py", "lib/gold.py", "lib/chain.py", "lib/arms.py", "lib/ra30_data.py",
          "tree/prod/steer_gen.py", "tree/prod/models/ouro.py", "tree/prod/tasks/__init__.py"]
    return {f: RD.sha256_file(os.path.join(HERE, f)) for f in fs}


def pilot_of_record(out_root, model, ds, plan_sha, pool_rec):
    """The full shards read the pilot: plan accepted, its ceiling gate G6 passed (else exit 3), the same pool file
    (sha256 pin, else exit 2), at least MIN_ELIGIBLE_FULL projected eligible items, the cap decision."""
    d = stage_dir(out_root, model, "pilot", ds)
    files = {n: os.path.join(d, n) for n in ("hash_line.json", "checks.json", "g6.json", "pool_record.json")}
    if not all(os.path.exists(p) for p in files.values()):
        R0.die(R0.EXIT_PLAN, "no finished Thinking pilot in %s" % d)
    J = {}
    for n, p in files.items():
        with open(p, encoding="utf-8") as f:
            J[n] = json.load(f)
    if J["hash_line.json"].get("plan_sha256") not in ACCEPTED_PILOT_PLANS + ((plan_sha,) if plan_sha else ()):
        R0.die(R0.EXIT_PLAN, "the pilot ran under plan %s, which is not accepted" % J["hash_line.json"].get("plan_sha256"))
    if not J["g6.json"].get("pass"):
        R0.die(R0.EXIT_GATE, "the pilot's text-edit ceiling %.3f is under %.2f" % (J["g6.json"]["ca_text"],
                                                                                     R0.CEILING_MIN))
    if J["pool_record.json"]["sha256"] != pool_rec["sha256"]:
        R0.die(R0.EXIT_PLAN, "the pool file differs from the one the pilot read (sha256 pin)")
    n = J["checks.json"].get("projected_eligible")
    if n is None or n < MIN_ELIGIBLE_FULL:
        R0.die(R0.EXIT_GATE, "%s under %d projected eligible: gate failure" % (ds, MIN_ELIGIBLE_FULL))
    return {"use_cap": bool((J["checks.json"].get("horizon_cap_rule") or {}).get("use_cap_in_full")),
            "plan_sha256": J["hash_line.json"].get("plan_sha256")}


def run(args, ctx=None, clean_fn=None, items=None, pool_rec=None):
    global HORIZON
    HORIZON = int(args.horizon)
    R0.HORIZON = HORIZON                 # ra30.cap_new and summarize read the horizon of this process
    model, ds, stage = args.model, args.ds, args.stage
    shard, shards = int(args.shard), int(args.shards)
    if stage == "pilot" and shards != 1:
        R0.die(R0.EXIT_PLAN, "the pilot is not sharded")
    if not 0 <= shard < shards:
        R0.die(R0.EXIT_PLAN, "shard %d of %d" % (shard, shards))
    out = stage_dir(args.out, model, stage, ds, shard, shards)
    os.makedirs(out, exist_ok=True)
    plan_sha = R0.check_plan(args.plan_sha256) if not args.no_plan_check else None
    src = check_data() if not args.no_plan_check else {"files": {}}
    if items is None:
        items, pool_rec = load_pool(model, ds, args.pool_root, args.pool_local)
    by = {int(it["idx"]): it for it in items}
    order = RD.draw_order(list(by), ds, model)[shard::shards]
    n_target = N_SCORED_PILOT[ds] if stage == "pilot" else None
    part, parts = int(getattr(args, "rows_part", 0)), int(getattr(args, "rows_parts", 1))
    if not 0 <= part < parts:
        R0.die(R0.EXIT_PLAN, "rows part %d of %d" % (part, parts))
    use_cap, pilot_plan = False, None
    if stage == "full":
        p = pilot_of_record(args.out, model, ds, plan_sha, pool_rec)
        use_cap, pilot_plan = p["use_cap"], p["plan_sha256"]
    cfg = {"ds": ds, "stage": stage, "model": model, "k": K, "horizon": HORIZON, "use_cap": use_cap,
           "pilot_plan_sha256": pilot_plan, "n_scored_pilot": n_target, "arms": ACTIVE_ARMS, "tie": R0.TIE,
           "shard": shard, "shards": shards, "pool": pool_rec, "matching": "gold intermediates (lib/gold.py)",
           "min_eligible_full": MIN_ELIGIBLE_FULL}
    hl = os.path.join(out, "hash_line.json")
    hline = {"plan_sha256": plan_sha, "config_sha256": R0.sha256_text(json.dumps(cfg, sort_keys=True)),
             "code_sha256": code_hashes(), "data_sha256": src["files"], "pool_sha256": pool_rec["sha256"]}
    if os.path.exists(hl):
        with open(hl, encoding="utf-8") as f:
            old = json.load(f)
        if {k: old.get(k) for k in hline} != hline:
            R0.die(R0.EXIT_PLAN, "inputs changed since the first row of %s (hash_line.json)" % out)
    else:
        R0.write_json(hl, hline)
        R0.write_json(os.path.join(out, "config.json"), cfg)
        R0.write_json(os.path.join(out, "pool_record.json"), pool_rec)
    signal.signal(signal.SIGTERM, R0.on_term)
    if args.phase == "merge":
        return merge_parts(out, stage)
    if ctx is None:
        ctx = load_ctx(model, ds, args.think_dir or os.environ.get("K17_THINK_DIR"))
    sp = os.path.join(out, "screen.jsonl")
    screened = {int(r["idx"]): r for r in R0.read_jsonl(sp)}
    elig = [i for i in order if i in screened and screened[i]["drop"] is None]
    if args.phase in ("all", "screen"):
        for i in order:
            if n_target is not None and len(elig) >= n_target:
                break
            if i in screened:
                continue
            if R0.STOP["flag"]:
                R0.die(R0.EXIT_STOPPED, "stopped during the screen")
            r = screen_item(ctx, by[i], clean_fn)
            R0.append_jsonl(sp, r)
            screened[i] = r
            if r["drop"] is None:
                elig.append(i)
        R0.write_json(os.path.join(out, "screen_done.json"), {"n_screened": len(screened), "n_eligible": len(elig)})
        R0.log("%s %s %s: scanned %d, eligible %d" % (model, stage, ds, len(screened), len(elig)))
        if args.phase == "screen":
            summarize(out)
            return 0
    elif not os.path.exists(os.path.join(out, "screen_done.json")):
        R0.die(R0.EXIT_PLAN, "phase rows needs the finished screen of %s" % out)
    if n_target is not None:
        elig = elig[:n_target]
    # RA: rows can run as `parts` parallel jobs over disjoint eligible items (elig[part::parts]), each with its
    # share of the gate items, written to rows_part_<p>_of_<n>; phase merge joins them and computes G6
    rout = out if parts <= 1 else os.path.join(out, "rows_part_%d_of_%d" % (part, parts))
    os.makedirs(rout, exist_ok=True)
    elig = elig[part::parts]
    n_gate = int(math.ceil(N_GATE[stage] / float(parts)))
    gates = [gates_item(ctx, by[i], screened[i]) for i in elig[:n_gate]]
    gp = {"items": gates, "n": len(gates), "pass": all(g["pass"] for g in gates),
          "G2_identical_share": (sum(1 for g in gates if g["G2"]["identical"]) / len(gates)) if gates else None}
    R0.write_json(os.path.join(rout, "gates.json"), gp)
    if not gp["pass"]:
        R0.die(R0.EXIT_GATE, "gate failed (gates.json)")
    rp = os.path.join(rout, "rows.jsonl")
    done = {int(r["idx"]) for r in R0.read_jsonl(os.path.join(rout, "items_done.jsonl"))}
    for i in elig:
        if i in done:
            continue
        if R0.STOP["flag"]:
            R0.die(R0.EXIT_STOPPED, "stopped after item %d" % i)
        rows, meta = run_item(ctx, by[i], screened[i], use_cap)
        clear_item(ctx)
        for r in rows:
            r["idx"] = i
            R0.append_jsonl(rp, r)
        R0.append_jsonl(os.path.join(rout, "items_done.jsonl"), meta)
    if parts > 1:
        R0.write_json(os.path.join(rout, "part_done.json"), {"items": elig})
        R0.log("rows part %d of %d done (%d items); run phase merge after every part" % (part, parts, len(elig)))
        return 0
    if stage == "pilot":
        txt = [r for r in R0.read_jsonl(rp) if r["arm"] == "text"]
        ca = sum(1 for r in txt if r["outcome"] == "B") / max(1, len(txt))
        R0.write_json(os.path.join(out, "g6.json"), {"ca_text": ca, "n": len(txt), "pass": ca >= R0.CEILING_MIN})
        summarize(out)
        if ca < R0.CEILING_MIN:
            R0.die(R0.EXIT_GATE, "G6 failed: text-edit ceiling %.3f < %.2f" % (ca, R0.CEILING_MIN))
    else:
        summarize(out)
    R0.log("done")
    return 0


def merge_parts(out, stage):
    """Phase merge (CPU): the rows parts of a pilot joined into the pilot folder (rows, items_done, gates), every
    part must be done; then G6 on the merged rows and the summary."""
    parts = sorted(d for d in os.listdir(out) if d.startswith("rows_part_"))
    if not parts:
        R0.die(R0.EXIT_PLAN, "no rows parts in %s" % out)
    n_exp = int(parts[0].rsplit("_of_", 1)[1])
    if len(parts) != n_exp or not all(os.path.exists(os.path.join(out, d, "part_done.json")) for d in parts):
        R0.die(R0.EXIT_PLAN, "rows parts missing or unfinished: %s" % parts)
    gates, seen = [], set()
    for name in ("rows.jsonl", "items_done.jsonl"):
        lines = []
        for d in parts:
            for r in R0.read_jsonl(os.path.join(out, d, name)):
                key = (int(r["idx"]), r.get("arm"))
                if key in seen:
                    R0.die(R0.EXIT_PLAN, "item %s twice across rows parts" % r["idx"])
                seen.add(key)
                lines.append(r)
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
    for d in parts:
        with open(os.path.join(out, d, "gates.json"), encoding="utf-8") as f:
            gates += json.load(f)["items"]
    gp = {"items": gates, "n": len(gates), "pass": all(g["pass"] for g in gates), "parts": parts,
          "G2_identical_share": (sum(1 for g in gates if g["G2"]["identical"]) / len(gates)) if gates else None}
    R0.write_json(os.path.join(out, "gates.json"), gp)
    if stage == "pilot":
        txt = [r for r in R0.read_jsonl(os.path.join(out, "rows.jsonl")) if r["arm"] == "text"]
        ca = sum(1 for r in txt if r["outcome"] == "B") / max(1, len(txt))
        R0.write_json(os.path.join(out, "g6.json"), {"ca_text": ca, "n": len(txt), "pass": ca >= R0.CEILING_MIN})
        summarize(out)
        if ca < R0.CEILING_MIN:
            R0.die(R0.EXIT_GATE, "G6 failed: text-edit ceiling %.3f < %.2f" % (ca, R0.CEILING_MIN))
    else:
        summarize(out)
    R0.log("merged %d rows parts; done" % len(parts))
    return 0


def summarize(out):
    """Merges shard folders (if any) into the folder, then ra30.summarize; adds the eligible projection from the
    pool record (ra30.summarize has no local pool file for these models) and the shard rule."""
    shards = sorted(d for d in (os.listdir(out) if os.path.isdir(out) else []) if d.startswith("shard_"))
    if shards:
        for name in ("rows.jsonl", "screen.jsonl", "items_done.jsonl"):
            seen, lines = set(), []
            for d in shards:
                for r in R0.read_jsonl(os.path.join(out, d, name)):
                    key = (int(r["idx"]), r.get("arm"))
                    if key in seen:
                        R0.die(R0.EXIT_PLAN, "item %s twice across shards (%s)" % (r["idx"], name))
                    seen.add(key)
                    lines.append(r)
            with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                for r in lines:
                    f.write(json.dumps(r) + "\n")
        for fn in ("config.json", "pool_record.json"):
            src = os.path.join(out, shards[0], fn)
            if os.path.exists(src) and not os.path.exists(os.path.join(out, fn)):
                with open(src, encoding="utf-8") as f:
                    R0.write_json(os.path.join(out, fn), json.load(f))
    if not os.path.exists(os.path.join(out, "config.json")):
        return None
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    R0.HORIZON = int(cfg.get("horizon", HORIZON))
    res = R0.summarize(out)
    with open(os.path.join(out, "checks.json"), encoding="utf-8") as f:
        chk = json.load(f)
    n_pool = None
    if os.path.exists(os.path.join(out, "pool_record.json")):
        with open(os.path.join(out, "pool_record.json"), encoding="utf-8") as f:
            n_pool = json.load(f).get("n")
    scr = R0.read_jsonl(os.path.join(out, "screen.jsonl"))
    metas = R0.read_jsonl(os.path.join(out, "items_done.jsonl"))
    rate = chk.get("eligible_rate") or 0.0
    sc = [r["sec_clean"] for r in scr if r.get("sec_clean") is not None]
    si = [m["sec_item"] for m in metas]
    proj = None
    if n_pool and sc and si:
        proj = round((n_pool * sum(sc) / len(sc) + n_pool * rate * sum(si) / len(si)) / 3600.0, 2)
    chk.update({"n_pool": n_pool, "projected_eligible": round(rate * n_pool) if n_pool else None,
                "projected_full_card_hours": proj, "gate_80": "not applied (breadth cell, n >= 20)",
                "full_shards_rule": {"shards": (int(math.ceil(proj / (9 * 0.6))) if proj else None),
                                     "rule": "ceil(projected single-card hours / 5.4), each shard under 9 h at 0.6x"},
                "shards_merged": shards, "matching": "gold intermediates"})
    # RA: the four arms with intervals and rewrite_back (plan 6e)
    rows = R0.read_jsonl(os.path.join(out, "rows.jsonl"))
    by = {}
    for r in rows:
        by.setdefault(int(r["idx"]), {})[r["arm"]] = r
    focus = {}
    for a in ("text", "cv1", "kv1", "span1"):
        its = [i for i in by if a in by[i]]
        focus["CA_" + a] = R0.boot_mean([1.0 if by[i][a]["outcome"] == "B" else 0.0 for i in its])
    rb = {}
    for a in ("text", "cv1", "cv2", "cv3", "kv1", "cvP", "rnd1", "rnd3"):
        its = [i for i in by if a in by[i] and "rewrite_back" in by[i][a]]
        rb[a] = R0.boot_mean([1.0 if by[i][a]["rewrite_back"] else 0.0 for i in its])
    chk["thinking_focus"] = focus
    chk["rewrite_back"] = rb
    R0.write_json(os.path.join(out, "checks.json"), chk)
    fmt = lambda v: "-" if v is None else "%.3f [%.3f, %.3f] (n %d)" % tuple(v)
    lines = ["", "## Thinking cell (plan 6e)", "",
             "| readout | estimate, 90% interval |", "|---|---|"]
    lines += ["| %s | %s |" % (k, fmt(v)) for k, v in focus.items()]
    lines += ["| rewrite_back %s | %s |" % (k, fmt(v)) for k, v in rb.items()]
    lines += ["", "rewrite_back = after P the chain writes value_A again before its answer marker."]
    if cfg.get("ds") == "asdiv":
        lines += ["", R0.ASDIV_NOTE]
    with open(os.path.join(out, "summary.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    # audit wording limits (W2, W10): no identity line; the notes replace ra30.summarize's identity line
    import ra30_flags
    nt = ra30_flags.notes(out)
    with open(os.path.join(out, "summary.md"), encoding="utf-8") as f:
        md = f.read().split("\n")
    with open(os.path.join(out, "summary_notes.md"), encoding="utf-8") as f:
        note_lines = [x for x in f.read().split("\n")[2:] if x]
    md = [x for x in md if not x.startswith("Identity cvP = cv1")]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md + ["", "## Notes (audit wording limits)", ""] + note_lines) + "\n")
    chk["notes"] = nt
    R0.write_json(os.path.join(out, "checks.json"), chk)
    return res


def host_tokens(model, ds, out, pool_root=None, pool_local=False, tok=None):
    import transformers
    if tok is None:
        if transformers.__version__ != R0.TRANSFORMERS_PIN:
            R0.die(R0.EXIT_PLAN, "transformers %s, pinned %s" % (transformers.__version__, R0.TRANSFORMERS_PIN))
        from prod import steer_gen as S
        tok = S.load_tokenizer(model, os.environ.get("K17_THINK_DIR"))
    items, rec = load_pool(model, ds, pool_root, pool_local)
    d = os.path.join(out, model, "tokens")
    os.makedirs(d, exist_ok=True)
    bad, n = [], 0
    with open(os.path.join(d, "%s.jsonl" % ds), "w", encoding="utf-8") as f:
        for it in items:
            g = geometry(tok, ds, it)
            n += 1
            if g is None:
                bad.append(int(it["idx"]))
            f.write(json.dumps({"idx": int(it["idx"]), "ok": g is not None,
                                "prompt_A": g["prompt_A"] if g else None, "span": g["span"] if g else None}) + "\n")
    chk = {"model": model, "dataset": ds, "n": n, "n_mismatch": len(bad), "mismatch_idx": bad, "pool": rec,
           "pass": len(bad) <= 0.01 * max(1, n)}
    R0.write_json(os.path.join(d, "%s_check.json" % ds), chk)
    R0.log("host tokens %s %s: %d items, %d mismatches" % (model, ds, n, len(bad)))
    if not chk["pass"]:
        R0.die(R0.EXIT_PLAN, "more than 1 percent of the Thinking prompts do not rebuild on the host tokenizer")
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--model", choices=sorted(MODELS), default="ouro_1_4b_think")
    ap.add_argument("--ds", choices=RD.DATASETS, default="gsm8k")
    ap.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--phase", choices=("all", "screen", "rows", "merge"), default="all")
    ap.add_argument("--rows_part", type=int, default=0)
    ap.add_argument("--rows_parts", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=4096)
    ap.add_argument("--pool_root", default=os.environ.get("OUT_ROOT"))
    ap.add_argument("--pool_local", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--think_dir", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--plan_sha256", default="")
    ap.add_argument("--no_plan_check", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.tokens:
        host_tokens(a.model, a.ds, a.out, a.pool_root, a.pool_local)
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
