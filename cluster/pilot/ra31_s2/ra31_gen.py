"""RA-31 Part B: the scaled branch write under free greedy decoding, Ouro-1.4B base, GSM8K pool of Table 2 (the a1_gsm8k
job of the cluster round, prompt A, 395 scored rows plus 22 sentinels, the r00 layout). Plan: plan_ra31.md section 13.

The write of a branch is its post-norm output (input_layernorm_2 after o_proj, post_attention_layernorm_2 after
down_proj). Ops, at every position from the last prompt token through every generated token, loops 2 to 4:
- "scale" (whole write): y = alpha * w, float32 math, cast once (an exact scale op, not a lerp to zero);
- "subset": y = w + (alpha - 1) c_S, c_S = g * (W[:, S] x_S) / rms(u) with the live input x (down_proj or o_proj
  input) and the full pre-norm vector u (lib/ra31_hooks.contrib arithmetic), S the frozen Part B set.
The hooks are GenWriteHooks(SteerHooks) of the bundle's prod/steer_gen.py (md5 pinned below), so prod.generate.main
runs unchanged through steer_gen.run_gmain (batch width 16, the r00 layout).

    python ra31_gen.py list                          shards, arms and card-hours
    python ra31_gen.py partb --shard NAME --out OUT  one shard (resumable; a done marker per arm)
    python ra31_gen.py summarize --out OUT           merge every shard folder under OUT
    python ra31_gen.py cell_list                     EXT6c cells (section 15): shards and card-hours
    python ra31_gen.py cell --cell C --shard S --out OUT [--top_sets F] [--pool_root R]   one cell shard
    python ra31_gen.py cell_summarize --cell C --out OUT [--top_sets F] [--pool_root R]
    python ra31_gen.py freeze26 --pilot P --out F     the 2.6B set file from the EXT6 2.6B Part A pilot (CPU)
    python ra31_gen.py dep26 --out_root R             submit helper: afterok id of that pilot, or done
Exit codes: 2 pin or data mismatch; 3 the alpha-1 identity failed; other nonzero: the G.main exit of an arm.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lib")
TREE = os.path.join(HERE, "tree")
for _p in (LIB, TREE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import ra31_hooks as H31  # noqa: E402

DATA_B = os.path.join(HERE, "data_partb")
STEER_GEN_MD5 = "3b6e315346a8a7cf54048bfe9cf52ad3"   # the cluster-bundle copy (review risk d); tree/prod/steer_gen.py
STEER_GEN_VERSION = "k17gen-1"                        # the version string r00's meta records (model_meta.steer_gen)
TRANSFORMERS = "4.56.2"
MODEL = "ouro_1_4b_base"
K = 4
LOOPS = (2, 3, 4)
ALPHAS_LO = (0.0, 0.25, 0.5, 0.75)
ALPHAS_HI = (1.25, 1.5, 2.0)
BRANCHES = ("mlp", "attn")
SETS = ("all", "top", "rand")
HORIZON = 512
BATCH_WIDTH = 16
# Card-hour model (printed per shard): the brief's 3 s per decoded chain taken as the single-card rate (r00 measured 1.43 s
# per row at batch width 16 on the faster host), an A100 at 0.6 x the single card's speed (the slow end of 0.6 to 1.0), the
# subset hooks at 1.2 x, two loops at 0.55 x four loops; 417 rows per arm.
CARD_S_PER_ROW = 3.0
A100_SPEED = 0.6
SUBSET_COST = 1.2
K2_COST = 0.55
N_ROWS = 417
EXIT_PIN, EXIT_IDENTITY = 2, 3


def arm_name(kind, br, s, a):
    return "%s_%s_%s_a%s" % (kind, br, s, ("%.2f" % a).rstrip("0").rstrip("."))


def shards():
    """{shard: [arm]}; arm = {"name", "k", "edits": [{branch, op, alpha, set, loops, positions}]}. 53 arms plus
    r00_rerun (the unedited run on the cluster card, plain SteerHooks as r00 had: the reference of the bit-exact
    alpha-1 identity, since token identity with the stored r00 across GPU types is not guaranteed)."""
    out = {}
    for br in BRANCHES:
        for s in SETS:
            for half, als in (("lo", ALPHAS_LO), ("hi", ALPHAS_HI)):
                out["%s_%s_%s" % (br, s, half)] = [
                    {"name": arm_name("k4", br, s, a), "k": K,
                     "edits": [{"branch": br, "op": "scale" if s == "all" else "subset", "alpha": a, "set": s,
                                "loops": list(LOOPS), "positions": "from_last"}]} for a in als]
    for half, als in (("lo", ALPHAS_LO), ("hi", (1.0,) + ALPHAS_HI)):
        out["k2_%s" % half] = [{"name": arm_name("k2", "mlp", "all", a), "k": 2,
                                "edits": [{"branch": "mlp", "op": "scale", "alpha": a, "set": "all", "loops": [2],
                                           "positions": "from_last"}]} for a in als]
    ident = {"name": "ident", "k": K, "edits": []}
    for br in BRANCHES:
        ident["edits"].append({"branch": br, "op": "scale", "alpha": 1.0, "set": "all", "loops": list(LOOPS),
                               "positions": "from_last"})
        for s in ("top", "rand"):
            ident["edits"].append({"branch": br, "op": "subset", "alpha": 1.0, "set": s, "loops": list(LOOPS),
                                   "positions": "from_last"})
    rerun = {"name": "r00_rerun", "k": K, "edits": [], "plain": True}
    out["ident_allpos"] = [rerun, ident] + [
        {"name": "allpos_%s_a0" % br, "k": K, "edits": [{"branch": br, "op": "scale", "alpha": 0.0, "set": "all",
                                                       "loops": list(LOOPS), "positions": "all"}]}
        for br in BRANCHES]
    return out


def shard_hours(arms):
    h = 0.0
    for a in arms:
        f = K2_COST if a["k"] == 2 else 1.0
        if any(e["op"] == "subset" for e in a["edits"]):
            f *= SUBSET_COST
        h += N_ROWS * CARD_S_PER_ROW / A100_SPEED * f / 3600.0
    return round(h, 2)


# ------------------------------------------------------------------ the hooks
def _sg():
    import prod.steer_gen as SG
    return SG


def make_hooks(k, edits, sets):
    """GenWriteHooks over the bundle's SteerHooks (imported lazily, so the module loads without the tree)."""
    SG = _sg()

    class GenWriteHooks(SG.SteerHooks):
        """SteerHooks plus the RA-31 write edits. Rows carry no SteerHooks edits, so the parent never acts; its
        per-forward context (row keys, absolute positions, current_ut, layer) drives these hooks."""

        def __init__(self, k_, edits_, sets_):
            SG.SteerHooks.__init__(self, k_)
            self.edits = [dict(e) for e in edits_]
            self.sets = sets_
            self.live = {}
            self.wf = {}
            self.n_edit_calls = 0

        def install(self, model, kv=False):
            SG.SteerHooks.install(self, model, kv=kv)
            layers = model.model.layers[:self.L]
            self.units = {}
            cfg = model.config
            self.H, self.hd, self.I = int(cfg.num_attention_heads), int(cfg.head_dim), int(cfg.intermediate_size)
            for li, layer in enumerate(layers):
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._stash(li, "mlp")))
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._stash(li, "attn")))
                self.handles.append(layer.post_attention_layernorm_2.register_forward_hook(
                    self._edit(li, "mlp", layer.mlp.down_proj)))
                self.handles.append(layer.input_layernorm_2.register_forward_hook(
                    self._edit(li, "attn", layer.self_attn.o_proj)))
            return self

        def _layer_idx(self, br, s, li, device):
            key = (br, s, li)
            if key not in self.units:
                if s == "all":
                    n = self.I if br == "mlp" else self.H
                    us = list(range(n))
                else:
                    n = self.I if br == "mlp" else self.H
                    us = [u % n for u in self.sets[br][s] if u // n == li]
                if not us:
                    self.units[key] = None
                elif br == "mlp":
                    self.units[key] = torch.tensor(sorted(us), dtype=torch.long)
                else:
                    self.units[key] = H31.head_cols(us, self.hd)
            v = self.units[key]
            return None if v is None else v.to(device)

        def _mask(self, f, positions, device):
            """[B, T] bool: the positions this forward edits, per row (rows without an entry: none)."""
            keys, absp = f.get("keys"), f["abs"]
            if keys is None:
                return None
            M = torch.zeros(absp.shape, dtype=torch.bool, device=absp.device)
            for b, key in enumerate(keys):
                info = self.rows.get(key)
                if info is None:
                    continue
                lo = 0 if positions == "all" else int(info["plen"]) - 1
                M[b] = absp[b] >= lo
            return M.to(device)

        def _stash(self, li, br):
            def hook(module, args):
                f = self.f
                if f is not None and f.get("li") == li:
                    self.live[br] = args[0]
                return None
            return hook

        def _edit(self, li, br, proj):
            def hook(module, args, out):
                f = self.f
                if f is None or f.get("ut") is None:
                    return None
                j = int(f["ut"]) + 1
                es = [e for e in self.edits if e["branch"] == br and j in e["loops"]]
                if not es:
                    return None
                y = None
                for e in es:
                    M = self._mask(f, e["positions"], out.device)
                    if M is None or not bool(M.any()):
                        continue
                    cur = (y if y is not None else out)
                    w = cur[M]
                    if e["op"] == "scale":
                        new = (w.float() * float(e["alpha"])).to(out.dtype)
                    else:
                        idx = self._layer_idx(br, e["set"], li, out.device)
                        if idx is None:
                            continue
                        x = self.live[br][M]
                        u = args[0][M]
                        wkey = (br, e["set"], li)
                        if wkey not in self.wf:                  # the set's projection columns, float32, once
                            self.wf[wkey] = proj.weight[:, idx].float()
                        Wf = self.wf[wkey]
                        c = module.weight.float() * (x[:, idx].float() @ Wf.t()) / \
                            H31.rms(u, H31.UH.norm_eps(module))[:, None]
                        new = (w.float() + (float(e["alpha"]) - 1.0) * c).to(out.dtype)
                        st = self.stats.setdefault(("cS_norm_%s_%s" % (br, e["set"]), "sum_n"), [0.0, 0])
                        st[0] += float(c.norm(dim=-1).sum())
                        st[1] += int(c.shape[0])
                    if y is None:
                        y = out.clone()
                    y[M] = new
                    self.n_edit_calls += 1
                return y
            return hook

    return GenWriteHooks(k, edits, sets)


# ------------------------------------------------------------------ data and pins
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def check_data(data_dir=DATA_B):
    src = json.load(open(os.path.join(data_dir, "sources_partb.json")))
    bad = [f for f, h in src["outputs"].items() if sha256_file(os.path.join(data_dir, f)) != h]
    if bad:
        raise SystemExit(EXIT_PIN)
    return src


def check_pins(tree=TREE, data_dir=DATA_B):
    """steer_gen md5 = the bundle copy; its VERSION = the version r00's meta records; transformers pin. r00's meta
    holds the version string and the modeling md5, not the steer_gen file md5 (recorded in the plan)."""
    sg = os.path.join(tree, "prod", "steer_gen.py")
    meta = json.load(open(os.path.join(data_dir, "r00_meta.json")))
    out = {"steer_gen_md5": md5_file(sg), "steer_gen_md5_pinned": STEER_GEN_MD5,
           "r00_meta_steer_gen": meta["model_meta"].get("steer_gen"),
           "r00_meta_modeling_md5": meta["model_meta"].get("file_md5", {}).get("modeling_ouro.py")}
    bad = []
    if out["steer_gen_md5"] != STEER_GEN_MD5:
        bad.append("steer_gen md5 %s" % out["steer_gen_md5"])
    if out["r00_meta_steer_gen"] != STEER_GEN_VERSION:
        bad.append("r00 meta steer_gen %s" % out["r00_meta_steer_gen"])
    return out, bad


def load_sets(data_dir=DATA_B):
    d = json.load(open(os.path.join(data_dir, "partb_sets_1.4b.json")))
    return {"mlp": {"top": d["mlp"]["top"], "rand": d["mlp"]["rand"]},
            "attn": {"top": d["attn"]["top"], "rand": d["attn"]["rand"]}}


def row_ids(data_dir=DATA_B):
    it = json.load(open(os.path.join(data_dir, "items.json")))
    scored = [int(x["idx"]) for x in it["scored"]]
    return scored, scored + [int(s["idx"]) for s in it["sentinels"]]


def row_infos(tok, ids, chat=False):
    """steer.rows for every decoded row: the prompt-A ids (G.main's own prompt builder) and plen; no SG edits."""
    SG = _sg()
    from prod.tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(SG.TASK_A)}
    infos = {}
    for i in ids:
        pA = SG.prompt_texts(tok, SG.TASK_A, [srows[int(i)]], chat)[0]
        a_ids = tok(pA, add_special_tokens=False)["input_ids"]
        infos[str(i)] = {"plen": len(a_ids), "a_ids": a_ids, "edits": [], "lcdl": None}
    return infos


# ------------------------------------------------------------------ one shard
def default_gen(arm, hooks, ids, out_dir, mem_fraction):
    SG = _sg()
    return SG.run_gmain(MODEL, SG.TASK_A, arm["k"], ids, out_dir, HORIZON, hooks, False, think_stop=True,
                        mem_fraction=mem_fraction, batch_width=BATCH_WIDTH)


def default_tok():
    return _sg().load_tokenizer(MODEL)


def chains_of(job_dir):
    return _sg().load_chains(job_dir)


def stored_r00(data_dir=DATA_B):
    ref = {}
    with open(os.path.join(data_dir, "r00_chains.jsonl"), encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                ref[int(r["idx"])] = r["chain_ids"]
    return ref


def identity_check(job_dir, ref):
    """Token by token: the chains of job_dir against ref {idx: chain_ids} (same rows, same layout)."""
    got = {i: r["chain_ids"] for i, r in chains_of(job_dir).items()}
    diff = sorted(i for i in ref if got.get(i) != ref[i])
    return {"n_ref": len(ref), "n_got": len(got), "n_identical": len(ref) - len(diff), "differ": diff[:20],
            "pass": not diff and len(got) == len(ref)}


def run_shard(shard, out, mem_fraction=None, gen=None, tok_fn=None, data_dir=DATA_B, tree=TREE):
    gen = gen or default_gen
    tok_fn = tok_fn or default_tok
    arms = shards()[shard]
    os.makedirs(out, exist_ok=True)
    src = check_data(data_dir)
    pins, bad = check_pins(tree, data_dir)
    if bad:
        print("[ra31_gen] pins differ: %s" % bad, flush=True)
        raise SystemExit(EXIT_PIN)
    try:
        import transformers
        tv = transformers.__version__
    except ImportError:
        tv = None
    sets = load_sets(data_dir)
    _, ids = row_ids(data_dir)
    tok = tok_fn()
    infos = row_infos(tok, ids)
    head = {"shard": shard, "arms": [a["name"] for a in arms], "card_hours_a100_0.6": shard_hours(arms),
            "pins": pins, "transformers": tv, "data_sha256": src["outputs"], "n_rows": len(ids),
            "code_sha256": sha256_file(os.path.abspath(__file__))}
    json.dump(head, open(os.path.join(out, "shard_%s.json" % shard), "w"), indent=1, sort_keys=True)
    for arm in arms:
        d = os.path.join(out, arm["name"])
        done = os.path.join(d, "DONE.json")
        if os.path.exists(done):
            print("[ra31_gen] %s done, skipped" % arm["name"], flush=True)
            continue
        hooks = _sg().SteerHooks(arm["k"]) if arm.get("plain") else make_hooks(arm["k"], arm["edits"], sets)
        hooks.rows = {k_: dict(v) for k_, v in infos.items()}
        hooks.n_edit_calls = getattr(hooks, "n_edit_calls", 0)
        t0 = time.time()
        summ = gen(arm, hooks, ids, d, mem_fraction)
        code = int(summ.get("exit", 1))
        rec = {"arm": arm, "exit": code, "seconds": round(time.time() - t0, 1), "n_edit_calls": hooks.n_edit_calls,
               "cS_norm": {kk[0]: (v[0] / max(1, v[1])) for kk, v in hooks.stats.items()
                           if isinstance(kk, tuple) and str(kk[0]).startswith("cS_norm")}}
        if code != 0:
            print("[ra31_gen] %s G.main exit %d" % (arm["name"], code), flush=True)
            raise SystemExit(code)
        if arm.get("plain") or arm["name"] == "ident":
            rec["vs_stored_r00"] = identity_check(d, stored_r00(data_dir))       # printed, across GPU types
        if arm["name"] == "ident":
            rec["identity"] = identity_check(d, {i: r["chain_ids"] for i, r in
                                                 chains_of(os.path.join(out, "r00_rerun")).items()})
            json.dump(rec, open(os.path.join(d, "identity.json"), "w"), indent=1)
            print("[ra31_gen] alpha-1 identity against r00_rerun %s: %s; against the stored r00: %s" % (
                "PASS" if rec["identity"]["pass"] else "FAIL", json.dumps(rec["identity"]),
                json.dumps(rec["vs_stored_r00"])), flush=True)
            if not rec["identity"]["pass"]:
                json.dump(rec, open(os.path.join(d, "FAILED.json"), "w"), indent=1)
                raise SystemExit(EXIT_IDENTITY)
        json.dump(rec, open(done, "w"), indent=1)
        print("[ra31_gen] %s done %.0fs" % (arm["name"], rec["seconds"]), flush=True)
    print("[ra31_gen] shard %s done" % shard, flush=True)
    return 0


# ------------------------------------------------------------------ summarize (CPU)
def cells_of(job_dir):
    return _sg().load_cells(job_dir)


def arm_readout(job_dir, scored):
    """Accuracy (the Table 2 parser: cells trace_correct), share with no parsable answer (trace_answer None), mean
    chain length (n_trace), over the scored rows; 90 percent cluster bootstrap (cluster = problem)."""
    import k17_freeze as KF
    KO = KF.stats_mod()
    c = cells_of(job_dir)
    recs = [dict(c[i], cluster=str(i)) for i in scored if i in c]
    if not recs:
        return None
    acc = KF.cell(KO, recs, lambda r: 1.0 if r.get("trace_correct") else 0.0)
    nop = KF.cell(KO, recs, lambda r: 1.0 if r.get("trace_answer") is None else 0.0)
    ln = sum(float(r.get("n_trace") or 0) for r in recs) / len(recs)
    return {"n": len(recs), "n_missing": len(scored) - len(recs), "accuracy": acc, "no_answer": nop,
            "mean_chain_length": round(ln, 2)}


def summarize(out, data_dir=DATA_B):
    import results_common as RC
    scored, _ = row_ids(data_dir)
    table = {}
    for shard, arms in shards().items():
        for arm in arms:
            d = os.path.join(out, arm["name"])
            done = os.path.join(d, "DONE.json")
            if not os.path.exists(done):
                table[arm["name"]] = {"status": "not run"}
                continue
            dn = json.load(open(done))
            r = arm_readout(d, scored) or {}
            r.update({"status": "done", "shard": shard, "seconds": dn["seconds"], "cS_norm": dn.get("cS_norm"),
                      "identity": dn.get("identity"), "vs_stored_r00": dn.get("vs_stored_r00")})
            table[arm["name"]] = r
    ref = arm_readout_ref(data_dir, scored)
    rows = []
    for nm, r in table.items():
        if r.get("status") != "done" or not r.get("accuracy"):
            continue
        st = r["accuracy"]
        parts = nm.split("_")
        a = float(nm.rsplit("_a", 1)[1]) if "_a" in nm else None          # ident and r00_rerun: alpha 1
        rows.append(RC.row(grid="ra31", card="scaled_writes_partB", script="ra31_gen.py", model=MODEL, dataset="gsm8k",
                           arm=nm if a is None else "_".join(parts[:3]), site="r..end|loop2-4" if "k2" not in nm
                           else "r..end|loop2", loop=2 if "k2" in nm else 4, strength=1.0 if a is None else a,
                           metric="accuracy", endpoint="parsed_answer", item_set="own", n_items=st["n_defined"],
                           n_clusters=st["n_clusters"], estimate=st["mean"], neither_rate=r["no_answer"]["mean"],
                           ci90_lo=st["ci90"][0], ci90_hi=st["ci90"][1], ci90_method="stored", ci95_lo=st["ci95"][0],
                           ci95_hi=st["ci95"][1], source_key="results_partb.json:arms.%s.accuracy" % nm))
    RC.write_results(rows, os.path.join(out, "results_partb.csv"))
    n_done = sum(1 for r in table.values() if r.get("status") == "done")
    ident = table.get("ident", {}).get("identity")
    sets_info = json.load(open(os.path.join(data_dir, "partb_sets_1.4b.json")))
    ck = {"n_arms": len(table), "n_done": n_done, "identity_pass": None if not ident else ident["pass"],
          "set_rule": sets_info.get("rule"), "set_deviation": sets_info.get("deviation"),
          "set_ties_at_cut": {br: (sets_info.get(br) or {}).get("info") for br in BRANCHES},
          "ident_vs_stored_r00": table.get("ident", {}).get("vs_stored_r00"),
          "rerun_vs_stored_r00": table.get("r00_rerun", {}).get("vs_stored_r00"),
          "unedited_r00": ref, "csv_rows": len(rows)}
    json.dump({"arms": table, "checks": ck}, open(os.path.join(out, "results_partb.json"), "w"), indent=1)
    json.dump(ck, open(os.path.join(out, "checks_partb.json"), "w"), indent=1)
    L = ["# RA-31 Part B summary (Ouro-1.4B base, GSM8K, scored rows %d)" % len(scored), "",
         "Accuracy = cells trace_correct (the Table 2 parser), mean [90%% cluster bootstrap]; r00 unedited %s." %
         _fmt(ref and ref["accuracy"]), "", "| arm | accuracy | no answer | mean chain length | mean ||c_S|| |",
         "|---|---|---|---|---|"]
    for nm, r in table.items():
        if r.get("status") != "done":
            L.append("| %s | not run | | | |" % nm)
            continue
        L.append("| %s | %s | %s | %s | %s |" % (nm, _fmt(r.get("accuracy")), _fmt(r.get("no_answer")),
                                                r.get("mean_chain_length"), json.dumps(r.get("cS_norm"))))
    L += ["", "Checks: " + json.dumps(ck)]
    open(os.path.join(out, "summary_partb.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(ck, indent=1))
    return ck


def arm_readout_ref(data_dir, scored):
    import k17_freeze as KF
    KO = KF.stats_mod()
    c = {}
    with open(os.path.join(data_dir, "r00_cells.jsonl"), encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if "_header" not in r:
                    c[int(r["row_idx"])] = r
    recs = [dict(c[i], cluster=str(i)) for i in scored if i in c]
    if not recs:
        return None
    return {"n": len(recs), "accuracy": KF.cell(KO, recs, lambda r: 1.0 if r.get("trace_correct") else 0.0)}


def _fmt(st):
    return "n/a" if not st else "%.3f [%.3f, %.3f]" % (st["mean"], st["ci90"][0], st["ci90"][1])


# ================================================================== EXT6c cells (plan section 15)
# The Part B breadth grid: Ouro-1.4B base and 2.6B base by SVAMP, ASDiv, MAWPS and (2.6B) GSM8K; the 1.4B GSM8K cell
# is section 13 (above, unchanged). MLP lines (whole write, top set, random set) at alpha 0.5, 0.75, 1.25, 1.5, the
# loop-count arm (two loops, edit at loop 2: whole write at alpha 1, 1.25, 1.5; top and random at 1.25, 1.5), the two
# all-position alpha-0 arms (MLP, attention), r00_rerun (the four-loop alpha-1 baseline); every shard independent with
# its own identity gate (gate_rerun, gate_ident on the cell's first 10 scored problems).
CELL_ALPHAS = (0.5, 0.75, 1.25, 1.5)
CELL_K2_ALL = (1.0, 1.25, 1.5)
CELL_K2_SUB = (1.25, 1.5)
CELL_MAX_SCORED = 395                      # the 1.4B GSM8K cell's scored count: every cell keeps at most this many
RTX_GSM8K_S_PER_ROW = 597.0 / 417          # r00 of a1_gsm8k on the faster host: the reference of CARD_S_PER_ROW
STEER_GEN_MATH_MD5 = "f76f3e722e364d2471148602debd1c1c"    # cluster/pilot/steer_gen_math.py of the bundle
N_TOP_26B = 13517                          # 5 percent of 48 x 5632
PILOT_26B = "E6/ra31/a_2.6b/pilot_2.6b"    # the EXT6 2.6B Part A pilot (group g21_e6ra31a, stage s1_pilot)
SETS_26B = "E6/ra31/partb_sets_2.6b.json"  # written by the freeze26 job
PILOT_26B_STAGE = "g21_e6ra31a.s1_pilot"
CELLS = {
    # rtx: r00 seconds / rows on the faster host (cluster round); MAWPS: no local r00, the same model's ASDiv rate
    # [belief]; pool: "data" = data_partb/<cell>/ (copied, hashed), "runtime" = <pool_root>/<pool_rel> on the cluster
    "svamp_1.4b": {"model": "ouro_1_4b_base", "task": "svamp", "rtx": 309.2 / 252, "pool": "data", "top": "1.4b"},
    "asdiv_1.4b": {"model": "ouro_1_4b_base", "task": "asdiv", "rtx": 668.5 / 1058, "pool": "data", "top": "1.4b"},
    "mawps_1.4b": {"model": "ouro_1_4b_base", "task": "mawps", "rtx": 668.5 / 1058, "pool": "runtime",
                   "pool_rel": "A/steer_gen_math/a1_mawps", "top": "1.4b"},
    "gsm8k_2.6b": {"model": "ouro_2_6b_base", "task": "gsm8k", "rtx": 1303.0 / 447, "pool": "data", "top": "2.6b"},
    "svamp_2.6b": {"model": "ouro_2_6b_base", "task": "svamp", "rtx": 371.3 / 252, "pool": "data", "top": "2.6b"},
    "asdiv_2.6b": {"model": "ouro_2_6b_base", "task": "asdiv", "rtx": 832.1 / 1081, "pool": "data", "top": "2.6b"},
    "mawps_2.6b": {"model": "ouro_2_6b_base", "task": "mawps", "rtx": 832.1 / 1081, "pool": "runtime",
                   "pool_rel": "A/steer_gen_math/a2_mawps", "top": "2.6b"},
}
RUNTIME_ROWS_ASSUMED = CELL_MAX_SCORED + 22          # card-hours of a runtime pool before it is read


def cell_data(cell, data_root=DATA_B):
    return os.path.join(data_root, cell)


def cell_pool_path(cell, data_root=DATA_B, pool_root=None):
    c = CELLS[cell]
    if c["pool"] == "data":
        return cell_data(cell, data_root)
    if not pool_root:
        print("[ra31_gen] %s reads its pool at run time: give --pool_root (the cluster OUT_ROOT)" % cell, flush=True)
        raise SystemExit(EXIT_PIN)
    return os.path.join(pool_root, c["pool_rel"])


def cell_rows(cell, data_root=DATA_B, pool_root=None):
    """(scored kept, every decoded id, {idx: plen}, items.json path): the first CELL_MAX_SCORED scored problems in
    items.json order and every sentinel. A missing pool exits 2."""
    p = os.path.join(cell_pool_path(cell, data_root, pool_root), "items.json")
    if not os.path.exists(p):
        print("[ra31_gen] %s: pool items.json missing at %s" % (cell, p), flush=True)
        raise SystemExit(EXIT_PIN)
    it = json.load(open(p))
    sc = it["scored"][:CELL_MAX_SCORED]
    scored = [int(x["idx"]) for x in sc]
    return scored, scored + [int(s["idx"]) for s in it["sentinels"]], {int(x["idx"]): x.get("plen") for x in sc}, p


def cell_n_rows(cell, data_root=DATA_B):
    if CELLS[cell]["pool"] == "runtime":
        return RUNTIME_ROWS_ASSUMED
    return len(cell_rows(cell, data_root)[1])


GATE_ITEMS = None                          # None: every shard's identity gate runs on the shard's own items (the
#                                            rule: a fixed 10-item gate only if the full rerun doubled the cost; it adds
#                                            about 40 percent); an int: the cell's first GATE_ITEMS scored problems
MAX_SHARD_PACKED_H = 4.5                   # a shard over this many hours at 0.6 x is split by items, order[s::n]
MAX_PARTS = 8
GATE_COST = 1.0 + SUBSET_COST              # gate_rerun (plain) + gate_ident (every hook, alpha 1), per gate row


def cell_groups(cell):
    """{group: [arm]}, the 22 measurement arms of a cell in four independent groups: all (the whole-write line), lc
    (r00_rerun, the four-loop alpha-1 baseline on every item; the whole-write loop-count arm; the two all-position
    alpha-0 arms), top and rand (each line with its loop-count arm)."""
    def ed(br, op, a, s, loops, pos="from_last"):
        return {"branch": br, "op": op, "alpha": a, "set": s, "loops": list(loops), "positions": pos}
    g = {"all": [{"name": arm_name("k4", "mlp", "all", a), "k": K, "edits": [ed("mlp", "scale", a, "all", LOOPS)]}
                 for a in CELL_ALPHAS],
         "lc": ([{"name": "r00_rerun", "k": K, "edits": [], "plain": True}] +
                [{"name": arm_name("k2", "mlp", "all", a), "k": 2, "edits": [ed("mlp", "scale", a, "all", [2])]}
                 for a in CELL_K2_ALL] +
                [{"name": "allpos_%s_a0" % br, "k": K, "edits": [ed(br, "scale", 0.0, "all", LOOPS, "all")]}
                 for br in BRANCHES])}
    for s in ("top", "rand"):
        g[s] = ([{"name": arm_name("k4", "mlp", s, a), "k": K, "edits": [ed("mlp", "subset", a, s, LOOPS)]}
                 for a in CELL_ALPHAS] +
                [{"name": arm_name("k2", "mlp", s, a), "k": 2, "edits": [ed("mlp", "subset", a, s, [2])]}
                 for a in CELL_K2_SUB])
    return g


def gate_arms():
    """The gate of every shard, on the cell's first GATE_ITEMS scored problems: an unedited rerun and the alpha-1
    identity with every hook installed (MLP scale, top and random subsets, attention scale); token by token equal."""
    def ed(br, op, s):
        return {"branch": br, "op": op, "alpha": 1.0, "set": s, "loops": list(LOOPS), "positions": "from_last"}
    return [{"name": "gate_rerun", "k": K, "edits": [], "plain": True},
            {"name": "gate_ident", "k": K, "edits": [ed("mlp", "scale", "all"), ed("mlp", "subset", "top"),
                                                    ed("mlp", "subset", "rand"), ed("attn", "scale", "all")]}]


def _arm_eq(arms):
    f = 0.0
    for a in arms:
        x = K2_COST if a["k"] == 2 else 1.0
        if any(e["op"] == "subset" for e in a["edits"]):
            x *= SUBSET_COST
        f += x
    return f


def _gate_reuses_rerun(arms):
    """With the own-items gate, a shard that already runs r00_rerun on its items compares gate_ident with it."""
    return GATE_ITEMS is None and any(a["name"] == "r00_rerun" for a in arms)


def _hours(cell, n_rows, arms, gate_rows):
    s_row = CELLS[cell]["rtx"] * CARD_S_PER_ROW / RTX_GSM8K_S_PER_ROW
    g = SUBSET_COST if _gate_reuses_rerun(arms) else GATE_COST
    h = (n_rows * _arm_eq(arms) + gate_rows * g) * s_row / 3600.0
    return round(h, 2), round(h / A100_SPEED, 2)


def cell_shards(cell, data_root=DATA_B):
    """{shard: {"group", "part", "n_parts", "arms"}}. Every shard is independent (its own gate; no ordering); a group
    whose packed hours exceed MAX_SHARD_PACKED_H is split by items into n parts, part s = decoded ids [s::n]."""
    n = cell_n_rows(cell, data_root)
    gate = min(GATE_ITEMS, n) if GATE_ITEMS else None
    out = {}
    for grp, arms in cell_groups(cell).items():
        k = 1
        while k < min(n, MAX_PARTS) and                 _hours(cell, -(-n // k), arms, gate if gate is not None else -(-n // k))[1] > MAX_SHARD_PACKED_H:
            k += 1
        for s in range(k):
            nm = "%s_%s" % (cell, grp) if k == 1 else "%s_%s_p%dof%d" % (cell, grp, s + 1, k)
            out[nm] = {"group": grp, "part": s, "n_parts": k, "arms": arms}
    return out


def all_cell_shard_names():
    names = set()
    for c in CELLS:
        names |= set(cell_shards(c))
        for grp in cell_groups(c):                 # any split a runtime pool may need
            for k in range(2, 5):
                names |= set("%s_%s_p%dof%d" % (c, grp, s + 1, k) for s in range(k))
    return sorted(names)


def cell_hours(cell, shard_or_arms, data_root=DATA_B):
    """(single-card hours, hours on an A100 at 0.6 x) of a shard (name) or of a list of arms over every row."""
    n = cell_n_rows(cell, data_root)
    if isinstance(shard_or_arms, str):
        sh = cell_shards(cell, data_root)[shard_or_arms]
        m = -(-n // sh["n_parts"])
        return _hours(cell, m, sh["arms"], min(GATE_ITEMS, n) if GATE_ITEMS else m)
    return _hours(cell, n, shard_or_arms, 0)


def shard_spec(cell, shard, data_root=DATA_B, pool_root=None):
    """The shard of a runtime pool is resolved from its name (p<s>of<n>) when the assumed row count differs."""
    sh = cell_shards(cell, data_root).get(shard)
    if sh is not None:
        return sh
    for grp, arms in cell_groups(cell).items():
        pre = "%s_%s_p" % (cell, grp)
        if shard.startswith(pre) and "of" in shard:
            s, n = shard[len(pre):].split("of")
            return {"group": grp, "part": int(s) - 1, "n_parts": int(n), "arms": arms}
    return None


def run_cell(cell, shard, out, mem_fraction=None, gen=None, tok_fn=None, data_root=DATA_B, tree=TREE,
             top_sets=None, infos_fn=None, pool_root=None):
    """One shard, independent of every other: the gate (gate_rerun, gate_ident on the cell's first GATE_ITEMS scored
    problems; token-by-token equal, exit 3 otherwise), then the shard's arms on its items. Output out/<shard>/<arm>."""
    c = CELLS[cell]
    sh = shard_spec(cell, shard, data_root, pool_root)
    if sh is None:
        print("[ra31_gen] shard %s is not part of cell %s" % (shard, cell), flush=True)
        raise SystemExit(EXIT_PIN)
    sets = cell_top_sets(cell, top_sets, data_root)
    sdir = os.path.join(out, shard)
    os.makedirs(sdir, exist_ok=True)
    data_sha, pins, bad = cell_pins(cell, tree, data_root)
    if bad:
        print("[ra31_gen] pins differ: %s" % bad, flush=True)
        raise SystemExit(EXIT_PIN)
    scored, ids, plens, items_path = cell_rows(cell, data_root, pool_root)
    part = ids[sh["part"]::sh["n_parts"]]
    gate = scored[:GATE_ITEMS] if GATE_ITEMS else list(part)
    reuse = _gate_reuses_rerun(sh["arms"])
    task = setup_task(cell, os.path.join(sdir, "_task_rows"))
    tok = (tok_fn or (lambda: _sg().load_tokenizer(c["model"])))()
    need = sorted(set(part) | set(gate))
    infos = (infos_fn or cell_row_infos)(tok, need, task)
    badp = [i for i in need if plens.get(i) is not None and infos[str(i)]["plen"] != plens[i]]
    if badp:
        print("[ra31_gen] prompt lengths differ from the pool's on %d rows (tokenizer?): %s" % (len(badp), badp[:10]),
              flush=True)
        raise SystemExit(EXIT_PIN)
    if gen is None:
        def gen(arm, hooks, ids_, d, mem):
            return _sg().run_gmain(c["model"], task, arm["k"], ids_, d, HORIZON, hooks, False, think_stop=True,
                                   mem_fraction=mem, batch_width=BATCH_WIDTH)
    hs = _hours(cell, len(part), sh["arms"], len(gate))
    head = {"cell": cell, "shard": shard, "group": sh["group"], "part": sh["part"], "n_parts": sh["n_parts"],
            "arms": [a["name"] for a in sh["arms"]], "gate_items": gate, "card_hours_single": hs[0],
            "card_hours_a100_0.6": hs[1], "pins": pins, "data_sha256": data_sha,
            "pool_items_sha256": sha256_file(items_path), "n_rows": len(part), "task": task, "model": c["model"],
            "set_file_sha256": sha256_file(top_sets) if top_sets else None,
            "code_sha256": sha256_file(os.path.abspath(__file__))}
    json.dump(head, open(os.path.join(sdir, "shard.json"), "w"), indent=1, sort_keys=True)
    order = gate_arms() + list(sh["arms"])
    if reuse:                                   # r00_rerun on the shard's items is the gate reference
        order = [a for a in sh["arms"] if a["name"] == "r00_rerun"] + [gate_arms()[1]] +             [a for a in sh["arms"] if a["name"] != "r00_rerun"]
    for arm in order:
        d = os.path.join(sdir, arm["name"])
        done = os.path.join(d, "DONE.json")
        if os.path.exists(done):
            print("[ra31_gen] %s %s done, skipped" % (shard, arm["name"]), flush=True)
            continue
        rows_ = gate if arm["name"].startswith("gate_") else part
        if reuse and arm["name"] == "gate_ident":
            rows_ = part
        hooks = _sg().SteerHooks(arm["k"]) if arm.get("plain") else make_hooks(arm["k"], arm["edits"], sets)
        hooks.rows = {str(i): dict(infos[str(i)]) for i in rows_}
        hooks.n_edit_calls = getattr(hooks, "n_edit_calls", 0)
        t0 = time.time()
        summ = gen(arm, hooks, rows_, d, mem_fraction)
        code = int(summ.get("exit", 1))
        rec = {"arm": arm, "exit": code, "seconds": round(time.time() - t0, 1), "n_edit_calls": hooks.n_edit_calls,
               "rows": list(rows_), "cS_norm": {kk[0]: (v[0] / max(1, v[1])) for kk, v in hooks.stats.items()
                                                if isinstance(kk, tuple) and str(kk[0]).startswith("cS_norm")}}
        if code != 0:
            print("[ra31_gen] %s %s G.main exit %d" % (shard, arm["name"], code), flush=True)
            raise SystemExit(code)
        if arm["name"] == "gate_ident":
            refd = os.path.join(sdir, "r00_rerun" if reuse else "gate_rerun")
            rec["identity"] = identity_check(d, {i: r["chain_ids"] for i, r in chains_of(refd).items()})
            rec["identity"]["reference"] = os.path.basename(refd)
            json.dump(rec, open(os.path.join(d, "identity.json"), "w"), indent=1)
            print("[ra31_gen] %s alpha-1 identity gate (%d items) %s: %s" % (
                shard, len(rows_), "PASS" if rec["identity"]["pass"] else "FAIL", json.dumps(rec["identity"])),
                flush=True)
            if not rec["identity"]["pass"]:
                json.dump(rec, open(os.path.join(d, "FAILED.json"), "w"), indent=1)
                raise SystemExit(EXIT_IDENTITY)
        json.dump(rec, open(done, "w"), indent=1)
        print("[ra31_gen] %s %s done %.0fs" % (shard, arm["name"], rec["seconds"]), flush=True)
    print("[ra31_gen] cell %s shard %s done" % (cell, shard), flush=True)
    return 0


def summarize_cell(cell, out, data_root=DATA_B, top_sets=None, pool_root=None):
    """Merge every shard of a cell: per arm, the cells and chains of its item parts; a problem met twice in one arm
    stops the merge (exit 2). Gates: every shard's gate_ident result. Agreement of the merged r00_rerun with the stored
    r00 printed where the pool has r00 chains."""
    import results_common as RC
    c = CELLS[cell]
    scored, ids, _, items_path = cell_rows(cell, data_root, pool_root)
    shards_ = {}
    if os.path.isdir(out):
        for nm in sorted(os.listdir(out)):
            if os.path.exists(os.path.join(out, nm, "shard.json")):
                shards_[nm] = json.load(open(os.path.join(out, nm, "shard.json")))
    merged, chains, seconds, cs, gates = {}, {}, {}, {}, {}
    for nm, head in shards_.items():
        if head.get("cell") != cell:
            continue
        g = os.path.join(out, nm, "gate_ident", "DONE.json")
        gates[nm] = json.load(open(g)).get("identity") if os.path.exists(g) else None
        for an in head["arms"]:
            d = os.path.join(out, nm, an)
            if not os.path.exists(os.path.join(d, "DONE.json")):
                continue
            dn = json.load(open(os.path.join(d, "DONE.json")))
            m = merged.setdefault(an, {})
            ch = chains.setdefault(an, {})
            cl, cc = cells_of(d), chains_of(d)
            dup = sorted(set(m) & set(cl))
            if dup:
                print("[ra31_gen] %s: arm %s holds problems twice (shard %s): %s" % (cell, an, nm, dup[:10]),
                      flush=True)
                raise SystemExit(EXIT_PIN)
            m.update(cl)
            ch.update({i: r["chain_ids"] for i, r in cc.items()})
            seconds[an] = seconds.get(an, 0.0) + dn["seconds"]
            for k_, v in (dn.get("cS_norm") or {}).items():
                cs.setdefault(an, {}).setdefault(k_, []).append(v)
    import k17_freeze as KF
    KO = KF.stats_mod()
    table = {}
    for grp, arms in cell_groups(cell).items():
        for arm in arms:
            an = arm["name"]
            m = merged.get(an)
            if not m:
                table[an] = {"status": "not run", "group": grp}
                continue
            recs = [dict(m[i], cluster=str(i)) for i in scored if i in m]
            r = {"status": "done" if len(recs) == len(scored) else "partial", "group": grp, "n": len(recs),
                 "n_missing": len(scored) - len(recs), "seconds": round(seconds.get(an, 0.0), 1),
                 "cS_norm": {k_: sum(v) / len(v) for k_, v in cs.get(an, {}).items()}}
            if recs:
                r["accuracy"] = KF.cell(KO, recs, lambda x: 1.0 if x.get("trace_correct") else 0.0)
                r["no_answer"] = KF.cell(KO, recs, lambda x: 1.0 if x.get("trace_answer") is None else 0.0)
                r["mean_chain_length"] = round(sum(float(x.get("n_trace") or 0) for x in recs) / len(recs), 2)
            table[an] = r
    ref = arm_readout_ref(cell_pool_path(cell, data_root, pool_root), scored) if c["pool"] == "data" else None
    r00 = cell_r00(cell, data_root, pool_root)
    vs = None
    if r00 and chains.get("r00_rerun"):
        got = chains["r00_rerun"]
        diff = sorted(i for i in got if r00.get(i) != got[i])
        vs = {"n": len(got), "n_identical": len(got) - len(diff), "differ": diff[:20]}
    rows = []
    for nm, r in table.items():
        if r.get("status") not in ("done", "partial") or not r.get("accuracy"):
            continue
        st = r["accuracy"]
        a = float(nm.rsplit("_a", 1)[1]) if "_a" in nm else None
        rows.append(RC.row(grid="ra31", card="scaled_writes_partB_%s" % cell, script="ra31_gen.py", model=c["model"],
                           dataset=c["task"], arm=nm if a is None else "_".join(nm.split("_")[:3]),
                           site=("all|loop2-4" if nm.startswith("allpos") else
                                 "r..end|loop2" if "k2" in nm else "r..end|loop2-4"),
                           loop=2 if "k2" in nm else 4, strength=1.0 if a is None else a, metric="accuracy",
                           endpoint="parsed_answer", item_set="own", n_items=st["n_defined"],
                           n_clusters=st["n_clusters"], estimate=st["mean"], neither_rate=r["no_answer"]["mean"],
                           ci90_lo=st["ci90"][0], ci90_hi=st["ci90"][1], ci90_method="stored", ci95_lo=st["ci95"][0],
                           ci95_hi=st["ci95"][1], source_key="results_partb_%s.json:arms.%s.accuracy" % (cell, nm)))
    RC.write_results(rows, os.path.join(out, "results_partb_%s.csv" % cell))
    exp = cell_shards(cell, data_root)
    ck = {"cell": cell, "model": c["model"], "task": c["task"], "n_scored": len(scored), "n_rows": len(ids),
          "pool_items_sha256": sha256_file(items_path), "n_arms": len(table),
          "n_done": sum(1 for r in table.values() if r.get("status") == "done"),
          "shards_found": sorted(n for n in shards_ if shards_[n].get("cell") == cell),
          "shards_planned": sorted(exp), "gates": gates,
          "identity_pass": bool(gates) and all(g and g.get("pass") for g in gates.values()),
          "rerun_vs_stored_r00": vs, "unedited_r00": ref, "csv_rows": len(rows)}
    if c["top"] == "1.4b":
        si = json.load(open(os.path.join(data_root, "partb_sets_1.4b.json")))
    else:
        si = json.load(open(top_sets)) if top_sets and os.path.exists(top_sets) else {}
        ck["set_file_sha256"] = sha256_file(top_sets) if si else None
    ck.update({"set_rule": si.get("rule"), "set_deviation": si.get("deviation"),
               "set_ties_at_cut": (si.get("mlp") or {}).get("info")})
    if c["top"] == "2.6b":
        ck["deviation_2.6b"] = ("top set frozen from the EXT6 2.6B Part A pilot (50 donor items; a cross-unit "
                                "dependency: freeze26 after g21_e6ra31a.s1_pilot), not from a full Part A run")
    if c["pool"] == "runtime":
        ck["deviation_pool"] = "pool read at run time from the cluster round (%s), hashed here" % c["pool_rel"]
    json.dump({"arms": table, "checks": ck}, open(os.path.join(out, "results_partb_%s.json" % cell), "w"), indent=1)
    json.dump(ck, open(os.path.join(out, "checks_partb_%s.json" % cell), "w"), indent=1)
    L = ["# RA-31 Part B cell %s (%s, %s, scored rows %d)" % (cell, c["model"], c["task"], len(scored)), "",
         "Accuracy = cells trace_correct (the Table 2 parser), mean [90%% cluster bootstrap]; r00 unedited %s." %
         _fmt(ref and ref["accuracy"]), "", "| arm | status | accuracy | no answer | mean chain length | mean ||c_S|| |",
         "|---|---|---|---|---|---|"]
    for nm, r in table.items():
        L.append("| %s | %s | %s | %s | %s | %s |" % (nm, r["status"], _fmt(r.get("accuracy")), _fmt(r.get("no_answer")),
                                                     r.get("mean_chain_length", ""), json.dumps(r.get("cS_norm", ""))))
    L += ["", "Checks: " + json.dumps(ck)]
    open(os.path.join(out, "summary_partb_%s.md" % cell), "w").write("\n".join(L) + "\n")
    print(json.dumps(ck, indent=1))
    return ck


def cell_top_sets(cell, top_sets=None, data_root=DATA_B):
    """The unit sets of a cell. 1.4B cells: the frozen 1.4B set. 2.6B cells: the file of the freeze26 job (from the
    EXT6 2.6B Part A pilot; model ouro_2_6b_base, 13,517 neurons), required; never a set from 1.4B data."""
    c = CELLS[cell]
    if c["top"] == "1.4b":
        return load_sets(data_root)
    if not top_sets or not os.path.exists(top_sets):
        print("[ra31_gen] %s needs the 2.6B set file (--top_sets, written by freeze26): %s" % (cell, top_sets),
              flush=True)
        raise SystemExit(EXIT_PIN)
    d = json.load(open(top_sets))
    mt, mr = d["mlp"]["top"], d["mlp"].get("rand", [])
    if d.get("model") != c["model"] or len(mt) != N_TOP_26B or len(mr) != N_TOP_26B or max(mt + mr) >= 48 * 5632:
        print("[ra31_gen] %s: the set file is not a 2.6B base set (model %s, %d / %d neurons)" % (
            cell, d.get("model"), len(mt), len(mr)), flush=True)
        raise SystemExit(EXIT_PIN)
    return {"mlp": {"top": mt, "rand": mr}, "attn": {"top": [], "rand": []}}


def setup_task(cell, rows_dir, here=HERE):
    """Point steer_gen at the cell's task: gsm8k and svamp are prod tasks; asdiv and mawps are registered by the
    bundle's steer_gen_math.install (copied unchanged, md5 pinned) from tree/data_steer/<task>."""
    task = CELLS[cell]["task"]
    if task in ("asdiv", "mawps"):
        p = os.path.join(here, "steer_gen_math.py")
        if md5_file(p) != STEER_GEN_MATH_MD5:
            print("[ra31_gen] steer_gen_math md5 differs", flush=True)
            raise SystemExit(EXIT_PIN)
        if here not in sys.path:
            sys.path.insert(0, here)
        import steer_gen_math
        steer_gen_math.install(task, rows_dir)
    return task


def cell_row_infos(tok, ids, task):
    SG = _sg()
    from prod.tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(task)}
    infos = {}
    for i in ids:
        pA = SG.prompt_texts(tok, task, [srows[int(i)]], False)[0]
        a_ids = tok(pA, add_special_tokens=False)["input_ids"]
        infos[str(i)] = {"plen": len(a_ids), "a_ids": a_ids, "edits": [], "lcdl": None}
    return infos


def cell_r00(cell, data_root=DATA_B, pool_root=None):
    """The stored r00 chains of the cell's pool, or None (a runtime pool without r00: agreement not printed)."""
    d = cell_pool_path(cell, data_root, pool_root)
    if CELLS[cell]["pool"] == "data":
        return stored_r00(d)
    fs = [f for f in os.listdir(os.path.join(d, "r00"))] if os.path.isdir(os.path.join(d, "r00")) else []
    ch = [f for f in fs if f.startswith("chains_") and f.endswith(".jsonl")]
    if len(ch) != 1:
        return None
    ref = {}
    with open(os.path.join(d, "r00", ch[0]), encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                ref[int(r["idx"])] = r["chain_ids"]
    return ref


def cell_pins(cell, tree=TREE, data_root=DATA_B):
    """Data cells: data hashes, steer_gen md5 and the r00 version string (check_pins). Runtime cells: the steer_gen
    md5 only (the pool is hashed into the shard record)."""
    if CELLS[cell]["pool"] == "data":
        src = check_data(cell_data(cell, data_root))
        pins, bad = check_pins(tree, cell_data(cell, data_root))
        return src["outputs"], pins, bad
    md5 = md5_file(os.path.join(tree, "prod", "steer_gen.py"))
    return {}, {"steer_gen_md5": md5}, ([] if md5 == STEER_GEN_MD5 else ["steer_gen md5 %s" % md5])


def freeze26(pilot, out):
    """CPU job: the 2.6B set file from the EXT6 2.6B Part A pilot (make_partb_cells.freeze_26b, the 1.4B rule with
    L = 48). Exit 2 unless the pilot's pilot_gate.json exists and passed."""
    pg = os.path.join(pilot, "pilot_gate.json")
    if not os.path.exists(pg) or not json.load(open(pg)).get("pass"):
        print("[ra31_gen] freeze26: no passing pilot gate at %s" % pg, flush=True)
        raise SystemExit(EXIT_PIN)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import make_partb_cells as MC
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    MC.freeze_26b(pilot, out)
    print("[ra31_gen] freeze26 wrote %s sha256 %s" % (out, sha256_file(out)), flush=True)
    return 0


def dep26(out_root):
    """Submit helper for the 2.6B cells: prints 'afterok:<job id>' of the EXT6 2.6B Part A pilot (the last line of
    $OUT_ROOT/_chain/submitted.tsv whose stage file is g21_e6ra31a.s1_pilot), or 'done' when that pilot's
    pilot_gate.json passed (and 'done set' when the set file also exists). Exit 2 if neither holds."""
    pg = os.path.join(out_root, PILOT_26B, "pilot_gate.json")
    if os.path.exists(pg) and json.load(open(pg)).get("pass"):
        print("done set" if os.path.exists(os.path.join(out_root, SETS_26B)) else "done")
        return 0
    sub = os.path.join(out_root, "_chain", "submitted.tsv")
    jid = None
    if os.path.exists(sub):
        for ln in open(sub):
            f = ln.split()
            if len(f) >= 2 and f[1].startswith(PILOT_26B_STAGE):
                jid = f[0]
    if jid:
        print("afterok:%s" % jid)
        return 0
    print("[ra31_gen] dep26: the EXT6 2.6B Part A pilot is neither submitted nor done", flush=True)
    raise SystemExit(EXIT_PIN)


def cells_main(argv):
    ap = argparse.ArgumentParser(prog="ra31_gen cells")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cell_list")
    p = sub.add_parser("cell")
    p.add_argument("--cell", required=True, choices=sorted(CELLS))
    p.add_argument("--shard", required=True, choices=all_cell_shard_names())
    p.add_argument("--out", required=True)
    p.add_argument("--mem_fraction", type=float, default=None)
    p.add_argument("--top_sets", default=None, help="2.6B cells: the freeze26 set file (required there)")
    p.add_argument("--pool_root", default=None, help="MAWPS cells: the cluster OUT_ROOT holding the round's pool")
    s = sub.add_parser("cell_summarize")
    s.add_argument("--cell", required=True, choices=sorted(CELLS))
    s.add_argument("--out", required=True)
    s.add_argument("--top_sets", default=None)
    s.add_argument("--pool_root", default=None)
    f = sub.add_parser("freeze26")
    f.add_argument("--pilot", required=True)
    f.add_argument("--out", required=True)
    d = sub.add_parser("dep26")
    d.add_argument("--out_root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "cell_list":
        tot = [0.0, 0.0]
        longest = (0.0, None)
        for cell in CELLS:
            ct = [0.0, 0.0]
            for nm, sh in cell_shards(cell).items():
                h = cell_hours(cell, nm)
                ct = [ct[0] + h[0], ct[1] + h[1]]
                longest = max(longest, (h[1], nm))
                print("%-22s %2d arms + gate  %.2f h single card, %.2f h on an A100 at %.1f x" % (
                    nm, len(sh["arms"]), h[0], h[1], A100_SPEED))
            print("%-22s cell total %.2f h single card, %.2f h at %.1f x%s" % (
                cell, ct[0], ct[1], A100_SPEED, " (rows assumed %d)" % RUNTIME_ROWS_ASSUMED
                if CELLS[cell]["pool"] == "runtime" else ""))
            tot = [tot[0] + ct[0], tot[1] + ct[1]]
        print("total %.2f h single card, %.2f h at %.1f x; longest shard %s %.2f h at %.1f x (2.6B: after freeze26)" % (
            tot[0], tot[1], A100_SPEED, longest[1], longest[0], A100_SPEED))
        return 0
    if a.cmd == "cell":
        return run_cell(a.cell, a.shard, a.out, a.mem_fraction, top_sets=a.top_sets, pool_root=a.pool_root)
    if a.cmd == "freeze26":
        return freeze26(a.pilot, a.out)
    if a.cmd == "dep26":
        return dep26(a.out_root)
    summarize_cell(a.cell, a.out, top_sets=a.top_sets, pool_root=a.pool_root)
    return 0


def main(argv=None):
    av = sys.argv[1:] if argv is None else list(argv)
    if av and av[0] in ("cell_list", "cell", "cell_summarize", "freeze26", "dep26"):
        return cells_main(av)
    ap = argparse.ArgumentParser(prog="ra31_gen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("partb")
    p.add_argument("--shard", required=True, choices=sorted(shards()))
    p.add_argument("--out", required=True)
    p.add_argument("--mem_fraction", type=float, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = ap.parse_args(av)
    if a.cmd == "list":
        tot = 0.0
        for nm, arms in shards().items():
            h = shard_hours(arms)
            tot += h
            print("%-16s %2d arms  %.2f card-h (A100 at %.1f x the single card)  %s" % (nm, len(arms), h, A100_SPEED,
                                                                          ",".join(x["name"] for x in arms)))
        print("total %d arms, %.2f card-h" % (sum(len(v) for v in shards().values()), tot))
        return 0
    if a.cmd == "partb":
        return run_shard(a.shard, a.out, a.mem_fraction)
    summarize(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
