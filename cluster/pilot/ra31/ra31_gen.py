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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ra31_gen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("partb")
    p.add_argument("--shard", required=True, choices=sorted(shards()))
    p.add_argument("--out", required=True)
    p.add_argument("--mem_fraction", type=float, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
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
