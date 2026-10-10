"""K17 FREEZE RAVEN (brief RA-19 item 3): the RA-3 branch-freeze rows on Huginn-0125 and McLeish r32 (K = 32).
Plan: loop-signal/notes/k17_freeze_raven_plan.md (sha256 frozen before any scored row; --plan_sha256 must match).

Hooks come from the production package only (prod/steer: RavenBinding via adapters.bind, HookRegistry,
interventions.from_spec / expand / freeze_spec, readouts.Recorder), imported from a frozen tree copy (--tree).
This file adds no hook. Statistics helpers come from k17_freeze.py and the BOS / position-0 item rule from
k17_public.prep_item (both imported, unchanged). Items: the PUBLIC candidate files (cands_<tok>_<ds>.json).

    python k17_freeze_raven.py --run --model mcleish --out OUT --tree TREE --data DATA --plan PLAN --plan_sha256 SHA
        [--debug N] [--datasets D1,D2]
    python k17_freeze_raven.py --summarize --out OUT
"""
import argparse
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import k17_freeze as KF  # noqa: E402

K = 32
GAP_E = KF.GAP_E
NB = KF.NB
D2_CAP = 4
DATASETS = ("D1", "D2")
ANS_ROWS = ("E0", "ans_frz_mlp", "ans_frz_attn")
MODELS = {"huginn": {"prod": "huginn_0125", "tok": "huginn", "bos": None, "repo": "tomg-group-umd/huginn-0125"},
          "mcleish": {"prod": "mcleish_llama32_r32", "tok": "llama", "bos": 128000,
                      "repo": "smcleish/Recurrent-Llama-3.2-train-recurrence-32"}}
G5_REPORT_LOOPS = (1, 2, 4, 8, 16, 24, 31, 32)
STOP = {"flag": False}
r6, vals_at, margin = KF.r6, KF.vals_at, KF.margin


# ------------------------------------------------------------------ production objects around one bound model
class Ctx(object):
    def __init__(self, ad, binding):
        (self.I, self.Reg, self.RowInfo, self.Recorder, self.C, self.Positions,
         self.Site) = KF.prod_imports(TREE[0])
        self.ad = ad
        self.b = binding
        self.m = binding.model_module()
        self.g = binding.geometry(K)
        self.dev = binding.device()
        self.dtype = binding.dtype()
        I = self.I
        self.s_ans = I.expand("loop1.loop_end", {"kind": "once"}, self.g)[0]
        self.s_src = I.expand("loop1.L0.resid_post", {"kind": "once"}, self.g)[0]
        self.s_le = [I.expand("loop%d.loop_end" % j, {"kind": "once"}, self.g)[0] for j in range(1, K + 1)]
        self.s_fin = I.expand("coda.L%d.resid_post" % (self.g.n_coda - 1), {"kind": "once"}, self.g)[0]
        self.F_ans = self.g.first_changed(self.s_ans)
        self.F_src = self.g.first_changed(self.s_src)
        self.frz_sites = list(self.g.all_sites(["attn_out", "mlp_out"]))
        self.n_fwd = 0

    def x0(self, key, T):
        return self.b.x0_table(key)[:T].to(device=self.dev, dtype=self.dtype)[None]

    def nohook(self, ids, key):
        """The model's own forward with the item's x0 passed as input_states and no registry installed."""
        T = ids.shape[1]
        with torch.no_grad():
            o = self.m(input_ids=ids, num_steps=K, input_states=self.x0(key, T), use_cache=False,
                       cache_position=torch.arange(T, device=ids.device))
        self.n_fwd += 1
        return o.logits.float()

    def run(self, ids, row, ivs=(), recs=(), ro_id=None):
        """One hooked batch-1 forward. ro_id: recorder id holding LOOP_END at all positions; its per-loop pure
        read-outs (coda, ln_f, lm_head in the model dtype) are taken while the registry is still installed."""
        reg = self.Reg(self.b, K)
        for r in recs:
            reg.add_recorder(r)
        for iv in ivs:
            reg.add_intervention(iv)
        reg.install()
        ro = None
        try:
            with reg.batch([row], [0]):
                z = self.b.forward(ids, K, [row.key], x0_policy="fixed_table")
            out = reg.take_records()
            if ro_id is not None:
                ro = {}
                for j in range(1, K + 1):
                    h = out[(row.key, ro_id, self.s_le[j - 1].key())]
                    with torch.no_grad():
                        ro[j] = self.b.readout(j, h.to(device=self.dev, dtype=self.dtype)[None], "pure")[0]
        finally:
            reg.remove()
        self.n_fwd += 1
        return z, out, ro


@torch.no_grad()
def process_item(X, A, B, src, tA, tB, key):
    I, P = X.I, X.Positions
    T = len(A)
    r = T - 1
    src = sorted(src)
    ids_A = torch.tensor([A], device=X.dev)
    ids_B = torch.tensor([B], device=X.dev)
    spans = {"source": tuple(src), "answer": (r,)}
    at_r = P("abs", abs=(r,))

    def row(payload=None):
        return X.RowInfo(key=key, prompt_len=T, spans=dict(spans), payload=payload or {})

    def le_rec(rid):
        return X.Recorder(id=rid, sites=tuple(X.s_le), positions=P("all"), what="state", when="post")

    def H_of(recs, rid):
        return {j: recs[(key, rid, X.s_le[j - 1].key())] for j in range(1, K + 1)}

    rec = {"key": key, "T": T, "pos": src, "tA": tA, "tB": tB, "rows": {}}
    t0 = time.time()
    f0 = X.n_fwd
    z_nh = X.nohook(ids_A, key)[0]
    frz_rec = X.Recorder(id="frz", sites=tuple(X.frz_sites), positions=at_r, what="state", when="post")
    zA, rA, roA = X.run(ids_A, row(), recs=[le_rec("le"), frz_rec], ro_id="le")
    zA = zA[0]
    HA = H_of(rA, "le")
    srcB_rec = X.Recorder(id="srcB", sites=(X.s_src,), positions=P("span", span="source"), what="state",
                          when="post")
    finB_rec = X.Recorder(id="finB", sites=(X.s_fin,), positions=P("all"), what="state", when="post")
    zB, rB, roB = X.run(ids_B, row(), recs=[le_rec("le"), srcB_rec, finB_rec], ro_id="le")
    zB = zB[0]
    HB = H_of(rB, "le")
    srcB = rB[(key, "srcB", X.s_src.key())]
    finB = rB[(key, "finB", X.s_fin.key())]

    def patch(iid, site, pos, a=1.0):
        return I.from_spec({"id": iid, "op": "patch", "site": site.key(), "a": a, "target": {"kind": "donor"},
                            "positions": pos}, X.g)

    pos_r = {"kind": "abs", "abs": [r]}
    pos_src = {"kind": "span", "span": "source"}
    ans = patch("ans", X.s_ans, pos_r)
    ans0 = patch("ans", X.s_ans, pos_r, a=0.0)
    srcp = patch("src", X.s_src, pos_src)
    pcf = patch("pcf", X.s_fin, pos_r)
    sz = patch("sz", X.s_fin, pos_src)

    def freeze(comp, F):
        iv = I.from_spec(dict(I.freeze_spec(comp, pos_r, F, X.g), id="frz_" + comp), X.g)
        pay = {s.key(): rA[(key, "frz", s.key())] for s in iv.sites}
        return iv, pay

    fm, pay_fm = freeze("mlp_out", X.F_ans)
    fa, pay_fa = freeze("attn_out", X.F_ans)
    fa_s, pay_fa_s = freeze("attn_out", X.F_src)
    pay = {"ans": {X.s_ans.key(): HB[1][r:r + 1]},
           "src": {X.s_src.key(): srcB},
           "pcf": {X.s_fin.key(): finB[r:r + 1]},
           "sz": {X.s_fin.key(): finB[src]}}

    def edited(ivs, extra, with_le, with_ro=False):
        p = {iv.id: pay[iv.id] for iv in ivs if iv.id in pay}
        p.update(extra)
        z, rr, ro = X.run(ids_A, row(p), ivs=ivs, recs=[le_rec("le")] if with_le else [],
                          ro_id="le" if with_ro else None)
        return z[0], (H_of(rr, "le") if with_le else None), ro

    z_zero, _, _ = edited([ans0], {}, False)
    z_E0, H_E0, ro_E0 = edited([ans], {}, True, True)
    z_am, H_am, ro_am = edited([ans, fm], {"frz_mlp_out": pay_fm}, True, True)
    z_aa, H_aa, ro_aa = edited([ans, fa], {"frz_attn_out": pay_fa}, True, True)
    z_fm, _, _ = edited([fm], {"frz_mlp_out": pay_fm}, False)
    z_fa, _, _ = edited([fa], {"frz_attn_out": pay_fa}, False)
    z_src, _, _ = edited([srcp], {}, False)
    z_sf, H_sf, _ = edited([srcp, fa_s], {"frz_attn_out": pay_fa_s}, True)
    z_pc, _, _ = edited([pcf], {}, False)
    z_sz, _, _ = edited([sz], {}, False)

    ident = {
        "hook_identity": bool(torch.equal(zA, z_nh) and torch.equal(z_zero, zA)),
        "readout_identity": bool(torch.equal(roA[K][r], zA[r]) and torch.equal(roB[K][r], zB[r])),
        "freeze_no_edit_identity": bool(torch.equal(z_fm, zA) and torch.equal(z_fa, zA)),
        "source_attn_frozen_identity": bool(torch.equal(z_sf[r], zA[r]) and
                                            all(torch.equal(H_sf[j][r], HA[j][r]) for j in range(1, K + 1))),
        "source_edit_reaches_r": not bool(torch.equal(z_src[r], zA[r])),
        "pc_exact": bool(torch.equal(z_pc[r], zB[r])),
        "sz_exact": bool(torch.equal(z_sz[r], zA[r])),
        "ans_loop1_state_is_donor": all(bool(torch.equal(H[1][r], HB[1][r])) for H in (H_E0, H_am, H_aa)),
    }
    cA = vals_at(zA, r, tA, tB)
    cB = vals_at(zB, r, tA, tB)
    D = margin(cB) - margin(cA)
    rec["clean"] = [vals_at(roA[j], r, tA, tB) for j in range(1, K + 1)]
    rec["cleanB"] = [vals_at(roB[j], r, tA, tB) for j in range(1, K + 1)]
    rec["clean_final"] = cA
    rec["cleanB_final"] = cB
    rec["D"] = r6(D)
    for nm, z in (("zero", z_zero), ("frz_mlp", z_fm), ("frz_attn", z_fa), ("src", z_src),
                  ("src_frz_attn", z_sf), ("pc_final", z_pc), ("sz", z_sz)):
        v = vals_at(z, r, tA, tB)
        rec["rows"][nm] = {"final": v, "dm": r6(margin(v) - margin(cA))}
    TB = {j: HB[j][r] - HA[j][r] for j in range(1, K + 1)}
    for nm, z, H, ro in (("E0", z_E0, H_E0, ro_E0), ("ans_frz_mlp", z_am, H_am, ro_am),
                         ("ans_frz_attn", z_aa, H_aa, ro_aa)):
        per = []
        for j in range(1, K + 1):
            v = vals_at(ro[j], r, tA, tB)
            dl = H[j][r] - HA[j][r]
            tn = float(TB[j].norm())
            per.append({"v": v, "dm": r6(margin(v) - margin(rec["clean"][j - 1])),
                        "norm_ratio": r6(float(dl.norm()) / tn) if tn > 0 else None,
                        "cos": r6(torch.nn.functional.cosine_similarity(dl, TB[j], dim=0)) if tn > 0 else None,
                        "delta_norm": r6(dl.norm()), "target_norm": r6(tn)})
        fv = vals_at(z, r, tA, tB)
        rec["rows"][nm] = {"final": fv, "dm": r6(margin(fv) - margin(cA)), "loops": per,
                           "final_equals_loopK_readout": bool(torch.equal(ro[K][r], z[r]))}
    rec["identity"] = ident
    rec["forwards"] = X.n_fwd - f0
    rec["seconds"] = round(time.time() - t0, 3)
    return rec, ident


# ------------------------------------------------------------------ items
def load_cands(data_dir, tok, ds):
    p = os.path.join(data_dir, "cands_%s_%s.json" % (tok, ds))
    return json.load(open(p))["cands"], p


def prepare(data_dir, model, ds):
    import k17_public as KPUB
    spec = MODELS[model]
    cands, path = load_cands(data_dir, spec["tok"], ds)
    out, pos0 = [], 0
    for c in cands:
        it = KPUB.prep_item(c, spec["bos"])
        if it is None:
            pos0 += 1
            continue
        assert it["r"] == len(it["A"]) - 1 and len(it["A"]) == len(it["B"]), it["pair_id"]
        out.append({"key": it["pair_id"], "A": it["A"], "B": it["B"], "pos": sorted(it["src"]), "tA": it["tA"],
                    "tB": it["tB"], "cluster": c["cl"][0], "sa": c["cl"][0],
                    "sb": c["cl"][1] if len(c["cl"]) > 1 else c["cl"][0], "stratum": it["rel"],
                    "half": it.get("split"), "pos_len": (it.get("strata") or {}).get("pos_len")})
    return out, {"candidates": len(cands), "pos0_dropped": pos0, "cands_sha256": KF.sha256_file(path),
                 "cands_file": os.path.basename(path)}


def scan(X, items, ds, debug):
    stats = {"A_fail": 0, "B_fail": 0, "skipped_by_cap": 0}
    n_c, kept, scanned = {}, [], 0
    for it in items:
        if debug and len(kept) >= debug:
            break
        if ds == "D2" and n_c.get(it["cluster"], 0) >= D2_CAP:
            stats["skipped_by_cap"] += 1
            continue
        scanned += 1
        a = X.nohook(torch.tensor([it["A"]], device=X.dev), it["key"])[0, -1]
        if int(a.argmax()) != it["tA"]:
            stats["A_fail"] += 1
            continue
        b = X.nohook(torch.tensor([it["B"]], device=X.dev), it["key"])[0, -1]
        if int(b.argmax()) != it["tB"]:
            stats["B_fail"] += 1
            continue
        n_c[it["cluster"]] = n_c.get(it["cluster"], 0) + 1
        kept.append(it)
    stats.update({"scanned": scanned, "kept": len(kept)})
    return kept, stats


# ------------------------------------------------------------------ run
TREE = [None]


def prod_hashes(tree):
    fs = ["prod/steer/hooks.py", "prod/steer/interventions.py", "prod/steer/sites.py", "prod/steer/readouts.py",
          "prod/steer/adapters/__init__.py", "prod/steer/adapters/raven.py", "prod/models/raven.py",
          "prod/models/huginn.py", "prod/models/mcleish.py", "prod/models/__init__.py", "prod/config.py",
          "prod/common.py", "prod/tasks/data/model_revisions.json"]
    return {f: KF.md5_file(os.path.join(tree, f)) for f in fs}


def run(args):
    t_start = time.time()
    out = args.out
    os.makedirs(out, exist_ok=True)
    if KF.sha256_file(args.plan) != args.plan_sha256:
        raise SystemExit("plan sha256 mismatch")
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))
    TREE[0] = os.path.abspath(args.tree)
    if TREE[0] not in sys.path:
        sys.path.insert(0, TREE[0])
    spec = MODELS[args.model]
    dss = args.datasets.split(",")
    prepared = {ds: prepare(args.data, args.model, ds) for ds in dss}
    t_max = max(len(it["A"]) for ds in dss for it in prepared[ds][0]) + 1
    from prod import models as PM
    from prod.steer.adapters import bind
    import transformers
    import k17_public as KPUB
    ad = PM.get(spec["prod"])
    ad.load()
    ad.set_depth(K)
    b = bind(ad)
    b.set_init_state("fixed_table", t_max)
    X = Ctx(ad, b)
    from huggingface_hub.constants import HF_HUB_CACHE
    snap = os.path.join(HF_HUB_CACHE, "models--" + spec["repo"].replace("/", "--"),
                        "snapshots")
    mfile = None
    for rv in os.listdir(snap):
        p = os.path.join(snap, rv, "raven_modeling_minimal.py")
        if os.path.exists(p):
            mfile = KF.sha256_file(p)[:16]
    cfg = {"brief": "RA-19 item 3", "plan_sha256": args.plan_sha256,
           "code_sha256": KF.sha256_file(os.path.abspath(__file__)),
           "k17_freeze_md5": KF.md5_file(KF.__file__), "k17_public_md5": KF.md5_file(KPUB.__file__),
           "prod_md5": prod_hashes(TREE[0]), "model": spec["prod"], "repo": spec["repo"],
           "revision": getattr(ad.model.config, "_commit_hash", None), "modeling_sha16": mfile,
           "dtype": str(ad.model.dtype), "k": K, "batch": 1, "gap_e": GAP_E, "d2_cap": D2_CAP, "bos": spec["bos"],
           "x0_policy": "fixed_table", "t_max": t_max, "datasets": args.datasets, "debug": args.debug,
           "F_ans": X.F_ans, "F_src": X.F_src, "site_ans": X.s_ans.key(), "site_src": X.s_src.key(),
           "site_final_input": X.s_fin.key(), "n_freeze_sites_recorded": len(X.frz_sites),
           "source_report": b.source_report(), "cands": {ds: prepared[ds][1] for ds in dss},
           "torch": torch.__version__, "transformers": transformers.__version__,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    cfg_path = os.path.join(out, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        for k_ in ("plan_sha256", "code_sha256", "prod_md5", "revision", "debug", "t_max", "cands"):
            if old.get(k_) != cfg.get(k_):
                raise SystemExit("resume with a changed config field %s" % k_)
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=1, sort_keys=True)
    print("[frzr] config sha256 %s" % KF.sha256_file(cfg_path), flush=True)
    prog_path = os.path.join(out, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"sessions": []}
    sess = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "load_seconds": round(time.time() - t_start, 1)}
    prog["sessions"].append(sess)

    def save_prog():
        sess["seconds"] = round(time.time() - t_start, 1)
        sess["forwards"] = X.n_fwd
        sess["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if torch.cuda.is_available() else None
        json.dump(prog, open(prog_path + ".tmp", "w"), indent=1)
        os.replace(prog_path + ".tmp", prog_path)
    for ds in dss:
        sel_path = os.path.join(out, "selection_%s.json" % ds)
        if os.path.exists(sel_path):
            sel = json.load(open(sel_path))
        else:
            ts = time.time()
            f0 = X.n_fwd
            items, info = prepared[ds]
            kept, stats = scan(X, items, ds, args.debug)
            info = dict(info, **stats)
            info["rule"] = ("every clean-passing candidate in file order" +
                            (", cap %d per A-side operand cluster" % D2_CAP if ds == "D2" else ""))
            info["scan_forwards"] = X.n_fwd - f0
            info["scan_seconds"] = round(time.time() - ts, 1)
            sel = {"dataset": ds, "info": info, "items": kept}
            json.dump(sel, open(sel_path + ".tmp", "w"))
            os.replace(sel_path + ".tmp", sel_path)
            print("[frzr] %s selection: %s" % (ds, json.dumps(info)), flush=True)
        rec_path = os.path.join(out, "items_%s.jsonl" % ds)
        done = set()
        if os.path.exists(rec_path):
            for ln in open(rec_path):
                if ln.strip():
                    try:
                        done.add(json.loads(ln)["key"])
                    except ValueError:
                        pass
        ta = time.time()
        n_new = 0
        with open(rec_path, "a") as fh:
            for it in sel["items"]:
                if STOP["flag"]:
                    break
                if it["key"] in done:
                    continue
                rec, ident = process_item(X, it["A"], it["B"], it["pos"], it["tA"], it["tB"], it["key"])
                for k_ in ("cluster", "sa", "sb", "stratum", "half", "pos_len"):
                    rec[k_] = it.get(k_)
                rec["dataset"] = ds
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_new += 1
                if not all(ident[k_] for k_ in ident):
                    print("[frzr] %s item %s identity flags %s" % (ds, it["key"], json.dumps(ident)), flush=True)
                if n_new % 20 == 1:
                    print("[frzr] %s %d/%d (%.2fs/item) %.0fs" % (ds, len(done) + n_new, len(sel["items"]),
                                                                  (time.time() - ta) / n_new, time.time() - t_start),
                          flush=True)
                    save_prog()
        save_prog()
        print("[frzr] %s done %d new items %.0fs" % (ds, n_new, time.time() - ta), flush=True)
        if STOP["flag"]:
            print("[frzr] SIGTERM: stopped after the current item", flush=True)
            return
    sess["finished"] = True
    save_prog()
    KF.write_hashes(out)
    print("[frzr] done %.0fs, %d forwards" % (time.time() - t_start, X.n_fwd), flush=True)


# ------------------------------------------------------------------ summary (CPU)
def e_of(row, j=None):
    def f(r):
        if r["D"] is None or r["D"] < GAP_E:
            return None
        rw = r["rows"][row]
        dm = rw["dm"] if j is None else rw["loops"][j]["dm"]
        return dm / r["D"]
    return f


def summarize(args):
    KO = KF.stats_mod()
    out = args.out
    cfg = json.load(open(os.path.join(out, "config.json")))
    hs = json.load(open(os.path.join(out, "hashes.json")))
    hash_ok = all(KF.sha256_file(os.path.join(out, f)) == h for f, h in hs.items())
    prog = json.load(open(os.path.join(out, "progress.json")))
    data, sel = {}, {}
    for ds in DATASETS:
        p = os.path.join(out, "items_%s.jsonl" % ds)
        if os.path.exists(p):
            data[ds] = [json.loads(x) for x in open(p) if x.strip()]
            sel[ds] = json.load(open(os.path.join(out, "selection_%s.json" % ds)))["info"]
    res = {"plan_sha256": cfg["plan_sha256"], "model": cfg["model"], "config_sha256": KF.sha256_file(
        os.path.join(out, "config.json")), "cells": {}, "selection": sel}
    ident_all = {}
    for name, ds in (("facts", "D1"), ("addition", "D2")):
        if ds not in data:
            continue
        recs = data[ds]
        dy = ds == "D1"
        c = {"n_items": len(recs), "n_clusters": len(set(r["cluster"] for r in recs)),
             "descriptive_lt_40": len(recs) < 40}
        c["strata"] = {s: sum(1 for r in recs if r["stratum"] == s) for s in sorted(set(r["stratum"] for r in recs))}
        idk = sorted(recs[0]["identity"]) if recs else []
        c["identity_counts"] = {k_: sum(1 for r in recs if r["identity"][k_]) for k_ in idk}
        c["e0"] = KF.cell(KO, recs, e_of("E0"), dyadic=dy)
        c["e_attn"] = KF.cell(KO, recs, e_of("ans_frz_mlp"), dyadic=dy)
        c["e_mlp"] = KF.cell(KO, recs, e_of("ans_frz_attn"), dyadic=dy)
        for nm in ("pc_final", "sz", "src", "src_frz_attn"):
            c[nm + "_e"] = KF.cell(KO, recs, e_of(nm))
        okD = [r for r in recs if r["D"] is not None and r["D"] >= GAP_E]
        mD = sum(r["D"] for r in okD) / max(1, len(okD))
        c["ratio_of_means"] = {nm: round(sum(r["rows"][nm]["dm"] for r in okD) / max(1, len(okD)) / mD, 4)
                               if okD else None for nm in ANS_ROWS}
        c["mean_D"] = round(mD, 4)
        c["n_D_ge_2"] = len(okD)
        c["g5"] = {nm: [{"loop": j, "norm_ratio": KF.cell(KO, recs, KF.g5_of(nm, j - 1, "norm_ratio")),
                         "cos": KF.cell(KO, recs, KF.g5_of(nm, j - 1, "cos")),
                         "e": KF.cell(KO, recs, e_of(nm, j - 1))} for j in G5_REPORT_LOOPS] for nm in ANS_ROWS}
        ea = c["e_attn"]
        lo, hi = (ea["dyadic"]["ci90"] if dy else ea["ci90"]) if ea else (None, None)
        ok_id = all(c["identity_counts"][k_] == len(recs) for k_ in idk if k_ != "source_edit_reaches_r")
        verdict = None if ea is None else ("R1 falsified" if lo >= 0.5 else ("R1 supported" if hi <= 0.15 else "mixed"))
        c["verdict"] = verdict if ok_id else "no verdict (identity failed)"
        c["bounds_used"] = "dyadic 90%" if dy else "A-side cluster 90%"
        res["cells"][name] = c
        for k_ in idk:
            ident_all[k_] = ident_all.get(k_, True) and c["identity_counts"][k_] == len(recs)
    allrecs = [r for ds in data for r in data[ds]]
    # Summary fix after the Huginn run (no row changes): the freeze rows freeze the coda writes at r in the forward
    # (freeze_spec covers coda calls), while the per-loop pure read-out runs the coda without hooks, so their
    # loop-32 read-out cannot equal their final logits. The read-out identity is therefore scored on clean A, clean B
    # and E0; the freeze-row equality is reported beside it.
    readout_eq = all(r["rows"]["E0"]["final_equals_loopK_readout"] for r in allrecs)
    frz_ro_eq = {nm: sum(1 for r in allrecs if r["rows"][nm]["final_equals_loopK_readout"])
                 for nm in ("ans_frz_mlp", "ans_frz_attn")}
    gpu_s = sum(s.get("seconds", 0) for s in prog["sessions"])
    F_, A_ = res["cells"].get("facts", {}), res["cells"].get("addition", {})

    def bnd(c, ename, side):
        if not c or not c.get(ename):
            return None
        ci = c[ename]["dyadic"]["ci90"] if "dyadic" in c[ename] else c[ename]["ci90"]
        return ci[0] if side == "lo" else ci[1]
    ck = {"model": cfg["model"],
          "zero_edit_bit_identical": ident_all.get("hook_identity"),
          "freeze_no_edit_bit_identical": ident_all.get("freeze_no_edit_identity"),
          "source_attn_frozen_e_zero_bit_exact": ident_all.get("source_attn_frozen_identity"),
          "source_edit_reaches_r_all": ident_all.get("source_edit_reaches_r"),
          "pc_exact_all": ident_all.get("pc_exact"), "sz_exact_all": ident_all.get("sz_exact"),
          "ans_loop1_state_is_donor_all": ident_all.get("ans_loop1_state_is_donor"),
          "readout_identity_all": bool(ident_all.get("readout_identity") and readout_eq),
          "freeze_rows_loopK_readout_equals_final_n": frz_ro_eq, "n_items_all": len(allrecs),
          "n_items_facts": F_.get("n_items"), "n_clusters_facts": F_.get("n_clusters"),
          "n_items_addition": A_.get("n_items"), "n_clusters_addition": A_.get("n_clusters"),
          "verdict_facts": F_.get("verdict"), "verdict_addition": A_.get("verdict"),
          "e_attn_facts_lower_bound_ge_0.5": None if bnd(F_, "e_attn", "lo") is None else bnd(F_, "e_attn", "lo") >= 0.5,
          "e_attn_facts_upper_bound_le_0.15": None if bnd(F_, "e_attn", "hi") is None else bnd(F_, "e_attn", "hi") <= 0.15,
          "e_mlp_facts_upper_bound_le_0.15": None if bnd(F_, "e_mlp", "hi") is None else bnd(F_, "e_mlp", "hi") <= 0.15,
          "gpu_minutes": round(gpu_s / 60.0, 2), "outputs_hashed": bool(hash_ok), "plan_sha256": cfg["plan_sha256"]}
    res["checks"] = ck
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    json.dump(ck, open(os.path.join(out, "checks.json"), "w"), indent=1)
    L = ["# K17 FREEZE RAVEN summary (RA-19 item 3, X4a branch freeze, %s, K = %d)" % (cfg["model"], K), "",
         "Plan sha256 %s. Mean of per-item e = dm / D (items with D >= %.1f) [90%% percentile cluster bootstrap, "
         "%d resamples, seed 0] (items with e defined / clusters)." % (cfg["plan_sha256"], GAP_E, NB), "",
         "| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    for nm, c in res["cells"].items():
        dy = nm == "facts"
        L.append("| %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
            nm, c["n_items"], c["n_clusters"], KF.fmt(c["e0"], dy), KF.fmt(c["e_attn"], dy), KF.fmt(c["e_mlp"], dy),
            KF.fmt(c["pc_final_e"]), KF.fmt(c["src_e"]), c["verdict"]))
    L += ["", "## Ratio of means, strata, identities", ""]
    for nm, c in res["cells"].items():
        L.append("- %s: %s; mean D %.3f (%d items with D >= 2); strata %s; identity counts %s" % (
            nm, json.dumps(c["ratio_of_means"]), c["mean_D"], c["n_D_ge_2"], json.dumps(c["strata"]),
            json.dumps(c["identity_counts"])))
    L += ["", "## G5 split at selected loop ends", ""]
    for nm, c in res["cells"].items():
        L += ["### %s" % nm, "", "| row | loop | norm ratio | cos | e_j |", "|---|---|---|---|---|"]
        for row, per in c["g5"].items():
            for x in per:
                L.append("| %s | %d | %s | %s | %s |" % (row, x["loop"], KF.fmt(x["norm_ratio"]), KF.fmt(x["cos"]),
                                                         KF.fmt(x["e"])))
        L.append("")
    L += ["## Checks", "", "```", json.dumps(ck, indent=1), "```"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(ck, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--tree")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "out", "k17_public", "data"))
    ap.add_argument("--out")
    ap.add_argument("--plan")
    ap.add_argument("--plan_sha256")
    ap.add_argument("--datasets", default="D1,D2")
    ap.add_argument("--debug", type=int, default=0)
    a = ap.parse_args()
    if a.run:
        run(a)
    elif a.summarize:
        summarize(a)
    else:
        ap.error("--run or --summarize")


if __name__ == "__main__":
    main()
