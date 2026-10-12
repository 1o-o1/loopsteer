"""K17 FREEZE (brief RA-3, matrix card X4a, check C13): branch-freeze 2x2 on Ouro-1.4B base.
Plan: loop-signal/notes/k17_freeze_plan.md (sha256 frozen before any scored row; --plan_sha256 must match).

Hooks come from the production package only (loopedcot prod/steer: EagerOuroAdapter, OuroBinding,
HookRegistry, interventions.from_spec / freeze_spec, readouts.Recorder); this file builds specs, payloads and
endpoints and adds no hook. Items: k15_data / k16_path_ouro candidate builders (imported, never modified) and
the D1 pooled_all identity list of the knockout run (k17_ko240), frozen in items_d1.json by --make_items.

    python k17_freeze.py --make_items OUT/inputs --ko_json K17KO.json --pilot_json K16P.json
    python k17_freeze.py --selftest --tree LOOPEDCOT_ROOT
    python k17_freeze.py --run --out OUT --tree TREE --plan PLAN.md --plan_sha256 SHA [--debug N] [--datasets ...]
    python k17_freeze.py --summarize --out OUT --plan_sha256 SHA

Cluster copy (RA-23 item 1): --model ouro_1_4b_think | ouro_2_6b_think runs the same rows and identities on an Ouro
Thinking checkpoint (D1: every d1_candidates pair, then that model's own clean filter; the 267-item list is Ouro-1.4B
base's clean set and stays the base model's D1); --full_pool lifts the D4/D5 quotas. Without these flags the code path
is the RA-3 run's.
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
sys.path.insert(0, HERE)

import torch  # noqa: E402

K = 4
GAP_E = 2.0
NB = 10000
D2_CAP = 4
D4_QUOTA = {("dev", "T1"): 40, ("test", "T1"): 40, ("dev", "T2"): 40, ("test", "T2"): 40}
D5_QUOTA = {("dev", "T2"): 80, ("test", "T2"): 80}
DATASETS = ("D1", "D2", "D4", "D5")
ANS_ROWS = ("E0", "ans_frz_mlp", "ans_frz_attn")
STOP = {"flag": False}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


# ------------------------------------------------------------------ D1 identity list (CPU)
def make_items(out_dir, ko_json, pilot_json):
    ko = json.load(open(ko_json))
    pl = json.load(open(pilot_json))
    p80 = pl["datasets"]["D1"]["items"]
    new = ko["arms"]["new"]["items"]
    rows = []
    for src, its in (("pilot80", p80), ("new", new)):
        for it in its:
            rows.append({"src": src, "rel": it["rel"], "a": it["a"], "b": it["b"], "half": it["half"],
                         "pos": list(it["pos"]), "T": it["T"], "tA": it["tA"], "tB": it["tB"],
                         "cluster": it["cluster"], "stored_clean_final": it["clean"][-1][:2],
                         "stored_cleanB_final": it["cleanB"][-1][:2]})
    keys = [(r["rel"], r["a"], r["b"], r["half"]) for r in rows]
    assert len(set(keys)) == len(keys), "duplicate D1 item"
    os.makedirs(out_dir, exist_ok=True)
    meta = {"n_items": len(rows), "n_clusters": len(set(r["cluster"] for r in rows)),
            "n_subjects": len(set("%s:%s" % (r["rel"], r["a"]) for r in rows) |
                              set("%s:%s" % (r["rel"], r["b"]) for r in rows)),
            "source_files_md5": {"k17ko_ouro.json": md5_file(ko_json), "k16p_ouro.json": md5_file(pilot_json)},
            "rule": "k17_ko240 pooled_all = pilot80_stored + every passing new item, in that order"}
    path = os.path.join(out_dir, "items_d1.json")
    json.dump({"meta": meta, "items": rows}, open(path, "w"), indent=0)
    print(json.dumps(meta))
    print("items_d1.json sha256 %s" % sha256_file(path))


# ------------------------------------------------------------------ production imports
def prod_imports(tree):
    if tree not in sys.path:
        sys.path.insert(0, tree)
    from prod.steer import interventions as I
    from prod.steer.hooks import HookRegistry, RowInfo
    from prod.steer.readouts import Recorder
    from prod.steer.sites import Component, Positions, Site
    return I, HookRegistry, RowInfo, Recorder, Component, Positions, Site


class Ctx(object):
    """Production objects around one bound model."""

    def __init__(self, binding, tree):
        (self.I, self.Reg, self.RowInfo, self.Recorder, self.C, self.Positions,
         self.Site) = prod_imports(tree)
        self.b = binding
        self.g = binding.geometry(K)
        self.W = binding.model_module().lm_head.weight
        self.dev = self.W.device
        I = self.I
        self.s_ans = I.expand("loop1.loop_end", {"kind": "once"}, self.g)[0]
        self.s_src = I.expand("loop1.L0.resid_post", {"kind": "once"}, self.g)[0]
        self.s_le = [I.expand("loop%d.loop_end" % j, {"kind": "once"}, self.g)[0] for j in range(1, K + 1)]
        self.F_ans = self.g.first_changed(self.s_ans)
        self.F_src = self.g.first_changed(self.s_src)
        self.frz_sites = [s for s in self.g.all_sites(["attn_out", "mlp_out"]) if s.stage == "loop"]
        self.n_fwd = 0

    def run(self, ids, row, ivs=(), recs=()):
        reg = self.Reg(self.b, K)
        for r in recs:
            reg.add_recorder(r)
        for iv in ivs:
            reg.add_intervention(iv)
        reg.install()
        try:
            with reg.batch([row], [0]):
                z = self.b.forward(ids, K, [row.key])
            out = reg.take_records()
        finally:
            reg.remove()
        self.n_fwd += 1
        return z, out

    def nohook(self, ids):
        m = self.b.model_module()
        with torch.no_grad():
            o = m(input_ids=ids, cache_position=torch.arange(ids.shape[1], device=ids.device), use_cache=False)
        self.n_fwd += 1
        return o.logits.float()

    def readouts(self, H):
        """H {j: [T, d] float32} -> {j: [T, V] float32}: exit read-out lm_head(LOOP_END_j) on the full [1, T, d]
        tensor in the model dtype (same GEMM shape as the model's own final logits)."""
        return {j: self.b.readout(j, H[j].to(device=self.dev, dtype=self.W.dtype)[None], "exit")[0]
                for j in H}


def vals_at(z, r, tA, tB):
    v = z[r]
    return [r6(v[tA]), r6(v[tB]), r6(torch.logsumexp(v, -1)), int(v.argmax())]


def margin(v):
    return v[1] - v[0]


@torch.no_grad()
def process_item(X, A, B, src, tA, tB, key):
    """All rows of plan section 6 for one item. Returns (record dict, identity flags)."""
    I, C, P = X.I, X.C, X.Positions
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
    # 1. no hook
    z_nh = X.nohook(ids_A)[0]
    # 2. clean A with recorders
    frz_rec = X.Recorder(id="frz", sites=tuple(X.frz_sites), positions=at_r, what="state", when="post")
    zA, rA = X.run(ids_A, row(), recs=[le_rec("le"), frz_rec])
    zA = zA[0]
    HA = H_of(rA, "le")
    roA = X.readouts(HA)
    # 3. clean B with recorders
    srcB_rec = X.Recorder(id="srcB", sites=(X.s_src,), positions=P("span", span="source"), what="state",
                          when="post")
    zB, rB = X.run(ids_B, row(), recs=[le_rec("le"), srcB_rec])
    zB = zB[0]
    HB = H_of(rB, "le")
    roB = X.readouts(HB)
    srcB = rB[(key, "srcB", X.s_src.key())]

    # interventions
    def patch(iid, site, pos, a=1.0):
        return I.from_spec({"id": iid, "op": "patch", "site": site.key(), "a": a, "target": {"kind": "donor"},
                            "positions": pos}, X.g)

    pos_r = {"kind": "abs", "abs": [r]}
    pos_src = {"kind": "span", "span": "source"}
    ans = patch("ans", X.s_ans, pos_r)
    ans0 = patch("ans", X.s_ans, pos_r, a=0.0)
    srcp = patch("src", X.s_src, pos_src)
    pcf = patch("pcf", X.s_le[K - 1], pos_r)
    sz = patch("sz", X.s_le[K - 1], pos_src)

    def freeze(comp, F):
        iv = I.from_spec(dict(I.freeze_spec(comp, pos_r, F, X.g), id="frz_" + comp), X.g)
        pay = {s.key(): rA[(key, "frz", s.key())] for s in iv.sites}
        return iv, pay

    fm, pay_fm = freeze("mlp_out", X.F_ans)
    fa, pay_fa = freeze("attn_out", X.F_ans)
    fa_s, pay_fa_s = freeze("attn_out", X.F_src)
    pay = {"ans": {X.s_ans.key(): HB[1][r:r + 1]},
           "src": {X.s_src.key(): srcB},
           "pcf": {X.s_le[K - 1].key(): HB[K][r:r + 1]},
           "sz": {X.s_le[K - 1].key(): HB[K][src]}}

    def edited(ivs, extra, with_le):
        p = {iv.id: pay[iv.id] for iv in ivs if iv.id in pay}
        p.update(extra)
        z, rr = X.run(ids_A, row(p), ivs=ivs, recs=[le_rec("le")] if with_le else [])
        return z[0], (H_of(rr, "le") if with_le else None)

    z_zero, _ = edited([ans0], {}, False)
    z_E0, H_E0 = edited([ans], {}, True)
    z_am, H_am = edited([ans, fm], {"frz_mlp_out": pay_fm}, True)
    z_aa, H_aa = edited([ans, fa], {"frz_attn_out": pay_fa}, True)
    z_fm, _ = edited([fm], {"frz_mlp_out": pay_fm}, False)
    z_fa, _ = edited([fa], {"frz_attn_out": pay_fa}, False)
    z_src, _ = edited([srcp], {}, False)
    z_sf, H_sf = edited([srcp, fa_s], {"frz_attn_out": pay_fa_s}, True)
    z_pc, _ = edited([pcf], {}, False)
    z_sz, _ = edited([sz], {}, False)

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
    rec["D"] = r6(D)
    for nm, z in (("zero", z_zero), ("frz_mlp", z_fm), ("frz_attn", z_fa), ("src", z_src),
                  ("src_frz_attn", z_sf), ("pc_final", z_pc), ("sz", z_sz)):
        v = vals_at(z, r, tA, tB)
        rec["rows"][nm] = {"final": v, "dm": r6(margin(v) - margin(cA))}
    TB = {j: HB[j][r] - HA[j][r] for j in range(1, K + 1)}
    for nm, z, H in (("E0", z_E0, H_E0), ("ans_frz_mlp", z_am, H_am), ("ans_frz_attn", z_aa, H_aa)):
        ro = X.readouts(H)
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
                           "final_equals_loop4_readout": bool(torch.equal(ro[K][r], z[r]))}
    rec["identity"] = ident
    rec["forwards"] = X.n_fwd - f0
    rec["seconds"] = round(time.time() - t0, 3)
    return rec, ident


# ------------------------------------------------------------------ item builders (GPU host)
def build_d1(enc, items_path):
    import k15_data as KD
    rows = json.load(open(items_path))["items"]
    cands = KD.d1_candidates(enc)
    by = {}
    for p in cands:
        by.setdefault((p["rel"], p["a"], p["b"], p["half"]), []).append(p)
    out, bad = [], []
    for rw in rows:
        k = (rw["rel"], rw["a"], rw["b"], rw["half"])
        ps = by.get(k, [])
        if len(ps) != 1:
            bad.append({"key": list(k), "n_match": len(ps)})
            continue
        p = ps[0]
        ok = sorted(p["pos"]) == rw["pos"] and len(p["A"]) == rw["T"] and p["tA"] == rw["tA"] and p["tB"] == rw["tB"]
        if not ok:
            bad.append({"key": list(k), "identity_mismatch": True})
            continue
        out.append({"key": "%s|%s|%s|%s" % k, "A": p["A"], "B": p["B"], "pos": sorted(p["pos"]), "tA": p["tA"],
                    "tB": p["tB"], "cluster": rw["cluster"], "sa": "%s:%s" % (rw["rel"], rw["a"]),
                    "sb": "%s:%s" % (rw["rel"], rw["b"]), "stratum": rw["rel"], "half": rw["half"],
                    "src": rw["src"], "stored_clean_final": rw["stored_clean_final"],
                    "stored_cleanB_final": rw["stored_cleanB_final"]})
    return out, {"n_rows": len(rows), "n_matched": len(out), "bad": bad, "candidates": len(cands)}


def build_d1_pool(enc, items_path):
    """Cluster D1 pool of a non-base model (RA-20 item 9, RA-23 'full pools'): every k15_data.d1_candidates pair, in
    candidate order, before any clean filter (items_d1.json holds the pairs Ouro-1.4B base solves, so it is not used
    as the pool). Fields as build_d1; cluster = rel:a (the k16_path_ouro / k17_ko240 D1 cluster); src = the items_d1
    src for a pair in that list, else "pool"."""
    import k15_data as KD
    src = {(r["rel"], r["a"], r["b"], r["half"]): r["src"] for r in json.load(open(items_path))["items"]}
    cands = KD.d1_candidates(enc)
    out = []
    for p in cands:
        k = (p["rel"], p["a"], p["b"], p["half"])
        out.append({"key": "%s|%s|%s|%s" % k, "A": p["A"], "B": p["B"], "pos": sorted(p["pos"]), "tA": p["tA"],
                    "tB": p["tB"], "cluster": "%s:%s" % (p["rel"], p["a"]), "sa": "%s:%s" % (p["rel"], p["a"]),
                    "sb": "%s:%s" % (p["rel"], p["b"]), "stratum": p["rel"], "half": p["half"],
                    "src": src.get(k, "pool")})
    keys = [it["key"] for it in out]
    assert len(set(keys)) == len(keys), "duplicate D1 candidate key"
    return out, {"candidates": len(cands), "n_in_items_d1": sum(1 for it in out if it["src"] != "pool"),
                 "rule": "every d1_candidates pair, then the clean filter of this model"}


def make_know(X, stats):
    def know(p):
        a = X.nohook(torch.tensor([p["A"]], device=X.dev))[0, -1]
        if int(a.argmax()) != p["tA"]:
            stats["A_fail"] += 1
            return False
        b = X.nohook(torch.tensor([p["B"]], device=X.dev))[0, -1]
        if int(b.argmax()) != p["tB"]:
            stats["B_fail"] += 1
            return False
        return True
    return know


def select_d2(cands, know, cap=D2_CAP):
    """Every candidate in order; skipped (no forward) when its cluster already holds `cap` kept items."""
    n_c, kept, scanned = {}, [], 0
    for p in cands:
        c = p["cluster"]
        if n_c.get(c, 0) >= cap:
            continue
        scanned += 1
        if know(p):
            n_c[c] = n_c.get(c, 0) + 1
            kept.append(p)
    return kept, scanned


def build_scan(X, ds, debug, full_pool=False):
    import k15_data as KD
    import k16_path_ouro as KP
    stats = {"A_fail": 0, "B_fail": 0}
    know = make_know(X, stats)
    if ds == "D2":
        cands = KD.d2_candidates(X.enc)
        for p in cands:
            a1, a2_ = [int(v) for v in p["a"].split(" + ")]
            p["cluster"] = "x=%d" % (a1 if p["rel"] == "op1" else a2_)
        pool = cands[:12] if debug else cands
        kept, scanned = select_d2(pool, know)
        info = {"candidates": len(cands), "rule": "all clean-passing, cap %d per A-side edited operand value" % D2_CAP}
    else:
        cands = KP.ext_candidates(ds, "ouro")
        quota = dict(D4_QUOTA if ds == "D4" else D5_QUOTA)
        if full_pool:                    # cluster: every candidate scanned (the cap of 2 per cluster side stays)
            quota = {k: 10 ** 9 for k in quota}
        if debug:
            quota = {k: 1 for k in quota}
        kept, scanned = KD.select(cands, know, quota)
        for p in kept:
            p["cluster"] = p["cluster_id"]
        info = {"candidates": len(cands), "quota": {"%s|%s" % k: v for k, v in quota.items()},
                "rule": "k15_data.select cap 2 per cluster side per (half, kind)",
                "data_md5": md5_file(os.path.join(KP.DATA_DIR, "%s_ouro.jsonl" % ds.lower()))}
    info.update({"scanned": scanned, "kept": len(kept), "know_stats": stats})
    items = []
    for p in kept:
        key = ("%s|%s|%s" % (p["rel"], p["a"], p["b"])) if ds == "D2" else str(p["id"])
        items.append({"key": key, "A": p["A"], "B": p["B"], "pos": sorted(p["pos"]), "tA": p["tA"], "tB": p["tB"],
                      "cluster": p["cluster"], "stratum": p["rel"], "half": p["half"]})
    return items, info


# ------------------------------------------------------------------ run
MODELS = ("ouro_1_4b_base", "ouro_1_4b_think", "ouro_2_6b_think")   # prod registry names (prod/models/ouro.py)


def load_model(tree, mem_fraction, name="ouro_1_4b_base"):
    if tree not in sys.path:
        sys.path.insert(0, tree)
    from prod.steer.adapters import bind
    from prod.steer.adapters.ouro import EagerOuroAdapter
    ad = EagerOuroAdapter(name, mem_fraction=mem_fraction).load()
    ad.set_depth(K)
    b = bind(ad)
    return ad, b


def prod_hashes(tree):
    fs = ["prod/steer/hooks.py", "prod/steer/interventions.py", "prod/steer/sites.py", "prod/steer/readouts.py",
          "prod/steer/adapters/__init__.py", "prod/steer/adapters/ouro.py", "prod/models/ouro.py", "prod/config.py"]
    return {f: md5_file(os.path.join(tree, f)) for f in fs}


def run(args):
    t_start = time.time()
    out = args.out
    os.makedirs(out, exist_ok=True)
    if sha256_file(args.plan) != args.plan_sha256:
        raise SystemExit("plan sha256 mismatch")
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))
    ad, b = load_model(args.tree, args.mem_fraction, args.model)
    X = Ctx(b, args.tree)
    tok = ad.tok
    X.enc = lambda t, ans=False: tok(t, add_special_tokens=False)["input_ids"]
    if b.attn_implementation() != "eager":
        raise SystemExit("not eager")
    import k15_data as KD
    import k16_path_ouro as KP
    import transformers
    cfg_path = os.path.join(out, "config.json")
    cfg = {"brief": "RA-3", "plan_sha256": args.plan_sha256, "code_sha256": sha256_file(os.path.abspath(__file__)),
           "prod_md5": prod_hashes(args.tree), "k15_data_md5": md5_file(KD.__file__),
           "k16_path_ouro_md5": md5_file(KP.__file__), "items_d1_sha256": sha256_file(args.items_d1),
           "model": ad.repo, "revision": getattr(ad.model.config, "_commit_hash", None),
           "dtype": str(ad.model.dtype), "attn_implementation": b.attn_implementation(), "k": K, "batch": 1,
           "gap_e": GAP_E, "d2_cap": D2_CAP, "d4_quota": {"%s|%s" % k: v for k, v in D4_QUOTA.items()},
           "d5_quota": {"%s|%s" % k: v for k, v in D5_QUOTA.items()}, "datasets": args.datasets,
           "debug": args.debug, "F_ans": X.F_ans, "F_src": X.F_src, "torch": torch.__version__,
           "transformers": transformers.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "n_freeze_sites": len(X.frz_sites)}
    if args.model != MODELS[0] or args.full_pool:     # cluster options (absent from the RA-3 base config)
        cfg.update({"model_key": args.model, "full_pool": bool(args.full_pool),
                    "d1_rule": "every d1_candidates pair, then the clean filter of this model" if args.model != MODELS[0]
                    else "items_d1 identity list (RA-3)"})
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        for k_ in ("plan_sha256", "code_sha256", "prod_md5", "items_d1_sha256", "revision", "debug"):
            if old.get(k_) != cfg.get(k_):
                raise SystemExit("resume with a changed config field %s" % k_)
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=1, sort_keys=True)
    print("[freeze] config sha256 %s" % sha256_file(cfg_path), flush=True)
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
    for ds in args.datasets.split(","):
        sel_path = os.path.join(out, "selection_%s.json" % ds)
        if os.path.exists(sel_path):
            sel = json.load(open(sel_path))
        else:
            ts = time.time()
            f0 = X.n_fwd
            if ds == "D1" and args.model == MODELS[0]:
                items, info = build_d1(X.enc, args.items_d1)        # RA-3: the 267-item list (base clean by construction)
                if args.debug:
                    items = items[:args.debug]
            elif ds == "D1":                     # any other model: the full candidate pool, its own clean filter
                items, info = build_d1_pool(X.enc, args.items_d1)
                st = {"A_fail": 0, "B_fail": 0}
                know = make_know(X, st)
                pool = items[:3 * args.debug] if args.debug else items
                items = [it for it in pool if know(it)]
                info.update({"scanned": len(pool), "kept": len(items), "know_stats": st})
                if args.debug:
                    items = items[:args.debug]
            else:
                items, info = build_scan(X, ds, args.debug, args.full_pool)
                if args.debug:
                    items = items[:args.debug]
            info["scan_forwards"] = X.n_fwd - f0
            info["scan_seconds"] = round(time.time() - ts, 1)
            sel = {"dataset": ds, "info": info, "items": items}
            json.dump(sel, open(sel_path + ".tmp", "w"))
            os.replace(sel_path + ".tmp", sel_path)
            print("[freeze] %s selection: %s" % (ds, json.dumps({k_: v for k_, v in info.items() if k_ != "bad"})),
                  flush=True)
        rec_path = os.path.join(out, "items_%s.jsonl" % ds)
        done = set()
        if os.path.exists(rec_path):
            for ln in open(rec_path):
                if ln.strip():
                    done.add(json.loads(ln)["key"])
        ta = time.time()
        n_new = 0
        with open(rec_path, "a") as fh:
            for it in sel["items"]:
                if STOP["flag"]:
                    break
                if it["key"] in done:
                    continue
                rec, ident = process_item(X, it["A"], it["B"], it["pos"], it["tA"], it["tB"], it["key"])
                for k_ in ("cluster", "stratum", "half", "sa", "sb", "src", "stored_clean_final", "stored_cleanB_final"):
                    if k_ in it:
                        rec[k_] = it[k_]
                rec["dataset"] = ds
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_new += 1
                if not all(ident[k_] for k_ in ident):
                    print("[freeze] %s item %s identity flags %s" % (ds, it["key"], json.dumps(ident)), flush=True)
                if n_new % 20 == 1:
                    print("[freeze] %s %d/%d (%.2fs/item) %.0fs" % (ds, len(done) + n_new, len(sel["items"]),
                                                                    (time.time() - ta) / n_new, time.time() - t_start),
                          flush=True)
                    save_prog()
        save_prog()
        print("[freeze] %s done %d new items %.0fs" % (ds, n_new, time.time() - ta), flush=True)
        if STOP["flag"]:
            print("[freeze] SIGTERM: stopped after the current item", flush=True)
            return
    sess["finished"] = True
    save_prog()
    write_hashes(out)
    print("[freeze] done %.0fs, %d forwards" % (time.time() - t_start, X.n_fwd), flush=True)


def write_hashes(out):
    hs = {}
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if os.path.isfile(p) and (f.startswith("items_") or f.startswith("selection_") or f == "config.json"):
            hs[f] = sha256_file(p)
    json.dump(hs, open(os.path.join(out, "hashes.json"), "w"), indent=1, sort_keys=True)


# ------------------------------------------------------------------ statistics (CPU)
def stats_mod():
    import k17_ko240 as KO
    return KO


def cell(KO, recs, fn, cl_key="cluster", dyadic=False):
    xs = [fn(r) for r in recs]
    cl = [r[cl_key] for r in recs]
    b = KO.boot_dist(xs, cl, NB, 0)
    if b is None:
        return None
    out = {"mean": round(b[0], 4), "ci90": KO.q(b[1], [0.05, 0.95]), "ci95": KO.q(b[1], [0.025, 0.975]),
           "n_items": len(recs), "n_defined": b[2], "n_clusters": b[3]}
    if dyadic:
        keep = [i for i, x in enumerate(xs) if x is not None]
        tw = KO.tost_twoway([xs[i] for i in keep], [recs[i]["sa"] for i in keep], [recs[i]["sb"] for i in keep])
        out["dyadic"] = {"ci90": tw["ci90"], "ci95": tw["ci95"], "n_subjects": tw["n_subjects"]}
    return out


def e_of(row, j=None):
    def f(r):
        if r["D"] is None or r["D"] < GAP_E:
            return None
        rw = r["rows"][row]
        dm = rw["dm"] if j is None else rw["loops"][j]["dm"]
        return dm / r["D"]
    return f


def g5_of(row, j, what):
    def f(r):
        return r["rows"][row]["loops"][j][what]
    return f


def summarize(args):
    KO = stats_mod()
    out = args.out
    cfg = json.load(open(os.path.join(out, "config.json")))
    hs = json.load(open(os.path.join(out, "hashes.json")))
    hash_ok = all(sha256_file(os.path.join(out, f)) == h for f, h in hs.items())
    prog = json.load(open(os.path.join(out, "progress.json")))
    data = {}
    sel = {}
    for ds in DATASETS:
        p = os.path.join(out, "items_%s.jsonl" % ds)
        if os.path.exists(p):
            data[ds] = [json.loads(x) for x in open(p) if x.strip()]
            sel[ds] = json.load(open(os.path.join(out, "selection_%s.json" % ds)))["info"]
    cells_def = [("facts", "D1", None), ("addition", "D2", None), ("code_reads", "D4", "T1"), ("code_sums", "D4", "T2"),
                 ("code_pooled", "D4", None), ("twostep", "D5", None)]
    res = {"plan_sha256": cfg["plan_sha256"], "config": cfg, "cells": {}, "selection": sel}
    ident_all = {}
    for name, ds, st in cells_def:
        if ds not in data:
            continue
        recs = [r for r in data[ds] if st is None or r["stratum"] == st]
        dy = ds == "D1"
        c = {"n_items": len(recs), "n_clusters": len(set(r["cluster"] for r in recs))}
        c["strata"] = {s: sum(1 for r in recs if r["stratum"] == s) for s in sorted(set(r["stratum"] for r in recs))}
        idk = sorted(recs[0]["identity"]) if recs else []
        c["identity_counts"] = {k_: sum(1 for r in recs if r["identity"][k_]) for k_ in idk}
        c["e0"] = cell(KO, recs, e_of("E0"), dyadic=dy)
        c["e_attn"] = cell(KO, recs, e_of("ans_frz_mlp"), dyadic=dy)
        c["e_mlp"] = cell(KO, recs, e_of("ans_frz_attn"), dyadic=dy)
        c["pc_e"] = cell(KO, recs, e_of("pc_final"))
        c["sz_e"] = cell(KO, recs, e_of("sz"))
        c["src_e"] = cell(KO, recs, e_of("src"))
        c["src_frz_attn_e"] = cell(KO, recs, e_of("src_frz_attn"))
        okD = [r for r in recs if r["D"] is not None and r["D"] >= GAP_E]
        mD = sum(r["D"] for r in okD) / max(1, len(okD))
        c["ratio_of_means"] = {nm: round(sum(r["rows"][nm]["dm"] for r in okD) / max(1, len(okD)) / mD, 4) if okD else None
                               for nm in ANS_ROWS}
        c["mean_D"] = round(mD, 4)
        c["clean_top1"] = {"A": sum(1 for r in recs if r["clean"][-1][3] == r["tA"]) / max(1, len(recs)),
                           "B": sum(1 for r in recs if r["cleanB"][-1][3] == r["tB"]) / max(1, len(recs))}
        g5 = {}
        for nm in ANS_ROWS:
            g5[nm] = []
            for j in range(K):
                g5[nm].append({"loop": j + 1, "norm_ratio": cell(KO, recs, g5_of(nm, j, "norm_ratio")),
                               "cos": cell(KO, recs, g5_of(nm, j, "cos")), "e": cell(KO, recs, e_of(nm, j))})
        c["g5"] = g5
        ea = c["e_attn"]
        lo, hi = (ea["dyadic"]["ci90"] if dy else ea["ci90"]) if ea else (None, None)
        c["verdict"] = None if ea is None else ("R1 falsified" if lo >= 0.5 else ("R1 supported" if hi <= 0.15 else "mixed"))
        c["bounds_used"] = "dyadic 90%" if dy else "A-side cluster 90%"
        if dy and cfg["model"] == "ByteDance/Ouro-1.4B":   # stored values are the base model's (k17_ko240)
            c["stored_clean_max_abs_diff"] = round(max(
                max(abs(r["clean"][-1][0] - r["stored_clean_final"][0]), abs(r["clean"][-1][1] - r["stored_clean_final"][1]),
                    abs(r["cleanB"][-1][0] - r["stored_cleanB_final"][0]), abs(r["cleanB"][-1][1] - r["stored_cleanB_final"][1]))
                for r in recs), 5)
        res["cells"][name] = c
        for k_ in idk:
            ident_all[k_] = ident_all.get(k_, True) and c["identity_counts"][k_] == len(recs)
    allrecs = [r for ds in data for r in data[ds]]
    readout_eq = all(r["rows"][nm]["final_equals_loop4_readout"] for r in allrecs for nm in ANS_ROWS)
    gpu_s = sum(s.get("seconds", 0) for s in prog["sessions"])
    F = res["cells"].get("facts", {})

    def bound(cname, ename, side):
        c = res["cells"].get(cname)
        if not c or not c.get(ename):
            return None
        ci = c[ename]["dyadic"]["ci90"] if "dyadic" in c[ename] else c[ename]["ci90"]
        return ci[0] if side == "lo" else ci[1]
    ck = {"zero_edit_bit_identical": ident_all.get("hook_identity"),
          "freeze_no_edit_bit_identical": ident_all.get("freeze_no_edit_identity"),
          "source_attn_frozen_e_zero_bit_exact": ident_all.get("source_attn_frozen_identity"),
          "source_edit_reaches_r_all": ident_all.get("source_edit_reaches_r"),
          "pc_exact_all": ident_all.get("pc_exact"), "sz_exact_all": ident_all.get("sz_exact"),
          "readout_identity_all": bool(ident_all.get("readout_identity") and readout_eq),
          "n_items_facts": F.get("n_items"), "n_clusters_facts": F.get("n_clusters"),
          "n_items_addition": res["cells"].get("addition", {}).get("n_items"),
          "n_clusters_addition": res["cells"].get("addition", {}).get("n_clusters"),
          "n_items_code_reads": res["cells"].get("code_reads", {}).get("n_items"),
          "n_items_code_sums": res["cells"].get("code_sums", {}).get("n_items"),
          "n_clusters_code": res["cells"].get("code_pooled", {}).get("n_clusters"),
          "n_items_twostep": res["cells"].get("twostep", {}).get("n_items"),
          "n_clusters_twostep": res["cells"].get("twostep", {}).get("n_clusters"),
          "e_attn_facts_lower_bound_ge_0.5": None if bound("facts", "e_attn", "lo") is None else bound("facts", "e_attn", "lo") >= 0.5,
          "e_mlp_facts_upper_bound_le_0.15": None if bound("facts", "e_mlp", "hi") is None else bound("facts", "e_mlp", "hi") <= 0.15,
          "gpu_minutes": round(gpu_s / 60.0, 2), "gpu_minutes_le_90": gpu_s / 60.0 <= 90,
          "outputs_hashed": bool(hash_ok), "plan_sha256": cfg["plan_sha256"]}
    res["checks"] = ck
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    json.dump(ck, open(os.path.join(out, "checks.json"), "w"), indent=1)
    write_summary(out, res)
    print(json.dumps(ck, indent=1))


def fmt(c, dy=False):
    if c is None:
        return "n/a"
    s = "%.3f [%.3f, %.3f]" % (c["mean"], c["ci90"][0], c["ci90"][1])
    if dy and "dyadic" in c:
        s += " dyadic [%.3f, %.3f]" % tuple(c["dyadic"]["ci90"])
    return s + " (%d / %d cl)" % (c["n_defined"], c["n_clusters"])


def write_summary(out, res):
    L = ["# K17 FREEZE summary (RA-3, X4a branch freeze, %s)" % res["config"]["model"], "",
         "Plan sha256 %s. Cells: mean of per-item e = dm / D (items with D >= %.1f) [90%% percentile cluster "
         "bootstrap, %d resamples, seed 0] (items with e defined / clusters)." % (res["plan_sha256"], GAP_E, NB), "",
         "| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    for nm, c in res["cells"].items():
        dy = nm == "facts"
        L.append("| %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
            nm, c["n_items"], c["n_clusters"], fmt(c["e0"], dy), fmt(c["e_attn"], dy), fmt(c["e_mlp"], dy),
            fmt(c["pc_e"]), fmt(c["src_e"]), c["verdict"]))
    L += ["", "## Ratio of means (CPU pilot form) and strata", ""]
    for nm, c in res["cells"].items():
        L.append("- %s: %s; mean D %.3f; strata %s; clean top-1 A %.3f B %.3f; identity counts %s" % (
            nm, json.dumps(c["ratio_of_means"]), c["mean_D"], json.dumps(c["strata"]), c["clean_top1"]["A"],
            c["clean_top1"]["B"], json.dumps(c["identity_counts"])))
    L += ["", "## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D", ""]
    for nm, c in res["cells"].items():
        L += ["### %s" % nm, "", "| row | loop | norm ratio | cos | e_j |", "|---|---|---|---|---|"]
        for row, per in c["g5"].items():
            for x in per:
                L.append("| %s | %d | %s | %s | %s |" % (row, x["loop"], fmt(x["norm_ratio"]), fmt(x["cos"]), fmt(x["e"])))
        L.append("")
    L += ["## Checks", "", "```", json.dumps(res["checks"], indent=1), "```"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ CPU self-test
def selftest(tree):
    os.environ.setdefault("PROD_DEV", "cpu")
    tests_dir = os.path.join(tree, "tests", "steer")
    sys.path.insert(0, tree)
    sys.path.insert(0, tests_dir)
    from conftest import OuroDecoderLayer, make_fake_ouro, real_ouro_adapter_over
    from prod.steer.adapters import bind
    from prod.steer.adapters.ouro import OuroBinding
    res = {}
    fake = make_fake_ouro("eager", L=2, total_ut_steps=K)
    ad = real_ouro_adapter_over(fake, K)
    with OuroBinding.trust_for_tests(OuroDecoderLayer):
        b = bind(ad)
        X = Ctx(b, tree)
        res["F_ans"], res["F_src"], res["n_frz_sites"] = X.F_ans, X.F_src, len(X.frz_sites)
        A = [5, 9, 7, 2, 11, 3, 6, 8]
        B = [5, 9, 13, 4, 11, 3, 6, 8]
        zA = X.nohook(torch.tensor([A]))[0, -1]
        zB = X.nohook(torch.tensor([B]))[0, -1]
        tA = int(zA.argmax())
        tB = int(zB.argmax()) if int(zB.argmax()) != tA else int(zB.topk(2).indices[1])
        rec, ident = process_item(X, A, B, [2, 3], tA, tB, "t0")
    res["identity"] = ident
    res["n_forwards"] = rec["forwards"]
    res["ans_loop1_g5_unit"] = all(abs(rec["rows"][nm]["loops"][0]["norm_ratio"] - 1) < 1e-5 and
                                   abs(rec["rows"][nm]["loops"][0]["cos"] - 1) < 1e-5 for nm in ANS_ROWS)
    res["pc_dm_equals_D"] = abs(rec["rows"]["pc_final"]["dm"] - rec["D"]) < 1e-5
    res["sz_dm_zero"] = rec["rows"]["sz"]["dm"] == 0.0
    res["src_frz_dm_zero"] = rec["rows"]["src_frz_attn"]["dm"] == 0.0
    res["freezes_change_edit"] = (rec["rows"]["ans_frz_mlp"]["dm"] != rec["rows"]["E0"]["dm"] and
                                  rec["rows"]["ans_frz_attn"]["dm"] != rec["rows"]["E0"]["dm"])
    # D2 cap rule
    cands = [{"cluster": "x=%d" % (i % 3), "i": i} for i in range(20)]
    kept, scanned = select_d2(cands, lambda p: p["i"] % 5 != 0, cap=4)
    cnt = {}
    for p in kept:
        cnt[p["cluster"]] = cnt.get(p["cluster"], 0) + 1
    res["d2_cap_ok"] = max(cnt.values()) <= 4 and all(p["i"] % 5 != 0 for p in kept) and \
        len(kept) == 12 and [p["i"] for p in kept] == sorted(p["i"] for p in kept)
    # statistics through k17_ko240's estimators
    KO = stats_mod()
    fake_recs = [{"D": 10.0, "rows": {"E0": {"dm": 1.0 * (i % 2)}}, "cluster": "c%d" % (i // 2), "sa": "s%d" % (i % 5),
                  "sb": "s%d" % ((i + 1) % 5)} for i in range(40)]
    c = cell(KO, fake_recs, e_of("E0"), dyadic=True)
    res["stats_ok"] = abs(c["mean"] - 0.05) < 1e-9 and c["n_clusters"] == 20 and c["n_defined"] == 40 and \
        c["ci90"][0] <= 0.05 <= c["ci90"][1] and "dyadic" in c
    res["ok"] = bool(all(v for k_, v in ident.items()) and res["ans_loop1_g5_unit"] and res["pc_dm_equals_D"]
                     and res["sz_dm_zero"] and res["src_frz_dm_zero"] and res["freezes_change_edit"]
                     and res["d2_cap_ok"] and res["stats_ok"])
    print(json.dumps(res, indent=1))
    return res["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_items")
    ap.add_argument("--ko_json")
    ap.add_argument("--pilot_json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tree")
    ap.add_argument("--out")
    ap.add_argument("--plan")
    ap.add_argument("--plan_sha256")
    ap.add_argument("--items_d1")
    ap.add_argument("--datasets", default="D1,D2,D4,D5")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--mem_fraction", type=float, default=0.12)
    ap.add_argument("--model", default=MODELS[0], choices=MODELS)
    ap.add_argument("--full_pool", action="store_true", help="cluster: D4/D5 quotas lifted (cap per cluster side stays)")
    a = ap.parse_args()
    if a.make_items:
        make_items(a.make_items, a.ko_json, a.pilot_json)
    elif a.selftest:
        sys.exit(0 if selftest(a.tree) else 1)
    elif a.run:
        run(a)
    elif a.summarize:
        summarize(a)


if __name__ == "__main__":
    main()
