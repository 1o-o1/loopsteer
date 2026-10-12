"""K17 FREEZE IFM (RA-23 item 2): the branch-freeze 2x2 of k17_freeze.py (RA-3, plan notes/k17_freeze_plan.md) on the
IFM no-injection model (ifm_p1_ouro, xllm, loaded through k15_smoke.IfmM / k10_ifm, pure PyTorch attention).

The rows, the identities and the record layout are those of k17_freeze.process_item, so `k17_freeze.py --summarize`
and cluster/extract/freeze.py read this script's output unchanged. Only the hooks differ (the IFM code has no
production binding with attention and MLP sites):
  state sites   the output of runner._layer at one (loop j, layer l) slot: loop1.L0 (source edit), loop j end
                (= layer L-1 output of loop j: LOOP_END_j, the answer edit at j = 1, the controls at j = K)
  attn write    the output of layers[l].attention.wo (the attention write before the residual add)
  MLP write     the output of layers[l].nffn.fc2 (the MLP write before the residual add)
A freeze replaces the write at the last position r by the clean-A value recorded at the same (loop, layer) call, at
every loop call from the first call after the edit on (answer edit: loop 2 layer 0; source edit: loop 1 layer 1).
A patch at strength 1 copies the donor value; at strength 0 it leaves the tensor unchanged (no arithmetic). Every
forward is batch 1, no KV cache; the final logits are the runner's own m.output at the last position.

    python k17_freeze_ifm.py --run --model ifm_p1_ouro --out OUT --plan PLAN.md --plan_sha256 SHA \\
        --items_d1 DATA/k17_freeze/items_d1.json [--datasets D1,D2,D4] [--debug N]
    python k17_freeze.py --summarize --out OUT          (statistics, summary.md, checks.json)
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import k17_freeze as FZ  # noqa: E402

STOP = {"flag": False}
MODELS = ("ifm_p1_ouro",)       # no injection: the loop is the whole stack and no module re-injects the input
# statistics of k17_freeze (cluster/extract/freeze.py verify loads this script and calls cell, e_of, stats_mod)
cell, e_of, stats_mod = FZ.cell, FZ.e_of, FZ.stats_mod


def sha256_file(p):
    return FZ.sha256_file(p)


class IfmCtx(object):
    """Hooks on one IFM runner: state edits at _layer outputs, write freezes and records at wo / fc2 outputs."""

    def __init__(self, name):
        import k15_smoke as S1
        M = S1.IfmM(name)
        self.M, self.r, self.m = M, M.r, M.m
        m = self.m
        self.K = int(self.r.k)
        self.s0, self.e0 = int(m.loop_start_layer), int(m.loop_end_layers)
        self.L = self.e0 - self.s0
        if not (self.s0 == 0 and self.e0 == int(m.num_layers) and m.loop_input_injection_module is None
                and m.dense_prelude_input_injection_module is None and self.r.arch != "huginn"):
            raise SystemExit("%s: not a no-injection looped stack without prelude or coda" % name)
        self.dev = self.r.dev
        self.enc = M.enc
        self.slot_pt = {slot: (int(tag[4:]), lid) for slot, lid, tag in self.r.schedule(self.K)
                        if tag.startswith("loop")}
        self.cur = [None]
        self.ops = {}            # (kind, j, l) -> fn(tensor) -> tensor ; kind in state, attn, mlp
        self.rec = {}            # (kind, j, l) -> recorded tensor (clone)
        self.want = set()        # (kind, j, l) to record
        self.n_fwd = 0
        self.F_ans = (2, self.s0)          # first layer call after the loop-1 end
        self.F_src = (1, self.s0 + 1)      # first layer call after loop1.L0
        self.loop_end = [(j, self.e0 - 1) for j in range(1, self.K + 1)]
        self.src_pt = (1, self.s0)
        self._install()

    def _install(self):
        orig = self._orig = self.r._layer
        X = self

        def lay(lid, h, freq, cache, slot):
            X.cur[0] = X.slot_pt.get(slot)
            out = orig(lid, h, freq, cache, slot)
            pt = X.cur[0]
            X.cur[0] = None
            if pt is not None:
                key = ("state",) + pt
                f = X.ops.get(key)
                if f is not None:
                    out = f(out)
                if key in X.want:
                    X.rec[key] = out.detach().clone()
            return out
        self.r._layer = lay
        self.handles = []
        for lid in range(self.s0, self.e0):
            lay_m = self.m.layers[lid]
            self.handles.append(lay_m.attention.wo.register_forward_hook(self._write_hook("attn")))
            self.handles.append(lay_m.nffn.fc2.register_forward_hook(self._write_hook("mlp")))

    def _write_hook(self, kind):
        X = self

        def hook(mod, inp, out):
            pt = X.cur[0]
            if pt is None:
                return None
            key = (kind,) + pt
            f = X.ops.get(key)
            if f is not None:
                out = f(out)
            if key in X.want:
                X.rec[key] = out.detach().clone()
            return out if f is not None else None
        return hook

    @torch.no_grad()
    def run(self, ids, ops=None, want=()):
        """-> (last-position logits [V] float32, records). ops {(kind, j, l): fn}."""
        self.ops = dict(ops or {})
        self.want = set(want)
        self.rec = {}
        try:
            z, _ = self.r.fwd(torch.tensor([ids], device=self.dev), self.K, cache=None,
                              example_ids=torch.arange(1, device=self.dev))
        finally:
            self.ops, self.want = {}, set()
        self.n_fwd += 1
        return z[0, -1].float(), self.rec

    @torch.no_grad()
    def nohook(self, ids):
        """Logits [1, 1, V] of the runner's own forward with every hook removed (reinstalled afterwards)."""
        if torch.is_tensor(ids):
            ids = ids[0].tolist()
        for h in self.handles:
            h.remove()
        lay, self.r._layer = self.r._layer, self._orig
        try:
            z, _ = self.r.fwd(torch.tensor([ids], device=self.dev), self.K, cache=None,
                              example_ids=torch.arange(1, device=self.dev))
        finally:
            self.r._layer = lay
            self.handles = []
            for lid in range(self.s0, self.e0):
                lay_m = self.m.layers[lid]
                self.handles.append(lay_m.attention.wo.register_forward_hook(self._write_hook("attn")))
                self.handles.append(lay_m.nffn.fc2.register_forward_hook(self._write_hook("mlp")))
        self.n_fwd += 1
        return z[:, -1:].float()

    @torch.no_grad()
    def readout(self, H, r):
        """Exit read-out at position r of a loop-end state H [T, d] (model dtype): m.output on H[None][:, r:r+1], the
        call the runner makes on the last loop-end state (r = T - 1), so the loop-K read-out equals the final logits
        bit for bit."""
        return self.m.output(H[None][:, r:r + 1], None, None)[0, -1].float()

    def calls_from(self, F):
        """Every (loop, layer) call at or after F in execution order."""
        return [(j, l) for j in range(1, self.K + 1) for l in range(self.s0, self.e0) if (j, l) >= F]


def copy_at(pos, src):
    """Patch at strength 1: rows `pos` of the [1, T, d] tensor become `src` [len(pos), d] (exact copy)."""
    def f(o):
        o = o.clone()
        o[0, pos] = src.to(o.dtype)
        return o
    return f


def r6(x):
    return FZ.r6(x)


def vals(z, tA, tB):
    return [r6(z[tA]), r6(z[tB]), r6(torch.logsumexp(z, -1)), int(z.argmax())]


@torch.no_grad()
def process_item(X, A, B, src, tA, tB, key):
    """k17_freeze.process_item on IFM: same rows, same identities, same record fields."""
    K = X.K
    T = len(A)
    r = T - 1
    src = sorted(src)
    LE = [("state",) + pt for pt in X.loop_end]
    SRC = ("state",) + X.src_pt
    loop_calls = X.calls_from((1, X.s0))
    rec = {"key": key, "T": T, "pos": src, "tA": tA, "tB": tB, "rows": {}}
    t0 = time.time()
    f0 = X.n_fwd
    z_nh = X.nohook(A)[0, -1]
    frz_keys = [(k,) + c for k in ("attn", "mlp") for c in loop_calls]
    zA, rA = X.run(A, want=LE + frz_keys)
    HA = {j: rA[LE[j - 1]][0] for j in range(1, K + 1)}
    zB, rB = X.run(B, want=LE + [SRC])
    HB = {j: rB[LE[j - 1]][0] for j in range(1, K + 1)}
    srcB = rB[SRC][0, src]
    roA = {j: X.readout(HA[j], r) for j in HA}
    roB = {j: X.readout(HB[j], r) for j in HB}

    def freeze(kind, F):
        return {(kind,) + c: copy_at([r], rA[(kind,) + c][0, r:r + 1]) for c in X.calls_from(F)}
    ans = {LE[0]: copy_at([r], HB[1][r:r + 1])}
    fm, fa, fa_s = freeze("mlp", X.F_ans), freeze("attn", X.F_ans), freeze("attn", X.F_src)
    srcp = {SRC: copy_at(src, srcB)}
    pcf = {LE[K - 1]: copy_at([r], HB[K][r:r + 1])}
    sz = {LE[K - 1]: copy_at(src, HB[K][src])}

    def H_of(rr):
        return {j: rr[LE[j - 1]][0] for j in range(1, K + 1)}

    def edited(ops, with_le):
        z, rr = X.run(A, ops=ops, want=LE if with_le else ())
        return z, (H_of(rr) if with_le else None)
    z_zero, _ = edited({LE[0]: (lambda o: o)}, False)          # strength 0: the tensor passes unchanged
    z_E0, H_E0 = edited(ans, True)
    z_am, H_am = edited({**ans, **fm}, True)
    z_aa, H_aa = edited({**ans, **fa}, True)
    z_fm, _ = edited(fm, False)
    z_fa, _ = edited(fa, False)
    z_src, _ = edited(srcp, False)
    z_sf, H_sf = edited({**srcp, **fa_s}, True)
    z_pc, _ = edited(pcf, False)
    z_sz, _ = edited(sz, False)
    ident = {
        "hook_identity": bool(torch.equal(zA, z_nh) and torch.equal(z_zero, zA)),
        "readout_identity": bool(torch.equal(roA[K], zA) and torch.equal(roB[K], zB)),
        "freeze_no_edit_identity": bool(torch.equal(z_fm, zA) and torch.equal(z_fa, zA)),
        "source_attn_frozen_identity": bool(torch.equal(z_sf, zA) and
                                            all(torch.equal(H_sf[j][r], HA[j][r]) for j in range(1, K + 1))),
        "source_edit_reaches_r": not bool(torch.equal(z_src, zA)),
        "pc_exact": bool(torch.equal(z_pc, zB)),
        "sz_exact": bool(torch.equal(z_sz, zA)),
        "ans_loop1_state_is_donor": all(bool(torch.equal(H[1][r], HB[1][r])) for H in (H_E0, H_am, H_aa)),
    }
    cA, cB = vals(zA, tA, tB), vals(zB, tA, tB)
    D = FZ.margin(cB) - FZ.margin(cA)
    rec["clean"] = [vals(roA[j], tA, tB) for j in range(1, K + 1)]
    rec["cleanB"] = [vals(roB[j], tA, tB) for j in range(1, K + 1)]
    rec["D"] = r6(D)
    for nm, z in (("zero", z_zero), ("frz_mlp", z_fm), ("frz_attn", z_fa), ("src", z_src),
                  ("src_frz_attn", z_sf), ("pc_final", z_pc), ("sz", z_sz)):
        v = vals(z, tA, tB)
        rec["rows"][nm] = {"final": v, "dm": r6(FZ.margin(v) - FZ.margin(cA))}
    TB = {j: (HB[j][r] - HA[j][r]).float() for j in range(1, K + 1)}
    for nm, z, H in (("E0", z_E0, H_E0), ("ans_frz_mlp", z_am, H_am), ("ans_frz_attn", z_aa, H_aa)):
        per = []
        for j in range(1, K + 1):
            v = vals(X.readout(H[j], r), tA, tB)
            dl = (H[j][r] - HA[j][r]).float()
            tn = float(TB[j].norm())
            per.append({"v": v, "dm": r6(FZ.margin(v) - FZ.margin(rec["clean"][j - 1])),
                        "norm_ratio": r6(float(dl.norm()) / tn) if tn > 0 else None,
                        "cos": r6(torch.nn.functional.cosine_similarity(dl, TB[j], dim=0)) if tn > 0 else None,
                        "delta_norm": r6(dl.norm()), "target_norm": r6(tn)})
        fv = vals(z, tA, tB)
        rec["rows"][nm] = {"final": fv, "dm": r6(FZ.margin(fv) - FZ.margin(cA)), "loops": per,
                           "final_equals_loop4_readout": bool(torch.equal(X.readout(H[K], r), z))}
    rec["identity"] = ident
    rec["forwards"] = X.n_fwd - f0
    rec["seconds"] = round(time.time() - t0, 3)
    return rec, ident


# ------------------------------------------------------------------ items (k17_freeze's cluster pools, IFM tokenizer)
def build_d1(X, items_path, debug):
    """The full D1 candidate pool on the IFM tokenizer (k17_freeze.build_d1_pool: every k15_data.d1_candidates pair;
    items_d1.json is Ouro-1.4B base's clean set and only labels src), then the IFM clean filter (RA-20 item 9,
    RA-23 'full pools')."""
    pool, info = FZ.build_d1_pool(X.enc, items_path)
    st = {"A_fail": 0, "B_fail": 0}
    know = FZ.make_know(X, st)
    scan = pool[:3 * debug] if debug else pool
    items = [it for it in scan if know(it)]
    info.update({"scanned": len(scan), "kept": len(items), "know_stats": st,
                 "rule": "every d1_candidates pair on the IFM tokenizer, then the IFM clean filter"})
    return items, info


def build_scan(X, ds, debug, data_root):
    """D2 and D4 as k17_freeze.build_scan (D2 every clean-passing candidate, cap 4 per edited operand value; D4 the
    k15_data.select cap of 2 per cluster side, quota lifted: every candidate scanned)."""
    import k15_data as KD
    import k16_path_ouro as KP
    if ds == "D4":
        KP.DATA_DIR = os.path.join(data_root, "k16_path_ifm", "d4")
    stats = {"A_fail": 0, "B_fail": 0}
    know = FZ.make_know(X, stats)
    if ds == "D2":
        cands = KD.d2_candidates(X.enc)
        for p in cands:
            a1, a2_ = [int(v) for v in p["a"].split(" + ")]
            p["cluster"] = "x=%d" % (a1 if p["rel"] == "op1" else a2_)
        kept, scanned = FZ.select_d2(cands[:12] if debug else cands, know)
        info = {"candidates": len(cands), "rule": "all clean-passing, cap %d per A-side edited operand value" % FZ.D2_CAP}
    else:
        cands = KP.ext_candidates(ds, "ifm")
        quota = {k: (1 if debug else 10 ** 9) for k in FZ.D4_QUOTA}
        kept, scanned = KD.select(cands, know, quota)
        for p in kept:
            p["cluster"] = p["cluster_id"]
        info = {"candidates": len(cands), "quota": {"%s|%s" % k: v for k, v in quota.items()},
                "rule": "k15_data.select cap 2 per cluster side per (half, kind), quota lifted",
                "data_md5": FZ.md5_file(os.path.join(KP.DATA_DIR, "d4_ifm.jsonl"))}
    info.update({"scanned": scanned, "kept": len(kept), "know_stats": stats})
    items = []
    for p in kept:
        key = ("%s|%s|%s" % (p["rel"], p["a"], p["b"])) if ds == "D2" else str(p["id"])
        items.append({"key": key, "A": p["A"], "B": p["B"], "pos": sorted(p["pos"]), "tA": p["tA"], "tB": p["tB"],
                      "cluster": p["cluster"], "stratum": p["rel"], "half": p["half"]})
    return items, info


# ------------------------------------------------------------------ run (k17_freeze.run on the IFM context)
def run(args):
    t_start = time.time()
    out = args.out
    os.makedirs(out, exist_ok=True)
    if sha256_file(args.plan) != args.plan_sha256:
        raise SystemExit("plan sha256 mismatch")
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))
    X = IfmCtx(args.model)
    import k10_ifm
    import k15_data as KD
    import k15_smoke as S1
    import k16_path_ouro as KP
    data_root = os.environ.get("DATA_ROOT") or os.path.join(HERE, "data")
    cfg_path = os.path.join(out, "config.json")
    cfg = {"brief": "RA-23 item 2 (RA-3 rows on IFM)", "plan_sha256": args.plan_sha256,
           "code_sha256": sha256_file(os.path.abspath(__file__)), "k17_freeze_sha256": sha256_file(FZ.__file__),
           "k10_ifm_sha256": sha256_file(k10_ifm.__file__), "k15_smoke_md5": FZ.md5_file(S1.__file__),
           "k15_data_md5": FZ.md5_file(KD.__file__), "k16_path_ouro_md5": FZ.md5_file(KP.__file__),
           "items_d1_sha256": sha256_file(args.items_d1), "model": args.model,
           "checkpoint": k10_ifm.NAMES[args.model], "k": X.K, "batch": 1, "n_layers_per_loop": X.L,
           "attn_backend": X.r.attn_backend, "gap_e": FZ.GAP_E, "d2_cap": FZ.D2_CAP, "datasets": args.datasets,
           "debug": args.debug, "F_ans": list(X.F_ans), "F_src": list(X.F_src),
           "hooks": "state: runner._layer output; attn write: layers[l].attention.wo output; MLP write: "
                    "layers[l].nffn.fc2 output (both before the residual add)",
           "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "n_freeze_sites": 2 * len(X.calls_from((1, X.s0)))}
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        for k_ in ("plan_sha256", "code_sha256", "items_d1_sha256", "checkpoint", "debug"):
            if old.get(k_) != cfg.get(k_):
                raise SystemExit("resume with a changed config field %s" % k_)
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=1, sort_keys=True)
    print("[freeze_ifm] config sha256 %s" % sha256_file(cfg_path), flush=True)
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
            items, info = build_d1(X, args.items_d1, args.debug) if ds == "D1" else build_scan(X, ds, args.debug, data_root)
            if args.debug:
                items = items[:args.debug]
            info["scan_forwards"] = X.n_fwd - f0
            info["scan_seconds"] = round(time.time() - ts, 1)
            sel = {"dataset": ds, "info": info, "items": items}
            json.dump(sel, open(sel_path + ".tmp", "w"))
            os.replace(sel_path + ".tmp", sel_path)
            print("[freeze_ifm] %s selection: %s" % (ds, json.dumps({k_: v for k_, v in info.items() if k_ != "bad"})),
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
                for k_ in ("cluster", "stratum", "half", "sa", "sb", "src"):
                    if k_ in it:
                        rec[k_] = it[k_]
                rec["dataset"] = ds
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_new += 1
                if not all(ident.values()):
                    print("[freeze_ifm] %s item %s identity flags %s" % (ds, it["key"], json.dumps(ident)), flush=True)
                if n_new % 20 == 1:
                    print("[freeze_ifm] %s %d/%d (%.2fs/item) %.0fs" % (ds, len(done) + n_new, len(sel["items"]),
                                                                        (time.time() - ta) / n_new, time.time() - t_start),
                          flush=True)
                    save_prog()
        save_prog()
        print("[freeze_ifm] %s done %d new items %.0fs" % (ds, n_new, time.time() - ta), flush=True)
        if STOP["flag"]:
            print("[freeze_ifm] SIGTERM: stopped after the current item", flush=True)
            return
    sess["finished"] = True
    save_prog()
    FZ.write_hashes(out)
    print("[freeze_ifm] done %.0fs, %d forwards" % (time.time() - t_start, X.n_fwd), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODELS[0], choices=MODELS)
    ap.add_argument("--out")
    ap.add_argument("--plan")
    ap.add_argument("--plan_sha256")
    ap.add_argument("--items_d1")
    ap.add_argument("--datasets", default="D1,D2,D4")
    ap.add_argument("--debug", type=int, default=0)
    a = ap.parse_args()
    if a.run:
        run(a)


if __name__ == "__main__":
    main()
