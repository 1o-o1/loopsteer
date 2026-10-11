"""steer_gen_raven: the McLeish r32 path for steer_gen arm (a) (generation placement on the GSM8K VarBench pairs).

prod/steer_gen.py is imported UNCHANGED (its hook kernel `SteerHooks._apply/_build_plan`, position sets, random
directions, pool, shuffle, sentinel placement, layout simulation, cells/chains readers, outcome parser, bootstrap
and `analyze_a`). This file adds only what is McLeish-specific:

  * `RavenSteerHooks(SteerHooks)`: the same edit kernel on McLeish module hooks. Sites (K = 32):
      emb@1        transformer.wte output (token embedding) at the edited positions; B's embedding there makes the
                   whole forward the B forward (harness identity, run-1 r02)
      pre_out@0    output of the last prelude block = the injected e of every loop (complete patches only)
      post@j       output of the last core block at loop j = the loop-end state (Llast@j, k17_public site
                   ("c", j, C-1)); it is loop j+1's state input, or the coda input at j = K
      kv@j         K and V parts of attn.Wqkv(x) before rotary, at prelude calls (j = 0) and loop-j calls (j >= 1)
  * `SteeredMcLeishAdapter(McLeishAdapter)`: prod's McLeish adapter whose `decode` is a port of
      prod/models/raven.py RavenAdapter.decode (378-480) with ONE change: the initial latent state is the fixed
      per-item table (the `fixed_table` policy of prod/steer/adapters/raven.py RavenBinding.x0_table 165-180 and the
      fixed per-item state of loop-flow/src/k17_public.py RavenR.set_init): row key r draws table_r [t_max, d] once
      on a CPU generator seeded 1000003 r (trunc_normal(0, std, -3 std, 3 std) x emb_scale), and absolute position
      p reads table_r[p]. So x0 is a function of (row, position) only: the A and B prompts of an item, every arm,
      every batch layout and G.main's re-prefill at each wave all see the same x0 (position purity).
  * Rows r00-r21 of run 1 mapped to K = 32 by role (loop 1 -> 1, loop 2 -> 2, loop 3 = K-1 -> 31, K -> 32), plus
    r20n (r20's zero edits without the layout shift: the neutral-hook token identity). Complete patches on McLeish
    must also carry the prelude (its K/V and its output e, which every later loop re-injects); see rows_raven.
  * Modes: selftest (CPU, tiny random McLeish), checks (forward-level identities on the real model), prep, screen,
    select, run --row, analyze. Output layout of the cluster bundle: $OUT_ROOT/A/steer_gen_raven/mcleish_gsm8k
    (the gate subset in .../mcleish_gsm8k/gate). results.csv (cluster/extract/common.py columns, neither_rate per
    row) is written in every row folder and in the setting folder by `analyze`.

Generation uses SETTLE's own McLeish prompt (prod/tasks gsm8k, natural protocol, add_special_tokens=True: BOS first)
through prod.generate.main (G.main) at K = 32, horizon 512, batch width 16 (run 1's batch), stop rule of record.
Python >= 3.10 (the McLeish modeling file uses PEP 604 unions).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
LOOPEDCOT = os.path.join(HERE, "tree")          # the bundle's prod tree: steer_gen and every prod module it imports
if LOOPEDCOT not in sys.path:
    sys.path.insert(0, LOOPEDCOT)
BUNDLE_TOP = os.path.dirname(os.path.dirname(HERE))   # the folder that holds cluster/ (cluster.extract, like merge.py)
if BUNDLE_TOP not in sys.path:
    sys.path.append(BUNDLE_TOP)                      # after the prod tree, so prod stays the bundle copy

import torch  # noqa: E402

from prod import steer_gen as S  # noqa: E402
from prod.models import raven as R  # noqa: E402
from prod.models.mcleish import McLeishAdapter  # noqa: E402

VERSION = "k17genraven-1"
MODEL = "mcleish_llama32_r32"
K = 32
MCLEISH_REV = "a5f6f126e9d9302445346844f1d7d29d111a06eb"
MODELING_SHA256_PREFIX = "4f7a86e9b4ce9e73"     # k17_public MODELING_SHA16["mcleish"]
SETTING = "mcl"
T_MAX = 2048                                       # x0 table rows: max prompt (< 800) + horizon 512 + suffix + answer
S.SETTINGS[SETTING] = {
    "arm": "a", "model": MODEL, "k": K, "horizon": 512, "chat": False, "think_stop": True,
    "n_scored": None, "n_sent": 22, "n_shift": 8, "batch_width": 16, "x0_policy": "fixed_table",
    "x0_t_max": T_MAX, "x0_seed_rule": "1000003 * row_idx (CPU generator, float32 table, cast to bf16)",
    "settle": "cells_mcleish_llama32_r32_gsm8k_natural_k32.jsonl", "settle_prefilter_cap": 4096,
    "screen_A_from_settle": False, "n_gate": 20, "n_gate_screen": 30,
    "rows": ["r00", "r01", "r02", "r20", "r21", "r10", "r03", "r04", "r05", "r06", "r07", "r08", "r09",
             "r11", "r12", "r13", "r19", "r14", "r15", "r16", "r17", "r18"],
    "gate_rows": ["r00", "r01", "r02", "r20n", "r20", "r21"],
    "loop_map": {"Llast@1": 1, "Llast@2": 2, "Llast@3 (= K-1 in run 1)": K - 1, "Llast@K": K},
}


# ====================================================================== rows (run 1 by role, K = 32)
def rows_raven(k=K):
    """Run-1 rows (steer_gen.rows_arm_a) on McLeish. Loop map by role: 1 -> 1, 2 -> 2, 3 (= K-1 on Ouro k4) -> k-1,
    K -> k. Complete patch @j (positive control, must reproduce swap B): O1 at post@j on span..prompt end, plus the
    prelude output e (pre_out@0) and the K/V of every prelude call and of loops 1..j on the same positions. Then every
    state after loop j, every injected e and every K/V the generation reads equal the B run (e is re-injected at
    every loop on raven, so the Ouro recipe without pre_out would leave loops > j reading A's e)."""
    P = lambda j, pos, **kw: dict({"op": "patch", "site": "post", "loop": j, "pos": pos, "a": 1.0}, **kw)
    Rr = lambda j, pos, seed=0: {"op": "rand", "site": "post", "loop": j, "pos": pos, "seed": seed, "ref": "patch"}
    comp = lambda j: [P(j, "span_to_end"), {"op": "patch", "site": "pre_out", "loop": 0, "pos": "span_to_end", "a": 1.0},
                      {"op": "kv", "loops": list(range(0, j + 1)), "pos": "span_to_end", "a": 1.0}]
    km1 = k - 1
    zero = [{"op": "patch", "site": "emb", "loop": 1, "pos": "span", "a": 0.0},
            {"op": "patch", "site": "pre_out", "loop": 0, "pos": "span", "a": 0.0},
            {"op": "patch", "site": "post", "loop": list(range(1, k + 1)), "pos": "span", "a": 0.0},
            {"op": "kv", "loops": list(range(0, k + 1)), "pos": "span", "a": 0.0}]
    return {
        "r00": {"prompt": "A", "edits": [], "role": "clean A"},
        "r01": {"prompt": "B", "edits": [], "role": "swap B"},
        "r02": {"prompt": "A", "edits": [{"op": "patch", "site": "emb", "loop": 1, "pos": "span", "a": 1.0}],
                "role": "harness identity: token embedding at the span (wte output)"},
        "r03": {"prompt": "A", "edits": [P(1, "span")], "role": "source Llast@1"},
        "r19": {"prompt": "A", "edits": [P(2, "span")], "role": "source Llast@2"},
        "r04": {"prompt": "A", "edits": [P(km1, "span")], "role": "source Llast@K-1 (run-1 Llast@3)"},
        "r05": {"prompt": "A", "edits": [P(list(range(1, k + 1)), "span")], "role": "persistent source Llast@1..K"},
        "r06": {"prompt": "A", "edits": [P(1, "last_prompt")], "role": "null site (no answer state) Llast@1"},
        "r07": {"prompt": "A", "edits": [P(k, "last_prompt")], "role": "null site (no answer state) Llast@K"},
        "r08": {"prompt": "A", "edits": [Rr(1, "span", 0)], "role": "random span Llast@1"},
        "r09": {"prompt": "A", "edits": comp(1), "role": "complete patch @1"},
        "r11": {"prompt": "A", "edits": comp(km1), "role": "complete patch @K-1 (run-1 @3)"},
        "r10": {"prompt": "A", "edits": comp(k), "role": "complete patch @K"},
        "r12": {"prompt": "A", "edits": [P(1, "span_to_end")], "role": "span to prompt end Llast@1"},
        "r13": {"prompt": "A", "edits": [{"op": "kv", "loops": [1], "pos": "span", "a": 1.0}], "role": "KV loop 1"},
        "r14": {"prompt": "A", "edits": [Rr(km1, "span", 0)], "role": "random span Llast@K-1"},
        "r15": {"prompt": "A", "edits": [Rr(list(range(1, k + 1)), "span", 0)], "role": "random persistent"},
        "r16": {"prompt": "A", "edits": [Rr(1, "last_prompt", 0)], "role": "random last prompt Llast@1"},
        "r17": {"prompt": "A", "edits": [Rr(1, "span", 1)], "role": "random span Llast@1 seed 1"},
        "r18": {"prompt": "A", "edits": [Rr(1, "span", 2)], "role": "random span Llast@1 seed 2"},
        "r21": {"prompt": "A", "self_donor": True, "edits": [P(1, "span")],
                "role": "self-donor null: O1 Llast@1 span, donor = A's own batch-1 state (CL16)"},
        "r20": {"prompt": "A", "shift": True, "edits": zero, "role": "layout-perturbed duplicate clean (CL3)"},
        "r20n": {"prompt": "A", "edits": zero, "role": "neutral hooks: every hook kind at strength 0, r00's layout"},
    }


# ====================================================================== x0: fixed per-item table
SEED_MULT = R.SEED_MULT


def x0_table(model, key, t_max=T_MAX):
    """RavenBinding.x0_table (prod/steer/adapters/raven.py 165-180) for an int row key: float32 CPU [t_max, d],
    trunc_normal_(0, std, -3 std, 3 std) on a CPU generator seeded 1000003 r, times emb_scale when it is not 1."""
    std = float(model.config.init_values["std"])
    d = int(model.config.n_embd)
    g = torch.Generator()
    g.manual_seed(SEED_MULT * int(key))
    t = torch.empty((int(t_max), d), dtype=torch.float32)
    torch.nn.init.trunc_normal_(t, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=g)
    es = float(model.emb_scale)
    if es != 1:
        t = t * es
    return t


# ====================================================================== hooks
class RavenSteerHooks(S.SteerHooks):
    """steer_gen's edit kernel on McLeish module hooks. Per forward: row geometry from cache_position and the
    row pads the decode loop sets (`pad_rows`; the raven forward has no attention_mask argument, prod masks
    through R._BIAS); block pre-hooks set the call (stage-loop j, layer); counts are asserted at the end."""

    def __init__(self, k=None):
        super(RavenSteerHooks, self).__init__(k)
        self.pad_rows = None

    def install(self, model, kv=False):
        assert self.model is None, "hooks already installed"
        self.model = model
        tr = model.transformer
        cfg = model.config
        self.P, self.C, self.Cd = int(cfg.n_layers_in_prelude), int(cfg.n_layers_in_recurrent_block), int(cfg.n_layers_in_coda)
        self.d = int(cfg.n_embd)
        self.L = self.P + self.C
        self._chunks = list(tr.prelude[0].attn.chunks)
        hs = self.handles
        hs.append(model.register_forward_pre_hook(self._pre_top, with_kwargs=True))
        hs.append(model.register_forward_hook(self._post_top, with_kwargs=True))
        hs.append(tr.wte.register_forward_hook(self._emb_hook))
        for i, blk in enumerate(tr.prelude):
            hs.append(blk.register_forward_pre_hook(partial(self._blk_pre, "prelude", i)))
        for l, blk in enumerate(tr.core_block):
            hs.append(blk.register_forward_pre_hook(partial(self._blk_pre, "core", l)))
        for c, blk in enumerate(tr.coda):
            hs.append(blk.register_forward_pre_hook(partial(self._blk_pre, "coda", c)))
        hs.append(tr.prelude[self.P - 1].register_forward_hook(self._prelude_out))
        hs.append(tr.core_block[self.C - 1].register_forward_hook(self._loop_end))
        if kv:
            for blk in list(tr.prelude) + list(tr.core_block):
                hs.append(blk.attn.Wqkv.register_forward_hook(self._wqkv_hook))
            self.kv_installed = True
        return self

    # ------------------------------------------------------------------ forward bracket
    def _pre_top(self, module, args, kwargs):
        if kwargs.get("labels") is not None:
            raise RuntimeError("steer_gen_raven never passes labels")
        ids = kwargs.get("input_ids", args[0] if args else None)
        B, T = int(ids.shape[0]), int(ids.shape[1])
        dev = ids.device
        cp = kwargs.get("cache_position")
        if cp is None:
            raise RuntimeError("raven forward without cache_position (prod passes it, raven.py 434, 470)")
        ns = kwargs.get("num_steps")
        if self.k is not None and (ns is None or int(ns) != self.k):
            raise RuntimeError("num_steps %r, hooks expect k=%d" % (ns, self.k))
        pad = self.pad_rows
        if pad is None:
            pad = [0] * B
        if len(pad) != B:
            raise RuntimeError("pad_rows has %d rows, forward has %d" % (len(pad), B))
        padt = torch.tensor([int(p) for p in pad], dtype=torch.long, device=dev)
        absp = cp.to(dev).view(1, -1).long() - padt.view(-1, 1)
        keys = self.batch_keys
        if keys is not None and len(keys) != B:
            raise RuntimeError("batch_keys has %d rows, forward has %d" % (len(keys), B))
        f = {"B": B, "T": T, "abs": absp, "ncall": 0, "loop": 0, "call": None, "plan": {}, "lcdl": [],
             "need_h1pre": False, "need_h1post": False, "keys": keys, "n_blocks": 0, "n_prelude_out": 0}
        if keys is not None:
            self._build_plan(f)
        if f["lcdl"] or f["need_h1pre"]:
            raise RuntimeError("LoopCD rows are not defined on the raven path")
        self.f = f
        self.n_forwards += 1
        return None

    def _post_top(self, module, args, kwargs, output):
        f = self.f
        self.f = None
        if f is None:
            return None
        if self.k is not None and f["loop"] != self.k:
            raise RuntimeError("core block ran %d loops, expected k=%d" % (f["loop"], self.k))
        exp = self.P + self.C * f["loop"] + self.Cd
        if f["n_blocks"] != exp or f["n_prelude_out"] != 1:
            raise RuntimeError("block calls %d (expected %d), prelude outputs %d" % (f["n_blocks"], exp,
                                                                                   f["n_prelude_out"]))
        rec = self.rec
        if rec is not None and "logits" in rec["want"]:
            rec["logits"].append(output.logits[:, -1].float().detach().clone())
        return None

    # ------------------------------------------------------------------ sites
    def _blk_pre(self, stage, i, module, args):
        f = self.f
        if f is None:
            return None
        f["n_blocks"] += 1
        if stage == "prelude":
            f["call"] = (0, i)
        elif stage == "core":
            if i == 0:
                f["loop"] += 1
            f["call"] = (f["loop"], i)
            exp = self.P + self.C * (f["loop"] - 1) + i
            if int(args[2]) != exp:
                raise RuntimeError("core block_idx %d, expected %d" % (int(args[2]), exp))
        else:
            f["call"] = (-1, i)
        return None

    def _emb_hook(self, module, args, out):
        f = self.f
        if f is None:
            return None
        items = f["plan"].get(("emb", 1))
        y = self._apply(out, "emb", 1, items) if items else out
        rec = self.rec
        if rec is not None and "emb" in rec["want"]:
            rec["emb"] = y[:, rec["cols"]].detach().clone()
        return None if y is out else y

    def _prelude_out(self, module, args, out):
        f = self.f
        if f is None:
            return None
        f["n_prelude_out"] += 1
        items = f["plan"].get(("pre_out", 0))
        y = self._apply(out, "pre_out", 0, items) if items else out
        rec = self.rec
        if rec is not None and "post" in rec["want"]:
            rec["post"][0] = y[:, rec["cols"]].detach().clone()
        return None if y is out else y

    def _loop_end(self, module, args, out):
        f = self.f
        if f is None:
            return None
        j = f["loop"]
        items = f["plan"].get(("post", j))
        y = self._apply(out, "post", j, items) if items else out
        rec = self.rec
        if rec is not None and "post" in rec["want"]:
            rec["post"][j] = y[:, rec["cols"]].detach().clone()
        return None if y is out else y

    def _wqkv_hook(self, module, args, out):
        f = self.f
        if f is None:
            return None
        j, li = f["call"]
        if j < 0:
            return None
        rec = self.rec
        items = f["plan"].get(("kv", j))
        want = rec is not None and "kv" in rec["want"]
        if not items and not want:
            return None
        q, k_, v = out.split(self._chunks, dim=2)
        k2 = self._apply(k_, "kv", j, items, dkind=(j, li, "k")) if items else k_
        v2 = self._apply(v, "kv", j, items, dkind=(j, li, "v")) if items else v
        if want:
            rec["kv"][(j, li, "k")] = k2[:, rec["cols"]].detach().clone()
            rec["kv"][(j, li, "v")] = v2[:, rec["cols"]].detach().clone()
        if k2 is k_ and v2 is v:
            return None
        return torch.cat([q, k2, v2], dim=2)


# ====================================================================== adapter
class SteeredMcLeishAdapter(McLeishAdapter):
    """prod's McLeishAdapter + RavenSteerHooks + the fixed per-item x0 table. Only `decode` differs from prod
    (raven.py 378-480): x0 and the per-step state come from the row's table at the row's absolute positions."""

    def __init__(self, name=MODEL, steer=None, kv=False, t_max=T_MAX, **kw):
        super(SteeredMcLeishAdapter, self).__init__(name=name, **kw)
        self.steer = steer
        self.kv = bool(kv)
        self.t_max = int(t_max)
        self._tables = {}
        self.file_sha = {}

    def load(self, strict_patch=False):
        super(SteeredMcLeishAdapter, self).load(strict_patch=strict_patch)
        noise = float(getattr(self.model.config, "test_time_noise", 0) or 0)
        if noise != 0.0:
            raise SystemExit("test_time_noise %r (must be 0)" % noise)
        mod = sys.modules[type(self.model).__module__]
        mf = getattr(mod, "__file__", None)
        if mf and os.path.exists(mf):
            self.file_sha["raven_modeling_minimal.py"] = S.sha256_file(mf)[:16]
        self.after_load()
        return self

    def after_load(self):
        if self.steer is not None:
            self.steer.install(self.model, kv=self.kv)
            self.steer.record_fn = self.record_donor

    def set_depth(self, k):
        d = super(SteeredMcLeishAdapter, self).set_depth(k)
        if self.steer is not None:
            self.steer.k = int(k)
        return d

    def table(self, key):
        t = self._tables.get(key)
        if t is None:
            t = x0_table(self.model, key, self.t_max).to(self.dtype)
            self._tables[key] = t
        return t

    def x0_rows(self, keys, pads, L):
        """[B, L, d] in the model dtype: row b, column c reads table_b[c - pad_b] (pad columns table_b[0])."""
        live = set(keys)
        for kk in [x for x in self._tables if x not in live]:
            del self._tables[kk]
        out = torch.empty((len(keys), L, int(self.model.config.n_embd)), dtype=self.dtype)
        for b, (kk, p) in enumerate(zip(keys, pads)):
            if L - p > self.t_max:
                raise RuntimeError("position %d >= t_max %d" % (L - p, self.t_max))
            t = self.table(kk)
            out[b, p:] = t[:L - p]
            if p:
                out[b, :p] = t[0]
        return out

    @torch.no_grad()
    def decode(self, seqs, n_new, k, eos_ids=None, stop_strings=None, pieces=None, row_ids=None,
               use_mask=None, cache=None, allow_unsafe_padding=False):
        """Port of prod RavenAdapter.decode (raven.py 378-480); x0 = fixed per-item table (see module doc)."""
        assert int(k) == self._k, "set_depth(%r) before decode" % (k,)
        m = self.model
        dev = next(m.parameters()).device
        pieces = self._pieces if pieces is None else pieces
        eos = set(eos_ids if eos_ids is not None else self.stop_ids())
        pad = self.pad_id()
        B = len(seqs)
        lens = [len(s) for s in seqs]
        L = max(lens)
        padded = any(n != L for n in lens)
        if use_mask is None:
            use_mask = padded and B > 1
        if padded and B > 1 and not use_mask and not allow_unsafe_padding:
            raise ValueError("a padded raven batch needs use_mask=True")
        row_ids = list(range(B)) if row_ids is None else list(row_ids)
        rkeys = [int(r) for r in row_ids]
        pads = [L - n for n in lens]
        st = self.steer
        keys = [str(r) for r in row_ids]
        if st is not None:
            for b, key in enumerate(keys):
                info = st.rows.get(key)
                if info is None or not info.get("a_ids"):
                    continue
                a = info["a_ids"]
                if list(seqs[b][:len(a)]) != list(a):
                    raise RuntimeError("row %s: the batch sequence does not start with the row's prompt" % key)
            st.f = None
            st.ensure_donors(keys)
            st.batch_keys = keys
            st.pad_rows = pads

        ids = torch.full((B, L), pad, dtype=torch.long, device=dev)
        valid = torch.zeros((B, L + n_new), dtype=torch.bool, device=dev)
        for r, s in enumerate(seqs):
            ids[r, L - len(s):] = torch.tensor(s, dtype=torch.long, device=dev)
            valid[r, L - len(s):L] = True
        own_cache = cache is None
        cache = self.make_cache(k, L + n_new + 1, B) if own_cache else cache
        cache.reset()
        t0 = time.time()
        gen = [[] for _ in range(B)]
        try:
            x0 = self.x0_rows(rkeys, pads, L).to(dev)
            R._BIAS[0] = R.build_bias(valid[:, :L], L, self.dtype, causal=True) if use_mask else None
            try:
                out = m(input_ids=ids, num_steps=int(k), past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(0, L, device=dev), input_states=x0)
                cur = out.logits[:, -1, :].float().argmax(-1)
                del out
            finally:
                R._BIAS[0] = None
            tails = [""] * B
            fin = [False] * B
            pos = L
            for t in range(n_new):
                nxt = cur.tolist()
                feed = [pad] * B
                for b in range(B):
                    if fin[b]:
                        continue
                    x = nxt[b]
                    gen[b].append(x)
                    feed[b] = x
                    if x in eos:
                        fin[b] = True
                        continue
                    if stop_strings:
                        tails[b] = (tails[b] + pieces[x])[-64:]
                        if any(s_ in tails[b] for s_ in stop_strings):
                            fin[b] = True
                if t == n_new - 1 or all(fin):
                    break
                valid[:, pos] = True
                x1 = torch.stack([self.table(kk)[pos - p] for kk, p in zip(rkeys, pads)], 0).unsqueeze(1).to(dev)
                R._BIAS[0] = (R.build_bias(valid[:, :pos + 1], 1, self.dtype, causal=False) if use_mask else None)
                try:
                    out = m(input_ids=torch.tensor(feed, dtype=torch.long, device=dev).unsqueeze(1),
                            num_steps=int(k), past_key_values=cache, use_cache=True,
                            cache_position=torch.tensor([pos], device=dev), input_states=x1)
                    cur = out.logits[:, -1, :].float().argmax(-1)
                    del out
                finally:
                    R._BIAS[0] = None
                pos += 1
        finally:
            if st is not None:
                st.batch_keys = None
                st.pad_rows = None
                st.f = None
        dt = time.time() - t0
        if own_cache:
            del cache
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        if st is not None:
            st.log_decode(keys, seqs, n_new, gen)
        return gen, dt

    @torch.no_grad()
    def forward_rows(self, seqs, keys, want_logits="last"):
        """Batch forward, no cache, left-padded with the mask (checks): float32 logits at the last column [B, V]
        (or every column, `want_logits='all'`, batch 1 only). Hooks act on rows in steer.rows."""
        m = self.model
        dev = next(m.parameters()).device
        B = len(seqs)
        lens = [len(s) for s in seqs]
        L = max(lens)
        pads = [L - n for n in lens]
        ids = torch.full((B, L), self.pad_id(), dtype=torch.long, device=dev)
        valid = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for r, s in enumerate(seqs):
            ids[r, L - len(s):] = torch.tensor(s, dtype=torch.long, device=dev)
            valid[r, L - len(s):] = True
        st = self.steer
        if st is not None:
            st.f = None
            st.ensure_donors([str(x) for x in keys])
            st.batch_keys = [str(x) for x in keys]
            st.pad_rows = pads
        try:
            x0 = self.x0_rows([int(x) for x in keys], pads, L).to(dev)
            R._BIAS[0] = R.build_bias(valid, L, self.dtype, causal=True) if (B > 1 and any(pads)) else None
            try:
                out = m(input_ids=ids, num_steps=int(self._k), use_cache=False,
                        cache_position=torch.arange(0, L, device=dev), input_states=x0)
            finally:
                R._BIAS[0] = None
        finally:
            if st is not None:
                st.batch_keys = None
                st.pad_rows = None
                st.f = None
        z = out.logits.float()
        return z[:, -1] if want_logits == "last" else z

    @torch.no_grad()
    def record_donor(self, key):
        """Donor states of a row: one batch-1, no-cache forward of the row's donor sequence `b_ids` with the row's
        own x0 table, recording post@0 (prelude output), post@1..K, the token embedding and (kv rows) every prelude
        and loop call's K/V before rotary, at donor positions [p0, p1). Target position t0 + i reads donor row i."""
        st = self.steer
        info = st.rows[key]
        p0, p1 = info["donor_range"]
        want = {"post", "emb"}
        if any(e["op"] == "kv" for e in info.get("edits") or []):
            want.add("kv")
        saved_keys, saved_rec, saved_pad = st.batch_keys, st.rec, st.pad_rows
        st.batch_keys = None
        st.pad_rows = None
        st.rec = {"cols": slice(int(p0), int(p1)), "want": want, "post": {}, "pre": {}, "kv": {}, "emb": None,
                  "logits": []}
        m = self.model
        dev = next(m.parameters()).device
        try:
            ids = torch.tensor([info["b_ids"]], dtype=torch.long, device=dev)
            T = ids.shape[1]
            x0 = self.table(int(key))[:T].unsqueeze(0).to(dev)
            m(input_ids=ids, num_steps=int(self._k), use_cache=False,
              cache_position=torch.arange(0, T, device=dev), input_states=x0)
            rec = st.rec
        finally:
            st.rec = saved_rec
            st.batch_keys = saved_keys
            st.pad_rows = saved_pad
        if (want & {"kv"}) and not rec["kv"]:
            raise RuntimeError("kv donor requested but the Wqkv hooks are not installed")
        return {"t0": int(info.get("t0", p0)), "n": int(p1) - int(p0),
                "post": {j: v[0] for j, v in rec["post"].items()},
                "emb": None if rec["emb"] is None else rec["emb"][0],
                "kv": {kk: v[0] for kk, v in rec["kv"].items()}}

    def meta(self):
        d = super(SteeredMcLeishAdapter, self).meta()
        d.update({"steer_gen_raven": VERSION, "steer_kv_hooks": self.kv, "x0_policy": "fixed_table",
                  "x0_t_max": self.t_max, "x0_seed_rule": "1000003 * row_idx, CPU generator, float32 -> bf16",
                  "file_sha16": dict(self.file_sha)})
        return d


# ====================================================================== tokenizer and geometry
def load_tokenizer():
    from transformers import AutoTokenizer
    from prod import config as cfgmod
    tok = AutoTokenizer.from_pretrained("smcleish/Recurrent-Llama-3.2-train-recurrence-32",
                                        revision=cfgmod.revision_for(MODEL))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def enc(tok, text):
    """G.main's encoding for McLeish: add_special_tokens=True (BOS first), asserted = [BOS] + plain ids."""
    a = tok(text, add_special_tokens=True)["input_ids"]
    b = tok(text, add_special_tokens=False)["input_ids"]
    if a != [tok.bos_token_id] + b:
        raise RuntimeError("McLeish encoding is not BOS + plain ids")
    return a


def geometry(tok, prow, srows):
    """steer_gen.item_geometry (plain ids, character offsets) shifted by the BOS: span +1, plen +1."""
    g = S.item_geometry(tok, prow, False, srows)
    if g["span"][0] is not None:
        g["span"] = [g["span"][0] + 1, g["span"][1] + 1]
    g["idsA"] = [tok.bos_token_id] + g["idsA"]
    g["idsB"] = [tok.bos_token_id] + g["idsB"]
    g["plen"] = len(g["idsA"])
    return g


# ====================================================================== prep / screen / select
def prep(cfg, out_dir, settle_dir):
    """Candidates in run 1's order: pool -> SETTLE pre-filter (McLeish k32 natural correct_v2 at cap 4096 on A) ->
    random.Random('k17_gen:20261007') shuffle of the sorted ids (steer_gen.shuffled) -> McLeish geometry checks."""
    from prod.tasks import rows as task_rows
    pool, n_core_rows = S.core_pool()
    settle = S.settle_cells(os.path.join(settle_dir, cfg["settle"]), cap=cfg["settle_prefilter_cap"])
    srows = {int(r["idx"]): r for r in task_rows(S.TASK_A)}
    tok = load_tokenizer()
    by = {int(r["idx"]): r for r in pool}
    order = [i for i in S.shuffled(by) if settle.get(i)]
    items, bad = [], {}
    for idx in order:
        pr = by[idx]
        g = geometry(tok, pr, srows)
        ps = pr["tok"]["mcleish"]          # the pool's McLeish span and length count the BOS
        g["ok"]["matches_pool_span"] = list(g["span"]) == list(ps["span_A"]) and g["plen"] == ps["len_A"]
        fails = [kk for kk, v in g["ok"].items() if not v]
        if fails:
            bad[idx] = fails
            continue
        s0, s1 = g["span"]
        items.append({"idx": idx, "var_name": pr["var_name"], "value_A": pr["value_A"], "value_B": pr["value_B"],
                      "gold_A": str(pr["gold_A"]), "gold_B": str(pr["gold_B"]), "span": [s0, s1], "plen": g["plen"],
                      "n_span_tokens": s1 - s0, "offset_from_end": g["plen"] - s1,
                      "value_first_digit_differs": bool(pr["value_first_digit_differs"]),
                      "first_digit_differs": bool(pr["first_digit_differs"]),
                      "gold_B_chain_status": pr["gold_B_chain_status"], "n_digits": pr["n_digits"]})
    res = {"setting": cfg["name"], "n_pool_rows": n_core_rows, "n_pool_problems": len(pool),
           "n_settle_correct": len(order), "n_geometry_ok": len(items), "geometry_fail": bad,
           "pool_sha256": S.sha256_file(S.POOL_DEFAULT), "settle_file": cfg["settle"],
           "settle_sha256": S.sha256_file(os.path.join(settle_dir, cfg["settle"])), "items": items}
    os.makedirs(out_dir, exist_ok=True)
    S.write_json(os.path.join(out_dir, "prep.json"), res)
    return res


def screen_dirs(root, side):
    return sorted(d for d in os.listdir(root) if d.startswith("screen_%s_" % side)
                  and os.path.isdir(os.path.join(root, d)))


def load_screen(root, side):
    out = {}
    for d in screen_dirs(root, side):
        out.update(S.load_cells(os.path.join(root, d)))
    return out


def shortest_eval_rows(tok, exclude, n):
    """steer_gen.shortest_eval_rows with McLeish's encoding (BOS included, as in every plen here)."""
    from prod.tasks import rows as task_rows, split_labels
    from prod import config as cfgmod
    rws = task_rows(S.TASK_A)
    lab = split_labels(S.TASK_A, len(rws), cfgmod.SPLIT_SEED, cfgmod.N_CAL)
    cand = [r for r, l in zip(rws, lab) if l == "eval" and int(r["idx"]) not in exclude]
    cand = sorted(cand, key=lambda r: len(r["input"]))[:max(4 * n, 40)]
    pr = S.prompt_texts(tok, S.TASK_A, cand, False)
    pl = [(int(r["idx"]), len(enc(tok, p))) for r, p in zip(cand, pr)]
    return sorted(pl, key=lambda t: (t[1], t[0]))[:n]


def select(cfg, root, out_dir, n_scored=None, n_screened=None):
    """steer_gen.select_a with both screens run here (McLeish x0 differs from SETTLE's stream, so SETTLE's A cells
    are not this harness's clean A). scored = items in prep order with own answer = gold_A on A and = gold_B on B
    (the first n_scored, or every one); sentinels placed by prompt length from the remaining prep items (CL3);
    shift rows = the 8 shortest GSM8K eval prompts outside both sets."""
    from prod.tasks import num_eq
    pr = json.load(open(os.path.join(root, "prep.json"), encoding="utf-8"))
    sA, sB = load_screen(root, "A"), load_screen(root, "B")
    items = pr["items"] if n_screened is None else pr["items"][:n_screened]
    scored, fails = [], {"A_wrong": 0, "B_wrong": 0, "not_screened": 0}
    for it in items:
        a, b = sA.get(it["idx"]), sB.get(it["idx"])
        if a is None or b is None:
            fails["not_screened"] += 1
            continue
        okA = a.get("trace_answer") is not None and num_eq(a["trace_answer"], it["gold_A"])
        okB = b.get("trace_answer") is not None and num_eq(b["trace_answer"], it["gold_B"])
        fails["A_wrong"] += int(not okA)
        fails["B_wrong"] += int(not okB)
        if okA and okB:
            scored.append(it)
    n_pass = len(scored)
    if n_scored:
        scored = scored[:int(n_scored)]
    used = {it["idx"] for it in scored}
    cand = [(it["idx"], it["plen"]) for it in pr["items"] if it["idx"] not in used]
    sent = S.place_sentinels([it["plen"] for it in scored], cand, cfg["n_sent"])
    sent_ids = {kk for kk, _p in sent}
    tok = load_tokenizer()
    shift = shortest_eval_rows(tok, used | sent_ids, cfg["n_shift"])
    rep = S.sentinel_report([it["plen"] for it in scored], [p for _k, p in sent])
    lay = S.layout_change([it["plen"] for it in scored] + [p for _k, p in sent], [p for _k, p in shift])
    nA_ok = sum(1 for it in items if sA.get(it["idx"]) is not None and sB.get(it["idx"]) is not None
                and sA[it["idx"]].get("trace_answer") is not None
                and num_eq(sA[it["idx"]]["trace_answer"], it["gold_A"]))
    nB_given_A = n_pass
    sel = {"setting": cfg["name"], "n_screened_A": sum(1 for it in items if it["idx"] in sA),
           "n_screened_B": sum(1 for it in items if it["idx"] in sB), "filter_fails": fails,
           "screen_A_source": "screen_A job (this harness: fixed per-item x0)",
           "n_pass_both": n_pass, "n_A_ok": nA_ok, "swap_ceiling_B_given_A": round(nB_given_A / float(max(1, nA_ok)), 4),
           "n_scored": len(scored), "scored": scored, "sentinels": [{"idx": kk, "plen": p} for kk, p in sent],
           "shift": [{"idx": kk, "plen": p} for kk, p in shift], "sentinel_layout": rep, "r20_layout_change": lay,
           "selection_rule": "random.Random('k17_gen:20261007') shuffle of sorted ids after the SETTLE k32 pre-filter; "
                             + ("first %d passing both filters" % n_scored if n_scored else "every problem passing both filters")}
    os.makedirs(out_dir, exist_ok=True)
    if out_dir != root:
        S.write_json(os.path.join(out_dir, "prep.json"), pr)
    S.write_json(os.path.join(out_dir, "items.json"), sel)
    return sel


# ====================================================================== jobs through G.main
def build_row_infos(tok, items, rowdef):
    """steer_gen.build_row_infos with McLeish's encoding (BOS) and the item's BOS-shifted span."""
    from prod.tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(S.TASK_A)}
    infos = {}
    for it in items:
        ra = srows[int(it["idx"])]
        rb = dict(ra, input=it["question_B"])
        pA, pB = S.prompt_texts(tok, S.TASK_A, [ra, rb], False)
        a_ids, b_ids = enc(tok, pA), enc(tok, pB)
        s0, s1 = it["span"]
        if len(a_ids) != it["plen"] or len(b_ids) != it["plen"]:
            raise RuntimeError("prompt length mismatch for idx %s" % it["idx"])
        if a_ids[:s0] != b_ids[:s0] or a_ids[s1:] != b_ids[s1:] or a_ids[s0:s1] == b_ids[s0:s1]:
            raise RuntimeError("A/B ids do not differ exactly in the span for idx %s" % it["idx"])
        prompt_ids = a_ids if rowdef.get("prompt", "A") == "A" else b_ids
        if rowdef.get("self_donor"):
            b_ids = list(a_ids)
        infos[str(it["idx"])] = {"plen": len(a_ids), "span": (s0, s1), "a_ids": prompt_ids, "b_ids": b_ids,
                                 "edits": list(rowdef.get("edits") or []) if rowdef.get("prompt", "A") == "A" else [],
                                 "lcdl": None, "donor_range": (s0, len(a_ids)), "t0": s0}
    return infos


def run_gmain(task, only_ids, out_dir, horizon, steer, kv, mem_fraction=None, batch_width=16, adapter_cls=None):
    """steer_gen.run_gmain for McLeish: G.main with SteeredMcLeishAdapter swapped in for prod.generate.get
    (restored in finally). Resumable after a kill: G.main skips cells already written; decode_log.jsonl is appended.
    After a recorded failure the folder starts clean (steer_gen.set_aside_failed)."""
    import prod.generate as G
    os.makedirs(out_dir, exist_ok=True)
    S.set_aside_failed(out_dir)
    only =os.path.join(out_dir, "only_rows.json")
    with open(only, "w", encoding="utf-8") as f:
        json.dump(sorted(int(i) for i in only_ids), f)
    steer.decode_log_path = os.path.join(out_dir, "decode_log.jsonl")
    orig = G.get
    holder = {}
    cls = adapter_cls or SteeredMcLeishAdapter

    def _get(name, adapter_dir=None, **kw):
        if name != MODEL:
            raise KeyError(name)
        ad = cls(name, steer=steer, kv=kv, **kw)
        holder["ad"] = ad
        return ad
    argv = ["--model", MODEL, "--task", task, "--k", str(K if steer.k is None else steer.k), "--only-rows", only,
            "--out", out_dir, "--caps", str(horizon), "--horizon", str(horizon), "--no-extra-caps",
            "--batch-width", str(int(batch_width))]
    if mem_fraction:
        argv += ["--mem-fraction", str(mem_fraction)]
    G.get = _get
    code = 0
    t0 = time.time()
    try:
        G.main(argv)
    except SystemExit as e:
        code = int(e.code or 0)
    finally:
        G.get = orig
        if holder.get("ad") is not None:
            holder["ad"].steer.remove()
    summ = {"argv": [a if not a.startswith("/") else os.path.basename(a) for a in argv], "exit": code,
            "seconds": round(time.time() - t0, 1),
            "stats": {"%s|%s" % kk: v for kk, v in steer.stats.items()}, "n_forwards": steer.n_forwards}
    S.write_json(os.path.join(out_dir, "steer_job.json"), summ)
    return summ


def job(cfg, root, row, mem_fraction=None, screen=None, lo=None, hi=None, adapter_cls=None, horizon=None):
    """One row job (or a screen chunk: screen = 'A' | 'B', prep items [lo, hi))."""
    rdefs = rows_raven(cfg["k"])
    pool = {int(r["idx"]): r for r in S.core_pool()[0]}
    if screen:
        pr = json.load(open(os.path.join(root, "prep.json"), encoding="utf-8"))
        items = pr["items"][lo:hi]
        rowdef = rdefs["r00"] if screen == "A" else rdefs["r01"]
        name = "screen_%s_%04d_%04d" % (screen, lo, min(hi, len(pr["items"])))
        all_ids = [it["idx"] for it in items]
        scored = items
    else:
        sel = json.load(open(os.path.join(root, "items.json"), encoding="utf-8"))
        rowdef = rdefs[row]
        name = row
        scored = sel["scored"]
        all_ids = [it["idx"] for it in scored] + [s_["idx"] for s_ in sel["sentinels"]]
        if rowdef.get("shift"):
            all_ids += [s_["idx"] for s_ in sel["shift"]]
    scored = [dict(it, question_B=pool[int(it["idx"])]["question_B"]) for it in scored]
    out_dir = os.path.join(root, name)
    tok = load_tokenizer()
    steer = RavenSteerHooks(cfg["k"])
    steer.rows = build_row_infos(tok, scored, rowdef)
    kv = any(e["op"] == "kv" for e in rowdef.get("edits") or [])
    if rowdef.get("prompt") == "B":
        from prod.tasks import rows as task_rows
        srows = {int(r["idx"]): r for r in task_rows(S.TASK_A)}
        sids = {x["idx"] for x in scored}
        brows = []
        for it in scored:
            r = dict(srows[int(it["idx"])])
            r["input"] = it["question_B"]
            r["target"] = str(it["gold_B"])
            brows.append(r)
        brows += [dict(srows[int(i)]) for i in all_ids if i not in sids]
        rows_p = os.path.join(root, "rows_%s_B.jsonl" % name)
        S.write_jsonl(rows_p, brows)
        task = "%s_cfB_%s_%s" % (S.TASK_A, cfg["name"], name)
        S.register_task(task, S.TASK_A, rows_p)
    else:
        task = S.TASK_A
    return run_gmain(task, all_ids, out_dir, horizon or cfg["horizon"], steer, kv, mem_fraction=mem_fraction,
                     batch_width=cfg.get("batch_width", 16), adapter_cls=adapter_cls)


# ====================================================================== analysis, results.csv, gates
ROWS_CSV = {   # row -> (arm, site, loop, strength, random row at the same site), K = 32 (cluster/extract/gen.py form)
    "r00": ("clean_A", "none", "", "", None), "r01": ("swap_B", "none", "", "", None),
    "r02": ("emb_patch", "emb|span", 1, 1.0, None), "r03": ("patch", "post|span", 1, 1.0, "r08"),
    "r19": ("patch", "post|span", 2, 1.0, None), "r04": ("patch", "post|span", K - 1, 1.0, "r14"),
    "r05": ("patch", "post|span", "1-%d" % K, 1.0, "r15"), "r06": ("patch", "post|last_prompt", 1, 1.0, "r16"),
    "r07": ("patch", "post|last_prompt", K, 1.0, None), "r08": ("random_s0", "post|span", 1, "norm-matched", None),
    "r17": ("random_s1", "post|span", 1, "norm-matched", None), "r18": ("random_s2", "post|span", 1, "norm-matched", None),
    "r14": ("random_s0", "post|span", K - 1, "norm-matched", None),
    "r15": ("random_s0", "post|span", "1-%d" % K, "norm-matched", None),
    "r16": ("random_s0", "post|last_prompt", 1, "norm-matched", None),
    "r09": ("complete_patch", "post+pre_out+kv|span_to_end", "0-1", 1.0, None),
    "r11": ("complete_patch", "post+pre_out+kv|span_to_end", "0-%d" % (K - 1), 1.0, None),
    "r10": ("complete_patch", "post+pre_out+kv|span_to_end", "0-%d" % K, 1.0, None),
    "r12": ("patch", "post|span_to_end", 1, 1.0, None), "r13": ("kv_patch", "kv|span", 1, 1.0, None),
    "r20": ("dup_clean_shifted", "all", "0-%d" % K, 0.0, None), "r20n": ("neutral_hooks", "all", "0-%d" % K, 0.0, None),
    "r21": ("self_donor", "post|span", 1, 1.0, None),
}


def results_rows(an, model=MODEL, dataset="gsm8k", only=None, r90=None):
    from cluster.extract import common
    r90 = r90 or {}
    pc = ((an["rows"].get("r01") or {}).get("gold_B") or [None])[0]
    out = []
    for r, cell in an["rows"].items():
        if r not in ROWS_CSV or (only and r != only):
            continue
        arm, site, loop, strength, rnd = ROWS_CSV[r]
        nr = (cell.get("neither") or [None])[0]
        for mname in ("gold_B", "gold_A", "neither"):
            v = cell.get(mname)
            if not v:
                continue
            rv = ((an["rows"].get(rnd) or {}).get(mname) or [None])[0] if rnd else None
            v90 = (r90.get(r) or {}).get(mname)
            out.append(common.row(
                grid="A", card="generation placement", script="steer_gen_raven.py", model=model, dataset=dataset,
                arm="%s:%s" % (r, arm), site=site, loop=loop, strength=strength, metric=mname, endpoint="parsed_answer",
                item_set="own", n_items=v[3], n_clusters=v[4], estimate=v[0], neither_rate=nr, ci95_lo=v[1],
                ci95_hi=v[2], ci90_lo=v90[1] if v90 else None, ci90_hi=v90[2] if v90 else None,
                ci90_method="recomputed" if v90 else "missing_95_only", random_control_estimate=rv,
                positive_control_value=pc if mname == "gold_B" else None,
                source_key="analysis.json:" + json.dumps(["rows", r, mname])))
    return out


def analyze(cfg, root, tok=None):
    rows = [r for r in list(rows_raven(cfg["k"])) if os.path.isdir(os.path.join(root, r))]
    order = [r for r in cfg["rows"] + ["r20n"] if r in rows]
    an = S.analyze_a(dict(cfg, rows=order, rows_ext=[]), root, rows=order, tok=tok)
    if "r20n" in an["rows"]:
        c = an["rows"]["r20n"]
        an["neutral_hooks"] = {"token_identical_vs_r00": c.get("token_identical_vs_r00"),
                               "first_divergence_median_vs_r00": c.get("first_divergence_median_vs_r00"),
                               "outcome_discordance_vs_r00": None}
        sc = json.load(open(os.path.join(root, "items.json"), encoding="utf-8"))["scored"]
        ids = [int(it["idx"]) for it in sc]
        ca, cb = S.load_cells(os.path.join(root, "r20n")), S.load_cells(os.path.join(root, "r00"))
        n = [i for i in ids if i in ca and i in cb]
        an["neutral_hooks"]["outcome_discordance_vs_r00"] = round(
            sum(1 for i in n if bool(ca[i].get("trace_correct")) != bool(cb[i].get("trace_correct"))) / float(max(1, len(n))), 4)
    S.write_json(os.path.join(root, "analysis.json"), an)
    from cluster.extract import common
    try:     # 90 percent intervals recomputed from the per-item cells (cluster/extract/gen.py recompute, run-1 rows)
        from cluster.extract import gen as XG
        r90, _h = XG.recompute(root, S.__file__, True)
    except Exception as e:                                    # noqa: BLE001
        print("[warn] 90 percent recompute failed: %r" % (e,), flush=True)
        r90 = {}
    common.write_results(results_rows(an, r90=r90), os.path.join(root, "results.csv"))
    for r in order:
        common.write_results(results_rows(an, only=r, r90=r90), os.path.join(root, r, "results.csv"))
    return an


def gates(an, sel):
    """Run-1 gates on the gate subset (plan section 11, CL3, U5): swap ceiling, harness identity r02 vs r01
    (token-identical >= 0.95), neutral-hook token identity r20n vs r00 (= 1.0: same layout, zero edits), floors
    r20 / r21 outcome discordance vs r00 (<= 0.04), sentinel discordance per job (<= 0.10)."""
    rw = an["rows"]
    g = {}
    g["swap_ceiling"] = {"screen_B_given_A": sel.get("swap_ceiling_B_given_A"),
                         "r01_gold_B": (rw.get("r01", {}).get("gold_B") or [None])[0]}
    g["swap_ceiling"]["pass"] = g["swap_ceiling"]["r01_gold_B"] is not None and g["swap_ceiling"]["r01_gold_B"] >= 0.9
    ti = rw.get("r02", {}).get("token_identical_vs_r01")
    g["harness_identity_r02"] = {"token_identical_vs_r01": ti,
                                 "gold_B": (rw.get("r02", {}).get("gold_B") or [None])[0],
                                 "pass": ti is not None and ti >= 0.95}
    tn = rw.get("r20n", {}).get("token_identical_vs_r00")
    g["neutral_hooks_r20n"] = {"token_identical_vs_r00": tn, "pass": tn is not None and tn >= 1.0}
    for fl in ("r20", "r21"):
        d = an.get("%s_floor" % fl) or {}
        v = d.get("outcome_discordance")
        g["floor_%s" % fl] = {"outcome_discordance": v, "token_identical_vs_r00": d.get("token_identical_vs_r00"),
                              "neither": (d.get("neither") or [None])[0], "pass": v is not None and v <= 0.04}
    sd = {r: (c.get("sentinel_discordance") or [None])[0] for r, c in rw.items() if r != "r00"}
    g["sentinel_discordance"] = {"per_job": sd, "max": max([v for v in sd.values() if v is not None] or [None]),
                                 "pass": all(v is not None and v <= 0.10 for v in sd.values()) and bool(sd)}
    g["all_pass"] = all(v.get("pass") for v in g.values() if isinstance(v, dict))
    return g


# ====================================================================== forward-level checks (real model or fake)
def check_suite(ad, steer, items):
    """Batch-1, no-cache forward identities and a mixed-batch zero-edit decode (G1-type):
    emb patch == B forward (bitwise); complete patch @1, @K-1, @K == B forward; self-donor == A forward (bitwise);
    zero edits on every hook kind == no rows (bitwise logits, identical 16 decode tokens in a mixed padded batch);
    position purity: cached decode of 8 tokens with an O1 Llast@1 span edit == argmax of full no-cache recomputes."""
    rd = rows_raven(steer.k)
    res = {"items": [it["idx"] for it in items], "cases": {}}
    ok_all = True

    def setrows(rowname):
        steer.rows = build_row_infos_ids(items, rd[rowname])
        steer.donors = {}

    def z1(seq, key):
        return ad.forward_rows([seq], [key])[0]
    for name, ref, rowname in (("emb_patch_eq_B", "B", "r02"), ("complete1_eq_B", "B", "r09"),
                               ("completeKm1_eq_B", "B", "r11"), ("completeK_eq_B", "B", "r10"),
                               ("self_donor_eq_A", "A", "r21"), ("zero_edit_eq_A", "A", "r20n"),
                               ("source1_differs_A", "A", "r03")):
        setrows(rowname)
        dz = []
        for it in items:
            za = z1(it["a_ids"], it["idx"])
            steer.rows_saved, steer.rows = steer.rows, {}
            zr = z1(it["a_ids"] if ref == "A" else it["b_ids"], it["idx"])
            steer.rows = steer.rows_saved
            dz.append(float((za - zr).abs().max()))
        if name == "source1_differs_A":
            passed = True
            res["cases"][name] = {"max_abs_dz": dz, "note": "reported only (a loop-1 span edit may be erased)"}
        else:
            passed = all(x == 0.0 for x in dz)
            res["cases"][name] = {"max_abs_dz": dz, "pass": passed}
        ok_all = ok_all and passed
    # mixed padded batch: rows absent vs rows with zero edits on every hook kind, prefill logits and 16 tokens
    seqs = [it["a_ids"] for it in items]
    keys = [it["idx"] for it in items]
    steer.rows = {}
    steer.donors = {}
    z0 = ad.forward_rows(seqs, keys)
    g0, _ = ad.decode(seqs, 16, steer.k, row_ids=keys)
    setrows("r20n")
    z1_ = ad.forward_rows(seqs, keys)
    g1, _ = ad.decode(seqs, 16, steer.k, row_ids=keys)
    zero_ok = bool(torch.equal(z0, z1_)) and g0 == g1
    res["cases"]["zero_edit_mixed_batch"] = {"prefill_bitwise": bool(torch.equal(z0, z1_)), "tokens_equal": g0 == g1,
                                             "pass": zero_ok}
    ok_all = ok_all and zero_ok
    # position purity: cached decode with an edit vs no-cache recompute of prefix + generated with the same edit
    setrows("r03")
    agree, n = 0, 0
    margins = []
    for it in items[:2]:
        g, _ = ad.decode([it["a_ids"]], 8, steer.k, row_ids=[it["idx"]])
        seq = list(it["a_ids"])
        for t in g[0]:
            z = ad.forward_rows([seq], [it["idx"]])[0]
            top = z.topk(2).values
            margins.append(float(top[0] - top[1]))
            agree += int(int(z.argmax()) == int(t))
            n += 1
            seq.append(int(t))
    res["cases"]["cache_vs_recompute_r03"] = {"argmax_agree": agree, "n": n, "pass": agree >= n - n // 8,
                                              "recompute_top2_margins_min": min(margins) if margins else None}
    ok_all = ok_all and agree >= n - n // 8
    steer.rows = {}
    steer.donors = {}
    res["pass"] = ok_all
    return res


def build_row_infos_ids(items, rowdef):
    """Row infos from items that already hold McLeish ids (a_ids, b_ids with BOS) and the BOS-shifted span."""
    infos = {}
    for it in items:
        s0, s1 = it["span"]
        b_ids = list(it["a_ids"]) if rowdef.get("self_donor") else list(it["b_ids"])
        infos[str(it["idx"])] = {"plen": len(it["a_ids"]), "span": (s0, s1), "a_ids": list(it["a_ids"]),
                                 "b_ids": b_ids, "edits": list(rowdef.get("edits") or []), "lcdl": None,
                                 "donor_range": (s0, len(it["a_ids"])), "t0": s0}
    return infos


def check_items(tok, n=4):
    from prod.tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(S.TASK_A)}
    pool = {int(r["idx"]): r for r in S.core_pool()[0]}
    out = []
    for idx in S.shuffled(pool):
        g = geometry(tok, pool[idx], srows)
        if all(g["ok"].values()):
            out.append({"idx": idx, "a_ids": g["idsA"], "b_ids": g["idsB"], "span": g["span"], "plen": g["plen"]})
        if len(out) >= n:
            break
    return out


# ====================================================================== CPU self-test on a tiny McLeish
def tiny_mcleish(snapdir, seed=0, dtype=torch.bfloat16):
    """Random-weight McLeish from the snapshot's own modeling code, small dims (k17_public.tiny_raven 4161-4176)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(snapdir, trust_remote_code=True, local_files_only=True)
    for kk, v in dict(n_embd=64, n_heads=4, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
                      head_dim=16).items():
        setattr(cfg, kk, v)
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()
    with torch.no_grad():
        for _n, p in m.named_parameters():
            p.copy_(1.0 + 0.1 * torch.randn_like(p) if p.dim() == 1 else 0.05 * torch.randn_like(p))
    m = m.to(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def fake_adapter_cls(snapdir):
    class FakeMcLeish(SteeredMcLeishAdapter):
        """SteeredMcLeishAdapter on a tiny random McLeish (CPU): prod's load path minus the hub weights."""

        def load(self, strict_patch=False):
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(snapdir, local_files_only=True)
            self.model = tiny_mcleish(snapdir)
            patch = R.install_attention_patch(self.model, strict=False)
            self._info = {"attention_patch": patch, "fake": "tiny random McLeish (CPU self-test)"}
            self.max_positions = None
            self.after_load()
            return self
    return FakeMcLeish


def selftest(out_dir, snapdir):
    """CPU: tiny random McLeish. (1) forward-level identities and zero-edit decode (check_suite); (2) x0 table
    position purity and pad independence; (3) end-to-end G.main jobs (r00, r01, r02, r20n, r10, r03) on 3 items
    with sentinels at horizon 24, analyze_a, results.csv, gates read."""
    os.environ.setdefault("PROD_DEV", "cpu")
    res = {}
    Fk = fake_adapter_cls(snapdir)
    steer = RavenSteerHooks(K)
    ad = Fk(MODEL, steer=steer, kv=True)
    ad.load()
    ad.set_depth(K)
    tok = load_tokenizer()
    its = check_items(tok, n=3)
    t0 = time.time()
    res["check_suite"] = check_suite(ad, steer, its)
    # x0: a row's table rows at its absolute positions, independent of padding and batch
    t = ad.x0_rows([7, 9], [0, 3], 10)
    res["x0_pad_independent"] = bool(torch.equal(t[1, 3:], ad.table(9)[:7])) and bool(torch.equal(t[0], ad.table(7)[:10]))
    tt = x0_table(ad.model, 7, 64)
    res["x0_prefix_stable"] = bool(torch.equal(tt[:10].to(ad.dtype), ad.table(7)[:10]))
    steer.remove()
    res["check_seconds"] = round(time.time() - t0, 1)
    # end to end through G.main on 3 scored items + 2 sentinels + 2 shift rows
    pool = {int(r["idx"]): r for r in S.core_pool()[0]}
    from prod.tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(S.TASK_A)}
    items = []
    for it in its:
        pr = pool[it["idx"]]
        items.append({"idx": it["idx"], "gold_A": str(pr["gold_A"]), "gold_B": str(pr["gold_B"]), "span": it["span"],
                      "plen": it["plen"], "n_span_tokens": it["span"][1] - it["span"][0],
                      "value_first_digit_differs": bool(pr["value_first_digit_differs"]),
                      "gold_B_chain_status": pr["gold_B_chain_status"]})
    others = [i for i in S.shuffled(pool) if i not in {x["idx"] for x in items}][:4]
    sel = {"scored": items, "sentinels": [{"idx": i} for i in others[:2]], "shift": [{"idx": i} for i in others[2:]]}
    root = os.path.join(out_dir, "e2e")
    os.makedirs(root, exist_ok=True)
    S.write_json(os.path.join(root, "items.json"), sel)
    cfg = S.setting_cfg(SETTING)
    jobs = {}
    for r in ("r00", "r01", "r02", "r20n", "r10", "r03", "r21", "r20"):
        jobs[r] = job(cfg, root, r, adapter_cls=Fk, horizon=24)
    res["e2e_exit"] = {r: j["exit"] for r, j in jobs.items()}
    an = analyze(cfg, root, tok=tok)
    res["e2e_rows"] = {r: {kk: c.get(kk) for kk in ("gold_B", "neither", "token_identical_vs_r00",
                                                    "token_identical_vs_r01", "sentinel_discordance")}
                       for r, c in an["rows"].items()}
    res["e2e_gates_read"] = gates(an, {"swap_ceiling_B_given_A": None})
    res["e2e_expect"] = {
        "r02_token_identical_vs_r01": an["rows"]["r02"].get("token_identical_vs_r01") == 1.0,
        "r10_token_identical_vs_r01": an["rows"]["r10"].get("token_identical_vs_r01"),
        "r20n_token_identical_vs_r00": an["rows"]["r20n"].get("token_identical_vs_r00") == 1.0,
        "r21_token_identical_vs_r00": an["rows"]["r21"].get("token_identical_vs_r00"),
        "results_csv": os.path.exists(os.path.join(root, "results.csv")) and os.path.exists(os.path.join(root, "r02", "results.csv"))}
    res["pass"] = bool(res["check_suite"]["pass"] and res["x0_pad_independent"] and res["x0_prefix_stable"]
                       and all(v == 0 for v in res["e2e_exit"].values())
                       and res["e2e_expect"]["r02_token_identical_vs_r01"]
                       and res["e2e_expect"]["r20n_token_identical_vs_r00"] and res["e2e_expect"]["results_csv"])
    S.write_json(os.path.join(out_dir, "selftest.json"), res)
    return res


# ====================================================================== hash line
def hash_line(cfg, root, settle_dir):
    """plan, config, code, data sha256 (no host name)."""
    plan = os.environ.get("K17_PLAN")
    cfg_s = json.dumps(cfg, sort_keys=True).encode("utf-8")
    d = {"plan_sha256": S.sha256_file(plan) if plan and os.path.exists(plan) else None,
         "config_sha256": hashlib.sha256(cfg_s).hexdigest(),
         "code_sha256": {"steer_gen_raven.py": S.sha256_file(os.path.abspath(__file__)),
                         "steer_gen.py": S.sha256_file(S.__file__.replace(".pyc", ".py"))},
         "data_sha256": {"pairs_gsm8k.jsonl": S.sha256_file(S.POOL_DEFAULT),
                         cfg["settle"]: S.sha256_file(os.path.join(settle_dir, cfg["settle"])) if settle_dir else None},
         "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = os.path.join(root, "hash_line.json")
    if not os.path.exists(p):
        S.write_json(p, d)
    return json.load(open(p, encoding="utf-8"))


# ====================================================================== main
def main(argv=None):
    p = argparse.ArgumentParser(prog="steer_gen_raven")
    p.add_argument("mode", choices=["selftest", "checks", "prep", "screen", "select", "run", "analyze", "gates", "hash"])
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=None, help="setting root holding prep.json and the screens (gate / select)")
    p.add_argument("--row", default=None)
    p.add_argument("--side", default=None)
    p.add_argument("--lo", type=int, default=0)
    p.add_argument("--hi", type=int, default=None)
    p.add_argument("--n-scored", type=int, default=None)
    p.add_argument("--n-screened", type=int, default=None)
    p.add_argument("--settle-dir", default=None)
    p.add_argument("--mem-fraction", type=float, default=None)
    p.add_argument("--snap", default=None, help="selftest: McLeish snapshot folder (modeling code + tokenizer)")
    p.add_argument("--n", type=int, default=4)
    a = p.parse_args(argv)
    cfg = S.setting_cfg(SETTING)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "selftest":
        r = selftest(a.out, a.snap)
        print(json.dumps(r, default=str)[:6000])
        return 0 if r["pass"] else 1
    S.write_json(os.path.join(a.out, "setting.json"), cfg)
    if a.mode == "hash":
        print(json.dumps(hash_line(cfg, a.out, a.settle_dir)))
        return 0
    if a.mode == "checks":
        kw = {"mem_fraction": a.mem_fraction} if a.mem_fraction else {}
        steer = RavenSteerHooks(K)
        ad = SteeredMcLeishAdapter(MODEL, steer=steer, kv=True, **kw)
        ad.load()
        ad.set_depth(K)
        its = check_items(ad.tok, n=a.n)
        t0 = time.time()
        r = check_suite(ad, steer, its)
        r["seconds"] = round(time.time() - t0, 1)
        r["env"] = S.env_block(None)
        r["env"].pop("steer_gen_sha256", None)
        r["env"]["file_sha16"] = dict(ad.file_sha)
        S.write_json(os.path.join(a.out, "checks_mcl.json"), r)
        print(json.dumps({kk: v for kk, v in r.items() if kk != "env"}, default=str))
        return 0 if r["pass"] else 1
    if a.mode == "prep":
        r = prep(cfg, a.out, a.settle_dir)
        print(json.dumps({kk: v for kk, v in r.items() if kk != "items"}, default=str))
        return 0
    if a.mode == "screen":
        pr = json.load(open(os.path.join(a.out, "prep.json"), encoding="utf-8"))
        hi = len(pr["items"]) if a.hi is None else min(a.hi, len(pr["items"]))
        codes = []
        for side in (a.side,) if a.side else ("A", "B"):
            summ = job(cfg, a.out, None, a.mem_fraction, screen=side, lo=a.lo, hi=hi)
            print(json.dumps(summ))
            codes.append(S.job_exit(summ))
        return next((c for c in codes if c), 0)
    if a.mode == "select":
        r = select(cfg, a.root or a.out, a.out, n_scored=a.n_scored, n_screened=a.n_screened)
        print(json.dumps({kk: v for kk, v in r.items() if kk != "scored"}, default=str))
        return 0
    if a.mode == "run":
        summ = job(cfg, a.out, a.row, a.mem_fraction)
        print(json.dumps(summ))
        return S.job_exit(summ)
    if a.mode == "analyze":
        an = analyze(cfg, a.out, tok=load_tokenizer())
        print(json.dumps(an, default=str)[:6000])
        return 0
    if a.mode == "gates":
        an = json.load(open(os.path.join(a.out, "analysis.json"), encoding="utf-8"))
        sel = json.load(open(os.path.join(a.out, "items.json"), encoding="utf-8"))
        g = gates(an, sel)
        S.write_json(os.path.join(a.out, "gates.json"), g)
        print(json.dumps(g))
        return 0 if g["all_pass"] else 1
    raise SystemExit("unknown mode %s" % a.mode)


if __name__ == "__main__":
    sys.exit(main())
