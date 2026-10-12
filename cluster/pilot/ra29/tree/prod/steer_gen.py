"""B-K17-GEN: steering under generation on Ouro, through SETTLE's own generation path.

Plan of record: loop-signal/notes/k17_gen_plan.md (sections 1-18, amendments A1-A8, Critic log CL1-CL9).

What this module adds to prod (no prod file is edited):
  * `SteerHooks`: instance hooks on an Ouro model (removable; after `remove()` the model is the fresh load).
      H0 top-level pre/post hook (row geometry, assertions, LoopCD-Logits), H1 pre-norm loop end (`pre@j`),
      H2 post-norm loop end (`post@j`, the k15/k16 Ouro Llast@j), H3 layer-0 input at loop 1 (`emb@1`, the
      token embedding), H4 k_proj / v_proj outputs per call (loop, layer) for KV patches.
  * `SteeredOuroAdapter(OuroAdapter)`: prod's adapter with the hooks installed; `decode` hands the batch's row
      ids to the hooks, so prod.generate.main (G.main) runs unchanged with edits.
  * Placement rows (arm a), LoopCD rows (arm b), the teacher-forced loop-2 test (CL1), the answer-site test
      (CL4 `anschain`), the presence read-out (CL4), the gates G1-G5 and the analysis.

LoopCD = arXiv 2610.02185 v1 (Liu et al., "Decoding Looped Transformers Better for (Almost) Free"):
  Eq.(1)  h_0 -f-> h_1 -f-> ... -f-> h_R, z_R = lm_head(coda(h_R)); "lm_head consists of the final layer
          normalization and vocabulary projection". Ouro has no prelude or coda (App A.2): the hidden-state
          update is "applied before the normalization and head", i.e. on the PRE-norm loop-end state.
  Eq.(2)  LoopCD-Hidden: h' = h_R + w (h_R - h_1), w >= 0, fixed w (Sec 3.2).
  Eq.(3)  LoopCD-Logits: z' = z_R + w (z_R - z_1), z_1 = lm_head(norm(h_1)), w >= 0.
  Eq.(4)  adaptive w = w_max [1 - (p_R,(1) - p_R,(2))], the two largest probabilities of softmax(z_R), T = 1.
  App A.3 (1+w) log p_R - w log p_1 = z' + const, so the processor acts before any temperature.
  Reference loop h_1 for Ouro (deterministic initial state; Sec 5.1, Table A4 caption).
  Table A4 strengths: generation Ouro-1.4B fixed 0.3 / adaptive cap 0.5; math reasoning Ouro-1.4B-Thinking
  0.5 / 1.0.

Ouro facts used (modeling_ouro.py 574fa66, md5 5a0c3755): the loop is `for current_ut in range(total_ut_steps)`
over the decoder layers with kwarg `current_ut`, then `hidden = self.norm(hidden)` once per loop (592-608);
attention writes K/V at cache index current_ut * L + layer_idx (320); read-out with exit_at_step is
lm_head(hidden_states_list[k-1]) (776-779); lm_head has no bias.

Python 3.9; no PEP 604 unions, no `match`.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from .models.ouro import OuroAdapter, leftpad, make_static_cache_cls

VERSION = "k17gen-1"
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG_DIR)
# RA-18 dataset switch for arm (a): K17_DATASET = gsm8k (default, VarBench pairs, eval split only) or svamp
# (data_steer/svamp/pairs_svamp.jsonl, SETTLE's svamp task; both SETTLE splits, since arm (a) tunes nothing).
DATASET = os.environ.get("K17_DATASET", "gsm8k")
_DATASETS = {"gsm8k": {"pool": ("varbench", "pairs_gsm8k.jsonl"), "task": "gsm8k", "splits": ("eval",)},
             "svamp": {"pool": ("svamp", "pairs_svamp.jsonl"), "task": "svamp", "splits": ("eval", "cal")}}
POOL_DEFAULT = os.path.join(ROOT, "data_steer", *_DATASETS[DATASET]["pool"])
TASK_A = _DATASETS[DATASET]["task"]
SHUFFLE_KEY = "k17_gen:20261007"
PERM_SEED = 20261007

# Pinned checkpoints (SETTLE revisions, prod/tasks/data/model_revisions.json). The Thinking checkpoint is not
# in the fast-host HF cache; plan U3 loads a copy of the hub snapshot from a local directory (read only).
CKPTS = {
    "ouro_1_4b_base": {"layers": 24, "modeling_md5": "5a0c375566aef57873de94ee32bfa265"},
    "ouro_2_6b_base": {"layers": 48, "modeling_md5": "5a0c375566aef57873de94ee32bfa265"},
    "ouro_1_4b_think": {"layers": 24, "modeling_md5_prefix": "7129d9d0", "local_env": "K17_THINK_DIR"},
    # cluster extension (2026-10-11): the hub snapshot f1edd81 (weights.json), loaded from its folder like 1.4B Thinking
    "ouro_2_6b_think": {"layers": 48, "modeling_md5_prefix": "4135d08e", "local_env": "K17_THINK_DIR"},
}


# ====================================================================== small utilities
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(path, obj):
    tmp = "%s.%d.tmp" % (path, os.getpid())     # per process: row tasks that start together all write setting.json
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pos_seed(row_key, site_key, pos, seed):
    """DESIGN 7.1 `random`: CPU generator per (row, site, position) seeded (seed + 1000003 h) mod 2^63 with
    h = int(sha256(f"{row_key}|{site}|{p}")[:8], 16). Python hash() is salted and never used."""
    h = int(hashlib.sha256(("%s|%s|%d" % (row_key, site_key, int(pos))).encode("utf-8")).hexdigest()[:8], 16)
    return (int(seed) + 1000003 * h) % (2 ** 63)


def unit_dirs(row_key, site_key, positions, seed, d):
    """Unit Gaussian directions, one per absolute position, float32 [n, d] on CPU. Position pure."""
    out = torch.empty(len(positions), d, dtype=torch.float32)
    for i, p in enumerate(positions):
        g = torch.Generator().manual_seed(pos_seed(row_key, site_key, p, seed))
        u = torch.randn(d, generator=g, dtype=torch.float32)
        out[i] = u / u.norm()
    return out


# ====================================================================== the hooks
EDIT_OPS = ("patch", "rand", "lcdh", "kv")


def pos_range(info, pos):
    """Absolute positions [lo, hi) of a position set; hi None = no upper bound.

    span        the edited-number tokens of the prompt (VarBench `span_A`)
    span_to_end span start .. last prompt position (state only; critic M1 fix 2)
    last_prompt the last prompt position (the answer position before generation)
    all         every position, prompt and generated (LoopCD rows)
    [lo, hi]    explicit (teacher-forced rows)
    """
    if isinstance(pos, (list, tuple)):
        return int(pos[0]), (None if pos[1] is None else int(pos[1]))
    s0, s1 = info.get("span") or (None, None)
    plen = info.get("plen")
    if pos == "span":
        return int(s0), int(s1)
    if pos == "span_to_end":
        return int(s0), int(plen)
    if pos == "last_prompt":
        return int(plen) - 1, int(plen)
    if pos == "all":
        return 0, None
    raise ValueError("unknown position set %r" % (pos,))


def expand_loops(e, k):
    lp = e.get("loop", e.get("loops"))
    if lp == "K":
        return [int(k)]
    if isinstance(lp, (list, tuple)):
        return [int(k) if x == "K" else int(x) for x in lp]
    return [int(lp)]


class SteerHooks(object):
    """Edits keyed by row. `rows[key]` = {"plen", "span", "a_ids", "b_ids", "edits": [...], "lcdl": None|{...},
    "donor_range": (p0, p1), "t0": int}. A row not in `rows` (or with no edits and no lcdl) is never touched.

    Edit dicts:
      {"op": "patch", "site": "emb"|"post", "loop": j|[j..]|"K", "pos": ..., "a": 1.0}   O1 donor patch
      {"op": "rand",  "site": "post"|"pre", "loop": ..., "pos": ..., "seed": s, "ref": "patch"|"lcdh"|"scale",
       "ref_w": w (lcdh), "scale": c (scale)}                                             random same-norm
      {"op": "lcdh",  "site": "pre", "loop": j (>= 2), "pos": ..., "w": w}                LoopCD-Hidden form
      {"op": "kv",    "loops": [j..], "pos": ..., "a": 1.0}                               K/V donor patch
    lcdl: {"mode": "fixed", "w": w} | {"mode": "adaptive", "wmax": w}                     LoopCD-Logits
    """

    def __init__(self, k=None):
        self.k = None if k is None else int(k)
        self.rows = {}
        self.donors = {}
        self.handles = []
        self.model = None
        self.kv_installed = False
        self.batch_keys = None
        self.rec = None
        self.f = None
        self.stats = {}
        self.prompt_counted = set()
        self.n_forwards = 0
        self.record_fn = None          # callable(key) -> donors dict, set by the adapter
        self.decode_log_path = None    # CL17: one jsonl row per decode call (row ids, tokens requested / produced)
        self.n_decode_calls = 0

    # ------------------------------------------------------------------ install / remove
    def install(self, model, kv=False):
        assert self.model is None, "hooks already installed"
        self.model = model
        inner = model.model
        L = int(model.config.num_hidden_layers)
        self.L = L
        self.d = int(model.config.hidden_size)
        hs = self.handles
        hs.append(model.register_forward_pre_hook(self._pre_top, with_kwargs=True))
        hs.append(model.register_forward_hook(self._post_top, with_kwargs=True))
        hs.append(inner.norm.register_forward_pre_hook(self._pre_norm))
        hs.append(inner.norm.register_forward_hook(self._post_norm))
        for li, layer in enumerate(inner.layers[:L]):
            hs.append(layer.register_forward_pre_hook(partial(self._pre_layer, li), with_kwargs=True))
        if kv:
            for li, layer in enumerate(inner.layers[:L]):
                hs.append(layer.self_attn.k_proj.register_forward_hook(partial(self._kv_hook, li, "k")))
                hs.append(layer.self_attn.v_proj.register_forward_hook(partial(self._kv_hook, li, "v")))
            self.kv_installed = True
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.model = None
        self.kv_installed = False

    def log_decode(self, keys, seqs, n_new, gen):
        """CL17: append one row per decode call to decode_log_path (G.main's wave groups are visible here)."""
        self.n_decode_calls += 1
        if self.decode_log_path is None or keys is None:
            return
        row = {"call": self.n_decode_calls, "rows": list(keys), "n_new": int(n_new),
               "n_out": [len(g) for g in gen], "max_len": max(len(s) for s in seqs)}
        with open(self.decode_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------ per-forward plan
    def _row_active(self, key):
        info = self.rows.get(key)
        return info is not None and (info.get("edits") or info.get("lcdl"))

    def _pre_top(self, module, args, kwargs):
        if kwargs.get("labels") is not None:
            raise RuntimeError("steer_gen never passes labels to Ouro (exit-mixture trap, DESIGN 3.1)")
        inner = module.model
        if self.k is not None:
            assert int(inner.total_ut_steps) == self.k, (inner.total_ut_steps, self.k)
            assert module.early_exit_step == self.k - 1, (module.early_exit_step, self.k)
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None:
            emb = kwargs.get("inputs_embeds")
            B, T = int(emb.shape[0]), int(emb.shape[1])
            dev = emb.device
        else:
            B, T = int(ids.shape[0]), int(ids.shape[1])
            dev = ids.device
        am = kwargs.get("attention_mask")
        cp = kwargs.get("cache_position")
        if cp is None:
            pkv = kwargs.get("past_key_values")
            base = int(pkv.get_seq_length()) if (pkv is not None and kwargs.get("use_cache", True)) else 0
            cp = torch.arange(base, base + T, device=dev)
        if am is None:
            pad = torch.zeros(B, dtype=torch.long, device=dev)
        else:
            if am.dim() != 2:
                raise RuntimeError("steer_gen expects a 2D attention mask")
            pad = (am.shape[1] - am.sum(-1)).long()
        absp = cp.view(1, -1).long() - pad.view(-1, 1)
        ltk = kwargs.get("logits_to_keep", 0)
        if not isinstance(ltk, int):
            raise RuntimeError("logits_to_keep must be an int here")
        keys = self.batch_keys
        if keys is not None and len(keys) != B:
            raise RuntimeError("batch_keys has %d rows, forward has %d" % (len(keys), B))
        f = {"B": B, "T": T, "abs": absp, "ncall": 0, "ut": None, "li": None, "h1pre": None, "h1post": None,
             "ltk": ltk, "plan": {}, "lcdl": [], "need_h1pre": False, "need_h1post": False, "keys": keys,
             "count_stats": False}
        if keys is not None:
            self._build_plan(f)
        if self.rec is not None and self.rec.get("h1post"):
            f["need_h1post"] = True
        self.f = f
        self.n_forwards += 1
        return None

    def _build_plan(self, f):
        absp, T, keys = f["abs"], f["T"], f["keys"]
        plan = f["plan"]
        for b, key in enumerate(keys):
            info = self.rows.get(key)
            if info is None:
                continue
            if info.get("lcdl"):
                f["lcdl"].append((b, info["lcdl"]))
                f["need_h1post"] = True
            edits = info.get("edits") or []
            if not edits:
                continue
            first_prefill = T > 1 and key not in self.prompt_counted
            if first_prefill:
                self.prompt_counted.add(key)
            for e in edits:
                lo, hi = pos_range(info, e["pos"])
                row_abs = absp[b]
                if T == 1:
                    a0 = int(row_abs[0])
                    if a0 < lo or (hi is not None and a0 >= hi):
                        continue
                    cols = torch.zeros(1, dtype=torch.long, device=absp.device)
                    acols = [a0]
                else:
                    m = row_abs >= max(lo, 0)
                    if hi is not None:
                        m = m & (row_abs < hi)
                    cols = m.nonzero().view(-1)
                    if cols.numel() == 0:
                        continue
                    acols = row_abs[cols].tolist()
                count = (T == 1) or first_prefill
                if e["op"] == "kv":
                    sites = [("kv", j) for j in expand_loops(e, self.k)]
                else:
                    sites = [(e["site"], j) for j in expand_loops(e, self.k)]
                for s in sites:
                    if s[0] == "pre" and s[1] < 2 and e["op"] == "lcdh":
                        raise ValueError("LoopCD-Hidden needs loop >= 2 (h_1 is the reference)")
                    plan.setdefault(s, []).append((b, key, e, cols, acols, count))
                    if e["op"] == "lcdh" or (e["op"] == "rand" and (e.get("ref") == "lcdh"
                                                                     or e.get("dir", "gauss") != "gauss")):
                        f["need_h1pre"] = True

    # ------------------------------------------------------------------ donors
    def needs_donor(self, key):
        info = self.rows.get(key) or {}
        for e in info.get("edits") or []:
            if e["op"] in ("patch", "kv") or (e["op"] == "rand" and e.get("ref", "patch") == "patch"):
                return True
        return False

    def ensure_donors(self, keys):
        """Record missing donors BEFORE a forward (a donor forward inside a hook would clobber the context)."""
        if self.f is not None:
            raise RuntimeError("ensure_donors called inside a forward")
        for key in keys or []:
            if key in self.donors or not self.needs_donor(key):
                continue
            if self.record_fn is None:
                raise RuntimeError("row %s needs donors and no recorder is set" % key)
            self.donors[key] = self.record_fn(key)

    def donor_for(self, key):
        d = self.donors.get(key)
        if d is None:
            raise RuntimeError("row %s has no recorded donor (call ensure_donors first)" % key)
        return d

    def _donor_rows(self, d, kind, acols):
        idx = torch.tensor([a - int(d["t0"]) for a in acols], dtype=torch.long)
        n = int(d["n"])
        if int(idx.min()) < 0 or int(idx.max()) >= n:
            raise RuntimeError("donor range [%d, %d) does not cover positions %s"
                               % (d["t0"], int(d["t0"]) + n, acols[:5]))
        return kind.index_select(0, idx.to(kind.device))

    # ------------------------------------------------------------------ edit kernel
    @staticmethod
    def _coef(e, name, acols, dev):
        """The LoopCD coefficient of an edit at the given absolute positions, shape [n, 1] float32: the scalar
        e[name], or, when e[name + '_vec'] is given (CL11 relnorm), one coefficient per absolute position
        (a list indexed by absolute position)."""
        vec = e.get(name + "_vec")
        if vec is None:
            return torch.tensor(float(e[name]), dtype=torch.float32, device=dev)
        if max(acols) >= len(vec):
            raise RuntimeError("%s_vec has %d entries, position %d requested" % (name, len(vec), max(acols)))
        return torch.tensor([float(vec[a]) for a in acols], dtype=torch.float32, device=dev).view(-1, 1)

    def _stat(self, key, skey, dn, hn, count):
        if not count:
            return
        st = self.stats.setdefault((key, skey), [0.0, 0.0, 0])
        dl = dn.tolist()
        hl = hn.tolist()
        st[0] += float(sum(dl))
        st[1] += float(sum(x / max(y, 1e-12) for x, y in zip(dl, hl)))
        st[2] += len(dl)

    def _apply(self, x, site, j, items, dkind=None):
        """Edit the rows of `items` in tensor x [B, T, D]; returns a new tensor (x is not written in place)."""
        f = self.f
        y = x.clone()
        skey = "%s@%d" % (site, j)
        for (b, key, e, cols, acols, count) in items:
            cur = y[b].index_select(0, cols).float()
            op = e["op"]
            if op == "patch" or op == "kv":
                d = self.donor_for(key)
                if op == "patch":
                    src = d["emb"] if site == "emb" else d["post"][j]
                else:
                    src = d["kv"][dkind]
                t = self._donor_rows(d, src, acols).to(cur.device).float()
                new = torch.lerp(cur, t, float(e.get("a", 1.0)))
            elif op == "lcdh":
                h1 = f["h1pre"][b].index_select(0, cols).float()
                new = cur + self._coef(e, "w", acols, cur.device) * (cur - h1)
            elif op == "rand":
                ref = e.get("ref", "patch")
                if ref == "patch":
                    d = self.donor_for(key)
                    src = d["emb"] if site == "emb" else d["post"][j]
                    t = self._donor_rows(d, src, acols).to(cur.device).float()
                    nrm = (t - cur).norm(dim=-1)
                elif ref == "lcdh":
                    h1 = f["h1pre"][b].index_select(0, cols).float()
                    nrm = (self._coef(e, "ref_w", acols, cur.device) * (cur - h1)).norm(dim=-1)
                elif ref == "scale":
                    nrm = float(e["scale"]) * cur.norm(dim=-1)
                else:
                    raise ValueError("unknown random reference %r" % ref)
                # rkey (optional): a shared direction key, so arms that differ only in strength or position set
                # use the same direction at the same position (common random numbers, CL13)
                u = unit_dirs(e.get("rkey", key), skey, acols, int(e.get("seed", 0)), cur.shape[-1]).to(cur.device)
                rdir = e.get("dir", "gauss")
                if rdir != "gauss" and site != "pre":
                    raise ValueError("aligned/orth random directions are defined at the pre-norm site only")
                if rdir != "gauss":
                    # RA-1 / EXPERIMENT_MATRIX X9 (CB-F5): the same Gaussian u split along v = unit(h_j - h_1).
                    # aligned: sign(u . v) v (a push along the loop trajectory with a random sign);
                    # orth: the part of u orthogonal to v, renormalised. Same per-position norm as "gauss".
                    h1 = f["h1pre"][b].index_select(0, cols).float()
                    v = cur - h1
                    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    c = (u * v).sum(-1, keepdim=True)
                    if rdir == "aligned":
                        u = torch.where(c >= 0, torch.ones_like(c), -torch.ones_like(c)) * v
                    elif rdir == "orth":
                        o = u - c * v
                        u = o / o.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    else:
                        raise ValueError("unknown random direction %r" % rdir)
                new = cur + nrm.view(-1, 1) * u
            else:
                raise ValueError("unknown op %r" % op)
            if count:
                self._stat(key, skey if op != "kv" else "kv%s@%d" % (dkind[2], j),
                           (new - cur).norm(dim=-1), cur.norm(dim=-1), True)
            y[b].index_copy_(0, cols, new.to(y.dtype))
        return y

    # ------------------------------------------------------------------ hooks
    def _pre_layer(self, li, module, args, kwargs):
        f = self.f
        if f is None:
            return None
        ut = kwargs.get("current_ut")
        f["ut"] = None if ut is None else int(ut)
        f["li"] = li
        if li != 0 or f["ut"] != 0:
            return None
        x = args[0] if args else kwargs.get("hidden_states")
        rec = self.rec
        items = f["plan"].get(("emb", 1))
        y = x
        if items:
            y = self._apply(x, "emb", 1, items)
        if rec is not None and "emb" in rec["want"]:
            rec["emb"] = y[:, rec["cols"]].detach().clone()
        if y is x:
            return None
        if args:
            return (y,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = y
        return args, kwargs

    def _pre_norm(self, module, args):
        f = self.f
        if f is None:
            return None
        j = f["ncall"] + 1
        x = args[0]
        if j == 1 and f["need_h1pre"]:
            f["h1pre"] = x
        items = f["plan"].get(("pre", j))
        y = x
        if items:
            y = self._apply(x, "pre", j, items)
        rec = self.rec
        if rec is not None and "pre" in rec["want"]:
            rec["pre"][j] = y[:, rec["cols"]].detach().clone()
        if y is x:
            return None
        return (y,)

    def _post_norm(self, module, args, out):
        f = self.f
        if f is None:
            return None
        f["ncall"] += 1
        j = f["ncall"]
        items = f["plan"].get(("post", j))
        y = out
        if items:
            y = self._apply(out, "post", j, items)
        if j == 1 and f["need_h1post"]:
            f["h1post"] = y
        rec = self.rec
        if rec is not None and "post" in rec["want"]:
            rec["post"][j] = y[:, rec["cols"]].detach().clone()
        if y is out:
            return None
        return y

    def _kv_hook(self, li, which, module, args, out):
        f = self.f
        if f is None:
            return None
        if f["li"] != li:
            raise RuntimeError("k/v hook out of order: layer %s vs %s" % (f["li"], li))
        ut = f["ut"]
        j = ut + 1
        y = out
        items = f["plan"].get(("kv", j))
        if items:
            y = self._apply(out, "kv", j, items, dkind=(ut, li, which))
        rec = self.rec
        if rec is not None and "kv" in rec["want"]:
            rec["kv"][(ut, li, which)] = y[:, rec["cols"]].detach().clone()
        if y is out:
            return None
        return y

    def _post_top(self, module, args, kwargs, output):
        f = self.f
        self.f = None
        if f is None:
            return None
        if self.k is not None and f["ncall"] != self.k:
            raise RuntimeError("norm called %d times, expected k=%d" % (f["ncall"], self.k))
        rec = self.rec
        if rec is not None and rec.get("h1post"):
            rec["h1post_logits"] = self._z1(module, f, output.logits.shape[1])
        if f["lcdl"]:
            z = output.logits.float()
            z1 = self._z1(module, f, z.shape[1])
            for (b, spec) in f["lcdl"]:
                zb = z[b]
                if spec["mode"] == "fixed":
                    w = torch.full(zb.shape[:-1], float(spec["w"]), device=zb.device)
                elif spec["mode"] == "adaptive":
                    p = torch.softmax(zb, dim=-1)
                    top2 = p.topk(2, dim=-1).values
                    w = float(spec["wmax"]) * (1.0 - (top2[..., 0] - top2[..., 1]))
                else:
                    raise ValueError(spec)
                z[b] = zb + w.unsqueeze(-1) * (zb - z1[b])
                st = self.stats.setdefault((self.batch_keys[b], "lcdl"), [0.0, 0.0, 0])
                st[0] += float(w.sum())
                st[2] += int(w.numel())
            output.logits = z
        if rec is not None and "logits" in rec["want"]:
            rec["logits"].append(output.logits[:, -1].float().detach().clone())
        return output

    def _z1(self, module, f, tk):
        """Loop-1 exit read-out at the kept columns: lm_head(post-norm loop-1 state) in the model dtype, then
        float32 (2610.02185 Eq.(3) z_1; the same kernel the model's own exit read-out uses)."""
        h1 = f["h1post"]
        if h1 is None:
            raise RuntimeError("loop-1 post-norm state was not recorded")
        hk = h1[:, -tk:] if tk < h1.shape[1] else h1
        return F.linear(hk, module.lm_head.weight).float()


# ====================================================================== the adapter
class SteeredOuroAdapter(OuroAdapter):
    """prod's OuroAdapter plus SteerHooks. `decode` passes `row_ids` to the hooks; nothing else changes."""

    def __init__(self, name, steer, kv=False, think_dir=None, **kw):
        super(SteeredOuroAdapter, self).__init__(name, **kw)
        self.steer = steer
        self.kv = bool(kv)
        self.think_dir = think_dir
        self.file_md5 = {}

    def load(self):
        spec = CKPTS.get(self.base_name, {})
        if spec.get("local_env"):
            local = self.think_dir or os.environ.get(spec["local_env"])
            if not local:
                raise SystemExit("%s needs a local snapshot directory (env %s)" % (self.base_name,
                                                                                spec["local_env"]))
            for fn in ("modeling_ouro.py", "config.json", "tokenizer.json", "model.safetensors"):
                p = os.path.join(local, fn)
                if os.path.exists(p):
                    self.file_md5[fn] = md5_file(p)
            mm = self.file_md5.get("modeling_ouro.py", "")
            if not mm.startswith(spec["modeling_md5_prefix"]):
                raise SystemExit("modeling_ouro.py md5 %s is not the pinned %s" % (mm, spec["modeling_md5_prefix"]))
            self.repo = local
        super(SteeredOuroAdapter, self).load()
        mod = sys.modules[type(self.model).__module__]
        mf = getattr(mod, "__file__", None)
        if mf and os.path.exists(mf):
            self.file_md5.setdefault("modeling_ouro.py", md5_file(mf))
        if spec.get("modeling_md5") and self.file_md5.get("modeling_ouro.py") != spec["modeling_md5"]:
            raise SystemExit("modeling_ouro.py md5 %s != pinned %s" % (self.file_md5.get("modeling_ouro.py"),
                                                                    spec["modeling_md5"]))
        self.steer.install(self.model, kv=self.kv)
        self.steer.record_fn = self.record_donor
        return self

    def set_depth(self, k):
        d = super(SteeredOuroAdapter, self).set_depth(k)
        self.steer.k = int(k)
        return d

    @torch.no_grad()
    def decode(self, seqs, n_new, k, eos_ids, stop_strings=None, pieces=None, row_ids=None, **kw):
        keys = [str(r) for r in row_ids] if row_ids is not None else None
        if keys is not None:
            for b, key in enumerate(keys):
                info = self.steer.rows.get(key)
                if info is None or not info.get("a_ids"):
                    continue
                a = info["a_ids"]
                if list(seqs[b][:len(a)]) != list(a):
                    raise RuntimeError("row %s: the batch sequence does not start with the row's prompt" % key)
        # A forward that raised inside the hooks (e.g. CUDA OOM, which G.main catches and retries at half width)
        # never reaches the top-level post-hook, so its per-forward context is stale here: drop it.
        self.steer.f = None
        if keys is not None:
            self.steer.ensure_donors(keys)
        self.steer.batch_keys = keys
        try:
            out = super(SteeredOuroAdapter, self).decode(seqs, n_new, k, eos_ids, stop_strings=stop_strings,
                                                         pieces=pieces, row_ids=row_ids, **kw)
        finally:
            self.steer.batch_keys = None
        self.steer.log_decode(keys, seqs, n_new, out[0])
        return out

    @torch.no_grad()
    def record_donor(self, key):
        """Donor states of a row: one batch-1, no-cache, SDPA bf16 forward of the row's donor sequence `b_ids`,
        recording the post-norm loop-end states, the loop-1 layer-0 input (embedding) and, when the row has a kv
        edit, every call's k_proj / v_proj output, at donor positions [p0, p1). Target position t0 + i reads donor
        row i."""
        info = self.steer.rows[key]
        p0, p1 = info["donor_range"]
        want = {"post", "emb"}
        if any(e["op"] == "kv" for e in info.get("edits") or []):
            want.add("kv")
        saved_keys, saved_rec = self.steer.batch_keys, self.steer.rec
        self.steer.batch_keys = None
        self.steer.rec = {"cols": slice(int(p0), int(p1)), "want": want, "post": {}, "pre": {}, "kv": {},
                          "emb": None, "logits": []}
        try:
            ids = torch.tensor([info["b_ids"]], dtype=torch.long, device=next(self.model.parameters()).device)
            self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            rec = self.steer.rec
        finally:
            self.steer.rec = saved_rec
            self.steer.batch_keys = saved_keys
        d = {"t0": int(info.get("t0", p0)), "n": int(p1) - int(p0),
             "post": {j: v[0] for j, v in rec["post"].items()},
             "emb": None if rec["emb"] is None else rec["emb"][0],
             "kv": {kk: v[0] for kk, v in rec["kv"].items()}}
        return d

    def meta(self):
        d = super(SteeredOuroAdapter, self).meta()
        d.update({"steer_gen": VERSION, "steer_kv_hooks": self.kv, "file_md5": dict(self.file_md5)})
        if self.base_name == "ouro_1_4b_think":
            d["repo"] = "local snapshot copy of ByteDance/Ouro-1.4B-Thinking 3aaa222 (plan U3)"
        elif self.base_name == "ouro_2_6b_think":
            d["repo"] = "hub snapshot folder of ByteDance/Ouro-2.6B-Thinking f1edd81"
        return d


# ====================================================================== direct forwards (checks, TF)
@torch.no_grad()
def fwd(model, ids, attn=None, cache=None, cache_position=None, position_ids=None, logits_to_keep=0):
    """One forward the way prod's decode does it (bf16, SDPA), with or without a StaticUTCache."""
    dev = next(model.parameters()).device
    ids = ids.to(dev)
    B, T = ids.shape
    if attn is None:
        attn = torch.ones((B, T), dtype=torch.long, device=dev)
    kw = {"input_ids": ids, "attention_mask": attn, "logits_to_keep": int(logits_to_keep)}
    if position_ids is None:
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)[:, -T:]
    kw["position_ids"] = position_ids
    if cache is not None:
        kw.update({"past_key_values": cache, "use_cache": True,
                   "cache_position": cache_position if cache_position is not None
                   else torch.arange(T, device=dev)})
    else:
        kw["use_cache"] = False
    return model(**kw)


def loop_logits(model, rec_post, j):
    """Loop-j exit read-out from recorded post-norm states (lm_head in the model dtype, then float)."""
    return F.linear(rec_post[j], model.lm_head.weight).float()


# ====================================================================== data: tasks, pool, spans
def register_task(name, base, rows_path):
    """Register a task in prod.tasks in-process (no prod file is written): same prompt, suffix, stops and
    parsers as `base`, rows from `rows_path` (absolute; os.path.join keeps an absolute second argument)."""
    from . import tasks as T
    cfg = dict(T.TASKS[base])
    cfg["rows_file"] = os.path.abspath(rows_path)
    T.TASKS[name] = cfg
    return cfg


def core_pool(path=POOL_DEFAULT):
    """VarBench core pool (plan F7): text_exact, eval, equal length and identical ids outside the span on the
    four tokenizers; one pair per problem, the lowest var_name (DESIGN 12.3)."""
    rows = read_jsonl(path)
    core = [r for r in rows if r.get("text_exact") and r.get("split") in _DATASETS[DATASET]["splits"]
            and all(r["tok"][t]["equal_len"] and r["tok"][t]["outside_equal"]
                    for t in ("ouro", "huginn", "mcleish", "ifm"))]
    by = {}
    for r in core:
        by.setdefault(int(r["idx"]), []).append(r)
    return [sorted(v, key=lambda r: r["var_name"])[0] for _k, v in sorted(by.items())], len(core)


def shuffled(idx_list):
    xs = sorted(int(i) for i in idx_list)
    random.Random(SHUFFLE_KEY).shuffle(xs)
    return xs


def settle_cells(path, cap=4096):
    """{row_idx: correct_v2} from a SETTLE cells file at budget `cap` (plan F5)."""
    out = {}
    for r in read_jsonl(path):
        if "_header" in r or int(r.get("B", -1)) != int(cap):
            continue
        out[int(r["row_idx"])] = bool(r.get("correct_v2"))
    return out


def load_tokenizer(name, think_dir=None):
    from transformers import AutoTokenizer
    from . import config as cfgmod
    from .models.ouro import OURO_CHECKPOINTS
    spec = CKPTS.get(name, {})
    repo = OURO_CHECKPOINTS[name]["repo"]
    rev = cfgmod.revision_for(name)
    if spec.get("local_env"):
        repo = think_dir or os.environ.get(spec["local_env"]) or repo
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def prompt_texts(tok, task, rows, chat):
    from .tasks import build_prompts
    prompts, _s, _st, _e, _x = build_prompts(tok, task, rows, chat_template=chat)
    return prompts


def char_span_tokens(tok, text, c0, c1):
    """Token indices whose character range overlaps [c0, c1) (fast-tokenizer offsets)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    sel = [i for i, (a, b) in enumerate(offs) if b > c0 and a < c1 and b > a]
    return ids, (sel[0], sel[-1] + 1) if sel else (None, None)


def item_geometry(tok, prow, chat, settle_rows):
    """A/B prompt ids, the number span (token positions) and checks, for one VarBench pair.

    The span is mapped from the pair's character span through the tokenizer's offsets in the full prompt.
    On base prompts it must equal the pool's own `tok.ouro.span_A` (asserted by the caller); under the chat
    template it is recomputed this way (plan CL5)."""
    from .tasks import task_cfg
    c = task_cfg(TASK_A)
    rowA = dict(settle_rows[int(prow["idx"])])
    if rowA["input"] != prow["question_A"]:
        raise ValueError("question_A is not the SETTLE text for idx %s" % prow["idx"])
    rowB = dict(rowA, input=prow["question_B"], target=str(prow["gold_B"]))
    pA, pB = prompt_texts(tok, TASK_A, [rowA, rowB], chat)
    qA = c["q_prefix"] + prow["question_A"]
    qB = c["q_prefix"] + prow["question_B"]
    oA = pA.rfind(qA)
    oB = pB.rfind(qB)
    if oA < 0 or oB < 0:
        raise ValueError("question not found in the prompt for idx %s" % prow["idx"])
    oA += len(c["q_prefix"])
    oB += len(c["q_prefix"])
    ca0, ca1 = prow["char_span_A"]
    cb0, cb1 = prow["char_span_B"]
    idsA, spA = char_span_tokens(tok, pA, oA + ca0, oA + ca1)
    idsB, spB = char_span_tokens(tok, pB, oB + cb0, oB + cb1)
    ok = {"equal_len": len(idsA) == len(idsB), "span_equal": spA == spB and spA[0] is not None}
    if ok["equal_len"] and ok["span_equal"]:
        s0, s1 = spA
        ok["outside_equal"] = idsA[:s0] == idsB[:s0] and idsA[s1:] == idsB[s1:]
        dA = tok.decode(idsA[s0:s1]).strip().replace(",", "")
        dB = tok.decode(idsB[s0:s1]).strip().replace(",", "")
        ok["span_text"] = dA == str(prow["value_A"]) and dB == str(prow["value_B"])
        ok["span_not_pos0"] = s0 > 0
    return {"idsA": idsA, "idsB": idsB, "span": list(spA), "ok": ok, "plen": len(idsA)}


def wave1_groups(plens, width=16, keys=None):
    """G.main's first-wave grouping: rows sorted by prompt length, ties in task-row order (gsm8k rows are in idx
    order, so `keys` = idx), chunks of `width` (prod/generate.py gen_pass: at_T sorted by plen, at_T[gi:gi+b])."""
    keys = list(range(len(plens))) if keys is None else list(keys)
    order = sorted(range(len(plens)), key=lambda i: (plens[i], keys[i]))
    return [order[g:g + width] for g in range(0, len(order), width)]


def place_sentinels(scored_plens, cand, n):
    """Pick n candidates (key, plen) nearest the (q + 0.5)/n prompt-length quantiles of the scored set (CL3).
    Deterministic: ties broken by key."""
    sp = sorted(scored_plens)
    left = sorted(cand, key=lambda t: (t[1], t[0]))
    out = []
    for q in range(n):
        if not left:
            break
        target = sp[min(len(sp) - 1, int(round((q + 0.5) / n * len(sp) - 0.5)))]
        best = min(range(len(left)), key=lambda i: (abs(left[i][1] - target), left[i][0]))
        out.append(left.pop(best))
    return out


def sentinel_report(plens_scored, plens_sent, width=16):
    pl = list(plens_scored) + list(plens_sent)
    is_s = [False] * len(plens_scored) + [True] * len(plens_sent)
    groups = wave1_groups(pl, width)
    per = [sum(1 for i in g if is_s[i]) for g in groups]
    return {"n_chunks": len(groups), "sentinels_per_chunk": per, "chunks_without": sum(1 for x in per if x == 0)}


def layout_change(plens_base, plens_shift_extra, width=16):
    """Per base row: did its wave-1 group membership and padded length change when the extra rows join?"""
    g0 = wave1_groups(plens_base, width)
    pl1 = list(plens_base) + list(plens_shift_extra)
    g1 = wave1_groups(pl1, width)
    n = len(plens_base)
    mem0, mem1, pad0, pad1 = {}, {}, {}, {}
    for g in g0:
        L = max(plens_base[i] for i in g)
        for i in g:
            mem0[i] = frozenset(g)
            pad0[i] = L
    for g in g1:
        L = max(pl1[i] for i in g)
        for i in g:
            if i < n:
                mem1[i] = frozenset(x for x in g)
                pad1[i] = L
    ch_m = sum(1 for i in range(n) if mem0[i] != mem1[i])
    ch_p = sum(1 for i in range(n) if pad0[i] != pad1[i])
    return {"n": n, "group_changed": ch_m, "padded_len_changed": ch_p}


# ====================================================================== arm definitions
def rows_arm_a(k):
    """Arm (a) rows (plan section 6; CL3 r20; CL4 labels). 'prompt' = which prompt the job runs."""
    P = lambda j, pos, **kw: dict({"op": "patch", "site": "post", "loop": j, "pos": pos, "a": 1.0}, **kw)
    R = lambda j, pos, seed=0: {"op": "rand", "site": "post", "loop": j, "pos": pos, "seed": seed, "ref": "patch"}
    comp = lambda j: [P(j, "span_to_end"), {"op": "kv", "loops": list(range(1, j + 1)), "pos": "span_to_end", "a": 1.0}]
    return {
        "r00": {"prompt": "A", "edits": [], "role": "clean A"},
        "r01": {"prompt": "B", "edits": [], "role": "swap B"},
        "r02": {"prompt": "A", "edits": [{"op": "patch", "site": "emb", "loop": 1, "pos": "span", "a": 1.0}],
                "role": "harness identity L0@1"},
        "r03": {"prompt": "A", "edits": [P(1, "span")], "role": "source Llast@1"},
        "r19": {"prompt": "A", "edits": [P(2, "span")], "role": "source Llast@2"},
        "r04": {"prompt": "A", "edits": [P(3, "span")], "role": "source Llast@3"},
        "r05": {"prompt": "A", "edits": [P(list(range(1, k + 1)), "span")], "role": "persistent source"},
        "r06": {"prompt": "A", "edits": [P(1, "last_prompt")], "role": "null site (no answer state) Llast@1"},
        "r07": {"prompt": "A", "edits": [P(k, "last_prompt")], "role": "null site (no answer state) Llast@K"},
        "r08": {"prompt": "A", "edits": [R(1, "span", 0)], "role": "random span Llast@1"},
        "r09": {"prompt": "A", "edits": comp(1), "role": "complete patch @1"},
        "r11": {"prompt": "A", "edits": comp(3), "role": "complete patch @3"},
        "r10": {"prompt": "A", "edits": comp(k), "role": "complete patch @K"},
        "r12": {"prompt": "A", "edits": [P(1, "span_to_end")], "role": "span to prompt end Llast@1"},
        "r13": {"prompt": "A", "edits": [{"op": "kv", "loops": [1], "pos": "span", "a": 1.0}], "role": "KV loop 1"},
        "r14": {"prompt": "A", "edits": [R(3, "span", 0)], "role": "random span Llast@3"},
        "r15": {"prompt": "A", "edits": [R(list(range(1, k + 1)), "span", 0)], "role": "random persistent"},
        "r16": {"prompt": "A", "edits": [R(1, "last_prompt", 0)], "role": "random last prompt Llast@1"},
        "r17": {"prompt": "A", "edits": [R(1, "span", 1)], "role": "random span Llast@1 seed 1"},
        "r18": {"prompt": "A", "edits": [R(1, "span", 2)], "role": "random span Llast@1 seed 2"},
        "r21": {"prompt": "A", "self_donor": True, "edits": [P(1, "span")],
                "role": "self-donor null: O1 Llast@1 span, donor = A's own batch-1 state (CL16)"},
        "r20": {"prompt": "A", "shift": True, "role": "layout-perturbed duplicate clean (CL3)",
                "edits": [{"op": "patch", "site": "emb", "loop": 1, "pos": "span", "a": 0.0},
                          {"op": "patch", "site": "post", "loop": list(range(1, k + 1)), "pos": "span", "a": 0.0},
                          {"op": "kv", "loops": list(range(1, k + 1)), "pos": "span", "a": 0.0}]},
    }


def rows_arm_b(k, tun):
    """Arm (b)/(c) rows (plan section 6, U8, CL1-CL3). `tun` = frozen tuning {w_star, w_abs, wmax, w_fixed}."""
    ws = tun.get("w_star")
    wa = tun.get("w_abs")
    out = {
        "unedited": {"edits": []},
        "pert": {"edits": [], "shift": True},
        "lcdl_ad": {"edits": [], "lcdl": {"mode": "adaptive", "wmax": float(tun["wmax"])}},
    }
    if tun.get("w_fixed") is not None:
        out["lcdl_fx"] = {"edits": [], "lcdl": {"mode": "fixed", "w": float(tun["w_fixed"])}}
    if ws is not None:
        out["lcdh"] = {"edits": [{"op": "lcdh", "site": "pre", "loop": k, "pos": "all", "w": float(ws)}]}
        out["rand%d" % k] = {"edits": [{"op": "rand", "site": "pre", "loop": k, "pos": "all", "seed": 0,
                                         "ref": "lcdh", "ref_w": float(ws)}]}
    if k >= 3 and ws is not None:
        out["lcdh2_rel"] = {"edits": [{"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": float(ws)}]}
        if wa is not None:
            out["lcdh2_abs"] = {"edits": [{"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": float(wa)}]}
            out["rand2"] = {"edits": [{"op": "rand", "site": "pre", "loop": 2, "pos": "all", "seed": 0,
                                       "ref": "lcdh", "ref_w": float(wa)}]}
        out["pc2"] = {"edits": [{"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": -1.0}]}
        out["k%d" % (k - 1)] = {"edits": [], "depth": k - 1}
        wa3 = tun.get("w_abs3")
        if k >= 4 and wa3 is not None:
            # CL12: the loop-3 (= K-1) contrast at absolute matching, and its matched random
            out["lcdh3_abs"] = {"edits": [{"op": "lcdh", "site": "pre", "loop": 3, "pos": "all", "w": float(wa3)}]}
            out["rand3"] = {"edits": [{"op": "rand", "site": "pre", "loop": 3, "pos": "all", "seed": 0,
                                       "ref": "lcdh", "ref_w": float(wa3)}]}
    elif ws is not None:
        out["rand2"] = out.pop("rand%d" % k)
    return out


# ====================================================================== settings (config-driven)
SETTINGS = {
    "a1": {"arm": "a", "model": "ouro_1_4b_base", "k": 4, "horizon": 512, "chat": False, "think_stop": True,
           "n_scored": 150, "n_sent": 22, "n_shift": 8, "n_screen": 240, "screen_extend": 80,
           "settle": "cells_ouro_1_4b_base_gsm8k_natural_k4.jsonl", "screen_A_from_settle": True,
           "rows": ["r00", "r01", "r02", "r03", "r19", "r04", "r05", "r06", "r07", "r08", "r09", "r10",
                    "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r20", "r21"]},
    "a2": {"arm": "a", "model": "ouro_2_6b_base", "k": 4, "horizon": 512, "chat": False, "think_stop": True,
           "n_scored": 150, "n_sent": 22, "n_shift": 8, "n_screen": 240, "screen_extend": 80,
           "settle": "cells_ouro_2_6b_base_gsm8k_natural_k4.jsonl", "screen_A_from_settle": True,
           "rows": ["r00", "r01", "r02", "r03", "r04", "r05", "r06", "r07", "r08", "r09", "r10", "r20", "r21"],
           "rows_ext": ["r11", "r12", "r13", "r19", "r14", "r15", "r16", "r17", "r18"]},
    "a3": {"arm": "a", "model": "ouro_1_4b_think", "k": 4, "horizon": 4096, "chat": True, "think_stop": False,
           "n_scored": 30, "n_sent": 8, "n_shift": 8, "n_screen": 45, "screen_extend": 20,
           "settle": "cells_ouro_1_4b_think_gsm8k_natural2_k4.jsonl",
           "rows": ["r00", "r01", "r02", "r03", "r07", "r10", "r20"]},
    # cluster extension (2026-10-11): setting a3 on Ouro-2.6B-Thinking (the cluster config sets rows and counts)
    "a4": {"arm": "a", "model": "ouro_2_6b_think", "k": 4, "horizon": 4096, "chat": True, "think_stop": False,
           "n_scored": 30, "n_sent": 8, "n_shift": 8, "n_screen": 45, "screen_extend": 20,
           "settle": "cells_ouro_2_6b_think_gsm8k_natural2_k4.jsonl",
           "rows": ["r00", "r01", "r02", "r03", "r07", "r10", "r20"]},
    "b2": {"arm": "b", "model": "ouro_1_4b_base", "k": 4, "horizon": 4096, "chat": False, "think_stop": True,
           "task": "gsm8k", "n_eval": 400, "n_sent": 50, "n_shift": 8, "n_train": 200,
           "settle": "cells_ouro_1_4b_base_gsm8k_natural_k4.jsonl",
           "settle_strata": ["cells_ouro_1_4b_base_gsm8k_natural_k2.jsonl"],
           "wmax": 0.5, "w_fixed": 0.3,
           # RA-1 (2026-10-08): the loop-2 test is teacher-forced on the tune chains (no eval generation row) and
           # covers loop 2 only; the eval accuracy rows below are cut by RA-1 (kept for the scaled configuration)
           "tf_on": "train", "tf_loops": [2],
           "rows": ["unedited", "pert", "lcdl_ad", "lcdl_fx", "lcdh", "lcdh2_rel", "lcdh2_abs", "rand4", "rand2",
                    "pc2", "k3"],
           "rows_ext": ["lcdh3_abs", "rand3"]},
    "b1": {"arm": "b", "model": "ouro_1_4b_think", "k": 2, "horizon": 4096, "chat": True, "think_stop": False,
           "task": "gsm8k", "n_eval": 400, "n_sent": 50, "n_shift": 8, "n_train": 200,
           "settle": "cells_ouro_1_4b_think_gsm8k_natural2_k2.jsonl",
           "settle_strata": ["cells_ouro_1_4b_think_gsm8k_natural2_k4.jsonl"],
           "wmax": 0.5, "w_fixed": 0.3, "rows": ["unedited", "lcdl_ad", "lcdl_fx", "lcdh", "rand2"],
           "rows_ext": ["pert"], "unedited_from_settle": True},
    "c": {"arm": "b", "model": "ouro_1_4b_think", "k": 2, "horizon": 4096, "chat": True, "think_stop": False,
          "task": "math500", "n_eval": 400, "n_sent": 50, "n_shift": 8, "n_train": 0,
          "settle": "cells_ouro_1_4b_think_math500_natural2_k2.jsonl",
          "settle_strata": ["cells_ouro_1_4b_think_math500_natural2_k4.jsonl"],
          "wmax": 0.5, "w_fixed": 0.3, "rows": ["unedited", "lcdl_fx", "lcdh", "lcdl_ad", "rand2"]},
}


def setting_cfg(name, override=None):
    cfg = json.loads(json.dumps(SETTINGS[name]))
    if override:
        cfg.update(override)
    cfg["name"] = name
    if cfg.get("arm") == "a" and cfg.get("dataset", "gsm8k") != DATASET:
        raise SystemExit("setting dataset %r != K17_DATASET %r" % (cfg.get("dataset", "gsm8k"), DATASET))
    return cfg


# ====================================================================== prep / select (CPU)
def prep_a(cfg, out_dir, settle_dir, think_dir=None, pool_path=POOL_DEFAULT):
    """Arm (a) candidates: pool -> SETTLE pre-filter (correct_v2 on A at the setting's k) -> shuffle order;
    geometry checked per item with the run tokenizer; writes prep.json and the screen row lists."""
    from .tasks import rows as task_rows
    pool, n_core_rows = core_pool(pool_path)
    settle = settle_cells(os.path.join(settle_dir, cfg["settle"]))
    srows = {int(r["idx"]): r for r in task_rows(TASK_A)}
    tok = load_tokenizer(cfg["model"], think_dir)
    tok_base = tok if not cfg["chat"] else load_tokenizer("ouro_1_4b_base")
    by = {int(r["idx"]): r for r in pool}
    order = [i for i in shuffled(by) if settle.get(i)]
    items, bad = [], {}
    for idx in order:
        pr = by[idx]
        g = item_geometry(tok, pr, cfg["chat"], srows)
        if not cfg["chat"]:
            pool_span = pr["tok"]["ouro"]["span_A"]
            g["ok"]["matches_pool_span"] = list(g["span"]) == list(pool_span) and g["plen"] == pr["tok"]["ouro"]["len_A"]
        else:
            gb = item_geometry(tok_base, pr, False, srows)
            g["ok"]["base_method_matches_pool"] = list(gb["span"]) == list(pr["tok"]["ouro"]["span_A"])
        fails = [k for k, v in g["ok"].items() if not v]
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
    prep = {"setting": cfg["name"], "n_pool_rows": n_core_rows, "n_pool_problems": len(pool),
            "n_settle_correct": len(order), "n_geometry_ok": len(items), "geometry_fail": bad,
            "pool_sha256": sha256_file(pool_path), "settle_file": cfg["settle"], "items": items}
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "prep.json"), prep)
    return prep


def write_rows_files(out_dir, tag, items_A, items_B_or_A):
    """Rows files for a job set: A rows (SETTLE gsm8k rows, used through the plain gsm8k task with --only-rows)
    and the B task rows (question_B for scored items, the A text for every other row)."""
    from .tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(TASK_A)}
    brows = []
    for it in items_B_or_A:
        r = dict(srows[int(it["idx"])])
        if it.get("B"):
            r["input"] = it["question_B"]
            r["target"] = str(it["gold_B"])
        brows.append(r)
    p = os.path.join(out_dir, "rows_%s_B.jsonl" % tag)
    write_jsonl(p, brows)
    ids = sorted(int(it["idx"]) for it in items_A)
    q = os.path.join(out_dir, "only_%s.json" % tag)
    with open(q, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    return p, q


def settle_screen_A(path, cap):
    """RA-1 no-rerun rule: the clean-A screen is SETTLE's own natural-protocol cell at budget `cap` (same prompt,
    same G.main path); {row_idx: cells row}."""
    out = {}
    for r in read_jsonl(path):
        if "_header" in r or int(r.get("B", -1)) != int(cap):
            continue
        out[int(r["row_idx"])] = r
    return out


def select_a(cfg, out_dir, pool_path=POOL_DEFAULT, settle_dir=None):
    """After the screen jobs: scored = first n_scored items (shuffle order) with own answer = gold_A on A and
    = gold_B on B; sentinels placed by prompt length from the remaining pool items (CL3); shift rows = the 8
    shortest GSM8K eval prompts outside both sets. With cfg screen_A_from_settle the A side is SETTLE's cell
    at budget = horizon (RA-1 no-rerun rule)."""
    prep = json.load(open(os.path.join(out_dir, "prep.json"), encoding="utf-8"))
    if cfg.get("screen_A_from_settle"):
        if not settle_dir:
            raise SystemExit("screen_A_from_settle needs --settle-dir")
        sA = settle_screen_A(os.path.join(settle_dir, cfg["settle"]), cfg["horizon"])
        n_scr = int(cfg["n_screen"]) + int(cfg.get("_screen_extra", 0))
        sA = {i: r for i, r in sA.items() if i in {it["idx"] for it in prep["items"][:n_scr]}}
    else:
        sA = load_cells(os.path.join(out_dir, "screen_A"))
    sB = load_cells(os.path.join(out_dir, "screen_B"))
    from .tasks import num_eq
    scored, fails = [], {"A_wrong": 0, "B_wrong": 0, "not_screened": 0}
    for it in prep["items"]:
        a, b = sA.get(it["idx"]), sB.get(it["idx"])
        if a is None or b is None:
            fails["not_screened"] += 1
            continue
        okA = a.get("trace_answer") is not None and num_eq(a["trace_answer"], it["gold_A"])
        okB = b.get("trace_answer") is not None and num_eq(b["trace_answer"], it["gold_B"])
        if not okA:
            fails["A_wrong"] += 1
        if not okB:
            fails["B_wrong"] += 1
        if okA and okB:
            scored.append(it)
    scored = scored[:cfg["n_scored"]]
    used = {it["idx"] for it in scored}
    cand = [(it["idx"], it["plen"]) for it in prep["items"] if it["idx"] not in used]
    sent = place_sentinels([it["plen"] for it in scored], cand, cfg["n_sent"])
    sent_ids = {k for k, _p in sent}
    shift = shortest_eval_rows(cfg, exclude=used | sent_ids, n=cfg["n_shift"])
    rep = sentinel_report([it["plen"] for it in scored], [p for _k, p in sent])
    lay = layout_change([it["plen"] for it in scored] + [p for _k, p in sent], [p for _k, p in shift])
    sel = {"setting": cfg["name"], "n_screened_A": len(sA), "n_screened_B": len(sB), "filter_fails": fails,
           "screen_A_source": ("SETTLE cells %s at budget %d" % (cfg["settle"], int(cfg["horizon"]))
                               if cfg.get("screen_A_from_settle") else "screen_A job"),
           "n_scored": len(scored), "scored": scored,
           "sentinels": [{"idx": k, "plen": p} for k, p in sent],
           "shift": [{"idx": k, "plen": p} for k, p in shift],
           "sentinel_layout": rep, "r20_layout_change": lay}
    write_json(os.path.join(out_dir, "items.json"), sel)
    return sel


def shortest_eval_rows(cfg, exclude, n):
    """The n shortest GSM8K eval prompts (A text, the setting's template) outside `exclude`."""
    from .tasks import rows as task_rows, split_labels
    from . import config as cfgmod
    tok = load_tokenizer(cfg["model"], os.environ.get("K17_THINK_DIR"))
    rws = task_rows(TASK_A)
    lab = split_labels(TASK_A, len(rws), cfgmod.SPLIT_SEED, cfgmod.N_CAL)
    cand = [r for r, l in zip(rws, lab) if l == "eval" and int(r["idx"]) not in exclude]
    cand = sorted(cand, key=lambda r: len(r["input"]))[:max(4 * n, 40)]
    pr = prompt_texts(tok, TASK_A, cand, cfg["chat"])
    pl = [(int(r["idx"]), len(tok(p, add_special_tokens=False)["input_ids"])) for r, p in zip(cand, pr)]
    return sorted(pl, key=lambda t: (t[1], t[0]))[:n]


def load_cells(job_dir):
    """{row_idx: cells row} of the single cap of a job directory (the only cells_*.jsonl there)."""
    fs = [f for f in os.listdir(job_dir) if f.startswith("cells_") and f.endswith(".jsonl")] if os.path.isdir(job_dir) else []
    if not fs:
        return {}
    if len(fs) > 1:
        raise RuntimeError("more than one cells file in %s" % job_dir)
    out = {}
    for r in read_jsonl(os.path.join(job_dir, fs[0])):
        if "_header" in r:
            continue
        out[int(r["row_idx"])] = r
    return out


def load_chains(job_dir):
    fs = [f for f in os.listdir(job_dir) if f.startswith("chains_") and f.endswith(".jsonl")] if os.path.isdir(job_dir) else []
    out = {}
    for fn in fs:
        for r in read_jsonl(os.path.join(job_dir, fn)):
            out[int(r["idx"])] = r
    return out


def prep_b(cfg, out_dir, train_jsonl=None):
    """Arm (b)/(c) rows: eval rows by default_rng(20261007) permutation of the sorted eval row_idx, first
    n_eval; sentinels = cal rows placed at the eval prompt-length quantiles (CL3); shift rows = the shortest
    remaining cal rows; train-200 (U6) from GSM8K train rows 4-7472."""
    from .tasks import rows as task_rows, split_labels
    from . import config as cfgmod
    task = cfg["task"]
    rws = task_rows(task)
    lab = split_labels(task, len(rws), cfgmod.SPLIT_SEED, cfgmod.N_CAL)
    ev = sorted(int(r["idx"]) for r, l in zip(rws, lab) if l == "eval")
    cal = sorted(int(r["idx"]) for r, l in zip(rws, lab) if l == "cal")
    perm = np.random.default_rng(PERM_SEED).permutation(len(ev))
    eval_ids = [ev[i] for i in perm[:cfg["n_eval"]]]
    tok = load_tokenizer(cfg["model"], os.environ.get("K17_THINK_DIR"))
    byidx = {int(r["idx"]): r for r in rws}

    def plens(ids):
        pr = prompt_texts(tok, task, [byidx[i] for i in ids], cfg["chat"])
        return [len(tok(p, add_special_tokens=False)["input_ids"]) for p in pr]
    pe = plens(eval_ids)
    pc = plens(cal)
    sent = place_sentinels(pe, list(zip(cal, pc)), cfg["n_sent"])
    sid = {k for k, _p in sent}
    rest = sorted([(k, p) for k, p in zip(cal, pc) if k not in sid], key=lambda t: (t[1], t[0]))
    shift = rest[:cfg["n_shift"]]
    out = {"setting": cfg["name"], "task": task, "eval": eval_ids, "eval_plen": pe,
           "sentinels": [{"idx": k, "plen": p} for k, p in sent], "shift": [{"idx": k, "plen": p} for k, p in shift],
           "cal_all": cal, "sentinel_layout": sentinel_report(pe, [p for _k, p in sent]),
           "pert_layout_change": layout_change(pe + [p for _k, p in sent], [p for _k, p in shift])}
    if cfg.get("n_train") and train_jsonl:
        tr = read_jsonl(train_jsonl)
        tr_ids = sorted(int(r["idx"]) for r in tr if int(r["idx"]) >= 4)
        tperm = np.random.default_rng(PERM_SEED).permutation(len(tr_ids))
        out["train"] = [tr_ids[i] for i in tperm[:cfg["n_train"]]]
        out["train_file_sha256"] = sha256_file(train_jsonl)
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "items.json"), out)
    return out


def gsm8k_train_rows(parquet_path, out_path):
    """GSM8K train (openai/gsm8k 740312a main/train) as prod rows: {idx, input, target}; target = the number
    after the last '####' with commas removed."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    rows = []
    for i, (q, a) in enumerate(zip(df["question"], df["answer"])):
        t = a.rsplit("####", 1)[1].strip().replace(",", "")
        rows.append({"idx": i, "src_idx": i, "input": q, "target": t, "options": [], "gold_text": ""})
    write_jsonl(out_path, rows)
    return len(rows)


# ====================================================================== jobs through G.main
def env_block(ad=None):
    """checks.json `env` (A7: no host names)."""
    import transformers
    d = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
         "steer_gen": VERSION, "steer_gen_sha256": sha256_file(os.path.abspath(__file__))}
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
    if ad is not None:
        d["file_md5"] = dict(getattr(ad, "file_md5", {}))
    return d


def build_row_infos(tok, cfg, items, rowdef, sent_ids=()):
    """steer.rows for an arm-(a) job: scored rows get the row's edits; sentinels and shift rows none."""
    from .tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(TASK_A)}
    infos = {}
    for it in items:
        ra = srows[int(it["idx"])]
        rb = dict(ra, input=it["question_B"])
        pA, pB = prompt_texts(tok, TASK_A, [ra, rb], cfg["chat"])
        a_ids = tok(pA, add_special_tokens=False)["input_ids"]
        b_ids = tok(pB, add_special_tokens=False)["input_ids"]
        s0, s1 = it["span"]
        if len(a_ids) != it["plen"] or len(b_ids) != it["plen"]:
            raise RuntimeError("prompt length mismatch for idx %s" % it["idx"])
        if a_ids[:s0] != b_ids[:s0] or a_ids[s1:] != b_ids[s1:] or a_ids[s0:s1] == b_ids[s0:s1]:
            raise RuntimeError("A/B ids do not differ exactly in the span for idx %s" % it["idx"])
        prompt_ids = a_ids if rowdef.get("prompt", "A") == "A" else b_ids
        if rowdef.get("self_donor"):
            b_ids = list(a_ids)                       # CL16 r21: the donor is A itself (batch-1 numerics only)
        infos[str(it["idx"])] = {"plen": len(a_ids), "span": (s0, s1), "a_ids": prompt_ids, "b_ids": b_ids,
                                 "edits": list(rowdef.get("edits") or []) if rowdef.get("prompt", "A") == "A" else [],
                                 "lcdl": None, "donor_range": (s0, len(a_ids)), "t0": s0}
    return infos


def set_aside_failed(out_dir):
    """A job folder whose steer_job.json records a failed run (exit not 0, or no readable exit) moves whole into
    out_dir/_failed_<n>/ (n = 1, 2, ...), so the rerun starts clean: G.main keeps the errors and incomplete-row
    count of an earlier run in meta_*.json and would exit 3 again. Readers list the job folder only, not its
    subfolders. A folder without steer_job.json (a killed run) still resumes. Returns the new folder or None."""
    sj = os.path.join(out_dir, "steer_job.json")
    if not os.path.exists(sj):
        return None
    try:
        with open(sj, encoding="utf-8") as f:
            code = int(json.load(f).get("exit"))
    except (ValueError, TypeError, AttributeError):
        code = None
    if code == 0:
        return None
    n = 1
    while os.path.exists(os.path.join(out_dir, "_failed_%d" % n)):
        n += 1
    dst = os.path.join(out_dir, "_failed_%d" % n)
    os.makedirs(dst)
    for fn in sorted(os.listdir(out_dir)):
        if not fn.startswith("_failed_"):
            os.replace(os.path.join(out_dir, fn), os.path.join(dst, fn))
    print("[steer_gen] %s recorded exit %s: its files moved to %s; the job starts clean" % (sj, code, dst),
          flush=True)
    return dst


def job_exit(summ):
    """Process exit code of a job: the exit steer_job.json records (G.main exits 3 when a row is incomplete)."""
    try:
        return int(summ.get("exit"))
    except (TypeError, ValueError):
        return 1


def run_gmain(model, task, k, only_ids, out_dir, horizon, steer, kv, think_stop=True, mem_fraction=None,
              extra=(), think_dir=None, shards=1, shard=0, batch_width=16):
    """One G.main job with SteeredOuroAdapter swapped in for prod.generate.get (restored in finally).
    Resumable after a kill: G.main skips cells already written; decode_log.jsonl (CL17) is appended. After a
    recorded failure the folder starts clean (set_aside_failed)."""
    import prod.generate as G
    os.makedirs(out_dir, exist_ok=True)
    set_aside_failed(out_dir)
    only = os.path.join(out_dir, "only_rows.json")
    with open(only, "w", encoding="utf-8") as f:
        json.dump(sorted(int(i) for i in only_ids), f)
    steer.decode_log_path = os.path.join(out_dir, "decode_log.jsonl")
    orig = G.get
    holder = {}

    def _get(name, adapter_dir=None, **kw):
        ad = SteeredOuroAdapter(name, steer=steer, kv=kv, think_dir=think_dir, adapter_dir=adapter_dir, **kw)
        holder["ad"] = ad
        return ad
    argv = ["--model", model, "--task", task, "--k", str(k), "--only-rows", only, "--out", out_dir,
            "--caps", str(horizon), "--horizon", str(horizon), "--no-extra-caps", "--batch-width",
            str(int(batch_width))]
    if not think_stop:
        argv.append("--no-think-tag-is-stop")
    if mem_fraction:
        argv += ["--mem-fraction", str(mem_fraction)]
    if shards > 1:
        argv += ["--shards", str(shards), "--shard", str(shard)]
    argv += list(extra)
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
    summ = {"argv": argv, "exit": code, "seconds": round(time.time() - t0, 1),
            "stats": {"%s|%s" % kk: v for kk, v in steer.stats.items()}, "n_forwards": steer.n_forwards}
    write_json(os.path.join(out_dir, "steer_job.json"), summ)
    return summ


def job_a(cfg, out_root, row, think_dir=None, mem_fraction=None, screen=None, batch_width=16, n_limit=None,
          job_name=None):
    """Run one arm-(a) row (or a screen job: screen = 'A' | 'B') for a setting. CL16 bw-check: row r00,
    n_limit 20 (the first 20 scored items, no sentinels), batch_width 1, job_name 'bw1_r00'."""
    from .models.ouro import OURO_CHECKPOINTS
    k = cfg["k"]
    rdefs = rows_arm_a(k)
    if screen:
        prep = json.load(open(os.path.join(out_root, "prep.json"), encoding="utf-8"))
        n = int(cfg["n_screen"]) + int(cfg.get("_screen_extra", 0))
        items = prep["items"][:n]
        rowdef = rdefs["r00"] if screen == "A" else rdefs["r01"]
        name = "screen_%s" % screen
        all_ids = [it["idx"] for it in items]
        scored = items
    else:
        sel = json.load(open(os.path.join(out_root, "items.json"), encoding="utf-8"))
        rowdef = rdefs[row]
        name = job_name or row
        scored = sel["scored"]
        if n_limit:
            scored = scored[:int(n_limit)]
            all_ids = [it["idx"] for it in scored]
        else:
            all_ids = [it["idx"] for it in scored] + [s["idx"] for s in sel["sentinels"]]
        if rowdef.get("shift") and not n_limit:
            all_ids += [s["idx"] for s in sel["shift"]]
    pool = {int(r["idx"]): r for r in core_pool()[0]}
    for it in scored:
        pr = pool[int(it["idx"])]
        it["question_B"] = pr["question_B"]
    out_dir = os.path.join(out_root, name)
    tok = load_tokenizer(cfg["model"], think_dir)
    steer = SteerHooks(k)
    steer.rows = build_row_infos(tok, cfg, scored, rowdef)
    kv = any(e["op"] == "kv" for e in rowdef.get("edits") or [])
    if rowdef.get("prompt") == "B":
        bitems = [dict(it, B=True) for it in scored] + [{"idx": i} for i in all_ids if i not in {x["idx"] for x in scored}]
        rows_p = os.path.join(out_root, "rows_%s_B.jsonl" % name)
        from .tasks import rows as task_rows
        srows = {int(r["idx"]): r for r in task_rows(TASK_A)}
        brows = []
        for it in bitems:
            r = dict(srows[int(it["idx"])])
            if it.get("B"):
                r["input"] = it["question_B"]
                r["target"] = str(it["gold_B"])
            brows.append(r)
        write_jsonl(rows_p, brows)
        task = "%s_cfB_%s_%s" % (TASK_A, cfg["name"], name)
        register_task(task, TASK_A, rows_p)
    else:
        task = TASK_A
    assert cfg["model"] in OURO_CHECKPOINTS
    return run_gmain(cfg["model"], task, k, all_ids, out_dir, cfg["horizon"], steer, kv,
                     think_stop=cfg["think_stop"], mem_fraction=mem_fraction, think_dir=think_dir,
                     batch_width=batch_width)


def job_b(cfg, out_root, row, tun, think_dir=None, mem_fraction=None, which="eval"):
    """Run one arm-(b)/(c) row: which = 'eval' (eval + sentinels [+ shift]), 'train' (train-200, no sentinels),
    'cal' (the 100 cal rows, CL9 drift)."""
    items = json.load(open(os.path.join(out_root, "items.json"), encoding="utf-8"))
    k = cfg["k"]
    rdefs = rows_arm_b(k, tun)
    rowdef = rdefs[row]
    task = cfg["task"]
    if which == "eval":
        scored = items["eval"]
        ids = scored + [s["idx"] for s in items["sentinels"]]
        if rowdef.get("shift"):
            ids += [s["idx"] for s in items["shift"]]
    elif which == "train":
        scored = items["train"]
        ids = list(scored)
        task = "gsm8k_train"
        register_task(task, "gsm8k", os.environ.get("K17_TRAIN_ROWS"))
    elif which == "cal":
        scored = []
        ids = items["cal_all"]
    else:
        raise ValueError(which)
    steer = SteerHooks(rowdef.get("depth", k))
    for i in scored:
        steer.rows[str(i)] = {"plen": None, "span": None, "a_ids": None, "edits": list(rowdef.get("edits") or []),
                              "lcdl": rowdef.get("lcdl")}
    out_dir = os.path.join(out_root, "%s_%s" % (which, row))
    return run_gmain(cfg["model"], task, rowdef.get("depth", k), ids, out_dir, cfg["horizon"], steer, False,
                     think_stop=cfg["think_stop"], mem_fraction=mem_fraction, think_dir=think_dir)


# ====================================================================== statistics and outcomes
RE_NUM = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\d])")


def boot_cluster(xs, cl, n=1000, seed=0):
    """Ratio-of-sums mean with a 95% percentile bootstrap over clusters (port of k16_path_ouro.boot_cluster
    142-157), returning [v, lo, hi, n_items, n_clusters]."""
    groups = {}
    for x, c in zip(xs, cl):
        if x is not None:
            groups.setdefault(c, []).append(float(x))
    if not groups:
        return None
    keys = sorted(groups, key=lambda z: str(z))
    sums = torch.tensor([sum(groups[kk]) for kk in keys], dtype=torch.float64)
    cnts = torch.tensor([len(groups[kk]) for kk in keys], dtype=torch.float64)
    v0 = float(sums.sum() / cnts.sum())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=g)
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    q = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [round(v0, 4), round(float(q[0]), 4), round(float(q[1]), 4), int(cnts.sum()), len(keys)]


def boot_ratio(num, den, cl, n=1000, seed=0):
    """sum(num)/sum(den) with a cluster bootstrap (FR-type ratios of two per-token rates)."""
    groups = {}
    for a, b, c in zip(num, den, cl):
        g = groups.setdefault(c, [0.0, 0.0])
        g[0] += float(a)
        g[1] += float(b)
    keys = sorted(groups, key=lambda z: str(z))
    A = torch.tensor([groups[kk][0] for kk in keys], dtype=torch.float64)
    Bt = torch.tensor([groups[kk][1] for kk in keys], dtype=torch.float64)
    if float(Bt.sum()) == 0:
        return None
    v0 = float(A.sum() / Bt.sum())
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=gen)
    den_b = Bt[idx].sum(1)
    ok = den_b > 0
    bs = A[idx].sum(1)[ok] / den_b[ok]
    q = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [round(v0, 4), round(float(q[0]), 4), round(float(q[1]), 4), len(num), len(keys)]


def mcnemar_exact(b, c):
    """Exact two-sided McNemar: binomial(n = b + c, 0.5) on the discordant pairs."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    m = min(int(b), int(c))
    p = sum(math.comb(n, i) for i in range(m + 1)) / float(2 ** n)
    return min(1.0, 2.0 * p)


def holm(ps):
    """Holm step-down adjusted p-values, in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def first_mention(text, x, x2):
    """DESIGN 11 first_mention: isolated numbers (digits with optional thousands commas, not adjacent to a
    digit); which of x2 ('B'), x ('A') occurs first, else 'neither'."""
    fx, fx2 = float(x), float(x2)
    for m in RE_NUM.finditer(text or ""):
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        if v == fx2:
            return "B"
        if v == fx:
            return "A"
    return "neither"


def outcome(own, gold_a, gold_b):
    from .tasks import num_eq
    if own is None:
        return "neither"
    if num_eq(own, gold_b):
        return "B"
    if num_eq(own, gold_a):
        return "A"
    return "neither"


def degenerate(gen_ids, natural_stop, horizon, window=512, thresh=0.3):
    """P2: hits the horizon with no stop marker and distinct-4-gram ratio over the last `window` tokens < 0.3."""
    if natural_stop is not None and int(natural_stop) < int(horizon):
        return False
    if len(gen_ids) < int(horizon):
        return False
    tail = list(gen_ids[-window:])
    grams = [tuple(tail[i:i + 4]) for i in range(len(tail) - 3)]
    if not grams:
        return False
    return len(set(grams)) / float(len(grams)) < thresh


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


# ====================================================================== teacher-forced loop-2 test (CL1, CL6)
def tf_arms(k, w_star, w_abs, wmax):
    """Arm table of the teacher-forced test: name -> (prefill edits, token edits, lcdl). Positions 'all' in the
    prefill cover [0, T) and the token edit acts on the single scored position (P_last / P_all)."""
    A = "all"
    lc = lambda w, loop=2: {"op": "lcdh", "site": "pre", "loop": loop, "pos": A, "w": float(w)}
    rnd = lambda w, loop=2, seed=0: {"op": "rand", "site": "pre", "loop": loop, "pos": A, "seed": seed,
                                     "ref": "lcdh", "ref_w": float(w)}
    tiny = {"op": "rand", "site": "pre", "loop": 2, "pos": A, "seed": 0, "ref": "scale", "scale": 1e-3}
    arms = {"clean": ([], [], None),
            "last_fwd": ([], [lc(w_star)], None), "last_back": ([], [lc(-w_star)], None),
            "last_rand": ([], [rnd(w_star)], None),
            "rest_fwd": ([lc(w_star)], [], None), "rest_back": ([lc(-w_star)], [], None),
            "rest_rand": ([rnd(w_star)], [], None),
            "all_fwd": ([lc(w_star)], [lc(w_star)], None), "all_back": ([lc(-w_star)], [lc(-w_star)], None),
            "all_rand": ([rnd(w_star)], [rnd(w_star)], None),
            "all_pc2": ([lc(-1.0)], [lc(-1.0)], None), "all_rand_pc2": ([rnd(-1.0)], [rnd(-1.0)], None),
            "all_tiny": ([tiny], [tiny], None),
            "lcdh": ([], [lc(w_star, loop=k)], None), "rand_k": ([], [rnd(w_star, loop=k)], None),
            "lcdl_ad": ([], [], {"mode": "adaptive", "wmax": float(wmax)})}
    if w_abs is not None:
        arms["all_abs"] = ([lc(w_abs)], [lc(w_abs)], None)
    return arms


@torch.no_grad()
def tf_margins(model, steer, ids, k):
    """Clean full no-cache forward: per position final and loop-2 exit top-2 margins (CL6 filter)."""
    steer.batch_keys = None
    steer.rec = {"cols": slice(None), "want": {"post"}, "post": {}, "pre": {}, "kv": {}, "emb": None, "logits": []}
    try:
        out = fwd(model, ids.view(1, -1), logits_to_keep=0)
        rec = steer.rec
    finally:
        steer.rec = None
    zf = out.logits[0].float()
    z2 = loop_logits(model, rec["post"][2] if k >= 2 else rec["post"][1], 1)[0] if False else \
        F.linear(rec["post"][min(2, k)][0], model.lm_head.weight).float()
    t2f = zf.topk(2, -1).values
    t22 = z2.topk(2, -1).values
    return (t2f[:, 0] - t2f[:, 1]).cpu(), (t22[:, 0] - t22[:, 1]).cpu(), zf.argmax(-1).cpu()


@torch.no_grad()
def tf_chain(ad, steer, chain_key, ids, positions, arms, k):
    """One chain: one batched prefill (row r = arm r's prefill edits), then one batched single-token forward
    per scored position in DESCENDING order (the forward at t writes cache slot t; smaller t never read it).
    Returns per position per arm: final argmax, loop-2 exit argmax, KL(p_clean || p_arm) final, and per-loop
    ||dh|| and cos(dh, delta_own) at the scored position (dh vs the clean row)."""
    model = ad.model
    dev = next(model.parameters()).device
    names = list(arms)
    Bn = len(names)
    T = int(ids.numel())
    keys = ["%s|%s" % (chain_key, nm) for nm in names]
    for key, nm in zip(keys, names):
        steer.rows[key] = {"plen": T, "span": None, "a_ids": None, "edits": arms[nm][0], "lcdl": None}
    cache = ad._cache_factory(Bn, T + 1)
    batch = ids.view(1, -1).expand(Bn, T).to(dev)
    steer.batch_keys = keys
    try:
        fwd(model, batch, cache=cache, cache_position=torch.arange(T, device=dev), logits_to_keep=1)
    finally:
        steer.batch_keys = None
    res = {}
    for key, nm in zip(keys, names):
        steer.rows[key] = {"plen": T, "span": None, "a_ids": None, "edits": arms[nm][1], "lcdl": arms[nm][2]}
    for t in sorted(positions, reverse=True):
        cache.lens = [int(t)] * len(cache.lens)
        tok_t = batch[:, t:t + 1]
        attn = torch.ones((Bn, t + 1), dtype=torch.long, device=dev)
        steer.batch_keys = keys
        steer.rec = {"cols": slice(-1, None), "want": {"post", "pre"}, "post": {}, "pre": {}, "kv": {},
                     "emb": None, "logits": []}
        try:
            out = model(input_ids=tok_t, attention_mask=attn, position_ids=torch.full((Bn, 1), t, device=dev),
                        past_key_values=cache, use_cache=True, cache_position=torch.tensor([t], device=dev),
                        logits_to_keep=1)
            rec = steer.rec
        finally:
            steer.rec = None
            steer.batch_keys = None
        zf = out.logits[:, -1].float()
        z2 = F.linear(rec["post"][min(2, k)][:, -1], model.lm_head.weight).float()
        lp = torch.log_softmax(zf, -1)
        kl = (lp[0].exp() * (lp[0] - lp)).sum(-1)
        h = torch.stack([rec["post"][j][:, -1].float() for j in range(1, k + 1)], 0)  # [k, Bn, d]
        dh = h - h[:, :1]
        dn = dh.norm(dim=-1)
        row = {"final": zf.argmax(-1).tolist(), "own2": z2.argmax(-1).tolist(), "kl": kl.tolist(),
               "dh": dn.t().tolist(), "zmargin": float((zf[0].topk(2).values[0] - zf[0].topk(2).values[1]))}
        # cos(dh_j, dh_2): survival of the loop-2 change direction at later loops
        c = []
        for j in range(k):
            num = (dh[j] * dh[min(1, k - 1)]).sum(-1)
            den = dh[j].norm(dim=-1) * dh[min(1, k - 1)].norm(dim=-1)
            c.append(torch.where(den > 0, num / den.clamp(min=1e-12), torch.zeros_like(num)).tolist())
        row["cos_to_loop2"] = [list(x) for x in zip(*c)]
        res[int(t)] = row
    for key in keys:
        steer.rows.pop(key, None)
    del cache
    return names, res


# ====================================================================== answer-site test (CL4)
def answer_site(tok, chain_ids, gold):
    """Token index of the first answer digit after the LAST '####' in a chain (None if the own answer is not
    `gold`). The site position is the token before it."""
    from .tasks import num_eq
    pieces = [tok.decode([int(t)], clean_up_tokenization_spaces=False) for t in chain_ids]
    text = "".join(pieces)
    i = text.rfind("####")
    if i < 0:
        return None
    m = RE_NUM.search(text, i + 4)
    if m is None or not num_eq(m.group(0).replace(",", ""), gold):
        return None
    c = m.start()
    acc = 0
    for ti, p in enumerate(pieces):
        if acc >= c:
            return ti
        acc += len(p)
    return None


# ====================================================================== checks G1-G4
@torch.no_grad()
def g3_decide(g3, clean_gap, clean_top, measured_rate_ok=False):
    """G3 verdict from the per-kind counts. Rule (Deviation D5): argmax agreement >= n - n/32 and every argmax
    disagreement at a recompute top-2 margin < 1 logit. measured_rate_ok (the cluster checks, --g3-measured-rate):
    when the margin part holds for every kind and only the agreement count misses, the gate passes as Deviation D5m
    (D5 rule at the measured rate; the code has no recompute decode path), the measured agreement per kind is recorded
    and pass_strict stays False."""
    for nm, v in g3.items():
        v["pass_argmax"] = bool(v["argmax_agree"] >= v["n"] - max(1, v["n"] // 32))
        v["pass_full_vocab_dz"] = bool(v["max_abs_dz"] <= 2 * clean_gap + 0.05)
        v["pass_top5_dz"] = bool(v["max_abs_dz_top5"] <= 2 * clean_top + 0.05)
        v["pass_ties_only"] = bool(v["disagree_at_margin_ge_1"] == 0)
        v["pass"] = bool(v["pass_argmax"] and v["pass_ties_only"])
    out = {"kinds": g3, "clean_gap": round(clean_gap, 4), "clean_gap_top5": round(clean_top, 4),
           "rule": "argmax >= n - n/32 and every argmax disagreement at a recompute margin < 1 "
                   "logit (Deviation D5); top-5 (D3) and full-vocabulary |dz| rules reported as "
                   "pass_top5_dz and pass_full_vocab_dz",
           "pass": all(v["pass"] for v in g3.values())}
    out["pass_strict"] = out["pass"]
    if measured_rate_ok and not out["pass"] and all(v["pass_ties_only"] for v in g3.values()):
        out["pass"] = True
        out["deviation"] = {
            "id": "D5m",
            "rule": "D5 at the measured rate: every argmax disagreement at a recompute top-2 margin < 1 logit (holds); "
                    "the argmax agreement rate is recorded, not bounded by n - n/32",
            "why": "cached decode vs full recompute numerics on this hardware; the generation code has no recompute "
                   "decode path, so the run uses the cached path and records the measured agreement",
            "measured_agreement": {nm: {"agree": v["argmax_agree"], "n": v["n"],
                                        "rate": round(v["argmax_agree"] / max(1, v["n"]), 4)}
                                   for nm, v in g3.items()},
            "kinds_below_n_minus_n_over_32": sorted(nm for nm, v in g3.items() if not v["pass_argmax"])}
    return out


def check_suite(ad, steer, items, k, out_path=None, n_steps=32, g3_items=3, g3_measured_rate=False):
    """G1 zero edit, G2 placement, G3 cached vs full recompute, G4 identities on a few items of one checkpoint.
    `items` carry a_ids, b_ids, span, plen (the geometry of item_geometry)."""
    model = ad.model
    dev = next(model.parameters()).device
    eos = [ad.tok.eos_token_id]
    res = {}
    # ---------------- G1: zero edits mixed with nonzero edits in one batch, vs hooks inert
    zero_edits = [{"op": "patch", "site": "emb", "loop": 1, "pos": "span", "a": 0.0},
                  {"op": "patch", "site": "post", "loop": [1, 2, k], "pos": "span_to_end", "a": 0.0},
                  {"op": "kv", "loops": list(range(1, k + 1)), "pos": "span_to_end", "a": 0.0},
                  {"op": "lcdh", "site": "pre", "loop": k, "pos": "all", "w": 0.0},
                  {"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": 0.0},
                  {"op": "rand", "site": "pre", "loop": 2, "pos": "all", "seed": 0, "ref": "scale", "scale": 0.0}]
    live_edits = [{"op": "patch", "site": "post", "loop": 1, "pos": "span", "a": 1.0},
                  {"op": "lcdh", "site": "pre", "loop": k, "pos": "all", "w": 0.5}]
    seqs, keys = [], []
    for i, it in enumerate(items):
        for kind in ("zero", "live"):
            key = "g1|%d|%s" % (i, kind)
            steer.rows[key] = {"plen": it["plen"], "span": tuple(it["span"]), "a_ids": it["a_ids"], "b_ids": it["b_ids"],
                               "edits": zero_edits if kind == "zero" else live_edits,
                               "lcdl": {"mode": "fixed", "w": 0.0} if kind == "zero" else {"mode": "adaptive", "wmax": 0.5},
                               "donor_range": (it["span"][0], it["plen"]), "t0": it["span"][0]}
            seqs.append(it["a_ids"])
            keys.append(key)
    ids, attn, L = leftpad(seqs, ad.tok.pad_token_id, dev)
    # reference: hooks REMOVED (the fresh-load model), then reinstalled
    steer.remove()
    base = fwd(model, ids, attn=attn, logits_to_keep=1).logits.float()
    g0, _t = OuroAdapter.decode(ad, seqs, n_steps, k, eos, row_ids=None)
    steer.install(model, kv=ad.kv)
    steer.record_fn = ad.record_donor
    inert = fwd(model, ids, attn=attn, logits_to_keep=1).logits.float()
    steer.ensure_donors(keys)
    steer.batch_keys = keys
    try:
        hooked = fwd(model, ids, attn=attn, logits_to_keep=1).logits.float()
    finally:
        steer.batch_keys = None
    zr = [i for i, kk in enumerate(keys) if kk.endswith("zero")]
    lv = [i for i, kk in enumerate(keys) if kk.endswith("live")]
    g1_prefill = bool(torch.equal(base[zr], hooked[zr]))
    g1_inert = bool(torch.equal(base, inert))
    live_moved = float((base[lv] - hooked[lv]).abs().max())
    g1t, _t = ad.decode(seqs, n_steps, k, eos, row_ids=keys)
    g1_tokens = all(g0[i] == g1t[i] for i in zr)
    gi, _t = ad_plain_decode(ad, seqs, n_steps, k, eos, None)
    g1_inert_tokens = all(g0[i] == gi[i] for i in range(len(seqs)))
    res["gate_zero_edit"] = {"prefill_bit_identical": g1_prefill, "tokens_identical_%d_steps" % n_steps: g1_tokens,
                             "inert_hooks_bit_identical": g1_inert, "inert_hooks_tokens_identical": g1_inert_tokens,
                             "hook_kinds": ["emb", "post", "kv", "pre(lcdh)", "pre(rand)", "lcdl"],
                             "live_rows_moved_max_abs_logit": round(live_moved, 4),
                             "pass": bool(g1_prefill and g1_tokens and g1_inert and g1_inert_tokens and live_moved > 0)}
    # ---------------- G2: placement on the K/V cache of a prefill (clean row vs edited row, same sequence)
    it = items[0]
    s0, s1 = it["span"]
    plen = it["plen"]
    L_ = ad.n_layers()
    cases = {"emb@1": [{"op": "patch", "site": "emb", "loop": 1, "pos": "span", "a": 1.0}],
             "post@1": [{"op": "patch", "site": "post", "loop": 1, "pos": "span", "a": 1.0}],
             "post@%d" % (k - 1): [{"op": "patch", "site": "post", "loop": k - 1, "pos": "span", "a": 1.0}],
             "post@K": [{"op": "patch", "site": "post", "loop": k, "pos": "span", "a": 1.0}],
             "kv@1": [{"op": "kv", "loops": [1], "pos": "span", "a": 1.0}],
             "lcdh@K": [{"op": "lcdh", "site": "pre", "loop": k, "pos": "all", "w": 0.5}],
             "lcdh@2": [{"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": 0.5}],
             "lcdh@%d" % (k - 1): [{"op": "lcdh", "site": "pre", "loop": max(2, k - 1), "pos": "all", "w": 0.5}]}
    # Deviation D6: when the donor token equals the A token at every edited position (the ans-chain geometry: the
    # site is a chain token shared by A and the aligned donor), the embedding patch is a structural zero and the
    # layer-0 loop-1 K/V (a function of the token alone, before rotary) is unchanged, so the first changed entry is
    # None for emb@1 and 1 (layer 1, loop 1) for kv@1.
    site_tok_differs = any(it["a_ids"][p] != it["b_ids"][p] for p in range(s0, s1))
    expect_F = {"emb@1": 0 if site_tok_differs else None, "post@1": L_, "post@%d" % (k - 1): (k - 1) * L_,
                "post@K": None, "kv@1": 0 if site_tok_differs else (1 if L_ > 1 else None),
                "lcdh@K": None, "lcdh@2": 2 * L_ if k > 2 else None,
                "lcdh@%d" % (k - 1): (max(2, k - 1) * L_ if max(2, k - 1) < k else None)}
    g2 = {}
    for nm, ed in cases.items():
        kc, ke = "g2|clean", "g2|" + nm
        for kk, e in ((kc, []), (ke, ed)):
            steer.rows[kk] = {"plen": plen, "span": (s0, s1), "a_ids": it["a_ids"], "b_ids": it["b_ids"], "edits": e,
                              "lcdl": None, "donor_range": (s0, plen), "t0": s0}
        # two separate batch-1 prefills: rows of one batch are not bit-identical to each other even with equal
        # inputs (row-dependent GEMM blocking), but equal shapes in separate calls are
        b1 = torch.tensor([it["a_ids"]], dtype=torch.long, device=dev)
        steer.donors.pop(ke, None)
        steer.ensure_donors([kc, ke])
        caches = []
        for kk in (kc, ke):
            cache = ad._cache_factory(1, plen + 1)
            steer.batch_keys = [kk]
            try:
                fwd(model, b1, cache=cache, logits_to_keep=1)
            finally:
                steer.batch_keys = None
            caches.append(cache)
        first_changed, bad = None, []
        lo_pos = s0 if not nm.startswith("lcdh") else 0
        for ent in range(L_ * k):
            for kvname, ci in (("k", "key_cache"), ("v", "value_cache")):
                a = getattr(caches[0], ci)[ent][0, :, :plen]
                b = getattr(caches[1], ci)[ent][0, :, :plen]
                diff = (a != b).any(0).any(-1)          # [plen] positions that differ
                if bool(diff[:lo_pos].any()):
                    bad.append("ent %d %s: positions before the edit changed" % (ent, kvname))
                if bool(diff.any()) and first_changed is None:
                    first_changed = ent
        ok = (first_changed == expect_F[nm]) and not bad
        g2[nm] = {"first_changed_entry": first_changed, "expected": expect_F[nm], "violations": bad[:5], "pass": ok}
        del caches
    res["gate_placement"] = {"cases": g2, "site_tokens_equal": not site_tok_differs,
                             "pass": all(v["pass"] for v in g2.values())}
    # ---------------- G3: cached decode vs full no-cache recompute, per edit kind (teacher-forced on the cached tokens)
    g3 = {}
    kinds = {"clean": [], "post@1": cases["post@1"], "lcdh@K": cases["lcdh@K"], "lcdh@2": cases["lcdh@2"],
             "kv@1": cases["kv@1"], "rand_pre@2": [{"op": "rand", "site": "pre", "loop": 2, "pos": "all", "seed": 0,
                                                   "ref": "lcdh", "ref_w": 0.5}],
             "lcdl": "lcdl"}
    clean_gap, clean_top = None, None
    for nm, ed in kinds.items():
        gaps, tops, agree, tot, bad_dis, dms = [], [], 0, 0, 0, []
        for i, itx in enumerate(items[:g3_items]):
            key = "g3|%d|%s" % (i, nm)
            steer.rows[key] = {"plen": itx["plen"], "span": tuple(itx["span"]), "a_ids": itx["a_ids"], "b_ids": itx["b_ids"],
                               "edits": [] if ed == "lcdl" else ed,
                               "lcdl": {"mode": "adaptive", "wmax": 0.5} if ed == "lcdl" else None,
                               "donor_range": (itx["span"][0], itx["plen"]), "t0": itx["span"][0]}
            steer.rec = {"cols": slice(-1, None), "want": {"logits"}, "post": {}, "pre": {}, "kv": {}, "emb": None,
                         "logits": []}
            try:
                gen, _t = ad.decode([itx["a_ids"]], n_steps, k, [], row_ids=[key])
                zc = torch.cat(steer.rec["logits"], 0)
            finally:
                steer.rec = None
            seq = list(itx["a_ids"]) + list(gen[0])
            ids1 = torch.tensor([seq], dtype=torch.long, device=dev)
            steer.ensure_donors([key])
            steer.batch_keys = [key]
            try:
                zf = fwd(model, ids1, logits_to_keep=0).logits[0].float()
            finally:
                steer.batch_keys = None
            P = itx["plen"]
            zr = zf[P - 1:P - 1 + len(gen[0])]
            zc = zc[:zr.shape[0]]
            agree += int((zr.argmax(-1) == zc.argmax(-1)).sum())
            tot += int(zr.shape[0])
            gaps.append(float((zr - zc).abs().max()))
            # decision-relevant gap: the recompute's top-5 tokens per step (the full-vocabulary max is set by
            # far-tail logits, where bf16 steps are several logits wide)
            top = zr.topk(5, dim=-1).indices
            tops.append(float((zr.gather(-1, top) - zc.gather(-1, top)).abs().max()))
            # D5: an argmax disagreement is allowed only at a near tie of the recompute (margin < 1 logit)
            t2 = zr.topk(2, dim=-1)
            mr = t2.values[:, 0] - t2.values[:, 1]
            mc = zc.gather(-1, t2.indices[:, :1])[:, 0] - zc.gather(-1, t2.indices[:, 1:2])[:, 0]
            dms += (mr - mc).abs().tolist()
            dis = zr.argmax(-1) != zc.argmax(-1)
            bad_dis += int((dis & (mr >= 1.0)).sum())
        g3[nm] = {"argmax_agree": agree, "n": tot, "max_abs_dz": round(max(gaps), 4),
                  "max_abs_dz_top5": round(max(tops), 4), "disagree_at_margin_ge_1": bad_dis,
                  "max_abs_dmargin_top2": round(max(dms), 4) if dms else None}
        if nm == "clean":
            clean_gap, clean_top = max(gaps), max(tops)
    res["gate_cache_equiv"] = g3_decide(g3, clean_gap, clean_top, measured_rate_ok=g3_measured_rate)
    res["gate_cache_equiv"]["g3_items"] = min(g3_items, len(items))
    # ---------------- G4: identities (PCK, PCL, PC2)
    g4 = {}
    it = items[0]
    ids1 = torch.tensor([it["a_ids"]], dtype=torch.long, device=dev)
    for nm, row in (("PCK", {"edits": [{"op": "lcdh", "site": "pre", "loop": k, "pos": "all", "w": -1.0}], "lcdl": None}),
                    ("PCL", {"edits": [], "lcdl": {"mode": "fixed", "w": -1.0}})):
        key = "g4|" + nm
        steer.rows[key] = dict(row, plen=it["plen"], span=tuple(it["span"]), a_ids=it["a_ids"])
        steer.rec = {"cols": slice(None), "want": set(), "post": {}, "pre": {}, "kv": {}, "emb": None, "logits": [],
                     "h1post": True}
        steer.batch_keys = [key]
        try:
            z = fwd(model, ids1, logits_to_keep=0).logits[0].float()
            z1 = steer.rec["h1post_logits"][0]
        finally:
            steer.batch_keys = None
            steer.rec = None
        g4[nm] = {"max_abs_z_minus_z1": round(float((z - z1).abs().max()), 5)}
        g4[nm]["pass"] = g4[nm]["max_abs_z_minus_z1"] <= 0.02
    if k >= 3:
        key = "g4|PC2"
        steer.rows[key] = {"plen": it["plen"], "span": tuple(it["span"]), "a_ids": it["a_ids"],
                           "edits": [{"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": -1.0}], "lcdl": None}
        steer.batch_keys = [key]
        try:
            z = fwd(model, ids1, logits_to_keep=0).logits[0].float()
        finally:
            steer.batch_keys = None
        ad.set_depth(k - 1)
        try:
            z3 = fwd(model, ids1, logits_to_keep=0).logits[0].float()
        finally:
            ad.set_depth(k)
        agree = float((z.argmax(-1) == z3.argmax(-1)).float().mean())
        g4["PC2"] = {"argmax_agree": round(agree, 5), "max_abs_dz": round(float((z - z3).abs().max()), 4)}
        g4["PC2"]["pass"] = agree >= 0.999 and g4["PC2"]["max_abs_dz"] <= 0.1
    res["gate_identities"] = dict(g4, **{"pass": all(v["pass"] for v in g4.values())})
    for kk in [x for x in steer.rows if x.startswith("g")]:
        steer.rows.pop(kk, None)
    steer.donors = {}
    if out_path:
        write_json(out_path, res)
    return res


@torch.no_grad()
def ad_plain_decode(ad, seqs, n_new, k, eos, row_ids):
    """prod's decode with the hooks inert (no row ids)."""
    saved = ad.steer.batch_keys
    ad.steer.batch_keys = None
    try:
        return OuroAdapter.decode(ad, seqs, n_new, k, eos, row_ids=None)
    finally:
        ad.steer.batch_keys = saved


# ====================================================================== drivers: teacher-forced test and tuning
def chain_seqs(tok, cfg, chains, idxs, task):
    """{idx: (prompt ids, chain ids)} from a G.main chains file; the chain row's prompt_sha256 must match the
    prompt this module rebuilds (prod/generate.py write_chains)."""
    from .tasks import rows as task_rows
    from .common import sha256_text
    byidx = {int(r["idx"]): r for r in task_rows(task)}
    out, bad = {}, []
    for i in idxs:
        cr = chains.get(int(i))
        if cr is None:
            continue
        pr = prompt_texts(tok, task, [byidx[int(i)]], cfg["chat"])[0]
        enc = tok(pr, add_special_tokens=False)["input_ids"]
        if cr.get("prompt_sha256") != sha256_text(json.dumps(enc)):
            bad.append(int(i))
            continue
        out[int(i)] = (enc, [int(x) for x in cr["chain_ids"]])
    return out, bad


@torch.no_grad()
def tf_select(ad, steer, seqs, k, n_tokens, per_chain, tau, seed=PERM_SEED, exclude=()):
    """CL1/CL6 token draw: chain positions t (predicting chain token t+1) whose clean margins at the final and
    the loop-2 exit read-outs are both >= tau; at most `per_chain` per chain, `n_tokens` in all. `exclude`
    (RA-1: the test runs on the tune chains) = (idx, t) pairs already used by the tune draw; never redrawn."""
    dev = next(ad.model.parameters()).device
    exclude = {(int(i), int(t)) for i, t in exclude}
    cand, excluded, n_all, n_tune = [], 0, 0, 0
    for idx in sorted(seqs):
        enc, ch = seqs[idx]
        ids = torch.tensor(enc + ch, dtype=torch.long, device=dev)
        mf, m2, _am = tf_margins(ad.model, steer, ids, k)
        P = len(enc)
        for t in range(P - 1, P + len(ch) - 1):
            n_all += 1
            if (int(idx), int(t)) in exclude:
                n_tune += 1
                continue
            if float(mf[t]) >= tau and float(m2[t]) >= tau:
                cand.append((idx, t))
            else:
                excluded += 1
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cand))
    per, pick = {}, []
    for o in order:
        idx, t = cand[int(o)]
        if per.get(idx, 0) >= per_chain:
            continue
        per[idx] = per.get(idx, 0) + 1
        pick.append((idx, t))
        if len(pick) >= n_tokens:
            break
    return pick, {"n_positions": n_all, "excluded_by_margin": excluded, "n_candidates": len(cand),
                  "n_picked": len(pick), "n_chains": len(per), "tau": tau, "excluded_tune_picks": n_tune}


def tf_run(ad, steer, seqs, picks, arms, k):
    by = {}
    for idx, t in picks:
        by.setdefault(idx, []).append(t)
    rows = []
    names = None
    for idx in sorted(by):
        enc, ch = seqs[idx]
        ids = torch.tensor(enc + ch, dtype=torch.long)
        names, res = tf_chain(ad, steer, str(idx), ids, by[idx], arms, k)
        for t, r in res.items():
            rows.append(dict(r, idx=idx, t=t))
    return names, rows


def boot_diff(a, b, den, cl, n=1000, seed=0):
    """(sum a - sum b) / sum den with a cluster bootstrap: FR_arm - FR_rand on a shared denominator."""
    groups = {}
    for x, y, z, c in zip(a, b, den, cl):
        g = groups.setdefault(c, [0.0, 0.0, 0.0])
        g[0] += x
        g[1] += y
        g[2] += z
    keys = sorted(groups, key=lambda q: str(q))
    A = torch.tensor([groups[q][0] - groups[q][1] for q in keys], dtype=torch.float64)
    D = torch.tensor([groups[q][2] for q in keys], dtype=torch.float64)
    if float(D.sum()) == 0:
        return None
    v0 = float(A.sum() / D.sum())
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=gen)
    dd = D[idx].sum(1)
    ok = dd > 0
    bs = A[idx].sum(1)[ok] / dd[ok]
    q = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [round(v0, 4), round(float(q[0]), 4), round(float(q[1]), 4), len(a), len(keys)]


def tf_summary(names, rows, k, ref="lcdh"):
    """Flip rates, FR vs the LCD-H reference, FR - FR_rand per position set, CF, KL ratio G (CL1, CL2, CL6)."""
    ci = names.index("clean")
    cl = [r["idx"] for r in rows]
    flip = {nm: [int(r["final"][j] != r["final"][ci]) for r in rows] for j, nm in enumerate(names)}
    own = {nm: [int(r["own2"][j] != r["own2"][ci]) for r in rows] for j, nm in enumerate(names)}
    kl = {nm: [float(r["kl"][j]) for r in rows] for j, nm in enumerate(names)}
    out = {"n_tokens": len(rows), "n_chains": len(set(cl)), "flip": {}, "own_flip": {}, "kl": {}}
    for nm in names:
        out["flip"][nm] = boot_cluster(flip[nm], cl)
        out["own_flip"][nm] = boot_cluster(own[nm], cl)
        out["kl"][nm] = boot_cluster(kl[nm], cl)
    if ref in names:
        out["FR"], out["CF"], out["G"] = {}, {}, {}
        for nm in names:
            if nm in ("clean", ref):
                continue
            out["FR"][nm] = boot_ratio(flip[nm], flip[ref], cl)
            out["CF"][nm] = boot_ratio(flip[nm], own[nm], cl)
            out["G"][nm] = boot_ratio(kl[nm], kl[ref], cl)
        out["FR_minus_rand"] = {}
        for st in ("last", "rest", "all"):
            for arm in ("fwd", "back", "abs", "pc2"):
                nm = "%s_%s" % (st, arm)
                rn = "%s_rand" % st if arm != "pc2" else "all_rand_pc2"
                if nm in names and rn in names:
                    out["FR_minus_rand"][nm] = boot_diff(flip[nm], flip[rn], flip[ref], cl)
    return out


def tune_reach(ad, steer, seqs, picks, k, wmax, grid=(0.1, 0.2, 0.3, 0.5, 1.0)):
    """Plan section 9: LCD-H w by reach matching on teacher-forced train chains: the w whose final-token flip
    rate is closest to LCD-L adaptive's on the same tokens (ties to the smaller w). w_abs (audit P1): the w at
    loop 2 whose mean per-position push equals LCD-H's: w* mean||h_k - h_1|| / mean||h_2 - h_1|| (pre-norm)."""
    arms = {"clean": ([], [], None), "lcdl_ad": ([], [], {"mode": "adaptive", "wmax": float(wmax)})}
    for w in grid:
        arms["lcdh_%g" % w] = ([], [{"op": "lcdh", "site": "pre", "loop": k, "pos": "all", "w": float(w)}], None)
    names, rows = tf_run(ad, steer, seqs, picks, arms, k)
    ci = names.index("clean")
    rate = {}
    for j, nm in enumerate(names):
        rate[nm] = sum(int(r["final"][j] != r["final"][ci]) for r in rows) / float(max(1, len(rows)))
    target = rate["lcdl_ad"]
    best = min(grid, key=lambda w: (abs(rate["lcdh_%g" % w] - target), w))
    dev = next(ad.model.parameters()).device
    n2, nk = [], []
    for idx in sorted({i for i, _t in picks}):
        enc, ch = seqs[idx]
        ids = torch.tensor([enc + ch], dtype=torch.long, device=dev)
        steer.rec = {"cols": slice(len(enc) - 1, None), "want": {"pre"}, "post": {}, "pre": {}, "kv": {},
                     "emb": None, "logits": []}
        try:
            fwd(ad.model, ids, logits_to_keep=1)
            pre = steer.rec["pre"]
        finally:
            steer.rec = None
        h1 = pre[1][0].float()
        n2.append(float((pre[2][0].float() - h1).norm(dim=-1).mean()))
        nk.append(float((pre[k][0].float() - h1).norm(dim=-1).mean()))
    ratio = (sum(nk) / len(nk)) / max(1e-12, sum(n2) / len(n2))
    return {"grid": list(grid), "flip_rate": rate, "target_lcdl_ad": target, "w_star": best,
            "w_abs": round(best * ratio, 4), "mean_norm_hk_minus_h1": sum(nk) / len(nk),
            "mean_norm_h2_minus_h1": sum(n2) / len(n2), "n_tokens": len(rows)}


# ====================================================================== drivers: answer site (CL4) and presence (CL4)
def _done_ids(path, key="idx"):
    """Items already written to an incremental jsonl (CL15)."""
    if not path or not os.path.exists(path):
        return set()
    return {int(r[key]) for r in read_jsonl(path) if key in r}


def anschain_geometry(tok, cfg, it, ca, cb, srows):
    """CL4/CL14 answer-site geometry for one item: preA = A prompt + A's own chain up to the first answer digit;
    donor preB = B prompt + B's own chain up to its first answer digit; preB2 = B prompt + A's chain (aligned,
    diagnostic donor). Returns (None, reason) when a chain has no '#### gold' site."""
    ra = srows[int(it["idx"])]
    rb = dict(ra, input=it["question_B"])
    pA, pB = prompt_texts(tok, "gsm8k", [ra, rb], cfg["chat"])
    a_ids = tok(pA, add_special_tokens=False)["input_ids"]
    b_ids = tok(pB, add_special_tokens=False)["input_ids"]
    tA = answer_site(tok, ca["chain_ids"], it["gold_A"])
    tB = answer_site(tok, cb["chain_ids"], it["gold_B"])
    if tA is None or tB is None:
        return None, ("no_site_A" if tA is None else "no_site_B")
    preA = a_ids + list(ca["chain_ids"][:tA])
    preB = b_ids + list(cb["chain_ids"][:tB])
    preB2 = b_ids + list(ca["chain_ids"][:tA])
    return {"preA": preA, "preB": preB, "preB2": preB2, "site": len(preA) - 1, "dsite": len(preB) - 1,
            "a_first": int(ca["chain_ids"][tA]), "b_first": int(cb["chain_ids"][tB])}, None


def anschain_defs(k):
    """CL14 rows: (name, edits, donor key). 'readout_identity' (was 'complete') = the Llast@K state plus that
    position's K/V at every call: it checks the decoder and the donor alignment only, never the Llast@1 claim."""
    P = lambda j: {"op": "patch", "site": "post", "loop": j, "pos": "last_prompt", "a": 1.0}
    S = lambda c: {"op": "rand", "site": "post", "loop": 1, "pos": "last_prompt", "seed": 0, "ref": "scale",
                   "scale": float(c)}
    ident = [P(k), {"op": "kv", "loops": list(range(1, k + 1)), "pos": "last_prompt", "a": 1.0}]
    return [("clean", [], "B"), ("o1_L1", [P(1)], "B"), ("o1_LK", [P(k)], "B"), ("readout_identity", ident, "B"),
            ("rand_L1", [{"op": "rand", "site": "post", "loop": 1, "pos": "last_prompt", "seed": 0,
                          "ref": "patch"}], "B"),
            ("reach_c1", [S(1.0)], "B"), ("reach_c2", [S(2.0)], "B"), ("tiny", [S(1e-3)], "B"),
            ("d2_o1_L1", [P(1)], "B2"), ("d2_o1_LK", [P(k)], "B2"), ("d2_readout_identity", ident, "B2")]


@torch.no_grad()
def anschain_run(ad, steer, cfg, items, chainsA, chainsB, n_new=12, rows_path=None):
    """CL4 + CL14 answer-site rows per item (incremental, resumable). Per row: the greedy continuation (prod's
    decode with the hooks, 12 tokens or a newline), first token, parsed answer and outcome; from a batched
    no-cache prefill at the site: the loop-1 exit argmax and KL(clean || row) at the loop-1 exit and at the final
    read-out. Per item: the donor's own loop-1 exit argmax at its site."""
    from .tasks import rows as task_rows
    tok = ad.tok
    k = cfg["k"]
    dev = next(ad.model.parameters()).device
    srows = {int(r["idx"]): r for r in task_rows("gsm8k")}
    eos = [tok.eos_token_id]
    done = _done_ids(rows_path)
    recs, skipped = [], {"no_chain": 0, "no_site_A": 0, "no_site_B": 0, "done_before": 0}
    defs = anschain_defs(k)
    for it in items:
        idx = int(it["idx"])
        if idx in done:
            skipped["done_before"] += 1
            continue
        ca, cb = chainsA.get(idx), chainsB.get(idx)
        if ca is None or cb is None:
            skipped["no_chain"] += 1
            continue
        g, why = anschain_geometry(tok, cfg, it, ca, cb, srows)
        if g is None:
            skipped[why] += 1
            continue
        preA, site = g["preA"], g["site"]
        keys = []
        for nm, ed, dk in defs:
            key = "ans|%d|%s" % (idx, nm)
            donor = g["preB"] if dk == "B" else g["preB2"]
            ds = g["dsite"] if dk == "B" else site
            steer.rows[key] = {"plen": len(preA), "span": (site, site + 1), "a_ids": preA, "b_ids": donor,
                               "edits": ed, "lcdl": None, "donor_range": (ds, ds + 1), "t0": site}
            keys.append(key)
        steer.ensure_donors(keys)
        batch = torch.tensor([preA] * len(keys), dtype=torch.long, device=dev)
        steer.rec = {"cols": slice(-1, None), "want": {"post"}, "post": {}, "pre": {}, "kv": {}, "emb": None,
                     "logits": []}
        steer.batch_keys = keys
        try:
            o = fwd(ad.model, batch, logits_to_keep=1)
            post = steer.rec["post"]
        finally:
            steer.rec = None
            steer.batch_keys = None
        zf = o.logits[:, -1].float()
        z1 = F.linear(post[1][:, -1], ad.model.lm_head.weight).float()

        def kl0(z):
            lp = torch.log_softmax(z, -1)
            return (lp[0].exp() * (lp[0] - lp)).sum(-1).clamp(min=0).tolist()
        klf, kl1 = kl0(zf), kl0(z1)
        am1 = z1.argmax(-1).tolist()
        steer.rec = {"cols": slice(-1, None), "want": {"post"}, "post": {}, "pre": {}, "kv": {}, "emb": None,
                     "logits": []}
        try:
            fwd(ad.model, torch.tensor([g["preB"]], dtype=torch.long, device=dev), logits_to_keep=1)
            donor_l1 = int(F.linear(steer.rec["post"][1][:, -1], ad.model.lm_head.weight).float().argmax(-1)[0])
        finally:
            steer.rec = None
        gen, _t = ad.decode([preA] * len(keys), n_new, k, eos, stop_strings=["\n"], row_ids=keys)
        rec = {"idx": idx, "site": site, "first_digit_differs": it.get("first_digit_differs"),
               "n_span_tokens": it.get("n_span_tokens"), "b_first_token": g["b_first"],
               "a_first_token": g["a_first"], "donor_loop1_argmax": donor_l1}
        for bi, ((nm, _e, _d), gg, key) in enumerate(zip(defs, gen, keys)):
            txt = tok.decode(gg, clean_up_tokenization_spaces=False)
            m = RE_NUM.search(txt)
            ans = m.group(0).replace(",", "") if m else None
            rec[nm] = {"first_token": int(gg[0]) if gg else None, "answer": ans,
                       "outcome": outcome(ans, it["gold_A"], it["gold_B"]), "loop1_argmax": int(am1[bi]),
                       "loop1_kl": round(float(kl1[bi]), 6), "final_kl": round(float(klf[bi]), 6)}
            steer.rows.pop(key, None)
            steer.donors.pop(key, None)
        recs.append(rec)
        if rows_path:
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return recs, skipped


def anschain_gate_items(tok, cfg, items, chainsA, chainsB, n=3):
    """CL14 (c): gate items on the ans-chain geometry itself (prefix = A prompt + A chain, site = the last prefix
    position, donor = B prompt + A chain, aligned), for check_suite."""
    from .tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows("gsm8k")}
    out = []
    for it in items:
        ca, cb = chainsA.get(int(it["idx"])), chainsB.get(int(it["idx"]))
        if ca is None or cb is None:
            continue
        g, _w = anschain_geometry(tok, cfg, it, ca, cb, srows)
        if g is None:
            continue
        out.append({"idx": int(it["idx"]), "a_ids": g["preA"], "b_ids": g["preB2"],
                    "span": [g["site"], g["site"] + 1], "plen": len(g["preA"])})
        if len(out) >= n:
            break
    return out


def anschain_summary(recs, gates=None):
    """CL4 + CL14 read-out with the frozen gates: (a) the edit landed at loop 1, (b) Llast@1 can reach the output,
    (c) G1-G3 on the ans-chain geometry. Llast@1 is 'not carried' only if (a), (b) and (c) pass."""
    out = {"rows": {}}
    names = [nm for nm, _e, _d in anschain_defs(4)]
    fdd = [r for r in recs if r.get("first_digit_differs")]
    ids = [r["idx"] for r in recs]
    fids = [r["idx"] for r in fdd]
    for nm in names:
        if not recs or nm not in recs[0]:
            continue
        gB = [int(r[nm]["outcome"] == "B") for r in recs]
        gA = [int(r[nm]["outcome"] == "A") for r in recs]
        cell = {"n_items": len(recs), "n_clusters": len(set(ids)), "gold_B": boot_cluster(gB, ids),
                "gold_A": boot_cluster(gA, ids), "n_first_digit_differs": len(fdd),
                "first_token_is_B_first_digit": boot_cluster([int(r[nm]["first_token"] == r["b_first_token"])
                                                              for r in fdd], fids) if fdd else None,
                "first_digit_flip_vs_clean": boot_cluster([int(r[nm]["first_token"] != r["clean"]["first_token"])
                                                           for r in fdd], fids) if fdd else None,
                "final_kl_median": float(np.median([r[nm]["final_kl"] for r in recs])),
                "loop1_kl_median": float(np.median([r[nm]["loop1_kl"] for r in recs]))}
        st = {}
        for r in recs:
            s = "1" if r.get("n_span_tokens") == 1 else ">=2"
            st.setdefault(s, []).append(int(r[nm]["outcome"] == "B"))
        cell["strata_gold_B_by_span_tokens"] = {kk: [round(sum(v) / float(len(v)), 4), len(v)]
                                                for kk, v in st.items()}
        out["rows"][nm] = cell
    if recs and "o1_L1" in recs[0] and "tiny" in recs[0]:
        n = float(len(recs))
        match = sum(int(r["o1_L1"]["loop1_argmax"] == r["donor_loop1_argmax"]) for r in recs) / n
        moved = sum(int(r["o1_L1"]["loop1_kl"] > 10 * r["tiny"]["loop1_kl"]) for r in recs) / n
        ga = {"loop1_argmax_matches_donor": round(match, 4), "loop1_kl_above_10x_tiny": round(moved, 4),
              "clean_vs_donor_loop1_differ": round(sum(int(r["clean"]["loop1_argmax"] != r["donor_loop1_argmax"])
                                                       for r in recs) / n, 4)}
        ga["pass"] = bool(match >= 0.95 and moved >= 0.9)
        rc2 = out["rows"].get("reach_c2", {})
        fl = (rc2.get("first_digit_flip_vs_clean") or [0.0])[0]
        tiny_med = out["rows"]["tiny"]["final_kl_median"]
        gb = {"reach_c2_first_digit_flip": fl, "reach_c2_final_kl_median": rc2.get("final_kl_median"),
              "tiny_final_kl_median": tiny_med,
              "reach_c1_first_digit_flip": ((out["rows"].get("reach_c1", {}).get("first_digit_flip_vs_clean")
                                             or [None])[0])}
        gb["pass"] = bool(fl >= 0.10 or (rc2.get("final_kl_median") or 0.0) >= 10 * max(tiny_med, 1e-12))
        gc = bool(gates and all((gates.get(g_) or {}).get("pass") for g_ in ("gate_zero_edit", "gate_placement",
                                                                             "gate_cache_equiv")))
        ff = (out["rows"]["o1_L1"].get("first_digit_flip_vs_clean") or [None])[0]
        if ga["pass"] and gb["pass"] and gc:
            verdict = "not carried" if ff is not None and ff <= 0.10 else "carried or partial (see value)"
        else:
            verdict = "untested (a control failed)"
        out["gates"] = {"a_landed_loop1": ga, "b_reach_Llast1": gb, "c_decoder_gates": gc, "Llast1_verdict": verdict}
    return out


@torch.no_grad()
def presence_run(ad, steer, cfg, items, rows_path=None):
    """CL4 presence read-out pair_ab_proj (DESIGN 9.1) at the last prompt position, per loop end, for clean A,
    clean B, r06 (O1 Llast@1) and r07 (O1 Llast@K); plus the batch-1 first-token flip of r06/r07 vs clean A."""
    from .tasks import rows as task_rows
    tok = ad.tok
    k = cfg["k"]
    dev = next(ad.model.parameters()).device
    srows = {int(r["idx"]): r for r in task_rows("gsm8k")}
    out = []
    done = _done_ids(rows_path)
    for it in items:
        idx = int(it["idx"])
        if idx in done:
            continue
        ra = srows[idx]
        rb = dict(ra, input=it["question_B"])
        pA, pB = prompt_texts(tok, "gsm8k", [ra, rb], cfg["chat"])
        a_ids = tok(pA, add_special_tokens=False)["input_ids"]
        b_ids = tok(pB, add_special_tokens=False)["input_ids"]
        s0, s1 = it["span"]
        P = len(a_ids)
        defs = [("A", [], a_ids), ("B", [], b_ids),
                ("r06", [{"op": "patch", "site": "post", "loop": 1, "pos": "last_prompt", "a": 1.0}], a_ids),
                ("r07", [{"op": "patch", "site": "post", "loop": k, "pos": "last_prompt", "a": 1.0}], a_ids)]
        keys = []
        for nm, ed, seq in defs:
            key = "pres|%d|%s" % (idx, nm)
            steer.rows[key] = {"plen": P, "span": (s0, s1), "a_ids": seq, "b_ids": b_ids, "edits": ed, "lcdl": None,
                               "donor_range": (s0, P), "t0": s0}
            keys.append(key)
        steer.ensure_donors(keys)
        # CL4: batch 1 per row, prefill only. One forward per row, so mu_B has the donor's own batch-1 numerics and
        # the r06 loop-1 value is exactly s(h_B) = 0.5 (a check that the edit landed); a shared batch would add
        # row-dependent bf16 noise comparable to |mu_B - mu_A| at the last prompt position.
        first, posts = [], []
        for (nm, ed, seq), key in zip(defs, keys):
            steer.rec = {"cols": slice(-1, None), "want": {"post"}, "post": {}, "pre": {}, "kv": {}, "emb": None,
                         "logits": []}
            steer.batch_keys = [key]
            try:
                o = fwd(ad.model, torch.tensor([seq], dtype=torch.long, device=dev), logits_to_keep=1)
                posts.append(steer.rec["post"])
            finally:
                steer.rec = None
                steer.batch_keys = None
            first.append(int(o.logits[0, -1].float().argmax(-1)))
        r = {"idx": idx, "first_token": dict(zip([d[0] for d in defs], first)), "s": {}, "dv_norm": {},
             "dv_rel": {}}
        for j in range(1, k + 1):
            h = torch.stack([p[j][0, -1].float() for p in posts])
            muA, muB = h[0], h[1]
            dv = muB - muA
            den = float((dv * dv).sum())
            sj = (((h - (muA + muB) / 2) * dv).sum(-1) / max(den, 1e-12)).tolist()
            r["s"][str(j)] = dict(zip([d[0] for d in defs], [round(x, 4) for x in sj]))
            r["dv_norm"][str(j)] = round(den ** 0.5, 5)
            r["dv_rel"][str(j)] = round(den ** 0.5 / max(float(muA.norm()), 1e-12), 6)
        out.append(r)
        if rows_path:
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        for key in keys:
            steer.rows.pop(key, None)
            steer.donors.pop(key, None)
    return out


# ====================================================================== analysis
def analyze_a(cfg, out_root, rows=None, tok=None):
    """Arm (a) cells -> outcomes, transfer T, first mention, token identity, floors, paired row comparisons."""
    sel = json.load(open(os.path.join(out_root, "items.json"), encoding="utf-8"))
    scored = {int(it["idx"]): it for it in sel["scored"]}
    pool = {int(r["idx"]): r for r in core_pool()[0]}
    sent = [int(s["idx"]) for s in sel["sentinels"]]
    rows = rows or [r for r in cfg["rows"] + cfg.get("rows_ext", []) if os.path.isdir(os.path.join(out_root, r))]
    cells = {r: load_cells(os.path.join(out_root, r)) for r in rows}
    chains = {r: load_chains(os.path.join(out_root, r)) for r in rows}
    res = {"setting": cfg["name"], "n_scored": len(scored), "rows": {}}
    oc = {}
    for r in ["r01"] + [x for x in rows if x != "r01"]:
        if r not in cells:
            continue
        c = cells[r]
        out = {}
        for idx, it in scored.items():
            row = c.get(idx)
            if row is None:
                continue
            out[idx] = outcome(row.get("trace_answer"), it["gold_A"], it["gold_B"])
        oc[r] = out
        ids = sorted(out)
        gB = [int(out[i] == "B") for i in ids]
        gA = [int(out[i] == "A") for i in ids]
        ne = [int(out[i] == "neither") for i in ids]
        cell = {"n_items": len(ids), "n_clusters": len(ids), "gold_B": boot_cluster(gB, ids),
                "gold_A": boot_cluster(gA, ids), "neither": boot_cluster(ne, ids)}
        refB = oc.get("r01")
        if refB:
            nb = sum(1 for i in ids if refB.get(i) == "B")
            cell["T"] = round(sum(gB) / float(max(1, nb)), 4)
        st = {}
        for name, fn in (("span_tokens", lambda it: "1" if it["n_span_tokens"] == 1 else ">=2"),
                         ("value_first_digit_differs", lambda it: str(it["value_first_digit_differs"])),
                         ("gold_B_chain_status", lambda it: it["gold_B_chain_status"])):
            g = {}
            for i in ids:
                g.setdefault(fn(scored[i]), []).append(int(out[i] == "B"))
            st[name] = {kk: [round(sum(v) / float(len(v)), 4), len(v)] for kk, v in g.items()}
        cell["strata_gold_B"] = st
        if tok is not None and chains.get(r):
            fm = {}
            for i in ids:
                ch = chains[r].get(i)
                if ch is None:
                    continue
                txt = tok.decode(ch["chain_ids"], clean_up_tokenization_spaces=False)
                fm[i] = first_mention(txt, pool[i]["value_A"], pool[i]["value_B"])
            cell["first_mention"] = {v: sum(1 for x in fm.values() if x == v) for v in ("B", "A", "neither")}
            xb = [i for i in fm if fm[i] == "B"]
            cell["p_goldB_given_xprime"] = round(sum(1 for i in xb if out.get(i) == "B") / float(len(xb)), 4) if xb else None
        for ref in ("r00", "r01"):
            if chains.get(ref) and chains.get(r):
                same, fd = 0, []
                for i in ids:
                    a, b = chains[r].get(i), chains[ref].get(i)
                    if a is None or b is None:
                        continue
                    d = first_divergence(a["chain_ids"], b["chain_ids"])
                    same += int(d is None)
                    if d is not None:
                        fd.append(d)
                cell["token_identical_vs_%s" % ref] = round(same / float(max(1, len(ids))), 4)
                cell["first_divergence_median_vs_%s" % ref] = (sorted(fd)[len(fd) // 2] if fd else None)
        if "r00" in cells and r != "r00":
            dis, n = 0, 0
            for i in sent:
                a, b = cells[r].get(i), cells["r00"].get(i)
                if a is None or b is None:
                    continue
                n += 1
                dis += int(bool(a.get("trace_correct")) != bool(b.get("trace_correct")))
            cell["sentinel_discordance"] = [round(dis / float(max(1, n)), 4), n]
        cell["parse_rate"] = round(sum(1 for i in ids if c[i].get("trace_answer") is not None) / float(max(1, len(ids))), 4)
        cell["mean_n_trace"] = round(sum(c[i]["n_trace"] for i in ids) / float(max(1, len(ids))), 1)
        disc = [i for i in ids if "r00" in oc and r != "r00" and oc["r00"].get(i) != out[i]]
        cell["sentinel_coverage"] = sentinel_coverage(os.path.join(out_root, r, "decode_log.jsonl"), set(scored),
                                                      set(sent), discordant=disc)
        res["rows"][r] = cell
    for fl in ("r20", "r21"):
        if fl in oc and "r00" in oc:
            ids = [i for i in oc[fl] if i in oc["r00"]]
            res["%s_floor" % fl] = {"outcome_discordance": round(sum(1 for i in ids if oc[fl][i] != oc["r00"][i])
                                                                 / float(max(1, len(ids))), 4), "n": len(ids),
                                    "token_identical_vs_r00": res["rows"][fl].get("token_identical_vs_r00"),
                                    "neither": res["rows"][fl].get("neither")}
    bwd = os.path.join(out_root, "bw1_r00")
    if os.path.isdir(bwd) and chains.get("r00"):
        cb = load_chains(bwd)
        cc = load_cells(bwd)
        same, fd, dis, n = 0, [], 0, 0
        for i, ch in cb.items():
            ref = chains["r00"].get(i)
            if ref is None or i not in scored:
                continue
            n += 1
            d = first_divergence(ch["chain_ids"], ref["chain_ids"])
            same += int(d is None)
            if d is not None:
                fd.append(d)
            if i in cc and i in oc.get("r00", {}):
                dis += int(outcome(cc[i].get("trace_answer"), scored[i]["gold_A"], scored[i]["gold_B"]) != oc["r00"][i])
        res["bw_check"] = {"n": n, "token_identical_bw1_vs_bw16": round(same / float(max(1, n)), 4),
                           "first_divergence_median": (sorted(fd)[len(fd) // 2] if fd else None),
                           "first_divergence_min": (min(fd) if fd else None),
                           "outcome_discordance": round(dis / float(max(1, n)), 4)}
    comps = [("r04", "r03"), ("r05", "r03"), ("r12", "r03"), ("r13", "r03"), ("r19", "r03")]
    pc, ps = {}, []
    for a_, b_ in comps:
        if a_ in oc and b_ in oc:
            ids = sorted(set(oc[a_]) & set(oc[b_]))
            d = [int(oc[a_][i] == "B") - int(oc[b_][i] == "B") for i in ids]
            bb = sum(1 for i in ids if oc[a_][i] == "B" and oc[b_][i] != "B")
            cc = sum(1 for i in ids if oc[a_][i] != "B" and oc[b_][i] == "B")
            p = mcnemar_exact(bb, cc)
            pc["%s-%s" % (a_, b_)] = {"diff": boot_cluster(d, ids), "b": bb, "c": cc, "p": p}
            ps.append(p)
    if ps:
        for kk, adj in zip(list(pc), holm(ps)):
            pc[kk]["p_holm"] = adj
    res["paired"] = pc
    return res


def sentinel_coverage(log_path, scored, sentinels, discordant=(), thresh=0.8):
    """CL17: from decode_log.jsonl, the fraction of scored-row decode steps that ran in a group with >= 1
    sentinel; the flag 'sentinel-sparse' below `thresh`; discordant items whose steps ran mostly (> 50%) in
    sentinel-free groups."""
    if not os.path.exists(log_path):
        return None
    tot, cov, per = 0, 0, {}
    groups_wo, n_calls = 0, 0
    for r in read_jsonl(log_path):
        ids = [int(x) if str(x).lstrip("-").isdigit() else x for x in r["rows"]]
        has = any(i in sentinels for i in ids)
        n_calls += 1
        groups_wo += int(not has)
        for i, n in zip(ids, r["n_out"]):
            if i in scored:
                tot += int(n)
                cov += int(n) if has else 0
                p = per.setdefault(i, [0, 0])
                p[0] += int(n)
                p[1] += int(n) if not has else 0
    frac = cov / float(tot) if tot else None
    mostly_free = sorted(i for i in discordant if i in per and per[i][0] and per[i][1] / float(per[i][0]) > 0.5)
    return {"steps_scored": tot, "frac_steps_with_sentinel": None if frac is None else round(frac, 4),
            "decode_calls": n_calls, "calls_without_sentinel": groups_wo,
            "flag": ("sentinel-sparse" if frac is not None and frac < thresh else None),
            "discordant_mostly_sentinel_free": mostly_free[:50], "n_discordant": len(list(discordant))}


def analyze_b(cfg, out_root, rows=None, which="eval", settle_dir=None):
    """Arm (b) cells -> accuracy, paired vs unedited (McNemar, Holm), termination, degenerate rate (P2), tokens,
    seconds per row, sentinel floor, SETTLE agreement of the unedited row and SETTLE-status strata (section 8, 11)."""
    items = json.load(open(os.path.join(out_root, "items.json"), encoding="utf-8"))
    scored = items["eval"] if which == "eval" else items.get(which, [])
    sent = [int(s["idx"]) for s in items.get("sentinels", [])]
    rows = rows or [r for r in cfg["rows"] + cfg.get("rows_ext", [])
                    if os.path.isdir(os.path.join(out_root, "%s_%s" % (which, r)))]
    cells = {r: load_cells(os.path.join(out_root, "%s_%s" % (which, r))) for r in rows}
    res = {"setting": cfg["name"], "rows": {}}
    ref = cells.get("unedited", {})
    if not ref and cfg.get("unedited_from_settle") and settle_dir and which == "eval":
        # RA-1 no-rerun rule: the unedited row is SETTLE's own cell (budget 4096, correct_v2). Paired differences
        # then include the cross-run drift; each job's sentinel discordance against SETTLE measures that floor.
        ref = {i: {"correct_v2": v} for i, v in settle_cells(os.path.join(settle_dir, cfg["settle"])).items()}
        res["reference"] = "SETTLE cells %s at budget 4096 (no rerun, RA-1)" % cfg["settle"]
    settle, strata_files = None, []
    if settle_dir and cfg.get("settle") and which in ("eval", "cal"):
        settle = settle_cells(os.path.join(settle_dir, cfg["settle"]))
        strata_files = [(f, settle_cells(os.path.join(settle_dir, f))) for f in cfg.get("settle_strata", [])]
    ps, keys = [], []
    for r in rows:
        jd = os.path.join(out_root, "%s_%s" % (which, r))
        c = cells[r]
        ch = load_chains(jd)
        ids = [i for i in scored if i in c]
        acc = [int(bool(c[i].get("correct_v2"))) for i in ids]
        degen = {i: degenerate((ch.get(i) or {}).get("chain_ids") or [],
                               c[i]["n_trace"] if c[i].get("stop_marker") is not None else None, cfg["horizon"])
                 for i in ids}
        acc_dw = [a_ * int(not degen[i]) for a_, i in zip(acc, ids)]
        cell = {"n_items": len(ids), "n_clusters": len(ids), "acc": boot_cluster(acc, ids),
                "acc_degenerate_wrong": boot_cluster(acc_dw, ids),
                "degenerate": round(sum(degen.values()) / float(max(1, len(ids))), 4),
                "unstopped": round(sum(1 for i in ids if c[i].get("stop_marker") is None)
                                   / float(max(1, len(ids))), 4),
                "mean_n_trace": round(sum(c[i]["n_trace"] for i in ids) / float(max(1, len(ids))), 1),
                "median_n_trace": float(np.median([c[i]["n_trace"] for i in ids])) if ids else None}
        sj = os.path.join(jd, "steer_job.json")
        if os.path.exists(sj):
            meta = json.load(open(sj, encoding="utf-8"))
            nrows = len(json.load(open(os.path.join(jd, "only_rows.json"), encoding="utf-8")))
            cell["seconds_per_row"] = round(float(meta.get("seconds", 0)) / max(1, nrows), 3)
            cell["forwards_per_row"] = round(float(meta.get("n_forwards", 0)) / max(1, nrows), 1)
        if settle is not None:
            both = [i for i in ids if i in settle]
            if r == "unedited":
                cell["settle_agreement"] = [round(sum(int(bool(c[i]["correct_v2"]) == settle[i]) for i in both)
                                                  / float(max(1, len(both))), 4), len(both)]
                cell["settle_acc_same_rows"] = round(sum(int(settle[i]) for i in both) / float(max(1, len(both))), 4)
            strata = {}
            for i in ids:
                key = "k%d_%s" % (cfg["k"], "solved" if settle.get(i) else "unsolved")
                for f, sc in strata_files:
                    key += "|%s_%s" % (f.split("_")[-1].split(".")[0], "solved" if sc.get(i) else "unsolved")
                strata.setdefault(key, []).append(i)
            cell["strata_acc"] = {kk: [round(sum(int(bool(c[i]["correct_v2"])) for i in v) / float(len(v)), 4), len(v)]
                                  for kk, v in sorted(strata.items())}
        if r != "unedited" and ref:
            both = [i for i in ids if i in ref]
            d = [int(bool(c[i]["correct_v2"])) - int(bool(ref[i]["correct_v2"])) for i in both]
            wr = sum(1 for i in both if c[i]["correct_v2"] and not ref[i]["correct_v2"])
            rw = sum(1 for i in both if not c[i]["correct_v2"] and ref[i]["correct_v2"])
            cell.update({"delta": boot_cluster(d, both), "wrong_to_right": wr, "right_to_wrong": rw,
                         "p_mcnemar": mcnemar_exact(wr, rw)})
            if "strata_acc" in cell:
                sw = {}
                for kk, v in strata.items():
                    vb = [i for i in v if i in ref]
                    sw[kk] = {"n": len(vb),
                              "wrong_to_right": sum(1 for i in vb if c[i]["correct_v2"] and not ref[i]["correct_v2"]),
                              "right_to_wrong": sum(1 for i in vb if not c[i]["correct_v2"] and ref[i]["correct_v2"])}
                cell["strata_paired"] = sw
            ps.append(cell["p_mcnemar"])
            keys.append(r)
            if sent:
                dis, n = 0, 0
                for i in sent:
                    if i in c and i in ref:
                        n += 1
                        dis += int(bool(c[i]["correct_v2"]) != bool(ref[i]["correct_v2"]))
                cell["sentinel_discordance"] = [round(dis / float(max(1, n)), 4), n]
        cell["sentinel_coverage"] = sentinel_coverage(os.path.join(out_root, "%s_%s" % (which, r), "decode_log.jsonl"),
                                                      set(int(i) for i in scored), set(sent))
        res["rows"][r] = cell
    for kk, adj in zip(keys, holm(ps) if ps else []):
        res["rows"][kk]["p_holm"] = adj
    return res


@torch.no_grad()
def tf_equiv(ad, steer, seqs, picks, k, w, n_chains=5):
    """Gate G3b (CL1): the cached descending single-token argmax agrees with a full no-cache forward, clean and
    with the loop-2 contrast at every position (pass: >= 0.999 of tokens, at most one miss below 1000)."""
    by = {}
    for idx, t in picks:
        by.setdefault(idx, []).append(t)
    dev = next(ad.model.parameters()).device
    lc = {"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": float(w)}
    arms = {"clean": ([], [], None), "all_fwd": ([lc], [lc], None)}
    agree, n, gaps = 0, 0, []
    for idx in sorted(by)[:n_chains]:
        enc, ch = seqs[idx]
        ids = torch.tensor(enc + ch, dtype=torch.long)
        names, res = tf_chain(ad, steer, "eq%d" % idx, ids, by[idx], arms, k)
        for j, nm in enumerate(names):
            key = "eqfull|%d|%s" % (idx, nm)
            steer.rows[key] = {"plen": None, "span": None, "a_ids": None, "edits": arms[nm][0], "lcdl": None}
            steer.batch_keys = [key]
            try:
                zf = fwd(ad.model, ids.view(1, -1).to(dev), logits_to_keep=0).logits[0].float()
            finally:
                steer.batch_keys = None
                steer.rows.pop(key, None)
            for t, r in res.items():
                n += 1
                agree += int(int(zf[t].argmax()) == int(r["final"][j]))
    ok = n > 0 and (agree >= 0.999 * n if n >= 1000 else agree >= n - 1)
    return {"argmax_agree": agree, "n": n, "pass": bool(ok)}


# ====================================================================== teacher-forced test v2 (CL10-CL13)
TF_GRID = (0.1, 0.2, 0.3, 0.5, 1.0)
TF_SETS = ("last", "rest", "all")
TF_MATCH = ("rel", "abs", "rn")


def tf_arm_name(j, st, dr, m, w):
    return "L%d_%s_%s_%s@%g" % (j, st, dr, m, w)


def tf_arms_v2(k, w_star, abs_ratio, wmax, grid=TF_GRID, loops=(2, 3), back_at=None, rdirs_at=None):
    """CL11-CL13 arm table: {name: {"pre": [edit], "tok": [edit], "lcdl": None | dict, "meta": {...}}}.

    "pre" edits act in the chain prefill (every position, hence the K/V of the positions before t); "tok" edits
    act on the scored token's own forward. Position sets (CL1): last = tok only, rest = pre only, all = both.
    Strengths per loop j (CL11): rel = w; abs = w x abs_ratio[j] with abs_ratio[j] = mean||h_K - h_1|| /
    mean||h_j - h_1|| (pre-norm); rn = per-position coefficient w x base_j(p) (marker "rn", resolved per chain by
    tf_resolve; base_j from tf_clean_states). rand = a random direction with the arm's own per-position norm,
    one direction per (chain, loop, position) for every strength and set (common random numbers, key '@rand<j>').
    back = -coef, at w* only. Reference per w: lcdh@w = LoopCD-Hidden h' = h_K + w (h_K - h_1) at loop K on the
    scored token (2610.02185 Eq.(2); a loop-K edit changes no K/V), and randK@w, its matched random.
    RA-1 (2026-10-08): `rdirs_at` (default w*) adds two more matched randoms per (loop, set, matching): ral = the
    same Gaussian projected on unit(h_j - h_1) with a random sign, rort = its part orthogonal to h_j - h_1
    (EXPERIMENT_MATRIX X9, CB-F5), each at the arm's own per-position norm."""
    back_at = w_star if back_at is None else back_at
    rdirs_at = w_star if rdirs_at is None else rdirs_at
    loops = [int(j) for j in loops if 2 <= int(j) < int(k)]
    arms = {"clean": {"pre": [], "tok": [], "lcdl": None, "meta": {"kind": "clean"}}}

    def ed(j, m, coef, rand):
        e = {"op": "rand" if rand else "lcdh", "site": "pre", "loop": int(j), "pos": "all"}
        if rand:
            e.update({"seed": 0, "ref": "lcdh", "rkey": "@rand%d" % j})
            if rand in ("aligned", "orth"):
                e["dir"] = rand
        key = "ref_w" if rand else "w"
        if m == "rn":
            e["rn"] = [int(j), float(coef), key]
        else:
            e[key] = float(coef)
        return e

    def ratio(j):
        return float(abs_ratio[j] if j in abs_ratio else abs_ratio[str(j)])

    for w in grid:
        w = float(w)
        arms["lcdh@%g" % w] = {"pre": [], "tok": [{"op": "lcdh", "site": "pre", "loop": int(k), "pos": "all", "w": w}],
                               "lcdl": None, "meta": {"kind": "lcdh", "w": w}}
        arms["randK@%g" % w] = {"pre": [], "tok": [{"op": "rand", "site": "pre", "loop": int(k), "pos": "all", "seed": 0,
                                                    "ref": "lcdh", "ref_w": w, "rkey": "@rand%d" % k}],
                                "lcdl": None, "meta": {"kind": "randK", "w": w}}
        for j in loops:
            for m in TF_MATCH:
                coef = w if m != "abs" else w * ratio(j)
                dirs = [("fwd", coef, False), ("rand", coef, True)]
                if abs(w - float(back_at)) < 1e-12:
                    dirs.append(("back", -coef, False))
                if abs(w - float(rdirs_at)) < 1e-12:
                    dirs += [("ral", coef, "aligned"), ("rort", coef, "orth")]
                for dr, c, rnd in dirs:
                    e = ed(j, m, c, rnd)
                    for st in TF_SETS:
                        pre = [e] if st in ("rest", "all") else []
                        tok = [e] if st in ("last", "all") else []
                        arms[tf_arm_name(j, st, dr, m, w)] = {
                            "pre": pre, "tok": tok, "lcdl": None,
                            "meta": {"kind": "arm", "j": j, "set": st, "dir": dr, "m": m, "w": w}}
    arms["lcdl_ad"] = {"pre": [], "tok": [], "lcdl": {"mode": "adaptive", "wmax": float(wmax)}, "meta": {"kind": "lcdl"}}
    if 2 in loops:
        pc = {"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": -1.0}
        pr = {"op": "rand", "site": "pre", "loop": 2, "pos": "all", "seed": 0, "ref": "lcdh", "ref_w": -1.0,
              "rkey": "@rand2"}
        tiny = {"op": "rand", "site": "pre", "loop": 2, "pos": "all", "seed": 0, "ref": "scale", "scale": 1e-3,
                "rkey": "@rand2"}
        arms["pc2"] = {"pre": [pc], "tok": [pc], "lcdl": None, "meta": {"kind": "pc2"}}
        arms["pc2_rand"] = {"pre": [pr], "tok": [pr], "lcdl": None, "meta": {"kind": "pc2_rand"}}
        arms["tiny"] = {"pre": [tiny], "tok": [tiny], "lcdl": None, "meta": {"kind": "tiny"}}
    return arms


def tf_grid_arms(k, wmax, grid=TF_GRID):
    """Tuning table (section 9, CL10): clean, LCD-L adaptive, LCD-H at every grid w (loop K, scored token)."""
    arms = {"clean": {"pre": [], "tok": [], "lcdl": None, "meta": {"kind": "clean"}},
            "lcdl_ad": {"pre": [], "tok": [], "lcdl": {"mode": "adaptive", "wmax": float(wmax)},
                        "meta": {"kind": "lcdl"}}}
    for w in grid:
        arms["lcdh@%g" % w] = {"pre": [], "tok": [{"op": "lcdh", "site": "pre", "loop": int(k), "pos": "all",
                                                   "w": float(w)}], "lcdl": None,
                               "meta": {"kind": "lcdh", "w": float(w)}}
    return arms


@torch.no_grad()
def tf_clean_states(model, steer, ids, k, p0=0):
    """Clean batch-1 no-cache forward: per position the relnorm base (CL11)
    base_j(p) = ||h_K - h_1|| ||h_j|| / (||h_K|| ||h_j - h_1||), and mean ||h_j - h_1|| over positions >= p0
    (pre-norm loop-end states; the w_abs rule of section 5, per loop)."""
    saved = steer.batch_keys
    steer.batch_keys = None
    steer.rec = {"cols": slice(None), "want": {"pre"}, "post": {}, "pre": {}, "kv": {}, "emb": None, "logits": []}
    try:
        fwd(model, ids.view(1, -1), logits_to_keep=1)
        pre = steer.rec["pre"]
    finally:
        steer.rec = None
        steer.batch_keys = saved
    h = {j: pre[j][0].float() for j in range(1, k + 1)}
    nK = (h[k] - h[1]).norm(dim=-1)
    hK = h[k].norm(dim=-1).clamp(min=1e-12)
    base, norms = {}, {}
    for j in range(1, k + 1):
        dj = (h[j] - h[1]).norm(dim=-1)
        if j >= 2:
            base[j] = (nK * h[j].norm(dim=-1) / (hK * dj.clamp(min=1e-6))).cpu().tolist()
        norms[j] = float(dj[p0:].mean()) if dj[p0:].numel() else 0.0
    return base, norms


def tf_resolve(arms, chain_key, base):
    """Concrete edits for one chain: relnorm markers -> per-position coefficient lists ("w_vec" / "ref_w_vec"),
    '@rand<j>' -> a chain-level direction key."""
    out = {}
    for nm, a in arms.items():
        conc = {}
        for part in ("pre", "tok"):
            lst = []
            for e in a[part]:
                e2 = dict(e)
                if "rn" in e2:
                    j, coef, key = e2.pop("rn")
                    e2[key + "_vec"] = [coef * x for x in base[int(j)]]
                if isinstance(e2.get("rkey"), str) and e2["rkey"].startswith("@"):
                    e2["rkey"] = "%s|%s" % (chain_key, e2["rkey"][1:])
                lst.append(e2)
            conc[part] = lst
        conc["lcdl"] = a.get("lcdl")
        out[nm] = conc
    return out


def tf_groups(names, pkey, gsize):
    """Arm groups for the single-token phase: arms sorted by prefill key (arms that share a prefill are adjacent);
    every group starts with 'clean', the group's own reference row (flips and KL compare rows of one batch)."""
    rest = [nm for nm in names if nm != "clean"]
    order = {nm: i for i, nm in enumerate(names)}
    rest.sort(key=lambda nm: (pkey[nm] != pkey["clean"], pkey[nm], order[nm]))
    groups, cur = [], ["clean"]
    for nm in rest:
        if len(cur) >= gsize:
            groups.append(cur)
            cur = ["clean"]
        cur.append(nm)
    if len(cur) > 1 or not groups:
        groups.append(cur)
    return groups


@torch.no_grad()
def tf_chain_v2(ad, steer, chain_key, ids, positions, arms, k, p0=0, budget_gb=20.0, gsize=None):
    """One chain of the CL11-CL13 test. Per arm group: one prefill row per distinct prefill edit set, copied into
    the group's cache rows; then one single-token forward per scored position t in DESCENDING order (the forward
    at t writes cache slot t; smaller t never read it), so P_rest(t) = edited prefill of [0, t) + unedited token.
    Returns (names, {t: row}, extra). Per arm: flip (final argmax vs the group's clean row), own (loop-j exit
    argmax flips, j = 1..K-1), kl (KL(p_clean || p_arm), final read-out), dh (||post_j(arm) - post_j(clean)||)."""
    model = ad.model
    dev = next(model.parameters()).device
    T = int(ids.numel())
    base, norms = tf_clean_states(model, steer, ids.to(dev), k, p0)
    conc = tf_resolve(arms, chain_key, base)
    names = list(arms)
    order = {nm: i for i, nm in enumerate(names)}
    pkey = {nm: json.dumps(conc[nm]["pre"], sort_keys=True) for nm in names}
    if gsize is None:
        row_bytes = float(T + 1) * ad.kv_bytes_per_token(k)
        gsize = int(max(4, min(48, (budget_gb * 1024 ** 3) / row_bytes / 2)))
    groups = tf_groups(names, pkey, gsize)
    batch_ids = ids.view(1, -1).to(dev)
    res = {int(t): {"flip": [None] * len(names), "own": [None] * len(names), "kl": [None] * len(names),
                    "dh": [None] * len(names)} for t in positions}
    clean_final = {}
    used, key_arms = [], {}
    for gi, grp in enumerate(groups):
        uniq = []
        for nm in grp:
            if pkey[nm] not in uniq:
                uniq.append(pkey[nm])
        pv = ["%s|g%d|pv%d" % (chain_key, gi, u) for u in range(len(uniq))]
        for u, pk in enumerate(uniq):
            users = [nm for nm in grp if pkey[nm] == pk]
            steer.rows[pv[u]] = {"plen": T, "span": None, "a_ids": None, "edits": conc[users[0]]["pre"],
                                 "lcdl": None}
            key_arms[pv[u]] = ("pre", [nm for nm in users if conc[nm]["pre"]])
        cache_u = ad._cache_factory(len(uniq), T + 1)
        steer.batch_keys = pv
        try:
            fwd(model, batch_ids.expand(len(uniq), T), cache=cache_u, cache_position=torch.arange(T, device=dev),
                logits_to_keep=1)
        finally:
            steer.batch_keys = None
        sel = torch.tensor([uniq.index(pkey[nm]) for nm in grp], dtype=torch.long, device=dev)
        cache = ad._cache_factory(len(grp), T + 1)
        for e in range(len(cache.key_cache)):
            cache.key_cache[e].copy_(cache_u.key_cache[e].index_select(0, sel))
            cache.value_cache[e].copy_(cache_u.value_cache[e].index_select(0, sel))
        del cache_u
        keys = ["%s|g%d|%s" % (chain_key, gi, nm) for nm in grp]
        for key, nm in zip(keys, grp):
            steer.rows[key] = {"plen": T, "span": None, "a_ids": None, "edits": conc[nm]["tok"],
                               "lcdl": conc[nm]["lcdl"]}
            key_arms[key] = ("tok", [nm])
        used += pv + keys
        Bn = len(grp)
        for t in sorted(positions, reverse=True):
            t = int(t)
            cache.lens = [t] * len(cache.lens)
            attn = torch.ones((Bn, t + 1), dtype=torch.long, device=dev)
            steer.batch_keys = keys
            steer.rec = {"cols": slice(-1, None), "want": {"post"}, "post": {}, "pre": {}, "kv": {}, "emb": None,
                         "logits": []}
            try:
                out = model(input_ids=batch_ids[:, t:t + 1].expand(Bn, 1), attention_mask=attn,
                            position_ids=torch.full((Bn, 1), t, dtype=torch.long, device=dev),
                            past_key_values=cache, use_cache=True, cache_position=torch.tensor([t], device=dev),
                            logits_to_keep=1)
                rec = steer.rec
            finally:
                steer.rec = None
                steer.batch_keys = None
            zf = out.logits[:, -1].float()
            am = zf.argmax(-1)
            lp = torch.log_softmax(zf, -1)
            kl = (lp[0].exp() * (lp[0] - lp)).sum(-1)
            posts = [rec["post"][j][:, -1] for j in range(1, k + 1)]
            if k > 1:
                own = torch.stack([F.linear(p, model.lm_head.weight).float().argmax(-1) for p in posts[:-1]], 1)
            else:
                own = torch.zeros((Bn, 0), dtype=torch.long, device=dev)
            dh = torch.stack([(p.float() - p[:1].float()).norm(dim=-1) for p in posts], 1)
            amL, ownL, klL, dhL = am.tolist(), own.tolist(), kl.tolist(), dh.tolist()
            r = res[t]
            for b, nm in enumerate(grp):
                if nm == "clean" and gi > 0:
                    continue
                ai = order[nm]
                r["flip"][ai] = int(amL[b] != amL[0])
                r["own"][ai] = [int(x != y) for x, y in zip(ownL[b], ownL[0])]
                r["kl"][ai] = round(max(0.0, float(klL[b])), 7)
                r["dh"][ai] = [round(float(x), 4) for x in dhL[b]]
            clean_final.setdefault(t, []).append(int(amL[0]))
            if gi == 0:
                r["clean_final"] = int(amL[0])
                top2 = zf[0].topk(2).values
                r["zmargin"] = round(float(top2[0] - top2[1]), 4)
        del cache
    stats = {}
    for kk in used:
        part, users = key_arms[kk]
        for sk, v in [(sk_, v_) for (k_, sk_), v_ in steer.stats.items() if k_ == kk]:
            for nm in users:
                a = stats.setdefault(nm, {})
                if part == "pre" and "pre" in a:
                    continue                  # one prefill row per arm (groups may repeat a prefill)
                cur = a.setdefault(part, [0.0, 0.0, 0])
                cur[0] += v[0]
                cur[1] += v[1]
                cur[2] += v[2]
    used_set = set(used)
    for kk in [x for x in steer.stats if x[0] in used_set]:
        steer.stats.pop(kk, None)
    for key in used:
        steer.rows.pop(key, None)
        steer.prompt_counted.discard(key)
    mism = sum(1 for v in clean_final.values() if len(set(v)) > 1)
    return names, res, {"norms": norms, "stats": stats, "clean_mismatch_positions": mism, "n_groups": len(groups),
                        "gsize": gsize, "T": T}


def tf_run_v2(ad, steer, seqs, picks, arms, k, rows_path, budget_gb=20.0, log=None):
    """Resumable runner (CL15): one jsonl row per (chain, t), then a chain_done row; a restart skips done chains
    and tf_load ignores rows of an unfinished chain. The header holds the arm names and a hash of the arm table
    (a restart with a different table is refused)."""
    names = list(arms)
    sig = hashlib.sha256(json.dumps({nm: [arms[nm]["pre"], arms[nm]["tok"], arms[nm]["lcdl"]] for nm in names},
                                    sort_keys=True).encode("utf-8")).hexdigest()
    done = set()
    if os.path.exists(rows_path):
        for r in read_jsonl(rows_path):
            if "_header" in r:
                if r["_header"]["arms_sha256"] != sig:
                    raise SystemExit("%s was written with a different arm table; refusing to mix" % rows_path)
            elif "chain_done" in r:
                done.add(int(r["chain_done"]))
    else:
        with open(rows_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_header": {"names": names, "arms_sha256": sig,
                                            "meta": {nm: arms[nm].get("meta") for nm in names}}}) + "\n")
    by = {}
    for idx, t in picks:
        by.setdefault(int(idx), []).append(int(t))
    n_new = 0
    for idx in sorted(by):
        if idx in done:
            continue
        enc, ch = seqs[idx]
        ids = torch.tensor(list(enc) + list(ch), dtype=torch.long)
        t0 = time.time()
        _n, res, extra = tf_chain_v2(ad, steer, str(idx), ids, by[idx], arms, k, p0=max(0, len(enc) - 1),
                                     budget_gb=budget_gb)
        with open(rows_path, "a", encoding="utf-8") as f:
            for t in sorted(res):
                f.write(json.dumps(dict(res[t], idx=idx, t=t)) + "\n")
            f.write(json.dumps(dict(extra, chain_done=idx, seconds=round(time.time() - t0, 2))) + "\n")
        n_new += 1
        if log is not None:
            log("chain %d done: %d tokens, %.1f s, %d groups" % (idx, len(res), time.time() - t0, extra["n_groups"]))
    return names, sig, n_new


def tf_load(rows_path):
    """(names, rows of finished chains, {idx: chain extra}) from a tf_run_v2 file."""
    names, last, extras = None, {}, {}
    for r in read_jsonl(rows_path):
        if "_header" in r:
            names = r["_header"]["names"]
        elif "chain_done" in r:
            extras[int(r["chain_done"])] = r
        else:
            last[(int(r["idx"]), int(r["t"]))] = r     # a re-run chain's rows come after its partial rows
    rows = [r for kk, r in sorted(last.items()) if kk[0] in extras]
    return names, rows, extras


def tf_class(ci):
    """CL10 bands on a [v, lo, hi, ...] difference (dG primary, dFR secondary)."""
    if ci is None:
        return None
    v, lo, hi = ci[0], ci[1], ci[2]
    if v <= 0.10 and hi <= 0.20:
        return "not carried"
    if v >= 0.40 and lo >= 0.25:
        return "carried"
    return "partial"


def boot_slope(ws, ys_by_cluster, ns_by_cluster, n=1000, seed=0):
    """CL13 dose-response: OLS slope over the grid w of y(w) = sum_c ys[c][w] / sum_c ns[c], with a 95% cluster
    bootstrap; returns [slope, lo, hi, n_tokens, n_clusters]."""
    keys = sorted(ys_by_cluster, key=lambda z: str(z))
    if not keys:
        return None
    Y = torch.tensor([ys_by_cluster[c] for c in keys], dtype=torch.float64)
    N = torch.tensor([ns_by_cluster[c] for c in keys], dtype=torch.float64)
    w = torch.tensor([float(x) for x in ws], dtype=torch.float64)
    wc = w - w.mean()
    den = float((wc * wc).sum())

    def slope_of(idx):
        y = Y[idx].sum(-2) / N[idx].sum(-1, keepdim=True).clamp(min=1e-12)
        return ((y - y.mean(-1, keepdim=True)) * wc).sum(-1) / den
    v0 = float(slope_of(torch.arange(len(keys)).view(1, -1))[0])
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=g)
    bs = slope_of(idx)
    q = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [round(v0, 5), round(float(q[0]), 5), round(float(q[1]), 5), int(N.sum()), len(keys)]


def ct_labels(summ, loops, w_star, pc2_carried):
    """EXPERIMENT_MATRIX X9 'pass' row, family CT (frozen labels), per (loop, matching) at w*:
    not carried(set) = FR <= 0.25 and CF <= 0.25 with upper bounds <= 0.35; carried(set) = FR >= 0.75 with
    lower bound >= 0.5. 'consistent with H2' = rest carried and last not carried; 'consistent with contraction'
    = last, rest and all not carried while PC2 is carried (CL2: PC2 flips >= 2x its matched random); else
    'unresolved'. FR and CF are the forward (aligned, + sign) contrast's ratios."""
    bw = (summ.get("by_w") or {}).get("%g" % float(w_star)) or {}
    cells = bw.get("cells") or {}
    out = {"w": float(w_star), "pc2_carried": pc2_carried, "fr_readable": bw.get("fr_readable"),
           "den_events": bw.get("lcdh_events"), "labels": {}}

    def notc(c):
        fr, cf = c.get("FR"), c.get("CF")
        return bool(fr is not None and cf is not None and fr[0] <= 0.25 and cf[0] <= 0.25 and fr[2] <= 0.35
                    and cf[2] <= 0.35)

    def carried(c):
        fr = c.get("FR")
        return bool(fr is not None and fr[0] >= 0.75 and fr[1] >= 0.5)
    for j in loops:
        for m in TF_MATCH:
            cs = {st: cells.get("L%d_%s_%s" % (j, st, m)) for st in TF_SETS}
            if any(v is None for v in cs.values()):
                continue
            st_ = {st: ("carried" if carried(c) else "not carried" if notc(c) else "neither") for st, c in cs.items()}
            if st_["rest"] == "carried" and st_["last"] == "not carried":
                lab = "consistent with H2"
            elif all(v == "not carried" for v in st_.values()) and pc2_carried:
                lab = "consistent with contraction"
            else:
                lab = "unresolved"
            out["labels"]["L%d_%s" % (j, m)] = {"label": lab, "by_set": st_,
                                                 "FR": {st: c.get("FR") for st, c in cs.items()},
                                                 "CF": {st: c.get("CF") for st, c in cs.items()}}
    return out


def tf_summary_v2(names, arms_meta, rows, extras, w_star, grid=TF_GRID, min_events=100):
    """CL10-CL13 read-out: per w the FR, G (primary), dFR, dG, CF and classes per (loop, set, matching), the agreed
    class (abs and relnorm agree), back-vs-forward at w*, dose slopes, strengths (P1), PC2, tiny and LCD-L."""
    ix = {nm: i for i, nm in enumerate(names)}
    cl = [int(r["idx"]) for r in rows]
    flip = {nm: [r["flip"][i] for r in rows] for nm, i in ix.items()}
    kl = {nm: [r["kl"][i] for r in rows] for nm, i in ix.items()}

    def own(nm, j):
        i = ix[nm]
        return [r["own"][i][j - 1] if j <= len(r["own"][i]) else r["flip"][i] for r in rows]
    out = {"n_tokens": len(rows), "n_chains": len(set(cl)), "flip": {}, "kl": {}, "by_w": {}, "back": {}, "dose": {},
           "strength": {}, "min_events": min_events,
           "clean_mismatch_positions": sum(int(e.get("clean_mismatch_positions", 0)) for e in extras.values())}
    for nm in names:
        if nm == "clean":
            continue
        out["flip"][nm] = boot_cluster(flip[nm], cl)
        out["kl"][nm] = boot_cluster(kl[nm], cl)
        agg = {}
        for e in extras.values():
            for part, v in ((e.get("stats") or {}).get(nm) or {}).items():
                a = agg.setdefault(part, [0.0, 0.0, 0])
                a[0] += v[0]
                a[1] += v[1]
                a[2] += v[2]
        if agg:
            out["strength"][nm] = {p: {"mean_norm_delta": round(v[0] / max(1, v[2]), 5),
                                       "mean_rel_delta": round(v[1] / max(1, v[2]), 6), "n": v[2]}
                                   for p, v in agg.items()}
    loops = sorted({m["j"] for m in arms_meta.values() if m and m.get("kind") == "arm"})
    for w in grid:
        ref = "lcdh@%g" % w
        if ref not in ix:
            continue
        ev = int(sum(flip[ref]))
        bw = {"lcdh_events": ev, "lcdh_flip": out["flip"][ref], "randK_flip": out["flip"].get("randK@%g" % w),
              "lcdh_kl": out["kl"][ref], "fr_readable": ev >= min_events, "cells": {}, "agreed": {}}
        for j in loops:
            for st in TF_SETS:
                cls_by_m = {}
                for m in TF_MATCH:
                    f_, r_ = tf_arm_name(j, st, "fwd", m, w), tf_arm_name(j, st, "rand", m, w)
                    if f_ not in ix or r_ not in ix:
                        continue
                    c = {"FR": boot_ratio(flip[f_], flip[ref], cl), "G": boot_ratio(kl[f_], kl[ref], cl),
                         "dFR": boot_diff(flip[f_], flip[r_], flip[ref], cl),
                         "dG": boot_diff(kl[f_], kl[r_], kl[ref], cl),
                         "CF": boot_ratio(flip[f_], own(f_, j), cl), "own_flip": boot_cluster(own(f_, j), cl),
                         "den_events": ev, "n_tokens": len(rows), "n_chains": len(set(cl))}
                    c["class_G"] = tf_class(c["dG"])
                    c["class_FR"] = tf_class(c["dFR"]) if ev >= min_events else "FR underpowered"
                    for r2 in ("ral", "rort"):
                        q_ = tf_arm_name(j, st, r2, m, w)
                        if q_ in ix:
                            c["dFR_" + r2] = boot_diff(flip[f_], flip[q_], flip[ref], cl)
                            c["dG_" + r2] = boot_diff(kl[f_], kl[q_], kl[ref], cl)
                            c["class_G_" + r2] = tf_class(c["dG_" + r2])
                            c["class_FR_" + r2] = (tf_class(c["dFR_" + r2]) if ev >= min_events
                                                   else "FR underpowered")
                    bw["cells"]["L%d_%s_%s" % (j, st, m)] = c
                    cls_by_m[m] = c["class_G"]
                if "abs" in cls_by_m and "rn" in cls_by_m:
                    bw["agreed"]["L%d_%s" % (j, st)] = (cls_by_m["abs"] if cls_by_m["abs"] == cls_by_m["rn"]
                                                       else "matching-dependent")
        if "tiny" in ix:
            # Q8 / RA-1: G is read only where LCD-H's median final KL is >= 10 x the tiny push's
            med_l, med_t = float(np.median(kl[ref])), float(np.median(kl["tiny"]))
            bw["median_kl_lcdh"] = round(med_l, 8)
            bw["median_kl_tiny"] = round(med_t, 8)
            bw["g_readable"] = bool(med_l >= 10.0 * med_t)
        out["by_w"]["%g" % w] = bw
    if "pc2" in ix and "pc2_rand" in ix:
        pc2_ok = bool(sum(flip["pc2"]) > 0 and sum(flip["pc2"]) >= 2.0 * sum(flip["pc2_rand"]))
    else:
        pc2_ok = None
    out["ct_labels"] = ct_labels(out, loops, w_star, pc2_ok)
    ref_s = "lcdh@%g" % w_star
    for j in loops:
        for st in TF_SETS:
            for m in TF_MATCH:
                f_, b_ = tf_arm_name(j, st, "fwd", m, w_star), tf_arm_name(j, st, "back", m, w_star)
                if f_ in ix and b_ in ix and ref_s in ix:
                    out["back"]["L%d_%s_%s" % (j, st, m)] = {
                        "G_back": boot_ratio(kl[b_], kl[ref_s], cl), "FR_back": boot_ratio(flip[b_], flip[ref_s], cl),
                        "G_fwd_minus_back": boot_diff(kl[f_], kl[b_], kl[ref_s], cl),
                        "FR_fwd_minus_back": boot_diff(flip[f_], flip[b_], flip[ref_s], cl)}
                ws = [w for w in grid if tf_arm_name(j, st, "fwd", m, w) in ix and tf_arm_name(j, st, "rand", m, w) in ix]
                if len(ws) >= 3:
                    yf, yk, nn = {}, {}, {}
                    pairs = [(ix[tf_arm_name(j, st, "fwd", m, w)], ix[tf_arm_name(j, st, "rand", m, w)]) for w in ws]
                    for ri, r in enumerate(rows):
                        c = cl[ri]
                        a = yf.setdefault(c, [0.0] * len(ws))
                        b = yk.setdefault(c, [0.0] * len(ws))
                        nn[c] = nn.get(c, 0) + 1
                        for wi, (fi, qi) in enumerate(pairs):
                            a[wi] += r["flip"][fi] - r["flip"][qi]
                            b[wi] += r["kl"][fi] - r["kl"][qi]
                    out["dose"]["L%d_%s_%s" % (j, st, m)] = {"w": list(ws),
                                                             "slope_flip_minus_rand": boot_slope(ws, yf, nn),
                                                             "slope_kl_minus_rand": boot_slope(ws, yk, nn)}
    for nm in ("pc2", "pc2_rand", "tiny", "lcdl_ad"):
        if nm in ix:
            out[nm] = {"flip": out["flip"][nm], "kl": out["kl"][nm], "events": int(sum(flip[nm]))}
    if "pc2" in ix and "pc2_rand" in ix:
        out["pc2"]["flip_ratio_vs_rand"] = round(sum(flip["pc2"]) / float(max(1, sum(flip["pc2_rand"]))), 4)
    return out


def tf_plan_from_tuning(tun, n_eval_chains, per_chain=20, base_chains=200, min_events=100, floor_tokens=2000):
    """CL10 sizing: tokens needed for >= min_events LCD-H final flips at w* on the eval set, from the filtered
    train-200 event rate (1.25 margin)."""
    ev = int(tun["events"]["lcdh@%g" % float(tun["w_star"])])
    nt = int(tun["n_tokens"])
    r = ev / float(max(1, nt))
    need = int(math.ceil(1.25 * min_events / r)) if r > 0 else None
    n_chains, pc = int(base_chains), int(per_chain)
    if need is None or need > base_chains * per_chain:
        n_chains = int(n_eval_chains)
        pc = min(60, int(math.ceil((need or 10 ** 9) / float(max(1, n_eval_chains)))))
    n_tokens = max(int(floor_tokens), need or 0)
    return {"train_rate_lcdh_w_star": round(r, 5), "train_events": ev, "train_tokens": nt, "n_needed": need,
            "n_chains": n_chains, "per_chain": pc, "n_tokens": n_tokens, "min_events": min_events,
            "fr_descriptive_only_expected": bool(need is None or need > n_chains * pc)}


@torch.no_grad()
def tf_equiv_v2(ad, steer, seqs, picks, arms, k, n_chains=3, n_tok=5, budget_gb=20.0):
    """Gate G3b for the v2 path (CL1, CL13): per arm, the cached descending result (prefill copy, group clean
    row) equals a full no-cache forward with the same edits written over explicit positions: P_last -> [t, t+1),
    P_rest -> [0, t), P_all -> every position. Compared: the final flip bit vs clean and KL (|dKL| <= 0.05 or
    5%). Pass: flip agreement >= n - 1 below 1000 comparisons, else >= 0.999."""
    by = {}
    for idx, t in picks:
        by.setdefault(int(idx), []).append(int(t))
    dev = next(ad.model.parameters()).device
    agree, n, kl_bad, worst = 0, 0, 0, 0.0
    for idx in sorted(by)[:n_chains]:
        enc, ch = seqs[idx]
        ids = torch.tensor(list(enc) + list(ch), dtype=torch.long)
        pos = sorted(by[idx])[-n_tok:]
        names, res, _x = tf_chain_v2(ad, steer, "eqv2|%d" % idx, ids, pos, arms, k, p0=max(0, len(enc) - 1),
                                     budget_gb=budget_gb)
        base, _norms = tf_clean_states(ad.model, steer, ids.to(dev), k)
        conc = tf_resolve(arms, "eqv2|%d" % idx, base)

        def full(edits, lcdl=None):
            steer.rows["eqfull"] = {"plen": None, "span": None, "a_ids": None, "edits": edits, "lcdl": lcdl}
            steer.batch_keys = ["eqfull"]
            try:
                return fwd(ad.model, ids.view(1, -1), logits_to_keep=0).logits[0].float()
            finally:
                steer.batch_keys = None
                steer.rows.pop("eqfull", None)
        zc = full([])
        for nm in names:
            if nm == "clean":
                continue
            m = arms[nm].get("meta") or {}
            for t in pos:
                if m.get("set") == "rest":
                    if t == 0:
                        continue
                    eds = [dict(e, pos=[0, t]) for e in conc[nm]["pre"]]
                elif m.get("set") == "all" or (conc[nm]["pre"] and not m.get("set")):
                    eds = [dict(e, pos=[0, None]) for e in conc[nm]["pre"]]
                else:
                    eds = [dict(e, pos=[t, t + 1]) for e in conc[nm]["tok"]]
                if not eds and conc[nm]["lcdl"] is None:
                    continue
                z = full(eds, conc[nm]["lcdl"])
                fl = int(int(z[t].argmax()) != int(zc[t].argmax()))
                lp0, lp1 = torch.log_softmax(zc[t], -1), torch.log_softmax(z[t], -1)
                klf = float((lp0.exp() * (lp0 - lp1)).sum())
                kc = res[t]["kl"][names.index(nm)]
                n += 1
                agree += int(fl == res[t]["flip"][names.index(nm)])
                gap = abs(klf - kc)
                worst = max(worst, gap)
                kl_bad += int(gap > max(0.05, 0.05 * abs(klf)))
    ok = n > 0 and (agree >= 0.999 * n if n >= 1000 else agree >= n - 1)
    return {"flip_agree": agree, "n": n, "kl_out_of_tol": kl_bad, "max_abs_dkl": round(worst, 5),
            "pass": bool(ok and kl_bad <= max(1, n // 100))}


@torch.no_grad()
def tf_pc2_k3(ad, steer, seqs, idxs, k):
    """RA-1 reset control on the test chains: LCD-H2 at w = -1 (every position, k = K) against the clean K-1 run,
    full no-cache forwards over every chain position. Pass = the G4 rule (argmax agreement >= 0.999, max |dz|
    <= 0.1)."""
    dev = next(ad.model.parameters()).device
    agree, n, worst = 0, 0, 0.0
    for idx in idxs:
        enc, ch = seqs[idx]
        ids = torch.tensor([list(enc) + list(ch)], dtype=torch.long, device=dev)
        steer.rows["pc2k3"] = {"plen": None, "span": None, "a_ids": None, "lcdl": None,
                               "edits": [{"op": "lcdh", "site": "pre", "loop": 2, "pos": "all", "w": -1.0}]}
        steer.batch_keys = ["pc2k3"]
        try:
            z = fwd(ad.model, ids, logits_to_keep=0).logits[0].float()
        finally:
            steer.batch_keys = None
            steer.rows.pop("pc2k3", None)
        ad.set_depth(k - 1)
        try:
            z3 = fwd(ad.model, ids, logits_to_keep=0).logits[0].float()
        finally:
            ad.set_depth(k)
        agree += int((z.argmax(-1) == z3.argmax(-1)).sum())
        n += int(z.shape[0])
        worst = max(worst, float((z - z3).abs().max()))
    fr = agree / float(max(1, n))
    return {"chains": list(idxs), "n_positions": n, "argmax_agree": round(fr, 6), "max_abs_dz": round(worst, 4),
            "pass": bool(n > 0 and fr >= 0.999 and worst <= 0.1)}


def _picks_cached(path, make):
    """Token picks are drawn once and reused on restart (CL15)."""
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        return [tuple(x) for x in d["picks"]], d["selection"]
    picks, sel = make()
    write_json(path, {"picks": [list(x) for x in picks], "selection": sel})
    return picks, sel


def tune_main(cfg, a, ad, steer, items, chains):
    """Section 9 + CL10: reach matching on the CL6-filtered train-200 tokens (LCD-H grid vs LCD-L adaptive), event
    counts per w, w_abs per loop (CL11/CL12), train floors at w* with the full eval arm set at w*, and the eval
    teacher-forced sizing (tf_plan.json). Resumable."""
    k = cfg["k"]
    task = "gsm8k_train"
    register_task(task, "gsm8k", os.environ.get("K17_TRAIN_ROWS"))
    ids = items["train"][:a.n] if a.n else items["train"]
    seqs, bad = chain_seqs(ad.tok, cfg, chains, ids, task)
    ntok = int(a.tf_tokens or cfg.get("tf_tokens", 2000))
    pc = int(a.per_chain or cfg.get("tf_per_chain", 20))
    picks, selinfo = _picks_cached(os.path.join(a.out, "tune_picks.json"),
                                   lambda: tf_select(ad, steer, seqs, k, ntok, pc, float(cfg.get("tau", 1.0))))
    t0 = time.time()
    grid_arms = tf_grid_arms(k, cfg["wmax"])
    rp = os.path.join(a.out, "tune_rows.jsonl")
    tf_run_v2(ad, steer, seqs, picks, grid_arms, k, rp, budget_gb=a.budget_gb)
    names, rows, extras = tf_load(rp)
    ix = {nm: i for i, nm in enumerate(names)}
    ev = {nm: int(sum(r["flip"][ix[nm]] for r in rows)) for nm in names if nm != "clean"}
    nt = len(rows)
    rate = {nm: ev[nm] / float(max(1, nt)) for nm in ev}
    target = rate["lcdl_ad"]
    grid = list(TF_GRID)
    w_flip = min(grid, key=lambda w: (abs(rate["lcdh@%g" % w] - target), w))
    # RA-1 decision (from plan Q8): fewer than 20 LCD-L adaptive flip events -> match on the median final KL
    # (ties to the smaller w). Both rules are recorded; the rule is fixed here, before any test row.
    med = {nm: (float(np.median([r["kl"][ix[nm]] for r in rows])) if rows else 0.0) for nm in names if nm != "clean"}
    w_kl = min(grid, key=lambda w: (abs(med["lcdh@%g" % w] - med["lcdl_ad"]), w))
    rule = "flip" if ev["lcdl_ad"] >= 20 else "median_kl"
    w_star = w_flip if rule == "flip" else w_kl
    nrm = {}
    for e in extras.values():
        for j, v in e["norms"].items():
            nrm.setdefault(int(j), []).append(float(v))
    mean = {j: sum(v) / len(v) for j, v in nrm.items()}
    ratio = {j: mean[k] / max(1e-12, mean[j]) for j in mean if 2 <= j < k}
    tun = {"grid": grid, "events": ev, "flip_rate": {kk: round(v, 5) for kk, v in rate.items()}, "n_tokens": nt,
           "n_chains": len(extras), "target_lcdl_ad": round(target, 5), "w_star": w_star,
           "reach_rule": rule, "w_star_flip_rule": w_flip, "w_star_kl_rule": w_kl,
           "median_kl": {kk: round(v, 8) for kk, v in med.items()},
           "reach_rule_text": "flip matching if LCD-L adaptive has >= 20 flip events on the filtered tune tokens, "
                              "else median final KL matching (RA-1)",
           "abs_ratio": {str(j): round(v, 5) for j, v in ratio.items()},
           "w_abs": round(w_star * ratio[2], 5) if 2 in ratio else None,
           "w_abs3": round(w_star * ratio[3], 5) if 3 in ratio else None,
           "mean_norm_hj_minus_h1": {str(j): round(v, 5) for j, v in mean.items()},
           "wmax": cfg["wmax"], "w_fixed": cfg.get("w_fixed"), "selection": selinfo, "prompt_mismatch": bad,
           "population": "CL6-filtered train-200 tokens (CL10)",
           "chains_from": os.path.basename(os.path.normpath(a.chains_from))}
    fl_arms = tf_arms_v2(k, w_star, ratio, cfg["wmax"], grid=(w_star,), loops=tuple(cfg.get("tf_loops", (2, 3))))
    fp = os.path.join(a.out, "tune_floor_rows.jsonl")
    tf_run_v2(ad, steer, seqs, picks, fl_arms, k, fp, budget_gb=a.budget_gb)
    fn, frows, fext = tf_load(fp)
    tun["floors"] = tf_summary_v2(fn, {nm: fl_arms[nm]["meta"] for nm in fn}, frows, fext, w_star, grid=(w_star,))
    tun["seconds"] = round(time.time() - t0, 1)
    tf_on = cfg.get("tf_on", "eval")
    if tf_on in items:
        tun["tf_plan"] = dict(tf_plan_from_tuning(tun, n_eval_chains=len(items[tf_on])), tf_on=tf_on)
        if not os.path.exists(os.path.join(a.out, "tf_plan.json")):
            write_json(os.path.join(a.out, "tf_plan.json"), tun["tf_plan"])
    write_json(os.path.join(a.out, a.tuning or "tuning.json"), tun)
    return {kk: tun.get(kk) for kk in ("w_star", "w_abs", "w_abs3", "events", "n_tokens", "selection", "tf_plan")}


def tf_main(cfg, a, ad, steer, items, chains):
    """CL10-CL13 eval teacher-forced test: frozen tuning.json and tf_plan.json, picks drawn once, gate G3b on 3
    chains before any scored row, then the full arm table over every grid w. Resumable."""
    k = cfg["k"]
    tp = a.tuning or os.path.join(a.out, "tuning.json")
    if not os.path.exists(tp):
        raise SystemExit("tf needs the frozen tuning.json")
    tun = json.load(open(tp, encoding="utf-8"))
    plan_p = os.path.join(a.out, "tf_plan.json")
    plan = json.load(open(plan_p, encoding="utf-8")) if os.path.exists(plan_p) else \
        tf_plan_from_tuning(tun, n_eval_chains=len(items["eval"]))
    n_chains = int(a.n or plan["n_chains"])
    ntok = int(a.tf_tokens or plan["n_tokens"])
    pc = int(a.per_chain or plan["per_chain"])
    tf_on = cfg.get("tf_on", "eval")
    exclude = []
    if tf_on == "train":
        # RA-1: the test runs on the tune chains (no eval generation row); tokens disjoint from the tune draw
        task = "gsm8k_train"
        register_task(task, "gsm8k", os.environ.get("K17_TRAIN_ROWS"))
        tpk = os.path.join(a.out, "tune_picks.json")
        if not os.path.exists(tpk):
            raise SystemExit("tf on the tune chains needs tune_picks.json (run tune first)")
        exclude = [tuple(x) for x in json.load(open(tpk, encoding="utf-8"))["picks"]]
    else:
        task = cfg["task"]
    seqs, bad = chain_seqs(ad.tok, cfg, chains, items[tf_on][:n_chains], task)
    picks, selinfo = _picks_cached(os.path.join(a.out, "tf_picks.json"),
                                   lambda: tf_select(ad, steer, seqs, k, ntok, pc, float(cfg.get("tau", 1.0)),
                                                     exclude=exclude))
    if set((int(i), int(t)) for i, t in picks) & set((int(i), int(t)) for i, t in exclude):
        raise SystemExit("test picks overlap the tune picks")
    ratio = {int(j): float(v) for j, v in tun["abs_ratio"].items()}
    w_star = float(tun["w_star"])
    loops = tuple(cfg.get("tf_loops", (2, 3)))
    arms = tf_arms_v2(k, w_star, ratio, cfg["wmax"], loops=loops)
    eqp = os.path.join(a.out, "tf_gate_g3b.json")
    if not os.path.exists(eqp):
        sub = {nm: arms[nm] for nm in ["clean", tf_arm_name(2, "all", "fwd", "rn", w_star),
                                       tf_arm_name(3, "rest", "fwd", "abs", w_star),
                                       tf_arm_name(2, "last", "rand", "rel", w_star),
                                       tf_arm_name(3, "all", "back", "rn", w_star),
                                       tf_arm_name(2, "rest", "ral", "abs", w_star),
                                       tf_arm_name(2, "all", "rort", "rn", w_star),
                                       "lcdh@%g" % w_star, "lcdl_ad", "pc2"] if nm in arms}
        eq = tf_equiv_v2(ad, steer, seqs, picks, sub, k, budget_gb=a.budget_gb)
        eq["pc2_k3"] = tf_pc2_k3(ad, steer, seqs, sorted({int(i) for i, _t in picks})[:3], k)
        eq["pass_g3b"] = eq["pass"]
        eq["pass"] = bool(eq["pass"] and eq["pc2_k3"]["pass"])
        write_json(eqp, eq)
        if not eq["pass"]:
            raise SystemExit("gate G3b / PC2-k3 failed: %s" % eq)
    t0 = time.time()
    rp = os.path.join(a.out, "tf_eval_rows.jsonl")
    tf_run_v2(ad, steer, seqs, picks, arms, k, rp, budget_gb=a.budget_gb, log=lambda msg: print(msg, flush=True))
    names, rows, extras = tf_load(rp)
    summ = tf_summary_v2(names, {nm: arms[nm]["meta"] for nm in names}, rows, extras, w_star,
                         min_events=int(plan.get("min_events", 100)))
    summ.update({"selection": selinfo, "gate_g3b": json.load(open(eqp, encoding="utf-8")), "prompt_mismatch": bad,
                 "tf_plan": plan, "w_star": w_star, "abs_ratio": ratio,
                 "seconds_this_process": round(time.time() - t0, 1),
                 "seconds_per_chain": [e.get("seconds") for e in extras.values()][:10]})
    write_json(os.path.join(a.out, "tf_eval.json"), summ)
    return {"n_tokens": summ["n_tokens"], "n_chains": summ["n_chains"],
            "events_w_star": summ["by_w"].get("%g" % w_star, {}).get("lcdh_events"), "gate_g3b": summ["gate_g3b"]}


# ====================================================================== CLI
def make_adapter(name, k, kv=True, think_dir=None, mem_fraction=None):
    kw = {}
    if mem_fraction:
        kw["mem_fraction"] = float(mem_fraction)
    steer = SteerHooks(k)
    ad = SteeredOuroAdapter(name, steer=steer, kv=kv, think_dir=think_dir, **kw)
    ad.load()
    ad.set_depth(k)
    return ad, steer


def check_items(tok, cfg, n=4, pool_path=POOL_DEFAULT):
    """The first n pool items (shuffle order) with valid geometry, for the gates (no correctness filter)."""
    from .tasks import rows as task_rows
    srows = {int(r["idx"]): r for r in task_rows(TASK_A)}
    pool = {int(r["idx"]): r for r in core_pool(pool_path)[0]}
    out = []
    for idx in shuffled(pool):
        g = item_geometry(tok, pool[idx], cfg["chat"], srows)
        if all(g["ok"].values()):
            out.append({"idx": idx, "a_ids": g["idsA"], "b_ids": g["idsB"], "span": g["span"], "plen": g["plen"]})
        if len(out) >= n:
            break
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="prod.steer_gen")
    p.add_argument("mode", choices=["checks", "prep", "screen", "select", "run", "tune", "tf", "anschain",
                                    "presence", "trainrows", "analyze"])
    p.add_argument("--setting", default=None)
    p.add_argument("--row", default=None)
    p.add_argument("--which", default="eval")
    p.add_argument("--out", required=True)
    p.add_argument("--settle-dir", default=None)
    p.add_argument("--think-dir", default=os.environ.get("K17_THINK_DIR"))
    p.add_argument("--mem-fraction", type=float, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--g3-items", dest="g3_items", type=int, default=3,
                   help="checks: items of gate G3 (cache equivalence; 32 steps each); the cluster runs 12")
    p.add_argument("--g3-measured-rate", dest="g3_measured_rate", action="store_true",
                   help="checks: Deviation D5m, G3 passes on the margin rule alone and records the measured agreement")
    p.add_argument("--tuning", default=None)
    p.add_argument("--config-json", default=None)
    p.add_argument("--chains-from", default=None)
    p.add_argument("--parquet", default=None)
    p.add_argument("--budget-gb", type=float, default=20.0, help="teacher-forced KV budget per arm group")
    p.add_argument("--tf-tokens", type=int, default=None, help="debug: override the token count")
    p.add_argument("--per-chain", type=int, default=None, help="debug: override the per-chain cap")
    p.add_argument("--rows", default=None, help="analyze: comma list of rows")
    a = p.parse_args(argv)
    override = json.load(open(a.config_json, encoding="utf-8")) if a.config_json else None
    cfg = setting_cfg(a.setting, override) if a.setting else None
    if a.mode == "trainrows":
        n = gsm8k_train_rows(a.parquet, a.out)
        print("wrote %d train rows to %s" % (n, a.out))
        return 0
    os.makedirs(a.out, exist_ok=True)
    if cfg is not None:
        write_json(os.path.join(a.out, "setting.json"), cfg)
    if a.mode == "prep":
        if cfg["arm"] == "a":
            r = prep_a(cfg, a.out, a.settle_dir, a.think_dir)
            print(json.dumps({kk: v for kk, v in r.items() if kk != "items"}, default=str))
        else:
            r = prep_b(cfg, a.out, os.environ.get("K17_TRAIN_ROWS"))
            print(json.dumps({kk: v for kk, v in r.items() if kk in ("sentinel_layout", "pert_layout_change")}))
        return 0
    if a.mode == "screen":
        if a.n is not None:
            cfg["n_screen"] = int(a.n)
        codes = []
        for side in (("B",) if cfg.get("screen_A_from_settle") else ("A", "B")):
            summ = job_a(cfg, a.out, None, a.think_dir, a.mem_fraction, screen=side)
            print(json.dumps(summ))
            codes.append(job_exit(summ))
        return next((c for c in codes if c), 0)
    if a.mode == "select":
        r = select_a(cfg, a.out, settle_dir=a.settle_dir)
        print(json.dumps({kk: v for kk, v in r.items() if kk not in ("scored",)}, default=str))
        return 0
    if a.mode == "run":
        if cfg["arm"] == "a" and a.row == "bw1_r00":
            # CL16 bw-check: r00 on the first 20 scored items at batch width 1 (compared with r00 at 16)
            summ = job_a(cfg, a.out, "r00", a.think_dir, a.mem_fraction, batch_width=1, n_limit=a.n or 20,
                         job_name="bw1_r00")
        elif cfg["arm"] == "a":
            summ = job_a(cfg, a.out, a.row, a.think_dir, a.mem_fraction)
        else:
            tun = json.load(open(a.tuning, encoding="utf-8")) if a.tuning else {}
            tun.setdefault("wmax", cfg["wmax"])
            tun.setdefault("w_fixed", cfg.get("w_fixed"))
            summ = job_b(cfg, a.out, a.row, tun, a.think_dir, a.mem_fraction, which=a.which)
        print(json.dumps(summ))
        return job_exit(summ)
    if a.mode == "checks":
        ad, steer = make_adapter(cfg["model"], cfg["k"], kv=True, think_dir=a.think_dir, mem_fraction=a.mem_fraction)
        its = check_items(ad.tok, cfg, n=a.n or 4)
        t0 = time.time()
        res = check_suite(ad, steer, its, cfg["k"], out_path=None, g3_items=a.g3_items,
                          g3_measured_rate=a.g3_measured_rate)
        res["items"] = [x["idx"] for x in its]
        res["seconds"] = round(time.time() - t0, 1)
        res["env"] = env_block(ad)
        write_json(os.path.join(a.out, "checks_%s.json" % cfg["name"]), res)
        print(json.dumps({kk: v for kk, v in res.items() if kk != "env"}, default=str))
        return 0
    if a.mode in ("tune", "tf"):
        ad, steer = make_adapter(cfg["model"], cfg["k"], kv=False, think_dir=a.think_dir, mem_fraction=a.mem_fraction)
        items = json.load(open(os.path.join(a.out, "items.json"), encoding="utf-8"))
        chains = load_chains(a.chains_from)
        r = tune_main(cfg, a, ad, steer, items, chains) if a.mode == "tune" else tf_main(cfg, a, ad, steer, items,
                                                                                          chains)
        print(json.dumps(r, default=str)[:4000])
        return 0
    if a.mode in ("anschain", "presence") and DATASET != "gsm8k":
        raise SystemExit("%s is built for the GSM8K pairs only (K17_DATASET=%s)" % (a.mode, DATASET))
    if a.mode in ("anschain", "presence"):
        ad, steer = make_adapter(cfg["model"], cfg["k"], kv=True, think_dir=a.think_dir, mem_fraction=a.mem_fraction)
        sel = json.load(open(os.path.join(a.out, "items.json"), encoding="utf-8"))
        pool = {int(r["idx"]): r for r in core_pool()[0]}
        items = [dict(it, question_B=pool[int(it["idx"])]["question_B"]) for it in sel["scored"]][:a.n or None]
        t0 = time.time()
        k = cfg["k"]
        if a.mode == "anschain":
            cA = load_chains(os.path.join(a.out, "r00"))
            cB = load_chains(os.path.join(a.out, "r01"))
            gp = os.path.join(a.out, "anschain_gates.json")
            if not os.path.exists(gp):
                gits = anschain_gate_items(ad.tok, cfg, items, cA, cB, n=3)
                gates = check_suite(ad, steer, gits, k, n_steps=12) if gits else {}
                gates["items"] = [g["idx"] for g in gits]
                write_json(gp, gates)
            gates = json.load(open(gp, encoding="utf-8"))
            if not all((gates.get(g_) or {}).get("pass") for g_ in ("gate_zero_edit", "gate_placement",
                                                                    "gate_cache_equiv")):
                print("[warn] ans-chain decoder gates failed; Llast@1 cells will read 'untested'", flush=True)
            rp = os.path.join(a.out, "anschain_rows.jsonl")
            _recs, skipped = anschain_run(ad, steer, cfg, items, cA, cB, rows_path=rp)
            recs = read_jsonl(rp) if os.path.exists(rp) else []
            summ = {"summary": anschain_summary(recs, gates), "skipped_this_process": skipped,
                    "seconds_this_process": round(time.time() - t0, 1),
                    "gates": {g_: (gates.get(g_) or {}).get("pass") for g_ in ("gate_zero_edit", "gate_placement",
                                                                             "gate_cache_equiv", "gate_identities")}}
            write_json(os.path.join(a.out, "anschain.json"), summ)
        else:
            rp = os.path.join(a.out, "presence_rows.jsonl")
            presence_run(ad, steer, cfg, items, rows_path=rp)
            recs = read_jsonl(rp) if os.path.exists(rp) else []
            ids = [r["idx"] for r in recs]
            summ = {"n_items": len(recs), "seconds_this_process": round(time.time() - t0, 1),
                    "first_token_flip_vs_A": {nm: boot_cluster([int(r["first_token"][nm] != r["first_token"]["A"])
                                                                for r in recs], ids) for nm in ("B", "r06", "r07")},
                    "s_by_loop": {nm: {str(j): boot_cluster([r["s"][str(j)][nm] for r in recs], ids)
                                       for j in range(1, k + 1)} for nm in ("A", "B", "r06", "r07")},
                    "dv_norm_median": {str(j): float(np.median([r["dv_norm"][str(j)] for r in recs if "dv_norm" in r]))
                                       for j in range(1, k + 1)} if any("dv_norm" in r for r in recs) else None,
                    "dv_rel_median": {str(j): float(np.median([r["dv_rel"][str(j)] for r in recs if "dv_rel" in r]))
                                      for j in range(1, k + 1)} if any("dv_rel" in r for r in recs) else None}
            write_json(os.path.join(a.out, "presence.json"), summ)
        print(json.dumps(summ, default=str)[:4000])
        return 0
    if a.mode == "analyze":
        if cfg["arm"] == "a":
            tok = load_tokenizer(cfg["model"], a.think_dir)
            r = analyze_a(cfg, a.out, tok=tok)
            name = "analysis.json"
        else:
            r = analyze_b(cfg, a.out, which=a.which, settle_dir=a.settle_dir)
            name = "analysis_%s.json" % a.which
        write_json(os.path.join(a.out, name), r)
        print(json.dumps(r, default=str)[:6000])
        return 0
    raise SystemExit("unknown mode %s" % a.mode)


if __name__ == "__main__":
    sys.exit(main())
