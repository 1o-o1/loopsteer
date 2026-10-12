"""Raven binding: hooks on Huginn-0125 and McLeish Recurrent-Llama-3.2 (DESIGN 3.2, 3.3, 6.2, 8.2).

What this module measures and why. A raven model runs a prelude, k steps of a core block whose
input is `adapter(cat[x, e])` (the state and the injected prelude output e), then a coda
[read huginn_raven.py 636-723, 762-779; McLeish 624-768]. To edit or read any (stage, loop,
layer, component) tensor, the SandwichBlock forward is transcribed per family (Huginn 525-535,
McLeish 458-469) as an instance `forward` with one `reg.site` call per component, guarded by the
normalized source hash of the original. Module hooks cover the rest: the top-level pre/post
hooks (forward bracket, initial-state policy, epilogue S2), the adapter input (INJ_INPUT),
`attn.proj` (HEAD_OUT) and `attn.Wqkv` (K_PROJ, V_PROJ). Knockout routes through prod's
attention bias `prod.models.raven._BIAS` [read raven.py:60-152], whose entries are ASSIGNED
finfo.min (DESIGN 6.2).

The initial latent state x0 is a confound unrelated to an edit unless every arm and row sees the
same x0 at the same absolute position (recon P12, F8). `fixed_table` gives that: row key r draws
`table_r` once on a CPU generator seeded 1000003 r with the formula of prod
`raven.StateStream.draw:241-247`, and position p reads `table_r[p]` (DESIGN 6.2).

Also here: the three coda read-outs `pure`, `cached_ref` and `cached` (DESIGN 8.2, MF5), the
float32 read-out `readout32` (DESIGN 22.2) and `McLeishRungAdapter`, prod's McLeish loader for
the r4/r8/r16 rungs at an explicit revision (DESIGN 22.2, M7a-c).
"""
import contextlib
import copy
import hashlib
import types
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ...models import raven as R
from ...models.mcleish import McLeishAdapter
from ...models.raven import DEV, RavenAdapter, install_attention_patch
from ..hooks import (CallCountError, FamilyBinding, HookRegistry, RowInfo, RowKey,
                     SteerConfigError, UnsupportedSite, source_sha256)
from ..readouts import no_tf32
from ..sites import Component, Geometry, Site

C = Component
X0_POLICIES = ("settle", "fixed_table", "zeros")
_BLOCK_COMPONENTS = (C.RESID_PRE, C.ATTN_OUT, C.HEAD_OUT, C.RESID_MID, C.MLP_OUT, C.RESID_POST,
                     C.K_PROJ, C.V_PROJ)
READOUT_MODES = ("pure", "cached_ref", "cached")


def row_seed_int(key: RowKey) -> int:
    """Integer row index of a row key for the x0 seed 1000003 r (raven.SEED_MULT).

    prod keys are task-global row indices (ints, PF1) and are used as they are. A string item key
    (forward mode) maps to int(sha256(key)[:8], 16), the hashing rule DESIGN 7.1 uses for the
    `random` op [decision: DESIGN gives no rule for string keys]."""
    if isinstance(key, bool):
        raise SteerConfigError("row key must be an int or a str, got a bool")
    if isinstance(key, int):
        if key < 0:
            raise SteerConfigError("negative row key %d" % key)
        return key
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16)


class RavenBinding(FamilyBinding):
    """Hooks of DESIGN 6.2 on a loaded prod HuginnAdapter or McLeishAdapter."""
    #: normalized sha256 of SandwichBlock.forward (DESIGN 6 rule), computed 2026-10-07 from
    #: SP/src/huginn_raven.py (snapshot bb6621b6) and the cached McLeish a5f6f126 file. The value
    #: names the family first; the transcription is chosen by it.
    KNOWN_SOURCES = {
        "3b324d8b3bb32a06e5e2274080e7c53c970bc3525683eec914f34ec63aa95617":
            "huginn: raven_modeling_minimal.py SandwichBlock.forward, tomg-group-umd/huginn-0125 "
            "bb6621b6 (525-535)",
        "d2ac932eb90ef94528d2f8cb06fde63212c3b5456487afbf59a28cf0bcea64f9":
            "mcleish: raven_modeling_minimal.py SandwichBlock.forward, smcleish/Recurrent-Llama-"
            "3.2-train-recurrence-32 a5f6f126 (458-469)",
    }

    def __init__(self, adapter: Any):
        if getattr(adapter, "model", None) is None:
            raise SteerConfigError("bind a loaded adapter (call load() first)")
        if adapter.family not in ("huginn", "mcleish"):
            raise SteerConfigError("RavenBinding binds huginn or mcleish, got %r" % adapter.family)
        self.family = adapter.family
        self.adapter = adapter
        self.model = adapter.model
        cfg = self.model.config
        self.P = int(cfg.n_layers_in_prelude)
        self.C = int(cfg.n_layers_in_recurrent_block)
        self.Cd = int(cfg.n_layers_in_coda)
        self.d = int(cfg.n_embd)
        self.reg = None             # type: Optional[HookRegistry]
        self._handles = []          # type: List[Any]
        self._sites = {}            # type: Dict[Tuple[str, Optional[int], int], Dict[Component, Site]]
        self._transcribed = None    # type: Optional[Tuple[str, str]]
        #: None until set_init_state: a decode never falls back to a policy nobody chose, so the
        #: recorded policy is the one that ran (DESIGN 6.2, recon P12; core fixer B-F2)
        self.x0_policy = None       # type: Optional[str]
        self.t_max = None           # type: Optional[int]
        self._tables = {}           # type: Dict[RowKey, Tensor]
        self._call_policy = None    # type: Optional[str]
        self._fw = {}               # type: Dict[str, Any]
        self._ln_f32 = None         # type: Optional[torch.nn.Module]
        self._chunks = []           # type: List[int]

    # ---------------------------------------------------------------- shape
    def model_module(self) -> torch.nn.Module:
        return self.model

    def geometry(self, k: int) -> Geometry:
        cfg = self.model.config
        nh = int(cfg.num_attention_heads)
        return Geometry(self.family, int(k), self.P, self.C, self.Cd, self.d, nh, self.d // nh,
                        True, True)

    def expected_counts(self, k: int) -> Dict[str, int]:
        return {"adapter": int(k), "blocks": self.P + self.C * int(k) + self.Cd}

    def check_intervention(self, iv: Any) -> None:
        if iv.op == "knockout":
            cls = getattr(R._module_of(self.model), "CausalSelfAttention")
            if not getattr(cls, "_prod_patched", False):
                raise UnsupportedSite("raven knockout routes through prod's attention bias; call "
                                      "prod.models.raven.install_attention_patch first")

    def check_recorder(self, rec: Any) -> None:
        if rec.what == "attn_mass":
            raise UnsupportedSite("attn_mass on raven: prod's patched attention runs SDPA and "
                                  "returns no probabilities (DESIGN 1.2, PF5)")
        if rec.what == "gate":
            raise UnsupportedSite("raven has no exit gate")
        if rec.what == "readout_stats" and rec.mode not in READOUT_MODES:
            raise UnsupportedSite("raven read-out mode %r (pure | cached_ref | cached)" % rec.mode)

    # ---------------------------------------------------------------- initial state policy
    def set_init_state(self, policy: str, t_max: Optional[int] = None) -> None:
        """`plan.init_state` (DESIGN 6.2). fixed_table needs T_max = max prompt + horizon + suffix
        + n_answer (REQUIRED), at most `max_positions` on Huginn."""
        if policy not in X0_POLICIES:
            raise SteerConfigError("init_state %r not in %s" % (policy, X0_POLICIES))
        if policy == "fixed_table":
            if t_max is None:
                raise SteerConfigError("init_state fixed_table needs t_max (REQUIRED)")
            cap = getattr(self.adapter, "max_positions", None)
            if cap is not None and int(t_max) > int(cap):
                raise SteerConfigError("t_max %d exceeds max_positions %d (Huginn rotary table)"
                                       % (int(t_max), int(cap)))
            if self.t_max is not None and int(t_max) != self.t_max:
                self._tables = {}
            self.t_max = int(t_max)
        self.x0_policy = policy

    def x0_table(self, key: RowKey) -> Tensor:
        """table_r, float32 CPU [T_max, d]: trunc_normal_(0, std, -3 std, 3 std) on a CPU generator
        seeded 1000003 r, times emb_scale when it is not 1 [raven.StateStream.draw:241-247]."""
        t = self._tables.get(key)
        if t is None:
            std = float(self.model.config.init_values["std"])
            g = torch.Generator()
            g.manual_seed(R.SEED_MULT * row_seed_int(key))
            t = torch.empty((self.t_max, self.d), dtype=torch.float32)
            torch.nn.init.trunc_normal_(t, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=g)
            emb_scale = float(self.model.emb_scale)
            if emb_scale != 1:
                t = t * emb_scale
            self._tables[key] = t
        return t

    def _x0(self, policy: str, B: int, T: int) -> Tensor:
        ctx = self.reg.ctx
        dt, dev = self.dtype(), self.device()
        if policy == "zeros":
            return torch.zeros((B, T, self.d), dtype=dt, device=dev)
        if self.t_max is None:
            raise SteerConfigError("fixed_table x0 without t_max (binding.set_init_state)")
        out = torch.empty((B, T, self.d), dtype=torch.float32)
        abs_pos = ctx.abs_pos.cpu()
        if int(abs_pos.max()) >= self.t_max:
            raise SteerConfigError("absolute position %d >= t_max %d" % (int(abs_pos.max()),
                                                                        self.t_max))
        # The cache holds this batch's rows only (a Huginn table is up to 86 MB). A table is a pure
        # function of (key, t_max), so a row that returns is redrawn bit-identically [B-F4].
        live = set(row.key for row in ctx.rows)
        for key in [k_ for k_ in self._tables if k_ not in live]:
            del self._tables[key]
        for b, row in enumerate(ctx.rows):
            out[b] = self.x0_table(row.key)[abs_pos[b].clamp(min=0)]   # pad columns: table_r[0]
        return out.to(device=dev, dtype=dt)

    # ---------------------------------------------------------------- install
    def install(self, reg: HookRegistry) -> None:
        if self.reg is not None:
            raise SteerConfigError("RavenBinding already has a registry installed")
        tr = self.model.transformer
        blocks = list(tr.prelude) + list(tr.core_block) + list(tr.coda)
        sha = self.check_source(type(blocks[0]).forward, "SandwichBlock.forward")
        desc = self.KNOWN_SOURCES[sha]
        if not desc.startswith(self.family):
            raise SteerConfigError("SandwichBlock source %s is recorded for %r, not %s"
                                   % (sha[:12], desc, self.family))
        self._transcribed = (sha, desc)
        self._chunks = list(blocks[0].attn.chunks)
        body = _HUGINN_BLOCK if self.family == "huginn" else _MCLEISH_BLOCK
        self.reg = reg
        h = self._handles
        h.append(self.model.register_forward_pre_hook(self._pre, with_kwargs=True))
        h.append(self.model.register_forward_hook(self._post, with_kwargs=True))
        h.append(tr.adapter.register_forward_pre_hook(self._on_adapter))
        for blk in blocks:
            blk.forward = types.MethodType(body(self), blk)
            h.append(blk.attn.proj.register_forward_pre_hook(self._on_proj))
            h.append(blk.attn.Wqkv.register_forward_hook(self._on_wqkv))

    def remove(self) -> None:
        tr = self.model.transformer
        for blk in list(tr.prelude) + list(tr.core_block) + list(tr.coda):
            if "forward" in blk.__dict__:
                del blk.__dict__["forward"]
        for hd in self._handles:
            hd.remove()
        self._handles = []
        self.reg = None

    def block_sites(self, stage: str, loop: Optional[int], layer: int) -> Dict[Component, Site]:
        key = (stage, loop, layer)
        s = self._sites.get(key)
        if s is None:
            s = {c: Site(stage, loop, layer, c) for c in _BLOCK_COMPONENTS}
            self._sites[key] = s
        return s

    # ---------------------------------------------------------------- forward bracket
    def _pre(self, module: Any, args: Any, kwargs: Dict[str, Any]) -> Any:
        """Forward bracket, x0 policy and the INIT_STATE site (DESIGN 6.2 hook 1)."""
        reg = self.reg
        cp = kwargs.get("cache_position")
        if cp is None:
            raise SteerConfigError("raven forward without cache_position (prod passes it, "
                                   "raven.py:434, 470)")
        ids = kwargs.get("input_ids", args[0] if args else None)
        B, T = int(ids.shape[0]), int(ids.shape[1])
        reg.begin_forward(cp, T, B=B)
        self._fw = {"cache": kwargs.get("past_key_values"), "n_blocks": 0}
        policy = self._call_policy or self.x0_policy
        if policy is None:
            raise SteerConfigError("raven x0 policy unset: pass init_state (and t_max for "
                                   "fixed_table) to SteeredAdapter or call binding.set_init_state "
                                   "(DESIGN 6.2)")
        if policy != "settle":
            kwargs["input_states"] = self._x0(policy, B, T)
        s = Site("input", None, None, C.INIT_STATE)
        if reg.wants(s):
            if kwargs.get("input_states") is None:
                raise UnsupportedSite("INIT_STATE edit with no input_states (the model would draw "
                                      "x0 inside iterate_forward)")
            kwargs["input_states"] = reg.site(s, kwargs["input_states"])
        return args, kwargs

    def _post(self, module: Any, args: Any, kwargs: Dict[str, Any], output: Any) -> Any:
        """`cached` read-outs, then the epilogue of DESIGN 5 (S2)."""
        reg = self.reg
        for rec, site, h_last in reg.ctx.store.pop("deferred_readouts", []):
            z = self._cached_readout(h_last)
            col = torch.tensor([reg.ctx.T - 1], device=z.device)
            reg.deliver_readout(rec, site, z, columns=col)
        output.logits = reg.logits(output.logits)
        reg.end_forward()
        return output

    def enter_block(self, step_idx: Any, freqs: Any) -> Tuple[str, Optional[int], int]:
        """Map block_idx to (stage, loop, layer) (DESIGN 6.2 hook 2) and, outside read-outs, check
        the block_idx sequence of DESIGN 4.2 and the adapter-call count."""
        idx = int(step_idx)
        P, Cc = self.P, self.C
        if idx < 0:
            call = ("coda", None, -idx - 1)
        elif idx < P:
            call = ("prelude", None, idx)
        else:
            call = ("loop", (idx - P) // Cc + 1, (idx - P) % Cc)
        reg = self.reg
        if not reg.live():
            return call
        n = self._fw["n_blocks"]
        k = reg.k
        want = n if n < P + Cc * k else -(n - P - Cc * k + 1)
        if idx != want:
            raise CallCountError("block_idx %d at block call %d, expected %d" % (idx, n, want))
        if call[0] == "loop" and reg.counts("adapter") != call[1]:
            raise CallCountError("core block of step %d after %d adapter calls"
                                 % (call[1], reg.counts("adapter")))
        self._fw["n_blocks"] = n + 1
        reg.count("blocks")
        ctx = reg.ctx
        ctx.stage, ctx.loop, ctx.layer = call[0], (call[1] or 0), call[2]
        if ctx.freqs is None:
            ctx.freqs = freqs
        return call

    @contextlib.contextmanager
    def knockout_bias(self, call_site: Site) -> Iterator[None]:
        """Route a knocked call through `_BIAS` (DESIGN 6.2): base = the installed bias, or
        build_bias(valid_2d, T, dtype, causal=T > 1) with valid_2d[b, c] = c >= pad[b] for
        c < KV = cache_position[-1] + 1 [read raven.py:155-177]; the copy with the knocked entries
        assigned finfo.min is installed for this call and the old value restored in finally."""
        reg = self.reg
        if not reg.routed(call_site):
            yield
            return
        ctx = reg.ctx
        old = R._BIAS[0]
        if old is None:
            KV = int(ctx.cache_position[-1]) + 1
            cols = torch.arange(KV, device=ctx.cache_position.device)
            pad = torch.tensor(ctx.pad, device=cols.device)
            valid_2d = cols[None, :] >= pad[:, None]
            base = R.build_bias(valid_2d, ctx.T, self.dtype(), causal=ctx.T > 1)
        else:
            base = old
        mask, _routed = reg.attention(call_site, base)
        R._BIAS[0] = mask
        try:
            yield
        finally:
            R._BIAS[0] = old

    # ---------------------------------------------------------------- module hooks
    def _call_site(self, comp: Component) -> Optional[Site]:
        reg = self.reg
        if reg is None or not reg.live():
            return None
        ctx = reg.ctx
        loop = ctx.loop if ctx.stage == "loop" else None
        return self.block_sites(ctx.stage, loop, ctx.layer)[comp]

    def _on_adapter(self, module: Any, args: Any) -> Any:
        """Adapter input cat[x, e] (DESIGN 6.2 hook 3): step s from the call count; INJ_INPUT(s)
        edits columns [d:] (INJ_STATE(s) is an alias of a hooked tensor, DESIGN 4.1)."""
        reg = self.reg
        if not reg.live():
            return None
        s = reg.count("adapter")
        if self._fw["n_blocks"] != self.P + self.C * (s - 1):
            raise CallCountError("adapter call %d after %d block calls" % (s, self._fw["n_blocks"]))
        ctx = reg.ctx
        ctx.stage, ctx.loop, ctx.layer = "loop", s, -1
        site = Site("loop", s, None, C.INJ_INPUT)
        if not reg.wants(site):
            return None
        u = args[0]
        e = reg.site(site, u[..., self.d:])
        return (torch.cat([u[..., :self.d], e], dim=-1),) + tuple(args[1:])

    def _on_proj(self, module: Any, args: Any) -> Any:
        """HEAD_OUT: attn.proj input viewed [B, T, H, hd] (DESIGN 6.2 hook 4)."""
        s = self._call_site(C.HEAD_OUT)
        if s is None or not self.reg.wants(s):
            return None
        x = args[0]
        B, T = x.shape[0], x.shape[1]
        g = self.reg.geom
        y = self.reg.site(s, x.view(B, T, g.n_heads, g.head_dim))
        return (y.reshape(B, T, -1),) + tuple(args[1:])

    def _on_wqkv(self, module: Any, args: Any, out: Tensor) -> Any:
        """K_PROJ / V_PROJ: the K and V parts of Wqkv(x).split(chunks, dim=2) before rotary
        [read raven.py:121] (DESIGN 6.2 hook 5, PF3)."""
        sk, sv = self._call_site(C.K_PROJ), self._call_site(C.V_PROJ)
        if sk is None or not (self.reg.wants(sk) or self.reg.wants(sv)):
            return None
        q, k, v = out.split(self._chunks, dim=2)
        k = self.reg.site(sk, k)
        v = self.reg.site(sv, v)
        return torch.cat([q, k, v], dim=2)

    # ---------------------------------------------------------------- forward and read-outs
    def forward(self, ids: Tensor, k: int, row_keys: List[RowKey], x0_policy: str = "fixed_table",
                grad: bool = False) -> Tensor:
        """model(input_ids, num_steps=k, input_states=x0, use_cache=False, cache_position=arange(T))
        -> float32 logits [B, T, V] (DESIGN 6.2). grad=True passes num_steps=tensor([0, k]) under
        enable_grad so every core step runs in the with-grad loop (MF1, S1)."""
        reg = self.reg
        if reg is None:
            raise SteerConfigError("binding.forward needs an installed registry (it applies x0)")
        if int(k) != reg.k:
            raise SteerConfigError("forward at k=%d on a registry of depth %d" % (k, reg.k))
        if x0_policy not in ("fixed_table", "zeros"):
            raise SteerConfigError("forward mode x0 policy must be fixed_table or zeros; settle "
                                   "applies to prod decode only")
        if grad:
            if getattr(self.model, "gradient_checkpointing", False):
                raise SteerConfigError("grad mode with gradient_checkpointing would replay the hooks")
            num_steps = torch.tensor([0, int(k)])                  # type: Any
            mode = torch.enable_grad()                              # type: Any
        else:
            num_steps = int(k)
            mode = torch.no_grad()
        T = ids.shape[1]
        self._call_policy = x0_policy
        try:
            with _rows_for(reg, ids, row_keys), mode:
                out = self.model(input_ids=ids, num_steps=num_steps, input_states=None,
                                 use_cache=False, cache_position=torch.arange(T, device=ids.device))
        finally:
            self._call_policy = None
        return out.logits.float()

    def _coda(self, x: Tensor, freqs: Any, past_key_values: Any) -> Tensor:
        """Family read-out body before lm_head: Huginn ln_f(coda(ln_f(h))) [read 868-898], McLeish
        ln_f(coda(h)) [read 684-688]; coda blocks with block_idx -1, -2, ..."""
        tr = self.model.transformer
        if self.family == "huginn":
            x = tr.ln_f(x)
        idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for blk in tr.coda:
            idx = idx - 1
            x = blk(x, freqs, idx, None, past_key_values)
        return x

    def _freqs(self, T: int) -> Any:
        ctx = self.reg.ctx if self.reg is not None else None
        if ctx is None or ctx.freqs is None:
            raise SteerConfigError("raven read-out needs the rotary freqs of a forward")
        if _freqs_len(ctx.freqs) != T:
            raise SteerConfigError("read-out over %d positions with freqs of %d" % (
                T, _freqs_len(ctx.freqs)))
        return ctx.freqs

    def readout(self, j: int, h: Tensor, mode: str) -> Tensor:
        """`pure`: no cache, all positions, no bias (k10_hf.coda_lens); `cached_ref`: the live cache
        over all positions of the forward, _BIAS as installed (V9 port check only); `cached` is run
        by the post-hook for the last column (DESIGN 8.2). Sets ctx.in_readout throughout (S10)."""
        if mode not in READOUT_MODES:
            raise UnsupportedSite("raven read-out mode %r" % (mode,))
        if mode == "cached":
            raise SteerConfigError("cached read-outs run deferred, at the last column, from the "
                                   "forward's post-hook (register a readout_stats recorder)")
        if mode == "pure":
            self._check_pure()
        freqs = self._freqs(h.shape[1])
        with _guard(self.reg):
            if mode == "pure":
                old = R._BIAS[0]
                R._BIAS[0] = None
                try:
                    x = self._coda(h, freqs, None)
                finally:
                    R._BIAS[0] = old
            else:
                cache = self._fw.get("cache")
                if cache is None:
                    raise UnsupportedSite("cached_ref read-out needs a cached decode forward")
                with _coda_slots_restored(cache, self.Cd, h.shape[1]):
                    x = self._coda(h, freqs, cache)
            z = self.model.lm_head(self.model.transformer.ln_f(x)).float()
        self.reg.add_extra_passes(self.Cd * h.shape[0] * h.shape[1])
        return z

    def _check_pure(self) -> None:
        """`pure` is forward mode (DESIGN 8.2): no cache and pad 0. Under a cache the lens sees only
        the forward's own columns (one token at a decode step), and with left pads the original
        attention runs is_causal with no key-padding mask [core review G2-F1: huginn
        CausalSelfAttention.forward 454-500, McLeish a5f6f126 387-428; measured on the CPU fakes by
        test_raven_pure_readout_refused_in_decode_and_padded], so the value would not be a function of (row,
        position) (DESIGN 4.4)."""
        if self._fw.get("cache") is not None:
            raise UnsupportedSite("raven pure read-out inside a cached forward; use cached")
        ctx = self.reg.ctx if self.reg is not None else None
        if ctx is not None and any(int(p) != 0 for p in ctx.pad):
            raise UnsupportedSite("raven pure read-out on a left-padded forward (pads %r)"
                                  % (ctx.pad,))

    def _cached_readout(self, h_last: Tensor) -> Tensor:
        """DESIGN 8.2 `cached`: save the coda slots at the last position, run the coda for that
        position as a T = 1 query over the cache, ln_f and lm_head, restore the slots."""
        cache = self._fw.get("cache")
        if cache is None:
            raise UnsupportedSite("cached read-out on a forward without a cache; use pure")
        pos = int(cache._seen_tokens) - 1
        slots = [-(l + 1) for l in range(self.Cd)]
        saved = [(cache.key_cache[s][:, :, pos].clone(), cache.value_cache[s][:, :, pos].clone(),
                  bool(cache.valid_mask[s, pos])) for s in slots]
        freqs = _freqs_last(self.reg.ctx.freqs)
        try:
            with _guard(self.reg):
                x = self._coda(h_last, freqs, cache)
                z = self.model.lm_head(self.model.transformer.ln_f(x)).float()
        finally:
            for s, (kk, vv, ok) in zip(slots, saved):
                cache.key_cache[s][:, :, pos] = kk
                cache.value_cache[s][:, :, pos] = vv
                cache.valid_mask[s, pos] = ok
        self.reg.add_extra_passes(self.Cd * h_last.shape[0])
        return z

    def readout32(self, h: Tensor, ro: Any) -> Tensor:
        """Float32 read-out (DESIGN 22.2): lm_head in float32 on the float32 output of the family
        read-out's final norm; for ("loop", j) the coda runs first (pure mode) in the model dtype
        (recorded states are float32 by default; the coda's Linear layers need the model dtype,
        core fixer B-F1). TF32 off."""
        if self._ln_f32 is None:
            self._ln_f32 = copy.deepcopy(self.model.transformer.ln_f).float()
        if ro != "final":
            self._check_pure()
        with _guard(self.reg), no_tf32():
            if ro != "final":
                old = R._BIAS[0]
                R._BIAS[0] = None
                try:
                    h = self._coda(h.to(self.dtype()), self._freqs(h.shape[1]), None)
                finally:
                    R._BIAS[0] = old
            n = self._ln_f32(h.float())
            return F.linear(n, self.model.lm_head.weight.float())

    def source_report(self) -> Dict[str, Any]:
        cls = getattr(R._module_of(self.model), "CausalSelfAttention")
        sha, desc = self._transcribed if self._transcribed else (None, None)
        return {"family": self.family, "loader": type(self.adapter).__name__,
                "transcribed": {"SandwichBlock.forward": sha, "source": desc},
                "attention_patch": bool(getattr(cls, "_prod_patched", False)),
                "attention_forward_sha256": (source_sha256(cls._prod_original_forward)
                                             if getattr(cls, "_prod_patched", False) else None),
                "num_steps_mode": "split_tensor",
                "x0_policy": self.x0_policy, "t_max": self.t_max,
                "x0_seed_rule": "1000003 * row_index; str keys: int(sha256(key)[:8], 16)",
                "emb_scale": float(self.model.emb_scale),
                "init_std": float(self.model.config.init_values["std"]),
                "test_time_noise": getattr(self.model.config, "test_time_noise", None)}


# ------------------------------------------------------------------ transcribed block forwards
def _HUGINN_BLOCK(binding: RavenBinding) -> Any:
    """SandwichBlock.forward transcribed from Huginn raven_modeling_minimal.py 525-535 with the
    registry calls of DESIGN 6.2: x = norm_2(attn_out + x); x = norm_4(mlp(norm_3(x)) + x)."""

    def forward(self, x, freqs_cis, step_idx, mask=None, past_key_values=None):
        reg = binding.reg
        stage, loop, layer = binding.enter_block(step_idx, freqs_cis)
        if reg.skipped(stage, loop, layer):
            if past_key_values is not None and stage == "prelude" and layer == 0:
                raise UnsupportedSite("skip of raven prelude L0 under a cache (S3)")
            return x
        S = binding.block_sites(stage, loop, layer)
        if reg.wants(S[C.RESID_PRE]):
            x = reg.site(S[C.RESID_PRE], x)
        with binding.knockout_bias(S[C.ATTN_OUT]):
            attn_out = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values)
        if reg.wants(S[C.ATTN_OUT]):
            attn_out = reg.site(S[C.ATTN_OUT], attn_out)
        x = self.norm_2(attn_out + x)
        if reg.wants(S[C.RESID_MID]):
            x = reg.site(S[C.RESID_MID], x)
        mlp_out = self.mlp(self.norm_3(x))
        if reg.wants(S[C.MLP_OUT]):
            mlp_out = reg.site(S[C.MLP_OUT], mlp_out)
        x = self.norm_4(mlp_out + x)
        if reg.wants(S[C.RESID_POST]):
            x = reg.site(S[C.RESID_POST], x)
        return x

    return forward


def _MCLEISH_BLOCK(binding: RavenBinding) -> Any:
    """SandwichBlock.forward transcribed from McLeish raven_modeling_minimal.py 458-469 with the
    registry calls of DESIGN 6.2: x = attn_out + x; x = mlp(norm_2(x)) + x."""

    def forward(self, x, freqs_cis, step_idx, mask=None, past_key_values=None):
        reg = binding.reg
        stage, loop, layer = binding.enter_block(step_idx, freqs_cis)
        if reg.skipped(stage, loop, layer):
            if past_key_values is not None and stage == "prelude" and layer == 0:
                raise UnsupportedSite("skip of raven prelude L0 under a cache (S3)")
            return x
        S = binding.block_sites(stage, loop, layer)
        if reg.wants(S[C.RESID_PRE]):
            x = reg.site(S[C.RESID_PRE], x)
        with binding.knockout_bias(S[C.ATTN_OUT]):
            attn_out = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values)
        if reg.wants(S[C.ATTN_OUT]):
            attn_out = reg.site(S[C.ATTN_OUT], attn_out)
        x = attn_out + x
        if reg.wants(S[C.RESID_MID]):
            x = reg.site(S[C.RESID_MID], x)
        mlp_out = self.mlp(self.norm_2(x))
        if reg.wants(S[C.MLP_OUT]):
            mlp_out = reg.site(S[C.MLP_OUT], mlp_out)
        x = mlp_out + x
        if reg.wants(S[C.RESID_POST]):
            x = reg.site(S[C.RESID_POST], x)
        return x

    return forward


# ------------------------------------------------------------------ helpers
def _freqs_len(freqs: Any) -> int:
    """Positions covered by a forward's rotary freqs: Huginn freqs_cis[:, cache_position], McLeish
    the (cos, sin) pair of rotary_emb(x, position_ids); the position axis is dim 1 in both."""
    f = freqs[0] if isinstance(freqs, (tuple, list)) else freqs
    return int(f.shape[1])


def _freqs_last(freqs: Any) -> Any:
    if isinstance(freqs, (tuple, list)):
        return type(freqs)(f[:, -1:] for f in freqs)
    return freqs[:, -1:]


@contextlib.contextmanager
def _coda_slots_restored(cache: Any, n_coda: int, T: int) -> Iterator[None]:
    """Save the coda cache slots (block_idx -1 .. -n_coda) at the forward's columns
    [seen - T, seen), where HuginnStaticCache.update writes [read huginn 356-381], and restore them
    after a lens pass. A `cached_ref` lens writes amateur K/V there; at a loop site the forward's
    own coda overwrites them, but at a coda site or a skipped coda block nothing does, and later
    decode steps would attend to them (DESIGN 5 neutral recorders) [core fixer B-F5]."""
    seen = int(cache._seen_tokens)
    a = seen - int(T)
    slots = [-(l + 1) for l in range(n_coda)]
    saved = [(cache.key_cache[s][:, :, a:seen].clone(), cache.value_cache[s][:, :, a:seen].clone(),
              cache.valid_mask[s, a:seen].clone()) for s in slots]
    try:
        yield
    finally:
        for s, (kk, vv, ok) in zip(slots, saved):
            cache.key_cache[s][:, :, a:seen] = kk
            cache.value_cache[s][:, :, a:seen] = vv
            cache.valid_mask[s, a:seen] = ok


def _guard(reg: Optional[HookRegistry]) -> Any:
    return reg.readout_guard() if reg is not None else contextlib.nullcontext()


def _rows_for(reg: HookRegistry, ids: Tensor, row_keys: List[RowKey]) -> Any:
    """The caller's open reg.batch (it carries spans and payloads), else plain rows of the keys."""
    if len(row_keys) != ids.shape[0]:
        raise SteerConfigError("%d row keys for a batch of %d" % (len(row_keys), ids.shape[0]))
    if reg.rows is not None:
        if [r.key for r in reg.rows] != list(row_keys):
            raise SteerConfigError("row keys %r differ from the open batch %r"
                                   % (list(row_keys), [r.key for r in reg.rows]))
        if any(int(p) != 0 for p in reg.pad):
            raise SteerConfigError("binding.forward is no-cache with pad 0 (DESIGN 5); the open "
                                   "batch has pads %r" % (reg.pad,))
        return contextlib.nullcontext()
    T = ids.shape[1]
    return reg.batch([RowInfo(key=k_, prompt_len=T) for k_ in row_keys], [0] * len(row_keys))


# ------------------------------------------------------------------ McLeish rungs (M7a-c)
class McLeishRungAdapter(McLeishAdapter):
    """prod McLeishAdapter for smcleish/Recurrent-Llama-3.2-train-recurrence-{4, 8, 16} at an
    explicit revision (DESIGN 22.2). `load` is ported line by line from prod/models/raven.py:275-
    308; the only change is `revision=self.revision` (prod's pins file has no entry for these
    names). revision and depths are REQUIRED: neither is recorded anywhere yet (matrix Q3, Q6)."""

    def __init__(self, rung: int, revision: str, depths: Tuple[int, ...], **kw: Any):
        if not revision:
            raise SteerConfigError("McLeishRungAdapter: revision is REQUIRED (matrix C11)")
        if not depths:
            raise SteerConfigError("McLeishRungAdapter: depths are REQUIRED")
        RavenAdapter.__init__(self, name="mcleish_llama32_r%d" % int(rung),
                              repo="smcleish/Recurrent-Llama-3.2-train-recurrence-%d" % int(rung),
                              depths=tuple(depths), trained_depth=int(rung), **kw)
        self.revision = str(revision)

    def load(self, strict_patch=False):
        if DEV == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.mem_fraction)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # load the REVISION that `prod.install_models` pinned, not whatever the repo's `main`
        # points at today. `revision=None` (no pins file yet) loads `main`.
        rev = self.revision
        tok = AutoTokenizer.from_pretrained(self.repo, revision=rev)
        m = AutoModelForCausalLM.from_pretrained(
            self.repo, dtype=self.dtype, trust_remote_code=True,
            revision=rev).to(DEV).eval()
        patch = install_attention_patch(m, strict=strict_patch)
        self.tok = tok
        self.model = m
        cfg = m.config
        self._info = {"attention_patch": patch,
                      "n_layers_in_prelude": int(cfg.n_layers_in_prelude),
                      "n_layers_in_recurrent_block": int(cfg.n_layers_in_recurrent_block),
                      "n_layers_in_coda": int(cfg.n_layers_in_coda),
                      "mean_recurrence": int(getattr(cfg, "mean_recurrence", 0)),
                      "test_time_noise": getattr(cfg, "test_time_noise", None),
                      "n_embd": int(cfg.n_embd), "n_heads": int(cfg.num_attention_heads),
                      "num_key_value_heads": int(cfg.num_key_value_heads),
                      "head_dim": int(cfg.n_embd // cfg.num_attention_heads),
                      "emb_scale": float(m.emb_scale),
                      "init_std": float(cfg.init_values["std"]),
                      "pad_id": self.pad_id(), "stop_ids": self.stop_ids(),
                      "block_size": int(getattr(cfg, "block_size", 0) or 0)}
        # Huginn's apply_rotary_emb_complex_like indexes a precomputed table of block_size
        # positions, so an absolute position at or past block_size raises; McLeish's llama-style
        # rotary is computed on the fly and its block_size (1024) is not a limit (S9f ran past it).
        self.max_positions = (int(cfg.block_size) if patch.get("rotary_style") == "complex"
                              and getattr(cfg, "block_size", None) else None)
        return self

    def meta(self):
        d = super(McLeishRungAdapter, self).meta()
        d["revision"] = self.revision
        return d
