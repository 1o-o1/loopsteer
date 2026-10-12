"""Ouro binding: hooks on Ouro-1.4B/2.6B, base and Thinking (DESIGN 3.1, 6.1, 8.1, 22.2, 22.11).

What this module measures and why. Ouro runs its L decoder layers k times with no input
injection; the post-norm state at the end of loop j is both the loop-j exit read-out and the
loop j+1 input (base 592-608). To edit or read any (loop, layer, component) tensor, the decoder
layer forward is transcribed as an instance `forward` (base 388-422) with one `reg.site` call per
component, guarded by the normalized source hash of the original, so every number this package
reports comes from the checkpoint's own arithmetic in the original order. Module hooks cover the
rest: the top-level pre/post hooks (forward bracket, exit traps S4, epilogue S2), `o_proj`
(HEAD_OUT), `k_proj`/`v_proj` (PF3), `model.norm` (LOOP_END) and `early_exit_gate` (gate).

Also here: `EagerOuroAdapter`, prod's loader with `attn_implementation="eager"` (K2: knockout and
attention probabilities need the eager mask path), `eager_vs_sdpa_gap`, the per-item gap every
knockout run records (K2, P14), and `SampledOuroAdapter`, prod's decode with temperature/top-p
sampling for the descriptive AIME row (DESIGN 22.11).
"""
import contextlib
import hashlib
import time
import types
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ... import config as cfgmod
from ...models.ouro import (DEV, OuroAdapter, leftpad, new_forced_states, copy_forced_states,
                            patch_universal_cache)
from ..hooks import (CallCountError, FamilyBinding, HookRegistry, RowInfo, RowKey,
                     SteerConfigError, UnsupportedSite, source_sha256)
from ..readouts import no_tf32
from ..sites import Component, Geometry, Site

C = Component
_LAYER_COMPONENTS = (C.RESID_PRE, C.ATTN_OUT, C.HEAD_OUT, C.RESID_MID, C.MLP_OUT, C.RESID_POST,
                     C.K_PROJ, C.V_PROJ)
READOUT_MODES = ("exit", "logit_lens")
_RESID = (C.RESID_PRE, C.RESID_MID, C.RESID_POST)


class OuroBinding(FamilyBinding):
    """Hooks of DESIGN 6.1 on a loaded prod OuroAdapter (or EagerOuroAdapter)."""
    family = "ouro"
    #: normalized sha256 of OuroDecoderLayer.forward (computed 2026-10-07 from the cached
    #: modeling_ouro.py of each revision with the inspect.getsource rule of DESIGN 6)
    KNOWN_SOURCES = {
        "ca0c9363e8e66cdda769f5220797820a839f4eb561ce370dc8d1dd8f238a386b":
            "modeling_ouro.py OuroDecoderLayer.forward: ByteDance/Ouro-1.4B 574fa66 (md5 5a0c3755, "
            "base 388-422), Ouro-2.6B (same file), Ouro-1.4B-Thinking 3aaa222 (md5 7129d9d0, "
            "410-446), Ouro-2.6B-Thinking f1edd81 (md5 4135d08e)",
    }
    #: originals whose structure the hooks rely on without transcribing them (reported only)
    CONTEXT_SOURCES = {
        "OuroModel.forward": "1db37da0b6b8cdf6fe83abaa487b7dc38f7119fb9652086bf655d5f5cd5b040a",
        "OuroForCausalLM.forward":
            "019d1e8134722fa02df7fef904eac6381ef66c7bcf24cfa554f20ddca35a2153",
    }

    def __init__(self, adapter: Any):
        if getattr(adapter, "model", None) is None:
            raise SteerConfigError("bind a loaded adapter (call load() first)")
        self.adapter = adapter
        self.model = adapter.model
        self.reg = None             # type: Optional[HookRegistry]
        self._handles = []          # type: List[Any]
        self._sites = {}            # type: Dict[Any, Dict[Component, Site]]
        self._transcribed_sha = None

    # ---------------------------------------------------------------- shape
    def model_module(self) -> torch.nn.Module:
        return self.model

    def geometry(self, k: int) -> Geometry:
        cfg = self.model.config
        return Geometry("ouro", int(k), 0, int(cfg.num_hidden_layers), 0, int(cfg.hidden_size),
                        int(cfg.num_attention_heads), int(cfg.head_dim), False, False)

    def expected_counts(self, k: int) -> Dict[str, int]:
        return {"loop_end": int(k)}

    def attn_implementation(self) -> Optional[str]:
        return getattr(self.model.config, "_attn_implementation", None)

    def check_intervention(self, iv: Any) -> None:
        if iv.op == "knockout" and self.attn_implementation() != "eager":
            raise UnsupportedSite("Ouro knockout needs the eager load (EagerOuroAdapter, K2); "
                                  "this model runs %r" % self.attn_implementation())

    def check_recorder(self, rec: Any) -> None:
        if rec.what == "attn_mass" and self.attn_implementation() != "eager":
            raise UnsupportedSite("attn_mass needs the eager load (PF5, K2)")
        if rec.what == "readout_stats" and rec.mode not in READOUT_MODES:
            raise UnsupportedSite("Ouro read-out mode %r (exit | logit_lens)" % (rec.mode,))
        if rec.what == "readout_stats":
            # DESIGN 8.1: `exit` is lm_head(loop_end_j), whose input model.norm already normalized
            # (base 606); `logit_lens` is lm_head(rmsnorm(h)) on RESID_* sites. The crossed pairs
            # are a second RMSNorm or none at all [core fixer A-F1].
            for s in rec.sites:
                c = Component(s.component)
                if rec.mode == "exit" and c != C.LOOP_END:
                    raise UnsupportedSite("Ouro exit read-out at %s: exit reads LOOP_END sites only "
                                          "(use logit_lens on RESID_* sites)" % s.key())
                if rec.mode == "logit_lens" and c not in _RESID:
                    raise UnsupportedSite("Ouro logit_lens read-out at %s: logit_lens reads RESID_* "
                                          "sites only (use exit at LOOP_END)" % s.key())

    # ---------------------------------------------------------------- install
    def install(self, reg: HookRegistry) -> None:
        if self.reg is not None:
            raise SteerConfigError("OuroBinding already has a registry installed")
        layers = self.model.model.layers
        self._transcribed_sha = self.check_source(type(layers[0]).forward,
                                                  "OuroDecoderLayer.forward")
        self.reg = reg
        m = self.model
        h = self._handles
        h.append(m.register_forward_pre_hook(self._pre, with_kwargs=True))
        h.append(m.register_forward_hook(self._post, with_kwargs=True))
        for l, layer in enumerate(layers):
            layer.forward = types.MethodType(_transcribed_layer_forward(self), layer)
            att = layer.self_attn
            h.append(att.o_proj.register_forward_pre_hook(self._head_hook(l)))
            h.append(att.k_proj.register_forward_hook(self._proj_hook(l, C.K_PROJ)))
            h.append(att.v_proj.register_forward_hook(self._proj_hook(l, C.V_PROJ)))
        h.append(m.model.norm.register_forward_hook(self._on_norm))
        h.append(m.model.early_exit_gate.register_forward_hook(self._on_gate))

    def remove(self) -> None:
        for layer in self.model.model.layers:
            if "forward" in layer.__dict__:
                del layer.__dict__["forward"]
        for hd in self._handles:
            hd.remove()
        self._handles = []
        self.reg = None

    def layer_sites(self, j: int, l: int) -> Dict[Component, Site]:
        key = (j, l)
        s = self._sites.get(key)
        if s is None:
            s = {c: Site("loop", j, l, c) for c in _LAYER_COMPONENTS}
            self._sites[key] = s
        return s

    # ---------------------------------------------------------------- hooks
    def _pre(self, module: Any, args: Any, kwargs: Dict[str, Any]) -> None:
        """Exit traps (S4) and the forward bracket."""
        reg, m = self.reg, self.model
        if kwargs.get("labels") is not None:
            raise SteerConfigError("steer code never passes labels to Ouro (S4: the labels path "
                                   "returns the exit-pdf mixture of all k exits)")
        if int(m.model.total_ut_steps) != reg.k or m.early_exit_step != reg.k - 1:
            raise SteerConfigError("Ouro forward with total_ut_steps=%r early_exit_step=%r; the "
                                   "registry depth is %d (S4)" % (m.model.total_ut_steps,
                                                                  m.early_exit_step, reg.k))
        cp = kwargs.get("cache_position")
        if cp is None:
            raise SteerConfigError("Ouro forward without cache_position")
        ids = kwargs.get("input_ids", args[0] if args else None)
        ref = ids if ids is not None else kwargs["inputs_embeds"]
        reg.begin_forward(cp, ref.shape[1], attn_2d=kwargs.get("attention_mask"), B=ref.shape[0])

    def _post(self, module: Any, args: Any, kwargs: Dict[str, Any], output: Any) -> Any:
        """Forward epilogue (S2): logits processors, then the call-count asserts."""
        output.logits = self.reg.logits(output.logits)
        self.reg.end_forward()
        return output

    def enter_layer(self, j: int, l: int) -> None:
        """Set ctx stage/loop/layer and check current_ut against the norm-call count (k9_skip
        73-80 pattern)."""
        reg = self.reg
        ctx = reg.ctx
        if ctx is None or ctx.in_readout:
            return
        if reg.counts("loop_end") != j - 1:
            raise CallCountError("layer call at loop %d after %d norm calls"
                                 % (j, reg.counts("loop_end")))
        ctx.stage, ctx.loop, ctx.layer = "loop", j, l

    def _head_hook(self, l: int) -> Any:
        def hook(module: Any, args: Any) -> Any:
            reg = self.reg
            if reg.ctx is None:
                return None
            s = self.layer_sites(reg.ctx.loop, l)[C.HEAD_OUT]
            if not reg.wants(s):
                return None
            x = args[0]
            B, T = x.shape[0], x.shape[1]
            y = reg.site(s, x.view(B, T, reg.geom.n_heads, reg.geom.head_dim))
            return (y.reshape(B, T, -1),) + tuple(args[1:])
        return hook

    def _proj_hook(self, l: int, comp: Component) -> Any:
        def hook(module: Any, args: Any, out: Tensor) -> Any:
            reg = self.reg
            if reg.ctx is None:
                return None
            s = self.layer_sites(reg.ctx.loop, l)[comp]
            return reg.site(s, out) if reg.wants(s) else None
        return hook

    def _on_norm(self, module: Any, args: Any, out: Tensor) -> Any:
        reg = self.reg
        if not reg.live():
            return None
        ctx = reg.ctx
        j = reg.count("loop_end")
        if j != ctx.loop:
            raise CallCountError("norm call %d during loop %d" % (j, ctx.loop))
        s = Site("loop", j, None, C.LOOP_END)
        return reg.site(s, out) if reg.wants(s) else None

    def _on_gate(self, module: Any, args: Any, out: Tensor) -> None:
        reg = self.reg
        j = reg.counts("loop_end")
        if reg.wants_gate(j):
            reg.gate(j, out)

    # ---------------------------------------------------------------- forward and read-outs
    def forward(self, ids: Tensor, k: int, row_keys: List[RowKey], x0_policy: str = "fixed_table",
                grad: bool = False) -> Tensor:
        """No-cache forward, batch rows unpadded: model(input_ids, cache_position=arange(T),
        use_cache=False); float32 logits [B, T, V]. grad=True runs under enable_grad (DESIGN 6.1:
        no inner no_grad on Ouro's state path, only the rotary forward is decorated)."""
        reg = self.reg
        if reg is not None and int(k) != reg.k:
            raise SteerConfigError("forward at k=%d on a registry of depth %d" % (k, reg.k))
        T = ids.shape[1]
        cp = torch.arange(T, device=ids.device)
        mode = torch.enable_grad() if grad else torch.no_grad()
        with _rows_for(reg, ids, row_keys), mode:
            out = self.model(input_ids=ids, cache_position=cp, use_cache=False)
        return out.logits.float()

    def readout(self, j: int, h: Tensor, mode: str) -> Tensor:
        """`exit` = lm_head(loop_end_j) (k16_path_ouro.py:301); `logit_lens` = lm_head(rmsnorm(h))
        with the class forward of model.model.norm, so the LOOP_END hook does not fire (S10)."""
        W = self.model.lm_head.weight
        with _guard(self.reg):
            if mode == "exit":
                return F.linear(h, W).float()
            if mode == "logit_lens":
                n = self.model.model.norm
                return F.linear(type(n).forward(n, h), W).float()
        raise UnsupportedSite("Ouro read-out mode %r (exit | logit_lens)" % (mode,))

    def readout32(self, h: Tensor, ro: Any) -> Tensor:
        """Float32 read-out from LOOP_END(j): F.linear(h.float(), lm_head.weight.float()), TF32
        off (DESIGN 22.2)."""
        with _guard(self.reg), no_tf32():
            return F.linear(h.float(), self.model.lm_head.weight.float())

    def source_report(self) -> Dict[str, Any]:
        mm = self.model.model
        ctx_rep = {}
        for name, want in self.CONTEXT_SOURCES.items():
            cls = type(mm) if name.startswith("OuroModel") else type(self.model)
            got = source_sha256(cls.forward)
            ctx_rep[name] = {"sha256": got, "known": got == want}
        return {"family": "ouro", "loader": type(self.adapter).__name__,
                "attn_implementation": self.attn_implementation(),
                "early_exit_threshold": getattr(self.model, "early_exit_threshold", None),
                "transcribed": {"OuroDecoderLayer.forward": self._transcribed_sha},
                "context_sources": ctx_rep}


def eager_vs_sdpa_gap(binding: OuroBinding, ids: Tensor, k: int, row_key: RowKey, read: int,
                      t_a: int, t_b: int) -> Dict[str, Any]:
    """Clean final logits under eager vs SDPA for one item (DESIGN 6.1 K2, P14), the k16 method
    [read k16_path_ouro.sdpa_gap:407-430]: a clean forward, the same forward with
    config._attn_implementation switched to "sdpa", then switched back. Reported at the read
    position as the margin gap |delta(z(tB) - z(tA))| and the max abs gap over the vocabulary,
    with a check that the restored eager forward is bit-identical. Runs on its own registry with no
    consumers, so it needs the binding free of any installed registry."""
    if binding.reg is not None:
        raise SteerConfigError("eager_vs_sdpa_gap runs on a binding with no installed registry")
    cfg = binding.model.config
    prev = cfg._attn_implementation
    if prev != "eager":
        raise SteerConfigError("eager_vs_sdpa_gap needs the eager load, got %r" % (prev,))
    reg = HookRegistry(binding, k)
    reg.install()
    try:
        ze = binding.forward(ids, k, [row_key])[0, read]
        try:
            cfg._attn_implementation = "sdpa"
            zs = binding.forward(ids, k, [row_key])[0, read]
        finally:
            cfg._attn_implementation = prev
        again = binding.forward(ids, k, [row_key])[0, read]
    finally:
        reg.remove()
    m_e, m_s = ze[t_b] - ze[t_a], zs[t_b] - zs[t_a]
    return {"margin_gap": float((m_e - m_s).abs()), "max_abs": float((ze - zs).abs().max()),
            "top1_agree": bool(int(ze.argmax()) == int(zs.argmax())),
            "impl_after": cfg._attn_implementation,
            "restored_eager_bit_identical": bool(torch.equal(again, ze))}


def _guard(reg: Optional[HookRegistry]) -> Any:
    return reg.readout_guard() if reg is not None else contextlib.nullcontext()


def _rows_for(reg: Optional[HookRegistry], ids: Tensor, row_keys: List[RowKey]) -> Any:
    """Use the caller's reg.batch when it is open (it carries spans and payloads); otherwise open
    one with plain rows of the given keys (pad 0, prompt_len T)."""
    if len(row_keys) != ids.shape[0]:
        raise SteerConfigError("%d row keys for a batch of %d" % (len(row_keys), ids.shape[0]))
    if reg is None:
        return contextlib.nullcontext()
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


def _transcribed_layer_forward(binding: OuroBinding) -> Any:
    """OuroDecoderLayer.forward transcribed from modeling_ouro base 388-422 (Thinking 410-446,
    bodies identical up to blank lines), with the registry calls of DESIGN 6.1."""

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        reg = binding.reg
        j = int(kwargs["current_ut"]) + 1
        l = int(self.self_attn.layer_idx)
        binding.enter_layer(j, l)
        if reg.skipped("loop", j, l):
            if past_key_value is not None and j == 1 and l == 0:
                raise UnsupportedSite("skip of Ouro (loop 1, layer 0) under a cache (S3)")
            return hidden_states                                       # k9_skip.py:83-86
        S = binding.layer_sites(j, l)
        if reg.wants(S[C.RESID_PRE]):
            hidden_states = reg.site(S[C.RESID_PRE], hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_mask, _routed = reg.attention(S[C.ATTN_OUT], attention_mask)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if reg.wants_attn(S[C.ATTN_OUT]):
            if attn_weights is None:
                raise UnsupportedSite("attn_mass: the attention module returned no "
                                      "probabilities (eager load needed, PF5)")
            reg.attn_probs(S[C.ATTN_OUT], attn_weights)
        hidden_states = self.input_layernorm_2(hidden_states)
        if reg.wants(S[C.ATTN_OUT]):
            hidden_states = reg.site(S[C.ATTN_OUT], hidden_states)
        hidden_states = residual + hidden_states
        if reg.wants(S[C.RESID_MID]):
            hidden_states = reg.site(S[C.RESID_MID], hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        if reg.wants(S[C.MLP_OUT]):
            hidden_states = reg.site(S[C.MLP_OUT], hidden_states)
        hidden_states = residual + hidden_states
        if reg.wants(S[C.RESID_POST]):
            hidden_states = reg.site(S[C.RESID_POST], hidden_states)
        return hidden_states

    return forward


# ------------------------------------------------------------------ loaders (K2, DESIGN 22.11)
class EagerOuroAdapter(OuroAdapter):
    """prod OuroAdapter with the eager attention load (K2; the k16 loader k16_path_ouro.py:211-220).
    `load` is ported line by line from prod/models/ouro.py:271-300; the only change is
    attn_implementation="eager" (checked by the normalized-source test)."""

    def load(self):
        if DEV == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.mem_fraction)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # load the REVISION that `prod.install_models` pinned, not whatever the repo's `main`
        # points at today. `revision=None` (no pins file yet) loads `main`.
        rev = cfgmod.revision_for(self.name)
        tok = AutoTokenizer.from_pretrained(self.repo, trust_remote_code=True, revision=rev)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(
            self.repo, dtype=self.dtype, trust_remote_code=True, revision=rev,
            attn_implementation="eager").to(DEV).eval()
        merged = False
        if self.adapter_dir:
            from peft import PeftModel                     # s32_common.load_arm
            pm = PeftModel.from_pretrained(m, self.adapter_dir, is_trainable=False)
            m = pm.merge_and_unload().to(DEV).eval()
            merged = True
        patched, already = patch_universal_cache(m)
        self.tok = tok
        self.model = m
        self._info = {"cache_patched_now": bool(patched),
                      "cache_already_had_get_mask_sizes": bool(already),
                      "lora_merged": merged, "adapter_dir": self.adapter_dir,
                      "n_layers": int(m.config.num_hidden_layers),
                      "hidden_size": int(m.config.hidden_size),
                      "vocab_size": int(m.config.vocab_size),
                      "early_exit_threshold": getattr(m, "early_exit_threshold", None)}
        return self


def sample_seed(key: Any, abs_pos: int) -> int:
    """Seed of the draw for the token at absolute position `abs_pos` of the (problem, sample) key
    "row_idx|s": int(sha256(f"{key}|{abs_pos}")[:16], 16) mod 2^63. DESIGN 22.11 seeds one
    generator per key with int(sha256(key)[:16], 16) mod 2^63; adding the position makes every
    draw a pure function of (row, absolute position) as DESIGN 4.4 requires, so a second decode()
    call of the same chain (forced read-out, later wave, OOM retry) does not reuse the first
    call's uniforms [decision, core review G2-F5]. Common random numbers across arms are kept."""
    s = str(key)
    if "|" not in s:
        raise SteerConfigError("sampled decode needs row ids of the form 'row_idx|sample', got %r"
                               % (key,))
    blob = "%s|%d" % (s, int(abs_pos))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 63)


class SampledOuroAdapter(OuroAdapter):
    """prod OuroAdapter whose decode samples with temperature and top-p instead of the greedy
    argmax (DESIGN 22.11, descriptive AIME row). Row ids are "row_idx|sample" keys; the token at
    absolute position p of row r is drawn on a CPU generator seeded by sample_seed(r, p).
    Logits processors (LoopCD) act in
    the forward epilogue, before the temperature (2610.02185 App A.3)."""

    def __init__(self, name, temperature, top_p, **kw):
        super(SampledOuroAdapter, self).__init__(name, **kw)
        if not temperature > 0 or not 0 < top_p <= 1:
            raise SteerConfigError("temperature > 0 and 0 < top_p <= 1 are REQUIRED")
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def _sample(self, z: Tensor, row_ids: List[Any], abs_pos: List[int]) -> Tensor:
        """Temperature, then top-p as transformers TopPLogitsWarper (keep the smallest set whose
        mass reaches top_p, the most likely token always kept), then one multinomial draw per
        row on CPU float32 from a generator seeded by sample_seed(row, absolute position)."""
        out = []
        for b in range(z.shape[0]):
            s = z[b].float() / self.temperature
            srt, idx = torch.sort(s, descending=False)
            drop = srt.softmax(-1).cumsum(-1) <= (1 - self.top_p)
            drop[-1] = False
            s = s.masked_fill(drop.scatter(0, idx, drop), float("-inf"))
            p = s.softmax(-1).cpu()
            g = torch.Generator().manual_seed(sample_seed(row_ids[b], abs_pos[b]))
            out.append(int(torch.multinomial(p, 1, generator=g)))
        return torch.tensor(out, dtype=torch.long, device=z.device)

    @torch.no_grad()
    def decode(self, seqs, n_new, k, eos_ids, stop_strings=None, pieces=None, row_ids=None,
               stopper=None, wait_ids=None, states=None):
        # Port of prod/models/ouro.py:358-452; only the sampling lines differ (DESIGN 22.11).
        """s28_common.decode with s13_common's forced-continuation branch folded in.

        stopper is None -> plain greedy, a row stops at any id in eos_ids or a stop string
        stopper given   -> forced continuation: no row stops; a token that would stop is NOT
                           emitted, `wait_ids` are queued instead, so every row emits n_new tokens
        Returns (gen, seconds) when stopper is None, else (gen, states, seconds).

        The forced states passed in are never written through: decode works on copies and returns
        them, so a call that raises (the OOM the caller retries at a smaller batch) leaves the
        caller's own states exactly as they were. The caller keeps the returned states.
        """
        assert int(k) == self._k, "set_depth(%r) before decode" % (k,)
        m, tok = self.model, self.tok
        pieces = self._pieces if pieces is None else pieces
        dev = next(m.parameters()).device
        pad = tok.pad_token_id
        B = len(seqs)
        ids, attn, L = leftpad(seqs, pad, dev)
        cache = self._cache_factory(B, L + n_new)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        cp = torch.arange(L, device=dev)
        out = m(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache,
                use_cache=True, cache_position=cp, logits_to_keep=1)
        cur = self._sample(out.logits[:, -1, :].float(), row_ids, [len(s) for s in seqs])
        del out

        gen = [[] for _ in range(B)]
        if stopper is not None:
            # copy before the first token: every write below lands on the copies, which are what the
            # caller gets back on success (copy_forced_states says why)
            states = new_forced_states(B) if states is None else copy_forced_states(states)
        fin = [False] * B
        tails = [""] * B
        eset = set(eos_ids or [])
        t0 = time.time()
        t = 0
        while t < n_new:
            nxt = cur.tolist()
            feed = [pad] * B
            for b in range(B):
                if fin[b]:
                    continue
                x = nxt[b]
                if stopper is None:
                    gen[b].append(x)
                    feed[b] = x
                    if x in eset:
                        fin[b] = True
                        continue
                    if stop_strings:
                        tails[b] = (tails[b] + pieces[x])[-64:]
                        if any(s in tails[b] for s in stop_strings):
                            fin[b] = True
                    continue
                st = states[b]
                if st["pending"]:
                    x = st["pending"].pop(0)
                else:
                    tail_old = "".join(st["tail"])
                    reason = stopper(x, tail_old)
                    if reason is not None:
                        absp = st["off"] + len(gen[b])
                        if st["natural_stop_pos"] is None:
                            st["natural_stop_pos"] = absp
                            st["natural_stop_reason"] = reason
                        st["n_forced_continuations"] += 1
                        st["forced_positions"].append([absp, reason, x])
                        st["tail"].clear()
                        st["pending"] = list(wait_ids[1:])
                        x = wait_ids[0]
                gen[b].append(x)
                feed[b] = x
                st["tail"].append(pieces[x])
            t += 1
            if t >= n_new or all(fin):
                break
            cur_in = torch.tensor(feed, dtype=torch.long, device=dev).unsqueeze(1)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=dev)], 1)
            o = m(input_ids=cur_in, attention_mask=attn,
                  position_ids=attn.sum(-1, keepdim=True) - 1,
                  past_key_values=cache, use_cache=True,
                  cache_position=torch.tensor([L + t - 1], device=dev), logits_to_keep=1)
            cur = self._sample(o.logits[:, -1, :].float(), row_ids,
                               [len(s) + len(g) for s, g in zip(seqs, gen)])
            del o
        dt = time.time() - t0
        del cache, ids, attn, cur
        torch.cuda.empty_cache()
        if stopper is None:
            return gen, dt
        for b, st in enumerate(states):
            st["off"] += len(gen[b])
        return gen, states, dt

    def meta(self):
        d = super(SampledOuroAdapter, self).meta()
        d.update({"sampling": {"temperature": self.temperature, "top_p": self.top_p,
                               "seed_rule": "per token: int(sha256('row_idx|s|abs_pos')[:16], 16) "
                                            "mod 2^63",
                               "top_p_rule": "transformers TopPLogitsWarper"}})
        return d
