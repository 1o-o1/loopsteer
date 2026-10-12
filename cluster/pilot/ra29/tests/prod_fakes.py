"""CPU fakes for the prod.steer tests (DESIGN 15.1): tiny looped models with the structure of the
real checkpoints, toy tokenizers and minimal interventions.

- FakeOuro (module `steer_fake_ouro`): the OuroDecoderLayer body of modeling_ouro base 388-422, the
  loop of base 592-608 (`total_ut_steps`, `norm`, `early_exit_gate`), the exit read-out and traps of
  base 723-799, a UniversalTransformerCache prod's StaticUTCache subclasses, K/V at
  `current_ut * L + layer_idx`, eager (probabilities returned, float 0/min mask) and sdpa
  (None probabilities, optional mask-None skip) attention.
- FakeRaven(variant) (modules `steer_fake_raven_huginn` / `steer_fake_raven_mcleish`): the raven
  forward of huginn 636-779 / McLeish 624-768 with the gradient split of iterate_forward, the
  HuginnStaticCache "full" update of huginn 356-380 and a CausalSelfAttention prod's attention patch
  applies to (Wqkv, chunks, proj, the module-level rotary function of each family).
- FakeIfmRunner: the k10_ifm.IfmRunner surface (schedule, _layer, _init_state, fwd decorated
  no_grad, coda_lens, generate) over a fake xllm model, a twin (arch "transformer",
  loop_start_layer 0) and a Huginn-class variant, with explicit-softmax CausalSoftmaxAttention
  modules (backend None, bool `segments` mask), and `fake_xllm()` for the xllm imports the binding
  makes (DepthControlledHuginn.identity_recurrent_state, repeat_kv, apply_rotary_embeddings).
- CharTok (single characters, numbers split into digits) and ChunkTok (3-digit chunks).
- Interventions: AddEdit, PatchEdit, RandomKeyEdit, Knock and Skip, duck-typed on the registry
  contract of DESIGN 5 and 7, so these tests do not depend on interventions.py.

Fakes are float32 and seeded; every builder runs inside torch.random.fork_rng.
"""
import contextlib
import hashlib
import math
import os
import sys
import types
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("PROD_DEV", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
import torch.nn.functional as F                                       # noqa: E402
from torch.nn.attention.bias import causal_lower_right                # noqa: E402

D_MODEL, N_HEADS, HEAD_DIM, VOCAB = 16, 2, 8, 64


class Out(object):
    """Model output with settable attributes (logits replaced in place by the epilogue)."""

    def __init__(self, **kw: Any):
        self.__dict__.update(kw)


def _perturb_norms(model: nn.Module, scale: float = 0.1) -> None:
    for name, p in model.named_parameters():
        if "norm" in name and name.endswith("weight"):
            with torch.no_grad():
                p.add_(scale * torch.randn_like(p))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _cos_sin(position_ids: torch.Tensor, dim: int, base: float = 10000.0
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = position_ids[..., None].float() * inv
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


# ===================================================================================== FakeOuro
class FakeOuroConfig(object):
    def __init__(self, L: int, attn: str, total_ut_steps: int):
        self.num_hidden_layers = L
        self.num_attention_heads = N_HEADS
        self.num_key_value_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.hidden_size = D_MODEL
        self.intermediate_size = 32
        self.vocab_size = VOCAB
        self.rms_norm_eps = 1e-6
        self.attention_dropout = 0.0
        self.use_cache = False
        self.total_ut_steps = total_ut_steps
        self.layer_types = ["full_attention"] * L
        self._attn_implementation = attn


class UniversalTransformerCache(object):
    """modeling_ouro UniversalTransformerCache (base 107-205): one growing K/V entry per index."""

    def __init__(self, max_cache_size: Optional[int] = None):
        self.key_cache, self.value_cache, self.layers = [], [], []
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = key_states, value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states],
                                                    dim=2)
        self._seen_tokens = self.key_cache[layer_idx].shape[2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0


class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class OuroMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def eager_attention_forward(module, query, key, value, attention_mask, scaling):
    """transformers eager path: matmul, + mask, softmax in float32, probabilities returned."""
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sdpa_attention_forward(module, query, key, value, attention_mask, scaling):
    """transformers sdpa path: a bool mask (True = attend) or None (causal skip); no probabilities."""
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    if attention_mask is None:
        out = F.scaled_dot_product_attention(query, key_states, value_states,
                                             is_causal=query.shape[2] > 1, scale=scaling)
    else:
        out = F.scaled_dot_product_attention(query, key_states, value_states,
                                             attn_mask=attention_mask[:, :, :, :key_states.shape[-2]],
                                             scale=scaling)
    return out.transpose(1, 2).contiguous(), None


class OuroAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size,
                                bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None,
                cache_position=None, current_ut=0, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = query_states * cos[:, None] + _rotate_half(query_states) * sin[:, None]
        key_states = key_states * cos[:, None] + _rotate_half(key_states) * sin[:, None]
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states,
                current_ut * self.config.num_hidden_layers + self.layer_idx, cache_kwargs)
        fn = (eager_attention_forward if self.config._attn_implementation == "eager"
              else sdpa_attention_forward)
        attn_output, attn_weights = fn(self, query_states, key_states, value_states,
                                       attention_mask, self.scaling)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OuroDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OuroAttention(config=config, layer_idx=layer_idx)
        self.mlp = OuroMLP(config)
        self.input_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OuroRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim

    @torch.no_grad()
    def forward(self, x, position_ids):
        cos, sin = _cos_sin(position_ids, self.dim)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class OuroModel(nn.Module):
    def __init__(self, config, mask_none_when_unpadded=False):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OuroDecoderLayer(config, i)
                                     for i in range(config.num_hidden_layers)])
        self.norm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OuroRotaryEmbedding(config)
        self.total_ut_steps = config.total_ut_steps
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        self.mask_none_when_unpadded = mask_none_when_unpadded

    def causal_mask(self, input_embeds, attention_mask, cache_position, past_key_values):
        """create_causal_mask: eager -> float 4D 0/finfo.min [B, 1, T, KV]; sdpa -> bool 4D (fully
        masked rows opened, the torch fix) or None when unpadded and the switch is on."""
        B, T = input_embeds.shape[:2]
        past = past_key_values.get_seq_length(0) if past_key_values is not None else 0
        KV = past + T
        impl = self.config._attn_implementation
        if impl == "sdpa" and self.mask_none_when_unpadded and (
                attention_mask is None or bool(attention_mask.bool().all())):
            return None
        kpos = torch.arange(KV, device=input_embeds.device)
        allowed = (kpos[None, :] <= cache_position[:, None])[None].expand(B, T, KV)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, :KV].bool()
        allowed = allowed[:, None]
        if impl == "eager":
            neg = torch.finfo(input_embeds.dtype).min
            return torch.where(allowed, torch.zeros((), dtype=input_embeds.dtype),
                               torch.full((), neg, dtype=input_embeds.dtype))
        return allowed | ~allowed.any(-1, keepdim=True)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, cache_position=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = UniversalTransformerCache(
                self.config.num_hidden_layers * self.config.total_ut_steps)
        if cache_position is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past, past + inputs_embeds.shape[1],
                                          device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        mask = self.causal_mask(inputs_embeds, attention_mask, cache_position, past_key_values)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states_list = []
        gate_list = []
        for current_ut in range(self.total_ut_steps):
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    current_ut=current_ut,
                    **kwargs,
                )
            hidden_states = self.norm(hidden_states)
            hidden_states_list.append(hidden_states)
            gate_list.append(self.early_exit_gate(hidden_states))
        return (Out(last_hidden_state=hidden_states,
                    past_key_values=past_key_values if use_cache else None),
                hidden_states_list, gate_list)


class OuroForCausalLM(nn.Module):
    def __init__(self, config, mask_none_when_unpadded=False):
        super().__init__()
        self.config = config
        self.model = OuroModel(config, mask_none_when_unpadded)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.early_exit_step = None
        self.early_exit_threshold = None

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, cache_position=None,
                logits_to_keep=0, use_weighted_exit=False, exit_at_step=None, exit_threshold=None,
                **kwargs):
        """Exit selection of modeling_ouro base 723-799: labels -> exit-pdf mixture; a valid
        exit_at_step -> that exit; else the threshold branch; else the last hidden state."""
        exit_at_step = exit_at_step if exit_at_step is not None else self.early_exit_step
        exit_threshold = exit_threshold if exit_threshold is not None else self.early_exit_threshold
        outputs, hs, gates = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                        position_ids=position_ids, past_key_values=past_key_values,
                                        inputs_embeds=inputs_embeds, use_cache=use_cache,
                                        cache_position=cache_position, **kwargs)
        sl = slice(-logits_to_keep, None)
        pdf, rem = [], torch.ones_like(gates[0].squeeze(-1))
        for i, g in enumerate(gates):
            lam = torch.sigmoid(g.squeeze(-1))
            if i < len(gates) - 1:
                pdf.append(lam * rem)
                rem = rem * (1.0 - lam)
            else:
                pdf.append(rem)
        pdf = torch.stack(pdf, dim=2)
        if labels is not None:
            logits = sum(self.lm_head(h[:, sl]) * pdf[:, sl, i:i + 1] for i, h in enumerate(hs))
        elif exit_at_step is not None and 0 <= exit_at_step < len(hs):
            logits = self.lm_head(hs[exit_at_step][:, sl])
        elif exit_threshold is not None:
            mask = torch.cumsum(pdf, dim=2) >= exit_threshold
            steps = torch.argmax(mask.float(), dim=2)
            steps[~mask.any(dim=2)] = pdf.shape[2] - 1
            st = torch.stack(hs, dim=2)
            idx = steps[..., None, None].expand(-1, -1, 1, st.size(-1))
            logits = self.lm_head(torch.gather(st, 2, idx).squeeze(2)[:, sl])
        else:
            logits = self.lm_head(outputs.last_hidden_state[:, sl])
        return Out(loss=None, logits=logits, past_key_values=outputs.past_key_values)


_OURO_MOD = types.ModuleType("steer_fake_ouro")
for _c in (UniversalTransformerCache, OuroRMSNorm, OuroMLP, OuroAttention, OuroDecoderLayer,
           OuroRotaryEmbedding, OuroModel, OuroForCausalLM):
    _c.__module__ = "steer_fake_ouro"
    setattr(_OURO_MOD, _c.__name__, _c)
sys.modules["steer_fake_ouro"] = _OURO_MOD


def make_fake_ouro(attn: str = "eager", L: int = 2, total_ut_steps: int = 4, seed: int = 0,
                   mask_none_when_unpadded: bool = False) -> OuroForCausalLM:
    """FakeOuro: hidden 16, heads 2, head_dim 8, vocab 64, float32 (DESIGN 15.1)."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = OuroForCausalLM(FakeOuroConfig(L, attn, total_ut_steps), mask_none_when_unpadded)
        _perturb_norms(m)
    return m.eval()


def real_ouro_adapter_over(fake: OuroForCausalLM, k: int, cls: Any = None) -> Any:
    """A prod OuroAdapter('ouro_1_4b_base') (or `cls`) with model and tok set, without load()."""
    from prod.models.ouro import OuroAdapter, patch_universal_cache
    ad = (cls or OuroAdapter)("ouro_1_4b_base")
    ad.model = fake
    ad.tok = CharTok()
    patch_universal_cache(fake)
    ad.set_depth(k)
    return ad


# ==================================================================================== FakeRaven
class RavenConfig(object):
    def __init__(self, variant: str):
        self.n_embd = D_MODEL
        self.num_attention_heads = N_HEADS
        self.num_key_value_heads = N_HEADS if variant == "huginn" else 1
        self.n_layers_in_prelude = 1
        self.n_layers_in_recurrent_block = 2
        self.n_layers_in_coda = 1
        self.intermediate_size = 32
        self.norm_eps = 1e-6
        self.init_values = {"std": 0.5, "embed_scale": 2.0 if variant == "mcleish" else 1.0}
        self.qk_bias = variant == "huginn"
        self.bias = False
        self.block_size = 64
        self.rope_base = 10000.0
        self.vocab_size = VOCAB
        self.padded_vocab_size = VOCAB
        self.mean_recurrence = 4
        self.test_time_noise = 0
        self.test_time_noise_type = "fixed"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class HuginnStaticCache(object):
    """huginn 300-445: [steps, B, heads, max_len, hd] buffers; `update` index-copies at
    [_seen_tokens - n, _seen_tokens) and advances _seen_tokens only when step_idx == 0."""

    def __init__(self, max_length, max_num_steps, num_heads, hidden_dim, batch_size=1,
                 lookup_strategy="full", device=None, dtype=torch.float32):
        self._seen_tokens = 0
        self.max_length = max_length
        self.lookup_strategy = lookup_strategy
        self.max_num_steps = max_num_steps
        shape = (max_num_steps, batch_size, num_heads, max_length, hidden_dim)
        self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.valid_mask = torch.zeros((max_num_steps, max_length), dtype=torch.bool, device=device)

    def update(self, key_states, value_states, step_idx, lookup_strategy=None):
        if step_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        start_idx = self._seen_tokens - key_states.shape[-2]
        indices = torch.arange(start_idx, start_idx + key_states.shape[-2],
                               device=key_states.device)
        self.key_cache[step_idx].index_copy_(2, indices, key_states)
        self.value_cache[step_idx].index_copy_(2, indices, value_states)
        self.valid_mask[step_idx, start_idx: start_idx + key_states.shape[-2]] = True
        return (self.key_cache[step_idx, :, :, : self._seen_tokens],
                self.value_cache[step_idx, :, :, : self._seen_tokens])

    def reset(self):
        self._seen_tokens = 0
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.valid_mask.zero_()

    def get_seq_length(self, step_idx=0):
        return self._seen_tokens


def apply_rotary_emb_complex_like(q, k, freqs_cis):
    """Huginn-style rotary before the head transpose: q, k [B, S, H, hd]; freqs_cis
    [1, S, hd/2, 2] holds (cos, sin) per pair."""
    def rot(x):
        xr = x.float().reshape(*x.shape[:-1], -1, 2)
        c, s = freqs_cis[..., 0].unsqueeze(2), freqs_cis[..., 1].unsqueeze(2)
        out = torch.stack([xr[..., 0] * c - xr[..., 1] * s, xr[..., 0] * s + xr[..., 1] * c], -1)
        return out.flatten(-2).type_as(x)
    return rot(q), rot(k)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Llama rotary after the head transpose: q, k [B, H, S, hd]; cos, sin [1, S, hd]."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class _RavenAttention(nn.Module):
    """Shared __init__ of the two attention classes. prod's install_attention_patch sets class
    attributes, so each family's class must not inherit the other's patch flag."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head
        shape = (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.chunks = [config.n_embd, self.n_kv_heads * self.head_dim,
                       self.n_kv_heads * self.head_dim]
        self.Wqkv = nn.Linear(config.n_embd, shape, bias=False)
        if config.qk_bias:
            self.qk_bias = nn.Parameter(0.1 * torch.randn(2, 1, self.n_head, self.head_dim))
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)


class HuginnCausalSelfAttention(_RavenAttention):
    def forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
        B, S, E = x.shape
        q, k, v = self.Wqkv(x).split(self.chunks, dim=2)
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, block_idx)
        if q.shape[2] < k.shape[2]:
            if q.shape[2] > 1:
                bias = causal_lower_right(q.shape[2], k.shape[2])
                y = F.scaled_dot_product_attention(q, k, v, bias, dropout_p=0.0)
            else:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).reshape(B, S, E).contiguous()
        return self.proj(y)


class McLeishCausalSelfAttention(_RavenAttention):
    def forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
        B, S, E = x.shape
        q, k, v = self.Wqkv(x).split(self.chunks, dim=2)
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = freqs_cis
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, block_idx)
        if q.shape[2] < k.shape[2]:
            if q.shape[2] > 1:
                bias = causal_lower_right(q.shape[2], k.shape[2])
                y = F.scaled_dot_product_attention(q, k, v, bias, dropout_p=0.0, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                                   enable_gqa=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True,
                                               enable_gqa=True)
        y = y.transpose(1, 2).reshape(B, S, E).contiguous()
        return self.proj(y)


class GatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size * 2, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.nonlin = nn.SiLU()

    def forward(self, x):
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        return self.proj(self.nonlin(x_fc_1) * x_fc_2)


class HuginnSandwichBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = HuginnCausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = GatedMLP(config)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.layer_id = layer_id

    def forward(self, x, freqs_cis, step_idx, mask=None, past_key_values=None):
        attn_out = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values)
        x = self.norm_2(attn_out + x)
        x = self.norm_4(self.mlp(self.norm_3(x)) + x)
        return x


class McLeishSandwichBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = McLeishCausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = GatedMLP(config)
        self.layer_id = layer_id

    def forward(self, x, freqs_cis, step_idx, mask=None, past_key_values=None):
        attn_out = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values)
        x = attn_out + x
        x = self.mlp(self.norm_2(x)) + x
        return x


class McLeishRotary(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd // config.num_attention_heads

    @torch.no_grad()
    def forward(self, x, position_ids):
        cos, sin = _cos_sin(position_ids, self.dim)
        return cos.to(x.dtype), sin.to(x.dtype)


class RavenForCausalLM(nn.Module):
    """huginn 540-779 / McLeish 472-768 with the variant switches named in DESIGN 3.2, 3.3."""

    def __init__(self, config, variant):
        super().__init__()
        self.config = config
        self.variant = variant
        blk = HuginnSandwichBlock if variant == "huginn" else McLeishSandwichBlock
        P, Cc, Cd = (config.n_layers_in_prelude, config.n_layers_in_recurrent_block,
                     config.n_layers_in_coda)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
            prelude=nn.ModuleList(blk(config, i) for i in range(P)),
            adapter=nn.Linear(config.n_embd * 2, config.n_embd, bias=config.bias),
            core_block=nn.ModuleList(blk(config, P + i) for i in range(Cc)),
            coda=nn.ModuleList(blk(config, P + Cc * config.mean_recurrence + i)
                               for i in range(Cd)),
            ln_f=RMSNorm(config.n_embd, eps=config.norm_eps)))
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.emb_scale = config.init_values["embed_scale"]
        hd = config.n_embd // config.num_attention_heads
        if variant == "huginn":
            cos, sin = _cos_sin(torch.arange(config.block_size), hd)
            self.register_buffer("freqs_cis", torch.stack([cos[:, :hd // 2], sin[:, :hd // 2]],
                                                          dim=-1)[None], persistent=True)
        else:
            self.rotary_emb = McLeishRotary(config)
        self.gradient_checkpointing = False

    def forward(self, input_ids, input_embeds=None, input_states=None, attention_mask=None,
                position_ids=None, labels=None, num_steps=None, past_key_values=None,
                use_cache=False, cache_position=None, init_scale=1.0, **kwargs):
        if self.variant == "huginn":
            if position_ids is None and cache_position is None:
                freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
            elif position_ids is not None:
                freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
            else:
                freqs_cis = self.freqs_cis[:, cache_position]
        else:
            if position_ids is None and cache_position is None:
                position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            elif cache_position is not None:
                position_ids = cache_position.unsqueeze(0)
        if input_embeds is None:
            input_embeds = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale
        if self.variant == "mcleish":
            freqs_cis = self.rotary_emb(input_embeds, position_ids)
        prepared_attn_mask = None
        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
        for block in self.transformer.prelude:
            block_idx += 1
            input_embeds = block(input_embeds, freqs_cis, block_idx, prepared_attn_mask,
                                 past_key_values)
        x, _ng, _wg, _xk, block_idx = self.iterate_forward(
            input_embeds, input_states, freqs_cis, block_idx, prepared_attn_mask, past_key_values,
            num_steps, init_scale)
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in self.transformer.coda:
            block_idx -= 1
            x = block(x, freqs_cis, block_idx, prepared_attn_mask, past_key_values)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if self.variant == "huginn":
            logits = logits.float()
        return Out(logits=logits, past_key_values=past_key_values)

    def iterate_forward(self, input_embeds, input_states, freqs_cis, block_idx, mask,
                        past_key_values=None, num_steps=None, init_scale=1.0):
        x = xk = (self.initialize_state(input_embeds, scale=init_scale) if input_states is None
                  else input_states.clone())
        if hasattr(num_steps, "__len__") and len(num_steps) > 1:
            num_steps_no_grad, num_steps_with_grad = num_steps
        else:
            num_steps_no_grad, num_steps_with_grad = num_steps, torch.tensor(0)
        with torch.no_grad():
            for no_grad_step in range(num_steps_no_grad):
                xk = x
                x, block_idx = self.core_block_forward(xk, input_embeds, freqs_cis, mask,
                                                       past_key_values, block_idx, no_grad_step)
        for grad_step in range(num_steps_with_grad):
            xk = x
            step = num_steps_no_grad + grad_step
            if self.variant == "huginn":
                x, block_idx = self._maybe_checkpoint_core_block(
                    xk, input_embeds, freqs_cis, mask, past_key_values, block_idx, step)
            else:
                x, block_idx = self.core_block_forward(xk, input_embeds, freqs_cis, mask,
                                                       past_key_values, block_idx, step)
        out = self.transformer.ln_f(x) if self.variant == "huginn" else x
        return out, num_steps_no_grad, num_steps_with_grad, xk.detach(), block_idx

    def _maybe_checkpoint_core_block(self, *args):
        if self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.core_block_forward, *args,
                                                     use_reentrant=False)
        return self.core_block_forward(*args)

    def core_block_forward(self, x, input_embeds, freqs_cis, mask, past_key_values, block_idx,
                           current_step):
        block_idx = block_idx.detach().clone()
        x = self._maybe_inject_noise(x, current_step)
        x = self.transformer.adapter(torch.cat([x, input_embeds.to(x.device)], dim=-1))
        for block in self.transformer.core_block:
            block_idx += 1
            x = block(x, freqs_cis, block_idx, mask, past_key_values)
        return x, block_idx

    def initialize_state(self, input_embeds, scale=1.0):
        x = torch.randn_like(input_embeds)
        std = self.config.init_values["std"] * scale
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        return x

    def _maybe_inject_noise(self, x, current_step):
        if self.config.test_time_noise > 0:
            raise NotImplementedError("the fake runs at test_time_noise = 0 (both checkpoints)")
        return x


def _raven_module(variant: str) -> types.ModuleType:
    name = "steer_fake_raven_" + variant
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    att = HuginnCausalSelfAttention if variant == "huginn" else McLeishCausalSelfAttention
    blk = HuginnSandwichBlock if variant == "huginn" else McLeishSandwichBlock
    cls = type("RavenForCausalLM", (RavenForCausalLM,), {"__module__": name})
    for obj, nm in ((att, "CausalSelfAttention"), (blk, "SandwichBlock"), (cls, "RavenForCausalLM"),
                    (HuginnStaticCache, "HuginnStaticCache"), (RMSNorm, "RMSNorm")):
        setattr(mod, nm, obj)
    if variant == "huginn":
        mod.apply_rotary_emb_complex_like = apply_rotary_emb_complex_like
    else:
        mod.apply_rotary_pos_emb = apply_rotary_pos_emb
    sys.modules[name] = mod
    return mod


def make_fake_raven(variant: str = "huginn", seed: int = 0) -> RavenForCausalLM:
    """FakeRaven: P 1, C 2, Cd 1, d 16, vocab 64, float32 (DESIGN 15.1)."""
    mod = _raven_module(variant)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = mod.RavenForCausalLM(RavenConfig(variant), variant)
        _perturb_norms(m)
    return m.eval()


def real_raven_adapter_over(fake: RavenForCausalLM, k: int) -> Any:
    """prod HuginnAdapter / McLeishAdapter with model and tok set after install_attention_patch;
    pad id 0 and stop id 1 fit the fake vocabulary."""
    from prod.models.huginn import HuginnAdapter
    from prod.models.mcleish import McLeishAdapter
    from prod.models.raven import install_attention_patch
    ad = HuginnAdapter() if fake.variant == "huginn" else McLeishAdapter()
    ad.model = fake
    ad.tok = CharTok()
    ad._pad_id, ad._stop_ids = 0, [1]
    install_attention_patch(fake)
    ad.max_positions = fake.config.block_size if fake.variant == "huginn" else None
    ad.dtype = torch.float32
    ad.set_depth(k)
    return ad


@contextlib.contextmanager
def raven_cpu() -> Iterator[None]:
    """prod.models.raven.DEV = 'cpu' for decode (StateStream reads it at call time), restored."""
    import prod.models.raven as R
    old = R.DEV
    R.DEV = "cpu"
    try:
        yield
    finally:
        R.DEV = old


# ================================================================================ FakeIfmRunner
def apply_rotary_embeddings(xq, xk, freqs_cis, backward=False):
    """xllm-style rotary on [B, S, H, hd] with freqs [S, hd/2, 2] (cos, sin)."""
    def rot(x):
        xr = x.float().reshape(*x.shape[:-1], -1, 2)
        c, s = freqs_cis[None, :, None, :, 0], freqs_cis[None, :, None, :, 1]
        out = torch.stack([xr[..., 0] * c - xr[..., 1] * s, xr[..., 0] * s + xr[..., 1] * c], -1)
        return out.flatten(-2).type_as(x)
    return rot(xq), rot(xk)


def repeat_kv(x, n_rep):
    """xllm repeat_kv on [B, S, H, hd]."""
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x[:, :, :, None].expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)


class FakeKV(object):
    """k10_ifm.FastKV: one preallocated buffer per schedule slot."""

    def __init__(self, B, maxlen, kvh, d, dtype, device):
        self.k = torch.empty((B, maxlen, kvh, d), dtype=dtype, device=device)
        self.v = torch.empty((B, maxlen, kvh, d), dtype=dtype, device=device)
        self.n = 0


class CausalSoftmaxAttention(nn.Module):
    """Backend-None explicit softmax with a bool `segments` mask (True = blocked) OR-ed with the
    causal mask (the ops k16_path_ifm.attn_recompute ports); a FakeKV cache takes the
    install_fast_attention path (causal at prefill from 0, no mask for one token)."""

    def __init__(self, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.rope_head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.backend = None

    def forward(self, xq, xk, xv, g, freqs_cis, segments, fp32_attn_output=False,
                deterministic=True, cache=None):
        bs, slen, _ = xq.shape
        q = xq.view(bs, slen, self.n_heads, self.head_dim)
        k = xk.view(bs, slen, self.n_kv_heads, self.head_dim)
        v = xv.view(bs, slen, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_embeddings(q, k, freqs_cis=freqs_cis)
        if cache is not None:
            assert segments is None
            n0 = cache.n
            cache.k[:, n0:n0 + slen], cache.v[:, n0:n0 + slen] = k, v
            cache.n = n0 + slen
            qq = q.transpose(1, 2)
            kk = cache.k[:, :cache.n].transpose(1, 2)
            vv = cache.v[:, :cache.n].transpose(1, 2)
            out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=slen > 1, scale=self.scale,
                                                 enable_gqa=True)
            return out.transpose(1, 2).reshape(bs, slen, -1), cache
        n_rep = self.n_heads // self.n_kv_heads
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1), v.permute(0, 2, 1, 3)
        mask = torch.triu(torch.full((bs, slen, slen), float("-inf")), diagonal=1).type_as(q)
        if segments is not None:
            mask = mask.masked_fill(segments, value=float("-inf"))
        scores = torch.matmul(q, k) * self.scale + mask.unsqueeze(1)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q)
        out = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(bs, slen, -1)
        return out, cache


class FakeIfmAttention(nn.Module):
    def __init__(self, d, n_heads, head_dim):
        super().__init__()
        self.wq = nn.Linear(d, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d, bias=False)
        self.causal_attention = CausalSoftmaxAttention(n_heads, n_heads, head_dim)


class FakeIfmLayer(nn.Module):
    """Pre-norm block with the xllm layer signature (h, freq, segments, ...) -> (h, aux, cache)."""

    def __init__(self, d):
        super().__init__()
        self.attention_norm = RMSNorm(d)
        self.attention = FakeIfmAttention(d, N_HEADS, HEAD_DIM)
        self.ffn_norm = RMSNorm(d)
        self.ffn = GatedMLP(types.SimpleNamespace(n_embd=d, intermediate_size=32))

    def forward(self, h, freq, segments, moe_router_load_balancing_type=None,
                fp32_attn_output=False, deterministic=True, cache=None):
        a = self.attention
        x = self.attention_norm(h)
        o, cache = a.causal_attention(a.wq(x), a.wk(x), a.wv(x), None, freq, segments,
                                      fp32_attn_output, deterministic, cache)
        h = h + a.wo(o)
        h = h + self.ffn(self.ffn_norm(h))
        return h, None, cache


class FakeRope(object):
    def __init__(self, hd):
        self.hd = hd

    def get_freqs_cis(self, start, end, device):
        cos, sin = _cos_sin(torch.arange(start, end), self.hd)
        return torch.stack([cos[:, :self.hd // 2], sin[:, :self.hd // 2]], dim=-1).to(device)


class FakeInjection(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.adapter = nn.Linear(2 * d, d, bias=False)

    def forward(self, state, emb):
        return self.adapter(torch.cat([state, emb], dim=-1))


class FakeOutputLayer(nn.Module):
    """xllm TransformerOutputLayer: final_norm, then output, float logits."""

    def __init__(self, d, vocab):
        super().__init__()
        self.final_norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab, bias=False)

    def forward(self, x, y, mask=None):
        return self.output(self.final_norm(x)).float()


class FakeIfmModel(nn.Module):
    def __init__(self, arch, P, Cc, Cd, K, inj, state_init, exit_norm):
        super().__init__()
        d = D_MODEL
        self.model_dim = d
        self.embed = nn.Embedding(VOCAB, d)
        self.layers = nn.ModuleList(FakeIfmLayer(d) for _ in range(P + Cc + Cd))
        self.rope = FakeRope(HEAD_DIM)
        self.output = FakeOutputLayer(d, VOCAB)
        self.loop_start_layer, self.loop_end_layers, self.num_layers = P, P + Cc, P + Cc + Cd
        self.loop_times = K
        self.huginn_state_init = state_init if arch == "huginn" else None
        self.input_injection = FakeInjection(d) if (inj and arch == "huginn") else None
        self.loop_input_injection_module = (FakeInjection(d) if (inj and arch == "transformer")
                                            else None)
        self.huginn_recurrent_exit_norm_module = RMSNorm(d) if exit_norm else None
        self.dense_prelude_input_injection_module = None
        self.dense_prelude_input_layers = []
        self.dense_prelude_source_layer = -1
        self.training = False

    def _recurrence_input(self, prelude_output):
        return prelude_output

    def initialize_recurrent_state(self, input_embeds, cache_len=0):
        """xllm looped.py:481-497 (eval branch)."""
        if self.huginn_state_init == "zero":
            return torch.zeros_like(input_embeds)
        if self.huginn_state_init == "input":
            return input_embeds
        batch = torch.arange(input_embeds.shape[0], dtype=torch.float32)
        return self._hashed_normal_state(input_embeds, cache_len, batch.view(-1, 1, 1) * 0.037719,
                                         math.sqrt(2.0 / self.model_dim))

    def _hashed_normal_state(self, input_embeds, cache_len, batch_term, scale):
        """xllm looped.py:499-527 (model-parallel rank 0)."""
        b, s, h = input_embeds.shape
        dim = torch.arange(h, dtype=torch.float32).view(1, 1, -1)
        pos = torch.arange(cache_len, cache_len + s, dtype=torch.float32).view(1, -1, 1)
        st = dim.expand(b, s, h).clone()
        st.mul_(0.013579).add_(pos * 0.021713).add_(batch_term).add_(0.123457)
        st.sin_().mul_(43758.5453123)
        st.sub_(st.floor()).clamp_(1e-6, 1.0 - 1e-6).mul_(2.0).sub_(1.0).erfinv_().mul_(scale)
        return st.to(dtype=input_embeds.dtype)


class DepthControlledHuginn(object):
    """Stand-in of xllm.models.looped.DepthControlledHuginn for its unbound
    identity_recurrent_state (looped.py:750-765)."""

    def identity_recurrent_state(self, input_embeds, cache_len, example_ids, seed):
        modulus = 16777213
        identity = torch.remainder(example_ids.to(dtype=torch.long), modulus)
        mixed = torch.remainder(identity * 104729 + (int(seed) % modulus) * 130363, modulus)
        phase = mixed.to(dtype=torch.float32).mul_(2.0 * math.pi / modulus).view(-1, 1, 1)
        return self._hashed_normal_state(input_embeds, cache_len, phase,
                                         math.sqrt(2.0) * (1.0 / (self.model_dim ** 0.5)))


@contextlib.contextmanager
def fake_xllm() -> Iterator[None]:
    """Register the xllm modules the IFM binding imports (looped, causal_attention, rotary)."""
    names = ["xllm", "xllm.models", "xllm.models.looped", "xllm.modules",
             "xllm.modules.causal_attention", "xllm.modules.rotary_positional_embedding"]
    saved = {n: sys.modules.get(n) for n in names}
    mods = {n: types.ModuleType(n) for n in names}
    mods["xllm.models.looped"].DepthControlledHuginn = DepthControlledHuginn
    mods["xllm.modules.causal_attention"].repeat_kv = repeat_kv
    mods["xllm.modules.rotary_positional_embedding"].apply_rotary_embeddings = (
        apply_rotary_embeddings)
    try:
        sys.modules.update(mods)
        yield
    finally:
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


class FakeIfmRunner(object):
    """k10_ifm.IfmRunner over FakeIfmModel; method bodies follow k10_ifm.py:191-362."""
    family = "ifm"

    def __init__(self, arch="huginn", P=1, Cc=2, Cd=1, K=2, inj=True, state_init="input",
                 exit_norm=False, seed=0):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.m = FakeIfmModel(arch, P, Cc, Cd, K, inj, state_init, exit_norm).eval()
            _perturb_norms(self.m)
        self.arch = arch
        self.tok = CharTok()
        self.dev = torch.device("cpu")
        self.K_trained = K
        self.k = None
        self.pieces = self.tok.batch_decode([[i] for i in range(VOCAB)])
        self.init_mode = "identity" if self.m.huginn_state_init == "normal" else "native"
        self.n_calls = 0

    def set_depth(self, k):
        self.k = int(k)

    def width(self, k, max_seq, cap=16):
        return cap

    def schedule(self, k):
        m = self.m
        out = []
        s, e, L = int(m.loop_start_layer), int(m.loop_end_layers), int(m.num_layers)
        for i in range(s):
            out.append(("prelude", i))
        for j in range(1, k + 1):
            for i in range(s, e):
                out.append(("loop%d" % j, i))
        for i in range(e, L):
            out.append(("coda", i))
        return [(n, lid, tag) for n, (tag, lid) in enumerate(out)]

    def _layer(self, lid, h, freq, cache, slot):
        c = None
        if cache is not None:
            c = cache.get(slot)
            if c is None:
                ca = self.m.layers[lid].attention.causal_attention
                c = FakeKV(h.shape[0], 256, ca.n_kv_heads, ca.head_dim, h.dtype, h.device)
        h, _aux, nc = self.m.layers[lid](h, freq, None, moe_router_load_balancing_type=None,
                                         fp32_attn_output=False, deterministic=True, cache=c)
        if cache is not None:
            cache[slot] = nc
        return h

    def _init_state(self, input_embeds, cache_len, example_ids, init):
        m = self.m
        if m.huginn_state_init == "normal" and init == "identity":
            from xllm.models.looped import DepthControlledHuginn as DCH
            return DCH.identity_recurrent_state(m, input_embeds, cache_len, example_ids, 0)
        return m.initialize_recurrent_state(input_embeds, cache_len)

    @torch.no_grad()
    def fwd(self, tokens, k, cache=None, cache_len=0, example_ids=None, record=False, init=None,
            all_logits=False):
        m = self.m
        init = init or self.init_mode
        B, S = tokens.shape
        if example_ids is None:
            example_ids = torch.arange(B, device=tokens.device)
        hidden = m.embed(tokens)
        freq = m.rope.get_freqs_cis(cache_len, cache_len + S, hidden.device)
        self.n_calls = 0
        sched = self.schedule(k)
        if self.arch == "huginn":
            slot = 0
            for i in range(int(m.loop_start_layer)):
                hidden = self._layer(i, hidden, freq, cache, slot)
                slot += 1
            input_embeds = m._recurrence_input(hidden)
            state = self._init_state(input_embeds, cache_len, example_ids, init)
            exit_norm = m.huginn_recurrent_exit_norm_module
            for j in range(k):
                if m.input_injection is not None:
                    state = m.input_injection(state, input_embeds)
                for i in range(m.loop_start_layer, m.loop_end_layers):
                    state = self._layer(i, state, freq, cache, slot)
                    slot += 1
                if exit_norm is not None:
                    state = exit_norm(state)
                self.n_calls += 1
            hidden = state
            for i in range(m.loop_end_layers, m.num_layers):
                hidden = self._layer(i, hidden, freq, cache, slot)
                slot += 1
        else:
            emb = hidden
            inj = m.loop_input_injection_module
            for slot, lid, tag in sched:
                if inj is not None and lid == m.loop_start_layer and tag.startswith("loop"):
                    hidden = inj(hidden, emb)
                hidden = self._layer(lid, hidden, freq, cache, slot)
                if tag.startswith("loop") and lid == m.loop_end_layers - 1:
                    self.n_calls += 1
        h = hidden if all_logits else hidden[:, -1:]
        return m.output(h, None, None), None

    def coda_lens(self, state, freq):
        m = self.m
        h = state
        for i in range(m.loop_end_layers, m.num_layers):
            h, _a, _c = m.layers[i](h, freq, None, moe_router_load_balancing_type=None,
                                    fp32_attn_output=False, deterministic=True, cache=None)
        return m.output(h, None, None)

    @torch.no_grad()
    def generate(self, seqs, n_new, eos_ids, stop_strings, row_ids):
        B = len(seqs)
        lens = [len(s) for s in seqs]
        start, end = min(lens), max(lens) + n_new
        tokens = torch.zeros((B, end), dtype=torch.long)
        for b, s in enumerate(seqs):
            tokens[b, :len(s)] = torch.tensor(s, dtype=torch.long)
        ex = torch.tensor(list(row_ids), dtype=torch.long)
        cache = {}
        logits, _ = self.fwd(tokens[:, :start], self.k, cache=cache, cache_len=0, example_ids=ex)
        eos = set(eos_ids or [])
        gen = [[] for _ in range(B)]
        fin = [False] * B
        for cur in range(start, end):
            nxt = logits[:, -1].argmax(-1).tolist()
            col = tokens[:, cur].tolist()
            for b in range(B):
                if cur < lens[b]:
                    continue
                if fin[b]:
                    col[b] = 0
                    continue
                x = int(nxt[b])
                col[b] = x
                gen[b].append(x)
                if x in eos or len(gen[b]) >= n_new:
                    fin[b] = True
            tokens[:, cur] = torch.tensor(col, dtype=torch.long)
            if cur == end - 1 or (all(fin) and cur >= max(lens) - 1):
                break
            logits, _ = self.fwd(tokens[:, cur:cur + 1], self.k, cache=cache, cache_len=cur,
                                 example_ids=ex)
        return gen


# =================================================================================== tokenizers
_CHARS = (" 0123456789abcdefghijklmnopqrstuvwxyz+-=.,:?\n#ABCDEFGHIJKLMNO")


class CharTok(object):
    """One id per character (numbers split into digits); 0 pad, 1 eos, 2 bos; vocab 64."""
    pad_token_id, eos_token_id, bos_token_id = 0, 1, 2
    pad_token, eos_token, bos_token = "<pad>", "<eos>", "<bos>"
    vocab_size = VOCAB

    def __init__(self):
        self.id_of = {c: i + 3 for i, c in enumerate(_CHARS)}
        self.piece = {i: c for c, i in self.id_of.items()}
        self.piece.update({0: "", 1: "", 2: ""})
        self.special = {self.pad_token: 0, self.eos_token: 1, self.bos_token: 2}

    def _split(self, text: str) -> List[Tuple[str, int, int]]:
        return [(c, i, i + 1) for i, c in enumerate(text)]

    def _ids(self, pieces: List[Tuple[str, int, int]]) -> List[int]:
        return [self.id_of.get(p, self.id_of["?"]) for p, _a, _b in pieces]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kw):
        pieces = self._split(text)
        ids = self._ids(pieces)
        offs = [(a, b) for _p, a, b in pieces]
        if add_special_tokens:
            ids, offs = [self.bos_token_id] + ids, [(0, 0)] + offs
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids, clean_up_tokenization_spaces=False, skip_special_tokens=False):
        return "".join(self.piece.get(int(i), "") for i in ids)

    def batch_decode(self, seqs, clean_up_tokenization_spaces=False, skip_special_tokens=False):
        return [self.decode(s) for s in seqs]

    def convert_tokens_to_ids(self, tok):
        if tok in self.special:
            return self.special[tok]
        return self.id_of.get(tok)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        text = "".join("%s: %s\n" % (m["role"], m["content"]) for m in messages)
        text = text + ("assistant: " if add_generation_prompt else "")
        return self(text)["input_ids"] if tokenize else text


class ChunkTok(CharTok):
    """CharTok with every number split into 3-digit chunks from the left (DESIGN 15.1, DR17,
    DR21); chunk ids are 64 + offset(len) + int(chunk), offsets 0 / 10 / 110."""
    vocab_size = VOCAB + 1110
    _OFF = {1: 0, 2: 10, 3: 110}

    def _split(self, text: str) -> List[Tuple[str, int, int]]:
        out, i = [], 0
        while i < len(text):
            if text[i].isdigit():
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                for a in range(i, j, 3):
                    out.append((text[a:min(a + 3, j)], a, min(a + 3, j)))
                i = j
            else:
                out.append((text[i], i, i + 1))
                i += 1
        return out

    def _ids(self, pieces: List[Tuple[str, int, int]]) -> List[int]:
        ids = []
        for p, _a, _b in pieces:
            if p.isdigit():
                ids.append(VOCAB + self._OFF[len(p)] + int(p))
            else:
                ids.append(self.id_of.get(p, self.id_of["?"]))
        return ids

    def decode(self, ids, clean_up_tokenization_spaces=False, skip_special_tokens=False):
        out = []
        for i in ids:
            i = int(i)
            if i >= VOCAB:
                r = i - VOCAB
                n = 3 if r >= 110 else (2 if r >= 10 else 1)
                out.append(str(r - self._OFF[n]).zfill(n))
            else:
                out.append(self.piece.get(i, ""))
        return "".join(out)


# ================================================================================ interventions
class _Iv(object):
    """Registry-contract intervention base (DESIGN 5, 7): id, op, sites, positions, rows."""
    op = "add"

    def __init__(self, ident, sites, positions, rows=None):
        self.id = ident
        self.sites = tuple(sites)
        self.positions = positions
        self.rows = None if rows is None else frozenset(rows)
        self.recorders = ()

    def knock(self, site, row, ctx):
        return None


class AddEdit(_Iv):
    """h <- h + delta (delta broadcast over the selected positions)."""

    def __init__(self, ident, sites, positions, delta, rows=None):
        super().__init__(ident, sites, positions, rows)
        self.delta = delta

    def edit(self, x, site, row, h_pos, ctx):
        return x + self.delta.to(x).reshape(1, *x.shape[1:]) if self.delta is not None else x


class PatchEdit(_Iv):
    """h <- lerp(h, t, a) with t = row.payload[id][site key] aligned to the selected positions."""
    op = "patch"

    def __init__(self, ident, sites, positions, a=1.0, rows=None):
        super().__init__(ident, sites, positions, rows)
        self.a = float(a)

    def edit(self, x, site, row, h_pos, ctx):
        t = row.payload[self.id][site.key()].to(x)
        return torch.lerp(x, t, self.a) if self.a != 0 else x


class RandomKeyEdit(_Iv):
    """h <- h + scale * u, u a unit Gaussian per (row, site, position) from a CPU generator."""
    op = "random"

    def __init__(self, ident, sites, positions, scale=1.0, seed=0):
        super().__init__(ident, sites, positions)
        self.scale, self.seed = float(scale), int(seed)

    def edit(self, x, site, row, h_pos, ctx):
        out = []
        for i, p in enumerate(h_pos.tolist()):
            hsh = int(hashlib.sha256(("%s|%s|%d" % (row.key, site.key(), p)).encode()).hexdigest()
                      [:8], 16)
            g = torch.Generator().manual_seed((self.seed + 1000003 * hsh) % (2 ** 63))
            u = torch.randn(x.shape[1:], generator=g)
            out.append(x[i] + self.scale * u / u.norm())
        return torch.stack(out)


class Knock(_Iv):
    """Knockout at block calls named by sites: queries and keys are absolute position lists, or
    callables row -> list."""
    op = "knockout"

    def __init__(self, ident, calls, queries, keys, rows=None):
        super().__init__(ident, calls, None, rows)
        self.q, self.k = queries, keys
        self._calls = {(s.stage, s.loop, s.layer) for s in calls}

    def knock(self, site, row, ctx):
        if (site.stage, site.loop, site.layer) not in self._calls:
            return None
        q = self.q(row) if callable(self.q) else self.q
        k = self.k(row) if callable(self.k) else self.k
        return torch.tensor(q, dtype=torch.long), torch.tensor(k, dtype=torch.long)

    def edit(self, x, site, row, h_pos, ctx):
        return x


class Skip(_Iv):
    op = "skip"

    def __init__(self, ident, calls):
        super().__init__(ident, calls, None)

    def edit(self, x, site, row, h_pos, ctx):
        return x


def tensors_equal(a: Dict[Any, torch.Tensor], b: Dict[Any, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
