"""Batched left-padded decoding for the raven family (Huginn-0125 and McLeish Recurrent-Llama-3.2).

Why this file exists. Both checkpoints run `prepared_attn_mask = None` in `RavenForCausalLM.forward`
(Huginn line 674, McLeish line 660: the `compile_mask(...)` call is commented out), so the stock
forward applies NO padding mask and a padded batch silently attends to pad tokens. Both spikes
therefore decoded at batch 1 (`s9c_common.decode_forced`, one sequence; `s9f_common.greedy`, which
asserts equal lengths), at 18 tok/s down to 1.9 tok/s on the Spark.

The design, and why it is exact up to floating point.

1. POSITIONS. Huginn selects its rotary frequencies with
   `self.freqs_cis.index_select(1, position_ids.squeeze())`, which takes a 1-D index only, so
   per-row position ids are impossible on that checkpoint. Both checkpoints do accept
   `cache_position`, a shared 1-D absolute position vector, which is what the spikes pass. So every
   row in a left-padded batch is given the SAME absolute positions: 0..L-1 at prefill and L+t while
   decoding. RoPE's contribution to the score of two tokens depends only on their position
   DIFFERENCE, and each row's real tokens are contiguous (all padding is on the left), so the
   attention scores among a row's real tokens are exactly those of the unpadded computation. Only
   bf16 rounding differs.

2. MASK. `CausalSelfAttention.forward` is replaced on the dynamically loaded module class. When no
   bias is installed the replacement calls the checkpoint's ORIGINAL forward, unchanged, so batch-1
   and equal-length batches run stock model code bit-for-bit. When a bias IS installed it runs a
   copy of that checkpoint's own body with exactly one change: SDPA is called with
   `attn_mask=bias, is_causal=False`. The bias is `float` with `finfo.min` on masked keys, causal at
   prefill and key-padding only while decoding, and the diagonal is always unmasked so a fully
   padded query row cannot make an all -inf softmax row and a NaN. No file under any HF cache is
   edited; `install()` records the sha256 of the original `forward` source and refuses to install
   the masked path when `strict=True` and that hash is unknown.

3. LATENT STATE. `initialize_state` is `randn_like` followed by `trunc_normal_`, which overwrites
   every element, so the `randn_like` only consumes RNG (s9f_common.StateStream's note, preflighted
   there against the model's own code). The state is therefore drawn per example from a
   `torch.Generator` seeded `1000003 * problem_index` on a `[1, S, H]` tensor and concatenated,
   which makes it independent of batch width and batch composition, and passed as `input_states`.
   Without this the batched draw would differ from the batch-1 draw for a reason that has nothing to
   do with attention.

Ported verbatim: the static cache subclass and StateStream from `s9f_common.py`; the stop rule,
STOP_IDS, PAD_ID, prompts and parses for Huginn from `s9c_common.py`; `passes_per_token` from both.
"""
import hashlib
import inspect
import os
import sys
import time

import torch

from .. import config as cfgmod

from ..common import BATCH_CAP, MEM_FRACTION, RESERVE_GB
from .base import Adapter

DEV = os.environ.get("PROD_DEV", "cuda")
SEED_MULT = 1000003          # per-example init seed for the latent draw: SEED_MULT * the row index
HUGINN_PAD_ID = 65509        # s9c_common.PAD_ID
HUGINN_STOP_IDS = [65504, 65505, 65508]   # s9c_common.STOP_IDS: begin_text, end_text, end_turn

#: the installed additive attention bias, or None. One entry per process; set by the decode loop
#: immediately before each model() call and cleared after it.
_BIAS = [None]

#: sha256 of the original `CausalSelfAttention.forward` source of every checkpoint revision this
#: package was written against, i.e. the revisions whose body the masked path below copies. An
#: install with `strict=True` refuses anything else, so a checkpoint update cannot silently run a
#: masked path derived from a different body. Recorded 2026-09-11 on the Spark from the cached
#: snapshots (Huginn bb6621b6, McLeish a5f6f126).
KNOWN_ATTENTION_FORWARD = {
    "tomg-group-umd/huginn-0125":
        "948132c035ccbdae0824e909b1676a1046961be77b7f90e18a7bacf7c0fde96d",
    "smcleish/Recurrent-Llama-3.2-train-recurrence-32":
        "2c825f00cad37b8971bad0bb72f1d6c146864d87fd05ce52d0e7080a3700aca6",
}


# ------------------------------------------------------------------ the attention patch
def _module_of(model):
    return sys.modules[type(model).__module__]


def _rotary_style(mod):
    """"complex" = Huginn's apply_rotary_emb_complex_like (rotary BEFORE the head transpose);
    "llama" = McLeish's apply_rotary_pos_emb with a (cos, sin) pair AFTER the transpose."""
    if hasattr(mod, "apply_rotary_emb_complex_like"):
        return "complex"
    if hasattr(mod, "apply_rotary_pos_emb"):
        return "llama"
    raise RuntimeError("no known rotary function in %s" % mod.__name__)


def install_attention_patch(model, strict=False):
    """Replace CausalSelfAttention.forward with the bias-aware copy. Idempotent.

    Returns {"installed", "already", "sha256_original_forward", "rotary_style", "known"}.
    """
    mod = _module_of(model)
    cls = getattr(mod, "CausalSelfAttention")
    src = inspect.getsource(cls.forward)
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    style = _rotary_style(mod)
    known = sha in KNOWN_ATTENTION_FORWARD.values()
    if getattr(cls, "_prod_patched", False):
        return {"installed": False, "already": True, "sha256_original_forward": sha,
                "rotary_style": style, "known": known}
    if strict and not known:
        raise RuntimeError(
            "CausalSelfAttention.forward sha256 %s is not one of the revisions this package was "
            "written against; refusing to install the masked path. Re-read the checkpoint's "
            "raven_modeling_minimal.py, confirm the mask branch, and add the hash." % sha)
    original = cls.forward
    apply_rot = (getattr(mod, "apply_rotary_emb_complex_like") if style == "complex"
                 else getattr(mod, "apply_rotary_pos_emb"))

    def forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
        bias = _BIAS[0]
        if bias is None:
            # No bias installed: run the checkpoint's own forward, untouched.
            return original(self, x, freqs_cis, block_idx, mask, past_key_values)
        B, S, E = x.shape
        q, k, v = self.Wqkv(x).split(self.chunks, dim=2)
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        if style == "complex":                      # Huginn: rotary, then transpose
            q, k = apply_rot(q, k, freqs_cis=freqs_cis)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:                                        # McLeish: transpose, then rotary
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            cos, sin = freqs_cis
            q, k = apply_rot(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, block_idx)
        gqa = q.shape[1] != k.shape[1]
        b = bias[:, :, -q.shape[2]:, :k.shape[2]].to(q.dtype)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=b, dropout_p=0.0, is_causal=False, enable_gqa=gqa)
        y = y.transpose(1, 2).reshape(B, S, E).contiguous()
        return self.proj(y)

    cls.forward = forward
    cls._prod_patched = True
    cls._prod_original_forward = original
    return {"installed": True, "already": False, "sha256_original_forward": sha,
            "rotary_style": style, "known": known}


def build_bias(attn_valid, n_query, dtype, causal):
    """Additive attention bias from a [B, KV] validity mask.

    attn_valid : 1 where the key position is a real token, 0 where it is left padding
    n_query    : rows of the bias (L at prefill, 1 while decoding)
    causal     : True at prefill (query i may not see key j > i)

    The diagonal of the query block is always unmasked so a fully padded query row cannot produce
    an all -inf softmax row (and a NaN that would then propagate through the residual stream).
    """
    B, KV = attn_valid.shape
    neg = torch.finfo(dtype).min
    key_ok = attn_valid.to(torch.bool)[:, None, None, :].expand(B, 1, n_query, KV)
    if causal:
        qpos = torch.arange(KV - n_query, KV, device=attn_valid.device)[:, None]
        kpos = torch.arange(KV, device=attn_valid.device)[None, :]
        ok = key_ok & (kpos <= qpos)[None, None, :, :]
    else:
        ok = key_ok.clone()
    diag = torch.arange(n_query, device=attn_valid.device)
    ok[:, :, diag, KV - n_query + diag] = True
    return torch.where(ok, torch.zeros((), dtype=dtype, device=attn_valid.device),
                       torch.full((), neg, dtype=dtype, device=attn_valid.device))


# ------------------------------------------------------------------ static cache (s9f_common)
_CACHE_CLS = {}


def static_cache_cls(model):
    """`HuginnStaticCache` with a constructor that works on transformers 4.56.

    s9f_common._static_cache_cls, verbatim: the checkpoint's `__init__` calls `super().__init__()`
    and 4.56's `Cache.__init__` raises "You should provide exactly one of `layers` or
    `layer_class_to_replicate`", so the subclass replicates the body for lookup_strategy='full' and
    inherits `update`, `reset`, `get_seq_length` and `get_memory_usage` unchanged.
    """
    mod = _module_of(model)
    if mod.__name__ in _CACHE_CLS:
        return _CACHE_CLS[mod.__name__]

    class ProdStaticCache(mod.HuginnStaticCache):
        def __init__(self, max_length, max_num_steps, num_heads, hidden_dim, batch_size=1,
                     lookup_strategy="full", device=None, dtype=torch.bfloat16):
            assert lookup_strategy == "full", lookup_strategy
            self._seen_tokens = 0
            self.max_length = max_length
            self.lookup_strategy = lookup_strategy
            self.max_num_steps = max_num_steps
            shape = (max_num_steps, batch_size, num_heads, max_length, hidden_dim)
            self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
            self.value_cache = torch.zeros(shape, dtype=dtype, device=device)
            self.valid_mask = torch.zeros((max_num_steps, max_length), dtype=torch.bool,
                                          device=device)
            torch._dynamo.mark_static_address(self.key_cache)
            torch._dynamo.mark_static_address(self.value_cache)
            torch._dynamo.mark_static_address(self.valid_mask)

    _CACHE_CLS[mod.__name__] = ProdStaticCache
    return ProdStaticCache


# ------------------------------------------------------------------ latent state (s9f_common)
class StateStream(object):
    """Per-example RNG stream for the random initial latent state (s9f_common.StateStream).

    `initialize_state` is `randn_like(input_embeds)` followed by `trunc_normal_`, which overwrites
    every element, so the state is exactly a truncated normal and the `randn_like` only consumes
    RNG. Drawing it here from a per-example Generator on a [1, S, H] tensor makes it independent of
    batch width and composition.
    """

    def __init__(self, model, idx, seed=None, device=DEV, dtype=torch.bfloat16):
        self.idx = int(idx)
        self.seed = SEED_MULT * int(idx) if seed is None else int(seed)
        self.H = int(model.config.n_embd)
        self.std = float(model.config.init_values["std"])
        self.emb_scale = float(model.emb_scale)
        self.device = device
        self.dtype = dtype
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(self.seed)

    def reset(self):
        self.gen.manual_seed(self.seed)

    def draw(self, S):
        x = torch.empty((1, S, self.H), dtype=self.dtype, device=self.device)
        torch.nn.init.trunc_normal_(x, mean=0.0, std=self.std, a=-3 * self.std, b=3 * self.std,
                                    generator=self.gen)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        return x


# ------------------------------------------------------------------ the adapter
class RavenAdapter(Adapter):
    """Shared implementation; `huginn.py` and `mcleish.py` only fill in the registry fields."""

    def __init__(self, name, repo, depths, trained_depth, pad_id=None, stop_ids=None,
                 dtype=torch.bfloat16, mem_fraction=MEM_FRACTION, batch_cap=BATCH_CAP,
                 reserve_gb=RESERVE_GB):
        self.name = name
        self.repo = repo
        self.depths = tuple(depths)
        self.trained_depth = trained_depth
        self.chat_template = False           # neither checkpoint ships an instruction-tuned sibling
        self._pad_id = pad_id
        self._stop_ids = list(stop_ids or [])
        self.dtype = dtype
        self.mem_fraction = float(mem_fraction)
        self.batch_cap = int(batch_cap)
        self.reserve_gb = float(reserve_gb)
        self.tok = None
        self.model = None
        self._k = None
        self._pieces = None
        self._info = {}

    # ---------------------------------------------------------------- lifecycle
    def load(self, strict_patch=False):
        if DEV == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.mem_fraction)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # load the REVISION that `prod.install_models` pinned, not whatever the repo's `main`
        # points at today. `revision=None` (no pins file yet) loads `main`.
        rev = cfgmod.revision_for(self.name)
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

    def set_depth(self, k):
        self._k = int(k)
        if self._pieces is None:
            # batch_decode once rather than 65k-128k single decodes (the spikes' loop cost ~60 s
            # per process on the McLeish tokenizer)
            n = int(self.model.config.vocab_size)
            self._pieces = self.tok.batch_decode([[i] for i in range(n)],
                                                 clean_up_tokenization_spaces=False)
        return {"num_steps": int(k), "passes_per_token": self.passes_per_token(k)}

    def pad_id(self):
        if self._pad_id is not None:
            return int(self._pad_id)
        for cand in (self.tok.pad_token_id, self.tok.eos_token_id):
            if cand is not None:
                return int(cand)
        return 0

    def stop_ids(self):
        if self._stop_ids:
            return list(self._stop_ids)
        return [int(self.tok.eos_token_id)]

    # ---------------------------------------------------------------- shapes and cost
    def n_layers(self):
        cfg = self.model.config
        return int(cfg.n_layers_in_prelude + cfg.n_layers_in_recurrent_block
                   + cfg.n_layers_in_coda)

    def passes_per_token(self, k):
        """prelude + k * core + coda (s9c_common.layer_passes, s9f_common.passes_per_token)."""
        cfg = self.model.config
        return int(cfg.n_layers_in_prelude + int(k) * cfg.n_layers_in_recurrent_block
                   + cfg.n_layers_in_coda)

    def kv_bytes_per_token(self, k):
        """The static cache holds one K and one V per (step slot, token): 2 * heads * head_dim *
        itemsize * passes_per_token(k)."""
        cfg = self.model.config
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        hd = cfg.n_embd // cfg.num_attention_heads
        return int(2 * cfg.num_key_value_heads * hd * itemsize * self.passes_per_token(k))

    def row_token_ceiling(self, k):
        free, total = torch.cuda.mem_get_info()
        ours = torch.cuda.memory_reserved()
        cap_left = self.mem_fraction * total - ours
        avail = min(float(free), max(0.0, cap_left)) - self.reserve_gb * 1024 ** 3
        return max(64, int(avail / self.kv_bytes_per_token(k)))

    def batch_for(self, k, max_seq):
        try:
            est = int(self.row_token_ceiling(k) / max(1, max_seq))
        except Exception:                                     # noqa: BLE001
            est = self.batch_cap
        return max(1, min(self.batch_cap, est)), est

    def make_cache(self, k, max_length, batch_size):
        cfg = self.model.config
        return static_cache_cls(self.model)(
            max_length=int(max_length), max_num_steps=self.passes_per_token(k),
            num_heads=int(cfg.num_key_value_heads),
            hidden_dim=int(cfg.n_embd // cfg.num_attention_heads),
            batch_size=int(batch_size), lookup_strategy="full",
            device=torch.device(DEV), dtype=self.dtype)

    # ---------------------------------------------------------------- decoding
    @torch.no_grad()
    def decode(self, seqs, n_new, k, eos_ids=None, stop_strings=None, pieces=None, row_ids=None,
               use_mask=None, cache=None, allow_unsafe_padding=False):
        """Greedy decode a left-padded batch. Returns (generated id lists, seconds).

        use_mask None -> a mask is installed iff the batch is padded (lengths differ) and B > 1.
                 False -> never install one, which is the stock model path (batch 1, or an
                          exact-equal-length group): used by gate G0's control arms.
        row_ids  -> problem indices, one per row, for the per-example latent state stream.
        """
        assert int(k) == self._k, "set_depth(%r) before decode" % (k,)
        m = self.model
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
            raise ValueError(
                "a padded batch of %d rows needs use_mask=True: the stock forward hard-codes "
                "prepared_attn_mask = None, so it would attend to pad tokens. Pass "
                "allow_unsafe_padding=True only to MEASURE that error (gate G0 arm B)." % B)
        row_ids = list(range(B)) if row_ids is None else list(row_ids)
        streams = [StateStream(m, i, device=DEV, dtype=self.dtype) for i in row_ids]

        dev = next(m.parameters()).device
        ids = torch.full((B, L), pad, dtype=torch.long, device=dev)
        valid = torch.zeros((B, L + n_new), dtype=torch.bool, device=dev)
        for r, s in enumerate(seqs):
            ids[r, L - len(s):] = torch.tensor(s, dtype=torch.long, device=dev)
            valid[r, L - len(s):L] = True
        own_cache = cache is None
        cache = self.make_cache(k, L + n_new + 1, B) if own_cache else cache
        cache.reset()

        t0 = time.time()
        # The initial latent state must be drawn at the row's REAL length, not at the padded length.
        # `StateStream.draw(n)` consumes n * H values, so drawing L for a row of length n would both
        # attach the wrong draw block to each real token (token i of the row sits at position
        # L - n + i, not i) and leave the generator at a different point before the first decode
        # draw. Drawing n and placing it at the row's real positions makes a padded batch's state
        # stream identical to that row's batch-1 stream. Pad positions get a throwaway draw rather
        # than zeros, so nothing degenerate enters the norms; they are masked out of attention.
        x0 = torch.empty((B, L, m.config.n_embd), dtype=self.dtype, device=dev)
        _std = float(m.config.init_values["std"])
        _g = torch.Generator(device=DEV)
        _g.manual_seed(0)
        torch.nn.init.trunc_normal_(x0, mean=0.0, std=_std, a=-3 * _std, b=3 * _std, generator=_g)
        for r, st in enumerate(streams):
            x0[r, L - lens[r]:, :] = st.draw(lens[r])[0]
        _BIAS[0] = build_bias(valid[:, :L], L, self.dtype, causal=True) if use_mask else None
        try:
            out = m(input_ids=ids, num_steps=int(k), past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(0, L, device=dev), input_states=x0)
            cur = out.logits[:, -1, :].float().argmax(-1)
            del out
        finally:
            _BIAS[0] = None

        gen = [[] for _ in range(B)]
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
                    if any(s in tails[b] for s in stop_strings):
                        fin[b] = True
            if t == n_new - 1 or all(fin):
                break
            valid[:, pos] = True
            x1 = torch.cat([st.draw(1) for st in streams], dim=0)
            _BIAS[0] = (build_bias(valid[:, :pos + 1], 1, self.dtype, causal=False)
                        if use_mask else None)
            try:
                out = m(input_ids=torch.tensor(feed, dtype=torch.long,
                                               device=dev).unsqueeze(1),
                        num_steps=int(k), past_key_values=cache, use_cache=True,
                        cache_position=torch.tensor([pos], device=dev), input_states=x1)
                cur = out.logits[:, -1, :].float().argmax(-1)
                del out
            finally:
                _BIAS[0] = None
            pos += 1
        dt = time.time() - t0
        if own_cache:
            del cache
            torch.cuda.empty_cache()
        return gen, dt

    def meta(self):
        d = {"name": self.name, "family": self.family, "repo": self.repo,
             "chat_template": False, "depths": list(self.depths),
             "trained_depth": self.trained_depth, "dtype": str(self.dtype),
             "mem_fraction": self.mem_fraction, "batch_cap": self.batch_cap,
             "reserve_gb": self.reserve_gb, "seed_mult": SEED_MULT}
        d.update(self._info)
        return d
