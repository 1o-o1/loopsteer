"""K10-A runner for the IFM looped checkpoints (xllm-loop, native format). Runs in the IFM venv (cluster/env/requirements_ifm.txt).

Loading follows xllm.paper_part1.native_inference.load_native_model (strict state dict, init_mode none,
fused blocks off) with one change: causal_attn_backend = None (pure PyTorch attention), set before the
model is built. The forward is re-implemented over the model's own modules so that (a) the state after
every pass is recorded at every position, (b) generation can use a KV cache with one slot per executed
layer call (the native forward takes no cache). Both are checked against model.forward (k10_smoke.py).

Architectures (xllm/models/looped.py):
  huginn       prelude layers, input_embeds = _recurrence_input(prelude output), state init, loop_times x
               [input_injection(state, input_embeds) -> recurrent layers -> optional exit norm], coda.
  transformer  LoopedTransformer: execution_schedule; loop_input_injection_module(hidden, embeddings) at
               every loop entry; dense-prelude injection at dense_prelude_input_layers.
"""
import json
import math
import os
import sys
import time

import torch

# cluster copy: the xllm-loop clone and the IFM checkpoint folders come from the environment
XLLM = os.environ.get("XLLM_ROOT", "")
CKPT = os.environ.get("IFM_CKPT_ROOT") or os.environ.get("WEIGHTS_ROOT", "")
NAMES = {
    "ifm_p2s": "LoopedLM-P2-huginn-s-fixed-r5",
    "ifm_p1_ouro": "LoopedLM-P1-dense-ouro-336b",
    "ifm_p1_ouro_inj": "LoopedLM-P1-dense-ouro-raw-injection-336b",
    "ifm_p1_mid": "LoopedLM-P1-dense-middle-336b",
    "ifm_p1_mid_inj": "LoopedLM-P1-dense-middle-prelude-injection-336b",
    "ifm_p1_hug": "LoopedLM-P1-dense-huginn-336b",
    "ifm_p2m_diag": "LoopedLM-P2-huginn-m-fixed-pln5",
    "ifm_p2m_ortho": "LoopedLM-P2-huginn-m-orthoinj",
    "ifm_p2m_r5": "LoopedLM-P2-huginn-m-fixed-r5",
    "ifm_p2s_r5": "LoopedLM-P2-huginn-s-fixed-r5",
    "ifm_p2s_pln5": "LoopedLM-P2-huginn-s-fixed-pln5",
}


KV_MAX = 4608        # positions held by one preallocated KV slot (prompt + 2048 + read-out < 4608)


class FastKV(object):
    """Preallocated KV buffer for one executed layer call (one schedule slot)."""

    def __init__(self, B, maxlen, kvh, d, dtype, device):
        self.k = torch.empty((B, maxlen, kvh, d), dtype=dtype, device=device)
        self.v = torch.empty((B, maxlen, kvh, d), dtype=dtype, device=device)
        self.n = 0


def install_fast_attention(model):
    """Instance-level replacement of every CausalSoftmaxAttention.forward: when the cache is a FastKV
    (K10's cached decode), rotary as in the original, the new keys/values written into the
    preallocated buffer, and SDPA with grouped-query heads (causal at prefill, which always starts
    at position 0; no mask for a one-token step). Any other call (cache None or a tuple) runs the
    original forward unchanged, so the no-cache forward used for the lens and for C1 is the checkpoint's
    own pure-PyTorch path."""
    from einops import rearrange
    from xllm.modules.causal_attention import CausalSoftmaxAttention
    from xllm.modules.rotary_positional_embedding import apply_rotary_embeddings
    n = 0
    for mod in model.modules():
        if not isinstance(mod, CausalSoftmaxAttention):
            continue
        orig = mod.forward

        def fwd(xq, xk, xv, g, freqs_cis, segments, fp32_attn_output=False, deterministic=True,
                cache=None, _orig=orig, _m=mod):
            if not isinstance(cache, FastKV):
                return _orig(xq, xk, xv, g, freqs_cis, segments, fp32_attn_output, deterministic, cache)
            assert segments is None
            bs, slen, _ = xq.shape
            xq = rearrange(xq, 'b l (h d) -> b l h d', h=_m.n_heads)
            xk = rearrange(xk, 'b l (h d) -> b l h d', h=_m.n_kv_heads)
            xv = rearrange(xv, 'b l (h d) -> b l h d', h=_m.n_kv_heads)
            if _m.rope_head_dim == _m.head_dim:
                xq, xk = apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=False)
            elif _m.rope_head_dim > 0:
                sp = [_m.rope_head_dim, _m.head_dim - _m.rope_head_dim]
                xq_r, xq_n = torch.split(xq, sp, dim=-1)
                xk_r, xk_n = torch.split(xk, sp, dim=-1)
                xq_r, xk_r = apply_rotary_embeddings(xq_r, xk_r, freqs_cis=freqs_cis, backward=False)
                xq = torch.cat([xq_r, xq_n], dim=-1)
                xk = torch.cat([xk_r, xk_n], dim=-1)
            n0 = cache.n
            cache.k[:, n0:n0 + slen] = xk
            cache.v[:, n0:n0 + slen] = xv
            cache.n = n0 + slen
            if slen > 1:
                assert n0 == 0, "K10 cached prefill must start at position 0"
            q = xq.transpose(1, 2)
            k = cache.k[:, :cache.n].transpose(1, 2)
            v = cache.v[:, :cache.n].transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=slen > 1, scale=_m.scale, enable_gqa=True)
            return out.transpose(1, 2).reshape(bs, slen, -1), cache
        mod.forward = fwd
        n += 1
    return n


class XTok(object):
    """The xllm HuggingFace tokenizer with the interface prod.tasks uses."""

    def __init__(self, xt):
        self.xt = xt
        self.hf = xt._tokenizer
        self.eos_token_id = int(xt.eos_id)
        self.bos_token_id = int(xt.bos_id)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(self.hf(text, add_special_tokens=False).input_ids)}

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return self.hf.decode(list(ids), clean_up_tokenization_spaces=False)


def load_ifm(name, backend=None, device="cuda"):
    """-> (model, xllm tokenizer, ModelConf, raw config dict). Strict load, pure PyTorch attention."""
    if XLLM not in sys.path:
        sys.path.insert(0, XLLM)
    stubs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs")
    if stubs not in sys.path:
        sys.path.append(stubs)                 # only reached when the real module is absent
    from xllm.config import ModelConf, TokenizerConf
    from xllm.data.dataset_streamer.tokenizer import build_tokenizer
    from xllm.models.build import get_model_cls
    root = os.path.join(CKPT, NAMES[name])
    raw = json.load(open(os.path.join(root, "config.json")))
    if "paper-part2" in open(os.path.join(root, "artifact_manifest.json")).read():
        from xllm.paper_part2.artifacts import load_artifact as la2
        fields, state_dict = la2(root)
        tcfg = dict(raw["tokenizer"])
        tcfg["path"] = os.path.realpath(os.path.join(root, tcfg["path"]))
        model_fields = dict(fields)
    else:
        from xllm.paper_part1.artifacts import load_artifact as la1
        config, state_dict = la1(root)
        tcfg = config["tokenizer"]
        model_fields = dict(config["model"])
    tok = build_tokenizer(TokenizerConf.from_dict(tcfg))
    cfg = ModelConf.from_dict(model_fields)
    cfg.init_mode = "none"
    cfg.fused_block = False
    cfg.fused_output_layer = False
    cfg.causal_attn_backend = backend
    if cfg.vocab_size == -1:
        cfg.vocab_size = tok.vocab_size
    dev = torch.device(device)
    od, odt = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device(dev)
        torch.set_default_dtype(torch.bfloat16)
        model = get_model_cls(cfg.arch, cfg)(cfg, tok)
    finally:
        torch.set_default_dtype(odt)
        torch.set_default_device(od)
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = True        # configure_native_runtime
    return model, tok, cfg, raw


class IfmRunner(object):
    family = "ifm"
    loader = "xllm"

    def __init__(self, name):
        self.name = name
        self.m, xt, self.cfg, self.raw = load_ifm(name)
        self.tok = XTok(xt)
        self.dev = next(self.m.parameters()).device
        self.arch = self.cfg.arch
        self.K_trained = int(self.m.loop_times)
        self.k = None
        self.max_positions = None
        self.attn_backend = ("None (pure PyTorch; artifact config says %r); cached decode: SDPA on "
                             "preallocated KV (k10_ifm.install_fast_attention)" % (
                                 self.raw["model"].get("causal_attn_backend"),))
        m = self.m
        assert getattr(m, "huginn_hierarchical_state", "none") in ("none", None)
        assert getattr(m, "context_parallel_size", 1) == 1
        self.dc = hasattr(m, "identity_recurrent_state")       # DepthControlledHuginn
        n = int(self.cfg.vocab_size)
        self.pieces = [self.tok.hf.decode([i], clean_up_tokenization_spaces=False) for i in range(n)]
        self.init_mode = ("identity" if getattr(m, "huginn_state_init", None) == "normal"
                          else "native")
        self.n_calls = 0
        self.n_fast_attention = install_fast_attention(self.m)

    # -------------------------------------------------------------- structure
    def set_depth(self, k):
        self.k = int(k)

    def schedule(self, k):
        """[(slot, layer_id, tag)] with tag in prelude | loop<j> | coda (1-based j)."""
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
                c = FastKV(h.shape[0], KV_MAX, ca.n_kv_heads, ca.head_dim, h.dtype, h.device)
        h, _aux, nc = self.m.layers[lid](h, freq, None, moe_router_load_balancing_type=None,
                                         fp32_attn_output=False, deterministic=True, cache=c)
        if cache is not None:
            cache[slot] = nc
        return h

    def _init_state(self, input_embeds, cache_len, example_ids, init):
        m = self.m
        if m.huginn_state_init == "normal" and init == "identity":
            # Part II's batch-invariant evaluation state; plain Huginn (Part I) has no such method,
            # so DepthControlledHuginn's is applied to it (it uses only _hashed_normal_state and
            # model_dim; the scale sqrt(2)/sqrt(d) equals the native sqrt(2/d))
            from xllm.models.looped import DepthControlledHuginn
            return DepthControlledHuginn.identity_recurrent_state(m, input_embeds, cache_len,
                                                                  example_ids, 0)
        return m.initialize_recurrent_state(input_embeds, cache_len)

    @torch.no_grad()
    def fwd(self, tokens, k, cache=None, cache_len=0, example_ids=None, record=False, init=None,
            all_logits=False):
        """One forward of `tokens` [B, S] at depth k. cache: dict slot -> (k, v, len) or None (no
        cache, positions from 0). Returns (logits [B, S or 1, V] float, rec or None)."""
        m = self.m
        init = init or self.init_mode
        B, S = tokens.shape
        if example_ids is None:
            example_ids = torch.arange(B, device=tokens.device)
        hidden = m.embed(tokens)
        freq = None if m.rope is None else m.rope.get_freqs_cis(cache_len, cache_len + S, hidden.device)
        rec = {"states": [], "h0": None, "freq": freq} if record else None
        self.n_calls = 0
        sched = self.schedule(k)
        if self.arch == "huginn":
            P = int(m.loop_start_layer)
            slot = 0
            for i in range(P):
                hidden = self._layer(i, hidden, freq, cache, slot)
                slot += 1
            input_embeds = m._recurrence_input(hidden)
            state = self._init_state(input_embeds, cache_len, example_ids, init)
            if record:
                rec["h0"] = state
                rec["inp"] = input_embeds
            exit_norm = getattr(m, "huginn_recurrent_exit_norm_module", None)
            for j in range(k):
                if m.input_injection is not None:
                    state = m.input_injection(state, input_embeds)
                for i in range(m.loop_start_layer, m.loop_end_layers):
                    state = self._layer(i, state, freq, cache, slot)
                    slot += 1
                if exit_norm is not None:
                    state = exit_norm(state)
                self.n_calls += 1
                if record:
                    rec["states"].append(state)
            hidden = state
            for i in range(m.loop_end_layers, m.num_layers):
                hidden = self._layer(i, hidden, freq, cache, slot)
                slot += 1
        else:
            emb = hidden
            dpe = None
            inj = m.loop_input_injection_module
            dinj = m.dense_prelude_input_injection_module
            for slot, lid, tag in sched:
                if inj is not None and lid == m.loop_start_layer and tag.startswith("loop"):
                    if record and tag == "loop1" and rec["h0"] is None:
                        rec["h0"] = hidden
                    hidden = inj(hidden, emb)
                elif record and tag == "loop1" and lid == m.loop_start_layer and rec["h0"] is None:
                    rec["h0"] = hidden
                if dinj is not None and lid in m.dense_prelude_input_layers:
                    hidden = dinj(hidden, dpe)
                hidden = self._layer(lid, hidden, freq, cache, slot)
                if dinj is not None and lid == m.dense_prelude_source_layer:
                    dpe = hidden
                if tag.startswith("loop") and lid == m.loop_end_layers - 1:
                    self.n_calls += 1
                    if record:
                        rec["states"].append(hidden)
        h = hidden if all_logits else hidden[:, -1:]
        return m.output(h, None, None), rec

    def coda_lens(self, state, freq):
        """Pass-j state -> logits through the layers after the loop (no cache) and the output head."""
        m = self.m
        h = state
        for i in range(m.loop_end_layers, m.num_layers):
            h, _a, _c = m.layers[i](h, freq, None, moe_router_load_balancing_type=None,
                                    fp32_attn_output=False, deterministic=True, cache=None)
        return m.output(h, None, None)

    # -------------------------------------------------------------- runner interface
    def prompt_ids(self, text):
        return [self.tok.bos_token_id] + self.tok(text)["input_ids"]

    def eos_ids(self, base):
        return sorted(set(base) | {self.tok.eos_token_id})

    def width(self, k, max_seq, cap=16):
        return cap

    @torch.no_grad()
    def generate(self, seqs, n_new, eos_ids, stop_strings, row_ids):
        """Greedy, KV cache per executed layer call. Right-padded rows: generation starts at the
        shortest prompt, a row keeps its own prompt token until its prompt ends (generate_native)."""
        B = len(seqs)
        lens = [len(s) for s in seqs]
        start, end = min(lens), max(lens) + n_new
        dev = self.dev
        tokens = torch.zeros((B, end), dtype=torch.long, device=dev)
        for b, s in enumerate(seqs):
            tokens[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
        ex = torch.tensor(list(row_ids), dtype=torch.long, device=dev)
        cache = {}
        logits, _ = self.fwd(tokens[:, :start], self.k, cache=cache, cache_len=0, example_ids=ex)
        eos = set(eos_ids or [])
        gen = [[] for _ in range(B)]
        fin = [False] * B
        tails = [""] * B
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
                elif stop_strings:
                    tails[b] = (tails[b] + self.pieces[x])[-64:]
                    if any(s_ in tails[b] for s_ in stop_strings):
                        fin[b] = True
            tokens[:, cur] = torch.tensor(col, dtype=torch.long, device=dev)
            if cur == end - 1 or (all(fin) and cur >= max(lens) - 1):
                break
            logits, _ = self.fwd(tokens[:, cur:cur + 1], self.k, cache=cache, cache_len=cur,
                                 example_ids=ex)
        del cache
        return gen

    @torch.no_grad()
    def hooked(self, ids, row_id, n_prompt):
        from k10_hf import HookedOut
        x = torch.tensor([ids], dtype=torch.long, device=self.dev)
        ex = torch.tensor([row_id], dtype=torch.long, device=self.dev)
        logits, rec = self.fwd(x, self.k, cache=None, example_ids=ex, record=True, all_logits=True)
        states = [s[0] for s in rec["states"]]
        freq = rec["freq"]
        memo = {}

        def lens(j, pos):
            if j not in memo:
                memo.clear()
                memo[j] = self.coda_lens(rec["states"][j - 1], freq)[0]
            return memo[j][pos]
        return HookedOut(states, rec["h0"][0], logits[0], lens, self.n_calls)

    def injection_summary(self):
        m = self.m
        mods = {}
        for nm in ("input_injection", "loop_input_injection_module",
                   "dense_prelude_input_injection_module"):
            mod = getattr(m, nm, None)
            if mod is not None:
                mods[nm] = mod
        out = {"modules": {}}
        for nm, mod in mods.items():
            d = {"class": type(mod).__name__}
            if hasattr(mod, "A_log"):
                dt = torch.nn.functional.softplus(mod.dt_bias.float())
                a = torch.exp(mod.A_log.float())
                decay = torch.exp(-dt * a)
                W = mod.input_adapter.weight.float()
                eye = torch.eye(W.shape[0], W.shape[1], device=W.device)

                def q(t):
                    v = torch.quantile(t.flatten().cpu(), torch.tensor([0.25, 0.5, 0.75]))
                    return {"q25": round(float(v[0]), 5), "median": round(float(v[1]), 5),
                            "q75": round(float(v[2]), 5), "min": round(float(t.min()), 5),
                            "max": round(float(t.max()), 5)}
                d.update({"decay_exp_minus_dt_a": q(decay), "gain_dt": q(dt),
                          "W_fro": round(float(W.norm()), 4),
                          "W_minus_identity_fro": round(float((W - eye).norm()), 4),
                          "W_diag": q(torch.diagonal(W)),
                          "def": "h <- decay * h + dt * (W e); decay = exp(-softplus(dt_bias) * "
                                 "exp(A_log)), dt = softplus(dt_bias), per channel"})
            if hasattr(mod, "adapter"):
                W = mod.adapter.weight.float()
                dd = W.shape[0]
                d.update({"state_half_fro": round(float(W[:, :dd].norm()), 4),
                          "input_half_fro": round(float(W[:, dd:].norm()), 4)})
            out["modules"][nm] = d
        if not mods:
            out["note"] = "no input injection in this checkpoint"
        out["state_init"] = getattr(m, "huginn_state_init", None)
        out["recurrent_exit_norm"] = getattr(self.cfg, "huginn_recurrent_exit_norm", None)
        return out

    def meta(self):
        m = self.m
        return {"family": "ifm", "checkpoint": NAMES[self.name], "arch": self.arch,
                "class": type(m).__name__, "k": self.k, "loop_times_trained": self.K_trained,
                "loop_start_layer": int(m.loop_start_layer), "loop_end_layers": int(m.loop_end_layers),
                "num_layers": int(m.num_layers), "init_mode": self.init_mode,
                "attention": self.attn_backend}


def make_runner(model):
    t = time.time()
    r = IfmRunner(model)
    r.load_seconds = time.time() - t
    return r
