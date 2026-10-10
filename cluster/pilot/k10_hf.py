"""K10-A runners for the transformers-loaded families: Ouro-1.4B (OuroAdapter) and the raven family
(Recurrent-Llama-3.2 retrofit, Huginn-0125; RavenAdapter). Generation is the prod adapters' own batched
KV-cache decode, unchanged. The hooked forward is one no-cache forward that records the state after every
pass at every position (Ouro: forward hook on model.model.norm; raven: instance wrapper of
core_block_forward), with a lens that reproduces the model's own head (C1).
"""
import time

import torch


class HookedOut(object):
    """Result of one hooked forward. states: list of K [T, H] tensors (the carried state after each
    pass); h0 [T, H]; logits [T, V] the model's own output; lens(j, pos) -> [n, V] float."""

    def __init__(self, states, h0, logits, lens_fn, n_calls, extra=None):
        self.states, self.h0, self.logits, self._lens = states, h0, logits, lens_fn
        self.n_calls = n_calls
        self.extra = extra or {}

    def lens(self, j, pos):
        return self._lens(j, pos)

    def logits_at(self, pos):
        return self.logits[pos].float()


# ------------------------------------------------------------------ Ouro
class OuroRunner(object):
    family = "ouro"
    loader = "transformers"

    def __init__(self, name="ouro_1_4b_base"):
        from prod.models import get
        self.ad = get(name)
        self.ad.load()
        self.m, self.tok = self.ad.model, self.ad.tok
        self.dev = next(self.m.parameters()).device
        self.k = None
        self.pieces = self.tok.batch_decode([[i] for i in range(int(self.m.config.vocab_size))],
                                            clean_up_tokenization_spaces=False)
        self.max_positions = None
        self.attn_backend = str(getattr(self.m.config, "_attn_implementation", "sdpa"))

    def set_depth(self, k):
        if self.k != int(k):
            self.ad.set_depth(int(k))
            self.k = int(k)

    def prompt_ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def eos_ids(self, base):
        return list(base)

    def width(self, k, max_seq, cap=16):
        b, _ = self.ad.batch_for(k, max_seq)
        return max(1, min(cap, b))

    def generate(self, seqs, n_new, eos_ids, stop_strings, row_ids):
        g, _ = self.ad.decode(seqs, n_new, self.k, eos_ids, stop_strings=stop_strings)
        return [list(x) for x in g]

    @torch.no_grad()
    def hooked(self, ids, row_id, n_prompt):
        k = self.k
        inner = self.m.model
        rec = {"post": [], "emb": None}
        hs = [inner.norm.register_forward_hook(lambda mod, i, o: rec["post"].append(o[0].detach())),
              inner.embed_tokens.register_forward_hook(
                  lambda mod, i, o: rec.__setitem__("emb", o[0].detach()))]
        try:
            x = torch.tensor([ids], dtype=torch.long, device=self.dev)
            out = self.m(input_ids=x, use_cache=False, logits_to_keep=0)
        finally:
            for h in hs:
                h.remove()
        logits = out.logits[0]
        posts = rec["post"]
        n_calls = len(posts)

        def lens(j, pos):
            return self.m.lm_head(posts[j - 1][pos]).float()
        return HookedOut(posts, rec["emb"], logits, lens, n_calls)

    def injection_summary(self):
        return {"note": "Ouro has no re-read of the input: pass 1 reads the embeddings, later passes "
                        "read only the carried state"}

    def meta(self):
        return {"family": "ouro", "repo": self.ad.repo, "k": self.k,
                "attention": self.attn_backend}


# ------------------------------------------------------------------ raven (retrofit, Huginn)
class RavenRunner(object):
    loader = "transformers"

    def __init__(self, name):
        from prod.models import get
        self.ad = get(name)
        self.ad.load()
        self.family = self.ad.family
        self.m, self.tok = self.ad.model, self.ad.tok
        cfg = self.m.config
        noise = float(getattr(cfg, "test_time_noise", 0) or 0)
        assert noise == 0.0, "test_time_noise %r" % noise
        self.dev = next(self.m.parameters()).device
        self.k = None
        self.ast = bool(getattr(self.ad, "add_special_tokens", False))
        self.pieces = None
        self.max_positions = getattr(self.ad, "max_positions", None)
        self.attn_backend = "sdpa (raven CausalSelfAttention; prod bias-mask patch for padded batches)"
        self._orig_cbf = self.m.core_block_forward          # bound method of the class
        self.rec = None                                     # None = generation, no recording
        self.intervene = None
        self.m.core_block_forward = self._cbf
        self.calls = 0

    def _cbf(self, x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
        step = self.calls
        self.calls += 1
        if self.intervene is not None:
            x, input_embeds = self.intervene(step, x, input_embeds)
        if self.rec is not None:
            if step == 0:
                self.rec["h0"] = x[0].detach()
                self.rec["inp"] = input_embeds.detach()
                self.rec["freqs"] = freqs_cis
        out, bidx = self._orig_cbf(x, input_embeds, freqs_cis, mask, past_key_values, block_idx,
                                   current_step)
        if self.rec is not None:
            self.rec["states"].append(out.detach())
        return out, bidx

    def set_depth(self, k):
        if self.k != int(k):
            self.ad.set_depth(int(k))
            self.k = int(k)
        if self.pieces is None:
            self.pieces = list(self.ad._pieces)

    def prompt_ids(self, text):
        return self.tok(text, add_special_tokens=self.ast)["input_ids"]

    def eos_ids(self, base):
        return sorted(set(base) | set(self.ad.stop_ids()))

    def width(self, k, max_seq, cap=16):
        b, _ = self.ad.batch_for(k, max_seq)
        return max(1, min(cap, b))

    def generate(self, seqs, n_new, eos_ids, stop_strings, row_ids):
        self.calls = 0
        g, _ = self.ad.decode(seqs, n_new, self.k, eos_ids, stop_strings=stop_strings,
                              row_ids=row_ids)
        return [list(x) for x in g]

    def init_states(self, row_ids, n_prompt, T):
        """The initial latent state the generation path uses for this row: StateStream(row idx), one
        draw of n_prompt for the prompt, then one draw of 1 per later position (decode's order)."""
        from prod.models.raven import StateStream
        out = []
        for rid in row_ids:
            st = StateStream(self.m, rid, device=str(self.dev), dtype=self.ad.dtype)
            parts = [st.draw(n_prompt)] + [st.draw(1) for _ in range(T - n_prompt)]
            out.append(torch.cat(parts, dim=1))
        return torch.cat(out, dim=0)

    def coda_lens(self, x, freqs):
        """The forward's path from a recurrence output to logits, all positions, no cache."""
        tr = self.m.transformer
        if self.family == "huginn":
            x = tr.ln_f(x)                     # iterate_forward returns ln_f(x) on Huginn
        bidx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in tr.coda:
            bidx -= 1
            x = block(x, freqs, bidx, None, None)
        x = tr.ln_f(x)
        return self.m.lm_head(x).float()

    @torch.no_grad()
    def forward_batch(self, ids_batch, x0, record=True):
        """No-cache forward of equal-length rows with given initial states; records every pass."""
        self.calls = 0
        self.rec = {"states": []} if record else None
        try:
            x = torch.tensor(ids_batch, dtype=torch.long, device=self.dev)
            out = self.m(input_ids=x, num_steps=self.k, input_states=x0, use_cache=False)
            rec = self.rec
        finally:
            self.rec = None
        return out.logits, rec

    @torch.no_grad()
    def hooked(self, ids, row_id, n_prompt):
        x0 = self.init_states([row_id], n_prompt, len(ids))
        logits, rec = self.forward_batch([ids], x0)
        states = [s[0] for s in rec["states"]]
        freqs = rec["freqs"]
        cache = {}

        def lens(j, pos):
            if j not in cache:
                cache.clear()
                cache[j] = self.coda_lens(rec["states"][j - 1], freqs)[0]
            return cache[j][pos]
        return HookedOut(states, rec["h0"], logits[0], lens, len(states),
                         extra={"inp": rec["inp"][0]})

    def injection_summary(self):
        W = self.m.transformer.adapter.weight.detach().float()
        d = W.shape[0]
        ws, wi = W[:, :d], W[:, d:]

        def nrm(a):
            return {"fro": round(float(a.norm()), 4),
                    "spectral": round(float(torch.linalg.matrix_norm(a, ord=2)), 4)}
        b = self.m.transformer.adapter.bias
        return {"adapter_shape": list(W.shape), "state_half": nrm(ws), "input_half": nrm(wi),
                "fro_ratio_input_over_state": round(float(wi.norm() / ws.norm()), 4),
                "bias_norm": None if b is None else round(float(b.detach().float().norm()), 4),
                "column_order": "adapter(torch.cat([state, prelude_output], -1)): columns [:d] state, "
                                "[d:] prelude output"}

    def meta(self):
        m = self.ad.meta()
        return {"family": self.family, "repo": self.ad.repo, "k": self.k,
                "attention": self.attn_backend,
                "n_layers_in_prelude": m.get("n_layers_in_prelude"),
                "n_layers_in_recurrent_block": m.get("n_layers_in_recurrent_block"),
                "n_layers_in_coda": m.get("n_layers_in_coda"),
                "add_special_tokens": self.ast}


def make_runner(model):
    t = time.time()
    if model == "ouro":
        r = OuroRunner("ouro_1_4b_base")
    elif model == "ouro26":
        r = OuroRunner("ouro_2_6b_base")
    elif model in ("retrofit", "retrofit_r32", "retrofit_r8"):
        r = RavenRunner("mcleish_llama32_r32")
    elif model == "huginn":
        r = RavenRunner("huginn_0125")
    else:
        raise KeyError(model)
    r.load_seconds = time.time() - t
    return r
