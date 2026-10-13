"""Ouro adapter: 1.4B and 2.6B, base and Thinking, plus any LoRA adapter directory.

Ported verbatim (source in each comment):
  * load / dtype / attn_implementation   s28_common.load_model
  * the step knob and the EXPLICIT read-out  s9b_common.set_steps
      `model.model.total_ut_steps = k` and `model.early_exit_step = k - 1`. Ouro-2.6B's default
      forward path mixes readouts (about 1% of tokens read step 1 under the shipped threshold of
      1.0), so every 2.6B number must come from exit_at_step=3 or lm_head on the last hidden
      state. The explicit read-out is set on 1.4B too, where it is a no-op
      (config.early_exit_threshold is None there), so one code path serves both scales.
  * the get_mask_sizes patch          s3_patch.patch_universal_cache (ByteDance's own fix, shipped
      in Ouro-1.4B-Thinking's modeling_ouro.py but not in Ouro-1.4B's); attach the override, then
      left-pad freely.
  * the preallocated UniversalTransformerCache   s28_common.make_static_cache_cls
  * the measured KV batch ceiling               s28_common.row_token_ceiling / batch_for
  * greedy decode over that cache               s28_common.decode
  * forced continuation ("Wait," injection)     s13_common.decode(stopper=...) / make_stopper
  * LoRA arms                                   s32_common.load_arm (merge_and_unload) and the
      S32 tag line "Loops: k." / "Loops: k. Tokens: T."
"""
import os
import sys
import time
from collections import deque

import torch

from .. import config as cfgmod

from ..common import BATCH_CAP, MEM_FRACTION, RESERVE_GB
from .base import Adapter

DEV = os.environ.get("PROD_DEV", "cuda")
TAIL_WINDOW = 32

# ------------------------------------------------------------------ registry rows
OURO_CHECKPOINTS = {
    "ouro_1_4b_base": {"repo": "ByteDance/Ouro-1.4B", "chat": False, "layers": 24},
    "ouro_1_4b_think": {"repo": "ByteDance/Ouro-1.4B-Thinking", "chat": True, "layers": 24},
    "ouro_2_6b_base": {"repo": "ByteDance/Ouro-2.6B", "chat": False, "layers": 48},
    "ouro_2_6b_think": {"repo": "ByteDance/Ouro-2.6B-Thinking", "chat": True, "layers": 48},
}
OURO_DEPTHS = (1, 2, 3, 4)
OURO_TRAINED_DEPTH = 4          # Ouro paper 2510.25741, T_max = 4


# ------------------------------------------------------------------ the cache patch (s3_patch)
def patch_universal_cache(model):
    """Attach Ouro-1.4B-Thinking's get_mask_sizes to the UniversalTransformerCache class the
    dynamic module already loaded. The cached file is never edited. (s3_patch.py, verbatim.)"""
    mod = sys.modules[type(model).__module__]
    cls = getattr(mod, "UniversalTransformerCache")
    if "get_mask_sizes" in cls.__dict__:
        return False, True

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        query_length = cache_position.shape[0]
        seq_length = self.get_seq_length(layer_idx)
        return seq_length + query_length, 0

    cls.get_mask_sizes = get_mask_sizes
    return True, False


def _cache_class(model):
    return getattr(sys.modules[type(model).__module__], "UniversalTransformerCache")


def make_static_cache_cls(model):
    """s28_common.make_static_cache_cls, verbatim: preallocate each of the k*L K and V entries once
    and write in place, returning a length-`filled` view, so attention and get_mask_sizes ->
    get_seq_length see exactly the same tensors the stock cache would return."""
    UTC = _cache_class(model)

    class StaticUTCache(UTC):
        def __init__(self, n_entries, batch, max_len, n_kv_heads, head_dim, dtype, device):
            self.key_cache, self.value_cache, self.layers = [], [], []
            self._seen_tokens = 0
            self.max_cache_size = n_entries
            self.lens = [0] * n_entries
            self.max_len = max_len
            for _ in range(n_entries):
                self.key_cache.append(torch.empty(batch, n_kv_heads, max_len, head_dim,
                                                  dtype=dtype, device=device))
                self.value_cache.append(torch.empty(batch, n_kv_heads, max_len, head_dim,
                                                    dtype=dtype, device=device))

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            n = key_states.shape[2]
            s = self.lens[layer_idx]
            if s + n > self.max_len:
                raise RuntimeError("StaticUTCache overflow: %d + %d > %d" % (s, n, self.max_len))
            self.key_cache[layer_idx][:, :, s:s + n, :] = key_states
            self.value_cache[layer_idx][:, :, s:s + n, :] = value_states
            self.lens[layer_idx] = s + n
            self._seen_tokens = s + n
            return (self.key_cache[layer_idx][:, :, :s + n, :],
                    self.value_cache[layer_idx][:, :, :s + n, :])

        def get_seq_length(self, layer_idx=0):
            if layer_idx is None:
                layer_idx = 0
            if layer_idx < 0 or layer_idx >= len(self.lens):
                return 0
            return self.lens[layer_idx]

        def get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx)

        def clear(self):
            self.lens = [0] * len(self.lens)
            self._seen_tokens = 0

    return StaticUTCache


# ------------------------------------------------------------------ left padding (s13/s26/s28)
def leftpad(seqs, pad, dev):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, L - len(s):] = torch.tensor(s, dtype=torch.long)
        att[r, L - len(s):] = 1
    return ids.to(dev), att.to(dev), L


def strip_tail(ids_out, eos_ids):
    """s9a, verbatim: keep the eos token, drop everything after it."""
    out = []
    for t in ids_out:
        cut = len(t)
        for j, v in enumerate(t):
            if v in eos_ids:
                cut = j + 1
                break
        out.append(t[:cut])
    return out


def token_pieces(tok, vocab_size):
    return tok.batch_decode([[i] for i in range(vocab_size)], clean_up_tokenization_spaces=False)


# ------------------------------------------------------------------ forced continuation (s13)
WAIT_TEXT = {False: "\nWait,", True: " Wait,"}       # keyed by chat_template, s13_common.WAIT_TEXT
import re                                            # noqa: E402  (kept next to its only user)
RE_HASH_LINE = re.compile(r"####[ \t]*-?\$?[\d,]*\.?\d[\d,]*[ \t]*\r?\n")


def make_stopper(chat_template, pieces, eos_ids, task):
    """s13_common.make_stopper, verbatim, generalised from (gsm8k|math500) to the task table.

    f(tok_id, tail_str) -> reason|None. `tail_str` is the decoded tail of the row's last
    TAIL_WINDOW generated tokens BEFORE this candidate token.
    """
    if chat_template:
        stop_set = set(eos_ids)

        def stopper(x, tail_old):
            return ("tok_%d" % x) if x in stop_set else None
        return stopper

    eos = eos_ids[0]
    if task in ("gsm8k", "svamp"):          # SVAMP uses the GSM8K prompt, marker and stop strings
        def stopper_gsm(x, tail_old):                                  # S9d verbatim
            if x == eos:
                return "eos"
            new = tail_old + pieces[x]
            if RE_HASH_LINE.search(new) and not RE_HASH_LINE.search(tail_old):
                return "hash_line"
            if "\n\nQuestion" in new and "\n\nQuestion" not in tail_old:
                return "new_question"
            return None
        return stopper_gsm

    if task == "math500":
        def stopper_math(x, tail_old):
            if x == eos:
                return "eos"
            new = tail_old + pieces[x]
            if "Final Answer:" in new and "Final Answer:" not in tail_old:
                return "final_answer"
            if "\n\nProblem" in new and "\n\nProblem" not in tail_old:
                return "new_problem"
            return None
        return stopper_math

    # Every other task uses the CoT layout ("... So the answer is (b)." / "yes" / "True" / a number,
    # then "\n\nQ:" for the next question). The natural stop rule cuts at the stop strings; the
    # forced stopper, like GSM8K's hash-line rule, fires one token earlier, at the terminator of the
    # answer sentence, so "Wait," replaces the period and the chain reads "So the answer is (b)\nWait,".
    from ..tasks import task_cfg
    tc = task_cfg(task)
    marker = re.escape(tc["own_marker"])
    re_answer = re.compile(marker + r"[ \t]*\(?(?:[A-Fa-f]|yes|no|true|false|-?\$?\d[\d,]*(?:\.\d+)?)"
                           r"\)?[ \t]*[.\n]", re.IGNORECASE)
    stops = list(tc.get("stops") or [])

    def stopper_generic(x, tail_old):
        if x == eos:
            return "eos"
        new = tail_old + pieces[x]
        if re_answer.search(new) and not re_answer.search(tail_old):
            return "answer_sentence"
        for s in stops:
            if s in new and s not in tail_old:
                return "new_question"
        return None
    return stopper_generic


def new_forced_states(b):
    return [{"pending": [], "natural_stop_pos": None, "natural_stop_reason": None,
             "n_forced_continuations": 0, "forced_positions": [],
             "tail": deque(maxlen=TAIL_WINDOW), "off": 0} for _ in range(b)]


def copy_forced_states(states):
    """Independent copies of the per-row forced states, `tail` rebuilt as the bounded deque.

    decode works on copies and hands the copies back (it never writes through to what it was given),
    because a decode can RAISE: the caller retries an OOM at half the batch with the same rows, and
    in-place mutation would make that retry count the injections and positions of the failed attempt
    a second time and drain a `pending` wait the model never proposed. Copying also accepts a state
    read back from a checkpoint, whose `tail` is a plain list.
    """
    out = []
    for st in states:
        out.append({"pending": list(st.get("pending") or []),
                    "natural_stop_pos": st.get("natural_stop_pos"),
                    "natural_stop_reason": st.get("natural_stop_reason"),
                    "n_forced_continuations": int(st.get("n_forced_continuations") or 0),
                    "forced_positions": [list(p) for p in (st.get("forced_positions") or [])],
                    "tail": deque(st.get("tail") or (), maxlen=TAIL_WINDOW),
                    "off": int(st.get("off") or 0)})
    return out


# ------------------------------------------------------------------ the adapter
class OuroAdapter(Adapter):
    family = "ouro"

    def __init__(self, name, adapter_dir=None, dtype=torch.bfloat16,
                 mem_fraction=MEM_FRACTION, batch_cap=BATCH_CAP, reserve_gb=RESERVE_GB,
                 explicit_readout=True):
        if name not in OURO_CHECKPOINTS:
            raise KeyError("unknown Ouro checkpoint %r" % name)
        spec = OURO_CHECKPOINTS[name]
        self.name = name if adapter_dir is None else "%s+%s" % (name,
                                                                os.path.basename(adapter_dir))
        self.base_name = name
        self.repo = spec["repo"]
        self.chat_template = spec["chat"]
        self.depths = OURO_DEPTHS
        self.trained_depth = OURO_TRAINED_DEPTH
        self.adapter_dir = adapter_dir
        self.dtype = dtype
        self.mem_fraction = float(mem_fraction)
        self.batch_cap = int(batch_cap)
        self.reserve_gb = float(reserve_gb)
        self.explicit_readout = bool(explicit_readout)
        self.tok = None
        self.model = None
        self._k = None
        self._cache_factory = None
        self._pieces = None
        self._info = {}

    # ---------------------------------------------------------------- lifecycle
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
            attn_implementation="sdpa").to(DEV).eval()
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

    def set_depth(self, k):
        """s9b_common.set_steps: the step knob plus the explicit read-out."""
        m = self.model
        m.model.total_ut_steps = int(k)
        assert int(m.model.total_ut_steps) == int(k)
        m.early_exit_step = (int(k) - 1) if self.explicit_readout else None
        self._k = int(k)
        self._cache_factory = self._make_cache_factory(int(k))
        if self._pieces is None:
            self._pieces = token_pieces(self.tok, m.config.vocab_size)
        return {"total_ut_steps": int(m.model.total_ut_steps),
                "early_exit_step": m.early_exit_step,
                "early_exit_threshold": getattr(m, "early_exit_threshold", None)}

    # ---------------------------------------------------------------- shapes and cost
    def n_layers(self):
        return int(self.model.config.num_hidden_layers)

    def passes_per_token(self, k):
        return int(k) * int(self.model.config.num_hidden_layers)

    def kv_bytes_per_token(self, k):
        """2 (K,V) * kv heads * head_dim * itemsize * L * k (s28_common.bytes_per_row_token)."""
        cfg = self.model.config
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return int(2 * cfg.num_key_value_heads * cfg.head_dim * itemsize
                   * cfg.num_hidden_layers * int(k))

    def row_token_ceiling(self, k):
        """s28_common.row_token_ceiling, verbatim: the largest (row x token) KV allocation we may
        make right now, from MEASURED free memory."""
        free, total = torch.cuda.mem_get_info()
        ours = torch.cuda.memory_reserved()
        cap_left = self.mem_fraction * total - ours
        avail = min(float(free), max(0.0, cap_left)) - self.reserve_gb * 1024 ** 3
        return max(256, int(avail / self.kv_bytes_per_token(k)))

    def batch_for(self, k, max_seq):
        try:
            est = int(self.row_token_ceiling(k) / max(1, max_seq))
        except Exception:                                     # noqa: BLE001
            est = self.batch_cap
        return max(1, min(self.batch_cap, est)), est

    def _make_cache_factory(self, k):
        cls = make_static_cache_cls(self.model)
        cfg = self.model.config
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        dtype = next(self.model.parameters()).dtype
        dev = next(self.model.parameters()).device

        def factory(batch, max_len):
            return cls(cfg.num_hidden_layers * int(k), batch, max_len, nkv, hd, dtype, dev)
        return factory

    # ---------------------------------------------------------------- decoding
    @torch.no_grad()
    def decode(self, seqs, n_new, k, eos_ids, stop_strings=None, pieces=None, row_ids=None,
               stopper=None, wait_ids=None, states=None):
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
        cur = out.logits[:, -1, :].float().argmax(-1)
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
            cur = o.logits[:, -1, :].float().argmax(-1)
            del o
        dt = time.time() - t0
        del cache, ids, attn, cur
        torch.cuda.empty_cache()
        if stopper is None:
            return gen, dt
        for b, st in enumerate(states):
            st["off"] += len(gen[b])
        return gen, states, dt

    def wait_ids(self):
        return self.tok(WAIT_TEXT[self.chat_template], add_special_tokens=False)["input_ids"]

    def meta(self):
        d = {"name": self.name, "base_name": self.base_name, "family": self.family,
             "repo": self.repo, "chat_template": self.chat_template,
             "depths": list(self.depths), "trained_depth": self.trained_depth,
             "dtype": str(self.dtype), "mem_fraction": self.mem_fraction,
             "batch_cap": self.batch_cap, "reserve_gb": self.reserve_gb,
             "explicit_readout": self.explicit_readout,
             "wait_text": WAIT_TEXT[self.chat_template]}
        d.update(self._info)
        return d
