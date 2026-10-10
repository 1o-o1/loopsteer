"""McLeish Recurrent-Llama-3.2 adapter (the retrofit S9f measured).

Facts of record:
  * repo `smcleish/Recurrent-Llama-3.2-train-recurrence-32`, trained at recurrence 32
  * structure prelude 4 / core 6 / coda 4, so layer passes per token = 8 + 6k, and the static cache
    needs max_num_steps = 6k + 8 (s9f_common.make_cache's note: the model's own
    `_prep_generate_args` would have used 4 + 4*num_steps, which is wrong for a 6-layer core).
    raven.py sizes the cache from `passes_per_token`, which reads the config, so it is right for
    both checkpoints.
  * depth set {1, 2, 4, 8} on every task plus {16, 32} on GSM8K: the loop axis saturates at k=4
    despite training at recurrence 32 (eight times the recurrent compute buys one point), which is
    why the extra depths are GSM8K-only.
  * prompts use `add_special_tokens=True`, following the model card's own usage example
    (`tokenizer.encode(..., add_special_tokens=True)`), which prepends <|begin_of_text|>
    (s9f_common.build_prompts). This is the one tokenisation difference from the Ouro path and it is
    recorded in every run meta as `add_special_tokens`.
  * no padding mask on the stock forward, so batched decoding needs raven.py's attention patch;
    even an UNPADDED equal-length batch of 2 is measured as not token-identical to batch 1 at k=4
    (2 of 8 rows diverge), which is a bf16 batch-width effect, not a mask problem.
"""
from .raven import RavenAdapter

REPO = "smcleish/Recurrent-Llama-3.2-train-recurrence-32"
DEPTHS = (1, 2, 4, 8, 16, 32)
DEPTHS_DEFAULT = (1, 2, 4, 8)
DEPTHS_GSM8K_EXTRA = (16, 32)
TRAINED_DEPTH = 32


class McLeishAdapter(RavenAdapter):
    family = "mcleish"
    #: the model card's usage example encodes with add_special_tokens=True (s9f_common)
    add_special_tokens = True

    def __init__(self, name="mcleish_llama32_r32", **kw):
        super(McLeishAdapter, self).__init__(
            name=name, repo=REPO, depths=DEPTHS, trained_depth=TRAINED_DEPTH, **kw)

    def depths_for(self, task):
        return tuple(DEPTHS_DEFAULT) + (tuple(DEPTHS_GSM8K_EXTRA) if task == "gsm8k" else ())
