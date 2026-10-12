"""Huginn-0125 adapter. Everything decoding-related is in raven.py; this file is the registry row.

Facts of record, with their source:
  * structure prelude 2 / core 4 / coda 2, so layer passes per token = 4 + 4k
    (s9c_common's header, confirmed from config.json: n_layers_in_prelude 2,
    n_layers_in_recurrent_block 4, n_layers_in_coda 2)
  * depth set {1, 2, 4, 8, 16, 32}, an exponential grid
  * trained depth 32 (config mean_recurrence = 32), which is past its optimum on GSM8K: 16 steps
    match 32 on average and beat it on about half the problems
  * PAD_ID 65509 and STOP_IDS [65504, 65505, 65508] (begin_text, end_text, end_turn) from
    s9c_common, which read them out of `_get_stops` line 1605
  * `compile_mask` is NEVER called: the released `compile_mask` has an operator-precedence bug that
    makes the prelude non-causal on padded batches when embed_inputs is used, and the normal forward
    avoids it because it never calls compile_mask, so any per-step analysis must build the prelude
    the forward() way. raven.py only ever calls `model(...)`, i.e. the normal forward, and supplies
    its own mask through the attention patch.
"""
from .raven import HUGINN_PAD_ID, HUGINN_STOP_IDS, RavenAdapter

REPO = "tomg-group-umd/huginn-0125"
DEPTHS = (1, 2, 4, 8, 16, 32)
TRAINED_DEPTH = 32


class HuginnAdapter(RavenAdapter):
    family = "huginn"

    def __init__(self, name="huginn_0125", **kw):
        super(HuginnAdapter, self).__init__(
            name=name, repo=REPO, depths=DEPTHS, trained_depth=TRAINED_DEPTH,
            pad_id=HUGINN_PAD_ID, stop_ids=HUGINN_STOP_IDS, **kw)
