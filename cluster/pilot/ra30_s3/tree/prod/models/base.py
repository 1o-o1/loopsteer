"""The one interface every family adapter implements.

`generate.py` and the gates talk only to this interface, so adding a checkpoint is a new adapter and
nothing else. A depth is always the number of loop executions k: Ouro's `total_ut_steps`, Huginn's
and McLeish's `num_steps`.
"""


class Adapter(object):
    #: registry name, e.g. "ouro_1_4b_base"
    name = None
    #: "ouro" | "huginn" | "mcleish"
    family = None
    #: HF repo id
    repo = None
    #: True when prompts go through the chat template with exemplars in the user turn
    chat_template = False
    #: the depths this adapter accepts
    depths = ()
    #: trained depth of record (from the card or the paper)
    trained_depth = None

    # ---------------------------------------------------------------- lifecycle
    def load(self):
        """Load tokenizer and model, apply every required patch, return self."""
        raise NotImplementedError

    def set_depth(self, k):
        """Set the loop count and the read-out. Returns a dict recorded in the run meta."""
        raise NotImplementedError

    # ---------------------------------------------------------------- shapes and cost
    def n_layers(self):
        """Layers executed per loop step (Ouro) -- see passes_per_token for the family formula."""
        raise NotImplementedError

    def passes_per_token(self, k):
        """Layer passes per generated token at depth k.

        Ouro:    k * L                       (L = num_hidden_layers)
        raven:   prelude + k * core + coda   (s9c_common.layer_passes, s9f_common.passes_per_token)
        """
        raise NotImplementedError

    def kv_bytes_per_token(self, k):
        """Bytes of KV cache a single token occupies at depth k, in the run's dtype."""
        raise NotImplementedError

    def batch_for(self, k, max_seq):
        """(batch, unclamped estimate) from the MEASURED free memory (s28_common.batch_for)."""
        raise NotImplementedError

    # ---------------------------------------------------------------- decoding
    def decode(self, seqs, n_new, k, eos_ids, stop_strings=None, pieces=None, row_ids=None):
        """Greedy decode a batch of prompts. Returns (list of generated id lists, seconds).

        `row_ids` are the PROBLEM indices of the rows, needed by the raven families to draw the
        per-example initial latent state, and ignored by Ouro.
        """
        raise NotImplementedError

    def meta(self):
        """Everything about this adapter that belongs in the run meta."""
        raise NotImplementedError
