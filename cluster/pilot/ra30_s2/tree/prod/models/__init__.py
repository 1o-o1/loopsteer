"""Model registry. `get(name)` returns an unloaded adapter; `depths_for(name, task)` the depth set.

Names of record (one per (model, checkpoint) the paper reports):
  ouro_1_4b_base   ouro_1_4b_think   ouro_2_6b_base   ouro_2_6b_think
  huginn_0125      mcleish_llama32_r32
A LoRA arm is `ouro_1_4b_base+<dir name>`, requested as `--model=ouro_1_4b_base --adapter=<path>`
with the S32 tag convention.
"""
from .huginn import DEPTHS as HUGINN_DEPTHS, HuginnAdapter
from .mcleish import DEPTHS_DEFAULT as MCLEISH_DEPTHS, DEPTHS_GSM8K_EXTRA, McLeishAdapter
from .ouro import OURO_CHECKPOINTS, OURO_DEPTHS, OuroAdapter

MODEL_ORDER = ["ouro_1_4b_base", "ouro_1_4b_think", "ouro_2_6b_base", "ouro_2_6b_think",
               "huginn_0125", "mcleish_llama32_r32"]

FAMILY = {}
for _n in OURO_CHECKPOINTS:
    FAMILY[_n] = "ouro"
FAMILY["huginn_0125"] = "huginn"
FAMILY["mcleish_llama32_r32"] = "mcleish"


def get(name, adapter_dir=None, **kw):
    if name in OURO_CHECKPOINTS:
        return OuroAdapter(name, adapter_dir=adapter_dir, **kw)
    if name == "huginn_0125":
        if adapter_dir:
            raise ValueError("no LoRA arms exist for Huginn")
        return HuginnAdapter(**kw)
    if name == "mcleish_llama32_r32":
        if adapter_dir:
            raise ValueError("no LoRA arms exist for McLeish")
        return McLeishAdapter(**kw)
    raise KeyError("unknown model %r; known: %s" % (name, ", ".join(MODEL_ORDER)))


def depths_for(name, task, cfg=None):
    """The depth set of record: Ouro {1,2,3,4}; McLeish {1,2,4,8} plus {16,32} on GSM8K; Huginn
    {1,2,4,8,16,32}, natural stop only (the forced block is restricted to the Ouro checkpoints by
    config.FORCED_BLOCK).

    The sets live in `prod/config.py` so the command line can override them; the per-family
    constants below stay as the adapters' own record of what the checkpoint supports and are what a
    name missing from the config falls back to.
    """
    from .. import config as cfgmod
    ks = cfgmod.depths_for(name, task, cfg)
    if ks:
        return ks
    if name in OURO_CHECKPOINTS:
        return tuple(OURO_DEPTHS)
    if name == "huginn_0125":
        return tuple(HUGINN_DEPTHS)
    if name == "mcleish_llama32_r32":
        return tuple(MCLEISH_DEPTHS) + (tuple(DEPTHS_GSM8K_EXTRA) if task == "gsm8k" else ())
    raise KeyError(name)


def uses_chat_template(name):
    return bool(OURO_CHECKPOINTS.get(name, {}).get("chat", False))
