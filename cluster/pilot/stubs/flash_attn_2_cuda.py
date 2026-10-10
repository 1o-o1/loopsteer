"""Import-time stand-in for the FlashAttention 2 CUDA module, which has no wheel for the Spark (aarch64).
xllm/modules/fused_ops/attention/flash.py imports it unconditionally; K10 builds every IFM model with
causal_attn_backend=None (pure PyTorch attention), so no function of this module is ever called. Any
attribute access raises, so an accidental use of the flash path fails loudly instead of silently."""


def __getattr__(name):
    raise RuntimeError("flash_attn_2_cuda.%s called: FlashAttention 2 is not installed; K10 runs "
                       "the IFM models with causal_attn_backend=None" % name)
