"""RA-28 unit hooks: neuron, head and final-norm level reads and edits on a bound Ouro model, installed beside the
production registry (prod/steer, unchanged) for one forward and removed after it.

The prod registry keeps its role: the donor and random patches at loop1.loop_end, the full MLP and attention
holds (interventions.freeze_spec), the loop-end recorders and the call counts. These hooks read the loop index from
the registry's live forward context (binding.reg.ctx.loop, set by the transcribed layer forward) and act only at
the last prompt token r unless a spec says otherwise. A hook with nothing to do returns None, so the module output
is the very same tensor (bit-exact when idle).

Hook points per layer l (OuroMLP: down_proj(SiLU(gate_proj(x)) * up_proj(x)); OuroAttention: o_proj(z)):
- gate_proj / up_proj forward hooks: record or hold the gate and up values (gate-only and up-only holds);
- down_proj forward-pre hook: record, hold or scale the activations a (the down_proj input), gradient leaf;
- down_proj forward hook: record u = down_proj(a) (the input of post_attention_layernorm_2);
- o_proj forward-pre hook: record or hold the per-head outputs z (the o_proj input), gradient leaf;
- o_proj forward hook: record o = o_proj(z) (the input of input_layernorm_2);
- self_attn forward-pre hook (kwargs): block attention from r to the source span in listed heads (per-head copy of
  the eager float mask, the listed entries assigned finfo.min, as the prod knockout does for all heads);
- model.norm forward-pre hook: gradient leaf at h_t; forward hook: record h_t and its inverse RMS, or hold the
  final-norm scale at its clean value (output times inv_rms_clean / inv_rms_now at r).
"""
import torch


class Spec(object):
    """What the unit hooks do in one forward. Keys of the dicts are (loop, layer) except hold_norm (loop)."""

    def __init__(self, r, record=False, grad=False, hold_act=None, hold_gate=None, hold_up=None, scale_act=None,
                 alpha=1.0, scale_pos="last", hold_heads=None, block_heads=None, span=(), hold_norm=None):
        self.r = int(r)
        self.record = bool(record)
        self.grad = bool(grad)
        self.hold_act = hold_act or {}       # (t, l) -> (idx long, values [n] model dtype)
        self.hold_gate = hold_gate or {}
        self.hold_up = hold_up or {}
        self.scale_act = scale_act or {}     # (t, l) -> idx long
        self.alpha = float(alpha)
        if scale_pos not in ("last", "all"):
            raise ValueError("scale_pos must be last or all")
        self.scale_pos = scale_pos
        self.hold_heads = hold_heads or {}   # (t, l) -> (heads long, values [n, hd] model dtype)
        self.block_heads = block_heads or {}  # (t, l) -> list of heads
        self.span = tuple(int(p) for p in span)
        self.hold_norm = hold_norm or {}     # t -> inv_rms_clean (float32 0-d)


REC_KEYS = ("gate", "up", "act", "u_mlp", "z", "o_attn")


def norm_eps(module):
    """The epsilon of an RMSNorm module (variance_epsilon in OuroRMSNorm); raises when absent."""
    for name in ("variance_epsilon", "eps"):
        v = getattr(module, name, None)
        if v is not None:
            return float(v)
    raise AttributeError("no epsilon on %s" % type(module).__name__)


class Units(object):
    def __init__(self, binding):
        m = binding.model_module()
        self.b = binding
        self.model = m
        self.layers = list(m.model.layers)
        self.final_norm = m.model.norm
        cfg = m.config
        self.L = int(cfg.num_hidden_layers)
        self.I = int(cfg.intermediate_size)
        self.H = int(cfg.num_attention_heads)
        self.hd = int(cfg.head_dim)
        self.d = int(cfg.hidden_size)
        self.handles = []
        self.spec = None
        self.rec = None
        self.leaves = None
        self.norm_eps = norm_eps(self.final_norm)

    # ---------------------------------------------------------------- lifecycle
    def install(self, spec):
        if self.handles:
            raise RuntimeError("unit hooks already installed")
        self.spec = spec
        self.rec = {k: {} for k in REC_KEYS}
        self.rec["h"] = {}
        self.rec["inv"] = {}
        self.leaves = {"act": {}, "z": {}, "h": {}}
        h = self.handles
        for l, layer in enumerate(self.layers):
            mlp, att = layer.mlp, layer.self_attn
            h.append(mlp.gate_proj.register_forward_hook(self._branch_hook(l, "gate", spec.hold_gate)))
            h.append(mlp.up_proj.register_forward_hook(self._branch_hook(l, "up", spec.hold_up)))
            h.append(mlp.down_proj.register_forward_pre_hook(self._down_pre(l)))
            h.append(mlp.down_proj.register_forward_hook(self._out_rec(l, "u_mlp")))
            h.append(att.o_proj.register_forward_pre_hook(self._o_pre(l)))
            h.append(att.o_proj.register_forward_hook(self._out_rec(l, "o_attn")))
            if spec.block_heads:
                h.append(att.register_forward_pre_hook(self._attn_pre(l), with_kwargs=True))
        h.append(self.final_norm.register_forward_pre_hook(self._norm_pre))
        h.append(self.final_norm.register_forward_hook(self._norm_post))

    def remove(self):
        for hd in self.handles:
            hd.remove()
        self.handles = []

    def loop(self):
        reg = self.b.reg
        if reg is None or not reg.live():
            return None
        return int(reg.ctx.loop)

    def take(self, K):
        """Records as float32 CPU tensors: gate, up, act [K, L, I]; u_mlp, o_attn [K, L, d]; z [K, L, H*hd];
        h [K, d]; inv [K]. Missing (loop, layer) cells raise."""
        out = {}
        for k in REC_KEYS:
            if not self.rec[k]:
                continue
            out[k] = torch.stack([torch.stack([self.rec[k][(t, l)] for l in range(self.L)])
                                  for t in range(1, K + 1)]).float().cpu()
        if self.rec["h"]:
            out["h"] = torch.stack([self.rec["h"][t] for t in range(1, K + 1)]).float().cpu()
            out["inv"] = torch.stack([self.rec["inv"][t] for t in range(1, K + 1)]).float().cpu()
        return out

    # ---------------------------------------------------------------- hooks
    def _branch_hook(self, l, name, holds):
        def hook(module, args, out):
            t = self.loop()
            if t is None:
                return None
            sp, r = self.spec, self.spec.r
            new = None
            if (t, l) in holds:
                idx, vals = holds[(t, l)]
                new = out.clone()
                new[0, r, idx] = vals
            if sp.record:
                self.rec[name][(t, l)] = (new if new is not None else out)[0, r].detach()
            return new
        return hook

    def _down_pre(self, l):
        def hook(module, args):
            t = self.loop()
            if t is None:
                return None
            sp, r = self.spec, self.spec.r
            x = args[0]
            new = None
            if (t, l) in sp.hold_act:
                idx, vals = sp.hold_act[(t, l)]
                new = x.clone()
                new[0, r, idx] = vals
            if (t, l) in sp.scale_act:
                idx = sp.scale_act[(t, l)]
                new = (new if new is not None else x).clone()
                if sp.scale_pos == "last":
                    new[0, r, idx] = (new[0, r, idx].float() * sp.alpha).to(new.dtype)
                else:
                    new[0, :, idx] = (new[0, :, idx].float() * sp.alpha).to(new.dtype)
            cur = new if new is not None else x
            if sp.record:
                self.rec["act"][(t, l)] = cur[0, r].detach()
            if sp.grad:
                leaf = torch.zeros_like(cur, requires_grad=True)
                self.leaves["act"][(t, l)] = leaf
                cur = cur + leaf
                new = cur
            return None if new is None else (new,) + tuple(args[1:])
        return hook

    def _out_rec(self, l, name):
        def hook(module, args, out):
            t = self.loop()
            if t is None or not self.spec.record:
                return None
            self.rec[name][(t, l)] = out[0, self.spec.r].detach()
            return None
        return hook

    def _o_pre(self, l):
        def hook(module, args):
            t = self.loop()
            if t is None:
                return None
            sp, r = self.spec, self.spec.r
            x = args[0]
            new = None
            if (t, l) in sp.hold_heads:
                heads, vals = sp.hold_heads[(t, l)]
                new = x.clone()
                B, T = new.shape[0], new.shape[1]
                v = new.view(B, T, self.H, self.hd)
                v[0, r, heads] = vals
            cur = new if new is not None else x
            if sp.record:
                self.rec["z"][(t, l)] = cur[0, r].detach()
            if sp.grad:
                leaf = torch.zeros_like(cur, requires_grad=True)
                self.leaves["z"][(t, l)] = leaf
                cur = cur + leaf
                new = cur
            return None if new is None else (new,) + tuple(args[1:])
        return hook

    def _attn_pre(self, l):
        def hook(module, args, kwargs):
            t = self.loop()
            if t is None or (t, l) not in self.spec.block_heads:
                return None
            mask = kwargs.get("attention_mask")
            if not isinstance(mask, torch.Tensor) or not torch.is_floating_point(mask) or mask.dim() != 4:
                raise RuntimeError("head block needs the eager float 4D mask, got %r" % (type(mask),))
            sp = self.spec
            if not sp.span or sp.r in sp.span:
                raise RuntimeError("head block needs a source span that excludes r")
            B, _, Tq, KV = mask.shape
            new = mask.expand(B, self.H, Tq, KV).clone()
            neg = torch.finfo(new.dtype).min
            span = torch.tensor(sp.span, dtype=torch.long, device=new.device)
            for h in self.spec.block_heads[(t, l)]:
                new[0, int(h), sp.r, span] = neg
            if not bool((new[0, :, sp.r] == 0).any(-1).all()):
                raise RuntimeError("head block leaves query r with no allowed key")
            kw = dict(kwargs)
            kw["attention_mask"] = new
            return args, kw
        return hook

    def _norm_pre(self, module, args):
        t = self.loop()
        if t is None or not self.spec.grad:
            return None
        leaf = torch.zeros_like(args[0], requires_grad=True)
        self.leaves["h"][t] = leaf
        return (args[0] + leaf,) + tuple(args[1:])

    def _inv(self, x_row):
        return torch.rsqrt(x_row.float().pow(2).mean(-1) + self.norm_eps)

    def _norm_post(self, module, args, out):
        t = self.loop()
        if t is None:
            return None
        sp, r = self.spec, self.spec.r
        x = args[0]
        if sp.record:
            self.rec["h"][t] = x[0, r].detach()
            self.rec["inv"][t] = self._inv(x[0, r].detach())
        if t in sp.hold_norm:
            ratio = sp.hold_norm[t].to(device=x.device, dtype=torch.float32) / self._inv(x[0, r].detach())
            new = out.clone()
            new[0, r] = (out[0, r].float() * ratio).to(out.dtype)
            return new
        return None

    # ---------------------------------------------------------------- spec builders
    def slots_to_hold(self, slots, rec, name, dev, dtype):
        """[(t, l, j)] -> {(t, l): (idx, values)} with values = rec[name][t - 1, l, idx] in the model dtype."""
        g = {}
        for t, l, j in slots:
            g.setdefault((t, l), []).append(j)
        out = {}
        for key, js in g.items():
            idx = torch.tensor(sorted(js), dtype=torch.long, device=dev)
            vals = rec[name][key[0] - 1, key[1]][idx.cpu()].to(device=dev, dtype=dtype)
            out[key] = (idx, vals)
        return out

    def slots_to_scale(self, slots, dev):
        g = {}
        for t, l, j in slots:
            g.setdefault((t, l), []).append(j)
        return {key: torch.tensor(sorted(js), dtype=torch.long, device=dev) for key, js in g.items()}

    def heads_to_hold(self, slots, rec, dev, dtype):
        g = {}
        for t, l, h in slots:
            g.setdefault((t, l), []).append(h)
        out = {}
        for key, hs in g.items():
            hs = sorted(hs)
            z = rec["z"][key[0] - 1, key[1]].view(self.H, self.hd)
            out[key] = (torch.tensor(hs, dtype=torch.long, device=dev),
                        z[torch.tensor(hs, dtype=torch.long)].to(device=dev, dtype=dtype))
        return out

    def heads_to_block(self, slots):
        g = {}
        for t, l, h in slots:
            g.setdefault((t, l), []).append(int(h))
        return {key: sorted(v) for key, v in g.items()}
