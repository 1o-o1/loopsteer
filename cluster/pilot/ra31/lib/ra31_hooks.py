"""RA-31 unit hooks: the subset write edit at the last prompt token r, on the post-norm branch write, beside the
production registry and on top of the RA-28 unit hooks (lib/ra28_hooks.py, unchanged copy; subclassed here).

Ouro ends each branch in an RMSNorm (input_layernorm_2 after o_proj, post_attention_layernorm_2 after down_proj), so
the write added to the residual is w = g * u / rms(u), u = W x (x = down_proj input a, or o_proj input z), no bias,
rms(u) = sqrt(mean(u^2) + eps) over the full vector. The write decomposes exactly over the input units with the
full-vector rms held fixed:
    c_S = g * (W[:, S] x_S) / rms(u),     sum of c_S over a partition of the units = w.
Interpolated write on a unit set S (units outside S stay at alpha 1):
    w' = w + (alpha - 1) (c_S - c_S_clean)
c_S from the live run (its own x and rms), c_S_clean from clean A (its own x and rms), float32 math, written back in
the model dtype. alpha 1 adds 0 and writes w unchanged (bit-exact); at S = all, c_S = w up to rounding, so the edit
reduces to the whole-write interpolation w_clean + alpha (w - w_clean). The whole-write arm itself is the prod patch
edit (lerp, a = 1 - alpha, target clean), not this module. Heads: S = the o_proj input columns of the heads.

Hooks per layer l: down_proj and o_proj forward-pre hooks (record the input at r; RA-28 pattern), and forward
hooks on post_attention_layernorm_2 (MLP write) and input_layernorm_2 (attention write) that apply the edit.
"""
import torch

import ra28_hooks as UH


class Spec31(UH.Spec):
    """edit_mlp / edit_attn: {(t, l): (idx long, c_clean [d] float32)}; alpha: the scalar of the arm; T: the prompt
    length (every hooked call must have it; teacher-forced, no cache)."""

    def __init__(self, r, record=False, edit_mlp=None, edit_attn=None, alpha=1.0, T=None, grad_w=None):
        UH.Spec.__init__(self, r, record=record, alpha=alpha)
        self.edit_mlp = edit_mlp or {}
        self.edit_attn = edit_attn or {}
        self.T = None if T is None else int(T)
        self.grad_w = set(grad_w or ())    # (t, l): a zero gradient leaf added to the post-norm MLP write


def rms(u, eps):
    """u [..., d] -> sqrt(mean(u^2) + eps) in float32 (the OuroRMSNorm denominator)."""
    return torch.sqrt(u.float().pow(2).mean(-1) + eps)


def contrib(x, W, idx, u, g, eps):
    """Contribution of input units idx to the post-norm write at one position: g * (W[:, idx] x[idx]) / rms(u).
    x [n_in], W [d, n_in], u [d] (the full pre-norm vector), g [d]. float32 [d]."""
    W_ = W[:, idx].float()
    return g.float() * (W_ @ x[idx].float()) / rms(u, eps)


def edit_row(w, c_live, c_clean, alpha):
    """w [d] (model dtype) -> w + (alpha - 1) (c_live - c_clean), float32 math, model dtype out."""
    return (w.float() + (float(alpha) - 1.0) * (c_live - c_clean)).to(w.dtype)


def head_cols(heads, hd, device=None):
    """Head indices -> o_proj input columns (head h owns h * hd ... h * hd + hd - 1), sorted long tensor."""
    cols = [h * hd + j for h in sorted(int(x) for x in heads) for j in range(hd)]
    return torch.tensor(cols, dtype=torch.long, device=device)


class Units31(UH.Units):
    """RA-28 Units with the RA-31 subset write edit. The RA-28 holds, scaling, gradients and head blocks are refused
    (RA-31 never uses them), so a misbuilt spec cannot act silently."""

    def install(self, spec):
        if spec.hold_act or spec.hold_gate or spec.hold_up or spec.scale_act or spec.hold_heads or \
                spec.block_heads or spec.hold_norm or spec.grad:
            raise ValueError("RA-31 specs carry only edit_mlp, edit_attn, alpha, T and record")
        self.live = {"act": {}, "z": {}}
        self.leaves_w = {}
        self.n_edits = 0
        UH.Units.install(self, spec)
        for l, layer in enumerate(self.layers):
            self.handles.append(layer.post_attention_layernorm_2.register_forward_hook(
                self._norm_edit(l, "mlp", layer.mlp.down_proj, "act")))
            self.handles.append(layer.input_layernorm_2.register_forward_hook(
                self._norm_edit(l, "attn", layer.self_attn.o_proj, "z")))

    def _check_T(self, x):
        sp = self.spec
        if int(x.shape[0]) != 1:
            raise RuntimeError("RA-31 unit hooks run batch 1")
        if sp.T is not None and int(x.shape[1]) != sp.T:
            raise RuntimeError("RA-31 unit hook saw sequence length %d, expected the prompt length %d"
                               % (int(x.shape[1]), sp.T))

    def _in_pre(self, l, name, table):
        def hook(module, args):
            t = self.loop()
            if t is None:
                return None
            sp, r = self.spec, self.spec.r
            x = args[0]
            self._check_T(x)
            if (t, l) in getattr(sp, table):
                self.live[name][(t, l)] = x[0, r].detach()
            if sp.record:
                self.rec[name][(t, l)] = x[0, r].detach()
            return None
        return hook

    def _down_pre(self, l):
        return self._in_pre(l, "act", "edit_mlp")

    def _o_pre(self, l):
        return self._in_pre(l, "z", "edit_attn")

    def _norm_edit(self, l, branch, proj, name):
        table = "edit_mlp" if branch == "mlp" else "edit_attn"

        def hook(module, args, out):
            t = self.loop()
            if t is None:
                return None
            sp, r = self.spec, self.spec.r
            tab = getattr(sp, table)
            new = None
            if (t, l) in tab:
                idx, c_clean = tab[(t, l)]
                x = self.live[name].pop((t, l))
                u = args[0][0, r]
                c = contrib(x, proj.weight, idx, u, module.weight, UH.norm_eps(module))
                new = out.clone()
                new[0, r] = edit_row(out[0, r], c, c_clean.to(c.device), sp.alpha)
                self.n_edits += 1
            if branch == "mlp" and (t, l) in sp.grad_w:
                leaf = torch.zeros_like(out, requires_grad=True)
                self.leaves_w[(t, l)] = leaf
                new = (new if new is not None else out) + leaf
            return new
        return hook
