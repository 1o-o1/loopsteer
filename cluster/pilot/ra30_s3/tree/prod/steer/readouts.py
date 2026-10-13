"""Read-outs: what the model would predict from an intermediate state, and the records of it.

What this module measures and why. The mechanism questions (DESIGN 1.3, 7.3) need the prediction
at every loop, not only at the end: an edit can move the loop-j read-out and be overwritten by
loop j+1. A per-loop read-out is evidence only where the model was trained to emit it (Ouro exits)
or where a lens passed its evidence gate (PF12). This module holds:

- `Recorder`: the spec the registry uses to capture states, read-out statistics, exit-gate logits
  and attention mass at sites (DESIGN 8);
- `readout_stats`: per position z[targets], logsumexp, top-1, top-1 log-prob and entropy
  (k16_path_ouro.vals:108-112 plus entropy);
- `exit_pdf`: Ouro's exit distribution p_i = lambda_i prod_{j<i}(1 - lambda_j), p_K = remainder
  [read modeling_ouro base 723-733];
- `TunedLens` and `train_tuned_lens`: the tuned lens TL_s(h) = R_s(h + W_s h + b_s) [arXiv
  2303.08112 Eq.8; tuned-lens nn/lenses.py transform_hidden h + self[idx](h)], trained on
  KL(p_final || p_lens) [Eq.9; train_loop.py:398-437] with the reference optimizer
  [scripts/ingredients.py:201-263] (DESIGN 8.4).

The family read-out functions themselves (`binding.readout`, `binding.readout32`) live in the
bindings, because each needs its family's modules.
"""
import contextlib
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .hooks import HookRegistry, RowInfo, RowKey, SteerConfigError
from .sites import Component, Positions, Site, canonical

WHATS = ("state", "readout_stats", "gate", "attn_mass")


@dataclass
class Recorder:
    """What to capture at which sites (DESIGN 8). `positions` is the query set for attn_mass."""
    id: str
    sites: Tuple[Site, ...]
    positions: Positions
    what: str
    when: str = "pre"
    mode: str = ""
    targets: Optional[Dict[RowKey, List[int]]] = None
    key_positions: Optional[Positions] = None
    dtype: str = "float32"
    to_cpu: bool = True

    def __post_init__(self) -> None:
        if self.what not in WHATS:
            raise SteerConfigError("recorder %s: unknown kind %r" % (self.id, self.what))
        if self.when not in ("pre", "post"):
            raise SteerConfigError("recorder %s: when must be 'pre' or 'post'" % self.id)
        if self.what == "readout_stats" and not self.mode:
            raise SteerConfigError("recorder %s: readout_stats needs a read-out mode" % self.id)
        self.sites = tuple(self.sites)


@contextlib.contextmanager
def no_tf32() -> Iterator[None]:
    """TF32 off for a float32 read-out, previous values restored (DESIGN 22.2 `readout32`;
    k17_varbench_plan.md section 2, review MS2)."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def readout_stats(z: Tensor, targets: Sequence[int]) -> Dict[str, Tensor]:
    """Per position of z [n, V] (float32): z[targets], logsumexp, top-1, top-1 log-prob, entropy."""
    z = z.float()
    logp = z.log_softmax(-1)
    top1 = z.argmax(-1)
    out = {"lse": torch.logsumexp(z, -1), "top1": top1,
           "top1_logp": logp.gather(-1, top1[:, None])[:, 0],
           "entropy": -(logp.exp() * logp).sum(-1)}
    if len(targets):
        out["z_targets"] = z[:, list(targets)]
    return out


def exit_pdf(gate_logits: List[Tensor]) -> Tensor:
    """[read modeling_ouro base 723-733] gate logits per loop ([B, T, 1] each) -> pdf [B, T, K]."""
    pdf = []
    remaining = torch.ones_like(gate_logits[0].squeeze(-1))
    for idx, g in enumerate(gate_logits):
        lam = torch.sigmoid(g.squeeze(-1))
        if idx < len(gate_logits) - 1:
            pdf.append(lam * remaining)
            remaining = remaining * (1.0 - lam)
        else:
            pdf.append(remaining)
    return torch.stack(pdf, dim=-1)


# ------------------------------------------------------------------ tuned lens (DESIGN 8.4)
class TunedLens(torch.nn.Module):
    """One affine translator per site, zero-initialised, applied residually: h + W_s h + b_s
    [2303.08112 Eq.8 with A = I + W; nn/lenses.py:160-164, 306-311]. Translators are float32."""

    def __init__(self, sites: Sequence[Site], d: int):
        super(TunedLens, self).__init__()
        self.sites = tuple(sites)
        self.translators = torch.nn.ModuleDict()
        for s in self.sites:
            lin = torch.nn.Linear(d, d, bias=True)
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
            self.translators[s.key().replace(".", "__")] = lin
        self.meta = {}   # type: Dict[str, Any]

    def forward(self, site: Site, h: Tensor) -> Tensor:
        t = self.translators[site.key().replace(".", "__")]
        hf = h.float()
        return (hf + t(hf)).to(h.dtype)


@dataclass
class LensCorpus:
    """Token rows of the lens training corpus with their sources (path and sha256 each, PF12)."""
    rows: List[List[int]]
    keys: List[RowKey]
    sources: List[Dict[str, str]] = field(default_factory=list)

    def prefix_len(self) -> int:
        """Length of the longest common token prefix (the shared exemplar block, PF12)."""
        if not self.rows:
            return 0
        n = min(len(r) for r in self.rows)
        first = self.rows[0]
        for i in range(n):
            if any(r[i] != first[i] for r in self.rows):
                return i
        return n


LENS_DEFAULTS = {"optimizer": "sgd_nesterov", "lr_scale": 1.0, "momentum": 0.9,
                 "weight_decay": 1e-3, "grad_clip": 1.0, "val_frac": 0.2, "split_seed": 20260908}
_REQUIRED = ("num_steps", "tokens_per_step", "epochs")


def _lens_mode(binding: Any, k: int, cfg: Dict[str, Any]) -> str:
    """R_s per family (MF6): Ouro lm_head(rmsnorm(.)); raven the pure coda read-out; IFM the
    final-norm or coda lens named by cfg['readout_mode']."""
    if cfg.get("readout_mode"):
        return str(cfg["readout_mode"])
    fam = binding.family
    if fam == "ouro":
        return "logit_lens"
    if fam in ("huginn", "mcleish"):
        return "pure"
    raise SteerConfigError("IFM lens training needs cfg['readout_mode'] "
                           "('final_norm_lens' or 'coda_lens')")


def _check_lens_sites(binding: Any, k: int, sites: Sequence[Site]) -> None:
    """Translators sit at the loop ends of loops 1..k only, on the family tensor (DESIGN 8.4)."""
    geom = binding.geometry(k)
    for s in sites:
        c = canonical(geom, s)
        if binding.family == "ouro":
            ok = c.component == Component.RESID_POST and c.layer == geom.n_core - 1
        else:
            ok = c == canonical(geom, Site("loop", c.loop, None, Component.LOOP_END))
        if c.stage != "loop" or not ok:
            raise SteerConfigError("tuned-lens site %s is not a loop-end site of family %s "
                                   "(Ouro: loop{j}.L{L-1}.resid_post; raven/IFM: loop_end)"
                                   % (s.key(), binding.family))


def _kl(z_final: Tensor, z_lens: Tensor) -> Tensor:
    """Per-position KL(p_final || p_lens) [2303.08112 Eq.9; train_loop.py:432-434]."""
    lf = z_final.float().log_softmax(-1)
    return (lf.exp() * (lf - z_lens.float().log_softmax(-1))).sum(-1)


class _Frozen(object):
    """Turns off requires_grad on the model's parameters and restores the flags."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.flags = []   # type: List[Tuple[torch.nn.Parameter, bool]]

    def __enter__(self) -> "_Frozen":
        for p in self.module.parameters():
            self.flags.append((p, p.requires_grad))
            p.requires_grad_(False)
        return self

    def __exit__(self, *exc: Any) -> None:
        for p, f in self.flags:
            p.requires_grad_(f)


def _row_pass(binding: Any, reg: HookRegistry, k: int, ids: List[int], key: RowKey,
              sites: Sequence[Site], rec_id: str) -> Tuple[Tensor, Dict[Site, Tensor]]:
    """One no-grad forward: final logits [1, T, V] and the states at the lens sites [1, T, d]."""
    dev = binding.device()
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        z = binding.forward(x, k, [key])
    rec = reg.take_records()
    states = {s: rec[(key, rec_id, s.key())][None].to(dev) for s in sites}
    return z, states


def train_tuned_lens(binding: Any, k: int, corpus: LensCorpus, sites: Sequence[Site],
                     cfg: Dict[str, Any]) -> TunedLens:
    """Train one translator per loop-end site (DESIGN 8.4). REQUIRED cfg: num_steps,
    tokens_per_step (MF4) and epochs (passes allowed over the training rows); the other fields
    default to the tuned-lens code values (LENS_DEFAULTS). Raises when the data cannot fill the
    requested steps (scripts_train_loop.py:279-283). Returns the lens of the epoch with the lowest
    validation KL; `lens.meta` records tokens used, passes, the shared prefix length and the
    validation KL and top-1 agreement per site and epoch (the PF12 lens report inputs)."""
    missing = [n for n in _REQUIRED if cfg.get(n) is None]
    if missing:
        raise SteerConfigError("train_tuned_lens: REQUIRED cfg fields missing: %s" % missing)
    c = dict(LENS_DEFAULTS, **cfg)
    if c["optimizer"] != "sgd_nesterov":
        raise SteerConfigError("optimizer %r is not implemented (sgd_nesterov only)"
                               % c["optimizer"])
    _check_lens_sites(binding, k, sites)
    geom = binding.geometry(k)
    sites = [canonical(geom, s) for s in sites]          # raven loop_end is an alias (DESIGN 8.4)
    mode = _lens_mode(binding, k, c)
    prefix = corpus.prefix_len()
    n = len(corpus.rows)
    perm = np.random.default_rng(int(c["split_seed"])).permutation(n)
    n_val = int(n * float(c["val_frac"]))
    val_idx, train_idx = [int(i) for i in perm[:n_val]], [int(i) for i in perm[n_val:]]
    tok_row = {i: max(0, len(corpus.rows[i]) - prefix) for i in train_idx}
    tps, epochs, n_steps = int(c["tokens_per_step"]), int(c["epochs"]), int(c["num_steps"])
    capacity = _lens_capacity([tok_row[i] for i in train_idx], tps, epochs)
    if capacity < n_steps:
        raise SteerConfigError("Can only take %d steps on the corpus with tokens_per_step=%d "
                               "and epochs=%d. Requested %d steps."
                               % (capacity, tps, epochs, n_steps))
    lens = TunedLens(sites, geom.d_model).to(binding.device())
    beta = float(c["momentum"])
    opt = torch.optim.SGD(lens.parameters(), lr=float(c["lr_scale"]) * (1 - beta), momentum=beta,
                          nesterov=True, weight_decay=float(c["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, (n_steps - s) / n_steps))
    rec = Recorder(id="tuned_lens_states", sites=tuple(sites), positions=Positions("all"),
                   what="state", dtype=str(binding.dtype()).replace("torch.", ""))
    reg = HookRegistry(binding, k)
    reg.add_recorder(rec)
    reg.install()
    hist, best, step, tokens, passes = [], None, 0, 0, 0
    acc = 0          # loss tokens in the pending step's gradient; carried across passes
    try:
        with _Frozen(binding.model_module()):
            while step < n_steps:
                if passes >= epochs:                     # guarded by _lens_capacity above
                    raise SteerConfigError("train_tuned_lens: %d passes took %d of %d steps"
                                           % (passes, step, n_steps))
                passes += 1
                for i in train_idx:
                    if step >= n_steps:
                        break
                    ids, key = corpus.rows[i], corpus.keys[i]
                    if tok_row[i] == 0:
                        continue
                    with reg.batch([_row(key, ids)], [0]):
                        z, states = _row_pass(binding, reg, k, ids, key, sites, rec.id)
                    loss = 0.0
                    for s in sites:
                        z_l = binding.readout(s.loop, lens(s, states[s]), mode)
                        loss = loss + _kl(z[0, prefix:], z_l[0, prefix:]).sum()
                    loss.backward()                      # token sum; the step divides by acc
                    acc += tok_row[i]
                    tokens += tok_row[i]
                    if acc >= tps:
                        # the step gradient is the per-token mean over the tokens it holds
                        # [train_loop.py:398-437: token-mean loss over each step's tokens]
                        for p in lens.parameters():
                            if p.grad is not None:
                                p.grad.div_(float(acc))
                        torch.nn.utils.clip_grad_norm_(lens.parameters(), float(c["grad_clip"]))
                        opt.step()
                        sched.step()
                        opt.zero_grad(set_to_none=True)
                        step += 1
                        acc = 0
                ev = _evaluate(binding, reg, k, corpus, val_idx, sites, lens, mode, prefix, rec.id)
                hist.append(dict(ev, epoch=passes, steps=step))
                total = sum(ev["val_kl"].values())
                if best is None or total < best[0]:
                    best = (total, passes, copy.deepcopy(lens.state_dict()))
    finally:
        reg.remove()
    lens.load_state_dict(best[2])
    lens.meta = {"tokens_used": tokens, "passes": passes, "prefix_len": prefix,
                 "n_train_rows": len(train_idx), "n_val_rows": len(val_idx),
                 "best_epoch": best[1], "history": hist, "readout_mode": mode,
                 "sources": list(corpus.sources), "cfg": c,
                 "deviation": "corpus is task text, not pretraining text (DESIGN 8.4, MF4)"}
    return lens


def _lens_capacity(tok_rows: Sequence[int], tokens_per_step: int, epochs: int) -> int:
    """Optimizer steps the training loop takes in `epochs` passes over rows of these loss-token
    counts: whole rows accumulate, a step fires once the pending count reaches tokens_per_step,
    and the pending count carries across passes (the loop of train_tuned_lens)."""
    acc, steps = 0, 0
    for _ in range(int(epochs)):
        for t in tok_rows:
            if t == 0:
                continue
            acc += int(t)
            if acc >= int(tokens_per_step):
                steps += 1
                acc = 0
    return steps


def _row(key: RowKey, ids: List[int]) -> RowInfo:
    return RowInfo(key=key, prompt_len=len(ids))


def _evaluate(binding: Any, reg: HookRegistry, k: int, corpus: LensCorpus, idx: List[int],
              sites: Sequence[Site], lens: TunedLens, mode: str, prefix: int,
              rec_id: str) -> Dict[str, Dict[str, float]]:
    """Validation KL per site and top-1 agreement of the lens with the model's final output."""
    kl = {s.key(): 0.0 for s in sites}
    agree = {s.key(): 0.0 for s in sites}
    n_tok = 0
    with torch.no_grad():
        for i in idx:
            ids, key = corpus.rows[i], corpus.keys[i]
            if len(ids) <= prefix:
                continue
            with reg.batch([_row(key, ids)], [0]):
                z, states = _row_pass(binding, reg, k, ids, key, sites, rec_id)
            zf = z[0, prefix:]
            n_tok += zf.shape[0]
            for s in sites:
                zl = binding.readout(s.loop, lens(s, states[s]), mode)[0, prefix:]
                kl[s.key()] += float(_kl(zf, zl).sum())
                agree[s.key()] += float((zl.argmax(-1) == zf.argmax(-1)).sum())
    n = max(n_tok, 1)
    return {"val_kl": {s: v / n for s, v in kl.items()},
            "val_top1_agreement": {s: v / n for s, v in agree.items()}, "val_tokens": n_tok}

