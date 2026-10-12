"""The hook registry: one object that holds every edit, knockout, skip, recorder, logits processor
and gradient leaf of a run, and that the family bindings call from inside the model forward.

What this module measures and why. A steering result is only interpretable when the edit acts
exactly where it is said to act and nowhere else (critic B4; DESIGN 0 rule 9). The registry
enforces that contract (DESIGN 5):

- an untouched site returns the very same tensor object, so neutral hooks are bit-exact;
- an edit selects (row, absolute position) by a pure function of the row key, the position, the
  site and the loop (DESIGN 4.4), never by a call count, so cached decode, re-prefilled waves and
  the forced read-out are steered identically;
- edit math runs in float32 and is written back in the model dtype (k15_smoke2.apply_entries
  :89-105), into one clone per site call;
- every edited position adds its absolute ||y - x||_2, y as written in the model dtype, to the
  strength statistics (P1, audit C3);
- knockouts assign finfo(dtype).min to mask entries (never add it), with query rows taken from
  `cache_position` (MF9, S5) and a full-row guard (S6);
- per-forward call counts are asserted at `end_forward`, which clears `ctx.store` last (S2);
- calls made while a read-out runs never edit, record or count (S10).
"""
import contextlib
import hashlib
import inspect
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .sites import (CallCountError, Component, Geometry, Positions, PromptMismatch, Site,
                    SourceMismatch, SteerConfigError, SteerError, UnsupportedSite, canonical)

__all__ = ["RowKey", "SteerError", "SteerConfigError", "UnsupportedSite", "SourceMismatch",
           "PromptMismatch", "CallCountError", "RowInfo", "ForwardCtx", "FamilyBinding",
           "HookRegistry", "GradLeaves", "normalized_source", "source_sha256"]

RowKey = Union[int, str]
#: one (stage, loop, layer) call of a block
Call = Tuple[str, Optional[int], int]
RECORDER_KINDS = ("state", "readout_stats", "gate", "attn_mass")
_MISSING = object()


# ------------------------------------------------------------------ rows and forward context
@dataclass
class RowInfo:
    """One batch row. `key`: prod `rws[i]["idx"]` (task-global row index, PF1) or the item id."""
    key: RowKey
    prompt_len: int
    spans: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardCtx:
    """Per-forward state the bindings and the interventions read (DESIGN 5).

    `rows`, `pad` and `cache_position` are the batch table of the forward; logits processors read
    the row keys there (DESIGN 13.4 sz rows).
    """
    B: int
    T: int
    abs_pos: Tensor
    valid: Tensor
    attn_2d: Optional[Tensor]
    stage: str = "input"
    loop: int = 0
    layer: int = -1
    in_readout: bool = False
    freqs: Any = None
    store: Dict[Any, Any] = field(default_factory=dict)
    rows: List[RowInfo] = field(default_factory=list)
    pad: List[int] = field(default_factory=list)
    cache_position: Optional[Tensor] = None


# ------------------------------------------------------------------ source guard (DESIGN 6)
def normalized_source(fn: Callable) -> str:
    """`inspect.getsource` with trailing whitespace stripped and blank and comment-only lines
    dropped (DESIGN 6 hash rule)."""
    out = []
    for ln in inspect.getsource(fn).splitlines():
        s = ln.rstrip()
        if s.strip() and not s.strip().startswith("#"):
            out.append(s)
    return "\n".join(out)


def source_sha256(fn: Callable) -> str:
    return hashlib.sha256(normalized_source(fn).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ binding interface (DESIGN 5)
class FamilyBinding(object):
    """Implemented in adapters/*.py. One per loaded model; the registry drives it.

    KNOWN_SOURCES maps the normalized sha256 of every original the binding transcribes to its
    file and revision. `trust_for_tests` is the only way to extend it (tests, restored in finally).
    """
    family = ""
    KNOWN_SOURCES = {}   # type: Dict[str, str]

    def geometry(self, k: int) -> Geometry:
        raise NotImplementedError

    def install(self, reg: "HookRegistry") -> None:
        raise NotImplementedError

    def remove(self) -> None:
        raise NotImplementedError

    def forward(self, ids: Tensor, k: int, row_keys: List[RowKey], x0_policy: str = "fixed_table",
                grad: bool = False) -> Tensor:
        raise NotImplementedError

    def readout(self, j: int, h: Tensor, mode: str) -> Tensor:
        raise NotImplementedError

    def readout32(self, h: Tensor, ro: Any) -> Tensor:
        raise NotImplementedError

    def source_report(self) -> Dict[str, Any]:
        raise NotImplementedError

    def expected_counts(self, k: int) -> Dict[str, int]:
        """Per-forward call counts asserted by `HookRegistry.end_forward`."""
        raise NotImplementedError

    def model_module(self) -> torch.nn.Module:
        """The torch module whose parameters the binding reads."""
        raise NotImplementedError

    def device(self) -> torch.device:
        return next(self.model_module().parameters()).device

    def dtype(self) -> torch.dtype:
        return next(self.model_module().parameters()).dtype

    def check_intervention(self, iv: Any) -> None:
        """Raise UnsupportedSite for an op this binding or load cannot host."""

    def check_recorder(self, rec: Any) -> None:
        """Raise UnsupportedSite for a recorder this binding or load cannot host."""

    @classmethod
    def check_source(cls, fn: Callable, what: str) -> str:
        sha = source_sha256(fn)
        if sha not in cls.KNOWN_SOURCES:
            raise SourceMismatch("%s: normalized source sha256 %s is not in %s.KNOWN_SOURCES; "
                                 "re-read the original and record its hash" % (what, sha,
                                                                              cls.__name__))
        return sha

    @classmethod
    @contextlib.contextmanager
    def trust_for_tests(cls, owner: type, method: str = "forward",
                        label: Optional[str] = None) -> Iterator[str]:
        """Temporarily add `owner.method`'s normalized hash to KNOWN_SOURCES (tests only).
        `label` is the table value; bindings that select a transcription by it (raven: the
        family name first) need it."""
        sha = source_sha256(getattr(owner, method))
        added = sha not in cls.KNOWN_SOURCES
        if added:
            desc = label if label is not None else "test:%s.%s" % (owner.__name__, method)
            cls.KNOWN_SOURCES = dict(cls.KNOWN_SOURCES, **{sha: desc})
        try:
            yield sha
        finally:
            if added:
                table = dict(cls.KNOWN_SOURCES)
                table.pop(sha, None)
                cls.KNOWN_SOURCES = table


# ------------------------------------------------------------------ gradient leaves
class GradLeaves(object):
    """Zero leaves added at sites (`h = h + leaf`, DESIGN 5 contract step 4) for attribution.

    The leaf has the model dtype (K1). One forward fills `leaves` and `states`; one backward then
    serves every site. `masks[site]` is the bool [B, T] position mask at attach time.
    """

    def __init__(self, sites: Sequence[Site], positions: Positions):
        self.sites = tuple(sites)
        self.positions = positions
        self.leaves = {}   # type: Dict[Site, Tensor]
        self.states = {}   # type: Dict[Site, Tensor]
        self.masks = {}    # type: Dict[Site, Tensor]

    def attach(self, site: Site, h: Tensor, mask: Tensor) -> Tensor:
        leaf = torch.zeros_like(h, requires_grad=True)
        self.leaves[site] = leaf
        self.states[site] = h.detach()
        self.masks[site] = mask
        return h + leaf

    def leaf(self, site: Site) -> Tensor:
        """The leaf added at `site` in the last forward (input of torch.autograd.grad)."""
        if site not in self.leaves:
            raise SteerConfigError("no grad leaf at %s in the last forward" % site.key())
        return self.leaves[site]

    def grad(self, site: Site) -> Optional[Tensor]:
        return self.leaves[site].grad

    def reset(self) -> None:
        self.leaves, self.states, self.masks = {}, {}, {}


# ------------------------------------------------------------------ the registry
class HookRegistry(object):
    """Holds a run's consumers; see the module docstring for the contract (DESIGN 5)."""

    def __init__(self, binding: FamilyBinding, k: int):
        self.binding = binding
        self.k = int(k)
        self.geom = binding.geometry(self.k)
        self.installed = False
        self.ctx = None          # type: Optional[ForwardCtx]
        self._active = False
        self._rows = None        # type: Optional[List[RowInfo]]
        self._pad = None         # type: Optional[List[int]]
        self.clear()

    # ---------------------------------------------------------------- configuration
    def clear(self) -> None:
        """Drop every consumer, statistic and record (between batches or before install)."""
        self._ids = set()
        self._edits = {}         # type: Dict[Site, List[Any]]
        self._recs = {}          # type: Dict[Site, List[Any]]
        self._gate_recs = {}     # type: Dict[int, List[Any]]
        self._attn_recs = {}     # type: Dict[Call, List[Tuple[Any, Site]]]
        self._knock = {}         # type: Dict[Call, List[Any]]
        self._skip = set()       # type: set
        self._procs = []         # type: List[Tuple[str, Callable]]
        self._grads = {}         # type: Dict[Site, GradLeaves]
        self._edit_stats = {}    # type: Dict[Tuple[RowKey, str, str], Dict[str, float]]
        self._knock_stats = Counter()
        self._records = {}       # type: Dict[Tuple[RowKey, str, str], Dict[int, Any]]
        self.taken_positions = {}  # type: Dict[Tuple[RowKey, str, str], List[int]]
        self._counts = Counter()
        self._forwards = 0
        self._last_counts = {}   # type: Dict[str, int]
        self._extra_passes = 0
        self._journal = None     # type: Optional[Dict[str, Any]]
        #: decode positions already in the statistics, per (row, intervention, site key)
        self._decode_seen = {}   # type: Dict[Tuple[RowKey, str, str], set]

    def _new_id(self, ident: str) -> str:
        if ident in self._ids:
            raise SteerConfigError("duplicate consumer id %r" % (ident,))
        self._ids.add(ident)
        return ident

    def _canonical_strict(self, site: Site, who: str) -> Site:
        c = canonical(self.geom, site)
        if c != site:
            raise SteerConfigError("%s: site %s is an alias of %s; register canonical sites"
                                   % (who, site.key(), c.key()))
        return c

    def _call_of(self, site: Site, who: str) -> Call:
        """(stage, loop, layer) of the block call a per-layer site names. Knockout, skip and
        attn_mass act on a call, not on a tensor, so the component only names the call
        (interventions.call_component: ATTN_OUT, RESID_POST on IFM) and is not canonicalized."""
        if site.layer is None:
            raise SteerConfigError("%s names a block call by a per-layer site, got %s"
                                   % (who, site.key()))
        n = self.geom.n_blocks(site.stage)
        loop_ok = site.stage != "loop" or 1 <= site.loop <= self.k
        if not (0 <= site.layer < n and loop_ok):
            raise SteerConfigError("%s: call %s does not exist at depth %d" % (who, site.key(),
                                                                               self.k))
        return (site.stage, site.loop, site.layer)

    def add_intervention(self, iv: Any) -> str:
        """Register an interventions.Intervention (duck-typed: id, op, sites, positions, rows,
        edit(x, site, row, h_pos, ctx), knock(site, row, ctx)) together with its own internal
        recorders (`iv.recorders`, e.g. the `extrapolate` reference state, DESIGN 7.1)."""
        self.binding.check_intervention(iv)
        ident = self._new_id(iv.id)
        if not iv.sites:
            raise SteerConfigError("intervention %s has no sites" % ident)
        for s in iv.sites:
            if iv.op == "skip":
                self._skip.add(self._call_of(s, "skip " + ident))
            elif iv.op == "knockout":
                self._knock.setdefault(self._call_of(s, "knockout " + ident), []).append(iv)
            else:
                self._edits.setdefault(self._canonical_strict(s, ident), []).append(iv)
        for rec in getattr(iv, "recorders", ()):
            self.add_recorder(rec)
        return ident

    def add_recorder(self, rec: Any) -> str:
        """Register a readouts.Recorder."""
        if rec.what not in RECORDER_KINDS:
            raise SteerConfigError("unknown recorder kind %r" % (rec.what,))
        self.binding.check_recorder(rec)
        ident = self._new_id(rec.id)
        for s in rec.sites:
            if rec.what == "attn_mass":
                if rec.key_positions is None:
                    raise SteerConfigError("attn_mass recorder %s needs key_positions" % ident)
                self._attn_recs.setdefault(self._call_of(s, "attn_mass " + ident), []).append(
                    (rec, s))
                continue
            c = self._canonical_strict(s, ident)
            if rec.what == "gate":
                if c.component != Component.LOOP_END:
                    raise SteerConfigError("gate recorder %s needs LOOP_END sites" % ident)
                self._gate_recs.setdefault(c.loop, []).append(rec)
            else:
                self._recs.setdefault(c, []).append(rec)
        return ident

    def add_logits_processor(self, proc: Callable[[Tensor, ForwardCtx], Tensor],
                             ident: Optional[str] = None) -> str:
        ident = self._new_id(ident or "proc%d" % len(self._procs))
        self._procs.append((ident, proc))
        return ident

    def add_grad_leaves(self, sites: Sequence[Site], positions: Positions) -> GradLeaves:
        gl = GradLeaves([self._canonical_strict(s, "grad leaves") for s in sites], positions)
        for s in gl.sites:
            if s in self._grads:
                raise SteerConfigError("grad leaves already registered at %s" % s.key())
            self._grads[s] = gl
        return gl

    def has_consumers(self) -> bool:
        return bool(self._edits or self._recs or self._gate_recs or self._attn_recs
                    or self._knock or self._skip or self._procs or self._grads)

    # ---------------------------------------------------------------- lifecycle
    def install(self) -> None:
        if self.installed:
            raise SteerConfigError("registry already installed")
        self.binding.install(self)
        self.installed = True

    def remove(self) -> None:
        if self.installed:
            self.binding.remove()
            self.installed = False
        self._active = False

    @contextlib.contextmanager
    def batch(self, rows: List[RowInfo], pad: List[int]) -> Iterator["HookRegistry"]:
        """Row table of one decode() call or one forward batch."""
        if len(rows) != len(pad):
            raise SteerConfigError("batch: %d rows but %d pads" % (len(rows), len(pad)))
        prev = (self._rows, self._pad)
        self._rows, self._pad = list(rows), [int(p) for p in pad]
        try:
            yield self
        finally:
            self._rows, self._pad = prev

    @property
    def rows(self) -> Optional[List[RowInfo]]:
        return self._rows

    @property
    def pad(self) -> Optional[List[int]]:
        """Left pads of the open batch (None outside reg.batch)."""
        return self._pad

    # ---------------------------------------------------------------- per decode call
    def begin_call(self) -> None:
        """Start journaling one decode() call: edit statistics are buffered per position and
        record writes keep the value they replace, so `end_call` can drop the forwards a finished
        row only ran because other rows were still decoding."""
        if self._journal is not None:
            raise SteerConfigError("begin_call inside an open decode call")
        self._journal = {"edits": [], "writes": {}}

    def end_call(self, cutoffs: Optional[Dict[RowKey, int]]) -> None:
        """Close the journal. `cutoffs[row]` is the first absolute position whose forward is not
        part of that row's output (SteeredAdapter.decode: len(seq) + max(len(gen) - 1, 0)); edit
        statistics and records at positions >= the cutoff are dropped and the replaced record
        values restored. A row absent from `cutoffs` keeps everything.

        cutoffs None means the decode raised: its output is discarded (prod retries an OOM at half
        the batch), so the call leaves no trace: no statistic is added and every journaled record
        write is restored [core fixer B-F3 / C-F1].

        Decode statistics count each (row, intervention, site, absolute position) once over the
        registry's life, so a wave re-prefill or a forced read-out that recomputes positions
        already counted adds nothing (DESIGN 5 per-row n positions and sum of ||delta||, P1):
        the first committed value of a position is kept, and n_calls counts the site calls that
        added at least one position [core fixer C-F1]."""
        j, self._journal = self._journal, None
        if j is None:
            raise SteerConfigError("end_call without begin_call")
        discard = cutoffs is None
        cut = cutoffs or {}
        if not discard:
            for key, iv_id, sk, pos, d in j["edits"]:
                seen = self._decode_seen.setdefault((key, iv_id, sk), set())
                keep = []
                for p, v in zip(pos, d):
                    if (key in cut and p >= cut[key]) or p in seen:
                        continue
                    seen.add(p)
                    keep.append(v)
                if keep:
                    self._add_stat(key, iv_id, sk, keep)
        for (slot_key, p), prev in j["writes"].items():
            rk = slot_key[0]
            if not discard and (rk not in cut or p < cut[rk]):
                continue
            slot = self._records.get(slot_key)
            if slot is None:
                continue
            if prev is _MISSING:
                slot.pop(p, None)
            else:
                slot[p] = prev
            if not slot:
                del self._records[slot_key]

    # ---------------------------------------------------------------- binding-facing
    def begin_forward(self, cache_position: Tensor, T: int, attn_2d: Optional[Tensor] = None,
                      B: Optional[int] = None) -> None:
        """Start a forward: abs_pos[b, c] = cache_position[c] - pad[b], valid = abs_pos >= 0
        (DESIGN 4.3 column mapping)."""
        rows, pad = self._rows, self._pad
        if rows is None:
            if self.has_consumers():
                raise SteerConfigError("a forward ran with consumers registered but outside "
                                       "reg.batch(rows, pad)")
            n = int(B if B is not None else (attn_2d.shape[0] if attn_2d is not None else 1))
            rows, pad = [RowInfo(key=b, prompt_len=0) for b in range(n)], [0] * n
        if B is not None and len(rows) != int(B):
            raise SteerConfigError("reg.batch has %d rows but the forward has B = %d"
                                   % (len(rows), int(B)))
        cp = cache_position.reshape(-1).long()
        if cp.numel() != int(T):
            raise SteerConfigError("cache_position has %d entries for T = %d" % (cp.numel(), T))
        pad_t = torch.tensor(pad, dtype=torch.long, device=cp.device)
        abs_pos = cp[None, :] - pad_t[:, None]
        self.ctx = ForwardCtx(B=len(rows), T=int(T), abs_pos=abs_pos, valid=abs_pos >= 0,
                              attn_2d=attn_2d, rows=rows, pad=list(pad), cache_position=cp)
        self._pmask = {}
        self._counts = Counter()
        for gl in set(self._grads.values()):
            gl.reset()
        self._active = True

    def end_forward(self) -> None:
        """Assert the per-forward call counts, then clear ctx.store as the last statement (S2)."""
        if not self._active:
            raise SteerConfigError("end_forward without begin_forward")
        self._active = False
        self._forwards += 1
        self._last_counts = dict(self._counts)
        want = self.binding.expected_counts(self.k)
        bad = {n: (self._counts.get(n, 0), v) for n, v in want.items()
               if self._counts.get(n, 0) != v}
        try:
            if bad:
                raise CallCountError("per-forward call counts (seen, expected): %r" % (bad,))
            if self.ctx.store.get("deferred_readouts"):
                raise SteerConfigError("deferred (cached) read-outs were left unrun at the end of "
                                       "the forward; the %s binding does not run them"
                                       % self.binding.family)
        finally:
            self.ctx.store.clear()

    def count(self, name: str, n: int = 1) -> int:
        """Increment a per-forward counter (no-op inside a read-out, S10); returns the count."""
        if self.ctx is not None and self.ctx.in_readout:
            return self._counts.get(name, 0)
        self._counts[name] += n
        return self._counts[name]

    def counts(self, name: str) -> int:
        return self._counts.get(name, 0)

    @contextlib.contextmanager
    def readout_guard(self) -> Iterator[None]:
        """Set ctx.in_readout for a read-out pass inside (or after) a forward (S10)."""
        ctx = self.ctx
        if ctx is None:
            yield
            return
        prev = ctx.in_readout
        ctx.in_readout = True
        try:
            yield
        finally:
            ctx.in_readout = prev

    def live(self) -> bool:
        """True inside a forward and outside any read-out."""
        return self._active and self.ctx is not None and not self.ctx.in_readout

    def wants(self, site: Site) -> bool:
        """True if any edit, state or read-out recorder or grad leaf sits at the canonical site."""
        return self.live() and (site in self._edits or site in self._recs or site in self._grads)

    def _mask(self, positions: Positions, b: int) -> Tensor:
        key = (positions, b)
        m = self._pmask.get(key)
        if m is None:
            ctx = self.ctx
            m = positions.resolve(ctx.rows[b], ctx.abs_pos[b], ctx.valid[b])
            self._pmask[key] = m
        return m

    @staticmethod
    def _in_scope(iv: Any, row: RowInfo) -> bool:
        return iv.rows is None or row.key in iv.rows

    def site(self, site: Site, h: Tensor) -> Tensor:
        """DESIGN 5 contract: pre recorders, edits in registration order (one clone per call,
        float32 math, written back in h.dtype), post recorders, grad leaves."""
        if not self.wants(site):
            return h
        ctx = self.ctx
        if h.shape[0] != ctx.B or h.shape[1] != ctx.T:
            raise SteerConfigError("%s: tensor %s does not match the forward (B=%d, T=%d)"
                                   % (site.key(), tuple(h.shape), ctx.B, ctx.T))
        self._record(site, h, "pre")
        edited = False
        for iv in self._edits.get(site, ()):
            for b, row in enumerate(ctx.rows):
                if not self._in_scope(iv, row):
                    continue
                t = self._mask(iv.positions, b).nonzero(as_tuple=True)[0]
                if t.numel() == 0:
                    continue
                x = h[b, t].float()
                y = iv.edit(x, site, row, ctx.abs_pos[b, t], ctx)
                if not edited:
                    h = h.clone()
                    edited = True
                y = y.to(h.dtype)                  # the value written: its delta is the one the
                h[b, t] = y                        # model sees (core fixer B-F6)
                self._account(row.key, iv.id, site, x, y, ctx.abs_pos[b, t])
        self._record(site, h, "post")
        gl = self._grads.get(site)
        if gl is not None:
            mask = torch.stack([self._mask(gl.positions, b) for b in range(ctx.B)])
            h = gl.attach(site, h, mask)
        return h

    def _account(self, key: RowKey, iv_id: str, site: Site, x: Tensor, y: Tensor,
                 abs_pos: Tensor) -> None:
        d = (y.detach().float() - x.detach()).reshape(x.shape[0], -1).norm(dim=1)
        if self._journal is not None:
            self._journal["edits"].append((key, iv_id, site.key(), abs_pos.tolist(), d.tolist()))
            return
        self._add_stat(key, iv_id, site.key(), d.tolist())

    def _add_stat(self, key: RowKey, iv_id: str, sk: str, d: List[float]) -> None:
        """One site call's ||y - x||_2 per edited position into the strength statistics."""
        st = self._edit_stats.setdefault((key, iv_id, sk), {
            "n_calls": 0, "n_positions": 0, "delta_norm_total": 0.0, "delta_norm_max": 0.0})
        st["n_calls"] += 1
        st["n_positions"] += len(d)
        st["delta_norm_total"] += float(sum(d))
        st["delta_norm_max"] = max(st["delta_norm_max"], float(max(d)))

    # ---------------------------------------------------------------- recorders
    def _put(self, key: Tuple[RowKey, str, str], abs_pos: List[int], vals: List[Any]) -> None:
        slot = self._records.setdefault(key, {})
        writes = self._journal["writes"] if self._journal is not None else None
        for p, v in zip(abs_pos, vals):
            if writes is not None and (key, int(p)) not in writes:
                writes[(key, int(p))] = slot.get(int(p), _MISSING)
            slot[int(p)] = v                     # position pure: a re-prefill rewrites the same

    def _record(self, site: Site, h: Tensor, when: str) -> None:
        for rec in self._recs.get(site, ()):
            if rec.when != when:
                continue
            if rec.what == "state":
                self._record_state(rec, site, h)
            elif rec.mode == "cached":
                # DESIGN 8.2: the cached lens runs the last column only; a request for any other
                # column of this forward would be dropped silently [core fixer C-F2]
                for b in range(self.ctx.B):
                    if bool(self._mask(rec.positions, b)[:-1].any()):
                        raise SteerConfigError(
                            "cached read-out %s at %s selects columns before the last of this "
                            "forward (row %r); cached mode reads the last column only (DESIGN 8.2)"
                            % (rec.id, site.key(), self.ctx.rows[b].key))
                self.ctx.store.setdefault("deferred_readouts", []).append(
                    (rec, site, h[:, -1:].detach()))
            else:
                z = self.binding.readout(site.loop, h, rec.mode)
                self.deliver_readout(rec, site, z, columns=None)

    def _record_state(self, rec: Any, site: Site, h: Tensor) -> None:
        ctx = self.ctx
        if not rec.to_cpu:
            ctx.store[("state", rec.id, site.key())] = h.detach()
            return
        dt = getattr(torch, rec.dtype)
        for b, row in enumerate(ctx.rows):
            t = self._mask(rec.positions, b).nonzero(as_tuple=True)[0]
            if t.numel():
                v = h[b, t].detach().to(dt).cpu()
                self._put((row.key, rec.id, site.key()), ctx.abs_pos[b, t].tolist(), list(v))

    def deliver_readout(self, rec: Any, site: Site, z: Tensor, columns: Optional[Tensor]) -> None:
        """Record read-out statistics of z [B, T_r, V] (float32). `columns` are the forward
        columns of z's T_r axis (None = every column; the raven `cached` mode passes the last)."""
        from .readouts import readout_stats
        ctx = self.ctx
        if not rec.to_cpu:
            ctx.store[("readout", rec.id, site.key())] = z
        cols = torch.arange(ctx.T, device=z.device) if columns is None else columns
        for b, row in enumerate(ctx.rows):
            m = self._mask(rec.positions, b)[cols]
            t = m.nonzero(as_tuple=True)[0]
            if t.numel() == 0:
                continue
            targets = (rec.targets or {}).get(row.key, [])
            st = readout_stats(z[b, t].float(), targets)
            vals = [{n: v[i].cpu() for n, v in st.items()} for i in range(t.numel())]
            self._put((row.key, rec.id, site.key()), ctx.abs_pos[b, cols[t]].tolist(), vals)

    def gate(self, loop: int, g: Tensor) -> None:
        """Ouro exit-gate logits of loop `loop` ([B, T, 1]) for `gate` recorders."""
        if not self.live():
            return
        ctx = self.ctx
        site = Site("loop", loop, None, Component.LOOP_END)
        for rec in self._gate_recs.get(loop, ()):
            for b, row in enumerate(ctx.rows):
                t = self._mask(rec.positions, b).nonzero(as_tuple=True)[0]
                if t.numel():
                    v = g[b, t].detach().float().reshape(t.numel(), -1)[:, 0].cpu()
                    self._put((row.key, rec.id, site.key()), ctx.abs_pos[b, t].tolist(), list(v))

    def wants_gate(self, loop: int) -> bool:
        return self.live() and loop in self._gate_recs

    def wants_attn(self, call_site: Site) -> bool:
        return self.live() and (call_site.stage, call_site.loop, call_site.layer) in self._attn_recs

    def attn_probs(self, call_site: Site, probs: Tensor) -> None:
        """Record the attention mass on key_positions per query and head (PF5; k16 W2 quantity:
        k16_path_ouro.py:302 captures out[1], w2_window:519 sums probs[:, src]). probs is the
        module's own [B, H, T_q, KV] post-softmax tensor; T_q equals the forward's columns."""
        ctx = self.ctx
        call = (call_site.stage, call_site.loop, call_site.layer)
        KV = probs.shape[-1]
        for rec, rec_site in self._attn_recs.get(call, ()):
            for b, row in enumerate(ctx.rows):
                t = self._mask(rec.positions, b).nonzero(as_tuple=True)[0]
                if t.numel() == 0:
                    continue
                kabs = torch.arange(KV, device=probs.device) - ctx.pad[b]
                kmask = rec.key_positions.resolve(row, kabs, kabs >= 0)
                q_rows = t + (probs.shape[-2] - ctx.T)
                mass = probs[b][:, q_rows][:, :, kmask].float().sum(-1)        # [H, n_q]
                vals = [{"per_head": mass[:, i].cpu(), "mean": mass[:, i].mean().cpu()}
                        for i in range(t.numel())]
                self._put((row.key, rec.id, rec_site.key()), ctx.abs_pos[b, t].tolist(), vals)

    # ---------------------------------------------------------------- knockout and skip
    def attention(self, site: Site, mask: Any, kind: str = "float") -> Tuple[Any, bool]:
        """(mask to use, routed?). Routes a call when any knockout names it; the routed baseline
        with no knocked entries is the comparison of record (DESIGN 6.1 routing rule).

        kind "float": a float additive mask [B, 1, T_q, KV] (Ouro eager mask, raven _BIAS); knocked
        entries are ASSIGNED finfo(dtype).min. kind "segments": an IFM bool mask [B, S, S]
        (True = blocked) OR-ed with the knocks (k16_path_ifm.py:320-331)."""
        if not self.live():
            return mask, False
        call = (site.stage, site.loop, site.layer)
        ivs = self._knock.get(call)
        if not ivs:
            return mask, False
        if kind == "float":
            return self._knock_float(site, ivs, mask), True
        if kind == "segments":
            return self._knock_segments(site, ivs, mask), True
        raise SteerConfigError("unknown mask kind %r" % (kind,))

    def _entries(self, site: Site, ivs: List[Any], Tq: int, KV: int,
                 device: torch.device) -> List[Tuple[int, Tensor, Tensor]]:
        """Knocked (row, query rows, key columns): q_row = q_abs + pad - cache_position[0],
        k_col = k_abs + pad (MF9, S5); queries outside this forward are skipped."""
        ctx = self.ctx
        cp0 = int(ctx.cache_position[0])
        out = []
        for iv in ivs:
            for b, row in enumerate(ctx.rows):
                if not self._in_scope(iv, row):
                    continue
                qk = iv.knock(site, row, ctx)
                if qk is None:
                    continue
                q = torch.as_tensor(qk[0], dtype=torch.long, device=device).reshape(-1)
                kk = torch.as_tensor(qk[1], dtype=torch.long, device=device).reshape(-1)
                qr = q + ctx.pad[b] - cp0
                qr = qr[(qr >= 0) & (qr < Tq)]
                kc = kk + ctx.pad[b]
                kc = kc[(kc >= 0) & (kc < KV)]
                self._knock_stats[(row.key, iv.id, site.key())] += int(qr.numel() * kc.numel())
                if qr.numel() and kc.numel():
                    out.append((b, qr, kc))
        return out

    def _knock_float(self, site: Site, ivs: List[Any], mask: Any) -> Tensor:
        if not isinstance(mask, Tensor) or not torch.is_floating_point(mask) or mask.dim() != 4:
            raise UnsupportedSite("%s: knockout needs a float 4D additive mask (the eager path "
                                  "or the raven bias); got %r" % (site.key(), type(mask)))
        B = self.ctx.B
        neg = torch.finfo(mask.dtype).min
        zero = torch.zeros((), dtype=mask.dtype, device=mask.device)
        fm = torch.where(mask == 0, zero, torch.full((), neg, dtype=mask.dtype, device=mask.device))
        if fm.shape[0] != B:
            fm = fm.expand(B, *fm.shape[1:]).clone()
        ents = self._entries(site, ivs, fm.shape[-2], fm.shape[-1], fm.device)
        for b, qr, kc in ents:
            fm[b, :, qr[:, None], kc[None, :]] = neg
        for b, qr, _kc in ents:
            if not bool((fm[b][:, qr] == 0).any(-1).all()):
                raise SteerConfigError("%s: a knocked query row of row %r keeps no allowed key "
                                       "(full-row guard, S6)" % (site.key(), self.ctx.rows[b].key))
        return fm

    def _knock_segments(self, site: Site, ivs: List[Any], segments: Optional[Tensor]) -> Tensor:
        ctx = self.ctx
        dev = ctx.abs_pos.device if segments is None else segments.device
        kom = torch.zeros((ctx.B, ctx.T, ctx.T), dtype=torch.bool, device=dev)
        ents = self._entries(site, ivs, ctx.T, ctx.T, dev)
        for b, qr, kc in ents:
            kom[b, qr[:, None], kc[None, :]] = True
        out = kom if segments is None else (segments | kom)
        cols = torch.arange(ctx.T, device=dev)
        for b, qr, _kc in ents:
            allowed = ~out[b, qr] & (cols[None, :] <= qr[:, None])
            if not bool(allowed.any(-1).all()):
                raise SteerConfigError("%s: a knocked query of row %r keeps no key k <= q "
                                       "(full-row guard, S6)" % (site.key(), ctx.rows[b].key))
        return out

    def skipped(self, stage: str, loop: Optional[int], layer: int) -> bool:
        return self.live() and (stage, loop, layer) in self._skip

    def routed(self, site: Site) -> bool:
        """True when a knockout routes the block call `site` names in this live forward."""
        return self.live() and (site.stage, site.loop, site.layer) in self._knock

    # ---------------------------------------------------------------- epilogue and results
    def logits(self, z: Tensor) -> Tensor:
        """Forward epilogue step 1 (S2): processors in registration order on z [B, T_out, V];
        the T_out columns are the forward's last T_out columns."""
        if not self._active:
            return z
        for _ident, proc in self._procs:
            z = proc(z, self.ctx)
        return z

    def add_extra_passes(self, n: int) -> None:
        """Read-out layer passes outside the executed slots (DESIGN 8.2 cost)."""
        self._extra_passes += int(n)

    def stats(self) -> Dict[str, Any]:
        edits = {}   # type: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
        for (key, iv_id, sk), st in self._edit_stats.items():
            edits.setdefault(str(key), {}).setdefault(iv_id, {})[sk] = dict(st)
        knock = {}   # type: Dict[str, Dict[str, Dict[str, int]]]
        for (key, iv_id, sk), n in self._knock_stats.items():
            knock.setdefault(str(key), {}).setdefault(iv_id, {})[sk] = int(n)
        return {"edits": edits, "knock_entries": knock, "forwards": self._forwards,
                "last_forward_counts": dict(self._last_counts),
                "extra_passes": self._extra_passes}

    def take_records(self) -> Dict[Tuple[RowKey, str, str], Any]:
        """(row, recorder id, site key) -> float32 CPU tensor [n_pos, ...] in ascending absolute
        position (a dict of such tensors for read-out and attention-mass records). The positions
        of each record are left in `taken_positions`."""
        out = {}
        self.taken_positions = {}
        for key, slot in self._records.items():
            pos = sorted(slot)
            vals = [slot[p] for p in pos]
            if isinstance(vals[0], dict):
                out[key] = {n: torch.stack([v[n] for v in vals]) for n in vals[0]}
            else:
                out[key] = torch.stack(vals)
            self.taken_positions[key] = pos
        self._records = {}
        return out

