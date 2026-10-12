"""Family bindings and the steered adapter that prod.generate drives (DESIGN 5, 6).

What this module provides and why. Every measurement of the study runs one of two ways: a
forward of fixed token ids (mechanism rows) or prod's own decode loop (transfer and accuracy rows,
so prompts, parsers and decoding are SETTLE's). `bind` gives the family binding that owns the
hooks of a loaded adapter; `SteeredAdapter` wraps a prod adapter so that `prod.generate.main`
decodes through a hook registry built per depth from a plan, with the batch's row table (row keys,
spans, payloads, left pads) set for every decode call. `rows_from_prompts` builds the row table
from the indexed prompts and refuses a decode sequence that does not start with its row's prompt
(PromptMismatch), so a steered row can never be matched to the wrong item.
"""
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..hooks import (FamilyBinding, HookRegistry, PromptMismatch, RowInfo, RowKey,
                     SteerConfigError)
from ..sites import Geometry

__all__ = ["bind", "SteeredAdapter", "rows_from_prompts", "plan_sha256"]


def bind(adapter_or_runner: Any) -> FamilyBinding:
    """Dispatch on `.family` (DESIGN 6): ouro -> OuroBinding, huginn / mcleish -> RavenBinding,
    ifm -> IfmBinding (an IfmAdapter binds its runner)."""
    fam = getattr(adapter_or_runner, "family", None)
    if fam == "ouro":
        from .ouro import OuroBinding
        return OuroBinding(adapter_or_runner)
    if fam in ("huginn", "mcleish"):
        from .raven import RavenBinding
        return RavenBinding(adapter_or_runner)
    if fam == "ifm":
        from .ifm import IfmBinding
        return IfmBinding(getattr(adapter_or_runner, "runner", None) or adapter_or_runner)
    raise SteerConfigError("no binding for family %r" % (fam,))


def plan_sha256(spec: Any) -> str:
    """sha256 of the plan spec as canonical JSON (sorted keys, no whitespace)."""
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def rows_from_prompts(prompts: Dict[RowKey, Sequence[int]],
                      spans: Optional[Dict[RowKey, Dict[str, Sequence[int]]]] = None,
                      payload: Optional[Dict[RowKey, Dict[str, Any]]] = None
                      ) -> Callable[[List[RowKey], List[List[int]]], List[RowInfo]]:
    """A `rows_of_call` for SteeredAdapter: RowInfo(key, prompt_len, spans, payload) per row id,
    raising PromptMismatch when seqs[b][:P] differs from the indexed prompt of row_ids[b]."""
    table = {k: list(v) for k, v in prompts.items()}

    def rows_of_call(row_ids: List[RowKey], seqs: List[List[int]]) -> List[RowInfo]:
        if row_ids is None or len(row_ids) != len(seqs):
            raise SteerConfigError("a steered decode needs one row id per sequence")
        out = []
        for key, seq in zip(row_ids, seqs):
            if key not in table:
                raise PromptMismatch("row %r has no indexed prompt" % (key,))
            p = table[key]
            if list(seq[:len(p)]) != p:
                raise PromptMismatch("decode sequence of row %r does not start with its indexed "
                                     "prompt" % (key,))
            sp = {n: tuple(int(i) for i in v) for n, v in ((spans or {}).get(key) or {}).items()}
            out.append(RowInfo(key=key, prompt_len=len(p), spans=sp,
                               payload=dict((payload or {}).get(key) or {})))
        return out
    return rows_of_call


def _row_cutoffs(rows: List[RowInfo], seqs: List[List[int]], gen: List[List[int]]
                 ) -> Dict[RowKey, int]:
    """First absolute position whose forward is not part of a row's output. prod decode feeds
    generated token i at position len(seq) + i [read OuroAdapter.decode prod/models/ouro.py:359,
    RavenAdapter.decode prod/models/raven.py:378 and the IFM runner.generate: a finished row keeps
    receiving pad or eos inputs while other rows decode],
    and the last emitted token's own forward feeds nothing that row uses, so the row's forwards end
    at len(seq) + len(gen) - 2; prompt positions are always kept."""
    out = {}   # type: Dict[RowKey, int]
    for r, s, g in zip(rows, seqs, gen):
        c = len(s) + max(len(g) - 1, 0)
        out[r.key] = max(out.get(r.key, c), c)
    return out


class SteeredAdapter(object):
    """The object prod.generate.main drives (DESIGN 5). Attribute reads fall through to the inner
    adapter; load / set_depth / decode / meta add the binding, the per-depth registry and the
    per-call row table."""

    def __init__(self, inner: Any, plan_builder: Callable[[HookRegistry, Geometry], None],
                 rows_of_call: Callable[[List[RowKey], List[List[int]]], List[RowInfo]],
                 plan_spec: Optional[Any] = None, init_state: Optional[str] = None,
                 t_max: Optional[int] = None):
        """`init_state` and `t_max` are `plan.init_state` and its T_max (DESIGN 6.2). A raven
        binding needs them (here or from the plan builder via binding.set_init_state): its x0
        policy has no default, so the policy a record names is the one that ran."""
        self.__dict__["inner"] = inner
        self.__dict__["plan_builder"] = plan_builder
        self.__dict__["rows_of_call"] = rows_of_call
        self.__dict__["plan_spec"] = plan_spec
        self.__dict__["init_state"] = init_state
        self.__dict__["t_max"] = t_max
        self.__dict__["binding"] = None
        self.__dict__["registry"] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["inner"], name)

    def load(self, *a: Any, **kw: Any) -> "SteeredAdapter":
        self.inner.load(*a, **kw)
        return self.bind_loaded()

    def bind_loaded(self) -> "SteeredAdapter":
        """Bind an inner adapter that is already loaded (tests and runners that load first)."""
        b = bind(self.inner)
        if self.__dict__["init_state"] is not None:
            if not hasattr(b, "set_init_state"):
                raise SteerConfigError("init_state %r given for family %r, which has no x0 policy"
                                       % (self.__dict__["init_state"], getattr(b, "family", None)))
            b.set_init_state(self.__dict__["init_state"], self.__dict__["t_max"])
        self.__dict__["binding"] = b
        return self

    def set_depth(self, k: int) -> Any:
        if self.binding is None:
            raise SteerConfigError("SteeredAdapter.set_depth before load")
        info = self.inner.set_depth(k)
        if self.registry is not None:
            self.registry.remove()
        reg = HookRegistry(self.binding, int(k))
        self.plan_builder(reg, reg.geom)
        if hasattr(self.binding, "set_init_state") and self.binding.x0_policy is None:
            raise SteerConfigError("a steered %s decode needs plan.init_state (and t_max for "
                                   "fixed_table): SteeredAdapter(init_state=..., t_max=...) or "
                                   "binding.set_init_state in the plan builder (DESIGN 6.2)"
                                   % self.binding.family)
        reg.install()
        self.__dict__["registry"] = reg
        return info

    def decode(self, seqs: List[List[int]], n_new: int, k: int, eos_ids: Any,
               stop_strings: Any = None, pieces: Any = None, row_ids: Any = None,
               **kw: Any) -> Any:
        if self.registry is None or self.registry.k != int(k):
            raise SteerConfigError("SteeredAdapter.decode at k=%r without set_depth(k)" % (k,))
        rows = self.rows_of_call(row_ids, seqs)
        L = max(len(s) for s in seqs)
        pad = [0] * len(seqs) if self.family == "ifm" else [L - len(s) for s in seqs]
        reg = self.registry
        with reg.batch(rows, pad=pad):
            reg.begin_call()
            cut = None
            try:
                out = self.inner.decode(seqs, n_new, k, eos_ids, stop_strings=stop_strings,
                                        pieces=pieces, row_ids=row_ids, **kw)
                cut = _row_cutoffs(rows, seqs, out[0])
            finally:
                reg.end_call(cut)
        return out

    def meta(self) -> Dict[str, Any]:
        d = dict(self.inner.meta())
        d["steer"] = {"plan": self.plan_spec,
                      "plan_sha256": plan_sha256(self.plan_spec),
                      "binding": self.binding.source_report() if self.binding else None}
        return d

    def passes_per_token(self, k: int) -> int:
        """The inner count; read-out extra passes are in registry.stats()['extra_passes']."""
        return self.inner.passes_per_token(k)
