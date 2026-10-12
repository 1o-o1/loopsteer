"""Sites, geometry and positions: where in a looped LM an edit, a knockout or a recorder acts.

What this module measures and why. The study asks where an edit changes the prediction of a looped
LM and which later pathway removes it (DESIGN.md section 4). Every tool of the package therefore
names its tensor by one coordinate system: stage x loop x layer x component (a `Site`) plus a set of
absolute token positions (a `Positions`). This module owns that system and the arithmetic that
follows from it, with no model code:

- `Site` and its canonical string (DESIGN 4.1);
- `canonical(geom, site)`: the alias table, so one tensor is hooked once (DESIGN 4.1);
- `Geometry`: executed index, rank, first changed K/V slot F and cache slot (DESIGN 4.2, 4.5);
- `Positions.resolve`: absolute positions of one row (DESIGN 4.3);
- `readout_identity` and `structural_zero` (DESIGN 4.5, DR13, DR15, DR22);
- `answer_site_set`, `matrix_site` and `ifm_matched_d112` (DESIGN 4.2, 4.5; EXPERIMENT_MATRIX 5.2).

Executed index, 0-based [DESIGN 4.2]: prelude layer l -> l; loop (j, l) -> P + C(j-1) + l; coda l ->
P + C k + l. Per-loop sites take the index of the layer they follow (LOOP_END(j), INJ_STATE(j+1):
layer (j, C-1), ranked after it) or precede (INJ_INPUT(j), INJ_POST(j), INIT_STATE: layer (j, 0),
ranked before it). `Geometry.rank` is the total order used by every "before or after" test.

The error classes live here because `hooks.py` imports this module; `hooks.py` re-exports them
under the names DESIGN section 5 lists.
"""
import enum
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor


# ------------------------------------------------------------------ errors (DESIGN 5)
class SteerError(Exception):
    """Base class of every prod.steer error."""


class SteerConfigError(SteerError):
    """A configuration is missing a REQUIRED field or is inconsistent."""


class UnsupportedSite(SteerError):
    """The family or the load cannot host this site, op or mode."""


class SourceMismatch(SteerError):
    """A transcribed original's normalized source hash is not in KNOWN_SOURCES."""


class PromptMismatch(SteerError):
    """A decode sequence does not start with the indexed prompt of its row."""


class CallCountError(SteerError):
    """The per-forward call counts contradict the geometry (DESIGN 5 end_forward)."""


# ------------------------------------------------------------------ sites (DESIGN 4.1)
STAGES = ("input", "prelude", "loop", "coda")


class Component(str, enum.Enum):
    EMBED = "embed"              # stage input: token embedding output (raven: wte * emb_scale)
    PRELUDE_OUT = "prelude_out"  # stage input: raven / IFM-huginn prelude output e (the injected input)
    INIT_STATE = "init_state"    # stage input: raven / IFM-huginn x0 before step 1
    RESID_PRE = "resid_pre"      # per layer: block input
    ATTN_OUT = "attn_out"        # per layer: attention branch output before its residual add
    HEAD_OUT = "head_out"        # per layer: o_proj / attn.proj INPUT, viewed [B, T, H, hd]
    RESID_MID = "resid_mid"      # per layer: residual after the attention add
    MLP_OUT = "mlp_out"          # per layer: MLP branch output before its residual add
    RESID_POST = "resid_post"    # per layer: block output
    LOOP_END = "loop_end"        # per loop: the carried state at the end of loop j
    INJ_STATE = "inj_state"      # per loop: state entering the injection of loop j
    INJ_INPUT = "inj_input"      # per loop: injected input argument at loop j
    INJ_POST = "inj_post"        # per loop: injection output
    K_PROJ = "k_proj"            # per layer: key projection output before rotary (PF3)
    V_PROJ = "v_proj"            # per layer: value projection output (PF3)


C = Component
PER_LAYER = frozenset([C.RESID_PRE, C.ATTN_OUT, C.HEAD_OUT, C.RESID_MID, C.MLP_OUT, C.RESID_POST,
                       C.K_PROJ, C.V_PROJ])
PER_LOOP = frozenset([C.LOOP_END, C.INJ_STATE, C.INJ_INPUT, C.INJ_POST])
INPUT_COMPONENTS = frozenset([C.EMBED, C.PRELUDE_OUT, C.INIT_STATE])
#: components whose edit changes the K/V of the call they belong to (DESIGN 4.5 table, row 1 and K/V)
_F_SELF = frozenset([C.RESID_PRE, C.INJ_POST, C.INJ_STATE, C.INIT_STATE, C.EMBED, C.INJ_INPUT,
                     C.K_PROJ, C.V_PROJ])
#: components written after the layer's own K/V (DESIGN 4.5 table, row 2)
_F_NEXT = frozenset([C.ATTN_OUT, C.HEAD_OUT, C.RESID_MID, C.MLP_OUT, C.RESID_POST, C.LOOP_END])
#: sub-rank inside one executed index (DESIGN 4.2 ties, DR3, DR22)
_SUB_RANK = {C.EMBED: -10, C.PRELUDE_OUT: 8, C.INIT_STATE: -5, C.INJ_STATE: -4, C.INJ_INPUT: -3,
             C.INJ_POST: -2, C.RESID_PRE: 0, C.K_PROJ: 1, C.V_PROJ: 1, C.HEAD_OUT: 2, C.ATTN_OUT: 3,
             C.RESID_MID: 4, C.MLP_OUT: 5, C.RESID_POST: 6, C.LOOP_END: 7}
#: components each family can host (DESIGN 4.1 table; IFM v1 list)
FAMILY_COMPONENTS = {
    "ouro": PER_LAYER | {C.LOOP_END, C.EMBED},
    "huginn": PER_LAYER | PER_LOOP | INPUT_COMPONENTS,
    "mcleish": PER_LAYER | PER_LOOP | INPUT_COMPONENTS,
    "ifm": frozenset([C.RESID_PRE, C.RESID_POST, C.LOOP_END, C.INJ_STATE, C.INJ_INPUT, C.INJ_POST,
                      C.EMBED, C.PRELUDE_OUT, C.INIT_STATE]),
}
RAVEN = ("huginn", "mcleish")

_KEY_RE = re.compile(r"^(?:(input)|(prelude|coda)\.L(\d+)|loop(\d+)(?:\.L(\d+))?)\.([a-z_]+)$")


@dataclass(frozen=True)
class Site:
    """One tensor of the forward: stage x loop x layer x component (DESIGN 4.1).

    `loop` is 1-based and set for stage "loop" only. `layer` is the 0-based index inside the
    stage's own block list, None for per-loop and input components.
    """
    stage: str
    loop: Optional[int]
    layer: Optional[int]
    component: Component

    def __post_init__(self) -> None:
        comp = Component(self.component)
        object.__setattr__(self, "component", comp)
        if self.stage not in STAGES:
            raise SteerConfigError("unknown stage %r" % (self.stage,))
        if (self.stage == "loop") != (self.loop is not None):
            raise SteerConfigError("loop must be set exactly for stage 'loop': %r" % (self,))
        if self.stage == "input":
            if comp not in INPUT_COMPONENTS or self.layer is not None:
                raise SteerConfigError("stage 'input' takes embed/prelude_out/init_state only")
        elif comp in PER_LAYER:
            if self.layer is None:
                raise SteerConfigError("per-layer component %s needs a layer" % comp.value)
        elif comp in PER_LOOP:
            if self.stage != "loop" or self.layer is not None:
                raise SteerConfigError("per-loop component %s needs stage 'loop' and no layer"
                                       % comp.value)
        else:
            raise SteerConfigError("component %s belongs to stage 'input'" % comp.value)

    def key(self) -> str:
        """Canonical string: input.embed, prelude.L1.resid_post, loop3.L12.attn_out, loop3.loop_end."""
        c = self.component.value
        if self.stage == "input":
            return "input.%s" % c
        if self.stage == "loop":
            if self.layer is None:
                return "loop%d.%s" % (self.loop, c)
            return "loop%d.L%d.%s" % (self.loop, self.layer, c)
        return "%s.L%d.%s" % (self.stage, self.layer, c)

    @staticmethod
    def parse(s: str) -> "Site":
        """Inverse of `key()` for concrete sites (integer layers and loops only)."""
        m = _KEY_RE.match(s.strip())
        if m is None:
            raise SteerConfigError("not a concrete site key: %r (templates go through "
                                   "interventions.expand, aliases through sites.resolve)" % (s,))
        inp, stg, lay, loop, loop_lay, comp = m.groups()
        if inp:
            return Site("input", None, None, Component(comp))
        if stg:
            return Site(stg, None, int(lay), Component(comp))
        return Site("loop", int(loop), None if loop_lay is None else int(loop_lay), Component(comp))

    def __str__(self) -> str:
        return self.key()


_ALIAS_RE = re.compile(r"\.L(first|mid|last)\.")


def resolve(geom: "Geometry", s: str) -> Site:
    """Parse a key whose layer may be an alias (`first` = 0, `mid` = n//2, `last` = n-1, the k15
    convention, DESIGN 4.1) and return the canonical site."""
    m = _ALIAS_RE.search(s)
    if m:
        stage = "loop" if s.startswith("loop") else s.split(".")[0]
        lay = geom.resolve_layer(stage, m.group(1))
        s = s[:m.start()] + ".L%d." % lay + s[m.end():]
    return canonical(geom, Site.parse(s))


# ------------------------------------------------------------------ geometry (DESIGN 4.2)
@dataclass(frozen=True)
class Geometry:
    """Static shape of one model at depth k (DESIGN 4.2).

    The last three fields are IFM-only facts the alias table needs (DESIGN 4.1 IFM list): whether a
    `huginn_recurrent_exit_norm_module` sits between the last core layer and the carried state,
    whether a dense-prelude injection acts at the loop's first layer, and the xllm arch.
    """
    family: str
    k: int
    n_prelude: int
    n_core: int
    n_coda: int
    d_model: int
    n_heads: int
    head_dim: int
    injection: bool
    has_init_state: bool
    state_init: Optional[str] = None
    exit_norm: bool = False
    dense_inj_at_loop_start: bool = False
    arch: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILY_COMPONENTS:
            raise SteerConfigError("unknown family %r" % (self.family,))
        if self.k < 1 or self.n_core < 1:
            raise SteerConfigError("k and n_core must be >= 1")

    # ---------------------------------------------------------------- sizes
    def total(self) -> int:
        """P + C k + Cd = adapter.passes_per_token(k) [read raven.py:339-343; ouro.py:320-321]."""
        return self.n_prelude + self.n_core * self.k + self.n_coda

    def n_blocks(self, stage: str) -> int:
        if stage == "prelude":
            return self.n_prelude
        if stage == "loop":
            return self.n_core
        if stage == "coda":
            return self.n_coda
        raise SteerConfigError("stage %r has no blocks" % (stage,))

    def resolve_layer(self, stage: str, alias_or_int: Union[str, int]) -> int:
        n = self.n_blocks(stage)
        if isinstance(alias_or_int, str) and not alias_or_int.lstrip("-").isdigit():
            table = {"first": 0, "mid": n // 2, "last": n - 1}
            if alias_or_int not in table:
                raise SteerConfigError("unknown layer alias %r" % (alias_or_int,))
            return table[alias_or_int]
        lay = int(alias_or_int)
        if not 0 <= lay < n:
            raise SteerConfigError("layer %d outside stage %s (n=%d)" % (lay, stage, n))
        return lay

    def check(self, site: Site) -> None:
        """Raise unless the site exists in this geometry and family."""
        if site.component not in FAMILY_COMPONENTS[self.family]:
            raise UnsupportedSite("%s: component %s is not supported on family %s"
                                  % (site.key(), site.component.value, self.family))
        if site.stage == "loop" and not 1 <= site.loop <= self.k:
            raise SteerConfigError("%s: loop outside 1..%d" % (site.key(), self.k))
        if site.layer is not None:
            n = self.n_blocks(site.stage)
            if not 0 <= site.layer < n:
                raise SteerConfigError("%s: layer outside stage %s (n=%d)"
                                       % (site.key(), site.stage, n))
        if site.component in (C.INJ_STATE, C.INJ_INPUT, C.INJ_POST) and not self.injection:
            raise UnsupportedSite("%s: no input injection in this geometry" % site.key())
        if site.component == C.INIT_STATE and not self.has_init_state:
            raise UnsupportedSite("%s: no initial state in this geometry" % site.key())
        if site.component == C.PRELUDE_OUT and not (
                (self.family in RAVEN and self.n_prelude > 0)
                or (self.family == "ifm" and self.has_init_state)):
            raise UnsupportedSite("%s: no prelude output in this geometry" % site.key())

    # ---------------------------------------------------------------- indices
    def _layer_index(self, stage: str, loop: Optional[int], layer: int) -> int:
        if stage == "prelude":
            return layer
        if stage == "loop":
            return self.n_prelude + self.n_core * (loop - 1) + layer
        return self.n_prelude + self.n_core * self.k + layer

    def executed_index(self, site: Site) -> int:
        """0-based executed index of the canonical site (DESIGN 4.2, DR3, DR22)."""
        s = canonical(self, site)
        c = s.component
        if s.layer is not None:
            return self._layer_index(s.stage, s.loop, s.layer)
        if c == C.LOOP_END:
            return self._layer_index("loop", s.loop, self.n_core - 1)
        if c in (C.INJ_INPUT, C.INJ_POST, C.INJ_STATE):
            # INJ_STATE stays canonical only at loop 1 of an IFM twin (DESIGN 4.1)
            return self._layer_index("loop", s.loop, 0)
        if c == C.INIT_STATE:
            return self._layer_index("loop", 1, 0)
        if c == C.PRELUDE_OUT:
            return max(self.n_prelude - 1, 0)
        return 0                                                    # EMBED

    def rank(self, site: Site) -> Tuple[int, int]:
        """Total order of sites: (executed index, sub-rank) with the DESIGN 4.2 tie rule."""
        s = canonical(self, site)
        sub = _SUB_RANK[s.component]
        if s.component == C.PRELUDE_OUT and self.n_prelude == 0:
            sub = -6                                                 # before INIT_STATE at (1, 0)
        return (self.executed_index(s), sub)

    def virtual_layer(self, site: Site) -> int:
        """= executed_index for per-layer sites (DESIGN 4.2)."""
        return self.executed_index(site)

    def first_changed(self, site: Site) -> Optional[int]:
        """F of DESIGN 4.5: the first executed slot whose K/V can change; None when no K/V changes."""
        s = canonical(self, site)
        if s.component == C.PRELUDE_OUT:
            return self.n_prelude                                    # (1, 0)
        e = self.executed_index(s)
        if s.component in _F_SELF:
            return e
        if s.component in _F_NEXT:
            return e + 1 if e + 1 < self.total() else None
        raise SteerConfigError("no F rule for %s" % s.key())

    def cache_slot(self, site: Site) -> Optional[int]:
        """Physical cache slot of the layer a per-layer site belongs to (DESIGN 4.2).

        Ouro (j-1)L + l [read base 320]; raven prelude l, core P + C(j-1) + l, coda the tensor index
        max_num_steps - (l+1) of block_idx -(l+1) [read huginn 675-698]; IFM the slot number.
        """
        s = canonical(self, site)
        if s.layer is None:
            return None
        if self.family == "ouro":
            return (s.loop - 1) * self.n_core + s.layer
        if self.family in RAVEN and s.stage == "coda":
            return self.total() - (s.layer + 1)
        return self.executed_index(s)

    def all_sites(self, components: Iterable[Union[str, Component]]) -> List[Site]:
        """Every canonical site with one of these components, in rank order, each tensor once."""
        comps = set(Component(c) for c in components)
        cands = []
        for c in comps:
            if c in INPUT_COMPONENTS:
                cands.append(Site("input", None, None, c))
            elif c in PER_LOOP:
                cands.extend(Site("loop", j, None, c) for j in range(1, self.k + 1))
            else:
                for stage, n in (("prelude", self.n_prelude), ("coda", self.n_coda)):
                    cands.extend(Site(stage, None, l, c) for l in range(n))
                cands.extend(Site("loop", j, l, c) for j in range(1, self.k + 1)
                             for l in range(self.n_core))
        out = {}
        for s in cands:
            try:
                cs = canonical(self, s)
            except UnsupportedSite:
                continue
            if cs.component in comps:
                out[cs] = None
        return sorted(out, key=self.rank)


# ------------------------------------------------------------------ aliases (DESIGN 4.1)
def canonical(geom: Geometry, site: Site) -> Site:
    """Map an alias to the one tensor that is actually hooked, so no tensor is hooked twice."""
    if geom.family == "ouro" and site.stage in ("prelude", "coda"):
        raise UnsupportedSite("%s: Ouro has no %s" % (site.key(), site.stage))
    geom.check(site)
    c, fam = site.component, geom.family
    if fam == "ouro":
        if c == C.EMBED:
            return Site("loop", 1, 0, C.RESID_PRE)
        return site
    if fam in RAVEN:
        return _canonical_raven(geom, site)
    return _canonical_ifm(geom, site)


def _canonical_raven(geom: Geometry, site: Site) -> Site:
    c, last = site.component, geom.n_core - 1
    if c == C.EMBED:
        return Site("prelude", None, 0, C.RESID_PRE)
    if c == C.PRELUDE_OUT:
        return Site("prelude", None, geom.n_prelude - 1, C.RESID_POST)
    if c == C.INJ_STATE:
        if site.loop == 1:
            return Site("input", None, None, C.INIT_STATE)
        return Site("loop", site.loop - 1, last, C.RESID_POST)    # iterate_forward passes xk = x
    if c == C.INJ_POST:
        return Site("loop", site.loop, 0, C.RESID_PRE)
    if c == C.LOOP_END:
        return Site("loop", site.loop, last, C.RESID_POST)
    return site


def _canonical_ifm(geom: Geometry, site: Site) -> Site:
    c, last = site.component, geom.n_core - 1
    if c == C.INIT_STATE and not geom.injection:
        # no injection call: x0 is the loop-1 input of the first core layer [derived from
        # k10_ifm.fwd:259-266, `state` goes straight into the loop-1 layers]
        return Site("loop", 1, 0, C.RESID_PRE)
    if c == C.LOOP_END:
        return site if geom.exit_norm else Site("loop", site.loop, last, C.RESID_POST)
    if (c == C.RESID_PRE and site.stage == "loop" and site.layer == 0 and geom.injection
            and not geom.dense_inj_at_loop_start):
        return Site("loop", site.loop, None, C.INJ_POST)
    if c == C.INJ_STATE:
        if site.loop >= 2:
            return _canonical_ifm(geom, Site("loop", site.loop - 1, None, C.LOOP_END))
        if geom.has_init_state:
            return Site("input", None, None, C.INIT_STATE)
        return site                                                  # twin: state arg of call 1
    if c == C.EMBED:
        if geom.has_init_state and geom.n_prelude >= 1:
            return Site("prelude", None, 0, C.RESID_PRE)
        if not geom.injection and not geom.has_init_state and not geom.dense_inj_at_loop_start:
            first = (Site("prelude", None, 0, C.RESID_PRE) if geom.n_prelude
                     else Site("loop", 1, 0, C.RESID_PRE))
            return first
        return site
    return site


# ------------------------------------------------------------------ positions (DESIGN 4.3)
POSITION_KINDS = ("abs", "span", "span_to_prompt_end", "after_span", "last_prompt", "generated",
                  "from_last_prompt", "all")


@dataclass(frozen=True)
class Positions:
    """A set of absolute positions of one row's unpadded sequence (DESIGN 4.3)."""
    kind: str
    abs: Tuple[int, ...] = ()
    span: str = ""

    def __post_init__(self) -> None:
        if self.kind not in POSITION_KINDS:
            raise SteerConfigError("unknown positions kind %r" % (self.kind,))
        if self.kind in ("span", "span_to_prompt_end", "after_span") and not self.span:
            raise SteerConfigError("positions kind %r needs a span name" % self.kind)
        object.__setattr__(self, "abs", tuple(int(p) for p in self.abs))

    def _span(self, row: Any) -> Tuple[int, ...]:
        sp = row.spans.get(self.span)
        if not sp:
            raise SteerConfigError("row %r has no span %r" % (row.key, self.span))
        return tuple(sp)

    def resolve(self, row: Any, abs_pos: Tensor, valid: Tensor) -> Tensor:
        """-> bool[T] for one row; abs_pos long[T] and valid bool[T] are that row's columns."""
        P = int(row.prompt_len)
        k = self.kind
        if k == "all":
            m = torch.ones_like(valid)
        elif k == "last_prompt":
            m = abs_pos == P - 1
        elif k == "generated":
            m = abs_pos >= P
        elif k == "from_last_prompt":
            m = abs_pos >= P - 1
        elif k == "abs":
            m = torch.isin(abs_pos, torch.tensor(self.abs, dtype=abs_pos.dtype,
                                                 device=abs_pos.device))
        else:
            sp = self._span(row)
            if k == "span":
                m = torch.isin(abs_pos, torch.tensor(sp, dtype=abs_pos.dtype, device=abs_pos.device))
            elif k == "span_to_prompt_end":
                m = (abs_pos >= min(sp)) & (abs_pos <= P - 1)
            else:                                                     # after_span
                m = abs_pos > max(sp)
        return m & valid


# ------------------------------------------------------------------ read-outs (DESIGN 4.5)
ReadoutSpec = Union[str, Tuple[Any, ...]]
#: per-loop read-out modes each family hosts (OuroBinding.readout, raven.READOUT_MODES,
#: ifm.READOUT_MODES; DESIGN 8.1-8.3)
FAMILY_READOUT_MODES = {"ouro": ("exit", "logit_lens"), "huginn": ("pure", "cached", "cached_ref"),
                        "mcleish": ("pure", "cached", "cached_ref"),
                        "ifm": ("final_norm_lens", "coda_lens")}
#: raven modes whose coda query attends to coda K/V of earlier positions in the cache (DESIGN 8.2)
_CACHED_MODES = ("cached", "cached_ref")


def _parse_ro(geom: Geometry, ro: ReadoutSpec) -> Tuple[Optional[int], str]:
    """-> (loop j or None for "final", lens mode). A mode the family does not host raises."""
    if ro == "final":
        return None, "final"
    if not (isinstance(ro, tuple) and len(ro) in (2, 3) and ro[0] == "loop"):
        raise SteerConfigError("read-out must be 'final' or ('loop', j[, mode]): %r" % (ro,))
    j = int(ro[1])
    if not 1 <= j <= geom.k:
        raise SteerConfigError("read-out loop %d outside 1..%d" % (j, geom.k))
    if len(ro) == 3:
        mode = str(ro[2])
        if mode not in FAMILY_READOUT_MODES[geom.family]:
            raise SteerConfigError("read-out mode %r is not a %s mode %s"
                                   % (mode, geom.family, FAMILY_READOUT_MODES[geom.family]))
        return j, mode
    if geom.family == "ouro":
        return j, "exit"
    if geom.family in RAVEN:
        return j, "pure"
    return j, "coda_lens" if geom.n_coda > 0 else "final_norm_lens"


def own_attn(geom: Geometry, ro: ReadoutSpec) -> bool:
    """True when the read-out runs attention of its own (raven and IFM coda lenses)."""
    j, mode = _parse_ro(geom, ro)
    if j is None or geom.n_coda == 0:
        return False
    return geom.family in RAVEN or mode == "coda_lens"


def readout_input_rank(geom: Geometry, ro: ReadoutSpec) -> Tuple[int, int]:
    """R(ro): the rank of the read-out's input state (DESIGN 4.5)."""
    j, _mode = _parse_ro(geom, ro)
    if j is None:
        if geom.n_coda > 0:
            return geom.rank(Site("coda", None, geom.n_coda - 1, C.RESID_POST))
        return geom.rank(Site("loop", geom.k, None, C.LOOP_END)
                         if geom.family == "ouro" or geom.exit_norm
                         else Site("loop", geom.k, geom.n_core - 1, C.RESID_POST))
    return geom.rank(Site("loop", j, None, C.LOOP_END))


def _identity_set(geom: Geometry, ro: ReadoutSpec) -> Set[Site]:
    j, _mode = _parse_ro(geom, ro)
    if j is None and geom.n_coda > 0:
        return {Site("coda", None, geom.n_coda - 1, C.RESID_POST)}
    if j is not None and own_attn(geom, ro):
        return set()                                                # coda blocks run in the lens
    j = geom.k if j is None else j
    out = {Site("loop", j, geom.n_core - 1, C.RESID_POST)}
    if geom.family == "ouro" or (geom.family == "ifm" and geom.exit_norm):
        out.add(Site("loop", j, None, C.LOOP_END))
    return out


def readout_identity(geom: Geometry, site: Site, ro: ReadoutSpec) -> bool:
    """True when only norms and the head run between the site and the read-out (DR13, DR15)
    [derived]: the read-out at the read position is then a function of the edited tensor alone."""
    s = canonical(geom, site)
    if s.component not in (C.RESID_POST, C.LOOP_END):
        return False
    return s in _identity_set(geom, ro)


def structural_zero(geom: Geometry, site: Site, S: Iterable[int], read: int,
                    ro: ReadoutSpec) -> bool:
    """True when an edit at `site` on positions S cannot change the read-out at `read` (DR22)
    [derived from DESIGN 4.5 items 1-3]: none of (a) read in S at a site ranked at or before R(ro),
    (b) an earlier edited position whose changed K/V (from F) is read at or before R(ro), (c) an
    own-attention read-out over an earlier edited position ranked at or before R(ro).

    Raven `cached` / `cached_ref`: the read-out's coda query attends to the coda K/V that the final (loop-k)
    coda pass wrote at earlier positions (DESIGN 8.2 "the amateur query attends to expert K/V of earlier
    positions"), so (b) uses the final read-out rank instead of R(loop j) [derived from DESIGN 8.2].
    (c) recomputes earlier positions from the lens input, so it keeps R(ro) and applies only where the lens
    runs over earlier positions: `cached` runs the last column alone (no (c), its only route to earlier
    positions is coda K/V, covered by (b)); `cached_ref` runs every column of the forward over the live
    cache (amateur K/V of the same forward, (c) with R(ro)) [core fixer A-F2 / B-F7]."""
    S = sorted(set(int(p) for p in S))
    if not S:
        raise SteerConfigError("structural_zero needs a non-empty position set")
    r_site = geom.rank(site)
    R = readout_input_rank(geom, ro)
    F = geom.first_changed(site)
    _j, mode = _parse_ro(geom, ro)
    R_early = readout_input_rank(geom, "final") if mode in _CACHED_MODES else R
    if int(read) in S and r_site <= R:
        return False
    if S[0] < read and F is not None and F <= R_early[0]:
        return False
    if own_attn(geom, ro) and mode != "cached" and S[0] < read and r_site <= R:
        return False
    return True


# ------------------------------------------------------------------ selector site set (MR2)
def _ifm_full_stack(geom: Geometry) -> bool:
    return geom.n_prelude == 0 and geom.n_coda == 0


def answer_site_set(geom: Geometry, below_k: bool = False, last_loop: str = "all") -> List[Site]:
    """The selector site set S of EXPERIMENT_MATRIX 5.2 (PLAN A7) at the read position, executed
    order, one site per (loop, layer) cell, never the site that feeds the read-out (DESIGN 4.5).

    Ouro: {LOOP_END(j) : j < k} U {RESID_POST(j, l) : l <= L-2}; |S| = 95 at L = 24, k = 4
    [derived: 3 + 23*4]. IFM full stack and d112: every block output but the last executed slot.
    raven and the IFM Huginn-class models with a prelude (the middle pair): the core outputs
    RESID_POST(j, l), |S| = C k.
    """
    if last_loop not in ("all", "l0_lmid"):
        raise SteerConfigError("last_loop must be 'all' or 'l0_lmid'")
    k, n = geom.k, geom.n_core
    if geom.family == "ouro":
        out = []
        for j in range(1, k + 1):
            out.extend(Site("loop", j, l, C.RESID_POST) for l in range(n - 1))
            if j < k:
                out.append(Site("loop", j, None, C.LOOP_END))
    elif geom.family in RAVEN or (geom.family == "ifm" and geom.n_prelude > 0):
        out = [Site("loop", j, l, C.RESID_POST) for j in range(1, k + 1) for l in range(n)]
    elif geom.family == "ifm" and _ifm_full_stack(geom):
        if k == 1 and n != 112:
            raise UnsupportedSite("answer_site_set: k = 1 with C = %d has no loop axis (d28)" % n)
        out = [Site("loop", j, l, C.RESID_POST) for j in range(1, k + 1) for l in range(n)][:-1]
    else:
        raise UnsupportedSite("answer_site_set does not cover family %s with this geometry"
                              % geom.family)
    if below_k:
        out = [s for s in out if s.loop < k]
    if last_loop == "l0_lmid":
        keep = {0, n // 2}
        out = [s for s in out if s.loop < k or (s.layer in keep and s.component == C.RESID_POST)]
    bad = [s.key() for s in out if readout_identity(geom, s, "final")]
    if bad:
        raise SteerConfigError("answer_site_set holds read-out identity sites %s (matrix C9)" % bad)
    return out


_MATRIX_RE = re.compile(r"^(L0|Lmid|Llast)@(\d+|K)$")


def _loop_named(geom: Geometry, lab: str, j: int) -> Site:
    if not 1 <= j <= geom.k:
        raise SteerConfigError("matrix site loop %d outside 1..%d" % (j, geom.k))
    if lab == "L0":
        return canonical(geom, Site("loop", j, 0, C.RESID_POST))
    if lab == "Lmid":
        mid = canonical(geom, Site("loop", j, geom.n_core // 2, C.RESID_POST))
        # one site per (loop, layer) cell (DESIGN 4.5): on C <= 2 (IFM P2, P/C/Cd = 1/2/1) the n//2
        # layer is the L0 or Llast tensor, and recon_protocol F10 says there is no Lmid there
        if mid in (canonical(geom, Site("loop", j, 0, C.RESID_POST)),
                   canonical(geom, Site("loop", j, None, C.LOOP_END))):
            raise UnsupportedSite("Lmid@%d is the L0 or Llast tensor %s at C = %d (no Lmid)"
                                  % (j, mid.key(), geom.n_core))
        return mid
    return canonical(geom, Site("loop", j, None, C.LOOP_END))


def matrix_site(geom: Geometry, name: str, mapped_from: Optional[Geometry] = None) -> Site:
    """Matrix site names (EXPERIMENT_MATRIX 5.2, CB-F13) -> canonical sites (DESIGN 4.5).

    L0@j, Lmid@j, Llast@j (j an integer or K), LAST, IDX-latest and input.embed. On d112 and d28 the
    names need the paired looped geometry `mapped_from`: d112 takes the layer at the paired site's
    executed index, d28 (the unique-layer control of the twin) the paired site's layer index.
    """
    if mapped_from is not None:
        return _matrix_site_mapped(geom, name, mapped_from)
    if name == "input.embed":
        return canonical(geom, Site("input", None, None, C.EMBED))
    if name == "LAST":
        middle = geom.family in RAVEN or (geom.family == "ifm" and not _ifm_full_stack(geom))
        return _loop_named(geom, "Llast" if middle else "Lmid", geom.k)
    if name == "IDX-latest":
        return max(answer_site_set(geom), key=geom.rank)
    m = _MATRIX_RE.match(name)
    if m is None:
        raise SteerConfigError("unknown matrix site name %r" % (name,))
    j = geom.k if m.group(2) == "K" else int(m.group(2))
    return _loop_named(geom, m.group(1), j)


def _matrix_site_mapped(geom: Geometry, name: str, pair: Geometry) -> Site:
    if not (geom.family == "ifm" and geom.k == 1 and _ifm_full_stack(geom)):
        raise SteerConfigError("mapped_from applies to the plain d28/d112 geometries only")
    ps = matrix_site(pair, name)
    if geom.n_core == pair.total():                                  # d112: same executed index
        return Site("loop", 1, pair.executed_index(ps), C.RESID_POST)
    if geom.n_core == pair.n_core and _ifm_full_stack(pair):         # d28: same layer of the loop
        cs = canonical(pair, ps)
        if cs.layer is None:
            cs = Site("loop", cs.loop, pair.n_core - 1, C.RESID_POST)
        return Site("loop", 1, cs.layer, C.RESID_POST)
    raise SteerConfigError("cannot map %s from a geometry with total %d onto C = %d"
                           % (name, pair.total(), geom.n_core))


def ifm_matched_d112(geom: Geometry, site: Site) -> int:
    """The d112 layer matched to a site of a geometry whose total is 112 (recon_protocol F14:
    28(j-1)+l, 12(j-1)+g, coda 84+g); raises for P2 and every other total."""
    if geom.family != "ifm" or geom.total() != 112:
        raise UnsupportedSite("ifm_matched_d112 needs an IFM geometry with 112 executed slots, "
                              "got %s with %d" % (geom.family, geom.total()))
    return geom.executed_index(site)

