# ---------------------------------------------------------------------------------------------------------
# EXACT-EDIT PLAN (builder B2, module interventions; written before the code below, kept for the reviewer)
#
# Files written: prod/steer/interventions.py (this file), tests/steer/test_interventions.py. Nothing else.
# Imports (read-only): prod.steer.sites (Site, Component, Positions, Geometry, canonical, matrix_site) and
# prod.steer.hooks (RowInfo, ForwardCtx, RowKey, SteerConfigError, UnsupportedSite), coded against DESIGN.md
# sections 4-5 because B0 has not written them yet. prod.steer.readouts.Recorder is imported lazily, only by
# from_spec for `extrapolate` (DESIGN 7.1 "auto-registers an internal recorder").
#
# 1. Intervention dataclass (DESIGN 7): fields id, op, sites, positions, a, params, rows, trainable, plus
#    `recorders` (internal recorders the registry must add with the intervention; extrapolate only).
#    edit(x, site, row, h_pos, ctx) dispatches to pure float32 op functions; knock(site, row, ctx) returns
#    absolute (queries, keys) at routed calls. Skips are read by the registry from iv.sites (hooks.py).
#    h_pos = absolute positions (long [n], ascending) of the n selected rows of x (DESIGN 4.3, 5 contract).
#    hooks.HookRegistry.add_intervention registers iv.recorders with iv (DESIGN 7.1 extrapolate).
# 2. Ops (DESIGN 7.1 table), each a pure function with its source cited:
#    patch (torch.lerp; k15_smoke.op_apply O1), add raw/relnorm (O2; k15_smoke2.caa_delta), normadd (O3),
#    project (ReFT DII 2404.03592 Eq.1), width (k16_path_ouro.proj_rescale), random (O4 with the per-(row,
#    site, position) sha256 generator rule), extrapolate (2610.02185 Eq.2), knockout (calls + Q/K sets; mask
#    work is the binding's), skip, kv_patch (expands to patch at K_PROJ and V_PROJ), loreft (2404.03592 Eq.2,
#    dropout on the edit term only). Every strength op returns x itself at strength 0.
# 3. Payload alignment (DESIGN 7 "Payload alignment"): payload rows map to the row's full resolved position
#    set in ascending order; open-ended kinds map by offset. A length mismatch raises (P4: never broadcast).
# 4. expand(template, schedule, geom): `loop*`, `L*`, layer aliases, kinds once / every_loop / from_loop /
#    loops / post_last_injection, `calls_from` (int F or {"site": key}); canonical + order-preserving dedupe.
# 5. from_spec(spec, geom): REQUIRED-field checks (SteerConfigError), U orthonormality, extrapolate ordering,
#    kv_patch expansion, knockout and skip call sets, loreft module lookup, IFM refusals (UnsupportedSite).
# 6. Strength matching (P1, DR9): UnitNorm (dict + op), unit_norm, abs_matched_a, abs_matched_arms.
# 7. Protocol builders: y2_grid (P13), ko_conditions (K3, MR8, path audit F3), restore_sites (K4, DR3, DR4,
#    DR16, MR7), suffix_grid + suffix_calls (matrix X6), slot_rule (port of k17_varbench.window_rule_v4, the v4
#    rule, plus the DESIGN 22.7 band clause and r_pre/r_post/last_read/W rules), matched_keys +
#    matched_cell_status (matrix X4b), freeze_spec (matrix X4a), mean_target, dev_item_donor and
#    replacement_ratio (matrix X4e), describe (record fields).
# 8. NOT built: tf_position_sets (DESIGN 22.11). It serves only LCD-H2 / the CT rows, which
#    EXPERIMENT_MATRIX.md section 14 (production gate, binding for every prod builder) lists as gated
#    "design only, no module until its pilot passes" (K17 GEN). Reported as a deviation.
# 9. Tests: tests/steer/test_interventions.py, every DESIGN 15.2 test-5 bullet plus T22.13, T22.14, T22.15
#    and T22.18, against the real sites/hooks/readouts modules.
# 10. Second pass (same role, on top of the first pass, after B0's sites/hooks/readouts landed): drop the
#    unused Intervention.skips (hooks.HookRegistry keeps its own skip set); suffix_grid takes
#    K from `mapped_from` on d28/d112 (their own k is 1); unit_norm of random/width raises SteerConfigError
#    when the reference Intervention carries no spec; tests drive the real HookRegistry through a local
#    fake looped binding, and (fixer round 2) the real Ouro and raven bindings over the conftest fakes.
# ---------------------------------------------------------------------------------------------------------
"""Interventions: the edit operators, schedules and pathway protocols of the steering study.

What this module measures and why. The study asks where an edit of a looped LM's hidden state changes the
prediction and which later pathway keeps or removes it (DESIGN.md goal G1), whether an edited value transfers
under generation (G2), and how steering compares with baselines on benchmarks (G3). Each question is answered
by running a forward clean and again with an intervention at a site (stage x loop x layer x component x
positions, DESIGN 4.1). This module defines the interventions and runs no model:

- the operators of DESIGN 7.1 (patch, add, normadd, project, width, random, extrapolate, knockout, skip,
  kv_patch, loreft) as pure float32 functions, each an exact identity at strength 0;
- schedules that place one site template at the loops of a looped model (DESIGN 7.2, 22.6);
- strength matching across schedules, relative and absolute (audit P1 / C3, DESIGN DR9): the residual norm
  grows across loops, so equal relative strength is not an equal push;
- the pathway protocols: the knockout condition set with the relay control (DESIGN 7.3, K3), restore sites for
  undo from the measured read window (K4), the matrix X6 suffix grid and slot rule, the X4a branch freeze, the
  X4b size-matched unrelated keys, the X4e ablation targets and the Y2 input-channel grid (audit P13).

The registry (prod.steer.hooks) calls `Intervention.edit` with the float32 rows of one row's selected
positions and writes the result back in the model dtype (DESIGN 5, "Contract of site"). Knockout and skip are
consumed by the registry through `Intervention.knock` and through the call sites in `Intervention.sites`; their
`edit` is the identity.

Contracts this module defines for its consumers (DESIGN leaves them to the owner of interventions.py):
- `h_pos` passed to `edit` holds the absolute positions (long [n], ascending) of the n rows of `x`.
- At HEAD_OUT, `x` is [n, H, hd] (the Ouro and raven bindings view the o_proj / attn.proj input as [B, T, H,
  hd], DESIGN 6.1 hook 3, 6.2 hook 4; adapters/raven.py _on_proj) or flat [n, H*hd]; `edit` works on the flat
  [n, H*hd] form and returns x's shape. A HEAD_OUT payload may be [n, H*hd] or [n, H, hd].
- `params["heads"]` keeps an edit on the listed heads. It is allowed only at HEAD_OUT sites, with distinct
  heads in 0..H-1, and only for ops whose equation still holds on those heads' coordinates: patch, raw add and
  extrapolate act per coordinate; project and width need U inside the heads; random restricts and renormalizes
  its draw. normadd, relnorm add and loreft refuse a head set (their norms are taken over the full vector).
  `random` and `width` use the head set of their reference (`params["of"]`).
- `random` and `width` read the payload of their reference under the reference's id
  (`row.payload[of.id]`); `payload_needs(iv)` lists every payload an intervention reads. They refuse a
  reference that has more than one site and whose delta depends on h (every op but `add`), because the delta
  is recomputed on the control's own, already moved state at later sites.
- Knockout calls are named by `call_component(geom)` sites (ATTN_OUT, as the `attn_mass` recorder of DESIGN 8;
  RESID_POST on IFM, which hosts no ATTN_OUT) and skipped layers by RESID_POST sites (SKIP_COMPONENT); only
  (stage, loop, layer) is matched.
- `extrapolate` reads its reference state from `ctx.store[("state", recorder_id, ref_site_key)]`, the full
  forward tensor [B, T, d] ([B, T, H, hd] at HEAD_OUT, flattened like x) that its internal recorder in
  `recorders` (when="post", to_cpu=False) stores through hooks.HookRegistry._record_state; the edit takes the
  row's batch index and the columns whose abs_pos equal h_pos. `ref_site` and every edited site have the same
  canonical component. The registry adds `iv.recorders` together with `iv`. `random` and `width` carry a copy
  of their reference whose recorder id is "<control id>|ref", so several controls of one reference can share a
  registry.
- Open-ended position kinds `all`, `after_span` and `from_last_prompt`: in a forward that holds the first
  position of the set (a prefill), the payload has at least one row per position from that start through the
  later of the prompt end and the forward's last position; further rows serve later decode steps, so one
  payload serves the prefill and every decode step of a generate call. A missing row raises (never broadcast).
- `loreft` reads per-site modules from `params["modules"]`; a module exposes tensors `R` [r, d], `W` [r, d],
  `b` [r] and a callable `dropout` applied to the edit term.
"""
import copy
import dataclasses
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from prod.steer.hooks import ForwardCtx, RowInfo, RowKey, SteerConfigError, UnsupportedSite
from prod.steer.readouts import Recorder
from prod.steer.sites import Component, Geometry, Positions, Site, canonical, matrix_site

__all__ = [
    "Intervention", "UnitNorm", "from_spec", "expand", "unit_norm", "abs_matched_a", "abs_matched_arms",
    "y2_grid", "ko_conditions", "restore_sites", "suffix_grid", "suffix_calls", "attention_calls",
    "slot_rule", "matched_keys", "matched_cell_status", "freeze_spec", "mean_target", "dev_item_donor",
    "replacement_ratio", "unit_gaussian", "first_changed", "describe", "payload_needs",
]

OPS = ("patch", "add", "normadd", "project", "width", "random", "extrapolate", "knockout", "skip",
       "kv_patch", "loreft")
TARGET_KINDS = ("donor", "clean", "mean", "dev_item")          # DESIGN 7.1 patch, 22.6 X4e
PROJECT_TARGETS = ("zero", "donor", "clean", "mean")           # DESIGN 7.1 project: t = 0, donor or mean
STRENGTH_KINDS = ("raw", "relnorm")                            # DESIGN 7.1 add
SCHEDULE_KINDS = ("once", "every_loop", "from_loop", "loops", "post_last_injection")   # DESIGN 7.2
REFERENCE_OPS = ("patch", "add", "normadd", "project", "width", "random", "extrapolate", "loreft")
PROMPT_BOUNDED = ("all", "after_span", "from_last_prompt")    # open-ended kinds whose set starts in the prompt
PER_LOOP = (Component.LOOP_END, Component.INJ_STATE, Component.INJ_INPUT, Component.INJ_POST)
INPUT_ONLY = (Component.EMBED, Component.PRELUDE_OUT, Component.INIT_STATE)
X_FREE_OPS = ("add",)                      # ops whose delta does not depend on h (DESIGN 7.1 add: delta = a v)
CALL_COMPONENT = Component.ATTN_OUT        # names a knockout call (DESIGN 8 attn_mass convention)
CALL_COMPONENT_IFM = Component.RESID_POST  # IFM v1 hosts no ATTN_OUT (DESIGN 4.1), so its calls use RESID_POST
SKIP_COMPONENT = Component.RESID_POST      # names a skipped layer; canonical on every family (DESIGN 4.1)
RESTORE_COMPONENTS = (Component.RESID_POST, Component.LOOP_END)   # DESIGN 7.3 "A restore site is ..."

RANDOM_SEED_MULT = 1000003                 # DESIGN 7.1 random: (seed + 1000003 h) mod 2^63
SEED_MOD = 2 ** 63
ORTHO_TOL = 1e-5                           # DESIGN 7.1 project: ||U^T U - I||_inf < 1e-5 (float32)
NORM_EPS = 1e-12                           # k15_smoke.op_apply O3 and k16_path_ouro.proj_rescale clamp_min
CARRIED_GAP = 2.0                          # DESIGN 22.7 e_carried "denominator >= 2" (k17_varbench GAP)
CARRIED_MIN_N = 10                         # matrix X6 rule (b) "over >= 10 items" (k17_varbench.window_rule_v4 min_n)
KU_FALLBACK_TOL = 0.25                     # matrix X4b fallback: |dm_clean(K_u)| within +-25%
KU_UNMATCHED_MAX = 0.30                    # matrix X4b pass: > 30% unmatched -> "no matched control"
VACUOUS_RATIO = 0.5                        # matrix X4e: replacement size ratio < 0.5 -> "vacuous"


# ------------------------------------------------------------------------------------------ small helpers
def _seed_int(s: str) -> int:
    """sha256 seed rule of DESIGN 7.3 (MR8) and 22.6: int(sha256(s).hexdigest()[:16], 16)."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def _req(d: Dict[str, Any], key: str, what: str) -> Any:
    """A REQUIRED field (DESIGN 0 rule 3): a missing or null value raises SteerConfigError."""
    if not isinstance(d, dict) or d.get(key) is None:
        raise SteerConfigError("%s: REQUIRED field '%s' is missing" % (what, key))
    return d[key]


def _key(site: Union[Site, str]) -> str:
    return site if isinstance(site, str) else site.key()


def _n_blocks(geom: Geometry, stage: str) -> int:
    return {"prelude": geom.n_prelude, "loop": geom.n_core, "coda": geom.n_coda}[stage]


def _call_index(geom: Geometry, stage: str, loop: Optional[int], layer: int) -> int:
    """Executed index of block call (stage, loop, layer), DESIGN 4.2: l; P + C(j-1) + l; P + C*k + l."""
    if stage == "prelude":
        return layer
    if stage == "loop":
        return geom.n_prelude + geom.n_core * (loop - 1) + layer
    if stage == "coda":
        return geom.n_prelude + geom.n_core * geom.k + layer
    raise SteerConfigError("stage %r has no block calls" % stage)


def _physical_key(site: Site) -> str:
    """Key shared by every loop of one physical layer (`tie_loops`, DESIGN 7.1 add and loreft)."""
    if site.stage != "loop":
        return site.key()
    if site.layer is None:
        return "loop*.%s" % site.component.value
    return "loop*.L%d.%s" % (site.layer, site.component.value)


def _positions(spec: Any) -> Positions:
    """Positions from a spec dict ({"kind", "abs", "span"}), a list of absolute ints, or a Positions."""
    if isinstance(spec, Positions):
        return spec
    if isinstance(spec, (list, tuple)):
        return Positions(kind="abs", abs=tuple(int(p) for p in spec))
    if not isinstance(spec, dict) or "kind" not in spec:
        raise SteerConfigError("positions must be a dict with 'kind', a list of ints or a Positions: %r" % (spec,))
    return Positions(kind=spec["kind"], abs=tuple(int(p) for p in spec.get("abs", ())), span=spec.get("span", ""))


def _positions_dict(p: Positions) -> Dict[str, Any]:
    out = {"kind": p.kind}  # type: Dict[str, Any]
    if p.abs:
        out["abs"] = list(p.abs)
    if p.span:
        out["span"] = p.span
    return out


def _item(item: Any, name: str) -> Any:
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _read_pos(item: Any, read_idx: int = 0) -> int:
    """Read position r of `item.reads[read_idx]` = (r, tA, tB); an empty `reads` means
    [(spans["answer"][-1], tA, tB)] (DESIGN 12 PairItem.reads, PF7)."""
    reads = item.get("reads") if isinstance(item, dict) else getattr(item, "reads", None)
    if not reads:
        reads = [(int(_item(item, "spans")["answer"][-1]),)]
    if not 0 <= int(read_idx) < len(reads):
        raise SteerConfigError("read_idx %d outside the item's %d reads" % (read_idx, len(reads)))
    return int(reads[int(read_idx)][0])


def _source(item: Any) -> List[int]:
    return sorted(int(p) for p in _item(item, "spans")["source"])


def _mean(xs: Sequence[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


# ------------------------------------------------------------------------------------------ templates
def _parse_template(template: str) -> Tuple[str, Any, Any, Component]:
    """'loop*.L*.mlp_out' -> ("loop", "*", "*", MLP_OUT). Loop token: int or "*"; layer token: int, "*" or an
    alias (first | mid | last, DESIGN 4.1); None where the component has no loop or layer."""
    parts = template.split(".")
    try:
        comp = Component(parts[-1])
    except ValueError:
        raise SteerConfigError("unknown component in site template %r" % template)
    stage_tok, middle = parts[0], parts[1:-1]
    if stage_tok == "input":
        if middle or comp not in INPUT_ONLY:
            raise SteerConfigError("bad input-stage template %r" % template)
        return "input", None, None, comp
    if stage_tok.startswith("loop"):
        stage, rest = "loop", stage_tok[4:]
        if rest != "*" and not rest.isdigit():
            raise SteerConfigError("bad loop token in %r" % template)
        loop_tok = "*" if rest == "*" else int(rest)  # type: Any
    elif stage_tok in ("prelude", "coda"):
        stage, loop_tok = stage_tok, None
    else:
        raise SteerConfigError("unknown stage in site template %r" % template)
    if comp in PER_LOOP:
        if middle or stage != "loop":
            raise SteerConfigError("per-loop component needs 'loop<j>.<component>': %r" % template)
        return stage, loop_tok, None, comp
    if len(middle) != 1 or not middle[0].startswith("L"):
        raise SteerConfigError("per-layer template needs one 'L<layer>' token: %r" % template)
    lt = middle[0][1:]
    layer_tok = "*" if lt == "*" else (int(lt) if lt.isdigit() else lt)  # type: Any
    return stage, loop_tok, layer_tok, comp


def _schedule_loops(schedule: Dict[str, Any], geom: Geometry) -> Optional[List[int]]:
    """Loops a `loop*` template expands to (DESIGN 7.2); None for `once`."""
    kind = _req(schedule, "kind", "schedule")
    if kind not in SCHEDULE_KINDS:
        raise SteerConfigError("unknown schedule kind %r" % kind)
    k = geom.k
    if kind == "once":
        return None
    if kind == "every_loop":
        return list(range(1, k + 1))
    if kind == "from_loop":
        j = int(_req(schedule, "j", "schedule from_loop"))
        if not 1 <= j <= k:
            raise SteerConfigError("from_loop j=%d outside 1..%d" % (j, k))
        return list(range(j, k + 1))
    if kind == "loops":
        loops = sorted(set(int(j) for j in _req(schedule, "loops", "schedule loops")))
        if not loops or loops[0] < 1 or loops[-1] > k:
            raise SteerConfigError("loops %r outside 1..%d" % (loops, k))
        return loops
    if not geom.injection:   # post_last_injection: loop k, after the last injection
        raise UnsupportedSite("post_last_injection needs an injection model (family %s)" % geom.family)
    return [k]


def _layers(geom: Geometry, stage: str, layer_tok: Any, template: str) -> List[Optional[int]]:
    if layer_tok is None:
        return [None]
    n = _n_blocks(geom, stage)
    if layer_tok == "*":
        return list(range(n))
    layer = geom.resolve_layer(stage, layer_tok)
    if not 0 <= layer < n:
        raise SteerConfigError("layer %r outside 0..%d in %r" % (layer_tok, n - 1, template))
    return [layer]


def _resolve_F(calls_from: Any, geom: Geometry) -> int:
    """`calls_from`: an executed index, or {"site": key} naming the edit whose F (DESIGN 4.5) is used."""
    if isinstance(calls_from, dict):
        F = geom.first_changed(Site.parse(_req(calls_from, "site", "calls_from")))
        if F is None:
            raise SteerConfigError("calls_from: the named edit changes no K/V (F is None)")
        return int(F)
    return int(calls_from)


def _expand_sites(templates: Union[str, Sequence[str]], schedule: Dict[str, Any], geom: Geometry,
                  canonicalize: bool) -> Tuple[Site, ...]:
    templates = [templates] if isinstance(templates, str) else list(templates)
    loops_sched = _schedule_loops(schedule, geom)
    out, any_star = [], False
    for t in templates:
        stage, loop_tok, layer_tok, comp = _parse_template(t)
        if stage == "loop" and loop_tok == "*":
            if loops_sched is None:
                raise SteerConfigError("schedule 'once' needs an explicit loop, got %r" % t)
            any_star, loops = True, loops_sched
        elif stage == "loop":
            if loops_sched is not None:
                raise SteerConfigError("template %r has an explicit loop; its schedule must be 'once'" % t)
            if not 1 <= loop_tok <= geom.k:
                raise SteerConfigError("loop %d outside 1..%d in %r" % (loop_tok, geom.k, t))
            loops = [loop_tok]
        else:
            loops = [None]
        for j in loops:
            for layer in _layers(geom, stage, layer_tok, t):
                s = Site(stage=stage, loop=j, layer=layer, component=comp)
                out.append(canonical(geom, s) if canonicalize else s)
    if loops_sched is not None and not any_star:
        raise SteerConfigError("schedule %r needs a 'loop*' template" % schedule.get("kind"))
    if schedule.get("calls_from") is not None:
        F = _resolve_F(schedule["calls_from"], geom)
        out = [s for s in out if _site_index(geom, s, canonicalize) >= F]
    seen, uniq = set(), []
    for s in out:
        if s.key() not in seen:
            seen.add(s.key())
            uniq.append(s)
    if not uniq:
        raise SteerConfigError("site templates %r expand to no site" % (templates,))
    return tuple(uniq)


def _site_index(geom: Geometry, site: Site, canonicalized: bool) -> int:
    return geom.executed_index(site) if canonicalized else _call_index(geom, site.stage, site.loop, site.layer)


def expand(template: Union[str, Sequence[str]], schedule: Dict[str, Any], geom: Geometry) -> Tuple[Site, ...]:
    """Concrete canonical sites of a site template under a schedule (DESIGN 7.2; v1.5 `L*` and `calls_from`,
    22.6). `loop*` takes the schedule's loops, `L*` every block of the stage, `Lfirst/Lmid/Llast` the aliases of
    DESIGN 4.1, `calls_from` keeps sites with executed index >= F. Templates without `loop*` are placed as
    written. Aliases are canonicalized and deduplicated, so no tensor is hooked twice (DESIGN 4.1)."""
    return _expand_sites(template, schedule, geom, canonicalize=True)


def _call_sites(calls: Any, schedule: Optional[Dict[str, Any]], geom: Geometry, comp: Component) -> Tuple[Site, ...]:
    """Block calls named by templates such as 'loop2.L3', 'loop*.L*' or a full site key (its component is
    replaced by comp)."""
    if isinstance(calls, str):
        calls = [calls]
    templates = []
    for c in calls:
        parts = c.split(".")
        if parts[-1] in {m.value for m in Component}:
            parts = parts[:-1]
        templates.append(".".join(parts + [comp.value]))
    return _expand_sites(templates, schedule or {"kind": "once"}, geom, canonicalize=False)


def call_component(geom: Geometry) -> Component:
    """Component that names a knockout call: ATTN_OUT (DESIGN 8), RESID_POST on IFM, which hosts no ATTN_OUT."""
    return CALL_COMPONENT_IFM if geom.family == "ifm" else CALL_COMPONENT


def attention_calls(geom: Geometry, min_index: int = 0) -> List[Site]:
    """Every attention call (prelude, loop, coda) with executed index >= min_index, in executed order, named by
    `call_component(geom)` sites. Coda calls are included: a relay or source cut that skipped them would leave
    later positions reading the edited source there (DESIGN 7.3 relay exactness)."""
    comp = call_component(geom)
    out = [Site(stage="prelude", loop=None, layer=l, component=comp) for l in range(geom.n_prelude)]
    for j in range(1, geom.k + 1):
        out += [Site(stage="loop", loop=j, layer=l, component=comp) for l in range(geom.n_core)]
    out += [Site(stage="coda", loop=None, layer=l, component=comp) for l in range(geom.n_coda)]
    return [s for s in out if _call_index(geom, s.stage, s.loop, s.layer) >= min_index]


# ------------------------------------------------------------------------------------------ operators
def op_patch(x: Tensor, t: Tensor, a: float) -> Tensor:
    """O1 patch h <- h + a (t - h) (k15_smoke.op_apply:56-57). torch.lerp gives exactly t at a = 1."""
    return x if a == 0 else torch.lerp(x, t, float(a))


def op_add(x: Tensor, delta: Tensor) -> Tensor:
    """O2 add h <- h + delta (k15_smoke.op_apply:58-59)."""
    return x + delta


def op_normadd(x: Tensor, delta: Tensor) -> Tensor:
    """O3 h <- (h + delta) ||h|| / ||h + delta|| per position (k15_smoke.op_apply:60-62)."""
    n = x + delta
    return n * (x.norm(dim=-1, keepdim=True) / n.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS))


def op_project(x: Tensor, U: Tensor, t: Tensor) -> Tensor:
    """Subspace replacement h <- h + U U^T (t - h), U [d, r] orthonormal (ReFT DII 2404.03592 Eq.1 with R = U^T;
    t = 0 is the subspace ablation of W&R 2609.01924 section 7.2)."""
    return x + ((t - x) @ U) @ U.T


def op_width(x: Tensor, delta: Tensor, U: Tensor) -> Tensor:
    """Width control: delta' = U U^T delta * ||delta|| / ||U U^T delta||, h <- h + delta'
    (k16_path_ouro.proj_rescale:82-85)."""
    pj = (delta @ U) @ U.T
    return x + pj * (delta.norm(dim=-1, keepdim=True) / pj.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS))


def op_random(x: Tensor, delta: Tensor, u: Tensor) -> Tensor:
    """O4 random direction at the reference edit's per-position norm: h <- h + ||delta|| u
    (k15_smoke.op_apply:63-64)."""
    return x + delta.norm(dim=-1, keepdim=True) * u


def op_extrapolate(x: Tensor, h_ref: Tensor, w: float) -> Tensor:
    """LoopCD-Hidden h <- h + w (h - h_ref) (arXiv 2610.02185 Eq.2)."""
    return x if w == 0 else x + float(w) * (x - h_ref)


def unit_gaussian(row_key: RowKey, site_key: str, p: int, seed: int, d: int) -> Tensor:
    """Unit Gaussian direction for one (row, site, position), float32 CPU. Generator seed
    (seed + 1000003 h) mod 2^63 with h = int(sha256(f"{row_key}|{site_key}|{p}").hexdigest()[:8], 16)
    [decision, DESIGN 7.1]; normalization as k15_smoke.unit_random:68-71. Position pure."""
    h = int(hashlib.sha256(("%s|%s|%d" % (row_key, site_key, p)).encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator(device="cpu").manual_seed((int(seed) + RANDOM_SEED_MULT * h) % SEED_MOD)
    u = torch.randn(d, generator=g, dtype=torch.float32)
    return u / u.norm()


# ------------------------------------------------------------------------------------------ alignment
def _span(row: RowInfo, name: str) -> Tuple[int, ...]:
    if not row.spans.get(name):
        raise SteerConfigError("row %r has no span %r" % (row.key, name))
    return row.spans[name]


def _open_start(pos: Positions, row: RowInfo) -> Optional[int]:
    """First absolute position of an open-ended kind (DESIGN 4.3); None for kinds with a finite set."""
    if pos.kind == "after_span":
        return max(_span(row, pos.span)) + 1
    if pos.kind == "generated":
        return row.prompt_len
    if pos.kind == "from_last_prompt":
        return row.prompt_len - 1
    if pos.kind == "all":
        return 0
    return None


def _static_positions(pos: Positions, row: RowInfo) -> Optional[List[int]]:
    """The row's full resolved position set, ascending, for kinds with a finite set (DESIGN 4.3)."""
    if pos.kind == "abs":
        return sorted(set(pos.abs))
    if pos.kind == "span":
        return sorted(set(_span(row, pos.span)))
    if pos.kind == "span_to_prompt_end":
        return list(range(min(_span(row, pos.span)), row.prompt_len))
    if pos.kind == "last_prompt":
        return [row.prompt_len - 1]
    if _open_start(pos, row) is not None:
        return None
    raise SteerConfigError("unknown positions kind %r" % pos.kind)


def _aligned(t: Tensor, pos: Positions, row: RowInfo, h_pos: Tensor, what: str) -> Tensor:
    """Rows of a per-row tensor t aligned with the row's resolved positions in ascending order, picked at the
    absolute positions h_pos (DESIGN 7 "Payload alignment"). A length mismatch raises: a tensor is never
    broadcast over positions it was not built for (audit P4 / C2)."""
    if t.dim() != 2:
        raise SteerConfigError("%s must be [n, d], got shape %s" % (what, tuple(t.shape)))
    hp = [int(p) for p in h_pos.tolist()]
    full = _static_positions(pos, row)
    if full is not None:
        if t.shape[0] != len(full):
            raise SteerConfigError("%s has %d rows but row %r resolves %d positions (%s); refusing to broadcast"
                                   % (what, t.shape[0], row.key, len(full), pos.kind))
        where = {p: i for i, p in enumerate(full)}
        if any(p not in where for p in hp):
            raise SteerConfigError("%s: edited positions %s outside the resolved set" % (what, hp))
        idx = [where[p] for p in hp]
    else:
        start = _open_start(pos, row)
        idx = [p - start for p in hp]
        if min(idx) < 0 or max(idx) >= t.shape[0]:
            raise SteerConfigError("%s has %d rows; positions %s (from %d) fall outside" % (what, t.shape[0], hp, start))
        if pos.kind in PROMPT_BOUNDED and min(hp) == start:
            # a prefill holds the set's first position, so the set through the prompt end is known; rows past
            # it serve the decode steps of the same generate call (one RowInfo per decode call, SteeredAdapter)
            want = max(max(hp), row.prompt_len - 1) - start + 1
            if t.shape[0] < want:
                raise SteerConfigError("%s has %d rows but row %r resolves %d positions (%s from %d) in this "
                                       "prefill; refusing to broadcast" % (what, t.shape[0], row.key, want,
                                                                           pos.kind, start))
    return t[torch.tensor(idx, dtype=torch.long, device=t.device)]


def _payload(iv: "Intervention", site: Site, row: RowInfo, h_pos: Tensor, x: Tensor) -> Tensor:
    pl = row.payload.get(iv.id)
    if pl is None or site.key() not in pl:
        raise SteerConfigError("missing payload row.payload[%r][%r] for row %r" % (iv.id, site.key(), row.key))
    t = pl[site.key()]
    if site.component == Component.HEAD_OUT and t.dim() == 3:
        t = t.reshape(t.shape[0], -1)            # recorded at HEAD_OUT as [n, H, hd]
    t = _aligned(t, iv.positions, row, h_pos, "payload %s@%s" % (iv.id, site.key()))
    return t.to(device=x.device, dtype=torch.float32)


def _per_site(table: Any, site: Site, tie: bool, what: str) -> Any:
    """A per-site entry (by site key, or by physical key under tie_loops); a bare tensor is shared."""
    if isinstance(table, Tensor):
        return table
    key = _physical_key(site) if tie else site.key()
    if not isinstance(table, dict) or key not in table:
        raise SteerConfigError("%s has no entry for %r" % (what, key))
    return table[key]


# ------------------------------------------------------------------------------------------ Intervention
@dataclass(eq=False)
class Intervention:
    """One intervention at concrete canonical sites (DESIGN 7). `edit` takes the float32 rows x [n, d] of ONE
    row at one site and returns float32 [n, d]; `knock` returns absolute (queries, keys) at routed calls."""
    id: str
    op: str
    sites: Tuple[Site, ...]
    positions: Positions
    a: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    rows: Optional[FrozenSet[RowKey]] = None
    trainable: bool = False
    recorders: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        knock = self.op == "knockout"
        self._calls = frozenset((s.stage, s.loop, s.layer) for s in self.sites) if knock else frozenset()
        of = self.params.get("of")
        if self.op in ("random", "width") and isinstance(of, Intervention) and len(of.sites) > 1 \
                and _x_dependent(of):
            raise SteerConfigError(
                "%s %s: reference %s (op %s) has %d sites and a delta that depends on h; the control would "
                "recompute it on its own moved state after the first site, so its norm would not match the "
                "reference (DESIGN 7.1 random). Use a single-site reference or an add reference"
                % (self.op, self.id, of.id, of.op, len(of.sites)))
        if self.op in ("random", "width") and isinstance(of, Intervention) and of.recorders:
            # The control recomputes the reference delta, so it needs the reference's internal recorder (the
            # extrapolate ref state). It gets its own copy, id "<control id>|ref", so two controls of one
            # reference (seeds, the sz neutral copy) can share a HookRegistry, which refuses duplicate ids.
            own = _own_reference(of, self.id)
            self.params = dict(self.params, of=own)
            self.recorders = tuple(own.recorders)

    def edit(self, x: Tensor, site: Site, row: RowInfo, h_pos: Tensor, ctx: ForwardCtx) -> Tensor:
        if x.shape[0] == 0:
            return x
        fn = _EDITS.get(self.op)
        if fn is None:
            raise SteerConfigError("intervention %s: op %r has no edit (kv_patch is a spec shorthand; "
                                   "subclasses with their own op override edit)" % (self.id, self.op))
        xf = x
        if site.component == Component.HEAD_OUT and x.dim() == 3:
            xf = x.reshape(x.shape[0], -1)       # Ouro [n, H, hd] view (DESIGN 6.1 hook 3) -> [n, H*hd]
        y = fn(self, xf, site, row, h_pos, ctx)
        if y is xf:
            return x
        heads, head_dim = _heads(self)
        if heads is not None and site.component == Component.HEAD_OUT:
            y = _restrict_heads(xf, y, heads, head_dim)
        return y.reshape(x.shape)

    def knock(self, site: Site, row: RowInfo, ctx: ForwardCtx) -> Optional[Tuple[Tensor, Tensor]]:
        """Absolute (query, key) positions knocked at this call, or None when the call is not routed. Raises
        SteerConfigError when a query would keep no key (full-row guard, DESIGN 6.1 S6); the binding repeats
        the guard on the real mask, where padding is known."""
        if not self.routes(site):
            return None
        T = int(ctx.abs_pos.max().item()) + 1
        q = _abs_set(self.params["queries"], row, T)
        k = _abs_set(self.params["keys"], row, T)
        kset = set(k)
        for p in q:
            if p in kset and all(c in kset for c in range(p + 1)):
                raise SteerConfigError("knockout %s leaves query %d of row %r with no allowed key" % (self.id, p, row.key))
        return torch.tensor(q, dtype=torch.long), torch.tensor(k, dtype=torch.long)

    def routes(self, site: Site) -> bool:
        """True when this knockout routes the call named by site (the registry's routed set, DESIGN 6.1)."""
        return self.op == "knockout" and (site.stage, site.loop, site.layer) in self._calls


def _own_reference(of: Intervention, owner_id: str) -> Intervention:
    """Shallow copy of a reference whose internal recorders (the extrapolate ref state, through any random /
    width chain) are renamed "<owner_id>|ref". The original reference keeps its own recorders."""
    new = copy.copy(of)
    if of.op in ("random", "width") and isinstance(of.params.get("of"), Intervention):
        inner = _own_reference(of.params["of"], owner_id)
        new.params = dict(of.params, of=inner)
        new.recorders = tuple(inner.recorders)
    else:
        new.recorders = tuple(dataclasses.replace(r, id="%s|ref" % owner_id) for r in of.recorders)
    return new


def _abs_set(pos: Positions, row: RowInfo, T: int) -> List[int]:
    if pos.kind == "abs":
        return sorted(set(p for p in pos.abs if 0 <= p < T))
    m = pos.resolve(row, torch.arange(T), torch.ones(T, dtype=torch.bool))
    return [int(p) for p in m.nonzero().flatten().tolist()]


def _x_dependent(iv: Intervention) -> bool:
    """True when the op's delta depends on h (every op but add; random / width follow their reference)."""
    if iv.op in ("random", "width") and isinstance(iv.params.get("of"), Intervention):
        return _x_dependent(iv.params["of"])
    return iv.op not in X_FREE_OPS


def _heads(iv: Intervention) -> Tuple[Optional[Sequence[int]], int]:
    """(head set, head_dim) of an intervention: its own, or for random / width the reference's, so a control
    built without from_spec (baselines.random_ctrl.random_direction) keeps the reference's heads."""
    if iv.params.get("heads") is not None:
        return iv.params["heads"], int(iv.params["head_dim"])
    of = iv.params.get("of")
    if iv.op in ("random", "width") and isinstance(of, Intervention):
        return _heads(of)
    return None, 0


def _head_mask(iv: "Intervention", site: Site, d: int, device: Any) -> Optional[Tensor]:
    """float32 [d] mask, 1 on the coordinates of the head set (`_heads`) at a HEAD_OUT site; None otherwise."""
    heads, hd = _heads(iv)
    if heads is None or site.component != Component.HEAD_OUT:
        return None
    m = torch.zeros(d, dtype=torch.float32, device=device)
    for h in heads:
        m[int(h) * hd:(int(h) + 1) * hd] = 1.0
    return m


def _restrict_heads(x: Tensor, y: Tensor, heads: Sequence[int], head_dim: int) -> Tensor:
    """Keep the edit on the listed heads of a flat HEAD_OUT tensor [n, H*hd] (DESIGN 6.1 hook 3)."""
    n, d = x.shape
    keep = torch.ones(d // head_dim, dtype=torch.bool, device=x.device)
    keep[torch.tensor(list(heads), dtype=torch.long, device=x.device)] = False
    out = y.clone().view(n, d // head_dim, head_dim)
    out[:, keep] = x.view(n, d // head_dim, head_dim)[:, keep]
    return out.view(n, d)


def _target(iv: Intervention, x: Tensor, site: Site, row: RowInfo, h_pos: Tensor) -> Tensor:
    """Patch / project target t: zero, a per-row payload (donor, clean, dev_item) or a shared mean."""
    kind = iv.params["target"]["kind"]
    if kind == "zero":
        return torch.zeros_like(x)
    if kind == "mean" and iv.id not in row.payload:
        m = _per_site(_req(iv.params, "mean", "mean target"), site, False, "params['mean']")
        m = m.to(device=x.device, dtype=torch.float32)
        if m.dim() == 1:
            if len(_static_positions(iv.positions, row) or ()) != 1:
                raise SteerConfigError("a [d] mean target needs exactly one edited position (no broadcast)")
            m = m[None]
        return _aligned(m, iv.positions, row, h_pos, "mean target")
    return _payload(iv, site, row, h_pos, x)


def _nbar_at(table: Any, entry_nbar: Any, site: Site, tie: bool, what: str) -> float:
    """n-bar of relnorm add at `site` (DESIGN 7.1 add: the mean ||h|| at the site; strength accounting nbar_s).
    A table is keyed by the virtual site key even under tie_loops, which ties only the vector. An nbar stored in
    a tied vector entry would be one value for every loop of the layer, so it is refused."""
    if table is not None:
        return float(_per_site(table, site, False, "nbar table"))
    if entry_nbar is not None and tie and _physical_key(site) != site.key():
        raise SteerConfigError("%s: under tie_loops the vector entry %s is shared across loops, so its nbar "
                               "cannot be nbar_s at %s; give a per-site 'nbar' table" % (what, _physical_key(site),
                                                                                         site.key()))
    if entry_nbar is None:
        raise SteerConfigError("%s needs nbar at %s" % (what, site.key()))
    return float(entry_nbar)


def _vector_delta(iv: Intervention, x: Tensor, site: Site, row: RowInfo, h_pos: Tensor) -> Tensor:
    """delta of add / normadd: a v (raw) or a nbar v / ||v|| (relnorm, k15_smoke2.caa_delta:165-166)."""
    vec = iv.params["vector"]
    tie = bool(iv.params.get("tie_loops", False))
    nbar = None
    if isinstance(vec, str):          # "para" / "payload": per-row aligned vectors (O2, audit P4)
        v = _payload(iv, site, row, h_pos, x)
    else:
        entry = _per_site(vec, site, tie, "params['vector']")
        v = entry["v"] if isinstance(entry, dict) else entry
        nbar = entry.get("nbar") if isinstance(entry, dict) else None
        v = v.to(device=x.device, dtype=torch.float32)
        if v.dim() != 1:
            raise SteerConfigError("a shared vector must be [d]; per-position vectors go in the payload")
    strength = iv.params["strength"]
    if strength["kind"] == "raw":
        return float(iv.a) * v
    nbar = _nbar_at(iv.params.get("nbar", strength.get("nbar")), nbar, site, tie, "relnorm %s %s" % (iv.op, iv.id))
    vn = v.norm(dim=-1, keepdim=True)
    if bool((vn == 0).any()):
        raise SteerConfigError("relnorm %s %s: zero vector at %s" % (iv.op, iv.id, site.key()))
    return float(iv.a) * float(nbar) * v / vn


def _edit_patch(iv, x, site, row, h_pos, ctx):
    return x if iv.a == 0 else op_patch(x, _target(iv, x, site, row, h_pos), iv.a)


def _edit_add(iv, x, site, row, h_pos, ctx):
    return x if iv.a == 0 else op_add(x, _vector_delta(iv, x, site, row, h_pos))


def _edit_normadd(iv, x, site, row, h_pos, ctx):
    return x if iv.a == 0 else op_normadd(x, _vector_delta(iv, x, site, row, h_pos))


def _edit_project(iv, x, site, row, h_pos, ctx):
    U = _per_site(iv.params["U"], site, False, "params['U']").to(device=x.device, dtype=torch.float32)
    return op_project(x, U, _target(iv, x, site, row, h_pos))


def _ref_delta(iv, x, site, row, h_pos, ctx) -> Tensor:
    """delta of the reference op of `random` / `width` at the same site and positions."""
    return iv.params["of"].edit(x, site, row, h_pos, ctx) - x


def _edit_width(iv, x, site, row, h_pos, ctx):
    U = _per_site(iv.params["U"], site, False, "params['U']").to(device=x.device, dtype=torch.float32)
    return op_width(x, _ref_delta(iv, x, site, row, h_pos, ctx), U)


def _edit_random(iv, x, site, row, h_pos, ctx):
    delta = _ref_delta(iv, x, site, row, h_pos, ctx)
    if not bool(delta.any()):
        return x
    u = torch.stack([unit_gaussian(row.key, site.key(), int(p), iv.params["seed"], x.shape[1])
                     for p in h_pos.tolist()]).to(x.device)
    m = _head_mask(iv, site, x.shape[1], x.device)
    if m is not None:   # [decision] the draw is kept on the reference's heads and renormalized, so ||delta|| holds
        u = u * m
        u = u / u.norm(dim=-1, keepdim=True)
    return op_random(x, delta, u)


def _edit_extrapolate(iv, x, site, row, h_pos, ctx):
    w = float(iv.params["w"])
    if w == 0:
        return x
    return op_extrapolate(x, _ref_state(iv, row, h_pos, ctx).to(device=x.device, dtype=torch.float32), w)


def _ref_state(iv: Intervention, row: RowInfo, h_pos: Tensor, ctx: ForwardCtx) -> Tensor:
    """h_ref of `extrapolate` at the edit's positions, from the internal state recorder's [B, T, d] tensor
    (hooks.HookRegistry._record_state stores to_cpu=False states at ("state", recorder id, site key))."""
    key = ("state", iv.recorders[0].id, iv.params["ref_site"].key())
    if key not in ctx.store:
        raise SteerConfigError("extrapolate %s: reference state %r not recorded in this forward (register "
                               "iv.recorders with the intervention)" % (iv.id, key))
    hits = [i for i, r in enumerate(ctx.rows) if r is row]
    if not hits:
        raise SteerConfigError("extrapolate %s: row %r is not in this forward" % (iv.id, row.key))
    if len(hits) > 1:   # edit() gets no batch index (hooks.HookRegistry.site), so the row must be unique
        raise SteerConfigError("extrapolate %s: one RowInfo object fills batch rows %s more than once; give each "
                               "batch row its own RowInfo" % (iv.id, hits))
    b = hits[0]
    hit = (ctx.abs_pos[b][None, :] == h_pos.to(ctx.abs_pos.device)[:, None]).nonzero()
    if hit.shape[0] != h_pos.numel():
        raise SteerConfigError("extrapolate %s: edited positions not found in the forward" % iv.id)
    # flat [n, d] like the x of edit(): a HEAD_OUT state is stored [B, T, H, hd] (the bindings' view)
    return ctx.store[key][b, hit[:, 1]].reshape(h_pos.numel(), -1)


def _edit_loreft(iv, x, site, row, h_pos, ctx):
    """LoReFT h <- h + dropout(R^T (W h + b - R h)) (ReFT 2404.03592 Eq.2; DESIGN 7.1, dropout on the edit term
    only). The equation is the module's forward (baselines.reft_editor.LoReFTSite, DESIGN 10.4): one module in
    params["module"], or a table params["modules"] keyed by site (physical key under tie_loops)."""
    m = iv.params.get("module")
    if m is None:
        m = _per_site(iv.params["modules"], site, bool(iv.params.get("tie_loops", False)), "params['modules']")
    return m(x)


def _edit_identity(iv, x, site, row, h_pos, ctx):
    return x


_EDITS = {
    "patch": _edit_patch, "add": _edit_add, "normadd": _edit_normadd, "project": _edit_project,
    "width": _edit_width, "random": _edit_random, "extrapolate": _edit_extrapolate, "loreft": _edit_loreft,
    "knockout": _edit_identity, "skip": _edit_identity,
}


# ------------------------------------------------------------------------------------------ from_spec
def _check_orthonormal(U: Tensor, what: str) -> None:
    """DESIGN 7.1 project: ||U^T U - I||_inf < 1e-5 in float32."""
    U = U.float()
    if U.dim() != 2:
        raise SteerConfigError("%s must be [d, r]" % what)
    err = (U.T @ U - torch.eye(U.shape[1], device=U.device, dtype=torch.float32)).abs().max().item()
    if not err < ORTHO_TOL:
        raise SteerConfigError("%s is not orthonormal: ||U^T U - I||_inf = %.3g" % (what, err))


def _rows(spec: Dict[str, Any]) -> Optional[FrozenSet[RowKey]]:
    return None if spec.get("rows") is None else frozenset(spec["rows"])


def _base_params(spec: Dict[str, Any], geom: Geometry) -> Dict[str, Any]:
    params = {"spec": spec, "schedule": spec.get("schedule", {"kind": "once"})}
    for k in ("label", "tie_loops", "heads"):
        if spec.get(k) is not None:
            params[k] = spec[k]
    if "heads" in params:
        params["head_dim"] = geom.head_dim
    return params


def _spec_sites(spec: Dict[str, Any], geom: Geometry, what: str) -> Tuple[Site, ...]:
    return expand(_req(spec, "site", what), spec.get("schedule", {"kind": "once"}), geom)


def _table_covers(table: Any, sites: Sequence[Site], tie: bool, what: str) -> None:
    if isinstance(table, (Tensor, str)):
        return
    for s in sites:
        _per_site(table, s, tie, what)


def _build_patch(spec, geom, iv_id, what):
    a = float(_req(spec, "a", what))
    target = _req(spec, "target", what)
    if _req(target, "kind", what + " target") not in TARGET_KINDS:
        raise SteerConfigError("%s: target kind must be one of %s" % (what, TARGET_KINDS))
    params = dict(_base_params(spec, geom), target=target)
    if spec.get("mean") is not None:
        params["mean"] = spec["mean"]
    return Intervention(iv_id, "patch", _spec_sites(spec, geom, what), _positions(_req(spec, "positions", what)),
                        a=a, params=params, rows=_rows(spec))


def _build_vector_op(spec, geom, iv_id, what):
    op = spec["op"]
    a = float(_req(spec, "a", what))
    strength = _req(spec, "strength", what)
    if _req(strength, "kind", what + " strength") not in STRENGTH_KINDS:
        raise SteerConfigError("%s: strength kind must be one of %s" % (what, STRENGTH_KINDS))
    vector = _req(spec, "vector", what)
    if isinstance(vector, str) and vector not in ("para", "payload"):
        raise SteerConfigError("%s: vector must be 'para', 'payload', a tensor or a per-site table" % what)
    sites = _spec_sites(spec, geom, what)
    params = dict(_base_params(spec, geom), strength=strength, vector=vector)
    tie = bool(spec.get("tie_loops", False))
    _table_covers(vector, sites, tie, what + " vector")
    if spec.get("nbar") is not None:
        params["nbar"] = spec["nbar"]
    return Intervention(iv_id, op, sites, _positions(_req(spec, "positions", what)), a=a, params=params,
                        rows=_rows(spec))


def _build_project(spec, geom, iv_id, what):
    U = _req(spec, "U", what)
    target = _req(spec, "target", what)
    if _req(target, "kind", what + " target") not in PROJECT_TARGETS:
        raise SteerConfigError("%s: project target kind must be one of %s" % (what, PROJECT_TARGETS))
    sites = _spec_sites(spec, geom, what)
    for s in sites:
        _check_orthonormal(_per_site(U, s, False, what + " U"), what + " U")
    params = dict(_base_params(spec, geom), U=U, target=target)
    if spec.get("mean") is not None:
        params["mean"] = spec["mean"]
    return Intervention(iv_id, "project", sites, _positions(_req(spec, "positions", what)), a=1.0, params=params,
                        rows=_rows(spec))


def _reference(spec, geom, what) -> Intervention:
    of = _req(spec, "of", what)
    ref = of if isinstance(of, Intervention) else from_spec(of, geom)
    if ref.op not in REFERENCE_OPS:
        raise SteerConfigError("%s: reference op %r changes no state" % (what, ref.op))
    if spec.get("site") is not None and [s.key() for s in _spec_sites(spec, geom, what)] != [s.key() for s in ref.sites]:
        raise SteerConfigError("%s: sites must equal the reference op's sites" % what)
    if spec.get("positions") is not None and _positions(spec["positions"]) != ref.positions:
        raise SteerConfigError("%s: positions must equal the reference op's positions" % what)
    return ref


def _inherit_heads(spec, ref: Intervention, params: Dict[str, Any], what: str) -> None:
    """A control keeps the reference's head set: DESIGN 7.1 random / width 'same sites ... as the reference',
    and a head set is part of a HEAD_OUT site (DESIGN 6.1 hook 3)."""
    rh = ref.params.get("heads")
    sh = spec.get("heads")
    if sh is not None and (rh is None or sorted(int(h) for h in sh) != sorted(int(h) for h in rh)):
        raise SteerConfigError("%s: heads %r must equal the reference op's heads %r" % (what, sh, rh))
    params.pop("heads", None)
    params.pop("head_dim", None)
    if rh is not None:
        params["heads"], params["head_dim"] = rh, ref.params["head_dim"]


def _build_width(spec, geom, iv_id, what):
    r = int(_req(spec, "r", what))
    U = _req(spec, "U", what)
    ref = _reference(spec, geom, what)
    params = dict(_base_params(spec, geom), U=U, r=r, of=ref, schedule=ref.params.get("schedule"))
    _inherit_heads(spec, ref, params, what)
    probe = Intervention(iv_id, "width", ref.sites, ref.positions, params=params)
    for s in ref.sites:
        Us = _per_site(U, s, False, what + " U")
        _check_orthonormal(Us, what + " U")
        if Us.shape[1] != r:
            raise SteerConfigError("%s: U has rank %d, r = %d" % (what, Us.shape[1], r))
        m = _head_mask(probe, s, Us.shape[0], Us.device)
        if m is not None and not float((Us.float() * (1 - m)[:, None]).abs().max()) < ORTHO_TOL:
            raise SteerConfigError("%s: U at %s leaves the reference's heads %r, so the rescaled projection "
                                   "would move other heads" % (what, s.key(), params["heads"]))
    return Intervention(iv_id, "width", ref.sites, ref.positions, a=ref.a, params=params,
                        rows=_rows(spec) if spec.get("rows") is not None else ref.rows)


def _build_random(spec, geom, iv_id, what):
    seed = int(_req(spec, "seed", what))
    ref = _reference(spec, geom, what)
    params = dict(_base_params(spec, geom), seed=seed, of=ref, schedule=ref.params.get("schedule"))
    _inherit_heads(spec, ref, params, what)
    return Intervention(iv_id, "random", ref.sites, ref.positions, a=ref.a, params=params,
                        rows=_rows(spec) if spec.get("rows") is not None else ref.rows)


def _build_extrapolate(spec, geom, iv_id, what):
    w = float(_req(spec, "w", what))
    sites = _spec_sites(spec, geom, what)
    refs = expand(_req(spec, "ref_site", what), {"kind": "once"}, geom)
    if len(refs) != 1:
        raise SteerConfigError("%s: ref_site must name one site" % what)
    ref = refs[0]
    for s in sites:
        if not geom.rank(ref) < geom.rank(s):
            raise SteerConfigError("%s: ref_site %s must execute before %s" % (what, ref.key(), s.key()))
        # [decision] Eq.2 of 2610.02185 extrapolates one state kind at two depths; a ref of another component
        # (e.g. resid_post for a head_out site, same width on Ouro) would subtract coordinates of two spaces
        if ref.component != s.component:
            raise SteerConfigError("%s: ref_site %s and site %s must have the same component (canonical %s vs "
                                   "%s)" % (what, ref.key(), s.key(), ref.component.value, s.component.value))
    positions = _positions(_req(spec, "positions", what))
    rec = Recorder(id="%s|ref" % iv_id, sites=(ref,), positions=positions, what="state", when="post", to_cpu=False)
    params = dict(_base_params(spec, geom), w=w, ref_site=ref)
    return Intervention(iv_id, "extrapolate", sites, positions, a=w, params=params, rows=_rows(spec),
                        recorders=(rec,))


def _build_knockout(spec, geom, iv_id, what):
    queries = _positions(_req(spec, "queries", what))
    keys = _positions(_req(spec, "keys", what))
    calls = _call_sites(_req(spec, "calls", what), spec.get("schedule"), geom, call_component(geom))
    params = dict(_base_params(spec, geom), queries=queries, keys=keys)
    if spec.get("condition") is not None:
        params["condition"] = spec["condition"]
    return Intervention(iv_id, "knockout", calls, queries, a=1.0, params=params, rows=_rows(spec))


def _build_skip(spec, geom, iv_id, what):
    if spec.get("rows") is not None:
        raise SteerConfigError("%s: skip acts on the whole batch and cannot be scoped to a row (DESIGN 13.4, sz "
                               "rows); drop 'rows'" % what)
    if spec.get("calls") is not None:
        calls = spec["calls"]
    else:
        loops, layers = _req(spec, "loops", what), _req(spec, "layers", what)
        calls = ["loop%d.L%d" % (int(j), int(l)) for j in loops for l in layers]
    sites = _call_sites(calls, spec.get("schedule"), geom, SKIP_COMPONENT)
    return Intervention(iv_id, "skip", sites, Positions(kind="all"), a=1.0, params=_base_params(spec, geom),
                        rows=_rows(spec))


def _build_kv_patch(spec, geom, iv_id, what):
    """kv_patch = patch at K_PROJ(j, l) and V_PROJ(j, l) of each call (DESIGN 7.1; recon_mathcode M1, M2)."""
    if geom.family == "ifm":
        raise UnsupportedSite("kv_patch: IFM has no K/V components in v1 (DESIGN 4.1)")
    a = float(_req(spec, "a", what))
    calls = _call_sites(_req(spec, "calls", what), spec.get("schedule"), geom, Component.K_PROJ)
    sites = []
    for c in calls:
        for comp in (Component.K_PROJ, Component.V_PROJ):
            sites.append(canonical(geom, Site(stage=c.stage, loop=c.loop, layer=c.layer, component=comp)))
    params = dict(_base_params(spec, geom), target={"kind": "donor"}, expanded_from="kv_patch")
    return Intervention(iv_id, "patch", tuple(sites), _positions(_req(spec, "positions", what)), a=a,
                        params=params, rows=_rows(spec))


def _build_loreft(spec, geom, iv_id, what):
    sites = _spec_sites(spec, geom, what)
    tie = bool(spec.get("tie_loops", False))
    if spec.get("module") is not None:
        params = dict(_base_params(spec, geom), module=spec["module"])
        mods = [spec["module"]]
    else:
        modules = _req(spec, "modules", what)
        params = dict(_base_params(spec, geom), modules=modules)
        mods = [_per_site(modules, s, tie, what + " modules") for s in sites]
    if not all(callable(m) for m in mods):
        raise SteerConfigError("%s: a loreft module must be callable (reft_editor.LoReFTSite)" % what)
    return Intervention(iv_id, "loreft", sites, _positions(_req(spec, "positions", what)), a=1.0, params=params,
                        rows=_rows(spec), trainable=True)


_BUILDERS = {
    "patch": _build_patch, "add": _build_vector_op, "normadd": _build_vector_op, "project": _build_project,
    "width": _build_width, "random": _build_random, "extrapolate": _build_extrapolate,
    "knockout": _build_knockout, "skip": _build_skip, "kv_patch": _build_kv_patch, "loreft": _build_loreft,
}


def from_spec(spec: Dict[str, Any], geom: Geometry) -> Intervention:
    """Build an Intervention from a spec (DESIGN 13.1 form: id, op, site, schedule, positions, a, target, ...).
    REQUIRED fields of each op (DESIGN 7.1 'defaults' column) raise SteerConfigError when missing."""
    what = "intervention spec %r" % (spec.get("id") if isinstance(spec, dict) else spec)
    iv_id = str(_req(spec, "id", what))
    op = _req(spec, "op", what)
    if op not in _BUILDERS:
        raise SteerConfigError("%s: unknown op %r" % (what, op))
    iv = _BUILDERS[op](spec, geom, iv_id, what)
    if spec.get("heads") is not None:
        _check_heads(spec["heads"], iv, geom, what)
    return iv


def _check_heads(heads: Any, iv: Intervention, geom: Geometry, what: str) -> None:
    """The head-set rule of the module contract: HEAD_OUT sites only, distinct heads in 0..H-1, and only ops
    whose equation holds on the heads' coordinates (DESIGN 6.1 hook 3; DESIGN 0 rule 3: config errors raise)."""
    if isinstance(heads, (str, bytes)) or not isinstance(heads, Sequence) or not heads:
        raise SteerConfigError("%s: heads must be a non-empty list of head indices, got %r" % (what, heads))
    hs = [int(h) for h in heads]
    if len(set(hs)) != len(hs):
        raise SteerConfigError("%s: heads %r repeat a head" % (what, list(heads)))
    if any(not 0 <= h < geom.n_heads for h in hs):
        raise SteerConfigError("%s: heads %r outside 0..%d" % (what, list(heads), geom.n_heads - 1))
    off = [s.key() for s in iv.sites if s.component != Component.HEAD_OUT]
    if off:
        raise SteerConfigError("%s: heads are defined only at head_out sites, not at %s" % (what, off))
    relnorm = iv.op == "add" and (iv.params.get("strength") or {}).get("kind") == "relnorm"
    if iv.op in ("normadd", "loreft") or relnorm:
        raise SteerConfigError("%s: op %s%s takes its norm over the full head_out vector, so a head set would "
                               "break its equation; edit the full vector or use patch / raw add"
                               % (what, iv.op, " relnorm" if relnorm else ""))
    if iv.op == "project":
        for s in iv.sites:
            Us = _per_site(iv.params["U"], s, False, what + " U").float()
            m = _head_mask(iv, s, Us.shape[0], Us.device)
            if not float((Us * (1 - m)[:, None]).abs().max()) < ORTHO_TOL:
                raise SteerConfigError("%s: U at %s leaves the heads %r, so the projection would move other "
                                       "heads" % (what, s.key(), list(heads)))


def first_changed(iv: Intervention, geom: Geometry) -> Optional[int]:
    """F of an edit (DESIGN 4.5): the smallest first changed executed index over its sites; None when no site
    changes any K/V (Ouro LOOP_END(k)) and for knockout and skip, which are described by their calls."""
    if iv.op in ("knockout", "skip"):
        return None
    fs = [geom.first_changed(s) for s in iv.sites]
    fs = [int(f) for f in fs if f is not None]
    return min(fs) if fs else None


def describe(iv: Intervention, geom: Geometry) -> Dict[str, Any]:
    """JSON fields of an intervention for records (DESIGN 13.3 forward row: op, site, positions, a, schedule,
    first_changed_exec_index). `site` is the site key, or the list of keys for a multi-site intervention;
    `sites` always lists them."""
    p = iv.params
    keys = [s.key() for s in iv.sites]
    return {
        "id": iv.id, "op": iv.op, "site": keys[0] if len(keys) == 1 else keys, "sites": keys,
        "positions": _positions_dict(iv.positions),
        "a": iv.a, "schedule": p.get("schedule"), "target": (p.get("target") or {}).get("kind"),
        "strength_kind": (p.get("strength") or {}).get("kind"), "label": p.get("label"),
        "expanded_from": p.get("expanded_from"), "condition": p.get("condition"),
        "of": p["of"].id if "of" in p else None, "seed": p.get("seed"),
        "rows": None if iv.rows is None else sorted(str(r) for r in iv.rows),
        "heads": None if _heads(iv)[0] is None else [int(h) for h in _heads(iv)[0]],
        "first_changed_exec_index": first_changed(iv, geom),
    }


def payload_needs(iv: Intervention) -> List[Tuple[str, str, Tuple[Site, ...], Positions]]:
    """Per-row payloads an intervention reads, as (payload id, kind, sites, positions): kind is the target kind
    (donor, clean, dev_item; mean only when no shared mean is given) or "vector" (add / normadd with a 'para' or
    'payload' vector). `random` and `width` read their reference's payload under the reference's id, so they
    list the reference's needs."""
    p = iv.params
    if iv.op in ("random", "width") and isinstance(p.get("of"), Intervention):
        return payload_needs(p["of"])
    if iv.op in ("patch", "project"):
        kind = (p.get("target") or {}).get("kind")
        if kind in ("donor", "clean", "dev_item") or (kind == "mean" and p.get("mean") is None):
            return [(iv.id, kind, tuple(iv.sites), iv.positions)]
    if iv.op in ("add", "normadd") and isinstance(p.get("vector"), str):
        return [(iv.id, "vector", tuple(iv.sites), iv.positions)]
    return []


# ------------------------------------------------------------------------------------------ strength matching
class UnitNorm(dict):
    """Per-site ||delta|| per position per unit strength (DESIGN 7.1 DR9). `op` names the op it was computed
    for; `additive` is True when the absolute total of a schedule is exactly a * sum of unit norms (add)."""

    def __init__(self, values: Dict[str, float], op: str, additive: bool) -> None:
        super().__init__(values)
        self.op = op
        self.additive = additive


def unit_norm(spec: Dict[str, Any], geom: Geometry, build: Dict[str, Any]) -> Optional[UnitNorm]:
    """Unit norm per site (DESIGN 7.1 "Unit norm"): relnorm add -> nbar_s; raw add (CAA native m) -> ||v_s||;
    patch -> mean over build items and edited positions of ||t_s - h_s,clean||; random and width -> the unit
    norm of their reference op; every other op -> None. A raw add with a per-row payload vector has an
    item-dependent norm, so its unit norm is None; relnorm add has nbar_s whatever v is. With a head set
    (module contract) the norms are taken over the heads' coordinates [derived: the edit changes only those].
    `build` holds {"vector": per-site table} and {"patch": {site_key: {"target": [Tensor [n, d]], "clean":
    [Tensor [n, d]]}}} as needed."""
    op = spec["op"]
    if op in ("random", "width"):
        of = _req(spec, "of", "unit_norm %s" % op)
        if isinstance(of, Intervention):
            of = _req(of.params, "spec", "unit_norm: reference intervention %s" % of.id)
        return unit_norm(of, geom, build)
    if op not in ("add", "patch"):
        return None
    sites = expand(spec["site"], spec.get("schedule", {"kind": "once"}), geom)
    if op == "add":
        return _unit_norm_add(spec, sites, build, geom)
    table = _req(build, "patch", "unit_norm build for patch")
    vals = {}
    for s in sites:
        rec = _req(table, s.key(), "unit_norm build patch")
        diffs = [_masked(t.float() - h.float(), spec, s, geom).norm(dim=-1) for t, h in zip(rec["target"], rec["clean"])]
        vals[s.key()] = float(torch.cat(diffs).mean())
    return UnitNorm(vals, "patch", additive=False)


def _masked(t: Tensor, spec: Dict[str, Any], site: Site, geom: Geometry) -> Tensor:
    """t ([..., d] or HEAD_OUT [n, H, hd]) flattened to [..., d] and kept on the spec's head set at HEAD_OUT."""
    if site.component == Component.HEAD_OUT and t.dim() == 3:
        t = t.reshape(t.shape[0], -1)
    heads = spec.get("heads")
    if heads is None or site.component != Component.HEAD_OUT:
        return t
    m = torch.zeros(t.shape[-1], dtype=t.dtype, device=t.device)
    for h in heads:
        m[int(h) * geom.head_dim:(int(h) + 1) * geom.head_dim] = 1
    return t * m


def _unit_norm_add(spec: Dict[str, Any], sites: Sequence[Site], build: Dict[str, Any],
                   geom: Geometry) -> Optional[UnitNorm]:
    vec = spec.get("vector") if spec.get("vector") is not None else build.get("vector")
    relnorm = spec["strength"]["kind"] == "relnorm"
    if vec is None or (isinstance(vec, str) and not relnorm):
        return None
    tie = bool(spec.get("tie_loops", False))
    table = spec.get("nbar", spec["strength"].get("nbar"))
    vals = {}
    for s in sites:
        if isinstance(vec, str):          # relnorm with a payload vector: DESIGN 7.1 "n-bar_s (exact per position)"
            vals[s.key()] = _nbar_at(table, None, s, tie, "unit_norm relnorm")
            continue
        entry = _per_site(vec, s, tie, "unit_norm vector")
        if relnorm:
            nbar = entry.get("nbar") if isinstance(entry, dict) else None
            vals[s.key()] = _nbar_at(table, nbar, s, tie, "unit_norm relnorm")
        else:
            v = entry["v"] if isinstance(entry, dict) else entry
            vals[s.key()] = float(_masked(v.float(), spec, s, geom).norm())
    return UnitNorm(vals, "add", additive=True)


def _additive(unit: Dict[str, float], op: Optional[str], what: str) -> bool:
    """Whether a * sum(unit) is the exact absolute total: a UnitNorm says so itself; a plain dict (a JSON
    round trip, or caa.unit_norms) needs the op it was computed for, and only `add` is additive (DESIGN 7.1)."""
    if isinstance(unit, UnitNorm):
        return bool(unit.additive)
    if op is None:
        raise SteerConfigError("%s: a plain dict unit norm needs op= (only add is additive; for other ops "
                               "ref_total is REQUIRED); pass a UnitNorm or the op" % what)
    return op == "add"


def abs_matched_a(a: float, sites_ref: Sequence[Union[Site, str]], site: Union[Site, str],
                  unit: Optional[Dict[str, float]], ref_total: Optional[float] = None,
                  op: Optional[str] = None) -> float:
    """Single-site strength with the reference arm's absolute total: ref_total / unit[site] (audit P1, DR9).
    For add, ref_total defaults to a * sum_{s in sites_ref} unit[s] [derived from delta = a v]. For every other
    op ref_total is REQUIRED (the reference arm's measured mean per-position delta_norm_total), because later
    calls of a multi-site patch act on an already moved h [derived]. The op is read from a UnitNorm, or from
    `op` for a plain dict; a plain dict without `op` and without ref_total raises."""
    if unit is None:
        raise SteerConfigError("abs_matched_a: this op has no unit norm (DESIGN 7.1)")
    u = unit.get(_key(site))
    if u is None or not u > 0:
        raise SteerConfigError("abs_matched_a: no positive unit norm at %s" % _key(site))
    if ref_total is None:
        if not _additive(unit, op, "abs_matched_a"):
            raise SteerConfigError("abs_matched_a: ref_total is REQUIRED for op %r" % getattr(unit, "op", op))
        ref_total = float(a) * sum(unit[_key(s)] for s in sites_ref)
    return float(ref_total) / u


def abs_matched_arms(ref_spec: Dict[str, Any], unit: Optional[Dict[str, float]], geom: Geometry,
                     ref_total: Optional[float] = None) -> List[Dict[str, Any]]:
    """One single-loop arm per loop of the reference schedule at the absolutely matched strength (DR9). A loop
    holding several reference sites gets a = ref_total / sum of their unit norms [derived: equal absolute total
    per position]. `ref_total` follows abs_matched_a; a plain dict unit is additive only when the reference op
    is add. Arms are grouped and labelled by the loop the schedule placed each site at, not by the loop of its
    canonical site (raven INJ_STATE(j) is canonical at RESID_POST(j-1, last), INJ_STATE(1) at input.init_state).
    For `random` and `width` (unit norm = their reference's, DESIGN 7.1) the arms are the reference's arms,
    each wrapped as the same control with `of` = that arm, id `<control id>|abs_loop<j>`."""
    if ref_spec.get("op") in ("random", "width"):
        of = _req(ref_spec, "of", "abs_matched_arms %s" % ref_spec.get("op"))
        if isinstance(of, Intervention):
            of = _req(of.params, "spec", "abs_matched_arms: reference intervention %s" % of.id)
        out = []
        for arm in abs_matched_arms(of, unit, geom, ref_total):
            j = arm["matching"]["loop"]
            ctrl = {k: v for k, v in ref_spec.items() if k not in ("site", "schedule", "positions", "heads")}
            ctrl.update(id="%s|abs_loop%s" % (ref_spec["id"], j), of=arm,
                        matching=dict(arm["matching"], ref=ref_spec["id"]))
            out.append(ctrl)
        return out
    schedule = ref_spec.get("schedule", {"kind": "once"})
    sites = expand(ref_spec["site"], schedule, geom)
    placed = _expand_sites(ref_spec["site"], {x: y for x, y in schedule.items() if x != "calls_from"}, geom,
                           canonicalize=False)
    loop_of = {}  # type: Dict[str, Optional[int]]
    for s in placed:
        loop_of.setdefault(canonical(geom, s).key(), s.loop)
    if unit is None:
        raise SteerConfigError("abs_matched_arms: this op has no unit norm (DESIGN 7.1)")
    if ref_total is None:
        if not _additive(unit, ref_spec.get("op"), "abs_matched_arms"):
            raise SteerConfigError("abs_matched_arms: ref_total is REQUIRED for op %r"
                                   % getattr(unit, "op", ref_spec.get("op")))
        ref_total = float(ref_spec["a"]) * sum(unit[s.key()] for s in sites)
    groups = {}  # type: Dict[Optional[int], List[Site]]
    for s in sites:
        groups.setdefault(loop_of[s.key()], []).append(s)
    arms = []
    for j, group in groups.items():
        denom = sum(unit[s.key()] for s in group)
        if not denom > 0:
            raise SteerConfigError("abs_matched_arms: zero unit norm in loop %s" % j)
        arm = dict(ref_spec)
        arm.update(id="%s|abs_loop%s" % (ref_spec["id"], j), site=[s.key() for s in group], schedule={"kind": "once"},
                   a=float(ref_total) / denom,
                   matching={"kind": "absolute", "ref": ref_spec["id"], "ref_total": float(ref_total), "loop": j})
        arms.append(arm)
    return arms


# ------------------------------------------------------------------------------------------ Y2 grid
def y2_grid(geom: Geometry) -> List[Dict[str, Any]]:
    """Input-channel protocol grid (audit P13; k16_path_ifm.item_y2:427-455, names at 7-8): a(j) =
    single_occurrence(j), b(j) = persistent_from(j), pre(j) = INJ_STATE once at j, post(j) = INJ_POST once at
    j, for every j in 1..K (K-1 included), plus a(1)+pre(1). The plan adds id and a, and may drop a j only by
    listing it.
    Where pre(1) is the INIT_STATE edit and x0 does not depend on the prompt (raven: zeros or table_r seeded by
    the row key, adapters/raven.py x0_table / _x0, and the runner records the donor under clean A's key; IFM
    with huginn_state_init "normal": a hashed normal state with example id 0 and seed 42, adapters/ifm.py
    _shared_init_state, which "uses only _hashed_normal_state and model_dim" per the comment at loop-flow
    k10_ifm.py:226-228 [belief: xllm's identity_recurrent_state not read]), the donor x0 equals
    clean A's, so pre(1) is an exact no-op and a(1)+pre(1) equals a(1). Both rows are kept, labelled
    "structural null: x0 is prompt-independent", with `structural_null` set and n_calls 0, so a consumer skips
    them [decision]. Other IFM state_init values are not flagged [belief: "input" mixes the prompt into x0]."""
    if not geom.injection:
        raise UnsupportedSite("y2_grid needs an injection model (family %s)" % geom.family)
    out = []
    for j in range(1, geom.k + 1):
        out.append(_y2("a", j, "loop%d.inj_input" % j, {"kind": "once"}, geom, "single_occurrence(%d)" % j,
                       "argument only" if j == 1 else None))
        out.append(_y2("b", j, "loop*.inj_input", {"kind": "from_loop", "j": j}, geom, "persistent_from(%d)" % j, None))
        out.append(_y2("pre", j, "loop%d.inj_state" % j, {"kind": "once"}, geom, None, None))
        out.append(_y2("post", j, "loop%d.inj_post" % j, {"kind": "once"}, geom, None, None))
    out.append(_y2("a+pre", 1, ["loop1.inj_input", "loop1.inj_state"], {"kind": "once"}, geom, None,
                   "argument+state"))
    return out


def _y2(protocol: str, j: int, site: Any, schedule: Dict[str, Any], geom: Geometry, input_protocol: Optional[str],
        label: Optional[str]) -> Dict[str, Any]:
    canon = [s.key() for s in expand(site, schedule, geom)]
    if protocol == "pre" and canon == ["input.init_state"]:
        label = "init_state"      # DESIGN 7.2: where loop1.inj_state is canonical at INIT_STATE, pre(1) edits x0
    null = None
    if protocol in ("pre", "a+pre") and j == 1 and "input.init_state" in canon and _x0_prompt_independent(geom):
        null = "pre(1) edits x0, which does not depend on the prompt: the donor x0 equals clean A's"
        label = "structural null: x0 is prompt-independent"
    name = "a(1)+pre(1)" if protocol == "a+pre" else "%s(%d)" % (protocol, j)
    return {"name": name, "protocol": protocol, "j": j, "op": "patch", "site": site, "schedule": schedule,
            "positions": {"kind": "span", "span": "source"}, "target": {"kind": "donor"},
            "input_protocol": input_protocol, "label": label, "canonical_sites": canon, "structural_null": null,
            "n_calls": 0 if null else (geom.k - j + 1 if protocol == "b" else 1)}


def _x0_prompt_independent(geom: Geometry) -> bool:
    """True where x0 carries no prompt information: raven (adapters/raven.py _x0: zeros or the row-key table)
    and IFM with huginn_state_init "normal" (adapters/ifm.py _shared_init_state; see y2_grid)."""
    return geom.family in ("huginn", "mcleish") or (geom.family == "ifm" and geom.state_init == "normal")


# ------------------------------------------------------------------------------------------ knockout set
def ko_conditions(item: Any, geom: Geometry, F: int, item_index: Optional[int] = None,
                  seed_rule: str = "sha256", read_idx: int = 0) -> Dict[str, Optional[Dict[str, Any]]]:
    """Knockout condition set of DESIGN 7.3 (port of k16_path_ouro.ko_conditions:88-106, plus relay K3, the
    sink-keeping variants of path audit F3, per-loop cuts and relay_rand0..2 of MR8). `calls_after` = every
    attention call with executed index >= F. Each value is a from_spec-ready knockout spec without `id`, or None
    ('none', a rand draw without enough candidates, and relay when it equals src_all). Key `_meta` holds
    src_has_pos0, relay_equals_src_all and the seed rule. The read is `item.reads[read_idx]` (DESIGN 12, PF7).
    Two [decision]s beyond the DESIGN 7.3 table (7.3 text to be updated by its owner): src_prelude and src_coda
    hold the prelude and coda calls of calls_after, which the core-layer bands src_lo / src_hi leave out
    (`_meta.lo_hi_omit`); relay_rand keys are drawn only from positions before max(src), so no relay query
    loses its own key, and a draw with fewer candidates than |src| is None as for rand*."""
    if F is None:
        raise SteerConfigError("ko_conditions: the edit changes no K/V (F is None)")
    if seed_rule not in ("sha256", "k16_index"):
        raise SteerConfigError("ko_conditions: unknown seed_rule %r" % seed_rule)
    if seed_rule == "k16_index" and item_index is None:
        raise SteerConfigError("ko_conditions: seed_rule 'k16_index' needs item_index")
    key, src, read = str(_item(item, "key")), _source(item), _read_pos(item, read_idx)
    if not src or max(src) >= read:
        raise SteerConfigError("ko_conditions: the source span must lie before the read position")
    calls = attention_calls(geom, int(F))
    if not calls:
        raise SteerConfigError("ko_conditions: no attention call at or after F = %d" % F)
    q_read = {"kind": "abs", "abs": [read]}
    q_relay = {"kind": "after_span", "span": "source"}

    def cond(name: str, q: Dict[str, Any], keys: Sequence[int], cs: Sequence[Site]) -> Dict[str, Any]:
        return {"op": "knockout", "condition": name, "queries": q, "keys": {"kind": "abs", "abs": list(keys)},
                "calls": [c.key() for c in cs]}

    # src_l2 = the first loop of calls_after (DESIGN 7.3; k16_path_ouro.ko_conditions:91 c_l2 is loop 2), so a
    # prelude F gives loop 1, not the remaining prelude calls
    loops_after = [c.loop for c in calls if c.stage == "loop"]
    first = [c for c in calls if c.stage == "loop" and c.loop == min(loops_after)] if loops_after else []
    half = geom.n_core // 2                                        # k16: L // 2
    lo = [c for c in calls if c.stage == "loop" and c.layer < half]
    hi = [c for c in calls if c.stage == "loop" and c.layer >= half]
    # [decision] DESIGN 7.3 bands are core layers; prelude and coda calls of calls_after are their own conditions,
    # so src_prelude + src_lo + src_hi + src_coda = src_all (the k16 Ouro pilot has neither stage)
    pre = [c for c in calls if c.stage == "prelude"]
    coda = [c for c in calls if c.stage == "coda"]
    out = {"none": None, "src_all": cond("src_all", q_read, src, calls), "src_l2": cond("src_l2", q_read, src, first) if first else None,
           "src_lo": cond("src_lo", q_read, src, lo) if lo else None,
           "src_hi": cond("src_hi", q_read, src, hi) if hi else None,
           "src_prelude": cond("src_prelude", q_read, src, pre) if pre else None,
           "src_coda": cond("src_coda", q_read, src, coda) if coda else None}  # type: Dict[str, Any]
    cand = [p for p in range(1, read) if p not in src]             # k16: range(1, last), not in src
    for s in range(3):
        seed = _seed_int("%s|%d" % (key, s)) if seed_rule == "sha256" else 1000 * s + int(item_index)
        out["rand%d" % s] = cond("rand%d" % s, q_read, sorted(random.Random(seed).sample(cand, len(src))), calls) \
            if len(cand) >= len(src) else None
    # [decision] relay_rand keys exclude the relay queries (positions > max(src)): a knocked key that is also a
    # query would remove that query's own key, which relay never does (every relay query keeps itself, 7.3)
    relay_cand = [p for p in cand if p < max(src)]
    for s in range(3):
        seed = _seed_int("%s|relay|%d" % (key, s))
        out["relay_rand%d" % s] = cond("relay_rand%d" % s, q_relay,
                                       sorted(random.Random(seed).sample(relay_cand, len(src))),
                                       calls) if len(relay_cand) >= len(src) else None
    out["nonsrc"] = cond("nonsrc", q_read, [p for p in range(0, read) if p not in src], calls)
    out["allbut"] = cond("allbut", q_read, list(range(0, read)), calls)
    out["nonsrc0"] = cond("nonsrc0", q_read, [p for p in range(1, read) if p not in src], calls)
    out["allbut0"] = cond("allbut0", q_read, list(range(1, read)), calls)
    relay_equal = max(src) + 1 == read
    out["relay"] = None if relay_equal else cond("relay", q_relay, src, calls)
    all_calls = attention_calls(geom)
    for j in sorted(set(c.loop for c in calls if c.stage == "loop")):
        loop_calls = [c for c in all_calls if c.stage == "loop" and c.loop == j]
        out["src_loop(%d)" % j] = cond("src_loop(%d)" % j, q_read, src, loop_calls)
        out["relay_loop(%d)" % j] = None if relay_equal else cond("relay_loop(%d)" % j, q_relay, src, loop_calls)
    out["_meta"] = {"item": key, "read": read, "source": src, "F": int(F), "src_has_pos0": 0 in src,
                    "relay_equals_src_all": relay_equal, "seed_rule": seed_rule,
                    "relay_rand_seed_rule": "sha256", "relay_rand_pool": "1..max(src)-1, not in src",
                    "relay_rand_n_candidates": len(relay_cand),
                    "band_partition": "src_prelude + src_lo + src_hi + src_coda = src_all",
                    "lo_hi_omit": [c.key() for c in pre + coda]}
    return out


# ------------------------------------------------------------------------------------------ undo restore sites
def _restore_candidates(geom: Geometry, F: int) -> List[Tuple[Site, int]]:
    """Canonical restore sites (LOOP_END, RESID_POST / IFM `_layer` output) with e(r) >= F, in executed order."""
    out = []
    for s in geom.all_sites(list(RESTORE_COMPONENTS)):     # canonical, each tensor once, rank order
        e = geom.executed_index(s)
        if e >= F:
            out.append((s, e))
    return out


def restore_sites(window: Dict[str, Dict[str, Any]], geom: Geometry, datasets: Sequence[str],
                  edit_site: Site) -> Dict[str, Any]:
    """Undo restore sites from the measured single-call read window (DESIGN 7.3 K4, DR3, DR4, DR16; v1.5 MR7).
    Secondary measure: production restore sites come from slot_rule (DESIGN 22.7).

    window[d] = {"read_window": [executed index passing under own or final scoring], "final_window": [passing
    under final scoring], "suffix": {s0: bool, relay suffix knockout from s0 passes window_test}, "dev_keys",
    "sha256"}. Per dataset: r_in (every candidate with first_read <= e < last_read whose suffix from e + 1 was
    measured and passes), r_pre (the latest candidate with F <= e < first_read) and r_post (the earliest with
    e >= last_read) [decision: the controls closest to the window are the most stringent]."""
    if geom.injection:
        raise SteerConfigError("restore_sites: undo is not run on injection designs (DESIGN DR4, PLAN A5)")
    F = geom.first_changed(edit_site)
    if F is None:
        raise SteerConfigError("restore_sites: edit %s changes no K/V, so no later read exists" % edit_site.key())
    cands = _restore_candidates(geom, int(F))
    out = {"F": int(F), "edit_site": edit_site.key(), "datasets": {}}  # type: Dict[str, Any]
    for d in datasets:
        if d not in window:
            raise SteerConfigError("restore_sites: no window for dataset %r" % d)
        out["datasets"][d] = _restore_one(window[d], cands, int(F))
    return out


def _restore_one(w: Dict[str, Any], cands: List[Tuple[Site, int]], F: int) -> Dict[str, Any]:
    rw = sorted(int(c) for c in w["read_window"] if int(c) >= F)
    fw = sorted(int(c) for c in w["final_window"] if int(c) >= F)
    suffix = {int(k): bool(v) for k, v in w["suffix"].items()}
    res = {"F": F, "n_candidates": len(cands), "window_sha256": w.get("sha256"), "dev_keys": w.get("dev_keys"),
           "first_read": None, "last_read": None, "r_in": [], "r_pre": None, "r_post": None,
           "r_pre_label": None, "site_labels": {}, "label": "no in-window slot", "reason": None}  # type: Dict[str, Any]
    if not fw:
        res["reason"] = "no_read_after_F"
        return res
    if not rw or not set(fw) <= set(int(c) for c in w["read_window"]):
        raise SteerConfigError("restore_sites: the final window must lie inside the read window")
    first_read, last_read = rw[0], fw[-1]

    def entry(c: Site, e: int) -> Dict[str, Any]:
        return {"site": c.key(), "e": e, "s0": e + 1}

    r_in = [entry(c, e) for c, e in cands if first_read <= e < last_read and suffix.get(e + 1) is True]
    pre = [entry(c, e) for c, e in cands if e < first_read]
    post = [entry(c, e) for c, e in cands if e >= last_read]
    res.update(first_read=first_read, last_read=last_read, r_in=r_in, r_pre=pre[-1] if pre else None,
               r_post=post[0] if post else None, r_pre_label=None if pre else "no_pre_window_restore",
               site_labels={c.key(): ("before" if e < first_read else "in" if e < last_read else "after")
                            for c, e in cands})
    if r_in:
        res["label"] = "in-window"
    else:
        res["reason"] = "no_candidate"
    return res


# ------------------------------------------------------------------------------------------ X6 suffix grid
def suffix_grid(geom: Geometry, extra: Sequence[Union[Site, str]] = (),
                mapped_from: Optional[Geometry] = None) -> List[Optional[Site]]:
    """Boundary grid of matrix X6: None = `start` (the full knockout from the first call), then after L0@1,
    Lmid@1, Llast@1, after Lmid@j and Llast@j for j = 2..K-1, after Lmid@K (matrix_site names; d28 and d112
    through `mapped_from`, whose K is the paired looped model's). `extra` adds plan points (DR16). Ouro K = 4
    gives 9 boundaries. Boundaries are kept in executed order and deduplicated."""
    K = (mapped_from or geom).k
    names = ["L0@1", "Lmid@1", "Llast@1"]
    for j in range(2, K):
        names += ["Lmid@%d" % j, "Llast@%d" % j]
    names.append("Lmid@%d" % K)
    sites = [matrix_site(geom, n, mapped_from=mapped_from) for n in names]
    sites += [canonical(geom, Site.parse(e) if isinstance(e, str) else e) for e in extra]
    seen, uniq = set(), []
    for s in sites:
        if s.key() not in seen:
            seen.add(s.key())
            uniq.append(s)
    uniq.sort(key=geom.rank)
    return [None] + uniq


def suffix_calls(geom: Geometry, after: Optional[Site]) -> List[Site]:
    """Attention calls after boundary `after` (DESIGN 22.7: the calls ranked after c by the rank of 4.2), or every
    call for the `start` boundary (None). A call ranks after c exactly when its K/V can see c, so the suffix
    starts at F(c) of DESIGN 4.5 (sites.Geometry.first_changed: e for components the own call reads, e + 1 for
    the others, n_prelude for a canonical PRELUDE_OUT) and is empty when F is None. first_changed canonicalizes,
    so an alias and its canonical site give the same suffix (DESIGN 4.1)."""
    if after is None:
        return attention_calls(geom)
    F = geom.first_changed(after)
    return attention_calls(geom, geom.total() if F is None else int(F))


def _boundary_key(b: Optional[Site]) -> str:
    return "start" if b is None else "after:" + b.key()


def slot_rule(curves: Dict[str, Any], candidates: Sequence[Site], geom: Geometry, edit_site: Site,
              thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Matrix X6 primary slot rule (DESIGN 22.7, matrix X6 endpoint). Port of k17_varbench.window_rule_v4 (the v4
    rule: necessity clause a, e_carried b2 >= 0.2 over >= 10 items; loop-flow/src/k17_varbench.py:431-436, read
    2026-10-08 at sha256 4377ede2) plus the DESIGN 22.7 band clause of (a) and the r_pre / r_post / last_read / W
    rules. The source keeps v4 only as a reported diagnostic since plan critic 3 F1; its primary is the v5
    influence rule (k17_varbench.window_rule:414-428). Which rule X6 uses is an open DESIGN 22.7 / matrix X6
    decision [belief: v4 kept here only because DESIGN 22.7 and matrix X6 still specify it].
    On an injection model (matrix X6 lists M9b; X3 undo is not run there, PLAN A5) rule (a), last_read and W
    are computed as usual and r_in / r_pre / r_post are None, labelled "no undo on injection (PLAN A5)", as
    k17_varbench.select_slots returns the dep curve without slots for NO_UNDO models (k17_varbench.py:2657-2664).

    curves (dev items, clean A, final read-out): "dep_rel" {boundary key: mean dep_rel}, "dep_rel_lo" {boundary
    key: lower bound of the simultaneous band}, with boundary keys "start" and "after:<site key>";
    "pair_ab_proj" {candidate key: mean pair_ab_proj at r under the source edit}; "carried" {candidate key:
    [(m(U_c + KO_c), m(B + KO_c), m(KO_c)) per item]}; optional "dev_keys" and "sha256".
    Rule (a): dep_rel(after c) >= x6.read_frac * dep_rel(start) and band lower bound > 0. Rule (b): presence
    s = pair_ab_proj + 0.5 >= x6.presence_min, or mean e_carried >= x6.carried_min over >= 10 items with
    denominator >= 2. r_in = every c with (a) and (b); r_post = the earliest c with (b) and not (a); r_pre = the
    earliest c without (b), or the first candidate labelled `early control` when (b) holds there."""
    F = geom.first_changed(edit_site)
    if F is None:
        raise SteerConfigError("slot_rule: edit %s changes no K/V" % edit_site.key())
    read_frac = float(_req(thresholds, "x6.read_frac", "slot_rule thresholds"))
    presence_min = float(_req(thresholds, "x6.presence_min", "slot_rule thresholds"))
    carried_min = float(_req(thresholds, "x6.carried_min", "slot_rule thresholds"))
    dep, lo = curves["dep_rel"], curves["dep_rel_lo"]
    full = dep.get("start")

    def rule_a(bkey: str) -> bool:
        da, lb = dep.get(bkey), lo.get(bkey)
        return full is not None and da is not None and lb is not None and da >= read_frac * full and lb > 0

    bounds = [None] + [Site.parse(k[len("after:"):]) for k in dep if k.startswith("after:")]
    passing = [b for b in bounds if rule_a(_boundary_key(b))]
    last = max(passing, key=lambda b: (-1, 0) if b is None else geom.rank(b)) if passing else None
    window = {"last_read": _boundary_key(last) if passing else None,
              "W": [c.key() for c in suffix_calls(geom, last)] if passing else []}
    th = {"x6.read_frac": read_frac, "x6.presence_min": presence_min, "x6.carried_min": carried_min}
    if geom.injection:
        return dict(window, F=int(F), edit_site=edit_site.key(), slots={}, r_in=None, r_pre=None, r_pre_label=None,
                    r_post=None, n_candidates=0, label="no undo on injection (PLAN A5)", reason="injection",
                    thresholds=th, dev_keys=curves.get("dev_keys"), sha256=curves.get("sha256"))
    cands = sorted([c for c in candidates if geom.executed_index(c) >= F], key=geom.rank)
    slots = {}
    for c in cands:
        bkey = _boundary_key(c)
        if bkey not in dep:
            raise SteerConfigError("slot_rule: candidate %s is not on the suffix grid" % c.key())
        slots[c.key()] = _slot(rule_a(bkey), curves, c.key(), presence_min, carried_min)
        slots[c.key()]["e"] = geom.executed_index(c)
    keys = [c.key() for c in cands]
    r_in = [k for k in keys if slots[k]["inside"]]
    r_post = next((k for k in keys if slots[k]["b"] and not slots[k]["a_still_read"]), None)
    r_pre = next((k for k in keys if not slots[k]["b"]), None)
    pre_label = None
    if keys and slots[keys[0]]["b"]:
        r_pre, pre_label = keys[0], "early control"
    return dict(window, F=int(F), edit_site=edit_site.key(), slots=slots, r_in=r_in, r_pre=r_pre,
                r_pre_label=pre_label, r_post=r_post, n_candidates=len(cands),
                label="in-window" if r_in else "no in-window slot", reason=None if r_in else "no_candidate",
                thresholds=th, dev_keys=curves.get("dev_keys"), sha256=curves.get("sha256"))


def _slot(a: bool, curves: Dict[str, Any], ckey: str, presence_min: float, carried_min: float) -> Dict[str, Any]:
    """Clauses of one candidate (k17_varbench.window_rule_v4: b1 presence, b2 e_carried; informative = a and b2
    and selected_by as k17_varbench.window_rule:425-428)."""
    proj = curves.get("pair_ab_proj", {}).get(ckey)
    pres = None if proj is None else float(proj) + 0.5
    ec = [(u - k) / (b - k) for u, b, k in curves.get("carried", {}).get(ckey, [])
          if None not in (u, b, k) and (b - k) >= CARRIED_GAP]
    ecm = _mean(ec)
    b1 = pres is not None and pres >= presence_min
    b2 = ecm is not None and len(ec) >= CARRIED_MIN_N and ecm >= carried_min
    inside = a and (b1 or b2)
    return {"a_still_read": a, "b_presence": b1, "b_carried": b2, "b": b1 or b2, "inside": inside,
            "presence": pres, "e_carried": ecm, "e_carried_n": len(ec),
            "selected_by": ("a+b2" if b2 else "a+b1_only") if inside else None, "informative": a and b2}


# ------------------------------------------------------------------------------------------ X4 rows
def matched_keys(item: Any, geom: Geometry, calls: Sequence[Union[Site, str]], mass: Optional[Dict[int, float]],
                 target: float, seed_key: str, max_draws: int = 50, tol: float = 0.10,
                 dm_fn: Optional[Callable[[List[int]], float]] = None, read_idx: int = 0) -> Optional[Dict[str, Any]]:
    """Size-matched unrelated keys K_u (matrix X4b, DESIGN 22.6). Eligible keys: positions < read, not in src,
    not 0 (the sink, path audit F3), not the read. Draw t permutes them with
    random.Random(int(sha256(f"{seed_key}|ku|{t}").hexdigest()[:16], 16)).
    - mass matching (mass given; target = mass(src)): the shortest prefix whose mass reaches the target,
      accepted when |mass - target| <= tol * target [decision there: draw procedure];
    - fallback (mass None, e.g. raven): the first |src| keys of the draw, accepted when |dm_fn(K)| is within
      25% of |target| = |dm_clean(src)|.
    The read is `item.reads[read_idx]` (DESIGN 12, PF7). Returns the accepted draw, or None when no draw in
    max_draws is accepted (the item is unmatched)."""
    src, read = _source(item), _read_pos(item, read_idx)
    eligible = [p for p in range(1, read) if p not in src]
    call_keys = [_key(c) for c in calls]
    for t in range(max_draws):
        perm = random.Random(_seed_int("%s|ku|%d" % (seed_key, t))).sample(eligible, len(eligible))
        if mass is not None:
            got = _mass_prefix(perm, mass, float(target), tol)
            if got is not None:
                keys, m = got
                return {"keys": sorted(keys), "kind": "mass", "draws": t + 1, "size_ratio": m / float(target),
                        "calls": call_keys}
            continue
        if dm_fn is None:
            raise SteerConfigError("matched_keys: the fallback without attention mass needs dm_fn")
        if len(eligible) < len(src):
            return None
        keys = sorted(perm[:len(src)])
        dm = abs(float(dm_fn(keys)))
        if abs(dm - abs(float(target))) <= KU_FALLBACK_TOL * abs(float(target)):
            return {"keys": keys, "kind": "dm", "draws": t + 1, "size_ratio": dm / abs(float(target)),
                    "calls": call_keys}
    return None


def _mass_prefix(perm: Sequence[int], mass: Dict[int, float], target: float, tol: float) -> Optional[Tuple[List[int], float]]:
    if not target > 0:
        raise SteerConfigError("matched_keys: the source mass must be > 0")
    cum = 0.0
    for i, p in enumerate(perm):
        if p not in mass:
            raise SteerConfigError("matched_keys: no attention mass for key %d" % p)
        cum += float(mass[p])
        if cum >= target:
            return (list(perm[:i + 1]), cum) if abs(cum - target) <= tol * target else None
    return None


def matched_cell_status(results: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Cell status of matrix X4b: unmatched items counted; more than 30% unmatched gives `no matched control`."""
    n = len(results)
    unmatched = sum(1 for r in results if r is None)
    share = unmatched / n if n else None
    label = "no matched control" if share is None or share > KU_UNMATCHED_MAX else "matched"
    return {"n_items": n, "n_matched": n - unmatched, "n_unmatched": unmatched, "unmatched_share": share,
            "label": label}


def freeze_spec(component: Union[Component, str], positions: Any, F: int, geom: Geometry) -> Dict[str, Any]:
    """X4a branch freeze (matrix X4a, DESIGN 22.6): `patch` with target clean, a = 1, at every call with
    executed index >= F of `component` (ATTN_OUT or MLP_OUT), prelude and coda calls included where the family
    has them, over `positions` (the read position). The payload is the clean-A state at the same calls and
    positions from a batch-1 clean forward, so the frozen values are bit-copies of clean A. The plan sets id."""
    comp = Component(component)
    if comp not in (Component.ATTN_OUT, Component.MLP_OUT):
        raise SteerConfigError("freeze_spec: component must be attn_out or mlp_out, got %s" % comp.value)
    if geom.family == "ifm":
        raise UnsupportedSite("freeze_spec: IFM v1 has no ATTN_OUT or MLP_OUT hooks (matrix Q4)")
    if F is None:
        raise SteerConfigError("freeze_spec: F is None")
    sites = ["loop*.L*.%s" % comp.value]
    if geom.n_prelude:
        sites.insert(0, "prelude.L*.%s" % comp.value)
    if geom.n_coda:
        sites.append("coda.L*.%s" % comp.value)
    pos = _positions_dict(_positions(positions))
    return {"id": "freeze_%s" % comp.value, "op": "patch", "a": 1.0, "target": {"kind": "clean"}, "site": sites,
            "schedule": {"kind": "every_loop", "calls_from": int(F)}, "positions": pos, "label": "freeze"}


def mean_target(states: Sequence[Tensor]) -> Tensor:
    """Dev-mean ablation target (matrix X4e (i); IOI 2211.00593 section 3 mean ablation): the float32 mean of
    the dev items' states at their own read position, [d]."""
    if not states:
        raise SteerConfigError("mean_target: no dev states")
    return torch.stack([s.float().reshape(-1) for s in states]).mean(dim=0)


def dev_item_donor(item: Any, dev_items: Sequence[Any], s: int) -> Optional[Any]:
    """Resample donor of matrix X4e (ii): a dev item whose tA differs from the item's, drawn by
    random.Random(int(sha256(f"{item.key}|x4e|{s}").hexdigest()[:16], 16)) from the eligible dev items sorted by
    key [decision: sorting makes the draw independent of list order]. None when no dev item is eligible."""
    key, tA = str(_item(item, "key")), _item(item, "tA")
    pool = sorted((d for d in dev_items if _item(d, "tA") != tA and str(_item(d, "key")) != key),
                  key=lambda d: str(_item(d, "key")))
    if not pool:
        return None
    return random.Random(_seed_int("%s|x4e|%d" % (key, s))).choice(pool)


def replacement_ratio(replacement: Tensor, clean: Tensor, hA: Tensor, hB: Tensor) -> Dict[str, Any]:
    """X4e replacement size ||replacement - clean|| / ||H^B - H^A|| at r; below 0.5 the cell is `vacuous`
    (matrix X4e [decision there])."""
    den = float((hB.float() - hA.float()).norm())
    if den == 0:
        return {"ratio": None, "label": "undefined"}
    ratio = float((replacement.float() - clean.float()).norm()) / den
    return {"ratio": ratio, "label": "vacuous" if ratio < VACUOUS_RATIO else "sized"}
