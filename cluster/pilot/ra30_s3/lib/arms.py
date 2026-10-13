"""RA-30 arms: the arm table, steer_gen row infos, the neuron hook and the neuron sets (plan_ra30.md sections 5, 7).

State edits use steer_gen.SteerHooks unchanged ("patch" and "rand" ops at the post-norm loop-end site post@j, the
"kv" op on the loop-1 (kv1) or loop-4 (kv4, identity row) k_proj / v_proj outputs, explicit absolute positions; the adapter is built with kv=True). The neuron arms add one forward-pre hook on mlp.down_proj of the last layer: at loop 1
(current_ut 0, read from SteerHooks' live forward context f["ut"], f["li"]), at the edited positions of a row, the
selected neurons of the down_proj input a take the donor's recorded values (model dtype, bit-exact), every other
neuron keeps its live value. That equals adding down_proj(d * mask), d = a_donor - a_A, to the MLP output of that
layer only: the arm carries only the last layer's MLP part of the loop-1 state difference (review N7; not a bound,
the decode is nonlinear).
"""
import torch

import ra28_score as SC

# ------------------------------------------------------------------ the arm table (default baseline first)
NEURON_SETS = ("full", "t1", "t5", "t20", "r5", "ds5")
FRAC = {"t1": 0.01, "t5": 0.05, "t20": 0.20, "r5": 0.05, "ds5": 0.05}

ARMS = [
    # name, batch, sequence, donor, edits (pos "P" = derived-value positions, "span", "last_prompt"), role
    ("base", "scratch", "A", "B", [("patch", 1, "last_prompt")], "default baseline: last prompt token, end of loop 1"),
    ("text", "prefix", "A+prefB", None, [], "text edit (ceiling)"),
    ("floor", "prefix", "A+prefA", None, [], "unedited (floor)"),
    ("cv1", "prefix", "A+prefA", "A+prefB", [("patch", 1, "P")], "chain-value patch, end of loop 1"),
    ("cv2", "prefix", "A+prefA", "A+prefB", [("patch", 2, "P")], "chain-value patch, end of loop 2"),
    ("cv3", "prefix", "A+prefA", "A+prefB", [("patch", 3, "P")], "chain-value patch, end of loop 3"),
    ("kv1", "prefix", "A+prefA", "A+prefB", [("kv", [1], "P")],
     "chain-value key-value patch, loop 1 (the only read of P the post arms never edit; text = cv1 + kv1)"),
    ("cvP", "prefix", "A+prefA", "A+prefB", [("patch", [1, 2, 3, 4], "P")], "persistent chain-value patch"),
    ("rnd1", "prefix", "A+prefA", "A+prefB", [("rand", 1, "P")], "random same-norm, derived positions, loop 1"),
    ("rnd3", "prefix", "A+prefA", "A+prefB", [("rand", 3, "P")], "random same-norm, derived positions, loop 3"),
    ("span1", "scratch", "A", "B", [("patch", 1, "span")], "span patch, end of loop 1 (reference)"),
    ("clean", "scratch", "A", None, [], "clean A in the scratch batch (in-batch reference)"),
]
# identity rows, decoded only on the gate items (pre-registered identities, review B1): kv4 = cv3 at the K/V level
IDENTITY_ARMS = [("kv4", "prefix", "A+prefA", "A+prefB", [("kv", [4], "P")], "loop-4 key-value patch (= cv3)")]
ARMS += [("nc_" + s, "prefix", "A+prefA", "A+prefB", [("neuron", 1, "P")], "chain value, MLP neurons " + s)
         for s in NEURON_SETS]
ARMS += [("ns_" + s, "scratch", "A", "B", [("neuron", 1, "span")], "span, MLP neurons " + s) for s in NEURON_SETS]
ARM_NAMES = [a[0] for a in ARMS]
ARM = {a[0]: {"batch": a[1], "seq": a[2], "donor": a[3], "edits": a[4], "role": a[5]} for a in ARMS + IDENTITY_ARMS}
IDENTITIES = (("cvP", "cv1"), ("kv4", "cv3"))     # (arm, its identity), token-identical share printed on the card
PREFIX_ARMS = [a for a in ARM_NAMES if ARM[a]["batch"] == "prefix"]
SCRATCH_ARMS = [a for a in ARM_NAMES if ARM[a]["batch"] == "scratch"]
DONOR_ROW = {"prefix": "kv1", "scratch": "base"}  # the row whose donor record serves its batch (kv1 records K/V)


def key_of(ds, idx, arm):
    """Per-item row key (review B2): '<ds>:<idx>|<arm>'."""
    return "%s:%s|%s" % (ds, int(idx), arm)
FULL_STATE = {"nc": "cv1", "ns": "span1"}      # the full-state arm each neuron arm's share is read against


def steer_edits(arm, P):
    """steer_gen edit dicts of an arm (neuron edits are the NeuronHook's, not SteerHooks')."""
    out = []
    for op, loop, pos in ARM[arm]["edits"]:
        if op == "neuron":
            continue
        p = [int(P[0]), int(P[1])] if pos == "P" else pos
        if op == "patch":
            out.append({"op": "patch", "site": "post", "loop": loop, "pos": p, "a": 1.0})
        elif op == "kv":
            out.append({"op": "kv", "loops": list(loop), "pos": p, "a": 1.0})
        else:
            out.append({"op": "rand", "site": "post", "loop": loop, "pos": p, "seed": 0, "ref": "patch"})
    return out


def row_info(arm, g, item_key):
    """steer.rows entry of one arm for one item. g: prompt_A, prompt_B (ids), prefix_A, prefix_B, P, span."""
    a = ARM[arm]
    pa, pb = list(g["prompt_A"]), list(g["prompt_B"])
    seqs = {"A": pa, "B": pb, "A+prefA": pa + list(g["prefix_A"]), "A+prefB": pa + list(g["prefix_B"])}
    seq = seqs[a["seq"]]
    donor = seqs[a["donor"]] if a["donor"] else None
    s0, s1 = int(g["span"][0]), int(g["span"][1])
    if a["batch"] == "scratch" and (len(pb) != len(pa) or pa[:s0] != pb[:s0] or pa[s1:] != pb[s1:]):
        raise ValueError("prompt B does not match prompt A outside the span (review B3)")
    rng = (int(g["P"][0]), int(g["P"][1])) if a["batch"] == "prefix" else (s0, len(pa))
    edits = steer_edits(arm, g["P"])
    for e in edits:
        if e["op"] == "rand":
            e["rkey"] = item_key                      # one seeded draw per item (and site)
    return {"plen": len(pa), "span": (s0, s1), "a_ids": seq, "b_ids": donor if donor is not None else list(seq),
            "edits": edits, "lcdl": None, "donor_range": rng, "t0": rng[0]}


def neuron_range(arm, g):
    if ARM[arm]["batch"] == "prefix":
        return int(g["P"][0]), int(g["P"][1])
    return int(g["span"][0]), int(g["span"][1])


# ------------------------------------------------------------------ the neuron hook
class NeuronHook(object):
    """down_proj forward-pre hook at (loop, layer). specs[key] = {"lo", "hi", "idx" (LongTensor or None = all),
    "donor" [hi - lo, I] model dtype}. record = (lo, hi) records x[0, lo:hi] of a forward with no batch keys."""

    def __init__(self, steer, layer, loop=1):
        self.steer = steer
        self.layer = int(layer)
        self.loop = int(loop)
        self.specs = {}
        self.record = None
        self.recorded = None
        self.handle = None
        self.n_applied = {}

    def install(self, model):
        if self.handle is not None:
            raise RuntimeError("neuron hook already installed")
        self.handle = model.model.layers[self.layer].mlp.down_proj.register_forward_pre_hook(self._pre)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None

    def _pre(self, module, args):
        f = self.steer.f
        if f is None or f.get("ut") != self.loop - 1 or f.get("li") != self.layer:
            return None
        x = args[0]
        keys = f.get("keys")
        if keys is None:
            if self.record is not None:
                lo, hi = self.record
                self.recorded = x[0, lo:hi].detach().clone()
            return None
        y = None
        for b, key in enumerate(keys):
            sp = self.specs.get(key)
            if sp is None:
                continue
            absp = f["abs"][b]
            cols = ((absp >= sp["lo"]) & (absp < sp["hi"])).nonzero().view(-1)
            if cols.numel() == 0:
                continue
            if y is None:
                y = x.clone()
            don = sp["donor"].to(x.device).index_select(0, (absp[cols] - sp["lo"]).long())
            if sp["idx"] is None:
                y[b, cols] = don.to(y.dtype)
            else:
                idx = sp["idx"].to(x.device)
                sub = y[b].index_select(0, cols)
                sub[:, idx] = don[:, idx].to(y.dtype)
                y[b].index_copy_(0, cols, sub)
            self.n_applied[key] = self.n_applied.get(key, 0) + int(cols.numel())
        return None if y is None else (y,) + tuple(args[1:])


@torch.no_grad()
def record_acts(model, steer, hook, ids, lo, hi):
    """a = the down_proj input of (hook.layer, hook.loop) at positions [lo, hi) of one batch-1, no-cache forward
    (the forward steer_gen's record_donor makes), model dtype."""
    if steer.f is not None:
        raise RuntimeError("record_acts inside a forward")
    saved = steer.batch_keys
    steer.batch_keys = None
    hook.record, hook.recorded = (int(lo), int(hi)), None
    try:
        dev = next(model.parameters()).device
        model(input_ids=torch.tensor([list(ids)], dtype=torch.long, device=dev), use_cache=False, logits_to_keep=1)
    finally:
        hook.record = None
        steer.batch_keys = saved
    if hook.recorded is None:
        raise RuntimeError("the neuron hook recorded nothing (loop %d, layer %d)" % (hook.loop, hook.layer))
    return hook.recorded


# ------------------------------------------------------------------ neuron sets
def k_of(frac, n):
    return int(round(frac * n))


def item_scores(a_A, a_D):
    """Per neuron, the L2 norm over the edited positions of d = a_D - a_A (float32): [I]."""
    return (a_D.float() - a_A.float()).norm(dim=0)


def top_set(score, k):
    """Indices of the k largest scores, ties to the lower index (stable), sorted."""
    order = torch.argsort(-score.double(), stable=True)
    return sorted(int(i) for i in order[:k])


def random_set(n, k, *parts):
    return SC.random_subset(n, k, *parts)


def dataset_scores(pairs):
    """Mean over split items of the per-neuron mean |d| over positions: [I]. pairs = [(a_A, a_D)]."""
    acc = None
    for a_A, a_D in pairs:
        s = (a_D.float() - a_A.float()).abs().mean(dim=0)
        acc = s if acc is None else acc + s
    if acc is None:
        raise ValueError("no split item")
    return acc / len(pairs)


def neuron_sets(a_A, a_D, ds_set, model_name, ds, setting, item_key):
    """{set name: sorted index list or None (all)} for one item and one setting (chain value or span)."""
    I = int(a_A.shape[-1])
    sc = item_scores(a_A, a_D)
    out = {"full": None}
    for s in ("t1", "t5", "t20"):
        out[s] = top_set(sc, k_of(FRAC[s], I))
    out["r5"] = random_set(I, k_of(FRAC["r5"], I), "ra30", model_name, ds, setting, item_key, "r5")
    if ds_set is not None:                       # split items get no dataset-level arm (no leakage)
        out["ds5"] = list(ds_set)
    if ds_set is not None and len(ds_set) != k_of(FRAC["ds5"], I):
        raise ValueError("dataset-level set has %d neurons, expected %d" % (len(ds_set), k_of(FRAC["ds5"], I)))
    return out
