"""K17 VarBench placement with the copy-read target (brief B-K17-VARBENCH, PLAN.md amendments A1-A6).
Plan (frozen definitions, thresholds, predictions): notes/k17_varbench_plan.md (v6).
Imports k15_smoke, k15_smoke2, k15_data, k13_edit and k10_ifm; never modifies them.

Copy-read target (A3): teacher-force A's gold GSM8K chain up to the token before the first restatement of the
edited number x; read the x' vs x preference at that position r. Every scored row is one batch-1 forward.
A site is (loop j, layer l). Ouro: l < L-1 is the layer-l output, l = L-1 is the model.norm output (carried
state, readout input). IFM / d112: executed slot 28(j-1) + l (block output).
Arms (plan section 4): V1 placement, V2 PC-site, V3 presence, V4 read window, V5 undo (phase W picks the slots,
phase M runs them), V6 carry gain (predict then verify) on VarBench and D1, D2, D4, V7 answer doses.

    python k17_varbench.py --selftest
    python k17_varbench.py --build_items --out DATA_DIR                      (laptop CPU, tokenizers only)
    python k17_varbench.py --cpu_check --models ouro --data DATA_DIR --out OUT [--device cuda]
    python k17_varbench.py --build_vbd --out DATA_DIR                        (v5 derived-use items, laptop CPU)
    python k17_varbench.py --fd_check --models ouro --data DATA_DIR --out OUT   (v5 per-family G check)
    python k17_varbench.py --bf16_check --models ouro --data DATA_DIR --out OUT [--fd_dir DIR] (v6 V6-gate check)
    python k17_varbench.py --models ouro --data DATA_DIR --out OUT --phase W|M|all [--debug]
    python k17_varbench.py --summarize --out OUT [--plan PLAN.md] [--cgcheck CGCHECK_checks.json]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re

import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import k15_smoke2 as S2  # noqa: E402  (apply_entries)

OURO_REPOS = {"ouro": "ByteDance/Ouro-1.4B", "ouro_think": "ByteDance/Ouro-1.4B-Thinking",
              "ouro26": "ByteDance/Ouro-2.6B", "ouro26_think": "ByteDance/Ouro-2.6B-Thinking"}
IFM_MODELS = ("ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_d112")
# Hub revisions pinned for the cluster run (values from cluster/weights.json); a local folder path gets None
OURO_REVISION = {"ByteDance/Ouro-1.4B": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
                 "ByteDance/Ouro-2.6B": "1ed04250da1a9936042725d302e81c8fa2ab5abd",
                 "ByteDance/Ouro-1.4B-Thinking": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
                 "ByteDance/Ouro-2.6B-Thinking": "f1edd81e7ac41355db670500ceaf204e0f73af68"}
MODEL_ORDER = list(OURO_REPOS) + list(IFM_MODELS)
GSET = {"ouro": "full", "ouro_think": "full", "ouro26": "red", "ouro26_think": "red",
        "ifm_p1_ouro": "red", "ifm_p1_ouro_inj": "red", "ifm_d112": "red"}
N_VB = {"ouro": 200, "ouro_think": 200, "ouro26": 200, "ouro26_think": 200,
        "ifm_p1_ouro": 60, "ifm_p1_ouro_inj": 60, "ifm_d112": 60}
N_SUB = {"ouro": 40, "ouro_think": 40, "ouro26": 40, "ouro26_think": 40,
         "ifm_p1_ouro": 30, "ifm_p1_ouro_inj": 20, "ifm_d112": 30}   # v4: IFM undo models 20 -> 30 (plan critic 2 F3)
A2_N = {"ouro": 80, "ouro_think": 80, "ouro26": 80, "ouro26_think": 80,
        "ifm_p1_ouro": 40, "ifm_p1_ouro_inj": 40, "ifm_d112": 40}
NO_UNDO = ("ifm_p1_ouro_inj",)
# v5 (plan critic 3 F2): RND runs at all 24 V1 cells on every family; RND_CELLS_IFM is the cut-Q5 fallback only
RND_CELLS_IFM = {("src", "L0@1"), ("src", "Llast@1"), ("src", "Llast@3"), ("src", "L0@4"), ("src", "Llast@4"),
                 ("ans", "Llast@1"), ("ans", "Llast@3")}
RND_ALL = True
CODE_VERSION = "v6"
# v6 (code review 3, plan section 14 v6)
PI_A = (0.25, 0.5, 0.75, 1.0)   # PLAN G4: path-integrated gain, mean gradient at the lerped single-site state
PI_N = 20                   # G_PI dev items per dataset (cap; pilot 20, S1 --pi_n 40); 0 = off
NO_GRAD = [False]          # cluster copy (--no_gradient_rank): skip the gradient score (carry gain), its verify rows, G_PI
PI_DATASETS = ("VarBench", "D2", "VBD")
BF16_N = 30                 # A-F3: fresh VarBench items per model in the bf16-vs-fp32 selector check
BF16_BAND_MIN, FN_AGREE_MIN = 0.9, 0.95
CGCHECK_SHA256 = "46a4e86580839de7793e291ef14ac681cd2ab33784e38732a23441d30149ee1f"   # A-F6: CGCHECK full-run checks.json
DKO_GUARD = 2.0             # C-F1: b2 = car / D_KO only where D_KO = m(BKO_c) - m(KO_c) >= 2 logits
# v5 (plan critic 3 F5): derived-use target. RA-20 A2 item 1: 0 = no cap (every item of the pool, the cluster
# default); --vbd_n N still caps at N for backward compatibility.
VBD_N = {"ouro": 0, "ouro_think": 0, "ouro26": 0, "ouro26_think": 0,
         "ifm_p1_ouro": 0, "ifm_p1_ouro_inj": 0, "ifm_d112": 0}
SHARD_SALT = "k17vb-shard:"   # RA-20 A2 item 1: stable item partition, shard = sha256(salt + pair_id)[:8] mod n
VBD_V1 = {"ouro": True, "ouro_think": True, "ouro26": True, "ouro26_think": True,
          "ifm_p1_ouro": False, "ifm_p1_ouro_inj": False, "ifm_d112": False}   # IFM: V6 only (shared-host cap)
# B-EXT2 item 5: --vbd_v1 name=1 turns the V1 cells on for the named models (default: the dict above)
FD_A = (1e-3, 1e-2, 0.1)    # v5 (plan critic 3 F4): finite-difference doses for the per-family G check (fp32 model)
FD_FLOOR = 1e-4             # |a G| below this (logits) is not scored in the FD check
FD_N = 4                    # VarBench items per model in the FD check
LO_RND = {("src", "L0@1"), ("src", "Llast@1"), ("ans", "L0@1"), ("ans", "Llast@1")}
WIN_BOUNDS = ["full", "after_Llast@1", "after_Lmid@2", "after_Llast@2", "after_Lmid@3", "after_Llast@3", "after_Lmid@4"]
CAND_SLOTS = ["Lmid@2", "Llast@2", "Lmid@3", "Llast@3", "Lmid@4"]   # undo candidates (v2.1: Lmid@2, Lmid@3 added after the debug run)
GAP = 2.0
NBOOT = 1000
TOST_M = 0.10
A_VERIFY = (1.0, 0.25)
THR = {1.0: 2.0, 0.25: 0.5}
HIT_TOL = 0.05              # v3 T7 (review MS2): hit if dm(top) >= max dm - HIT_TOL * D
FN_G = 0.05                 # v3 A7 false-negative threshold: |G| < FN_G * D counts as "predicted not carried"
CARRIED_E = 0.5             # a verified site is "carried" when its e >= 0.5
# v5 (code review 2, plan section 14): undo and window in first-token logits on the same items
REMOVE_MIN = 0.2            # V5 clause (a), influence: post-c reads remove >= 0.2 of the source edit's dm
CARRY_MIN, CARRY_LO = 0.3, 0.15   # b2 (informative): v6 mean (u - k) / D_KO >= 0.3, 95% CI lower > 0.15 (C-F1)
SURV_REV, SURV_STAY = 0.3, 0.7    # T6 v5: reverts if the 95% CI upper of survival s(c) <= 0.3; stays if lower >= 0.7
MIN_CL = 10                 # item AND cluster minimum for b2 and the A2-1 loops < K row (pool pairs: up to 6 per problem)
T8_MIN = 40                 # T8: a model with fewer than 40 kept clusters is reported and not interpreted
PLANT_F = 0.5               # probe power control: planted digit code with norm 0.5 x the item's own |h_A - h_B|
MODEL_DIR = {}
ALLOW_HOST_MIX = False      # v5 (plan critic 3 F3): a results file never mixes host roles unless this flag is set
ROLE = ""                   # host role written instead of the host name (A7): ouro_host, ifm_host, laptop
DROP_BLOCK = False          # laptop debug only: zero-shot copy-read (block removed) so the backward fits in 8 GB

# build inputs (laptop)
from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE  # noqa: E402
DS = os.path.join(HERE, "..", "..", "data_steer")   # cluster copy: build inputs relative to the bundle
PAIRS = DS + "/varbench/pairs_gsm8k.jsonl"
GSM_DIR = os.path.join(_HF_HUB_CACHE, "datasets--openai--gsm8k/snapshots/"
                             "740312add88f781978c0658806c59bc2815b9866/main")
PROMPT_FILE = os.path.join(HERE, "..", "..", "prod", "tasks", "data", "prompt_gsm8k.txt")
TOK_OURO = os.path.join(_HF_HUB_CACHE, "models--ByteDance--Ouro-1.4B/snapshots/"
                              "574fa66cb8bf5abdc979642d01cf2b79b16bfab1/tokenizer.json")
TOK_IFM = DS + "/_cache/tok_ifm/tokenizer.json"
IFM_BOS = 0
K16_ART = os.environ.get("PILOT_ARTIFACTS", "")   # cluster copy: pilot artifact folder (build only)
D4_DIR = os.path.join(HERE, "..", "data", "d4")


def r5(x):
    return None if x is None else round(float(x), 5)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def skey(site):
    return "%d.%d" % tuple(site)


def ksite(k):
    a, b = k.split(".")
    return (int(a), int(b))


# ------------------------------------------------------------------ pure helpers: text, items
NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def numval(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def first_iso(sol, x):
    """First isolated match of str(x): not adjacent to a digit or to a digit-joined '.' or ','."""
    s = str(x)
    for mt in re.finditer(re.escape(s), sol):
        a, b = mt.start(), mt.end()
        if a > 0 and (sol[a - 1].isdigit() or (sol[a - 1] in ".," and a > 1 and sol[a - 2].isdigit())):
            continue
        if b < len(sol) and (sol[b].isdigit() or (sol[b] in ".," and b + 1 < len(sol) and sol[b + 1].isdigit())):
            continue
        return a
    return None


def in_annot(sol, a):
    return sol.rfind("<<", 0, a) > sol.rfind(">>", 0, a)


def contradictory_prefix(C, x):
    """Prefix-validity filter (GPT critic, EXPERIMENTS 3c): True if the prefix already states x's value in another
    format, or a calculator step takes an input derived from x (taint: x, then results of tainted steps)."""
    xv = float(x)
    if any(numval(t) == xv for t in NUM.findall(C)):
        return True
    taint = {xv}
    for mt in re.finditer(r"<<([^=>]*)=([^>]*)>>", C):
        ins = [numval(t) for t in NUM.findall(mt.group(1))]
        if any(v in taint for v in ins if v is not None):
            return True
        res = numval(mt.group(2).strip().rstrip("."))
        if res is not None and any(v in taint for v in ins if v is not None):
            taint.add(res)
    return False


def ambiguous_source(q, span, x):
    """x also occurs elsewhere in the question (another copy source)."""
    xv = float(x)
    for mt in NUM.finditer(q):
        if mt.start() == span[0]:
            continue
        if numval(mt.group()) == xv:
            return True
    return False


UNITS_W = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                      "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS_W = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES_W = {"hundred": 100, "thousand": 1000, "million": 10 ** 6}
_NW = "|".join(list(UNITS_W) + list(TENS_W) + list(SCALES_W))
WORD_RE = re.compile(r"\b(?:%s)(?:[\s-]+(?:and[\s-]+)?(?:%s))*\b" % (_NW, _NW), re.I)


def words_val(s):
    """Value of an English cardinal number phrase ('three', 'twenty-five', 'two hundred and ten')."""
    total, cur = 0, 0
    for t in [t for t in re.split(r"[\s-]+", s.lower()) if t and t != "and"]:
        if t in UNITS_W:
            cur += UNITS_W[t]
        elif t in TENS_W:
            cur += TENS_W[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        else:
            total += (cur or 1) * SCALES_W[t]
            cur = 0
    return float(total + cur)


def word_nums(s):
    """Number words in s -> [(start, text, value)] (v6, review C-F3: numeric regexes miss 'Three')."""
    return [(mt.start(), mt.group(), words_val(mt.group())) for mt in WORD_RE.finditer(s)]


def prose_flags(C, q, x, extra_ok=()):
    """v6 prefix rule (review C-F3, the VBD prose rule applied to every chain prefix). Returns drop reasons:
    'x_word_form' = C states x as a number word; 'prose_unverified_number' = a number outside the <<...>>
    annotations (digits or number words) that is neither x, nor a number of the question (digits or words), nor
    the result of an earlier annotation in C, nor in extra_ok (it may be an unannotated value derived from x)."""
    xv = float(x)
    out = []
    if any(v == xv for _, _, v in word_nums(C)):
        out.append("x_word_form")
    qn = {numval(t) for t in NUM.findall(q)} | {v for _, _, v in word_nums(q)}
    earlier = {numval(mt.group(2).strip().rstrip(".")) for mt in STEP_RE.finditer(C)} | set(extra_ok)
    prose = STEP_RE.sub("", C)
    pv = [numval(t) for t in NUM.findall(prose)] + [v for _, _, v in word_nums(prose)]
    if any(v is not None and v != xv and v not in qn and v not in earlier for v in pv):
        out.append("prose_unverified_number")
    return out


def tertile(v, cuts):
    return 0 if v <= cuts[0] else (1 if v <= cuts[1] else 2)


def rr_order(items, key):
    """Round-robin over groups (sorted names), keeping each group's own order (critic F9d)."""
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    names = sorted(groups)
    out, k = [], 0
    while len(out) < len(items):
        for g in names:
            if k < len(groups[g]):
                out.append(groups[g][k])
        k += 1
    return out


# ------------------------------------------------------------------ pure helpers: operators, metrics, stats
def e_lerp(pos, tgt, a, dev):
    P = len(pos)
    return ("lerp", torch.zeros(P, dtype=torch.long, device=dev), torch.tensor(pos, device=dev),
            tgt.float().to(dev), torch.full((P, 1), float(a), device=dev))


def e_add(pos, delta, dev):
    P = len(pos)
    return ("add", torch.zeros(P, dtype=torch.long, device=dev), torch.tensor(pos, device=dev), delta.float().to(dev))


def rnd_delta(pair_id, site, pos_name, positions, dnorm, H):
    """Unit Gaussian per position, seed = first 8 hex of sha256(pair_id|site|pos|p), scaled to dnorm -> [P, H]."""
    rows = []
    for p, n in zip(positions, dnorm):
        g = torch.Generator().manual_seed(int(sha("%s|%s|%s|%d" % (pair_id, skey(site), pos_name, p))[:8], 16))
        u = torch.randn(H, generator=g)
        rows.append(u / u.norm() * float(n))
    return torch.stack(rows)


def vals(z, tA, tB):
    """[z(tA), z(tB), logsumexp, top-1] for one logit vector."""
    return [r5(z[tA]), r5(z[tB]), r5(torch.logsumexp(z.float(), -1)), int(z.argmax())]


def logp(z, t):
    return float(z[t].float() - torch.logsumexp(z.float(), -1))


def proj(h, mA, mB):
    """Pair projection: 0 at clean A, 1 at clean B; None if |mB - mA| < 0.01 |mA|."""
    d = (mB - mA).float()
    if float(d.norm()) < 0.01 * float(mA.float().norm()):
        return None
    return float(((h.float() - (mA.float() + mB.float()) / 2) @ d) / (d @ d) + 0.5)


def boot2(xs, cl, n=NBOOT, seed=0, q=(0.025, 0.975)):
    """Dyadic (two-way) bootstrap for pairs of units (review MS3/F6, audit_path_ouro F8): cl[i] = (unit_A, unit_B).
    Units (subjects, A or B side) are resampled with replacement; item i gets weight c[A_i] * c[B_i].
    -> [mean, lo, hi, n_items, n_units]."""
    pr = [(float(x), c) for x, c in zip(xs, cl) if x is not None]
    if not pr:
        return None
    units = sorted({u for _, c in pr for u in c})
    ix = {u: i for i, u in enumerate(units)}
    x = torch.tensor([v for v, _ in pr], dtype=torch.float64)
    ia = torch.tensor([ix[c[0]] for _, c in pr])
    ib = torch.tensor([ix[c[1]] for _, c in pr])
    v0 = float(x.mean())
    if len(units) < 2:
        return [round(v0, 4), None, None, len(pr), len(units)]
    g = torch.Generator().manual_seed(seed)
    draw = torch.randint(len(units), (n, len(units)), generator=g)
    cnt = torch.zeros(n, len(units), dtype=torch.float64).scatter_add_(1, draw, torch.ones(n, len(units), dtype=torch.float64))
    w = cnt[:, ia] * cnt[:, ib]
    sw = w.sum(1)
    ok = sw > 0
    bs = (w[ok] * x[None, :]).sum(1) / sw[ok]
    qq = torch.quantile(bs, torch.tensor(list(q), dtype=torch.float64))
    return [round(v0, 4), round(float(qq[0]), 4), round(float(qq[1]), 4), len(pr), len(units)]


def boot(xs, cl, n=NBOOT, seed=0, q=(0.025, 0.975)):
    """Ratio-of-sums mean with a percentile bootstrap over clusters -> [mean, lo, hi, n_items, n_clusters].
    Tuple cluster labels (unit_A, unit_B) switch to the dyadic bootstrap (boot2)."""
    if any(isinstance(c, (tuple, list)) for x, c in zip(xs, cl) if x is not None):
        return boot2(xs, [tuple(c) for c in cl], n=n, seed=seed, q=q)
    groups = {}
    for x, c in zip(xs, cl):
        if x is not None:
            groups.setdefault(c, []).append(float(x))
    if not groups:
        return None
    keys = sorted(groups)
    sums = torch.tensor([sum(groups[k]) for k in keys], dtype=torch.float64)
    cnts = torch.tensor([len(groups[k]) for k in keys], dtype=torch.float64)
    v0 = float(sums.sum() / cnts.sum())
    if len(keys) < 2:
        return [round(v0, 4), None, None, int(cnts.sum()), len(keys)]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=g)
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    qq = torch.quantile(bs, torch.tensor(list(q), dtype=torch.float64))
    return [round(v0, 4), round(float(qq[0]), 4), round(float(qq[1]), 4), int(cnts.sum()), len(keys)]


def tost(xs, cl, margin=TOST_M):
    """Null by TOST if the 90% cluster-bootstrap CI lies inside [-margin, margin]; else 'below <95% upper>'."""
    b90 = boot(xs, cl, q=(0.05, 0.95))
    b95 = boot(xs, cl)
    if b90 is None or b90[1] is None:
        return None
    ok = b90[1] > -margin and b90[2] < margin
    return {"ci90": b90[:3], "null_by_tost": bool(ok),
            "label": ("null (TOST, margin %.2f)" % margin) if ok else ("below %.3f" % b95[2])}


def boot_group_diff(xs, cl, grp, g1, g2, n=NBOOT, seed=0, q=(0.025, 0.975)):
    """Difference of two stratum means (ratio of sums), mean(g1) - mean(g2), with a percentile bootstrap that
    resamples clusters jointly (review MS4: P-ST strata differences). -> [diff, lo, hi, n_g1, n_g2, n_clusters]."""
    S = {}
    for x, c, gr in zip(xs, cl, grp):
        if x is None or gr not in (g1, g2):
            continue
        s = S.setdefault(c, [0.0, 0, 0.0, 0])
        k = 0 if gr == g1 else 2
        s[k] += float(x)
        s[k + 1] += 1
    if not S:
        return None
    keys = sorted(S)
    T = torch.tensor([S[k] for k in keys], dtype=torch.float64)
    n1, n2 = float(T[:, 1].sum()), float(T[:, 3].sum())
    if n1 == 0 or n2 == 0:
        return None
    d0 = float(T[:, 0].sum()) / n1 - float(T[:, 2].sum()) / n2
    if len(keys) < 2:
        return [round(d0, 4), None, None, int(n1), int(n2), len(keys)]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=g)
    B = T[idx].sum(1)
    ok = (B[:, 1] > 0) & (B[:, 3] > 0)
    bs = B[ok, 0] / B[ok, 1] - B[ok, 2] / B[ok, 3]
    qq = torch.quantile(bs, torch.tensor(list(q), dtype=torch.float64))
    return [round(d0, 4), round(float(qq[0]), 4), round(float(qq[1]), 4), int(n1), int(n2), len(keys)]


def max_band(cols, cl, n=NBOOT, seed=0):
    """Simultaneous 95% band (max-statistic cluster bootstrap) for several per-item columns."""
    names = list(cols)
    clus = sorted(set(cl))
    if len(clus) < 2:
        return None
    ix = {c: i for i, c in enumerate(clus)}
    S = torch.zeros(len(names), len(clus), dtype=torch.float64)
    N = torch.zeros(len(names), len(clus), dtype=torch.float64)
    for a, nm in enumerate(names):
        for x, c in zip(cols[nm], cl):
            if x is not None:
                S[a, ix[c]] += x
                N[a, ix[c]] += 1
    mean = S.sum(1) / N.sum(1).clamp_min(1)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(clus), (n, len(clus)), generator=g)
    bs = S[:, idx].sum(2) / N[:, idx].sum(2).clamp_min(1)
    sd = bs.std(1).clamp_min(1e-12)
    dev = ((bs - mean[:, None]).abs() / sd[:, None]).max(0).values
    c = float(torch.quantile(dev, 0.95))
    return {nm: [round(float(mean[a]), 4), round(float(mean[a] - c * sd[a]), 4), round(float(mean[a] + c * sd[a]), 4)]
            for a, nm in enumerate(names)}


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    pr = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pr) < 3:
        return None
    rx, ry = rankdata([a for a, _ in pr]), rankdata([b for _, b in pr])
    n = len(pr)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else round(sxy / (sx * sy), 4)


def verify_sites(G):
    """G {site_key: predicted} -> (keys ranked by G desc, the 9 verify keys: top-3, median-3, bottom-3)."""
    ks = sorted(G, key=lambda k: (-G[k], k))
    n = len(ks)
    m = n // 2
    out = []
    for i in (0, 1, 2, m - 1, m, m + 1, n - 3, n - 2, n - 1):
        if 0 <= i < n and ks[i] not in out:
            out.append(ks[i])
    return ks, out


def hit_metrics(G, dm, D, thr, tol=HIT_TOL):
    """One item, one dose, over the sites in G that have a verified dm (T7 v3, review MS2).
    Eligible: D >= GAP and verified range (max - min dm) > thr. Hit (tolerance rule): dm at the top-predicted site
    (argmax G) >= max dm - tol * D. Chance = fraction of verified sites inside that band (the hit rate of a site
    drawn at random from the verified set). top3_hit = expected hit of a site drawn at random from the selector's
    own top 3 (test-item baseline). strict_hit (v2 rule: strictly larger than every other site) is a diagnostic."""
    ks = [k for k in G if k in dm and dm[k] is not None]
    if len(ks) < 3:
        return None
    rk = sorted(ks, key=lambda k: (-G[k], k))
    top = rk[0]
    acts = [dm[k] for k in ks]
    mx = max(acts)
    okD = D is not None and D >= GAP
    elig = bool(okD and (mx - min(acts)) > thr)
    band = {k for k in ks if dm[k] >= mx - tol * D} if okD else set()
    return {"eligible": elig, "hit": int(top in band) if elig else None,
            "chance": (len(band) / len(ks)) if elig else None,
            "top3_hit": (sum(1 for k in rk[:3] if k in band) / 3.0) if elig else None,
            "strict_hit": int(all(dm[top] > dm[k] for k in ks if k != top)) if elig else None,
            "regret": (mx - dm[top]) / D if okD else None,
            "spearman": spearman([G[k] for k in ks], acts) if elig else None, "n_verified": len(ks), "top": top,
            "n_band": len(band)}


def window_rule(removal, carried, pres, n_items, n_clusters, min_n=MIN_CL, n_b2=None, ncl_b2=None):
    """Frozen V5 rule for one candidate slot, v5 (plan critic 3 F1; plan section 4 V5). All in first-token logits on
    the same subset items (D >= GAP). removal = [1 - sum(u - k) / sum(m_src - m_A), lo, hi]: the share of the source
    edit's dm that the post-c span reads carry (clause a, influence). carried = cluster boot of (u - k) / D_KO (b2;
    v6, review C-F1: under the relay knockout after c the reachable range is D_KO = m(BKO_c) - m(KO_c), not D; items
    with D_KO >= DKO_GUARD; n_b2 / ncl_b2 are its counts). pres = mean presence under the source edit (b1).
    (a) removal >= REMOVE_MIN; b2 mean >= CARRY_MIN and 95% CI lower > CARRY_LO; (a) and b2 each need >= min_n items
    AND clusters. Only (a) and b2 make a slot informative for 'undo reverts' (b1 alone can select a slot where nothing
    carried survives, so a revert is forced)."""
    enough = n_items >= min_n and n_clusters >= min_n
    n_b2 = n_items if n_b2 is None else n_b2
    ncl_b2 = n_clusters if ncl_b2 is None else ncl_b2
    enough_b2 = n_b2 >= min_n and ncl_b2 >= min_n
    a = bool(enough and removal is not None and removal[0] is not None and removal[0] >= REMOVE_MIN)
    b1 = bool(pres is not None and pres >= 0.3)
    b2 = bool(enough_b2 and carried is not None and carried[0] is not None and carried[0] >= CARRY_MIN
              and carried[1] is not None and carried[1] > CARRY_LO)
    sel_by = ("a+b2" if b2 else "a+b1_only") if (a and (b1 or b2)) else None
    return {"a_still_read": a, "b_presence": b1, "b_carried": b2, "inside": bool(a and (b1 or b2)),
            "selected_by": sel_by, "informative": bool(a and b2), "enough_items": bool(enough),
            "enough_items_b2": bool(enough_b2), "min_n": min_n}


def window_rule_v4(dep_rel_full, dep_rel_after, pres, ecar, ecar_n, min_n=10):
    """v4 rule (necessity clause a, e_carried b2 >= 0.2), kept as a reported diagnostic only (plan critic 3 F1)."""
    a = dep_rel_full is not None and dep_rel_after is not None and dep_rel_after >= 0.2 * dep_rel_full
    b1 = pres is not None and pres >= 0.3
    b2 = ecar is not None and ecar_n >= min_n and ecar >= 0.2
    return {"a_still_read": bool(a), "b_presence": bool(b1), "b_carried": bool(b2), "inside": bool(a and (b1 or b2))}


def slot_quantities(items, cn):
    """v5 per-item undo/window quantities at slot cn (first-token margins m = z(tB) - z(tA); items with D >= GAP).
    tot = m(src O1 Llast@1) - m_A; car = m(UKO_c) - m(KO_c) (what reached non-span positions before c; the relay
    knockout after c blocks every later span read, so the restore at c is invisible under it); U = m(U_c) - m_A.
    v6 (review C-F1): DKO = m(BKO_c) - m(KO_c), the reachable range under the same knockout (None if BKO is missing).
    -> dict of aligned lists (None where a row is missing)."""
    out = {"tot": [], "car": [], "U": [], "D": [], "DKO": [], "cl": []}
    for it in items:
        D = Dgap(it)
        rows = it["rows"]

        def m(key):
            rv = rows.get(key)
            return None if rv is None else rv["f"][1] - rv["f"][0]
        mA = it["clean_A"][1] - it["clean_A"][0]
        s, u, k, uu = m("O1|src|Llast@1|1"), m("UKO|%s" % cn), m("KO|rel|after_%s" % cn), m("U|%s" % cn)
        b = m("BKO|%s" % cn)
        ok = D >= GAP
        out["tot"].append(None if (not ok or s is None) else s - mA)
        out["car"].append(None if (not ok or u is None or k is None) else u - k)
        out["U"].append(None if (not ok or uu is None) else uu - mA)
        out["D"].append(D if ok else None)
        out["DKO"].append(None if (not ok or b is None or k is None) else b - k)
        out["cl"].append(it["cluster"])
    return out


def slot_stats(items, cn):
    """v5 slot statistics on the given items: removal (clause a), carried / D (b2), survival s(c) = sum U / sum car
    (T6), each with a 95% cluster CI. -> dict."""
    q = slot_quantities(items, cn)
    both = [(c, t, cl) for c, t, cl in zip(q["car"], q["tot"], q["cl"]) if c is not None and t is not None]
    rr = boot_ratio([c for c, _, _ in both], [t for _, t, _ in both], [cl for _, _, cl in both]) if both else None
    removal = None if rr is None else [round(1 - rr[0], 4), None if rr[2] is None else round(1 - rr[2], 4),
                                        None if rr[1] is None else round(1 - rr[1], 4), rr[3], rr[4]]
    cod = [None if (c is None or d is None) else c / d for c, d in zip(q["car"], q["D"])]
    carried_D = boot(cod, q["cl"])
    # v6 (review C-F1, audit C8/P8): b2 is scored per D_KO (guard D_KO >= DKO_GUARD); per-D value reported beside it
    cok = [None if (c is None or dk is None or dk < DKO_GUARD) else c / dk for c, dk in zip(q["car"], q["DKO"])]
    carried = boot(cok, q["cl"])
    dkd = boot([None if (dk is None or d is None) else dk / d for dk, d in zip(q["DKO"], q["D"])], q["cl"])
    sv = [(u, c, cl) for u, c, cl in zip(q["U"], q["car"], q["cl"]) if u is not None and c is not None]
    surv = boot_ratio([u for u, _, _ in sv], [c for _, c, _ in sv], [cl for _, _, cl in sv]) if sv else None
    n_it = sum(1 for c, t in zip(q["car"], q["tot"]) if c is not None and t is not None)
    n_cl = len({cl for c, t, cl in zip(q["car"], q["tot"], q["cl"]) if c is not None and t is not None})
    n_b2 = sum(1 for x in cok if x is not None)
    ncl_b2 = len({cl for x, cl in zip(cok, q["cl"]) if x is not None})
    return {"removal": removal, "carried_over_DKO": carried, "carried_over_D": carried_D, "DKO_over_D": dkd,
            "survival": surv, "n_items": n_it, "n_clusters": n_cl, "n_b2": n_b2, "n_clusters_b2": ncl_b2,
            "dko_guard": DKO_GUARD, "n_dko_below_guard": sum(1 for dk in q["DKO"] if dk is not None and dk < DKO_GUARD)}


def t6_verdict(surv):
    """T6 v5 (plan critic 3 F1): reverts if the 95% CI upper of s(c) <= SURV_REV; stays if the lower >= SURV_STAY."""
    if surv is None or surv[1] is None or surv[2] is None:
        return None
    if surv[2] <= SURV_REV:
        return "reverts"
    if surv[1] >= SURV_STAY:
        return "stays"
    return "partial"


def boot_ratio(num, den, cl, n=NBOOT, seed=0, q=(0.025, 0.975)):
    """sum(num) / sum(den) over items where both are defined, percentile bootstrap over clusters (plan critic 2 F3:
    e_U / e_carried per undo slot). -> [ratio, lo, hi, n_items, n_clusters]; None when sum(den) is 0."""
    S = {}
    for x, y, c in zip(num, den, cl):
        if x is None or y is None:
            continue
        s = S.setdefault(c, [0.0, 0.0, 0])
        s[0] += float(x)
        s[1] += float(y)
        s[2] += 1
    if not S:
        return None
    keys = sorted(S)
    T = torch.tensor([S[k] for k in keys], dtype=torch.float64)
    if float(T[:, 1].sum()) == 0:
        return None
    v0 = float(T[:, 0].sum() / T[:, 1].sum())
    if len(keys) < 2:
        return [round(v0, 4), None, None, int(T[:, 2].sum()), len(keys)]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=g)
    B = T[idx].sum(1)
    ok = B[:, 1] != 0
    bs = B[ok, 0] / B[ok, 1]
    qq = torch.quantile(bs, torch.tensor(list(q), dtype=torch.float64))
    return [round(v0, 4), round(float(qq[0]), 4), round(float(qq[1]), 4), int(T[:, 2].sum()), len(keys)]


def plan_hashes(path):
    """Plan hashes at launch (plan critic 2 F4, P14): sha256 of the whole file and of the file without section 13
    (the run log, the only section that may change between phase W and phase M)."""
    raw = open(path, "rb").read()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    out, skip = [], False
    for ln in lines:
        if ln.startswith("## "):
            skip = ln.startswith("## 13.")
        if not skip:
            out.append(ln)
    m = CODE_SHA_RE.search("".join(out))
    return {"plan_sha256": hashlib.sha256(raw).hexdigest(),
            "plan_frozen_sha256": hashlib.sha256("".join(out).encode("utf-8")).hexdigest(),
            "plan_code_sha256": m.group(1) if m else None}


CODE_SHA_RE = re.compile(r"Launch code sha256: `?([0-9a-f]{64})`?")
IMPORTS = ("k15_smoke2.py", "k15_smoke.py", "k15_data.py", "k10_ifm.py")


def code_hashes(path=None):
    """v5 (plan critic 3 F1): sha256 of this file (checked against the plan at launch) and of the imported modules
    (recorded only; they are never modified)."""
    path = path or os.path.abspath(__file__)
    out = {"code_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    for f in IMPORTS:
        p = os.path.join(os.path.dirname(path), f)
        out["sha256_" + f] = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    return out


def first_diff_digits(xa, xb):
    """Probe labels (plan critic 2 F5/F7): the digits of x and x' at their first differing character, or None."""
    a, b = str(xa).replace(",", ""), str(xb).replace(",", "")
    for ca, cb in zip(a, b):
        if ca != cb:
            return (ca, cb) if (ca.isdigit() and cb.isdigit()) else None
    return None


def probe_site(H, labels, cands, cl):
    """Parameter-free digit probe, leave one cluster out (plan critic 2 F7). H [n, d] states, labels[n] digit of the
    state's read number, cands[n] = (digit_x, digit_x') of its item, cl[n] cluster. For each state: centre on the
    mean of the other clusters' states, compare by cosine with the other clusters' class means of the two candidate
    digits, predict the closer one (chance 0.5). -> per-state correctness (None where a candidate has no
    other-cluster state)."""
    H = H.double()
    n = H.shape[0]
    labs = sorted(set(labels))
    li = {d: i for i, d in enumerate(labs)}
    L = torch.tensor([li[d] for d in labels])
    S_tot = torch.zeros(len(labs), H.shape[1], dtype=torch.float64).index_add_(0, L, H)
    N_tot = torch.zeros(len(labs), dtype=torch.float64).index_add_(0, L, torch.ones(n, dtype=torch.float64))
    out = [None] * n
    for c in sorted(set(cl)):
        ix = [i for i in range(n) if cl[i] == c]
        t = torch.tensor(ix)
        S = S_tot - torch.zeros_like(S_tot).index_add_(0, L[t], H[t])
        N = N_tot - torch.zeros_like(N_tot).index_add_(0, L[t], torch.ones(len(ix), dtype=torch.float64))
        if float(N.sum()) == 0:
            continue
        mu = S.sum(0) / N.sum()
        for i in ix:
            a, b = cands[i]
            if N[li[a]] == 0 or N[li[b]] == 0:
                continue
            h = H[i] - mu
            ca, cb = S[li[a]] / N[li[a]] - mu, S[li[b]] / N[li[b]] - mu
            sa = float(F.cosine_similarity(h, ca, dim=0))
            sb = float(F.cosine_similarity(h, cb, dim=0))
            out[i] = int((a if sa > sb else b) == labels[i])
    return out


def ko_add_mask(am, Q, keys):
    Kt = torch.zeros_like(am)
    Kt[:, :, torch.tensor(Q, device=am.device)[:, None], torch.tensor(keys, device=am.device)[None, :]] = \
        torch.finfo(am.dtype).min
    return Kt


# ------------------------------------------------------------------ site sets
def v1_sites(K, Lb):
    out = {}
    for j in range(1, K + 1):
        out["L0@%d" % j] = (j, 0)
        out["Lmid@%d" % j] = (j, Lb // 2)
        out["Llast@%d" % j] = (j, Lb - 1)
    return out


def gset(kind, K, Lb):
    """Answer-site set (PLAN A7, review F2): loops 1..K-1 at every layer ('full') or at 5 depths ('red'); in the
    last loop only L0@K and Lmid@K. The readout-feeding tensor (K, Lb-1) and every other last-loop layer are out."""
    ls = list(range(Lb)) if kind == "full" else [0, Lb // 4, Lb // 2, (3 * Lb) // 4, Lb - 1]
    return [(j, l) for j in range(1, K) for l in ls] + [(K, 0), (K, Lb // 2)]


def dep_val(m_ko, m_a):
    """Read-window dependence (V4) = (m_KO - m_A) / (-m_A); None when -m_A < GAP (review MS6 guard)."""
    if m_ko is None or m_a is None or -m_a < GAP:
        return None
    return (m_ko - m_a) / (-m_a)


RKO_POOL = "positions 1 .. min(src) - 1 (before the span; v6, review B-F1)"


def rand_ko_keys(pair_id, src, r):
    """Equal-size random non-span keys for the relay control (V4, F11). v6 (review B-F1): drawn only before the span,
    so every relay query (after the span) can see each key, as it sees the span; the v5 pool range(1, r) also held
    chain positions after the span, hidden only from the few queries after them."""
    pool = [k for k in range(1, min(src)) if k < r]
    return sorted(random.Random(int(sha(pair_id + "|ko_rand")[:8], 16)).sample(pool, min(len(src), len(pool))))


def st1e(A, B, pos):
    """v5 stratum ST1e (plan critic 3 F7): number of source positions whose A and B tokens differ, '1' vs '>=2'. A
    shared leading token has identical causal context in A and B, so O1 and RND edits there are exact no-ops."""
    n = sum(1 for p in pos if p < len(A) and p < len(B) and A[p] != B[p])
    return "1" if n <= 1 else ">=2"


def d1_split(rel, subj):
    """D1 split by subject (review MS3/F6): sha256 of the subject's cluster key, even first hex digit -> dev."""
    return "dev" if int(sha("k17vb-d1:%s:%s" % (rel, subj))[0], 16) % 2 == 0 else "test"


def pick_pair(rows):
    """One pair per problem (review F8): lowest var_name in Python string order (prod DESIGN 12.3 rule)."""
    return min(rows, key=lambda v: (v["var_name"], v["pair_id"]))


def rank_cuts(vals_):
    """Tertile cuts by rank (review F5): values <= cut0 -> 0, <= cut1 -> 1, else 2."""
    s = sorted(vals_)
    return [s[len(s) // 3 - 1], s[2 * len(s) // 3 - 1]]


def per_model(spec, base, cast=str):
    """CLI override (review F9): '' keeps base; 'v' sets every model; 'name=v,name=v' sets the named models."""
    out = dict(base)
    if not spec:
        return out
    if "=" not in spec:
        return {k: cast(spec) for k in out}
    for kv in spec.split(","):
        k, v = kv.split("=", 1)
        if k not in out:
            raise KeyError(k)
        out[k] = cast(v)
    return out


def role_of(models):
    """Host role from the model list (A7: no host names in checks.json)."""
    ms = models if isinstance(models, (list, tuple)) else [x for x in str(models).split(",") if x]
    return "ifm_host" if any(m in IFM_MODELS for m in ms) else "ouro_host"


HOME_RE = re.compile(r"(/home/|/Users/|[A-Za-z]:[\\/]+Users[\\/]+)[^/\\\s\"']+")


def scrub(obj, hosts):
    """Replace host names (hosts: {name: role}) and user home prefixes in every string of obj (A7)."""
    if isinstance(obj, dict):
        return {scrub(k, hosts): scrub(v, hosts) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, hosts) for v in obj]
    if isinstance(obj, str):
        s = HOME_RE.sub("~", obj)
        for h, rl in hosts.items():
            if h:
                s = s.replace(h, rl)
        return s
    return obj


def order_of(site, Lb):
    return (site[0] - 1) * Lb + site[1]


def calls_after(site, K, Lb):
    """Attention calls (j, l) executed after the site tensor; None = every call (full knockout)."""
    o = order_of(site, Lb) if site is not None else -1
    return {(j, l) for j in range(1, K + 1) for l in range(Lb) if order_of((j, l), Lb) > o}


def r_state(c, s, r):
    """State at r from a clean record stored as all positions or as [r]."""
    h = c["H"][s]
    return h[0] if h.shape[0] == 1 else h[r]


# ------------------------------------------------------------------ fp32 readout (review MS2/F2, plan section 2)
class Fp32MM(object):
    """Exact fp32 matmul for the readout: TF32 off inside (k10_ifm.load_ifm turns it on), previous value restored."""
    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        return self

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev
        return False


def readout32(h, Wf):
    """Logits in float32 from a bf16 state: F.linear(h.float(), W.float()) with TF32 off."""
    with Fp32MM():
        return F.linear(h.float(), Wf)


def fp32_readout_selftest():
    g = torch.Generator().manual_seed(1)
    h = (torch.randn(4, 64, generator=g) * 3).to(torch.bfloat16).requires_grad_(True)
    W = torch.randn(300, 64, generator=g).to(torch.bfloat16)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    z32 = readout32(h, W.float())
    restored = torch.backends.cuda.matmul.allow_tf32 is True
    torch.backends.cuda.matmul.allow_tf32 = prev
    zbf = F.linear(h, W).float()
    ref = h.double() @ W.double().T
    z32.sum().backward()
    out = {"dtype": str(z32.dtype), "fp32_vs_float64_max": r5((z32.detach().double() - ref.detach()).abs().max()),
           "bf16_vs_float64_max": r5((zbf.detach().double() - ref.detach()).abs().max()),
           "grad_flows": h.grad is not None and bool(h.grad.abs().sum() > 0), "tf32_restored": restored}
    out["ok"] = bool(z32.dtype == torch.float32 and out["fp32_vs_float64_max"] < 1e-3 and restored
                     and out["bf16_vs_float64_max"] > out["fp32_vs_float64_max"] and out["grad_flows"])
    return out


# ------------------------------------------------------------------ selftest
def selftest():
    res = {}
    dev = "cpu"
    torch.manual_seed(0)
    o = torch.randn(1, 6, 16).to(torch.bfloat16)
    t = torch.randn(2, 16)
    e = S2.apply_entries(o, [e_lerp([1, 4], t, 1.0, dev)])
    res["lerp_a1_exact"] = bool(torch.equal(e[0, [1, 4]], t.to(o.dtype)) and torch.equal(e[0, [0, 2, 3, 5]], o[0, [0, 2, 3, 5]]))
    e2 = S2.apply_entries(o, [e_lerp([2], o[0, [2]].float() + 1.0, 2.0, dev)])
    res["lerp_a2_extrapolates"] = abs(float((e2[0, 2].float() - o[0, 2].float()).mean()) - 2.0) < 0.05
    res["zero_add_identity"] = bool(torch.equal(S2.apply_entries(o, [e_add(list(range(6)), torch.zeros(6, 16), dev)]), o))
    d = rnd_delta("p", (1, 3), "src", [5, 6], [2.0, 3.0], 16)
    res["rnd_norms"] = [r5(x) for x in d.norm(dim=-1)]
    res["rnd_deterministic"] = bool(torch.equal(d, rnd_delta("p", (1, 3), "src", [5, 6], [2.0, 3.0], 16)))
    res["rnd_differs_by_site"] = not torch.equal(d, rnd_delta("p", (1, 4), "src", [5, 6], [2.0, 3.0], 16))
    mA, mB = torch.zeros(8), torch.ones(8)
    res["proj"] = [proj(mA, mA, mB), proj(mB, mA, mB), proj((mA + mB) / 2, mA, mB), proj(mA + 1, mA + 1, mA + 1)]
    b = boot([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"])
    res["boot"] = b
    res["boot_ok"] = (abs(b[0] - 4.0) < 1e-9 and b[3] == 4 and b[4] == 3
                      and boot([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"]) == b)
    tz = tost([0.01, -0.02, 0.0, 0.015, -0.01] * 8, list(range(40)))
    tb = tost([0.2, 0.1, 0.15, 0.12, 0.18] * 8, list(range(40)))
    res["tost"] = [tz["null_by_tost"], tb["null_by_tost"], tb["label"]]
    mb = max_band({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 0.0, 1.0, 1.0]}, ["w", "x", "y", "z"])
    res["max_band_ok"] = mb is not None and all(v[1] <= v[0] <= v[2] for v in mb.values())
    res["spearman"] = [spearman([1, 2, 3, 4], [10, 20, 30, 40]), spearman([1, 2, 3, 4], [4, 3, 2, 1])]
    G = {"%d.0" % i: float(i) for i in range(19)}
    ks, ver = verify_sites(G)
    res["verify_sites"] = ver
    res["verify_ok"] = ver == ["18.0", "17.0", "16.0", "10.0", "9.0", "8.0", "2.0", "1.0", "0.0"]
    Gv = {k: G[k] for k in ver}
    h = hit_metrics(Gv, dict(Gv), 10.0, 2.0)
    dmt = dict(Gv)
    dmt["17.0"] = dmt["18.0"]
    ht = hit_metrics(Gv, dmt, 10.0, 2.0)
    hz = hit_metrics(Gv, {k: 0.001 for k in ver}, 10.0, 2.0)
    dmr = dict(Gv)
    dmr["18.0"] = 5.0
    hr = hit_metrics(Gv, dmr, 10.0, 2.0)
    hd = hit_metrics(Gv, dict(Gv), 1.5, 0.5)                      # D < GAP: not eligible
    res["hit"] = [h["hit"], round(h["chance"], 4), h["regret"], round(h["top3_hit"], 4), ht["hit"], round(ht["chance"], 4),
                  ht["strict_hit"], round(ht["top3_hit"], 4), hz["eligible"], hz["hit"], hr["hit"], hr["regret"],
                  hd["eligible"]]
    res["hit_ok"] = (h["hit"] == 1 and abs(h["chance"] - 1 / 9) < 1e-9 and h["regret"] == 0.0
                     and abs(h["top3_hit"] - 1 / 3) < 1e-9
                     and ht["hit"] == 1 and abs(ht["chance"] - 2 / 9) < 1e-9 and ht["strict_hit"] == 0
                     and abs(ht["top3_hit"] - 2 / 3) < 1e-9
                     and hz["eligible"] is False and hz["hit"] is None and hr["hit"] == 0
                     and abs(hr["regret"] - 1.2) < 1e-9 and hd["eligible"] is False)
    # dyadic bootstrap, stratum difference, read-window guard, D1 subject split, pair rule, cuts, CLI overrides, scrub
    b2 = boot2([1.0, 2.0, 3.0, 4.0], [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    res["boot2"] = b2
    res["boot2_ok"] = (abs(b2[0] - 2.5) < 1e-9 and b2[3] == 4 and b2[4] == 4 and b2[1] <= b2[0] <= b2[2]
                       and boot([1.0, 2.0, 3.0, 4.0], [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]) == b2)
    gd = boot_group_diff([1.0, 1.0, 3.0, 3.0], [0, 1, 2, 3], ["x", "x", "y", "y"], "x", "y")
    res["group_diff"] = gd
    res["group_diff_ok"] = gd[0] == -2.0 and gd[3] == 2 and gd[4] == 2 and gd[5] == 4
    res["dep_guard"] = [dep_val(0.0, -1.0), dep_val(0.0, -4.0), dep_val(None, -4.0)]
    res["dep_guard_ok"] = res["dep_guard"] == [None, 1.0, None]
    toy = [{"rel": "capital", "a": "S%d" % i, "b": "S%d" % ((i + d) % 12), "idx": 10 * i + d, "half": "x"}
           for i in range(12) for d in (1, 2, 3)]
    kept_d1, ncross = d1_assign(toy)
    subj = {s: {(p["rel"], p[w]) for p in kept_d1 if p["split"] == s for w in ("a", "b")} for s in ("dev", "test")}
    res["d1_split"] = [len(kept_d1), ncross, len(subj["dev"] & subj["test"])]
    res["d1_split_ok"] = (len(kept_d1) + ncross == len(toy) and not (subj["dev"] & subj["test"])
                          and all(p["half"] == ("dev" if p["split"] == "dev" else "heldout") for p in kept_d1))
    res["pick_pair_ok"] = pick_pair([{"var_name": "b", "pair_id": "g-1-b"}, {"var_name": "a", "pair_id": "g-1-a"}])["pair_id"] == "g-1-a"
    res["rank_cuts_ok"] = rank_cuts(list(range(9, 0, -1))) == [3, 6]
    pm = [per_model("full", {"a": "red", "b": "red"}), per_model("a=full", {"a": "red", "b": "red"}),
          per_model("30", {"a": 40, "b": 20}, int)]
    res["per_model_ok"] = pm == [{"a": "full", "b": "full"}, {"a": "full", "b": "red"}, {"a": 30, "b": 30}]
    sc_ = scrub({"node-a.example.org": "/home/user0/proj/x", "w": "C:\\Users\\user0\\.cache\\y"},
                {"node-a.example.org": "ouro_host"})
    res["scrub"] = sc_
    res["scrub_ok"] = sc_ == {"ouro_host": "~/proj/x", "w": "~\\.cache\\y"}
    # dev-tuned site by mean dm/D (review MS7): X = 0.9 D on D = 20 items and 0.1 D on D = 4 items, Y = 0.6 D on all
    def toy_item(D, xs):
        it_ = {"clean_A": [0.0, -D / 2.0], "clean_B": [0.0, D / 2.0], "rows": {}}
        for k_, x_ in xs.items():
            it_["rows"]["O1|ans|%s|1" % k_] = {"f": [0.0, -D / 2.0 + x_ * D]}
        return it_
    tdev = [toy_item(20.0, {"1.0": 0.9, "1.1": 0.6}), toy_item(4.0, {"1.0": 0.1, "1.1": 0.6})]
    res["dev_tuned_ok"] = dev_tuned_site(tdev, [(1, 0), (1, 1)])[0] == "1.1"
    res["window_rule"] = [window_rule_v4(1.0, 0.3, 0.5, None, 0)["inside"], window_rule_v4(1.0, 0.1, 0.5, 0.5, 20)["inside"],
                          window_rule_v4(1.0, 0.3, 0.1, 0.25, 12)["inside"], window_rule_v4(1.0, 0.3, 0.1, 0.25, 9)["inside"]]
    res["sets"] = [len(gset("full", 4, 24)), len(gset("red", 4, 24)), len(gset("red", 4, 48)), len(gset("red", 4, 28)),
                   len(gset("full", 4, 48)), (4, 23) in gset("full", 4, 24), (4, 12) in gset("red", 4, 24),
                   (4, 14) in gset("red", 4, 28), sorted(set(l for _, l in gset("red", 4, 48))),
                   [sorted(l for j, l in gset(kd, 4, lb) if j == 4) for kd, lb in (("full", 24), ("red", 28), ("full", 48))]]
    res["sets_ok"] = res["sets"] == [74, 17, 17, 17, 146, False, True, True, [0, 12, 24, 36, 47],
                                     [[0, 12], [0, 14], [0, 24]]]
    res["calls_after"] = [len(calls_after((1, 23), 4, 24)), len(calls_after((4, 12), 4, 24)), len(calls_after(None, 4, 24)),
                          len(calls_after((4, 14), 4, 28))]
    res["calls_after_ok"] = res["calls_after"] == [72, 11, 96, 13]
    res["first_iso"] = [first_iso("a 16 b 160 c", 16), first_iso("x 160 16", 16), first_iso("1.16 and 16.5", 16),
                        first_iso("no", 3)]
    res["first_iso_ok"] = res["first_iso"] == [2, 6, None, None]
    res["contradictory"] = [contradictory_prefix("She has 1,000 apples", 1000),
                            contradictory_prefix("He pays <<3*4=12>>12 dollars", 5),
                            contradictory_prefix("He pays <<10/2=5>>", 7)]
    res["contradictory_ok"] = res["contradictory"] == [True, False, False]
    res["ambiguous"] = [ambiguous_source("after 3 weeks she had 3 cups", [6, 7], 3),
                        ambiguous_source("after 3 weeks she had 4 cups", [6, 7], 3)]
    res["ambiguous_ok"] = res["ambiguous"] == [True, False]
    items = [{"g": "a", "i": 0}, {"g": "a", "i": 1}, {"g": "a", "i": 2}, {"g": "b", "i": 3}]
    res["rr_order"] = [x["i"] for x in rr_order(items, lambda x: x["g"])]
    res["rr_ok"] = res["rr_order"] == [0, 3, 1, 2]
    T = 7
    sc = torch.randn(1, 2, T, T).to(torch.bfloat16)
    causal = torch.full((T, T), torch.finfo(torch.bfloat16).min).triu(1).to(torch.bfloat16)[None, None]
    Kt = ko_add_mask(causal, [4, 5, 6], [1, 2])
    p0 = torch.softmax((sc + causal).float(), -1)
    p1 = torch.softmax((sc + causal + Kt).float(), -1)
    res["toy_relay_zero"] = bool((p1[0, :, 4:, [1, 2]] == 0).all()) and bool(torch.equal(p1[0, :, :4], p0[0, :, :4]))
    res["toy_zero_identity"] = bool(torch.equal(torch.softmax((sc + (causal + torch.zeros_like(causal))).float(), -1), p0))
    res["v4"] = v4_selftest()
    res["v5"] = v5_selftest()
    res["v6"] = v6_selftest()
    res["fp32_readout"] = fp32_readout_selftest()
    pc = predictions_coverage_selftest()
    res["predictions_coverage"] = pc
    ok = all(res[k] for k in ("lerp_a1_exact", "lerp_a2_extrapolates", "zero_add_identity", "rnd_deterministic",
                              "rnd_differs_by_site", "boot_ok", "max_band_ok", "verify_ok", "hit_ok", "sets_ok",
                              "calls_after_ok", "first_iso_ok", "contradictory_ok", "ambiguous_ok", "rr_ok",
                              "toy_relay_zero", "toy_zero_identity", "boot2_ok", "group_diff_ok", "dep_guard_ok",
                              "d1_split_ok", "pick_pair_ok", "rank_cuts_ok", "per_model_ok", "scrub_ok",
                              "dev_tuned_ok"))
    ok = ok and res["fp32_readout"]["ok"] and pc["ok"] and res["v4"]["ok"] and res["v5"]["ok"] and res["v6"]["ok"]
    ok = ok and all(abs(a - b) < 1e-4 for a, b in zip(res["rnd_norms"], [2.0, 3.0]))
    ok = ok and res["proj"][:3] == [0.0, 1.0, 0.5] and res["proj"][3] is None
    ok = ok and res["tost"][0] is True and res["tost"][1] is False and res["spearman"] == [1.0, -1.0]
    ok = ok and res["window_rule"] == [True, False, True, False]
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


# ------------------------------------------------------------------ build (laptop CPU)
def build_items(out_dir):
    import pandas as pd
    from tokenizers import Tokenizer
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(x) for x in open(PAIRS, encoding="utf-8")]
    core = [r for r in rows if r["split"] == "eval" and r["text_exact"]
            and all(r["tok"][t]["outside_equal"] for t in ("ouro", "huginn", "mcleish", "ifm"))]
    test = pd.read_parquet(os.path.join(GSM_DIR, "test-00000-of-00001.parquet"))
    train = pd.read_parquet(os.path.join(GSM_DIR, "train-00000-of-00001.parquet"))
    block = open(PROMPT_FILE, encoding="utf-8").read()

    def fmt(i):
        return "Question: %s\nAnswer: %s\n\n" % (train.iloc[i]["question"], train.iloc[i]["answer"])
    assert "".join(fmt(i) for i in range(4)) == block, "SETTLE block != gsm8k train rows 0-3"
    block_alt = "".join(fmt(i) for i in range(4, 8))
    tk = {"ouro": Tokenizer.from_file(TOK_OURO), "ifm": Tokenizer.from_file(TOK_IFM)}
    assert tk["ifm"].token_to_id("<|begin_of_text|>") == IFM_BOS

    def enc(name, text):
        ids = tk[name].encode(text, add_special_tokens=False).ids
        return ([IFM_BOS] + ids) if name == "ifm" else ids
    stats = {"core_rows": len(core), "core_problems": len({r["idx"] for r in core})}
    cnt = {"no_restatement": 0, "in_annotation": 0, "contradictory_prefix": 0, "x_word_form": 0,
           "prose_unverified_number": 0, "ambiguous_source": 0, "assert_fail": {"ouro": 0, "ifm": 0},
           "dropped_ids": {"contradictory_prefix": [], "x_word_form": [], "prose_unverified_number": [],
                           "ambiguous_source": []},
           "prefix_rule": "v6 (review C-F3): contradictory prefix (x in another digit format or a tainted annotation), "
                          "then x as a number word, then a prose number (digits or words) that is not a question number "
                          "or an earlier annotation result; each row counted under its first reason"}
    def block_ctx(n, b):
        """Block ids as tokenized in context: Ouro merges the trailing '\\n\\n' into one token when the block
        stands alone but splits it before 'Question'; take the tokens whose characters lie inside the block."""
        e = tk[n].encode(b + "Question: x", add_special_tokens=False)
        k = next(i for i, (s, _) in enumerate(e.offsets) if s >= len(b))
        return ([IFM_BOS] + e.ids[:k]) if n == "ifm" else e.ids[:k]
    blk = {n: block_ctx(n, block) for n in tk}
    blk_alt = {n: block_ctx(n, block_alt) for n in tk}
    stats["block_ctx_len"] = {n: len(blk[n]) for n in tk}
    stats["block_alone_len"] = {n: len(enc(n, block)) for n in tk}
    valid = []
    for r in core:
        sol = test.iloc[r["idx"]]["answer"]
        x, xb = r["value_A"], r["value_B"]
        c = first_iso(sol, x)
        if c is None:
            cnt["no_restatement"] += 1
            continue
        if in_annot(sol, c):
            cnt["in_annotation"] += 1
            continue
        C = sol[:c]
        if contradictory_prefix(C, x):
            cnt["contradictory_prefix"] += 1
            cnt["dropped_ids"]["contradictory_prefix"].append(r["pair_id"])
            continue
        pf = prose_flags(C, r["question_A"], x)            # v6 (review C-F3)
        if pf:
            cnt[pf[0]] += 1
            cnt["dropped_ids"][pf[0]].append(r["pair_id"])
            continue
        if ambiguous_source(r["question_A"], r["char_span_A"], x):
            cnt["ambiguous_source"] += 1
            cnt["dropped_ids"]["ambiguous_source"].append(r["pair_id"])
            continue
        rec = {"pair_id": r["pair_id"], "idx": r["idx"], "var_name": r["var_name"], "value_A": x, "value_B": xb,
               "n_digits": r["n_digits"],
               "first_digit_shared": str(x)[0] == str(xb)[0], "gold_first_digit_differs": r["first_digit_differs"],
               "q_A": r["question_A"], "q_B": r["question_B"], "C": C, "tok": {}}
        ok = True
        for n in tk:
            pa = block + "Question: " + r["question_A"] + "\nAnswer:"
            pb = block + "Question: " + r["question_B"] + "\nAnswer:"
            ipa = enc(n, pa)
            fa = enc(n, pa + " " + C + str(x))
            fb = enc(n, pb + " " + C + str(xb))
            s, e = r["tok"][n]["span_A"]
            good = (len(ipa) == r["tok"][n]["len_A"] and fa[:len(ipa)] == ipa and len(fa) == len(fb)
                    and fa[:len(blk[n])] == blk[n])
            ts = None
            if good:
                diffs = [i for i in range(len(fa)) if fa[i] != fb[i]]
                after = [i for i in diffs if i >= len(ipa)]
                bad = [i for i in diffs if i < len(ipa) and not (s <= i < e)]
                good = bool(after) and not bad
                if good:
                    ts = after[0]
                    good = fa[ts] != fb[ts] and ts > len(ipa)
            if good:
                r2a = enc(n, block_alt + pa[len(block):] + " " + C + str(x))
                r2b = enc(n, block_alt + pb[len(block):] + " " + C + str(xb))
                good = r2a == blk_alt[n] + fa[len(blk[n]):] and r2b == blk_alt[n] + fb[len(blk[n]):]
            if not good:
                cnt["assert_fail"][n] += 1
                ok = False
                break
            rec["tok"][n] = {"span": [s, e], "plen": len(ipa), "full_A": fa, "full_B": fb, "tstar": ts,
                             "n_rest": len(fa) - ts, "n_chain": ts - len(ipa),
                             "r2_off": len(blk_alt[n]) - len(blk[n])}
        if ok:
            valid.append(rec)
    by = {}
    for v in valid:
        by.setdefault(v["idx"], []).append(v)
    core_by = {}
    for r in core:
        core_by.setdefault(r["idx"], []).append(r)
    gen_pick = {i: pick_pair(rs)["pair_id"] for i, rs in core_by.items()}   # GEN arm (a): DESIGN 12.3 on the core pool
    sel = {i: pick_pair(vs) for i, vs in by.items()}                        # v3 (review F8): same rule on valid rows
    order = sorted(sel, key=lambda i: sha("k17vb-order:%d" % i))
    rank = {i: k for k, i in enumerate(order)}
    for v in valid:
        v["selected"] = sel[v["idx"]]["pair_id"] == v["pair_id"]
        v["gen_pair_id"] = gen_pick[v["idx"]]
        v["split"] = "dev" if int(sha("k17vb-split:%d" % v["idx"])[0], 16) % 2 == 0 else "test"
    stats.update(cnt)
    stats.update({"valid_rows": len(valid), "valid_problems": len(by), "s200": [sel[i]["pair_id"] for i in order[:200]],
                  "pair_rule": "lowest var_name among valid rows (prod DESIGN 12.3)",
                  "pair_equals_gen_all": sum(1 for i in order if sel[i]["pair_id"] == gen_pick[i]),
                  "pair_equals_gen_s200": sum(1 for i in order[:200] if sel[i]["pair_id"] == gen_pick[i])})
    for n in tk:
        items = []
        for v in sorted(valid, key=lambda v: (rank[v["idx"]], not v["selected"], v["pair_id"])):
            it = {k: v[k] for k in ("pair_id", "idx", "var_name", "value_A", "value_B", "n_digits", "first_digit_shared",
                                    "gold_first_digit_differs", "q_A", "q_B", "C", "selected", "split", "gen_pair_id")}
            it.update(v["tok"][n])
            it["order"] = rank[v["idx"]]
            items.append(it)
        sel_items = [it for it in items if it["selected"]]
        cuts = rank_cuts([it["plen"] for it in sel_items[:200]])            # v3 (review F5): by rank on plen, both tokenizers
        meta = {"tokenizer": n, "block_text": block, "block_ids": blk[n], "block_alt_ids": blk_alt[n],
                "len_tertile_cuts": cuts, "bos": IFM_BOS if n == "ifm" else None,
                "source": {"pairs": PAIRS, "pairs_sha256": hashlib.sha256(open(PAIRS, "rb").read()).hexdigest(),
                           "prompt": PROMPT_FILE, "gsm8k": GSM_DIR, "tokenizer_file": TOK_OURO if n == "ouro" else TOK_IFM}}
        s200 = sel_items[:200]
        meta["s200_strata"] = {
            "span_1_vs_ge2": [sum(1 for it in s200 if it["span"][1] - it["span"][0] == 1),
                              sum(1 for it in s200 if it["span"][1] - it["span"][0] >= 2)],
            "first_digit_shared": sum(1 for it in s200 if it["first_digit_shared"]),
            "digits_1_2_ge3": [sum(1 for it in s200 if it["n_digits"] == 1), sum(1 for it in s200 if it["n_digits"] == 2),
                               sum(1 for it in s200 if it["n_digits"] >= 3)],
            "len_tertile": [sum(1 for it in s200 if tertile(it["plen"], cuts) == t) for t in (0, 1, 2)],
            "cont": sum(1 for it in s200 if it["n_rest"] >= 2), "dev": sum(1 for it in s200 if it["split"] == "dev"),
            "plen_median": sorted(it["plen"] for it in s200)[len(s200) // 2], "chain_tok_median": sorted(it["n_chain"] for it in s200)[len(s200) // 2]}
        json.dump({"meta": meta, "stats": stats, "items": items}, open(os.path.join(out_dir, "items_%s.json" % n), "w"))
        print(n, json.dumps(meta["s200_strata"]), flush=True)
    excl = k16_exclusions()
    json.dump(excl, open(os.path.join(out_dir, "k16_exclusions.json"), "w"), indent=1)
    for n in ("ouro", "ifm"):
        with open(os.path.join(D4_DIR, "d4_%s.jsonl" % n), "rb") as f_in, \
                open(os.path.join(out_dir, "d4_%s.jsonl" % n), "wb") as f_out:
            f_out.write(f_in.read())
    print(json.dumps({k: v for k, v in stats.items() if k not in ("s200", "dropped_ids")}), flush=True)
    print("dropped", json.dumps(cnt["dropped_ids"]), flush=True)
    print("excl", {k: len(v) for k, v in excl.items()}, flush=True)
    build_vbd(out_dir)
    return stats


# ------------------------------------------------------------------ v5 derived-use target (plan critic 3 F5)
STEP_RE = re.compile(r"<<([^=>]*)=([^>]*)>>")


def safe_eval(expr):
    """Arithmetic only (+ - * / // % ** and parentheses on numbers); None if anything else appears."""
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
           ast.Mod: op.mod, ast.Pow: op.pow}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 8:
                raise ValueError("pow")
            return ops[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        raise ValueError(type(n).__name__)
    try:
        return ev(ast.parse(expr.replace(",", "").strip(), mode="eval"))
    except Exception:  # noqa: BLE001
        return None


def fmt_num(v):
    """GSM8K-style number text: integer without a decimal point, else at most 4 decimals with zeros stripped."""
    if v is None:
        return None
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = ("%.4f" % v).rstrip("0").rstrip(".")
    return s


def iso_all(s, x):
    """Every isolated match of str(x) in s (same isolation rule as first_iso)."""
    out, t = [], str(x)
    for mt in re.finditer(re.escape(t), s):
        a, b = mt.start(), mt.end()
        if a > 0 and (s[a - 1].isdigit() or (s[a - 1] in ".," and a > 1 and s[a - 2].isdigit())):
            continue
        if b < len(s) and (s[b].isdigit() or (s[b] in ".," and b + 1 < len(s) and s[b + 1].isdigit())):
            continue
        out.append(a)
    return out


def vbd_text(sol, q, span, x, xb):
    """Derived-use text (plan critic 3 F5, LEDGER FINAL FRAME D2). First calculator step of A's gold whose inputs
    include x; the shared prefix is the gold chain up to that step's '<<expr=' with every isolated x rendered as each
    side's own value (A: x, B: x'), so each side's prefix is self-consistent and holds no x-derived result. Returns
    (record, None) or (None, drop reason)."""
    xv = float(x)
    if len(str(x)) != len(str(xb)):
        return None, "unequal_length"
    first, earlier = None, []
    for mt in STEP_RE.finditer(sol):
        ins = [numval(t) for t in NUM.findall(mt.group(1))]
        if any(v == xv for v in ins if v is not None):
            first = mt
            break
        earlier.append(numval(mt.group(2).strip().rstrip(".")))
    if first is None:
        return None, "no_derived_step"
    if any(v == xv for v in earlier):
        return None, "coincident_value"
    C = sol[:first.start(2)]
    if any(numval(t.group()) == xv and t.group() != str(x) for t in NUM.finditer(C)):
        return None, "x_other_format"
    pf = prose_flags(C, q, x, extra_ok=[v for v in earlier if v is not None])   # v6 (review C-F3): number words too
    if "x_word_form" in pf:
        return None, "x_other_format"
    if "prose_unverified_number" in pf:
        return None, "prose_unverified_number"
    if ambiguous_source(q, span, x):
        return None, "ambiguous_source"
    occ = iso_all(C, x)
    CB = C
    for a in reversed(occ):
        CB = CB[:a] + str(xb) + CB[a + len(str(x)):]
    expr_A = first.group(1)
    expr_B = expr_A
    for a in reversed(iso_all(expr_A, x)):
        expr_B = expr_B[:a] + str(xb) + expr_B[a + len(str(x)):]
    rA_text = first.group(2).strip()
    res_A, eA, res_B = numval(rA_text.rstrip(".")), safe_eval(expr_A), safe_eval(expr_B)
    if res_A is None or eA is None or abs(eA - res_A) > 1e-6 * max(1.0, abs(res_A)):
        return None, "annotation_mismatch"
    if fmt_num(res_A) != rA_text:
        return None, "result_format"
    if res_B is None or abs(res_B - res_A) < 1e-9:
        return None, "result_unchanged"
    return {"C_A": C, "C_B": CB, "occ": occ, "expr_A": expr_A, "expr_B": expr_B, "res_A": rA_text,
            "res_B": fmt_num(res_B)}, None


def build_vbd(out_dir):
    """v5 (plan critic 3 F5): derived-use VarBench items -> items_vbd_<tok>.json (laptop CPU, tokenizers only)."""
    import pandas as pd
    from tokenizers import Tokenizer
    rows = [json.loads(x) for x in open(PAIRS, encoding="utf-8")]
    core = [r for r in rows if r["split"] == "eval" and r["text_exact"]
            and all(r["tok"][t]["outside_equal"] for t in ("ouro", "huginn", "mcleish", "ifm"))]
    test = pd.read_parquet(os.path.join(GSM_DIR, "test-00000-of-00001.parquet"))
    train = pd.read_parquet(os.path.join(GSM_DIR, "train-00000-of-00001.parquet"))
    block = open(PROMPT_FILE, encoding="utf-8").read()

    def fmt(i):
        return "Question: %s\nAnswer: %s\n\n" % (train.iloc[i]["question"], train.iloc[i]["answer"])
    block_alt = "".join(fmt(i) for i in range(4, 8))
    tk = {"ouro": Tokenizer.from_file(TOK_OURO), "ifm": Tokenizer.from_file(TOK_IFM)}

    def enc_o(name, text):
        e = tk[name].encode(text, add_special_tokens=False)
        if name == "ifm":
            return [IFM_BOS] + e.ids, [(0, 0)] + list(e.offsets)
        return list(e.ids), list(e.offsets)

    def block_ctx(n, b):
        ids, offs = enc_o(n, b + "Question: x")
        k = next(i for i, (s, _) in enumerate(offs) if s >= len(b) and (n != "ifm" or i > 0))
        return ids[:k]
    blk = {n: block_ctx(n, block) for n in tk}
    blk_alt = {n: block_ctx(n, block_alt) for n in tk}
    drops, valid = {}, []
    for r in core:
        sol = test.iloc[r["idx"]]["answer"]
        rec, why = vbd_text(sol, r["question_A"], r["char_span_A"], r["value_A"], r["value_B"])
        if rec is None:
            drops[why] = drops.get(why, 0) + 1
            continue
        v = {"pair_id": r["pair_id"], "idx": r["idx"], "var_name": r["var_name"], "value_A": r["value_A"],
             "value_B": r["value_B"], "n_digits": r["n_digits"],
             "first_digit_shared": str(r["value_A"])[0] == str(r["value_B"])[0],
             "gold_first_digit_differs": r["first_digit_differs"], "q_A": r["question_A"], "q_B": r["question_B"],
             "tok": {}}
        v.update(rec)
        ok = True
        for n in tk:
            pa = block + "Question: " + r["question_A"] + "\nAnswer:"
            pb = block + "Question: " + r["question_B"] + "\nAnswer:"
            ipa, _ = enc_o(n, pa)
            preA, _ = enc_o(n, pa + " " + rec["C_A"])
            preB, _ = enc_o(n, pb + " " + rec["C_B"])
            fa, offa = enc_o(n, pa + " " + rec["C_A"] + rec["res_A"])
            fb, _ = enc_o(n, pb + " " + rec["C_B"] + rec["res_B"])
            s, e = r["tok"][n]["span_A"]
            good = (len(ipa) == r["tok"][n]["len_A"] and fa[:len(ipa)] == ipa and fa[:len(preA)] == preA
                    and fb[:len(preB)] == preB and len(preA) == len(preB) and fa[:len(blk[n])] == blk[n])
            ts, chain = None, []
            if good:
                base = len(pa) + 1
                rng = [(base + a, base + a + len(str(r["value_A"]))) for a in rec["occ"]]
                chain = [i for i, (cs, ce) in enumerate(offa) if i >= len(ipa) and i < len(preA)
                         and any(cs < b and ce > a for a, b in rng)]
                srcset = set(range(s, e)) | set(chain)
                m = min(len(fa), len(fb))
                diffs = [i for i in range(m) if fa[i] != fb[i]]
                good = (bool(diffs) and all(i in srcset for i in diffs if i < len(preA))
                        and any(i >= len(preA) for i in diffs) and len(chain) >= 1)
                if good:
                    ts = next(i for i in diffs if i >= len(preA))
            if not good:
                drops["tokenizer_assert_" + n] = drops.get("tokenizer_assert_" + n, 0) + 1
                ok = False
                break
            v["tok"][n] = {"span": [s, e], "plen": len(ipa), "full_A": fa[:ts + 1], "full_B": fb[:ts + 1], "tstar": ts,
                           "n_rest": 1, "n_chain": ts - len(ipa), "r2_off": len(blk_alt[n]) - len(blk[n]),
                           "src_extra": chain, "text_tail_A": " " + rec["C_A"] + rec["res_A"],
                           "text_tail_B": " " + rec["C_B"] + rec["res_B"]}
        if ok:
            valid.append(v)
    by = {}
    for v in valid:
        by.setdefault(v["idx"], []).append(v)
    sel = {i: pick_pair(vs) for i, vs in by.items()}
    order = sorted(sel, key=lambda i: sha("k17vb-order:%d" % i))
    rank = {i: k for k, i in enumerate(order)}
    stats = {"core_rows": len(core), "drops": drops, "valid_rows": len(valid), "valid_problems": len(by),
             "rule": "first calculator step whose inputs include x; prefix = gold up to '<<expr='; every isolated x "
                     "rendered with each side's own value; read = first differing token of the step result"}
    for n in tk:
        items = []
        for v in sorted(valid, key=lambda v: (rank[v["idx"]], v["pair_id"])):
            it = {k: v[k] for k in ("pair_id", "idx", "var_name", "value_A", "value_B", "n_digits", "first_digit_shared",
                                    "gold_first_digit_differs", "q_A", "q_B", "res_A", "res_B", "expr_A", "expr_B")}
            it.update(v["tok"][n])
            it["selected"] = sel[v["idx"]]["pair_id"] == v["pair_id"]
            it["split"] = "dev" if int(sha("k17vb-split:%d" % v["idx"])[0], 16) % 2 == 0 else "test"
            it["order"] = rank[v["idx"]]
            it["kind"] = "derived"
            items.append(it)
        si = [it for it in items if it["selected"]]
        cuts = rank_cuts([it["plen"] for it in si[:200]]) if len(si) >= 3 else [0, 0]
        meta = {"tokenizer": n, "block_text": block, "block_ids": blk[n], "block_alt_ids": blk_alt[n],
                "len_tertile_cuts": cuts, "bos": IFM_BOS if n == "ifm" else None, "target": "derived-use (v5, F5)"}
        stats["selected_%s" % n] = len(si)
        stats["chain_positions_median_%s" % n] = sorted(len(it["src_extra"]) for it in si)[len(si) // 2] if si else None
        json.dump({"meta": meta, "stats": stats, "items": items}, open(os.path.join(out_dir, "items_vbd_%s.json" % n), "w"))
    print("vbd", json.dumps(stats), flush=True)
    return stats


def d4_content(r, side="A"):
    """D4 content key (P10(5)): kind | vals of one side | index of src_var | indices of query_vars (letters
    ignored). side 'A' is the v3 key; side 'B' is the B-side problem (plan critic 2 F8)."""
    idx_src = r["names"].index(r["src_var"])
    idx_q = [r["names"].index(q) for q in r["query_vars"]]
    vs = r["vals_A"] if side == "A" else r["vals_B"]
    return sha("%s|%s|%d|%s" % (r["kind"], ",".join(str(v) for v in vs), idx_src,
                                ",".join(str(i) for i in idx_q)))[:16]


def d4_components(rows):
    """v4 D4 clusters (plan critic 2 F8, C16): connected components of the graph whose nodes are content keys and
    whose edges are rows (A-side key - B-side key). -> {key: component id (smallest key in the component)}."""
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for r in rows:
        a, b = find(d4_content(r, "A")), find(d4_content(r, "B"))
        if a != b:
            par[max(a, b)] = min(a, b)
    return {k: find(k) for k in list(par)}


def k16_exclusions():
    """D2 problems and D4 template ids / content clusters used by any K16 run (fresh A2 items, critic F9b)."""
    d2, d4ids, d4tpl = set(), set(), set()
    files = sorted(glob.glob(os.path.join(K16_ART, "k16_path_ouro", "k16p_*.json"))
                   + glob.glob(os.path.join(K16_ART, "k16_path_ifm", "k16_*.json"))
                   + glob.glob(os.path.join(K16_ART, "k16_path_ifm", "d45", "k16_*.json")))
    for f in files:
        d = json.load(open(f))
        for ds, D in d.get("datasets", {}).items():
            for it in D.get("items", []):
                if ds == "D2":
                    d2.add(it["a"])
                elif ds == "D4":
                    if it.get("id"):
                        d4ids.add(it["id"])
                        d4tpl.add(re.sub(r"-p\d+$", "", it["id"]))
                    if it.get("cluster"):
                        d4tpl.add(it["cluster"])
    rows = []
    for n in ("ouro", "ifm"):
        rows += [json.loads(x) for x in open(os.path.join(D4_DIR, "d4_%s.jsonl" % n), encoding="utf-8")]
    content = sorted({d4_content(r) for r in rows if r["cluster_id"] in d4tpl or r["id"] in d4ids})
    return {"files": [os.path.relpath(f, K16_ART).replace("\\", "/") for f in files], "D2": sorted(d2),
            "D4_ids": sorted(d4ids), "D4_templates": sorted(d4tpl), "D4_content": content}


# ------------------------------------------------------------------ runners
class OuroR(object):
    family = "ouro"

    def __init__(self, name, dev, fp32=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tokenizers import Tokenizer
        path = MODEL_DIR.get(name, OURO_REPOS[name])
        self.name, self.path, self.dev = name, path, dev
        self.adt = torch.float32 if fp32 else torch.bfloat16      # v5 (F4): fp32 only for the FD check
        rev = OURO_REVISION.get(path)                            # cluster copy: pinned hub revision
        self.m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=self.adt, revision=rev,
                                                      attn_implementation="eager").to(dev).eval()
        for prm in self.m.parameters():
            prm.requires_grad_(False)
        self.hf_tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, revision=rev)
        if os.path.isdir(path):
            tj = os.path.join(path, "tokenizer.json")
        else:
            from huggingface_hub import hf_hub_download
            tj = hf_hub_download(path, "tokenizer.json", revision=rev)
        self.tok = Tokenizer.from_file(tj)
        self.tok_file = tj
        self.tok_md5 = hashlib.md5(open(tj, "rb").read()).hexdigest()
        self.inner = self.m.model
        self.L = int(self.m.config.num_hidden_layers)
        self.K = int(self.inner.total_ut_steps)
        self.Lb = self.L
        self.Hd = int(self.m.config.hidden_size)
        self.W = self.m.lm_head.weight
        assert getattr(self.m.lm_head, "bias", None) is None
        self.Wf = self.W.detach().float()            # v3 readout in float32 (review MS2/F2)
        self.attn = [ly.self_attn for ly in self.inner.layers[:self.L]]
        self.n_fwd = 0

    def enc(self, t, ans=False):
        return self.hf_tok(t, add_special_tokens=False)["input_ids"]

    def enc_vb(self, t):
        return self.tok.encode(t, add_special_tokens=False).ids

    def run(self, ids, r, edits=None, rec=None, ko=None, leaves=None, cont=False, cap=None, grad=False,
            mask_log=None, bf16_ref=False):
        """One batch-1 forward. edits / rec / leaves keyed by site (j, l); ko {call (j, l): (Q, keys)};
        cap: calls whose module attention probabilities are returned."""
        T = len(ids)
        x = torch.tensor([ids], device=self.dev)
        edits, rec, leaves = edits or {}, rec or {}, leaves or {}
        L, K = self.L, self.K
        cnt, ncnt, H, hr, out = {}, [0], {}, {}, {}
        calls, probs, masks = {}, {}, {}

        def site(s, o):
            if s in leaves:
                o = o + leaves[s]
            if s in edits:
                o = S2.apply_entries(o, edits[s])
            if s in rec:
                H[s] = (o[0] if rec[s] == "all" else o[0, rec[s]]).detach().clone()
            return o

        def mk(l):
            def h(mod, i, o):
                c = cnt.get(l, 0) + 1
                cnt[l] = c
                return site((c, l), o)
            return h

        def norm_hook(mod, i, o):
            ncnt[0] += 1
            j = ncnt[0]
            o = site((j, L - 1), o)
            hr[j] = o[0, r]
            if j == K and cont and T > r + 1:
                out["cont_h"] = o[0, r + 1:]
            return o
        hs = [self.inner.layers[l].register_forward_hook(mk(l)) for l in range(L - 1)]
        hs.append(self.inner.norm.register_forward_hook(norm_hook))
        do_wrap = bool(ko) or bool(cap) or (mask_log is not None)
        if do_wrap:
            for l, att in enumerate(self.attn):
                def fwd(*args, _o=att.forward, _l=l, **kw):
                    key = (int(kw.get("current_ut", 0)) + 1, _l)
                    calls[key] = calls.get(key, 0) + 1
                    am = kw.get("attention_mask")
                    if mask_log is not None:
                        mask_log.append(None if am is None else [list(am.shape), str(am.dtype)])
                    if ko and key in ko:
                        spec = ko[key]
                        mk_ = (id(spec), tuple(am.shape))
                        if mk_ not in masks:
                            masks[mk_] = ko_add_mask(am, spec[0], spec[1])
                        kw["attention_mask"] = am + masks[mk_]
                    res = _o(*args, **kw)
                    if cap and key in cap:
                        probs[key] = res[1][0].detach().float().clone()
                    return res
                att.forward = fwd
        try:
            with torch.set_grad_enabled(grad):
                self.m(input_ids=x, use_cache=False)
                zl = [readout32(hr[j], self.Wf) for j in range(1, K + 1)]
                zc = readout32(out["cont_h"], self.Wf) if "cont_h" in out else None
                zbf = F.linear(hr[K], self.W).float().detach() if bf16_ref else None
        finally:
            for h in hs:
                h.remove()
            if do_wrap:
                for att in self.attn:
                    att.__dict__.pop("forward", None)
        self.n_fwd += 1
        return {"z_r": zl[-1], "z_loops": zl, "z_cont": zc, "H": H, "probs": probs, "calls": calls, "z_r_bf16": zbf,
                "layer_calls": dict(cnt), "norm_calls": ncnt[0], "chk": {}}


class IfmR(object):
    family = "ifm"

    def __init__(self, name, dev=None, fp32=False):
        import k15_smoke as S1
        M = S1.IfmM(name)
        if fp32:                      # v5 (F4): fp32 copy of the weights for the FD check only
            M.m.float()
        self.adt = next(M.m.parameters()).dtype
        self.name, self.M, self.r, self.m, self.dev = name, M, M.r, M.m, M.dev
        self.path = name
        self.enc = M.enc
        m = self.m
        assert int(m.loop_end_layers) == int(m.num_layers), "coda present: readout would skip it (audit M2)"
        self.plain = bool(M.plain)
        self.n_slots = len(self.r.schedule(self.r.k))
        assert self.n_slots == 112, self.n_slots
        if self.plain:
            self.K, self.Lb = 4, 28
        else:
            assert int(m.loop_start_layer) == 0
            self.K, self.Lb = int(self.r.k), int(m.loop_end_layers) - int(m.loop_start_layer)
        assert self.K * self.Lb == self.n_slots
        self.Hd = int(m.model_dim)
        import copy
        cuda = str(self.dev).startswith("cuda") and torch.cuda.is_available()
        mem0 = torch.cuda.memory_allocated(self.dev) if cuda else 0
        self.out32 = copy.deepcopy(m.output).float().eval()   # v3 readout in float32 (review MS2/F2)
        for prm in self.out32.parameters():
            prm.requires_grad_(False)
        self.out32_params = int(sum(p_.numel() for p_ in self.out32.parameters()))
        assert self.out32_params == int(sum(p_.numel() for p_ in m.output.parameters()))
        self.out32_bytes = (torch.cuda.memory_allocated(self.dev) - mem0) if cuda else None
        if cuda:   # the copy must be the head alone (no parent module dragged in by deepcopy)
            assert self.out32_bytes <= 4 * self.out32_params + (64 << 20), (self.out32_bytes, self.out32_params)
        self.ctx = {"slot": None, "edits": {}, "rec": {}, "leaves": {}, "ko": {}, "H": {}, "final": None,
                    "chk": None, "chk_out": {}, "slots_run": 0}
        self.n_fwd = 0
        self._install()

    def slot(self, site):
        return self.Lb * (site[0] - 1) + site[1]

    def site_of(self, slot):
        return (slot // self.Lb + 1, slot % self.Lb)

    def _install(self):
        r, C = self.r, self.ctx
        o_layer = r._layer
        last = self.n_slots - 1

        def lay(lid, h, freq, cache, slot, _o=o_layer):
            C["slot"] = slot
            out = _o(lid, h, freq, cache, slot)
            C["slot"] = None
            C["slots_run"] += 1
            s = self.site_of(slot)
            if s in C["leaves"]:
                out = out + C["leaves"][s]
            if s in C["edits"]:
                out = S2.apply_entries(out, C["edits"][s])
            if s in C["rec"]:
                C["H"][s] = (out[0] if C["rec"][s] == "all" else out[0, C["rec"][s]]).detach().clone()
            if slot == last:
                C["final"] = out
            return out
        r._layer = lay
        from xllm.modules.causal_attention import CausalSoftmaxAttention
        n = 0
        for mod in self.m.modules():
            if not isinstance(mod, CausalSoftmaxAttention):
                continue
            assert mod.backend is None, mod.backend
            prev = mod.forward

            def att(xq, xk, xv, g, freqs_cis, segments, fp32_attn_output=False, deterministic=True, cache=None,
                    _p=prev, _m=mod):
                s = C["slot"]
                km = C["ko"].get(s) if C["ko"] else None
                if km is not None:
                    assert cache is None, "knockout only on the no-cache forward"
                    segments = km if segments is None else (segments | km)
                res = _p(xq, xk, xv, g, freqs_cis, segments, fp32_attn_output, deterministic, cache)
                if C["chk"] is not None and s in C["chk"] and cache is None:
                    pr, o2 = attn_recompute_full(_m, xq, xk, xv, freqs_cis, segments)
                    C["chk_out"][s] = (float((res[0].float() - o2.float()).abs().max()), pr)
                return res
            mod.forward = att
            n += 1
        self.n_attn = n

    def run(self, ids, r, edits=None, rec=None, ko=None, leaves=None, cont=False, cap=None, grad=False,
            mask_log=None, bf16_ref=False):
        C = self.ctx
        T = len(ids)
        x = torch.tensor([ids], device=self.dev)
        kom = {}
        if ko:
            cache = {}
            for call, spec in ko.items():
                k_ = id(spec)
                if k_ not in cache:
                    mm = torch.zeros(1, T, T, dtype=torch.bool, device=self.dev)
                    mm[0, torch.tensor(spec[0], device=self.dev)[:, None],
                       torch.tensor(spec[1], device=self.dev)[None, :]] = True
                    cache[k_] = mm
                kom[self.slot(call)] = cache[k_]
        C.update({"edits": edits or {}, "rec": rec or {}, "leaves": leaves or {}, "ko": kom, "H": {}, "final": None,
                  "chk": ({self.slot(c) for c in cap} if cap else None), "chk_out": {}, "slots_run": 0})
        ex = torch.zeros(1, dtype=torch.long, device=self.dev)
        zc, zbf = None, None
        try:
            if grad:
                with torch.enable_grad():
                    zr_raw, _ = type(self.r).fwd.__wrapped__(self.r, x, self.r.k, cache=None, example_ids=ex)
                    with Fp32MM():
                        z_r = self.out32(C["final"][:, r:r + 1].float(), None, None)[0, 0]
            else:
                with torch.no_grad():
                    zr_raw, _ = self.r.fwd(x, self.r.k, cache=None, example_ids=ex)
                    fin = C["final"]
                    with Fp32MM():
                        z_r = self.out32(fin[:, r:r + 1].float(), None, None)[0, 0]
                        if cont and T > r + 1:
                            zc = self.out32(fin[:, r + 1:].float(), None, None)[0]
                    zbf = self.m.output(fin[:, r:r + 1], None, None)[0, 0].float() if bf16_ref else None
            assert z_r.dtype == torch.float32, z_r.dtype
            self.last_final_dtype = str(C["final"].dtype)
            H, chk, nslots = C["H"], C["chk_out"], C["slots_run"]
        finally:
            C.update({"edits": {}, "rec": {}, "leaves": {}, "ko": {}, "H": {}, "final": None, "chk": None,
                      "chk_out": {}})
        self.n_fwd += 1
        return {"z_r": z_r, "z_loops": None, "z_cont": zc, "H": H, "probs": {}, "chk": chk, "slots_run": nslots,
                "z_internal_last": zr_raw[0, -1].float().detach(), "z_r_bf16": zbf}


def attn_recompute_full(mod, xq, xk, xv, freqs_cis, segments):
    """Backend-None CausalSoftmaxAttention ops (as k16_path_ifm.attn_recompute), all query rows."""
    from einops import rearrange
    from xllm.modules.causal_attention import repeat_kv
    from xllm.modules.rotary_positional_embedding import apply_rotary_embeddings
    bs, slen, _ = xq.shape
    q = rearrange(xq, 'b l (h d) -> b l h d', h=mod.n_heads)
    k = rearrange(xk, 'b l (h d) -> b l h d', h=mod.n_kv_heads)
    v = rearrange(xv, 'b l (h d) -> b l h d', h=mod.n_kv_heads)
    if mod.rope_head_dim == mod.head_dim:
        q, k = apply_rotary_embeddings(q, k, freqs_cis=freqs_cis, backward=False)
    elif mod.rope_head_dim > 0:
        sp = [mod.rope_head_dim, mod.head_dim - mod.rope_head_dim]
        q_r, q_n = torch.split(q, sp, dim=-1)
        k_r, k_n = torch.split(k, sp, dim=-1)
        q_r, k_r = apply_rotary_embeddings(q_r, k_r, freqs_cis=freqs_cis, backward=False)
        q, k = torch.cat([q_r, q_n], dim=-1), torch.cat([k_r, k_n], dim=-1)
    n_rep = mod.n_heads // mod.n_kv_heads
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    q = rearrange(q, 'b l h d -> b h l d')
    k = rearrange(k, 'b l h d -> b h d l')
    v = rearrange(v, 'b l h d -> b h l d')
    mask = torch.full((bs, slen, slen), float("-inf"), device=q.device)
    mask = torch.triu(mask, diagonal=1).type_as(q)
    if segments is not None:
        mask = mask.masked_fill(segments, value=float("-inf"))
    scores = torch.matmul(q, k) * mod.scale + mask.unsqueeze(1)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q)
    out = rearrange(torch.matmul(probs, v), 'b h l d -> b l (h d)')
    return probs[0].float(), out


def model_copy(name, path):
    """Repo id plus the local copy as a path relative to the home directory (A7: no host or user names)."""
    repo = OURO_REPOS.get(name, name)
    if path and os.path.isdir(str(path)):
        try:
            rel = os.path.relpath(os.path.abspath(path), os.path.expanduser("~")).replace("\\", "/")
        except ValueError:
            rel = os.path.basename(os.path.abspath(path))
        return {"repo": repo, "copy": "~/" + rel if not rel.startswith("..") else os.path.basename(os.path.abspath(path))}
    return {"repo": repo, "copy": "hub cache" if name in OURO_REPOS else "k10_ifm checkpoint (%s)" % name}


def load_runner(name, dev, fp32=False):
    return OuroR(name, dev, fp32=fp32) if name in OURO_REPOS else IfmR(name, dev, fp32=fp32)


def sync(R):
    if str(R.dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------ VarBench items at runtime
def load_vb(data_dir, R):
    return json.load(open(os.path.join(data_dir, "items_%s.json" % ("ouro" if R.family == "ouro" else "ifm"))))


def vb_item(raw, meta):
    """Stored item -> runtime dict: the four row sequences (I2), R2 sequences, strata."""
    fa, fb, ts = raw["full_A"], raw["full_B"], raw["tstar"]
    restA, restB = fa[ts:], fb[ts:]
    nb = len(meta["block_ids"])
    it = {k: raw[k] for k in ("pair_id", "idx", "split", "value_A", "value_B", "n_digits", "first_digit_shared",
                              "gold_first_digit_differs", "plen", "tstar", "n_chain", "order", "r2_off")}
    it["cluster"] = str(raw["idx"])
    it["kind"] = raw.get("kind", "copy")
    it["var_name"], it["gen_pair_id"] = raw.get("var_name"), raw.get("gen_pair_id")
    it["restA"], it["restB"] = restA, restB
    it["tA"], it["tB"] = restA[0], restB[0]
    it["cont"] = len(restA) >= 2
    SA, SA2 = fa[:ts] + restA[:-1], fa[:ts] + restB[:-1]
    SB, SB2 = fb[:ts] + restB[:-1], fb[:ts] + restA[:-1]
    R2A = meta["block_alt_ids"] + SA[nb:]
    R2B = meta["block_alt_ids"] + SB[nb:]
    span = list(raw["span"])
    shift = 0
    if DROP_BLOCK:                       # laptop debug only (plan DV9)
        keep0 = 1 if meta.get("bos") is not None else 0
        cut = nb - keep0
        SA, SA2, SB, SB2 = [s[:keep0] + s[nb:] for s in (SA, SA2, SB, SB2)]
        R2A, R2B = SA, SB
        shift = cut
        it["r2_off"] = 0
    it.update({"SA": SA, "SA2": SA2, "SB": SB, "SB2": SB2, "R2A": R2A, "R2B": R2B})
    it["r"] = ts - 1 - shift
    # v5 derived-use items (F5): the source is the prompt span plus every chain restatement of x (own value per side)
    it["src"] = sorted(set(range(span[0] - shift, span[1] - shift)) | {p - shift for p in raw.get("src_extra", [])})
    it["span"] = [span[0] - shift, span[1] - shift]
    n_e = sum(1 for p in it["src"] if SA[p] != SB[p])     # v5 (plan critic 3 F7): positions whose A, B tokens differ
    it["n_src_differ"] = n_e
    it["strata"] = {"ST1": "1" if len(it["src"]) == 1 else ">=2", "ST1e": "1" if n_e <= 1 else ">=2",
                    "ST2": "shared" if raw["first_digit_shared"] else "differs",
                    "ST3": str(raw["n_digits"]) if raw["n_digits"] < 3 else ">=3",
                    "ST4": str(tertile(raw["plen"], meta["len_tertile_cuts"])),
                    "ST5": "<=10" if raw["n_chain"] <= 10 else ("11-40" if raw["n_chain"] <= 40 else ">40"),
                    "ST6": "aligned", "cont": "yes" if it["cont"] else "no"}
    return it


def check_tokenization(R, data, n_check):
    """Load-time assert (I3): the model's own tokenizer reproduces the stored ids."""
    meta = data["meta"]
    block = meta["block_text"]
    items = [x for x in data["items"] if x["selected"]][:n_check]
    enc = R.enc_vb if R.family == "ouro" else R.enc
    bad = []
    for raw in items:
        ta = block + "Question: " + raw["q_A"] + "\nAnswer:" + " " + raw["C"] + str(raw["value_A"])
        tb = block + "Question: " + raw["q_B"] + "\nAnswer:" + " " + raw["C"] + str(raw["value_B"])
        if enc(ta) != raw["full_A"] or enc(tb) != raw["full_B"]:
            bad.append(raw["pair_id"])
    out = {"checked": len(items), "mismatch": len(bad), "first_mismatch": bad[:3]}
    if R.family == "ouro":
        hf_bad = 0
        for raw in items[:20]:
            ta = block + "Question: " + raw["q_A"] + "\nAnswer:" + " " + raw["C"] + str(raw["value_A"])
            hf_bad += int(R.enc(ta) != raw["full_A"])
        out["transformers_tokenizer_mismatch_of_20"] = hf_bad
        out["tokenizer_md5"] = R.tok_md5
    else:
        out["bos_id"] = int(R.r.tok.bos_token_id)
        if out["bos_id"] != IFM_BOS:
            out["mismatch"] += 1
    return out


# ------------------------------------------------------------------ per-item engine
class Item(object):
    """Clean A/B runs (with records) and the row runner for one item (VarBench or A2)."""

    def __init__(self, R, it, rec, vb=True):
        self.R, self.it, self.vb = R, it, vb
        self.rows = it.setdefault("rows", {})
        self.r = it["r"]
        self.tA, self.tB = it["tA"], it["tB"]
        self.seqA = it["SA"] if vb else it["A"]
        self.seqB = it["SB"] if vb else it["B"]
        cont = vb and it["cont"]
        sync(R)
        t0 = time.time()
        self.cA = R.run(self.seqA, self.r, rec=rec, cont=cont)
        sync(R)
        self.t_fwd = time.time() - t0
        self.cB = R.run(self.seqB, self.r, rec=rec, cont=cont)
        it["clean_A"] = vals(self.cA["z_r"], self.tA, self.tB)
        it["clean_B"] = vals(self.cB["z_r"], self.tA, self.tB)
        if R.family == "ouro":
            it["clean_A_loops"] = [vals(z, self.tA, self.tB) for z in self.cA["z_loops"]]
            it["clean_B_loops"] = [vals(z, self.tA, self.tB) for z in self.cB["z_loops"]]
        self.passed = (int(self.cA["z_r"].argmax()) == self.tA and int(self.cB["z_r"].argmax()) == self.tB)

    @staticmethod
    def lp(o, rest):
        s = logp(o["z_r"], rest[0])
        for k in range(1, len(rest)):
            s += logp(o["z_cont"][k - 1], rest[k])
        return s

    def finish_clean(self):
        """Clean full-span log-odds LO_A, LO_B (SA2 / SB2 forwards on cont items) and the causality check."""
        it = self.it
        lpA_A = self.lp(self.cA, it["restA"]) if it["cont"] else logp(self.cA["z_r"], it["restA"][0])
        lpB_B = self.lp(self.cB, it["restB"]) if it["cont"] else logp(self.cB["z_r"], it["restB"][0])
        if it["cont"]:
            a2 = self.R.run(it["SA2"], self.r, cont=True)
            b2 = self.R.run(it["SB2"], self.r, cont=True)
            lpB_A, lpA_B = self.lp(a2, it["restB"]), self.lp(b2, it["restA"])
            it["causality_ok"] = bool(torch.equal(a2["z_r"], self.cA["z_r"]) and torch.equal(b2["z_r"], self.cB["z_r"]))
        else:
            lpB_A, lpA_B = logp(self.cA["z_r"], it["restB"][0]), logp(self.cB["z_r"], it["restA"][0])
        it["LO_A"] = r5(lpB_A - lpA_A)
        it["LO_B"] = r5(lpB_B - lpA_B)

    def row(self, key, edits, lo=False, rec=None, ko=None, seq=None, exact_vs=None, pres=None, extra=None):
        """Run one row on SA (or seq) and store it under key; existing keys are skipped."""
        if key in self.rows:
            return None
        it, R = self.it, self.R
        cont = bool(lo and self.vb and it["cont"] and seq is None)
        o = R.run(seq or self.seqA, self.r, edits=edits, rec=rec, ko=ko, cont=cont)
        rv = {"f": vals(o["z_r"], self.tA, self.tB)}
        if o["z_loops"] is not None:
            rv["L"] = [vals(z, self.tA, self.tB) for z in o["z_loops"][:-1]]
        if lo and self.vb and seq is None:
            if cont:
                lpA = self.lp(o, it["restA"])
                o2 = R.run(it["SA2"], self.r, edits=edits, ko=ko, cont=True)
                lpB = self.lp(o2, it["restB"])
                rv["caus"] = bool(torch.equal(o2["z_r"], o["z_r"]))
            else:
                lpA, lpB = logp(o["z_r"], it["restA"][0]), logp(o["z_r"], it["restB"][0])
            rv["lo"] = r5(lpB - lpA)
        if exact_vs is not None:
            rv["exact"] = bool(torch.equal(o["z_r"], exact_vs))
        if pres:
            rv["pres"] = {skey(s): r5(proj(r_state(o, s, self.r), r_state(self.cA, s, self.r), r_state(self.cB, s, self.r)))
                          for s in pres if s in o["H"]}
        if extra:
            rv.update(extra)
        self.rows[key] = rv
        return o


def vb_rec(r, v1, gsites, cslots):
    rec = {s: "all" for s in v1.values()}
    for s in list(gsites) + list(cslots):
        rec.setdefault(s, [r])
    return rec


# ------------------------------------------------------------------ carry gain (V6)
def carry_gain(R, X, sites):
    """One clean-A forward + backward with zero leaves at every site; G(site) = g . (H^B - H^A) at r."""
    it = X.it
    T = len(X.seqA)
    leaves = {s: torch.zeros(1, T, R.Hd, dtype=R.adt, device=R.dev, requires_grad=True) for s in sites}
    sync(R)
    t0 = time.time()
    o = R.run(X.seqA, X.r, leaves=leaves, grad=True)
    m = o["z_r"][X.tB] - o["z_r"][X.tA]
    m.backward()
    sync(R)
    G = {}
    for s in sites:
        g = leaves[s].grad[0, X.r].float()
        d = r_state(X.cB, s, X.r).float() - r_state(X.cA, s, X.r).float()
        G[skey(s)] = r5(g @ d)
    it["grad_run_bit_identical"] = bool(torch.equal(o["z_r"].detach(), X.cA["z_r"]))
    it["t_grad"] = round(time.time() - t0, 4)
    it["t_fwd"] = round(X.t_fwd, 4)
    del leaves, o, m
    return G


def pi_gain(R, X, sites):
    """PLAN G4 (review C-F4): path-integrated gain per site, G_PI(s) = mean over a in PI_A of g_s(a) . delta_s, where
    g_s(a) = d m_final / d H_s[r] evaluated with only site s set to lerp(H^A_s, H^B_s, a) at r (a Riemann mean of the
    gradient along the single-site path; AtP*'s QK fix and GradDrop are not applied). The edit is a leaf holding
    a . delta_s at r, so the gradient is taken at the edited state (the state at s before the edit is clean A's,
    since only s is edited). 4 forward + backward passes per site. -> {site_key: G_PI}."""
    it = X.it
    T, r = len(X.seqA), X.r
    out = {}
    sync(R)
    t0 = time.time()
    for s in sites:
        d = r_state(X.cB, s, r).float() - r_state(X.cA, s, r).float()
        acc = 0.0
        for a in PI_A:
            lv = torch.zeros(1, T, R.Hd, dtype=R.adt, device=R.dev)
            lv[0, r] = (a * d).to(R.adt)
            lv.requires_grad_(True)
            o = R.run(X.seqA, r, leaves={s: lv}, grad=True)
            (o["z_r"][X.tB] - o["z_r"][X.tA]).backward()
            acc += float(lv.grad[0, r].float() @ d)
            del lv, o
        out[skey(s)] = r5(acc / len(PI_A))
    sync(R)
    it["t_pi"] = round(time.time() - t0, 3)
    it["pi_passes"] = len(sites) * len(PI_A)
    return out


def pi_summary(items, red_keys, K, cl_of):
    """G4 row (review C-F4) on dev items with G_PI and every G_red site verified at a = 1: tolerance hit, hit - latest
    site, regret and Spearman for G_PI and for the first-order G on the same sites and items, their paired hit
    difference, and the same inside loops 1..K-1 (latest = Llast@K-1)."""
    rk = list(red_keys)
    latest = max(rk, key=ksite)
    ltk = [k for k in rk if ksite(k)[0] < K]
    latest_l = max(ltk, key=ksite) if ltk else None
    cols = {n_: [] for n_ in ("pi_hit", "g_hit", "lat_hit", "pi_reg", "g_reg", "pi_sp", "g_sp", "cl", "l_pi", "l_g",
                              "l_lat", "l_cl", "cost", "el_cl")}
    for it in items:
        if it.get("split") != "dev" or not it.get("G_PI") or not it.get("G"):
            continue
        D = Dgap(it)
        dms = {k: dm_of(it, "O1|ans|%s|1" % k) for k in rk}
        if D < GAP or any(v is None for v in dms.values()) or any(k not in it["G_PI"] for k in rk):
            continue
        hp = hit_metrics({k: it["G_PI"][k] for k in rk}, dms, D, THR[1.0])
        hg = hit_metrics({k: it["G"][k] for k in rk}, dms, D, THR[1.0])
        if hp is None or hg is None:
            continue
        cols["cost"].append(it.get("pi_passes"))
        cols["pi_reg"].append(hp["regret"])
        cols["g_reg"].append(hg["regret"])
        cols["cl"].append(cl_of(it))
        if hp["eligible"]:
            mx = max(dms.values())
            cols["pi_hit"].append(hp["hit"])
            cols["g_hit"].append(hg["hit"])
            cols["lat_hit"].append(int(dms[latest] >= mx - HIT_TOL * D))
            cols["pi_sp"].append(hp["spearman"])
            cols["g_sp"].append(hg["spearman"])
            cols["el_cl"].append(cl_of(it))
        if ltk:
            lp = hit_metrics({k: it["G_PI"][k] for k in ltk}, {k: dms[k] for k in ltk}, D, THR[1.0])
            lg = hit_metrics({k: it["G"][k] for k in ltk}, {k: dms[k] for k in ltk}, D, THR[1.0])
            if lp is not None and lp["eligible"]:
                mxl = max(dms[k] for k in ltk)
                cols["l_pi"].append(lp["hit"])
                cols["l_g"].append(lg["hit"])
                cols["l_lat"].append(int(dms[latest_l] >= mxl - HIT_TOL * D))
                cols["l_cl"].append(cl_of(it))
    cle, el_cl = cols["cl"], cols["el_cl"]

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None
    return {"sites": "G_red (%d)" % len(rk), "n_dev_complete": len(cle), "eligible": len(el_cl),
            "pi_top1_hit": boot(cols["pi_hit"], el_cl), "g_top1_hit": boot(cols["g_hit"], el_cl),
            "latest_site_hit": boot(cols["lat_hit"], el_cl),
            "pi_minus_g_hit": boot([a - b for a, b in zip(cols["pi_hit"], cols["g_hit"])], el_cl),
            "pi_hit_minus_latest": boot([a - b for a, b in zip(cols["pi_hit"], cols["lat_hit"])], el_cl),
            "pi_regret": boot(cols["pi_reg"], cle), "g_regret": boot(cols["g_reg"], cle),
            "pi_spearman_median": med(cols["pi_sp"]), "g_spearman_median": med(cols["g_sp"]),
            "loops_lt_K": {"eligible": len(cols["l_cl"]), "baseline_site": latest_l,
                           "pi_top1_hit": boot(cols["l_pi"], cols["l_cl"]), "g_top1_hit": boot(cols["l_g"], cols["l_cl"]),
                           "pi_minus_g_hit": boot([a - b for a, b in zip(cols["l_pi"], cols["l_g"])], cols["l_cl"]),
                           "pi_hit_minus_latest": boot([a - b for a, b in zip(cols["l_pi"], cols["l_lat"])], cols["l_cl"])},
            "cost_fwd_bwd_passes_per_item": med(cols["cost"]),
            "rule": "PLAN G4: mean gradient at the single-site lerp, a in %s; reported, not a frozen prediction" % (list(PI_A),)}


def v6_rows(R, X, G, is_dev, uni_site):
    """Verify rows (top/median/bottom 3 at each dose), uniform and random-site baselines; dev items: dev search."""
    it = X.it
    ks, ver = verify_sites(G)
    it["G"], it["verify"] = G, ver
    rnd = ks[int(sha(it.get("pair_id", it.get("key", "")) + "|rnd_site")[:8], 16) % len(ks)]
    it["rnd_site"], it["uni_site"] = rnd, skey(uni_site)
    todo = [(k, a) for a in A_VERIFY for k in ver + [skey(uni_site), rnd]]
    if is_dev:
        todo += [(k, 1.0) for k in ks]
    for k, a in todo:
        s = ksite(k)
        X.row("O1|ans|%s|%g" % (k, a), {s: [e_lerp([X.r], r_state(X.cB, s, X.r)[None], a, R.dev)]})


def dev_tuned_site(items_dev, sites):
    """v3 (review MS7): the dev-tuned site maximizes mean dm / D over dev items with D >= GAP (the reported metric),
    not mean raw dm. Ties go to the earlier site in the list."""
    best, bv, table = None, None, {}
    for s in sites:
        k = skey(s)
        xs = []
        for it in items_dev:
            x, D = dm_of(it, "O1|ans|%s|1" % k), Dgap(it)
            if x is not None and D >= GAP:
                xs.append(x / D)
        if not xs:
            continue
        v = sum(xs) / len(xs)
        table[k] = round(v, 4)
        if bv is None or v > bv:
            best, bv = k, v
    return best, table


def dev_tuned_pass(R, items_test, site_key, seq_fn):
    """Pass 2: O1 at the dev-tuned site on test items that lack it (clean B rerun for the donor)."""
    s = ksite(site_key)
    n = 0
    for it in items_test:
        need = [a for a in A_VERIFY if "O1|ans|%s|%g" % (site_key, a) not in it["rows"]]
        if not need:
            continue
        r = it["r"]
        cB = R.run(seq_fn(it, "B"), r, rec={s: [r]})
        for a in need:
            o = R.run(seq_fn(it, "A"), r, edits={s: [e_lerp([r], cB["H"][s], a, R.dev)]})
            it["rows"]["O1|ans|%s|%g" % (site_key, a)] = {"f": vals(o["z_r"], it["tA"], it["tB"]), "pass2": True}
            n += 1
    return n


def dm_of(it, key):
    rv = it.get("rows", {}).get(key)
    if rv is None:
        return None
    return (rv["f"][1] - rv["f"][0]) - (it["clean_A"][1] - it["clean_A"][0])


def Dgap(it):
    return (it["clean_B"][1] - it["clean_B"][0]) - (it["clean_A"][1] - it["clean_A"][0])


# ------------------------------------------------------------------ VarBench arms
def src_o1_row(R, X, v1, cslots, K):
    s1, src, r = v1["Llast@1"], X.it["src"], X.r
    loops = [v1["Llast@%d" % j] for j in range(1, K + 1)]
    pres = list(dict.fromkeys(list(cslots) + loops))
    dn = (X.cB["H"][s1][src].float() - X.cA["H"][s1][src].float()).norm(dim=-1)
    X.row("O1|src|Llast@1|1", {s1: [e_lerp(src, X.cB["H"][s1][src], 1.0, R.dev)]}, lo=True,
          rec={q: [r] for q in pres}, pres=pres, extra={"dn": [r5(x) for x in dn], "dn_sum": r5(dn.sum())})


def run_vb_item_W(R, X, v1, cslots, cnames, model):
    """Phase W on a subset item: V2 PC-site, V4 window, V5-W undo candidates, V7 answer doses."""
    it, dev, r, src = X.it, R.dev, X.r, X.it["src"]
    T = len(X.seqA)
    K, Lb = R.K, R.Lb
    for nm, s in v1.items():
        X.row("PCS|%s" % nm, {s: [e_lerp(list(range(T)), X.cB["H"][s], 1.0, dev)]}, exact_vs=X.cB["z_r"])
    Qrel = list(range(max(src) + 1, T))
    spec_dir, spec_rel = ([r], src), (Qrel, src)
    bsite = {"full": None}
    bsite.update({"after_%s" % nm: v1[nm] for nm in ("Llast@1", "Lmid@2", "Llast@2", "Lmid@3", "Llast@3", "Lmid@%d" % K)})
    for bn in WIN_BOUNDS:
        calls = calls_after(bsite[bn], K, Lb)
        X.row("KO|dir|%s" % bn, {}, ko={c: spec_dir for c in calls})
        X.row("KO|rel|%s" % bn, {}, ko={c: spec_rel for c in calls})
    rkeys = rand_ko_keys(it["pair_id"], src, r)
    X.row("KOR|rel|full", {}, ko={c: (Qrel, rkeys) for c in calls_after(None, K, Lb)},
          extra={"keys": rkeys, "pool": RKO_POOL})
    src_o1_row(R, X, v1, cslots, K)
    if model not in NO_UNDO:
        s1 = v1["Llast@1"]
        se = [e_lerp(src, X.cB["H"][s1][src], 1.0, dev)]
        for cn in cnames:
            c = v1[cn]
            ed = {s1: se, c: [e_lerp(src, X.cA["H"][c][src], 1.0, dev)]}
            calls = calls_after(c, K, Lb)
            X.row("U|%s" % cn, ed, lo=True)
            X.row("BKO|%s" % cn, {}, ko={cc: spec_rel for cc in calls}, seq=X.seqB)
            X.row("UKO|%s" % cn, ed, ko={cc: spec_rel for cc in calls})
    for j in range(1, K):
        s = v1["Llast@%d" % j]
        for a in (2.0, 4.0):
            X.row("V7|a%g|Llast@%d" % (a, j), {s: [e_lerp([r], X.cB["H"][s][[r]], a, dev)]})
        dr = X.cB["H"][s][r].float() - X.cA["H"][s][r].float()
        nsrc = float((X.cB["H"][s][src].float() - X.cA["H"][s][src].float()).norm(dim=-1).sum())
        if float(dr.norm()) > 0:
            X.row("V7|mn|Llast@%d" % j, {s: [e_add([r], (dr / dr.norm() * nsrc)[None], dev)]},
                  extra={"norm": r5(nsrc), "ans_delta_norm": r5(dr.norm())})


def run_vb_item_M(R, X, v1, gsites, cslots, model, slots_sel, is_dev, uni_site, rnd_cells, o2_ok, do_pi=False):
    """Phase M: V1 placement (+ full span), O2 context check, presence, undo at the selected slots, V6."""
    it, dev, r, src = X.it, R.dev, X.r, X.it["src"]
    K = R.K
    pid = it["pair_id"]
    loops = [v1["Llast@%d" % j] for j in range(1, K + 1)]
    src_o1_row(R, X, v1, cslots, K)
    for pos_name, P in (("src", src), ("ans", [r])):
        for nm, s in v1.items():
            dn = (X.cB["H"][s][P].float() - X.cA["H"][s][P].float()).norm(dim=-1)
            pres = loops if (pos_name == "ans" and nm in ["Llast@%d" % j for j in range(1, K)]) else None
            ex = X.cB["z_r"] if (pos_name, nm) == ("ans", "Llast@%d" % K) else (
                X.cA["z_r"] if (pos_name, nm) == ("src", "Llast@%d" % K) else None)
            X.row("O1|%s|%s|1" % (pos_name, nm), {s: [e_lerp(P, X.cB["H"][s][P], 1.0, dev)]}, lo=True,
                  rec={q: [r] for q in pres} if pres else None, pres=pres, exact_vs=ex,
                  extra={"dn": [r5(x) for x in dn], "dn_sum": r5(dn.sum())})
            if rnd_cells is None or (pos_name, nm) in rnd_cells:
                delta = rnd_delta(pid, s, pos_name, P, dn.tolist(), R.Hd)
                exr = X.cA["z_r"] if (pos_name, nm) == ("src", "Llast@%d" % K) else None
                X.row("RND|%s|%s|1" % (pos_name, nm), {s: [e_add(P, delta, dev)]}, lo=(pos_name, nm) in LO_RND,
                      exact_vs=exr)
    if o2_ok:
        off = it["r2_off"]
        for nm in ("Llast@1", "Llast@3"):
            s = v1[nm]
            for pos_name, P in (("src", src), ("ans", [r])):
                Pp = [p + off for p in P]
                v = X.r2B["H"][s][Pp].float() - X.r2A["H"][s][Pp].float()
                own = X.cB["H"][s][P].float() - X.cA["H"][s][P].float()
                on = float(own.norm())
                diag = {"cos": r5(F.cosine_similarity(v.flatten(), own.flatten(), dim=0)) if on > 0 else None,
                        "norm_ratio": r5(float(v.norm()) / on) if on > 0 else None}
                X.row("O2|%s|%s|1" % (pos_name, nm), {s: [e_add(P, v, dev)]}, extra=diag)
    if model not in NO_UNDO and slots_sel:
        s1 = v1["Llast@1"]
        se = [e_lerp(src, X.cB["H"][s1][src], 1.0, dev)]
        T = len(X.seqA)
        spec_rel = (list(range(max(src) + 1, T)), src)
        for cn in slots_sel:
            c = v1[cn]
            ed = {s1: se, c: [e_lerp(src, X.cA["H"][c][src], 1.0, dev)]}
            X.row("U|%s" % cn, ed, lo=True)
            # v5 (plan critic 3 F6): the carried denominator of s(c) on every phase-M item (rows exist on W items)
            calls = calls_after(c, K, R.Lb)
            X.row("UKO|%s" % cn, ed, ko={cc: spec_rel for cc in calls})
            X.row("KO|rel|after_%s" % cn, {}, ko={cc: spec_rel for cc in calls})
            # v6 (review C-F1): the D_KO ceiling of b2 on M-only items too (1 forward per selected slot)
            X.row("BKO|%s" % cn, {}, ko={cc: spec_rel for cc in calls}, seq=X.seqB)
    if not NO_GRAD[0]:
        G = carry_gain(R, X, gsites)
        v6_rows(R, X, G, is_dev, uni_site)
        if do_pi:                                    # PLAN G4 (review C-F4): dev items only, G_red sites
            it["G_PI"] = pi_gain(R, X, gset("red", K, R.Lb))


def r2_runs(R, X, v1, cslots):
    """R2 clean runs: clean-pass filter (P4), O2 vectors, presence-axis validation (F5)."""
    it = X.it
    rp = X.r + it["r2_off"]
    rec = {s: "all" for s in (v1["Llast@1"], v1["Llast@3"])}
    for s in list(v1.values()) + list(cslots):
        rec.setdefault(s, [rp])
    X.r2A = R.run(it["R2A"], rp, rec=rec)
    X.r2B = R.run(it["R2B"], rp, rec=rec)
    ok = int(X.r2A["z_r"].argmax()) == X.tA and int(X.r2B["z_r"].argmax()) == X.tB
    it["r2_pass"] = bool(ok)
    it["r2_pres"] = {skey(s): [r5(proj(r_state(X.r2A, s, rp), r_state(X.cA, s, X.r), r_state(X.cB, s, X.r))),
                               r5(proj(r_state(X.r2B, s, rp), r_state(X.cA, s, X.r), r_state(X.cB, s, X.r)))]
                     for s in dict.fromkeys(list(v1.values()) + list(cslots))}
    return ok


# ------------------------------------------------------------------ model checks (plan section 2, a-g)
@torch.no_grad()
def model_checks(R, it, v1):
    dev, r, src = R.dev, it["r"], it["src"]
    T = len(it["SA"])
    K, Lb = R.K, R.Lb
    res = {"T": T, "r": r, "src": src, "cont": it["cont"]}
    base = R.run(it["SA"], r, cont=it["cont"], bf16_ref=True)
    res["readout"] = {"dtype": str(base["z_r"].dtype),
                      "fp32_vs_bf16_head_max_abs": r5((base["z_r"] - base["z_r_bf16"]).abs().max()),
                      "bf16_head_distinct_top10": len(set(base["z_r_bf16"].topk(10).values.tolist())),
                      "fp32_head_distinct_top10": len(set(base["z_r"].topk(10).values.tolist())),
                      "tf32_after": bool(torch.backends.cuda.matmul.allow_tf32)}
    if R.family == "ifm":
        res["readout"]["out32_params"] = R.out32_params
        res["readout"]["out32_bytes"] = R.out32_bytes
    allsites = [(j, l) for j in range(1, K + 1) for l in range(Lb)]
    if R.family == "ouro":
        res["call_counts_ok"] = bool(all(base["layer_calls"].get(l) == K for l in range(R.L - 1)) and base["norm_calls"] == K)
        mlog = []
        wb = R.run(it["SA"], r, cont=it["cont"], mask_log=mlog)
        res["wrapper_bit_identical"] = bool(torch.equal(wb["z_r"], base["z_r"]))
        res["attn_calls_ok"] = bool(len(wb["calls"]) == K * R.L and all(v == 1 for v in wb["calls"].values()))
        res["eager_mask_none_calls"] = sum(1 for x in mlog if x is None)
        res["eager_mask_shape"] = mlog[0] if mlog else None
        own = R.m(input_ids=torch.tensor([it["SA"]], device=dev), use_cache=False).logits[0, r].float()
        res["model_own_logits_max_abs_diff"] = r5((own - base["z_r"]).abs().max())
    else:
        res["call_counts_ok"] = bool(base["slots_run"] == R.n_slots)
        res["n_attention_wrapped"] = R.n_attn
        res["internal_head_max_abs_diff"] = (r5((base["z_internal_last"] - base["z_r"]).abs().max())
                                             if r == T - 1 else "n/a (continuation)")
    zt = torch.zeros(T, R.Hd, device=dev)
    zero_e = [e_add(list(range(T)), zt, dev)]
    z1 = R.run(it["SA"], r, edits={s: zero_e for s in allsites}, cont=it["cont"])
    res["zero_add_every_site_bit_identical"] = bool(torch.equal(z1["z_r"], base["z_r"]) and (
        z1["z_cont"] is None or torch.equal(z1["z_cont"], base["z_cont"])))
    res["repeat_bit_identical"] = bool(torch.equal(R.run(it["SA"], r, cont=it["cont"])["z_r"], base["z_r"]))
    rec_all = {s: "all" for s in v1.values()}
    rr = R.run(it["SA"], r, rec=rec_all, cont=it["cont"])
    res["rec_bit_identical"] = bool(torch.equal(rr["z_r"], base["z_r"]))
    with torch.enable_grad():
        leaves = {s: torch.zeros(1, T, R.Hd, dtype=R.adt, device=dev, requires_grad=True)
                  for s in allsites if s != (K, Lb - 1)}
        og = R.run(it["SA"], r, leaves=leaves, grad=True)
        (og["z_r"][it["tB"]] - og["z_r"][it["tA"]]).backward()
        res["grad_leaf_bit_identical"] = bool(torch.equal(og["z_r"].detach(), base["z_r"]))
        res["grad_nonzero_sites"] = [sum(1 for lv in leaves.values() if float(lv.grad[0, r].abs().sum()) > 0), len(leaves)]
        del leaves, og
    if it["cont"]:
        res["causality_r_bit_identical"] = bool(torch.equal(R.run(it["SA2"], r, cont=True)["z_r"], base["z_r"]))
    order = sorted(v1.values(), key=lambda s: order_of(s, Lb))
    s_e, p = v1["Lmid@1"], src[0]
    d = torch.randn(1, R.Hd, generator=torch.Generator().manual_seed(0))
    ed = R.run(it["SA"], r, edits={s_e: [e_add([p], d, dev)]}, rec=rec_all)
    i0 = order.index(s_e)
    ep = {"site": skey(s_e), "position": p,
          "earlier_sites_identical": all(torch.equal(rr["H"][s], ed["H"][s]) for s in order[:i0]),
          "later_sites_positions_before_p_identical": all(torch.equal(rr["H"][s][:p], ed["H"][s][:p]) for s in order[i0 + 1:]),
          "position_p_changed_next_site": not torch.equal(rr["H"][order[i0 + 1]][p], ed["H"][order[i0 + 1]][p])}
    ep["ok"] = bool(ep["earlier_sites_identical"] and ep["later_sites_positions_before_p_identical"]
                    and ep["position_p_changed_next_site"])
    res["edit_placement"] = ep
    kc = (2, Lb // 2)
    Q = list(range(max(src) + 1, T))
    later = [s for s in order if order_of(s, Lb) >= order_of(kc, Lb)]
    earlier = [s for s in order if order_of(s, Lb) < order_of(kc, Lb)]
    kp = {}
    for var, spec in (("direct", ([r], src)), ("relay", (Q, src))):
        cc = R.run(it["SA"], r, rec=rec_all, cap={kc})
        kk = R.run(it["SA"], r, rec=rec_all, ko={kc: spec}, cap={kc})
        qset = spec[0]
        keep = [q for q in range(T) if q not in qset]
        x = {"call": list(kc), "Q": [min(qset), max(qset), len(qset)],
             "earlier_sites_identical": all(torch.equal(cc["H"][s], kk["H"][s]) for s in earlier),
             "later_sites_positions_before_Q_identical": all(torch.equal(cc["H"][s][:min(qset)], kk["H"][s][:min(qset)])
                                                             for s in later),
             "Q_changed_at_call_site": not torch.equal(cc["H"][kc][qset], kk["H"][kc][qset]),
             "logits_changed": not torch.equal(cc["z_r"], kk["z_r"])}
        ok = x["earlier_sites_identical"] and x["later_sites_positions_before_Q_identical"] and x["Q_changed_at_call_site"]
        if R.family == "ouro":
            P0, P1 = cc["probs"][kc], kk["probs"][kc]
            x["module_probs_knocked_exactly_zero"] = bool((P1[:, qset][:, :, src] == 0).all())
            x["module_probs_other_rows_identical"] = bool(torch.equal(P0[:, keep], P1[:, keep]))
            ok = ok and x["module_probs_knocked_exactly_zero"] and x["module_probs_other_rows_identical"]
        else:
            mv, pr = kk["chk"].get(R.slot(kc), (None, None))
            x["module_vs_recompute_max_abs"] = r5(mv) if mv is not None else None
            x["recompute_knocked_mass_max"] = r5(pr[:, qset][:, :, src].sum(-1).max()) if pr is not None else None
            x["note"] = "module output vs recompute under the mask; functional span-vs-random check in the window (F11)"
        x["ok"] = bool(ok)
        kp[var] = x
    res["knockout_placement"] = kp
    res["identity_ok"] = bool(res["call_counts_ok"] and res["zero_add_every_site_bit_identical"]
                              and res["repeat_bit_identical"] and res["rec_bit_identical"]
                              and res["grad_leaf_bit_identical"] and res.get("wrapper_bit_identical", True)
                              and res.get("attn_calls_ok", True) and res.get("eager_mask_none_calls", 0) == 0
                              and res["readout"]["dtype"] == "torch.float32")
    res["all_ok"] = bool(res["identity_ok"] and ep["ok"] and all(v["ok"] for v in kp.values())
                         and res.get("causality_r_bit_identical", True))
    return res


def d1_assign(cands):
    """v3 D1 split (review MS3/F6): split by the A subject's hash; a pair is kept only when the B subject hashes
    to the same split, so dev and test share no subject on either side. 'half' (the k15_data.select quota key)
    follows the split. Candidates are scanned in pair order. -> (kept, n_cross_split_dropped)."""
    out, n_cross = [], 0
    for p in sorted(cands, key=lambda p: p["idx"]):
        sa, sb = d1_split(p["rel"], p["a"]), d1_split(p["rel"], p["b"])
        if sa != sb:
            n_cross += 1
            continue
        p["k15_half"] = p.get("half")
        p["split"] = sa
        p["half"] = "dev" if sa == "dev" else "heldout"
        out.append(p)
    return out, n_cross


# ------------------------------------------------------------------ A2 datasets (V6 only)
def a2_candidates(R, ds, data_dir, n_total, excl):
    import k15_data as KD
    if ds == "D1":
        allc = KD.d1_candidates(R.enc)
        cands, n_cross = d1_assign(allc)
        q = n_total // 4
        quota = {("heldout", "capital"): q, ("heldout", "city"): q, ("dev", "capital"): q, ("dev", "city"): q}
        for p in cands:
            p["cluster"] = "%s:%s" % (p["rel"], p["a"])
            p["cluster_b"] = "%s:%s" % (p["rel"], p["b"])
            p["strata"] = {"pos_len": "1" if len(p["pos"]) == 1 else ">=2", "ST1": "1" if len(p["pos"]) == 1 else ">=2",
                           "ST1e": st1e(p["A"], p["B"], p["pos"]),
                           "subject_pos0": "yes" if 0 in p["pos"] else "no"}
            p["kind"] = "retrieval"
            p["key"] = "%s|%s|%s" % (p["rel"], p["a"], p["b"])
        return cands, quota, "k15", {
            "label": "same pool as K16 D1 (49 countries, 33 cities: no fresh subjects)",
            "split": "by subject sha256 (A and B subject in the same split; dev and test share no subject)",
            "candidates_all": len(allc), "dropped_cross_split": n_cross,
            "candidates_by_split": {s: sum(1 for p in cands if p["split"] == s) for s in ("dev", "test")},
            "ci": "dyadic bootstrap over subjects (A and B side)"}
    if ds == "D2":
        ex = set(excl["D2"])
        allc = KD.d2_candidates(R.enc)
        cands = [p for p in allc if p["a"] not in ex]
        quota = {("all", "op1"): n_total // 2, ("all", "op2"): n_total // 2}
        for p in cands:
            a1, a2_ = [int(v) for v in p["a"].split(" + ")]
            p["cluster"] = "x=%d" % (a1 if p["rel"] == "op1" else a2_)
            p["split"] = "dev" if int(sha("k17vb-d2:" + p["cluster"])[0], 16) % 2 == 0 else "test"
            p["strata"] = {"op": p["rel"], "ST1": "1" if len(p["pos"]) == 1 else ">=2",
                          "ST1e": st1e(p["A"], p["B"], p["pos"])}
            p["kind"] = "computed"
            p["key"] = "%s|%s|%s" % (p["rel"], p["a"], p["b"])
        return cands, quota, "k15", {"candidates_all": len(allc), "excluded_k16": len(allc) - len(cands)}
    if ds == "D4":
        tok = "ouro" if R.family == "ouro" else "ifm"
        rows = [json.loads(x) for x in open(os.path.join(data_dir, "d4_%s.jsonl" % tok), encoding="utf-8")]
        cont_ex, tpl_ex, id_ex = set(excl["D4_content"]), set(excl["D4_templates"]), set(excl["D4_ids"])
        # v4 (plan critic 2 F8): cluster = connected component of shared A/B content keys over every row of the
        # file; a component that holds any K16 key (A or B side of a K16 row, or the v3 K16 content set) is out.
        comp = d4_components(rows)
        k16_keys = set(cont_ex)
        for rr in rows:
            if rr["id"] in id_ex or rr["cluster_id"] in tpl_ex:
                k16_keys.update((d4_content(rr, "A"), d4_content(rr, "B")))
        k16_comps = {comp[k] for k in k16_keys if k in comp}
        out, n_bad, n_ex = [], 0, 0
        for rr in sorted(rows, key=lambda x: x["id"]):
            A, B, src = rr["ids_A"], rr["ids_B"], rr["source_span"]
            if len(A) != len(B) or [i for i in range(len(A)) if A[i] != B[i]] != list(src):
                n_bad += 1
                continue
            cc = comp[d4_content(rr, "A")]
            if rr["id"] in id_ex or rr["cluster_id"] in tpl_ex or cc in k16_comps:
                n_ex += 1
                continue
            out.append({"A": A, "B": B, "pos": src, "tA": rr["answer_first_token_A"], "tB": rr["answer_first_token_B"],
                        "rel": rr["kind"], "id": rr["id"], "key": rr["id"], "cluster": cc, "a": rr["id"], "b": rr["id"],
                        "split": "dev" if int(sha("k17vb-d4c:" + cc)[0], 16) % 2 == 0 else "test",
                        "strata": {"kind": rr["kind"], "ST1": "1" if len(src) == 1 else ">=2", "ST1e": st1e(A, B, src)},
                        "kind": "retrieval" if rr["kind"] == "T1" else "computed",
                        "prompt_A": rr["prompt_A"], "suffix_ids": rr["suffix_ids"]})
        csz = {}
        for k, c in comp.items():
            csz[c] = csz.get(c, 0) + 1
        q = n_total // 4
        quota = {("dev", "T1"): q, ("test", "T1"): q, ("dev", "T2"): q, ("test", "T2"): q}
        if R.family == "ifm":
            nb = sum(1 for p in out[:50] if R.enc(p["prompt_A"]) + p["suffix_ids"] != p["A"])
        else:
            nb = sum(1 for p in out[:50] if R.enc(p["prompt_A"]) + p["suffix_ids"] != p["A"])
        dev_c = {p["cluster"] for p in out if p["split"] == "dev"}
        tst_c = {p["cluster"] for p in out if p["split"] == "test"}
        dev_k = {d4_content(rr, s) for rr in rows if comp[d4_content(rr, "A")] in dev_c for s in ("A", "B")}
        tst_k = {d4_content(rr, s) for rr in rows if comp[d4_content(rr, "A")] in tst_c for s in ("A", "B")}
        return out, quota, "own", {"rows": len(rows), "unequal_or_bad_span": n_bad, "excluded_k16": n_ex,
                                   "encode_mismatch_of_50": nb,
                                   "cluster": "connected component of shared A/B content keys (v4, plan critic 2 F8)",
                                   "n_components": len(csz), "largest_component_keys": max(csz.values()) if csz else 0,
                                   "k16_components_excluded": len(k16_comps),
                                   "dev_test_shared_content_keys": len(dev_k & tst_k)}
    raise KeyError(ds)


def a2_select(cands, know, quota, mode):
    import k15_data as KD
    if mode == "k15":
        return KD.select(cands, know, quota)
    n_q, seen, kept, scanned = {}, set(), [], 0
    for p in cands:
        qk = (p["split"], p["rel"])
        if qk not in quota or n_q.get(qk, 0) >= quota[qk] or p["cluster"] in seen:
            continue
        scanned += 1
        if not know(p):
            continue
        n_q[qk] = n_q.get(qk, 0) + 1
        seen.add(p["cluster"])
        kept.append(p)
        if all(n_q.get(k, 0) >= v for k, v in quota.items()):
            break
    return kept, scanned


class _Budget(Exception):
    pass


def run_a2(R, model, ds, data_dir, gsites, uni_site, res_ds, debug, deadline, excl, dump):
    t0 = time.time()
    n_total = 4 if debug else A2_N[model]
    cands, quota, mode, binfo = a2_candidates(R, ds, data_dir, n_total, excl)
    if debug:                                    # one dev-side and one test-side cell where the split is in the key
        ks = list(quota)
        dev_k = [k for k in ks if k[0] in ("dev",)]
        oth_k = [k for k in ks if k[0] not in ("dev",)]
        quota = {k: 1 for k in (dev_k[:1] + oth_k[:1] if dev_k else ks[:2])}
    stats = {"A_fail": 0, "B_fail": 0}
    K, Lb = R.K, R.Lb
    sK = (K, Lb - 1)
    done = res_ds.get("items", {})

    def know(p):
        if deadline is not None and time.time() > deadline:
            raise _Budget()
        p["r"] = len(p["A"]) - 1
        X = Item(R, p, {s: [p["r"]] for s in list(gsites) + [sK]}, vb=False)
        if int(X.cA["z_r"].argmax()) != p["tA"]:
            stats["A_fail"] += 1
            return False
        if int(X.cB["z_r"].argmax()) != p["tB"]:
            stats["B_fail"] += 1
            return False
        p["_X"] = X
        return True
    truncated = False
    try:
        kept, scanned = a2_select(cands, know, quota, mode)
    except _Budget:
        kept, scanned, truncated = [p for p in cands if "_X" in p], None, True
    kept = rr_order(kept, lambda p: (p["split"], p["rel"]))
    res_ds.update({"candidates": len(cands), "scanned": scanned, "kept": len(kept), "know_stats": stats,
                   "quota": {"%s|%s" % k: v for k, v in quota.items()}, "truncated_selection": truncated,
                   "build": binfo, "items": done})
    print("[k17vb %s] %s kept %d scanned %s %.0fs" % (model, ds, len(kept), scanned, time.time() - t0), flush=True)
    for i, p in enumerate(kept):
        X = p.pop("_X")
        if p["key"] in done and done[p["key"]].get("done"):
            continue
        if deadline is not None and time.time() > deadline:
            res_ds["truncated_items"] = True
            break
        it = {k: p[k] for k in ("key", "rel", "a", "b", "split", "cluster", "kind", "strata", "pos", "r", "tA", "tB",
                                "clean_A", "clean_B", "rows")}
        if "cluster_b" in p:
            it["cluster_b"] = p["cluster_b"]
        it["T"] = len(p["A"])
        if "clean_A_loops" in p:
            it["clean_A_loops"], it["clean_B_loops"] = p["clean_A_loops"], p["clean_B_loops"]
        X.it, X.rows = it, it["rows"]
        f0, ti = R.n_fwd, time.time()
        X.row("O1|ans|%s|1" % skey(sK), {sK: [e_lerp([X.r], X.cB["H"][sK], 1.0, R.dev)]}, exact_vs=X.cB["z_r"])
        if not NO_GRAD[0]:
            G = carry_gain(R, X, gsites)
            v6_rows(R, X, G, it["split"] == "dev", uni_site)
            if (ds in PI_DATASETS and it["split"] == "dev"
                    and sum(1 for v in done.values() if v.get("G_PI")) < PI_N):      # PLAN G4 (review C-F4)
                it["G_PI"] = pi_gain(R, X, gset("red", K, Lb))
        sync(R)
        it["forwards"], it["seconds"] = R.n_fwd - f0, round(time.time() - ti, 2)
        it["done"] = True
        done[p["key"]] = it
        if i % 10 == 0 or i == len(kept) - 1:
            print("[k17vb %s] %s item %d/%d (%d fwd, %.1fs)" % (model, ds, i + 1, len(kept), it["forwards"],
                                                               it["seconds"]), flush=True)
            dump()
        del X
    items = [v for v in done.values() if v.get("done")]
    best, table = dev_tuned_site([v for v in items if v["split"] == "dev"], gsites)
    res_ds["dev_tuned"] = {"site": best, "n_dev": sum(1 for v in items if v["split"] == "dev"), "mean_dm_over_D_by_site": table}
    if best is not None:
        pmap = {p["key"]: p for p in kept}

        def seq_fn(it, w):
            p = pmap[it["key"]]
            return p["A"] if w == "A" else p["B"]
        res_ds["dev_tuned"]["pass2_rows"] = dev_tuned_pass(R, [v for v in items if v["split"] == "test" and v["key"] in pmap],
                                                           best, seq_fn)
    res_ds["seconds"] = round(time.time() - t0, 1)


def check_tokenization_vbd(R, data, n_check):
    """Load-time assert for derived-use items: the model's tokenizer reproduces the stored ids (prefix up to t*)."""
    block = data["meta"]["block_text"]
    enc = R.enc_vb if R.family == "ouro" else R.enc
    items = [x for x in data["items"] if x["selected"]][:n_check]
    bad = []
    for raw in items:
        for q, tail, full in ((raw["q_A"], raw["text_tail_A"], raw["full_A"]), (raw["q_B"], raw["text_tail_B"], raw["full_B"])):
            if enc(block + "Question: " + q + "\nAnswer:" + tail)[:len(full)] != full:
                bad.append(raw["pair_id"])
                break
    return {"checked": len(items), "mismatch": len(bad), "first_mismatch": bad[:3]}


def vbd_raws(data, pool):
    """RA-20 A2 item 1: --pool pairs = every valid derived-use pair (cluster = problem); else one pair per problem."""
    return list(data["items"]) if pool == "pairs" else [x for x in data["items"] if x["selected"]]


def vbd_open(R, model, data_dir, res_v, debug, pool):
    """Load the derived-use items, check tokenization, fix n_target. -> (raws, meta, n_target, with_v1) or None."""
    tok = "ouro" if R.family == "ouro" else "ifm"
    fp = os.path.join(data_dir, "items_vbd_%s.json" % tok)
    if not os.path.exists(fp):
        res_v["error"] = "items_vbd_%s.json missing (run --build_vbd)" % tok
        return None
    data = json.load(open(fp))
    meta = data["meta"]
    res_v["build"] = data["stats"]
    res_v["tokenization"] = check_tokenization_vbd(R, data, 20 if debug else 400)
    if res_v["tokenization"]["mismatch"]:
        res_v["error"] = "tokenization mismatch"
        return None
    raws = vbd_raws(data, pool)
    n_target = 2 if debug else (VBD_N[model] or len(raws))      # RA-20 A2 item 1: VBD_N 0 = no cap
    with_v1 = VBD_V1[model]
    res_v["config"] = {"n_target": n_target, "v1": with_v1, "endpoint": "first-token e (no continuation)",
                       "gset": "same as VarBench"}
    return raws, meta, n_target, with_v1


def vbd_load(R, raw, meta, v1, gsites, role, S):
    """Clean A/B runs of one derived-use item; the clean-pass result goes to the scan dict S."""
    it = vb_item(raw, meta)
    it["host_role"] = role
    X = Item(R, it, vb_rec(it["r"], v1, gsites, []))
    S[raw["pair_id"]] = {"A_pass": int(X.cA["z_r"].argmax()) == X.tA, "B_pass": int(X.cB["z_r"].argmax()) == X.tB}
    return X, it


def vbd_item_rows(R, X, it, v1, gsites, uni_site, rnd_cells, with_v1, do_pi):
    """Rows of one kept derived-use item (V1 where with_v1, V6, G_PI where do_pi)."""
    K = R.K
    sK = (K, R.Lb - 1)
    f0, ti = R.n_fwd, time.time()
    dev, r, src, pid = R.dev, X.r, it["src"], it["pair_id"]
    X.row("O1|ans|Llast@%d|1" % K, {sK: [e_lerp([r], X.cB["H"][sK][[r]], 1.0, dev)]}, exact_vs=X.cB["z_r"])
    X.row("O1|src|Llast@%d|1" % K, {sK: [e_lerp(src, X.cB["H"][sK][src], 1.0, dev)]}, exact_vs=X.cA["z_r"])
    if with_v1:
        for pos_name, P in (("src", src), ("ans", [r])):
            for nm, s in v1.items():
                dn = (X.cB["H"][s][P].float() - X.cA["H"][s][P].float()).norm(dim=-1)
                X.row("O1|%s|%s|1" % (pos_name, nm), {s: [e_lerp(P, X.cB["H"][s][P], 1.0, dev)]},
                      extra={"dn_sum": r5(dn.sum())})
                if rnd_cells is None or (pos_name, nm) in rnd_cells:
                    delta = rnd_delta(pid, s, pos_name, P, dn.tolist(), R.Hd)
                    X.row("RND|%s|%s|1" % (pos_name, nm), {s: [e_add(P, delta, dev)]},
                          exact_vs=X.cA["z_r"] if (pos_name, nm) == ("src", "Llast@%d" % K) else None)
    if not NO_GRAD[0]:
        G = carry_gain(R, X, gsites)
        v6_rows(R, X, G, it["split"] == "dev", uni_site)
        if do_pi:                                                                      # PLAN G4 (review C-F4)
            it["G_PI"] = pi_gain(R, X, gset("red", K, R.Lb))
    sync(R)
    it["forwards"], it["seconds"], it["done"] = R.n_fwd - f0, round(time.time() - ti, 2), True


def vbd_close(R, res_v, raws, meta, gsites, t0):
    """Dev-tuned site over the done dev items and its pass 2 on the test items (needs every item: one process)."""
    I = res_v["items"]
    items = [v for v in I.values() if v.get("done")]
    best, table = dev_tuned_site([v for v in items if v["split"] == "dev"], gsites)
    res_v["dev_tuned"] = {"site": best, "n_dev": sum(1 for v in items if v["split"] == "dev"), "mean_dm_over_D_by_site": table}
    if best is not None:
        rawmap = {x["pair_id"]: x for x in raws}

        def seq_fn(it, w):
            full = vb_item(rawmap[it["pair_id"]], meta)
            return full["SA"] if w == "A" else full["SB"]
        res_v["dev_tuned"]["pass2_rows"] = dev_tuned_pass(R, [v for v in items if v["split"] == "test"], best, seq_fn)
    res_v["n_kept"], res_v["seconds"] = len(items), round(time.time() - t0, 1)


def run_vbd(R, model, data_dir, v1, gsites, uni_site, res_v, debug, deadline, dump, role, rnd_cells, pool="s200"):
    """v5 derived-use target (plan critic 3 F5): Ouro V1 (O1 and RND at the source = prompt span + chain
    restatements, and at the read position, 12 sites, first-token e) plus V6; IFM V6 only (shared-host cap). Clean filter:
    top-1 at r is the first differing token of res_A on A and of res_B on B."""
    t0 = time.time()
    op = vbd_open(R, model, data_dir, res_v, debug, pool)
    if op is None:
        return
    raws, meta, n_target, with_v1 = op
    I, S = res_v.setdefault("items", {}), res_v.setdefault("scan", {})
    n_kept = 0
    for raw in raws:
        if n_kept >= n_target:
            break
        if deadline is not None and time.time() > deadline:
            res_v["truncated"] = True
            break
        st = S.get(raw["pair_id"])
        if st is not None and not (st["A_pass"] and st["B_pass"]):
            continue
        if (I.get(raw["pair_id"]) or {}).get("done"):
            n_kept += 1
            continue
        X, it = vbd_load(R, raw, meta, v1, gsites, role, S)
        if not X.passed:
            continue
        n_kept += 1
        pid = it["pair_id"]
        do_pi = ("VBD" in PI_DATASETS and it["split"] == "dev"
                 and sum(1 for v in I.values() if v.get("G_PI")) < PI_N)
        vbd_item_rows(R, X, it, v1, gsites, uni_site, rnd_cells, with_v1, do_pi)
        I[pid] = {k: v for k, v in it.items() if k not in STRIP}
        dump()
        print("[k17vb %s] VBD %d/%d %s (%d fwd, %.1fs)" % (model, n_kept, n_target, pid, it["forwards"], it["seconds"]),
              flush=True)
        del X
    vbd_close(R, res_v, raws, meta, gsites, t0)


def vbd_scan(R, model, data_dir, v1, gsites, res_v, debug, deadline, role, pool):
    """Stage scan (RA-20 A2 item 1): the clean-pass scan of run_vbd with its stopping rule, no rows.
    -> (kept pair_ids in pool order, complete, split by pair_id)."""
    op = vbd_open(R, model, data_dir, res_v, debug, pool)
    if op is None:
        raise SystemExit("VBD: %s" % res_v.get("error"))
    raws, meta, n_target, _ = op
    split_of = {x["pair_id"]: x["split"] for x in raws}
    I, S = res_v.setdefault("items", {}), res_v.setdefault("scan", {})
    kept = []
    for raw in raws:
        if len(kept) >= n_target:
            break
        if deadline is not None and time.time() > deadline:
            res_v["truncated"] = True
            return kept, False, split_of
        st = S.get(raw["pair_id"])
        if st is not None and not (st["A_pass"] and st["B_pass"]):
            continue
        if (I.get(raw["pair_id"]) or {}).get("done") or st is not None:
            kept.append(raw["pair_id"])
            continue
        X, _ = vbd_load(R, raw, meta, v1, gsites, role, S)
        if X.passed:
            kept.append(raw["pair_id"])
        del X
    print("[k17vb %s] VBD scan kept %d of %d scanned" % (model, len(kept), len(S)), flush=True)
    return kept, True, split_of


def shard_of(pair_id, n):
    """RA-20 A2 item 1: deterministic partition by the stable item id (independent of order and of the kept set)."""
    return int(sha(SHARD_SALT + pair_id)[:8], 16) % n


def shard_sfx(shard):
    return ".shard%dof%d" % tuple(shard) if shard else ""


def pi_set(kept, split_of, ds):
    """G_PI items of a fresh one-process run: the first PI_N dev items of the kept list (PLAN G4)."""
    if ds not in PI_DATASETS:
        return []
    return [p for p in kept if split_of[p] == "dev"][:PI_N]


def atomic_json(path, obj, **kw):
    json.dump(obj, open(path + ".tmp", "w"), **kw)
    os.replace(path + ".tmp", path)


def load_shards(out_dir, name, splan, launch):
    """Merge inputs (RA-20 A2 item 1): every shard file k of n present once, complete, same code and config as the
    scan stage and this launch, each item in its own partition. -> (n, {k: shard dict}, file names)."""
    pre = "vb_%s.shard" % name
    files = sorted(f for f in os.listdir(out_dir) if f.startswith(pre) and f.endswith(".json"))
    if not files:
        raise SystemExit("merge: no shard files %s*.json in the output folder" % pre)
    shards, ns = {}, set()
    for f in files:
        d = json.load(open(os.path.join(out_dir, f)))
        k, n = d["shard"]
        ns.add(n)
        if k in shards:
            raise SystemExit("merge: shard %d/%d appears twice" % (k, n))
        if not d.get("complete"):
            raise SystemExit("merge: shard %s is not complete" % f)
        for key in ("code_sha256", "config_sha256"):
            if d["launch"].get(key) != splan[key] or launch.get(key) != splan[key]:
                raise SystemExit("merge: %s of %s or of this launch differs from the scan stage" % (key, f))
        for ds in ("varbench", "vbd"):
            for p in d[ds + "_items"]:
                if shard_of(p, n) != k:
                    raise SystemExit("merge: item %s in shard %d/%d belongs to shard %d" % (p, k, n, shard_of(p, n)))
        shards[k] = d
    if len(ns) != 1 or set(shards) != set(range(next(iter(ns)))):
        raise SystemExit("merge: shard set %s does not cover 0..n-1 for one n %s" % (sorted(shards), sorted(ns)))
    return next(iter(ns)), shards, files


def merged_items(shards, ds, kept):
    """Union of the shard items of one dataset, asserted equal to the planned set, no duplicates; in kept order."""
    have = {}
    for k in sorted(shards):
        for p, v in shards[k][ds + "_items"].items():
            if p in have:
                raise SystemExit("merge: %s item %s in two shards" % (ds, p))
            have[p] = v
    if len(kept) != len(set(kept)) or set(have) != set(kept):
        raise SystemExit("merge: %s union %d items != planned set %d (missing %s, extra %s)" % (
            ds, len(have), len(set(kept)), sorted(set(kept) - set(have))[:5], sorted(set(have) - set(kept))[:5]))
    return [(p, have[p]) for p in kept]


# ------------------------------------------------------------------ one model
def nvsmi():
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        procs = []
        for ln in q:
            pid = ln.split(",")[0].strip()
            script = ""
            try:
                cmd = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True,
                                     timeout=10).stdout.strip()
                mt = re.search(r"([\w.-]+\.py)", cmd)
                script = mt.group(1) if mt else (os.path.basename(cmd.split()[0]) if cmd else "")
            except Exception:  # noqa: BLE001
                pass
            procs.append({"used_memory": ln.split(",")[1].strip() if "," in ln else ln, "script": script})
        return {"gpus": g, "procs": procs, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as ex:  # noqa: BLE001
        return {"error": repr(ex)}


STRIP = ("SA", "SA2", "SB", "SB2", "R2A", "R2B", "restA", "restB")


DATA_FILES = ("items_ouro.json", "items_ifm.json", "items_vbd_ouro.json", "items_vbd_ifm.json", "d4_ouro.jsonl",
              "d4_ifm.jsonl", "k16_exclusions.json")


def data_hashes(data_dir):
    """PLAN G8 (review C-F4): sha256 of every input data file, recorded in the launch record before any row."""
    out = {}
    for f in DATA_FILES:
        p = os.path.join(data_dir, f) if data_dir else None
        out[f] = hashlib.sha256(open(p, "rb").read()).hexdigest() if (p and os.path.exists(p)) else None
    return out


def plan_launch_check(name, out_dir, phase, debug, plan, plan_sha, code_path=None, data_dir=None, config=None,
                      gpu_name=None):
    """v4 (plan critic 2 F4, P14): hash the plan at launch, before any scored row. Full runs need --plan and
    --plan_sha (the hash the coordinator wrote to the LEDGER) and they must match; phase M also needs the phase-W
    launch record and an unchanged plan outside section 13. Debug runs record the hash when given. v6 (PLAN G8,
    review C-F4): the record also holds the sha256 of every data file and of the run config; phase M refuses data
    files that differ from phase W. v6 (review C-F5): the record holds the GPU name."""
    launch = {"phase": phase, "debug": bool(debug), "t": time.strftime("%Y-%m-%d %H:%M:%S"),
              "code_version": CODE_VERSION, "host_role": ROLE or role_of([name]), "gpu_name": gpu_name}
    launch.update(code_hashes(code_path))
    if data_dir is not None:
        launch["data_sha256"] = data_hashes(data_dir)
    if config is not None:
        launch["config"] = config
        launch["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if plan and os.path.exists(plan):
        launch.update(plan_hashes(plan))
    if plan_sha:
        launch["plan_sha_arg"] = plan_sha
    if debug:
        launch["code_matches_plan"] = (launch.get("plan_code_sha256") == launch["code_sha256"]) if plan else None
        return launch
    if not launch.get("plan_sha256"):
        raise SystemExit("full run needs --plan (plan critic 2 F4)")
    if not plan_sha or plan_sha != launch["plan_sha256"]:
        raise SystemExit("--plan_sha %s does not match the plan file %s" % (plan_sha, launch["plan_sha256"]))
    if launch.get("plan_code_sha256") != launch["code_sha256"]:      # v5 (plan critic 3 F1)
        raise SystemExit("code sha256 %s differs from the plan's 'Launch code sha256' %s" % (
            launch["code_sha256"], launch.get("plan_code_sha256")))
    launch["code_matches_plan"] = True
    if phase == "all":
        raise SystemExit("full runs go phase W, then the slots are appended and the plan re-hashed, then phase M")
    if phase == "M":
        path = os.path.join(out_dir, "vb_%s.json" % name)
        prev = json.load(open(path)).get("plan_launch", []) if os.path.exists(path) else []
        w = [x for x in prev if x.get("phase") == "W" and not x.get("debug")]
        if not w:
            raise SystemExit("phase M needs a phase-W launch record in vb_%s.json" % name)
        if any(x.get("plan_frozen_sha256") != launch["plan_frozen_sha256"] for x in w):
            raise SystemExit("plan changed outside section 13 since phase W")
        if any(x.get("code_sha256") != launch["code_sha256"] for x in w):
            raise SystemExit("code changed since phase W")
        if data_dir is not None and any(x.get("data_sha256") not in (None, launch["data_sha256"]) for x in w):
            raise SystemExit("data files changed since phase W (PLAN G8)")
    return launch


def host_mix_check(res, role, name, allow=False, gpu_name=None):
    """v5 (plan critic 3 F3): one host role per results file. A model moved to another host restarts in a new --out
    dir; --allow_host_mix continues (items are tagged and summarize stratifies). v6 (review C-F5): the role label is
    manual, so the GPU name of every earlier launch must also match. -> host roles seen before."""
    prev = sorted({x.get("host_role") or res.get("host_role") for x in res.get("plan_launch", [])} - {None})
    if prev and prev != [role] and not allow:
        raise SystemExit("vb_%s.json holds runs from host role %s, this launch is %s: start a new --out dir "
                         "(plan section 7 migration rule) or pass --allow_host_mix" % (name, prev, role))
    gp = sorted({x.get("gpu_name") for x in res.get("plan_launch", [])} | {res.get("gpu_name")} - {None})
    if gpu_name is not None and gp and gp != [gpu_name] and not allow:
        raise SystemExit("vb_%s.json holds runs on GPU %s, this launch runs on %s: start a new --out dir "
                         "(plan section 7 migration rule) or pass --allow_host_mix" % (name, gp, gpu_name))
    return prev


def run_model(name, data_dir, out_dir, dev, phase, debug, budget_min, datasets, n_vb=None, pool="s200", plan=None,
              plan_sha=None, stage="all", shard=None):
    """stage 'all' = one process (unchanged). RA-20 A2 item 1, cluster path (k17_public.py pattern): 'scan' = phase W
    (if asked) plus the clean-pass scans of VarBench and VBD with the one-process stopping rules, written to
    shardplan_<model>.json; 'rows' with shard (k, n) = rows of the planned items whose shard_of(pair_id, n) == k, to
    vb_<model>.shardKofN.json and states_<model>.shardKofN.pt; 'merge' = shard files into vb_<model>.json and
    states_<model>.pt (union == planned set, asserted), then the dev-tuned passes as in one process."""
    if stage != "all":
        if any(ds != "VBD" for ds in datasets):
            raise SystemExit("--stage %s takes --datasets VBD only (A2 datasets run with --stage all)" % stage)
        if stage in ("rows", "merge") and phase == "W":
            raise SystemExit("phase W runs in --stage all or --stage scan")
        if (stage == "rows") != (shard is not None):
            raise SystemExit("--shard k/n goes with --stage rows only")
    t0 = time.time()
    gpu = gpu_name_of(dev)
    run_cfg = {"model": name, "phase": phase, "debug": bool(debug), "datasets": list(datasets), "n_vb": n_vb,
               "pool": pool, "gset": GSET[name], "n_sub": N_SUB[name], "a2_n": A2_N[name], "vbd_n": VBD_N[name],
               "pi_n": PI_N, "pi_datasets": list(PI_DATASETS), "budget_min": budget_min, "drop_block": DROP_BLOCK,
               "role": ROLE or role_of([name])}
    if NO_GRAD[0]:
        run_cfg["no_gradient_rank"] = True
    launch = plan_launch_check(name, out_dir, phase, debug, plan, plan_sha, data_dir=data_dir, config=run_cfg,
                               gpu_name=gpu)
    R = load_runner(name, dev)
    load_s = time.time() - t0
    K, Lb = R.K, R.Lb
    v1 = v1_sites(K, Lb)
    gsites = gset(GSET[name], K, Lb)
    uni_site = (K, Lb // 2)
    cnames = [c.replace("@4", "@%d" % K) for c in CAND_SLOTS]
    cslots = [v1[c] for c in cnames]
    path = os.path.join(out_dir, "vb_%s.json" % name)
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.pop("host", None)
    res.pop("path", None)
    role = ROLE or role_of([name])
    prev_roles = host_mix_check(res, role, name, ALLOW_HOST_MIX, gpu_name=gpu)
    res.update({"model": name, "family": R.family, "host_role": role, "device": str(dev), "debug": debug,
                "drop_block": DROP_BLOCK, "K": K, "Lb": Lb, "gset": GSET[name], "n_gsites": len(gsites),
                "v1_sites": {k: skey(v) for k, v in v1.items()}, "uni_site": skey(uni_site),
                "cand_slots": {c: skey(v1[c]) for c in cnames}, "load_seconds": round(load_s, 1), "dtype": "bfloat16",
                "model_copy": model_copy(name, R.path),
                # rows and merge keep the scan stage's header; their launch records go to the shard / merge files
                "phase_runs": res.get("phase_runs", []) + ([] if stage in ("rows", "merge") else [phase]),
                "code_version": CODE_VERSION,
                "plan_launch": res.get("plan_launch", []) + ([] if stage in ("rows", "merge") else [launch]),
                "host_roles": sorted(set(prev_roles) | {role}),
                "gpu_name": gpu})
    res.setdefault("varbench", {"items": {}, "scan": {}})
    res.setdefault("a2", {})
    res.setdefault("nvsmi", []).append(nvsmi())
    sfx = shard_sfx(shard)                       # '' except in stage rows
    spath = os.path.join(out_dir, "states_%s%s.pt" % (name, sfx))
    states = torch.load(spath) if os.path.exists(spath) else {}
    splan_path = os.path.join(out_dir, "shardplan_%s.json" % name)
    sh_path = os.path.join(out_dir, "vb_%s%s.json" % (name, sfx))
    sh = {"model": name, "stage": "rows", "shard": list(shard) if shard else None, "launch": launch,
          "partition": "int(sha256('%s' + pair_id)[:8], 16) %% n" % SHARD_SALT, "complete": False,
          "varbench_items": {}, "vbd_items": {}}
    if stage == "rows":
        if not os.path.exists(splan_path):
            raise SystemExit("stage rows needs %s from --stage scan" % os.path.basename(splan_path))
        if os.path.exists(sh_path):              # resume: this shard's finished items
            prev_sh = json.load(open(sh_path))
            sh["varbench_items"], sh["vbd_items"] = prev_sh["varbench_items"], prev_sh["vbd_items"]
        res["varbench"]["items"].update(sh["varbench_items"])
        if sh["vbd_items"]:
            res.setdefault("vbd", {"items": {}, "scan": {}})["items"].update(sh["vbd_items"])

    def dump():
        if states:
            torch.save(states, spath + ".tmp")
            os.replace(spath + ".tmp", spath)
        if stage == "rows":                      # a shard never writes vb_<model>.json
            sh["seconds_run"], sh["forwards_run"] = round(time.time() - t0, 1), R.n_fwd
            atomic_json(sh_path, sh)
            json.dump({"t": time.time(), "model": name, "fwd": R.n_fwd, "pid": os.getpid()},
                      open(os.path.join(out_dir, "heartbeat_%s%s.json" % (name, sfx)), "w"))
            return
        res["seconds_total"] = round(time.time() - t0, 1)
        res["forwards"] = R.n_fwd
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        json.dump({"t": time.time(), "model": name, "fwd": R.n_fwd, "pid": os.getpid()},
                  open(os.path.join(out_dir, "heartbeat_%s.json" % name), "w"))
    data = load_vb(data_dir, R)
    meta = data["meta"]
    res["tokenization"] = check_tokenization(R, data, 20 if debug else 400)
    res["prefix_filter"] = {k: data["stats"].get(k) for k in ("core_rows", "core_problems", "no_restatement",
                                                              "in_annotation", "contradictory_prefix", "x_word_form",
                                                              "prose_unverified_number", "prefix_rule",
                                                              "ambiguous_source", "assert_fail", "valid_rows",
                                                              "valid_problems", "dropped_ids")}
    print("[k17vb %s] tokenization %s" % (name, res["tokenization"]), flush=True)
    if res["tokenization"]["mismatch"]:
        dump()
        raise SystemExit("tokenization mismatch")
    if pool == "s200":                       # pilot: first 200 problems in the frozen order, one pair each
        raws = [x for x in data["items"] if x["selected"]][:200]
    elif pool == "problems":                 # S1: every problem, one pair each
        raws = [x for x in data["items"] if x["selected"]]
    else:                                    # S1 option: every valid pair (cluster = problem)
        raws = list(data["items"])
    res["pool"] = pool
    n_target = n_vb or (3 if debug else N_VB[name])
    n_sub = max(1, n_target - 1) if debug else min(N_SUB[name], n_target)   # v5: debug keeps one M-only item (F6 path)
    res["config"] = {"gset": GSET[name], "n_gsites": len(gsites), "n_vb": n_target, "n_sub": n_sub,
                     "a2_n": 4 if debug else A2_N[name], "pool": pool, "datasets": list(datasets),
                     "readout": "float32 (TF32 off)", "hit_tol_D": HIT_TOL, "fn_threshold_D": FN_G,
                     "vbd_n": 2 if debug else VBD_N[name], "rnd_all_cells": RND_ALL, "code_version": CODE_VERSION}
    deadline = t0 + 60.0 * budget_min if budget_min > 0 else None
    VB = res["varbench"]
    # v5 (plan critic 3 F3, F1): hook checks rerun whenever the host role or the code hash differs from the last check
    cbr = res.setdefault("checks_by_role", {})
    rkey = "%s|%s" % (role, gpu)                  # v6 (review C-F5): checks keyed on (role, GPU name)
    prevc = cbr.get(rkey) or {}
    if not (prevc.get("all_ok") and prevc.get("code_sha256") == launch["code_sha256"]):
        first = next((x for x in raws if x["n_rest"] >= 2), raws[0])
        ck_ = model_checks(R, vb_item(first, meta), v1)
        ck_["code_sha256"] = launch["code_sha256"]
        ck_["host_role"] = role
        ck_["gpu_name"] = gpu
        cbr[rkey] = ck_
        res["checks"] = ck_
        dump()
        print("[k17vb %s] checks %s" % (name, json.dumps(ck_)), flush=True)
        if not ck_["identity_ok"]:
            raise SystemExit("hook identity failed")
    rnd_cells = None if RND_ALL else (RND_CELLS_IFM if R.family == "ifm" else None)

    def load_item(raw):
        it = vb_item(raw, meta)
        it["host_role"] = role
        X = Item(R, it, vb_rec(it["r"], v1, gsites, cslots))
        VB["scan"][raw["pair_id"]] = {"A_pass": int(X.cA["z_r"].argmax()) == X.tA,
                                      "B_pass": int(X.cB["z_r"].argmax()) == X.tB}
        prev = VB["items"].get(raw["pair_id"])
        if prev is not None:
            for k_, v_ in prev.items():
                if k_ not in it or k_ == "rows":
                    it[k_] = v_
            X.rows = it["rows"]
        return X, it

    def store(it):
        VB["items"][it["pair_id"]] = {k: v for k, v in it.items() if k not in STRIP}

    def failed_scan(raw):
        st = VB["scan"].get(raw["pair_id"])
        return st is not None and not (st["A_pass"] and st["B_pass"])
    if phase in ("W", "all") and stage in ("all", "scan"):
        n_kept, win = 0, []
        for raw in raws:
            if n_kept >= n_sub:
                break
            if failed_scan(raw):
                continue
            prev = VB["items"].get(raw["pair_id"])
            if prev is not None and prev.get("done_W"):
                n_kept += 1
                win.append(prev)
                continue
            X, it = load_item(raw)
            if not X.passed:
                continue
            n_kept += 1
            X.finish_clean()
            f0, ti = R.n_fwd, time.time()
            run_vb_item_W(R, X, v1, cslots, cnames, name)
            sync(R)
            it["seconds_W"], it["forwards_W"], it["done_W"] = round(time.time() - ti, 2), R.n_fwd - f0, True
            store(it)
            win.append(VB["items"][it["pair_id"]])
            dump()
            print("[k17vb %s] W %d/%d %s (%d fwd, %.1fs)" % (name, n_kept, n_sub, raw["pair_id"], it["forwards_W"],
                                                            it["seconds_W"]), flush=True)
            del X
        VB["window"] = select_slots(win, cnames, {c: skey(v1[c]) for c in cnames}, name, min_n=1 if debug else MIN_CL)
        json.dump(VB["window"], open(os.path.join(out_dir, "window_%s.json" % name), "w"), indent=1)
        print("[k17vb %s] window selected %s" % (name, VB["window"]["selected"]), flush=True)
        dump()
    if phase in ("M", "all"):
        if name in NO_UNDO:
            slots_sel = []
        else:
            wf = os.path.join(out_dir, "window_%s.json" % name)
            if not os.path.exists(wf):
                raise SystemExit("phase M needs window_%s.json from phase W" % name)
            slots_sel = json.load(open(wf))["selected"]
        VB["slots_used"] = slots_sel

        def m_rows(raw, X, it, do_pi):
            """Phase-M rows of one kept item (the one-process sequence; stage rows calls the same)."""
            X.finish_clean()
            o2_ok = r2_runs(R, X, v1, cslots)
            states[raw["pair_id"]] = {"sites": [skey(v1[k]) for k in v1], "tA": X.tA, "tB": X.tB, "split": it["split"],
                                      "AB": torch.stack([torch.stack([r_state(X.cA, v1[k], X.r), r_state(X.cB, v1[k], X.r)])
                                                         for k in v1]).half().cpu()}
            f0, ti = R.n_fwd, time.time()
            run_vb_item_M(R, X, v1, gsites, cslots, name, slots_sel, it["split"] == "dev", uni_site, rnd_cells, o2_ok,
                          do_pi=do_pi)
            sync(R)
            it["seconds_M"], it["forwards_M"], it["done_M"] = round(time.time() - ti, 2), R.n_fwd - f0, True
            store(it)
        n_kept = 0
        stopped = False
        splan = None
        if stage in ("rows", "merge"):
            if not os.path.exists(splan_path):
                raise SystemExit("stage %s needs %s from --stage scan" % (stage, os.path.basename(splan_path)))
            splan = json.load(open(splan_path))
            for key in ("code_sha256", "config_sha256"):
                if launch.get(key) != splan[key]:
                    raise SystemExit("stage %s: %s differs from the scan stage (same code and flags needed)" % (
                        stage, key))
            if not splan["varbench"]["complete"] or ("VBD" in datasets and not splan["vbd"]["complete"]):
                raise SystemExit("stage %s needs a complete --stage scan first" % stage)
        if stage == "all":
            for raw in raws:
                if n_kept >= n_target:
                    break
                if deadline is not None and time.time() > deadline:
                    VB["truncated"] = True
                    break
                if failed_scan(raw):
                    continue
                prev = VB["items"].get(raw["pair_id"])
                if prev is not None and prev.get("done_M"):
                    n_kept += 1
                    continue
                X, it = load_item(raw)
                if not X.passed:
                    continue
                n_kept += 1
                n_pi = sum(1 for v in VB["items"].values() if v.get("G_PI"))
                do_pi = bool("VarBench" in PI_DATASETS and it["split"] == "dev" and n_pi < PI_N)
                m_rows(raw, X, it, do_pi)
                dump()
                print("[k17vb %s] M %d/%d %s (%d fwd, %.1fs)" % (name, n_kept, n_target, raw["pair_id"],
                                                                it["forwards_M"], it["seconds_M"]), flush=True)
                del X
                if n_kept % 25 == 0:
                    res["nvsmi"].append(nvsmi())
        elif stage == "scan":                # the M loop's clean-pass scan and stopping rule, no rows
            kept, complete = [], True
            for raw in raws:
                if len(kept) >= n_target:
                    break
                if deadline is not None and time.time() > deadline:
                    VB["truncated"] = True
                    complete = False
                    break
                if failed_scan(raw):
                    continue
                pid = raw["pair_id"]
                prev = VB["items"].get(pid)
                if (prev is not None and prev.get("done_M")) or pid in VB["scan"]:   # passed in phase W or before
                    kept.append(pid)
                    continue
                X, _ = load_item(raw)
                if X.passed:
                    kept.append(pid)
                del X
            split_of = {x["pair_id"]: x["split"] for x in raws}
            splan = {"model": name, "code_sha256": launch["code_sha256"], "config_sha256": launch["config_sha256"],
                     "partition": sh["partition"], "pool": pool, "datasets": list(datasets),
                     "varbench": {"n_target": n_target, "kept": kept, "pi": pi_set(kept, split_of, "VarBench"),
                                  "complete": complete}}
            print("[k17vb %s] VarBench scan kept %d of %d scanned" % (name, len(kept), len(VB["scan"])), flush=True)
            dump()
        elif stage == "rows":
            k_sh, n_sh = shard
            rawmap = {x["pair_id"]: x for x in raws}
            pi_vb = set(splan["varbench"]["pi"])
            mine = [p for p in splan["varbench"]["kept"] if shard_of(p, n_sh) == k_sh]
            for i, pid in enumerate(mine):
                if pid in sh["varbench_items"]:
                    continue
                if deadline is not None and time.time() > deadline:
                    stopped = True
                    break
                X, it = load_item(rawmap[pid])
                if not X.passed:
                    raise SystemExit("clean status changed between scan and rows: %s %s" % (name, pid))
                m_rows(rawmap[pid], X, it, pid in pi_vb)
                sh["varbench_items"][pid] = VB["items"][pid]
                dump()
                print("[k17vb %s] M shard %d/%d item %d/%d %s (%d fwd, %.1fs)" % (
                    name, k_sh, n_sh, i + 1, len(mine), pid, it["forwards_M"], it["seconds_M"]), flush=True)
                del X
        else:                                # merge
            n_sh, shards, files = load_shards(out_dir, name, splan, launch)
            kept = splan["varbench"]["kept"]
            for pid, v in merged_items(shards, "varbench", kept):
                VB["items"][pid] = v
            st_all = {}
            for k_ in range(n_sh):
                sp_ = os.path.join(out_dir, "states_%s%s.pt" % (name, shard_sfx((k_, n_sh))))
                for pid, v in (torch.load(sp_) if os.path.exists(sp_) else {}).items():
                    if pid in st_all:
                        raise SystemExit("merge: states of %s in two shards" % pid)
                    st_all[pid] = v
            if set(st_all) != set(kept):
                raise SystemExit("merge: states union %d != planned set %d" % (len(st_all), len(kept)))
            for pid in kept:                 # interned keys: the pickle then matches the one-process file byte for byte
                states[pid] = {sys.intern(k): v for k, v in st_all[pid].items()}
            n_kept = len(kept)
            mrec = {"model": name, "stage": "merge", "launch": launch, "partition": splan["partition"],
                    "n_shards": n_sh, "shard_files": files, "shard_launches": [shards[k]["launch"] for k in sorted(shards)],
                    "varbench": {"planned": len(kept), "merged": len(kept), "union_equals_planned": True,
                                 "duplicates": 0}}
        if stage in ("all", "merge"):
            VB["n_target"], VB["n_kept"] = n_target, n_kept
            items = [v for v in VB["items"].values() if v.get("done_M")]
            best, table = dev_tuned_site([v for v in items if v["split"] == "dev"], gsites)
            VB["dev_tuned"] = {"site": best, "n_dev": sum(1 for v in items if v["split"] == "dev"), "mean_dm_over_D_by_site": table}
            if best is not None:
                rawmap = {x["pair_id"]: x for x in raws}

                def seq_fn(it, w):
                    full = vb_item(rawmap[it["pair_id"]], meta)
                    return full["SA"] if w == "A" else full["SB"]
                VB["dev_tuned"]["pass2_rows"] = dev_tuned_pass(R, [v for v in items if v["split"] == "test"], best, seq_fn)
            dump()
        excl = json.load(open(os.path.join(data_dir, "k16_exclusions.json")))
        for ds in datasets:
            if ds == "VBD":                  # v5 (plan critic 3 F5): derived-use target
                res.setdefault("vbd", {"items": {}, "scan": {}})
                rv = res["vbd"]
                if stage == "all":
                    run_vbd(R, name, data_dir, v1, gsites, uni_site, rv, debug, deadline, dump, role, rnd_cells, pool)
                elif stage == "scan":
                    kv, cv, split_v = vbd_scan(R, name, data_dir, v1, gsites, rv, debug, deadline, role, pool)
                    splan["vbd"] = {"n_target": rv["config"]["n_target"], "kept": kv, "pi": pi_set(kv, split_v, "VBD"),
                                    "complete": cv}
                else:
                    t_v = time.time()
                    op = vbd_open(R, name, data_dir, rv, debug, pool)
                    if op is None:
                        raise SystemExit("VBD: %s" % rv.get("error"))
                    raws_v, meta_v, _, with_v1 = op
                    if stage == "rows":
                        rmap = {x["pair_id"]: x for x in raws_v}
                        pi_v = set(splan["vbd"]["pi"])
                        mine = [p for p in splan["vbd"]["kept"] if shard_of(p, shard[1]) == shard[0]]
                        S_, I_ = rv.setdefault("scan", {}), rv.setdefault("items", {})
                        for i, pid in enumerate(mine):
                            if pid in sh["vbd_items"] or stopped:
                                continue
                            if deadline is not None and time.time() > deadline:
                                stopped = True
                                break
                            X, it = vbd_load(R, rmap[pid], meta_v, v1, gsites, role, S_)
                            if not X.passed:
                                raise SystemExit("clean status changed between scan and rows: VBD %s %s" % (name, pid))
                            vbd_item_rows(R, X, it, v1, gsites, uni_site, rnd_cells, with_v1, pid in pi_v)
                            I_[pid] = {k: v for k, v in it.items() if k not in STRIP}
                            sh["vbd_items"][pid] = I_[pid]
                            dump()
                            print("[k17vb %s] VBD shard %d/%d item %d/%d %s (%d fwd, %.1fs)" % (
                                name, shard[0], shard[1], i + 1, len(mine), pid, it["forwards"], it["seconds"]),
                                flush=True)
                            del X
                    else:                    # merge
                        kv = splan["vbd"]["kept"]
                        for pid, v in merged_items(shards, "vbd", kv):
                            rv["items"][pid] = v
                        mrec["vbd"] = {"planned": len(kv), "merged": len(kv), "union_equals_planned": True,
                                       "duplicates": 0}
                        vbd_close(R, rv, raws_v, meta_v, gsites, t_v)
                dump()
                continue
            res["a2"].setdefault(ds, {})
            run_a2(R, name, ds, data_dir, gsites, uni_site, res["a2"][ds], debug, deadline, excl, dump)
            dump()
        if stage == "scan":
            atomic_json(splan_path, splan, indent=1)
        elif stage == "rows":
            sh["complete"] = not stopped
        elif stage == "merge":
            atomic_json(os.path.join(out_dir, "merge_%s.json" % name), mrec, indent=1)
    res["nvsmi"].append(nvsmi())
    dump()
    print("[k17vb %s] done %.0fs, %d forwards" % (name, time.time() - t0, R.n_fwd), flush=True)


def select_slots(items, cnames, ckeys, model, min_n=MIN_CL):
    """Frozen V5 rule (plan section 4, v5) on the phase-W subset. Debug runs pass min_n = 1 (code path only)."""
    def mA(it):
        return it["clean_A"][1] - it["clean_A"][0]

    def m(it, key):
        rv = it["rows"].get(key)
        return None if rv is None else rv["f"][1] - rv["f"][0]

    def dep(it, key):
        return dep_val(m(it, key), mA(it))          # review MS6: None when -m_A < GAP

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None
    out = {"n_items": len(items), "dep": {}, "slots": {}, "selected": [], "rule": "plan section 4 V5",
           "dep_excluded_small_margin": sum(1 for it in items if -mA(it) < GAP)}
    for var in ("dir", "rel"):
        for bn in WIN_BOUNDS:
            out["dep"]["%s|%s" % (var, bn)] = r5(mean([dep(it, "KO|%s|%s" % (var, bn)) for it in items]))
    out["dep"]["rand_rel|full"] = r5(mean([dep(it, "KOR|rel|full") for it in items]))
    if model in NO_UNDO:
        out["note"] = "no undo on the injection twin (A5)"
        return out
    full = out["dep"]["rel|full"]
    out["rule"] = ("plan section 4 V5 v6: (a) influence removal >= %.2f; b2 carried / D_KO >= %.2f with 95%% CI lower > "
                   "%.2f on items with D_KO >= %g (review C-F1); (a), b2 each need >= %d items and clusters; carried / D, "
                   "D_KO / D and necessity (v4 clause a) reported" % (REMOVE_MIN, CARRY_MIN, CARRY_LO, DKO_GUARD, min_n))
    for cn in cnames:
        da = out["dep"].get("rel|after_%s" % cn)
        pres = mean([((it["rows"].get("O1|src|Llast@1|1") or {}).get("pres") or {}).get(ckeys[cn]) for it in items])
        ec = []
        for it in items:
            u, b, k = m(it, "UKO|%s" % cn), m(it, "BKO|%s" % cn), m(it, "KO|rel|after_%s" % cn)
            if None in (u, b, k) or (b - k) < GAP:
                continue
            ec.append((u - k) / (b - k))
        st = slot_stats(items, cn)
        rule = window_rule(st["removal"], st["carried_over_DKO"], pres, st["n_items"], st["n_clusters"], min_n=min_n,
                           n_b2=st["n_b2"], ncl_b2=st["n_clusters_b2"])
        out["slots"][cn] = dict(rule, removal=st["removal"], carried_over_DKO=st["carried_over_DKO"],
                                carried_over_D=st["carried_over_D"], DKO_over_D=st["DKO_over_D"], n_b2=st["n_b2"],
                                n_clusters_b2=st["n_clusters_b2"], n_dko_below_guard=st["n_dko_below_guard"],
                                survival_subset=st["survival"], n_items=st["n_items"], n_clusters=st["n_clusters"],
                                presence=r5(pres), necessity_dep_rel_after=r5(da), necessity_dep_rel_full=r5(full),
                                v4_rule=window_rule_v4(full, da, pres, mean(ec), len(ec)),
                                e_carried_v4=r5(mean(ec)), e_carried_v4_n=len(ec))
        if rule["inside"]:
            out["selected"].append(cn)
    # only (a) and b2 slots can show a rebuild; b1-only slots are run but uninformative (plan critic 2 F3)
    out["selected_informative"] = [cn for cn in out["selected"] if out["slots"][cn]["informative"]]
    return out


def cpu_check(name, data_dir, out_dir, dev="cpu"):
    t0 = time.time()
    R = load_runner(name, dev)
    data = load_vb(data_dir, R)
    tk = check_tokenization(R, data, 20)
    raws = [x for x in data["items"] if x["selected"]]
    first = next(x for x in raws if x["n_rest"] >= 2)
    ck = model_checks(R, vb_item(first, data["meta"]), v1_sites(R.K, R.Lb))
    out = {"model": name, "device": dev, "host_role": ROLE or role_of([name]), "drop_block": DROP_BLOCK, "tokenization": tk,
           "checks": ck, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(out_dir, "check_%s_%s.json" % (str(dev).split(":")[0], name)), "w"), indent=1)
    print(json.dumps(out), flush=True)
    return bool(ck["all_ok"] and not tk["mismatch"])


def fd_summary(recs, a_pass=1e-2, floor=FD_FLOOR):
    """FD rule (plan section 4 V6, v5, same form as B-K17-CGCHECK H11): at a = a_pass, over (item, site) with
    |a G| >= floor, median |dm - a G| / |a G| <= 0.05 and sign agreement >= 0.95. recs: dicts with a, aG, dm."""
    out = {}
    for a in sorted({x["a"] for x in recs}):
        xs = [x for x in recs if x["a"] == a and abs(x["aG"]) >= floor]
        rel = sorted(abs(x["dm"] - x["aG"]) / abs(x["aG"]) for x in xs)
        sg = [int((x["dm"] > 0) == (x["aG"] > 0)) for x in xs]
        out["a=%g" % a] = {"n_scored": len(xs), "n_below_floor": sum(1 for x in recs if x["a"] == a) - len(xs),
                           "median_rel_err": round(rel[len(rel) // 2], 5) if rel else None,
                           "p90_rel_err": round(rel[min(len(rel) - 1, int(0.9 * len(rel)))], 5) if rel else None,
                           "sign_agree": round(sum(sg) / len(sg), 4) if sg else None}
    p = out.get("a=%g" % a_pass) or {}
    out["pass"] = bool(p.get("n_scored") and p["median_rel_err"] <= 0.05 and p["sign_agree"] >= 0.95)
    out["rule"] = "a = %g: median rel err <= 0.05 and sign agreement >= 0.95 on |aG| >= %g" % (a_pass, floor)
    return out


def cp_lower(k, n, alpha=0.05):
    """Clopper-Pearson two-sided (1 - alpha) lower bound of a binomial rate k / n (bisection on the tail)."""
    if n == 0 or k == 0:
        return 0.0

    def tail(p):                      # P(X >= k | n, p)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    lo, hi = 0.0, k / n
    for _ in range(60):
        mid = (lo + hi) / 2
        if tail(mid) < alpha / 2:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def bf16_agree(pairs, K=None, min_items=0):
    """bf16 vs fp32 G. pairs: {pid: {"D", "G_fp32", "G_bf16"}}. v5.1: the selector check is the T7 tolerance band,
    not exact top-site equality: per item with D >= GAP, the fp32 G at the bf16 top site is within HIT_TOL x D of the
    fp32 maximum. v6 (review A-F3): the same band for the argmax inside loops 1..K-1 (site keys 'j.l' with j < K; the
    'A2-1 ltK' row's selector), a Clopper-Pearson lower bound per band, and min_items items with D >= GAP.
    ok = n_D >= min_items, band agreement >= 0.9 (full set and loops < K) and FN-class agreement >= 0.95."""
    diff, top_same, absD, fn_same, band, band_l = [], [], [], [], [], []
    for pid, rec in sorted(pairs.items()):
        G, D, Gb = rec["G_fp32"], rec["D"], rec.get("G_bf16")
        if not Gb:
            continue
        for k, v in G.items():
            if abs(v) > 1e-6:
                diff.append(abs(Gb[k] - v) / abs(v))
            if D >= GAP:
                absD.append(abs(Gb[k] - v) / D)
                fn_same.append(int((abs(Gb[k]) < FN_G * D) == (abs(v) < FN_G * D)))
        t32, t16 = max(G, key=lambda k: (G[k], k)), max(Gb, key=lambda k: (Gb[k], k))
        top_same.append(int(t32 == t16))
        if D >= GAP:
            band.append(int(G[t16] >= G[t32] - HIT_TOL * D))
            if K is not None:
                lk = [k for k in G if ksite(k)[0] < K]
                if lk:
                    l32, l16 = max(lk, key=lambda k: (G[k], k)), max(lk, key=lambda k: (Gb[k], k))
                    band_l.append(int(G[l16] >= G[l32] - HIT_TOL * D))
    diff.sort()
    absD.sort()

    def q(xs, f):
        return round(xs[min(len(xs) - 1, int(f * len(xs)))], 5) if xs else None
    st = {"n": len(diff), "median_rel": q(diff, 0.5), "p90_rel": q(diff, 0.9),
          "abs_over_D_median": q(absD, 0.5), "abs_over_D_p90": q(absD, 0.9),
          "fn_class_agree": round(sum(fn_same) / len(fn_same), 4) if fn_same else None, "fn_n": len(fn_same),
          "top_site_same": [sum(top_same), len(top_same)], "top_in_fp32_band": [sum(band), len(band)],
          "top_in_fp32_band_cp95_lower": cp_lower(sum(band), len(band)),
          "band": "fp32 G at the bf16 top site >= fp32 max G - %g D (items with D >= %g)" % (HIT_TOL, GAP),
          "min_items": min_items}
    okb = bool(band and len(band) >= min_items and sum(band) / len(band) >= BF16_BAND_MIN)
    if K is not None:
        st["loops_lt_K_top_in_fp32_band"] = [sum(band_l), len(band_l)]
        st["loops_lt_K_cp95_lower"] = cp_lower(sum(band_l), len(band_l))
        okb = okb and bool(band_l and sum(band_l) / len(band_l) >= BF16_BAND_MIN)
    ok = bool(okb and st["fn_class_agree"] is not None and st["fn_class_agree"] >= FN_AGREE_MIN)
    return st, ok


class Fp32Strict(object):
    """Exact fp32 for a whole pass: TF32 off for matmul and cudnn, precision 'highest'; restored on exit (v5.1)."""
    def __enter__(self):
        self.prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                     torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return self

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.prev[0], self.prev[1]
        torch.set_float32_matmul_precision(self.prev[2])
        return False


def bf16_check(name, data_dir, out_dir, dev, n=BF16_N, fd_dir=""):
    """v6 (review A-F3): the V6 gate's bf16 selector condition, scored on fresh items. The model's run set (GSET:
    G_full on the 1.4B models), the first n VarBench S200 items that pass the clean filter and are not FD items
    (pair_ids from fd_<model>.json in fd_dir or out_dir), one clean A/B pass and one forward + backward per item per
    precision (bf16 as the run; fp32 with TF32 off). Rule: bf16_agree(K, min_items = n). -> bf16_<model>.json."""
    t0 = time.time()
    fd_ids, fd_src = set(), None
    for dd in [x for x in (fd_dir, out_dir) if x]:
        fp = os.path.join(dd, "fd_%s.json" % name)
        if os.path.exists(fp):
            fd_ids, fd_src = set(json.load(open(fp)).get("pair_ids") or []), os.path.basename(fp)
            break
    out = {"model": name, "host_role": ROLE or role_of([name]), "n_target": n, "gset": GSET[name],
           "fresh_vs_fd": {"fd_pair_ids_excluded": sorted(fd_ids), "fd_file": fd_src}}
    G16, D16, pids, K = {}, {}, None, None
    G32, D32 = {}, {}
    for prec in ("bf16", "fp32"):
        R = load_runner(name, dev, fp32=(prec == "fp32"))
        out["dtype_%s" % prec] = str(R.adt)
        ctx = Fp32Strict() if prec == "fp32" else None
        if ctx:
            ctx.__enter__()
        try:
            out["gpu_name"] = gpu_name_of(dev)
            K, Lb = R.K, R.Lb
            sites = gset(GSET[name], K, Lb)
            out["sites"] = {"n": len(sites), "set": GSET[name]}
            data = load_vb(data_dir, R)
            meta = data["meta"]
            raws = [x for x in data["items"] if x["selected"]][:200]
            raws = [x for x in raws if x["pair_id"] not in fd_ids]
            if pids is not None:
                raws = [x for x in raws if x["pair_id"] in pids]
            got, scanned = [], 0
            for raw in raws:
                if pids is None and len(got) >= n:
                    break
                it = vb_item(raw, meta)
                X = Item(R, it, {s: [it["r"]] for s in sites})
                scanned += 1
                if pids is None and not X.passed:
                    continue
                G = carry_gain(R, X, sites)
                got.append(raw["pair_id"])
                (G16 if prec == "bf16" else G32)[raw["pair_id"]] = G
                (D16 if prec == "bf16" else D32)[raw["pair_id"]] = Dgap(it)
                X = None
            if pids is None:
                pids = set(got)
                out["pair_ids"], out["scanned"] = got, scanned
            out["final_hidden_dtype_%s" % prec] = getattr(R, "last_final_dtype", str(R.adt))
            out["seconds_%s" % prec] = round(time.time() - t0, 1)
        finally:
            if ctx:
                ctx.__exit__()
        R = None
        gc_cuda()
    pairs = {pid: {"D": D32[pid], "D_bf16": D16.get(pid), "G_fp32": G32[pid], "G_bf16": G16.get(pid)} for pid in G32}
    st, ok = bf16_agree(pairs, K=K, min_items=n)
    out.update({"G_pairs": pairs, "stats": st, "ok": ok, "n_items": len(pairs),
                "n_items_D_ge_gap": sum(1 for v in pairs.values() if v["D"] >= GAP),
                "rule": "fresh items (not FD), run set; fp32 G at the bf16 top site within %g D of the fp32 max on >= %g "
                        "of items (full set and loops < K), FN class agreement >= %g, >= %d items with D >= %g" % (
                            HIT_TOL, BF16_BAND_MIN, FN_AGREE_MIN, n, GAP),
                "seconds": round(time.time() - t0, 1)})
    json.dump(out, open(os.path.join(out_dir, "bf16_%s.json" % name), "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "G_pairs"}), flush=True)
    return out


def cgcheck_gate(cgc, sha_):
    """v6 (review A-F6): the CGCHECK part of the V6 gate opens only on the full-run checks.json: verdict PASS, no
    failed or insufficient cell, and sha256 equal to CGCHECK_SHA256 (the debug checks.json also says PASS)."""
    if cgc is None:
        return {"pass": None, "note": "no --cgcheck file"}
    pv = cgc.get("pass")
    verdict = pv is True or pv == "PASS" or (isinstance(pv, dict) and pv.get("verdict") == "PASS")
    fails = (pv.get("failed_cells") if isinstance(pv, dict) else None) or []
    insuf = (pv.get("insufficient_cells") if isinstance(pv, dict) else None) or []
    sha_ok = sha_ == CGCHECK_SHA256
    return {"pass": bool(verdict and not fails and not insuf and sha_ok), "verdict_pass": bool(verdict),
            "failed_cells": fails, "insufficient_cells": insuf, "sha256": sha_, "sha256_expected": CGCHECK_SHA256,
            "sha256_ok": sha_ok}


def gpu_name_of(dev):
    """GPU model name for the launch record and the host-role key (v6, review C-F5); 'cpu' off GPU."""
    return torch.cuda.get_device_name(0) if (str(dev).startswith("cuda") and torch.cuda.is_available()) else "cpu"


def fd_check(name, data_dir, out_dir, dev, n=FD_N):
    """v5 (plan critic 3 F4): per-family carry-gain check on VarBench items. Pass 1 (bf16 model, as the run): G at
    the G_red sites. Pass 2 (fp32 model): G in fp32 and the exact O1 dm(a) at a in FD_A, compared with a G. Also
    the bf16-vs-fp32 G difference and whether the bf16 top site equals the fp32 top site. -> fd_<model>.json."""
    t0 = time.time()
    out = {"model": name, "host_role": ROLE or role_of([name]), "n_target": n, "sites": "G_red", "a": list(FD_A)}
    G16, pids, recs, g32 = {}, None, [], {}
    for prec in ("bf16", "fp32"):
        R = load_runner(name, dev, fp32=(prec == "fp32"))
        out["dtype_%s" % prec] = str(R.adt)
        tf32_prev = None
        if prec == "fp32":
            # v5.1: k10_ifm.load_ifm turns TF32 on, so the IFM "fp32" body ran its matmuls in TF32 (10-bit mantissa)
            # and the FD dm at small a was rounding noise (v5 debug: IFM fail, Ouro pass). Exact fp32 for this pass.
            tf32_prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                         torch.get_float32_matmul_precision())
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            out["tf32_fp32_pass"] = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                     "before": list(tf32_prev)}
        K, Lb = R.K, R.Lb
        sites = gset("red", K, Lb)
        data = load_vb(data_dir, R)
        meta = data["meta"]
        raws = [x for x in data["items"] if x["selected"]][:200]
        if pids is not None:
            raws = [x for x in raws if x["pair_id"] in pids]
        got = []
        for raw in raws:
            if pids is None and len(got) >= n:
                break
            it = vb_item(raw, meta)
            X = Item(R, it, {s: [it["r"]] for s in sites})
            if pids is None and not X.passed:
                continue
            G = carry_gain(R, X, sites)
            got.append(raw["pair_id"])
            if prec == "bf16":
                G16[raw["pair_id"]] = G
                continue
            mA = float(X.cA["z_r"][X.tB] - X.cA["z_r"][X.tA])
            g32[raw["pair_id"]] = (G, float(X.cB["z_r"][X.tB] - X.cB["z_r"][X.tA]) - mA)
            rep = R.run(X.seqA, X.r)
            out["final_hidden_dtype_fp32"] = getattr(R, "last_final_dtype", str(R.adt))
            noise = abs(float(rep["z_r"][X.tB] - rep["z_r"][X.tA]) - mA)
            for s in sites:
                hb = r_state(X.cB, s, X.r)[None]
                for a in FD_A:
                    o = R.run(X.seqA, X.r, edits={s: [e_lerp([X.r], hb, a, R.dev)]})
                    dm = float(o["z_r"][X.tB] - o["z_r"][X.tA]) - mA
                    recs.append({"pair_id": raw["pair_id"], "site": skey(s), "a": a, "aG": a * G[skey(s)], "dm": dm,
                                 "noise": noise})
        if pids is None:
            pids = set(got)
            out["pair_ids"] = got
        out["load_and_run_seconds_%s" % prec] = round(time.time() - t0, 1)
        if tf32_prev is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32_prev[0], tf32_prev[1]
            torch.set_float32_matmul_precision(tf32_prev[2])
        X = o = rep = None   # noqa: F841  (drop references to the runner before freeing it)
        del R
        gc_cuda()
    out["records"] = recs
    out["fd"] = fd_summary(recs)
    out["noise_floor_max"] = max((x["noise"] for x in recs), default=None)
    out["G_pairs"] = {pid: {"D": D, "G_fp32": G, "G_bf16": G16.get(pid)} for pid, (G, D) in g32.items()}
    out["bf16_vs_fp32_G"], out["bf16_selector_ok"] = bf16_agree(out["G_pairs"])
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(out_dir, "fd_%s.json" % name), "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "records"}), flush=True)
    return out


def gc_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ summary
def pos_of(key):
    """Edited position of a row key: 'O1|ans|Lmid@2|1' -> 'ans'. Undo, knockout and PC-site rows are source-family."""
    p = key.split("|")
    return p[1] if len(p) > 1 and p[1] in ("src", "ans") else "src"


def primary_f(key):
    """v3 endpoint rule (review MS1/F1, plan section 5): answer-position rows -> first-token e. The edit at r directly
    controls only the token at r; the continuation positions on SA / SA' still read A's prompt span, so e_LO of an
    answer edit is capped at D / D_LO per item. Source rows, undo rows and SZ -> full-span e_LO."""
    return "e" if pos_of(key) == "ans" else "e_LO"


def e_row(it, key, lo=False):
    rv = it["rows"].get(key)
    if rv is None:
        return None
    if lo:
        if rv.get("lo") is None or it.get("LO_A") is None:
            return None
        D = it["LO_B"] - it["LO_A"]
        return None if D < GAP else (rv["lo"] - it["LO_A"]) / D
    D = Dgap(it)
    return None if D < GAP else dm_of(it, key) / D


def e_lo_rel(it, key):
    """Answer rows: dLO / D. Per item this equals e_LO / e_LO(PC-final), since dLO(PC-final) = D exactly."""
    rv = it["rows"].get(key)
    if rv is None or rv.get("lo") is None or it.get("LO_A") is None:
        return None
    D = Dgap(it)
    return None if D < GAP else (rv["lo"] - it["LO_A"]) / D


def prim_row(it, key):
    """Per-item value of the key's primary endpoint (source rows without a full-span value fall back to e)."""
    rv = it["rows"].get(key)
    if rv is None:
        return None
    if primary_f(key) == "e_LO" and "lo" in rv:
        return e_row(it, key, lo=True)
    return e_row(it, key)


def flip_row(it, key):
    rv = it["rows"].get(key)
    return None if rv is None else int(rv["f"][3] == it["tB"])


def cell(items, key, cl, null_cited=False):
    """One cell: e_LO (full span) and e (first token), flip, dm, dz(A), dz(B), n_items, n_clusters; the primary
    endpoint follows primary_f(key); TOST on the primary endpoint for null-cited cells. Answer cells also carry
    e_LO_over_pc = dLO / D (the full-span effect relative to its per-item ceiling)."""
    if not any(key in it["rows"] for it in items):
        return None
    xl = [e_row(it, key, lo=True) for it in items]
    xe = [e_row(it, key) for it in items]
    use_lo = primary_f(key) == "e_LO" and any(x is not None for x in xl)
    prim = xl if use_lo else xe
    out = {"endpoint": "e_LO" if use_lo else "e", "n_items": sum(1 for it in items if key in it["rows"]),
           "e_LO": boot(xl, cl), "e": boot(xe, cl), "flip": boot([flip_row(it, key) for it in items], cl),
           "dm": boot([dm_of(it, key) for it in items], cl)}
    out["primary"] = out[out["endpoint"]]
    if pos_of(key) == "ans":
        out["e_LO_over_pc"] = boot([e_lo_rel(it, key) for it in items], cl)
    rvs = [it["rows"].get(key) for it in items]
    out["dzA"] = boot([None if rv is None else rv["f"][0] - it["clean_A"][0] for rv, it in zip(rvs, items)], cl)
    out["dzB"] = boot([None if rv is None else rv["f"][1] - it["clean_A"][1] for rv, it in zip(rvs, items)], cl)
    out["n_clusters"] = len(set(c for rv, c in zip(rvs, cl) if rv is not None))
    if null_cited:
        out["tost"] = tost(prim, cl)
    return out


def t3_pass(c):
    """T3 on a RND cell: |mean primary| <= 0.05 and flip <= 0.02."""
    if not c or not c.get("primary") or not c.get("flip"):
        return None
    return bool(abs(c["primary"][0]) <= 0.05 and c["flip"][0] <= 0.02)


def rnd_pair(items, key, cl):
    """Citation rule T3b, v6 (review C-F2; replaces the v5 subtraction). For an O1 cell, the RND cell at the same site
    and position: its T3 verdict, dz(A) and dz(B), and the paired per-item difference e(O1) - e(RND) (cluster CI; on
    e_LO where both rows have it, else first-token e). The cited value is always O1: O1 at a = 1 sets the state to
    clean B's own state, so it has no random-damage component to subtract. RND is a gate: where it fails T3, a
    lower-bound ('works', 'carried', working-site) claim also needs the O1 - RND CI lower bound > 0 (`lower_gate`);
    upper-bound, null and 'not carried' claims cite O1 alone with the label 'RND fails T3' (damage moves the margin
    toward the tie, i.e. raises e). Where RND did not run the cell is not citable.
    -> (record, per-item cited series = O1, cited endpoint)."""
    rk = key.replace("O1|", "RND|", 1)
    o1 = [prim_row(it, key) for it in items]
    if not any(rk in it["rows"] for it in items):
        return ({"rnd_run": False, "cite": "not citable (RND not run at site)", "citable": False, "lower_gate": None},
                o1, primary_f(key))
    rc = cell(items, rk, cl)
    t3 = t3_pass(rc)
    lo_ok = primary_f(key) == "e_LO" and any(e_row(it, rk, lo=True) is not None for it in items)
    ep = "e_LO" if lo_ok else "e"

    def d(lo):
        out = []
        for it in items:
            a, b = e_row(it, key, lo=lo), e_row(it, rk, lo=lo)
            out.append(None if (a is None or b is None) else a - b)
        return out
    de, dl = d(False), (d(True) if lo_ok else None)
    rec = {"rnd_run": True, "rnd_T3": t3, "rnd_endpoint": rc["endpoint"], "rnd_primary": rc["primary"],
           "rnd_flip": rc["flip"], "rnd_dzA": rc["dzA"], "rnd_dzB": rc["dzB"], "diff_e": boot(de, cl),
           "diff_e_LO": boot(dl, cl) if dl else None, "diff_endpoint": ep,
           "cite": "O1" if t3 else "O1 (RND fails T3: lower bounds gated on O1 - RND)", "citable": True}
    rec["diff_primary"] = rec["diff_e_LO"] if lo_ok else rec["diff_e"]
    dp = rec["diff_primary"]
    rec["lower_gate"] = True if t3 else bool(dp and dp[1] is not None and dp[1] > 0)
    rec["rule"] = "T3b v6 (review C-F2): cite O1; RND gates lower-bound claims only"
    return rec, o1, primary_f(key) if any(x is not None for x in o1) else "e"


def cited_value(c):
    """The value a prediction or class cites for an O1 cell (T3b v6, rnd_pair): always the O1 primary (where RND ran).
    -> ([mean, lo, hi, ...] or None, label, lower_gate)."""
    rp = (c or {}).get("rnd_pair") or {}
    if not c:
        return None, None, None
    if not rp.get("citable"):
        return None, rp.get("cite"), None
    return c.get("primary"), rp.get("cite"), rp.get("lower_gate")


def answer_class(items, key, j, family, gate, rnd_fail=False, lower_gate=None):
    """Answer-cell class (plan section 5, F5). Ouro: e_own(j) = (m_edit(j) - m_A(j)) / D read at the edited loop's
    own exit (final D); 'not carried' needs e_own >= 0.5 and final e / e_own < 0.25. IFM / d112: the R2 presence
    gate stands in for e_own (the edited state is B at the site by construction). v6 (review C-F2, A-F5): final e and
    e_own are both O1 values (the v5 rule subtracted RND from final e only, which mixed two bases). Where the RND cell
    fails T3 (rnd_fail) the record carries that label, and 'carried' (a lower bound) is citable only if the O1 - RND
    CI lower bound is > 0 (lower_gate)."""
    fe_l, eo_l = [], []
    for it in items:
        rv = it["rows"].get(key)
        D = Dgap(it)
        if rv is None or D < GAP:
            continue
        x = dm_of(it, key) / D
        fe_l.append(x)
        if family == "ouro" and "L" in rv:
            zA, zB = rv["L"][j - 1][0], rv["L"][j - 1][1]
            cA_ = it["clean_A_loops"][j - 1]
            eo_l.append(((zB - zA) - (cA_[1] - cA_[0])) / D)
    if not fe_l:
        return None
    fe = sum(fe_l) / len(fe_l)
    rec = {"n": len(fe_l), "final_e": r5(fe)}
    if family == "ouro":
        eo = sum(eo_l) / len(eo_l) if eo_l else None
        rec["e_own"] = r5(eo)
        present = eo is not None and eo >= 0.5
        rec["ratio"] = r5(fe / eo) if present else None
        notc = present and fe / eo < 0.25
    else:
        rec["presence_gate"] = bool(gate)
        present = bool(gate)
        notc = present and fe < 0.25
    rec["class"] = ("carried" if fe >= 0.5 else ("not carried" if notc else ("partial" if present else GATE_FAILS)))
    rec["final_e_basis"] = "O1 (RND fails T3 at site)" if rnd_fail else "O1"
    rec["rnd_fails_T3"] = bool(rnd_fail)
    if rec["class"] == "carried" and rnd_fail:
        rec["carried_rnd_gate"] = lower_gate
    return rec


# v4 (plan critic 2 F5): a failed gate cannot tell an absent answer state from one the rest of loop j overwrites
GATE_FAILS = "gate fails (absent or overwritten within loop j)"
PROBE_PRESENT, PROBE_ABSENT, PROBE_PC = 0.75, 0.60, 0.80


def probe_table(states, items, v1, K, cl_of=lambda it: it["cluster"]):
    """Parameter-free digit probe (plan critic 2 F5/F7) on the stored clean A and B states at r, every V1 site.
    Labels: the digits of x and x' at their first differing character (items where they differ). Positive control
    (the known-present case): the readout-feeding state Llast@K, from which the clean model reads tA / tB.
    v5 (plan critic 3 F1): a planted-digit power control at EVERY probed site: each state gets a fixed random unit
    code of its own digit (seed 'k17vb-plant:<digit>') scaled to PLANT_F x its item's |h_A - h_B| at that site, and
    the same probe runs on the planted states. Labels: 'present' (acc >= 0.75); 'absent' (acc <= 0.60) only where
    the planted control is detected (planted acc >= 0.80), else 'no power at site'; otherwise 'unclear'."""
    byp = {it["pair_id"]: it for it in items}
    use = []
    for pid, st in (states or {}).items():
        it = byp.get(pid)
        if it is None:
            continue
        lab = first_diff_digits(it.get("value_A"), it.get("value_B"))
        if lab is not None:
            use.append((pid, st, lab, cl_of(it)))
    out = {"n_items": len(use), "n_clusters": len(set(u[3] for u in use)), "chance": 0.5,
           "rule": "LOCO nearest centred class mean by cosine over {digit(x), digit(x')}; present >= %.2f, absent <= %.2f, "
                   "PC (Llast@K) >= %.2f" % (PROBE_PRESENT, PROBE_ABSENT, PROBE_PC), "sites": {}}
    if len(use) < 4:
        out.update({"pc_Llast@K": None, "pc_ok": None, "labels": {}, "note": "fewer than 4 items with differing digits"})
        return out
    sites = use[0][1]["sites"]
    for si, sk in enumerate(sites):
        H = torch.cat([torch.stack([u[1]["AB"][si][0], u[1]["AB"][si][1]]) for u in use]).float()
        labels = [d for u in use for d in (u[2][0], u[2][1])]
        cands = [u[2] for u in use for _ in (0, 1)]
        cl = [u[3] for u in use for _ in (0, 1)]
        corr = probe_site(H, labels, cands, cl)
        per = []
        for i in range(len(use)):
            xs = [x for x in corr[2 * i:2 * i + 2] if x is not None]
            per.append(sum(xs) / len(xs) if xs else None)
        out["sites"][sk] = boot(per, [u[3] for u in use])
        nrm = torch.stack([(u[1]["AB"][si][0].float() - u[1]["AB"][si][1].float()).norm() for u in use])
        Hp = H + torch.stack([plant_code(d, H.shape[1]) * PLANT_F * float(nrm[i // 2])
                              for i, d in enumerate(labels)])
        corr_p = probe_site(Hp, labels, cands, cl)
        per_p = []
        for i in range(len(use)):
            xs = [x for x in corr_p[2 * i:2 * i + 2] if x is not None]
            per_p.append(sum(xs) / len(xs) if xs else None)
        out.setdefault("planted", {})[sk] = boot(per_p, [u[3] for u in use])
    names = {skey(v): k for k, v in v1.items()}
    out["sites"] = {names.get(k, k): v for k, v in out["sites"].items()}
    out["planted"] = {names.get(k, k): v for k, v in out.get("planted", {}).items()}
    pc = out["sites"].get("Llast@%d" % K)
    out["pc_Llast@K"] = pc
    out["pc_ok"] = bool(pc and pc[0] >= PROBE_PC)
    out["planted_rule"] = "planted code norm %.2f x |h_A - h_B| per item and site; power if planted acc >= %.2f" % (
        PLANT_F, PROBE_PC)
    lab = {}
    for nm, v in out["sites"].items():
        pw = out["planted"].get(nm)
        power = bool(pw and pw[0] >= PROBE_PC)
        if not v:
            lab[nm] = "unclear"
        elif v[0] >= PROBE_PRESENT:
            lab[nm] = "present"
        elif v[0] <= PROBE_ABSENT:
            lab[nm] = "absent" if power else "no power at site"
        else:
            lab[nm] = "unclear"
    out["labels"] = lab
    return out


def plant_code(d, H):
    """Fixed random unit vector for digit d (planted power control, plan critic 3 F1)."""
    g = torch.Generator().manual_seed(int(sha("k17vb-plant:%s" % d)[:8], 16))
    u = torch.randn(H, generator=g)
    return u / u.norm()


STRATA = ("ST1", "ST1e", "ST2", "ST3", "ST4", "ST5", "cont")


def strata_cells(items, key, cl):
    """Per-stratum means of the key's primary endpoint (A1, P3; ST4 added in v3, review F5)."""
    out = {}
    for st in STRATA:
        vals_ = sorted(set(str(it["strata"].get(st)) for it in items))
        out[st] = {v: boot([prim_row(it, key) if str(it["strata"].get(st)) == v else None for it in items], cl)
                   for v in vals_}
    return out


def strata_diff(items, key, st, g1, g2):
    """P-ST (review MS4): mean(g1) - mean(g2) of the primary endpoint, cluster bootstrap of the difference."""
    return boot_group_diff([prim_row(it, key) for it in items], [it["cluster"] for it in items],
                           [str(it["strata"].get(st)) for it in items], g1, g2)


XPAIRS = [("ouro", "ouro_think", "Thinking - base"), ("ouro26", "ouro26_think", "Thinking - base"),
          ("ouro", "ouro26", "2.6B - 1.4B"), ("ouro_think", "ouro26_think", "2.6B - 1.4B"),
          ("ifm_p1_ouro", "ifm_p1_ouro_inj", "inj - no-inj"), ("ifm_p1_ouro", "ifm_d112", "d112 - twin"),
          ("ifm_p1_ouro_inj", "ifm_d112", "d112 - inj")]


def paired_cells(items_a, items_b, keys):
    """Cross-model cells on the intersection of kept items (P3; review MS4/F4): per item, B - A of the key's
    primary endpoint, problem-cluster bootstrap. Both models of a pair share a tokenizer, so every intersection
    item has the same strata in both; the strata counts of the intersection are reported."""
    ia = {it["pair_id"]: it for it in items_a}
    ib = {it["pair_id"]: it for it in items_b}
    common = sorted(set(ia) & set(ib))
    out = {"n_A": len(ia), "n_B": len(ib), "n_intersection": len(common),
           "strata": {st: {v: sum(1 for p in common if str(ia[p]["strata"].get(st)) == v)
                           for v in sorted(set(str(ia[p]["strata"].get(st)) for p in common))} for st in ("ST1", "ST2")},
           "cells": {}}
    cl = [ia[p]["cluster"] for p in common]
    for key in keys:
        xa = [prim_row(ia[p], key) for p in common]
        xb = [prim_row(ib[p], key) for p in common]
        both = [a is not None and b is not None for a, b in zip(xa, xb)]
        if not any(both):
            continue
        out["cells"][key] = {"endpoint": primary_f(key),
                             "A": boot([a if k else None for a, k in zip(xa, both)], cl),
                             "B": boot([b if k else None for b, k in zip(xb, both)], cl),
                             "diff_B_minus_A": boot([(b - a) if k else None for a, b, k in zip(xa, xb, both)], cl)}
    return out


CG_STRATA = ("ST1", "ST1e", "ST2")


def cg_st(it, st):
    """Carry-gain stratum (plan critic 2 F6): ST1 = source tokens 1 vs >= 2 (A2 items: from 'pos'), ST2 = first
    digit of x and x' shared vs differs (VarBench only; None elsewhere)."""
    v = (it.get("strata") or {}).get(st)
    if v is None and st == "ST1" and it.get("pos") is not None:
        v = "1" if len(it["pos"]) == 1 else ">=2"
    return None if v is None else str(v)


def carry_summary(items, gsites_keys, uni_key, dev_site, cl_of, K, rnd_adj=None):
    """V6 metrics for one dataset (plan T7 v3, review MS2/MS5/MS7/MS8/F2). items: done items with G / verify / rows.
    (1) Verified-9 analysis at each dose (all items): tolerance hit, band chance, random-in-top-3 hit, strict hit
    (diagnostic), regret, Spearman. (2) Test items: dm / D at the predicted site vs dev-tuned, uniform Lmid@K,
    latest site in the set, random site, random site in the selector's top 3. (3) Dev items, full set at a = 1
    (every site verified): A2-1 hit vs band chance, regret, Spearman, the same within loops 1..K-1, and the A7
    false-negative rate."""
    gk = [k for k in gsites_keys]
    gks = set(gk)
    latest = max(gk, key=ksite)
    ltk = [k for k in gk if ksite(k)[0] < K]
    latest_ltk = max(ltk, key=ksite) if ltk else None
    out = {"set_size": len(gk), "latest_site": latest, "latest_equals_uniform": latest == uni_key,
           "latest_site_loops_lt_K": latest_ltk}

    def topk(it, n):
        G = {k: v for k, v in it["G"].items() if k in gks}
        return sorted(G, key=lambda k: (-G[k], k))[:n]

    def at(it, k, a):
        x = dm_of(it, "O1|ans|%s|%g" % (k, a))
        D = Dgap(it)
        return None if (x is None or D < GAP) else x / D

    by_dose = {}
    for a in A_VERIFY:
        per = []
        for it in items:
            if not it.get("G"):
                continue
            ver = [k for k in it["verify"] if k in gks]
            dm = {k: dm_of(it, "O1|ans|%s|%g" % (k, a)) for k in ver}
            # v6 (review A-F1): the band scales with the dose like the eligibility threshold (tol x a x D)
            h = hit_metrics({k: it["G"][k] for k in ver}, dm, Dgap(it), THR[a], tol=HIT_TOL * a)
            if h is not None:
                per.append((it, h))
                by_dose.setdefault(id(it), {"it": it})[a] = h
        cl = [cl_of(it) for it, _ in per]
        el = [(it, h) for it, h in per if h["eligible"]]
        cle = [cl_of(it) for it, _ in el]
        sp = sorted(h["spearman"] for _, h in el if h["spearman"] is not None)
        rec = {"set": "verified 9 (top-3, median-3, bottom-3 by G)", "n_items": len(per), "eligible": len(el),
               "top1_hit": boot([h["hit"] for _, h in el], cle), "chance": boot([h["chance"] for _, h in el], cle),
               "random_in_top3_hit": boot([h["top3_hit"] for _, h in el], cle),
               "strict_hit": boot([h["strict_hit"] for _, h in el], cle),
               "regret": boot([h["regret"] for _, h in per], cl),
               "spearman_median": sp[len(sp) // 2] if sp else None, "spearman_n": len(sp)}
        # v4 (plan critic 2 F6, A1): ST1 (and ST2 on VarBench) strata of hit, regret and test dm / D
        rec["strata"] = {}
        for st in CG_STRATA:
            vs = sorted(set(cg_st(it, st) for it, _ in per) - {None})
            if not vs:
                continue
            rec["strata"][st] = {v: {"top1_hit": boot([h["hit"] if cg_st(it, st) == v else None for it, h in el], cle),
                                     "regret": boot([h["regret"] if cg_st(it, st) == v else None for it, h in per], cl)}
                                 for v in vs}
        tests = [it for it in items if it["split"] == "test" and it.get("G")]
        clt = [cl_of(it) for it in tests]
        t3 = []
        for it in tests:
            vs = [at(it, k, a) for k in topk(it, 3)]
            t3.append(None if any(v is None for v in vs) else sum(vs) / len(vs))
        cols = {"predicted": [at(it, topk(it, 1)[0], a) if topk(it, 1) else None for it in tests],
                "dev_tuned": [at(it, dev_site, a) if dev_site else None for it in tests],
                "uniform": [at(it, uni_key, a) for it in tests],
                "latest_site": [at(it, latest, a) for it in tests],
                "random_site": [at(it, it["rnd_site"], a) if it.get("rnd_site") in gks else None for it in tests],
                "random_in_top3": t3}
        rec["test_dm_over_D"] = {k: boot(v, clt) for k, v in cols.items()}
        for st, sv in rec["strata"].items():
            for v in sv:
                sv[v]["test_dm_over_D_predicted"] = boot([x if cg_st(it, st) == v else None
                                                          for x, it in zip(cols["predicted"], tests)], clt)
        for b in ("dev_tuned", "uniform", "latest_site", "random_site", "random_in_top3"):
            diff = [None if (p is None or q is None) else p - q for p, q in zip(cols["predicted"], cols[b])]
            rec["test_dm_over_D"]["predicted_minus_%s" % b] = boot(diff, clt)
        rec["band_tol_D"] = HIT_TOL * a
        out["a=%g" % a] = rec
    # v6 (review A-F1): dose clause paired on items eligible at both doses (band tol x a x D at each dose)
    both = [v for v in by_dose.values() if all(a_ in v and v[a_]["eligible"] for a_ in A_VERIFY)]
    clb = [cl_of(v["it"]) for v in both]
    out["dose_pair"] = {"n_items": len(both), "n_clusters": len(set(clb)),
                        "hit_a0.25": boot([v[0.25]["hit"] for v in both], clb),
                        "hit_a1": boot([v[1.0]["hit"] for v in both], clb),
                        "hit_diff_a0.25_minus_a1": boot([v[0.25]["hit"] - v[1.0]["hit"] for v in both], clb),
                        "hit_minus_chance_a0.25": boot([v[0.25]["hit"] - v[0.25]["chance"] for v in both], clb),
                        "hit_minus_chance_a1": boot([v[1.0]["hit"] - v[1.0]["chance"] for v in both], clb),
                        "rule": "items eligible at both doses; band = %g x a x D at dose a (review A-F1)" % HIT_TOL}
    devs = [it for it in items if it["split"] == "dev" and it.get("G")]
    frac = frac_g = None
    if devs:
        # T3b v6 (review C-F2): working = mean O1 e >= 0.5 (no subtraction). The gated fraction also needs, at a V1
        # site where RND fails T3 (rnd_adj: site -> RND row), the O1 - RND cluster CI lower bound > 0. Upper-bound
        # rows (retrieval <= 0.20) use the O1 fraction, lower-bound rows (computed >= 0.35) the gated one.
        mean_e, gate_ok = {}, {}
        for k in gk:
            xs = [e_row(it, "O1|ans|%s|1" % k) for it in devs]
            ok_ = True
            if rnd_adj and k in rnd_adj:
                rs = [e_row(it, rnd_adj[k]) for it in devs]
                dd = boot([None if (x is None or r_ is None) else x - r_ for x, r_ in zip(xs, rs)], [cl_of(it) for it in devs])
                ok_ = bool(dd and dd[1] is not None and dd[1] > 0)
            xs = [x for x in xs if x is not None]
            if xs:
                mean_e[k] = sum(xs) / len(xs)
                gate_ok[k] = ok_
        if mean_e:
            frac = round(sum(1 for v in mean_e.values() if v >= CARRIED_E) / len(mean_e), 4)
            frac_g = round(sum(1 for k, v in mean_e.items() if v >= CARRIED_E and gate_ok[k]) / len(mean_e), 4)
        out["working_site_rnd_failing_sites"] = sorted(rnd_adj) if rnd_adj else []
        out["working_site_rnd_gate_closed"] = sorted(k for k, v in gate_ok.items() if not v)
    full = {"hit": [], "chance": [], "latest_hit": [], "regret": [], "sp": [], "cl": [], "cl_el": [],
            "l_hit": [], "l_chance": [], "l_latest": [], "l_sp": [], "l_cl": [], "l_dmd": [], "l_dmd_cl": [],
            "fn": [], "fn_cl": [], "miss": [], "miss_cl": [], "it": [], "it_el": []}
    for it in devs:
        D = Dgap(it)
        dms = {k: dm_of(it, "O1|ans|%s|1" % k) for k in gk}
        if any(v is None for v in dms.values()) or D < GAP:
            continue
        G = {k: it["G"][k] for k in gk}
        c = cl_of(it)
        h = hit_metrics(G, dms, D, THR[1.0])
        if h is None:
            continue
        mx = max(dms.values())
        full["regret"].append(h["regret"])
        full["cl"].append(c)
        full["it"].append(it)
        if h["eligible"]:
            full["hit"].append(h["hit"])
            full["chance"].append(h["chance"])
            full["latest_hit"].append(int(dms[latest] >= mx - HIT_TOL * D))
            full["sp"].append(h["spearman"])
            full["cl_el"].append(c)
            full["it_el"].append(it)
        if ltk:
            # option (a) diagnostic (plan critic 2 F1): selector top-1 within loops 1..K-1 minus Llast@K-1, dm / D
            tl = sorted(ltk, key=lambda k: (-G[k], k))[0]          # same tie rule as hit_metrics
            full["l_dmd"].append((dms[tl] - dms[latest_ltk]) / D)
            full["l_dmd_cl"].append(c)
            hl = hit_metrics({k: G[k] for k in ltk}, {k: dms[k] for k in ltk}, D, THR[1.0])
            if hl is not None and hl["eligible"]:
                mxl = max(dms[k] for k in ltk)
                full["l_hit"].append(hl["hit"])
                full["l_chance"].append(hl["chance"])
                full["l_latest"].append(int(dms[latest_ltk] >= mxl - HIT_TOL * D))
                full["l_sp"].append(hl["spearman"])
                full["l_cl"].append(c)
        for k in gk:
            carried = dms[k] / D >= CARRIED_E
            low = abs(G[k]) < FN_G * D
            if low:
                full["fn"].append(1.0 if carried else 0.0)
                full["fn_cl"].append(c)
            if carried:
                full["miss"].append(1.0 if low else 0.0)
                full["miss_cl"].append(c)

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None
    out["dev_full_set_a1"] = {
        "n_dev_complete": len(full["cl"]), "eligible": len(full["cl_el"]),
        "top1_hit": boot(full["hit"], full["cl_el"]), "chance": boot(full["chance"], full["cl_el"]),
        "latest_site_hit": boot(full["latest_hit"], full["cl_el"]),
        # v4 A2-1 (plan critic 2 F2): selector hit minus readout-proximity (latest site) hit, paired per item
        "hit_minus_latest": boot([a - b for a, b in zip(full["hit"], full["latest_hit"])], full["cl_el"]),
        "regret": boot(full["regret"], full["cl"]), "spearman_median": med(full["sp"]),
        "loops_lt_K": {"eligible": len(full["l_cl"]), "n_clusters": len(set(full["l_cl"])),
                       "top1_hit": boot(full["l_hit"], full["l_cl"]),
                       "chance": boot(full["l_chance"], full["l_cl"]),
                       "latest_site_hit": boot(full["l_latest"], full["l_cl"]),
                       "baseline_site": latest_ltk,
                       "hit_minus_latest": boot([a - b for a, b in zip(full["l_hit"], full["l_latest"])], full["l_cl"]),
                       "dm_over_D_selector_minus_Llast_Km1": boot(full["l_dmd"], full["l_dmd_cl"]),
                       "spearman_median": med(full["l_sp"])},
        "strata": {st: {v: {"top1_hit": boot([h if cg_st(it, st) == v else None
                                              for h, it in zip(full["hit"], full["it_el"])], full["cl_el"]),
                            "hit_minus_latest": boot([(h - lh) if cg_st(it, st) == v else None
                                                      for h, lh, it in zip(full["hit"], full["latest_hit"], full["it_el"])],
                                                     full["cl_el"]),
                            "regret": boot([g if cg_st(it, st) == v else None for g, it in zip(full["regret"], full["it"])],
                                           full["cl"])}
                        for v in sorted(set(cg_st(it, st) for it in full["it"]) - {None})}
                   for st in CG_STRATA},
        "false_negative": {"rule": "|G| < %.2f D; carried = verified e >= %.1f (A7)" % (FN_G, CARRIED_E),
                           "fn_rate": boot(full["fn"], full["fn_cl"]), "n_low_G_sites": len(full["fn"]),
                           "carried_with_low_G": boot(full["miss"], full["miss_cl"]),
                           "n_carried_sites": len(full["miss"])}}
    fs_ = out["dev_full_set_a1"]

    def lo_pos(b):
        return bool(b and b[1] is not None and b[1] > 0)
    fs_["adds_beyond_latest"] = {"full_set": lo_pos(fs_["hit_minus_latest"]),
                                 "loops_lt_K": lo_pos(fs_["loops_lt_K"]["hit_minus_latest"]),
                                 "rule": "95% cluster CI lower bound of hit - latest-site hit > 0 (claim, not prediction)"}
    out["working_site_fraction_dev"] = frac
    out["working_site_fraction_dev_rnd_gated"] = frac_g
    out["full_set_regret_dev"] = out["dev_full_set_a1"]["regret"]
    out["dev_tuned_site"] = dev_site
    rec_site, _ = dev_tuned_site(devs, [ksite(k) for k in gk])
    out["dev_tuned_recomputed_v3"] = rec_site
    out["dev_tuned_matches_v3_rule"] = (rec_site == dev_site) if dev_site else None
    tg = [it["t_grad"] / it["t_fwd"] for it in items if it.get("t_grad") and it.get("t_fwd")]
    out["cost_fwd_equiv"] = {"selector_nominal": 3, "selector_measured_grad_over_fwd": round(sum(tg) / len(tg), 3) if tg else None,
                             "dev_search_forwards": len(gk) * len(devs), "n_dev": len(devs), "set_size": len(gk)}
    # PLAN G4 (review C-F4): path-integrated gain row on dev items with G_PI (G_red sites)
    if any(it.get("G_PI") for it in items):
        Lb_ = max(ksite(k)[1] for k in gk) + 1
        red = [skey(x) for x in gset("red", K, Lb_)]
        out["path_integrated"] = pi_summary(items, red, K, cl_of)
        tp = [it["t_pi"] / it["t_fwd"] for it in items if it.get("t_pi") and it.get("t_fwd")]
        out["cost_fwd_equiv"]["path_integrated_passes_per_item"] = len(red) * len(PI_A)
        out["cost_fwd_equiv"]["path_integrated_measured_fwd_equiv"] = round(sum(tp) / len(tp), 1) if tp else None
    gi = [it.get("grad_run_bit_identical") for it in items if "grad_run_bit_identical" in it]
    out["grad_run_bit_identical_all"] = all(gi) if gi else None
    return out


def co_tenant_memory(snaps):
    """nvidia-smi snapshots reduced to memory numbers only (A7, review F7)."""
    out = []
    for sn in snaps or []:
        if "error" in sn:
            out.append({"error": "nvidia-smi failed"})
            continue
        mem = []
        for p in sn.get("procs", []):
            if "used_memory" in p:
                mem.append(p["used_memory"])
            elif "line" in p and "," in p["line"]:
                mem.append(p["line"].split(",")[1].strip())
        out.append({"t": sn.get("t"), "gpus": sn.get("gpus"), "procs_used_memory": mem})
    return out


def summarize(out_dir, plan=None, cgcheck=None):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "vb_*.json"))):
        d = json.load(open(f))
        res[d["model"]] = d
    fds = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "fd_*.json"))):
        d = json.load(open(f))
        fds[d["model"]] = d
    cgc, cgc_sha = None, None
    if cgcheck and os.path.exists(cgcheck):
        try:
            cgc = json.load(open(cgcheck))
            cgc_sha = hashlib.sha256(open(cgcheck, "rb").read()).hexdigest()
        except Exception:  # noqa: BLE001
            cgc = None
    # PLAN G8 (review C-F4): hash every model output file before any analysis reads it
    out_sha = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()
               for pat in ("vb_*.json", "fd_*.json", "bf16_*.json", "window_*.json", "states_*.pt")
               for f in sorted(glob.glob(os.path.join(out_dir, pat)))}
    bfs = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "bf16_*.json"))):
        d = json.load(open(f))
        bfs[d["model"]] = d
    procs = {os.path.relpath(f, out_dir).replace("\\", "/"): json.load(open(f))
             for f in glob.glob(os.path.join(out_dir, "**", "proc_*.json"), recursive=True)}
    hosts = {}
    for m, d in res.items():
        if d.get("host"):
            hosts[d["host"]] = role_of([m])
    for v in procs.values():
        if v.get("host"):
            hosts[v["host"]] = role_of(v.get("models", ""))
    launches = {m: d.get("plan_launch") for m, d in res.items()}
    fro = {x.get("plan_frozen_sha256") for v in launches.values() for x in (v or []) if not x.get("debug")}
    ck = {"plan_sha256": hashlib.sha256(open(plan, "rb").read()).hexdigest() if plan and os.path.exists(plan) else None,
          "plan_sha256_at_launch": launches,
          "plan_frozen_consistent": (len(fro) == 1) if fro else None,
          "endpoint": {"source_rows_undo_SZ": "full_span_logodds e_LO (fields e_LO, dLO)",
                       "answer_rows": "first_token_margin e (fields e, dm); e_LO_over_pc = dLO / D reported beside it",
                       "secondary": "first_token_margin (fields e, dm) on every row", "rule": "plan section 5 v3 (review MS1)",
                       "readout": "float32, TF32 off (v3, review MS2)"},
          "tokenization": {}, "prefix_filter": None, "hook_identity": {}, "readout": {}, "edit_placement": {},
          "knockout_placement": {}, "causality": {}, "items": {}, "pc_final": {}, "pc_site": {}, "structural_zero": {},
          "placement": {}, "strata_diff": {}, "cross_model": {}, "random_control": {}, "o2_context": {}, "presence": {},
          "answer_classes": {}, "window": {}, "undo_slots": {}, "undo": {}, "carry_gain": {},
          "predictions_vs_outcomes": [], "predictions_tally": {}, "model_copies": {}, "downloads": "none from the hub",
          "gpu_minutes": {}, "code_version": CODE_VERSION, "host_roles": {}, "t8": {}, "placement_vbd": {},
          "carry_gain_gate": {}, "citation_rule": "T3b v6 (review C-F2): the cited value is always O1; where the RND cell "
                                                  "at the same site and position fails T3, a lower-bound claim also needs "
                                                  "the O1 - RND paired CI lower bound > 0; upper-bound and null claims cite "
                                                  "O1 alone", "output_sha256": {}, "bf16_check": {}}
    L = ["# K17 VarBench placement (B-K17-VARBENCH, code %s)" % CODE_VERSION, "",
         "Copy-read target (A3). Source cells: e_LO = full-span log-odds effect; answer cells: e = first-token margin "
         "effect (plan section 5 v3). Both only where the clean gap >= %.1f. Cells: primary mean [95%% cluster bootstrap "
         "CI] / flip. '*' = the RND cell at that site fails T3: the O1 value is cited, and a lower-bound claim there also "
         "needs the O1 - RND CI lower bound > 0 (checks.json placement.<cell>.rnd_pair, T3b v6, review C-F2)." % GAP, ""]
    items_by = {}
    for m in [x for x in MODEL_ORDER if x in res]:
        d = res[m]
        VB = d["varbench"]
        ck["tokenization"][m] = d.get("tokenization")
        ck["prefix_filter"] = ck["prefix_filter"] or d.get("prefix_filter")
        c0 = d.get("checks", {})
        ck["hook_identity"][m] = {k: c0.get(k) for k in ("identity_ok", "all_ok", "call_counts_ok", "wrapper_bit_identical",
                                                         "zero_add_every_site_bit_identical", "repeat_bit_identical",
                                                         "rec_bit_identical", "grad_leaf_bit_identical",
                                                         "eager_mask_none_calls", "model_own_logits_max_abs_diff",
                                                         "internal_head_max_abs_diff", "grad_nonzero_sites")}
        ck["readout"][m] = c0.get("readout") or "bf16 head cast to float (pre-v3 run)"
        ck["edit_placement"][m] = c0.get("edit_placement")
        ck["knockout_placement"][m] = c0.get("knockout_placement")
        ck["hook_identity"][m]["by_role"] = {rl: {k: v.get(k) for k in ("identity_ok", "all_ok", "code_sha256")}
                                             for rl, v in (d.get("checks_by_role") or {}).items()}
        ck["model_copies"][m] = d.get("model_copy") or {"repo": OURO_REPOS.get(m, m), "copy": d.get("path")}
        items = [it for it in VB["items"].values() if it.get("done_M")]
        items_by[m] = items
        witems = [it for it in VB["items"].values() if it.get("done_W")]
        cl = [it["cluster"] for it in items]
        # v5 (plan critic 3 F3, T8): host roles of the items; T8 counts clusters
        ck["host_roles"][m] = {"file": d.get("host_roles") or [d.get("host_role")],
                               "items": {rl: sum(1 for it in items if (it.get("host_role") or d.get("host_role")) == rl)
                                         for rl in sorted({it.get("host_role") or d.get("host_role") for it in items} - {None})}}
        ck["host_roles"][m]["mixed"] = len(ck["host_roles"][m]["items"]) > 1
        ck["t8"][m] = {"n_clusters": len(set(cl)), "min": T8_MIN, "interpretable": len(set(cl)) >= T8_MIN}
        caus_rows = [rv.get("caus") for it in items for rv in it["rows"].values() if "caus" in rv]
        ck["causality"][m] = {"check_item": c0.get("causality_r_bit_identical"),
                              "clean_items_ok": sum(1 for it in items if it.get("causality_ok") is True),
                              "clean_items_cont": sum(1 for it in items if "causality_ok" in it),
                              "rows_ok": sum(1 for x in caus_rows if x), "rows": len(caus_rows)}
        scan = VB.get("scan", {})
        ck["items"][m] = {"scanned": len(scan), "kept": len(items), "A_fail": sum(1 for s in scan.values() if not s["A_pass"]),
                          "B_fail": sum(1 for s in scan.values() if s["A_pass"] and not s["B_pass"]),
                          "n_clusters": len(set(cl)), "subset_W": len(witems), "config": d.get("config"),
                          "strata": {st: {v: sum(1 for it in items if str(it["strata"].get(st)) == v)
                                          for v in sorted(set(str(it["strata"].get(st)) for it in items))}
                                     for st in STRATA} if items else {},
                          "e_excluded": sum(1 for it in items if Dgap(it) < GAP),
                          "e_LO_excluded": sum(1 for it in items if it.get("LO_B") is None or it["LO_B"] - it["LO_A"] < GAP),
                          "r2_pass": sum(1 for it in items if it.get("r2_pass")),
                          "pair_equals_gen": sum(1 for it in items if it.get("gen_pair_id") == it["pair_id"]), "a2": {}}
        K = d["K"]
        names = list(d["v1_sites"])
        pck = "O1|ans|Llast@%d|1" % K
        pcf = [it["rows"].get(pck, {}).get("exact") for it in items]
        ck["pc_final"][m] = {"exact": sum(1 for x in pcf if x), "n": sum(1 for x in pcf if x is not None),
                             "e": boot([e_row(it, pck) for it in items], cl),
                             "e_LO_ceiling": boot([e_row(it, pck, lo=True) for it in items], cl)}
        sz = [it["rows"].get("O1|src|Llast@%d|1" % K, {}).get("exact") for it in items] + \
             [it["rows"].get("RND|src|Llast@%d|1" % K, {}).get("exact") for it in items]
        ck["structural_zero"][m] = {"exact": sum(1 for x in sz if x), "n": sum(1 for x in sz if x is not None)}
        pcs = {}
        for nm in names:
            xs = [it["rows"].get("PCS|%s" % nm, {}).get("exact") for it in witems]
            ee = boot([e_row(it, "PCS|%s" % nm) for it in witems], [it["cluster"] for it in witems])
            pcs[nm] = {"exact": sum(1 for x in xs if x), "n": sum(1 for x in xs if x is not None), "e": ee}
        ck["pc_site"][m] = pcs
        pl = {}
        L += ["## %s (K %d, layers per loop %d, set %s %d sites, %s)" % (
                  m, K, d["Lb"], d["gset"], len(gset(d["gset"], K, d["Lb"])), d.get("host_role") or role_of([m])), "",
              "Items: %s. Hook checks all_ok %s. PC-final exact %s, SZ exact %s." % (
                  json.dumps({k: ck["items"][m][k] for k in ("scanned", "kept", "A_fail", "B_fail", "subset_W")}),
                  c0.get("all_ok"), {k: ck["pc_final"][m][k] for k in ("exact", "n")}, ck["structural_zero"][m]), "",
              "| op pos (endpoint) | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for op in ("O1", "RND", "O2"):
            for pos in ("src", "ans"):
                row_cells, ep = [], None
                for nm in names:
                    key = "%s|%s|%s|1" % (op, pos, nm)
                    nullc = op == "RND" or (pos == "ans" and not nm.endswith("@%d" % K)) or (
                        m == "ifm_p1_ouro_inj" and pos == "src" and not nm.endswith("@%d" % K))
                    c = cell(items, key, cl, null_cited=(nullc and op != "O1"))
                    if c is None:
                        row_cells.append("-")
                        continue
                    if op == "O1":
                        c["strata"] = strata_cells(items, key, cl)
                        # T3b v6 (review C-F2): cite O1; RND gates lower-bound claims; TOST on the O1 series
                        rp, series, cep = rnd_pair(items, key, cl)
                        c["rnd_pair"] = rp
                        c["cited"], c["cited_label"], c["lower_gate"] = cited_value(c)
                        c["cited_endpoint"] = cep
                        if nullc:
                            c["tost"] = tost(series, cl) if rp.get("citable") else None
                            c["tost_basis"] = rp.get("cite")
                    if m == "ifm_p1_ouro_inj" and op == "O1" and pos == "src":
                        _, prim, _ = rnd_pair(items, key, cl)
                        for stn in ("ST1", "ST1e"):       # v5 (F7): ST1e = differing source tokens; ST1 = span length
                            c["tost_by_%s" % stn] = {v: tost([x if it["strata"].get(stn) == v else None
                                                              for x, it in zip(prim, items)], cl)
                                                     for v in sorted(set(str(it["strata"].get(stn)) for it in items))}
                    c["pc_final"] = {k: ck["pc_final"][m][k] for k in ("exact", "n")}
                    if pos == "ans":
                        c["pc_final_e_LO_ceiling"] = ck["pc_final"][m]["e_LO_ceiling"]
                    c["pc_site"] = pcs.get(nm)
                    pl["%s|%s|%s" % (op, pos, nm)] = c
                    ep = c["endpoint"]
                    v = c["primary"] or c["e"]
                    star = "*" if (op == "O1" and (c.get("rnd_pair") or {}).get("rnd_T3") is False) else ""
                    row_cells.append("%.2f [%.2f, %.2f] / %.2f%s" % (v[0], v[1] if v[1] is not None else float("nan"),
                                                                      v[2] if v[2] is not None else float("nan"),
                                                                      c["flip"][0] if c["flip"] else float("nan"), star))
                L.append("| %s %s (%s) | " % (op, pos, ep or "-") + " | ".join(row_cells) + " |")
        ck["placement"][m] = pl
        sd = {}
        for nm in ("Llast@1",):
            key = "O1|src|%s|1" % nm
            sd["src %s ST1 1 - >=2" % nm] = strata_diff(items, key, "ST1", "1", ">=2")
            sd["src %s ST1e 1 - >=2" % nm] = strata_diff(items, key, "ST1e", "1", ">=2")     # v5 (F7): P-ST uses ST1e
            sd["src %s ST2 shared - differs" % nm] = strata_diff(items, key, "ST2", "shared", "differs")
            sd["basis"] = "O1 primary (T3b v6, review A-F2/C-F2); RND T3 at the cell: %s" % (
                ((pl.get("O1|src|%s" % nm) or {}).get("rnd_pair") or {}).get("rnd_T3"),)
        ck["strata_diff"][m] = sd
        ck["random_control"][m] = {k: {"endpoint": v["endpoint"], "e_LO": v["e_LO"], "e": v["e"], "flip": v["flip"],
                                       "tost": v.get("tost")} for k, v in pl.items() if k.startswith("RND")}
        o2 = {}
        for k, v in pl.items():
            if k.startswith("O2"):
                o1 = pl.get(k.replace("O2", "O1", 1))
                key = k + "|1"
                k1 = key.replace("O2", "O1", 1)
                cs = [it["rows"][key].get("cos") for it in items if key in it["rows"]]
                nr = [it["rows"][key].get("norm_ratio") for it in items if key in it["rows"]]
                pd_ = [None if (key not in it["rows"] or k1 not in it["rows"] or e_row(it, key) is None
                                or e_row(it, k1) is None) else e_row(it, key) - e_row(it, k1) for it in items]
                o2[k] = {"e": v["e"], "e_O1": o1["e"] if o1 else None,
                         "paired_diff_e_O2_minus_O1": boot(pd_, cl),
                         "cos_mean": r5(sum(x for x in cs if x is not None) / max(1, sum(1 for x in cs if x is not None))),
                         "norm_ratio_mean": r5(sum(x for x in nr if x is not None) / max(1, sum(1 for x in nr if x is not None))),
                         "label": "context-robustness check (not vector transfer); paired on r2_pass items, first-token e"}
        ck["o2_context"][m] = o2
        # presence gate and answer classes
        gate = {}
        for nm, sk in d["v1_sites"].items():
            ps = [it.get("r2_pres", {}).get(sk) for it in items if it.get("r2_pass")]
            ps = [p for p in ps if p and None not in p]
            npass = sum(1 for a_, b_ in ps if b_ - a_ >= 0.5)
            gate[nm] = {"n": len(ps), "pass_frac": round(npass / len(ps), 4) if ps else None,
                        "gate": bool(ps and npass / len(ps) >= 0.5)}
        ck["presence"][m] = {"gate": gate}
        under = {}
        for key in ["O1|src|Llast@1|1"] + ["O1|ans|Llast@%d|1" % j for j in range(1, K)]:
            pk = [it["rows"].get(key, {}).get("pres") for it in items]
            pk = [p for p in pk if p]
            if pk:
                under[key] = {s: r5(sum(p[s] for p in pk if p.get(s) is not None) / max(1, sum(1 for p in pk if p.get(s) is not None)))
                              for s in pk[0]}
        ck["presence"][m]["under_edit"] = under
        classes = {}
        for nm in names:
            j = int(nm.split("@")[1])
            if j == K:
                continue                 # last-loop answer cells: e is reported, no carry class
            rp_ = ((pl.get("O1|ans|%s" % nm) or {}).get("rnd_pair") or {})
            classes[nm] = answer_class(items, "O1|ans|%s|1" % nm, j, d["family"], gate[nm]["gate"],
                                       rnd_fail=(rp_.get("rnd_T3") is False), lower_gate=rp_.get("lower_gate"))
            if classes[nm] is not None:
                classes[nm]["rnd_cite"] = rp_.get("cite")
        dose = {}
        for j in range(1, K):
            for tag in ("a2", "a4", "mn"):
                key = "V7|%s|Llast@%d" % (tag, j)
                dose[key] = answer_class(witems, key, j, d["family"], gate["Llast@%d" % j]["gate"])
        sp_ = os.path.join(out_dir, "states_%s.pt" % m)
        try:
            states_ = torch.load(sp_) if os.path.exists(sp_) else {}
        except Exception as ex:  # noqa: BLE001
            states_ = {}
            print("[summarize] states_%s.pt unreadable: %r" % (m, ex), flush=True)
        prb = probe_table(states_, items, {nm: ksite(sk) for nm, sk in d["v1_sites"].items()}, K)
        for nm, rec in classes.items():
            if rec is None:
                continue
            lab = (prb.get("labels") or {}).get(nm)
            rec["probe"] = {"acc": (prb.get("sites") or {}).get(nm), "label": lab}
            if rec["class"] == GATE_FAILS:            # F5: 'no answer state' only where the probe says absent
                rec["cite_as"] = "no answer state at site" if lab == "absent" else (
                    "answer state present, overwritten within loop j" if lab == "present" else GATE_FAILS)
            elif rec["class"] == "not carried" and d["family"] != "ouro":   # IFM: needs the probe (F7)
                rec["cite_as"] = "not carried" if lab == "present" else "not citable (probe not 'present')"
            elif rec["class"] == "carried" and rec.get("carried_rnd_gate") is False:     # v6 T3b gate (review C-F2)
                rec["cite_as"] = "not citable as carried (RND fails T3, O1 - RND CI lower <= 0)"
            else:
                rec["cite_as"] = rec["class"]
        ck["answer_classes"][m] = {"rule": ("ouro: e_own at the edited loop's own trained exit (every answer site); "
                                            "ifm/d112: R2 presence gate (weak, see report). v4: a failed gate is "
                                            "'%s'; cite_as uses the digit probe" % GATE_FAILS),
                                   "a=1": classes, "subset_doses": dose, "probe": prb}
        # window and undo (review MS6: items with -m_A < GAP are excluded from dep, the band and rule (a))
        win = VB.get("window", {})
        wc = [it["cluster"] for it in witems]

        def dep(it, key):
            rv = it["rows"].get(key)
            return None if rv is None else dep_val(rv["f"][1] - rv["f"][0], it["clean_A"][1] - it["clean_A"][0])
        cols = {"%s|%s" % (v, b): [dep(it, "KO|%s|%s" % (v, b)) for it in witems] for v in ("dir", "rel") for b in WIN_BOUNDS}
        band = max_band(cols, wc) if witems else None
        dsp, drn = [dep(it, "KO|rel|full") for it in witems], [dep(it, "KOR|rel|full") for it in witems]
        ms = sum(x for x in dsp if x is not None) / max(1, sum(1 for x in dsp if x is not None))
        mr = sum(x for x in drn if x is not None) / max(1, sum(1 for x in drn if x is not None))
        # v6 (review A-F4, C-F7): dep = 1 is the tA / tB tie (m_KO = 0), not clean B (dep = D / -m_A). Past the tie the
        # order of two guesses no longer measures span reading, so 'relay >= direct' is scored on per-item min(dep, 1)
        clip = {k_: boot([None if x is None else min(x, 1.0) for x in v_], wc) for k_, v_ in cols.items()} if witems else {}
        ck["window"][m] = {"n": len(witems), "dep_excluded_small_margin": sum(1 for it in witems if
                                                                             -(it["clean_A"][1] - it["clean_A"][0]) < GAP),
                           "dep_band": band, "dep_clipped_at_1": clip,
                           "dep_note": "dep = (m_KO - m_A) / (-m_A); dep = 1 is the tA/tB tie, clean B is dep = D / (-m_A)",
                           "functional_ko_check": {
                               "dep_span_relay_full": r5(ms), "dep_random_relay_full": r5(mr),
                               "ok": bool(ms >= 0.5 and abs(mr) < 0.5 * ms)}}
        # v5 (plan critic 3 F1): slot clauses as frozen at phase W (window file); pre-v5 files are re-scored with v5
        slot_tab = {}
        v6_file = bool(win.get("slots")) and all("carried_over_DKO" in sv for sv in win["slots"].values())
        for cn, sv in (win.get("slots") or {}).items():
            if v6_file:
                slot_tab[cn] = dict(sv)
            else:                        # v6 (review C-F1): earlier window files are re-scored with the D_KO b2
                st5 = slot_stats(witems, cn)
                rl = window_rule(st5["removal"], st5["carried_over_DKO"], sv.get("presence"), st5["n_items"],
                                 st5["n_clusters"], min_n=1 if d.get("debug") else MIN_CL, n_b2=st5["n_b2"],
                                 ncl_b2=st5["n_clusters_b2"])
                slot_tab[cn] = dict(rl, removal=st5["removal"], carried_over_DKO=st5["carried_over_DKO"],
                                    carried_over_D=st5["carried_over_D"], DKO_over_D=st5["DKO_over_D"],
                                    n_items=st5["n_items"], n_clusters=st5["n_clusters"], presence=sv.get("presence"),
                                    necessity_dep_rel_after=sv.get("necessity_dep_rel_after", sv.get("dep_rel_after")),
                                    rescored_from="pre-v6 window file (selection in the file: %s)" % sv.get("selected_by"))
        used = VB.get("slots_used")
        ck["undo_slots"][m] = {"selected": win.get("selected"), "slots": slot_tab, "used_in_M": used,
                               "informative_in_M": None if used is None else [c for c in used if (slot_tab.get(c) or {}).get("informative")],
                               "rule": win.get("rule"), "dep_excluded_small_margin": win.get("dep_excluded_small_margin")}
        und = {}
        src_e = cell(items, "O1|src|Llast@1|1", cl)
        m_only = [it for it in items if not it.get("done_W")]           # v5 (plan critic 3 F6): scored items
        mo_cl = [it["cluster"] for it in m_only]
        for cn in [c.replace("@4", "@%d" % K) for c in CAND_SLOTS]:
            key = "U|%s" % cn
            c_all = cell(items, key, cl, null_cited=True)
            c_sub = cell(witems, key, wc, null_cited=True)
            c_mo = cell(m_only, key, mo_cl, null_cited=True)
            st = slot_tab.get(cn) or {}
            sc = slot_stats(m_only, cn)
            sel_ = slot_stats(witems, cn)
            und[cn] = {"selected": cn in (used or []), "selected_by": st.get("selected_by"),
                       "informative": st.get("informative"),
                       "scored_items": "phase-M items outside the W subset (plan critic 3 F6)",
                       "survival": sc["survival"], "t6": t6_verdict(sc["survival"]), "n_items": sc["n_items"],
                       "n_clusters": sc["n_clusters"], "removal_scored": sc["removal"],
                       "carried_over_D_scored": sc["carried_over_D"],
                       "carried_over_DKO_scored": sc["carried_over_DKO"], "DKO_over_D_scored": sc["DKO_over_D"],
                       "n_b2_scored": sc["n_b2"],
                       "selection_data_W": {"survival": sel_["survival"], "t6": t6_verdict(sel_["survival"]),
                                            "removal": sel_["removal"], "carried_over_D": sel_["carried_over_D"],
                                            "carried_over_DKO": sel_["carried_over_DKO"], "DKO_over_D": sel_["DKO_over_D"],
                                            "n_items": sel_["n_items"]},
                       "cite_as_rebuild_evidence": bool(cn in (used or []) and st.get("informative")
                                                        and t6_verdict(sc["survival"]) == "reverts"),
                       "e_U_M_only": c_mo, "all_items": c_all, "subset": c_sub,
                       "e_U_over_e_src": (r5(c_all["e_LO"][0] / src_e["e_LO"][0]) if c_all and src_e and c_all["e_LO"]
                                          and src_e["e_LO"] and src_e["e_LO"][0] else None)}
        ck["undo"][m] = und
        # carry gain
        gk = [skey(s) for s in gset(d["gset"], K, d["Lb"])]
        def rnd_adj_of(plc):
            v1m = {nm: sk for nm, sk in d["v1_sites"].items()}
            return {v1m[nm]: "RND|ans|%s|1" % nm for nm in v1m
                    if ((plc.get("O1|ans|%s" % nm) or {}).get("rnd_pair") or {}).get("rnd_T3") is False}
        cg = {"VarBench": carry_summary(items, gk, d["uni_site"], (VB.get("dev_tuned") or {}).get("site"),
                                        lambda it: it["cluster"], K, rnd_adj=rnd_adj_of(pl))}
        cg["VarBench"]["dev_tuned_table"] = (VB.get("dev_tuned") or {}).get("mean_dm_over_D_by_site",
                                                                             (VB.get("dev_tuned") or {}).get("mean_dm_by_site"))
        cg["VarBench"]["pc_final"] = ck["pc_final"][m]          # v4 (plan critic 2 F6, A1)
        for ds, D_ in d.get("a2", {}).items():
            its = [it for it in D_.get("items", {}).values() if it.get("done")]
            dyad = ds == "D1" and all("cluster_b" in it for it in its) and bool(its)
            cl_of = (lambda it: (it["cluster"], it["cluster_b"])) if dyad else (lambda it: it["cluster"])
            ck["items"][m]["a2"][ds] = {"kept": len(its), "scanned": D_.get("scanned"), "know_stats": D_.get("know_stats"),
                                        "n_clusters": len(set(it["cluster"] for it in its)),
                                        "n_subjects_A_or_B": (len(set(u for it in its for u in (it["cluster"], it["cluster_b"])))
                                                              if dyad else None),
                                        "ci": "dyadic over subjects" if dyad else "cluster", "build": D_.get("build"),
                                        "split": {s: sum(1 for it in its if it["split"] == s) for s in ("dev", "test")},
                                        "strata": {k: {v: sum(1 for it in its if str(it["strata"].get(k)) == v)
                                                       for v in sorted(set(str(it["strata"].get(k)) for it in its))}
                                                   for k in sorted(set(k for it in its for k in it["strata"]))}}
            if dyad:
                dev_s = {u for it in its if it["split"] == "dev" for u in (it["cluster"], it["cluster_b"])}
                tst_s = {u for it in its if it["split"] == "test" for u in (it["cluster"], it["cluster_b"])}
                ck["items"][m]["a2"][ds]["dev_test_shared_subjects"] = len(dev_s & tst_s)
            dsite = (D_.get("dev_tuned") or {}).get("site")
            cg[ds] = carry_summary(its, gk, d["uni_site"], dsite, cl_of, K)
            for kind in ("retrieval", "computed"):
                sub = [it for it in its if it["kind"] == kind]
                if sub and ds == "D4":
                    cg["%s/%s" % (ds, kind)] = carry_summary(sub, gk, d["uni_site"], dsite, cl_of, K)
            pk2 = "O1|ans|%s|1" % skey((K, d["Lb"] - 1))
            pcf2 = [it["rows"].get(pk2, {}).get("exact") for it in its]
            cg[ds]["pc_final_exact"] = [sum(1 for x in pcf2 if x), sum(1 for x in pcf2 if x is not None)]
            pcv = {"exact": cg[ds]["pc_final_exact"][0], "n": cg[ds]["pc_final_exact"][1],
                   "e": boot([e_row(it, pk2) for it in its], [cl_of(it) for it in its])}
            for kk in [ds] + ["%s/%s" % (ds, kind) for kind in ("retrieval", "computed")]:
                if kk in cg:
                    cg[kk]["pc_final"] = pcv              # v4 (plan critic 2 F6, A1)
        # v5 (plan critic 3 F5): derived-use target, V1 cells (Ouro) and V6
        VD = d.get("vbd") or {}
        vits = [it for it in (VD.get("items") or {}).values() if it.get("done")]
        if VD:
            vcl = [it["cluster"] for it in vits]
            vsc = VD.get("scan") or {}
            ck["items"][m]["vbd"] = {"kept": len(vits), "scanned": len(vsc),
                                     "A_fail": sum(1 for x in vsc.values() if not x["A_pass"]),
                                     "B_fail": sum(1 for x in vsc.values() if x["A_pass"] and not x["B_pass"]),
                                     "n_clusters": len(set(vcl)), "build": VD.get("build"),
                                     "tokenization": VD.get("tokenization"), "config": VD.get("config"),
                                     "error": VD.get("error"),
                                     "split": {sp: sum(1 for it in vits if it["split"] == sp) for sp in ("dev", "test")},
                                     "strata": {st: {v: sum(1 for it in vits if str(it["strata"].get(st)) == v)
                                                     for v in sorted(set(str(it["strata"].get(st)) for it in vits))}
                                                for st in STRATA} if vits else {}}
            vpk, vsz = "O1|ans|Llast@%d|1" % K, "O1|src|Llast@%d|1" % K
            pv = {"exact": sum(1 for it in vits if (it["rows"].get(vpk) or {}).get("exact")),
                  "n": sum(1 for it in vits if (it["rows"].get(vpk) or {}).get("exact") is not None),
                  "e": boot([e_row(it, vpk) for it in vits], vcl)}
            szv = [(it["rows"].get(k_) or {}).get("exact") for it in vits for k_ in (vsz, "RND|src|Llast@%d|1" % K)]
            plv = {"pc_final": pv, "structural_zero": {"exact": sum(1 for x in szv if x),
                                                       "n": sum(1 for x in szv if x is not None)}}
            for op in ("O1", "RND"):
                for pos in ("src", "ans"):
                    for nm in names:
                        key = "%s|%s|%s|1" % (op, pos, nm)
                        nullc = op == "RND" or (pos == "ans" and not nm.endswith("@%d" % K))
                        c = cell(vits, key, vcl, null_cited=(nullc and op != "O1"))
                        if c is None:
                            continue
                        if op == "O1":
                            c["strata"] = strata_cells(vits, key, vcl)
                            rp, series, cep = rnd_pair(vits, key, vcl)
                            c["rnd_pair"] = rp
                            c["cited"], c["cited_label"], c["lower_gate"] = cited_value(c)
                            c["cited_endpoint"] = cep
                            if nullc:
                                c["tost"] = tost(series, vcl) if rp.get("citable") else None
                                c["tost_basis"] = rp.get("cite")
                        c["pc_final"] = {"exact": pv["exact"], "n": pv["n"]}
                        plv["%s|%s|%s" % (op, pos, nm)] = c
            ck["placement_vbd"][m] = plv
            cg["VBD"] = carry_summary(vits, gk, d["uni_site"], (VD.get("dev_tuned") or {}).get("site"),
                                      lambda it: it["cluster"], K, rnd_adj=rnd_adj_of(plv))
            cg["VBD"]["pc_final"] = pv
        # v5 (plan critic 3 F4): V6 is interpreted only after B-K17-CGCHECK passes and this family's FD row passes
        fd_ = fds.get(m) or {}
        cgp = cgcheck_gate(cgc, cgc_sha)
        fdp = (fd_.get("fd") or {}).get("pass")
        bf_ = bfs.get(m) or {}
        # the frozen item count, not the file's own --bf16_n (a debug check with n = 3 must not open the gate)
        bso = None if not bf_ else bool(bf_.get("ok") and (bf_.get("n_items_D_ge_gap") or 0) >= BF16_N)
        ck["carry_gain_gate"][m] = {"cgcheck_pass": cgp["pass"], "cgcheck": cgp, "fd_pass": fdp, "fd": fd_.get("fd"),
                                    "fd_bf16_vs_fp32_G_4_items": fd_.get("bf16_vs_fp32_G"),
                                    "fd_bf16_selector_ok_4_items": fd_.get("bf16_selector_ok"),
                                    "bf16_check": {k_: bf_.get(k_) for k_ in ("n_items", "n_items_D_ge_gap", "sites",
                                                                               "stats", "ok", "rule", "fresh_vs_fd")},
                                    "bf16_selector_ok": bso, "fd_n_items": len(fd_.get("pair_ids") or []),
                                    "fd_dtypes": [fd_.get("dtype_bf16"), fd_.get("dtype_fp32")],
                                    "interpretable": bool(cgp["pass"] and fdp and bso),
                                    "rule": "no V6 row is interpreted until the CGCHECK full run passes (sha256 = %s..., no "
                                            "failed or insufficient cell), the family FD row passes, and the bf16 check "
                                            "on >= %d fresh items with D >= 2 passes (v6, review A-F3, A-F6)" % (
                                                CGCHECK_SHA256[:12], BF16_N)}
        ck["bf16_check"][m] = bf_.get("stats")
        if d["gset"] == "full":
            red = [skey(s) for s in gset("red", K, d["Lb"])]
            sub, scl = [], []
            for it in items:
                if it["split"] != "dev" or not it.get("G"):
                    continue
                t = max(red, key=lambda k: (it["G"][k], k))
                dms = {k: dm_of(it, "O1|ans|%s|1" % k) for k in red}
                if None in dms.values() or Dgap(it) < GAP:
                    continue
                sub.append((max(dms.values()) - dms[t]) / Dgap(it))
                scl.append(it["cluster"])
            cg["VarBench"]["G_red_subset_regret_dev_a1"] = boot(sub, scl) if sub else None
        ck["carry_gain"][m] = cg
        L += ["", "Answer classes (a = 1): " + json.dumps({k: (v or {}).get("class") for k, v in classes.items()}),
              "", "Window dep (subset n %d): %s" % (len(witems), json.dumps(win.get("dep"))),
              "Undo slots selected: %s; slot table %s" % (win.get("selected"), json.dumps(win.get("slots"))), "",
              "Carry gain (dev full set at a = 1: hit / band chance / regret; test: predicted - dev-tuned, - uniform):"]
        for ds, v in cg.items():
            fs = v.get("dev_full_set_a1") or {}
            a1 = v.get("a=1", {})
            L.append("- %s: dev eligible %s, hit %s, hit - latest %s, hit - Llast@K-1 (loops < K) %s, chance %s, regret %s, "
                     "FN %s; verified-9 eligible %s/%s; working-site fraction %s; dev-tuned %s; pred-dev %s; pred-uniform %s" % (
                         ds, fs.get("eligible"), fs.get("top1_hit"), fs.get("hit_minus_latest"),
                         (fs.get("loops_lt_K") or {}).get("hit_minus_latest"), fs.get("chance"), fs.get("regret"),
                         (fs.get("false_negative") or {}).get("fn_rate"), a1.get("eligible"), a1.get("n_items"),
                         v.get("working_site_fraction_dev"), v.get("dev_tuned_site"),
                         (a1.get("test_dm_over_D") or {}).get("predicted_minus_dev_tuned"),
                         (a1.get("test_dm_over_D") or {}).get("predicted_minus_uniform")))
        gt = ck["carry_gain_gate"].get(m) or {}
        L.append("- V6 gate (plan critic 3 F4): CGCHECK pass %s, family FD pass %s (%s), bf16 vs fp32 G %s -> "
                 "interpretable %s" % (gt.get("cgcheck_pass"), gt.get("fd_pass"),
                                       json.dumps((gt.get("fd") or {}).get("a=0.01")), json.dumps(gt.get("bf16_vs_fp32_G")),
                                       gt.get("interpretable")))
        us = ck["undo"].get(m) or {}
        L.append("- Undo v5 (scored on M-only items): " + json.dumps({cn: [v.get("selected_by"), v.get("t6"), v.get("survival")]
                                                                     for cn, v in us.items() if v.get("selected")}))
        pv_ = ck["placement_vbd"].get(m)
        if pv_:
            def cv(k_):
                c_ = pv_.get(k_)
                if not c_:
                    return None
                return [((c_.get("cited") or c_.get("primary") or [None])[0]), c_.get("cited_label")]
            L.append("- VBD (derived-use, first-token e): items %s; PC-final %s; SZ %s; src Llast@1 %s; ans Llast@1 %s, "
                     "Llast@2 %s, Llast@3 %s, Lmid@%d %s" % (
                         json.dumps({k_: (ck["items"][m].get("vbd") or {}).get(k_) for k_ in ("scanned", "kept", "A_fail", "B_fail")}),
                         pv_["pc_final"], pv_["structural_zero"], cv("O1|src|Llast@1"), cv("O1|ans|Llast@1"),
                         cv("O1|ans|Llast@2"), cv("O1|ans|Llast@3"), K, cv("O1|ans|Lmid@%d" % K)))
        L.append("")
    # cross-model cells on the intersection of kept items (P3, review MS4/F4)
    for ma, mb, lab in XPAIRS:
        if ma in items_by and mb in items_by:
            K = res[ma]["K"]
            keys = ["O1|%s|%s|1" % (pos, nm) for pos in ("src", "ans") for nm in res[ma]["v1_sites"]]
            pc_ = paired_cells(items_by[ma], items_by[mb], keys)
            pc_["label"] = lab
            hr_ = [res[ma].get("host_role"), res[mb].get("host_role")]       # v5 (plan critic 3 F3)
            pc_["host_roles"] = hr_
            pc_["cross_host"] = bool(hr_[0] != hr_[1] or ck["host_roles"][ma]["mixed"] or ck["host_roles"][mb]["mixed"])
            pc_["harness_cells"] = ["O1|src|Llast@%d|1" % K, "O1|ans|Llast@%d|1" % K]
            # v6 (review A-F2 under T3b v6): the paired difference is on O1 in both models; each side's RND verdict
            for k_, cv_ in pc_["cells"].items():
                ck_ = k_[:-2] if k_.endswith("|1") else k_
                cv_["basis"] = "O1 in both models (T3b v6)"
                cv_["rnd_T3_A"] = (((ck["placement"].get(ma) or {}).get(ck_) or {}).get("rnd_pair") or {}).get("rnd_T3")
                cv_["rnd_T3_B"] = (((ck["placement"].get(mb) or {}).get(ck_) or {}).get("rnd_pair") or {}).get("rnd_T3")
            ck["cross_model"]["%s|%s" % (ma, mb)] = pc_
    roles = {}
    for p, v in procs.items():
        rl = v.get("host_role") or hosts.get(v.get("host"), role_of(v.get("models", "")))
        if v.get("device", "").startswith("cuda") and v.get("mode") != "cpu_check":
            r_ = roles.setdefault(rl, {"total": 0.0})
            mode = v.get("mode", "?")
            r_[mode] = round(r_.get(mode, 0.0) + v["seconds"] / 60.0, 2)
            r_["total"] = round(r_["total"] + v["seconds"] / 60.0, 2)
    ck["gpu_minutes"] = roles
    ck["co_tenants"] = {m: co_tenant_memory(d.get("nvsmi")) for m, d in res.items()}
    ck["output_sha256"] = out_sha
    ck["predictions_vs_outcomes"] = predictions(ck, res)
    tally = {}
    for rw in ck["predictions_vs_outcomes"]:
        t = tally.setdefault("harness" if rw["harness"] else "tallied", {"held": 0, "missed": 0, "not scored": 0})
        t[rw["verdict"]] += 1
    ck["predictions_tally"] = tally
    ck = scrub(ck, hosts)
    json.dump(scrub({"models": res, "processes": procs}, hosts), open(os.path.join(out_dir, "results.json"), "w"))
    with open(os.path.join(out_dir, "checks.json"), "w") as f:
        f.write("{\n" + ",\n".join('"%s": %s' % (k, json.dumps(v)) for k, v in ck.items()) + "\n}\n")
    L += ["## Predictions", "", "Tally (harness rows apart): %s" % json.dumps(tally), ""]
    open(os.path.join(out_dir, "summary.md"), "w").write(scrub("\n".join(L), hosts))
    print(json.dumps({"items": {m: {k: v[k] for k in ("scanned", "kept")} for m, v in ck["items"].items()},
                      "gpu_minutes": ck["gpu_minutes"], "tally": tally}), flush=True)
    return ck


# ------------------------------------------------------------------ frozen predictions (plan section 8)
A2_DATASETS = [("VarBench", "retrieval"), ("D1", "retrieval"), ("D2", "computed"), ("D4/retrieval", "retrieval"),
               ("D4/computed", "computed"), ("VBD", "computed")]     # v5 (plan critic 3 F5): VBD = real-math computed


def predictions(ck, res):
    """Every plan section 8 row, with its harness tag and endpoint (review F3). A row whose inputs are missing gets
    verdict 'not scored'; [harness] rows are kept out of the held/missed tally (F10). Point estimates decide the
    verdict; CIs sit in the cited checks.json cells."""
    out = []

    def add(pid, model, desc, val, held, harness=False, endpoint=None, note=None):
        verdict = "not scored" if (val is None or held is None) else ("held" if held else "missed")
        out.append({"id": pid, "model": model, "prediction": desc, "endpoint": endpoint, "outcome": val,
                    "verdict": verdict, "harness": harness, "note": note})

    def pc(m, key):
        return (ck.get("placement", {}).get(m) or {}).get(key)

    def pv(m, key, place="placement"):
        """T3b v6 (review C-F2): the cited value (always O1), its endpoint label and the RND lower-bound gate."""
        c = (ck.get(place, {}).get(m) or {}).get(key)
        if c is None:
            return (None, None, None)
        if "rnd_pair" not in c:
            return (None, None, None) if c.get("primary") is None else (c["primary"][0], c["endpoint"], None)
        v, lab, gate = cited_value(c)
        return ((None, lab, None) if v is None else
                (v[0], "%s (%s)" % (c.get("cited_endpoint") or c["endpoint"], lab), gate))

    def flip(m, key):
        c = pc(m, key)
        return None if c is None or not c.get("flip") else c["flip"][0]

    def cmp_(v, lo=None, hi=None):
        if v is None:
            return None
        return (lo is None or v >= lo) and (hi is None or v <= hi)

    def gated(v, lo, hi, gate):
        """Verdict with the T3b v6 gate: a lower bound also needs the RND gate (False where RND fails T3 and the
        O1 - RND CI lower bound is <= 0); an upper bound alone is scored on O1."""
        ok = cmp_(v, lo, hi)
        if ok is None or lo is None or gate is not False:
            return ok, None
        return False, ("O1 meets the bound but RND fails T3 and the O1 - RND CI lower bound is <= 0" if ok else None)

    def src(m, nm, lo=None, hi=None, need_flip=None, place="placement", tag=""):
        v, ep, gate = pv(m, "O1|src|%s" % nm, place)
        rng = ("in [%g, %g]" % (lo, hi)) if (lo is not None and hi is not None) else (
            (">= %g" % lo) if lo is not None else "<= %g" % hi)
        ok, note = gated(v, lo, hi, gate)
        if need_flip is None:
            add("%ssrc %s" % (tag, nm), m, "%ssource %s %s" % (tag, nm, rng), v, ok, endpoint=ep, note=note)
        else:
            f = flip(m, "O1|src|%s" % nm)
            add("%ssrc %s" % (tag, nm), m, "%ssource %s %s and flip >= %g" % (tag, nm, rng, need_flip),
                None if v is None else [v, f],
                None if (v is None or f is None) else bool(ok and f >= need_flip), endpoint=ep, note=note)

    def ans(m, nm, lo=None, hi=None, place="placement", tag=""):
        v, ep, gate = pv(m, "O1|ans|%s" % nm, place)
        rng = ("in [%g, %g]" % (lo, hi)) if (lo is not None and hi is not None) else (
            (">= %g" % lo) if lo is not None else "<= %g" % hi)
        ok, note = gated(v, lo, hi, gate)
        add("%sans %s" % (tag, nm), m, "%sanswer %s final e %s" % (tag, nm, rng), v, ok, endpoint=ep, note=note)

    def ans_class(m, nm, want, eown_min=None):
        c = ((ck.get("answer_classes", {}).get(m) or {}).get("a=1") or {}).get(nm)
        if c is None:
            add("ans %s class" % nm, m, "answer %s class '%s'" % (nm, want), None, None, endpoint="e (class rule)")
            return
        held = c["class"] == want
        desc = "answer %s class '%s'" % (nm, want)
        if eown_min is not None:
            held = held and c.get("e_own") is not None and c["e_own"] >= eown_min
            desc += " with e_own >= %g" % eown_min
        add("ans %s class" % nm, m, desc, [c["class"], c.get("e_own"), c.get("final_e")], held, endpoint="e (class rule)")

    def harness_rows(m, K):
        p = ck.get("pc_final", {}).get(m)
        add("PC-final", m, "answer O1 at Llast@K = clean B exactly", None if p is None else [p["exact"], p["n"]],
            None if not p or not p["n"] else p["exact"] == p["n"], harness=True, endpoint="bit-exact z at r")
        z = ck.get("structural_zero", {}).get(m)
        add("SZ", m, "source O1 and RND at Llast@K = clean A exactly", None if z is None else [z["exact"], z["n"]],
            None if not z or not z["n"] else z["exact"] == z["n"], harness=True, endpoint="bit-exact z at r")

    def undo_rows(m, slots_bounds, only_if_selected=False):
        """v5 (plan critic 3 F1/F6): scored only at slots selected by (a) and b2, on phase-M items outside the W
        subset. bound 'T6' = T6 verdict 'reverts' (95% CI upper of s(c) <= 0.3); a number = s(c) point <= bound."""
        u = ck.get("undo", {}).get(m) or {}
        us = ck.get("undo_slots", {}).get(m) or {}
        sel, inf = us.get("used_in_M"), us.get("informative_in_M")
        for cn, bound in slots_bounds:
            rec = u.get(cn) or {}
            sv = rec.get("survival")
            v = None if not sv else [sv[0], sv[1], sv[2]]
            desc = ("undo at %s reverts (T6: s(c) 95%% CI upper <= %.1f)" % (cn, SURV_REV) if bound == "T6" else
                    "undo at %s s(c) <= %g" % (cn, bound)) + " (if selected by (a) and b2)"
            if sel is None:
                add("undo %s" % cn, m, desc, None, None, endpoint="s(c) first-token")
            elif cn not in sel:
                add("undo %s" % cn, m, desc, None, None, endpoint="s(c) first-token", note="slot not selected by the V5 rule")
            elif cn not in (inf or []):
                add("undo %s" % cn, m, desc, v, None, endpoint="s(c) first-token", note="selected by b1 only: uninformative")
            else:
                held = (None if rec.get("t6") is None else rec["t6"] == "reverts") if bound == "T6" else (
                    None if v is None else v[0] <= bound)
                add("undo %s" % cn, m, desc, v, held, endpoint="s(c) first-token")

    def undo_all(m):
        u = ck.get("undo", {}).get(m) or {}
        us = ck.get("undo_slots", {}).get(m) or {}
        sel, inf = us.get("used_in_M"), us.get("informative_in_M")
        desc = "undo reverts (T6 v5: s(c) 95% CI upper <= 0.3) at every informative slot (selected by (a) and b2)"
        if not inf:
            note = None
            if sel == []:
                note = "no in-window slot"
            elif sel:
                note = "no informative slot (b1-only: %s)" % ",".join(sel)
            add("undo", m, desc, None, None, endpoint="s(c) first-token", note=note)
            return
        vv = {cn: (u.get(cn) or {}).get("t6") for cn in inf}
        held = None if any(x is None for x in vv.values()) else all(x == "reverts" for x in vv.values())
        add("undo", m, desc, {cn: [(u.get(cn) or {}).get("t6"), (u.get(cn) or {}).get("survival")] for cn in inf}, held,
            endpoint="s(c) first-token",
            note=("b1-only slots not scored: %s" % ",".join(c for c in sel if c not in inf)) if set(sel) - set(inf) else None)

    for m in MODEL_ORDER:
        K = (res.get(m) or {}).get("K", 4)
        sl = ["L0@%d", "Lmid@%d", "Llast@%d"]
        if m in OURO_REPOS:
            for nm in ("L0@1", "Lmid@1", "Llast@1"):
                src(m, nm, lo=0.85, need_flip=0.85)
            for j in (2, 3):
                for s_ in sl:
                    src(m, s_ % j, lo=0.8)
            src(m, "L0@%d" % K, lo=0.7)
            src(m, "Lmid@%d" % K, lo=0.2, hi=0.8)
            harness_rows(m, K)
            for j in (1, 2, 3):
                ans_class(m, "Llast@%d" % j, "not carried", eown_min=0.5 if j in (2, 3) else None)
            for j in (1, 2, 3):
                for s_ in ("L0@%d", "Lmid@%d"):
                    ans_class(m, s_ % j, GATE_FAILS)
                    ans(m, s_ % j, hi=0.10)
            ans(m, "L0@%d" % K, hi=0.2)
            ans(m, "Lmid@%d" % K, lo=0.2, hi=0.8)
            undo_all(m)
        elif m in ("ifm_p1_ouro", "ifm_p1_ouro_inj"):
            if m == "ifm_p1_ouro":
                for j in (1, 2, 3):
                    for s_ in sl:
                        src(m, s_ % j, lo=0.8)
                src(m, "L0@4", lo=0.7)
                src(m, "Lmid@4", lo=0.1, hi=0.6)
            else:
                for j in (1, 2, 3):
                    for s_ in sl:
                        nm = s_ % j
                        c = pc(m, "O1|src|%s" % nm)
                        t = ((c or {}).get("tost_by_ST1e") or {}).get("1")      # v5 (F7): ST1e stratum
                        add("inj src %s" % nm, m, "source %s <= 0.10, null by TOST on the 1-token stratum (ST1e)" % nm,
                            None if t is None else t["ci90"], None if t is None else t["null_by_tost"],
                            endpoint="cited series (ST1e = 1)")
                src(m, "L0@4", lo=0.7)
            harness_rows(m, 4)
            for j in (1, 2, 3):
                for s_ in sl:
                    ans(m, s_ % j, hi=0.10)
            for nm in ("L0@1", "Lmid@1", "L0@2", "Lmid@2"):
                ans_class(m, nm, GATE_FAILS)
            ans(m, "L0@4", hi=0.15)
            ans(m, "Lmid@4", lo=0.1, hi=0.6)
            if m == "ifm_p1_ouro":
                undo_all(m)
            else:
                ps = ck.get("pc_site", {}).get(m) or {}
                for j in (1, 2, 3):
                    for s_ in sl:
                        nm = s_ % j
                        e_ = ((ps.get(nm) or {}).get("e") or [None])[0]
                        add("PC-site %s" % nm, m, "PC-site %s e in [0.3, 0.95] (low confidence)" % nm, e_,
                            cmp_(e_, 0.3, 0.95), endpoint="e")
                for nm in ("L0@4", "Lmid@4", "Llast@4"):
                    p = ps.get(nm)
                    add("PC-site %s" % nm, m, "PC-site %s = clean B exactly (after the last injection)" % nm,
                        None if not p else [p["exact"], p["n"]], None if not p or not p["n"] else p["exact"] == p["n"],
                        harness=True, endpoint="bit-exact z at r")
        else:                                                   # ifm_d112, mapped layers
            for nm in ("L0@1", "Lmid@1", "Llast@1", "L0@2", "Lmid@2", "Llast@2", "L0@3", "Lmid@3"):
                src(m, nm, lo=0.8)                              # layers 0-70
            src(m, "Llast@3", lo=0.5)                           # layer 83
            src(m, "L0@4", lo=0.5)                              # layer 84
            src(m, "Lmid@4", hi=0.4)                            # layer 98
            harness_rows(m, 4)
            for nm in ("L0@1", "Lmid@1", "Llast@1", "L0@2", "Lmid@2", "Llast@2"):
                ans(m, nm, hi=0.10)                             # layers <= 55
            ans(m, "Llast@3", lo=0.1, hi=0.5)                   # 83
            ans(m, "L0@4", lo=0.2, hi=0.7)                      # 84
            ans(m, "Lmid@4", lo=0.5)                            # 98
            undo_rows(m, [("Llast@2", "T6"), ("Llast@3", 0.3)], only_if_selected=True)
        # global rows per model
        rnd = ck.get("random_control", {}).get(m)
        if rnd:
            okr = []
            for k, x in rnd.items():
                pe = x.get(x.get("endpoint") or "e")
                t3 = pe is not None and x.get("flip") is not None and abs(pe[0]) <= 0.05 and x["flip"][0] <= 0.02
                okr.append(bool(t3 and (x.get("tost") or {}).get("null_by_tost")))
            add("P-RND", m, "every RND cell passes T3 (|e| <= 0.05, flip <= 0.02) and is null by TOST",
                "%d/%d" % (sum(okr), len(okr)), all(okr) if okr else None, endpoint="primary per cell")
        else:
            add("P-RND", m, "every RND cell passes T3 and is null by TOST", None, None, endpoint="primary per cell")
        o2 = ck.get("o2_context", {}).get(m) or {}
        dd = {k: ((v.get("paired_diff_e_O2_minus_O1") or [None])[0]) for k, v in o2.items()}
        add("P-O2", m, "O2 within 0.15 of O1 at Llast@1 and Llast@3 (paired, r2_pass items)", dd or None,
            (all(x is not None and abs(x) <= 0.15 for x in dd.values()) if dd else None), harness=True, endpoint="e")
        b = ((ck.get("window", {}).get(m) or {}).get("dep_band") or {})
        x = (b.get("rel|full") or [None])[0]
        add("P-W full", m, "full relay knockout dep >= 0.8 (control)", x, cmp_(x, 0.8), endpoint="dep (first-token)")
        if m in OURO_REPOS:
            # v5 (plan critic 3 F1): 'the last loop still reads the span' scored by influence, necessity reported
            sl3 = ((ck.get("undo_slots", {}).get(m) or {}).get("slots") or {}).get("Llast@3") or {}
            x = (sl3.get("removal") or [None])[0]
            add("P-W L3", m, "influence: post-Llast@3 span reads carry >= 0.3 of the source edit's dm (removal)", x,
                cmp_(x, 0.3), endpoint="removal (first-token)",
                note="necessity dep_rel(after Llast@3) = %s" % ((b.get("rel|after_Llast@3") or [None])[0],))
        bc = ((ck.get("window", {}).get(m) or {}).get("dep_clipped_at_1") or {})
        pr = {bn: [(bc.get("rel|%s" % bn) or [None])[0], (bc.get("dir|%s" % bn) or [None])[0]] for bn in WIN_BOUNDS}
        okp = [None if (r_ is None or d_ is None) else r_ >= d_ for r_, d_ in pr.values()]
        add("P-W relay>=direct", m, "relay dep >= direct dep at every boundary, per-item dep clipped at 1 (v6, review "
            "A-F4/C-F7: tallied, not harness)", pr if bc else None, (all(okp) if (bc and None not in okp) else None),
            endpoint="min(dep, 1) (first-token)")
        if m in OURO_REPOS:
            sd = ck.get("strata_diff", {}).get(m) or {}
            for lab, k in (("ST1e", "src Llast@1 ST1e 1 - >=2"), ("ST2", "src Llast@1 ST2 shared - differs")):
                v = sd.get(k)
                add("P-ST %s" % lab, m, "source Llast@1 e differs by <= 0.1 between %s strata" % lab,
                    v, None if v is None else abs(v[0]) <= 0.1, endpoint="e_LO")
        if m in OURO_REPOS:                 # v5 (plan critic 3 F5): derived-use V1 rows (cited values, first-token e)
            src(m, "Llast@1", lo=0.8, place="placement_vbd", tag="VBD ")
            for j in (2, 3):
                ans(m, "Llast@%d" % j, lo=0.5, place="placement_vbd", tag="VBD ")
        cg = ck.get("carry_gain", {}).get(m) or {}
        gate_ok = ((ck.get("carry_gain_gate") or {}).get(m) or {}).get("interpretable")
        n_before = len(out)
        wfs = {}
        for ds, kind in A2_DATASETS:
            v = cg.get(ds) or {}
            mm = "%s/%s" % (m, ds)
            fs = v.get("dev_full_set_a1") or {}
            dh = (fs.get("hit_minus_latest") or [None])[0]
            add("A2-1 hit", mm, "dev full set, a = 1: selector hit - latest-site hit >= -0.10 (v4, plan critic 2 F2)",
                dh, cmp_(dh, -0.10), endpoint="tolerance hit (first-token dm)",
                note="band chance %s is a diagnostic" % ((fs.get("chance") or [None])[0],))
            lt = fs.get("loops_lt_K") or {}
            dl = (lt.get("hit_minus_latest") or [None])[0]
            okn = (lt.get("eligible") or 0) >= MIN_CL and (lt.get("n_clusters") or 0) >= MIN_CL
            add("A2-1 ltK", mm, "dev, loops 1..K-1, a = 1: selector hit - Llast@K-1 hit >= -0.10 (>= 10 eligible)",
                dl if okn else None, cmp_(dl, -0.10) if okn else None, endpoint="tolerance hit (first-token dm)",
                note=None if okn else "eligible %s < 10" % lt.get("eligible"))
            rg = (fs.get("regret") or [None])[0]
            add("A2-1 regret", mm, "dev full set, a = 1: mean regret <= 0.15", rg, cmp_(rg, None, 0.15),
                endpoint="dm / D")
            dp_ = v.get("dose_pair") or {}
            hd = (dp_.get("hit_diff_a0.25_minus_a1") or [None])[0]
            add("A2-1 dose", mm, "verified 9, items eligible at both doses (v6, review A-F1: band 0.05 x a x D): "
                "hit(a = 0.25) - hit(a = 1) >= -0.10", None if hd is None else [hd, dp_.get("n_items")],
                cmp_(hd, -0.10), endpoint="paired tolerance hit (first-token dm)")
            wf, wfg = v.get("working_site_fraction_dev"), v.get("working_site_fraction_dev_rnd_gated")
            wfs[ds] = (wf, wfg)
            if m.startswith("ifm_p1"):
                add("A2-2", mm, "working-site fraction <= 0.20 (twin, every dataset)", wf, cmp_(wf, None, 0.20),
                    endpoint="e (O1)")
            elif kind == "retrieval":
                add("A2-2", mm, "working-site fraction <= 0.20 (retrieval)", wf, cmp_(wf, None, 0.20), endpoint="e (O1)")
            else:                   # T3b v6: a lower bound counts a site only where the RND gate is open
                add("A2-2", mm, "working-site fraction >= 0.35 (computed)", wfg, cmp_(wfg, 0.35),
                    endpoint="e (O1, RND-gated sites)")
            td = (v.get("a=1") or {}).get("test_dm_over_D") or {}
            dd_ = (td.get("predicted_minus_dev_tuned") or [None])[0]
            add("A2-4", mm, "|selector - dev-tuned| <= 0.10 D", dd_, None if dd_ is None else abs(dd_) <= 0.10,
                endpoint="dm / D (test)")
            a1 = v.get("a=1") or {}
            el, n_ = a1.get("eligible"), a1.get("n_items")
            add("A2-5", mm, "eligible fraction (verified 9, a = 1) >= 0.6 [belief]", None if not n_ else [el, n_],
                None if not n_ else el / n_ >= 0.6, endpoint="dm")
        if not m.startswith("ifm_p1"):
            # v5 (plan critic 3 F5): the regime claim needs real math on both sides; D2 alone gives no regime claim
            rv_, cv_ = (wfs.get("VarBench") or (None, None))[0], (wfs.get("VBD") or (None, None))[1]
            add("A2-2 regime", m, "real-math regime contrast: VarBench (retrieval) <= 0.20 and VBD (computed) >= 0.35",
                None if (rv_ is None or cv_ is None) else [rv_, cv_],
                None if (rv_ is None or cv_ is None) else bool(rv_ <= 0.20 and cv_ >= 0.35), endpoint="e",
                note="no regime claim from D2 alone")
        if not gate_ok:                     # v5 (plan critic 3 F4): no V6 row is interpreted before the gate passes
            for rw in out[n_before:]:
                if rw["id"].startswith("A2") and rw["verdict"] != "not scored":
                    rw["verdict"] = "not scored"
                    rw["note"] = ((rw["note"] + "; ") if rw["note"] else "") + "V6 gate closed (CGCHECK / FD, plan critic 3 F4)"
                elif rw["id"].startswith("A2"):
                    rw["note"] = ((rw["note"] + "; ") if rw["note"] else "") + "V6 gate closed"

    # cross-model rows (P-ST second part) and the global undo-slot row (P-W)
    xm = ck.get("cross_model", {})
    for ma, mb in (("ouro", "ouro_think"), ("ouro26", "ouro26_think"), ("ouro", "ouro26"), ("ouro_think", "ouro26_think")):
        x = xm.get("%s|%s" % (ma, mb))
        if not x or not x["cells"]:
            add("P-ST %s" % ("think" if "think" in mb and "think" not in ma else "size"), "%s|%s" % (ma, mb),
                "paired difference within 0.15 at every V1 cell (intersection)", None, None, endpoint="primary per cell")
            continue
        if x.get("cross_host"):                      # v5 (plan critic 3 F3): pairs only within one host role
            add("P-ST %s" % ("think" if "think" in mb and "think" not in ma else "size"), "%s|%s" % (ma, mb),
                "paired difference within 0.15 at every V1 cell (intersection)", None, None, endpoint="primary per cell",
                note="cross-host pair %s: not scored" % x.get("host_roles"))
            continue
        hk = set(x.get("harness_cells", []))
        diffs = {k: (v["diff_B_minus_A"] or [None])[0] for k, v in x["cells"].items() if k not in hk}
        held = None if any(v is None for v in diffs.values()) else all(abs(v) <= 0.15 for v in diffs.values())
        worst = max(diffs, key=lambda k: abs(diffs[k]) if diffs[k] is not None else -1) if diffs else None
        add("P-ST %s" % ("think" if "think" in mb and "think" not in ma else "size"), "%s|%s" % (ma, mb),
            "paired difference within 0.15 at every V1 cell (intersection n %d)" % x["n_intersection"],
            {"max_abs_cell": worst, "max_abs": diffs.get(worst) if worst else None, "n_cells": len(diffs)}, held,
            endpoint="primary per cell")
    sel = {m: (ck.get("undo_slots", {}).get(m) or {}).get("used_in_M") for m in OURO_REPOS}
    have = {m: s for m, s in sel.items() if s is not None}
    cnt = sum(1 for s in have.values() if ("Llast@2" in s or "Llast@3" in s))
    add("P-W slots", "ouro x4", "selected undo slots include Llast@2 or Llast@3 on >= 3 of 4 Ouro checkpoints",
        None if len(have) < 4 else "%d/4" % cnt, None if len(have) < 4 else cnt >= 3, endpoint="V5 rule")
    t8 = ck.get("t8") or {}                 # v5 (T8 in clusters): a model under T8_MIN clusters is not interpreted
    for rw in out:
        mod = rw["model"].split("/")[0]
        bases = list(OURO_REPOS) if mod == "ouro x4" else mod.split("|")
        low = [m_ for m_ in bases if m_ in t8 and not t8[m_]["interpretable"]]
        if not rw["harness"] and low and rw["verdict"] != "not scored":
            rw["verdict"] = "not scored"
            rw["note"] = ((rw["note"] + "; ") if rw["note"] else "") + "T8: %s under %d clusters" % (",".join(low), T8_MIN)
    return out


def v4_selftest():
    """Plan critic 2 fixes F1-F8 (plan v4)."""
    import tempfile
    out = {}
    # F3: clause per slot (v5 rule; the v4 rule is in selftest window_rule)
    w1 = window_rule([0.5, 0.4, 0.6], [0.05, 0.0, 0.1], 0.5, 20, 20)
    w2 = window_rule([0.5, 0.4, 0.6], [0.4, 0.2, 0.6], 0.1, 12, 12)
    w3 = window_rule([0.1, 0.0, 0.2], [0.5, 0.3, 0.7], 0.5, 20, 20)
    w4 = window_rule([0.5, 0.4, 0.6], [0.4, 0.2, 0.6], 0.1, 12, 9)
    out["window_clause"] = [w1["selected_by"], w1["informative"], w2["selected_by"], w2["informative"], w3["selected_by"],
                            w4["selected_by"]]
    out["window_clause_ok"] = out["window_clause"] == ["a+b1_only", False, "a+b2", True, None, None]
    br = boot_ratio([0.1, 0.1, 0.0, None], [0.5, 0.5, 0.5, 1.0], ["a", "b", "c", "d"])
    out["boot_ratio"] = br
    out["boot_ratio_ok"] = (br is not None and abs(br[0] - 0.2 / 1.5) < 1e-4 and br[3] == 3 and br[4] == 3
                            and boot_ratio([1.0], [0.0], ["a"]) is None)
    # F4: plan hashes; section 13 edits keep the frozen hash, other edits change it; launch checks
    td = tempfile.mkdtemp()
    p1 = os.path.join(td, "plan.md")
    cp = os.path.join(td, "code.py")
    open(cp, "w").write("print(1)\n")
    csha = hashlib.sha256(open(cp, "rb").read()).hexdigest()
    base = "# t\n## 12. x\nA\nLaunch code sha256: `%s`\n## 13. Run log\nslots: none\n## 14. Review\nR\n" % csha
    open(p1, "w").write(base)
    h0 = plan_hashes(p1)
    open(p1, "w").write(base.replace("slots: none", "slots: Llast@2"))
    h1 = plan_hashes(p1)
    open(p1, "w").write(base.replace("R\n", "R2\n"))
    h2 = plan_hashes(p1)
    out["plan_hash_ok"] = (h0["plan_sha256"] != h1["plan_sha256"] and h0["plan_frozen_sha256"] == h1["plan_frozen_sha256"]
                           and h2["plan_frozen_sha256"] != h0["plan_frozen_sha256"])
    open(p1, "w").write(base)
    errs = []
    for args in ((None, None, "W"), (p1, "bad", "W"), (p1, h0["plan_sha256"], "all"), (p1, h0["plan_sha256"], "M")):
        try:
            plan_launch_check("ouro", td, args[2], False, args[0], args[1], code_path=cp)
            errs.append("passed")
        except SystemExit:
            errs.append("refused")
    lw = plan_launch_check("ouro", td, "W", False, p1, h0["plan_sha256"], code_path=cp)
    json.dump({"plan_launch": [lw]}, open(os.path.join(td, "vb_ouro.json"), "w"))
    open(p1, "w").write(base.replace("slots: none", "slots: Llast@2"))
    lm = plan_launch_check("ouro", td, "M", False, p1, h1["plan_sha256"], code_path=cp)
    open(p1, "w").write(base.replace("R\n", "R2\n"))
    try:
        plan_launch_check("ouro", td, "M", False, p1, h2["plan_sha256"], code_path=cp)
        errs.append("passed")
    except SystemExit:
        errs.append("refused")
    open(p1, "w").write(base)                       # v5 (plan critic 3 F1): code edited after the plan froze its hash
    open(cp, "w").write("print(2)\n")
    try:
        plan_launch_check("ouro", td, "W", False, p1, h0["plan_sha256"], code_path=cp)
        errs.append("passed")
    except SystemExit:
        errs.append("refused")
    dbg = plan_launch_check("ouro", td, "all", True, None, None, code_path=cp)
    out["plan_launch"] = errs
    out["plan_launch_ok"] = (errs == ["refused"] * 6 and lm["plan_sha256"] == h1["plan_sha256"]
                             and lw["code_matches_plan"] is True and lw["plan_code_sha256"] == csha
                             and dbg["debug"] is True and "plan_sha256" not in dbg and dbg["code_sha256"] != csha)
    # F5/F7: probe labels and the parameter-free probe (planted digit direction vs pure noise)
    out["first_diff"] = [first_diff_digits(1234, 1284), first_diff_digits("1,500", "1,700"), first_diff_digits(5, 5),
                         first_diff_digits(9, 12)]
    out["first_diff_ok"] = out["first_diff"] == [("3", "8"), ("5", "7"), None, ("9", "1")]
    g = torch.Generator().manual_seed(0)
    dirs = torch.randn(10, 32, generator=g)
    labs, cands, cl, Hs, Hn = [], [], [], [], []
    for i in range(60):
        a_, b_ = i % 10, (i * 3 + 1) % 10
        if a_ == b_:
            b_ = (b_ + 1) % 10
        for d_ in (a_, b_):
            Hs.append(dirs[d_] * 2.0 + torch.randn(32, generator=g))
            Hn.append(torch.randn(32, generator=g))
            labs.append(str(d_))
            cands.append((str(a_), str(b_)))
            cl.append("c%d" % (i // 2))
    cs = probe_site(torch.stack(Hs), labs, cands, cl)
    cn_ = probe_site(torch.stack(Hn), labs, cands, cl)
    acc_s = sum(x for x in cs if x is not None) / sum(1 for x in cs if x is not None)
    acc_n = sum(x for x in cn_ if x is not None) / sum(1 for x in cn_ if x is not None)
    out["probe_acc_signal_noise"] = [round(acc_s, 3), round(acc_n, 3)]
    out["probe_ok"] = acc_s >= 0.9 and 0.3 <= acc_n <= 0.7
    # probe_table end to end on toy states: the PC site (Llast@K of a K = 1 toy) carries the digit, L0 is noise
    items_, states_ = [], {}
    for i in range(40):
        xa, xb = 100 + i % 10, 100 + (i * 3 + 1) % 10
        if xa == xb:
            xb = 100 + (xb + 1 - 100) % 10
        da, db = str(xa)[2], str(xb)[2]
        items_.append({"pair_id": "p%d" % i, "cluster": "c%d" % i, "value_A": xa, "value_B": xb})
        AB = torch.stack([torch.stack([torch.randn(32, generator=g), torch.randn(32, generator=g)]),
                          torch.stack([torch.randn(32, generator=g), torch.randn(32, generator=g)]),
                          torch.stack([dirs[int(da)] * 2 + torch.randn(32, generator=g) * 0.5,
                                       dirs[int(db)] * 2 + torch.randn(32, generator=g) * 0.5])])
        states_["p%d" % i] = {"sites": ["1.0", "1.1", "1.2"], "AB": AB.half()}
    pt = probe_table(states_, items_, {"L0@1": (1, 0), "Lmid@1": (1, 1), "Llast@1": (1, 2)}, 1)
    out["probe_table"] = [pt["pc_ok"], pt["labels"].get("Llast@1"), pt["labels"].get("L0@1"), pt["n_items"]]
    out["probe_table_ok"] = (pt["pc_ok"] is True and pt["labels"]["Llast@1"] == "present"
                             and pt["labels"]["L0@1"] in ("absent", "unclear") and pt["n_items"] == 40)
    # F8: D4 components join A and B keys across rows

    def d4row(va, vb):
        return {"kind": "T2", "names": ["a", "b"], "src_var": "a", "query_vars": ["a", "b"], "vals_A": va, "vals_B": vb}
    rows_ = [d4row([1, 2], [3, 2]), d4row([3, 2], [5, 2]), d4row([7, 7], [8, 7])]
    cp = d4_components(rows_)
    k = [d4_content(r_, "A") for r_ in rows_]
    out["d4_components"] = [cp[k[0]] == cp[k[1]], cp[k[0]] == cp[k[2]], len(set(cp.values()))]
    out["d4_components_ok"] = out["d4_components"] == [True, False, 2]
    # F6: carry-gain strata keys; F2: hit - latest-site hit and the loops < K row
    out["cg_st"] = [cg_st({"strata": {"ST1": "1"}}, "ST1"), cg_st({"pos": [3, 4]}, "ST1"), cg_st({"pos": [3]}, "ST2")]
    out["cg_st_ok"] = out["cg_st"] == ["1", ">=2", None]

    def cg_item(pid, D, dm, G, split="dev"):
        it_ = {"pair_id": pid, "cluster": pid, "split": split, "clean_A": [0.0, -D / 2.0], "clean_B": [0.0, D / 2.0],
               "rows": {}, "G": G, "verify": list(G), "strata": {"ST1": "1", "ST2": "shared"}, "rnd_site": "1.0"}
        for k_, x_ in dm.items():
            for a_ in A_VERIFY:
                it_["rows"]["O1|ans|%s|%g" % (k_, a_)] = {"f": [0.0, -D / 2.0 + x_ * a_]}
        return it_
    # K = 2, set {1.0, 1.1, 1.2, 2.0}; the selector picks 1.1 on both items. Full set: latest 2.0 is best on i1,
    # 1.1 on i2 -> hit 0.5, latest hit 0.5. Loops < K: 1.1 is best on both, baseline Llast@K-1 = 1.2 never ->
    # hit 1.0, baseline 0.0; dm / D selector - baseline (0.2 + 0.6) / 2 = 0.4
    G_ = {"1.0": 0.0, "1.1": 1.0, "1.2": 0.2, "2.0": 0.5}
    toy_cg = [cg_item("i1", 10.0, {"1.0": 0.0, "1.1": 3.0, "1.2": 1.0, "2.0": 9.0}, G_),
              cg_item("i2", 10.0, {"1.0": 0.0, "1.1": 9.0, "1.2": 3.0, "2.0": 3.0}, G_)]
    cs_ = carry_summary(toy_cg, ["1.0", "1.1", "1.2", "2.0"], "2.0", "1.1", lambda it: it["cluster"], 2)
    fs = cs_["dev_full_set_a1"]
    out["carry_v4"] = [fs["top1_hit"][0], fs["latest_site_hit"][0], fs["hit_minus_latest"][0],
                       fs["loops_lt_K"]["baseline_site"], fs["loops_lt_K"]["hit_minus_latest"][0],
                       fs["loops_lt_K"]["dm_over_D_selector_minus_Llast_Km1"][0], sorted(fs["strata"]),
                       sorted(cs_["a=1"]["strata"])]
    out["carry_v4_ok"] = out["carry_v4"] == [0.5, 0.5, 0.0, "1.2", 1.0, 0.4, ["ST1", "ST1e", "ST2"], ["ST1", "ST2"]]
    out["ok"] = all(out[k_] for k_ in out if k_.endswith("_ok"))
    return out


def v5_selftest():
    """Plan critic 3 fixes F1-F8 (plan v5)."""
    out = {}

    def toy(i, s_=3.0, k_=-4.0, u_=0.0, U_=-4.0, rnd=None, o1=None):
        it = {"pair_id": "p%d" % i, "cluster": "c%d" % i, "clean_A": [0.0, -5.0], "clean_B": [0.0, 5.0], "rows": {
            "O1|src|Llast@1|1": {"f": [0.0, s_]}, "UKO|Llast@2": {"f": [0.0, u_]}, "KO|rel|after_Llast@2": {"f": [0.0, k_]},
            "U|Llast@2": {"f": [0.0, U_]}, "BKO|Llast@2": {"f": [0.0, k_ + 10.0]}}, "strata": {}, "tA": 0, "tB": 1}
        if o1 is not None:
            it["rows"]["O1|ans|Llast@1|1"] = {"f": [0.0, -5.0 + 10.0 * o1, 0.0, 1]}
        if rnd is not None:
            it["rows"]["RND|ans|Llast@1|1"] = {"f": [0.0, -5.0 + 10.0 * rnd, 0.0, 0]}
        return it
    # F1: v5 slot statistics (D = 10, tot = 8, car = 4, U = 1): removal 0.5, carried / D 0.4, survival 0.25
    # (v6: b2 on car / D_KO; here D_KO = D = 10, so the v5 expectations hold)
    its = [toy(i, u_=0.0 + 0.01 * (i % 3), U_=-4.0 + 0.01 * (i % 2)) for i in range(12)]
    st = slot_stats(its, "Llast@2")
    w = window_rule(st["removal"], st["carried_over_DKO"], 0.0, st["n_items"], st["n_clusters"], n_b2=st["n_b2"],
                    ncl_b2=st["n_clusters_b2"])
    out["slot_stats"] = [st["removal"][0], st["carried_over_D"][0], st["survival"][0], t6_verdict(st["survival"]),
                         w["selected_by"], w["informative"]]
    out["slot_stats_ok"] = (abs(st["removal"][0] - 0.5) < 0.01 and abs(st["carried_over_D"][0] - 0.4) < 0.01
                            and abs(st["survival"][0] - 0.25) < 0.01 and t6_verdict(st["survival"]) == "reverts"
                            and w["selected_by"] == "a+b2" and w["informative"] is True)
    out["t6_ok"] = (t6_verdict([0.8, 0.75, 0.9]) == "stays" and t6_verdict([0.5, 0.2, 0.8]) == "partial"
                    and t6_verdict([0.5, None, None]) is None)
    # F2: RND pairing and the citation rule
    a = [toy(i, o1=0.3 + 0.01 * (i % 2), rnd=0.2) for i in range(12)]
    b = [toy(i, o1=0.3 + 0.01 * (i % 2), rnd=0.0) for i in range(12)]
    cla = [it["cluster"] for it in a]
    ca, cb = cell(a, "O1|ans|Llast@1|1", cla), cell(b, "O1|ans|Llast@1|1", cla)
    ca["rnd_pair"], _, _ = rnd_pair(a, "O1|ans|Llast@1|1", cla)
    cb["rnd_pair"], _, _ = rnd_pair(b, "O1|ans|Llast@1|1", cla)
    va, la, ga = cited_value(ca)
    vb_, lb, gb = cited_value(cb)
    nr = rnd_pair([toy(0, o1=0.3)], "O1|ans|Llast@1|1", ["c0"])[0]
    ac = answer_class(a, "O1|ans|Llast@1|1", 1, "ifm", True, rnd_fail=True)
    # v6 (review C-F2): the cited value stays O1 (0.305) where RND fails; the gate (O1 - RND = 0.105, CI > 0) is open
    out["rnd_pair"] = [ca["rnd_pair"]["rnd_T3"], round(va[0], 4), la, ga, cb["rnd_pair"]["rnd_T3"], round(vb_[0], 4), lb,
                       gb, nr["citable"], ac["final_e"], ac["final_e_basis"]]
    out["rnd_pair_ok"] = (ca["rnd_pair"]["rnd_T3"] is False and abs(va[0] - 0.305) < 1e-3 and "RND fails" in la
                          and ga is True and cb["rnd_pair"]["rnd_T3"] is True and abs(vb_[0] - 0.305) < 1e-3
                          and lb == "O1" and gb is True and nr["citable"] is False and abs(ac["final_e"] - 0.305) < 1e-3
                          and ac["final_e_basis"].startswith("O1"))
    # F1: planted power control. Site 0: A and B states identical (planted code has norm 0: no power); site 1: noise
    # with distinct A/B (planted detected); site 2 = PC (digit code present)
    g = torch.Generator().manual_seed(3)
    dirs = torch.randn(10, 32, generator=g)
    items_, states_ = [], {}
    for i in range(40):
        xa, xb = 100 + i % 10, 100 + (i * 3 + 1) % 10
        if xa == xb:
            xb = 100 + (xb + 1 - 100) % 10
        da, db = str(xa)[2], str(xb)[2]
        items_.append({"pair_id": "p%d" % i, "cluster": "c%d" % i, "value_A": xa, "value_B": xb})
        same = torch.randn(32, generator=g)
        AB = torch.stack([torch.stack([same, same]),
                          torch.stack([torch.randn(32, generator=g), torch.randn(32, generator=g)]),
                          torch.stack([dirs[int(da)] * 2 + torch.randn(32, generator=g) * 0.5,
                                       dirs[int(db)] * 2 + torch.randn(32, generator=g) * 0.5])])
        states_["p%d" % i] = {"sites": ["1.0", "1.1", "1.2"], "AB": AB.half()}
    pt = probe_table(states_, items_, {"L0@1": (1, 0), "Lmid@1": (1, 1), "Llast@1": (1, 2)}, 1)
    out["probe_planted"] = [pt["labels"].get("L0@1"), pt["labels"].get("Lmid@1"), pt["labels"].get("Llast@1"),
                            (pt["planted"].get("L0@1") or [None])[0], (pt["planted"].get("Lmid@1") or [None])[0]]
    out["probe_planted_ok"] = (pt["labels"]["L0@1"] in ("no power at site", "unclear")
                               and (pt["planted"]["L0@1"][0] or 0) < PROBE_PC
                               and pt["planted"]["Lmid@1"][0] >= PROBE_PC and pt["labels"]["Lmid@1"] in ("absent", "unclear")
                               and pt["labels"]["Llast@1"] == "present")
    # F7: ST1e
    out["st1e"] = [st1e([1, 2, 3], [1, 5, 3], [0, 1, 2]), st1e([1, 2], [3, 4], [0, 1]), st1e([1, 2], [1, 2], [0, 1])]
    out["st1e_ok"] = out["st1e"] == ["1", ">=2", "1"]
    # F5: derived-use text
    q = "Tom has 3 bags with 4 apples each."
    sol = "Each bag has 4 apples, so 3*4 = <<3*4=12>>12 apples.\n#### 12"
    rec, why = vbd_text(sol, q, [8, 9], 3, 5)
    rec2, why2 = vbd_text("He gets 7 more, 3*4 = <<3*4=12>>12.\n#### 12", q, [8, 9], 3, 5)
    rec3, why3 = vbd_text("Each bag has 4 apples, so 4+1 = <<4+1=5>>5.\n#### 5", q, [8, 9], 3, 5)
    out["vbd_text"] = [rec and rec["C_A"][-12:], rec and rec["C_B"][-12:], rec and rec["res_A"], rec and rec["res_B"],
                       why2, why3]
    out["vbd_text_ok"] = (rec is not None and rec["C_A"].endswith("3*4 = <<3*4=") and rec["C_B"].endswith("5*4 = <<5*4=")
                          and rec["res_A"] == "12" and rec["res_B"] == "20" and why2 == "prose_unverified_number"
                          and why3 == "no_derived_step")
    out["safe_eval_ok"] = (safe_eval("3*4+2") == 14.0 and safe_eval("10/4") == 2.5 and safe_eval("__import__('os')") is None
                           and fmt_num(2.5) == "2.5" and fmt_num(14.0) == "14" and fmt_num(1 / 3) == "0.3333")
    # F4: FD rule
    good = [{"a": 0.01, "aG": 0.01 * (k + 1), "dm": 0.01 * (k + 1) * 1.01} for k in range(20)]
    bad = [{"a": 0.01, "aG": 0.01 * (k + 1), "dm": -0.01 * (k + 1)} for k in range(20)]
    low = [{"a": 0.01, "aG": 1e-6, "dm": 0.5}]
    out["fd"] = [fd_summary(good)["pass"], fd_summary(bad)["pass"], fd_summary(good + low)["a=0.01"]["n_below_floor"]]
    out["fd_ok"] = out["fd"] == [True, False, 1]
    # v5.1: bf16 selector check by the T7 band. Item 1: near-tie (bf16 picks b, fp32 b is 0.05 below a, D 10 ->
    # inside the 0.5 band); item 2: same top; item 3: bf16 picks a site 2 logits below the fp32 top (outside).
    g32a = {"a": 3.0, "b": 2.95, "c": 0.0}
    tie = {"p1": {"D": 10.0, "G_fp32": g32a, "G_bf16": {"a": 2.9, "b": 2.96, "c": 0.0}},
           "p2": {"D": 10.0, "G_fp32": g32a, "G_bf16": {"a": 3.1, "b": 2.9, "c": 0.0}}}
    far = dict(tie, p3={"D": 10.0, "G_fp32": {"a": 3.0, "b": 1.0, "c": 0.0}, "G_bf16": {"a": 0.5, "b": 1.2, "c": 0.0}})
    s1, ok1 = bf16_agree(tie)
    s2, ok2 = bf16_agree(far)
    out["bf16_agree"] = [s1["top_site_same"], s1["top_in_fp32_band"], ok1, s2["top_in_fp32_band"], ok2]
    out["bf16_agree_ok"] = out["bf16_agree"] == [[1, 2], [2, 2], True, [2, 3], False]
    # F3: host-mix refusal
    rs = {"host_role": "ouro_host", "plan_launch": [{"host_role": "ouro_host"}]}
    try:
        host_mix_check(rs, "ifm_host", "ouro")
        hm = "passed"
    except SystemExit:
        hm = "refused"
    out["host_mix"] = [hm, host_mix_check(rs, "ifm_host", "ouro", allow=True), host_mix_check(rs, "ouro_host", "ouro"),
                       host_mix_check({}, "ifm_host", "ouro")]
    out["host_mix_ok"] = out["host_mix"] == ["refused", ["ouro_host"], ["ouro_host"], []]
    ch = code_hashes()
    out["code_hash_ok"] = ch["code_sha256"] == hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    out["ok"] = all(out[k_] for k_ in out if k_.endswith("_ok"))
    return out


def v6_selftest():
    """Code review 3 fixes (plan section 14 v6)."""
    import tempfile
    out = {}
    # A-F1: the reviewer's case. 9 sites, D = 10, dm(0.25) = 0.25 dm(1), the top-G site 0.1 D below the best.
    G9 = {"%d.0" % i: float(9 - i) for i in range(9)}
    dm1 = {k: v for k, v in zip(G9, [9.0, 10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
    dm25 = {k: 0.25 * v for k, v in dm1.items()}
    h1 = hit_metrics(G9, dm1, 10.0, THR[1.0], tol=HIT_TOL * 1.0)
    h25_old = hit_metrics(G9, dm25, 10.0, THR[0.25])
    h25 = hit_metrics(G9, dm25, 10.0, THR[0.25], tol=HIT_TOL * 0.25)
    out["dose_band"] = [h1["hit"], h25_old["hit"], h25["hit"], round(h1["chance"], 4), round(h25["chance"], 4)]
    out["dose_band_ok"] = out["dose_band"] == [0, 1, 0, round(1 / 9, 4), round(1 / 9, 4)]

    def cg_item(pid, D, dm, G, split="dev"):
        it_ = {"pair_id": pid, "cluster": pid, "split": split, "clean_A": [0.0, -D / 2.0], "clean_B": [0.0, D / 2.0],
               "rows": {}, "G": G, "verify": list(G), "strata": {}, "rnd_site": "1.0"}
        for k_, x_ in dm.items():
            for a_ in A_VERIFY:
                it_["rows"]["O1|ans|%s|%g" % (k_, a_)] = {"f": [0.0, -D / 2.0 + x_ * a_]}
        return it_
    G_ = {"1.0": 0.0, "1.1": 1.0, "1.2": 0.2, "2.0": 0.5}
    toy = [cg_item("i1", 10.0, {"1.0": 0.0, "1.1": 9.0, "1.2": 1.0, "2.0": 10.0}, G_),
           cg_item("i2", 10.0, {"1.0": 0.0, "1.1": 9.0, "1.2": 3.0, "2.0": 3.0}, G_),
           cg_item("i3", 10.0, {"1.0": 0.0, "1.1": 1.0, "1.2": 1.2, "2.0": 1.1}, G_)]   # i3: range 1.2, not eligible
    dp = carry_summary(toy, list(G_), "2.0", "1.1", lambda it: it["cluster"], 2)["dose_pair"]
    out["dose_pair"] = [dp["n_items"], dp["hit_a1"][0], dp["hit_a0.25"][0], dp["hit_diff_a0.25_minus_a1"][0]]
    out["dose_pair_ok"] = out["dose_pair"] == [2, 0.5, 0.5, 0.0]
    # C-F2: T3b v6 gate in predictions. O1 0.9 at source Llast@1 on ouro; RND fails T3; gate closed -> missed with note
    def ck_with(gate):
        c = {"primary": [0.9, 0.85, 0.95, 20, 20], "endpoint": "e_LO", "flip": [0.9, 0.85, 0.95, 20, 20],
             "cited_endpoint": "e_LO", "rnd_pair": {"citable": True, "rnd_T3": False, "lower_gate": gate,
                                                    "cite": "O1 (RND fails T3: lower bounds gated on O1 - RND)"}}
        c2 = dict(c, primary=[0.05, 0.0, 0.1, 20, 20])
        return {"placement": {"ouro": {"O1|src|Llast@1": c, "O1|ans|L0@1": c2}}}
    rows_c = {(r_["id"], r_["model"]): r_ for r_ in predictions(ck_with(False), {})}
    rows_o = {(r_["id"], r_["model"]): r_ for r_ in predictions(ck_with(True), {})}
    out["t3b_gate"] = [rows_c[("src Llast@1", "ouro")]["verdict"], rows_o[("src Llast@1", "ouro")]["verdict"],
                       rows_c[("ans L0@1", "ouro")]["verdict"]]
    out["t3b_gate_ok"] = out["t3b_gate"] == ["missed", "held", "held"]
    pr0 = predictions({}, {})
    out["relay_not_harness_ok"] = all(r_["harness"] is False for r_ in pr0 if r_["id"] == "P-W relay>=direct")
    # B-F1: random relay keys only before the span
    rk = rand_ko_keys("gsm8k-188-buy_rate", [591, 592], 655)
    out["rand_keys"] = rk
    out["rand_keys_ok"] = (len(rk) == 2 and all(1 <= k < 591 for k in rk)
                           and rk == rand_ko_keys("gsm8k-188-buy_rate", [591, 592], 655))
    # C-F3: number words in the prefix
    q357 = "Jack has a budget of $90. Each ticket costs $30 and each meal $5."
    C357 = "Each person costs $30 because 5 + 5 + 20 = <<5+5+20=30>>30\nThree total people can attend because "
    out["prose"] = [words_val("twenty-five"), words_val("two hundred and ten"), prose_flags(C357, q357, 90),
                    prose_flags("In three years Jerry will be ", "Jerry is 10 and in 3 years", 3),
                    prose_flags("He buys <<4*5=20>>20 apples and ", "He buys 4 bags of 5 apples and 7 pears", 7)]
    out["prose_ok"] = out["prose"] == [25.0, 210.0, ["prose_unverified_number"], ["x_word_form"], []]
    # C-F1: b2 on car / D_KO. D = 10, D_KO = 2.5, car = 1.5: car / D 0.15 fails b2, car / D_KO 0.6 passes it
    def ut(i, dko):
        return {"pair_id": "p%d" % i, "cluster": "c%d" % i, "clean_A": [0.0, -5.0], "clean_B": [0.0, 5.0], "rows": {
            "O1|src|Llast@1|1": {"f": [0.0, 3.0]}, "UKO|Llast@2": {"f": [0.0, -2.5 + 0.01 * (i % 3)]},
            "KO|rel|after_Llast@2": {"f": [0.0, -4.0]}, "U|Llast@2": {"f": [0.0, -4.5]},
            "BKO|Llast@2": {"f": [0.0, -4.0 + dko]}}, "strata": {}, "tA": 0, "tB": 1}
    st = slot_stats([ut(i, 2.5) for i in range(12)], "Llast@2")
    w = window_rule(st["removal"], st["carried_over_DKO"], 0.0, st["n_items"], st["n_clusters"], n_b2=st["n_b2"],
                    ncl_b2=st["n_clusters_b2"])
    st_g = slot_stats([ut(i, 1.5) for i in range(12)], "Llast@2")          # D_KO 1.5 < guard: b2 has no items
    out["b2_dko"] = [round(st["carried_over_D"][0], 3), round(st["carried_over_DKO"][0], 3), round(st["DKO_over_D"][0], 3),
                     w["b_carried"], st_g["n_b2"], st_g["n_dko_below_guard"]]
    out["b2_dko_ok"] = (abs(st["carried_over_D"][0] - 0.151) < 0.01 and abs(st["carried_over_DKO"][0] - 0.604) < 0.01
                        and abs(st["DKO_over_D"][0] - 0.25) < 1e-6 and w["b_carried"] is True and st_g["n_b2"] == 0
                        and st_g["n_dko_below_guard"] == 12)
    # A-F3: bf16 check bands (full set and loops < K) and the Clopper-Pearson bound
    Gf = {"1.0": 3.0, "1.1": 2.0, "2.0": 3.5}
    pr = {"p%d" % i: {"D": 10.0, "G_fp32": Gf, "G_bf16": {"1.0": 2.0, "1.1": 2.6, "2.0": 3.6}} for i in range(3)}
    s1, ok1 = bf16_agree(pr, K=2, min_items=3)
    s2, ok2 = bf16_agree(pr, K=2, min_items=4)
    out["bf16_v6"] = [s1["top_in_fp32_band"], s1["loops_lt_K_top_in_fp32_band"], ok1, ok2, cp_lower(4, 4),
                      cp_lower(30, 30)]
    out["bf16_v6_ok"] = (s1["top_in_fp32_band"] == [3, 3] and s1["loops_lt_K_top_in_fp32_band"] == [0, 3]
                         and ok1 is False and ok2 is False and abs(cp_lower(4, 4) - 0.3976) < 2e-3
                         and abs(cp_lower(30, 30) - 0.8843) < 2e-3)
    pr2 = {k_: dict(v_, G_bf16={"1.0": 3.1, "1.1": 2.0, "2.0": 3.6}) for k_, v_ in pr.items()}
    out["bf16_v6_pass_ok"] = bf16_agree(pr2, K=2, min_items=3)[1] is True
    # A-F6: CGCHECK gate needs the full-run file
    dbg = {"pass": {"verdict": "PASS", "failed_cells": [], "insufficient_cells": ["D1/SRC"]}}
    full = {"pass": {"verdict": "PASS", "failed_cells": [], "insufficient_cells": []}}
    out["cg_gate"] = [cgcheck_gate(dbg, CGCHECK_SHA256)["pass"], cgcheck_gate(full, "0" * 64)["pass"],
                      cgcheck_gate(full, CGCHECK_SHA256)["pass"], cgcheck_gate(None, None)["pass"]]
    out["cg_gate_ok"] = out["cg_gate"] == [False, False, True, None]
    # C-F5: a GPU change refuses the results file; C-F4 (G8): data hashes in the launch record
    rs = {"host_role": "ouro_host", "gpu_name": "GPU-A", "plan_launch": [{"host_role": "ouro_host", "gpu_name": "GPU-A"}]}
    try:
        host_mix_check(rs, "ouro_host", "ouro", gpu_name="GPU-B")
        hm = "passed"
    except SystemExit:
        hm = "refused"
    out["host_gpu"] = [hm, host_mix_check(rs, "ouro_host", "ouro", gpu_name="GPU-A")]
    out["host_gpu_ok"] = out["host_gpu"] == ["refused", ["ouro_host"]]
    td = tempfile.mkdtemp()
    open(os.path.join(td, "items_ouro.json"), "w").write("{}")
    dh = data_hashes(td)
    ln = plan_launch_check("ouro", td, "all", True, None, None, data_dir=td, config={"a": 1}, gpu_name="GPU-A")
    out["g8"] = [dh["items_ouro.json"] == hashlib.sha256(b"{}").hexdigest(), dh["items_ifm.json"],
                 ln["data_sha256"] == dh, len(ln["config_sha256"]), ln["gpu_name"]]
    out["g8_ok"] = out["g8"] == [True, None, True, 64, "GPU-A"]
    # C-F4 (G4): path-integrated gain on a toy runner with m = |h|^2 at one site: G_PI = 2 hA.d + 2 mean(a) |d|^2

    class ToyR(object):
        Hd, adt, dev, n_fwd = 3, torch.float32, "cpu", 0

        def run(self, ids, r, edits=None, rec=None, ko=None, leaves=None, cont=False, cap=None, grad=False, **kw):
            h = torch.tensor([[[0.0] * 3, [1.0, 2.0, 0.5]]])
            s_ = (1, 0)
            with torch.set_grad_enabled(grad):
                if leaves and s_ in leaves:
                    h = h + leaves[s_]
                m_ = (h[0, r] ** 2).sum()
                return {"z_r": torch.stack([torch.zeros(()), m_])}

    class ToyX(object):
        seqA, r, tA, tB = [0, 1], 1, 0, 1
        cA = {"H": {(1, 0): torch.tensor([[1.0, 2.0, 0.5]])}}
        cB = {"H": {(1, 0): torch.tensor([[2.0, 1.0, 1.5]])}}
        it = {}
    gp = pi_gain(ToyR(), ToyX(), [(1, 0)])["1.0"]
    hA, d_ = torch.tensor([1.0, 2.0, 0.5]), torch.tensor([1.0, -1.0, 1.0])
    want = float(2 * hA @ d_ + 2 * (sum(PI_A) / len(PI_A)) * (d_ @ d_))
    out["pi_gain"] = [gp, round(want, 5), ToyX.it.get("pi_passes")]
    out["pi_gain_ok"] = abs(gp - want) < 1e-4 and ToyX.it.get("pi_passes") == 4
    # pi_summary: on two dev items the PI ranking hits where first-order G misses
    t1 = cg_item("j1", 10.0, {"1.0": 0.0, "1.1": 9.0, "1.2": 1.0, "2.0": 3.0}, {"1.0": 0.0, "1.1": 0.1, "1.2": 0.9, "2.0": 0.5})
    t2 = cg_item("j2", 10.0, {"1.0": 0.0, "1.1": 9.0, "1.2": 1.0, "2.0": 3.0}, {"1.0": 0.0, "1.1": 0.1, "1.2": 0.9, "2.0": 0.5})
    for t_ in (t1, t2):
        t_["G_PI"] = {"1.0": 0.0, "1.1": 2.0, "1.2": 0.5, "2.0": 0.4}
        t_["pi_passes"] = 16
    ps = pi_summary([t1, t2], ["1.0", "1.1", "1.2", "2.0"], 2, lambda it: it["cluster"])
    out["pi_summary"] = [ps["eligible"], ps["pi_top1_hit"][0], ps["g_top1_hit"][0], ps["pi_minus_g_hit"][0]]
    out["pi_summary_ok"] = out["pi_summary"] == [2, 1.0, 0.0, 1.0]
    out["ok"] = all(out[k_] for k_ in out if k_.endswith("_ok"))
    return out


def predictions_coverage_selftest():
    """Every section 8 row is emitted, as 'not scored' when nothing ran (review F3)."""
    rows = predictions({}, {})
    by = {}
    for rw in rows:
        m = rw["model"].split("/")[0]
        by[m] = by.get(m, 0) + 1
    # per model: model rows + 7 (Ouro) or 4 (IFM) global rows; A2 rows counted under "<model>/<dataset>" -> m
    # v4 (plan critic 2 F1/F2): A2-3 dropped from the tally (Ouro: -5), A2-1 ltK added on every model (+5)
    # v5 (plan critic 3 F5): VBD adds 7 A2 rows on every model, 3 V1 rows on Ouro, and one A2-2 regime row on Ouro
    # and d112 (not on the IFM twins, whose A2-2 rule is 'every dataset <= 0.20')
    want = {"ouro": 84, "ouro_think": 84, "ouro26": 84, "ouro26_think": 84, "ifm_p1_ouro": 75, "ifm_p1_ouro_inj": 85,
            "ifm_d112": 71}
    cross = sum(1 for rw in rows if "|" in rw["model"] or rw["model"] == "ouro x4")
    ok = all(by.get(m) == n for m, n in want.items()) and cross == 5 and all(rw["verdict"] == "not scored" for rw in rows)
    return {"rows": len(rows), "per_model": {m: by.get(m) for m in want}, "cross_and_global": cross, "ok": bool(ok)}


def main():
    global DROP_BLOCK, ROLE, ALLOW_HOST_MIX, PI_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build_items", action="store_true")
    ap.add_argument("--cpu_check", action="store_true")
    ap.add_argument("--fd_check", action="store_true", help="v5 F4: per-family FD check of G (bf16 then fp32)")
    ap.add_argument("--build_vbd", action="store_true", help="v5 F5: build only the derived-use items")
    ap.add_argument("--allow_host_mix", action="store_true", help="v5 F3: continue a results file from another host role")
    ap.add_argument("--cgcheck", default="", help="B-K17-CGCHECK checks.json (V6 gate at summarize)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", default="ouro")
    ap.add_argument("--datasets", default="VBD,D2,D1,D4")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "out", "k17_varbench", "data"))
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="p")
    ap.add_argument("--phase", default="all", choices=["W", "M", "all"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--budget_min", type=float, default=0.0)
    ap.add_argument("--n_vb", type=int, default=0)
    ap.add_argument("--pool", default="s200", choices=["s200", "problems", "pairs"])
    ap.add_argument("--model_dir", default="", help="local snapshot for the first model in --models")
    ap.add_argument("--model_dirs", default="", help="name=path,name=path (local snapshots, no hub download)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plan", default="", help="plan file: hashed at launch (runs) and at summarize")
    ap.add_argument("--plan_sha", default="", help="plan sha256 the coordinator wrote to the LEDGER (full runs)")
    ap.add_argument("--drop_block", action="store_true", help="laptop debug only (plan DV9)")
    ap.add_argument("--gset", default="", help="answer-site set: 'full' | 'red' | 'name=full,name=red' (review F9)")
    ap.add_argument("--n_sub", default="", help="phase-W subset size: 'N' | 'name=N,...' (review F9)")
    ap.add_argument("--a2_n", default="", help="A2 items per dataset: 'N' | 'name=N,...' (review F9)")
    ap.add_argument("--vbd_n", default="", help="v5 VBD items: 'N' | 'name=N,...'")
    ap.add_argument("--vbd_v1", default="", help="B-EXT2 item 5: VBD V1 cells on (1) or off (0): '0|1' | "
                    "'name=0|1,...' (default VBD_V1)")
    ap.add_argument("--role", default="", help="host role in records instead of the host name (A7)")
    ap.add_argument("--bf16_check", action="store_true", help="v6 A-F3: bf16 vs fp32 selector check on fresh items")
    ap.add_argument("--bf16_n", type=int, default=BF16_N, help="v6: items in the bf16 check")
    ap.add_argument("--fd_dir", default="", help="v6: directory holding fd_<model>.json (FD items are excluded)")
    ap.add_argument("--no_gradient_rank", action="store_true",
                    help="cluster copy: skip the gradient score, its verify rows and G_PI (site selection is not run)")
    ap.add_argument("--pi_n", type=int, default=-1, help="v6 G4: path-integrated gain dev items per dataset (0 = off)")
    ap.add_argument("--stage", default="all", choices=("all", "scan", "rows", "merge"),
                    help="all = one process; cluster (RA-20): scan (phase W if asked + clean-pass scans), rows "
                         "--shard k/n, merge (one --out for every stage)")
    ap.add_argument("--shard", default="", help="k/n for --stage rows: items with sha256 partition of pair_id == k")
    a = ap.parse_args()
    shard = None
    if a.shard:
        k_, n_ = (int(t) for t in a.shard.split("/"))
        assert 0 <= k_ < n_, a.shard
        shard = (k_, n_)
    assert (a.stage == "rows") == (shard is not None), "--shard k/n goes with --stage rows only"
    if a.stage != "all":
        a.tag = "%s.%s%s" % (a.tag, a.stage, shard_sfx(shard))
    if a.pi_n >= 0:
        PI_N = a.pi_n
    NO_GRAD[0] = a.no_gradient_rank
    DROP_BLOCK = a.drop_block
    ROLE = a.role
    ALLOW_HOST_MIX = a.allow_host_mix
    GSET.update(per_model(a.gset, GSET))
    N_SUB.update(per_model(a.n_sub, N_SUB, int))
    A2_N.update(per_model(a.a2_n, A2_N, int))
    VBD_N.update(per_model(a.vbd_n, VBD_N, int))
    VBD_V1.update(per_model(a.vbd_v1, VBD_V1, lambda v: {"0": False, "1": True}[v]))
    assert all(v in ("full", "red") for v in GSET.values()), GSET
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.build_items:
        build_items(a.out)
        return
    if a.build_vbd:
        build_vbd(a.out)
        return
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out, a.plan, a.cgcheck or None)
        return
    if a.model_dir:
        MODEL_DIR[a.models.split(",")[0]] = a.model_dir
    for kv in [x for x in a.model_dirs.split(",") if x]:
        k_, v_ = kv.split("=", 1)
        MODEL_DIR[k_] = os.path.expanduser(v_)
    t0 = time.time()
    ok = True
    try:
        for name in [x for x in a.models.split(",") if x]:
            if a.cpu_check:
                ok = cpu_check(name, a.data, a.out, a.device) and ok
            elif a.fd_check:
                fd_check(name, a.data, a.out, a.device)
            elif a.bf16_check:
                bf16_check(name, a.data, a.out, a.device, n=a.bf16_n, fd_dir=a.fd_dir)
            else:
                run_model(name, a.data, a.out, a.device, a.phase, a.debug, a.budget_min,
                          [x for x in a.datasets.split(",") if x], a.n_vb or None, a.pool, a.plan, a.plan_sha,
                          stage=a.stage, shard=shard)
    finally:
        json.dump({"models": a.models, "seconds": round(time.time() - t0, 1), "host_role": ROLE or role_of(a.models),
                   "device": a.device, "mode": "cpu_check" if a.cpu_check else ("fd_check" if a.fd_check else (
                       "bf16_check" if a.bf16_check else ("debug" if a.debug else "full"))),
                   "phase": a.phase, "stage": a.stage, "shard": list(shard) if shard else None, "drop_block": DROP_BLOCK,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                  open(os.path.join(a.out, "proc_%s.json" % a.tag), "w"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
