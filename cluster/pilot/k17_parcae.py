"""K17 Parcae as a fifth looped family (brief B-K17-PARCAE; PLAN.md amendments A1-A8).
Plan (frozen definitions, thresholds, predictions): notes/k17_parcae_plan.md (v2).

Model: SandyResearch parcae-1.3b (main) and parcae-370m (replication), fp32, K = 8 recurrences, batch 1.
Structure (parcae_lm commit 69284c13): wte -> prelude -> ln_prelude = e -> 8 x [adapter(x, e) = x*decay +
dt*(e @ B.T), core blocks with value embeds] -> C -> coda (value embeds) -> ln_f -> lm_head.
Sites: post_inj(j) = adapter output at loop j, core block l output at loop j ("c<j>.<l>"), coda block c output.
Arms (plan section 8): E1 placement, E5 input protocols (+ AD5 downstream-input and state-only rows), E8 carry
gain (predict then verify), presence and answer classes. Controls: PC-final, SZ, PC-site [harness], RND, STA
(informative positive control). Datasets D1, D2, D4, D5 and the VarBench copy-read target (VB).

    python k17_parcae.py --selftest                                   (pure helpers, CPU)
    python k17_parcae.py --selftest_model                             (tiny random Parcae, CPU; needs parcae_lm)
    python k17_parcae.py --build_items --out DATA_DIR --tok TOKENIZER_JSON          (laptop CPU)
    python k17_parcae.py --run --model parcae-370m --data DATA_DIR --out OUT [--debug N] [--datasets D1,D2]
    python k17_parcae.py --scan_only --model parcae-140m --data DATA_DIR --out OUT
    python k17_parcae.py --summarize --out OUT [--plan PLAN.md]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get("PARCAE_LM_PATH"):            # CPU unit test against a local copy of the installed source
    sys.path.insert(0, os.environ["PARCAE_LM_PATH"])

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ------------------------------------------------------------------ constants (plan sections 5-10)
K = 8
E1_LOOPS = (1, 2, 4, 7, 8)
VB_LOOPS = (1, 4, 7, 8)
O2_SRC = ("Llast@1", "Llast@7", "Llast@8", "post_inj@8")
O2_ANS = ("Llast@1", "Llast@7", "Lmid@8", "Llast@8")
STA_NAMES = ("Llast@1", "Llast@4", "Llast@7", "post_inj@8")
TARGET = {"D1": 80, "D2": 60, "D4": 60, "D5": 60, "VB": 60}
DATASETS = ("D1", "D2", "D4", "D5", "VB")
# A1 / plan section 9 strata on every reported cell (review round 2 F7); O2 cells add para_fmt
STRATA = {"D1": ("pos_len", "rel"), "D2": ("pos_len", "rel"), "D4": ("pos_len", "rel"), "D5": ("pos_len", "rel"),
          "VB": ("pos_len", "ST2", "ST3", "cont")}
D1_MULTI_MAX = 40
D1_SUBJ_CAP = 2
D2_PER_OP = 30
D2_CLUSTER_CAP = 2
D4_PER_CELL = 15
D5_PER_SPLIT = 30
MIN_KEPT = 40
MIN_E8_TEST = 20
N_SUB = 20
VB_DEV_SEARCH = 20
GAP = 2.0
NBOOT = 2000
TOST_M = 0.10
PR3_M = 0.15
A_VERIFY = (1.0, 0.25)
THR = {1.0: 2.0, 0.25: 0.5}
SEED_OFFSETS = (0, 1, 2)
PARCAE_SHA16 = "c0d792379c3e965e"
PARCAE_COMMIT = "69284c13746e849104f738d6d1a347b1f457df76"
from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE  # noqa: E402  (honours HF_HOME)
HUB = _HF_HUB_CACHE
NO_GRAD = [False]     # cluster copy (--no_gradient_rank): skip E8 (gradient score, verify rows, dev search) and the dev-tuned pass
SNAP = {
    "parcae-140m": "models--SandyResearch--parcae-140m/snapshots/236739ce93ba1afc1b3f172fb4a509f4500d2852",
    "parcae-370m": "models--SandyResearch--parcae-370m/snapshots/439284464ee4999bd1f762da7d044613a4828efe",
    "parcae-770m": "models--SandyResearch--parcae-770m/snapshots/86c1403b16941e746ecbfbcc35885f51d060892e",
    "parcae-1.3b": "models--SandyResearch--parcae-1.3b/snapshots/517f8e6add927e3d73e15beec13a5b500537b9cc",
}
TOK_REMOTE = os.path.join(HUB, "models--SandyResearch--parcae-tokenizer/snapshots/"
                               "6247b5d0592876b73660f34c2dd16c9db9e9045c/tokenizer.json")
TOK_MD5 = "d1979ceeb54b76acadfc6884f90ea029"
BOS_OPTIONS = (("bos0", 0), ("bos1_eos", 1), ("none", None))


def _trunc_std_factor(c=3.0):
    phi = math.exp(-c * c / 2) / math.sqrt(2 * math.pi)
    Z = math.erf(c / math.sqrt(2))
    return math.sqrt(1 - 2 * c * phi / Z)


TRUNC_STD = _trunc_std_factor()          # 0.98658: std of a standard normal truncated at +-3


# ------------------------------------------------------------------ pure helpers (VARBENCH-compatible; parity in selftest)
def r5(x):
    return None if x is None else round(float(x), 5)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def seed_of(pair_id, off=0):
    """Initial-state seed (plan DC5, critic F7): depends only on the pair, shared by scan, rows and models."""
    return int(sha(pair_id)[:8], 16) + int(off)


def split_hex(prefix, key):
    """dev if the first hex digit of sha256(prefix + key) is even (VARBENCH convention)."""
    return "dev" if int(sha(prefix + key)[0], 16) % 2 == 0 else "test"


def rr_order(items, key):
    """Round-robin over groups (sorted names), keeping each group's own order."""
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


def boot2(xs, cl, n=NBOOT, seed=0, q=(0.025, 0.975)):
    """Dyadic (two-way) bootstrap for pairs of units, copied from k17_varbench.boot2 (review round 2 F2):
    cl[i] = (unit_A, unit_B). Units are resampled with replacement; item i gets weight c[A_i] * c[B_i].
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
    Tuple cluster labels (unit_A, unit_B) switch to the dyadic bootstrap (boot2), as k17_varbench.boot."""
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


def diff_tost(xs, cl, grp, g1, g0, margin=PR3_M, n=NBOOT, seed=0):
    """Stratum difference mean(g1) - mean(g0) (ratio of sums within each stratum) with a cluster bootstrap that
    resamples clusters jointly (a cluster may hold items of both strata). Equivalent if the 90% CI lies inside
    [-margin, margin] (plan section 9, PR3). Tuple clusters (unit_A, unit_B) use dyadic weights c[A] * c[B]
    (review round 2 F2)."""
    if any(isinstance(c, (tuple, list)) for x, c in zip(xs, cl) if x is not None):
        return _diff_tost_dyadic(xs, [tuple(c) for c in cl], grp, g1, g0, margin, n, seed)
    by = {}
    for x, c, s in zip(xs, cl, grp):
        if x is None or s not in (g1, g0):
            continue
        d = by.setdefault(c, {g1: [0.0, 0], g0: [0.0, 0]})
        d[s][0] += float(x)
        d[s][1] += 1
    keys = sorted(by)
    if len(keys) < 2:
        return None
    S = torch.tensor([[by[k][g1][0], by[k][g1][1], by[k][g0][0], by[k][g0][1]] for k in keys], dtype=torch.float64)
    if S[:, 1].sum() == 0 or S[:, 3].sum() == 0:
        return None
    v0 = float(S[:, 0].sum() / S[:, 1].sum() - S[:, 2].sum() / S[:, 3].sum())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(keys), (n, len(keys)), generator=g)
    R = S[idx].sum(1)
    ok = (R[:, 1] > 0) & (R[:, 3] > 0)
    R = R[ok]
    bs = R[:, 0] / R[:, 1] - R[:, 2] / R[:, 3]
    q = torch.quantile(bs, torch.tensor([0.05, 0.95, 0.025, 0.975], dtype=torch.float64))
    eq = bool(q[0] > -margin and q[1] < margin)
    return {"diff": round(v0, 4), "ci90": [round(float(q[0]), 4), round(float(q[1]), 4)],
            "ci95": [round(float(q[2]), 4), round(float(q[3]), 4)], "equivalent": eq, "margin": margin,
            "n_g1": int(S[:, 1].sum()), "n_g0": int(S[:, 3].sum()), "n_clusters": len(keys)}


def _diff_tost_dyadic(xs, cl, grp, g1, g0, margin, n, seed):
    pr = [(float(x), c, s) for x, c, s in zip(xs, cl, grp) if x is not None and s in (g1, g0)]
    if not pr or not any(s == g1 for _, _, s in pr) or not any(s == g0 for _, _, s in pr):
        return None
    units = sorted({u for _, c, _ in pr for u in c})
    if len(units) < 2:
        return None
    ix = {u: i for i, u in enumerate(units)}
    x = torch.tensor([v for v, _, _ in pr], dtype=torch.float64)
    m1 = torch.tensor([1.0 if s == g1 else 0.0 for _, _, s in pr], dtype=torch.float64)
    m0 = 1.0 - m1
    ia = torch.tensor([ix[c[0]] for _, c, _ in pr])
    ib = torch.tensor([ix[c[1]] for _, c, _ in pr])
    v0 = float((x * m1).sum() / m1.sum() - (x * m0).sum() / m0.sum())
    g = torch.Generator().manual_seed(seed)
    draw = torch.randint(len(units), (n, len(units)), generator=g)
    cnt = torch.zeros(n, len(units), dtype=torch.float64).scatter_add_(1, draw, torch.ones(n, len(units), dtype=torch.float64))
    w = cnt[:, ia] * cnt[:, ib]
    w1, w0 = (w * m1).sum(1), (w * m0).sum(1)
    ok = (w1 > 0) & (w0 > 0)
    bs = (w[ok] * (x * m1)[None]).sum(1) / w1[ok] - (w[ok] * (x * m0)[None]).sum(1) / w0[ok]
    q = torch.quantile(bs, torch.tensor([0.05, 0.95, 0.025, 0.975], dtype=torch.float64))
    eq = bool(q[0] > -margin and q[1] < margin)
    return {"diff": round(v0, 4), "ci90": [round(float(q[0]), 4), round(float(q[1]), 4)],
            "ci95": [round(float(q[2]), 4), round(float(q[3]), 4)], "equivalent": eq, "margin": margin,
            "n_g1": int(m1.sum()), "n_g0": int(m0.sum()), "n_clusters": len(units), "dyadic": True}


def cl_of(x):
    """Bootstrap cluster of an item: (A subject, B subject) on D1 (dyadic, review round 2 F2), else the cluster."""
    return (x["cluster"], x["cluster_b"]) if x.get("cluster_b") else x["cluster"]


def _units(x):
    return (x["cluster"], x["cluster_b"]) if x.get("cluster_b") else (x["cluster"],)


def n_units(items):
    return len({u for x in items for u in _units(x)})


def unit_overlap(items):
    """dev/test overlap of bootstrap units (D1: A or B subject; else the cluster)."""
    return len({u for x in items if x["split"] == "dev" for u in _units(x)}
               & {u for x in items if x["split"] == "test" for u in _units(x)})


def paired_boot(xs, ys, cl, n=NBOOT, seed=0):
    """Paired difference mean(x - y) over items with both values, cluster bootstrap -> boot list."""
    d, c = [], []
    for x, y, k in zip(xs, ys, cl):
        if x is not None and y is not None:
            d.append(x - y)
            c.append(k)
    return boot(d, c, n=n, seed=seed)


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
    """G {site_key: predicted} -> (keys ranked by G desc, the verify keys: top-3, median-3, bottom-3)."""
    ks = sorted(G, key=lambda k: (-G[k], k))
    n = len(ks)
    m = n // 2
    out = []
    for i in (0, 1, 2, m - 1, m, m + 1, n - 3, n - 2, n - 1):
        if 0 <= i < n and ks[i] not in out:
            out.append(ks[i])
    return ks, out


def hit_metrics(G, dm, D, thr):
    """One item, one dose, over the verified sites (VARBENCH T7): eligibility, strict hit, permutation chance,
    regret, Spearman. The top-predicted site is the argmax of G."""
    ks = [k for k in G if k in dm and dm[k] is not None]
    if len(ks) < 3:
        return None
    top = max(ks, key=lambda k: (G[k], k))
    acts = [dm[k] for k in ks]
    mx = max(acts)
    uniq = sum(1 for a in acts if a == mx) == 1
    elig = (mx - min(acts)) > thr
    hit = int(all(dm[top] > dm[k] for k in ks if k != top))
    return {"eligible": bool(elig), "hit": hit if elig else None,
            "chance": ((1.0 / len(ks)) if uniq else 0.0) if elig else None,
            "regret": (mx - dm[top]) / D if (D is not None and D >= GAP) else None,
            "spearman": spearman([G[k] for k in ks], acts) if elig else None, "n_verified": len(ks), "top": top}


HIT_TOL = 0.05


def hit_metrics_v3(G, dm, D, thr, tol=HIT_TOL):
    """VARBENCH T7 v3 (review MS2), copied verbatim for cross-model comparability; added at the run step as a
    secondary E8 metric (plan DV7). Hit = dm(top) >= max dm - tol * D; chance = band fraction; top3_hit =
    expected hit of a site drawn at random from the selector's own top 3. The frozen plan's strict hit stays."""
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


def proj(h, mA, mB):
    """Pair projection: 0 at clean A, 1 at clean B; None if |mB - mA| < 0.01 |mA|."""
    d = (mB - mA).float()
    if float(d.norm()) < 0.01 * float(mA.float().norm()):
        return None
    return float(((h.float() - (mA.float() + mB.float()) / 2) @ d) / (d @ d) + 0.5)


def fn_rate(pairs):
    """A7 false-negative rate over (G, dm) site pairs of items: among |G| < 0.05 D, fraction with |dm| >= 0.25 D.
    pairs: list of (G, dm, D)."""
    low = [(g, d, DD) for g, d, DD in pairs if g is not None and d is not None and DD is not None and abs(g) < 0.05 * DD]
    if not low:
        return {"n_sites_total": len(pairs), "n_low_G": 0, "fn_rate": None}
    fn = sum(1 for g, d, DD in low if abs(d) >= 0.25 * DD)
    return {"n_sites_total": len(pairs), "n_low_G": len(low), "fn_rate": round(fn / len(low), 4)}


def answer_class(a, e_final, e_own=None, gate=None):
    """Plan section 8 (critic F3, DV6): thresholds scale with the dose a. e_own given -> Llast@j rule; gate given
    (presence gate passed: True/False/None) -> L0/Lmid/post_inj rule."""
    if e_final is None:
        return None
    if e_final >= 0.5 * a:
        return "carried"
    if e_own is not None:
        if e_own >= 0.5 * a:
            return "not carried" if e_final < 0.25 * a else "partial"
        return "no answer state at site"
    if gate is True:
        return "not carried" if e_final < 0.25 * a else "partial"
    if gate is False:
        return "no answer state at site"
    return "unvalidated"


def attribute(src_e, pc_e, channels):
    """Frozen reading rule v3 for a source-cell null (plan section 11 PR10, review round 2 F1). src_e, pc_e: boot
    lists [mean, lo, hi] or None; channels {name: boot list} for VE-coda, VE-loop, e re-injection (same items as
    the control). Null = mean <= 0.15 (T1 fails)."""
    if not src_e or src_e[0] is None:
        return None
    if src_e[0] > 0.15:
        return "not a null"
    if not pc_e or pc_e[0] is None:
        return "no informative control"
    if pc_e[0] >= 0.5:
        return "decay at the source"
    if pc_e[0] > 0.15:
        return "control partial (0.15 < PC < 0.5): not attributed"
    hit = [n for n in ("VE-coda", "VE-loop", "e re-injection")
           if channels.get(n) and channels[n][0] is not None and channels[n][0] >= 0.5]
    return ("input channel: " + ", ".join(hit)) if hit else "unattributed (no single input channel >= 0.5)"


def info_pc_name(loop):
    """Informative positive control (state-only, all positions) for a source cell at loop j (plan section 7)."""
    if loop <= 2:
        return "Llast@1"
    if loop <= 5:
        return "Llast@4"
    if loop <= 7:
        return "Llast@7"
    return "post_inj@8"


# ------------------------------------------------------------------ sites
def name2site(name, C):
    kind, j = name.split("@")
    j = int(j)
    if kind == "post_inj":
        return ("p", j)
    return ("c", j, {"L0": 0, "Lmid": C // 2, "Llast": C - 1}[kind])


def skey(site):
    if site[0] == "p":
        return "p%d" % site[1]
    if site[0] == "c":
        return "c%d.%d" % (site[1], site[2])
    return "k%d" % site[1]


def ksite(k):
    if k[0] == "p":
        return ("p", int(k[1:]))
    if k[0] == "c":
        a, b = k[1:].split(".")
        return ("c", int(a), int(b))
    return ("k", int(k[1:]))


def site_loop(site):
    return site[1] if site[0] in ("p", "c") else K + 1


def exec_order(site, C):
    """Execution order index: post_inj(j) < core(j, 0..C-1) < post_inj(j+1) ... < coda."""
    if site[0] == "p":
        return (site[1] - 1) * (C + 1)
    if site[0] == "c":
        return (site[1] - 1) * (C + 1) + 1 + site[2]
    return K * (C + 1) + site[1]


def e1_names(ds):
    loops = VB_LOOPS if ds == "VB" else E1_LOOPS
    out = []
    for j in loops:
        out += ["post_inj@%d" % j, "L0@%d" % j]
        if ds != "VB" or j == K:
            out.append("Lmid@%d" % j)
        out.append("Llast@%d" % j)
    return out


def e8_sets(C):
    """Primary S (all core outputs except Llast@K, critic F1) and secondary S+ (with Llast@K)."""
    Sp = [("c", j, l) for j in range(1, K + 1) for l in range(C)]
    S = [s for s in Sp if s != ("c", K, C - 1)]
    return S, Sp


def vals(z, tA, tB):
    """[z(tA), z(tB), logsumexp, top-1] for one logit vector."""
    return [r5(z[tA]), r5(z[tB]), r5(torch.logsumexp(z.float(), -1)), int(z.argmax())]


def logp(z, t):
    return float(z[t].float() - torch.logsumexp(z.float(), -1))


def apply_entries(out, ents):
    """ents: list of (kind, pos LongTensor, value [P, d], a). 'set' = lerp toward value (a = 1 exact copy),
    'add' = add value. Applied to batch row 0 of a cloned output."""
    out = out.clone()
    for kind, pos, val, a in ents:
        v = val.to(out.dtype)
        if kind == "set":
            out[0, pos] = v if a == 1.0 else torch.lerp(out[0, pos], v, float(a))
        else:
            out[0, pos] = out[0, pos] + v
    return out


def rnd_delta(pair_id, key, pos_name, positions, dnorm, H):
    """Unit Gaussian per position, seed = first 8 hex of sha256(pair_id|site|pos|p), scaled to dnorm -> [P, H]."""
    rows = []
    for p, n in zip(positions, dnorm):
        g = torch.Generator().manual_seed(int(sha("%s|%s|%s|%d" % (pair_id, key, pos_name, p))[:8], 16))
        u = torch.randn(H, generator=g)
        rows.append(u / u.norm() * float(n))
    return torch.stack(rows)


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ------------------------------------------------------------------ selection rules (plan section 9)
class Selector(object):
    """Quota, cap and problem-disjointness rules per dataset. want(c) -> scan it; accept(c) after it passes."""

    def __init__(self, ds, target=None):
        self.ds = ds
        self.target = target if target is not None else TARGET[ds]
        self.n = 0
        self.n_multi = 0
        self.subj = {}
        self.grp = {}
        self.clus = {}
        self.problems = set()

    def done(self):
        if self.n >= self.target:
            return True
        if self.ds == "D2":
            return all(self.grp.get(g, 0) >= D2_PER_OP for g in ("op1", "op2"))
        if self.ds == "D4":
            return all(self.grp.get((s, k), 0) >= D4_PER_CELL for s in ("dev", "test") for k in ("T1", "T2"))
        if self.ds == "D5":
            return all(self.grp.get(s, 0) >= D5_PER_SPLIT for s in ("dev", "test"))
        return False

    def want(self, c):
        if self.done():
            return False
        ds = self.ds
        if ds == "D1":
            if c["strata"]["pos_len"] == ">=2" and self.n_multi >= D1_MULTI_MAX:
                return False
            for s in c["sel"]["subjects"]:
                if self.subj.get(s, 0) >= D1_SUBJ_CAP:
                    return False
            return True
        if ds == "D2":
            return self.grp.get(c["rel"], 0) < D2_PER_OP and self.clus.get(c["cluster"], 0) < D2_CLUSTER_CAP
        if ds == "D4":
            return self.grp.get((c["split"], c["rel"]), 0) < D4_PER_CELL and c["cluster"] not in self.clus
        if ds == "D5":
            return self.grp.get(c["split"], 0) < D5_PER_SPLIT and not (set(c["sel"]["problems"]) & self.problems)
        return True

    def accept(self, c):
        self.n += 1
        ds = self.ds
        if ds == "D1":
            if c["strata"]["pos_len"] == ">=2":
                self.n_multi += 1
            for s in c["sel"]["subjects"]:
                self.subj[s] = self.subj.get(s, 0) + 1
        elif ds == "D2":
            self.grp[c["rel"]] = self.grp.get(c["rel"], 0) + 1
            self.clus[c["cluster"]] = self.clus.get(c["cluster"], 0) + 1
        elif ds == "D4":
            k = (c["split"], c["rel"])
            self.grp[k] = self.grp.get(k, 0) + 1
            self.clus[c["cluster"]] = 1
        elif ds == "D5":
            self.grp[c["split"]] = self.grp.get(c["split"], 0) + 1
            self.problems |= set(c["sel"]["problems"])


# ------------------------------------------------------------------ build (laptop CPU, Parcae tokenizer)
class ParcaeTok(object):
    """k16_data.Tok-compatible wrapper around the Parcae tokenizer.json (no BOS at build; the runtime adds it)."""
    name = "parcae"
    bos = None

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(path)
        self.path = path
        self.md5 = md5_file(path)

    def enc(self, text, ans=False):
        return list(self.t.encode(text, add_special_tokens=False).ids)

    def enc_spans(self, text, ranges):
        e = self.t.encode(text, add_special_tokens=False)
        spans = [[i for i, (s, t) in enumerate(e.offsets) if s < c1 and t > c0] for c0, c1 in ranges]
        return list(e.ids), spans

    def dec(self, ids):
        return [self.t.decode([i]) for i in ids]


def subj_span(x, y):
    n = min(len(x), len(y))
    p = 0
    while p < n and x[p] == y[p]:
        p += 1
    s = 0
    while s < n - p and x[-1 - s] == y[-1 - s]:
        s += 1
    return list(range(p, len(x) - s)), list(range(p, len(y) - s))


def para_forms(main_src, forms):
    """forms: list of (fmt, idsA', idsB') -> aligned forms (P4: same span length, equal length, nonempty)."""
    out = []
    for fmt, a, b in forms:
        if len(a) != len(b):
            continue
        sa, sb = subj_span(a, b)
        if not sa or sa != sb or len(sa) != len(main_src):
            continue
        out.append({"fmt": fmt, "A": list(a), "B": list(b), "src": sa, "r": len(a) - 1})
    return out


def single_token_answer(tk, ans, t_first):
    """The full answer ' N' is the shared space token plus exactly one number token equal to t_first."""
    ids = tk.enc(" " + str(ans), ans=True)
    return len(ids) == 2 and tk.dec([ids[0]]) == [" "] and ids[1] == t_first


def d4_content(r, side="A"):
    """VARBENCH content key (P10(5)): kind | vals of one side | index of src_var | indices of query_vars.
    Same as k17_varbench.d4_content (side 'B' = the B-side problem, VARBENCH critic 2 F8)."""
    idx_src = r["names"].index(r["src_var"])
    idx_q = [r["names"].index(q) for q in r["query_vars"]]
    vs = r["vals_A"] if side == "A" else r["vals_B"]
    return sha("%s|%s|%d|%s" % (r["kind"], ",".join(str(v) for v in vs), idx_src,
                                ",".join(str(i) for i in idx_q)))[:16]


def d4_components(rows):
    """VARBENCH v4 D4 clusters (k17_varbench.d4_components, review round 2 F2): connected components of the graph
    whose nodes are content keys and whose edges are rows (A-side key - B-side key). -> {key: component id}."""
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


def build_d1(tk):
    import k15_data as KD
    cands = KD.d1_candidates(tk.enc)
    multi, one = [], []
    n_cross = {"1": 0, ">=2": 0}
    bos_cal = []
    for p in cands:
        pid = "D1|%s|%s|%s" % (p["rel"], p["a"], p["b"])
        cl = "%s:%s" % (p["rel"], p["a"])
        clb = "%s:%s" % (p["rel"], p["b"])
        if p["rel"] == "capital" and p["half"] == "dev":     # DC3 pool (115 k15-dev capital pairs), before the drop
            bos_cal.append({"ds": "D1", "pair_id": pid, "A": list(p["A"]), "B": list(p["B"]), "r": len(p["A"]) - 1,
                            "src": list(p["pos"]), "tA": int(p["tA"]), "tB": int(p["tB"]), "rel": p["rel"],
                            "cluster": cl, "split": "n/a", "k15_half": p["half"], "strata": {}, "label": pid,
                            "para": []})
        if split_hex("k17pc-d1:", cl) != split_hex("k17pc-d1:", clb):     # review round 2 F2 (VARBENCH d1_assign)
            n_cross["1" if len(p["pos"]) == 1 else ">=2"] += 1
            continue
        forms = para_forms(p["pos"], [("para%d" % i, a, b) for i, (a, b) in enumerate(p["para"])])
        c = {"ds": "D1", "pair_id": pid, "A": list(p["A"]), "B": list(p["B"]), "r": len(p["A"]) - 1,
             "src": list(p["pos"]), "tA": int(p["tA"]), "tB": int(p["tB"]), "rel": p["rel"], "cluster": cl,
             "cluster_b": clb, "split": split_hex("k17pc-d1:", cl), "k15_half": p["half"],
             "strata": {"pos_len": "1" if len(p["pos"]) == 1 else ">=2", "rel": p["rel"],
                        "subject_pos0": bool(0 in p["pos"])},
             "para": forms, "sel": {"subjects": ["%s:%s" % (p["rel"], p["a"]), "%s:%s" % (p["rel"], p["b"])]},
             "label": "%s -> %s" % (p["a"], p["b"])}
        (multi if len(p["pos"]) >= 2 else one).append(c)
    out = multi + one
    st = {"candidates_all": len(cands), "dropped_cross_split": n_cross,
          "split_rule": "A and B subject hash to the same split (k17pc-d1:rel:subject); CI dyadic over subjects",
          "candidates": len(out), "multi": len(multi), "one": len(one),
          "by_split_pos_len": {"%s|%s" % (s, pl): sum(1 for c in out if c["split"] == s and c["strata"]["pos_len"] == pl)
                               for s in ("dev", "test") for pl in ("1", ">=2")},
          "pos_len_x_rel": {"%s|%s" % (c["strata"]["pos_len"], c["rel"]): 0 for c in out}}
    for c in out:
        st["pos_len_x_rel"]["%s|%s" % (c["strata"]["pos_len"], c["rel"])] += 1
    st["subject_pos0"] = sum(1 for c in out if c["strata"]["subject_pos0"])
    st["bos_cal"] = bos_cal
    st["with_para"] = sum(1 for c in out if c["para"])
    return out, st


def build_d2(tk):
    import k15_data as KD
    cands = KD.d2_candidates(tk.enc)
    out, bad = [], 0
    for p in cands:
        a1, b1 = [int(v) for v in p["a"].split(" + ")]
        a2, b2 = [int(v) for v in p["b"].split(" + ")]
        if not (single_token_answer(tk, a1 + b1, p["tA"]) and single_token_answer(tk, a2 + b2, p["tB"])):
            bad += 1
            continue
        cl = "x=%d" % (a1 if p["rel"] == "op1" else b1)
        forms = para_forms(p["pos"], [("para%d" % i, a, b) for i, (a, b) in enumerate(p["para"])])
        out.append({"ds": "D2", "pair_id": "D2|%s|%s|%s" % (p["rel"], p["a"], p["b"]), "A": list(p["A"]),
                    "B": list(p["B"]), "r": len(p["A"]) - 1, "src": list(p["pos"]), "tA": int(p["tA"]),
                    "tB": int(p["tB"]), "rel": p["rel"], "cluster": cl, "split": split_hex("k17vb-d2:", cl),
                    "strata": {"pos_len": "1" if len(p["pos"]) == 1 else ">=2", "rel": p["rel"]}, "para": forms,
                    "sel": {}, "label": "%s -> %s" % (p["a"], p["b"])})
    return out, {"candidates": len(cands), "kept": len(out), "answer_not_single_token": bad,
                 "with_para": sum(1 for c in out if c["para"])}


def build_d4(tk):
    import k16_data as KD6
    items = KD6.d4_items("T1", KD6.D4_N_TEMPLATES["T1"]) + KD6.d4_items("T2", KD6.D4_N_TEMPLATES["T2"])
    out, drops = [], {}
    rows = []
    for it in items:
        row, why = KD6.d4_row(tk, it)
        if row is None:
            drops[why] = drops.get(why, 0) + 1
            continue
        rows.append(row)
    comp = d4_components(rows)
    for row in rows:
        A, B, src = row["ids_A"], row["ids_B"], row["source_span"]
        if len(A) != len(B) or [i for i in range(len(A)) if A[i] != B[i]] != list(src):
            drops["unequal_or_bad_span"] = drops.get("unequal_or_bad_span", 0) + 1
            continue
        ta, tb = row["answer_first_token_A"], row["answer_first_token_B"]
        if not (single_token_answer(tk, row["answer_A"], ta) and single_token_answer(tk, row["answer_B"], tb)):
            drops["answer_not_single_token"] = drops.get("answer_not_single_token", 0) + 1
            continue
        cc = comp[d4_content(row, "A")]
        forms = para_forms(src, [(q["format"], q["ids_A"], q["ids_B"]) for q in row["para"]])
        out.append({"ds": "D4", "pair_id": "D4|" + row["id"], "A": list(A), "B": list(B), "r": len(A) - 1,
                    "src": list(src), "tA": int(ta), "tB": int(tb), "rel": row["kind"], "cluster": cc,
                    "split": split_hex("k17vb-d4c:", cc), "template": row["cluster_id"],
                    "keys": [d4_content(row, "A"), d4_content(row, "B")],
                    "strata": {"pos_len": "1" if len(src) == 1 else ">=2", "rel": row["kind"]}, "para": forms,
                    "sel": {}, "label": row["id"]})
    out = rr_order(out, lambda c: (c["split"], c["rel"]))
    csz = {}
    for k, c in comp.items():
        csz[c] = csz.get(c, 0) + 1
    dev_k = {k for c in out if c["split"] == "dev" for k in c["keys"]}
    tst_k = {k for c in out if c["split"] == "test" for k in c["keys"]}
    return out, {"items": len(items), "kept": len(out), "dropped": drops, "with_para": sum(1 for c in out if c["para"]),
                 "cluster": "connected component of shared A/B content keys (VARBENCH v4 d4_components)",
                 "clusters": len(set(c["cluster"] for c in out)), "n_components": len(csz),
                 "largest_component_keys": max(csz.values()) if csz else 0,
                 "by_split_kind": {"%s|%s" % (s, k): len({c["cluster"] for c in out if c["split"] == s and c["rel"] == k})
                                   for s in ("dev", "test") for k in ("T1", "T2")},
                 "dev_test_shared_content_keys": len(dev_k & tst_k)}


def build_d5(tk):
    import k16_data as KD6
    items = KD6.d5_items()
    demos = set((a, b, m) for a, b, m in KD6.D5_DEMOS)
    out, drops = [], {}
    edges = []
    for it in items:
        row, why = KD6.d5_row(tk, it)
        if row is None:
            drops[why] = drops.get(why, 0) + 1
            continue
        A, B, src = row["ids_A"], row["ids_B"], row["source_span"]
        if len(A) != len(B) or [i for i in range(len(A)) if A[i] != B[i]] != list(src):
            drops["unequal_or_bad_span"] = drops.get("unequal_or_bad_span", 0) + 1
            continue
        ta, tb = row["answer_first_token_A"], row["answer_first_token_B"]
        if not (single_token_answer(tk, row["answer_A"], ta) and single_token_answer(tk, row["answer_B"], tb)):
            drops["answer_not_single_token"] = drops.get("answer_not_single_token", 0) + 1
            continue
        if {ta, tb} & {A[src[0]], B[src[0]]}:
            drops["answer_equals_source_token"] = drops.get("answer_equals_source_token", 0) + 1
            continue
        m = row["mult"]
        pa = (row["operands_A"][0], row["operands_A"][1], m)
        pb = (row["operands_B"][0], row["operands_B"][1], m)
        if pa in demos or pb in demos:
            drops["demo_problem"] = drops.get("demo_problem", 0) + 1
            continue
        kA, kB = "%d,%d,%d" % pa, "%d,%d,%d" % pb
        edges.append((kA, kB))
        forms = para_forms(src, [(q["format"], q["ids_A"], q["ids_B"]) for q in row["para"]])
        out.append({"ds": "D5", "pair_id": "D5|" + row["id"], "A": list(A), "B": list(B), "r": len(A) - 1,
                    "src": list(src), "tA": int(ta), "tB": int(tb), "rel": "T2", "cluster": row["id"],
                    "split": split_hex("k17pc-d5:", row["id"]), "template": row["cluster_id"],
                    "strata": {"pos_len": "1" if len(src) == 1 else ">=2", "rel": "T2",
                               "src_operand": row["src_operand"]},
                    "para": forms, "sel": {"problems": [kA, kB]}, "label": row["id"]})
    par = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    comps = len(set(find(x) for x in list(par)))
    out = rr_order(out, lambda c: c["split"])
    return out, {"items": len(items), "kept": len(out), "dropped": drops, "with_para": sum(1 for c in out if c["para"]),
                 "problems": len(par), "shared_problem_components": comps}


def build_vb(tk):
    """VARBENCH I2 + I3 rules on the Parcae tokenizer (plan T3, critic F5)."""
    import pandas as pd
    import k17_varbench as VB
    rows = [json.loads(x) for x in open(VB.PAIRS, encoding="utf-8")]
    core = [r for r in rows if r["split"] == "eval" and r["text_exact"]
            and all(r["tok"][t]["outside_equal"] for t in ("ouro", "huginn", "mcleish", "ifm"))]
    test = pd.read_parquet(os.path.join(VB.GSM_DIR, "test-00000-of-00001.parquet"))
    train = pd.read_parquet(os.path.join(VB.GSM_DIR, "train-00000-of-00001.parquet"))
    block = open(VB.PROMPT_FILE, encoding="utf-8").read()

    def fmt(i):
        return "Question: %s\nAnswer: %s\n\n" % (train.iloc[i]["question"], train.iloc[i]["answer"])
    assert "".join(fmt(i) for i in range(4)) == block, "SETTLE block != gsm8k train rows 0-3"
    block_alt = "".join(fmt(i) for i in range(4, 8))

    def block_ctx(b):
        e = tk.t.encode(b + "Question: x", add_special_tokens=False)
        k = next(i for i, (s, _) in enumerate(e.offsets) if s >= len(b))
        return list(e.ids[:k])
    blk, blk_alt = block_ctx(block), block_ctx(block_alt)
    cnt = {"core_rows": len(core), "core_problems": len({r["idx"] for r in core}), "no_restatement": 0,
           "in_annotation": 0, "contradictory_prefix": 0, "ambiguous_source": 0, "assert_fail": {}}
    valid = []
    for r in core:
        sol = test.iloc[r["idx"]]["answer"]
        x, xb = r["value_A"], r["value_B"]
        c = VB.first_iso(sol, x)
        if c is None:
            cnt["no_restatement"] += 1
            continue
        if VB.in_annot(sol, c):
            cnt["in_annotation"] += 1
            continue
        C = sol[:c]
        if VB.contradictory_prefix(C, x):
            cnt["contradictory_prefix"] += 1
            continue
        if VB.ambiguous_source(r["question_A"], r["char_span_A"], x):
            cnt["ambiguous_source"] += 1
            continue
        pa = block + "Question: " + r["question_A"] + "\nAnswer:"
        pb = block + "Question: " + r["question_B"] + "\nAnswer:"
        off = len(block + "Question: ")
        ea = tk.t.encode(pa, add_special_tokens=False)
        eb = tk.t.encode(pb, add_special_tokens=False)
        ipa, ipb = list(ea.ids), list(eb.ids)
        ca = (off + r["char_span_A"][0], off + r["char_span_A"][1])
        cb = (off + r["char_span_B"][0], off + r["char_span_B"][1])
        sa = [i for i, (s, t) in enumerate(ea.offsets) if s < ca[1] and t > ca[0]]
        sb = [i for i, (s, t) in enumerate(eb.offsets) if s < cb[1] and t > cb[0]]
        why = None
        if len(ipa) != len(ipb):
            why = "prompt_len_mismatch"
        elif not sa or sa != sb or sa != list(range(sa[0], sa[-1] + 1)):
            why = "span_mismatch"
        else:
            diffs = [i for i in range(len(ipa)) if ipa[i] != ipb[i]]
            if not diffs or any(i not in sa for i in diffs):
                why = "outside_unequal"
        if why is None:
            fa = tk.enc(pa + " " + C + str(x))
            fb = tk.enc(pb + " " + C + str(xb))
            if fa[:len(ipa)] != ipa or fb[:len(ipb)] != ipb:
                why = "prefix_not_kept"
            elif len(fa) != len(fb):
                why = "full_len_mismatch"
            elif fa[:len(blk)] != blk:
                why = "block_ctx"
            else:
                diffs = [i for i in range(len(fa)) if fa[i] != fb[i]]
                after = [i for i in diffs if i >= len(ipa)]
                bad = [i for i in diffs if i < len(ipa) and i not in sa]
                if not after or bad:
                    why = "full_outside_unequal"
                else:
                    ts = after[0]
                    if ts <= len(ipa):
                        why = "tstar_at_prompt_end"
                    else:
                        r2a = tk.enc(block_alt + pa[len(block):] + " " + C + str(x))
                        r2b = tk.enc(block_alt + pb[len(block):] + " " + C + str(xb))
                        if r2a != blk_alt + fa[len(blk):] or r2b != blk_alt + fb[len(blk):]:
                            why = "r2_mismatch"
        if why is not None:
            cnt["assert_fail"][why] = cnt["assert_fail"].get(why, 0) + 1
            continue
        restA, restB = fa[ts:], fb[ts:]
        SA, SA2 = fa[:ts] + restA[:-1], fa[:ts] + restB[:-1]
        SB, SB2 = fb[:ts] + restB[:-1], fb[:ts] + restA[:-1]
        off2 = len(blk_alt) - len(blk)
        valid.append({"ds": "VB", "pair_id": r["pair_id"], "idx": int(r["idx"]), "A": SA, "B": SB, "A2": SA2,
                      "B2": SB2, "restA": restA, "restB": restB, "cont": len(restA) >= 2, "r": ts - 1, "src": sa,
                      "tA": int(restA[0]), "tB": int(restB[0]), "rel": "gsm8k", "cluster": str(r["idx"]),
                      "split": split_hex("k17vb-split:", "%d" % r["idx"]), "plen": len(ipa), "n_chain": ts - len(ipa),
                      "value_A": x, "value_B": xb,
                      "strata": {"pos_len": "1" if len(sa) == 1 else ">=2",
                                 "ST2": "shared" if str(x)[0] == str(xb)[0] else "differs",
                                 "ST3": str(r["n_digits"]) if r["n_digits"] < 3 else ">=3",
                                 "cont": "yes" if len(restA) >= 2 else "no"},
                      "para": [{"fmt": "R2", "A": blk_alt + SA[len(blk):], "B": blk_alt + SB[len(blk):],
                                "src": [i + off2 for i in sa], "r": ts - 1 + off2}],
                      "sel": {}, "label": r["pair_id"]})
    by = {}
    for v in valid:
        by.setdefault(v["idx"], []).append(v)
    sel = {i: min(vs, key=lambda v: sha("k17vb-pair:" + v["pair_id"])) for i, vs in by.items()}
    order = sorted(sel, key=lambda i: sha("k17vb-order:%d" % i))
    out = []
    for k, i in enumerate(order):
        c = sel[i]
        c["order"] = k
        out.append(c)
    cnt.update({"valid_rows": len(valid), "valid_problems": len(by), "selected": len(out),
                "block_ctx_len": len(blk), "block_alt_ctx_len": len(blk_alt),
                "k17_varbench_md5": md5_file(os.path.join(HERE, "k17_varbench.py")),
                "pairs_sha256": sha256_file(VB.PAIRS)})
    vbf = os.path.join(HERE, "..", "out", "k17_varbench", "data", "items_ouro.json")
    if os.path.exists(vbf):
        d = json.load(open(vbf))
        vsel = [x["pair_id"] for x in sorted((x for x in d["items"] if x["selected"]), key=lambda x: x["order"])]
        mine = [c["pair_id"] for c in out]
        cnt["varbench_overlap"] = {"varbench_selected": len(vsel), "parcae_selected": len(mine),
                                   "same_pair_ids": len(set(vsel) & set(mine)),
                                   "same_order_first200": mine[:200] == vsel[:200],
                                   "varbench_items_sha256": sha256_file(vbf)}
    else:
        cnt["varbench_overlap"] = None
    sp = [len(c["src"]) for c in out[:200]]
    cnt["first200"] = {"span_1": sp.count(1), "span_ge2": len(sp) - sp.count(1),
                       "cont": sum(1 for c in out[:200] if c["cont"]),
                       "dev": sum(1 for c in out[:200] if c["split"] == "dev"),
                       "plen_median": sorted(c["plen"] for c in out[:200])[len(out[:200]) // 2]}
    return out, cnt


def build_items(out_dir, tok_path):
    tk = ParcaeTok(tok_path)
    assert tk.md5 == TOK_MD5, "Parcae tokenizer md5 %s != %s" % (tk.md5, TOK_MD5)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"tokenizer_md5": tk.md5, "tokenizer_checks": {}, "datasets": {}}
    for s in ("The capital of France is", " Paris", " 12", " 1234", "Paris is a city"):
        meta["tokenizer_checks"][s] = tk.enc(s)
    for ds, fn in (("D1", build_d1), ("D2", build_d2), ("D4", build_d4), ("D5", build_d5), ("VB", build_vb)):
        t0 = time.time()
        cands, st = fn(tk)
        bos_cal = st.pop("bos_cal", None)
        if bos_cal is not None:
            st["bos_cal_n"] = len(bos_cal)
        st["seconds"] = round(time.time() - t0, 1)
        st["dev_test_cluster_overlap"] = len({c["cluster"] for c in cands if c["split"] == "dev"}
                                             & {c["cluster"] for c in cands if c["split"] == "test"})
        st["dev_test_unit_overlap_A_or_B"] = unit_overlap(cands)
        p = os.path.join(out_dir, "cands_%s.json" % ds)
        atomic_json(p, dict({"ds": ds, "stats": st, "cands": cands}, **({"bos_cal": bos_cal} if bos_cal is not None else {})))
        st["file_sha256"] = sha256_file(p)
        meta["datasets"][ds] = st
        print(ds, json.dumps({k: v for k, v in st.items() if k not in ("dropped_ids",)}), flush=True)
    for m in ("k15_data.py", "k16_data.py", "k13_edit.py", "k15_smoke.py", "k17_varbench.py"):
        meta.setdefault("source_md5", {})[m] = md5_file(os.path.join(HERE, m))
    atomic_json(os.path.join(out_dir, "build_meta.json"), meta)
    return meta


# ------------------------------------------------------------------ runner (Parcae venv)
def parcae_sha16():
    import inspect
    import parcae_lm.models.parcae.parcae as PP
    return hashlib.sha256(open(inspect.getsourcefile(PP), "rb").read()).hexdigest()[:16]


def load_model(size, dev):
    """Strict load of the pinned checkpoint, fp32, eval, all parameters frozen (plan DC4, F12)."""
    from parcae_lm.models.parcae.config import ParcaeConfig
    d = os.path.join(HUB, SNAP[size])
    cfgd = json.load(open(os.path.join(d, "config.json")))
    cls = cfgd.pop("_class_name")
    cfgd.pop("rope_settings", None)               # as parcae_lm.from_pretrained (__init__.py:40-41)
    assert cls == "ParcaeConfig"
    cfg = ParcaeConfig(**cfgd)
    t0 = time.time()
    with torch.device(dev):
        model = cfg.construct_model()
    sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    msd = model.state_dict()
    audit = {"missing": sorted(set(msd) - set(sd)), "unexpected": sorted(set(sd) - set(msd)),
             "shape_mismatch": sorted(k for k in set(sd) & set(msd) if tuple(sd[k].shape) != tuple(msd[k].shape))}
    model.load_state_dict(sd, strict=True)
    del sd
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    audit["load_seconds"] = round(time.time() - t0, 1)
    audit["snapshot"] = SNAP[size].split("/")[-1]
    return model, cfg, audit


class Ctx(object):
    def __init__(self, r, edits, e_edits, ve_edits, rec, rec_e, rec_eargs, leaves):
        self.cnt = {}
        self.r = r
        self.edits = edits or {}
        self.e_edits = e_edits or {}
        self.ve_edits = ve_edits or {}
        self.rec = rec or {}
        self.rec_e = rec_e
        self.rec_eargs = rec_eargs
        self.leaves = leaves or {}
        self.H = {}
        self.E = None
        self.eargs = {}


class PR(object):
    """Parcae runner: hooks H1-H8 (plan section 6), fixed initial state per item, per-loop readout."""

    def __init__(self, model, cfg, dev, bos, name):
        self.m, self.cfg, self.dev, self.bos, self.name = model, cfg, dev, bos, name
        tr = model.transformer
        self.P, self.C, self.Cd = cfg.n_layers_in_prelude, cfg.n_layers_in_recurrent_block, cfg.n_layers_in_coda
        self.d = cfg.recurrent_embedding_dimension
        self.de = cfg.n_embd
        self.lmid, self.llast = self.C // 2, self.C - 1
        self.ve_core = {l: str(self.P + l) for l in range(self.C) if str(self.P + l) in model.value_embeds}
        self.ve_coda = {c: str(self.P + self.C + c) for c in range(self.Cd) if str(self.P + self.C + c) in model.value_embeds}
        self.ctx = None
        self.x0 = None
        self.std = cfg.init.get_std("embedding")
        self.hs = []
        for l, b in enumerate(tr.core_block):
            self.hs.append(b.register_forward_hook(self._core_hook(l)))
        self.hs.append(tr.adapter.register_forward_pre_hook(self._adapter_pre))
        self.hs.append(tr.adapter.register_forward_hook(self._adapter_post))
        for c, b in enumerate(tr.coda):
            self.hs.append(b.register_forward_hook(self._coda_hook(c)))
        self.hs.append(tr.ln_prelude.register_forward_hook(self._lnp_hook))
        for k in list(self.ve_core.values()) + list(self.ve_coda.values()):
            self.hs.append(model.value_embeds[k].register_forward_hook(self._ve_hook(k)))
        model.initialize_state = self._init_state          # H6: instance attribute override (DC5)

    # --- initial state
    def set_init(self, T, seed):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.empty(1, T, self.d)
        torch.nn.init.trunc_normal_(x, mean=0.0, std=self.std, a=-3 * self.std, b=3 * self.std, generator=g)
        if self.m.emb_scale != 1:
            x = x * self.m.emb_scale
        if T * self.d >= 4096:
            rel = float(x.std()) / (TRUNC_STD * self.std * float(self.m.emb_scale))
            assert abs(rel - 1) < 0.05, "initial-state draw std off by %.3f" % rel
        self.x0 = x.to(self.dev)
        self.x0_seed = int(seed)
        return x

    def _init_state(self, input_embeds):
        T = input_embeds.shape[1]
        assert self.x0 is not None and self.x0.shape[1] >= T, "initial state not set or too short"
        return self.x0[:, :T].to(dtype=input_embeds.dtype, device=input_embeds.device).clone()

    # --- hooks
    def _site(self, c, site, out):
        ents = c.edits.get(site)
        if ents:
            out = apply_entries(out, ents)
        lf = c.leaves.get(site)
        if lf is not None:
            r = c.r
            out = torch.cat([out[:, :r], out[:, r:r + 1] + lf, out[:, r + 1:]], 1)
        rp = c.rec.get(site)
        if rp is not None:
            c.H[site] = (out[0] if rp == "all" else out[0, rp]).detach().clone()
        return out

    def _core_hook(self, l):
        def h(mod, args, out):
            c = self.ctx
            if c is None:
                return None
            n = c.cnt.get(("c", l), 0) + 1
            c.cnt[("c", l)] = n
            return self._site(c, ("c", n, l), out)
        return h

    def _coda_hook(self, cc):
        def h(mod, args, out):
            c = self.ctx
            if c is None:
                return None
            n = c.cnt.get(("k", cc), 0) + 1
            c.cnt[("k", cc)] = n
            return self._site(c, ("k", cc), out)
        return h

    def _adapter_pre(self, mod, args):
        c = self.ctx
        if c is None:
            return None
        n = c.cnt.get("pre", 0) + 1
        c.cnt["pre"] = n
        x, e = args
        ed = c.e_edits.get(n)
        if ed is not None:
            pos, rows = ed
            e = e.clone()                       # critic F6: never write into the shared input_embeds tensor
            e[0, pos] = rows.to(e.dtype)
        if c.rec_eargs:
            c.eargs[n] = e[0].detach().clone()
        return (x, e) if ed is not None else None

    def _adapter_post(self, mod, args, out):
        c = self.ctx
        if c is None:
            return None
        n = c.cnt.get("post", 0) + 1
        c.cnt["post"] = n
        return self._site(c, ("p", n), out)

    def _lnp_hook(self, mod, args, out):
        c = self.ctx
        if c is None:
            return None
        c.cnt["lnp"] = c.cnt.get("lnp", 0) + 1
        if c.rec_e:
            c.E = out[0].detach().clone()
        return None

    def _ve_hook(self, key):
        def h(mod, args, out):
            c = self.ctx
            if c is None:
                return None
            n = c.cnt.get(("ve", key), 0) + 1
            c.cnt[("ve", key)] = n
            ed = c.ve_edits.get((key, n))
            if ed is None:
                return None
            pos, rows = ed
            out = out.clone()
            out[0, pos] = rows.to(out.dtype)
            return out
        return h

    def _check_counts(self, c, nK):
        bad = []
        for l in range(self.C):
            if c.cnt.get(("c", l)) != nK:
                bad.append(("core", l, c.cnt.get(("c", l))))
        for k in ("pre", "post"):
            if c.cnt.get(k) != nK:
                bad.append((k, c.cnt.get(k)))
        for cc in range(self.Cd):
            if c.cnt.get(("k", cc)) != 1:
                bad.append(("coda", cc, c.cnt.get(("k", cc))))
        for k in self.ve_core.values():
            if c.cnt.get(("ve", k)) != nK:
                bad.append(("ve", k, c.cnt.get(("ve", k))))
        for k in self.ve_coda.values():
            if c.cnt.get(("ve", k)) != 1:
                bad.append(("ve", k, c.cnt.get(("ve", k))))
        if c.cnt.get("lnp") != 1:
            bad.append(("ln_prelude", c.cnt.get("lnp")))
        assert not bad, "hook call counts wrong: %s" % bad

    # --- forward
    def run(self, ids, r, edits=None, e_edits=None, ve_edits=None, rec=None, rec_e=False, rec_eargs=False,
            leaves=None, grad=False, cont=False, steps=None, full=False):
        x = torch.tensor([ids], device=self.dev)
        nK = steps or K
        pair = (0, K) if grad else (nK, 0)
        c = Ctx(r, edits, e_edits, ve_edits, rec, rec_e, rec_eargs, leaves)
        self.ctx = c
        try:
            with (torch.enable_grad() if grad else torch.no_grad()):
                out = self.m(x, return_logits=True, num_steps_pair=pair)
        finally:
            self.ctx = None
        self._check_counts(c, nK)
        z = out["logits"][0]
        res = {"z_r": z[r] if grad else z[r].detach().clone(), "H": c.H, "E": c.E, "eargs": c.eargs}
        if cont:
            res["z_cont"] = z[r + 1:].detach().clone()
        if full:
            res["z"] = z.detach().clone()
        return res

    @torch.no_grad()
    def readout(self, ids, h):
        """H8: C -> coda (value embeds) -> ln_f -> lm_head * logit_scale on a loop-end state h [T, d] (plan DC6).
        Hooks are off (no context). Returns logits [T, V]."""
        assert self.ctx is None
        m = self.m
        input_ids = torch.tensor([ids], device=self.dev)
        freqs = m.freqs_cis[:, :len(ids)]
        x = m.transformer.C(h[None])
        off = self.P + self.C
        for i, blk in enumerate(m.transformer.coda):
            k = str(off + i)
            ve = m.value_embeds[k](input_ids) if k in m.value_embeds else None
            x = blk(x, freqs, None, ve=ve)
        x = m.transformer.ln_f(x)
        z = m.lm_head(x).float() * m.config.init.logit_scale
        if m.config.logit_softcap is not None:
            sc = m.config.logit_softcap
            z = sc * torch.tanh(z / sc)
        return z[0]

    def ve_rows(self, key, ids, pos):
        return self.m.value_embeds[key].weight[torch.tensor([ids[p] for p in pos], device=self.dev)]

    def all_sites(self):
        return [("p", j) for j in range(1, K + 1)] + [("c", j, l) for j in range(1, K + 1) for l in range(self.C)] + \
               [("k", self.Cd - 1)]


# ------------------------------------------------------------------ items at runtime
class HarnessFail(Exception):
    pass


def sync(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def prep_item(c, bos):
    """Stored candidate (no BOS) -> runtime item with BOS prepended and positions shifted (DC3). None if the
    edited span includes position 0 without BOS (P5); paraphrase forms with a position-0 span are dropped then."""
    sh = 0 if bos is None else 1
    pre = [] if bos is None else [int(bos)]
    if bos is None and 0 in c["src"]:
        return None
    it = {k: c[k] for k in ("ds", "pair_id", "tA", "tB", "rel", "cluster", "cluster_b", "split", "strata", "label")
          if k in c}
    it["A"], it["B"] = pre + c["A"], pre + c["B"]
    it["r"] = c["r"] + sh
    it["src"] = [p + sh for p in c["src"]]
    seqs = [it["A"], it["B"]]
    if c["ds"] == "VB":
        it["A2"], it["B2"] = pre + c["A2"], pre + c["B2"]
        it["restA"], it["restB"], it["cont"] = c["restA"], c["restB"], c["cont"]
        seqs += [it["A2"], it["B2"]]
    paras = []
    for f in c["para"]:
        if bos is None and 0 in f["src"]:
            continue
        paras.append({"fmt": f["fmt"], "A": pre + f["A"], "B": pre + f["B"], "src": [p + sh for p in f["src"]],
                      "r": f["r"] + sh})
        seqs += [paras[-1]["A"], paras[-1]["B"]]
    it["para"] = paras
    it["Tmax"] = max(len(s) for s in seqs)
    return it


class Item(object):
    """Clean A/B runs (records at every site, e) and the row runner for one item (plan sections 7-8)."""

    def __init__(self, R, it, seed_off=0, rec="all"):
        self.R, self.it = R, it
        self.vb = it["ds"] == "VB"
        self.r, self.tA, self.tB = it["r"], it["tA"], it["tB"]
        self.seed = seed_of(it["pair_id"], seed_off)
        R.set_init(it["Tmax"], self.seed)
        if rec == "all":
            rec = {s: "all" for s in R.all_sites()}
        full = bool(rec) and len(rec) > 1
        sync(R.dev)
        t0 = time.time()
        self.cA = R.run(it["A"], self.r, rec=rec, rec_e=full, cont=self.vb and full)
        sync(R.dev)
        self.t_fwd = time.time() - t0
        self.cB = R.run(it["B"], self.r, rec=rec, rec_e=full, cont=self.vb and full)
        zA, zB = self.cA["z_r"], self.cB["z_r"]
        self.mA = float(zA[self.tB] - zA[self.tA])
        self.mB = float(zB[self.tB] - zB[self.tA])
        self.D = self.mB - self.mA
        okA, okB = int(zA.argmax()) == self.tA, int(zB.argmax()) == self.tB
        self.status = "A_fail" if not okA else ("B_fail" if not okB else ("D_small" if self.D < GAP else "kept"))
        self.rows = {}
        self.out = {k: it[k] for k in ("ds", "pair_id", "cluster", "cluster_b", "split", "rel", "strata", "label", "r",
                                       "src", "tA", "tB", "Tmax") if k in it}
        self.out.update({"seed": self.seed, "seed_off": seed_off, "T": len(it["A"]), "clean_A": vals(zA, self.tA, self.tB),
                         "clean_B": vals(zB, self.tA, self.tB), "D": r5(self.D), "status": self.status,
                         "t_fwd": round(self.t_fwd, 4), "rows": self.rows})

    def _t(self, pos):
        return torch.tensor(pos, dtype=torch.long, device=self.R.dev)

    def allpos(self):
        return list(range(len(self.it["A"])))

    # --- edit builders
    def o1(self, site, pos, a=1.0):
        return {site: [("set", self._t(pos), self.cB["H"][site][pos], float(a))]}

    def rnd(self, site, pos, pos_name):
        hA, hB = self.cA["H"][site][pos], self.cB["H"][site][pos]
        dn = (hB - hA).float().norm(dim=-1).tolist()
        d = rnd_delta(self.it["pair_id"], skey(site), pos_name, pos, dn, hA.shape[-1]).to(self.R.dev)
        return {site: [("add", self._t(pos), d, 1.0)]}

    def e_ed(self, loops, pos):
        t, rows = self._t(pos), self.cB["E"][pos]
        return {j: (t, rows) for j in loops}

    def ve_ed(self, pos, after=None, coda=True, core=True):
        """B's value embeds at pos for every core call after `after` (execution order; all if None) and the coda.
        core=False: coda keys only (VE-coda); coda=False: core keys only (VE-core) (review round 2 F1)."""
        R, t, out = self.R, self._t(pos), {}
        for l, k in (R.ve_core.items() if core else ()):
            rows = R.ve_rows(k, self.it["B"], pos)
            for j in range(1, K + 1):
                if after is None or exec_order(("c", j, l), R.C) > exec_order(after, R.C):
                    out[(k, j)] = (t, rows)
        if coda:
            for c, k in R.ve_coda.items():
                out[(k, 1)] = (t, R.ve_rows(k, self.it["B"], pos))
        return out

    # --- rows
    @staticmethod
    def lp(o, rest):
        s = logp(o["z_r"], rest[0])
        for k in range(1, len(rest)):
            s += logp(o["z_cont"][k - 1], rest[k])
        return s

    def finish_clean_vb(self):
        it, R = self.it, self.R
        lpA_A = self.lp(self.cA, it["restA"]) if it["cont"] else logp(self.cA["z_r"], it["restA"][0])
        lpB_B = self.lp(self.cB, it["restB"]) if it["cont"] else logp(self.cB["z_r"], it["restB"][0])
        if it["cont"]:
            a2 = R.run(it["A2"], self.r, cont=True)
            b2 = R.run(it["B2"], self.r, cont=True)
            lpB_A, lpA_B = self.lp(a2, it["restB"]), self.lp(b2, it["restA"])
            self.out["causality_ok"] = bool(torch.equal(a2["z_r"], self.cA["z_r"]) and torch.equal(b2["z_r"], self.cB["z_r"]))
        else:
            lpB_A, lpA_B = logp(self.cA["z_r"], it["restB"][0]), logp(self.cB["z_r"], it["restA"][0])
        self.LO_A, self.LO_B = lpB_A - lpA_A, lpB_B - lpA_B
        self.out.update({"LO_A": r5(self.LO_A), "LO_B": r5(self.LO_B), "D_LO": r5(self.LO_B - self.LO_A),
                         "cont": it["cont"], "n_rest": len(it["restA"])})

    def row(self, key, edits=None, e_edits=None, ve_edits=None, lo=False, exact_vs=None, eown=None, rec_eargs=False,
            keep=False):
        if key in self.rows:
            return None
        it, R, r = self.it, self.R, self.r
        L = R.llast
        rec = {("c", j, L): [r] for j in range(1, K + 1)}
        cont = bool(lo and self.vb and it["cont"])
        o = R.run(it["A"], r, edits=edits, e_edits=e_edits, ve_edits=ve_edits, rec=rec, cont=cont, rec_eargs=rec_eargs)
        rv = {"f": vals(o["z_r"], self.tA, self.tB)}
        rv["pres"] = [r5(proj(o["H"][("c", j, L)][0], self.cA["H"][("c", j, L)][r], self.cB["H"][("c", j, L)][r]))
                      for j in range(1, K + 1)]
        if lo and self.vb:
            if cont:
                lpA = self.lp(o, it["restA"])
                o2 = R.run(it["A2"], r, edits=edits, e_edits=e_edits, ve_edits=ve_edits, cont=True)
                lpB = self.lp(o2, it["restB"])
                rv["caus"] = bool(torch.equal(o2["z_r"], o["z_r"]))
            else:
                lpA, lpB = logp(o["z_r"], it["restA"][0]), logp(o["z_r"], it["restB"][0])
            rv["lo"] = r5(lpB - lpA)
        if exact_vs is not None:
            rv["exact"] = bool(torch.equal(o["z_r"], exact_vs))
        if eown is not None:
            j, a = eown
            s = ("c", j, L)
            h = self.cA["H"][s].clone()
            h[r] = self.cB["H"][s][r] if a == 1.0 else torch.lerp(h[r], self.cB["H"][s][r], float(a))
            z = R.readout(it["A"], h)[r]
            rv["mj"] = r5(z[self.tB] - z[self.tA])
            rv["mj_state_ok"] = bool(torch.equal(o["H"][s][0], h[r]))
        self.rows[key] = rv
        return o if keep else None

    def depth_margins(self):
        """Clean A and B depth-j margins through the per-loop readout (critic F3); j = K must equal the model."""
        R, r, L = self.R, self.r, self.R.llast
        la, lb = [], []
        for j in range(1, K + 1):
            za = R.readout(self.it["A"], self.cA["H"][("c", j, L)])[r]
            zb = R.readout(self.it["B"], self.cB["H"][("c", j, L)])[r]
            la.append(vals(za, self.tA, self.tB))
            lb.append(vals(zb, self.tA, self.tB))
            if j == K:
                self.out["readout_exact_K"] = bool(torch.equal(za, self.cA["z_r"]) and torch.equal(zb, self.cB["z_r"]))
        self.out["loopsA"], self.out["loopsB"] = la, lb
        if not self.out["readout_exact_K"]:
            raise HarnessFail("per-loop readout at j=K differs from the model logits: %s" % self.it["pair_id"])


# ------------------------------------------------------------------ arms
def run_controls(X, subset):
    """PC-final, SZ (every item) and PC-site (subset) [harness]: any inexact row stops the run."""
    R, r, src = X.R, X.r, X.it["src"]
    sK = ("k", R.Cd - 1)
    X.row("PCF", edits=X.o1(sK, [r]), exact_vs=X.cB["z_r"], lo=X.vb)
    X.row("SZ|O1", edits=X.o1(sK, src), exact_vs=X.cA["z_r"], lo=X.vb)
    X.row("SZ|RND", edits=X.rnd(sK, src, "src"), exact_vs=X.cA["z_r"], lo=X.vb)
    bad = [k for k in ("PCF", "SZ|O1", "SZ|RND") if not X.rows[k]["exact"]]
    if subset:
        allp = X.allpos()
        for name in e1_names(X.it["ds"]):
            s = name2site(name, R.C)
            j = site_loop(s)
            k = "PCS|%s" % skey(s)
            X.row(k, edits=X.o1(s, allp), e_edits=X.e_ed(range(j + 1, K + 1), allp), ve_edits=X.ve_ed(allp, after=s),
                  exact_vs=X.cB["z_r"])
            if not X.rows[k]["exact"]:
                bad.append(k)
    if bad:
        raise HarnessFail("harness rows not exact on %s: %s" % (X.it["pair_id"], bad))


def run_e1(X):
    R, r, src, vb = X.R, X.r, X.it["src"], X.vb
    for name in e1_names(X.it["ds"]):
        s = name2site(name, R.C)
        k = skey(s)
        lo_rnd = vb and name in ("L0@1", "Llast@1")
        eo = (s[1], s[2]) if (s[0] == "c" and s[2] == R.llast and s[1] < K) else None
        X.row("O1|src|%s|1" % k, edits=X.o1(s, src), lo=vb)
        X.row("RND|src|%s" % k, edits=X.rnd(s, src, "src"), lo=lo_rnd)
        X.row("O1|ans|%s|1" % k, edits=X.o1(s, [r]), lo=vb, eown=(eo[0], 1.0) if eo else None)
        X.row("O1|ans|%s|0.25" % k, edits=X.o1(s, [r], 0.25), eown=(eo[0], 0.25) if eo else None)
        X.row("RND|ans|%s" % k, edits=X.rnd(s, [r], "ans"), lo=lo_rnd)


def run_e5(X, ad5):
    src, allp = X.it["src"], X.allpos()
    for j in range(1, K + 1):
        X.row("IN1|%d" % j, e_edits=X.e_ed([j], src))
    for j in range(1, K):
        X.row("INP|%d" % j, e_edits=X.e_ed(range(j, K + 1), src))
    X.row("VEP", ve_edits=X.ve_ed(src))
    X.row("VEC", ve_edits=X.ve_ed(src, coda=False))           # review round 2 F1: loop value embeds only
    X.row("VECD", ve_edits=X.ve_ed(src, core=False))          # coda value embeds only
    X.row("INVEP", e_edits=X.e_ed(range(1, K + 1), src), ve_edits=X.ve_ed(src))
    if ad5:
        X.row("STAVC|p%d" % K, edits=X.o1(("p", K), allp), ve_edits=X.ve_ed(src, core=False))
        for j in (1, 7):
            X.row("INPA|%d" % j, e_edits=X.e_ed(range(j, K + 1), allp))
        down = [p for p in allp if p > max(src)]
        X.row("INPD|1", e_edits=X.e_ed(range(1, K + 1), down))
        for name in STA_NAMES:
            s = name2site(name, X.R.C)
            X.row("STA|%s" % skey(s), edits=X.o1(s, allp))


def run_para(X, force=False):
    """O2 vectors and the presence validation from the first paraphrase form that passes the clean filter (P4).
    force (CPU unit test only) skips the filter."""
    R, it, r = X.R, X.it, X.r
    names = e1_names(it["ds"])
    sites = [name2site(n, R.C) for n in names]
    tried = []
    for f in it["para"]:
        rec = {s: "all" for s in sites}
        pa = R.run(f["A"], f["r"], rec=rec)
        pb = R.run(f["B"], f["r"], rec=rec)
        Dp = float((pb["z_r"][X.tB] - pb["z_r"][X.tA]) - (pa["z_r"][X.tB] - pa["z_r"][X.tA]))
        ok = (int(pa["z_r"].argmax()) == X.tA and int(pb["z_r"].argmax()) == X.tB and Dp >= GAP) or force
        tried.append([f["fmt"], bool(ok), r5(Dp)])          # review round 2 F5: P4 filter includes D' >= 2
        if not ok:
            continue
        X.out["para_fmt"] = f["fmt"]
        pv = {}
        for s in sites:
            mA, mB = X.cA["H"][s][r], X.cB["H"][s][r]
            pv[skey(s)] = [r5(proj(pa["H"][s][f["r"]], mA, mB)), r5(proj(pb["H"][s][f["r"]], mA, mB))]
        X.out["pval"] = pv
        diag = {}
        for name in O2_SRC:
            s = name2site(name, R.C)
            v = pb["H"][s][f["src"]] - pa["H"][s][f["src"]]
            own = X.cB["H"][s][it["src"]] - X.cA["H"][s][it["src"]]
            diag["src|" + skey(s)] = [r5(torch.nn.functional.cosine_similarity(v.flatten().float(), own.flatten().float(), 0)),
                                      r5(v.float().norm() / own.float().norm().clamp_min(1e-12))]
            X.row("O2|src|%s" % skey(s), edits={s: [("add", X._t(it["src"]), v, 1.0)]})
        for name in O2_ANS:
            s = name2site(name, R.C)
            v = pb["H"][s][[f["r"]]] - pa["H"][s][[f["r"]]]
            own = X.cB["H"][s][[r]] - X.cA["H"][s][[r]]
            diag["ans|" + skey(s)] = [r5(torch.nn.functional.cosine_similarity(v.flatten().float(), own.flatten().float(), 0)),
                                      r5(v.float().norm() / own.float().norm().clamp_min(1e-12))]
            X.row("O2|ans|%s" % skey(s), edits={s: [("add", X._t([r]), v, 1.0)]})
        X.out["o2diag"] = diag
        break
    X.out["para_tried"] = tried


def run_e8(X, dev_search):
    """Carry gain (plan section 8): one forward + backward with zero leaves at r at every site of S+, verify rows
    at the top/median/bottom 3 of S and of S+, random site of S, dev search on dev items."""
    R, it, r = X.R, X.it, X.r
    S, Sp = e8_sets(R.C)
    sync(R.dev)
    t0 = time.time()
    R.run(it["A"], r)                        # review round 2 F9: plain forward (no recording) = cost denominator
    sync(R.dev)
    X.out["t_fwd_plain"] = round(time.time() - t0, 4)
    leaves = {s: torch.zeros(1, 1, R.d, device=R.dev, requires_grad=True) for s in Sp}
    sync(R.dev)
    t0 = time.time()
    o = R.run(it["A"], r, leaves=leaves, grad=True)
    m = o["z_r"][X.tB] - o["z_r"][X.tA]
    m.backward()
    sync(R.dev)
    X.out["t_grad"] = round(time.time() - t0, 4)
    G = {}
    for s in Sp:
        g = leaves[s].grad[0, 0].float()
        d = (X.cB["H"][s][r] - X.cA["H"][s][r]).float()
        G[skey(s)] = r5(g @ d)
    X.out["grad_identical"] = bool(torch.equal(o["z_r"].detach(), X.cA["z_r"]))
    del leaves, o, m
    GS = {skey(s): G[skey(s)] for s in S}
    ksS, verS = verify_sites(GS)
    _, verP = verify_sites(G)
    rnd = ksS[int(sha(it["pair_id"] + "|rnd_site")[:8], 16) % len(ksS)]
    X.out.update({"G": G, "ver_S": verS, "ver_Sp": verP, "rnd_site": rnd, "dev_search": bool(dev_search)})
    todo = []
    for k in verS + [x for x in verP if x not in verS] + [rnd]:
        if k not in todo:
            todo.append(k)
    for a in A_VERIFY:
        for k in todo:
            X.row("O1|ans|%s|%g" % (k, a), edits=X.o1(ksite(k), [r], a))
    if dev_search:
        for s in Sp:
            X.row("O1|ans|%s|1" % skey(s), edits=X.o1(s, [r]))


def seq_B(it):
    return it["B"]


def dev_tuned_site(items, keys):
    best, bv, table = None, None, {}
    for k in keys:
        xs = [dm_of(x, "O1|ans|%s|1" % k) for x in items if x.get("dev_search")]
        xs = [v for v in xs if v is not None]
        if not xs:
            continue
        v = sum(xs) / len(xs)
        table[k] = round(v, 4)
        if bv is None or v > bv:
            best, bv = k, v
    return best, table


def dm_of(x, key):
    rv = x.get("rows", {}).get(key)
    if rv is None:
        return None
    return (rv["f"][1] - rv["f"][0]) - (x["clean_A"][1] - x["clean_A"][0])


def Dgap(x):
    return (x["clean_B"][1] - x["clean_B"][0]) - (x["clean_A"][1] - x["clean_A"][0])


def dev_tuned_pass(R, res, cmap):
    """Pass 2 (plan section 8): the dev-tuned site per set from dev items, then O1 at that site at both doses on
    test items that lack it (clean B rerun for the donor, same initial state)."""
    items = res["items"]
    S, Sp = e8_sets(R.C)
    out, n = {}, 0
    for name, keys in (("S", [skey(s) for s in S]), ("Sp", [skey(s) for s in Sp])):
        best, table = dev_tuned_site([x for x in items if x["split"] == "dev"], keys)
        out[name] = {"site": best, "n_dev": sum(1 for x in items if x["split"] == "dev" and x.get("dev_search")),
                     "table": table}
        if best is None:
            continue
        s = ksite(best)
        for x in items:
            if x["split"] != "test":
                continue
            need = [a for a in A_VERIFY if "O1|ans|%s|%g" % (best, a) not in x["rows"]]
            if not need:
                continue
            it = cmap[x["pair_id"]]
            R.set_init(it["Tmax"], x["seed"])
            cB = R.run(it["B"], it["r"], rec={s: [it["r"]]})
            for a in need:
                o = R.run(it["A"], it["r"], edits={s: [("set", torch.tensor([it["r"]], device=R.dev), cB["H"][s], a)]})
                x["rows"]["O1|ans|%s|%g" % (best, a)] = {"f": vals(o["z_r"], it["tA"], it["tB"]), "pass2": True}
                n += 1
    out["rows_added"] = n
    res["dev_tuned"] = out
    return out


# ------------------------------------------------------------------ hook checks (plan section 6, a-j)
def check_hooks(R, it):
    out = {}
    X = Item(R, it)
    A, r, src = it["A"], it["r"], it["src"]
    T = len(A)
    allp = list(range(T))
    tall = X._t(allp)
    ref = R.run(A, r, full=True)
    out["a_counts"] = {"core_calls_per_block": K, "adapter_calls": K, "ve_core_keys": sorted(R.ve_core.values()),
                       "ve_coda_keys": sorted(R.ve_coda.values()), "coda_calls_per_block": 1,
                       "asserted_in_every_forward": True}
    sites = R.all_sites() + [("k", c) for c in range(R.Cd - 1)]
    zero = {s: [("add", tall, torch.zeros(T, R.de if s[0] == "k" else R.d, device=R.dev), 1.0)] for s in sites}
    e0 = {j: (tall, X.cA["E"]) for j in range(1, K + 1)}
    v0 = {}
    for l, k in R.ve_core.items():
        for j in range(1, K + 1):
            v0[(k, j)] = (tall, R.ve_rows(k, A, allp))
    for c, k in R.ve_coda.items():
        v0[(k, 1)] = (tall, R.ve_rows(k, A, allp))
    o = R.run(A, r, edits=zero, e_edits=e0, ve_edits=v0, full=True)
    out["b_zero_edit_identity"] = bool(torch.equal(o["z"], ref["z"]))
    o = R.run(A, r, full=True)
    out["c_repeat_identical"] = bool(torch.equal(o["z"], ref["z"]))
    s_ed, p = ("c", 4, R.lmid), src[0]
    rec = {s: "all" for s in R.all_sites()}
    oe = R.run(A, r, edits=X.o1(s_ed, [p]), rec=rec)
    o_ed = exec_order(s_ed, R.C)
    earlier = [s for s in R.all_sites() if exec_order(s, R.C) < o_ed]
    later = [s for s in R.all_sites() if exec_order(s, R.C) > o_ed]
    oth = [q for q in allp if q != p]
    out["d_edit_locality"] = {
        "earlier_identical": all(torch.equal(oe["H"][s], X.cA["H"][s]) for s in earlier),
        "later_prefix_identical": all(torch.equal(oe["H"][s][:p], X.cA["H"][s][:p]) for s in later),
        "at_site": bool(torch.equal(oe["H"][s_ed][p], X.cB["H"][s_ed][p])
                        and torch.equal(oe["H"][s_ed][oth], X.cA["H"][s_ed][oth])),
        "later_changed_from_p": any(not torch.equal(oe["H"][s][p:], X.cA["H"][s][p:]) for s in later),
        "n_earlier": len(earlier), "n_later": len(later)}
    S, Sp = e8_sets(R.C)
    leaves = {s: torch.zeros(1, 1, R.d, device=R.dev, requires_grad=True) for s in Sp}
    og = R.run(A, r, leaves=leaves, grad=True, full=True)
    m = og["z_r"][it["tB"]] - og["z_r"][it["tA"]]
    m.backward()
    out["e_grad_run_identical"] = bool(torch.equal(og["z"], ref["z"]))
    out["e_grad_nonzero_sites"] = sum(1 for s in Sp if float(leaves[s].grad.abs().sum()) > 0)
    del leaves, og, m
    out["f_parcae_py_sha16"] = parcae_sha16()
    out["f_ok"] = out["f_parcae_py_sha16"] == PARCAE_SHA16
    g = {}
    for j in (1, 4, K):
        oj = R.run(A, r, steps=j, full=True)
        zr = R.readout(A, X.cA["H"][("c", j, R.llast)])
        g[str(j)] = bool(torch.equal(zr, oj["z"]))
    out["g_depth_readout_identical"] = g
    oh = R.run(A, r, e_edits=X.e_ed([3], src), rec_eargs=True)
    nsrc = [q for q in allp if q not in src]
    out["h_e_single_call_only"] = {
        "other_calls_equal_clean_A": all(torch.equal(oh["eargs"][n], X.cA["E"]) for n in range(1, K + 1) if n != 3),
        "call3_src_equals_B": bool(torch.equal(oh["eargs"][3][src], X.cB["E"][src])),
        "call3_rest_equals_A": bool(torch.equal(oh["eargs"][3][nsrc], X.cA["E"][nsrc]))}
    oi = R.run(A, r, e_edits={j: (tall, X.cB["E"]) for j in range(1, K + 1)}, ve_edits=X.ve_ed(allp))
    out["i_all_inputs_B_equals_clean_B"] = bool(torch.equal(oi["z_r"], X.cB["z_r"]))
    x = R.set_init(512, 12345)
    rel = float(x.std()) / (TRUNC_STD * R.std * float(R.m.emb_scale))
    out["j_init_std"] = {"rel_to_truncated_std": round(rel, 5), "within_2pct": abs(rel - 1) < 0.02,
                         "cfg_std": R.std, "emb_scale": float(R.m.emb_scale)}
    R.set_init(it["Tmax"], X.seed)
    ok = (out["b_zero_edit_identity"] and out["c_repeat_identical"] and all(v for k, v in out["d_edit_locality"].items()
          if not k.startswith("n_")) and out["e_grad_run_identical"] and out["f_ok"] and all(g.values())
          and all(out["h_e_single_call_only"].values()) and out["i_all_inputs_B_equals_clean_B"]
          and out["j_init_std"]["within_2pct"] and out["e_grad_nonzero_sites"] == len(Sp))
    out["all_ok"] = bool(ok)
    out["item"] = it["pair_id"]
    return out


# ------------------------------------------------------------------ scans, datasets, model driver
def bos_calibration(R, cands_d1):
    cal = [c for c in cands_d1 if c["rel"] == "capital" and c.get("k15_half") == "dev"]
    acc = {}
    for label, b in BOS_OPTIONS:
        n = 0
        for c in cal:
            it = prep_item(c, b)
            R.set_init(it["Tmax"], seed_of(c["pair_id"]))
            o = R.run(it["A"], it["r"])
            n += int(int(o["z_r"].argmax()) == c["tA"])
        acc[label] = round(n / max(1, len(cal)), 4)
    best = max(BOS_OPTIONS, key=lambda lb: acc[lb[0]])   # first maximum: ties go to BOS 0
    return {"n": len(cal), "acc": acc, "chosen": best[0], "chosen_id": best[1]}


def scan_dataset(R, ds, cands, target, deadline=None, seed_off=0, statuses=None):
    """Clean filter in selection order (plan section 9). -> (kept pair_ids, scan log, stopped by the deadline)."""
    sel = Selector(ds, target=target)
    cmap = {c["pair_id"]: c for c in cands}
    log = list(statuses or [])
    done = set()
    stopped = False
    for pid, st in log:
        done.add(pid)
        if st == "kept":
            sel.accept(cmap[pid])
    for c in cands:
        if sel.done():
            break
        if c["pair_id"] in done or not sel.want(c):
            continue
        if deadline is not None and time.time() > deadline:
            stopped = True                         # review round 2 F8: a budget stop is not a yield result
            break
        it = prep_item(c, R.bos)
        if it is None:
            log.append([c["pair_id"], "pos0_dropped"])
            continue
        X = Item(R, it, seed_off=seed_off, rec=None)
        log.append([c["pair_id"], X.status])
        if X.status == "kept":
            sel.accept(c)
    kept = [pid for pid, st in log if st == "kept"]
    return kept, log, stopped


def scan_stats(log, cmap):
    st = {"scanned": 0, "A_fail": 0, "B_fail": 0, "D_small": 0, "pos0_dropped": 0, "kept": 0}
    for pid, s in log:
        st["scanned"] += 1
        st[s] = st.get(s, 0) + 1
    kept = [cmap[pid] for pid, s in log if s == "kept"]
    st["n_clusters"] = len(set(c["cluster"] for c in kept))
    st["n_units"] = n_units(kept)
    st["dev_test_unit_overlap_A_or_B"] = unit_overlap(kept)
    pl = {}
    for c in kept:
        k = "%s|%s" % (c["strata"]["pos_len"], c["rel"])
        pl[k] = pl.get(k, 0) + 1
    st["pos_len_x_rel"] = pl
    st["split"] = {s: sum(1 for c in kept if c["split"] == s) for s in ("dev", "test")}
    st["dev_test_cluster_overlap"] = len({c["cluster"] for c in kept if c["split"] == "dev"}
                                         & {c["cluster"] for c in kept if c["split"] == "test"})
    return st


def run_dataset(R, ds, cands, out_dir, debug, deadline, logf):
    path = os.path.join(out_dir, "res_%s_%s.json" % (R.name, ds))
    res = json.load(open(path)) if os.path.exists(path) else {"ds": ds, "model": R.name, "bos": R.bos, "scan": [],
                                                                "items": [], "meta": {}}
    assert res["bos"] == R.bos, "resume with a different BOS"
    cmap = {c["pair_id"]: c for c in cands}
    target = debug if debug else TARGET[ds]
    if res["meta"].get("target") not in (None, target):
        raise HarnessFail("resume with a different target: %s vs %s" % (res["meta"]["target"], target))
    t0 = time.time()
    kept, log, stopped = scan_dataset(R, ds, cands, target, deadline, statuses=res["scan"])
    res["scan"] = log
    res["meta"]["scan_stopped_budget"] = stopped
    res["meta"]["scan_stats"] = scan_stats(log, cmap)
    res["meta"]["target"] = target
    res["meta"]["scan_seconds"] = round(res["meta"].get("scan_seconds", 0) + time.time() - t0, 1)
    if stopped and len(kept) < MIN_KEPT and not debug:
        res["meta"]["skipped"] = None
        res["meta"]["stopped_budget"] = "scan stopped by the budget deadline at kept %d (not a yield result)" % len(kept)
        atomic_json(path, res)
        logf("%s %s %s" % (R.name, ds, res["meta"]["stopped_budget"]))
        return res
    if len(kept) < MIN_KEPT and not debug:
        res["meta"]["skipped"] = "kept %d < %d after the D >= 2 filter (full scan)" % (len(kept), MIN_KEPT)
        atomic_json(path, res)
        logf("%s %s skipped: %s" % (R.name, ds, res["meta"]["skipped"]))
        return res
    res["meta"]["skipped"] = None
    done = {x["pair_id"] for x in res["items"]}
    atomic_json(path, res)
    for k, pid in enumerate(kept):
        if pid in done:
            continue
        if time.time() > deadline:
            res["meta"]["stopped_budget"] = True
            break
        t1 = time.time()
        it = prep_item(cmap[pid], R.bos)
        X = Item(R, it)
        if X.status != "kept":
            raise HarnessFail("clean status changed between scan and main pass: %s %s" % (pid, X.status))
        subset = k < N_SUB
        X.depth_margins()
        if X.vb:
            X.finish_clean_vb()
        run_controls(X, subset)
        run_e1(X)
        run_e5(X, ds in ("D1", "D2") or subset)
        run_para(X)
        n_ds = sum(1 for x in res["items"] if x.get("dev_search"))
        if not NO_GRAD[0]:
            run_e8(X, it["split"] == "dev" and (ds != "VB" or n_ds < VB_DEV_SEARCH))
        X.out.update({"subset": subset, "ad5": ds in ("D1", "D2") or subset, "n_rows": len(X.rows),
                      "t_item": round(time.time() - t1, 2), "kept_index": k})
        res["items"].append(X.out)
        atomic_json(path, res)
        logf("%s %s item %d/%d %s rows %d %.1fs" % (R.name, ds, k + 1, len(kept), pid, len(X.rows), time.time() - t1))
        del X
    if not res["meta"].get("stopped_budget") and time.time() > deadline:
        res["meta"]["stopped_budget"] = "deadline reached before the dev-tuned pass"
    if not res["meta"].get("stopped_budget") and not NO_GRAD[0]:
        cm = {x["pair_id"]: prep_item(cmap[x["pair_id"]], R.bos) for x in res["items"]}
        dt = dev_tuned_pass(R, res, cm)
        logf("%s %s dev-tuned %s" % (R.name, ds, json.dumps({k: v.get("site") if isinstance(v, dict) else v
                                                             for k, v in dt.items()})))
    res["meta"]["seconds"] = round(res["meta"].get("seconds", 0) + time.time() - t0, 1)
    atomic_json(path, res)
    return res


def seed_sensitivity(R, cands, res, deadline):
    """PR7: D1 clean-scan counts and the source O1 Llast@7 cell at seed offsets 1 and 2 (plan DC5)."""
    cmap = {c["pair_id"]: c for c in cands}
    kept0 = [pid for pid, s in res["scan"] if s == "kept"]
    s7 = ("c", 7, R.llast)
    k7 = "O1|src|%s|1" % skey(s7)
    out = {"offsets": {}}
    for off in SEED_OFFSETS[1:]:
        if time.time() > deadline:
            out["offsets"][str(off)] = {"stopped_budget": "deadline reached before this offset"}
            continue
        kept, log, stopped = scan_dataset(R, "D1", cands, TARGET["D1"], deadline, seed_off=off)
        e = {}
        n_skip = 0
        for x in res["items"]:
            if time.time() > deadline:
                n_skip += 1
                continue
            it = prep_item(cmap[x["pair_id"]], R.bos)
            X = Item(R, it, seed_off=off, rec={s7: "all"})
            if X.D < GAP:
                continue
            o = R.run(it["A"], it["r"], edits=X.o1(s7, it["src"]))
            e[x["pair_id"]] = r5((float(o["z_r"][X.tB] - o["z_r"][X.tA]) - X.mA) / X.D)
        out["offsets"][str(off)] = {"kept": len(kept), "kept_off0": len(kept0), "scan_stopped_budget": stopped,
                                    "items_not_run_budget": n_skip,
                                    "sym_diff": len(set(kept) ^ set(kept0)), "cell_key": k7, "e": e}
    res["meta"]["seed_sensitivity"] = out
    return out


def gpu_mem():
    if not torch.cuda.is_available():
        return None
    f, t = torch.cuda.mem_get_info()
    return {"free_gb": round(f / 1e9, 1), "total_gb": round(t / 1e9, 1),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def run_model(name, data_dir, out_dir, dev, debug, datasets, budget_min, scan_only=False, deadline_epoch=None):
    """budget_min caps this process; deadline_epoch (absolute, shared by every launch of one run) caps the run
    total (review round 2 F8). The earlier of the two applies."""
    t_start = time.time()
    deadline = t_start + 60 * budget_min
    if deadline_epoch:
        deadline = min(deadline, float(deadline_epoch))
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    lf = open(os.path.join(out_dir, "logs", "%s%s.log" % (name, "_scan" if scan_only else "")), "a")

    def logf(s):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
    meta = {"model": name, "debug": debug, "scan_only": scan_only, "budget_min": budget_min,
            "deadline_epoch": deadline_epoch, "deadline_used": round(deadline, 1), "targets": dict(TARGET),
            "torch": torch.__version__, "python": sys.version.split()[0], "gpu_mem_start": gpu_mem(),
            "tokenizer_md5_ok": (md5_file(TOK_REMOTE) == TOK_MD5) if os.path.exists(TOK_REMOTE) else None,
            "parcae_py_sha16": parcae_sha16(), "k17_parcae_md5": md5_file(os.path.abspath(__file__))}
    assert meta["parcae_py_sha16"] == PARCAE_SHA16, "installed parcae.py differs from the audited source"
    if NO_GRAD[0]:
        meta["no_gradient_rank"] = True
    bm = json.load(open(os.path.join(data_dir, "build_meta.json")))
    meta["build_meta_sha256"] = sha256_file(os.path.join(data_dir, "build_meta.json"))
    assert bm["tokenizer_md5"] == TOK_MD5
    files = {ds: json.load(open(os.path.join(data_dir, "cands_%s.json" % ds))) for ds in DATASETS}
    cands = {ds: f["cands"] for ds, f in files.items()}
    bos_pool = files["D1"].get("bos_cal") or cands["D1"]        # DC3 pool kept fixed across the round-2 D1 drop
    model, cfg, audit = load_model(name, dev)
    meta["load_audit"] = audit
    R = PR(model, cfg, dev, bos=0, name=name)
    meta["arch"] = {"P": R.P, "C": R.C, "Cd": R.Cd, "d": R.d, "lmid": R.lmid, "ve_core": R.ve_core, "ve_coda": R.ve_coda,
                    "init_std": R.std, "emb_scale": float(model.emb_scale)}
    mpath = os.path.join(out_dir, "meta_%s%s.json" % (name, "_scan" if scan_only else ""))
    if os.path.exists(mpath):                         # a second invocation keeps the earlier run's meta (minutes)
        prev = json.load(open(mpath))
        meta["previous_runs"] = prev.pop("previous_runs", []) + [prev]
    cal = bos_calibration(R, bos_pool)
    R.bos = cal["chosen_id"]
    meta["bos_calibration"] = cal
    logf("%s BOS %s" % (name, json.dumps(cal)))
    if scan_only:
        meta["scans"] = {}
        for ds in datasets:
            kept, log, stopped = scan_dataset(R, ds, cands[ds], TARGET[ds], deadline)
            meta["scans"][ds] = scan_stats(log, {c["pair_id"]: c for c in cands[ds]})
            meta["scans"][ds]["scan_stopped_budget"] = stopped
            logf("%s scan %s %s" % (name, ds, json.dumps(meta["scans"][ds])))
            atomic_json(mpath, meta)
    else:
        first = next(prep_item(c, R.bos) for c in cands["D1"] if prep_item(c, R.bos) is not None)
        hk = check_hooks(R, first)
        meta["hook_identity"] = hk
        logf("%s hook checks %s" % (name, json.dumps(hk)))
        atomic_json(mpath, meta)
        if not hk["all_ok"]:
            raise HarnessFail("hook checks failed")
        meta["datasets"] = {}
        for ds in datasets:
            res = run_dataset(R, ds, cands[ds], out_dir, debug, deadline, logf)
            meta["datasets"][ds] = {"kept": len(res["items"]), "skipped": res["meta"].get("skipped"),
                                    "stopped_budget": res["meta"].get("stopped_budget", False)}
            if ds == "D1" and name == "parcae-1.3b" and not debug and res["items"] and time.time() < deadline:
                ss = seed_sensitivity(R, cands["D1"], res, deadline)
                atomic_json(os.path.join(out_dir, "res_%s_%s.json" % (name, ds)), res)
                logf("seed sensitivity %s" % json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "e"}
                                                        for k, v in ss["offsets"].items()}))
            atomic_json(mpath, meta)
    meta["gpu_mem_end"] = gpu_mem()
    meta["process_minutes"] = round((time.time() - t_start) / 60, 2)
    atomic_json(mpath, meta)
    logf("%s done in %.1f min" % (name, meta["process_minutes"]))
    lf.close()
    return meta


# ------------------------------------------------------------------ summary (checks.json keys, plan section 12)
def e_of(x, key):
    rv = x["rows"].get(key)
    D = Dgap(x)
    if rv is None or D < GAP:
        return None
    return ((rv["f"][1] - rv["f"][0]) - (x["clean_A"][1] - x["clean_A"][0])) / D


def flip_of(x, key):
    rv = x["rows"].get(key)
    return None if rv is None else float(rv["f"][3] == x["tB"])


def elo_of(x, key):
    rv = x["rows"].get(key)
    if rv is None or rv.get("lo") is None or x.get("D_LO") is None or x["D_LO"] < GAP:
        return None
    return (rv["lo"] - x["LO_A"]) / x["D_LO"]


def eown_of(x, key):
    rv = x["rows"].get(key)
    D = Dgap(x)
    if rv is None or rv.get("mj") is None or D < GAP:
        return None
    j = int(key.split("|")[2][1:].split(".")[0])
    mAj = x["loopsA"][j - 1][1] - x["loopsA"][j - 1][0]
    return (rv["mj"] - mAj) / D


def mean(xs):
    xs = [v for v in xs if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def median(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    n = len(xs)
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 4)


def _stratum(x, sk):
    v = x["strata"].get(sk)
    return x.get(sk) if v is None else v


def cell(items, key, null=False, strata=None, lo=False, rnd=False, extra_strata=()):
    """One reported cell: e with its cluster CI (dyadic on D1), flip, n_items, n_clusters (units), strata.
    strata None -> the dataset default STRATA (A1, review round 2 F7); extra_strata adds item-level keys
    (para_fmt on O2 cells)."""
    if strata is None:
        strata = STRATA.get(items[0]["ds"], ("pos_len",)) if items else ()
    strata = tuple(strata) + tuple(extra_strata)
    xs = [e_of(x, key) for x in items]
    cl = [cl_of(x) for x in items]
    b = boot(xs, cl)
    if b is None:
        return None
    out = {"n_items": b[3], "n_clusters": b[4], "e": b[:3], "flip": mean(flip_of(x, key) for x in items)}
    if lo:
        bl = boot([elo_of(x, key) for x in items], cl)
        out["e_LO"] = bl[:3] if bl else None
        out["n_LO"] = bl[3] if bl else 0
    if null or rnd:
        out["tost"] = tost(xs, cl)
    if rnd:
        out["T3_pass"] = bool(b[0] is not None and abs(b[0]) <= 0.05 and (out["flip"] or 0) <= 0.02)
    st = {}
    for sk in strata:
        has = [x for x in items if e_of(x, key) is not None]
        vals_ = sorted(set(str(_stratum(x, sk)) for x in has if _stratum(x, sk) is not None))
        if len(vals_) < 2:
            if vals_:
                st[sk] = {"single_value": vals_[0]}
            continue
        st[sk] = {}
        for v in vals_:
            sub = [x for x in has if str(_stratum(x, sk)) == v]
            bb = boot([e_of(x, key) for x in sub], [cl_of(x) for x in sub])
            st[sk][v] = None if bb is None else {"e": bb[:3], "n_items": bb[3], "n_clusters": bb[4]}
    if st:
        out["strata"] = st
    return out


def load_results(out_dir):
    res, metas = {}, {}
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("res_") and f.endswith(".json"):
            d = json.load(open(os.path.join(out_dir, f)))
            res[(d["model"], d["ds"])] = d
        elif f.startswith("meta_") and f.endswith(".json"):
            d = json.load(open(os.path.join(out_dir, f)))
            metas[f[5:-5]] = d
    return res, metas


def summarize_e8(items, C, a, set_name):
    S, Sp = e8_sets(C)
    keys = [skey(s) for s in (S if set_name == "S" else Sp)]
    vf = "ver_S" if set_name == "S" else "ver_Sp"
    uni = skey(("c", K, C // 2)) if set_name == "S" else skey(("c", K, C - 1))
    its = [x for x in items if x.get("G")]
    per, tops = [], []
    for x in its:
        G = {k: x["G"][k] for k in keys}
        ver = x[vf]
        Gv = {k: G[k] for k in ver}
        dmv = {k: dm_of(x, "O1|ans|%s|%g" % (k, a)) for k in ver}
        h = hit_metrics(Gv, dmv, Dgap(x), THR[a])
        top_all = max(keys, key=lambda k: (G[k], k))
        tops.append(ksite(top_all))
        per.append((x, h))
    elig = [(x, h) for x, h in per if h and h["eligible"]]
    elig_test = [(x, h) for x, h in elig if x["split"] == "test"]
    # review round 2 F4: hit, chance and Spearman on the eligible TEST items (the pool the gate counts); the
    # dev + test values are kept under *_devtest
    out = {"n_sites": len(keys), "n_items": len(its), "eligible": len(elig), "eligible_test": len(elig_test),
           "interpreted": len(elig_test) >= MIN_E8_TEST, "pool": "eligible test items",
           "top1_hit": mean(h["hit"] for _, h in elig_test), "chance": mean(h["chance"] for _, h in elig_test),
           "regret": mean(h["regret"] for x, h in per if h and x["split"] == "test"),
           "spearman_median": median(h["spearman"] for _, h in elig_test),
           "spearman_n": sum(1 for _, h in elig_test if h["spearman"] is not None),
           "devtest": {"eligible": len(elig), "top1_hit": mean(h["hit"] for _, h in elig),
                       "chance": mean(h["chance"] for _, h in elig),
                       "regret": mean(h["regret"] for _, h in per if h),
                       "spearman_median": median(h["spearman"] for _, h in elig),
                       "spearman_n": sum(1 for _, h in elig if h["spearman"] is not None)},
           "top_site_loop_hist": {str(j): sum(1 for s in tops if s[1] == j) for j in range(1, K + 1)},
           "top_is_Llast8": sum(1 for s in tops if s == ("c", K, C - 1)),
           "top_in_loop8_or_late7": sum(1 for s in tops if s[1] == K or (s[1] == K - 1 and s[2] >= C // 2))}
    test = [x for x in its if x["split"] == "test"]
    cl = [cl_of(x) for x in test]
    D = [Dgap(x) for x in test]
    pred = [dm_of(x, "O1|ans|%s|%g" % (max(x[vf], key=lambda k: (x["G"][k], k)), a)) for x in test]
    base = {"uniform": [dm_of(x, "O1|ans|%s|%g" % (uni, a)) for x in test],
            "random": [dm_of(x, "O1|ans|%s|%g" % (x["rnd_site"], a)) for x in test]}
    out["uniform_site"] = uni
    out["_pred_dmD"] = [None if p is None else p / d for p, d in zip(pred, D)]
    out["pred_dmD"] = boot(out["_pred_dmD"], cl)
    for nm, vs in base.items():
        vv = [None if v is None else v / d for v, d in zip(vs, D)]
        out["%s_dmD" % nm] = boot(vv, cl)
        out["pred_minus_%s" % nm] = paired_boot(out["_pred_dmD"], vv, cl)
    dev = [x for x in its if x["split"] == "dev" and x.get("dev_search")]
    pairs = []
    for x in dev:
        for k in [skey(s) for s in S]:
            pairs.append((x["G"][k], dm_of(x, "O1|ans|%s|1" % k), Dgap(x)))
    out["fn_rate_dev"] = fn_rate(pairs)
    pairs = []
    for x in test:
        for k in x["ver_S"]:
            pairs.append((x["G"][k], dm_of(x, "O1|ans|%s|1" % k), Dgap(x)))
    out["fn_rate_test9"] = fn_rate(pairs)
    tf = [x["t_grad"] / x["t_fwd_plain"] for x in its if x.get("t_grad") and x.get("t_fwd_plain")]
    out["cost_fwd_equiv"] = {"selector_per_item": mean(tf), "denominator": "plain forward, no recording (t_fwd_plain)",
                             "n_items_timed": len(tf), "dev_search_total": len(Sp) * len(dev),
                             "dev_search_per_dev_item": len(Sp)}
    # DV7 (run step): VARBENCH T7 v3 tolerance hit, secondary. (1) on the verified 9 of every item at dose a;
    # (2) at a = 1 on dev-search items over the full set (every site of the set verified).
    v9 = []
    for x in its:
        G = {k: x["G"][k] for k in x[vf]}
        dmv = {k: dm_of(x, "O1|ans|%s|%g" % (k, a)) for k in x[vf]}
        v9.append((x, hit_metrics_v3(G, dmv, Dgap(x), THR[a])))
    e9 = [(x, h) for x, h in v9 if h and h["eligible"]]
    out["v3_verified9"] = {"eligible": len(e9), "eligible_test": sum(1 for x, _ in e9 if x["split"] == "test"),
                           "hit_tol": mean(h["hit"] for _, h in e9), "chance_band": mean(h["chance"] for _, h in e9),
                           "top3_random_hit": mean(h["top3_hit"] for _, h in e9),
                           "strict_hit": mean(h["strict_hit"] for _, h in e9),
                           "regret": mean(h["regret"] for _, h in v9 if h)}
    if a == 1.0:
        full = []
        for x in dev:
            G = {k: x["G"][k] for k in keys}
            dmf = {k: dm_of(x, "O1|ans|%s|1" % k) for k in keys}
            full.append(hit_metrics_v3(G, dmf, Dgap(x), THR[a]))
        ef = [h for h in full if h and h["eligible"]]
        out["v3_dev_fullset"] = {"n_dev_search": len(dev), "eligible": len(ef),
                                 "hit_tol": mean(h["hit"] for h in ef), "chance_band": mean(h["chance"] for h in ef),
                                 "strict_hit": mean(h["strict_hit"] for h in ef),
                                 "regret": mean(h["regret"] for h in full if h),
                                 "spearman_median": median(h["spearman"] for h in ef),
                                 "n_band_median": median(h["n_band"] for h in ef)}
    return out


def add_dev_tuned(out, items, res, C, a, set_name):
    dt = res.get("dev_tuned", {}).get(set_name, {})
    site = dt.get("site")
    out["dev_tuned_site"] = site
    if site is None:
        return
    test = [x for x in items if x.get("G") and x["split"] == "test"]
    cl = [cl_of(x) for x in test]
    vv = [None if dm_of(x, "O1|ans|%s|%g" % (site, a)) is None else dm_of(x, "O1|ans|%s|%g" % (site, a)) / Dgap(x)
          for x in test]
    out["dev_tuned_dmD"] = boot(vv, cl)
    out["pred_minus_dev_tuned"] = paired_boot(out["_pred_dmD"], vv, cl)


def summarize(out_dir, plan=None):
    res, metas = load_results(out_dir)
    ck = {"plan_sha256": sha256_file(plan) if plan and os.path.exists(plan) else None,
          "install": {"venv": "Parcae venv, cluster/env/requirements_parcae.txt (Python 3.12.3)", "parcae_source_commit": PARCAE_COMMIT,
                      "pypi_wheel_empty": True, "installed_equals_clone": True,
                      "parcae_py_sha16": sorted({m.get("parcae_py_sha16") for m in metas.values()}),
                      "torch": sorted({m.get("torch") for m in metas.values()})},
          "checkpoints": {k: v.split("/")[-1] for k, v in SNAP.items()},
          "endpoint": {"D_sets": "first-token margin at r (D2/D4/D5: the full answer is one token)",
                       "VB": {"primary": "full_span_logodds (e_LO)", "secondary": "first_token_margin (e)"}},
          "strict_load": {}, "depth_assert": "8 adapter calls asserted in every forward", "hook_identity": {},
          "bos_calibration": {}, "items": {}, "clusters": {}, "pc_final": {}, "sz": {}, "pc_site": {},
          "informative_pc": {}, "e1_cells": {}, "e5_cells": {}, "ad5_cells": {}, "answer_classes": {},
          "presence_validation": {}, "pr3": {}, "e8": {}, "seed_sensitivity": {}, "gpu_minutes": {},
          "scans_other_sizes": {}}
    for nm, m in metas.items():
        ck["gpu_minutes"][nm] = round((m.get("process_minutes") or 0)
                                      + sum(p.get("process_minutes") or 0 for p in m.get("previous_runs", [])), 2)
        if m.get("scan_only"):
            ck["scans_other_sizes"][m["model"]] = {"bos": m.get("bos_calibration"), "scans": m.get("scans")}
            continue
        mo = m["model"]
        la = m.get("load_audit", {})
        ck["strict_load"][mo] = not la.get("missing") and not la.get("unexpected") and not la.get("shape_mismatch")
        ck["hook_identity"][mo] = m.get("hook_identity")
        ck["bos_calibration"][mo] = m.get("bos_calibration")
    for (mo, ds), d in sorted(res.items()):
        items = d["items"]
        key = "%s|%s" % (mo, ds)
        C = metas.get(mo, {}).get("arch", {}).get("C") or (8 if "1.3b" in mo else 4)
        ck["items"][key] = dict(d["meta"].get("scan_stats", {}), target=d["meta"].get("target"),
                                skipped=d["meta"].get("skipped"), scored=len(items),
                                stopped_budget=d["meta"].get("stopped_budget", False))
        ck["clusters"][key] = {"dev_test_cluster_overlap": len({x["cluster"] for x in items if x["split"] == "dev"}
                                                               & {x["cluster"] for x in items if x["split"] == "test"}),
                               "dev_test_unit_overlap_A_or_B": unit_overlap(items),
                               "n_clusters": len(set(x["cluster"] for x in items)), "n_units": n_units(items),
                               "bootstrap": "dyadic over (A, B) subjects" if any(x.get("cluster_b") for x in items)
                               else "cluster"}
        if not items:
            continue
        vb = ds == "VB"
        pc = [x["rows"]["PCF"]["exact"] for x in items if "PCF" in x["rows"]]
        ck["pc_final"][key] = {"n": len(pc), "all_exact": all(pc), "tag": "harness"}
        sz = [x["rows"][k]["exact"] for x in items for k in ("SZ|O1", "SZ|RND") if k in x["rows"]]
        ck["sz"][key] = {"n": len(sz), "all_exact": all(sz), "tag": "harness"}
        ps = [v["exact"] for x in items for k, v in x["rows"].items() if k.startswith("PCS|")]
        ck["pc_site"][key] = {"n_rows": len(ps), "n_items": sum(1 for x in items if x.get("subset")),
                              "all_exact": all(ps), "tag": "harness"}
        ad5 = [x for x in items if x.get("ad5")]
        ipc = {}
        for name in STA_NAMES:
            ipc[name] = cell(ad5, "STA|%s" % skey(name2site(name, C)))
        ck["informative_pc"][key] = ipc
        e1 = {}
        for name in e1_names(ds):
            s = name2site(name, C)
            k = skey(s)
            j = site_loop(s)
            lab = "coda-gated" if (s == ("c", K, C - 1)) else ("last-loop answer cell" if s == ("c", K, C // 2) else None)
            ipn = info_pc_name(j)
            e1[name] = {
                "src_O1": cell(items, "O1|src|%s|1" % k, null=j <= 4, lo=vb),
                "src_RND": cell(items, "RND|src|%s" % k, rnd=True, lo=vb and name in ("L0@1", "Llast@1")),
                "ans_O1": cell(items, "O1|ans|%s|1" % k, null=j <= 4, lo=vb),
                "ans_O1_a0.25": cell(items, "O1|ans|%s|0.25" % k),
                "ans_RND": cell(items, "RND|ans|%s" % k, rnd=True, lo=vb and name in ("L0@1", "Llast@1")),
                "ans_label": lab,
                "src_informative_pc": {"site": ipn, "value": (ipc.get(ipn) or {}).get("e")}}
            if any(("O2|src|%s" % k) in x["rows"] for x in items):
                e1[name]["src_O2"] = cell(items, "O2|src|%s" % k, extra_strata=("para_fmt",))
                e1[name]["src_O2_diag"] = [mean(x.get("o2diag", {}).get("src|" + k, [None, None])[0] for x in items),
                                           mean(x.get("o2diag", {}).get("src|" + k, [None, None])[1] for x in items)]
            if any(("O2|ans|%s" % k) in x["rows"] for x in items):
                e1[name]["ans_O2"] = cell(items, "O2|ans|%s" % k, extra_strata=("para_fmt",))
                e1[name]["ans_O2_diag"] = [mean(x.get("o2diag", {}).get("ans|" + k, [None, None])[0] for x in items),
                                           mean(x.get("o2diag", {}).get("ans|" + k, [None, None])[1] for x in items)]
        ck["e1_cells"][key] = e1
        e5 = {}
        for j in range(1, K + 1):
            e5["IN-single(%d)" % j] = cell(items, "IN1|%d" % j, null=j <= 6)
        for j in range(1, K):
            e5["IN-persist(%d)" % j] = cell(items, "INP|%d" % j)
        e5["VE-persist"] = cell(items, "VEP")
        e5["VE-core"] = cell(items, "VEC")
        e5["VE-coda"] = cell(items, "VECD")
        e5["INVE-persist"] = cell(items, "INVEP")
        k7, k8 = "O1|src|%s|1" % skey(("c", K - 1, C - 1)), "O1|src|%s|1" % skey(("p", K))
        e5["post_inj(8)_minus_Llast@7_src"] = paired_boot([e_of(x, k8) for x in items], [e_of(x, k7) for x in items],
                                                          [cl_of(x) for x in items])
        ck["e5_cells"][key] = e5
        ck["ad5_cells"][key] = {"n_items": len(ad5),
                                "STA@post_inj(8)+VE-coda": cell(ad5, "STAVC|p%d" % K),
                                "VE-core_same_items": cell(ad5, "VEC"),
                                "VE-coda_same_items": cell(ad5, "VECD"),
                                "IN-persist-all(1)": cell(ad5, "INPA|1"),
                                "IN-persist-all(7)": cell(ad5, "INPA|7"),
                                "IN-persist-down(1)": cell(ad5, "INPD|1"),
                                "IN-persist(1)_same_items": cell(ad5, "INP|1"),
                                "STA": ipc}
        a5c = ck["ad5_cells"][key]
        ch = {"VE-coda": a5c["VE-coda_same_items"], "VE-loop": a5c["VE-core_same_items"],
              "e re-injection": a5c["IN-persist-all(1)"]}
        for name, cc in e1.items():
            cc["src_attribution"] = attribute((cc.get("src_O1") or {}).get("e"), cc["src_informative_pc"]["value"],
                                              {n: (c or {}).get("e") for n, c in ch.items()})
        pv, cls = {}, {}
        for name in e1_names(ds):
            s = name2site(name, C)
            k = skey(s)
            ps = [x["pval"][k] for x in items if x.get("pval") and k in x["pval"]]
            ok = [p[1] - p[0] >= 0.5 for p in ps if p[0] is not None and p[1] is not None]
            gate = (sum(ok) / len(ok) >= 0.5) if len(ok) >= 10 else None
            pv[name] = {"n_items_with_form": len(ps), "n_valid": len(ok), "n_pass": sum(ok), "gate": gate}
            for a in A_VERIFY:
                rk = "O1|ans|%s|%g" % (k, a)
                ef = mean(e_of(x, rk) for x in items)
                if s == ("c", K, C - 1):
                    cls["%s|a=%g" % (name, a)] = {"e_final": ef, "class": "coda-gated"}
                elif s[0] == "c" and s[2] == C - 1:
                    eo = mean(eown_of(x, rk) for x in items)
                    cls["%s|a=%g" % (name, a)] = {"e_final": ef, "e_own": eo, "class": answer_class(a, ef, e_own=eo)}
                else:
                    cls["%s|a=%g" % (name, a)] = {"e_final": ef, "presence_gate": gate,
                                                  "class": answer_class(a, ef, gate=gate)}
        ck["presence_validation"][key] = pv
        ck["answer_classes"][key] = cls
        if ds == "D1":
            k1 = "O1|src|%s|1" % skey(("c", 1, C - 1))
            m2 = [x for x in items if x["strata"]["pos_len"] == ">=2"]
            testable = len(m2) >= 20 and n_units(m2) >= 10
            ck["pr3"][mo] = {"cell": "src O1 Llast@1", "n_ge2": len(m2), "n_clusters_ge2": n_units(m2),
                             "n_1": sum(1 for x in items if x["strata"]["pos_len"] == "1"), "testable": testable,
                             "tost": diff_tost([e_of(x, k1) for x in items], [cl_of(x) for x in items],
                                               [x["strata"]["pos_len"] for x in items], ">=2", "1") if testable else None,
                             "note": "pos_len >= 2 coincides with the city relation on the Parcae tokenizer (plan DV5)"}
            if d["meta"].get("seed_sensitivity"):
                ss = d["meta"]["seed_sensitivity"]
                k7 = "O1|src|%s|1" % skey(("c", 7, C - 1))
                e0 = {x["pair_id"]: e_of(x, k7) for x in items}
                summ = {}
                for off, v in ss["offsets"].items():
                    common = [p for p in v["e"] if e0.get(p) is not None and v["e"][p] is not None]
                    summ[off] = {"kept": v["kept"], "kept_off0": v["kept_off0"], "sym_diff": v["sym_diff"],
                                 "cell_e_off": mean(v["e"][p] for p in common), "cell_e_off0": mean(e0[p] for p in common),
                                 "max_abs_item_diff": max([abs(v["e"][p] - e0[p]) for p in common] or [None])}
                ck["seed_sensitivity"][mo] = summ
        e8 = {}
        for set_name in ("S", "Sp"):
            for a in A_VERIFY:
                o = summarize_e8(items, C, a, set_name)
                add_dev_tuned(o, items, d, C, a, set_name)
                o.pop("_pred_dmD", None)
                e8["%s|a=%g" % (set_name, a)] = o
        ck["e8"][key] = e8
    ck["predictions_vs_outcomes"] = predictions(ck)
    atomic_json(os.path.join(out_dir, "checks.json"), ck)
    write_summary_md(out_dir, ck)
    return ck


def _e(ck, key, name, field):
    c = ck["e1_cells"].get(key, {}).get(name, {}).get(field)
    return None if not c else c["e"][0]


def predictions(ck):
    """PR1-PR10 (plan section 11) on parcae-1.3b unless stated; None where the cell is missing."""
    out = {}
    m = "parcae-1.3b"
    d1 = "%s|D1" % m

    def chk(cond):
        return None if cond is None else bool(cond)
    vals1 = {}
    for mo in ("parcae-1.3b", "parcae-370m"):
        for ds in DATASETS:
            k = "%s|%s" % (mo, ds)
            if k in ck["e1_cells"] and (ck["items"].get(k, {}).get("scored", 0) >= MIN_KEPT):
                for j in (1, 2, 4):
                    vals1["%s Llast@%d" % (k, j)] = _e(ck, k, "Llast@%d" % j, "src_O1")
    pr1a = [v for v in vals1.values() if v is not None]
    out["PR1"] = {"src Llast@<=4 <= 0.10 (all models, datasets >= 40)": chk(pr1a and all(v <= 0.10 for v in pr1a)) if pr1a else None,
                  "src Llast@7 <= 0.25 (1.3b D1)": chk(_e(ck, d1, "Llast@7", "src_O1") is not None and _e(ck, d1, "Llast@7", "src_O1") <= 0.25),
                  "src post_inj@8, L0@8, Lmid@8 >= 0.4 (1.3b D1)": chk(all(_e(ck, d1, n, "src_O1") is not None and _e(ck, d1, n, "src_O1") >= 0.4
                                                                         for n in ("post_inj@8", "L0@8", "Lmid@8"))),
                  "src Llast@8 >= 0.3 (1.3b D1)": chk(_e(ck, d1, "Llast@8", "src_O1") is not None and _e(ck, d1, "Llast@8", "src_O1") >= 0.3),
                  "values": vals1}
    lm = _e(ck, d1, "Lmid@8", "ans_O1")
    cls = ck["answer_classes"].get(d1, {})
    out["PR2"] = {"ans Lmid@8 in [0.3, 0.9]": chk(lm is not None and 0.3 <= lm <= 0.9),
                  "ans Llast@8 [coda-gated] >= 0.6": chk(_e(ck, d1, "Llast@8", "ans_O1") is not None and _e(ck, d1, "Llast@8", "ans_O1") >= 0.6),
                  "ans Llast@7 not carried": chk(cls.get("Llast@7|a=1", {}).get("class") == "not carried") if cls else None,
                  # review round 2 F6: every D1 answer cell with loop <= 4 (12 cells)
                  "ans loops <= 4 <= 0.10": chk(all(_e(ck, d1, n, "ans_O1") is not None and _e(ck, d1, n, "ans_O1") <= 0.10
                                                    for n in e1_names("D1") if site_loop(name2site(n, 8)) <= 4)),
                  "ans Llast@1 not carried": chk(cls.get("Llast@1|a=1", {}).get("class") == "not carried") if cls else None}
    p3 = ck["pr3"].get(m)
    out["PR3"] = {"testable": p3.get("testable") if p3 else None,
                  "equivalent within 0.15": (p3["tost"]["equivalent"] if p3 and p3.get("tost") else None)}
    e5 = ck["e5_cells"].get(d1, {})

    def g5(n):
        c = e5.get(n)
        return None if not c else c["e"][0]
    ins = [g5("IN-single(%d)" % j) for j in range(1, 7)]
    out["PR4"] = {"IN-single(j<=6) <= 0.10": chk(all(v is not None and v <= 0.10 for v in ins)) if e5 else None,
                  "IN-single(7) <= 0.25": chk(g5("IN-single(7)") is not None and g5("IN-single(7)") <= 0.25),
                  "IN-single(8) >= 0.3": chk(g5("IN-single(8)") is not None and g5("IN-single(8)") >= 0.3),
                  "IN-persist flat (|1-7| <= 0.15)": chk(g5("IN-persist(1)") is not None and g5("IN-persist(7)") is not None
                                                         and abs(g5("IN-persist(1)") - g5("IN-persist(7)")) <= 0.15),
                  "IN-persist(1) >= 0.5": chk(g5("IN-persist(1)") is not None and g5("IN-persist(1)") >= 0.5),
                  "INVE - IN-persist(1) in [0.1, 0.3]": chk(g5("INVE-persist") is not None and g5("IN-persist(1)") is not None
                                                            and 0.1 <= g5("INVE-persist") - g5("IN-persist(1)") <= 0.3),
                  "VE-persist in [0.1, 0.4]": chk(g5("VE-persist") is not None and 0.1 <= g5("VE-persist") <= 0.4)}
    pd_ = e5.get("post_inj(8)_minus_Llast@7_src")
    out["PR5"] = {"post_inj(8) - Llast@7 >= 0.3": chk(pd_ is not None and pd_[0] >= 0.3)}
    pr6 = {}
    for ds in ("D1", "D2"):
        e8 = ck["e8"].get("%s|%s" % (m, ds), {})
        s1, s25, sp = e8.get("S|a=1"), e8.get("S|a=0.25"), e8.get("Sp|a=1")
        if not s1:
            continue
        pr6[ds] = {"top in loop 8 or late 7 >= 80%": chk(s1["n_items"] and s1["top_in_loop8_or_late7"] / s1["n_items"] >= 0.8),
                   "hit >= 3x chance": chk(s1["top1_hit"] is not None and s1["chance"] and s1["top1_hit"] >= 3 * s1["chance"]),
                   "fn_rate_dev <= 0.10": chk(s1["fn_rate_dev"]["fn_rate"] is not None and s1["fn_rate_dev"]["fn_rate"] <= 0.10),
                   "spearman median >= 0.5": chk(s1["spearman_median"] is not None and s1["spearman_median"] >= 0.5),
                   "hit(0.25) >= hit(1) - 0.15": chk(s25 and s1["top1_hit"] is not None and s25["top1_hit"] is not None
                                                     and s25["top1_hit"] >= s1["top1_hit"] - 0.15),
                   "|pred - dev_tuned| <= 0.10 D": chk(s1.get("pred_minus_dev_tuned") is not None and abs(s1["pred_minus_dev_tuned"][0]) <= 0.10),
                   "S+: top = Llast@8 on >= 50%": chk(sp and sp["n_items"] and sp["top_is_Llast8"] / sp["n_items"] >= 0.5),
                   "interpreted": s1["interpreted"]}
    out["PR6"] = pr6
    ss = ck["seed_sensitivity"].get(m)
    out["PR7"] = {"clean count within 2": chk(ss and all(abs(v["kept"] - v["kept_off0"]) <= 2 for v in ss.values())) if ss else None,
                  "cell within 0.02": chk(ss and all(v["cell_e_off"] is not None and abs(v["cell_e_off"] - v["cell_e_off0"]) <= 0.02
                                                     for v in ss.values())) if ss else None}
    rnd_ok = [c.get("T3_pass") for k, e1 in ck["e1_cells"].items() for n, cc in e1.items()
              for f, c in cc.items() if f.endswith("RND") and c]
    out["PR8"] = {"RND passes T3 at every cell": chk(rnd_ok and all(rnd_ok)) if rnd_ok else None,
                  "PC-final, SZ, PC-site exact [harness]": chk(all(v["all_exact"] for d_ in (ck["pc_final"], ck["sz"], ck["pc_site"])
                                                                   for v in d_.values()))}
    it = ck["items"]

    def kept(k):
        return it.get(k, {}).get("kept")

    def ge(k, n):
        v = kept(k)
        return None if v is None else bool(v >= n)

    def lt(k, n):
        v = kept(k)
        return None if v is None else bool(v < n)
    p3s = ck["pr3"].get("parcae-370m")
    sc140 = (ck["scans_other_sizes"].get("parcae-140m") or {}).get("scans") or {}
    out["PR9"] = {"D1 1.3b >= 40": ge("parcae-1.3b|D1", 40), "D1 370m >= 40": ge("parcae-370m|D1", 40),
                  "D1 370m >=2 stratum < 20": None if not p3s else bool(p3s["n_ge2"] < 20),
                  "D2 1.3b >= 40": ge("parcae-1.3b|D2", 40), "D2 370m < 40": lt("parcae-370m|D2", 40),
                  "D4 >= 40 both": None if kept("parcae-1.3b|D4") is None or kept("parcae-370m|D4") is None
                  else bool(kept("parcae-1.3b|D4") >= 40 and kept("parcae-370m|D4") >= 40),
                  "D5 370m < 40": lt("parcae-370m|D5", 40), "VB 1.3b >= 40": ge("parcae-1.3b|VB", 40),
                  "D1 140m < 40": None if "D1" not in sc140 else bool(sc140["D1"]["kept"] < 40),
                  "values": {k: kept(k) for k in it}}
    a5 = ck["ad5_cells"].get(d1, {})

    def g10(n):
        c = a5.get(n)
        return None if not c else c["e"][0]
    sta = a5.get("STA", {}) if a5 else {}

    def gs(n):
        c = sta.get(n)
        return None if not c else c["e"][0]
    out["PR10"] = {"STA@Llast@1, @4 <= 0.10": chk(gs("Llast@1") is not None and gs("Llast@4") is not None
                                                   and gs("Llast@1") <= 0.10 and gs("Llast@4") <= 0.10),
                   "STA@Llast@7 in [0.05, 0.40]": chk(gs("Llast@7") is not None and 0.05 <= gs("Llast@7") <= 0.40),
                   "STA@post_inj(8) >= 0.8": chk(gs("post_inj@8") is not None and gs("post_inj@8") >= 0.8),
                   "IN-persist-all(1) >= 0.8": chk(g10("IN-persist-all(1)") is not None and g10("IN-persist-all(1)") >= 0.8),
                   "IN-persist-all(7) >= 0.6": chk(g10("IN-persist-all(7)") is not None and g10("IN-persist-all(7)") >= 0.6),
                   "IN-persist-down(1) >= 0.5 and >= IN-persist(1)": chk(g10("IN-persist-down(1)") is not None
                                                                         and g10("IN-persist(1)_same_items") is not None
                                                                         and g10("IN-persist-down(1)") >= 0.5
                                                                         and g10("IN-persist-down(1)") >= g10("IN-persist(1)_same_items"))}
    # PR11 (review round 2 F1; written after the step-2 debug rows were seen, so tagged post-debug, not frozen
    # before data): value-embed channel split, per dataset with >= 40 scored items, both models
    pr11 = {}
    for k, e5_ in ck["e5_cells"].items():
        if ck["items"].get(k, {}).get("scored", 0) < MIN_KEPT:
            continue
        a5_ = ck["ad5_cells"].get(k, {})

        def gv(c):
            return None if not c else c["e"][0]
        vep, vec, vecd = gv(e5_.get("VE-persist")), gv(e5_.get("VE-core")), gv(e5_.get("VE-coda"))
        stavc, sta8 = gv(a5_.get("STA@post_inj(8)+VE-coda")), gv((a5_.get("STA") or {}).get("post_inj@8"))
        pr11[k] = {"VE-coda >= VE-core": chk(vecd is not None and vec is not None and vecd >= vec),
                   "|VE-persist - max(VE-core, VE-coda)| <= 0.15": chk(None not in (vep, vec, vecd)
                                                                        and abs(vep - max(vec, vecd)) <= 0.15),
                   "STA@post_inj(8)+VE-coda >= 0.8": chk(stavc is not None and stavc >= 0.8),
                   "STA@post_inj(8)+VE-coda - STA@post_inj(8) >= 0.3": chk(stavc is not None and sta8 is not None
                                                                            and stavc - sta8 >= 0.3),
                   "values": {"VE-persist": vep, "VE-core": vec, "VE-coda": vecd, "STA@post_inj(8)": sta8,
                              "STA@post_inj(8)+VE-coda": stavc}}
    out["PR11"] = {"tag": "post-debug (written after the step-2 debug rows, before any scored run)", "cells": pr11}
    out["harness_tags"] = ["PR8: PC-final, SZ, PC-site"]
    return out


def write_summary_md(out_dir, ck):
    L = ["# K17 Parcae summary (generated by k17_parcae.py --summarize)", ""]
    L.append("Plan sha256: %s" % ck.get("plan_sha256"))
    L.append("")
    L.append("## Items")
    for k, v in sorted(ck["items"].items()):
        L.append("- %s: scanned %s, A_fail %s, B_fail %s, D_small %s, kept %s, scored %s, clusters %s, skipped %s, "
                 "pos_len x rel %s" % (k, v.get("scanned"), v.get("A_fail"), v.get("B_fail"), v.get("D_small"),
                                      v.get("kept"), v.get("scored"), v.get("n_clusters"), v.get("skipped"),
                                      v.get("pos_len_x_rel")))
    L.append("")
    L.append("## Harness checks")
    for nm in ("pc_final", "sz", "pc_site"):
        for k, v in sorted(ck[nm].items()):
            L.append("- %s %s: %s" % (nm, k, json.dumps(v)))
    for k, v in sorted(ck["hook_identity"].items()):
        L.append("- hook_identity %s: all_ok %s" % (k, (v or {}).get("all_ok")))
    L.append("")
    L.append("## E1 (e mean [95% CI], n items / clusters)")
    for k, e1 in sorted(ck["e1_cells"].items()):
        L.append("### %s" % k)
        L.append("| site | src O1 | src RND | ans O1 | ans O1 a=0.25 | ans RND | label | src informative PC | src reading |")
        L.append("|---|---|---|---|---|---|---|---|---|")

        def f(c):
            return "-" if not c else "%s [%s, %s] (%d/%d)" % (c["e"][0], c["e"][1], c["e"][2], c["n_items"], c["n_clusters"])
        for name, cc in e1.items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s=%s | %s |" % (
                name, f(cc.get("src_O1")), f(cc.get("src_RND")), f(cc.get("ans_O1")), f(cc.get("ans_O1_a0.25")),
                f(cc.get("ans_RND")), cc.get("ans_label") or "", cc["src_informative_pc"]["site"],
                cc["src_informative_pc"]["value"], cc.get("src_attribution")))
    L.append("")
    L.append("## E5 and AD5")
    for k, e5 in sorted(ck["e5_cells"].items()):
        L.append("- %s: %s" % (k, json.dumps({n: (c["e"] if isinstance(c, dict) else c) for n, c in e5.items() if c})))
    for k, a5 in sorted(ck["ad5_cells"].items()):
        L.append("- AD5 %s: %s" % (k, json.dumps({n: (c["e"] if isinstance(c, dict) and "e" in c else None)
                                                   for n, c in a5.items() if n not in ("STA", "n_items")})))
    L.append("")
    L.append("## E8")
    for k, e8 in sorted(ck["e8"].items()):
        for sk, o in e8.items():
            L.append("- %s %s: sites %s, eligible %s (test %s, interpreted %s), hit %s, chance %s, regret %s, spearman %s "
                     "(n %s), pred %s, uniform %s, dev-tuned %s (%s), random %s, FN dev %s, cost %s" % (
                         k, sk, o["n_sites"], o["eligible"], o["eligible_test"], o["interpreted"], o["top1_hit"],
                         o["chance"], o["regret"], o["spearman_median"], o["spearman_n"], o.get("pred_dmD"),
                         o.get("uniform_dmD"), o.get("dev_tuned_dmD"), o.get("dev_tuned_site"), o.get("random_dmD"),
                         o["fn_rate_dev"], o["cost_fwd_equiv"]))
    L.append("")
    L.append("## Predictions vs outcomes")
    L.append("```")
    L.append(json.dumps(ck["predictions_vs_outcomes"], indent=1)[:20000])
    L.append("```")
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ self-tests
def selftest():
    res = {}
    torch.manual_seed(0)
    o = torch.randn(1, 6, 16)
    t = torch.randn(2, 16)
    e = apply_entries(o, [("set", torch.tensor([1, 4]), t, 1.0)])
    res["set_a1_exact"] = bool(torch.equal(e[0, [1, 4]], t) and torch.equal(e[0, [0, 2, 3, 5]], o[0, [0, 2, 3, 5]]))
    e2 = apply_entries(o, [("set", torch.tensor([2]), o[0, [2]] + 1.0, 0.25)])
    res["set_a025"] = abs(float((e2[0, 2] - o[0, 2]).mean()) - 0.25) < 1e-5
    res["add_zero_identity"] = bool(torch.equal(apply_entries(o, [("add", torch.arange(6), torch.zeros(6, 16), 1.0)]), o))
    res["input_not_modified"] = bool(torch.equal(o, o.clone())) and not torch.equal(e, o)
    d = rnd_delta("p", "c1.3", "src", [5, 6], [2.0, 3.0], 16)
    res["rnd_norms"] = [r5(x) for x in d.norm(dim=-1)]
    res["rnd_deterministic"] = bool(torch.equal(d, rnd_delta("p", "c1.3", "src", [5, 6], [2.0, 3.0], 16)))
    res["rnd_differs_by_site"] = not torch.equal(d, rnd_delta("p", "c1.4", "src", [5, 6], [2.0, 3.0], 16))
    mA, mB = torch.zeros(8), torch.ones(8)
    res["proj"] = [proj(mA, mA, mB), proj(mB, mA, mB), proj((mA + mB) / 2, mA, mB), proj(mA + 1, mA + 1, mA + 1)]
    b = boot([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"])
    res["boot_ok"] = abs(b[0] - 4.0) < 1e-9 and b[3] == 4 and b[4] == 3
    tz = tost([0.01, -0.02, 0.0, 0.015, -0.01] * 8, list(range(40)))
    tb = tost([0.2, 0.1, 0.15, 0.12, 0.18] * 8, list(range(40)))
    res["tost"] = [tz["null_by_tost"], tb["null_by_tost"]]
    xs = [0.5, 0.52, 0.48, 0.51] * 10 + [0.49, 0.5, 0.53, 0.47] * 10
    grp = [">=2"] * 40 + ["1"] * 40
    cl = ["c%d" % (i % 20) for i in range(80)]
    dq = diff_tost(xs, cl, grp, ">=2", "1")
    xs2 = [0.8] * 40 + [0.2] * 40
    dn = diff_tost(xs2, cl, grp, ">=2", "1")
    res["diff_tost"] = [dq["equivalent"], dn["equivalent"], dn["diff"]]
    pb = paired_boot([1.0, 2.0, 3.0], [0.5, 1.0, None], ["a", "b", "c"])
    res["paired_boot"] = pb[:1] + pb[3:]
    # review round 2 F2: dyadic bootstrap and dyadic stratum difference on (A, B) subject tuples
    b2 = boot([1.0, 2.0, 3.0, 4.0], [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    res["boot2"] = b2
    xd = [0.5 + 0.01 * (i % 3) for i in range(60)]
    cld = [("s%d" % (i % 15), "s%d" % ((i + 7) % 15)) for i in range(60)]
    grd = [">=2" if i % 2 else "1" for i in range(60)]
    dd = diff_tost(xd, cld, grd, ">=2", "1")
    dd2 = diff_tost([0.9 if g == ">=2" else 0.1 for g in grd], cld, grd, ">=2", "1")
    res["diff_tost_dyadic"] = [dd["equivalent"], dd["dyadic"], dd["n_clusters"], dd2["equivalent"], dd2["diff"]]
    xi = [{"cluster": "city:Paris", "cluster_b": "city:Rome", "split": "dev"},
          {"cluster": "city:Rome", "cluster_b": "city:Oslo", "split": "test"}, {"cluster": "k", "split": "dev"}]
    res["cl_of"] = [cl_of(xi[0]), cl_of(xi[2]), n_units(xi), unit_overlap(xi)]
    res["dyadic_ok"] = (b2[3] == 4 and b2[4] == 4 and abs(b2[0] - 2.5) < 1e-9 and res["diff_tost_dyadic"][:3] == [True, True, 15]
                        and res["diff_tost_dyadic"][3] is False and abs(res["diff_tost_dyadic"][4] - 0.8) < 1e-9
                        and res["cl_of"] == [("city:Paris", "city:Rome"), "k", 4, 1])
    # review round 2 F1: reading rule v3
    res["attribute"] = [attribute([0.05], [0.7], {}), attribute([0.05], [0.05], {"VE-coda": [0.8], "VE-loop": [0.1]}),
                        attribute([0.05], [0.05], {"VE-coda": [0.2], "VE-loop": [0.6], "e re-injection": [0.55]}),
                        attribute([0.05], [0.05], {"VE-coda": [0.2]}), attribute([0.05], [0.3], {}),
                        attribute([0.4], [0.05], {}), attribute([0.05], None, {})]
    res["attribute_ok"] = res["attribute"] == [
        "decay at the source", "input channel: VE-coda", "input channel: VE-loop, e re-injection",
        "unattributed (no single input channel >= 0.5)", "control partial (0.15 < PC < 0.5): not attributed",
        "not a null", "no informative control"]
    # review round 2 F7: default strata and item-level para_fmt stratum
    its = [{"ds": "D1", "cluster": "c%d" % i, "cluster_b": "d%d" % i, "split": "test", "tB": 2,
            "strata": {"pos_len": "1" if i % 2 else ">=2", "rel": "city" if i % 2 else "capital"},
            "para_fmt": "para%d" % (i % 3), "clean_A": [0.0, -3.0, 0, 1], "clean_B": [0.0, 3.0, 0, 2],
            "rows": {"X": {"f": [0.0, 0.0, 0, 1]}}} for i in range(12)]
    cc_ = cell(its, "X", extra_strata=("para_fmt",))
    res["cell_strata"] = sorted(cc_["strata"]) if cc_ else None
    res["cell_strata_ok"] = (res["cell_strata"] == ["para_fmt", "pos_len", "rel"] and cc_["n_clusters"] == 24
                             and abs(cc_["e"][0] - 0.5) < 1e-9)
    res["spearman"] = [spearman([1, 2, 3, 4], [10, 20, 30, 40]), spearman([1, 2, 3, 4], [4, 3, 2, 1])]
    G = {"k%02d" % i: float(i) for i in range(63)}
    ks, ver = verify_sites(G)
    res["verify_n63"] = ver
    res["verify_ok"] = ver == ["k62", "k61", "k60", "k32", "k31", "k30", "k02", "k01", "k00"]
    Gv = {k: G[k] for k in ver}
    h = hit_metrics(Gv, dict(Gv), 10.0, 2.0)
    dmt = dict(Gv)
    dmt["k61"] = dmt["k62"]
    ht = hit_metrics(Gv, dmt, 10.0, 2.0)
    hz = hit_metrics(Gv, {k: 0.001 for k in ver}, 10.0, 2.0)
    res["hit_ok"] = (h["hit"] == 1 and abs(h["chance"] - 1 / 9) < 1e-9 and h["regret"] == 0.0 and ht["hit"] == 0
                     and ht["chance"] == 0.0 and hz["eligible"] is False and hz["hit"] is None)
    fr = fn_rate([(0.01, 3.0, 10.0), (0.01, 0.1, 10.0), (5.0, 3.0, 10.0), (0.2, 0.0, 10.0)])
    res["fn_rate"] = fr
    res["fn_rate_ok"] = fr["n_low_G"] == 3 and abs(fr["fn_rate"] - 1 / 3) < 1e-3
    res["classes"] = [answer_class(1.0, 0.1, e_own=0.8), answer_class(1.0, 0.1, e_own=0.2), answer_class(1.0, 0.7, e_own=0.9),
                      answer_class(0.25, 0.05, e_own=0.2), answer_class(0.25, 0.05, e_own=0.1),
                      answer_class(1.0, 0.1, gate=True), answer_class(1.0, 0.1, gate=False), answer_class(1.0, 0.1, gate=None),
                      answer_class(1.0, 0.3, e_own=0.8)]
    res["classes_ok"] = res["classes"] == ["not carried", "no answer state at site", "carried", "not carried",
                                           "no answer state at site", "not carried", "no answer state at site",
                                           "unvalidated", "partial"]
    res["info_pc"] = [info_pc_name(j) for j in (1, 2, 4, 7, 8)]
    res["info_pc_ok"] = res["info_pc"] == ["Llast@1", "Llast@1", "Llast@4", "Llast@7", "post_inj@8"]
    sites = [name2site(n, 8) for n in e1_names("D1")]
    res["e1_counts"] = [len(e1_names("D1")), len(e1_names("VB"))]
    res["sites_ok"] = (res["e1_counts"] == [20, 13] and all(ksite(skey(s)) == s for s in sites + [("k", 7)])
                       and name2site("Lmid@8", 8) == ("c", 8, 4) and name2site("Llast@7", 4) == ("c", 7, 3)
                       and "Lmid@8" in e1_names("VB") and "Lmid@4" not in e1_names("VB"))
    allS = [("p", j) for j in range(1, K + 1)] + [("c", j, l) for j in range(1, K + 1) for l in range(8)] + [("k", 0)]
    ords = [exec_order(s, 8) for s in sorted(allS, key=lambda s: (s[1] if s[0] != "k" else 99,
                                                                   -1 if s[0] == "p" else (s[2] if s[0] == "c" else 0)))]
    res["exec_order_monotone"] = ords == sorted(ords) and len(set(ords)) == len(ords)
    S8, Sp8 = e8_sets(8)
    S4, Sp4 = e8_sets(4)
    res["e8_sets"] = [len(S8), len(Sp8), len(S4), len(Sp4), ("c", 8, 7) in S8, ("c", 8, 7) in Sp8, ("c", 8, 4) in S8]
    res["e8_sets_ok"] = res["e8_sets"] == [63, 64, 31, 32, False, True, True]
    res["seed"] = [seed_of("x"), seed_of("x", 1) - seed_of("x"), seed_of("x") == seed_of("x")]
    res["seed_ok"] = res["seed"][1] == 1 and res["seed"][2]
    res["trunc_std"] = round(TRUNC_STD, 5)
    # selection rules
    c1 = [{"pair_id": "m%d" % i, "strata": {"pos_len": ">=2"}, "sel": {"subjects": ["s%d" % i, "t%d" % i]}} for i in range(50)]
    c1 += [{"pair_id": "o%d" % i, "strata": {"pos_len": "1"}, "sel": {"subjects": ["u%d" % i, "v%d" % i]}} for i in range(5)]
    c1 += [{"pair_id": "w%d" % i, "strata": {"pos_len": "1"}, "sel": {"subjects": ["w", "y%d" % i]}} for i in range(3)]
    s = Selector("D1", target=80)
    kept = []
    for c in c1:
        if s.want(c):
            s.accept(c)
            kept.append(c["pair_id"])
    res["sel_d1"] = [len([k for k in kept if k[0] == "m"]), len([k for k in kept if k[0] == "o"]),
                     len([k for k in kept if k[0] == "w"])]
    s = Selector("D2", target=60)
    c2 = [{"rel": "op1" if i % 2 else "op2", "cluster": "x=%d" % (i % 3)} for i in range(20)]
    kept2 = []
    for c in c2:
        if s.want(c):
            s.accept(c)
            kept2.append(c["cluster"])
    res["sel_d2_cluster_cap"] = max(kept2.count(x) for x in set(kept2))
    s = Selector("D5", target=60)
    c5 = [{"split": "dev", "sel": {"problems": ["a", "b"]}}, {"split": "dev", "sel": {"problems": ["b", "c"]}},
          {"split": "test", "sel": {"problems": ["d", "e"]}}]
    kept5 = []
    for i, c in enumerate(c5):
        if s.want(c):
            s.accept(c)
            kept5.append(i)
    res["sel_d5"] = kept5
    res["sel_ok"] = res["sel_d1"] == [40, 5, 2] and res["sel_d2_cluster_cap"] == 2 and res["sel_d5"] == [0, 2]
    s = Selector("D1", target=3)
    res["sel_target"] = [s.want(c1[0])] + [s.accept(c1[i]) or s.done() for i in range(3)]
    cand = {"ds": "D1", "pair_id": "p", "A": [5, 6, 7], "B": [5, 8, 7], "r": 2, "src": [1], "tA": 1, "tB": 2,
            "rel": "capital", "cluster": "c", "split": "dev", "strata": {"pos_len": "1"}, "label": "x",
            "para": [{"fmt": "f", "A": [9, 5, 6, 7], "B": [9, 5, 8, 7], "src": [2], "r": 3},
                     {"fmt": "g", "A": [6, 7], "B": [8, 7], "src": [0], "r": 1}]}
    p0, pn = prep_item(cand, 0), prep_item(cand, None)
    c0 = dict(cand, src=[0])
    res["prep"] = [p0["A"], p0["src"], p0["r"], len(p0["para"]), pn["A"], len(pn["para"]), prep_item(c0, None), p0["Tmax"]]
    res["prep_ok"] = (p0["A"] == [0, 5, 6, 7] and p0["src"] == [2] and p0["r"] == 3 and len(p0["para"]) == 2
                      and pn["A"] == [5, 6, 7] and len(pn["para"]) == 1 and prep_item(c0, None) is None and p0["Tmax"] == 5)
    rws = [{"kind": "T1", "names": ["x", "y"], "src_var": "y", "query_vars": ["x"], "vals_A": a_, "vals_B": b_}
           for a_, b_ in (([1, 2], [1, 3]), ([1, 3], [1, 4]), ([5, 6], [5, 7]))]
    cmp_ = d4_components(rws)
    res["d4_comp"] = len(set(cmp_.values()))
    res["d4_comp_ok"] = res["d4_comp"] == 2 and cmp_[d4_content(rws[0], "A")] == cmp_[d4_content(rws[1], "B")]
    try:
        import k17_varbench as VB
        res["varbench_parity"] = {
            "boot": VB.boot([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"], n=NBOOT) == boot([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"]),
            "spearman": VB.spearman([1, 3, 2, 5], [2, 1, 4, 3]) == spearman([1, 3, 2, 5], [2, 1, 4, 3]),
            "verify": VB.verify_sites(G) == verify_sites(G),
            "hit_v3": VB.hit_metrics(Gv, dmt, 10.0, 2.0) == hit_metrics_v3(Gv, dmt, 10.0, 2.0),
            "strict_hit": VB.hit_metrics(Gv, dmt, 10.0, 2.0)["strict_hit"] == hit_metrics(Gv, dmt, 10.0, 2.0)["hit"],
            "proj": VB.proj(mA + 0.3, mA, mB) == proj(mA + 0.3, mA, mB),
            "boot2": VB.boot2([1.0, 2.0, 3.0, 4.0], [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]) == b2,
            "d4_content": all(VB.d4_content(r_, sd) == d4_content(r_, sd) for sd in ("A", "B")
                              for r_ in [{"kind": "T1", "names": ["x", "y"], "src_var": "y", "query_vars": ["x"],
                                          "vals_A": [3, 4], "vals_B": [3, 7]}]),
            "rr": [x["i"] for x in VB.rr_order([{"g": "a", "i": 0}, {"g": "a", "i": 1}, {"g": "b", "i": 2}], lambda x: x["g"])]
            == [x["i"] for x in rr_order([{"g": "a", "i": 0}, {"g": "a", "i": 1}, {"g": "b", "i": 2}], lambda x: x["g"])]}
        res["varbench_parity_ok"] = all(res["varbench_parity"].values())
    except Exception as ex:  # noqa: BLE001  (the parity check is optional off the laptop)
        res["varbench_parity"] = "k17_varbench not importable here: %s" % type(ex).__name__
        res["varbench_parity_ok"] = True
    ok = all(res[k] for k in ("set_a1_exact", "set_a025", "add_zero_identity", "rnd_deterministic", "rnd_differs_by_site",
                              "boot_ok", "verify_ok", "hit_ok", "fn_rate_ok", "classes_ok", "info_pc_ok", "sites_ok",
                              "exec_order_monotone", "e8_sets_ok", "seed_ok", "sel_ok", "prep_ok", "varbench_parity_ok",
                              "dyadic_ok", "attribute_ok", "cell_strata_ok", "d4_comp_ok"))
    ok = ok and all(abs(a - b) < 1e-4 for a, b in zip(res["rnd_norms"], [2.0, 3.0]))
    ok = ok and res["proj"][:3] == [0.0, 1.0, 0.5] and res["proj"][3] is None
    ok = ok and res["tost"] == [True, False] and res["spearman"] == [1.0, -1.0]
    ok = ok and res["diff_tost"][0] is True and res["diff_tost"][1] is False and abs(res["diff_tost"][2] - 0.6) < 1e-9
    ok = ok and res["paired_boot"] == [0.75, 2, 2] and abs(res["trunc_std"] - 0.98658) < 1e-4
    ok = ok and res["sel_target"] == [True, False, False, True]
    res["ok"] = bool(ok)
    print(json.dumps(res, default=str))
    return ok


def tiny_model(seed=0):
    """Random-weight Parcae from the installed source (370m config shape, small dims) for CPU hook tests."""
    from parcae_lm.models.parcae.config import ParcaeConfig
    cfg = ParcaeConfig(name="tiny", block_size=64, n_embd=64, intermediate_size=128, num_attention_heads=4,
                       num_key_value_heads=4, vocab_size=128, padding_multiple=64, padded_vocab_size=128,
                       block_class_name="TransformerPreNormBlock", norm_class_name="RMSNorm", attn_impl="flash",
                       norm_eps=1e-5, mlp_class_name="BaseMLP", nonlin_name="ReLU2", bias=False, init_strategy="scaled-zero",
                       init_orthogonal=True, use_fused_head="pytorch", qk_norm=True, injection_type="diagonal",
                       n_layers_in_recurrent_block=4, n_layers_in_prelude=2, n_layers_in_coda=2, state_init="like-init",
                       recurrent_embedding_dimension=64, recurrent_intermediation_embedding_dimension=128, prelude_norm=True,
                       sampling_scheme="poisson-truncated-full", mean_recurrence=8, mean_backprop_depth=4,
                       recurrent_iteration_method="per-sequence", tie_embeddings=True)
    torch.manual_seed(seed)
    model = cfg.construct_model()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("A_log") or n.endswith("dt_bias"):
                p.normal_(0.0, 0.5)
            elif p.dim() == 1:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0.0, 0.08)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def selftest_model():
    model, cfg = tiny_model()
    R = PR(model, cfg, "cpu", bos=0, name="tiny")
    res = {"ve_core": R.ve_core, "ve_coda": R.ve_coda}
    A = [0, 5, 6, 7, 8, 9, 10, 11, 12]
    B = [0, 5, 13, 14, 8, 9, 10, 11, 12]
    R.set_init(16, 1)
    za = R.run(A, 8)["z_r"]
    zb = R.run(B, 8)["z_r"]
    tA = int(za.argmax())
    tB = int(zb.argmax()) if int(zb.argmax()) != tA else int(zb.topk(2).indices[1])
    it = {"ds": "D1", "pair_id": "tiny|1", "A": A, "B": B, "r": 8, "src": [2, 3], "tA": tA, "tB": tB, "rel": "capital",
          "cluster": "c1", "split": "dev", "strata": {"pos_len": ">=2"}, "label": "tiny",
          "para": [{"fmt": "p0", "A": [0, 20] + A[1:], "B": [0, 20] + B[1:], "src": [3, 4], "r": 9}]}
    it["Tmax"] = 10
    hk = check_hooks(R, it)
    res["hook_checks"] = hk
    X = Item(R, it)
    X.depth_margins()
    res["readout_exact_K"] = X.out["readout_exact_K"]
    run_controls(X, subset=True)
    res["controls_exact"] = all(v["exact"] for k, v in X.rows.items())
    allp = X.allpos()
    s = ("c", 2, R.lmid)
    o = R.run(A, 8, edits=X.o1(s, allp), e_edits=X.e_ed(range(3, K + 1), allp))
    res["pcs_without_ve_inexact"] = not torch.equal(o["z_r"], X.cB["z_r"])
    o = R.run(A, 8, edits=X.o1(s, allp), ve_edits=X.ve_ed(allp, after=s))
    res["pcs_without_e_inexact"] = not torch.equal(o["z_r"], X.cB["z_r"])
    o = R.run(A, 8, edits=X.o1(s, allp), e_edits=X.e_ed(range(2, K + 1), allp), ve_edits=X.ve_ed(allp, after=s))
    res["pcs_with_e_from_same_loop_exact"] = bool(torch.equal(o["z_r"], X.cB["z_r"]))
    o = R.run(A, 8, edits=X.o1(s, allp), e_edits=X.e_ed(range(3, K + 1), allp), ve_edits=X.ve_ed(allp, after=("c", 2, R.lmid + 1)))
    res["pcs_ve_one_call_late_inexact"] = not torch.equal(o["z_r"], X.cB["z_r"])
    # review round 2 F1: VE-core and VE-coda partition VE-persist; STA@post_inj(8) + B's value embeds at src after it
    # (loop-8 core + coda) is B's computation (A and B differ only at src here)
    src = it["src"]
    vfull, vcore, vcoda = X.ve_ed(src), X.ve_ed(src, coda=False), X.ve_ed(src, core=False)
    res["ve_split_partition"] = (set(vcore) | set(vcoda)) == set(vfull) and not (set(vcore) & set(vcoda))         and all(k[0] in R.ve_coda.values() for k in vcoda) and all(k[0] in R.ve_core.values() for k in vcore)
    o = R.run(A, 8, edits=X.o1(("p", K), allp), ve_edits=X.ve_ed(src, after=("p", K)))
    res["sta8_plus_after_ve_exact"] = bool(torch.equal(o["z_r"], X.cB["z_r"]))
    run_e1(X)
    run_e5(X, True)
    run_para(X, force=True)
    run_e8(X, dev_search=True)
    res["n_rows"] = len(X.rows)
    res["o2_rows"] = sorted(k for k in X.rows if k.startswith("O2|"))
    res["pval_n"] = len(X.out.get("pval", {}))
    res["para"] = X.out.get("para_tried")
    res["grad_identical"] = X.out["grad_identical"]
    res["G_n"] = len(X.out["G"])
    res["ver_S"] = X.out["ver_S"]
    res["eown_state_ok"] = all(v.get("mj_state_ok", True) for v in X.rows.values())
    res["eown_rows"] = sorted(k for k, v in X.rows.items() if "mj" in v)
    res["sta_rows"] = [k for k in X.rows if k.startswith("STA|")]
    res["ve_rows"] = sorted(k for k in X.rows if k.startswith("VE") or k.startswith("STAVC"))
    res["t_fwd_plain"] = X.out.get("t_fwd_plain")
    d = dict(X.out)
    d["dev_search"] = True
    resd = {"items": [d, dict(d, pair_id="tiny|1", split="test", rows={k: v for k, v in X.rows.items()
                                                                       if not k.startswith("O1|ans|c")})]}
    cm = {"tiny|1": it}
    dt = dev_tuned_pass(R, resd, cm)
    res["dev_tuned"] = {k: (v.get("site") if isinstance(v, dict) else v) for k, v in dt.items()}
    vb = dict(it, ds="VB", A=A[:8] + [20], A2=A[:8] + [21], B=B[:8] + [21], B2=B[:8] + [20], restA=[20, 3],
              restB=[21, 4], cont=True, r=7, tA=20, tB=21, para=[])
    Xv = Item(R, vb)
    Xv.finish_clean_vb()
    Xv.row("O1|ans|c1.3|1", edits=Xv.o1(("c", 1, R.llast), [7]), lo=True)
    res["vb_lo"] = {"causality_ok": Xv.out["causality_ok"], "row_caus": Xv.rows["O1|ans|c1.3|1"].get("caus"),
                    "lo": Xv.rows["O1|ans|c1.3|1"].get("lo"), "LO_A": Xv.out["LO_A"], "LO_B": Xv.out["LO_B"]}
    ok = (hk["all_ok"] and res["readout_exact_K"] and res["controls_exact"] and res["pcs_without_ve_inexact"]
          and res["pcs_without_e_inexact"] and res["pcs_ve_one_call_late_inexact"] and res["grad_identical"]
          and res["G_n"] == 32 and res["eown_state_ok"] and res["vb_lo"]["causality_ok"] and res["vb_lo"]["row_caus"]
          and len(res["eown_rows"]) == 8 and res["dev_tuned"]["S"] is not None and len(res["o2_rows"]) == 8
          and res["pval_n"] == 20 and res["ve_split_partition"] and res["sta8_plus_after_ve_exact"]
          and res["ve_rows"] == ["STAVC|p8", "VEC", "VECD", "VEP"] and res["t_fwd_plain"] is not None)
    res["ok"] = bool(ok)
    print(json.dumps(res, default=str))
    return ok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest_model", action="store_true")
    ap.add_argument("--build_items", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--scan_only", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--model", default="parcae-370m")
    ap.add_argument("--data", default="../out/k17_parcae/data")
    ap.add_argument("--out", default="../out/k17_parcae")
    ap.add_argument("--tok", default=TOK_REMOTE)
    ap.add_argument("--plan", default="")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--no_gradient_rank", action="store_true",
                    help="cluster copy: skip E8 (gradient score, verify rows, dev search) and the dev-tuned pass")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--budget_min", type=float, default=150.0)
    ap.add_argument("--deadline_epoch", type=float, default=0.0,
                    help="absolute end time (epoch s) shared by every launch of one run (review round 2 F8)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--targets", default="", help="plan section 10 cut rules, e.g. VB=40 (overrides TARGET)")
    a = ap.parse_args()
    NO_GRAD[0] = a.no_gradient_rank
    for kv in [t for t in a.targets.split(",") if t]:
        k, v = kv.split("=")
        assert k in TARGET and int(v) >= MIN_KEPT, kv
        TARGET[k] = int(v)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.selftest_model:
        sys.exit(0 if selftest_model() else 1)
    if a.build_items:
        build_items(a.out, a.tok)
        return
    if a.summarize:
        summarize(a.out, a.plan or None)
        return
    if a.run or a.scan_only:
        ds = [d for d in a.datasets.split(",") if d]
        assert all(d in DATASETS for d in ds), ds
        run_model(a.model, a.data, a.out, a.device, a.debug, ds, a.budget_min, scan_only=a.scan_only,
                  deadline_epoch=a.deadline_epoch or None)


if __name__ == "__main__":
    main()
