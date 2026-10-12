"""K17-PUBLIC: core mechanism tests on Huginn-0125, McLeish r32, its same-data non-recurrent Llama, Ouro-Thinking.
Brief B-K17-PUBLIC with PLAN.md amendments A1-A8; plan (frozen definitions, thresholds, predictions):
notes/k17_public_plan.md (v3: amendments, critic round 1 and critic round 2 applied).

Models (bf16 weights, float32 readout, batch 1):
  huginn     tomg-group-umd/huginn-0125 @ bb6621b6, K = 32 (and K = 16 reduced, model key huginn16), no BOS
  mcleish    smcleish/Recurrent-Llama-3.2-train-recurrence-32 @ a5f6f126, K = 32, BOS 128000
  llama_nr   smcleish/Llama-3.2-non-recurrent-posttrained @ 004e96ac, 16 layers, BOS 128000 (same-data
             non-recurrent reference, never a depth control; joint clean filter with mcleish)
  ouro_think ByteDance/Ouro-1.4B-Thinking @ 3aaa2224, K = 4, no BOS (E1 anchor only)
  ouro26_think ByteDance/Ouro-2.6B-Thinking @ f1edd81e, K = 4, C = 48, no BOS (cluster fill: the ouro_think run on
             the 2.6B checkpoint; same tokenizer file, candidates and targets)
Sites: ("p", j) adapter output at loop j (post_inj), ("c", j, l) core block l output at loop j, ("k", c) coda block c
output. NR is K = 1 with C = 16 layers; Ouro K = 4 with C = 24 (l = 23 is the model.norm output = carried state).
Arms: E1 placement, E5 input protocols (+ SC state ceiling, SRCIN decomposition), E8 carry gain (predict, verify),
presence and answer classes. Controls: PC-final, SZ, PC-site [harness], RND, SC (informative).

    python k17_public.py --selftest                        (pure helpers, CPU)
    python k17_public.py --selftest_model                  (tiny random raven + Llama models, CPU)
    python k17_public.py --build_items --out DATA_DIR      (laptop CPU, three tokenizers)
    python k17_public.py --run --model mcl_nr|huginn|ouro_think|ouro26_think --data DATA --out OUT [--debug N] [--datasets ..]
    python k17_public.py --fdcheck --model mcl_nr|huginn --data DATA --out OUT [--fd_items 10]   (fp32 FD check)
    python k17_public.py --summarize --out OUT [--plan PLAN.md]
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ------------------------------------------------------------------ pure helpers copied verbatim from k17_parcae.py
# (md5 70d1d754 at copy time; that file is edited by another track, so k17_public is self-contained. --selftest
# checks parity with k17_parcae and k17_varbench where they are importable.)
HIT_TOL = 0.05
GAP = 2.0


def _trunc_std_factor(c=3.0):
    phi = math.exp(-c * c / 2) / math.sqrt(2 * math.pi)
    Z = math.erf(c / math.sqrt(2))
    return math.sqrt(1 - 2 * c * phi / Z)


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


def proj_many(hs, mAs, mBs):
    """proj() for a list of sites in one device pass and one host transfer (same arithmetic in float32)."""
    if not hs:
        return []
    h, a, b = (torch.stack([t.float() for t in v]) for v in (hs, mAs, mBs))
    d = b - a
    dn, an = d.norm(dim=-1), a.norm(dim=-1)
    val = ((h - (a + b) / 2) * d).sum(-1) / (d * d).sum(-1) + 0.5
    out = torch.stack([val, dn, an]).tolist()
    return [None if dn_ < 0.01 * an_ else v for v, dn_, an_ in zip(*out)]


def proj(h, mA, mB):
    """Pair projection: 0 at clean A, 1 at clean B; None if |mB - mA| < 0.01 |mA|."""
    d = (mB - mA).float()
    if float(d.norm()) < 0.01 * float(mA.float().norm()):
        return None
    return float(((h.float() - (mA.float() + mB.float()) / 2) @ d) / (d @ d) + 0.5)


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


def vals(z, tA, tB):
    """[z(tA), z(tB), logsumexp, top-1] for one logit vector (one host transfer)."""
    zf = z.float()
    a, b, l, t = torch.stack([zf[tA], zf[tB], torch.logsumexp(zf, -1), zf.argmax().to(zf.dtype)]).tolist()
    return [r5(a), r5(b), r5(l), int(t)]


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


TRUNC_STD = _trunc_std_factor()


def hit_v3(G, dm, D, thr, tol=HIT_TOL):
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


def hit_v4(G, dm, D, thr, latest, tol=HIT_TOL):
    """Plan v3 T7 (critic 2 F1 = VARBENCH v4 T7): hit_v3 plus the hit of the fixed latest site (readout-proximity
    rule) under the same tolerance band. The scored A2-1 endpoint is hit - latest_hit; band chance is a diagnostic."""
    h = hit_v3(G, dm, D, thr, tol)
    if h is None:
        return None
    if h["eligible"] and latest in dm and dm[latest] is not None:
        mx = max(dm[k] for k in G if k in dm and dm[k] is not None)
        h["latest_hit"] = int(dm[latest] >= mx - tol * D)
        h["latest_regret"] = (mx - dm[latest]) / D
    else:
        h["latest_hit"] = h["latest_regret"] = None
    h["latest"] = latest
    return h


# ------------------------------------------------------------------ constants (plan sections 2-11)
from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE  # noqa: E402  (honours HF_HOME)
HUB = _HF_HUB_CACHE
CKPT = {
    "huginn": ("tomg-group-umd/huginn-0125", "bb6621b65e90b6a4b9b29ef88dc83866d450470c"),
    "mcleish": ("smcleish/Recurrent-Llama-3.2-train-recurrence-32", "a5f6f126e9d9302445346844f1d7d29d111a06eb"),
    "llama_nr": ("smcleish/Llama-3.2-non-recurrent-posttrained", "004e96ace360c1a37c08f4555227136e845087b3"),
    "ouro_think": ("ByteDance/Ouro-1.4B-Thinking", "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"),
    "ouro26_think": ("ByteDance/Ouro-2.6B-Thinking", "f1edd81e7ac41355db670500ceaf204e0f73af68"),
}
TOK_SHA16 = {"huginn": "9cc201a5061b70ab", "llama": "6b9e4e7fb171f92f", "ouro": "fcb808fe5e7642f5"}
MODELING_SHA16 = {"huginn": "a1d447da93c605a6", "mcleish": "4f7a86e9b4ce9e73"}
TOKFAM = {"huginn": "huginn", "huginn16": "huginn", "huginn8": "huginn", "mcleish": "llama", "llama_nr": "llama", "ouro_think": "ouro",
          "ouro26_think": "ouro"}
BOS = {"huginn": None, "llama": 128000, "ouro": None}
KDEF = {"huginn": 32, "huginn16": 16, "huginn8": 8, "mcleish": 32, "llama_nr": 1, "ouro_think": 4, "ouro26_think": 4}
JSET = {32: (1, 2, 8, 16, 31, 32), 16: (1, 2, 4, 8, 15, 16), 8: (1, 4, 6, 7, 8), 4: (1, 2, 3, 4)}
IN1_J = {32: (1, 2, 4, 8, 16, 24, 28, 30, 31, 32), 16: (1, 4, 8, 15, 16)}
INP_J = {32: (1, 2, 3, 4, 8, 9, 16, 17, 24, 28, 30, 31), 16: (1, 2, 4, 8, 15)}
SRCIN_J = {32: (1, 2, 8, 16, 31), 16: ()}
DATASETS = ("D1", "CF", "D2", "D4", "D5", "VB")
E8_DATASETS = ("D1", "D2", "D4")
PB8_ONLY = ("huginn8",)
TARGET = {"mcl_nr": {"D1": 80, "CF": 60, "D2": 80, "D4": 60, "D5": 60, "VB": 60},
          "huginn": {"D1": 60, "CF": 40, "D2": 60, "D4": 40, "D5": 40, "VB": 40},
          "ouro_think": {"D1": 80, "D2": 80, "D4": 60},
          "ouro26_think": {"D1": 80, "D2": 80, "D4": 60}}
HUGINN16_DS = ("D2", "D4")
# plan v3 (critic 2 F1, F9): every site of S+ is verified at a = 1 on "search items" = dev and cross items (the
# selector and the latest-site rule are not fitted, so these items are fair for A2-1; test items are kept for the
# dev-tuned baseline only). Caps per model and dataset.
DEV_FULL_CAP = {"huginn": 25, "mcleish": 60, "llama_nr": 60}
SEARCH_SPLITS = ("dev", "cross")
IN1S_J = {32: (1, 16, 31), 16: (1, 8, 15), 8: (1, 4, 7)}          # F3: one call at e_A + n (e_B - e_A), n = K-j+1
PB8_R = (1, 2, 4, 8)                                                # F10: Llast@(K-r), equal loops remaining
TINY_A = 1e-3                                                       # F13: tiny-edit noise floor dose
N_TINY = 20
FD_A = (1e-6, 1e-3, 1e-2)                                           # F2: finite-difference doses (1e-6 = floor probe)
FD_N = 10                                                           # items per dataset (D1, D2)
# review 3 F1 (PLAN G4): path-integrated gain, input path (EAP-IG style): the source-span token embeddings are moved
# from A to B by a, the whole network follows, and the gradient at every answer leaf is averaged over these doses.
PIG_A = (0.25, 0.5, 0.75, 1.0)
N_PIG_VER = 3                                                       # PIG top-3 sites of S verified at both doses
# review 3 F4: E8 pilot gate (raven full search). The first PILOT_N search items are always fully searched; later ones
# only if (pilot eligible rate) x (search items under the cap) >= MIN_E8, else the cell is descriptive.
PILOT_N = 10
A025_NOISE_CUT = 0.5 * 0.25 * 0.25                                  # half the a = 0.25 "not carried" cut (F7)
O2_LABEL = "cross-template same-pair transfer (built from the item's own A/B pair; not a transferable vector)"
SUBJ_CAP = 2
D2_CLUSTER_CAP = 2
MIN_KEPT = 40
MIN_E8 = 20
N_SUB = {"huginn": 10, "mcleish": 20, "llama_nr": 20, "ouro_think": 20, "huginn16": 10, "ouro26_think": 20}
# review 3 F2, F12: command-line overrides (--full_search_cap, --pc_site_n, --descriptive); filled in main()
FS_CAP_OVR, PCS_OVR, DESCRIPTIVE = {}, {}, {"targets": {}}
# RA-1 scope cut (2026-10-08): --no_reduced skips the Huginn K = 16 / K = 8 runners and the K = 32 PB8 rows
RUN_OPTS = {"no_reduced": False}


def full_cap(name, ds):
    """Full-search cap for runner `name` on dataset ds: --full_search_cap 'name.ds=N', 'name=N' or '*=N', else
    DEV_FULL_CAP."""
    for k in ("%s.%s" % (name, ds), name, "*"):
        if k in FS_CAP_OVR:
            return FS_CAP_OVR[k]
    return DEV_FULL_CAP.get(name, 0)


def n_sub(name):
    """PC-site / SC-identity subset size (first n kept items): --pc_site_n 'name=N' or 'N', else N_SUB."""
    return PCS_OVR.get(name, PCS_OVR.get("*", N_SUB.get(name, 10)))


def parse_kv(s, keys_ok=None):
    out = {}
    for kv in [t for t in (s or "").split(",") if t]:
        if "=" not in kv:
            out["*"] = int(kv)
            continue
        k, v = kv.split("=")
        assert keys_ok is None or k in keys_ok, kv
        out[k] = int(v)
    return out
GAP = 2.0
NBOOT = 1000
TOST_M = 0.10
A_VERIFY = (1.0, 0.25)
THR = {1.0: 2.0, 0.25: 0.5}
HIT_TOL = 0.05
FN_G = 0.05
SEED_OFFSETS = (0, 1, 2)
CF_RECORDS = os.path.join(os.environ.get("DATA_ROOT") or os.path.join(HERE, "data"), "k15_full", "cf_records.json")
CF_RELS = ("P17", "P103", "P364", "P641", "P176")
D1_TEMPLATES = (("capital", "The capital of {} is", "country"),
                ("city", "{} is a city in the country of", "city"),
                ("city_mid", "The city of {} is located in the country of", "city"))
D1_PARA = {"capital": ["The capital city of {} is", "Q: What is the capital of {}? A:"],
           "city": ["The city of {} is located in the country of", "Q: In which country is the city of {}? A:"],
           "city_mid": ["{} is a city in the country of", "Q: In which country is the city of {}? A:"]}
LAPTOP_TOK = {"huginn": HUB + "/models--tomg-group-umd--huginn-0125/snapshots/"
                        "bb6621b65e90b6a4b9b29ef88dc83866d450470c/tokenizer.json",
              "llama": HUB + "/models--smcleish--Recurrent-Llama-3.2-train-recurrence-32/snapshots/"
                       "a5f6f126e9d9302445346844f1d7d29d111a06eb/tokenizer.json",
              "ouro": HUB + "/models--ByteDance--Ouro-1.4B-Thinking/snapshots/"
                      "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d/tokenizer.json"}
REUSED = ("k15_data.py", "k16_data.py", "k13_edit.py", "k15_smoke.py", "k17_varbench.py")


def snap(key):
    repo, rev = CKPT[key]
    return os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots", rev)


# ------------------------------------------------------------------ statistics (one-way or dyadic clusters)
def _units(cl):
    return [c if isinstance(c, (tuple, list)) else (c,) for c in cl]


def bootc(xs, cl, n=NBOOT, seed=0, q=(0.025, 0.975)):
    """Mean with a percentile cluster bootstrap -> [mean, lo, hi, n_items, n_units]. cl[i] is one cluster key
    (one-way, ratio of sums) or a pair (unit_A, unit_B) (dyadic: units resampled, weight c[A] * c[B]; audit_path_ouro
    F8, VARBENCH boot2). Pairs whose two units are equal count once (weight c[A])."""
    pr = [(float(x), tuple(u)) for x, u in zip(xs, _units(cl)) if x is not None]
    if not pr:
        return None
    units = sorted({u for _, c in pr for u in c}, key=str)
    ix = {u: i for i, u in enumerate(units)}
    x = torch.tensor([v for v, _ in pr], dtype=torch.float64)
    v0 = float(x.mean())
    if len(units) < 2:
        return [round(v0, 4), None, None, len(pr), len(units)]
    g = torch.Generator().manual_seed(seed)
    draw = torch.randint(len(units), (n, len(units)), generator=g)
    cnt = torch.zeros(n, len(units), dtype=torch.float64).scatter_add_(1, draw, torch.ones(n, len(units),
                                                                                         dtype=torch.float64))
    w = torch.ones(n, len(pr), dtype=torch.float64)
    for k in range(max(len(c) for _, c in pr)):
        idx = torch.tensor([ix[c[min(k, len(c) - 1)]] for _, c in pr])
        same = torch.tensor([k > 0 and (len(c) <= k or c[k] == c[0]) for _, c in pr])
        wk = cnt[:, idx]
        w = w * torch.where(same[None, :], torch.ones_like(wk), wk)
    sw = w.sum(1)
    ok = sw > 0
    bs = (w[ok] * x[None, :]).sum(1) / sw[ok]
    qq = torch.quantile(bs, torch.tensor(list(q), dtype=torch.float64))
    return [round(v0, 4), round(float(qq[0]), 4), round(float(qq[1]), 4), len(pr), len(units)]


def tostc(xs, cl, margin=TOST_M):
    b90, b95 = bootc(xs, cl, q=(0.05, 0.95)), bootc(xs, cl)
    if b90 is None or b90[1] is None:
        return None
    ok = b90[1] > -margin and b90[2] < margin
    return {"ci90": b90[:3], "null_by_tost": bool(ok),
            "label": ("null (TOST, margin %.2f)" % margin) if ok else ("below %.3f" % b95[2])}


def pairedc(xs, ys, cl):
    d, c = [], []
    for x, y, k in zip(xs, ys, cl):
        if x is not None and y is not None:
            d.append(x - y)
            c.append(k)
    return bootc(d, c)


def mean(xs):
    xs = [v for v in xs if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def median(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    n = len(xs)
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 4)


# ------------------------------------------------------------------ sites (plan sections 4, 7)
def name2site(name, C):
    kind, j = name.split("@")
    j = int(j)
    if kind == "post_inj":
        return ("p", j)
    if kind == "layer":
        return ("c", 1, j)
    return ("c", j, {"L0": 0, "Lmid": C // 2, "Llast": C - 1}[kind])


def site_name(s, C, family):
    if family == "nr":
        return "layer@%d" % s[2]
    if s[0] == "p":
        return "post_inj@%d" % s[1]
    if s[0] == "k":
        return "coda@%d" % s[1]
    nm = {0: "L0", C // 2: "Lmid", C - 1: "Llast"}.get(s[2])
    return "%s@%d" % (nm, s[1]) if nm else "c%d.%d" % (s[1], s[2])


def exec_order(site, C, K):
    if site[0] == "p":
        return (site[1] - 1) * (C + 1)
    if site[0] == "c":
        return (site[1] - 1) * (C + 1) + 1 + site[2]
    return K * (C + 1) + site[1]


def e1_names(family, K, ds):
    if family == "nr":
        return ["layer@%d" % l for l in range(16)]
    if family == "ouro":
        return ["%s@%d" % (k, j) for j in range(1, K + 1) for k in ("L0", "Lmid", "Llast")]
    J = JSET[K]
    if ds == "VB":
        out = []
        for j in (1, K // 2, K - 1, K):
            out += ["post_inj@%d" % j, "Llast@%d" % j]
        return out + ["Lmid@%d" % K]
    out = []
    for j in J:
        out += ["post_inj@%d" % j]
        if K == 32:
            out += ["Lmid@%d" % j]
        out += ["Llast@%d" % j]
    if K == 32:
        out += ["L0@1", "L0@%d" % K]
    return out


def rnd_names(family, K, ds):
    """Plan v3 (critic 2 F5): RND same-norm at every E1 site and both positions, so every cell has its own floor
    e_RND and the 'works' rule (e - e_RND, EXPERIMENT_MATRIX 5.4) can be applied at every site."""
    return e1_names(family, K, ds)


def a025_names(family, K, ds):
    if ds == "VB":
        return []
    if family in ("nr", "ouro"):
        return e1_names(family, K, ds)
    return ["Llast@%d" % j for j in JSET[K]] + (["Lmid@%d" % K] if K == 32 else [])


def sc_names(family, K, ds):
    """Informative state ceiling SC at Llast@j for every E1 loop j < K (raven). Plan v3 (critic 2 F7): SC at
    post_inj(j), L0@j and Lmid@j equals SC at Llast@j bit for bit, because no layer inside loop j reads e after the
    adapter call; those rows run as harness identities (SCID) on the subset items, and the attribution rule uses
    SC(Llast@j) for every E1 site of loop j."""
    if family != "raven":
        return []
    return [n for n in e1_names(family, K, ds) if n.startswith("Llast@") and name2site(n, 6)[1] < K]


def latest_site(sites, C, K):
    """The site of a set nearest the readout in execution order (VARBENCH v4 readout-proximity baseline)."""
    return max(sites, key=lambda s: exec_order(s, C, K))


def last_loop_cell(family, K, C):
    """Last-loop answer cell (A7, paper EXPERIMENTS 3b2): raven Lmid@K, Ouro Lmid@4, NR layer 14."""
    if family == "nr":
        return ("c", 1, 14)
    return ("c", K, C // 2)


def e8_sets(K, C):
    """Primary S: every core-block output except the readout-feeding / coda-gated Llast@K; S+ adds it (raven).
    NR: layers 0..14 (S), S+ = S + layer 15 is NOT used (layer 15 at r is the readout-feeding site)."""
    Sp = [("c", j, l) for j in range(1, K + 1) for l in range(C)]
    S = [s for s in Sp if s != ("c", K, C - 1)]
    return S, Sp


# ------------------------------------------------------------------ tokenizers and item build (laptop CPU)
class Tok(object):
    """tokenizers.Tokenizer wrapper; enc never adds BOS (the runtime prepends it, DC3)."""

    def __init__(self, fam, path):
        from tokenizers import Tokenizer
        self.name, self.path = fam, os.path.expanduser(path)
        self.t = Tokenizer.from_file(self.path)
        self.sha16 = sha256_file(self.path)[:16]
        self.bos = BOS[fam]

    def enc(self, text, ans=False):
        return list(self.t.encode(text, add_special_tokens=False).ids)

    def enc_spans(self, text, ranges):
        e = self.t.encode(text, add_special_tokens=False)
        spans = [[i for i, (s, t) in enumerate(e.offsets) if s < c1 and t > c0] for c0, c1 in ranges]
        return list(e.ids), spans

    def dec(self, ids):
        return [self.t.decode([i]) for i in ids]


def answer_parts(enc, ans_a, ans_b):
    """-> (suffix, tA, tB, restA, restB): suffix = shared answer prefix ids (appended to the prompt), tA/tB the first
    differing tokens, rest = the remaining answer ids (full-answer filter, critic F10). None if a prefix."""
    ta, tb = enc(" " + ans_a, ans=True), enc(" " + ans_b, ans=True)
    c = 0
    while c < min(len(ta), len(tb)) and ta[c] == tb[c]:
        c += 1
    if c >= len(ta) or c >= len(tb):
        return None
    return list(ta[:c]), ta[c], tb[c], list(ta[c + 1:]), list(tb[c + 1:])


def ent_key(kind, name):
    return "%s:%s" % (kind, name)


def split_e8(units, prefix):
    """E8-only split (critic F1(1)): dev if every unit hashes dev, test if every unit hashes test, else cross."""
    s = {split_hex(prefix, u) for u in units}
    return s.pop() if len(s) == 1 else "cross"


def build_d1(tk):
    import k13_edit as KE
    tables = {"capital": KE.CAPITALS, "city": KE.CITIES, "city_mid": KE.CITIES}
    by_t = {}
    for rel, tmpl, kind in D1_TEMPLATES:
        tab = tables[rel]
        keys = list(tab)
        for i, a in enumerate(keys):
            for b in keys[i + 1:i + 6]:
                if tab[a] == tab[b]:
                    continue
                ia, ib = tk.enc(tmpl.format(a)), tk.enc(tmpl.format(b))
                ta, tb = tk.enc(" " + tab[a]), tk.enc(" " + tab[b])
                if len(ia) != len(ib) or ta[0] == tb[0]:
                    continue
                diff = [k for k in range(len(ia)) if ia[k] != ib[k]]
                if not diff or diff[-1] - diff[0] + 1 != len(diff):
                    continue
                ua, ub = ent_key(kind, a), ent_key(kind, b)
                forms = para_forms(diff, [("para%d" % n, tk.enc(t.format(a)), tk.enc(t.format(b)))
                                          for n, t in enumerate(D1_PARA[rel])])
                by_t.setdefault(rel, []).append({
                    "ds": "D1", "pair_id": "D1|%s|%s|%s" % (rel, a, b), "A": ia, "B": ib, "r": len(ia) - 1,
                    "src": diff, "tA": int(ta[0]), "tB": int(tb[0]), "restA": [], "restB": [], "rel": rel,
                    "cl": [ua, ub], "split": split_e8([ua, ub], "k17pub-d1:"),
                    "strata": {"pos_len": "1" if len(diff) == 1 else ">=2", "rel": rel,
                               "subject_pos0": "yes" if 0 in diff else "no"},
                    "para": forms, "sel": {"subjects": ["%s|%s" % (rel, ua), "%s|%s" % (rel, ub)]},
                    "label": "%s -> %s" % (a, b)})
    out = rr_order([c for rel in sorted(by_t) for c in by_t[rel]], lambda c: c["rel"])
    st = {"candidates": len(out), "pos_len_x_rel": {}, "split": {}}
    for c in out:
        k = "%s|%s" % (c["strata"]["pos_len"], c["rel"])
        st["pos_len_x_rel"][k] = st["pos_len_x_rel"].get(k, 0) + 1
        st["split"][c["split"]] = st["split"].get(c["split"], 0) + 1
    st["subject_pos0"] = sum(1 for c in out if c["strata"]["subject_pos0"] == "yes")
    return out, st


def build_cf(tk):
    import k15_data as KD
    recs = json.load(open(CF_RECORDS))["records"]
    cands, cst = KD.cf_candidates(tk.enc, recs, rels=list(CF_RELS))
    by = {}
    for p in cands:
        ua, ub = ent_key("cf", p["a"]), ent_key("cf", p["b"])
        forms = para_forms(p["pos"], [("para%d" % n, a, b) for n, (a, b) in enumerate(p["para"])])
        by.setdefault(p["rel"], []).append({
            "ds": "CF", "pair_id": "CF|%s|%d|%d" % (p["rel"], p["case_a"], p["case_b"]), "A": list(p["A"]),
            "B": list(p["B"]), "r": len(p["A"]) - 1, "src": list(p["pos"]), "tA": int(p["tA"]), "tB": int(p["tB"]),
            "restA": [], "restB": [], "rel": p["rel"], "cl": [ua, ub], "split": split_e8([ua, ub], "k17pub-cf:"),
            "strata": {"pos_len": "1" if len(p["pos"]) == 1 else ">=2", "rel": p["rel"],
                       "subject_pos0": "yes" if 0 in p["pos"] else "no"},
            "para": forms, "sel": {"subjects": [ua, ub]}, "label": "%s -> %s" % (p["a"], p["b"])})
    out = rr_order([c for rel in CF_RELS for c in by.get(rel, [])], lambda c: c["rel"])
    return out, {"candidates": len(out), "per_rel": {r: len(by.get(r, [])) for r in CF_RELS}, "cf_stats": cst,
                 "with_para": sum(1 for c in out if c["para"])}


def build_d2(tk):
    import k15_data as KD
    out = []
    for op, a, b, a2, b2 in KD.d2_raw():
        at = answer_parts(tk.enc, str(a + b), str(a2 + b2))
        if at is None:
            continue
        suf, tA, tB, rA, rB = at
        ia, ib = tk.enc(KD.D2_PROMPT.format(a, b)) + suf, tk.enc(KD.D2_PROMPT.format(a2, b2)) + suf
        if len(ia) != len(ib):
            continue
        sa, sb = subj_span(ia, ib)
        if not sa or len(sa) != len(sb):
            continue
        xa, xb = (a, a2) if op == "op1" else (b, b2)
        ua, ub = "x=%d" % xa, "x=%d" % xb
        forms = para_forms(sa, [("para%d" % n, tk.enc(t.format(a, b)) + suf, tk.enc(t.format(a2, b2)) + suf)
                                for n, t in enumerate(KD.D2_PARA)])
        out.append({"ds": "D2", "pair_id": "D2|%s|%d + %d|%d + %d" % (op, a, b, a2, b2), "A": ia, "B": ib,
                    "r": len(ia) - 1, "src": sa, "tA": int(tA), "tB": int(tB), "restA": rA, "restB": rB, "rel": op,
                    "cl": [ua, ub], "split": split_e8([ua, ub], "k17pub-d2:"), "ans": [a + b, a2 + b2],
                    "strata": {"pos_len": "1" if len(sa) == 1 else ">=2", "rel": op,
                               "endpoint": "full answer" if not (rA or rB or suf and len(suf) > 1) else
                               "answer token %d" % len(suf)},
                    "para": forms, "sel": {"clusters": [ua, ub]}, "label": "%d + %d -> %d + %d" % (a, b, a2, b2)})
    out = rr_order(out, lambda c: c["rel"])
    seen, ded = set(), []                       # d2_raw can draw a pair twice: keep the first, order unchanged
    for c in out:
        if c["pair_id"] not in seen:
            seen.add(c["pair_id"])
            ded.append(c)
    out = ded
    assert len({c["pair_id"] for c in out}) == len(out), "duplicate D2 item"
    return out, {"candidates": len(out), "with_rest": sum(1 for c in out if c["restA"] or c["restB"]),
                 "with_para": sum(1 for c in out if c["para"])}


def enc_pair(tk, tA, tB, ansA, ansB, rngA, rngB):
    """k16_data.encode_pair generalised (A5: space-digit merging): the shared answer prefix may be empty (Huginn
    merges the space into the first digit) or longer than one token; the source span is the tokens overlapping the
    value's characters and must equal the A/B difference."""
    at = answer_parts(tk.enc, ansA, ansB)
    if at is None:
        return None, "answer_prefix_or_none"
    suf, ta, tb, rA, rB = at
    ia, sa = tk.enc_spans(tA.s, rngA)
    ib, sb = tk.enc_spans(tB.s, rngB)
    ia, ib = ia + suf, ib + suf
    if len(ia) != len(ib):
        return None, "len_mismatch"
    span, span_b = subj_span(ia, ib)
    fa = sorted(set(i for s in sa for i in s))
    fb = sorted(set(i for s in sb for i in s))
    if span != span_b or span != fa or span != fb:
        return None, "span_mismatch"
    return {"ids_A": ia, "ids_B": ib, "suffix_ids": suf, "source_span": span, "tA": ta, "tB": tb,
            "restA": rA, "restB": rB}, None


def _val_ok(tk, ids, span, val):
    return len(span) == 1 and tk.dec([ids[span[0]]])[0].strip() == str(val)


def d4_row_any(tk, it):
    import k16_data as KD6
    kind, names, query, src = it["kind"], it["names"], it["query"], it["src"]
    ansA, ansB = KD6.d4_answer(it["vals_A"], query), KD6.d4_answer(it["vals_B"], query)
    bA = KD6.d4_prompt(KD6.D4_MAIN, kind, names, it["vals_A"], query)
    bB = KD6.d4_prompt(KD6.D4_MAIN, kind, names, it["vals_B"], query)
    part, why = enc_pair(tk, bA, bB, str(ansA), str(ansB), [bA.tags["v%d" % src][0]], [bB.tags["v%d" % src][0]])
    if part is None:
        return None, why
    if not (_val_ok(tk, part["ids_A"], part["source_span"], it["vals_A"][src])
            and _val_ok(tk, part["ids_B"], part["source_span"], it["vals_B"][src])):
        return None, "value_not_single_token"
    para = []
    for f in [f for f in "ABC" if f != KD6.D4_MAIN]:
        pA, pB = KD6.d4_prompt(f, kind, names, it["vals_A"], query), KD6.d4_prompt(f, kind, names, it["vals_B"], query)
        pp, why = enc_pair(tk, pA, pB, str(ansA), str(ansB), [pA.tags["v%d" % src][0]], [pB.tags["v%d" % src][0]])
        if pp is None or pp["suffix_ids"] != part["suffix_ids"]:
            continue
        para.append((f, pp["ids_A"], pp["ids_B"]))
    return dict(part, answer_A=str(ansA), answer_B=str(ansB), para=para), None


def d5_row_any(tk, it):
    import k16_data as KD6
    vx, vy, m, src = it["vx"], it["vy"], it["m"], it["src"]
    (a, b), (a2, b2) = it["ops_A"], it["ops_B"]
    ansA, ansB = KD6.d5_answer(a, b, m), KD6.d5_answer(a2, b2, m)
    tag = "opa" if src == "a" else "opb"
    pA, pB = KD6.d5_prompt("P0", vx, vy, a, b, m), KD6.d5_prompt("P0", vx, vy, a2, b2, m)
    part, why = enc_pair(tk, pA, pB, str(ansA), str(ansB), [pA.tags[tag][0]], [pB.tags[tag][0]])
    if part is None:
        return None, why
    sv_a, sv_b = (a, b)[src == "b"], (a2, b2)[src == "b"]
    if not (_val_ok(tk, part["ids_A"], part["source_span"], sv_a) and _val_ok(tk, part["ids_B"], part["source_span"], sv_b)):
        return None, "value_not_single_token"
    para = []
    for f in ("P1", "P2"):
        qA, qB = KD6.d5_prompt(f, vx, vy, a, b, m), KD6.d5_prompt(f, vx, vy, a2, b2, m)
        pp, why = enc_pair(tk, qA, qB, str(ansA), str(ansB), [qA.tags[tag][0]], [qB.tags[tag][0]])
        if pp is None or pp["suffix_ids"] != part["suffix_ids"]:
            continue
        para.append((f, pp["ids_A"], pp["ids_B"]))
    return dict(part, answer_A=str(ansA), answer_B=str(ansB), src_vals=[sv_a, sv_b], mult=m,
                problems=["%d,%d,%d" % (a, b, m), "%d,%d,%d" % (a2, b2, m)], para=para), None


def d4_key(it, side):
    v = it["vals_%s" % side]
    return sha("%s|%s|%d|%s" % (it["kind"], ",".join(str(x) for x in v), it["src"],
                                ",".join(str(i) for i in it["query"])))[:16]


def build_d4(tk):
    import k16_data as KD6
    items = KD6.d4_items("T1", KD6.D4_N_TEMPLATES["T1"]) + KD6.d4_items("T2", KD6.D4_N_TEMPLATES["T2"])
    out, drops = [], {}
    for it in items:
        row, why = d4_row_any(tk, it)
        if row is None:
            drops[why] = drops.get(why, 0) + 1
            continue
        A, B, src = row["ids_A"], row["ids_B"], row["source_span"]
        if [i for i in range(len(A)) if A[i] != B[i]] != list(src):
            drops["unequal_or_bad_span"] = drops.get("unequal_or_bad_span", 0) + 1
            continue
        ua, ub = "d4:" + d4_key(it, "A"), "d4:" + d4_key(it, "B")
        out.append({"ds": "D4", "pair_id": "D4|" + it["id"], "A": A, "B": B, "r": len(A) - 1, "src": list(src),
                    "tA": int(row["tA"]), "tB": int(row["tB"]), "restA": row["restA"], "restB": row["restB"],
                    "rel": it["kind"], "cl": [ua, ub], "split": split_e8([ua, ub], "k17pub-d4:"),
                    "template": it["cluster_id"],
                    "strata": {"pos_len": "1" if len(src) == 1 else ">=2", "rel": it["kind"],
                               "endpoint": "full answer" if not (row["restA"] or row["restB"]) else "first differing token"},
                    "para": para_forms(src, row["para"]), "sel": {"clusters": [ua]}, "label": it["id"]})
    out = rr_order(out, lambda c: (c["split"], c["rel"]))
    return out, {"items": len(items), "kept": len(out), "dropped": drops, "with_para": sum(1 for c in out if c["para"]),
                 "with_rest": sum(1 for c in out if c["restA"] or c["restB"])}


def build_d5(tk):
    import k16_data as KD6
    demos = set((a, b, m) for a, b, m in KD6.D5_DEMOS)
    out, drops, edges = [], {}, []
    for it in KD6.d5_items():
        row, why = d5_row_any(tk, it)
        if row is None:
            drops[why] = drops.get(why, 0) + 1
            continue
        A, B, src = row["ids_A"], row["ids_B"], row["source_span"]
        if [i for i in range(len(A)) if A[i] != B[i]] != list(src):
            drops["unequal_or_bad_span"] = drops.get("unequal_or_bad_span", 0) + 1
            continue
        lead = {row["answer_A"][0], row["answer_B"][0]}
        if lead & {str(v) for v in row["src_vals"]}:
            drops["answer_lead_equals_source_digit"] = drops.get("answer_lead_equals_source_digit", 0) + 1
            continue
        (a, b), (a2, b2) = it["ops_A"], it["ops_B"]
        if (a, b, row["mult"]) in demos or (a2, b2, row["mult"]) in demos:
            drops["demo_problem"] = drops.get("demo_problem", 0) + 1
            continue
        edges.append(tuple(row["problems"]))
        out.append({"ds": "D5", "pair_id": "D5|" + it["id"], "A": A, "B": B, "r": len(A) - 1, "src": list(src),
                    "tA": int(row["tA"]), "tB": int(row["tB"]), "restA": row["restA"], "restB": row["restB"],
                    "rel": "T2", "template": it["cluster_id"],
                    "strata": {"pos_len": "1" if len(src) == 1 else ">=2", "rel": "T2", "src_operand": it["src"],
                               "endpoint": "full answer" if not (row["restA"] or row["restB"]) else "leading digit"},
                    "para": para_forms(src, row["para"]), "sel": {"problems": row["problems"]}, "label": it["id"]})
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
    for c in out:
        comp = find(c["sel"]["problems"][0])
        c["cl"] = ["d5:" + c["sel"]["problems"][0], "d5:" + c["sel"]["problems"][1]]   # dyadic over problems
        c["component"] = comp
        c["split"] = "n/a (D5 is not an E8 dataset; critic F9)"
    comps = len({c["component"] for c in out})
    return out, {"kept": len(out), "dropped": drops, "with_para": sum(1 for c in out if c["para"]),
                 "problems": len(par), "components": comps,
                 "largest_component_items": max([sum(1 for c in out if c["component"] == k)
                                                 for k in {c["component"] for c in out}] or [0])}


def build_vb(tk):
    """VARBENCH I1-I6 (v3) per tokenizer: copy-read target, one pair per problem (lowest var_name), k17vb-order."""
    import pandas as pd
    import k17_varbench as VB
    rows = [json.loads(x) for x in open(VB.PAIRS, encoding="utf-8")]
    core = [r for r in rows if r["split"] == "eval" and r["text_exact"]
            and all(r["tok"][t]["outside_equal"] for t in ("ouro", "huginn", "mcleish", "ifm"))]
    test = pd.read_parquet(os.path.join(VB.GSM_DIR, "test-00000-of-00001.parquet"))
    block = open(VB.PROMPT_FILE, encoding="utf-8").read()
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
        ea, eb = tk.t.encode(pa, add_special_tokens=False), tk.t.encode(pb, add_special_tokens=False)
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
        elif any(i not in sa for i in range(len(ipa)) if ipa[i] != ipb[i]):
            why = "outside_unequal"
        if why is None:
            fa, fb = tk.enc(pa + " " + C + str(x)), tk.enc(pb + " " + C + str(xb))
            if fa[:len(ipa)] != ipa or fb[:len(ipb)] != ipb:
                why = "prefix_not_kept"
            elif len(fa) != len(fb):
                why = "full_len_mismatch"
            else:
                diffs = [i for i in range(len(fa)) if fa[i] != fb[i]]
                after = [i for i in diffs if i >= len(ipa)]
                bad = [i for i in diffs if i < len(ipa) and i not in sa]
                if not after or bad:
                    why = "full_outside_unequal"
                elif after[0] <= len(ipa):
                    why = "tstar_at_prompt_end"
                else:
                    ts = after[0]
        if why is not None:
            cnt["assert_fail"][why] = cnt["assert_fail"].get(why, 0) + 1
            continue
        restA, restB = fa[ts:], fb[ts:]
        valid.append({"ds": "VB", "pair_id": r["pair_id"], "idx": int(r["idx"]), "var_name": r["var_name"],
                      "A": fa[:ts] + restA[:-1], "B": fb[:ts] + restB[:-1], "A2": fa[:ts] + restB[:-1],
                      "B2": fb[:ts] + restA[:-1], "restA_seq": restA, "restB_seq": restB, "cont": len(restA) >= 2,
                      "r": ts - 1, "src": sa, "tA": int(restA[0]), "tB": int(restB[0]), "restA": [], "restB": [],
                      "rel": "gsm8k", "cl": ["gsm8k:%d" % r["idx"]],
                      "split": split_hex("k17vb-split:", "%d" % r["idx"]), "plen": len(ipa),
                      "n_chain": ts - len(ipa), "value_A": x, "value_B": xb,
                      "strata": {"pos_len": "1" if len(sa) == 1 else ">=2",
                                 "ST2": "shared" if str(x)[0] == str(xb)[0] else "differs",
                                 "ST3": str(r["n_digits"]) if r["n_digits"] < 3 else ">=3",
                                 "cont": "yes" if len(restA) >= 2 else "no"},
                      "para": [], "sel": {}, "label": r["pair_id"]})
    by = {}
    for v in valid:
        by.setdefault(v["idx"], []).append(v)
    sel = {i: VB.pick_pair(vs) for i, vs in by.items()}
    order = sorted(sel, key=lambda i: sha("k17vb-order:%d" % i))
    out = []
    for k, i in enumerate(order):
        c = sel[i]
        c["order"] = k
        out.append(c)
    cnt.update({"valid_rows": len(valid), "valid_problems": len(by), "selected": len(out),
                "pair_rule": "lowest var_name (VARBENCH v3 / prod DESIGN 12.3)",
                "first60": {"span_ge2": sum(1 for c in out[:60] if len(c["src"]) >= 2),
                            "cont": sum(1 for c in out[:60] if c["cont"]),
                            "T_median": sorted(len(c["A"]) for c in out[:60])[30] if len(out) >= 60 else None}})
    return out, cnt


BUILDERS = (("D1", build_d1), ("CF", build_cf), ("D2", build_d2), ("D4", build_d4), ("D5", build_d5), ("VB", build_vb))


def llama_parity(tk):
    """On the Llama tokenizer the generalised D4/D5 builders must reproduce k16_data's rows (ids, span, tA/tB)."""
    import k16_data as KD6

    class _T(object):
        name, bos = "llama", None

        def __init__(s, t):
            s.t = t

        def enc(s, text, ans=False):
            return tk.enc(text)

        def enc_spans(s, text, ranges):
            return tk.enc_spans(text, ranges)

        def dec(s, ids):
            return tk.dec(ids)
    t2 = _T(tk)
    out = {}
    for name, items, f_old, f_new in (("D4", KD6.d4_items("T1", 40) + KD6.d4_items("T2", 40), KD6.d4_row, d4_row_any),
                                      ("D5", KD6.d5_items(40), KD6.d5_row, d5_row_any)):
        n = same = 0
        for it in items:
            old, _ = f_old(t2, it)
            new, _ = f_new(tk, it)
            if old is None:
                continue
            n += 1
            same += int(new is not None and new["ids_A"] == old["ids_A"] and new["ids_B"] == old["ids_B"]
                        and new["source_span"] == old["source_span"] and new["tA"] == old["answer_first_token_A"]
                        and new["tB"] == old["answer_first_token_B"])
        out[name] = {"k16_rows": n, "identical": same}
    return out


def build_items(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {"tokenizers": {}, "datasets": {}, "source_md5": {m: md5_file(os.path.join(HERE, m)) for m in REUSED}}
    for fam in ("huginn", "llama", "ouro"):
        tk = Tok(fam, LAPTOP_TOK[fam])
        assert tk.sha16 == TOK_SHA16[fam], "%s tokenizer sha %s" % (fam, tk.sha16)
        chk = {s: tk.enc(s) for s in ("The capital of France is", " Paris", " 7", " 45", " 123", "Paris is",
                                     "12 + 25 = 37\n41 + 33 =", "a = 7\nprint(a)")}
        meta["tokenizers"][fam] = {"sha16": tk.sha16, "bos": tk.bos, "checks": chk,
                                   "space_digit": {"' 45'": tk.dec(tk.enc(" 45")), "' 7'": tk.dec(tk.enc(" 7"))}}
        if fam == "llama":
            meta["tokenizers"][fam]["k16_parity"] = llama_parity(tk)
        for ds, fn in BUILDERS:
            if fam == "ouro" and ds in ("CF", "D5", "VB"):
                continue
            t0 = time.time()
            cands, st = fn(tk)
            pos0 = sum(1 for c in cands if 0 in c["src"]) if BOS[fam] is None else 0
            st.update({"seconds": round(time.time() - t0, 1), "pos0_dropped_at_runtime": pos0,
                       "dev_test_unit_overlap": len({u for c in cands if c["split"] == "dev" for u in c["cl"]}
                                                    & {u for c in cands if c["split"] == "test" for u in c["cl"]})})
            p = os.path.join(out_dir, "cands_%s_%s.json" % (fam, ds))
            atomic_json(p, {"ds": ds, "tok": fam, "stats": st, "cands": cands})
            st["file_sha256"] = sha256_file(p)
            meta["datasets"]["%s|%s" % (fam, ds)] = st
            print(fam, ds, json.dumps({k: v for k, v in st.items() if k != "file_sha256"})[:600], flush=True)
    atomic_json(os.path.join(out_dir, "build_meta.json"), meta)
    return meta


# ------------------------------------------------------------------ runners
class Ctx(object):
    def __init__(self, r, edits, e_edits, rec, rec_e, rec_eargs, leaves, steps):
        self.cnt, self.r, self.steps = {}, r, steps
        self.edits, self.e_edits, self.rec = edits or {}, e_edits or {}, rec or {}
        self.rec_e, self.rec_eargs, self.leaves = rec_e, rec_eargs, leaves or {}
        self.H, self.E, self.eargs, self.hout, self.freqs, self.bidx_bad = {}, None, {}, None, None, []
        self.emb = None                     # review 3 F1: token-embedding entries (path-integrated gain only)


class Base(object):
    """Shared site edit / record / leaf logic for every family."""
    ctx = None

    def _emb_hook(self, mod, args, out):
        """Review 3 F1: edit the token-embedding output (before Huginn's emb_scale, which is linear) at listed rows."""
        c = self.ctx
        if c is None or c.emb is None:
            return None
        c.cnt["emb"] = c.cnt.get("emb", 0) + 1
        return apply_entries(out, c.emb)

    def emb_rows(self, ids):
        """Token-embedding rows of ids (the value the embedding module returns for them)."""
        return self.emb_mod.weight[torch.tensor(ids, device=self.emb_mod.weight.device)].detach()

    def _emb_check(self, c):
        if c.emb is not None:
            assert c.cnt.get("emb") == 1, "embedding edit applied %r times" % c.cnt.get("emb")

    def _site(self, c, site, out):
        tup = isinstance(out, tuple)
        o = out[0] if tup else out
        ents = c.edits.get(site)
        if ents:
            o = apply_entries(o, ents)
        lf = c.leaves.get(site)
        if lf is not None:
            r = c.r
            o = torch.cat([o[:, :r], o[:, r:r + 1] + lf, o[:, r + 1:]], 1)
        rp = c.rec.get(site)
        if rp is not None:
            c.H[site] = (o[0] if rp == "all" else o[0, rp]).detach().clone()
        if ents or lf is not None:
            return (o,) + tuple(out[1:]) if tup else o
        return None

    def z32(self, h):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return F.linear(h.float(), self.Wf)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def pres_sites(self):
        return [("c", j, self.C - 1) for j in range(1, self.K + 1)] if self.family != "nr" else \
            [("c", 1, l) for l in range(self.C)]


class RavenR(Base):
    """Huginn / McLeish: hooks R1-R6 (plan section 3). The model is called with num_steps = steps (int) or
    torch.tensor([0, K]) for gradients (raven facts F7); input_states = the fixed per-item initial state (DC4)."""

    def __init__(self, key, model, dev, K):
        self.key, self.m, self.dev, self.K = key, model, dev, K
        self.family = "raven"
        self.name = key if K == KDEF[key] else "%s%d" % (key, K)
        cfg = model.config
        self.P, self.C, self.Cd = int(cfg.n_layers_in_prelude), int(cfg.n_layers_in_recurrent_block), int(cfg.n_layers_in_coda)
        self.d = int(cfg.n_embd)
        self.std = float(cfg.init_values["std"])
        self.emb_scale = float(model.emb_scale)
        self.huginn = key.startswith("huginn")
        self.lnf_calls = 2 if self.huginn else 1
        self.W = model.lm_head.weight
        self.Wf = self.W.detach().float()
        self.dtype = self.W.dtype
        Wa = model.transformer.adapter.weight.detach().float()
        self.W_state, self.W_inp = Wa[:, :self.d], Wa[:, self.d:]
        tr = model.transformer
        self.hs = []
        for l, b in enumerate(tr.core_block):
            self.hs.append(b.register_forward_pre_hook(self._core_pre(l)))
            self.hs.append(b.register_forward_hook(self._core_hook(l)))
        self.hs.append(tr.adapter.register_forward_pre_hook(self._adapter_pre))
        self.hs.append(tr.adapter.register_forward_hook(self._adapter_post))
        for c, b in enumerate(tr.coda):
            self.hs.append(b.register_forward_pre_hook(self._coda_pre(c)))
            self.hs.append(b.register_forward_hook(self._coda_hook(c)))
        self.hs.append(tr.ln_f.register_forward_hook(self._lnf_hook))
        self.hs.append(tr.prelude[0].register_forward_pre_hook(self._prelude_pre))
        self.emb_mod = tr.wte
        self.hs.append(tr.wte.register_forward_hook(self._emb_hook))
        self.x0 = None

    def set_init(self, T, seed):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.empty(1, T, self.d)
        torch.nn.init.trunc_normal_(x, mean=0.0, std=self.std, a=-3 * self.std, b=3 * self.std, generator=g)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        # the bf16-rounded draw is the state for every dtype (bf16 and the fp32 FD check share the same values)
        self.x0 = x.to(torch.bfloat16).to(self.dtype).to(self.dev)
        return x

    def exec_idx(self, j, l):
        return self.P + self.C * (j - 1) + l

    def _prelude_pre(self, mod, args):
        c = self.ctx
        if c is not None:
            c.cnt["prelude0"] = c.cnt.get("prelude0", 0) + 1

    def _core_pre(self, l):
        def h(mod, args):
            c = self.ctx
            if c is None:
                return None
            n = c.cnt.get(("c", l), 0) + 1
            if int(args[2]) != self.exec_idx(n, l):
                c.bidx_bad.append((n, l, int(args[2])))
            if c.freqs is None:
                c.freqs = args[1]
            return None
        return h

    def _core_hook(self, l):
        def h(mod, args, out):
            c = self.ctx
            if c is None:
                return None
            n = c.cnt.get(("c", l), 0) + 1
            c.cnt[("c", l)] = n
            return self._site(c, ("c", n, l), out)
        return h

    def _coda_pre(self, cc):
        def h(mod, args):
            c = self.ctx
            if c is not None and int(args[2]) != -(cc + 1):
                c.bidx_bad.append(("coda", cc, int(args[2])))
        return h

    def _coda_hook(self, cc):
        def h(mod, args, out):
            c = self.ctx
            if c is None:
                return None
            c.cnt[("k", cc)] = c.cnt.get(("k", cc), 0) + 1
            return self._site(c, ("k", cc), out)
        return h

    def _adapter_pre(self, mod, args):
        c = self.ctx
        if c is None:
            return None
        n = c.cnt.get("pre", 0) + 1
        c.cnt["pre"] = n
        xe = args[0]
        if n == 1 and c.rec_e:
            c.E = xe[0, :, self.d:].detach().clone()
        ed = c.e_edits.get(n)
        if ed is not None:
            pos, rows = ed
            xe = xe.clone()                    # never write into the shared input_embeds tensor
            xe[0, pos, self.d:] = rows.to(xe.dtype)
        if c.rec_eargs:
            c.eargs[n] = xe[0, :, self.d:].detach().clone()
        return (xe,) if ed is not None else None

    def _adapter_post(self, mod, args, out):
        c = self.ctx
        if c is None:
            return None
        n = c.cnt.get("post", 0) + 1
        c.cnt["post"] = n
        return self._site(c, ("p", n), out)

    def _lnf_hook(self, mod, args, out):
        c = self.ctx
        if c is None:
            return None
        n = c.cnt.get("lnf", 0) + 1
        c.cnt["lnf"] = n
        if n == self.lnf_calls:
            c.hout = out[0]
        return None

    def _check(self, c, nK):
        bad = [("core", l, c.cnt.get(("c", l))) for l in range(self.C) if c.cnt.get(("c", l)) != nK]
        bad += [(k, c.cnt.get(k)) for k in ("pre", "post") if c.cnt.get(k) != nK]
        bad += [("coda", cc, c.cnt.get(("k", cc))) for cc in range(self.Cd) if c.cnt.get(("k", cc)) != 1]
        if c.cnt.get("lnf") != self.lnf_calls:
            bad.append(("ln_f", c.cnt.get("lnf")))
        if c.cnt.get("prelude0") != 1:
            bad.append(("prelude0", c.cnt.get("prelude0")))
        if c.bidx_bad:
            bad.append(("block_idx", c.bidx_bad[:3]))
        assert not bad, "raven hook call counts / executed index wrong: %s" % bad

    def run(self, ids, r, edits=None, e_edits=None, rec=None, rec_e=False, rec_eargs=False, leaves=None, grad=False,
            cont=False, steps=None, full=False, emb=None):
        x = torch.tensor([ids], device=self.dev)
        nK = steps or self.K
        assert self.x0 is not None and self.x0.shape[1] >= len(ids), "initial state not set or too short"
        x0 = self.x0[:, :len(ids)]
        c = Ctx(r, edits, e_edits, rec, rec_e, rec_eargs, leaves, nK)
        c.emb = emb
        self.ctx = c
        try:
            with (torch.enable_grad() if grad else torch.no_grad()):
                ns = torch.tensor([0, nK]) if grad else int(nK)
                self.m(input_ids=x, input_states=x0, num_steps=ns, use_cache=False)
                z_r = self.z32(c.hout[r:r + 1])[0]
                z_cont = self.z32(c.hout[r + 1:]).detach() if cont else None
                z_full = self.z32(c.hout).detach() if full else None
        finally:
            self.ctx = None
        self._check(c, nK)
        self._emb_check(c)
        return {"z_r": z_r if grad else z_r.detach(), "z_cont": z_cont, "z": z_full, "H": c.H, "E": c.E,
                "eargs": c.eargs, "freqs": c.freqs}

    @torch.no_grad()
    def readout(self, h, freqs, r=None):
        """R7: the forward's own path from a loop-end state h [T, d]: (ln_f on Huginn), coda, ln_f, float32 head.
        r given -> the logits at r only, with the same matmul shape as run() (bit-identity)."""
        assert self.ctx is None
        tr = self.m.transformer
        x = h[None]
        if self.huginn:
            x = tr.ln_f(x)
        bidx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for blk in tr.coda:
            bidx -= 1
            x = blk(x, freqs, bidx, None, None)
        x = tr.ln_f(x)
        return self.z32(x[0]) if r is None else self.z32(x[0, r:r + 1])[0]

    def all_sites(self):
        return ([("p", j) for j in range(1, self.K + 1)] + [("c", j, l) for j in range(1, self.K + 1) for l in range(self.C)]
                + [("k", c) for c in range(self.Cd)])

    def feed_site(self):
        return ("k", self.Cd - 1)


class LlamaR(Base):
    """The non-recurrent Llama (NR): forward hooks on model.model.layers[l]; readout = model.norm output at r."""

    def __init__(self, model, dev):
        self.m, self.dev, self.family, self.name, self.key = model, dev, "nr", "llama_nr", "llama_nr"
        self.K, self.C, self.Cd, self.P = 1, int(model.config.num_hidden_layers), 0, 0
        self.d = int(model.config.hidden_size)
        self.W = model.lm_head.weight
        self.Wf = self.W.detach().float()
        self.dtype = self.W.dtype
        self.hs = [ly.register_forward_hook(self._layer(l)) for l, ly in enumerate(model.model.layers)]
        self.hs.append(model.model.norm.register_forward_hook(self._norm))
        self.emb_mod = model.model.embed_tokens
        self.hs.append(self.emb_mod.register_forward_hook(self._emb_hook))

    def set_init(self, T, seed):
        return None

    def _layer(self, l):
        def h(mod, args, out):
            c = self.ctx
            if c is None:
                return None
            c.cnt[l] = c.cnt.get(l, 0) + 1
            return self._site(c, ("c", 1, l), out)
        return h

    def _norm(self, mod, args, out):
        c = self.ctx
        if c is None:
            return None
        c.cnt["norm"] = c.cnt.get("norm", 0) + 1
        c.hout = out[0]
        return None

    def run(self, ids, r, edits=None, e_edits=None, rec=None, rec_e=False, rec_eargs=False, leaves=None, grad=False,
            cont=False, steps=None, full=False, emb=None):
        assert not e_edits
        x = torch.tensor([ids], device=self.dev)
        c = Ctx(r, edits, None, rec, False, False, leaves, 1)
        c.emb = emb
        self.ctx = c
        try:
            with (torch.enable_grad() if grad else torch.no_grad()):
                self.m(input_ids=x, use_cache=False)
                z_r = self.z32(c.hout[r:r + 1])[0]
                z_cont = self.z32(c.hout[r + 1:]).detach() if cont else None
                z_full = self.z32(c.hout).detach() if full else None
        finally:
            self.ctx = None
        bad = [l for l in range(self.C) if c.cnt.get(l) != 1] + ([] if c.cnt.get("norm") == 1 else ["norm"])
        assert not bad, "llama hook counts wrong: %s" % bad
        self._emb_check(c)
        return {"z_r": z_r if grad else z_r.detach(), "z_cont": z_cont, "z": z_full, "H": c.H, "E": None, "eargs": {},
                "freqs": None}

    def all_sites(self):
        return [("c", 1, l) for l in range(self.C)]

    def feed_site(self):
        return ("c", 1, self.C - 1)


class OuroR(Base):
    """Ouro-Thinking: layer forward hooks (l < L-1) and the model.norm hook (l = L-1, carried state); readout =
    float32 head on the norm output of each loop (trained exits)."""

    def __init__(self, model, dev, name="ouro_think"):
        self.m, self.dev, self.family, self.name, self.key = model, dev, "ouro", name, name
        self.inner = model.model
        self.C = int(model.config.num_hidden_layers)
        self.K = int(self.inner.total_ut_steps)
        self.Cd, self.P = 0, 0
        self.d = int(model.config.hidden_size)
        self.W = model.lm_head.weight
        self.Wf = self.W.detach().float()
        self.dtype = self.W.dtype
        self.hs = [self.inner.layers[l].register_forward_hook(self._layer(l)) for l in range(self.C - 1)]
        self.hs.append(self.inner.norm.register_forward_hook(self._norm))

    def set_init(self, T, seed):
        return None

    def _layer(self, l):
        def h(mod, args, out):
            c = self.ctx
            if c is None:
                return None
            n = c.cnt.get(l, 0) + 1
            c.cnt[l] = n
            return self._site(c, ("c", n, l), out)
        return h

    def _norm(self, mod, args, out):
        c = self.ctx
        if c is None:
            return None
        n = c.cnt.get("norm", 0) + 1
        c.cnt["norm"] = n
        res = self._site(c, ("c", n, self.C - 1), out)
        o = out if res is None else res
        o = o[0] if isinstance(o, tuple) else o
        c.loops = getattr(c, "loops", {})
        c.loops[n] = o[0]
        return res

    def run(self, ids, r, edits=None, e_edits=None, rec=None, rec_e=False, rec_eargs=False, leaves=None, grad=False,
            cont=False, steps=None, full=False, emb=None):
        assert not e_edits and emb is None
        x = torch.tensor([ids], device=self.dev)
        c = Ctx(r, edits, None, rec, False, False, leaves, self.K)
        self.ctx = c
        try:
            with (torch.enable_grad() if grad else torch.no_grad()):
                self.m(input_ids=x, use_cache=False)
                hK = c.loops[self.K]
                z_r = self.z32(hK[r:r + 1])[0]
                z_loops = [self.z32(c.loops[j][r:r + 1])[0].detach() for j in range(1, self.K + 1)]
                z_cont = self.z32(hK[r + 1:]).detach() if cont else None
                z_full = self.z32(hK).detach() if full else None
        finally:
            self.ctx = None
        bad = [l for l in range(self.C - 1) if c.cnt.get(l) != self.K] + ([] if c.cnt.get("norm") == self.K else ["norm"])
        assert not bad, "ouro hook counts wrong: %s" % bad
        return {"z_r": z_r if grad else z_r.detach(), "z_cont": z_cont, "z": z_full, "H": c.H, "E": None, "eargs": {},
                "freqs": None, "z_loops": z_loops}

    def all_sites(self):
        return [("c", j, l) for j in range(1, self.K + 1) for l in range(self.C)]

    def feed_site(self):
        return ("c", self.K, self.C - 1)


def load_raven(key, dev, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    p = snap(key)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(p, dtype=dtype, trust_remote_code=True,
                                             local_files_only=True).to(dev).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    noise = float(getattr(m.config, "test_time_noise", 0) or 0)
    assert noise == 0.0, "test_time_noise %r" % noise
    return m, {"snapshot": CKPT[key][1][:8], "load_seconds": round(time.time() - t0, 1),
               "modeling_sha16": sha256_file(os.path.join(p, "raven_modeling_minimal.py"))[:16],
               "tokenizer_sha16": sha256_file(os.path.join(p, "tokenizer.json"))[:16],
               "config_sha16": sha256_file(os.path.join(p, "config.json"))[:16]}


def load_hf(key, dev, dtype=torch.bfloat16, **kw):
    from transformers import AutoModelForCausalLM
    p = snap(key)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(p, dtype=dtype, local_files_only=True, **kw).to(dev).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, {"snapshot": CKPT[key][1][:8], "load_seconds": round(time.time() - t0, 1),
               "tokenizer_sha16": sha256_file(os.path.join(p, "tokenizer.json"))[:16],
               "config_sha16": sha256_file(os.path.join(p, "config.json"))[:16]}


# ------------------------------------------------------------------ items at runtime
class HarnessFail(Exception):
    pass


def sync(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def prep_item(c, bos):
    """Stored candidate (no BOS) -> runtime item with BOS prepended and positions shifted (DC3); None if the edited
    span includes position 0 on a no-BOS model (P5). Paraphrase forms with a position-0 span are dropped then."""
    sh = 0 if bos is None else 1
    pre = [] if bos is None else [int(bos)]
    if bos is None and 0 in c["src"]:
        return None
    it = {k: c[k] for k in ("ds", "pair_id", "tA", "tB", "rel", "cl", "split", "strata", "label", "restA", "restB")
          if k in c}
    it["cluster"] = tuple(c["cl"]) if len(c["cl"]) > 1 else c["cl"][0]
    it["A"], it["B"] = pre + c["A"], pre + c["B"]
    it["r"] = c["r"] + sh
    it["src"] = [p + sh for p in c["src"]]
    seqs = [it["A"], it["B"]]
    if c["ds"] == "VB":
        it["A2"], it["B2"] = pre + c["A2"], pre + c["B2"]
        it["restA_seq"], it["restB_seq"], it["cont"] = c["restA_seq"], c["restB_seq"], c["cont"]
        seqs += [it["A2"], it["B2"]]
    if c.get("restA") or c.get("restB"):
        it["FA"] = it["A"] + [c["tA"]] + c["restA"][:-1] if c["restA"] else None
        it["FB"] = it["B"] + [c["tB"]] + c["restB"][:-1] if c["restB"] else None
        seqs += [s for s in (it["FA"], it["FB"]) if s]
    paras = []
    for f in c.get("para", []):
        if bos is None and 0 in f["src"]:
            continue
        paras.append({"fmt": f["fmt"], "A": pre + f["A"], "B": pre + f["B"], "src": [p + sh for p in f["src"]],
                      "r": f["r"] + sh})
        seqs += [paras[-1]["A"], paras[-1]["B"]]
    it["para"] = paras
    it["Tmax"] = max(len(s) for s in seqs)
    return it


class Item(object):
    """Clean A/B runs (records at every site, e) and the row runner for one item on one model (plan sections 5-7)."""

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
        self.mA, self.mB = float(zA[self.tB] - zA[self.tA]), float(zB[self.tB] - zB[self.tA])
        self.D = self.mB - self.mA
        okA, okB = int(zA.argmax()) == self.tA, int(zB.argmax()) == self.tB
        st = "A_fail" if not okA else ("B_fail" if not okB else ("D_small" if self.D < GAP else "kept"))
        self.full_answer = None
        if st == "kept" and (it.get("FA") or it.get("FB")):
            self.full_answer = self._full_answer()
            if not all(self.full_answer.values()):
                st = "full_answer_fail"
        self.status = st
        self.rows = {}
        self.out = {k: it[k] for k in ("ds", "pair_id", "cluster", "split", "rel", "strata", "label", "r", "src", "tA",
                                       "tB", "Tmax") if k in it}
        self.out.update({"seed": self.seed, "seed_off": seed_off, "T": len(it["A"]), "clean_A": vals(zA, self.tA, self.tB),
                         "clean_B": vals(zB, self.tA, self.tB), "D": r5(self.D), "status": self.status,
                         "full_answer": self.full_answer, "t_fwd": round(self.t_fwd, 4), "rows": self.rows})

    def _full_answer(self):
        """Critic F10: teacher-forced greedy check of the remaining answer digits on A and B."""
        out = {}
        for side, seq, rest in (("A", self.it.get("FA"), self.it["restA"]), ("B", self.it.get("FB"), self.it["restB"])):
            if not seq:
                continue
            o = self.R.run(seq, self.r, full=True)
            out[side] = all(int(o["z"][self.r + 1 + k].argmax()) == int(t) for k, t in enumerate(rest))
        return out

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

    def e_ed(self, loops, pos, n=None):
        """B's e rows at pos at adapter calls `loops`; n given -> e_A + n (e_B - e_A) (critic 2 F3: one call carrying
        the summed input delta of a persistent edit, n = K - j + 1; off-manifold for large n)."""
        t = self._t(pos)
        if n is None:
            rows = self.cB["E"][pos]
        else:
            eA = self.cA["E"][pos].float()
            rows = eA + float(n) * (self.cB["E"][pos].float() - eA)
        return {j: (t, rows) for j in loops}

    def e_rnd(self, loops, pos):
        """Random input control (critic 2 F5): e_A + a unit Gaussian per position scaled to |e_B - e_A| there."""
        eA = self.cA["E"][pos].float()
        dn = (self.cB["E"][pos].float() - eA).norm(dim=-1).tolist()
        d = rnd_delta(self.it["pair_id"], "e", "src", pos, dn, eA.shape[-1]).to(self.R.dev)
        return {j: (self._t(pos), eA + d) for j in loops}

    @staticmethod
    def lp(o, rest):
        s = logp(o["z_r"], rest[0])
        for k in range(1, len(rest)):
            s += logp(o["z_cont"][k - 1], rest[k])
        return s

    def finish_clean_vb(self):
        it, R = self.it, self.R
        if it["cont"]:
            lpA_A, lpB_B = self.lp(self.cA, it["restA_seq"]), self.lp(self.cB, it["restB_seq"])
            a2, b2 = R.run(it["A2"], self.r, cont=True), R.run(it["B2"], self.r, cont=True)
            lpB_A, lpA_B = self.lp(a2, it["restB_seq"]), self.lp(b2, it["restA_seq"])
            self.out["causality_ok"] = bool(torch.equal(a2["z_r"], self.cA["z_r"]) and torch.equal(b2["z_r"], self.cB["z_r"]))
        else:
            lpA_A, lpB_B = logp(self.cA["z_r"], it["restA_seq"][0]), logp(self.cB["z_r"], it["restB_seq"][0])
            lpB_A, lpA_B = logp(self.cA["z_r"], it["restB_seq"][0]), logp(self.cB["z_r"], it["restA_seq"][0])
        self.LO_A, self.LO_B = lpB_A - lpA_A, lpB_B - lpA_B
        self.out.update({"LO_A": r5(self.LO_A), "LO_B": r5(self.LO_B), "D_LO": r5(self.LO_B - self.LO_A),
                         "cont": it["cont"], "n_rest": len(it["restA_seq"])})

    def depth_margins(self):
        """Clean A/B depth-j margins (raven: coda readout of the loop-j state; Ouro: trained exits)."""
        R, r = self.R, self.r
        if R.family == "nr":
            return
        la, lb = [], []
        if R.family == "ouro":
            for j in range(R.K):
                la.append(vals(self.cA["z_loops"][j], self.tA, self.tB))
                lb.append(vals(self.cB["z_loops"][j], self.tA, self.tB))
            self.out["readout_exact_K"] = bool(torch.equal(self.cA["z_loops"][-1], self.cA["z_r"]))
        else:
            L = R.C - 1
            for j in range(1, R.K + 1):
                za = R.readout(self.cA["H"][("c", j, L)], self.cA["freqs"], r)
                zb = R.readout(self.cB["H"][("c", j, L)], self.cB["freqs"], r)
                la.append(vals(za, self.tA, self.tB))
                lb.append(vals(zb, self.tA, self.tB))
                if j == R.K:
                    self.out["readout_exact_K"] = bool(torch.equal(za, self.cA["z_r"]) and torch.equal(zb, self.cB["z_r"]))
        self.out["loopsA"], self.out["loopsB"] = la, lb
        if not self.out["readout_exact_K"]:
            raise HarnessFail("per-loop readout at j=K differs from the model logits: %s" % self.it["pair_id"])

    def deltas(self):
        """P1 / critic F6: |H^B - H^A| per source position at every E1 site and |delta e|, |W_e delta e| (the
        post-adapter size of one input-edit call), |W_s delta h| for source state edits at Llast@j (raven)."""
        R, src, out = self.R, self.it["src"], {}
        for s in R.all_sites():
            if s in self.cA["H"]:
                out[skey(s)] = r5((self.cB["H"][s][src] - self.cA["H"][s][src]).float().norm(dim=-1).sum())
        if R.family == "raven":
            de = (self.cB["E"][src] - self.cA["E"][src]).float()
            out["e"] = r5(de.norm(dim=-1).sum())
            out["We_e"] = r5((de @ R.W_inp.T).norm(dim=-1).sum())
            down = [p for p in range(len(self.it["A"])) if p > max(src)]
            dd = (self.cB["E"][down] - self.cA["E"][down]).float()
            out["We_e_down"] = r5((dd @ R.W_inp.T).norm(dim=-1).sum())
            for j in JSET[R.K]:
                s = ("c", j, R.C - 1)
                dh = (self.cB["H"][s][src] - self.cA["H"][s][src]).float()
                out["Ws_" + skey(s)] = r5((dh @ R.W_state.T).norm(dim=-1).sum())
        self.out["dnorm"] = out

    def row(self, key, edits=None, e_edits=None, lo=False, exact_vs=None, eown=None, rec_eargs=False, keep=False):
        if key in self.rows:
            return None
        it, R, r = self.it, self.R, self.r
        ps = R.pres_sites()
        rec = {s: [r] for s in ps}
        cont = bool(lo and self.vb and it["cont"])
        o = R.run(it["A"], r, edits=edits, e_edits=e_edits, rec=rec, cont=cont, rec_eargs=rec_eargs)
        rv = {"f": vals(o["z_r"], self.tA, self.tB)}
        rv["pres"] = [r5(v) for v in proj_many([o["H"][s][0] for s in ps], [self.cA["H"][s][r] for s in ps],
                                                [self.cB["H"][s][r] for s in ps])]
        if R.family == "ouro":
            rv["loops"] = [r5(z[self.tB] - z[self.tA]) for z in o["z_loops"]]
        if lo and self.vb:
            if cont:
                lpA = self.lp(o, it["restA_seq"])
                o2 = R.run(it["A2"], r, edits=edits, e_edits=e_edits, cont=True)
                lpB = self.lp(o2, it["restB_seq"])
                rv["caus"] = bool(torch.equal(o2["z_r"], o["z_r"]))
            else:
                lpA, lpB = logp(o["z_r"], it["restA_seq"][0]), logp(o["z_r"], it["restB_seq"][0])
            rv["lo"] = r5(lpB - lpA)
        if exact_vs is not None:
            rv["exact"] = bool(torch.equal(o["z_r"], exact_vs))
        if eown is not None and R.family == "raven":
            j = eown
            s = ("c", j, R.C - 1)
            h = self.cA["H"][s].clone()
            h[r] = o["H"][s][0]
            z = R.readout(h, self.cA["freqs"], r)
            rv["mj"] = r5(z[self.tB] - z[self.tA])
        self.rows[key] = rv
        return o if keep else None


# ------------------------------------------------------------------ arms
def vb_inp(K):
    """IN-persist(j) loops on VB: 1, K/2 and (review 3 F8) K-1."""
    return (1, K // 2, K - 1)


def run_controls(X, subset):
    """PC-final, SZ (every item) and PC-site (subset) [harness]: any inexact row stops the run."""
    R, r, src = X.R, X.r, X.it["src"]
    fs = R.feed_site()
    X.row("PCF", edits=X.o1(fs, [r]), exact_vs=X.cB["z_r"], lo=X.vb)
    X.row("SZ|O1", edits=X.o1(fs, src), exact_vs=X.cA["z_r"])
    X.row("SZ|RND", edits=X.rnd(fs, src, "src"), exact_vs=X.cA["z_r"])
    bad = [k for k in ("PCF", "SZ|O1", "SZ|RND") if not X.rows[k]["exact"]]
    if R.family == "raven":
        X.row("SC|p%d" % R.K, edits=X.o1(("p", R.K), X.allpos()), exact_vs=X.cB["z_r"])
        X.row("SC|c%d.%d" % (R.K, R.C - 1), edits=X.o1(("c", R.K, R.C - 1), X.allpos()), exact_vs=X.cB["z_r"])
        bad += [k for k in ("SC|p%d" % R.K, "SC|c%d.%d" % (R.K, R.C - 1)) if not X.rows[k]["exact"]]
    if subset:
        allp = X.allpos()
        for name in e1_names(R.family, R.K, X.it["ds"]):
            s = name2site(name, R.C)
            k = "PCS|%s" % skey(s)
            ee = X.e_ed(range(s[1] + 1, R.K + 1), allp) if R.family == "raven" else None
            X.row(k, edits=X.o1(s, allp), e_edits=ee, exact_vs=X.cB["z_r"])
            if not X.rows[k]["exact"]:
                bad.append(k)
    if bad:
        raise HarnessFail("harness rows not exact on %s %s: %s" % (R.name, X.it["pair_id"], bad))


def run_e1(X, tiny=False):
    """E1 placement: O1 source and answer at every site, RND at every site (critic 2 F5), answer a = 0.25 at the class
    sites, and on the first N_TINY items the tiny-edit floor row (a = 1e-3, critic 2 F13) at the same sites."""
    R, r, src, vb, ds = X.R, X.r, X.it["src"], X.vb, X.it["ds"]
    rn, an = set(rnd_names(R.family, R.K, ds)), set(a025_names(R.family, R.K, ds))
    for name in e1_names(R.family, R.K, ds):
        s = name2site(name, R.C)
        k = skey(s)
        j = s[1]
        eo = j if (R.family == "raven" and j < R.K and s[0] in ("p", "c")) else None
        X.row("O1|src|%s|1" % k, edits=X.o1(s, src), lo=vb)
        X.row("O1|ans|%s|1" % k, edits=X.o1(s, [r]), lo=vb, eown=eo)
        if name in an:
            X.row("O1|ans|%s|0.25" % k, edits=X.o1(s, [r], 0.25), eown=eo)
            if tiny:
                key = "O1|ans|%s|%g" % (k, TINY_A)
                X.row(key, edits=X.o1(s, [r], TINY_A))
                hA, hB = X.cA["H"][s][[r]], X.cB["H"][s][[r]]
                X.rows[key]["changed"] = r5((torch.lerp(hA, hB, TINY_A) != hA).float().mean())
        if name in rn:
            X.row("RND|src|%s" % k, edits=X.rnd(s, src, "src"), lo=vb)
            X.row("RND|ans|%s" % k, edits=X.rnd(s, [r], "ans"))


def run_pb8(X):
    """Critic 2 F10 (Huginn, PB8): source O1 at Llast@(K-r), r in PB8_R, at every K run (8, 16, 32)."""
    R, src = X.R, X.it["src"]
    for rr in PB8_R:
        j = R.K - rr
        if j >= 1:
            X.row("O1|src|%s|1" % skey(("c", j, R.C - 1)), edits=X.o1(("c", j, R.C - 1), src))


def run_e5(X, subset=False):
    """Raven only (A5): single, persistent, downstream-persistent input edits, SC at every Llast@j (critic F3),
    SRCIN = source O1 at Llast@j plus B's e at calls j+1..K (critic F8 decomposition)."""
    R, src, allp, K = X.R, X.it["src"], X.allpos(), X.R.K
    vb = X.vb
    in1 = (K - 1, K) if vb else IN1_J[K]
    inp = vb_inp(K) if vb else INP_J[K]          # review 3 F8: VB runs IN-persist(K-1) too (P13, E5 rule at j = K-1)
    for j in in1:
        X.row("IN1|%d" % j, e_edits=X.e_ed([j], src), lo=vb)
    for j in inp:
        X.row("INP|%d" % j, e_edits=X.e_ed(range(j, K + 1), src), lo=vb)
    for j in ((K // 2, K - 1) if vb else IN1S_J[K]):          # critic 2 F3: absolute matching to IN-persist(j)
        X.row("IN1S|%d" % j, e_edits=X.e_ed([j], src, n=K - j + 1), lo=vb)
    X.row("RNDIN|1", e_edits=X.e_rnd(range(1, K + 1), src), lo=vb)   # critic 2 F5: input-edit floor
    if not vb:
        down = [p for p in allp if p > max(src)]
        X.row("INPD|1", e_edits=X.e_ed(range(1, K + 1), down))
    bad = []
    for name in sc_names(R.family, K, X.it["ds"]):               # SC at Llast@j, j < K (critic 2 F7)
        s = name2site(name, R.C)
        o = X.row("SC|%s" % skey(s), edits=X.o1(s, allp), lo=vb, keep=True)
        if subset and o is not None:                             # SC identity: every site of loop j gives SC(Llast@j)
            j = s[1]
            for s2 in [("p", j), ("c", j, R.C // 2)] + ([("c", j, 0)] if j == 1 else []):
                kk = "SCID|%s" % skey(s2)
                X.row(kk, edits=X.o1(s2, allp), exact_vs=o["z_r"])
                if not X.rows[kk]["exact"]:
                    bad.append(kk)
    if bad:
        raise HarnessFail("SC identity rows not exact on %s %s: %s" % (R.name, X.it["pair_id"], bad))
    if not vb:
        for j in SRCIN_J[K]:
            s = ("c", j, R.C - 1)
            X.row("SRCIN|%d" % j, edits=X.o1(s, src), e_edits=X.e_ed(range(j + 1, K + 1), src))


def run_para(X, force=False):
    """O2 aligned paraphrase vectors (P4) and the presence validation from the first form that passes."""
    R, it, r = X.R, X.it, X.r
    names = e1_names(R.family, R.K, it["ds"])
    sites = [name2site(n, R.C) for n in names]
    if R.family == "raven":
        o2s = ["Llast@1", "Llast@%d" % (R.K - 1), "Llast@%d" % R.K, "post_inj@%d" % R.K]
        o2a = ["Llast@1", "Llast@%d" % (R.K - 1), "Lmid@%d" % R.K, "Llast@%d" % R.K]
    elif R.family == "nr":
        o2s = o2a = ["layer@3", "layer@7", "layer@11", "layer@14"]
    else:
        o2s = o2a = ["Llast@1", "Llast@3", "Lmid@4"]
    o2s = [n for n in o2s if n in names]
    o2a = [n for n in o2a if n in names]
    tried = []
    for f in it["para"]:
        rec = {s: "all" for s in sites}
        pa, pb = R.run(f["A"], f["r"], rec=rec), R.run(f["B"], f["r"], rec=rec)
        ok = (int(pa["z_r"].argmax()) == X.tA and int(pb["z_r"].argmax()) == X.tB) or force
        tried.append([f["fmt"], bool(ok)])
        if not ok:
            continue
        X.out["para_fmt"] = f["fmt"]
        X.out["pval"] = {skey(s): [r5(proj(pa["H"][s][f["r"]], X.cA["H"][s][r], X.cB["H"][s][r])),
                                   r5(proj(pb["H"][s][f["r"]], X.cA["H"][s][r], X.cB["H"][s][r]))] for s in sites}
        diag = {}
        for pos_name, nl in (("src", o2s), ("ans", o2a)):
            for name in nl:
                s = name2site(name, R.C)
                pp, po = (f["src"], it["src"]) if pos_name == "src" else ([f["r"]], [r])
                v = pb["H"][s][pp] - pa["H"][s][pp]
                own = X.cB["H"][s][po] - X.cA["H"][s][po]
                diag["%s|%s" % (pos_name, skey(s))] = [
                    r5(F.cosine_similarity(v.flatten().float(), own.flatten().float(), 0)),
                    r5(v.float().norm() / own.float().norm().clamp_min(1e-12))]
                X.row("O2|%s|%s" % (pos_name, skey(s)), edits={s: [("add", X._t(po), v, 1.0)]})
        X.out["o2diag"] = diag
        break
    X.out["para_tried"] = tried


def grad_gain(X, sites, emb_a=None):
    """One forward and backward of m = z(tB) - z(tA) at r on prompt A with zero leaves at r at every site. emb_a None:
    the clean A run (first-order carry gain G). emb_a = a: the source-span token embeddings are set to
    lerp(emb_A, emb_B, a) first (review 3 F1, path-integrated gain on the input path; a = 0 is the clean A run bit for
    bit, a = 1 is the clean B run). -> ({site: grad [d] float32}, z_r detached)."""
    R, it, r = X.R, X.it, X.r
    leaves = {s: torch.zeros(1, 1, R.d, device=R.dev, dtype=R.dtype, requires_grad=True) for s in sites}
    emb = None
    if emb_a is not None:
        src = it["src"]
        emb = [("set", X._t(src), R.emb_rows([it["B"][p] for p in src]), float(emb_a))]
    o = R.run(it["A"], r, leaves=leaves, grad=True, emb=emb)
    m = o["z_r"][X.tB] - o["z_r"][X.tA]
    m.backward()
    out = {s: leaves[s].grad[0, 0].float() for s in sites}
    z = o["z_r"].detach()
    del leaves, o, m
    return out, z


def run_e8(X, full_search):
    """Carry gain (A2, A6, A7; plan section 7): one forward + backward with zero leaves at r at every site of S+,
    verify rows at the top/median/bottom 3 of S and of S+ and a random site of S at both doses; on dev items with
    full_search, O1 at a = 1 at every site of S+ (critic F4: hit and chance over the full set)."""
    R, it, r = X.R, X.it, X.r
    S, Sp = e8_sets(R.K, R.C)
    if R.family == "nr":
        Sp = S
    sync(R.dev)
    t0 = time.time()
    grads, z0 = grad_gain(X, Sp)
    sync(R.dev)
    X.out["t_grad"] = round(time.time() - t0, 4)
    dH = {s: (X.cB["H"][s][r] - X.cA["H"][s][r]).float() for s in Sp}
    G = {skey(s): r5(grads[s] @ dH[s]) for s in Sp}
    X.out["grad_identical"] = bool(torch.equal(z0, X.cA["z_r"]))
    # review 3 F1 (PLAN G4): path-integrated gain on the input path, PIG(s) = mean_a grad_a(s) . dH_s, a in PIG_A;
    # Ga keeps every dose's term so other quadratures can be read later without rerunning
    t1 = time.time()
    acc = {s: torch.zeros_like(dH[s]) for s in Sp}
    Ga = {}
    pig_a1 = None
    for a in PIG_A:
        ga, za = grad_gain(X, Sp, emb_a=a)
        for s in Sp:
            acc[s] += ga[s]
        Ga["%g" % a] = {skey(s): r5(ga[s] @ dH[s]) for s in Sp}
        if a == 1.0:
            pig_a1 = bool(torch.equal(za, X.cB["z_r"]))
        del ga
    sync(R.dev)
    PIG = {skey(s): r5((acc[s] / len(PIG_A)) @ dH[s]) for s in Sp}
    nsrc = [p for p in range(len(it["A"])) if p not in it["src"]]
    X.out.update({"PIG": PIG, "Ga": Ga, "t_pig": round(time.time() - t1, 4), "pig_a1_equals_B": pig_a1,
                  "pig_ids_differ_only_at_src": all(it["A"][p] == it["B"][p] for p in nsrc)})
    del acc, grads
    GS = {skey(s): G[skey(s)] for s in S}
    ksS, verS = verify_sites(GS)
    _, verP = verify_sites(G)
    pig_top = sorted((skey(s) for s in S), key=lambda k: (-PIG[k], k))[:N_PIG_VER]
    rnd = ksS[int(sha(it["pair_id"] + "|rnd_site")[:8], 16) % len(ksS)]
    X.out.update({"G": G, "ver_S": verS, "ver_Sp": verP, "pig_top": pig_top, "rnd_site": rnd,
                  "full_search": bool(full_search)})
    todo = []
    for k in verS + [x for x in verP if x not in verS] + pig_top + [rnd]:
        if k not in todo:
            todo.append(k)
    for a in A_VERIFY:
        for k in todo:
            X.row("O1|ans|%s|%g" % (k, a), edits=X.o1(ksite(k), [r], a))
    if full_search:
        for s in Sp:
            X.row("O1|ans|%s|1" % skey(s), edits=X.o1(s, [r]))


def dm_of(x, key):
    rv = x.get("rows", {}).get(key)
    if rv is None:
        return None
    return (rv["f"][1] - rv["f"][0]) - (x["clean_A"][1] - x["clean_A"][0])


def Dgap(x):
    return (x["clean_B"][1] - x["clean_B"][0]) - (x["clean_A"][1] - x["clean_A"][0])


def dev_tuned_site(items, keys):
    """argmax over sites of mean dm / D on dev items with a full search and D >= 2 (VARBENCH v3 MS7)."""
    best, bv, table = None, None, {}
    for k in keys:
        xs = [dm_of(x, "O1|ans|%s|1" % k) / Dgap(x) for x in items
              if x.get("full_search") and x["split"] == "dev" and Dgap(x) >= GAP and dm_of(x, "O1|ans|%s|1" % k) is not None]
        if not xs:
            continue
        v = sum(xs) / len(xs)
        table[k] = round(v, 4)
        if bv is None or v > bv:
            best, bv = k, v
    return best, table


def dev_tuned_pass(R, res, cmap):
    """Pass 2: the dev-tuned site (from dev items) at both doses on test items that lack it."""
    items = res["items"]
    S, Sp = e8_sets(R.K, R.C)
    if R.family == "nr":
        Sp = S
    out, n = {}, 0
    for name, keys in (("S", [skey(s) for s in S]), ("Sp", [skey(s) for s in Sp])):
        best, table = dev_tuned_site(items, keys)
        out[name] = {"site": best, "n_dev": sum(1 for x in items if x["split"] == "dev" and x.get("full_search")),
                     "table": table}
        if best is None:
            continue
        s = ksite(best)
        for x in items:
            if x["split"] != "test" or not x.get("G"):
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


# ------------------------------------------------------------------ hook checks (plan section 3, a-k)
def check_hooks(R, it):
    out = {"model": R.name, "item": it["pair_id"]}
    X = Item(R, it)
    A, r, src = it["A"], it["r"], it["src"]
    T = len(A)
    allp = list(range(T))
    tall = X._t(allp)
    ref = R.run(A, r, full=True)
    out["a_counts_asserted_in_every_forward"] = True
    sites = R.all_sites()
    zero = {s: [("add", tall, torch.zeros(T, R.d, device=R.dev), 1.0)] for s in sites}
    e0 = {j: (tall, X.cA["E"]) for j in range(1, R.K + 1)} if R.family == "raven" else None
    o = R.run(A, r, edits=zero, e_edits=e0, full=True)
    out["b_zero_edit_identity"] = bool(torch.equal(o["z"], ref["z"]))
    o = R.run(A, r, full=True)
    out["c_repeat_identical"] = bool(torch.equal(o["z"], ref["z"]))
    jm = max(1, R.K // 2)
    s_ed, p = ("c", jm, R.C // 2), src[0]
    rec = {s: "all" for s in sites}
    oe = R.run(A, r, edits=X.o1(s_ed, [p]), rec=rec)
    o_ed = exec_order(s_ed, R.C, R.K)
    earlier = [s for s in sites if exec_order(s, R.C, R.K) < o_ed]
    later = [s for s in sites if exec_order(s, R.C, R.K) > o_ed]
    oth = [q for q in allp if q != p]
    out["d_edit_locality"] = {
        "earlier_identical": all(torch.equal(oe["H"][s], X.cA["H"][s]) for s in earlier),
        "later_prefix_identical": all(torch.equal(oe["H"][s][:p], X.cA["H"][s][:p]) for s in later),
        "at_site": bool(torch.equal(oe["H"][s_ed][p], X.cB["H"][s_ed][p]) and torch.equal(oe["H"][s_ed][oth], X.cA["H"][s_ed][oth])),
        "later_changed_from_p": any(not torch.equal(oe["H"][s][p:], X.cA["H"][s][p:]) for s in later),
        "n_earlier": len(earlier), "n_later": len(later), "site": skey(s_ed)}
    S, Sp = e8_sets(R.K, R.C)
    if R.family == "nr":
        Sp = S
    leaves = {s: torch.zeros(1, 1, R.d, device=R.dev, dtype=R.dtype, requires_grad=True) for s in Sp}
    og = R.run(A, r, leaves=leaves, grad=True, full=True)
    (og["z_r"][it["tB"]] - og["z_r"][it["tA"]]).backward()
    out["e_grad_run_identical"] = bool(torch.equal(og["z"], ref["z"]))
    out["e_grad_nonzero_sites"] = sum(1 for s in Sp if float(leaves[s].grad.float().abs().sum()) > 0)
    out["e_grad_n_sites"] = len(Sp)
    del leaves, og
    ok_g = True
    if R.family == "raven":
        p_ = snap(R.key)
        mf = os.path.join(p_, "raven_modeling_minimal.py")
        out["f_modeling_sha16"] = sha256_file(mf)[:16] if os.path.exists(mf) else None
        out["f_ok"] = (out["f_modeling_sha16"] == MODELING_SHA16[R.key]) if out["f_modeling_sha16"] else None
        g = {}
        for j in sorted({1, R.K // 2, R.K}):
            oj = R.run(A, r, steps=j, full=True)
            zr = R.readout(X.cA["H"][("c", j, R.C - 1)], X.cA["freqs"])
            g[str(j)] = bool(torch.equal(zr, oj["z"]))
        out["g_depth_readout_identical"] = g
        ok_g = all(g.values())
        jh = max(2, R.K // 4)
        oh = R.run(A, r, e_edits=X.e_ed([jh], src), rec_eargs=True)
        nsrc = [q for q in allp if q not in src]
        out["h_e_single_call_only"] = {
            "call": jh,
            "other_calls_equal_clean_A": all(torch.equal(oh["eargs"][n], X.cA["E"]) for n in range(1, R.K + 1) if n != jh),
            "call_src_equals_B": bool(torch.equal(oh["eargs"][jh][src], X.cB["E"][src])),
            "call_rest_equals_A": bool(torch.equal(oh["eargs"][jh][nsrc], X.cA["E"][nsrc]))}
        oi = R.run(A, r, e_edits={j: (tall, X.cB["E"]) for j in range(1, R.K + 1)})
        out["i_all_e_B_equals_clean_B"] = bool(torch.equal(oi["z_r"], X.cB["z_r"]))
        x = R.set_init(512, 12345)
        rel = float(x.std()) / (TRUNC_STD * R.std * R.emb_scale)
        out["j_init_std"] = {"rel_to_truncated_std": round(rel, 5), "within_2pct": abs(rel - 1) < 0.02,
                             "cfg_std": R.std, "emb_scale": R.emb_scale}
        x1, x2 = R.set_init(40, 7), R.set_init(100, 7)
        out["j_prefix_property"] = bool(torch.equal(x1, x2[:, :40]))
        R.set_init(it["Tmax"], X.seed)
        out["k_exec_index_asserted"] = {"formula": "%d + %d(j-1) + l" % (R.P, R.C), "loop_end_K": R.exec_idx(R.K, R.C - 1)}
        sl = R.run(A, r, edits=X.o1(("c", jm, R.C // 2), allp))
        sk = R.run(A, r, edits=X.o1(("c", jm, R.C - 1), allp))
        out["sc_identity_Lmid_eq_Llast"] = bool(torch.equal(sl["z_r"], sk["z_r"]))
    ok = (out["b_zero_edit_identity"] and out["c_repeat_identical"]
          and all(v for k, v in out["d_edit_locality"].items() if not k.startswith("n_") and k != "site")
          and out["e_grad_run_identical"] and out["e_grad_nonzero_sites"] == len(Sp) and ok_g)
    if R.family == "raven":
        ok = ok and out["f_ok"] is not False and all(v for k, v in out["h_e_single_call_only"].items() if k != "call") \
            and out["i_all_e_B_equals_clean_B"] and out["j_init_std"]["within_2pct"] and out["j_prefix_property"] \
            and out["sc_identity_Lmid_eq_Llast"]
    out["all_ok"] = bool(ok)
    return out


# ------------------------------------------------------------------ selection, scans, dataset driver
class Selector(object):
    """Quota and cap rules (plan section 8): D1/CF subject cap 2 per (template, entity) on either side; D2 cap 2 per
    edited-operand unit; D4 one item per content unit; D5 no problem in two kept items. want() before the clean
    filter, accept() after a pass."""

    def __init__(self, ds, target):
        self.ds, self.target, self.n, self.cnt, self.problems = ds, target, 0, {}, set()

    def done(self):
        return self.n >= self.target

    def want(self, c):
        if self.done():
            return False
        if self.ds in ("D1", "CF"):
            return all(self.cnt.get(s, 0) < SUBJ_CAP for s in c["sel"]["subjects"])
        if self.ds == "D2":
            return all(self.cnt.get(s, 0) < D2_CLUSTER_CAP for s in c["sel"]["clusters"])
        if self.ds == "D4":
            return all(self.cnt.get(s, 0) < 1 for s in self.units(c))
        if self.ds == "D5":
            return not (set(c["sel"]["problems"]) & self.problems)
        return True

    def units(self, c):
        """Capped units of a candidate. D4 (review 3 F5): both content keys (A and B side, c['cl']); the stored
        sel.clusters holds the A side only, so 'one item per content key on either side' was not enforced."""
        if self.ds == "D4":
            return list(c["cl"])
        return list(c["sel"].get("subjects", [])) + list(c["sel"].get("clusters", []))

    def accept(self, c):
        self.n += 1
        for s in self.units(c):
            self.cnt[s] = self.cnt.get(s, 0) + 1
        self.problems |= set(c["sel"].get("problems", []))


def item_status(Rs, c, seed_off=0):
    st = {}
    for R in Rs:
        it = prep_item(c, R.bos)
        if it is None:
            st[R.name] = "pos0_dropped"
            continue
        st[R.name] = Item(R, it, seed_off=seed_off, rec=None).status
    return st


def scan(Rs, ds, cands, target, deadline=None, seed_off=0, log=None, stats=None):
    """Joint clean filter in selection order over the runners Rs (McLeish and NR: identical items, paired).
    -> (kept pair_ids, log [pid, {model: status}]). stats (dict) receives 'stopped_budget' = True when the deadline
    ends the scan before the target is met (review 3 F10)."""
    sel = Selector(ds, target)
    cmap = {c["pair_id"]: c for c in cands}
    log = list(log or [])
    done = set()
    for pid, st in log:
        done.add(pid)
        if all(v == "kept" for v in st.values()):
            sel.accept(cmap[pid])
    stopped = False
    for c in cands:
        if sel.done():
            break
        if c["pair_id"] in done or not sel.want(c):
            continue
        if deadline is not None and time.time() > deadline:
            stopped = True
            break
        st = item_status(Rs, c, seed_off)
        log.append([c["pair_id"], st])
        if all(v == "kept" for v in st.values()):
            sel.accept(c)
    if stats is not None:
        stats["stopped_budget"] = stopped
    kept = [pid for pid, st in log if all(v == "kept" for v in st.values())]
    return kept, log


def kept_of(res):
    """Joint kept pair_ids of a res file in scan (selection) order."""
    return [pid for pid, st in res.get("scan", []) if st and all(v == "kept" for v in st.values())]


def scan_stats(log, cmap, names):
    st = {"scanned": len(log), "joint_kept": 0, "solo": {n: {} for n in names}}
    for pid, s in log:
        for n in names:
            v = s.get(n, "not_run")
            st["solo"][n][v] = st["solo"][n].get(v, 0) + 1
        st["joint_kept"] += int(all(v == "kept" for v in s.values()))
    kept = [cmap[pid] for pid, s in log if all(v == "kept" for v in s.values())]
    st["n_units"] = len({u for c in kept for u in c["cl"]})
    pl = {}
    for c in kept:
        k = "%s|%s" % (c["strata"]["pos_len"], c["rel"])
        pl[k] = pl.get(k, 0) + 1
    st["pos_len_x_rel"] = pl
    st["split"] = {s: sum(1 for c in kept if c["split"] == s) for s in ("dev", "test", "cross")}
    st["dev_test_unit_overlap"] = len({u for c in kept if c["split"] == "dev" for u in c["cl"]}
                                      & {u for c in kept if c["split"] == "test" for u in c["cl"]})
    return st


def run_item_rows(X, k, ds, full):
    """All rows of one kept item; k = its index in the global kept list (subset, tiny rows), full = full E8 search
    (decided from the global kept list by search_plan, review 3 F2/F4)."""
    R = X.R
    subset = k < n_sub(R.name)
    if R.name == "huginn8":                       # critic 2 F10: K = 8 runs only the harness rows and PB8 rows
        run_controls(X, False)
        run_pb8(X)
        X.out.update({"subset": False, "n_rows": len(X.rows), "kept_index": k})
        return
    X.depth_margins()
    X.deltas()
    if X.vb:
        X.finish_clean_vb()
    run_controls(X, subset)
    run_e1(X, tiny=k < N_TINY)
    if R.family == "raven":
        run_e5(X, subset)
        if R.key == "huginn" and ds in HUGINN16_DS and not RUN_OPTS["no_reduced"]:
            run_pb8(X)
    if not X.vb:
        run_para(X)
    if ds in E8_DATASETS and R.family != "ouro" and R.name in ("huginn", "mcleish", "llama_nr") \
            and not RUN_OPTS.get("no_gradient_rank"):
        run_e8(X, bool(full))
    X.out.update({"subset": subset, "n_rows": len(X.rows), "kept_index": k})


def res_path(out_dir, name, ds, shard=None):
    if shard:
        return os.path.join(out_dir, "res_%s_%s.shard%dof%d.json" % (name, ds, shard[0], shard[1]))
    return os.path.join(out_dir, "res_%s_%s.json" % (name, ds))


def load_res(out_dir, R, ds, shard=None):
    p = res_path(out_dir, R.name, ds, shard)
    return json.load(open(p)) if os.path.exists(p) else {"ds": ds, "model": R.name, "bos": R.bos, "K": R.K,
                                                         "scan": [], "items": [], "meta": {}}


def elig_S(x, K, C):
    """T7 eligibility over S at a = 1 on a fully searched item: D >= 2 and verified dm range over S > 2."""
    keys = [skey(s) for s in e8_sets(K, C)[0]]
    G = {k: x["G"][k] for k in keys if k in (x.get("G") or {})}
    h = hit_v3(G, {k: dm_of(x, "O1|ans|%s|1" % k) for k in keys}, Dgap(x), THR[1.0])
    return bool(h and h["eligible"])


def search_plan(R, ds, kept, cmap):
    """Review 3 F2/F4: the full-search set comes from the global kept list, not from a running counter, so shards and
    the sequential run choose the same items. Search items = kept dev/cross items in kept order, up to the cap
    (--full_search_cap). Raven: the first min(PILOT_N, cap) are the pilot (always searched); the rest are searched only
    if the pilot gate passes. NR: no gate (its full search is 15 forwards)."""
    if ds not in E8_DATASETS or R.name not in ("huginn", "mcleish", "llama_nr"):
        return {"applies": False, "cap": 0, "n_search_items": 0, "pilot": [], "rest": []}
    search = [p for p in kept if cmap[p]["split"] in SEARCH_SPLITS]
    cap = min(full_cap(R.name, ds), len(search))
    npil = min(PILOT_N, cap) if R.family == "raven" else 0      # NR: no pilot, so its search rows shard too
    return {"applies": True, "cap": cap, "n_search_items": len(search), "pilot": search[:npil],
            "rest": search[npil:cap]}


def e8_gate(R, ds, items, plan):
    """Pilot gate (review 3 F4), decided once every pilot item has rows: full search continues past the pilot iff
    (eligible rate over S at a = 1 on the pilot) x (search items under the cap) >= MIN_E8; else E8 on this
    (model, dataset) is descriptive and later search items get the verify rows only."""
    if not plan["applies"] or not plan["rest"]:
        return {"decided": True, "continue": True, "reason": "no search items beyond the pilot"}
    if not plan["pilot"]:
        return {"decided": True, "continue": True, "reason": "no pilot gate on this model (NR)"}
    done = {x["pair_id"]: x for x in items}
    if any(p not in done for p in plan["pilot"]):
        return {"decided": False, "continue": None, "n_pilot_done": sum(1 for p in plan["pilot"] if p in done)}
    el = [elig_S(done[p], R.K, R.C) for p in plan["pilot"]]
    rate = sum(el) / float(len(el))
    return {"decided": True, "n_pilot": len(el), "eligible": sum(el), "rate": round(rate, 4),
            "n_search_cap": plan["cap"], "projected_eligible": round(rate * plan["cap"], 2), "min_e8": MIN_E8,
            "continue": bool(rate * plan["cap"] >= MIN_E8),
            "rule": "full search past the first %d search items only if eligible rate x cap >= %d; otherwise the "
                    "E8 cell is descriptive" % (PILOT_N, MIN_E8)}


def do_rows(R, ds, cmap, kept, idxs, res_main, res_out, out_path, deadline, logf, plan):
    """Rows for the kept items at global kept indices idxs, skipping items present in res_main or res_out; results go
    to res_out (= res_main in the sequential run, a shard file otherwise). -> True if the deadline stopped it."""
    done = {x["pair_id"] for x in res_main["items"]} | {x["pair_id"] for x in res_out["items"]}
    pilot, rest = set(plan["pilot"]), set(plan["rest"])
    gate = None
    for k in idxs:
        pid = kept[k]
        if pid in done:
            continue
        if time.time() > deadline:
            return True
        full = pid in pilot
        if pid in rest:
            if gate is None:
                its = res_main["items"] + ([] if res_out is res_main else res_out["items"])
                gate = e8_gate(R, ds, its, plan)
                if not gate["decided"]:
                    raise HarnessFail("E8 pilot gate undecided for %s %s (%s): run --stage scan first" % (
                        R.name, ds, json.dumps(gate)))
                res_out["meta"]["e8_gate"] = gate
                logf("%s %s E8 pilot gate %s" % (R.name, ds, json.dumps(gate)))
            full = bool(gate["continue"])
        t1 = time.time()
        it = prep_item(cmap[pid], R.bos)
        X = Item(R, it)
        if X.status != "kept":
            raise HarnessFail("clean status changed between scan and rows: %s %s %s" % (R.name, pid, X.status))
        run_item_rows(X, k, ds, full)
        X.out["t_item"] = round(time.time() - t1, 2)
        res_out["items"].append(X.out)
        atomic_json(out_path, res_out)
        logf("%s %s item %d/%d %s rows %d %.1fs" % (R.name, ds, k + 1, len(kept), pid, len(X.rows), time.time() - t1))
        del X
    return False


def shard_files(out_dir, name, ds):
    pre = "res_%s_%s.shard" % (name, ds)
    return sorted(f for f in os.listdir(out_dir) if f.startswith(pre) and f.endswith(".json"))


def run_dataset(Rs, ds, cands, out_dir, target, deadline, logf, order_first=None, only=None, stage="all", shard=None):
    """Review 3 F2. stage 'all' (sequential, one host): joint scan, rows for every kept item, dev-tuned pass. Cluster:
    'scan' = the joint scan (one global pass, so the subject / content / problem caps are applied once) plus the E8
    pilot rows; 'rows' with shard (k, n) = rows for the kept items with index % n == k, written to their own shard file
    in the same directory; 'merge' = shard files into the res file (kept order), then the dev-tuned pass. Resumable by
    pair_id at every stage."""
    cmap = {c["pair_id"]: c for c in cands}
    if only is not None:
        cands = [cmap[p] for p in only if p in cmap]
    elif order_first:
        pri = [cmap[p] for p in order_first if p in cmap]
        ps = {c["pair_id"] for c in pri}
        cands = pri + [c for c in cands if c["pair_id"] not in ps]
    ress = {R.name: load_res(out_dir, R, ds) for R in Rs}
    r0 = ress[Rs[0].name]
    t0 = time.time()
    scan_stopped = False
    if stage in ("all", "scan"):
        sst = {}
        kept, log = scan(Rs, ds, cands, target, deadline, log=r0["scan"], stats=sst)
        scan_stopped = sst["stopped_budget"]
        stats = scan_stats(log, cmap, [R.name for R in Rs])
        for R in Rs:
            res = ress[R.name]
            res["scan"], res["meta"]["scan_stats"], res["meta"]["target"] = log, stats, target
            res["meta"]["joint_with"] = [x.name for x in Rs]
            res["meta"]["scan_complete"] = not scan_stopped      # F10: a deadline stop is never silent
            res["meta"]["stopped_budget"] = scan_stopped
            res["meta"]["scan_seconds"] = round(res["meta"].get("scan_seconds", 0) + time.time() - t0, 1)
            atomic_json(res_path(out_dir, R.name, ds), res)
        logf("%s %s scan %s%s" % ("+".join(R.name for R in Rs), ds, json.dumps(stats),
                                  " STOPPED BY BUDGET" if scan_stopped else ""))
    else:
        if not r0["scan"] or not r0["meta"].get("scan_complete"):
            raise HarnessFail("%s %s: stage %s needs a complete --stage scan first" % (Rs[0].name, ds, stage))
        kept = kept_of(r0)
    for R in Rs:
        res = ress[R.name]
        plan = search_plan(R, ds, kept, cmap)
        cur = {k: (len(v) if isinstance(v, list) else v) for k, v in plan.items()}
        prev = res["meta"].get("search_plan")
        if stage in ("rows", "merge") and prev is not None and prev != cur:
            raise HarnessFail("%s %s: search plan %s differs from the scan stage %s (same --full_search_cap needed)" % (
                R.name, ds, cur, prev))
        res["meta"]["search_plan"] = cur
        stopped = scan_stopped
        if stage == "scan":
            idxs = sorted(kept.index(p) for p in plan["pilot"])
            stopped = do_rows(R, ds, cmap, kept, idxs, res, res, res_path(out_dir, R.name, ds), deadline, logf,
                              plan) or stopped
            res["meta"]["e8_gate"] = e8_gate(R, ds, res["items"], plan)
        elif stage == "rows":
            sres = load_res(out_dir, R, ds, shard)
            idxs = [i for i in range(len(kept)) if i % shard[1] == shard[0]]
            st = do_rows(R, ds, cmap, kept, idxs, res, sres, res_path(out_dir, R.name, ds, shard), deadline, logf, plan)
            sres["meta"].update({"shard": list(shard), "stopped_budget": st, "kept_total": len(kept),
                                 "n_assigned": len(idxs), "seconds": round(sres["meta"].get("seconds", 0)
                                                                           + time.time() - t0, 1)})
            atomic_json(res_path(out_dir, R.name, ds, shard), sres)
            continue
        elif stage == "merge":
            have = {x["pair_id"]: x for x in res["items"]}
            files = shard_files(out_dir, R.name, ds)
            for f in files:
                for x in json.load(open(os.path.join(out_dir, f)))["items"]:
                    have.setdefault(x["pair_id"], x)
            res["items"] = [have[p] for p in kept if p in have]
            res["meta"]["merged_shards"] = files
        else:
            stopped = do_rows(R, ds, cmap, kept, range(len(kept)), res, res, res_path(out_dir, R.name, ds), deadline,
                              logf, plan) or stopped
        res["meta"]["stopped_budget"] = bool(stopped)
        res["meta"]["rows_missing"] = sum(1 for p in kept if p not in {x["pair_id"] for x in res["items"]})
        if stage in ("all", "merge"):
            if not (res["meta"].get("e8_gate") or {}).get("decided"):
                res["meta"]["e8_gate"] = e8_gate(R, ds, res["items"], plan)
            if res["meta"]["rows_missing"] == 0 and ds in E8_DATASETS and any(x.get("G") for x in res["items"]):
                cm = {x["pair_id"]: prep_item(cmap[x["pair_id"]], R.bos) for x in res["items"]}
                dt = dev_tuned_pass(R, res, cm)
                logf("%s %s dev-tuned %s" % (R.name, ds, json.dumps({k: v.get("site") if isinstance(v, dict) else v
                                                                    for k, v in dt.items()})))
        res["meta"]["seconds"] = round(res["meta"].get("seconds", 0) + time.time() - t0, 1)
        atomic_json(res_path(out_dir, R.name, ds), res)
    return ress


def seed_sensitivity(R, cands, res, deadline, logf):
    """PB7 (review 3 F9): the pair_ids scanned at offset 0 are re-run at offsets 1 and 2 with no Selector, and the
    clean-filter status of R is compared item by item (a Selector rescan would apply the subject cap with
    offset-specific acceptances and skip offset-0 items unscanned). The source O1 Llast@K/2 cell is recomputed on the
    scored items."""
    cmap = {c["pair_id"]: c for c in cands}
    scanned = [(pid, st[R.name]) for pid, st in res["scan"] if R.name in st]
    sK = ("c", R.K // 2, R.C - 1)
    out = {"offsets": {}, "cell": "O1|src|%s|1" % skey(sK), "rule": "same scanned pair_ids, no Selector (review 3 F9)"}
    for off in SEED_OFFSETS[1:]:
        st_off = {}
        for pid, _ in scanned:
            if time.time() > deadline:
                break
            st_off[pid] = item_status([R], cmap[pid], off)[R.name]
        cmp = [(pid, s0, st_off[pid]) for pid, s0 in scanned if pid in st_off]
        k_off = {pid for pid, _, s in cmp if s == "kept"}
        k_0 = {pid for pid, s, _ in cmp if s == "kept"}
        e = {}
        for x in res["items"]:
            if time.time() > deadline:
                break
            it = prep_item(cmap[x["pair_id"]], R.bos)
            X = Item(R, it, seed_off=off, rec={sK: "all"})
            if X.D < GAP:
                continue
            o = R.run(it["A"], it["r"], edits=X.o1(sK, it["src"]))
            e[x["pair_id"]] = r5((float(o["z_r"][X.tB] - o["z_r"][X.tA]) - X.mA) / X.D)
        out["offsets"][str(off)] = {"n_rescanned": len(cmp), "n_scanned_off0": len(scanned),
                                    "complete": len(cmp) == len(scanned), "kept_on_same_scan_set": len(k_off),
                                    "kept_off0": len(k_0), "sym_diff": len(k_off ^ k_0),
                                    "status_changed": sum(1 for _, a, b in cmp if a != b),
                                    "changes": [[p, a, b] for p, a, b in cmp if a != b][:100], "e": e}
        logf("seed offset %d: kept %d vs %d on %d rescanned items, sym_diff %d" % (
            off, len(k_off), len(k_0), len(cmp), len(k_off ^ k_0)))
    res["meta"]["seed_sensitivity"] = out
    return out


def gpu_mem():
    if not torch.cuda.is_available():
        return None
    f, t = torch.cuda.mem_get_info()
    return {"free_gb": round(f / 1e9, 1), "total_gb": round(t / 1e9, 1),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def load_cands(data_dir, fam, ds):
    p = os.path.join(data_dir, "cands_%s_%s.json" % (fam, ds))
    return json.load(open(p))["cands"] if os.path.exists(p) else None


def make_runners(group, dev):
    """group: mcl_nr -> [McLeish K=32, NR]; huginn -> [Huginn K=32] (+ K=16 runner on the same weights);
    ouro_think -> [Ouro]."""
    meta = {}
    if group == "mcl_nr":
        m1, a1 = load_raven("mcleish", dev)
        R1 = RavenR("mcleish", m1, dev, 32)
        R1.bos, R1.fam = BOS["llama"], "llama"
        m2, a2 = load_hf("llama_nr", dev)
        R2 = LlamaR(m2, dev)
        R2.bos, R2.fam = BOS["llama"], "llama"
        meta = {"mcleish": a1, "llama_nr": a2}
        return [R1, R2], [], meta
    if group == "huginn":
        m1, a1 = load_raven("huginn", dev)
        R1 = RavenR("huginn", m1, dev, 32)
        R1.bos, R1.fam = BOS["huginn"], "huginn"
        R16 = RavenR("huginn", m1, dev, 16)       # its own hooks; each runner acts only on its own context
        R16.bos, R16.fam = BOS["huginn"], "huginn"
        R8 = RavenR("huginn", m1, dev, 8)         # critic 2 F10: PB8 rows only
        R8.bos, R8.fam = BOS["huginn"], "huginn"
        return [R1], [R16, R8], {"huginn": a1}
    if group in OURO_GROUPS:
        m, a = load_hf(group, dev, trust_remote_code=True, attn_implementation="eager")
        R = OuroR(m, dev, group)
        R.bos, R.fam = BOS["ouro"], "ouro"
        return [R], [], {group: a}
    raise KeyError(group)


OURO_GROUPS = ("ouro_think", "ouro26_think")
GROUP_FAM = {"mcl_nr": "llama", "huginn": "huginn", "ouro_think": "ouro", "ouro26_think": "ouro"}


def pre_run_hashes(group, data_dir, plan, cfg):
    """PLAN G8 (review 3 F1): inputs, configuration and the written hypotheses (plan file) hashed before execution."""
    fam = GROUP_FAM[group]
    cs = {}
    for ds in DATASETS:
        p = os.path.join(data_dir, "cands_%s_%s.json" % (fam, ds))
        if os.path.exists(p):
            cs[ds] = sha256_file(p)
    bm = os.path.join(data_dir, "build_meta.json")
    return {"group": group, "plan_sha256": sha256_file(plan) if plan and os.path.exists(plan) else None,
            "code_md5": md5_file(os.path.abspath(__file__)), "cands_sha256": cs,
            "build_meta_sha256": sha256_file(bm) if os.path.exists(bm) else None,
            "config_sha256": sha(json.dumps(cfg, sort_keys=True)), "config": cfg,
            "hashed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_config(group, debug, datasets, budget_min, seeds, stage, shard):
    return {"group": group, "debug": debug, "datasets": list(datasets), "budget_min": budget_min, "seeds": seeds,
            "stage": stage, "shard": list(shard) if shard else None, "targets": dict(TARGET[group]),
            "descriptive_targets": dict(DESCRIPTIVE["targets"]), "full_search_cap_override": dict(FS_CAP_OVR),
            "full_search_cap_default": dict(DEV_FULL_CAP), "pc_site_n_override": dict(PCS_OVR), "n_sub": dict(N_SUB),
            "pig_a": list(PIG_A), "pilot_n": PILOT_N, "min_e8": MIN_E8, "min_kept": MIN_KEPT, "a_verify": list(A_VERIFY),
            "thr": {str(k): v for k, v in THR.items()}, "hit_tol": HIT_TOL, "gap": GAP, "nboot": NBOOT,
            "seed_offsets": list(SEED_OFFSETS), "no_reduced": RUN_OPTS["no_reduced"],
            **({"no_gradient_rank": True} if RUN_OPTS.get("no_gradient_rank") else {})}


def run_group(group, data_dir, out_dir, dev, debug, datasets, budget_min, seeds=True, stage="all", shard=None,
              plan=None):
    t_start = time.time()
    deadline = t_start + 60 * budget_min
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    tag = group if stage == "all" else ("%s.%s" % (group, stage) + (".shard%dof%d" % tuple(shard) if shard else ""))
    lf = open(os.path.join(out_dir, "logs", "%s.log" % tag), "a")

    def logf(s):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
    cfg = run_config(group, debug, datasets, budget_min, seeds, stage, shard)
    hashes = pre_run_hashes(group, data_dir, plan, cfg)           # G8: before the models load and any row is scored
    ipath = os.path.join(out_dir, "inputs_%s.json" % tag)
    prev_in = json.load(open(ipath)) if os.path.exists(ipath) else {"runs": []}
    prev_in["runs"].append(hashes)
    atomic_json(ipath, prev_in)
    logf("pre-run hashes %s" % json.dumps({k: v for k, v in hashes.items() if k != "config"}))
    Rs, Rx, audit = make_runners(group, dev)
    meta = {"group": group, "stage": stage, "shard": list(shard) if shard else None, "debug": debug,
            "budget_min": budget_min, "torch": torch.__version__, "pre_run": hashes,
            "python": sys.version.split()[0], "gpu_mem_start": gpu_mem(), "load_audit": audit,
            "k17_public_md5": md5_file(os.path.abspath(__file__)),
            "reused_md5": {m: md5_file(os.path.join(HERE, m)) for m in REUSED if os.path.exists(os.path.join(HERE, m))},
            "arch": {R.name: {"K": R.K, "C": R.C, "P": R.P, "Cd": R.Cd, "d": R.d, "bos": R.bos} for R in Rs + Rx}}
    mpath = os.path.join(out_dir, "meta_%s.json" % tag)
    if os.path.exists(mpath):
        prev = json.load(open(mpath))
        meta["previous_runs"] = prev.pop("previous_runs", []) + [prev]
    fam = Rs[0].fam
    d1 = load_cands(data_dir, fam, "D1")
    meta["hook_identity"] = {}
    for R in Rs + Rx:
        first = next(prep_item(c, R.bos) for c in d1 if prep_item(c, R.bos) is not None)
        hk = check_hooks(R, first)
        meta["hook_identity"][R.name] = hk
        logf("%s hook checks %s" % (R.name, json.dumps(hk)))
        atomic_json(mpath, meta)
        if not hk["all_ok"]:
            raise HarnessFail("hook checks failed on %s" % R.name)
    meta["datasets"] = {}
    tg = TARGET[group]
    prior = {}
    if group in OURO_GROUPS:              # McLeish's kept items first; needs the same --out as mcl_nr (review 3 F2)
        for ds in datasets:
            p = res_path(out_dir, "mcleish", ds)
            if os.path.exists(p):
                prior[ds] = kept_of(json.load(open(p)))
            else:
                logf("WARNING: no %s in --out; %s %s items are not aligned to McLeish's" % (p, group, ds))
    for ds in datasets:
        if ds not in tg:
            continue
        cands = load_cands(data_dir, fam, ds)
        if cands is None:
            logf("no candidates for %s %s" % (fam, ds))
            continue
        if ds == "D1":       # critic 2 F8: balance on relation x subject token count (round-robin over the cells)
            cands = rr_order(cands, lambda c: (c["rel"], c["strata"]["pos_len"]))
        target = debug if debug else tg[ds]
        ress = run_dataset(Rs, ds, cands, out_dir, target, deadline, logf, order_first=prior.get(ds), stage=stage,
                           shard=shard)
        meta["datasets"][ds] = {n: {"kept": len(kept_of(r)), "items": len(r["items"]),
                                    "stopped_budget": r["meta"].get("stopped_budget", False),
                                    "rows_missing": r["meta"].get("rows_missing")} for n, r in ress.items()}
        if seeds and ds == "D1" and group not in OURO_GROUPS and not debug and stage in ("all", "merge"):
            for R in Rs:
                if R.family == "raven":
                    seed_sensitivity(R, cands, ress[R.name], deadline, logf)
                    atomic_json(res_path(out_dir, R.name, ds), ress[R.name])
        atomic_json(mpath, meta)
    for Rk in ([] if RUN_OPTS["no_reduced"] else Rx):  # Huginn K = 16 (reduced arms) and K = 8 (PB8 only), same items
        for ds in HUGINN16_DS:
            if ds not in datasets:
                continue
            p = res_path(out_dir, "huginn", ds)
            if not os.path.exists(p):
                continue
            only = kept_of(json.load(open(p)))
            cands = load_cands(data_dir, fam, ds)
            ress = run_dataset([Rk], ds, cands, out_dir, len(only), deadline, logf, only=only, stage=stage,
                               shard=shard)
            meta["datasets"]["%s|%s" % (Rk.name, ds)] = {"kept": len(kept_of(ress[Rk.name])),
                                                         "items": len(ress[Rk.name]["items"])}
            atomic_json(mpath, meta)
    meta["gpu_mem_end"] = gpu_mem()
    meta["process_minutes"] = round((time.time() - t_start) / 60, 2)
    atomic_json(mpath, meta)
    logf("%s done in %.1f min" % (tag, meta["process_minutes"]))
    lf.close()
    return meta


def count_caps(data_dir):
    """Review 3 F2: upper bounds of kept items per (tokenizer, dataset) from the Selector over the candidate files,
    assuming every candidate passes the clean filter (no-BOS position-0 drops applied; D1 in run order). D4 is also
    counted with the A-side-only rule of v3 for comparison."""
    out = {}
    for fam in ("huginn", "llama", "ouro"):
        for ds in DATASETS:
            cands = load_cands(data_dir, fam, ds)
            if cands is None:
                continue
            if ds == "D1":
                cands = rr_order(cands, lambda c: (c["rel"], c["strata"]["pos_len"]))
            ok = [c for c in cands if not (BOS[fam] is None and 0 in c["src"])]
            sel = Selector(ds, 10 ** 9)
            kept = []
            for c in ok:
                if sel.want(c):
                    sel.accept(c)
                    kept.append(c)
            row = {"candidates": len(cands), "after_pos0": len(ok), "upper_bound_kept": len(kept),
                   "split": {s: sum(1 for c in kept if c["split"] == s) for s in ("dev", "test", "cross")}}
            if ds == "D4":
                cnt, ka = {}, 0
                for c in ok:
                    if all(cnt.get(u, 0) < 1 for u in c["sel"]["clusters"]):
                        ka += 1
                        for u in c["sel"]["clusters"]:
                            cnt[u] = cnt.get(u, 0) + 1
                row["upper_bound_kept_A_side_only_v3"] = ka
            out["%s|%s" % (fam, ds)] = row
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ carry-gain finite-difference check (critic 2 F2)
def fd_sites(R):
    """Site classes of the FD check: raven Llast@1, Llast@K/2, Llast@K-1, Lmid@K; NR their DC6 images (layers 4, 7,
    11, 14). Llast@K-1 added after the v3 debug: on McLeish, G and FD are both below the floor at Llast@1 and Llast@K/2
    (contraction), so without it no counted class tests the backward through a recurrence step."""
    if R.family == "nr":
        return [("Llast@1~layer@4", ("c", 1, 4)), ("Llast@16~layer@7", ("c", 1, 7)),
                ("Llast@31~layer@11", ("c", 1, 11)), ("Lmid@32~layer@14", ("c", 1, 14))]
    return [("Llast@1", ("c", 1, R.C - 1)), ("Llast@%d" % (R.K // 2), ("c", R.K // 2, R.C - 1)),
            ("Llast@%d" % (R.K - 1), ("c", R.K - 1, R.C - 1)), ("Lmid@%d" % R.K, ("c", R.K, R.C // 2))]


def fd_grad(R, it, S):
    """Clean A/B (records at r for every site of S) and G(s) = grad_{H_s[r]} m . (H^B - H^A)_s[r]."""
    r = it["r"]
    R.set_init(it["Tmax"], seed_of(it["pair_id"]))
    rec = {s: [r] for s in S}
    cA, cB = R.run(it["A"], r, rec=rec), R.run(it["B"], r, rec=rec)
    leaves = {s: torch.zeros(1, 1, R.d, device=R.dev, dtype=R.dtype, requires_grad=True) for s in S}
    o = R.run(it["A"], r, leaves=leaves, grad=True)
    (o["z_r"][it["tB"]] - o["z_r"][it["tA"]]).backward()
    G = {skey(s): float(leaves[s].grad[0, 0].float() @ (cB["H"][s][0] - cA["H"][s][0]).float()) for s in S}
    same = bool(torch.equal(o["z_r"].detach(), cA["z_r"]))
    del leaves, o
    return cA, cB, G, same


def sig7(x):
    """7 significant digits (FD files keep G below r5's 1e-5 resolution for the rank comparison, review 3 F6)."""
    return None if x is None else float("%.7g" % x)


def class_sites(R):
    """Answer-class sites of summarize (raven): Llast@j for j in JSET[K], j < K (review 3 F7)."""
    if R.family != "raven":
        return []
    return [("Llast@%d" % j, ("c", j, R.C - 1)) for j in JSET[R.K] if j < R.K]


def a025_rows(R, it, cA, cB, sites):
    """Answer O1 at a = 0.25 at r at the class sites, in the precision of R (review 3 F7)."""
    r, tA, tB = it["r"], it["tA"], it["tB"]
    mA = float(cA["z_r"][tB] - cA["z_r"][tA])
    out = {"D": sig7(float(cB["z_r"][tB] - cB["z_r"][tA]) - mA), "dm": {}}
    for nm, s in sites:
        ed = {s: [("set", torch.tensor([r], device=R.dev), cB["H"][s], 0.25)]}
        o = R.run(it["A"], r, edits=ed)["z_r"]
        out["dm"][nm] = sig7(float(o[tB] - o[tA]) - mA)
    return out


def fd_item(R, it, S):
    r, tA, tB = it["r"], it["tA"], it["tB"]
    cA, cB, G, same = fd_grad(R, it, S)
    mA = float(cA["z_r"][tB] - cA["z_r"][tA])
    D = float(cB["z_r"][tB] - cB["z_r"][tA]) - mA
    z0 = R.run(it["A"], r)["z_r"]
    out = {"pair_id": it["pair_id"], "cluster": it["cluster"], "D": r5(D), "Z": abs(float(z0[tB] - z0[tA]) - mA),
           "grad_run_identical": same, "G": {k: sig7(v) for k, v in G.items()}, "cells": {},
           "a025": {"fp32": a025_rows(R, it, cA, cB, class_sites(R))}}
    for cls, s in fd_sites(R):
        row = {"G": G[skey(s)], "dm": {}}
        for a in FD_A:
            ed = {s: [("set", torch.tensor([r], device=R.dev), cB["H"][s], float(a))]}
            o = R.run(it["A"], r, edits=ed)["z_r"]
            row["dm"]["%g" % a] = float(o[tB] - o[tA]) - mA
        out["cells"][cls] = row
    return out


def fd_rank_agreement(rows, Fl, a=1e-2):
    """Review 3 F6: bf16 vs fp32 agreement of G where it matters for E8. Spearman over the above-floor sites of S
    (|a G_fp32| > F, the FD floor at a = 1e-2; below it G is 1e-13..1e-5 and its rank is noise that E8 never selects),
    top-1 agreement and top-3 set agreement over all of S (E8 picks the argmax over S). The all-S Spearman is kept as
    a diagnostic."""
    per = []
    for x in rows:
        if not x.get("G_bf16"):
            continue
        ks = [k for k in x["G"] if k in x["G_bf16"] and x["G"][k] is not None and x["G_bf16"][k] is not None]
        if len(ks) < 3:
            continue
        ab = [k for k in ks if abs(a * x["G"][k]) > Fl]
        t32 = sorted(ks, key=lambda k: (-x["G"][k], k))
        t16 = sorted(ks, key=lambda k: (-x["G_bf16"][k], k))
        per.append({"pair_id": x["pair_id"], "n_sites": len(ks), "n_above_floor": len(ab),
                    "spearman_above_floor": spearman([x["G"][k] for k in ab], [x["G_bf16"][k] for k in ab]),
                    "spearman_all_S": spearman([x["G"][k] for k in ks], [x["G_bf16"][k] for k in ks]),
                    "top1_agree": t32[0] == t16[0], "top3_same_set": set(t32[:3]) == set(t16[:3])})
    sa = [p["spearman_above_floor"] for p in per if p["spearman_above_floor"] is not None]
    sl = [p["spearman_all_S"] for p in per if p["spearman_all_S"] is not None]
    return {"rule": fd_rank_agreement.__doc__, "floor_F": Fl, "n_items": len(per),
            "median_spearman_above_floor": median(sa), "min_spearman_above_floor": min(sa) if sa else None,
            "top1_agree_frac": mean(float(p["top1_agree"]) for p in per),
            "top3_same_frac": mean(float(p["top3_same_set"]) for p in per),
            "diagnostic_median_spearman_all_S": median(sl), "diagnostic_min_spearman_all_S": min(sl) if sl else None,
            "per_item": per}


def a025_noise(rows):
    """Review 3 F7: bf16 rounding noise of the a = 0.25 answer row, measured directly: |e_bf16 - e_fp32| per item and
    class site (each precision's own D, D >= 2 in both). A class site's a = 0.25 classes are scored only where the mean
    is below A025_NOISE_CUT (half the a = 0.25 'not carried' cut)."""
    out = {}
    names = sorted({n for x in rows for n in ((x.get("a025") or {}).get("fp32") or {}).get("dm", {})})
    for n in names:
        d = []
        for x in rows:
            f, b = (x.get("a025") or {}).get("fp32"), (x.get("a025") or {}).get("bf16")
            if not f or not b or f["D"] is None or b["D"] is None or f["D"] < GAP or b["D"] < GAP:
                continue
            if f["dm"].get(n) is None or b["dm"].get(n) is None:
                continue
            d.append(abs(b["dm"][n] / b["D"] - f["dm"][n] / f["D"]))
        out[n] = {"n": len(d), "mean_abs_de": mean(d), "max_abs_de": round(max(d), 5) if d else None,
                  "cut": A025_NOISE_CUT, "gate_pass": bool(d) and mean(d) < A025_NOISE_CUT}
    return out


def fd_summary(items):
    """CGCHECK H8/H11 rule: floor F = max(Z, N99), N99 = 99th percentile of |dm(1e-6) - 1e-6 G| over items and site
    classes; a cell is above floor iff |a G| > F; a class passes at a = 1e-2 iff, on its above-floor cells, median
    rel <= 0.05 and sign agreement >= 0.95 (>= 10 above-floor cells, else 'insufficient'). Verdict PASS iff every
    counted class passes (D1 and D2 pooled per class; per dataset reported)."""
    res = [abs(c["dm"]["1e-06"] - 1e-6 * c["G"]) for x in items for c in x["cells"].values()]
    Z = max([x["Z"] for x in items] or [0.0])
    n99 = sorted(res)[min(len(res) - 1, int(math.ceil(0.99 * len(res))) - 1)] if res else 0.0
    Fl = max(Z, n99)
    out = {"floor": {"Z": Z, "N99": n99, "F": Fl}, "classes": {}}
    classes = sorted({k for x in items for k in x["cells"]})
    verdict = []
    for cls in classes:
        oc = {}
        for a in FD_A[1:]:
            for scope in ("pooled", "D1", "D2"):
                its = [x for x in items if scope == "pooled" or x["pair_id"].startswith(scope + "|")]
                cells = [x["cells"][cls] for x in its if cls in x["cells"]]
                ab = [c for c in cells if abs(a * c["G"]) > Fl]
                rel = [abs(c["dm"]["%g" % a] - a * c["G"]) / max(abs(a * c["G"]), 1e-12) for c in ab]
                sg = [float((c["dm"]["%g" % a] > 0) == (c["G"] > 0)) for c in ab]
                # v3 debug: a zero-G cell whose FD moves above the floor would mean a detached leaf (reported)
                zm = [c for c in cells if abs(a * c["G"]) <= Fl and abs(c["dm"]["%g" % a]) > Fl]
                st = {"n_cells": len(cells), "n_above_floor": len(ab), "median_rel": median(rel),
                      "sign_agreement": mean(sg), "n_G_below_FD_above": len(zm)}
                if a == 1e-2:
                    st["pass"] = ("insufficient" if len(ab) < 10 else
                                  bool(st["median_rel"] <= 0.05 and st["sign_agreement"] >= 0.95))
                    if scope == "pooled":
                        verdict.append(st["pass"])
                oc["%s|a=%g" % (scope, a)] = st
        out["classes"][cls] = oc
    counted = [v for v in verdict if v != "insufficient"]
    out["verdict"] = "PASS" if counted and all(counted) else ("insufficient" if not counted else "FAIL")
    return out


def fd_models(group):
    return {"mcl_nr": ("mcleish", "llama_nr"), "huginn": ("huginn",)}[group]


def fdcheck(group, data_dir, out_dir, dev, n_items=FD_N, models=None):
    """Critic 2 F2 (LEDGER FINAL FRAME D3): fp32 weights, n_items D1 + n_items D2 items per model, FD vs G at the
    site classes, then the same items in bf16 (model cast in place) for Spearman(G_bf16, G_fp32) over S. Items: the
    kept items of the main run first (res files), then candidates in build order (no clean filter needed for FD)."""
    t0 = time.time()
    out = {"group": group, "doses": list(FD_A), "rule": fd_summary.__doc__, "models": {}}
    for key in (models or fd_models(group)):
        if isinstance(key, tuple):                  # (name, model, runner_factory) for the CPU selftest
            name, m, mk = key
            aud = {"selftest": True}
            R = mk(m)
            data = None
        else:
            if key == "llama_nr":
                m, aud = load_hf("llama_nr", dev, dtype=torch.float32)
                R = LlamaR(m, dev)
            else:
                m, aud = load_raven(key, dev, dtype=torch.float32)
                R = RavenR(key, m, dev, KDEF[key])
            R.bos = BOS[TOKFAM[key]]
            data = data_dir
        S = e8_sets(R.K, R.C)[0]
        its = []
        if data is None:
            its = list(n_items)                       # selftest passes prepared items
        else:
            for ds in ("D1", "D2"):
                cands = {c["pair_id"]: c for c in load_cands(data, TOKFAM[key], ds)}
                p = res_path(out_dir, R.name, ds)
                pids = [x["pair_id"] for x in json.load(open(p))["items"]] if os.path.exists(p) else []
                pids += [c for c in cands if c not in pids]
                k = 0
                for pid in pids:
                    if k >= n_items:
                        break
                    it = prep_item(cands[pid], R.bos)
                    if it is not None:
                        its.append(it)
                        k += 1
        rows = [fd_item(R, it, S) for it in its]
        mo = {"load_audit": aud, "n_items": len(rows), "summary": fd_summary(rows),
              "grad_run_identical_all": all(x["grad_run_identical"] for x in rows), "items": rows}
        for h in R.hs:
            h.remove()
        m.to(torch.bfloat16)
        if R.family == "nr":
            Rb = LlamaR(m, R.dev)
        else:
            Rb = RavenR(R.key, m, R.dev, R.K)
        Rb.bos = R.bos
        for it, x in zip(its, rows):
            cAb, cBb, Gb, _ = fd_grad(Rb, it, S)
            x["G_bf16"] = {k: sig7(v) for k, v in Gb.items()}
            x["a025"]["bf16"] = a025_rows(Rb, it, cAb, cBb, class_sites(Rb))
            del cAb, cBb
        for h in Rb.hs:
            h.remove()
        ra = fd_rank_agreement(rows, mo["summary"]["floor"]["F"])
        mo["rank_agreement_bf16_vs_fp32"] = ra
        # kept under the v3 key for continuity; the primary numbers are now the above-floor ones (review 3 F6)
        mo["spearman_G_bf16_vs_fp32"] = {"median_above_floor": ra["median_spearman_above_floor"],
                                         "min_above_floor": ra["min_spearman_above_floor"],
                                         "top1_agree_frac": ra["top1_agree_frac"], "top3_same_frac": ra["top3_same_frac"],
                                         "diagnostic_median_all_S": ra["diagnostic_median_spearman_all_S"],
                                         "diagnostic_min_all_S": ra["diagnostic_min_spearman_all_S"],
                                         "n_items": ra["n_items"]}
        mo["a025_noise"] = a025_noise(rows)
        out["models"][R.name if data is not None else name] = mo
        del m, R, Rb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out["minutes"] = round((time.time() - t0) / 60, 2)
    if out_dir:
        atomic_json(os.path.join(out_dir, "fd_%s.json" % group), out)
    print(json.dumps({k: {"verdict": v["summary"]["verdict"], "floor": v["summary"]["floor"],
                          "spearman": v["spearman_G_bf16_vs_fp32"]} for k, v in out["models"].items()}), flush=True)
    return out


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


def Dj_of(x, j):
    """Depth-j clean gap D_j = m_B(j) - m_A(j) (raven: coda readout of the loop-j state; Ouro: loop-j exit)."""
    if not x.get("loopsA") or j < 1 or j > len(x["loopsA"]):
        return None
    a, b = x["loopsA"][j - 1], x["loopsB"][j - 1]
    return (b[1] - b[0]) - (a[1] - a[0])


def eown_of(x, key):
    """e_own(j) = dm_own(j) / D_j with the guard D_j >= 2 (plan v3, critic 2 F4); None below the guard ("not testable
    at site"). raven: coda readout of the edited loop-j state; Ouro: the edited run's own loop-j exit."""
    rv = x["rows"].get(key)
    if rv is None or Dgap(x) < GAP or not x.get("loopsA"):
        return None
    j = ksite(key.split("|")[2])[1]
    Dj = Dj_of(x, j)
    if Dj is None or Dj < GAP:
        return None
    mAj = x["loopsA"][j - 1][1] - x["loopsA"][j - 1][0]
    if rv.get("mj") is not None:
        return (rv["mj"] - mAj) / Dj
    if rv.get("loops") is not None:
        return (rv["loops"][j - 1] - mAj) / Dj
    return None


def ratio_of(x, key, ref):
    """Critic F7: e / e(ref) per item (ref = IN-persist(1)), guard dm(ref) >= 2."""
    d, dr = dm_of(x, key), dm_of(x, ref)
    if d is None or dr is None or dr < GAP:
        return None
    return d / dr


def _ulist(c):
    return list(c) if isinstance(c, (list, tuple)) else [c]


def label_vs_rnd(items, key, rkey, fn=None):
    """EXPERIMENT_MATRIX 5.4 (plan v3 T1, critic 2 F5): works = mean(e - e_RND) >= 0.5, flip >= 0.5 and the 95%
    cluster-CI lower bound of e - e_RND >= 0.3; fails = TOST null at 0.10 on e; else partial."""
    fn = fn or e_of
    cl = [x["cluster"] for x in items]
    d = pairedc([fn(x, key) for x in items], [fn(x, rkey) for x in items], cl)
    fl = mean(flip_of(x, key) for x in items)
    t = tostc([fn(x, key) for x in items], cl)
    if d is None:
        return {"e_minus_rnd": None, "label": None}
    lab = ("works" if (d[0] >= 0.5 and (fl or 0) >= 0.5 and d[1] is not None and d[1] >= 0.3) else
           ("fails" if t and t["null_by_tost"] else "partial"))
    return {"e_minus_rnd": d, "label": lab, "rnd_key": rkey}


def cell(items, key, null=False, strata=("pos_len",), lo=False, rnd=False, fn=None):
    fn = fn or e_of
    xs = [fn(x, key) for x in items]
    cl = [x["cluster"] for x in items]
    b = bootc(xs, cl)
    if b is None:
        return None
    out = {"n_items": b[3], "n_clusters": b[4], "e": b[:3], "flip": mean(flip_of(x, key) for x in items)}
    if lo:
        bl = bootc([elo_of(x, key) for x in items], cl)
        out["e_LO"] = bl[:3] if bl else None
        out["n_LO"] = bl[3] if bl else 0
    if null or rnd:
        out["tost"] = tostc(xs, cl)
    if rnd:      # plan v3 (critic 2 F5): validity = flip <= 0.02; e is the floor, reported, not gated
        out["valid_flip_le_0.02"] = bool((out["flip"] or 0) <= 0.02)
    st = {}
    for sk in strata:
        if sk == "pos_len_x_rel":
            def get(x):
                return "%s|%s" % (x["strata"].get("pos_len"), x["strata"].get("rel"))
        else:
            def get(x, sk=sk):
                return x["strata"].get(sk)
        vs = sorted(set(str(get(x)) for x in items if get(x) is not None))
        if len(vs) < 2:
            continue
        st[sk] = {}
        for v in vs:
            sub = [x for x in items if str(get(x)) == v]
            bb = bootc([fn(x, key) for x in sub], [x["cluster"] for x in sub])
            st[sk][v] = None if bb is None else {"e": bb[:3], "n_items": bb[3], "n_clusters": bb[4]}
    if st:
        out["strata"] = st
    return out


def load_results(out_dir):
    res, metas = {}, {}
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("res_") and f.endswith(".json") and ".shard" not in f:     # shard files merge into res_*
            d = json.load(open(os.path.join(out_dir, f)))
            res[(d["model"], d["ds"])] = d
        elif f.startswith("meta_") and f.endswith(".json"):
            metas[f[5:-5]] = json.load(open(os.path.join(out_dir, f)))
    return res, metas


def arch_of(mo):
    """-> (family, K, C). Review 3 F3: huginn8 (the K = 8 runner of PB8) added; tiny_huginn8 for the selftest."""
    fam = {"huginn": "raven", "huginn16": "raven", "huginn8": "raven", "mcleish": "raven", "llama_nr": "nr",
           "ouro_think": "ouro", "ouro26_think": "ouro", "tiny_raven": "raven", "tiny_huginn": "raven",
           "tiny_llama": "nr"}[mo]
    C = {"huginn": 4, "huginn16": 4, "huginn8": 4, "mcleish": 6, "llama_nr": 16, "ouro_think": 24, "ouro26_think": 48,
         "tiny_raven": 6, "tiny_huginn": 4, "tiny_llama": 16}[mo]
    K = {"tiny_raven": 32, "tiny_huginn": 32, "tiny_llama": 1}.get(mo) or KDEF[mo]
    return fam, K, C


def summarize_pig(its, full, keys, latest, a):
    """Review 3 F1 (PLAN G4): the path-integrated gain next to G. Full set (a = 1, search items): top-1 hit of each, the
    paired PIG - G hit difference and PIG hit - latest-site hit (cluster CIs). All items, both doses: dm / D at each
    estimator's top-1 site of the set (verified rows: G's and PIG's top 3 of S are verified at both doses)."""
    pi = [x for x in its if x.get("PIG")]
    if not pi:
        return None
    out = {"n_items": len(pi), "estimator": "mean over a in %s of grad_a(s) . dH_s; grad_a taken with the source-span "
                                            "token embeddings at lerp(A, B, a) (input path, EAP-IG style)" % (list(PIG_A),),
           "pig_a1_equals_clean_B_all": all(bool(x.get("pig_a1_equals_B")) for x in pi),
           "ids_differ_only_at_src_all": all(bool(x.get("pig_ids_differ_only_at_src")) for x in pi)}
    if a == 1.0:
        rows = []
        for x in full:
            if not x.get("PIG"):
                continue
            dmf = {k: dm_of(x, "O1|ans|%s|1" % k) for k in keys}
            hp = hit_v4({k: x["PIG"][k] for k in keys}, dmf, Dgap(x), THR[a], latest, tol=HIT_TOL)
            hg = hit_v4({k: x["G"][k] for k in keys}, dmf, Dgap(x), THR[a], latest, tol=HIT_TOL)
            if hp and hg and hp["eligible"] and hp["latest_hit"] is not None:
                rows.append((x, hp, hg))
        cl = [x["cluster"] for x, _, _ in rows]
        out["full_set_search"] = {
            "eligible": len(rows), "interpreted": len(rows) >= MIN_E8,
            "pig_top1_hit": bootc([hp["hit"] for _, hp, _ in rows], cl),
            "G_top1_hit": bootc([hg["hit"] for _, _, hg in rows], cl),
            "pig_minus_G_hit": bootc([hp["hit"] - hg["hit"] for _, hp, hg in rows], cl),
            "pig_hit_minus_latest": bootc([hp["hit"] - hp["latest_hit"] for _, hp, _ in rows], cl),
            "pig_regret": bootc([hp["regret"] for _, hp, _ in rows], cl),
            "pig_spearman_median": median(hp["spearman"] for _, hp, _ in rows),
            "G_spearman_median": median(hg["spearman"] for _, _, hg in rows)}
    tp, tg, cl2, same = [], [], [], []
    for x in pi:
        D = Dgap(x)
        if D < GAP:
            continue
        kp = max(keys, key=lambda k: (x["PIG"][k], k))
        kg = max(keys, key=lambda k: (x["G"][k], k))
        vp, vg = dm_of(x, "O1|ans|%s|%g" % (kp, a)), dm_of(x, "O1|ans|%s|%g" % (kg, a))
        if vp is None or vg is None:
            continue
        tp.append(vp / D)
        tg.append(vg / D)
        cl2.append(x["cluster"])
        same.append(float(kp == kg))
    out["top1_dmD_all_items"] = {"n": len(tp), "pig": bootc(tp, cl2), "G": bootc(tg, cl2),
                                 "pig_minus_G": pairedc(tp, tg, cl2), "same_top_site_frac": mean(same)}
    return out


def summarize_e8(items, K, C, family, a, set_name):
    """Plan v3 T7 (critic 2 F1, F9; VARBENCH v4 T7). Full-set endpoints on search items (dev + cross items with every
    site of the set verified at a = 1; the selector and the latest-site rule are not fitted, so these items are fair):
    the scored A2-1 endpoint is selector hit - latest-site hit (same tolerance band, cluster CI), over the set (latest
    = the set's site nearest the readout) and within loops 1..K-1 (latest = Llast@K-1; raven only). Band chance,
    strict hit and the verified-9 numbers are diagnostics. Test items: selector vs dev-tuned, uniform, random."""
    S, Sp = e8_sets(K, C)
    if family == "nr":
        Sp = S
    sset = S if set_name == "S" else Sp
    keys = [skey(s) for s in sset]
    latest = skey(latest_site(sset, C, K))
    vf = "ver_S" if set_name == "S" else "ver_Sp"
    uni = skey(last_loop_cell(family, K, C))
    its = [x for x in items if x.get("G")]
    out = {"n_sites": len(keys), "n_items": len(its), "uniform_site": uni, "latest_site": latest}
    tops = [ksite(max(keys, key=lambda k: (x["G"][k], k))) for x in its]
    out["top_site_loop_hist"] = {str(j): sum(1 for s in tops if s[1] == j) for j in sorted({s[1] for s in tops})}
    out["top_in_last_loop"] = sum(1 for s in tops if s[1] == K)
    out["top_is_latest"] = sum(1 for s in tops if skey(s) == latest)
    if family == "raven":
        out["top_is_Llast_K"] = sum(1 for s in tops if s == ("c", K, C - 1))
    full = [x for x in its if x.get("full_search")]
    if a == 1.0:
        fh = []
        for x in full:
            G = {k: x["G"][k] for k in keys}
            dmf = {k: dm_of(x, "O1|ans|%s|1" % k) for k in keys}
            fh.append((x, hit_v4(G, dmf, Dgap(x), THR[a], latest, tol=HIT_TOL)))
        ef = [(x, h) for x, h in fh if h and h["eligible"] and h["latest_hit"] is not None]
        cl = [x["cluster"] for x, _ in ef]
        hml = bootc([h["hit"] - h["latest_hit"] for _, h in ef], cl)
        out["full_set_search"] = {
            "n_items_full_search": len(full), "split_counts": {sp: sum(1 for x in full if x["split"] == sp)
                                                               for sp in SEARCH_SPLITS},
            "eligible": len(ef), "interpreted": len(ef) >= MIN_E8,
            "top1_hit": bootc([h["hit"] for _, h in ef], cl), "latest_hit": bootc([h["latest_hit"] for _, h in ef], cl),
            "hit_minus_latest": hml,
            "adds_beyond_latest": bool(hml and hml[1] is not None and hml[1] > 0),
            "diagnostic_chance_band": mean(h["chance"] for _, h in ef),
            "diagnostic_top3_random_hit": mean(h["top3_hit"] for _, h in ef),
            "diagnostic_strict_hit": mean(h["strict_hit"] for _, h in ef),
            "regret": bootc([h["regret"] for _, h in fh if h], [x["cluster"] for x, h in fh if h]),
            "latest_regret": bootc([h["latest_regret"] for _, h in fh if h], [x["cluster"] for x, h in fh if h]),
            "spearman_median": median(h["spearman"] for _, h in ef), "n_band_median": median(h["n_band"] for _, h in ef)}
        if family == "raven":
            klt = [k for k in keys if ksite(k)[1] <= K - 1]
            lt_latest = skey(("c", K - 1, C - 1))
            lh = []
            for x in full:
                G = {k: x["G"][k] for k in klt}
                dmf = {k: dm_of(x, "O1|ans|%s|1" % k) for k in klt}
                h = hit_v4(G, dmf, Dgap(x), THR[a], lt_latest, tol=HIT_TOL)
                if h and h["eligible"] and h["latest_hit"] is not None:
                    lh.append((x, h))
            lcl = [x["cluster"] for x, _ in lh]
            lml = bootc([h["hit"] - h["latest_hit"] for _, h in lh], lcl)
            out["full_set_search"]["loops_lt_K"] = {
                "baseline_site": lt_latest, "eligible": len(lh), "scored": len(lh) >= 10,
                "top1_hit": bootc([h["hit"] for _, h in lh], lcl), "latest_hit": bootc([h["latest_hit"] for _, h in lh], lcl),
                "hit_minus_latest": lml, "adds_beyond_latest": bool(lml and lml[1] is not None and lml[1] > 0),
                "diagnostic_chance_band": mean(h["chance"] for _, h in lh)}
        n_low = fn = n_all = 0
        miss = carried = 0
        fcl = []
        fxs = []
        for x in full:
            D = Dgap(x)
            if D < GAP:
                continue
            for k in keys:
                g, d = x["G"].get(k), dm_of(x, "O1|ans|%s|1" % k)
                if g is None or d is None:
                    continue
                n_all += 1
                if abs(g) < FN_G * D:
                    n_low += 1
                    fn += int(d / D >= 0.5)
                    fxs.append(float(d / D >= 0.5))
                    fcl.append(x["cluster"])
                if d / D >= 0.5:
                    carried += 1
                    miss += int(abs(g) < FN_G * D)
        out["fn_rate_search"] = {"n_sites": n_all, "n_low_G": n_low, "fn_rate": bootc(fxs, fcl),
                                 "n_carried_sites": carried,
                                 "carried_with_low_G": round(miss / carried, 4) if carried else None,
                                 "rule": "|G| < %.2f D; carried = verified e >= 0.5 (A7)" % FN_G}
    out["pig"] = summarize_pig(its, full, keys, latest, a)
    v9 = []
    for x in its:
        G = {k: x["G"][k] for k in x[vf]}
        dmv = {k: dm_of(x, "O1|ans|%s|%g" % (k, a)) for k in x[vf]}
        v9.append((x, hit_v3(G, dmv, Dgap(x), THR[a], tol=HIT_TOL)))
    e9 = [(x, h) for x, h in v9 if h and h["eligible"]]
    out["verified9"] = {"eligible": len(e9), "hit_tol": mean(h["hit"] for _, h in e9),
                        "diagnostic_chance_band_9": mean(h["chance"] for _, h in e9),
                        "top3_random_hit": mean(h["top3_hit"] for _, h in e9),
                        "spearman_median": median(h["spearman"] for _, h in e9),
                        "note": "the dose clause (a = 0.25 vs 1) is read here on all items; chance over 9 verified "
                                "sites is set by the verify design"}
    test = [x for x in its if x["split"] == "test"]
    cl = [x["cluster"] for x in test]
    D = [Dgap(x) for x in test]

    def dmD(xs):
        return [None if v is None or d < GAP else v / d for v, d in zip(xs, D)]
    pred = dmD([dm_of(x, "O1|ans|%s|%g" % (max(x[vf], key=lambda k: (x["G"][k], k)), a)) for x in test])
    top3 = []
    for x in test:
        rk = sorted(x[vf], key=lambda k: (-x["G"][k], k))[:3]
        v = [dm_of(x, "O1|ans|%s|%g" % (k, a)) for k in rk]
        top3.append(None if any(t is None for t in v) else sum(v) / 3.0)
    base = {"uniform": dmD([dm_of(x, "O1|ans|%s|%g" % (uni, a)) for x in test]),
            "random": dmD([dm_of(x, "O1|ans|%s|%g" % (x["rnd_site"], a)) for x in test]),
            "random_in_top3": dmD(top3)}
    out["test"] = {"n_test": len(test), "pred_dmD": bootc(pred, cl)}
    for nm, vv in base.items():
        out["test"]["%s_dmD" % nm] = bootc(vv, cl)
        out["test"]["pred_minus_%s" % nm] = pairedc(pred, vv, cl)
    out["_pred"] = pred
    tf = [x["t_grad"] / x["t_fwd"] for x in its if x.get("t_grad") and x.get("t_fwd")]
    tpg = [x["t_pig"] / x["t_fwd"] for x in its if x.get("t_pig") and x.get("t_fwd")]
    out["cost_fwd_equiv"] = {"selector_per_item": mean(tf), "pig_per_item": mean(tpg), "full_search_per_item": len(keys)}
    return out


def add_dev_tuned(out, items, res, a, set_name):
    dt = res.get("dev_tuned", {}).get(set_name, {})
    site = dt.get("site")
    out["dev_tuned_site"] = site
    if site is None:
        return
    test = [x for x in items if x.get("G") and x["split"] == "test"]
    cl = [x["cluster"] for x in test]
    vv = [None if dm_of(x, "O1|ans|%s|%g" % (site, a)) is None or Dgap(x) < GAP
          else dm_of(x, "O1|ans|%s|%g" % (site, a)) / Dgap(x) for x in test]
    out["test"]["dev_tuned_dmD"] = bootc(vv, cl)
    out["test"]["pred_minus_dev_tuned"] = pairedc(out["_pred"], vv, cl)


def nr_window(items):
    """L* = last NR layer with mean source e >= 0.5; cluster-bootstrap interval of L* (200 resamples)."""
    cl = [x["cluster"] for x in items]
    means = []
    for l in range(16):
        b = bootc([e_of(x, "O1|src|c1.%d|1" % l) for x in items], cl)
        means.append(None if b is None else b[0])
    ls = [l for l, v in enumerate(means) if v is not None and v >= 0.5]
    units = sorted({u for c in cl for u in _ulist(c)}, key=str)
    by = {u: [x for x in items if u in _ulist(x["cluster"])] for u in units}
    g = torch.Generator().manual_seed(0)
    draws = []
    for _ in range(200 if len(units) > 1 else 0):
        pick = torch.randint(len(units), (len(units),), generator=g).tolist()
        sub = [x for i in pick for x in by[units[i]]]
        ms = [mean(e_of(x, "O1|src|c1.%d|1" % l) for x in sub) for l in range(16)]
        lsb = [l for l, v in enumerate(ms) if v is not None and v >= 0.5]
        draws.append(max(lsb) if lsb else -1)
    draws.sort()
    return {"means": means, "L_star": max(ls) if ls else None,
            "ci95": [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]] if draws else None}


def summarize(out_dir, plan=None):
    res, metas = load_results(out_dir)
    ck = {"plan_sha256": sha256_file(plan) if plan and os.path.exists(plan) else None,
          "code_md5": sorted({m.get("k17_public_md5") for m in metas.values()}),
          "reused_md5": {k: v for m in metas.values() for k, v in (m.get("reused_md5") or {}).items()},
          "checkpoints": {k: {"repo": v[0], "revision": v[1][:8]} for k, v in CKPT.items()},
          "load_audit": {k: m.get("load_audit") for k, m in metas.items()},
          "endpoint": {"D_sets": "first differing answer token at r, float32 readout; Llama: the full answer (one "
                                 "token after the shared space); Huginn D2/D5: leading digit, D4 T2: first differing "
                                 "digit; full-answer filter on the remaining digits (critic F10)",
                       "VB": {"source_rows": "full-span log-odds e_LO", "answer_rows": "first-token e"}},
          "exec_index_map": {"huginn": "2 + 4(j-1) + l", "mcleish": "4 + 6(j-1) + l", "asserted": "every forward"},
          "nr_map": {"prelude_end": "NR layer 3 <-> McLeish e", "loop_end_j": "NR layer 3 + ceil(8 j / K)",
                     "table_K32": {str(j): 3 + -(-8 * j // 32) for j in JSET[32]}, "coda_input": "NR layer 11",
                     "label": "same-data non-recurrent reference, not a depth control (A5)"},
          "hook_identity": {}, "items": {}, "clusters": {}, "pc_final": {}, "sz": {}, "sc_harness": {}, "pc_site": {},
          "e1_cells": {}, "e5_cells": {}, "decomposition": {}, "sc_cells": {}, "answer_classes": {},
          "presence_validation": {}, "e8": {}, "seed_sensitivity": {}, "nr_window": {}, "gpu_minutes": {},
          "timing": {}, "tiny_floor": {}, "e5_rules": {}, "cross_model_D1_capitals": {},
          "tiny_floor_note": "diagnostic only (review 3 F7): the bf16 lerp at a = 1e-3 rounds most elements away, so "
                             "it cannot fail; the a = 0.25 class gate uses a025_noise (bf16 vs fp32 on the FD items)",
          "a025_noise": {}, "pre_run": {k: m.get("pre_run") for k, m in metas.items() if m.get("pre_run")},
          "datasets_short_by_budget": [],
          "o2_label": O2_LABEL,
          "split_fraction": {"rule": "unit -> dev if the first hex digit of sha256(prefix + unit) is even, else test; "
                                     "an item is dev (test) only if every unit hashes dev (test), else cross",
                             "expected": {"one-unit items (VB)": {"dev": 0.5, "test": 0.5},
                                          "dyadic items (D1, CF, D2, D4)": {"dev": 0.25, "test": 0.25, "cross": 0.5}},
                             "use": "E8 full search on dev + cross items (A2-1 endpoints); dev-tuned site fitted on dev "
                                    "items, scored on test items; PN3 (b), (c) scored on non-dev items"},
          "fd_check": {}}
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("fd_") and f.endswith(".json"):
            fdd = json.load(open(os.path.join(out_dir, f)))
            for mk, v in fdd.get("models", {}).items():
                ra = v.get("rank_agreement_bf16_vs_fp32") or fd_rank_agreement(v.get("items", []),
                                                                               v["summary"]["floor"]["F"])
                ck["fd_check"][mk] = {"verdict": v["summary"]["verdict"], "floor": v["summary"]["floor"],
                                      "classes": v["summary"]["classes"], "n_items": v["n_items"],
                                      "grad_run_identical_all": v.get("grad_run_identical_all"),
                                      "rank_agreement_bf16_vs_fp32": {k: w for k, w in ra.items() if k != "per_item"},
                                      "spearman_G_bf16_vs_fp32": v.get("spearman_G_bf16_vs_fp32")}
                ck["a025_noise"][mk] = v.get("a025_noise") or a025_noise(v.get("items", []))
    for nm, m in metas.items():
        ck["gpu_minutes"][nm] = round((m.get("process_minutes") or 0)
                                      + sum(p.get("process_minutes") or 0 for p in m.get("previous_runs", [])), 2)
        for k, v in (m.get("hook_identity") or {}).items():
            ck["hook_identity"][k] = v
    for (mo, ds), d in sorted(res.items()):
        items = d["items"]
        key = "%s|%s" % (mo, ds)
        fam, K, C = arch_of(mo)
        tgt = d["meta"].get("target")
        short = tgt is not None and len(items) < tgt
        stp = bool(d["meta"].get("stopped_budget", False))
        ck["items"][key] = dict(d["meta"].get("scan_stats", {}), target=tgt, scored=len(items), stopped_budget=stp,
                                scan_complete=d["meta"].get("scan_complete"), rows_missing=d["meta"].get("rows_missing"),
                                descriptive=bool(len(items) < MIN_KEPT),
                                flag=("short of target: stopped by budget" if short and stp else
                                      ("short of target: pool or clean filter" if short else None)),
                                search_plan=d["meta"].get("search_plan"), e8_gate=d["meta"].get("e8_gate"))
        if short and stp:                          # review 3 F10: never silent
            ck["datasets_short_by_budget"].append(key)
        dv = {u for x in items if x["split"] == "dev" for u in _ulist(x["cluster"])}
        ts = {u for x in items if x["split"] == "test" for u in _ulist(x["cluster"])}
        ck["clusters"][key] = {"dev_test_unit_overlap": len(dv & ts),
                               "n_units": len({u for x in items for u in _ulist(x["cluster"])}),
                               "split": {s: sum(1 for x in items if x["split"] == s) for s in ("dev", "test", "cross")}}
        if not items:
            continue
        ck["timing"][key] = {"t_fwd_median": median(x["t_fwd"] for x in items),
                             "t_item_median": median(x.get("t_item") for x in items),
                             "t_grad_median": median(x.get("t_grad") for x in items),
                             "T_median": median(x["T"] for x in items), "rows_median": median(x["n_rows"] for x in items)}
        vb = ds == "VB"
        pcf = [x["rows"]["PCF"]["exact"] for x in items if "PCF" in x["rows"]]
        ck["pc_final"][key] = {"n": len(pcf), "all_exact": all(pcf), "tag": "harness"}
        sz = [x["rows"][k]["exact"] for x in items for k in ("SZ|O1", "SZ|RND") if k in x["rows"]]
        ck["sz"][key] = {"n": len(sz), "all_exact": all(sz), "tag": "harness"}
        sh = [v["exact"] for x in items for k, v in x["rows"].items() if k.startswith(("SC|", "SCID|")) and "exact" in v]
        ck["sc_harness"][key] = {"n": len(sh), "all_exact": all(sh), "tag": "harness"}
        ps = [v["exact"] for x in items for k, v in x["rows"].items() if k.startswith("PCS|")]
        ck["pc_site"][key] = {"n_rows": len(ps), "n_items": sum(1 for x in items if x.get("subset")),
                              "all_exact": all(ps), "tag": "harness"}
        if mo in PB8_ONLY:                         # the K = 8 runner has harness and PB8 rows only (k16_vs_k32)
            continue
        strata =("pos_len", "rel", "pos_len_x_rel") if ds in ("D1", "CF") else \
            (("pos_len", "ST2", "ST3", "cont") if vb else ("pos_len", "rel", "endpoint"))
        sc = {}
        if fam == "raven":
            for nm_ in sc_names(fam, K, ds):
                sc[nm_] = cell(items, "SC|%s" % skey(name2site(nm_, C)), strata=(), lo=vb)
            ck["sc_cells"][key] = sc

        def sc_at(nm_):
            c = sc.get(nm_)
            if not c:
                return None
            return (c.get("e_LO") or c["e"])[0] if vb else c["e"][0]

        def sc_of(j):
            return sc_at("Llast@%d" % j)
        efn = elo_of if vb else e_of
        e1 = {}
        for name in e1_names(fam, K, ds):
            s = name2site(name, C)
            k = skey(s)
            lab = None
            if fam == "raven" and s == ("c", K, C - 1):
                lab = "coda-gated"
            elif s == last_loop_cell(fam, K, C):
                lab = "last-loop answer cell"
            elif s == ("c", K, C - 1):
                lab = "readout-feeding"
            e1[name] = {"src_O1": cell(items, "O1|src|%s|1" % k, null=True, lo=vb, strata=strata),
                        "ans_O1": cell(items, "O1|ans|%s|1" % k, null=True, strata=())}
            if any(("O1|ans|%s|0.25" % k) in x["rows"] for x in items):
                e1[name]["ans_O1_a0.25"] = cell(items, "O1|ans|%s|0.25" % k, strata=())
            if any(("RND|src|%s" % k) in x["rows"] for x in items):
                e1[name]["src_RND"] = cell(items, "RND|src|%s" % k, rnd=True, strata=())
                e1[name]["ans_RND"] = cell(items, "RND|ans|%s" % k, rnd=True, strata=())
            for pos in ("src", "ans"):
                if any(("O2|%s|%s" % (pos, k)) in x["rows"] for x in items):
                    dk = "%s|%s" % (pos, k)
                    e1[name]["%s_O2" % pos] = cell(items, "O2|%s|%s" % (pos, k), strata=())
                    e1[name]["%s_O2_diag" % pos] = [mean((x.get("o2diag") or {}).get(dk, [None, None])[0] for x in items),
                                                    mean((x.get("o2diag") or {}).get(dk, [None, None])[1] for x in items)]
            e1[name]["label"] = lab
            if any(("RND|src|%s" % k) in x["rows"] for x in items):
                e1[name]["src_vs_RND"] = label_vs_rnd(items, "O1|src|%s|1" % k, "RND|src|%s" % k, fn=efn)
                e1[name]["ans_vs_RND"] = label_vs_rnd(items, "O1|ans|%s|1" % k, "RND|ans|%s" % k)
            if fam == "raven" and s[0] in ("p", "c") and s[1] < K:
                scs = sc_of(s[1])
                e1[name]["SC_same_loop"] = scs
                se = (e1[name]["src_O1"] or {}).get("e_LO" if vb else "e") or (e1[name]["src_O1"] or {}).get("e")
                tn = ((e1[name]["src_O1"] or {}).get("tost") or {}).get("null_by_tost")
                null = bool(se and se[0] is not None and (se[0] <= 0.15 or tn))
                e1[name]["src_attribution"] = (
                    "n/a (source not null)" if not null else
                    ("recurrence forgets state (SC <= 0.15)" if scs is not None and scs <= 0.15 else
                     ("source position (SC >= 0.5)" if scs is not None and scs >= 0.5 else
                      "unattributed (SC between 0.15 and 0.5, or missing)")))
            if any(("O2|src|%s" % k) in x["rows"] or ("O2|ans|%s" % k) in x["rows"] for x in items):
                e1[name]["O2_label"] = O2_LABEL
            e1[name]["delta_src_sum_mean"] = mean((x.get("dnorm") or {}).get(k) for x in items)
            tk = "O1|ans|%s|%g" % (k, TINY_A)
            tv = [abs(dm_of(x, tk)) / Dgap(x) for x in items if dm_of(x, tk) is not None and Dgap(x) >= GAP]
            if tv:
                ck["tiny_floor"].setdefault(key, {})[name] = {
                    "floor_abs_dm_over_D": mean(tv), "max": round(max(tv), 5), "n": len(tv),
                    "changed_frac_mean": mean((x["rows"].get(tk) or {}).get("changed") for x in items),
                    "a": TINY_A, "diagnostic_only": True}
        ck["e1_cells"][key] = e1
        if fam == "raven":
            e5 = {}
            ref = "INP|1"
            for j in ((K - 1, K) if vb else IN1_J[K]):
                e5["IN-single(%d)" % j] = cell(items, "IN1|%d" % j, null=j <= K - 2, strata=(), lo=vb)
                e5["IN-single(%d)/IN-persist(1)" % j] = cell(items, "IN1|%d" % j, strata=(),
                                                              fn=lambda x, kk: ratio_of(x, kk, ref))
            for j in (vb_inp(K) if vb else INP_J[K]):
                e5["IN-persist(%d)" % j] = cell(items, "INP|%d" % j, strata=(), lo=vb)
                e5["IN-persist(%d)/IN-persist(1)" % j] = cell(items, "INP|%d" % j, strata=(),
                                                              fn=lambda x, kk: ratio_of(x, kk, ref))
            e5["IN-persist-down(1)"] = cell(items, "INPD|1", strata=())
            dn = [x.get("dnorm") or {} for x in items]
            we = mean(v.get("We_e") for v in dn)
            wd = mean(v.get("We_e_down") for v in dn)

            def absd(n):
                return None if we is None else round(n * we, 4)
            for nm_, c in list(e5.items()):        # sum |W_e delta e| over adapter calls next to every E5 cell (P1)
                if not isinstance(c, dict) or "/" in nm_:
                    continue
                if nm_.startswith("IN-single("):
                    c["abs_delta_sum_We"] = absd(1)
                elif nm_.startswith("IN-persist-down"):
                    c["abs_delta_sum_We"] = None if wd is None else round(K * wd, 4)
                elif nm_.startswith("IN-persist("):
                    c["abs_delta_sum_We"] = absd(K - int(nm_[11:-1]) + 1)
            e5["RND-input-persist(1)"] = cell(items, "RNDIN|1", rnd=True, strata=(), lo=vb)
            if e5["RND-input-persist(1)"]:
                e5["RND-input-persist(1)"]["abs_delta_sum_We"] = "same |delta e| per position as IN-persist(1); " \
                                                                 "W_e image not logged"
            for j in ((K // 2, K - 1) if vb else IN1S_J[K]):
                c = cell(items, "IN1S|%d" % j, strata=(), lo=vb)
                if c:
                    c["abs_delta_sum_We"] = absd(K - j + 1)
                    c["n_scale"] = K - j + 1
                    c["label"] = "off-manifold (n = K)" if j == 1 else "absolutely matched to IN-persist(%d)" % j
                e5["IN-single(%d)x%d" % (j, K - j + 1)] = c
                e5["abs-matched IN-persist(%d) - IN-single(%d)x%d" % (j, j, K - j + 1)] = pairedc(
                    [efn(x, "INP|%d" % j) for x in items], [efn(x, "IN1S|%d" % j) for x in items],
                    [x["cluster"] for x in items])
            for j in sorted(set((K - 1, K) if vb else IN1_J[K]) & set((vb_inp(K) if vb else INP_J[K]))):
                e5["rel-matched IN-persist(%d) - IN-single(%d)" % (j, j)] = pairedc(
                    [efn(x, "INP|%d" % j) for x in items], [efn(x, "IN1|%d" % j) for x in items],
                    [x["cluster"] for x in items])
            rules = {}
            for j in ((K - 1, K) if vb else IN1_J[K]):
                c1 = e5.get("IN-single(%d)" % j) or {}
                cp = e5.get("IN-persist(%d)" % j) if j < K else e5.get("IN-single(%d)" % K)
                v1 = (c1.get("e_LO") or c1.get("e") or [None])[0] if vb else (c1.get("e") or [None])[0]
                vp = None if not cp else ((cp.get("e_LO") or cp.get("e") or [None])[0] if vb else cp["e"][0])
                t1 = (c1.get("tost") or {}).get("null_by_tost")
                null = v1 is not None and (v1 <= 0.15 or bool(t1))
                rules["IN-single(%d)" % j] = {
                    "e_single": v1, "e_persist_same_j": vp,
                    "reading": ("not erased" if v1 is not None and not null else
                                ("erased by refresh (IN-persist(j) >= 0.5)" if vp is not None and vp >= 0.5 else
                                 ("not testable (IN-persist(j) not run)" if v1 is not None and vp is None else
                                  ("no working input edit at j" if v1 is not None else None))))}
            for j in (vb_inp(K) if vb else INP_J[K]):
                rules["IN-persist(%d) vs RND input" % j] = label_vs_rnd(items, "INP|%d" % j, "RNDIN|1", fn=efn)
            ck["e5_rules"][key] = rules
            kk1, kk2 = "O1|src|%s|1" % skey(("c", K - 1, C - 1)), "O1|src|%s|1" % skey(("p", K))
            e5["post_inj(K)_minus_Llast@(K-1)_src"] = pairedc([efn(x, kk2) for x in items],
                                                             [efn(x, kk1) for x in items], [x["cluster"] for x in items])
            e5["post_inj(K)_minus_Llast@(K-1)_src_abs_delta"] = {
                "post_inj(K)_dh_sum": mean(v.get(skey(("p", K))) for v in dn),
                "Llast@(K-1)_dh_sum": mean(v.get(skey(("c", K - 1, C - 1))) for v in dn),
                "Llast@(K-1)_Ws_dh_sum (its push into loop K)": mean(v.get("Ws_" + skey(("c", K - 1, C - 1))) for v in dn)}
            e5["norms"] = {"We_de_per_call_sum": mean(v.get("We_e") for v in dn), "de_sum": mean(v.get("e") for v in dn),
                           "state_dh_Llast@(K-1)_sum": mean(v.get(skey(("c", K - 1, C - 1))) for v in dn),
                           "Ws_dh_Llast@(K-1)_sum": mean(v.get("Ws_" + skey(("c", K - 1, C - 1))) for v in dn),
                           "post_inj(K)_dh_sum": mean(v.get(skey(("p", K))) for v in dn),
                           "We_de_down_per_call_sum": wd,
                           "note": "IN-persist(j) applies K-j+1 calls of We_de, IN-single(j) one, IN-single(j)xn one "
                                   "call of n We_de (P1, critic 2 F3)"}
            ck["e5_cells"][key] = e5
            dec = {}
            for j in (() if vb else SRCIN_J[K]):
                ks = "O1|src|%s|1" % skey(("c", j, C - 1))
                kin = "INP|%d" % (j + 1) if (j + 1) in INP_J[K] else ("IN1|%d" % K if j + 1 == K else None)
                if kin is None:
                    continue
                dec["j=%d" % j] = {"SC": sc_of(j), "src_O1": (cell(items, ks, strata=()) or {}).get("e"),
                                   "IN-persist(j+1)": (cell(items, kin, strata=()) or {}).get("e"),
                                   "SRCIN": (cell(items, "SRCIN|%d" % j, strata=()) or {}).get("e"),
                                   "state_own=SRCIN-INP(j+1)": pairedc([e_of(x, "SRCIN|%d" % j) for x in items],
                                                                       [e_of(x, kin) for x in items],
                                                                       [x["cluster"] for x in items])}
            ck["decomposition"][key] = dec
        pv, cls = {}, {}
        for name in e1_names(fam, K, ds):
            s = name2site(name, C)
            k = skey(s)
            pss = [x["pval"][k] for x in items if x.get("pval") and k in x["pval"]]
            ok = [p[1] - p[0] >= 0.5 for p in pss if p[0] is not None and p[1] is not None]
            gate = (sum(ok) / len(ok) >= 0.5) if len(ok) >= 10 else None
            pv[name] = {"n_items_with_form": len(pss), "n_valid": len(ok), "n_pass": sum(ok), "gate": gate}
            if fam == "nr" or s[1] >= K:
                continue
            for a in A_VERIFY:
                rk = "O1|ans|%s|%g" % (k, a)
                if not any(rk in x["rows"] for x in items):
                    continue
                ef_all = mean(e_of(x, rk) for x in items)
                gated = [x for x in items if eown_of(x, rk) is not None]
                ef, eo = mean(e_of(x, rk) for x in gated), mean(eown_of(x, rk) for x in gated)
                djd = mean((Dj_of(x, s[1]) or 0) / Dgap(x) for x in items if Dgap(x) >= GAP and x.get("loopsA"))
                if ef_all is not None and ef_all >= 0.5 * a:
                    c0 = "carried"
                elif len(gated) < 10:
                    c0 = "not testable at site (D_j < 2 on all but %d items)" % len(gated)
                else:
                    c0 = answer_class(a, ef, e_own=eo)
                scj = sc_of(s[1]) if fam == "raven" else None
                if fam == "raven" and c0 in ("not carried", "partial", "no answer state at site"):
                    if scj is not None and scj <= 0.15:
                        c0 = "state forgotten"            # critic 1 F3: both hypotheses predict the null here
                    elif scj is None or scj < 0.5:
                        c0 = "not read (SC < 0.5)"
                nz = (ck["a025_noise"].get(mo) or {}).get(name)        # review 3 F7: bf16 vs fp32 at a = 0.25
                if a == 0.25 and c0 != "carried" and not (nz and nz["gate_pass"]):
                    c0 = ("not scored (a = 0.25 bf16 noise >= %.4f)" % A025_NOISE_CUT if nz else
                          "not scored (a = 0.25 bf16 noise not measured at this site)")
                cls["%s|a=%g" % (name, a)] = {"e_final_all": ef_all, "n_items": len(items), "n_gated_Dj_ge_2": len(gated),
                                              "e_final_gated": ef, "e_own_gated": eo, "mean_Dj_over_D": djd,
                                              "class": c0, "SC_same_loop": scj,
                                              "a025_noise_mean": (nz or {}).get("mean_abs_de") if a == 0.25 else None}
        ck["presence_validation"][key] = pv
        ck["answer_classes"][key] = cls
        if fam == "nr":
            ck["nr_window"][key] = {"all": nr_window(items),
                                    "dev": nr_window([x for x in items if x["split"] == "dev"])}
        if d["meta"].get("seed_sensitivity"):
            ss = d["meta"]["seed_sensitivity"]
            e0 = {x["pair_id"]: e_of(x, ss["cell"]) for x in items}
            sso = {}
            for off, v in ss["offsets"].items():
                cm = [p for p in v["e"] if e0.get(p) is not None and v["e"][p] is not None]
                sso[off] = {"kept": v["kept_on_same_scan_set"], "kept_off0": v["kept_off0"], "sym_diff": v["sym_diff"],
                            "cell_e_off": mean(v["e"][p] for p in cm), "cell_e_off0": mean(e0[p] for p in cm),
                            "max_abs_item_diff": max([abs(v["e"][p] - e0[p]) for p in cm] or [None]),
                            "status_changed": v.get("status_changed"), "n_rescanned": v.get("n_rescanned"),
                            "rescan_complete": v.get("complete")}
            ck["seed_sensitivity"][key] = sso
        if any(x.get("G") for x in items):
            e8 = {}
            for set_name in (("S", "Sp") if fam == "raven" else ("S",)):
                for a in A_VERIFY:
                    o = summarize_e8(items, K, C, fam, a, set_name)
                    add_dev_tuned(o, items, d, a, set_name)
                    o.pop("_pred", None)
                    fdv = (ck["fd_check"].get(mo) or {}).get("verdict")
                    o["fd_verdict"] = fdv
                    gate = d["meta"].get("e8_gate") or {}
                    o["e8_gate"] = gate                  # review 3 F4: pilot gate; a stopped search = descriptive
                    o["descriptive_by_gate"] = gate.get("continue") is False
                    o["interpreted"] = bool(fdv == "PASS" and gate.get("continue") is not False
                                            and (o.get("full_set_search") or {}).get("interpreted", True))
                    e8["%s|a=%g" % (set_name, a)] = o
            ck["e8"][key] = e8
    caps = {}
    for mo in ("huginn", "mcleish", "llama_nr", "ouro_think"):
        d = res.get((mo, "D1"))
        if d:
            caps[mo] = {x["pair_id"]: x for x in d["items"] if x.get("rel") == "capital"}
    if caps:
        common = sorted(set.intersection(*[set(v) for v in caps.values()]))
        cm = {"models": sorted(caps), "n_common_capital_items": len(common), "cells": {}}
        for mo, v in caps.items():
            fam, K, C = arch_of(mo)
            its = [v[p] for p in common]
            cm["cells"][mo] = {nm: {"src_O1": (cell(its, "O1|src|%s|1" % skey(name2site(nm, C)), strata=()) or {}).get("e"),
                                    "ans_O1": (cell(its, "O1|ans|%s|1" % skey(name2site(nm, C)), strata=()) or {}).get("e")}
                               for nm in e1_names(fam, K, "D1")} if its else {}
        ck["cross_model_D1_capitals"] = cm
    ck["pn3_paired"] = pn3(res)
    ck["k16_vs_k32"] = k16_vs_k32(res)
    ck["predictions_vs_outcomes"] = predictions(ck)
    atomic_json(os.path.join(out_dir, "checks.json"), ck)
    write_summary_md(out_dir, ck)
    return ck


def _paired(res, m1, k1, m2, k2, ds, filt=None, d2=None):
    a, b = res.get((m1, ds)), res.get((m2, d2 or ds))
    if not a or not b:
        return None
    ib = {x["pair_id"]: x for x in b["items"]}
    common = [x for x in a["items"] if x["pair_id"] in ib and (filt is None or filt(x))]
    if not common:
        return None
    xs = [e_of(x, k1) for x in common]
    ys = [e_of(ib[x["pair_id"]], k2) for x in common]
    cl = [x["cluster"] for x in common]
    return {"n": len(common), "first": bootc(xs, cl), "second": bootc(ys, cl), "first_minus_second": pairedc(xs, ys, cl)}


def pn3(res):
    """PN3 (paired McLeish vs NR on identical items). Plan v3 (critic 2 F6): NR's read window L* is measured on dev
    items first; (b) and (c) compare at NR layer min(N(j), lower 95% bound of L*_dev) and are scored on non-dev items;
    the contrast NR src layer l minus McLeish src Llast@16 is reported for every l <= L*_dev (curve)."""
    out = {}
    nd = lambda x: x["split"] != "dev"   # noqa: E731
    for ds in ("D1", "D2", "D4"):
        nr, mc = res.get(("llama_nr", ds)), res.get(("mcleish", ds))
        if not nr or not mc:
            continue
        dev = [x for x in nr["items"] if x["split"] == "dev"]
        w = nr_window(dev) if dev else None
        lo = None
        if w and w.get("L_star") is not None:
            lo = w["ci95"][0] if w.get("ci95") else w["L_star"]
        bl = None if lo is None or lo < 0 else min(7, lo)
        cl_ = None if lo is None or lo < 0 else min(4, lo)
        d = {"window_dev": w, "b_layer": bl, "c_layer": cl_, "scored_on": "non-dev items (test + cross)",
             "a_prelude_end: NR src L3 vs McLeish IN-persist(1)": _paired(res, "llama_nr", "O1|src|c1.3|1", "mcleish", "INP|1", ds)}
        d["b_mid: NR src L(b) vs McLeish src Llast@16"] = None if bl is None else _paired(
            res, "llama_nr", "O1|src|c1.%d|1" % bl, "mcleish", "O1|src|c16.5|1", ds, filt=nd)
        d["c_early: NR src L(c) vs McLeish src Llast@1"] = None if cl_ is None else _paired(
            res, "llama_nr", "O1|src|c1.%d|1" % cl_, "mcleish", "O1|src|c1.5|1", ds, filt=nd)
        d["d_coda_input: NR ans L11 vs McLeish ans Llast@32"] = _paired(res, "llama_nr", "O1|ans|c1.11|1", "mcleish",
                                                                       "O1|ans|c32.5|1", ds)
        if w and w.get("L_star") is not None:
            d["curve_b: NR src layer l minus McLeish src Llast@16"] = {
                str(l): (_paired(res, "llama_nr", "O1|src|c1.%d|1" % l, "mcleish", "O1|src|c16.5|1", ds, filt=nd)
                         or {}).get("first_minus_second") for l in range(0, w["L_star"] + 1)}
        out[ds] = d
    return out


def k16_vs_k32(res):
    """PB8 (plan v3, critic 2 F10): source O1 at Llast@(K-r) at equal loops remaining r across K in {8, 16, 32}, same
    items. A pair is scored only when both cells are off the floor (mean e >= 0.2 in each)."""
    out = {}
    pairs = (("huginn16", "huginn"), ("huginn8", "huginn16"), ("huginn8", "huginn"))
    Ks = {"huginn": 32, "huginn16": 16, "huginn8": 8}
    for ds in HUGINN16_DS:
        d = {}
        for m1, m2 in pairs:
            for rr in PB8_R:
                j1, j2 = Ks[m1] - rr, Ks[m2] - rr
                if j1 < 1:
                    continue
                v = _paired(res, m1, "O1|src|c%d.3|1" % j1, m2, "O1|src|c%d.3|1" % j2, ds)
                if v:
                    f, sc = (v.get("first") or [None])[0], (v.get("second") or [None])[0]
                    v["off_floor"] = None if f is None or sc is None else (f >= 0.2 and sc >= 0.2)
                    v["depth_fraction"] = [round(j1 / Ks[m1], 3), round(j2 / Ks[m2], 3)]
                d["r=%d %s vs %s (Llast@%d vs Llast@%d)" % (rr, m1, m2, j1, j2)] = v
        d["IN-single(K-1) huginn16 vs huginn"] = _paired(res, "huginn16", "IN1|15", "huginn", "IN1|31", ds)
        out[ds] = d
    return out


def _g(ck, key, name, field):
    c = (ck["e1_cells"].get(key, {}).get(name) or {}).get(field)
    return None if not c else c["e"][0]


def _tost(ck, key, name, field):
    c = (ck["e1_cells"].get(key, {}).get(name) or {}).get(field)
    return None if not c or not c.get("tost") else c["tost"]["null_by_tost"]


def _e5(ck, key, n):
    c = ck["e5_cells"].get(key, {}).get(n)
    return None if not isinstance(c, dict) or "e" not in c else c["e"][0]


def _le(v, t):
    return None if v is None else v <= t


def _ge(v, t):
    return None if v is None else v >= t


def _in(v, lo, hi):
    return None if v is None else lo <= v <= hi


def _all(xs):
    xs = list(xs)
    return None if not xs or any(v is None for v in xs) else all(xs)


def predictions(ck):
    """Plan section 10 (v3). A cell is scored only with >= 40 items (T8); None = not scored."""
    out = {}

    def ok_n(key):
        return (ck["items"].get(key, {}).get("scored") or 0) >= MIN_KEPT
    p1 = {}
    for ds in ("D1", "D2", "D4", "VB"):
        k = "mcleish|%s" % ds
        if not ok_n(k):
            continue
        js = [j for j in (1, 2, 8, 16) if "Llast@%d" % j in ck["e1_cells"].get(k, {})]
        p1["%s src Llast@j<=16 <= 0.10" % ds] = _all(_le(_g(ck, k, "Llast@%d" % j, "src_O1"), 0.10) for j in js)
        if ds in ("D1", "D2"):
            p1["%s null by TOST at Llast@1,2,8,16" % ds] = _all(_tost(ck, k, "Llast@%d" % j, "src_O1") for j in js)
    k = "mcleish|D1"
    if ok_n(k):
        p1["D1 src Llast@31 <= 0.30"] = _le(_g(ck, k, "Llast@31", "src_O1"), 0.30)
        p1["D1 src post_inj@32 in [0.3, 0.9]"] = _in(_g(ck, k, "post_inj@32", "src_O1"), 0.3, 0.9)
        p1["D1 src Llast@32 (coda-gated) in [0.1, 0.7]"] = _in(_g(ck, k, "Llast@32", "src_O1"), 0.1, 0.7)
    out["PB1"] = p1
    p2 = {}
    for ds in ("D1", "D2", "D4"):
        k = "huginn|%s" % ds
        if not ok_n(k):
            continue
        p2["%s src Llast@1, Llast@2 <= 0.15" % ds] = _all(_le(_g(ck, k, "Llast@%d" % j, "src_O1"), 0.15) for j in (1, 2))
        p2["%s src Llast@16 <= 0.30" % ds] = _le(_g(ck, k, "Llast@16", "src_O1"), 0.30)
        p2["%s src Llast@31 in [0.2, 0.8]" % ds] = _in(_g(ck, k, "Llast@31", "src_O1"), 0.2, 0.8)
        if ds == "D1":
            p2["D1 src post_inj@32 >= 0.4"] = _ge(_g(ck, k, "post_inj@32", "src_O1"), 0.4)
        h, mc = _g(ck, k, "Llast@16", "src_O1"), _g(ck, "mcleish|%s" % ds, "Llast@16", "src_O1")
        if ds != "D1":       # critic 2 F8: cross-model D1 comparisons only on capitals (cross_model_D1_capitals)
            p2["%s Huginn - McLeish at Llast@16 >= 0.05" % ds] = None if h is None or mc is None else h - mc >= 0.05
    out["PB2"] = p2
    p3 = {}
    for k in ("mcleish|D1", "huginn|D1", "huginn|D2"):
        if not ok_n(k):
            continue
        p3["%s ans Llast@j<=16 final <= 0.10" % k] = _all(_le(_g(ck, k, "Llast@%d" % j, "ans_O1"), 0.10) for j in (1, 2, 8, 16))
        cls = ck["answer_classes"].get(k, {})
        tested = [j for j in (2, 8, 16) if (cls.get("Llast@%d|a=1" % j) or {}).get("n_gated_Dj_ge_2", 0) >= 10]
        p3["%s ans Llast@j (j >= 2, gated D_j >= 2) class in {not carried, state forgotten}" % k] = _all(
            cls["Llast@%d|a=1" % j]["class"] in ("not carried", "state forgotten") for j in tested) if tested else None
        p3["%s ans Lmid@32 in [0.3, 0.9]" % k] = _in(_g(ck, k, "Lmid@32", "ans_O1"), 0.3, 0.9)
        p3["%s ans Llast@32 (coda-gated) >= 0.6" % k] = _ge(_g(ck, k, "Llast@32", "ans_O1"), 0.6)
    if ok_n("mcleish|D2"):
        p3["mcleish|D2 ans Llast@j<=16 <= 0.15"] = _all(_le(_g(ck, "mcleish|D2", "Llast@%d" % j, "ans_O1"), 0.15)
                                                        for j in (1, 2, 8, 16))
    out["PB3"] = p3
    p4 = {}
    for k in ("mcleish|D1", "mcleish|D2", "huginn|D1", "huginn|D2", "huginn|D4"):
        if not ok_n(k) or k not in ck["e5_cells"]:
            continue
        e5 = ck["e5_cells"][k]
        p4["%s rel IN-single(j<=30) <= 0.10" % k] = _all(_le(_e5(ck, k, "IN-single(%d)" % j), 0.10) for j in IN1_J[32] if j <= 30)
        p4["%s rel IN-single(31) <= 0.25" % k] = _le(_e5(ck, k, "IN-single(31)"), 0.25)
        p4["%s rel IN-single(32) >= 0.3" % k] = _ge(_e5(ck, k, "IN-single(32)"), 0.3)
        p4["%s rel IN-persist(j<=16) >= 0.7" % k] = _all(_ge(_e5(ck, k, "IN-persist(%d)" % j), 0.7) for j in (1, 2, 4, 8, 16))
        a, b = _e5(ck, k, "IN-persist(1)"), _e5(ck, k, "IN-persist(16)")
        p4["%s rel |IN-persist(1) - IN-persist(16)| <= 0.15" % k] = None if a is None or b is None else abs(a - b) <= 0.15
        c, s1 = _e5(ck, k, "IN-persist(31)"), _e5(ck, k, "IN-single(32)")
        p4["%s rel IN-single(32) <= IN-persist(31) <= IN-persist(1)" % k] = None if None in (a, c, s1) else s1 <= c <= a
        # critic 2 F3: absolute matching (one call carrying the summed delta of IN-persist(j))
        p4["%s abs IN-single(16)x17 <= 0.15" % k] = _le(_e5(ck, k, "IN-single(16)x17"), 0.15)
        d = e5.get("abs-matched IN-persist(16) - IN-single(16)x17")
        p4["%s abs IN-persist(16) - IN-single(16)x17 >= 0.5" % k] = None if not d else d[0] >= 0.5
        p4["%s abs IN-single(31)x2 <= IN-persist(31)" % k] = None if None in (_e5(ck, k, "IN-single(31)x2"), c) else \
            _e5(ck, k, "IN-single(31)x2") <= c
        rl = ck["e5_rules"].get(k, {})
        p4["%s rule: IN-single(j) nulls for j <= 16 read 'erased by refresh'" % k] = _all(
            (rl.get("IN-single(%d)" % j) or {}).get("reading") == "erased by refresh (IN-persist(j) >= 0.5)"
            for j in (1, 2, 4, 8, 16)) if rl else None
        if k.endswith("D1"):
            p4["%s IN-persist-down(1) >= 0.5" % k] = _ge(_e5(ck, k, "IN-persist-down(1)"), 0.5)
    out["PB4"] = p4
    p5 = {}
    for k in ("mcleish|D1", "huginn|D1", "huginn|D2"):
        d = ck["e5_cells"].get(k, {}).get("post_inj(K)_minus_Llast@(K-1)_src")
        if ok_n(k) and d:
            p5["%s post_inj(K) - Llast@(K-1) >= 0.3" % k] = d[0] >= 0.3
    out["PB5"] = p5
    p6 = {}
    for k in ("mcleish|D1", "huginn|D1", "huginn|D2"):
        sc = ck["sc_cells"].get(k)
        if not sc or not ok_n(k):
            continue
        v = {n: ((c or {}).get("e") or [None])[0] for n, c in sc.items()}
        if k.startswith("mcleish"):
            p6["%s SC(Llast@1), SC(Llast@16) <= 0.10" % k] = _all([_le(v.get("Llast@1"), 0.10), _le(v.get("Llast@16"), 0.10)])
            p6["%s SC(Llast@2), SC(Llast@8) <= 0.10" % k] = _all([_le(v.get("Llast@2"), 0.10), _le(v.get("Llast@8"), 0.10)])
        else:
            p6["%s SC(Llast@1) <= 0.15" % k] = _le(v.get("Llast@1"), 0.15)
        p6["%s SC(Llast@31) in [0.1, 0.7]" % k] = _in(v.get("Llast@31"), 0.1, 0.7)
    out["PB6"] = p6
    p7 = {}
    for mo, dk, dc in (("mcleish", 1, 0.02), ("huginn", 3, 0.05)):
        ss = ck["seed_sensitivity"].get("%s|D1" % mo) or ck["seed_sensitivity"].get("%s|D2" % mo)
        if ss:
            p7["%s kept within %d" % (mo, dk)] = all(abs(v["kept"] - v["kept_off0"]) <= dk for v in ss.values())
            p7["%s item-level sym_diff <= %d (same scanned items, review 3 F9)" % (mo, dk)] = all(
                v.get("sym_diff") is not None and v["sym_diff"] <= dk for v in ss.values())
            p7["%s cell within %.2f" % (mo, dc)] = _all(None if v["cell_e_off"] is None else
                                                        abs(v["cell_e_off"] - v["cell_e_off0"]) <= dc for v in ss.values())
    out["PB7"] = p7
    p8 = {}
    for ds, d in ck["k16_vs_k32"].items():
        for nm, v in d.items():
            if not v or not v.get("first_minus_second") or v["n"] < MIN_KEPT:
                continue
            if nm.startswith("IN-single"):
                p8["%s %s within 0.15" % (ds, nm)] = abs(v["first_minus_second"][0]) <= 0.15
            elif v.get("off_floor"):        # critic 2 F10: scored only where both cells are off the floor
                p8["%s %s within 0.10" % (ds, nm)] = abs(v["first_minus_second"][0]) <= 0.10
    out["PB8"] = p8
    pn = {}
    for ds in ("D1", "D2"):
        k = "llama_nr|%s" % ds
        if not ok_n(k):
            continue
        w = (ck["nr_window"].get(k) or {}).get("all") or {}
        if ds == "D1":
            pn["D1 NR src e >= 0.8 at layers 0..7"] = _all(_ge(_g(ck, k, "layer@%d" % l, "src_O1"), 0.8) for l in range(8))
            pn["D1 NR src 'works' (e - e_RND, 5.4) at layers 0..7"] = _all(
                ((ck["e1_cells"][k].get("layer@%d" % l) or {}).get("src_vs_RND") or {}).get("label") == "works"
                for l in range(8))
            pn["D1 L* in [9, 13]"] = _in(w.get("L_star"), 9, 13)
            pn["D1 NR src layer 14 <= 0.15"] = _le(_g(ck, k, "layer@14", "src_O1"), 0.15)
            pn["D1 NR ans layers <= 7 <= 0.15"] = _all(_le(_g(ck, k, "layer@%d" % l, "ans_O1"), 0.15) for l in range(8))
            pn["D1 NR ans layers 12..14 >= 0.5"] = _all(_ge(_g(ck, k, "layer@%d" % l, "ans_O1"), 0.5) for l in range(12, 15))
            pn["D1 NR ans layer 14 >= 0.8"] = _ge(_g(ck, k, "layer@14", "ans_O1"), 0.8)
        else:
            pn["D2 L* in [5, 11]"] = _in(w.get("L_star"), 5, 11)
    out["PN1_PN2"] = pn
    p3n = {}
    for ds, d in ck["pn3_paired"].items():
        a = d.get("a_prelude_end: NR src L3 vs McLeish IN-persist(1)")
        if a and a["first"] and a["second"] and a["first_minus_second"] and a["n"] >= MIN_KEPT:
            p3n["%s (a) both >= 0.7, |diff| <= 0.15" % ds] = (a["first"][0] >= 0.7 and a["second"][0] >= 0.7
                                                               and abs(a["first_minus_second"][0]) <= 0.15)
        if ds in ("D1", "D2"):
            b = d.get("b_mid: NR src L(b) vs McLeish src Llast@16")
            if b and b["first_minus_second"] and b["n"] >= 20:
                p3n["%s (b) NR src L%s - McLeish src Llast@16 >= 0.5 (non-dev)" % (ds, d["b_layer"])] = \
                    b["first_minus_second"][0] >= 0.5
            c = d.get("c_early: NR src L(c) vs McLeish src Llast@1")
            if c and c["first_minus_second"] and c["n"] >= 20:
                p3n["%s (c) NR src L%s - McLeish src Llast@1 >= 0.5 (non-dev)" % (ds, d["c_layer"])] = \
                    c["first_minus_second"][0] >= 0.5
        dd = d.get("d_coda_input: NR ans L11 vs McLeish ans Llast@32")
        if dd and dd["first"] and dd["second"] and dd["n"] >= MIN_KEPT:
            p3n["%s (d) both >= 0.5" % ds] = dd["first"][0] >= 0.5 and dd["second"][0] >= 0.5
    out["PN3"] = p3n
    pdd = {}
    for k in ("mcleish|D1", "mcleish|D2"):
        dec = ck["decomposition"].get(k, {})
        if not ok_n(k) or not dec:
            continue
        d16, d31 = dec.get("j=16", {}), dec.get("j=31", {})
        so16 = (d16.get("state_own=SRCIN-INP(j+1)") or [None])[0]
        need = [d16.get("SC"), (d16.get("src_O1") or [None])[0], (d16.get("IN-persist(j+1)") or [None])[0], so16]
        pdd["%s headline j=16: SC <= 0.10, src O1 <= 0.10, IN-persist(17) >= 0.7, |state_own| <= 0.10" % k] = \
            None if None in need else (need[0] <= 0.10 and need[1] <= 0.10 and need[2] >= 0.7 and abs(need[3]) <= 0.10)
        pdd["%s j=31 state_own in [0.05, 0.6]" % k] = _in((d31.get("state_own=SRCIN-INP(j+1)") or [None])[0], 0.05, 0.6)
    out["PD_decomposition"] = pdd
    po = {}
    for ds in ("D1", "D2", "D4"):
        k = "ouro_think|%s" % ds
        if not ok_n(k):
            continue
        if ds in ("D1", "D4"):
            po["%s src Llast@1 >= 0.8" % ds] = _ge(_g(ck, k, "Llast@1", "src_O1"), 0.8)
            po["%s src Llast@1 'works' (5.4)" % ds] = ((ck["e1_cells"][k].get("Llast@1") or {}).get("src_vs_RND")
                                                       or {}).get("label") == "works"
        if ds == "D1":
            v = _g(ck, k, "Llast@1", "ans_O1")
            c = (ck["answer_classes"].get(k, {}).get("Llast@1|a=1") or {}).get("class")
            po["D1 ans Llast@1 <= 0.10 and not carried (gated)"] = None if v is None else (v <= 0.10 and c == "not carried")
        if ds == "D2":
            po["D2 ans Llast@1 >= 0.5"] = _ge(_g(ck, k, "Llast@1", "ans_O1"), 0.5)
    out["PO1"] = po
    pe = {}
    for k, e8 in ck["e8"].items():
        s1, s25 = e8.get("S|a=1"), e8.get("S|a=0.25")
        if not s1:
            continue
        fs = s1.get("full_set_search") or {}
        hml = fs.get("hit_minus_latest")
        lt = fs.get("loops_lt_K") or {}
        pdt = s1["test"].get("pred_minus_dev_tuned")
        h1, h25 = s1["verified9"]["hit_tol"], (s25 or {}).get("verified9", {}).get("hit_tol")
        row = {"interpreted (FD PASS and >= 20 eligible search items)": s1.get("interpreted"),
               "A2-1 hit - latest-site hit >= -0.10 (full set)": None if not hml or not fs.get("interpreted") else hml[0] >= -0.10,
               "A2-1 hit - Llast@K-1 hit >= -0.10 (loops < K, >= 10 eligible)": None if not lt.get("scored") or not lt.get(
                   "hit_minus_latest") else lt["hit_minus_latest"][0] >= -0.10,
               "mean regret <= 0.15": None if not fs.get("regret") or not fs.get("interpreted") else fs["regret"][0] <= 0.15,
               "fn_rate <= 0.10": _le(((s1.get("fn_rate_search") or {}).get("fn_rate") or [None])[0], 0.10),
               "|selector - dev-tuned| <= 0.10 D (test)": None if not pdt else abs(pdt[0]) <= 0.10,
               "hit9(0.25) >= hit9(1) - 0.15": None if h1 is None or h25 is None else h25 >= h1 - 0.15,
               "claim: adds beyond readout proximity (CI lower > 0; not a prediction)": fs.get("adds_beyond_latest")}
        if k.startswith(("mcleish", "huginn")):
            row["top site in last loop on >= 80% (descriptive)"] = None if not s1["n_items"] else \
                s1["top_in_last_loop"] / s1["n_items"] >= 0.8
        pg = ((s1.get("pig") or {}).get("full_set_search") or {})
        pgd = pg.get("pig_minus_G_hit")
        # PG4 (review 3 F1, written before any PIG row ran; belief): the path-integrated gain is at least as good
        # as G at a = 1 on the full set; scored only with >= MIN_E8 eligible items and an interpreted cell
        row["PG4 PIG top-1 hit - G top-1 hit >= 0 (full set, a = 1)"] = None if not pgd or not pg.get("interpreted") \
            or not s1.get("interpreted") else pgd[0] >= 0
        pe[k] = row
    out["PE8"] = pe
    pf = {}
    for mo, v in ck["fd_check"].items():
        pf["%s FD verdict PASS" % mo] = v["verdict"] == "PASS"
        ra = v.get("rank_agreement_bf16_vs_fp32") or {}
        # review 3 F6: rank agreement over above-floor sites and top-1 agreement (the all-S value is a diagnostic)
        pf["%s Spearman(G bf16, G fp32) above floor, median >= 0.9" % mo] = _ge(ra.get("median_spearman_above_floor"), 0.9)
        pf["%s top-1 site agrees bf16 vs fp32 on >= 90%% of items" % mo] = _ge(ra.get("top1_agree_frac"), 0.9)
    out["PF_fd"] = pf
    pt = {}
    for mo, sites in ck["a025_noise"].items():      # review 3 F7: replaces the tiny-edit floor (which cannot fail)
        if sites:
            pt["%s a = 0.25 bf16 noise < %.5f at every class site" % (mo, A025_NOISE_CUT)] = all(
                v["gate_pass"] for v in sites.values())
    out["PT_a025_noise"] = pt
    rnd_ok = [c.get("valid_flip_le_0.02") for e1 in ck["e1_cells"].values() for cc in e1.values()
              if isinstance(cc, dict) for f, c in cc.items() if f in ("src_RND", "ans_RND") and isinstance(c, dict)]
    out["PC"] = {"RND valid (flip <= 0.02) at every cell; its e is the reported floor": _all(rnd_ok) if rnd_ok else None,
                 "PC-final, SZ, SC harness, PC-site exact [harness]": all(
                     v["all_exact"] for d_ in (ck["pc_final"], ck["sz"], ck["sc_harness"], ck["pc_site"]) for v in d_.values())}
    it = ck["items"]

    def kept(k):
        return it.get(k, {}).get("joint_kept")
    out["PY"] = {"mcleish/NR D1, D2, D4, VB >= 40": _all(_ge(kept("mcleish|%s" % d), 40) for d in ("D1", "D2", "D4", "VB")),
                 "Huginn D1 >= 40 (capitals + city_mid)": _ge(kept("huginn|D1"), 40),
                 "CF < 40 on every model": _all(None if kept(k) is None else kept(k) < 40 for k in it if k.endswith("|CF")),
                 "Huginn D5 < 40": None if kept("huginn|D5") is None else kept("huginn|D5") < 40,
                 "values": {k: kept(k) for k in it}}
    out["harness_tags"] = ["PC: PC-final, SZ, SC at post_inj(K) and Llast@K, PC-site"]
    return out


def write_summary_md(out_dir, ck):
    L = ["# K17-PUBLIC summary (generated by k17_public.py --summarize)", "", "Plan sha256: %s" % ck.get("plan_sha256"), "",
         "## Items"]
    for k, v in sorted(ck["items"].items()):
        L.append("- %s: scanned %s, joint kept %s, scored %s, units %s, solo %s, pos_len x rel %s, split %s" % (
            k, v.get("scanned"), v.get("joint_kept"), v.get("scored"), v.get("n_units"), json.dumps(v.get("solo")),
            json.dumps(v.get("pos_len_x_rel")), json.dumps(v.get("split"))))
    L += ["", "## Harness"]
    for nm in ("pc_final", "sz", "sc_harness", "pc_site"):
        for k, v in sorted(ck[nm].items()):
            L.append("- %s %s: %s" % (nm, k, json.dumps(v)))
    for k, v in sorted(ck["hook_identity"].items()):
        L.append("- hook_identity %s: all_ok %s" % (k, (v or {}).get("all_ok")))
    L += ["", "## Timing (s)"] + ["- %s: %s" % (k, json.dumps(v)) for k, v in sorted(ck["timing"].items())]

    def f(c):
        return "-" if not c else "%s [%s, %s] (%d/%d)" % (c["e"][0], c["e"][1], c["e"][2], c["n_items"], c["n_clusters"])
    L += ["", "## E1 (e mean [95% CI] (n items / units))"]
    for k, e1 in sorted(ck["e1_cells"].items()):
        L += ["### %s" % k, "| site | src O1 | src RND | src label (5.4) | ans O1 | ans a=0.25 | ans RND | ans label | "
              "cell | SC(Llast@j) | src-null attribution |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, cc in e1.items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                name, f(cc.get("src_O1")), f(cc.get("src_RND")), (cc.get("src_vs_RND") or {}).get("label"),
                f(cc.get("ans_O1")), f(cc.get("ans_O1_a0.25")), f(cc.get("ans_RND")),
                (cc.get("ans_vs_RND") or {}).get("label"), cc.get("label") or "", cc.get("SC_same_loop"),
                cc.get("src_attribution") or ""))
    L += ["", "## E5, SC, decomposition"]
    for k, e5 in sorted(ck["e5_cells"].items()):
        L.append("- %s: %s" % (k, json.dumps({n: (c["e"] if isinstance(c, dict) and "e" in c else c)
                                              for n, c in e5.items() if c})))
    for k, v in sorted(ck["sc_cells"].items()):
        L.append("- SC %s: %s" % (k, json.dumps({n: (c or {}).get("e") for n, c in v.items()})))
    for k, v in sorted(ck["decomposition"].items()):
        L.append("- decomposition %s: %s" % (k, json.dumps(v)))
    for k, v in sorted(ck["e5_rules"].items()):
        L.append("- E5 rules %s: %s" % (k, json.dumps(v)[:3000]))
    L += ["", "## Answer classes (gated on D_j >= 2)"]
    for k, v in sorted(ck["answer_classes"].items()):
        L.append("- %s: %s" % (k, json.dumps({n: [c["class"], c["n_gated_Dj_ge_2"], c["mean_Dj_over_D"]]
                                              for n, c in v.items()})))
    L += ["", "## Tiny-edit floor (a = %g)" % TINY_A]
    for k, v in sorted(ck["tiny_floor"].items()):
        L.append("- %s: %s" % (k, json.dumps(v)))
    L += ["", "## FD check (fp32)"] + ["- %s: %s" % (k, json.dumps(v)[:2500]) for k, v in sorted(ck["fd_check"].items())]
    L += ["", "## NR window and PN3"] + ["- window %s: %s" % (k, json.dumps(v)) for k, v in sorted(ck["nr_window"].items())]
    L += ["- PN3 %s: %s" % (k, json.dumps(v)[:3000]) for k, v in sorted(ck["pn3_paired"].items())]
    L += ["", "O2 rows: %s" % ck.get("o2_label")]
    L += ["", "## E8"]
    for k, e8 in sorted(ck["e8"].items()):
        for sk, o in e8.items():
            L.append("- %s %s: %s" % (k, sk, json.dumps(o)[:1500]))
    L += ["", "## Predictions vs outcomes", "```", json.dumps(ck["predictions_vs_outcomes"], indent=1)[:20000], "```"]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ self-tests
def selftest():
    res = {}
    b1 = bootc([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"])
    try:
        import k17_parcae as PC
        res["boot_oneway_matches_parcae"] = b1 == PC.boot([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "c"], n=NBOOT)
        G9 = {"k%d" % i: float(i) for i in range(9)}
        dm9 = dict(G9, k7=8.0)
        res["helpers_match_parcae"] = (PC.hit_metrics_v3(G9, dm9, 10.0, 2.0) == hit_v3(G9, dm9, 10.0, 2.0)
                                       and PC.verify_sites(G9) == verify_sites(G9) and PC.seed_of("x", 2) == seed_of("x", 2)
                                       and PC.split_hex("p:", "u") == split_hex("p:", "u")
                                       and torch.equal(PC.rnd_delta("p", "c1.3", "src", [1, 2], [2.0, 3.0], 8),
                                                       rnd_delta("p", "c1.3", "src", [1, 2], [2.0, 3.0], 8)))
    except ImportError as ex:
        res["boot_oneway_matches_parcae"] = res["helpers_match_parcae"] = "not checked here: %s" % type(ex).__name__
    xs = [0.1, 0.5, 0.3, 0.9, 0.2, 0.4]
    cl2 = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "a"), ("e", "b"), ("d", "e")]
    try:
        import k17_varbench as VB
        res["boot_dyadic_matches_varbench"] = bootc(xs, cl2) == VB.boot2(xs, cl2, n=NBOOT)
    except (ImportError, AttributeError) as ex:      # remote hosts may hold an older k17_varbench
        res["boot_dyadic_matches_varbench"] = "not checked here: %s" % type(ex).__name__
    res["boot_dyadic_list_ok"] = bootc(xs, [list(c) for c in cl2]) == bootc(xs, cl2)
    t0 = tostc([0.01, -0.02, 0.0, 0.015, -0.01] * 8, list(range(40)))
    t1 = tostc([0.2, 0.1, 0.15, 0.12, 0.18] * 8, list(range(40)))
    res["tost"] = [t0["null_by_tost"], t1["null_by_tost"]]
    pb = pairedc([1.0, 2.0, 3.0], [0.5, 1.0, None], ["a", "b", "c"])
    res["paired"] = [pb[0], pb[3], pb[4]]
    # sites
    res["e1_counts"] = {"raven32": len(e1_names("raven", 32, "D1")), "raven16": len(e1_names("raven", 16, "D1")),
                        "nr": len(e1_names("nr", 1, "D1")), "ouro": len(e1_names("ouro", 4, "D1")),
                        "raven32_VB": len(e1_names("raven", 32, "VB"))}
    res["e1_counts_ok"] = res["e1_counts"] == {"raven32": 20, "raven16": 12, "nr": 16, "ouro": 12, "raven32_VB": 9}
    res["rnd_counts"] = [len(rnd_names("raven", 32, "D1")), len(a025_names("raven", 32, "D1")), len(rnd_names("raven", 32, "VB"))]
    res["rnd_counts_ok"] = res["rnd_counts"] == [20, 7, 9]
    gq = torch.Generator().manual_seed(3)
    hq, aq, bq = (torch.randn(5, 16, generator=gq) for _ in range(3))
    bq[2] = aq[2]
    pm = proj_many(list(hq), list(aq), list(bq))
    p1 = [proj(hq[i], aq[i], bq[i]) for i in range(5)]
    zq = torch.randn(300, generator=gq)
    res["proj_many_ok"] = (pm[2] is None and p1[2] is None and all(abs(u - v) < 1e-5 for u, v in zip(pm, p1) if u is not None)
                           and vals(zq, 3, 7) == [r5(zq[3]), r5(zq[7]), r5(torch.logsumexp(zq, -1)), int(zq.argmax())])
    # v3 (critic 2 F7): SC at every E1 site with j < K; F1: latest site of a set
    res["sc_counts"] = [len(sc_names("raven", 32, "D1")), len(sc_names("raven", 16, "D1")), len(sc_names("raven", 32, "VB")),
                        len(sc_names("nr", 1, "D1"))]
    res["sc_counts_ok"] = res["sc_counts"] == [5, 5, 3, 0]
    res["latest"] = [latest_site(e8_sets(32, 4)[0], 4, 32), latest_site(e8_sets(32, 6)[0], 6, 32),
                     latest_site(e8_sets(1, 16)[0], 16, 1), latest_site(e8_sets(32, 4)[1], 4, 32)]
    res["latest_ok"] = res["latest"] == [("c", 32, 2), ("c", 32, 4), ("c", 1, 14), ("c", 32, 3)]
    # v3 (critic 2 F1): hit - latest-site hit; toy: selector = latest on item 1 (diff 0), beats it on item 2 (diff 1)
    Gt = {"a": 1.0, "b": 2.0, "z": 3.0}
    h1 = hit_v4(Gt, {"a": 0.0, "b": 1.0, "z": 9.0}, 10.0, 2.0, "z")
    h2 = hit_v4({"a": 1.0, "b": 5.0, "z": 3.0}, {"a": 0.0, "b": 9.0, "z": 1.0}, 10.0, 2.0, "z")
    res["hit_v4"] = [h1["hit"] - h1["latest_hit"], h2["hit"] - h2["latest_hit"], h2["chance"]]
    res["hit_v4_ok"] = res["hit_v4"] == [0, 1, 1.0 / 3]
    # v3 (critic 2 F4): e_own = dm_own(j) / D_j with the guard D_j >= 2
    xo = {"clean_A": [0.0, 0.0, 0.0, 0], "clean_B": [0.0, 10.0, 0.0, 1], "loopsA": [[0.0, 0.0, 0, 0], [0.0, 0.0, 0, 0]],
          "loopsB": [[0.0, 1.0, 0, 0], [0.0, 4.0, 0, 0]],
          "rows": {"O1|ans|c1.3|1": {"f": [0, 0, 0, 0], "mj": 0.5}, "O1|ans|c2.3|1": {"f": [0, 0, 0, 0], "mj": 3.0}}}
    res["eown_guard"] = [eown_of(xo, "O1|ans|c1.3|1"), eown_of(xo, "O1|ans|c2.3|1"), Dj_of(xo, 2)]
    res["eown_guard_ok"] = res["eown_guard"] == [None, 0.75, 4.0]
    # v3 (critic 2 F5): works = e - e_RND >= 0.5 with CI lower >= 0.3 and flip >= 0.5; fails = TOST null
    wi = []
    for i in range(30):
        wi.append({"cluster": "c%d" % i, "tB": 1, "clean_A": [0.0, 0.0, 0.0, 0], "clean_B": [0.0, 10.0, 0.0, 1],
                   "rows": {"W": {"f": [0.0, 9.0 + (i % 3) * 0.1, 0.0, 1]}, "N": {"f": [0.0, 0.05 * (i % 3 - 1), 0.0, 0]},
                            "R": {"f": [0.0, 1.5, 0.0, 0]}}})
    res["works_rule"] = [label_vs_rnd(wi, "W", "R")["label"], label_vs_rnd(wi, "N", "R")["label"],
                         label_vs_rnd(wi, "R", "N")["label"]]
    res["works_rule_ok"] = res["works_rule"] == ["works", "fails", "partial"]
    # v3 (critic 2 F2): FD pass rule on synthetic cells (linear -> PASS; sign flipped -> FAIL)
    fi_ok, fi_bad = [], []
    for i in range(12):
        g = 2.0 + i
        lin = {"%g" % a: a * g * (1 + 0.01 * ((i % 3) - 1)) for a in FD_A}
        lin["1e-06"] = 1e-6 * g + 1e-9
        fi_ok.append({"pair_id": ("D1|%d" if i < 6 else "D2|%d") % i, "Z": 0.0, "cells": {"x": {"G": g, "dm": lin}}})
        fi_bad.append({"pair_id": ("D1|%d" if i < 6 else "D2|%d") % i, "Z": 0.0,
                       "cells": {"x": {"G": g, "dm": dict(lin, **{"0.01": -0.01 * g})}}})
    res["fd_rule"] = [fd_summary(fi_ok)["verdict"], fd_summary(fi_bad)["verdict"], fd_summary(fi_ok[:5])["verdict"]]
    res["fd_rule_ok"] = res["fd_rule"] == ["PASS", "FAIL", "insufficient"]
    S, Sp = e8_sets(32, 4)
    S6, Sp6 = e8_sets(32, 6)
    Sn, _ = e8_sets(1, 16)
    res["e8_sets"] = [len(S), len(Sp), len(S6), len(Sp6), len(Sn), ("c", 32, 3) in S, ("c", 32, 2) in S, ("c", 1, 15) in Sn]
    res["e8_sets_ok"] = res["e8_sets"] == [127, 128, 191, 192, 15, False, True, False]
    res["names"] = [name2site("Lmid@32", 4), name2site("Llast@31", 6), name2site("post_inj@8", 6), name2site("layer@14", 16),
                    name2site("Lmid@2", 24)]
    res["names_ok"] = res["names"] == [("c", 32, 2), ("c", 31, 5), ("p", 8), ("c", 1, 14), ("c", 2, 12)]
    res["last_loop_cell"] = [last_loop_cell("raven", 32, 4), last_loop_cell("raven", 32, 6), last_loop_cell("nr", 1, 16),
                             last_loop_cell("ouro", 4, 24)]
    res["last_loop_ok"] = res["last_loop_cell"] == [("c", 32, 2), ("c", 32, 3), ("c", 1, 14), ("c", 4, 12)]
    allS = [("p", j) for j in range(1, 33)] + [("c", j, l) for j in range(1, 33) for l in range(4)] + [("k", 0), ("k", 1)]
    srt = sorted(allS, key=lambda s: exec_order(s, 4, 32))
    res["exec_order_ok"] = srt[:6] == [("p", 1), ("c", 1, 0), ("c", 1, 1), ("c", 1, 2), ("c", 1, 3), ("p", 2)] and \
        srt[-2:] == [("k", 0), ("k", 1)] and len({exec_order(s, 4, 32) for s in allS}) == len(allS)
    # executed-index formula (raven facts F2): Huginn 2 + 4(j-1) + l, McLeish 4 + 6(j-1) + l
    class _R(object):
        P, C = 2, 4
    res["exec_idx"] = [RavenR.exec_idx(_R(), 1, 0), RavenR.exec_idx(_R(), 32, 3)]
    _R.P, _R.C = 4, 6
    res["exec_idx"] += [RavenR.exec_idx(_R(), 1, 0), RavenR.exec_idx(_R(), 32, 5)]
    res["exec_idx_ok"] = res["exec_idx"] == [2, 129, 4, 195]
    res["nr_map"] = {j: 3 + -(-8 * j // 32) for j in JSET[32]}
    res["nr_map_ok"] = res["nr_map"] == {1: 4, 2: 4, 8: 5, 16: 7, 31: 11, 32: 11}
    res["e5_sets_ok"] = all(j + 1 in INP_J[32] or j + 1 == 32 for j in SRCIN_J[32]) and 31 in IN1_J[32] and 15 in IN1_J[16]
    # splits and answers
    res["split_e8"] = [split_e8(["a"], "p:"), split_e8(["a", "a"], "p:")]
    us = ["u%d" % i for i in range(40)]
    sp = {u: split_hex("p:", u) for u in us}
    du, tu = [u for u in us if sp[u] == "dev"], [u for u in us if sp[u] == "test"]
    res["split_cross_ok"] = split_e8([du[0], tu[0]], "p:") == "cross" and split_e8([du[0], du[1]], "p:") == "dev"

    def enc(t, ans=False):
        return [ord(ch) for ch in t]
    res["answer_parts"] = [answer_parts(enc, "45", "47"), answer_parts(enc, "4", "45")]
    res["answer_parts_ok"] = res["answer_parts"][0] == ([32, 52], 53, 55, [], []) and res["answer_parts"][1] is None

    def enc2(t, ans=False):
        out = []
        for ch in t:
            if ch == " ":
                out.append(-1)
            elif out and out[-1] == -1:
                out[-1] = 1000 + ord(ch)
            else:
                out.append(ord(ch))
        return out
    res["answer_parts_merged"] = answer_parts(enc2, "45", "65")
    res["answer_parts_merged_ok"] = res["answer_parts_merged"] == ([], 1052, 1054, [53], [53])
    # selector
    c1 = [{"pair_id": "p%d" % i, "sel": {"subjects": ["capital|s", "capital|t%d" % i]}} for i in range(5)]
    s = Selector("D1", 80)
    kept = []
    for c in c1:
        if s.want(c):
            s.accept(c)
            kept.append(c["pair_id"])
    res["sel_d1_cap"] = kept
    s5 = Selector("D5", 80)
    c5 = [{"sel": {"problems": ["a", "b"]}}, {"sel": {"problems": ["b", "c"]}}, {"sel": {"problems": ["d", "e"]}}]
    k5 = []
    for i, c in enumerate(c5):
        if s5.want(c):
            s5.accept(c)
            k5.append(i)
    res["sel_d5"] = k5
    s4 = Selector("D4", 80)
    # review 3 F5: item 1 reuses item 0's A key on its B side (the v3 A-side rule kept it), item 2 its A key
    c4 = [{"cl": ["x", "y"], "sel": {"clusters": ["x"]}}, {"cl": ["z", "x"], "sel": {"clusters": ["z"]}},
          {"cl": ["x", "w"], "sel": {"clusters": ["x"]}}, {"cl": ["v", "u"], "sel": {"clusters": ["v"]}}]
    k4 = [i for i, c in enumerate(c4) if s4.want(c) and (s4.accept(c) or True)]
    res["sel_d4"] = k4
    res["sel_ok"] = res["sel_d1_cap"] == ["p0", "p1"] and res["sel_d5"] == [0, 2] and res["sel_d4"] == [0, 3]
    # prep_item
    cand = {"ds": "D2", "pair_id": "p", "A": [5, 6, 7], "B": [5, 8, 7], "r": 2, "src": [1], "tA": 1, "tB": 2,
            "restA": [9], "restB": [9], "rel": "op1", "cl": ["x=1", "x=2"], "split": "dev", "strata": {"pos_len": "1"},
            "label": "x", "para": [{"fmt": "f", "A": [9, 5, 6, 7], "B": [9, 5, 8, 7], "src": [2], "r": 3},
                                   {"fmt": "g", "A": [6, 7], "B": [8, 7], "src": [0], "r": 1}]}
    p0, pn = prep_item(cand, 128000), prep_item(cand, None)
    res["prep"] = [p0["A"], p0["src"], p0["r"], len(p0["para"]), p0["FA"], p0["Tmax"], pn["A"], len(pn["para"]),
                   prep_item(dict(cand, src=[0]), None), p0["cluster"]]
    res["prep_ok"] = (p0["A"] == [128000, 5, 6, 7] and p0["src"] == [2] and p0["r"] == 3 and len(p0["para"]) == 2
                      and p0["FA"] == [128000, 5, 6, 7, 1] and p0["Tmax"] == 5 and pn["A"] == [5, 6, 7]
                      and len(pn["para"]) == 1 and res["prep"][8] is None and p0["cluster"] == ("x=1", "x=2"))
    # e8 summary on a synthetic item set (hit, chance over the full set, dev-tuned)
    S4, _ = e8_sets(2, 2)
    keys = [skey(s) for s in S4]
    items = []
    for i in range(3):
        G = {k: float(n) for n, k in enumerate(keys)}
        rows = {"O1|ans|%s|1" % k: {"f": [0.0, 3.0 * n, 0.0, 0]} for n, k in enumerate(keys)}
        rows.update({"O1|ans|%s|0.25" % k: {"f": [0.0, 3.0 * n / 4, 0.0, 0]} for n, k in enumerate(keys)})
        items.append({"pair_id": "q%d" % i, "cluster": "c%d" % i, "split": "dev" if i < 2 else "test", "G": G,
                      "ver_S": keys, "ver_Sp": keys, "rnd_site": keys[0], "full_search": i < 2,
                      "clean_A": [0.0, 0.0, 0.0, 0], "clean_B": [0.0, 10.0, 0.0, 1], "rows": rows, "t_fwd": 1.0, "t_grad": 3.0})
    o = summarize_e8(items, 2, 2, "nr", 1.0, "S")
    fs = o["full_set_search"]
    res["e8_full"] = [fs["eligible"], fs["top1_hit"][0], fs["diagnostic_chance_band"], fs["hit_minus_latest"][0],
                      o["latest_site"]]
    res["e8_full_ok"] = (res["e8_full"][:2] == [2, 1.0] and abs(res["e8_full"][2] - 1.0 / 3) < 1e-3
                         and res["e8_full"][3] == 0.0 and res["e8_full"][4] == keys[-1])
    bt, tab = dev_tuned_site(items, keys)
    res["dev_tuned_ok"] = bt == keys[-1]
    # review 3 F1: PIG summary on the synthetic items (PIG ranks the last site first, G the first site)
    for x in items:
        x["PIG"] = {k: -v for k, v in x["G"].items()}
        x["pig_a1_equals_B"], x["pig_ids_differ_only_at_src"] = True, True
    op = summarize_e8(items, 2, 2, "nr", 1.0, "S")["pig"]
    res["pig_summary"] = [op["full_set_search"]["eligible"], op["full_set_search"]["pig_top1_hit"][0],
                          op["full_set_search"]["G_top1_hit"][0], op["full_set_search"]["pig_minus_G_hit"][0]]
    res["pig_summary_ok"] = res["pig_summary"] == [2, 0.0, 1.0, -1.0]
    # review 3 F3: arch_of covers the K = 8 runner
    res["arch_huginn8"] = list(arch_of("huginn8"))
    res["arch_ok"] = res["arch_huginn8"] == ["raven", 8, 4]
    # review 3 F2, F12: command-line overrides
    FS_CAP_OVR.clear()
    FS_CAP_OVR.update(parse_kv("huginn=15,mcleish.D4=0"))
    res["full_cap"] = [full_cap("huginn", "D1"), full_cap("mcleish", "D4"), full_cap("mcleish", "D2"),
                       full_cap("llama_nr", "D1")]
    FS_CAP_OVR.clear()
    PCS_OVR.update(parse_kv("50"))
    res["n_sub"] = n_sub("mcleish")
    PCS_OVR.clear()
    res["overrides_ok"] = res["full_cap"] == [15, 0, 60, 60] and res["n_sub"] == 50 and n_sub("mcleish") == 20
    # review 3 F4: search plan from the global kept list and the pilot gate
    class _RR(object):
        name, family, K, C = "mcleish", "raven", 2, 2
    kept_t = ["k%d" % i for i in range(30)]
    cmap_t = {p: {"split": ("dev", "test", "cross")[i % 3]} for i, p in enumerate(kept_t)}
    pl = search_plan(_RR(), "D2", kept_t, cmap_t)
    res["search_plan"] = [pl["cap"], len(pl["pilot"]), len(pl["rest"]), pl["pilot"][:3]]
    keys2 = [skey(s_) for s_ in e8_sets(2, 2)[0]]

    def _fx(pid, rng):
        return {"pair_id": pid, "G": {k: 1.0 for k in keys2}, "clean_A": [0.0, 0.0, 0.0, 0],
                "clean_B": [0.0, 10.0, 0.0, 1],
                "rows": {"O1|ans|%s|1" % k: {"f": [0.0, rng * n / 2.0, 0.0, 0]} for n, k in enumerate(keys2)}}
    g_none = e8_gate(_RR(), "D2", [], pl)
    g_lo = e8_gate(_RR(), "D2", [_fx(p_, 0.5) for p_ in pl["pilot"]], pl)          # range 1.0 < 2: none eligible
    g_hi = e8_gate(_RR(), "D2", [_fx(p_, 9.0) for p_ in pl["pilot"]], pl)          # range 9 > 2: all eligible
    res["e8_gate"] = [g_none["decided"], g_lo["continue"], g_lo["eligible"], g_hi["continue"], g_hi["projected_eligible"]]
    res["e8_gate_ok"] = (res["search_plan"] == [20, 10, 10, ["k0", "k2", "k3"]]
                         and res["e8_gate"] == [False, False, 0, True, 20.0])
    # review 3 F10: a scan that starts after the deadline reports stopped_budget
    st_ = {}
    kk, lg = scan([], "D2", [{"pair_id": "q", "sel": {"clusters": ["a"]}}], 5, deadline=time.time() - 1, stats=st_)
    res["scan_stop"] = [kk, lg, st_.get("stopped_budget")]
    res["scan_stop_ok"] = res["scan_stop"] == [[], [], True]
    # review 3 F8: VB IN-persist includes K-1
    res["vb_inp_ok"] = vb_inp(32) == (1, 16, 31) and 31 in IN1_J[32]
    # review 3 F6: rank agreement above the floor ignores the sub-floor ranks
    ra_rows = [{"pair_id": "f%d" % i, "G": dict({"a": 1.0, "b": 0.5, "c": 0.25, "d": 0.1},
                                                 **{"z%d" % j: 1e-9 * j for j in range(6)}),
                "G_bf16": dict({"a": 1.01, "b": 0.49, "c": 0.26, "d": 0.11},
                               **{"z%d" % j: -1e-9 * j for j in range(6)})} for i in range(3)]
    ra = fd_rank_agreement(ra_rows, 1e-5)
    res["rank_agree"] = [ra["median_spearman_above_floor"], ra["top1_agree_frac"], ra["top3_same_frac"],
                         ra["diagnostic_median_spearman_all_S"] < 0.9]
    res["rank_agree_ok"] = res["rank_agree"] == [1.0, 1.0, 1.0, True]
    # review 3 F7: a = 0.25 noise from fp32 vs bf16 rows
    nrows = [{"a025": {"fp32": {"D": 10.0, "dm": {"Llast@1": 0.1, "Llast@2": 1.0}},
                       "bf16": {"D": 10.0, "dm": {"Llast@1": 0.12, "Llast@2": 2.0}}}} for _ in range(3)]
    nz = a025_noise(nrows)
    res["a025_noise"] = [nz["Llast@1"]["mean_abs_de"], nz["Llast@1"]["gate_pass"], nz["Llast@2"]["gate_pass"]]
    res["a025_noise_ok"] = res["a025_noise"] == [0.002, True, False]
    ok = all(res[k] for k in ("boot_oneway_matches_parcae", "helpers_match_parcae", "boot_dyadic_matches_varbench", "boot_dyadic_list_ok",
                              "e1_counts_ok", "rnd_counts_ok", "e8_sets_ok", "names_ok", "last_loop_ok", "exec_order_ok",
                              "exec_idx_ok", "nr_map_ok", "e5_sets_ok", "split_cross_ok", "answer_parts_ok",
                              "answer_parts_merged_ok", "sel_ok", "prep_ok", "e8_full_ok", "dev_tuned_ok",
                              "sc_counts_ok", "latest_ok", "proj_many_ok", "hit_v4_ok", "eown_guard_ok", "works_rule_ok", "fd_rule_ok",
                              "pig_summary_ok", "arch_ok", "overrides_ok", "e8_gate_ok", "scan_stop_ok", "vb_inp_ok",
                              "rank_agree_ok", "a025_noise_ok"))
    ok = ok and res["tost"] == [True, False] and res["paired"] == [0.75, 2, 2] and res["split_e8"][0] == res["split_e8"][1]
    res["ok"] = bool(ok)
    print(json.dumps(res, default=str))
    return ok


def tiny_raven(snapdir, seed=0, dtype=torch.bfloat16, **over):
    """Random-weight raven model from a cached snapshot's own modeling code, small dims (CPU hook tests)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(snapdir, trust_remote_code=True, local_files_only=True)
    for k, v in dict(n_embd=64, n_heads=4, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
                     head_dim=16, **over).items():
        setattr(cfg, k, v)
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(1.0 + 0.1 * torch.randn_like(p) if p.dim() == 1 else 0.05 * torch.randn_like(p))
    m = m.to(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tiny_llama(seed=0, dtype=torch.bfloat16):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=16, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg).to(dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _tiny_item(R, V, seed=1):
    A = [1, 5, 6, 7, 8, 9, 10, 11, 12]
    B = [1, 5, 13, 14, 8, 9, 10, 11, 12]
    R.set_init(16, seed)
    za, zb = R.run(A, 8)["z_r"], R.run(B, 8)["z_r"]
    tA = int(za.argmax())
    tB = int(zb.argmax()) if int(zb.argmax()) != tA else int(zb.topk(2).indices[1])
    it = {"ds": "D1", "pair_id": "tiny|1", "A": A, "B": B, "r": 8, "src": [2, 3], "tA": tA, "tB": tB, "rel": "capital",
          "cl": ["u1", "u2"], "cluster": ("u1", "u2"), "split": "dev", "strata": {"pos_len": ">=2", "rel": "capital"},
          "label": "tiny", "restA": [], "restB": [],
          "para": [{"fmt": "p0", "A": [1, 20] + A[1:], "B": [1, 20] + B[1:], "src": [3, 4], "r": 9}], "Tmax": 12}
    return it


def _driver_test(R):
    """Review 3 F2/F4/F10: the cluster path (stage scan with the E8 pilot, stage rows with shards 0/2 and 1/2, stage
    merge) gives the same kept list, full-search set, gate and rows as the sequential path in another directory; a
    scan that starts past the deadline reports stopped_budget and the rows stage refuses an incomplete scan. The clean
    margin guard is lifted (GAP) so random tiny models keep items; PILOT_N = 1, MIN_E8 = 1, full-search cap 3."""
    import tempfile
    g = globals()
    old = (g["GAP"], g["PILOT_N"], g["MIN_E8"], dict(FS_CAP_OVR))
    cands = []
    for i in range(7):
        A = [1, 5, 20 + i, 30 + i, 8, 9, 10, 11, 12]
        B = [1, 5, 40 + i, 50 + i, 8, 9, 10, 11, 12]
        pid = "D2|t%d" % i
        R.set_init(len(A), seed_of(pid))
        za, zb = R.run(A, 8)["z_r"], R.run(B, 8)["z_r"]
        tA, tB = int(za.argmax()), int(zb.argmax())
        if tB == tA:
            tB = int(zb.topk(2).indices[1])
        ua, ub = "x=%d" % i, "x=%d" % (i + 10)
        cands.append({"ds": "D2", "pair_id": pid, "A": A, "B": B, "r": 8, "src": [2, 3], "tA": tA, "tB": tB,
                      "restA": [], "restB": [], "rel": "op1", "cl": [ua, ub], "split": ("dev", "cross", "test")[i % 3],
                      "strata": {"pos_len": ">=2", "rel": "op1"}, "para": [], "sel": {"clusters": [ua, ub]},
                      "label": pid})
    out = {}

    def lg(_s):
        return None

    def rowsig(x):
        return (x["pair_id"], bool(x.get("full_search")), sorted((k, tuple(v["f"])) for k, v in x["rows"].items()))
    try:
        g["GAP"], g["PILOT_N"], g["MIN_E8"] = -1e9, 1, 1
        FS_CAP_OVR.clear()
        FS_CAP_OVR["*"] = 3
        far = time.time() + 3600
        d1, d2, d3 = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
        run_dataset([R], "D2", cands, d1, 7, far, lg)
        run_dataset([R], "D2", cands, d2, 7, far, lg, stage="scan")
        for k in range(2):
            run_dataset([R], "D2", cands, d2, 7, far, lg, stage="rows", shard=(k, 2))
        run_dataset([R], "D2", cands, d2, 7, far, lg, stage="merge")
        r1 = json.load(open(res_path(d1, R.name, "D2")))
        r2 = json.load(open(res_path(d2, R.name, "D2")))
        out["n_items"] = [len(r1["items"]), len(r2["items"])]
        out["kept"] = len(kept_of(r1))
        out["same_items_rows"] = [rowsig(x) for x in r1["items"]] == [rowsig(x) for x in r2["items"]]
        out["full_search_n"] = sum(1 for x in r1["items"] if x.get("full_search"))
        out["gate"] = r1["meta"].get("e8_gate")
        out["gate_equal"] = r1["meta"].get("e8_gate") == r2["meta"].get("e8_gate")
        out["merged_shards"] = r2["meta"].get("merged_shards")
        out["dev_tuned_equal"] = ((r1.get("dev_tuned") or {}).get("S") or {}).get("site") ==             ((r2.get("dev_tuned") or {}).get("S") or {}).get("site")
        run_dataset([R], "D2", cands, d3, 7, time.time() - 1, lg)
        r3 = json.load(open(res_path(d3, R.name, "D2")))
        out["deadline_flag"] = [r3["meta"].get("stopped_budget"), r3["meta"].get("scan_complete"), len(r3["items"])]
        try:
            run_dataset([R], "D2", cands, d3, 7, far, lg, stage="rows", shard=(0, 2))
            out["rows_refused_incomplete_scan"] = False
        except HarnessFail:
            out["rows_refused_incomplete_scan"] = True
    finally:
        g["GAP"], g["PILOT_N"], g["MIN_E8"] = old[0], old[1], old[2]
        FS_CAP_OVR.clear()
        FS_CAP_OVR.update(old[3])
    out["ok"] = bool(out["n_items"][0] == out["n_items"][1] == out["kept"] and out["kept"] >= 3
                     and out["same_items_rows"] and out["gate_equal"] and out["dev_tuned_equal"]
                     and out["deadline_flag"] == [True, False, 0] and out["rows_refused_incomplete_scan"]
                     and len(out["merged_shards"] or []) == 2)
    return out


def _pig_checks(X, sites):
    """Review 3 F1: the PIG path at a = 0 is the G path bit for bit; at a = 1 its logits are clean B's; PIG's top-3
    sites carry verify rows at both doses."""
    g0, z0 = grad_gain(X, sites)
    gz, zz = grad_gain(X, sites, emb_a=0.0)
    return {"a0_equals_G": bool(all(torch.equal(g0[s_], gz[s_]) for s_ in sites) and torch.equal(z0, zz)),
            "a1_equals_B": X.out.get("pig_a1_equals_B"), "n_pig": len(X.out.get("PIG") or {}),
            "n_Ga": len(X.out.get("Ga") or {}), "ids_differ_only_at_src": X.out.get("pig_ids_differ_only_at_src"),
            "pig_top_verified": all(("O1|ans|%s|%g" % (k, a)) in X.rows for k in X.out["pig_top"] for a in A_VERIFY)}


def selftest_model():
    """CPU: tiny random raven model (McLeish modeling code; Huginn too when its snapshot is cached) and a tiny Llama:
    hook checks a-k, controls, every arm, the E8 grad path, the dev-tuned pass and summarize."""
    res = {}
    snaps = [k for k in ("mcleish", "huginn") if os.path.exists(os.path.join(snap(k), "raven_modeling_minimal.py"))]
    if os.environ.get("MCLEISH_SNAP"):
        snaps = ["mcleish_env"]
    ok = True
    import tempfile
    tmp = tempfile.mkdtemp()
    for key in snaps:
        sd = os.environ["MCLEISH_SNAP"] if key == "mcleish_env" else snap(key)
        rk = "mcleish" if key.startswith("mcleish") else "huginn"
        m = tiny_raven(sd)
        R = RavenR(rk, m, "cpu", 32)
        tn = "tiny_raven" if rk == "mcleish" else "tiny_huginn"
        R.bos, R.fam, R.name = None, "x", tn
        it = _tiny_item(R, 300)
        hk = check_hooks(R, it)
        X = Item(R, it)
        X.depth_margins()
        X.deltas()
        run_controls(X, subset=True)
        run_e1(X, tiny=True)
        run_e5(X, subset=True)
        run_pb8(X)
        run_para(X, force=True)
        run_e8(X, full_search=True)
        allp = X.allpos()
        v3keys = ["IN1S|1", "IN1S|16", "IN1S|31", "RNDIN|1", "SCID|p16", "SCID|c16.%d" % (R.C // 2), "SCID|c1.0",
                  "O1|src|c30.%d|1" % (R.C - 1), "O1|src|c24.%d|1" % (R.C - 1), "RND|src|c16.%d" % (R.C // 2),
                  "O1|ans|c16.%d|0.001" % (R.C - 1)]
        v3_rows = {k: k in X.rows for k in v3keys}
        # IN1S(j) with n = 1 is IN-single(j) bit for bit; IN1S(K) = n of 1 as well (K - K + 1)
        o_n1 = R.run(it["A"], 8, e_edits=X.e_ed([20], it["src"], n=1))
        o_s1 = R.run(it["A"], 8, e_edits=X.e_ed([20], it["src"]))
        in1s_n1_exact = bool(torch.equal(o_n1["z_r"], o_s1["z_r"]))
        # K = 8 runner on the same weights: PB8-only path
        R8 = RavenR(rk, m, "cpu", 8)
        R8.bos, R8.fam, R8.name = None, "x", "huginn8"
        X8 = Item(R8, dict(it))
        run_item_rows(X8, 0, "D2", False)
        k8_rows = sorted(X8.rows)
        atomic_json(os.path.join(tmp, "res_huginn8_D2.json"), {"ds": "D2", "model": "huginn8", "items": [
            dict(X8.out, n_rows=len(X8.rows), t_item=0.0)], "meta": {}})     # review 3 F3: summarize reads it
        for h in R8.hs:
            h.remove()
        s = ("c", 16, R.C // 2)
        o = R.run(it["A"], 8, edits=X.o1(s, allp))
        neg = not torch.equal(o["z_r"], X.cB["z_r"])
        o = R.run(it["A"], 8, edits=X.o1(s, allp), e_edits=X.e_ed(range(18, R.K + 1), allp))
        neg2 = not torch.equal(o["z_r"], X.cB["z_r"])          # call 17 still reads A's e: must be inexact
        o = R.run(it["A"], 8, edits=X.o1(s, allp), e_edits=X.e_ed(range(17, R.K + 1), allp))
        pos2 = bool(torch.equal(o["z_r"], X.cB["z_r"]))
        r_ = {"hook_checks_ok": hk["all_ok"], "hook_checks": hk, "readout_exact_K": X.out["readout_exact_K"],
              "controls_exact": all(v["exact"] for v in X.rows.values() if "exact" in v),
              "pcs_without_e_inexact": neg, "pcs_e_one_call_late_inexact": neg2, "pcs_exact": pos2, "n_rows": len(X.rows),
              "grad_identical": X.out["grad_identical"], "G_n": len(X.out["G"]),
              "eown_rows": sum(1 for v in X.rows.values() if "mj" in v), "o2_rows": sum(1 for k in X.rows if k.startswith("O2|")),
              "dnorm_keys": len(X.out["dnorm"]), "v3_rows": v3_rows, "in1s_n1_exact": in1s_n1_exact,
              "k8_rows": k8_rows}
        r_["pig"] = _pig_checks(X, e8_sets(R.K, R.C)[1])
        R16 = RavenR(rk, m, "cpu", 16)              # review 3 driver test on a K = 16 runner named as an E8 model
        R16.bos, R16.fam, R16.name = None, "x", "mcleish"
        r_["driver"] = _driver_test(R16)
        for h in R16.hs:
            h.remove()
        X.out.update({"n_rows": len(X.rows), "t_item": 0.0})
        d = dict(X.out)
        resd = {"items": [d, dict(d, pair_id="tiny|1", split="test", full_search=False,
                                  rows={k: v for k, v in X.rows.items() if not k.startswith("O1|ans|c")})]}
        dt = dev_tuned_pass(R, resd, {"tiny|1": it})
        r_["dev_tuned"] = dt["S"]["site"]
        atomic_json(os.path.join(tmp, "res_%s_D1.json" % tn), {"ds": "D1", "model": tn, "items": resd["items"],
                                                                 "meta": {}, "dev_tuned": dt})
        r_["D"] = X.out["D"]
        # FD check path on an fp32 copy of the tiny model (critic 2 F2); the model is cast to bf16 inside
        mf = tiny_raven(sd, dtype=torch.float32)

        def mk(mm, rk=rk):
            Rf = RavenR(rk, mm, "cpu", 32)
            Rf.bos, Rf.fam, Rf.name = None, "x", "tiny_fd"
            return Rf
        fdo = fdcheck("selftest", None, None, "cpu", n_items=[it], models=[("tiny_fd", mf, mk)])
        fm = fdo["models"]["tiny_fd"]
        fdc = fm["summary"]["classes"]
        rels = [v["pooled|a=0.01"]["median_rel"] for v in fdc.values() if v["pooled|a=0.01"]["n_above_floor"]]
        r_["fd"] = {"floor": fm["summary"]["floor"], "median_rel_a0.01": rels, "grad_identical": fm["grad_run_identical_all"],
                    "spearman_bf16_fp32": fm["spearman_G_bf16_vs_fp32"], "verdict_1item": fm["summary"]["verdict"]}
        fd_ok = bool(fm["grad_run_identical_all"] and rels and max(rels) < 0.05
                     and fm["summary"]["verdict"] == "insufficient")
        r_["ok"] = bool(hk["all_ok"] and r_["readout_exact_K"] and r_["controls_exact"] and neg and neg2 and pos2
                        and (r_["dev_tuned"] is not None or X.D < GAP) and all(v3_rows.values()) and in1s_n1_exact
                        and "O1|src|c7.%d|1" % (R.C - 1) in k8_rows and "O1|src|c4.%d|1" % (R.C - 1) in k8_rows
                        and "PCF" in k8_rows and len(k8_rows) <= 8 and fd_ok
                        and r_["grad_identical"] and r_["G_n"] == 32 * R.C and r_["eown_rows"] > 0 and r_["o2_rows"] == 8
                        and r_["pig"]["a0_equals_G"] and r_["pig"]["a1_equals_B"] and r_["pig"]["n_pig"] == 32 * R.C
                        and r_["pig"]["n_Ga"] == len(PIG_A) and r_["pig"]["pig_top_verified"]
                        and r_["pig"]["ids_differ_only_at_src"] and r_["driver"]["ok"]
                        and bool(fm.get("a025_noise")) and fm.get("rank_agreement_bf16_vs_fp32") is not None)
        res[key] = r_
        ok = ok and r_["ok"]
    m = tiny_llama()
    R = LlamaR(m, "cpu")
    R.bos, R.fam, R.name = None, "x", "tiny_llama"
    it = _tiny_item(R, 300)
    hk = check_hooks(R, it)
    X = Item(R, it)
    X.depth_margins()
    X.deltas()
    run_controls(X, subset=True)
    run_e1(X, tiny=True)
    run_para(X, force=True)
    run_e8(X, full_search=True)
    X.out.update({"n_rows": len(X.rows), "t_item": 0.0})
    r_ = {"hook_checks_ok": hk["all_ok"], "controls_exact": all(v["exact"] for v in X.rows.values() if "exact" in v),
          "n_rows": len(X.rows), "G_n": len(X.out["G"]), "grad_identical": X.out["grad_identical"],
          "rnd_all_layers": all(("RND|src|c1.%d" % l) in X.rows for l in range(16))}
    r_["pig"] = _pig_checks(X, e8_sets(1, 16)[0])
    R2 = LlamaR(m, "cpu")
    R2.bos, R2.fam, R2.name = None, "x", "llama_nr"
    r_["driver"] = _driver_test(R2)
    for h in R2.hs:
        h.remove()
    mf = tiny_llama(dtype=torch.float32)

    def mkl(mm):
        Rf = LlamaR(mm, "cpu")
        Rf.bos, Rf.fam, Rf.name = None, "x", "tiny_fd_llama"
        return Rf
    fdo = fdcheck("selftest", None, None, "cpu", n_items=[it], models=[("tiny_fd_llama", mf, mkl)])
    fm = fdo["models"]["tiny_fd_llama"]
    rels = [v["pooled|a=0.01"]["median_rel"] for v in fm["summary"]["classes"].values()
            if v["pooled|a=0.01"]["n_above_floor"]]
    r_["fd"] = {"floor": fm["summary"]["floor"], "median_rel_a0.01": rels, "spearman": fm["spearman_G_bf16_vs_fp32"]}
    r_["ok"] = bool(hk["all_ok"] and r_["controls_exact"] and r_["G_n"] == 15 and r_["grad_identical"]
                    and r_["rnd_all_layers"] and fm["grad_run_identical_all"] and rels and max(rels) < 0.05
                    and r_["pig"]["a0_equals_G"] and r_["pig"]["a1_equals_B"] and r_["pig"]["n_pig"] == 15
                    and r_["pig"]["pig_top_verified"] and r_["driver"]["ok"]
                    and fm.get("rank_agreement_bf16_vs_fp32") is not None)
    atomic_json(os.path.join(tmp, "fd_selftest.json"), fdo)          # summarize reads the FD fields (F6, F7)
    res["llama"] = r_
    ok = ok and r_["ok"]
    atomic_json(os.path.join(tmp, "res_tiny_llama_D1.json"), {"ds": "D1", "model": "tiny_llama", "items": [X.out], "meta": {}})
    try:
        ck = summarize(tmp)
        res["summarize_ok"] = all(("tiny_raven|D1" if k.startswith("mcleish") else "tiny_huginn|D1") in ck["e1_cells"]
                                  for k in snaps) and "tiny_llama|D1" in ck["e1_cells"] \
            and (not snaps or "huginn8|D2" in ck["items"]) and "tiny_fd_llama" in ck["fd_check"] \
            and all((v.get("S|a=1") or {}).get("pig") is not None for v in ck["e8"].values())
        res["summarize_keys"] = sorted(ck["e1_cells"])
    except Exception as ex:  # noqa: BLE001
        res["summarize_ok"] = False
        res["summarize_error"] = repr(ex)
    ok = ok and res["summarize_ok"]
    res["raven_snapshots_tested"] = snaps
    res["ok"] = bool(ok)
    print(json.dumps(res, default=str)[:6000])
    print(json.dumps({"selftest_model_ok": res["ok"], "summarize_ok": res.get("summarize_ok"),      # last line, untruncated
                      "parts_ok": {k: v.get("ok") for k, v in res.items() if isinstance(v, dict) and "ok" in v},
                      "driver_ok": {k: (v.get("driver") or {}).get("ok") for k, v in res.items() if isinstance(v, dict)},
                      "raven_snapshots_tested": snaps}))
    return ok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest_model", action="store_true")
    ap.add_argument("--build_items", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--fdcheck", action="store_true", help="fp32 finite-difference check of G (critic 2 F2)")
    ap.add_argument("--fd_items", type=int, default=FD_N)
    ap.add_argument("--model", default="mcl_nr", choices=sorted(TARGET))
    ap.add_argument("--data", default="../out/k17_public/data")
    ap.add_argument("--out", default="../out/k17_public")
    ap.add_argument("--plan", default="")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--budget_min", type=float, default=120.0)
    ap.add_argument("--no_gradient_rank", action="store_true",
                    help="cluster copy: skip E8 (gradient score, PIG, verify rows; site selection is not run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_seeds", action="store_true")
    ap.add_argument("--targets", default="", help="cut rules, e.g. VB=40,CF=40 (overrides TARGET[model])")
    ap.add_argument("--descriptive", action="store_true",
                    help="allow --targets below %d; those datasets are recorded and reported as descriptive" % MIN_KEPT)
    ap.add_argument("--full_search_cap", default="",
                    help="E8 full-search cap: 'N' (every model), 'model=N' or 'model.DS=N', comma list (default "
                         "DEV_FULL_CAP); e.g. huginn=15 is cut rule (8), mcleish.D4=0 drops the McLeish D4 search")
    ap.add_argument("--pc_site_n", default="", help="PC-site / SC-identity subset size: 'N' or 'model=N' (default N_SUB)")
    ap.add_argument("--stage", default="all", choices=("all", "scan", "rows", "merge"),
                    help="all = sequential; cluster: scan (global selection + E8 pilot), rows --shard k/n, merge")
    ap.add_argument("--shard", default="", help="k/n for --stage rows (kept index %% n == k), one --out for all shards")
    ap.add_argument("--count_caps", action="store_true", help="Selector upper bounds over the candidate files (CPU)")
    ap.add_argument("--hash_inputs", action="store_true", help="print the pre-run hashes (G8) without running")
    ap.add_argument("--no_reduced", action="store_true", help="RA-1 cut: no Huginn K = 16 / K = 8 runners, no PB8 rows")
    a = ap.parse_args()
    for kv in [t for t in a.targets.split(",") if t]:
        k, v = kv.split("=")
        assert k in TARGET[a.model], kv
        assert int(v) >= MIN_KEPT or a.descriptive, "%s below %d needs --descriptive (review 3 F12)" % (kv, MIN_KEPT)
        if int(v) < MIN_KEPT:
            DESCRIPTIVE["targets"][k] = int(v)
        TARGET[a.model][k] = int(v)
    FS_CAP_OVR.update(parse_kv(a.full_search_cap))
    RUN_OPTS["no_reduced"] = bool(a.no_reduced)
    if a.no_gradient_rank:
        RUN_OPTS["no_gradient_rank"] = True
    PCS_OVR.update(parse_kv(a.pc_site_n))
    shard = None
    if a.shard:
        k, n = (int(t) for t in a.shard.split("/"))
        assert 0 <= k < n, a.shard
        shard = (k, n)
    assert (a.stage == "rows") == (shard is not None), "--shard k/n goes with --stage rows only"
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.selftest_model:
        sys.exit(0 if selftest_model() else 1)
    if a.build_items:
        build_items(a.out)
        return
    if a.count_caps:
        count_caps(a.data)
        return
    if a.summarize:
        summarize(a.out, a.plan or None)
        return
    if a.fdcheck:
        fdcheck(a.model, a.data, a.out, a.device, n_items=a.fd_items)
        return
    ds = [d for d in a.datasets.split(",") if d]
    assert all(d in DATASETS for d in ds), ds
    if a.hash_inputs:
        cfg = run_config(a.model, a.debug, ds, a.budget_min, not a.no_seeds, a.stage, shard)
        print(json.dumps(pre_run_hashes(a.model, a.data, a.plan or None, cfg)))
        return
    if a.run:
        # G8: the written hypotheses are hashed before a scored run; only a --debug run may go without the plan file
        assert a.debug or (a.plan and os.path.exists(a.plan)), "--plan PLAN.md is required for a scored run (G8)"
        run_group(a.model, a.data, a.out, a.device, a.debug, ds, a.budget_min, seeds=not a.no_seeds, stage=a.stage,
                  shard=shard, plan=a.plan or None)


if __name__ == "__main__":
    main()
