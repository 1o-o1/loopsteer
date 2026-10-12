"""RA-28 scoring, selection and set statistics (pure tensor code, CPU-testable; plan_ra28.md sections 5 to 7).

Notation at the last prompt token r, loop t, layer l (Ouro layer: attn -> input_layernorm_2 -> add; MLP ->
post_attention_layernorm_2 -> add, the sandwich norms of the transcribed decoder layer):
- a = SiLU(gate) * up, the down_proj input [I]; u = down_proj(a), the pre-norm MLP write [d].
- z = the o_proj input [H * hd] (per-head attention outputs before the output projection); o = o_proj(z) [d].
- h_t = the residual stream at the end of loop t, the input of the loop-end norm (model.norm).
- d_t = dM / dh_t on the clean run, M = z(tB) - z(tA) at r of the final logits ("the gradient of the margin at the
  residual"). For loop 4, d_4 = J_norm(h_4)^T (W_U[tB] - W_U[tA]) (rmsnorm_jt gives the closed form).
- The write of a unit enters h_t through the residual (the direct path inside loop t); the sandwich norm after the
  branch is linearized at its clean RMS (the norm scale frozen at clean): v = (g * d_t) / rms(u_clean).
Primary neuron score: s_j = (a_j^patched - a_j^clean) (W_down[:, j] . v)   (sum over j = (g * W_down da / rms) . d_t).
Primary head score:   s_h = (z_h^patched - z_h^clean) . (W_O[:, h] ^T v_attn).
Gradient-times-activation-change scores (attribution patching): s'_j = da_j dM/da_j, s'_h = dz_h . dM/dz_h (clean).
Restoring units are the most negative scores (they push the margin back toward the clean answer A).
"""
import hashlib
import math

import torch


# ------------------------------------------------------------------ norms
def inv_rms(x, eps):
    """1 / RMS of the last axis in float32, the arithmetic of OuroRMSNorm (x.float().pow(2).mean(-1) + eps)."""
    return torch.rsqrt(x.float().pow(2).mean(-1) + eps)


def frozen_norm_dir(d, g, u_clean, eps):
    """Direction for a pre-norm branch output when the norm scale is frozen at its clean value:
    v = g * d * inv_rms(u_clean). d, g, u_clean [..., D]."""
    return g.float() * d.float() * inv_rms(u_clean, eps)[..., None]


def rmsnorm_jt(x, g, eps, v):
    """J^T v for y = g * x * s(x), s = (mean(x^2) + eps)^(-1/2): s (g v) - (s^3 / n) x (x . g v)."""
    x = x.float()
    gv = g.float() * v.float()
    s = inv_rms(x, eps)[..., None]
    n = x.shape[-1]
    return s * gv - (s ** 3) / n * x * (x * gv).sum(-1, keepdim=True)


# ------------------------------------------------------------------ scores
def neuron_primary(da, W_down, v):
    """da [n, I], W_down [D, I] (nn.Linear weight), v [n, D] -> [n, I]: da_j (W_down[:, j] . v)."""
    return da.float() * (v.float() @ W_down.float())


def neuron_atp(da, ga):
    return da.float() * ga.float()


def head_primary(dz, W_o, v, H, hd):
    """dz [n, H*hd], W_o [D, H*hd], v [n, D] -> [n, H]: dz_h . (W_O[:, h-slice]^T v)."""
    q = v.float() @ W_o.float()
    return (dz.float() * q).reshape(-1, H, hd).sum(-1)


def head_atp(dz, gz, H, hd):
    return (dz.float() * gz.float()).reshape(-1, H, hd).sum(-1)


# ------------------------------------------------------------------ selection
def restoring_order(score):
    """Flat indices of `score` from most negative to most positive (stable: ties keep index order)."""
    return torch.argsort(score.reshape(-1).double(), stable=True)


def top_restoring(score, k):
    return sorted(int(i) for i in restoring_order(score)[:k])


def top_positive(score, k):
    return sorted(int(i) for i in torch.argsort(-score.reshape(-1).double(), stable=True)[:k])


def seed_of(*parts):
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:15], 16)


def random_subset(n, k, *parts):
    """k distinct indices of range(n), torch CPU generator seeded by sha256 of the parts (sorted)."""
    g = torch.Generator(device="cpu").manual_seed(seed_of(*parts))
    return sorted(int(i) for i in torch.randperm(n, generator=g)[:k])


def unravel(flat, shape):
    out = []
    for f in flat:
        idx = []
        for s in reversed(shape):
            idx.append(f % s)
            f //= s
        out.append(tuple(reversed(idx)))
    return out


def neuron_sets(mean_score, loops, k, L, I):
    """mean_score [K, L, I] (dev mean, loop t at index t - 1). Returns
    per_loop: sorted slots (t, l, j), the top k of each loop t in `loops` by its own score;
    shared:   sorted identities (l, j), the top k by the score summed over `loops`, as slots (t, l, j) in every loop.
    Both sets hold k * len(loops) (loop, unit) slots."""
    per = []
    for t in loops:
        per += [(t, l, j) for (l, j) in unravel(top_restoring(mean_score[t - 1], k), (L, I))]
    tot = sum(mean_score[t - 1] for t in loops)
    ids = unravel(top_restoring(tot, k), (L, I))
    shared = [(t, l, j) for t in loops for (l, j) in ids]
    return {"per_loop": sorted(per), "shared": sorted(shared), "shared_ids": sorted(ids)}


def random_neuron_sets(loops, k, L, I, *parts):
    per = []
    for t in loops:
        per += [(t, l, j) for (l, j) in unravel(random_subset(L * I, k, "loop", t, *parts), (L, I))]
    ids = unravel(random_subset(L * I, k, "shared", *parts), (L, I))
    return {"per_loop": sorted(per), "shared": sorted((t, l, j) for t in loops for (l, j) in ids)}


def head_slots(region, K, L, H):
    """Region slots (t, l, h): 'A' = layers 16 to 19 of loop K (the last-read loop), 'B' = loops 2 to K, all layers."""
    if region == "A":
        if L < 20:
            raise ValueError("region A needs layers 16 to 19 (model has %d layers)" % L)
        return [(K, l, h) for l in range(16, 20) for h in range(H)]
    if region == "B":
        return [(t, l, h) for t in range(2, K + 1) for l in range(L) for h in range(H)]
    raise ValueError(region)


def head_set(mean_score, region, k, K, L, H):
    """mean_score [K, L, H]; top k region slots by the most negative score."""
    slots = head_slots(region, K, L, H)
    v = torch.tensor([float(mean_score[t - 1, l, h]) for (t, l, h) in slots], dtype=torch.float64)
    return sorted(slots[i] for i in top_restoring(v, k))


def random_head_set(region, k, K, L, H, *parts):
    slots = head_slots(region, K, L, H)
    return sorted(slots[i] for i in random_subset(len(slots), k, "heads", region, *parts))


def group(slots):
    """[(t, l, x)] -> {(t, l): [x, ...]} with sorted x."""
    g = {}
    for t, l, x in slots:
        g.setdefault((t, l), []).append(x)
    return {key: sorted(v) for key, v in g.items()}


# ------------------------------------------------------------------ agreement
def ranks(x):
    x = x.reshape(-1).double()
    r = torch.empty_like(x)
    r[torch.argsort(x, stable=True)] = torch.arange(x.numel(), dtype=torch.float64)
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(ra.norm() * rb.norm())
    return None if den == 0 else float((ra * rb).sum()) / den


def topk_overlap(a, b, k):
    return len(set(top_restoring(a, k)) & set(top_restoring(b, k))) / float(k)


def set_overlap(s1, s2):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / float(max(1, min(len(s1), len(s2))))


def _lchoose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(x, N, K, n):
    """P(X >= x) for X ~ Hypergeometric(population N, K successes, n draws)."""
    lo, hi = max(x, 0, n - (N - K)), min(K, n)
    if lo > hi:
        return 0.0
    den = _lchoose(N, n)
    return float(min(1.0, sum(math.exp(_lchoose(K, i) + _lchoose(N - K, n - i) - den) for i in range(lo, hi + 1))))


def first_diff_tokens(a, b):
    """First position where token lists a and b differ -> (i, a[i], b[i]); None when one is a prefix of the other."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i, a[i], b[i]
    return None
