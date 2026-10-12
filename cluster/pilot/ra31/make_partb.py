"""RA-31 Part B data (CPU, run once before the plan hash): the frozen Part B unit sets from the Part A pilot, and
copies of the unedited reference run (cluster job a1_gsm8k: items.json, r00 chains, cells and meta), with the
sha256 of every source and output in data_partb/sources_partb.json. Token ids in the copies were made on the cluster
host (transformers 4.56.2); nothing here tokenizes.

Sets (plan section 13; a recorded deviation: frozen from the Part A pilot, 50 donor items, because the Part A full
run ends after the push): rank neurons (heads) by the number of donor items whose top set holds them; ties by the
lower flat index (the pilot records store the top sets only, review N7, so the summed score the brief names for the
tie-break is not available); keep 5 percent of the L x I neurons (6,758) and 2 of the L x H heads. rand: one seeded
draw of each size, the same for every problem.

    python make_partb.py [--pilot out/pilot_1.4b] [--a1 <a1_gsm8k job folder>]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L_14B, I_14B, H_14B = 24, 5632, 16
N_HEADS_TOP = 2


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rank_by_count(lists, N, n):
    cnt = [0] * N
    for xs in lists:
        for u in xs:
            cnt[u] += 1
    order = sorted(range(N), key=lambda u: (-cnt[u], u))
    return sorted(order[:n]), {"min_count_kept": cnt[order[n - 1]], "max_count": cnt[order[0]],
                               "n_tied_at_cut": sum(1 for u in range(N) if cnt[u] == cnt[order[n - 1]])}


def freeze_sets(pilot_dir, L=L_14B, I=I_14B, H=H_14B):
    import ra31 as R
    recs = []
    for ds in ("D1", "D2"):
        with open(os.path.join(pilot_dir, "items_%s.jsonl" % ds), encoding="utf-8") as f:
            recs += [json.loads(x) for x in f if x.strip()]
    n5 = R.n_top_neurons(L, I)
    mlp = [r["sets"]["donor"]["mlp_top"] for r in recs]
    attn = [r["sets"]["donor"]["attn_top"] for r in recs]
    assert all(len(x) == n5 for x in mlp) and all(max(x) < L * I for x in mlp), "pilot neuron sets"
    top_m, im = rank_by_count(mlp, L * I, n5)
    top_h, ih = rank_by_count(attn, L * H, N_HEADS_TOP)
    return {"rule": "count of Part A pilot donor items whose top set holds the unit; ties by the lower flat index",
            "deviation": "frozen from the Part A pilot (50 donor items), not the full run; tie-break by index, not "
                         "summed score (the pilot stores the top sets only)",
            "n_items": len(recs), "L": L, "I": I, "H": H,
            "mlp": {"top": top_m, "rand": R.rand_units("ra31:partb:neurons", L * I, n5), "info": im},
            "attn": {"top": top_h, "rand": R.rand_units("ra31:partb:heads", L * H, N_HEADS_TOP), "info": ih}}


def build(pilot_dir, a1_dir, out):
    os.makedirs(out, exist_ok=True)
    src = {}
    copies = {"items.json": "items.json", "r00_chains.jsonl": "r00/chains_ouro_1_4b_base_gsm8k_k4.jsonl",
              "r00_cells.jsonl": "r00/cells_ouro_1_4b_base_gsm8k_natural_k4_sub417.jsonl",
              "r00_meta.json": "r00/meta_ouro_1_4b_base_gsm8k_natural_k4_sub417.json"}
    for dst, rel in copies.items():
        p = os.path.join(a1_dir, rel)
        src["a1_gsm8k/" + rel] = sha(p)
        if dst == "r00_meta.json":               # an extract: the full meta carries host paths (env, argv)
            m = json.load(open(p))
            keep = {"model_meta": {"steer_gen": m["model_meta"].get("steer_gen"),
                                   "file_md5": m["model_meta"].get("file_md5"), "name": m["model_meta"].get("name"),
                                   "dtype": m["model_meta"].get("dtype")},
                    "env": {k: m["env"].get(k) for k in ("torch", "transformers", "device")},
                    "config_sha256": m.get("config_sha256"), "n_problems": m.get("n_problems"),
                    "horizon": m.get("horizon"), "batch_width_pinned": m.get("batch_width_pinned"),
                    "protocol": m.get("protocol"), "k": m.get("k"), "source_sha256": src["a1_gsm8k/" + rel]}
            json.dump(keep, open(os.path.join(out, dst), "w"), indent=1, sort_keys=True)
        else:
            shutil.copyfile(p, os.path.join(out, dst))
    for ds in ("D1", "D2"):
        src["pilot_1.4b/items_%s.jsonl" % ds] = sha(os.path.join(pilot_dir, "items_%s.jsonl" % ds))
    sets = freeze_sets(pilot_dir)
    json.dump(sets, open(os.path.join(out, "partb_sets_1.4b.json"), "w"), indent=0, sort_keys=True)
    outs = {f: sha(os.path.join(out, f)) for f in sorted(list(copies) + ["partb_sets_1.4b.json"])}
    json.dump({"sources": src, "outputs": outs}, open(os.path.join(out, "sources_partb.json"), "w"), indent=1,
              sort_keys=True)
    return outs, sets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=os.path.join(HERE, "out", "pilot_1.4b"))
    ap.add_argument("--a1", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "artifacts_spark",
                                                 "cluster_local", "ra20_g2", "A", "steer_gen", "a1_gsm8k"))
    ap.add_argument("--out", default=os.path.join(HERE, "data_partb"))
    a = ap.parse_args()
    outs, sets = build(a.pilot, a.a1, a.out)
    print(json.dumps({"outputs": outs, "mlp_info": sets["mlp"]["info"], "attn_top": sets["attn"]["top"],
                      "attn_info": sets["attn"]["info"]}, indent=1))
