"""Build the one CounterFact pool of the cluster run and the k17_public CF candidate files from it (CPU).

Decision (2026-10-09, option A): every model scans the same CF pool, the 3,741 pairs of the production CF builder
that the Ouro-1.4B and 2.6B base CF cells were run on, each with its own clean filter. This script ports that
builder (prod/steer/data.py: build_cf with _cf_candidates and cf_half, then the runner's pos0 = exclude through
_strata and _apply_pos0) and checks it against the production run's kept keys when --check_kept is given.

    python cluster/build_cf_pool.py --records cluster/pilot/data/k15_full/cf_records.json \
        --ouro_tok <Ouro-1.4B tokenizer.json> --public_dir cluster/pilot/data/public [--check_kept M1__CF.json ...]

Writes cluster/pilot/data/cf_pool/cf_pool.json (header + pairs as text) and, for the llama (McLeish, twin) and
huginn families, cands_<fam>_CF.json in --public_dir (the k17_public build_cf format, from the pool instead of
k15_data.cf_candidates, with no subject selector cap: "sel": {"subjects": []}), and updates build_meta.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, "pilot")
sys.path.insert(0, PILOT)

import cf_pool as CFP  # noqa: E402

CF_WINDOW = 5                                    # prod/steer/data.py CF_WINDOW (k15_data CF_WINDOW)

NOTE = ("CounterFact pool of the cluster run: the 3,741 candidate pairs the production CF builder made from "
        "cf_records.json (prod/steer/data.py build_cf, then the runner's pos0 = exclude). The filters are applied "
        "on the Ouro tokenizer (ByteDance/Ouro-1.4B tokenizer.json, no BOS), not before tokenization: (1) records of "
        "relations P17, P103, P364, P641, P176 with at least one paraphrase prompt, split by subject half "
        "(int(sha256('cf-split:' + subject)[:8], 16) % 2; 0 = dev, 1 = test; pairs only inside a half); (2) the "
        "answer ' ' + target_true is one Ouro token; (3) pairs within one relation and one prompt template, record A "
        "paired with the next records of the template in case-id order, at most 5 partners per record A (window 5), "
        "skipping pairs whose Ouro prompts differ in length, whose answer tokens are equal, whose subjects are equal, "
        "or whose differing token span (common prefix and suffix removed) is empty or of unequal length: 9,125 "
        "pairs; (4) pos0 = exclude: pairs whose differing span starts at position 0 of the Ouro prompt (the "
        "subject opens the prompt) are removed: 5,384 removed, 3,741 kept. No sampling, cap or quota. On another "
        "tokenizer a pair is kept when its answers are single, different tokens and prompts A and B have equal "
        "length with an aligned source span (cluster/pilot/cf_pool.py); the model's own clean filter follows.")


def ouro_enc(tok_path):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(tok_path)

    def enc(text, ans=False):
        return list(t.encode(text, add_special_tokens=False).ids)
    return enc, hashlib.sha256(open(tok_path, "rb").read()).hexdigest()


def cf_half(subject):
    return int(hashlib.sha256(("cf-split:" + subject).encode("utf-8")).hexdigest()[:8], 16) % 2


def production_pool(recs, enc):
    """Port of prod/steer/data.py _cf_candidates + build_cf (both halves) + pos0 exclude (first content pos 0)."""
    pairs, stats = [], {"per_half": {}, "candidates": 0, "pos0_removed": 0}
    for h in (0, 1):
        st = {}
        for rel in CFP.CF_RELS:
            rows = []
            for r in recs:
                if r["rel"] != rel or not r["para"] or cf_half(r["subject"]) != h:
                    continue
                ta = enc(" " + r["answer"], ans=True)
                if len(ta) != 1:
                    continue
                rows.append((r, enc(r["template"].format(r["subject"])), ta[0]))
            by_t = {}
            for x in rows:
                by_t.setdefault(x[0]["template"], []).append(x)
            cands = []
            for grp in by_t.values():
                for i, (ra, ia, ta) in enumerate(grp):
                    k = 0
                    for rb, ib, tb in grp[i + 1:]:
                        if k >= CF_WINDOW:
                            break
                        if len(ia) != len(ib) or ta == tb or ra["subject"] == rb["subject"]:
                            continue
                        sa, sb = CFP.subj_span(ia, ib)
                        if not sa or len(sa) != len(sb):
                            continue
                        k += 1
                        cands.append((ra, rb, sa))
            cands.sort(key=lambda c: (c[0]["case_id"], c[1]["case_id"]))
            st[rel] = {"single_token_records": len(rows), "templates": len(by_t), "candidates": len(cands)}
            stats["candidates"] += len(cands)
            for ra, rb, sa in cands:
                if sa[0] == 0:                       # subject_pos0 on the Ouro tokenizer (no BOS): pos0 = exclude
                    stats["pos0_removed"] += 1
                    continue
                pairs.append({"key": "cf-%d-%d" % (ra["case_id"], rb["case_id"]), "rel": rel,
                              "half": "dev" if h == 0 else "test", "case_a": ra["case_id"], "case_b": rb["case_id"],
                              "template": ra["template"], "subject_a": ra["subject"], "subject_b": rb["subject"],
                              "answer_a": ra["answer"], "answer_b": rb["answer"], "para": list(ra["para"])})
        stats["per_half"]["dev" if h == 0 else "test"] = st
    stats["pairs"] = len(pairs)
    return pairs, stats


def public_cands(KP, tk, pool):
    """k17_public build_cf on the pool (same item format), subject selector cap off (sel subjects empty)."""
    enc = tk.enc
    cands, cst = CFP.pool_candidates(enc, pool)
    by = {}
    for p in cands:
        ua, ub = KP.ent_key("cf", p["a"]), KP.ent_key("cf", p["b"])
        forms = KP.para_forms(p["pos"], [("para%d" % n, a, b) for n, (a, b) in enumerate(p["para"])])
        by.setdefault(p["rel"], []).append({
            "ds": "CF", "pair_id": "CF|%s|%d|%d" % (p["rel"], p["case_a"], p["case_b"]), "A": list(p["A"]),
            "B": list(p["B"]), "r": len(p["A"]) - 1, "src": list(p["pos"]), "tA": int(p["tA"]), "tB": int(p["tB"]),
            "restA": [], "restB": [], "rel": p["rel"], "cl": [ua, ub], "split": KP.split_e8([ua, ub], "k17pub-cf:"),
            "strata": {"pos_len": "1" if len(p["pos"]) == 1 else ">=2", "rel": p["rel"],
                       "subject_pos0": "yes" if 0 in p["pos"] else "no"},
            "para": forms, "sel": {"subjects": []}, "label": "%s -> %s" % (p["a"], p["b"]),
            "pool_key": p["pool_key"], "pool_half": p["pool_half"]})
    out = KP.rr_order([c for rel in CFP.CF_RELS for c in by.get(rel, [])], lambda c: c["rel"])
    return out, {"candidates": len(out), "per_rel": {r: len(by.get(r, [])) for r in CFP.CF_RELS}, "pool_stats": cst,
                 "with_para": sum(1 for c in out if c["para"]), "source": "cf_pool.json (cluster/build_cf_pool.py)",
                 "subject_cap": "none (sel subjects empty)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--ouro_tok", required=True)
    ap.add_argument("--out", default=os.path.join(PILOT, "data", "cf_pool", "cf_pool.json"))
    ap.add_argument("--public_dir", default="")
    ap.add_argument("--check_kept", default="", help="comma list of production kept/<model>__CF.json files")
    a = ap.parse_args()
    recs = json.load(open(a.records, encoding="utf-8"))["records"]
    enc, tok_sha = ouro_enc(a.ouro_tok)
    t0 = time.time()
    pairs, stats = production_pool(recs, enc)
    rec_sha = hashlib.sha256(open(a.records, "rb").read()).hexdigest()
    checks = {}
    keys = {q["key"] for q in pairs}
    for f in [x for x in a.check_kept.split(",") if x]:
        k = json.load(open(f, encoding="utf-8"))
        checks[os.path.basename(f)] = {"kept_keys": len(k["keys"]), "in_pool": sum(1 for x in k["keys"] if x in keys),
                                       "production_n_candidates": k.get("n_candidates")}
        assert checks[os.path.basename(f)]["in_pool"] == len(k["keys"]) and k.get("n_candidates") == len(pairs), checks
    header = {"note": NOTE, "n_pairs": len(pairs), "builder_stats": stats, "records_sha256": rec_sha,
              "ouro_tokenizer_sha256": tok_sha, "port_of": "prod/steer/data.py build_cf, _cf_candidates, cf_half; "
              "runner pos0 = exclude", "checks_against_production": checks}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"header": header, "pairs": pairs}, f)
    print(json.dumps({"pool": len(pairs), "seconds": round(time.time() - t0, 1), "checks": checks}))
    if a.public_dir:
        import k17_public as KP
        meta_p = os.path.join(a.public_dir, "build_meta.json")
        meta = json.load(open(meta_p, encoding="utf-8"))
        pool = {"header": header, "pairs": pairs}
        for fam in ("llama", "huginn"):
            tk = KP.Tok(fam, KP.LAPTOP_TOK[fam])
            assert tk.sha16 == KP.TOK_SHA16[fam], fam
            cands, st = public_cands(KP, tk, pool)
            p = os.path.join(a.public_dir, "cands_%s_CF.json" % fam)
            KP.atomic_json(p, {"ds": "CF", "tok": fam, "stats": st, "cands": cands})
            st["file_sha256"] = KP.sha256_file(p)
            st["pool_sha256"] = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
            meta["datasets"]["%s|CF" % fam] = st
            print(fam, json.dumps({k: v for k, v in st.items() if k not in ("pool_stats",)}))
        KP.atomic_json(meta_p, meta)


if __name__ == "__main__":
    main()
