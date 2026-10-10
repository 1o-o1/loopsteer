"""RA-23 item 5: the linear-relations facts set (data_steer/lre, built in the facts format) as k17_public D1 candidate
files, one per tokenizer family, so k17_public runs it through its facts path with --data <this folder> (no code
change; the cell is registered as dataset LRE, the records say D1 / facts).

Rules (coordinator, ledger 2026-10-09):
  1. answer_in_prompt pairs are excluded;
  2. the runner's position-0 rule is kept: on a family without BOS (Ouro, Huginn) a pair whose source span holds
     position 0 is dropped here as k17_public.prep_item drops it at run time (so the caps count usable pairs);
  3. at most 2 pairs per subject (either side) inside a relation;
  4. a per-relation cap of CAP pairs, filled round-robin over the relation's (template, half) groups in file order,
     so the largest relations do not dominate; the file is then ordered round-robin over relations.
Each kept pair becomes a k17_public D1 candidate: ids without BOS (the runner prepends it), source = the pair's
source_span, single-token answers, cluster units rel:subject, split = the set's subject-first half, strata pos_len /
rel / subject_pos0, paraphrase forms encoded here with the family tokenizer (k17_public.para_forms rule).

    python cluster/build_lre_pool.py --lre loopedcot/data_steer/lre --out cluster/pilot/data/public_lre \\
        --tok ouro=TOKENIZER.json --tok huginn=TOKENIZER.json --tok llama=TOKENIZER.json [--cap 10]
"""
import argparse
import hashlib
import json
import os

FAMS = {"ouro": "ouro", "huginn": "huginn", "llama": "mcleish"}      # k17_public family -> LRE pairs file suffix
BOS = {"ouro": None, "huginn": None, "llama": 128000}
CAP_DEFAULT = 10
SUBJ_CAP = 2


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
    """k17_public.para_forms: same span length, equal length, nonempty aligned span."""
    out = []
    for fmt, a, b in forms:
        if len(a) != len(b):
            continue
        sa, sb = subj_span(a, b)
        if not sa or sa != sb or len(sa) != len(main_src):
            continue
        out.append({"fmt": fmt, "A": list(a), "B": list(b), "src": sa, "r": len(a) - 1})
    return out


def rr_order(items, key):
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


def build(fam, rows, enc, cap):
    st = {"rows": len(rows), "answer_in_prompt": 0, "pos0_no_bos": 0, "bos_mismatch": 0}
    ok = []
    for q in rows:
        if q["answer_in_prompt"]:
            st["answer_in_prompt"] += 1
            continue
        A, B, src = list(q["ids_A"]), list(q["ids_B"]), list(q["source_span"])
        if BOS[fam] is not None:
            if A[0] != BOS[fam] or B[0] != BOS[fam]:
                st["bos_mismatch"] += 1
                continue
            A, B, src = A[1:], B[1:], [p - 1 for p in src]
        elif 0 in src:
            st["pos0_no_bos"] += 1
            continue
        ok.append((q, A, B, src))
    by_rel = {}
    for x in ok:
        by_rel.setdefault(x[0]["rel"], []).append(x)
    kept = []
    st["per_relation"] = {}
    for rel in sorted(by_rel):
        groups = rr_order(by_rel[rel], lambda x: (x[0]["template_idx"], x[0]["half"]))
        n_s, take = {}, []
        for x in groups:
            if len(take) >= cap:
                break
            q = x[0]
            sa, sb = "%s|%s" % (rel, q["subject_a"]), "%s|%s" % (rel, q["subject_b"])
            if n_s.get(sa, 0) >= SUBJ_CAP or n_s.get(sb, 0) >= SUBJ_CAP:
                continue
            n_s[sa] = n_s.get(sa, 0) + 1
            n_s[sb] = n_s.get(sb, 0) + 1
            take.append(x)
        st["per_relation"][rel] = {"usable": len(by_rel[rel]), "kept": len(take)}
        kept += take
    cands = []
    for q, A, B, src in kept:
        ua, ub = "%s:%s" % (q["rel"], q["subject_a"]), "%s:%s" % (q["rel"], q["subject_b"])
        forms = para_forms(src, [("para%d" % n, enc(p["template"].format(q["subject_a"])),
                                  enc(p["template"].format(q["subject_b"]))) for n, p in enumerate(q["para"])])
        assert len(A) == len(B) and q["tA"] != q["tB"] and src and all(A[i] == B[i] for i in range(len(A)) if i not in src)
        cands.append({"ds": "D1", "pair_id": "LRE|%s" % q["key"], "A": A, "B": B, "r": len(A) - 1, "src": src,
                      "tA": int(q["tA"]), "tB": int(q["tB"]), "restA": [], "restB": [], "rel": q["rel"],
                      "cl": [ua, ub], "split": q["half"],
                      "strata": {"pos_len": "1" if len(src) == 1 else ">=2", "rel": q["rel"],
                                 "subject_pos0": "yes" if 0 in src else "no"},
                      "para": forms, "sel": {"subjects": ["%s|%s" % (q["rel"], ua), "%s|%s" % (q["rel"], ub)]},
                      "label": "%s -> %s (%s)" % (q["subject_a"], q["subject_b"], q["rel"]),
                      "lre": {"key": q["key"], "template_idx": q["template_idx"], "relation_type": q["relation_type"]}})
    cands = rr_order(cands, lambda c: c["rel"])
    st.update({"usable": len(ok), "candidates": len(cands), "relations": len(st["per_relation"]),
               "relations_with_candidates": sum(1 for v in st["per_relation"].values() if v["kept"]),
               "split": {h: sum(1 for c in cands if c["split"] == h) for h in ("dev", "test")},
               "with_para": sum(1 for c in cands if c["para"])})
    return cands, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lre", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", action="append", default=[], help="fam=tokenizer.json (ouro, huginn, llama)")
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    toks = dict(x.split("=", 1) for x in a.tok)
    os.makedirs(a.out, exist_ok=True)
    meta = {"built_by": "cluster/build_lre_pool.py", "rules": __doc__.split("Rules")[1].split("Each kept")[0].strip(),
            "cap_per_relation": a.cap, "subject_cap": SUBJ_CAP, "inputs": {}, "tokenizers": {}, "outputs": {}}
    man = json.load(open(os.path.join(a.lre, "MANIFEST.json"), encoding="utf-8"))
    meta["lre_source"] = {k: man["source"][k] for k in ("repo", "commit", "license")}
    for fam, suf in FAMS.items():
        t = Tokenizer.from_file(toks[fam])
        enc = lambda s, t=t: list(t.encode(s, add_special_tokens=False).ids)
        pf = os.path.join(a.lre, "pairs_lre_%s.jsonl" % suf)
        rows = [json.loads(x) for x in open(pf, encoding="utf-8") if x.strip()]
        cands, st = build(fam, rows, enc, a.cap)
        name = "cands_%s_D1.json" % fam
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"ds": "D1", "tok": fam, "set": "LRE", "stats": st, "cands": cands}, f)
        meta["inputs"]["pairs_lre_%s.jsonl" % suf] = sha256_file(pf)
        meta["tokenizers"][fam] = sha256_file(toks[fam])
        meta["outputs"][name] = {"sha256": sha256_file(os.path.join(a.out, name)), "candidates": len(cands)}
        print(json.dumps({"fam": fam, **{k: v for k, v in st.items() if k != "per_relation"}}))
    with open(os.path.join(a.out, "build_meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
