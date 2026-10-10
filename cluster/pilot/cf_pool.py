"""The CounterFact pool shared by every model of the cluster run (one pool file per dataset, RA-20 item 9).

The pool is the 3,741 candidate pairs of the production CF builder (prod/steer/data.py build_cf, then the
runner's pos0 = exclude; the same pairs the Ouro-1.4B and 2.6B base CF cells were scanned on). The pool file
(cluster/pilot/data/cf_pool/cf_pool.json, written by cluster/build_cf_pool.py) holds each pair as text: relation,
template, both subjects, both answers and the paraphrase templates of record A. Its header states how the 3,741
were built.

pool_candidates() turns the pool into a model's candidates on that model's tokenizer, in the k15_data.cf_candidates
format, keeping only the pairs whose prompts are well formed on that tokenizer (single-token answers that differ,
prompts A and B of equal length, a source span of equal length on both sides); the model's own clean filter is then
applied by its script. The pairs are not re-sampled, re-windowed or capped.
"""
import json

CF_RELS = ["P17", "P103", "P364", "P641", "P176"]


def subj_span(x, y):
    """Positions between the common prefix and the common suffix of x and y (k15_data.subj_span)."""
    n = min(len(x), len(y))
    p = 0
    while p < n and x[p] == y[p]:
        p += 1
    s = 0
    while s < n - p and x[-1 - s] == y[-1 - s]:
        s += 1
    return list(range(p, len(x) - s)), list(range(p, len(y) - s))


def load_pool(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pool_candidates(enc, pool, rels=CF_RELS):
    """-> (candidates in the k15_data.cf_candidates format, stats per relation). enc(text, ans=False) is the
    model's encoder (BOS rule of the script); answers are encoded with ans=True. Order: relation by relation in
    CF_RELS, then (case_a, case_b), as k15_data.cf_candidates."""
    out, stats = [], {}
    pairs = pool["pairs"]
    for rel in rels:
        st = {"pool_pairs": 0, "kept": 0, "answer_not_single_token": 0, "same_answer_token": 0,
              "unequal_prompt_length": 0, "unaligned_span": 0}
        cands = []
        for q in pairs:
            if q["rel"] != rel:
                continue
            st["pool_pairs"] += 1
            ta, tb = enc(" " + q["answer_a"], ans=True), enc(" " + q["answer_b"], ans=True)
            if len(ta) != 1 or len(tb) != 1:
                st["answer_not_single_token"] += 1
                continue
            if ta[0] == tb[0]:
                st["same_answer_token"] += 1
                continue
            ia, ib = enc(q["template"].format(q["subject_a"])), enc(q["template"].format(q["subject_b"]))
            if len(ia) != len(ib):
                st["unequal_prompt_length"] += 1
                continue
            sa, sb = subj_span(ia, ib)
            if not sa or len(sa) != len(sb):
                st["unaligned_span"] += 1
                continue
            para = [(enc(t.format(q["subject_a"])), enc(t.format(q["subject_b"]))) for t in q["para"]]
            cands.append({"A": ia, "B": ib, "pos": sa, "tA": ta[0], "tB": tb[0], "a": q["subject_a"],
                          "b": q["subject_b"], "rel": rel, "half": "cf", "suffix": [], "para": para,
                          "case_a": q["case_a"], "case_b": q["case_b"], "template": q["template"],
                          "ans_a": q["answer_a"], "ans_b": q["answer_b"], "pool_key": q["key"],
                          "pool_half": q["half"]})
            st["kept"] += 1
        cands.sort(key=lambda p: (p["case_a"], p["case_b"]))
        out += cands
        stats[rel] = st
    return out, stats
