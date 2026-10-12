"""K15 dataset builders (brief B-K15-FULL-A; read-only for B-K15-FULL-B). Plan: loop-signal/notes/k15_fullA_plan.md.

Pair dict (k13 keys plus three): A, B (prompt ids), pos (edit positions in A = B span), tA, tB (answer token ids),
a, b (subject labels), rel, half, para (list of (idsA', idsB') paraphrase prompts), suffix (common answer prefix
ids already appended to A, B and every paraphrase; usually []).
`enc(text, ans=False)` is the model's encoder as in k13/k15 (IFM adds BOS when ans is False, Ouro adds nothing).

  D1  d1_candidates(enc): k13 build_pairs, held-out (odd index) then dev (even), k15_smoke PARA paraphrases.
      select(cands, know, D1_QUOTA) gives smoke 2's 80 pairs.
  CF  cf_candidates(enc, recs): CounterFact (azhx/counterfact) records from cf_records.json; same template,
      equal token length, single-token answers that differ; record A's two paraphrase prompts.
  D2  d2_candidates(enc): 2-shot "a + b =" (D2_PROMPT), two-digit operands, sum <= 99, the tens digit of one
      operand differs; paraphrases D2_PARA.
      select(cands, know, D2_QUOTA).

    python k15_data.py --selftest [--tok_ouro REPO_OR_DIR] [--tok_ifm DIR]
    python k15_data.py --prepare_cf OUT.json        (reads azhx/counterfact from the HF cache or hub)
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CF_REPO = "azhx/counterfact"
CF_FILES = ["data/train-00000-of-00001-05d11247db7abce8.parquet", "data/test-00000-of-00001-bacb83500fca49a9.parquet"]
CF_RELS = ["P17", "P103", "P364", "P641", "P176"]
CF_PER_REL = 80
CF_WINDOW = 5
D1_QUOTA = {("heldout", "capital"): 20, ("heldout", "city"): 20, ("dev", "capital"): 20, ("dev", "city"): 20}
D2_QUOTA = {("all", "op1"): 40, ("all", "op2"): 40}
D2_PROMPT = "12 + 25 = 37\n41 + 33 = 74\n{} + {} ="       # 2-shot: bare "a + b =" gave 30/100 (Ouro-1.4B), 3/100 (2.6B)
D2_PARA = ["What is {} plus {}? Answer:",
           "Q: What is 12 plus 25? A: 37\nQ: What is 41 plus 33? A: 74\nQ: What is {} plus {}? A:"]
D2_SEED = 15
D2_N = 600


def subj_span(x, y):
    """Positions between the common prefix and the common suffix of token lists x and y (k15_smoke.subj_span)."""
    n = min(len(x), len(y))
    p = 0
    while p < n and x[p] == y[p]:
        p += 1
    s = 0
    while s < n - p and x[-1 - s] == y[-1 - s]:
        s += 1
    return list(range(p, len(x) - s)), list(range(p, len(y) - s))


def answer_tokens(enc, ans_a, ans_b):
    """-> (suffix, tA, tB): suffix = common prefix of the two answer tokenisations (appended to the prompt),
    tA, tB = the first differing tokens. None if one answer is a prefix of the other."""
    ta, tb = enc(" " + ans_a, ans=True), enc(" " + ans_b, ans=True)
    c = 0
    while c < min(len(ta), len(tb)) and ta[c] == tb[c]:
        c += 1
    if c >= len(ta) or c >= len(tb):
        return None
    return list(ta[:c]), ta[c], tb[c]


def select(cands, know, quota, cap=2):
    """Scan cands in order. Skip when the (half, rel) quota is full or a subject was used `cap` times in this
    (half, rel) (checked before know, as smoke 2). know(p) -> bool. Returns (kept, n_scanned)."""
    n_q, n_s, kept, scanned = {}, {}, [], 0
    for p in cands:
        q = (p["half"], p["rel"])
        if q not in quota or n_q.get(q, 0) >= quota[q]:
            continue
        ka, kb = q + (p["a"],), q + (p["b"],)
        if n_s.get(ka, 0) >= cap or n_s.get(kb, 0) >= cap:
            continue
        scanned += 1
        if not know(p):
            continue
        n_q[q] = n_q.get(q, 0) + 1
        n_s[ka] = n_s.get(ka, 0) + 1
        n_s[kb] = n_s.get(kb, 0) + 1
        kept.append(p)
        if all(n_q.get(k, 0) >= v for k, v in quota.items()):
            break
    return kept, scanned


# ------------------------------------------------------------------ D1
def d1_candidates(enc):
    from k13_edit import build_pairs
    from k15_smoke import PARA
    pairs = build_pairs(enc)
    out = []
    for half, parity in (("heldout", 1), ("dev", 0)):
        for i, p in enumerate(pairs):
            if i % 2 != parity:
                continue
            q = dict(p, half=half, idx=i, suffix=[],
                     para=[(enc(t.format(p["a"])), enc(t.format(p["b"]))) for t in PARA[p["rel"]]])
            out.append(q)
    return out


# ------------------------------------------------------------------ CounterFact
def prepare_cf(out_path, rels=CF_RELS):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    fs = [hf_hub_download(CF_REPO, f, repo_type="dataset") for f in CF_FILES]
    df = pd.concat([pd.read_parquet(f) for f in fs])
    recs, dropped = [], {"no_subject_in_para": 0}
    for _, r in df.iterrows():
        rr = r["requested_rewrite"]
        if rr["relation_id"] not in rels:
            continue
        subj = rr["subject"]
        paras = []
        for pp in r["paraphrase_prompts"]:
            if subj not in pp:
                dropped["no_subject_in_para"] += 1
                continue
            esc = pp.replace("{", "{{").replace("}", "}}")
            paras.append(esc.replace(subj.replace("{", "{{").replace("}", "}}"), "{}", 1))
        recs.append({"case_id": int(r["case_id"]), "rel": rr["relation_id"], "template": rr["prompt"],
                     "subject": subj, "answer": rr["target_true"]["str"], "para": paras})
    recs.sort(key=lambda x: x["case_id"])
    meta = {"source": CF_REPO, "files": CF_FILES, "relations": rels, "n_records": len(recs), "dropped": dropped,
            "per_relation": {k: sum(1 for x in recs if x["rel"] == k) for k in rels}}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump({"meta": meta, "records": recs}, open(out_path, "w"))
    print(json.dumps(meta))
    return meta


def cf_candidates(enc, recs, rels=CF_RELS, window=CF_WINDOW):
    """Candidate pairs in order (relation by relation, record order, then partner order). half = "cf"."""
    out, stats = [], {}
    for rel in rels:
        rows = []
        for r in recs:
            if r["rel"] != rel or not r["para"]:
                continue
            ta = enc(" " + r["answer"], ans=True)
            if len(ta) != 1:
                continue
            rows.append((r, enc(r["template"].format(r["subject"])), ta[0]))
        by_t = {}
        for x in rows:
            by_t.setdefault(x[0]["template"], []).append(x)
        n = 0
        cands = []
        for grp in by_t.values():
            for i, (ra, ia, ta) in enumerate(grp):
                k = 0
                for rb, ib, tb in grp[i + 1:]:
                    if k >= window:
                        break
                    if len(ia) != len(ib) or ta == tb or ra["subject"] == rb["subject"]:
                        continue
                    sa, sb = subj_span(ia, ib)
                    if not sa or len(sa) != len(sb):
                        continue
                    k += 1
                    para = [(enc(t.format(ra["subject"])), enc(t.format(rb["subject"]))) for t in ra["para"]]
                    cands.append({"A": ia, "B": ib, "pos": sa, "tA": ta, "tB": tb, "a": ra["subject"],
                                  "b": rb["subject"], "rel": rel, "half": "cf", "suffix": [], "para": para,
                                  "case_a": ra["case_id"], "case_b": rb["case_id"], "template": ra["template"],
                                  "ans_a": ra["answer"], "ans_b": rb["answer"]})
                    n += 1
        cands.sort(key=lambda p: (p["case_a"], p["case_b"]))
        out += cands
        stats[rel] = {"single_token_records": len(rows), "templates": len(by_t), "candidates": n}
    return out, stats


def cf_quota(rels=CF_RELS, per_rel=CF_PER_REL):
    return {("cf", r): per_rel for r in rels}


# ------------------------------------------------------------------ D2 arithmetic
def d2_raw(n=D2_N, seed=D2_SEED):
    """[(op, a, b, a2, b2)] with a, b in 10..89, sums <= 99, the tens digit of the op operand changed."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        op = "op1" if len(out) % 2 == 0 else "op2"
        a, b = rng.randint(10, 89), rng.randint(10, 89)
        if a + b > 99:
            continue
        x, y = (a, b) if op == "op1" else (b, a)
        ts = [t for t in range(1, 9) if t != x // 10 and 10 * t + x % 10 + y <= 99]
        if not ts:
            continue
        x2 = 10 * rng.choice(ts) + x % 10
        a2, b2 = (x2, b) if op == "op1" else (a, x2)
        out.append((op, a, b, a2, b2))
    return out


def d2_candidates(enc, n=D2_N, seed=D2_SEED):
    out, seen = [], set()
    for op, a, b, a2, b2 in d2_raw(n, seed):
        key = "all|%s|%d + %d|%d + %d" % (op, a, b, a2, b2)     # extractor item id half|rel|a|b
        if key in seen:                                            # d2_raw can draw a pair twice: keep the first
            continue
        seen.add(key)
        at = answer_tokens(enc, str(a + b), str(a2 + b2))
        if at is None:
            continue
        suf, tA, tB = at
        ia, ib = enc(D2_PROMPT.format(a, b)) + suf, enc(D2_PROMPT.format(a2, b2)) + suf
        if len(ia) != len(ib):
            continue
        sa, sb = subj_span(ia, ib)
        if not sa or len(sa) != len(sb):
            continue
        para = [(enc(t.format(a, b)) + suf, enc(t.format(a2, b2)) + suf) for t in D2_PARA]
        out.append({"A": ia, "B": ib, "pos": sa, "tA": tA, "tB": tB, "a": "%d + %d" % (a, b),
                    "b": "%d + %d" % (a2, b2), "rel": op, "half": "all", "suffix": suf, "para": para,
                    "sum_a": a + b, "sum_b": a2 + b2})
    assert len({"%s|%s|%s|%s" % (p["half"], p["rel"], p["a"], p["b"]) for p in out}) == len(out), "duplicate D2 item"
    return out


# ------------------------------------------------------------------ arm B prompts
NEUTRAL150 = [
    # smoke 2's 30
    "I went to the store and", "The weather today is", "My neighbor told me that", "Yesterday I watched a movie and",
    "The meeting this morning was", "When I opened the door,", "The food at the restaurant was",
    "Our trip to the city was", "The new phone I bought", "After school, the kids",
    "The book I am reading is", "Last weekend we drove to", "The train arrived at the station and",
    "My brother called me to say", "The hotel room was", "At the office today,", "The concert last night was",
    "I spent the afternoon", "The teacher explained that", "This morning I woke up and",
    "The park near my house is", "We had dinner with friends and", "The package was delivered and",
    "My first day at the new job was", "The flight to Chicago was", "The museum downtown is",
    "I tried the new coffee shop and", "The game on Sunday was", "Our garden this year is",
    "The doctor said that",
    # 120 more
    "The bus was late this morning, so", "My sister moved to a new apartment and", "The library opened at nine and",
    "We walked along the river and", "The software update changed", "On Tuesday the committee discussed",
    "The kitchen in our house is", "My cousin visited us last month and", "The lecture on history was",
    "I picked up my keys and", "The store on the corner sells", "During lunch, my colleague mentioned",
    "The bridge across the bay is", "Our dog spent the day", "The movie theater downtown",
    "My grandmother used to say that", "The road to the village", "After the rain stopped, we",
    "The children built a sandcastle and", "The report was submitted on", "Every Saturday, my father",
    "The new restaurant on Main Street", "I read an article about", "The office printer",
    "My friend and I decided to", "The bakery near the station", "In the evening, the town square",
    "The class started at eight and", "Our neighbors have a cat that", "The airport was",
    "I bought a ticket for", "The hospital on the hill", "My roommate cooked dinner and",
    "The beach in the summer is", "The city council announced that", "When the phone rang, I",
    "The bookstore had a sale on", "We stayed at a small inn and", "The project deadline is",
    "The farmer's market on Sunday", "My boss asked me to", "The subway was crowded and",
    "The painting in the hallway", "Our team played against", "The documentary was about",
    "My laptop is", "The guests arrived at seven and", "The mountain trail was",
    "I called the customer service line and", "The new album by the band", "The school bus stopped at",
    "After dinner, we watched", "The weather forecast for tomorrow", "My uncle works as a",
    "The coffee at the office is", "The festival in the town square", "I cleaned the garage and",
    "The car needed new tires, so", "The play at the local theater", "We ordered pizza and",
    "The gym near my office", "My daughter started kindergarten and", "The river behind the house",
    "The election results were", "I planted tomatoes in", "The waiter brought us",
    "My parents visited the museum and", "The radio station played", "The apartment building on Elm Street",
    "The team meeting was moved to", "The cat jumped onto the table and", "I finished the puzzle and",
    "The ferry to the island", "The new policy at work", "My friend recommended a book about",
    "The village has a small", "The hike up the hill took", "The store manager said that",
    "The children played in the yard and", "The town library is", "I watched the news and",
    "The dinner party on Friday", "My phone battery", "The highway was",
    "The students listened to the speaker and", "I took the train to", "The fountain in the park",
    "Our holiday plans for this year", "The restaurant menu", "The movie started at",
    "My sister bought a new car and", "The sunset over the lake", "The hotel lobby was",
    "I went for a walk and", "The cafeteria at school serves", "The neighbors are building",
    "The weekend market sells", "The software engineer explained", "My grandfather told me about",
    "The traffic on the way home", "We visited the zoo and", "The shop owner said",
    "The band played a song about", "The pharmacy closes at", "The kids came home from school and",
    "The art class this week", "The office building across the street", "I made a cup of tea and",
    "The tour guide showed us", "The soccer match on Saturday", "The garden center had",
    "My colleague is moving to", "The weather in the mountains", "The lake was",
    "I opened the letter and", "The bakery sells bread that", "The meeting with the client",
    "The new teacher at school", "We cleaned the house and", "The power went out and",
]
assert len(NEUTRAL150) == 150 and len(set(NEUTRAL150)) == 150


# ------------------------------------------------------------------ self-test (tokenizer level, CPU)
def selftest(tok_ouro=None, tok_ifm=None):
    res = {}

    def fake_enc(t, ans=False):           # char-level, BOS = -1
        ids = [ord(c) for c in t]
        return ids if ans else [-1] + ids
    s = answer_tokens(fake_enc, "68", "78")
    res["answer_tokens_char"] = s == ([ord(" ")], ord("6"), ord("7"))
    res["answer_tokens_prefix_none"] = answer_tokens(fake_enc, "6", "68") is None
    raw = d2_raw()
    res["d2_raw_rules"] = all(10 <= x <= 89 for r in raw for x in r[1:]) and all(
        a + b <= 99 and a2 + b2 <= 99 and (a + b) // 10 != (a2 + b2) // 10 and
        ((op == "op1" and b == b2 and a % 10 == a2 % 10) or (op == "op2" and a == a2 and b % 10 == b2 % 10))
        for op, a, b, a2, b2 in raw)
    res["d2_raw_balance"] = sum(1 for r in raw if r[0] == "op1") == sum(1 for r in raw if r[0] == "op2")
    c2 = d2_candidates(fake_enc)
    p = c2[0]
    res["d2_char_pos_one"] = all(len(q["pos"]) == 1 for q in c2) and p["A"][-1] == ord(" ")
    res["d2_char_answer"] = all(q["tA"] == ord(str(q["sum_a"])[0]) and q["tB"] == ord(str(q["sum_b"])[0]) for q in c2)
    recs = [{"case_id": 1, "rel": "P17", "template": "{} is located in", "subject": "Abc", "answer": "France",
             "para": ["Noise text. {} is in"]},
            {"case_id": 2, "rel": "P17", "template": "{} is located in", "subject": "Abd", "answer": "Spain",
             "para": ["x {} y"]},
            {"case_id": 3, "rel": "P17", "template": "{} is located in", "subject": "Xbc", "answer": "France",
             "para": ["z {} w"]}]

    def word_enc(t, ans=False):             # one token per word, answers single tokens
        ids = [hash(w) % 100003 for w in t.split(" ") if w]
        return ids if ans else [-1] + ids
    cc, st = cf_candidates(word_enc, recs, rels=["P17"])
    res["cf_pairs"] = [(q["a"], q["b"]) for q in cc] == [("Abc", "Abd"), ("Abd", "Xbc")]
    res["cf_pos"] = all(q["pos"] == [1] for q in cc)
    res["cf_para_from_A"] = cc[0]["para"][0][0] == word_enc("Noise text. Abc is in")
    kept, sc = select([dict(q, half="h", rel="r") for q in
                       [{"a": "x", "b": "y"}, {"a": "x", "b": "z"}, {"a": "x", "b": "w"}, {"a": "u", "b": "v"},
                        {"a": "u", "b": "t"}]], lambda q: q["b"] != "v", {("h", "r"): 3})
    res["select_cap_quota"] = [(q["a"], q["b"]) for q in kept] == [("x", "y"), ("x", "z"), ("u", "t")] and sc == 4
    res["neutral150"] = len(NEUTRAL150)
    for name, path in (("ouro", tok_ouro), ("ifm", tok_ifm)):
        if not path:
            continue
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        tk = (PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json")) if name == "ifm"
              else AutoTokenizer.from_pretrained(path))
        bos = tk.convert_tokens_to_ids("<|begin_of_text|>") if name == "ifm" else None

        def enc(t, ans=False, tk=tk, bos=bos):
            ids = tk(t, add_special_tokens=False)["input_ids"]
            return ids if (ans or bos is None) else [bos] + ids
        c2 = d2_candidates(enc)
        p = c2[0]
        dec = lambda ids: [tk.decode([i]) for i in ids]  # noqa: E731
        res["d2_%s_n" % name] = len(c2)
        res["d2_%s_example" % name] = {"A": dec(p["A"]), "B": dec(p["B"]), "pos": p["pos"], "tA": dec([p["tA"]]),
                                       "tB": dec([p["tB"]]), "para0": dec(p["para"][0][0])}
        res["d2_%s_pos_len" % name] = sorted(set(len(q["pos"]) for q in c2))
        res["d2_%s_suffix" % name] = sorted(set(tuple(dec(q["suffix"])) for q in c2))
        d1 = d1_candidates(enc)
        res["d1_%s_n" % name] = len(d1)
    ok = all(v for k, v in res.items() if k in ("answer_tokens_char", "answer_tokens_prefix_none", "d2_raw_rules",
                                                 "d2_raw_balance", "d2_char_pos_one", "d2_char_answer", "cf_pairs",
                                                 "cf_pos", "cf_para_from_A", "select_cap_quota"))
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tok_ouro", default="")
    ap.add_argument("--tok_ifm", default="")
    ap.add_argument("--prepare_cf", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.tok_ouro, a.tok_ifm) else 1)
    if a.prepare_cf:
        prepare_cf(a.prepare_cf)


if __name__ == "__main__":
    main()
