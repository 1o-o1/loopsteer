"""K16 dataset builders (brief B-K16-DATA, plan loop-signal/PLAN.md). CPU only, no model forward.

Outputs under loop-flow/data/: d4/, d5/ (program-generated), d3/ (Wikidata + RAVEL), one JSONL per dataset per
tokenizer (d4_ouro.jsonl, d4_ifm.jsonl, ...), a split file per dataset (cluster -> dev/test) and
data/k16_data_manifest.json (sha256 of every output, tokenizer md5, counts). Format: notes/k16_data_report.md.

Row fields shared by all datasets: id, dataset, cluster_id, split, kind (T1 source-read | T2 computed), tokenizer,
prompt_A, prompt_B (text, without the answer's leading space), answer_A, answer_B (strings), ids_A, ids_B (model
input ids = enc(prompt) + suffix_ids; IFM has BOS at index 0, Ouro none), suffix_ids (common answer prefix, the
single space token for D4/D5), source_span (token indices into ids, same in A and B, the tokens that differ),
answer_first_token_A/B (ids of the first token after the suffix), para (list of {prompt_A, prompt_B, ids_A, ids_B,
source_span}: same item in other surface forms, for paraphrase vectors).

    python k16_data.py --selftest [--tok_ouro DIR] [--tok_ifm DIR]
    python k16_data.py --build d4,d5 [--tok_ouro DIR] [--tok_ifm DIR] [--out_dir DIR]
    python k16_data.py --build d3 ...                        (see the D3 section)
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from k15_data import answer_tokens, subj_span  # noqa: E402  (k15 conventions, read-only import)

DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
SEED = 16
IFM_TOK_SRC = os.environ.get("K16_TOK_IFM_SRC", "")   # cluster copy: scp source of the IFM tokenizer (build only)
OURO_REPO = "ByteDance/Ouro-1.4B"


# ------------------------------------------------------------------ tokenizers
class Tok(object):
    """name: ouro | ifm. enc(text, ans=False) is the k13/k15 encoder (IFM adds BOS unless ans)."""

    def __init__(self, name, hf, bos, path):
        self.name, self.hf, self.bos, self.path = name, hf, bos, path
        self.md5 = file_md5(os.path.join(path, "tokenizer.json")) if path else None

    def enc(self, text, ans=False):
        ids = self.hf(text, add_special_tokens=False)["input_ids"]
        return list(ids) if (ans or self.bos is None) else [self.bos] + list(ids)

    def enc_spans(self, text, ranges):
        """-> (ids as enc, [token indices overlapping each char range [c0, c1)])."""
        e = self.hf(text, add_special_tokens=False, return_offsets_mapping=True)
        shift = 0 if self.bos is None else 1
        spans = [[i + shift for i, (s, t) in enumerate(e["offset_mapping"]) if s < c1 and t > c0] for c0, c1 in ranges]
        return ([] if self.bos is None else [self.bos]) + list(e["input_ids"]), spans

    def dec(self, ids):
        return [self.hf.decode([i]) for i in ids]


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def find_ouro_tok(path=""):
    """Local dir with Ouro's tokenizer files: given path, else the HF cache snapshot, else a hub download of the
    tokenizer files only."""
    if path:
        return path
    from huggingface_hub.constants import HF_HUB_CACHE
    snaps = sorted(glob.glob(os.path.join(HF_HUB_CACHE, "models--ByteDance--Ouro-1.4B", "snapshots", "*")))
    for s in snaps:
        if os.path.exists(os.path.join(s, "tokenizer.json")):
            return s
    from huggingface_hub import snapshot_download
    return snapshot_download(OURO_REPO, allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"])


def find_ifm_tok(path=""):
    """Local dir with the IFM tokenizer (all 7 P1 dense checkpoints share one tokenizer.json, md5 0457edc7...).
    Else read-only scp of the two tokenizer files from K16_TOK_IFM_SRC into <HF_HOME>/k16_tok_ifm."""
    path = path or os.environ.get("K16_TOK_IFM", "")
    if path:
        return path
    from huggingface_hub.constants import HF_HOME
    dest = os.path.join(HF_HOME, "k16_tok_ifm")
    if not os.path.exists(os.path.join(dest, "tokenizer.json")):
        os.makedirs(dest, exist_ok=True)
        for f in ("tokenizer.json", "tokenizer_config.json"):
            subprocess.check_call(["scp", "-q", "%s/%s" % (IFM_TOK_SRC, f), os.path.join(dest, f)])
    return dest


def load_tokenizers(tok_ouro="", tok_ifm=""):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    po, pi = find_ouro_tok(tok_ouro), find_ifm_tok(tok_ifm)
    ouro = Tok("ouro", AutoTokenizer.from_pretrained(po), None, po)
    hf = PreTrainedTokenizerFast(tokenizer_file=os.path.join(pi, "tokenizer.json"))
    ifm = Tok("ifm", hf, hf.convert_tokens_to_ids("<|begin_of_text|>"), pi)
    return {"ouro": ouro, "ifm": ifm}


# ------------------------------------------------------------------ common
def split_of(cluster_id):
    """dev / test by sha256 of the cluster id (first 8 hex digits, parity)."""
    return "test" if int(hashlib.sha256(cluster_id.encode()).hexdigest()[:8], 16) % 2 else "dev"


class Text(object):
    """Text builder that remembers the char range of tagged pieces."""

    def __init__(self):
        self.s = ""
        self.tags = collections.defaultdict(list)

    def add(self, t, tag=None):
        if tag:
            self.tags[tag].append((len(self.s), len(self.s) + len(t)))
        self.s += t
        return self


def dump_jsonl(path, rows):
    """Atomic write (tmp file, then rename): readers never see a partial file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def to_pairs(rows):
    """JSONL rows -> k15_data pair dicts (A, B, pos, tA, tB, a, b, rel, half, para, suffix) for harnesses built on
    k15_data. para = list of (idsA', idsB'); half = the split."""
    out = []
    for r in rows:
        out.append(dict(r, A=r["ids_A"], B=r["ids_B"], pos=r["source_span"], tA=r["answer_first_token_A"],
                        tB=r["answer_first_token_B"], a=r["answer_A"], b=r["answer_B"], rel=r["kind"], half=r["split"],
                        suffix=r["suffix_ids"], para=[(q["ids_A"], q["ids_B"]) for q in r["para"]]))
    return out


def update_manifest(out_dir, key, entry):
    p = os.path.join(out_dir, "k16_data_manifest.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    m[key] = entry
    json.dump(m, open(p, "w"), indent=1, sort_keys=True)


def encode_pair(tk, tA, tB, ansA, ansB, rngA, rngB):
    """Shared tokenisation + checks for one (A, B) prompt pair. tA/tB: Text objects. rngA/rngB: char ranges
    (list) whose tokens must be the source span. Returns (row_part, reason) with reason None if ok."""
    at = answer_tokens(tk.enc, ansA, ansB)
    if at is None:
        return None, "answer_prefix_or_none"
    suf, ta, tb = at
    if len(suf) != 1 or tk.dec(suf) != [" "]:
        return None, "suffix_not_single_space"
    ia, sa = tk.enc_spans(tA.s, rngA)
    ib, sb = tk.enc_spans(tB.s, rngB)
    ia, ib = ia + suf, ib + suf
    if len(ia) != len(ib):
        return None, "len_mismatch"
    span, span_b = subj_span(ia, ib)
    flat_a = sorted(set(i for s in sa for i in s))
    flat_b = sorted(set(i for s in sb for i in s))
    if span != span_b or span != flat_a or span != flat_b:
        return None, "span_mismatch"
    return {"ids_A": ia, "ids_B": ib, "suffix_ids": suf, "source_span": span, "answer_first_token_A": ta,
            "answer_first_token_B": tb, "answer_first_token_A_str": tk.dec([ta])[0],
            "answer_first_token_B_str": tk.dec([tb])[0]}, None


def span_is(tk, part, strA, strB):
    """The source span tokens decode to exactly strA in A and strB in B (lists of token strings)."""
    sp = part["source_span"]
    return (tk.dec([part["ids_A"][i] for i in sp]) == strA and tk.dec([part["ids_B"][i] for i in sp]) == strB)


# ------------------------------------------------------------------ D4 code variable binding
D4_SEED = SEED * 100 + 4
D4_NAMES = list("abcdefghkmnpqrstuvwxyz")
D4_MAIN = "A"
# fixed 2-shot demos: (variable names, values, query indices); answers are computed by the program
D4_DEMOS = {"T1": [(["m", "n", "p"], [7, 2, 5], [1]), (["x", "y", "z"], [3, 8, 1], [0])],
            "T2": [(["m", "n", "p"], [1, 5, 3], [0, 1]), (["x", "y", "z"], [3, 2, 8], [1, 0])]}
D4_N_TEMPLATES = {"T1": 300, "T2": 420}
D4_PAIRS_PER_TEMPLATE = 3


def d4_answer(vals, query):
    return sum(vals[i] for i in query)


def d4_render(bld, fmt, names, vals, query, final):
    """Append one snippet to bld. fmt A: output on the next line after '#'; B: after 'Output:'; C: inline '#'.
    final: no output appended; value tokens tagged v0..v2, query variable names tagged q0, q1."""
    for i, (n, v) in enumerate(zip(names, vals)):
        bld.add("%s = " % n)
        bld.add(str(v), ("v%d" % i) if final else None)
        bld.add("\n")
    bld.add("print(")
    for k, qi in enumerate(query):
        if k:
            bld.add(" + ")
        bld.add(names[qi], ("q%d" % k) if final else None)
    bld.add(")")
    bld.add({"A": "\n#", "B": "\nOutput:", "C": " #"}[fmt])
    if not final:
        bld.add(" %d" % d4_answer(vals, query))


def d4_prompt(fmt, kind, names, vals, query):
    b = Text()
    sep = "\n" if fmt == "C" else "\n\n"
    for dn, dv, dq in D4_DEMOS[kind]:
        d4_render(b, fmt, dn, dv, dq, final=False)
        b.add(sep)
    d4_render(b, fmt, names, vals, query, final=True)
    return b


def d4_templates(kind, n, seed=D4_SEED):
    """Template dicts: names (3), fixed values (None at the source slot), query indices, src index, value pairs.
    T1: print(v), the queried variable's value is the source. T2: print(u + v), the source is the value of one of
    the two operands; both sums are <= 9 (single digit answer). All three values differ within a prompt."""
    rng = random.Random(seed + (0 if kind == "T1" else 1))
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 100 * n:
        tries += 1
        names = rng.sample(D4_NAMES, 3)
        if kind == "T1":
            qi = len(out) % 3
            query, src = [qi], qi
            others = rng.sample(range(1, 10), 2)
            fixed = {i: others[k] for k, i in enumerate([j for j in range(3) if j != qi])}
            cand = [x for x in range(1, 10) if x not in others]
        else:
            i, j = rng.sample(range(3), 2)
            d = [k for k in range(3) if k not in (i, j)][0]
            query = [i, j]
            src = query[len(out) % 2]
            oth = query[1 - len(out) % 2]
            u = rng.randint(1, 6)
            dv = rng.choice([x for x in range(1, 10) if x != u])
            fixed = {oth: u, d: dv}
            cand = [x for x in range(1, 10) if x not in (u, dv) and x + u <= 9]
        npair = min(D4_PAIRS_PER_TEMPLATE, len(cand) // 2)
        if npair < 1:
            continue
        vals = rng.sample(cand, 2 * npair)
        key = (tuple(names), tuple(sorted(fixed.items())), tuple(query), src)
        if key in seen:
            continue
        seen.add(key)
        out.append({"names": names, "fixed": fixed, "query": query, "src": src,
                    "pairs": [(vals[2 * k], vals[2 * k + 1]) for k in range(npair)]})
    return out


def d4_items(kind, n):
    """Tokenizer-independent item specs."""
    items = []
    for ti, t in enumerate(d4_templates(kind, n)):
        cid = "d4-%s-t%04d" % (kind, ti)
        for pi, (xa, xb) in enumerate(t["pairs"]):
            va, vb = [t["fixed"].get(i) for i in range(3)], [t["fixed"].get(i) for i in range(3)]
            va[t["src"]], vb[t["src"]] = xa, xb
            items.append({"id": "%s-p%d" % (cid, pi), "cluster_id": cid, "kind": kind, "names": t["names"],
                          "query": t["query"], "src": t["src"], "vals_A": va, "vals_B": vb})
    return items


def d4_row(tk, it):
    kind, names, query, src = it["kind"], it["names"], it["query"], it["src"]
    ansA, ansB = d4_answer(it["vals_A"], query), d4_answer(it["vals_B"], query)
    bA = d4_prompt(D4_MAIN, kind, names, it["vals_A"], query)
    bB = d4_prompt(D4_MAIN, kind, names, it["vals_B"], query)
    part, why = encode_pair(tk, bA, bB, str(ansA), str(ansB), [bA.tags["v%d" % src][0]], [bB.tags["v%d" % src][0]])
    if part is None:
        return None, why
    if not span_is(tk, part, [str(it["vals_A"][src])], [str(it["vals_B"][src])]):
        return None, "value_not_single_token"
    _, spA = tk.enc_spans(bA.s, [bA.tags["v%d" % i][0] for i in range(3)] + bA.tags["q0"] + bA.tags["q1"])
    if any(len(s) != 1 for s in spA[:3]):
        return None, "value_not_single_token"
    para = []
    for f in [f for f in "ABC" if f != D4_MAIN]:
        pA, pB = (d4_prompt(f, kind, names, it["vals_A"], query), d4_prompt(f, kind, names, it["vals_B"], query))
        pp, why = encode_pair(tk, pA, pB, str(ansA), str(ansB), [pA.tags["v%d" % src][0]], [pB.tags["v%d" % src][0]])
        if pp is None:
            return None, "para_" + why
        if pp["suffix_ids"] != part["suffix_ids"]:
            return None, "para_suffix_differs"
        para.append({"format": f, "prompt_A": pA.s, "prompt_B": pB.s, "ids_A": pp["ids_A"], "ids_B": pp["ids_B"],
                     "source_span": pp["source_span"]})
    qspan = sorted(set(i for s in spA[3:] for i in s))
    row = {"id": it["id"], "dataset": "d4", "cluster_id": it["cluster_id"], "split": split_of(it["cluster_id"]),
           "kind": kind, "tokenizer": tk.name, "prompt_A": bA.s, "prompt_B": bB.s, "answer_A": str(ansA),
           "answer_B": str(ansB)}
    row.update(part)
    row.update({"format": D4_MAIN, "names": names, "vals_A": it["vals_A"], "vals_B": it["vals_B"],
                "query_vars": [names[i] for i in query], "src_var": names[src],
                "src_operand": "only" if kind == "T1" else ("first" if src == query[0] else "second"),
                "value_spans": [s[0] for s in spA[:3]], "query_span": qspan, "last_pos": len(part["ids_A"]) - 1,
                "para": para})
    return row, None


# ------------------------------------------------------------------ D5 two-step math
D5_SEED = SEED * 100 + 5
D5_DEMOS = [(2, 5, 3), (4, 1, 2)]       # (a, b, m): x = a + b, y = x * m
D5_N_TEMPLATES = 400
D5_PAIRS_PER_TEMPLATE = 3
D5_MULT = [2, 3, 4]


def d5_answer(a, b, m):
    return (a + b) * m


def d5_prompt(fmt, vx, vy, a, b, m):
    """P0 main: 2-shot 'x = a + b. y = x * m. y = N'. P1: zero-shot question. P2: 2-shot Q/A.
    Tags: opa, opb (operand values), xref (the intermediate's variable inside the y clause), xclose ('.' that ends
    the x clause; P0 only)."""
    t = Text()
    if fmt == "P0":
        for da, db, dm in D5_DEMOS:
            t.add("%s = %d + %d. %s = %s * %d. %s = %d\n" % (vx, da, db, vy, vx, dm, vy, d5_answer(da, db, dm)))
        t.add("%s = " % vx).add(str(a), "opa").add(" + ").add(str(b), "opb").add(".", "xclose")
        t.add(" %s = " % vy).add(vx, "xref").add(" * %d. %s =" % (m, vy))
    elif fmt == "P1":
        t.add("Let %s = " % vx).add(str(a), "opa").add(" + ").add(str(b), "opb")
        t.add(" and %s = " % vy).add(vx, "xref").add(" * %d. What is %s? Answer:" % (m, vy))
    else:
        for da, db, dm in D5_DEMOS:
            t.add("Q: %s = %d + %d and %s = %s * %d. What is %s? A: %d\n" % (vx, da, db, vy, vx, dm, vy,
                                                                          d5_answer(da, db, dm)))
        t.add("Q: %s = " % vx).add(str(a), "opa").add(" + ").add(str(b), "opb")
        t.add(" and %s = " % vy).add(vx, "xref").add(" * %d. What is %s? A:" % (m, vy))
    return t


def d5_pair_ok(xa, xb, u, m):
    ya, yb = d5_answer(xa, u, m), d5_answer(xb, u, m)
    return xa != xb and len(str(ya)) == len(str(yb)) and str(ya)[0] != str(yb)[0]


def d5_items(n=D5_N_TEMPLATES, seed=D5_SEED):
    """Template = (variable names, multiplier, which operand is edited, the fixed operand). Pairs differ in the
    edited operand; the two answers have equal digit count and a different leading digit."""
    rng = random.Random(seed)
    items, seen, ti = [], set(), 0
    tries = 0
    while ti < n and tries < 100 * n:
        tries += 1
        vx, vy = rng.sample(D4_NAMES, 2)
        m, u, src = rng.choice(D5_MULT), rng.randint(1, 9), ("a", "b")[ti % 2]
        key = (vx, vy, m, u, src)
        if key in seen:
            continue
        cand = [(xa, xb) for xa in range(1, 10) for xb in range(xa + 1, 10) if d5_pair_ok(xa, xb, u, m)]
        rng.shuffle(cand)
        chosen, used = [], set()
        for xa, xb in cand:
            if xa in used or xb in used:
                continue
            chosen.append((xa, xb))
            used |= {xa, xb}
            if len(chosen) == D5_PAIRS_PER_TEMPLATE:
                break
        if not chosen:
            continue
        chosen = [(xb, xa) if rng.random() < 0.5 else (xa, xb) for xa, xb in chosen]
        seen.add(key)
        cid = "d5-T2-t%04d" % ti
        for pi, (xa, xb) in enumerate(chosen):
            ops_a = (xa, u) if src == "a" else (u, xa)
            ops_b = (xb, u) if src == "a" else (u, xb)
            items.append({"id": "%s-p%d" % (cid, pi), "cluster_id": cid, "kind": "T2", "vx": vx, "vy": vy, "m": m,
                          "src": src, "ops_A": list(ops_a), "ops_B": list(ops_b)})
        ti += 1
    return items


def d5_row(tk, it):
    vx, vy, m, src = it["vx"], it["vy"], it["m"], it["src"]
    (a, b), (a2, b2) = it["ops_A"], it["ops_B"]
    ansA, ansB = d5_answer(a, b, m), d5_answer(a2, b2, m)
    tag = "opa" if src == "a" else "opb"
    pA, pB = d5_prompt("P0", vx, vy, a, b, m), d5_prompt("P0", vx, vy, a2, b2, m)
    part, why = encode_pair(tk, pA, pB, str(ansA), str(ansB), [pA.tags[tag][0]], [pB.tags[tag][0]])
    if part is None:
        return None, why
    if not span_is(tk, part, [str((a, b)[src == "b"])], [str((a2, b2)[src == "b"])]):
        return None, "value_not_single_token"
    _, sp = tk.enc_spans(pA.s, [pA.tags["opa"][0], pA.tags["opb"][0], pA.tags["xclose"][0], pA.tags["xref"][0]])
    if any(len(s) != 1 for s in sp[:4]) or tk.dec([part["ids_A"][sp[2][0]]]) != ["."]:
        return None, "span_not_single_token"
    para = []
    for f in ("P1", "P2"):
        qA, qB = d5_prompt(f, vx, vy, a, b, m), d5_prompt(f, vx, vy, a2, b2, m)
        pp, why = encode_pair(tk, qA, qB, str(ansA), str(ansB), [qA.tags[tag][0]], [qB.tags[tag][0]])
        if pp is None:
            return None, "para_" + why
        para.append({"format": f, "prompt_A": qA.s, "prompt_B": qB.s, "ids_A": pp["ids_A"], "ids_B": pp["ids_B"],
                     "source_span": pp["source_span"]})
    row = {"id": it["id"], "dataset": "d5", "cluster_id": it["cluster_id"], "split": split_of(it["cluster_id"]),
           "kind": "T2", "tokenizer": tk.name, "prompt_A": pA.s, "prompt_B": pB.s, "answer_A": str(ansA),
           "answer_B": str(ansB)}
    row.update(part)
    row.update({"format": "P0", "var_x": vx, "var_y": vy, "mult": m, "src_operand": src, "operands_A": [a, b],
                "operands_B": [a2, b2], "inter_A": a + b, "inter_B": a2 + b2,
                "operand_spans": {"a": sp[0], "b": sp[1]}, "inter_close_pos": sp[2], "inter_ref_span": sp[3],
                "last_pos": len(part["ids_A"]) - 1, "para": para})
    return row, None


# ------------------------------------------------------------------ build driver for D4 / D5
def build_rows(tokenizers, items, row_fn):
    """-> {tok: (rows, drop_counter)}"""
    res = {}
    for name, tk in tokenizers.items():
        rows, drops = [], collections.Counter()
        for it in items:
            r, why = row_fn(tk, it)
            if r is None:
                drops[why] += 1
            else:
                rows.append(r)
        rows.sort(key=lambda r: r["id"])
        res[name] = (rows, drops)
    return res


def describe(rows, drops, n_gen):
    cl = collections.defaultdict(set)
    for r in rows:
        cl[r["split"]].add(r["cluster_id"])
    by_kind = collections.Counter(r["kind"] for r in rows)
    by_split = collections.Counter(r["split"] for r in rows)
    ft = collections.Counter((r["answer_first_token_A"], r["answer_first_token_B"]) for r in rows)
    return {"generated": n_gen, "kept": len(rows), "dropped": dict(drops), "by_kind": dict(by_kind),
            "items_by_split": dict(by_split), "clusters_by_split": {k: len(v) for k, v in sorted(cl.items())},
            "first_token_differs_in_pair": all(r["answer_first_token_A"] != r["answer_first_token_B"] for r in rows),
            "equal_len": all(len(r["ids_A"]) == len(r["ids_B"]) for r in rows),
            "distinct_first_token_pairs": len(ft)}


def write_dataset(out_dir, ds, items, row_fn, tokenizers, extra_split_meta=None):
    sub = os.path.join(out_dir, ds)
    res = build_rows(tokenizers, items, row_fn)
    clusters = sorted(set(it["cluster_id"] for it in items))
    split = {c: split_of(c) for c in clusters}
    split_path = os.path.join(sub, "%s_split.json" % ds)
    os.makedirs(sub, exist_ok=True)
    json.dump({"rule": "dev if int(sha256(cluster_id)[:8], 16) is even else test", "clusters": split},
              open(split_path, "w"), indent=0, sort_keys=True)
    entry = {"seed": SEED, "generated_items": len(items), "generated_clusters": len(clusters),
             "split_file": split_path, "split_sha256": sha256_file(split_path), "files": {}, "per_tokenizer": {},
             "tokenizer_md5": {n: t.md5 for n, t in tokenizers.items()}}
    for name, (rows, drops) in res.items():
        p = os.path.join(sub, "%s_%s.jsonl" % (ds, name))
        dump_jsonl(p, rows)
        entry["files"][name] = {"path": p, "sha256": sha256_file(p), "rows": len(rows)}
        entry["per_tokenizer"][name] = describe(rows, drops, len(items))
    update_manifest(out_dir, ds, entry)
    return entry


def build_d4(out_dir, tokenizers):
    items = d4_items("T1", D4_N_TEMPLATES["T1"]) + d4_items("T2", D4_N_TEMPLATES["T2"])
    return write_dataset(out_dir, "d4", items, d4_row, tokenizers)


def build_d5(out_dir, tokenizers):
    return write_dataset(out_dir, "d5", d5_items(), d5_row, tokenizers)


# ------------------------------------------------------------------ D3 same-subject attributes (Wikidata + RAVEL)
D3_SEED = SEED * 100 + 3
D3_CITIES_PER_COUNTRY = 3
WD_RESPONSE = "wikidata_response.json"          # raw SPARQL response, query in wikidata_query.rq (same folder)
WD_FIELD = {"capital": "capitals", "language": "languages", "continent": "continents", "currency": "currencies"}
WD_PROP = {"capital": "P36", "language": "P37", "continent": "P30", "currency": "P38"}
RAVEL_REPO = "hij/ravel"
RAVEL_ATTR = {"Country": "country", "Continent": "continent", "Language": "language", "Timezone": "timezone"}
# RAVEL name -> Wikidata English label (naming differences only, applied only if the label exists in the response)
RAVEL_COUNTRY_ALIAS = {"China": "People's Republic of China", "Macedonia": "North Macedonia", "Swaziland": "Eswatini",
                       "Bahamas": "The Bahamas", "Netherlands": "Kingdom of the Netherlands",
                       "Denmark": "Kingdom of Denmark", "Micronesia": "Federated States of Micronesia"}
# RAVEL zero-shot templates dropped because the answer format does not fit the stored value
RAVEL_EXCLUDE = {"city: %s, UTC offset:": "answer is a UTC offset, values are IANA ids",
                 "city: %s, timezone: UTC": "prefix fixes a UTC-offset answer format",
                 "The city of %s's government offer services in the following languages:": "list-type answer"}
# authored country templates (text only; every value comes from Wikidata). first: subject at position 0 of the
# prompt (no BOS on Ouro). mid: text precedes the subject. shared: the same prefix 'Country: ' before the subject
# for all attributes, so the subject-position states are identical across the four attribute prompts.
D3_COUNTRY_TEMPLATES = {
    "capital": [("first", "{S}'s capital city is"), ("mid", "The capital of {S} is"),
                ("shared", "Country: {S}\nCapital city:")],
    "language": [("first", "{S}'s official language is"), ("mid", "The official language of {S} is"),
                 ("shared", "Country: {S}\nOfficial language:")],
    "continent": [("first", "{S} is in the continent of"), ("mid", "The country of {S} is in the continent of"),
                  ("shared", "Country: {S}\nContinent:")],
    "currency": [("first", "{S}'s currency is the"), ("mid", "The currency of {S} is the"),
                 ("shared", "Country: {S}\nCurrency:")],
}


def wd_entities(path):
    """Wikidata response -> country entity dicts. vals[attr] = list of English labels, ids[attr] = their QIDs."""
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for r in d["results"]["bindings"]:
        g = lambda k: r[k]["value"] if k in r else ""  # noqa: E731
        qid = r["c"]["value"].rsplit("/", 1)[1]
        vals, ids = {}, {}
        for a, f in WD_FIELD.items():
            pairs = [x.split("::", 1) for x in g(f).split("|") if x]
            ids[a], vals[a] = [p[0] for p in pairs], [p[1] for p in pairs]
        out.append({"etype": "country", "eid": qid, "subject": g("cLabel"), "cluster": qid, "vals": vals, "ids": ids,
                    "un_member_wd": g("un") == "true"})
    out.sort(key=lambda e: e["eid"])
    return out


def ravel_zero_shot(t):
    """True for a template with one %s and no earlier demonstration (no braces, no sentence/clause end before %s)."""
    import re
    pre = t.split("%s")[0]
    return t.count("%s") == 1 and not re.search(r"[{}\[\]]|\. |; |: [A-Z]", pre)


def ravel_type(t):
    return "first" if t.startswith("%s") else ("shared" if t.startswith("city: ") else "mid")


def load_ravel(countries):
    """-> (city entities (capped, unique names), city templates {attr: [(tid, type, text, ravel_split)]}, stats)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def ld(d, s):
        return pd.read_parquet(hf_hub_download(RAVEL_REPO, "%s/%s-00000-of-00001.parquet" % (d, s), repo_type="dataset"))
    ent = pd.concat([ld("city_entity", s).assign(rsplit=s) for s in ("train", "val", "test")], ignore_index=True)
    pr = pd.concat([ld("city_prompt", s).assign(rsplit=s) for s in ("train", "val", "test")], ignore_index=True)
    label_to_q = {c["subject"]: c["eid"] for c in countries}
    cmap = {}
    for name in sorted(ent["Country"].unique()):
        lab = name if name in label_to_q else RAVEL_COUNTRY_ALIAS.get(name)
        cmap[name] = label_to_q.get(lab) if lab else None
    # which attributes the country fixes: all RAVEL entities of the country share one value (>= 2 entities)
    fixed = {}
    for a in ("Continent", "Language", "Timezone"):
        g = ent.groupby("Country")[a].nunique()
        n = ent.groupby("Country").size()
        fixed[a.lower()] = {c: bool(g[c] == 1 and n[c] >= 2) for c in g.index}
    cnt = ent.groupby("City")["City"].transform("size")
    uniq = ent[cnt == 1]
    cities = []
    for country, grp in uniq.groupby("Country"):
        rows = sorted(grp.to_dict("records"), key=lambda r: tuple(int(x) for x in r["ID"].split("-")))
        random.Random("%d|%s" % (D3_SEED, country)).shuffle(rows)
        for r in sorted(rows[:D3_CITIES_PER_COUNTRY], key=lambda r: r["ID"]):
            q = cmap[country]
            vals = {"country": [r["Country"]], "continent": [r["Continent"]],
                    "language": [x.strip() for x in r["Language"].split(",") if x.strip()], "timezone": [r["Timezone"]]}
            cities.append({"etype": "city", "eid": r["ID"], "subject": r["City"], "cluster": q or "ravel:" + country,
                           "vals": vals, "ravel_split": r["rsplit"], "country": country, "country_qid": q,
                           "ripple_attrs": [a for a in ("continent", "language", "timezone") if fixed[a][country]]})
    cities.sort(key=lambda e: (e["cluster"], e["eid"]))
    templ, dropped = {}, {}
    sub = pr[(pr["Source"] == "RAVEL") & pr["Attribute"].isin(list(RAVEL_ATTR))]
    for _, r in sub.iterrows():
        t = r["Template"].strip()
        if not ravel_zero_shot(r["Template"]):
            continue
        if t in RAVEL_EXCLUDE:
            dropped[t] = RAVEL_EXCLUDE[t]
            continue
        templ.setdefault(RAVEL_ATTR[r["Attribute"]], set()).add((t.replace("%s", "{S}"), r["rsplit"]))
    out_t = {}
    for a, ts in templ.items():
        out_t[a] = [("r-%s-%d" % (a, i), ravel_type(t), t, sp) for i, (t, sp) in enumerate(sorted(ts))]
    stats = {"ravel_entities": int(len(ent)), "ravel_unique_names": int((ent.groupby("City").size() == 1).sum()),
             "ravel_countries": int(ent["Country"].nunique()), "ravel_countries_matched_to_wikidata":
                 int(sum(1 for v in cmap.values() if v)), "ravel_countries_unmatched": sorted(k for k, v in cmap.items() if not v),
             "ravel_country_alias_used": {k: v for k, v in RAVEL_COUNTRY_ALIAS.items() if cmap.get(k)},
             "unique_name_cities": int(len(uniq)), "cities_selected": len(cities),
             "country_fixes": {a: {"countries_with_ge2_entities": int((ent.groupby("Country").size() >= 2).sum()),
                                   "countries_where_all_share_one_value": int(sum(fixed[a].values())),
                                   "share_of_entities_in_such_countries": round(float(
                                       ent["Country"].map(fixed[a]).mean()), 3)} for a in fixed},
             "templates_dropped": dropped, "templates_kept": {a: len(v) for a, v in out_t.items()}}
    return cities, out_t, stats


def d3_templates():
    """{etype: {attr: [(tid, type, text with {S}, template provenance)]}}"""
    return {"country": {a: [("c-%s-%s" % (a, ty), ty, t, "authored") for ty, t in v]
                        for a, v in D3_COUNTRY_TEMPLATES.items()}}


def d3_render(text, subject):
    """-> (prompt string, (c0, c1) char range of the subject)"""
    c0 = text.index("{S}")
    return text.replace("{S}", subject), (c0, c0 + len(subject))


def aligned(ia, ib, sa, sb):
    """Diff span of two prompt id lists if they differ only inside the subject spans sa, sb (equal length), else None."""
    if len(ia) != len(ib):
        return None
    da, db = subj_span(ia, ib)
    if not da or da != db or not (set(da) <= set(sa) and set(db) <= set(sb)):
        return None
    return da


def single(ent, attr):
    v = ent["vals"].get(attr, [])
    return v[0] if len(v) == 1 else None


def d3_flags(countries):
    f = {"multi_language": [c["subject"] for c in countries if len(c["vals"]["language"]) > 1],
         "transcontinental": [c["subject"] for c in countries if len(c["vals"]["continent"]) > 1],
         "multi_capital": [c["subject"] for c in countries if len(c["vals"]["capital"]) > 1],
         "multi_currency": [c["subject"] for c in countries if len(c["vals"]["currency"]) > 1],
         "no_capital": [c["subject"] for c in countries if not c["vals"]["capital"]],
         "no_language": [c["subject"] for c in countries if not c["vals"]["language"]],
         "no_currency": [c["subject"] for c in countries if not c["vals"]["currency"]],
         "not_un_member_in_wikidata": [c["subject"] for c in countries if not c["un_member_wd"]]}
    return f


def d3_rows(tk, ents, templ):
    """All D3 rows for one tokenizer + usable-count stats. ents: country + city entities (tokenizer independent)."""
    cache = {}
    pos0, spaced = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for e in ents:
        tl = templ[e["etype"]]
        for a in tl:
            for tid, ttype, text, _src in tl[a]:
                s, rg = d3_render(text, e["subject"])
                ids, sp = tk.enc_spans(s, [rg])
                cache[(e["eid"], tid)] = (ids, sp[0], s)
        pos0[e["etype"]][len(tk.enc_spans(e["subject"], [(0, len(e["subject"]))])[1][0])] += 1
        spaced[e["etype"]][len(tk.enc_spans(" " + e["subject"], [(1, 1 + len(e["subject"]))])[1][0])] += 1
    ft = {}
    for e in ents:
        for a in templ[e["etype"]]:
            v = single(e, a)
            if v is not None:
                ft[(e["eid"], a)] = tk.enc(" " + v, ans=True)[0]
    # strict uniqueness: the value's first token is used by one distinct value label only, within (etype, attribute)
    lab_by_ft = collections.defaultdict(set)
    for e in ents:
        for a in templ[e["etype"]]:
            if (e["eid"], a) in ft:
                lab_by_ft[(e["etype"], a, ft[(e["eid"], a)])].add(single(e, a))
    split = {e["eid"]: split_of(e["cluster"]) for e in ents}
    pools = collections.defaultdict(list)
    for e in ents:
        pools[(e["etype"], split[e["eid"]])].append(e)
    rows, stats = [], collections.defaultdict(collections.Counter)
    ua, us = collections.defaultdict(set), collections.defaultdict(set)
    for e in ents:
        tl = templ[e["etype"]]
        for a in tl:
            st = stats[(e["etype"], a)]
            st["entities"] += 1
            v = single(e, a)
            if v is None:
                st["not_single_valued"] += 1
                continue
            st["single_valued"] += 1
            uniq = len(lab_by_ft[(e["etype"], a, ft[(e["eid"], a)])]) == 1
            st["first_token_unique_in_attr"] += int(uniq)
            tids = [t[0] for t in tl[a]]
            rng = random.Random(int(hashlib.sha256(("%d|%s|%s|%s" % (D3_SEED, tk.name, e["eid"], a)).encode())
                                    .hexdigest()[:12], 16))
            best, best_n = [], -1
            for c in pools[(e["etype"], split[e["eid"]])]:
                if c["eid"] == e["eid"] or (c["eid"], a) not in ft or ft[(c["eid"], a)] == ft[(e["eid"], a)]:
                    continue
                n = sum(aligned(cache[(e["eid"], t)][0], cache[(c["eid"], t)][0], cache[(e["eid"], t)][1],
                                cache[(c["eid"], t)][1]) is not None for t in tids)
                if n > best_n:
                    best, best_n = [c], n
                elif n == best_n:
                    best.append(c)
            if not best:
                st["no_donor"] += 1
                continue
            d = rng.choice(best)
            ua[e["eid"]].add(a)
            if uniq:
                us[e["eid"]].add(a)
            st["donor_found"] += 1
            st["donor_aligned_any_template"] += int(best_n > 0)
            st["donor_aligned_all_templates"] += int(best_n == len(tids))
            al = {t: aligned(cache[(e["eid"], t)][0], cache[(d["eid"], t)][0], cache[(e["eid"], t)][1],
                             cache[(d["eid"], t)][1]) for t in tids}
            others = {a2: single(e, a2) for a2 in tl if a2 != a and single(e, a2) is not None}
            d_others = {a2: single(d, a2) for a2 in tl if a2 != a and single(d, a2) is not None}
            for tid, ttype, text, src in tl[a]:
                ia, sa, pa = cache[(e["eid"], tid)]
                ib, sb, pb = cache[(d["eid"], tid)]
                para = [{"template_id": t2, "template_type": ty2, "prompt_A": cache[(e["eid"], t2)][2],
                         "prompt_B": cache[(d["eid"], t2)][2], "ids_A": cache[(e["eid"], t2)][0],
                         "ids_B": cache[(d["eid"], t2)][0], "source_span": al[t2]}
                        for t2, ty2, _tx, _s in tl[a] if t2 != tid and al[t2] is not None]
                tA, tB = ft[(e["eid"], a)], ft[(d["eid"], a)]
                row = {"id": "d3-%s-%s-%s-%s" % (e["etype"], e["eid"], a, tid), "dataset": "d3",
                       "cluster_id": e["cluster"], "split": split[e["eid"]], "kind": "T1", "tokenizer": tk.name,
                       "prompt_A": pa, "prompt_B": pb, "answer_A": v, "answer_B": single(d, a), "ids_A": ia,
                       "ids_B": ib, "suffix_ids": [], "source_span": al[tid] if al[tid] is not None else sa,
                       "answer_first_token_A": tA,
                       "answer_first_token_B": tB, "answer_first_token_A_str": tk.dec([tA])[0],
                       "answer_first_token_B_str": tk.dec([tB])[0],
                       "etype": e["etype"], "subject": e["subject"], "subject_id": e["eid"], "attribute": a,
                       "template_id": tid, "template_type": ttype, "template": text, "template_source": src,
                       "subject_pos0": ttype == "first" or text.startswith("{S}"), "equal_len": al[tid] is not None,
                       "subject_span": sa, "subject_span_B": sb, "donor_subject": d["subject"], "donor_id": d["eid"],
                       "value_first_token_unique_in_attr": uniq, "other_attrs": others, "donor_other_attrs": d_others,
                       "last_pos": len(ia) - 1, "para": para}
                if e["etype"] == "country":
                    row["flags"] = {k: e["subject"] in v for k, v in d3_flags_cache(ents).items()}
                else:
                    row.update({"country": e["country"], "ripple_attrs": e["ripple_attrs"], "ravel_split": e["ravel_split"],
                                "donor_country": d["country"]})
                rows.append(row)
                st["rows"] += 1
                st["rows_equal_len"] += int(al[tid] is not None)
    sib = {}
    for r in rows:
        sib.setdefault((r["subject_id"], r["attribute"], r["template_type"]), r["id"])
    for r in rows:
        r["siblings"] = {a2: sib[(r["subject_id"], a2, r["template_type"])] for a2 in D3_ATTRS_OF[r["etype"]]
                         if a2 != r["attribute"] and (r["subject_id"], a2, r["template_type"]) in sib}
    for et, na in (("country", 4), ("city", 4)):
        es = [e for e in ents if e["etype"] == et]
        for tag, d_ in (("usable_attrs", ua), ("usable_attrs_first_token_unique", us)):
            cnt = collections.Counter(len(d_[e["eid"]]) for e in es)
            stats[(et, "_" + tag)] = collections.Counter({"entities": len(es), "ge2": sum(v for k, v in cnt.items() if k >= 2),
                                                          "ge3": sum(v for k, v in cnt.items() if k >= 3),
                                                          "all4": cnt.get(4, 0)})
    rows.sort(key=lambda r: r["id"])
    tok_stats = {et: {"subject_tokens_at_pos0": dict(sorted(pos0[et].items())),
                      "subject_tokens_after_space": dict(sorted(spaced[et].items()))} for et in pos0}
    return rows, stats, tok_stats


D3_ATTRS_OF = {"country": ("capital", "language", "continent", "currency"),
               "city": ("country", "continent", "language", "timezone")}
_FLAGS = {}


def d3_flags_cache(ents):
    if "f" not in _FLAGS:
        _FLAGS["f"] = d3_flags([e for e in ents if e["etype"] == "country"])
    return _FLAGS["f"]


def build_d3(out_dir, tokenizers):
    sub = os.path.join(out_dir, "d3")
    countries = wd_entities(os.path.join(sub, WD_RESPONSE))
    cities, city_t, rstats = load_ravel(countries)
    templ = d3_templates()
    templ["city"] = {a: [(tid, ty, t, "ravel:" + sp) for tid, ty, t, sp in v] for a, v in city_t.items()}
    ents = countries + cities
    clusters = collections.defaultdict(lambda: {"countries": 0, "cities": 0})
    for e in ents:
        clusters[e["cluster"]]["countries" if e["etype"] == "country" else "cities"] += 1
    split = {c: dict(v, split=split_of(c)) for c, v in sorted(clusters.items())}
    split_path = os.path.join(sub, "d3_split.json")
    json.dump({"rule": "dev if int(sha256(cluster_id)[:8], 16) is even else test; cluster = country QID with its RAVEL "
                       "cities (ravel:<name> if the country is not a Wikidata sovereign state)", "clusters": split},
              open(split_path, "w"), indent=0, sort_keys=True)
    ent_path = os.path.join(sub, "d3_entities.json")
    json.dump({"countries": countries, "cities": cities, "flags": d3_flags(countries), "ravel": rstats,
               "templates": {et: {a: [list(t) for t in v] for a, v in tl.items()} for et, tl in templ.items()}},
              open(ent_path, "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    entry = {"seed": D3_SEED, "split_file": split_path, "split_sha256": sha256_file(split_path),
             "entities_file": ent_path, "entities_sha256": sha256_file(ent_path),
             "wikidata_response_sha256": sha256_file(os.path.join(sub, WD_RESPONSE)),
             "wikidata_query_sha256": sha256_file(os.path.join(sub, "wikidata_query.rq")),
             "wikidata_rows": len(countries), "ravel": rstats, "flags_counts": {k: len(v) for k, v in
                                                                            d3_flags(countries).items()},
             "clusters": len(split), "clusters_by_split": dict(collections.Counter(v["split"] for v in split.values())),
             "files": {}, "per_tokenizer": {}, "tokenizer_md5": {n: t.md5 for n, t in tokenizers.items()}}
    for name, tk in tokenizers.items():
        rows, stats, tok_stats = d3_rows(tk, ents, templ)
        p = os.path.join(sub, "d3_%s.jsonl" % name)
        dump_jsonl(p, rows)
        entry["files"][name] = {"path": p, "sha256": sha256_file(p), "rows": len(rows)}
        entry["per_tokenizer"][name] = {"usable": {"%s|%s" % k: dict(v) for k, v in sorted(stats.items())},
                                        "subject_tokenisation": tok_stats,
                                        "rows_by_split": dict(collections.Counter(r["split"] for r in rows))}
    update_manifest(out_dir, "d3", entry)
    return entry


# ------------------------------------------------------------------ self-test (no model, no network)
class FakeTok(object):
    """Char-level stand-in with the Tok interface (BOS = -1 when bos=True)."""

    def __init__(self, bos=False, name="fake"):
        self.name, self.bos, self.md5 = name, (-1 if bos else None), None

    def enc(self, text, ans=False):
        ids = [ord(c) for c in text]
        return ids if (ans or self.bos is None) else [self.bos] + ids

    def enc_spans(self, text, ranges):
        shift = 0 if self.bos is None else 1
        return self.enc(text), [[i + shift for i in range(c0, c1)] for c0, c1 in ranges]

    def dec(self, ids):
        return [chr(i) if i >= 0 else "<bos>" for i in ids]


def selftest(tok_ouro="", tok_ifm=""):
    res = {}
    for bos in (False, True):
        tk = FakeTok(bos)
        r4 = [d4_row(tk, it) for it in d4_items("T1", 30) + d4_items("T2", 30)]
        rows = [r for r, _ in r4 if r]
        res["d4_fake_kept_%s" % bos] = (len(rows), len(r4))
        ok = True
        for r in rows:
            va, vb, src = r["vals_A"], r["vals_B"], r["names"].index(r["src_var"])
            qi = [r["names"].index(n) for n in r["query_vars"]]
            diff = [i for i in range(3) if va[i] != vb[i]]
            ok &= diff == [src] and src in qi and len(set(va)) == 3 and len(set(vb)) == 3
            ok &= int(r["answer_A"]) == sum(va[i] for i in qi) and int(r["answer_B"]) == sum(vb[i] for i in qi)
            ok &= r["kind"] == ("T1" if len(qi) == 1 else "T2") and (len(qi) == 1 or int(r["answer_A"]) <= 9 and int(r["answer_B"]) <= 9)
            ok &= len(r["source_span"]) == 1 and r["ids_A"][r["source_span"][0]] == ord(str(va[src]))
            ok &= r["ids_B"][r["source_span"][0]] == ord(str(vb[src]))
            ok &= [i for i in range(len(r["ids_A"])) if r["ids_A"][i] != r["ids_B"][i]] == r["source_span"]
            ok &= r["ids_A"][-1] == ord(" ") and r["suffix_ids"] == [ord(" ")] and r["last_pos"] == len(r["ids_A"]) - 1
            ok &= len(r["para"]) == 2 and all(len(q["source_span"]) == 1 for q in r["para"])
        res["d4_invariants_%s" % bos] = bool(ok)
        r5 = [d5_row(tk, it) for it in d5_items(40)]
        rows5 = [r for r, _ in r5 if r]
        res["d5_fake_kept_%s" % bos] = (len(rows5), len(r5))
        ok = True
        for r in rows5:
            (a, b), (a2, b2), m = r["operands_A"], r["operands_B"], r["mult"]
            ok &= int(r["answer_A"]) == (a + b) * m and int(r["answer_B"]) == (a2 + b2) * m
            ok &= len(r["answer_A"]) == len(r["answer_B"]) and r["answer_A"][0] != r["answer_B"][0]
            ok &= (a == a2) != (b == b2) and r["inter_A"] == a + b
            ok &= r["ids_A"][r["inter_close_pos"][0]] == ord(".") and r["ids_A"][r["inter_ref_span"][0]] == ord(r["var_x"])
            ok &= [i for i in range(len(r["ids_A"])) if r["ids_A"][i] != r["ids_B"][i]] == r["source_span"]
        res["d5_invariants_%s" % bos] = bool(ok)
    res["det_d4"] = json.dumps(d4_items("T1", 20)) == json.dumps(d4_items("T1", 20))
    res["det_d5"] = json.dumps(d5_items(20)) == json.dumps(d5_items(20))
    res["split_example"] = [split_of("Q142"), split_of("d4-T1-t0000"), split_of("d5-T2-t0001")]
    sa, sb = [3, 4, 5, 6], [3, 9, 5, 6]
    res["aligned"] = (aligned(sa, sb, [1, 2], [1, 2]) == [1] and aligned(sa, sb, [2], [2]) is None
                      and aligned([1, 2, 3], [1, 9, 3], [1], [1]) == [1]
                      and aligned([1, 2, 3], [1, 9, 3, 4], [1], [1]) is None)
    res["ravel_zero_shot"] = (ravel_zero_shot("%s is in the country of") and not ravel_zero_shot("Paris is in France. %s is in")
                              and ravel_zero_shot("city: %s, country:") and not ravel_zero_shot('[{"city": "%s"'))
    res["ravel_type"] = [ravel_type("%s is a city"), ravel_type("city: %s, x:"), ravel_type("Time zone in %s is")]
    ok = all(v for k, v in res.items() if k.startswith(("d4_invariants", "d5_invariants", "det_", "aligned", "ravel_zero")))
    ok &= all(v[0] > 0 for k, v in res.items() if k.startswith(("d4_fake_kept", "d5_fake_kept")))
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return ok


# ------------------------------------------------------------------ independent verification of the written files
def eval_d4(txt, fmt):
    """Independent evaluation of a D4 prompt: (value of the final print, all demo outputs correct)."""
    import re
    if fmt == "C":
        lines = txt.split("\n")
        blocks = ["\n".join(lines[i:i + 4]) for i in range(0, len(lines), 4)]
    else:
        blocks = txt.split("\n\n")

    def run(blk):
        env = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(\w) = (\d)$", blk, re.M)}
        return eval(re.search(r"print\((.*?)\)", blk).group(1), {}, env)
    ok = all(run(b) == int(re.search(r"(\d+)\s*$", b).group(1)) for b in blocks[:-1])
    return run(blocks[-1]), ok


def eval_d5(txt, fmt):
    """Independent evaluation of a D5 prompt: (value of y in the final problem, all demo answers correct)."""
    import re
    pat = {"P0": r"(\w) = (\d) \+ (\d)\. (\w) = (\w) \* (\d)\. (\w) =",
           "P1": r"Let (\w) = (\d) \+ (\d) and (\w) = (\w) \* (\d)\. What is (\w)\? Answer:",
           "P2": r"Q: (\w) = (\d) \+ (\d) and (\w) = (\w) \* (\d)\. What is (\w)\? A:"}[fmt]
    lines = txt.split("\n")
    m = re.match(pat + "$", lines[-1])
    got = (int(m.group(2)) + int(m.group(3))) * int(m.group(6)) if m else None
    ok = True
    for ln in lines[:-1]:
        m2 = re.match(pat + r" (\d+)$", ln)
        ok &= bool(m2) and (int(m2.group(2)) + int(m2.group(3))) * int(m2.group(6)) == int(m2.group(8))
    return got, ok


def verify(out_dir, tokenizers):
    """Re-reads the JSONL files and checks them against the raw sources and a fresh tokenisation (does not reuse the
    row builders). Returns a dict of check -> count (rows when all pass)."""
    import re
    res = {}
    for ds in ("d4", "d5", "d3"):
        for name, tk in tokenizers.items():
            rows = load_jsonl(os.path.join(out_dir, ds, "%s_%s.jsonl" % (ds, name)))
            split_file = json.load(open(os.path.join(out_dir, ds, "%s_split.json" % ds)))["clusters"]
            c = collections.Counter()
            for r in rows:
                suf = r["suffix_ids"]
                c["ids_match_fresh_tokenisation"] += int(r["ids_A"] == tk.enc(r["prompt_A"]) + suf
                                                         and r["ids_B"] == tk.enc(r["prompt_B"]) + suf)
                ia, ib, sp = r["ids_A"], r["ids_B"], r["source_span"]
                if len(ia) == len(ib):
                    diff = [i for i in range(len(ia)) if ia[i] != ib[i]]
                    c["diff_positions_in_source_span"] += int(set(diff) <= set(sp) and (ds == "d3" or diff == sp))
                else:
                    c["unequal_length"] += 1
                ta, tb = tk.enc(" " + r["answer_A"], ans=True), tk.enc(" " + r["answer_B"], ans=True)
                k = len(suf)
                c["answer_tokens_consistent"] += int(ta[:k] == suf and tb[:k] == suf and ta[k] == r["answer_first_token_A"]
                                                     and tb[k] == r["answer_first_token_B"]
                                                     and r["answer_first_token_A"] != r["answer_first_token_B"])
                sf = split_file[r["cluster_id"]]
                sf = sf["split"] if isinstance(sf, dict) else sf
                c["split_matches_hash_and_file"] += int(r["split"] == split_of(r["cluster_id"]) == sf)
                c["last_pos_ok"] += int(r["last_pos"] == len(ia) - 1)
                c["bos_ok"] += int((ia[0] == tk.bos) if tk.bos is not None else True)
                pok = True
                for p in r["para"]:
                    pa, pb, ps = p["ids_A"], p["ids_B"], p["source_span"]
                    pok &= pa == tk.enc(p["prompt_A"]) + suf and pb == tk.enc(p["prompt_B"]) + suf and len(pa) == len(pb)
                    pd_ = [i for i in range(len(pa)) if pa[i] != pb[i]]
                    pok &= (set(pd_) <= set(ps)) if ds == "d3" else (pd_ == ps)
                c["para_ids_fresh_and_diff_equals_span"] += int(pok)
                if ds in ("d4", "d5"):
                    texts = [(r["prompt_A"], r["answer_A"], r["format"]), (r["prompt_B"], r["answer_B"], r["format"])]
                    for p in r["para"]:
                        texts += [(p["prompt_A"], r["answer_A"], p["format"]), (p["prompt_B"], r["answer_B"], p["format"])]
                    for txt, ans, fmt in texts:
                        got, demos_ok = (eval_d4 if ds == "d4" else eval_d5)(txt, fmt)
                        c["answer_reevaluated"] += int(got == int(ans))
                        c["demos_reevaluated"] += int(demos_ok)
            n_txt = sum(2 + 2 * len(r["para"]) for r in rows)
            expect = {"answer_reevaluated": n_txt, "demos_reevaluated": n_txt}
            if ds == "d3":
                expect["diff_positions_in_source_span"] = len(rows) - c["unequal_length"]
                c["unequal_length_equals_flag_false"] = int(c["unequal_length"] == sum(1 for r in rows if not r["equal_len"]))
                expect["unequal_length_equals_flag_false"] = 1
            res["%s|%s" % (ds, name)] = dict(c, rows=len(rows), all_pass=all(
                v == expect.get(k, len(rows)) for k, v in c.items() if k != "unequal_length")
                and (ds == "d3" or c["unequal_length"] == 0))
    # D3 values against the raw sources
    wd = json.load(open(os.path.join(out_dir, "d3", WD_RESPONSE), encoding="utf-8"))["results"]["bindings"]
    raw = {}
    for b in wd:
        q = b["c"]["value"].rsplit("/", 1)[1]
        raw[q] = {a: [x.split("::", 1)[1] for x in b.get(f, {}).get("value", "").split("|") if x] for a, f in WD_FIELD.items()}
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rv = pd.concat([pd.read_parquet(hf_hub_download(RAVEL_REPO, "city_entity/%s-00000-of-00001.parquet" % s,
                                                    repo_type="dataset")) for s in ("train", "val", "test")])
    rv = rv.set_index("ID")
    rmap = {"country": "Country", "continent": "Continent", "timezone": "Timezone"}
    for name in tokenizers:
        c = collections.Counter()
        for r in load_jsonl(os.path.join(out_dir, "d3", "d3_%s.jsonl" % name)):
            for ans, sid in ((r["answer_A"], r["subject_id"]), (r["answer_B"], r["donor_id"])):
                if r["etype"] == "country":
                    c["values_equal_raw_source"] += int(raw[sid][r["attribute"]] == [ans])
                elif r["attribute"] == "language":
                    c["values_equal_raw_source"] += int([x.strip() for x in rv.loc[sid, "Language"].split(",")] == [ans])
                else:
                    c["values_equal_raw_source"] += int(rv.loc[sid, rmap[r["attribute"]]] == ans)
            c["subjects_in_prompts"] += int(r["subject"] in r["prompt_A"] and r["donor_subject"] in r["prompt_B"])
            c["rows"] += 1
        c["all_pass"] = c["values_equal_raw_source"] == 2 * c["rows"] and c["subjects_in_prompts"] == c["rows"]
        res["d3_sources|%s" % name] = dict(c)
    return res


def compile_checks(out_dir):
    """checks JSON of the brief from the manifest, the entity file and the verify results -> k16_data_checks.json"""
    m = json.load(open(os.path.join(out_dir, "k16_data_manifest.json")))
    v = json.load(open(os.path.join(out_dir, "k16_data_verify.json")))
    ent = json.load(open(os.path.join(out_dir, "d3", "d3_entities.json"), encoding="utf-8"))
    d3 = m["d3"]

    def items(ds):
        return {t: {k: x[k] for k in ("generated", "kept", "by_kind", "items_by_split", "clusters_by_split", "dropped",
                                      "first_token_differs_in_pair", "equal_len")}
                for t, x in m[ds]["per_tokenizer"].items()}
    out = {"wikidata_rows": d3["wikidata_rows"],
           "ravel_cities": {k: d3["ravel"][k] for k in ("ravel_entities", "ravel_unique_names", "ravel_countries",
                                                       "ravel_countries_matched_to_wikidata", "unique_name_cities",
                                                       "cities_selected")},
           "d3_usable": {t: x["usable"] for t, x in d3["per_tokenizer"].items()},
           "d3_rows": {t: {"rows": d3["files"][t]["rows"], "rows_by_split": d3["per_tokenizer"][t]["rows_by_split"]}
                       for t in d3["files"]},
           "d3_subject_tokenisation": {t: x["subject_tokenisation"] for t, x in d3["per_tokenizer"].items()},
           "d3_country_fixes": d3["ravel"]["country_fixes"],
           "d4_items": items("d4"), "d5_items": items("d5"),
           "split_sha256": {ds: m[ds]["split_sha256"] for ds in ("d3", "d4", "d5")},
           "split_clusters": {"d3": d3["clusters_by_split"], "d4": {t: x["clusters_by_split"] for t, x in m["d4"]["per_tokenizer"].items()},
                              "d5": {t: x["clusters_by_split"] for t, x in m["d5"]["per_tokenizer"].items()}},
           "file_sha256": {ds: {t: f["sha256"] for t, f in m[ds]["files"].items()} for ds in ("d3", "d4", "d5")},
           "tokenizer_md5": d3["tokenizer_md5"],
           "wikidata_sha256": {"response": d3["wikidata_response_sha256"], "query": d3["wikidata_query_sha256"]},
           "flags": {"counts": d3["flags_counts"], "countries": ent["flags"]},
           "ravel_unmatched_countries": d3["ravel"]["ravel_countries_unmatched"],
           "verify_all_pass": {k: x["all_pass"] for k, x in v.items()}}
    json.dump(out, open(os.path.join(out_dir, "k16_data_checks.json"), "w", encoding="utf-8"), indent=1, sort_keys=True,
              ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", default="", help="comma list of d3,d4,d5")
    ap.add_argument("--verify", action="store_true", help="re-read written files, check against raw sources")
    ap.add_argument("--checks", action="store_true", help="compile k16_data_checks.json (after --build and --verify)")
    ap.add_argument("--out_dir", default=DATA_DIR)
    ap.add_argument("--tok_ouro", default="")
    ap.add_argument("--tok_ifm", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.tok_ouro, a.tok_ifm) else 1)
    if a.checks:
        print(json.dumps(compile_checks(a.out_dir))[:500])
    if a.verify:
        r = verify(a.out_dir, load_tokenizers(a.tok_ouro, a.tok_ifm))
        json.dump(r, open(os.path.join(a.out_dir, "k16_data_verify.json"), "w"), indent=1, sort_keys=True)
        print(json.dumps(r, indent=1, sort_keys=True))
    if a.build:
        tks = load_tokenizers(a.tok_ouro, a.tok_ifm)
        for ds in a.build.split(","):
            e = {"d3": build_d3, "d4": build_d4, "d5": build_d5}[ds](a.out_dir, tks)
            print(ds, json.dumps({k: v for k, v in e.items() if k in ("files", "per_tokenizer")})[:3000])


if __name__ == "__main__":
    main()
