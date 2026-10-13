"""RA-29 item files (CPU, laptop, read only on every source). Plan: plan_ra29.md section 3.

    python make_data.py [--root RESEARCH_ROOT] [--datasets aqua csqa arc]

Sources (read only, under --root): the grid's frozen rows and exemplar prompts (loopedcot/prod/tasks/data), the
grid's prompt builder (loopedcot prod.tasks.build_prompts, base branch), and the grid's k = 4 cells at budget 0
(loopedcot/artifacts/cells_<model>_<task>_natural_k4.jsonl, joined on row_idx) for the prompt-length anchor and
the grid's own forced letter. Tokenizers: Ouro-1.4B and Ouro-2.6B base at the pinned revisions, local cache only.

Prompt A ids = enc(build_prompts(tok, task, [row])[0][0]) + the grid's forced suffix ids (the cue) + enc(" (")
(one token), so the next token is the letter. Prompt B = the same with the gold option's text
and one other option's text exchanged (one transposition). The new gold letter is assigned so that it is uniform
over the letters: items are walked in the seeded order of record and each takes, among the letters other than its
gold letter, the one used least so far (ties in a seeded per-item order); if that partner fails a check, the next
least used. Writes data/items_<task>.json, data/prefix_<task>.txt and data/sources.json. Dropped items stay in the
file with their reason.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TASKS = ("aqua", "csqa", "arc", "mmlu", "hellaswag")
# option layout of the grid's rows: "inline" = ' Answer Choices: (a) t1 (b) t2 ...' (lowercase labels),
# "lines" = newline + 'Options:' + newline + '(A) t1' + newline + '(B) t2 ...' (uppercase labels, MMLU and HellaSwag)
TASK_FORMAT = {"aqua": "inline", "csqa": "inline", "arc": "inline", "mmlu": "lines", "hellaswag": "lines"}
REPOS = {"ouro_1_4b_base": ("ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"),
         "ouro_2_6b_base": ("ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd")}
# Other tokenizer families (--tok): repo, revision, BOS id prepended to every prompt (None = no BOS; the grid's own
# prompt-length rule, checked against n_prompt_tokens), the grid's model name and the depth of its cells file.
TOK_VARIANTS = {
    "huginn": {"repo": "tomg-group-umd/huginn-0125", "revision": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
               "bos": None, "grid": "huginn_0125", "k": 32},
    "mcleish": {"repo": "smcleish/Recurrent-Llama-3.2-train-recurrence-32",
                "revision": "a5f6f126e9d9302445346844f1d7d29d111a06eb", "bos": 128000,
                "grid": "mcleish_llama32_r32", "k": 8},
    # Ouro Thinking: the grid's own Thinking prompt (build_prompts chat_template=True: the exemplars and the question
    # in the user turn, the generation prompt, then the forced read-out </think> + cue); per-model tokenizer config
    "think14": {"repo": "ByteDance/Ouro-1.4B-Thinking", "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
                "bos": None, "grid": "ouro_1_4b_think", "k": 4, "chat": True, "trust": True},
    "think26": {"repo": "ByteDance/Ouro-2.6B-Thinking", "revision": "f1edd81e7ac41355db670500ceaf204e0f73af68",
                "bos": None, "grid": "ouro_2_6b_think", "k": 4, "chat": True, "trust": True},
}
CUE = " ("
LETTERS = "abcdef"
ORDER_SEED = 29
SEP_CHOICES = " Answer Choices: "
RE_OPT = re.compile(r"\(([a-f])\) ")
SEP_LINES = "\nOptions:\n"
RE_OPT_LINES = re.compile(r"(?:^|\n)\(([A-F])\) ")
UPPER = "ABCDEF"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def ids_sha(ids):
    return sha256_bytes(json.dumps([int(i) for i in ids]).encode("utf-8"))


def item_seed(task, key):
    return int(hashlib.sha256(("ra29:%s:%s" % (task, key)).encode("utf-8")).hexdigest()[:15], 16)


def build_input(q, texts):
    return q + SEP_CHOICES + " ".join("(%s) %s" % (LETTERS[j], t) for j, t in enumerate(texts))


def parse_input(s, n_opt):
    """'<question> Answer Choices: (a) t1 (b) t2 ...' -> (question, [t1, ...]) or None; the rebuild must equal s."""
    i = s.rfind(SEP_CHOICES)
    if i < 0:
        return None
    q, rest = s[:i], s[i + len(SEP_CHOICES):]
    marks = list(RE_OPT.finditer(rest))
    if len(marks) != n_opt or [m.group(1) for m in marks] != list(LETTERS[:n_opt]) or marks[0].start() != 0:
        return None
    texts = []
    for j, m in enumerate(marks):
        end = marks[j + 1].start() - 1 if j + 1 < n_opt else len(rest)
        texts.append(rest[m.end():end])
    return (q, texts) if build_input(q, texts) == s else None


def build_lines(q, texts):
    return q + SEP_LINES + "\n".join("(%s) %s" % (UPPER[j], t) for j, t in enumerate(texts))


def parse_lines(s, n_opt):
    """'<question>' + SEP_LINES + '(A) t1' newline '(B) t2 ...' -> (question, [t1, ...]) or None; the rebuild must
    equal s."""
    i = s.rfind(SEP_LINES)
    if i < 0:
        return None
    q, rest = s[:i], s[i + len(SEP_LINES):]
    marks = list(RE_OPT_LINES.finditer(rest))
    if len(marks) != n_opt or [m.group(1) for m in marks] != list(UPPER[:n_opt]) or marks[0].start() != 0:
        return None
    texts = [rest[m.end():(marks[j + 1].start() if j + 1 < n_opt else len(rest))] for j, m in enumerate(marks)]
    return (q, texts) if build_lines(q, texts) == s else None


def block_chars_lines(q, texts):
    """Candidate starts of the option block (the newline before '(A)', then '(A)' itself) and its end."""
    c = len(q) + len(SEP_LINES)
    return (c - 1, c), len(build_lines(q, texts))


def fmt_fns(fmt):
    if fmt == "lines":
        return parse_lines, build_lines, block_chars_lines
    return parse_input, build_input, lambda q, t: ((block_chars(q, t)[0],), block_chars(q, t)[1])


def block_chars(q, texts):
    """Character span [c0, c1) of the option block in the row input: from the space before '(a)' (the token
    ' (' carries it) to the end of the last option text."""
    return len(q) + len(SEP_CHOICES) - 1, len(build_input(q, texts))


def prefix_len(enc, text, ids):
    """Token count of text when its ids are a prefix of ids, else None (a token straddles the boundary)."""
    p = enc(text)
    return len(p) if ids[:len(p)] == p else None


def build_item(task, row, head, a_prefix, tail_ids, enc, cells_row, counts, fmt="inline"):
    """One item with its checks; it['drop'] holds the reason when it cannot enter. counts: letter -> times used as
    the new gold letter so far (updated when the item is kept). Letters are stored lowercase for both layouts."""
    parse, build, bchars = fmt_fns(fmt)
    n_opt = len(row["options"])
    key = "%s:%d" % (task, row["idx"])
    it = {"key": key, "idx": row["idx"], "n_opt": n_opt, "drop": None}
    if cells_row is not None:
        it.update({"grid_n_prompt_tokens": cells_row["n_prompt_tokens"], "grid_pred_k4_b0": cells_row["pred"],
                   "grid_correct_k4_b0": bool(cells_row["correct_v2"])})
    parsed = parse(row["input"], n_opt)
    if parsed is None:
        it["drop"] = "parse_fail"
        return it
    q, texts = parsed
    ca = LETTERS.index(row["target"].lower())
    it["lA"] = LETTERS[ca]
    if sum(1 for t in texts if t == texts[ca]) > 1:
        it["drop"] = "duplicate_gold_text"
        return it
    ids_pA = enc(head + row["input"] + a_prefix)
    it["n_prompt_tokens"] = len(ids_pA)
    if cells_row is not None and cells_row["n_prompt_tokens"] != len(ids_pA):
        it["drop"] = "grid_token_count_mismatch"
        return it
    ids_A = ids_pA + list(tail_ids)
    starts, c1 = bchars(q, texts)
    fullA = row["input"] + a_prefix
    c0, L0 = starts[0], None
    for c0 in starts:                                   # the first start that is a token boundary
        L0 = prefix_len(enc, head + fullA[:c0], ids_A)
        if L0 is not None:
            break
    # the block end: the end of the last option text, or one character later when the tokenizer merges that end
    # with the next character (Llama merges '.' and the newline); the first that is a token boundary
    L1 = None
    for c1 in (c1, c1 + 1):
        L1 = prefix_len(enc, head + fullA[:c1], ids_A)
        if L1 is not None:
            break
    if L0 is None or L1 is None or not L0 < L1:
        it["drop"] = "block_boundary_A"
        return it
    others = [j for j in range(n_opt) if j != ca]
    random.Random(item_seed(task, key)).shuffle(others)
    others.sort(key=lambda j: counts.get(LETTERS[j], 0))          # stable: ties keep the seeded order
    it["partner_order"] = [LETTERS[j] for j in others]
    tried = []
    for j in others:
        tB = list(texts)
        tB[ca], tB[j] = texts[j], texts[ca]
        inB = build(q, tB)
        ids_B = enc(head + inB + a_prefix) + list(tail_ids)
        why = None
        if len(ids_B) != len(ids_A):
            why = "length"
        else:
            fullB = inB + a_prefix
            if (prefix_len(enc, head + fullB[:c0], ids_B), prefix_len(enc, head + fullB[:c1], ids_B)) != (L0, L1):
                why = "block_boundary_B"
            elif ids_B[:L0] != ids_A[:L0] or ids_B[L1:] != ids_A[L1:]:
                why = "outside_block_differs"
        if why:
            tried.append([LETTERS[j], why])
            continue
        counts[LETTERS[j]] = counts.get(LETTERS[j], 0) + 1
        it.update({"lB": LETTERS[j], "input_A": row["input"], "input_B": inB, "T": len(ids_A),
                   "r": len(ids_A) - 1, "block": [L0, L1], "ids_A_sha256": ids_sha(ids_A),
                   "ids_B_sha256": ids_sha(ids_B), "first_partner": j == others[0], "tried": tried,
                   "n_block_diff": sum(1 for x, y in zip(ids_A[L0:L1], ids_B[L0:L1]) if x != y)})
        return it
    it["tried"] = tried
    it["drop"] = "no_valid_transposition"
    return it


def load_cells(root, model, task, k=4):
    p = os.path.join(root, "loopedcot", "artifacts", "cells_%s_%s_natural_k%d.jsonl" % (model, task, k))
    with open(p, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    return {r["row_idx"]: r for r in lines[1:] if r.get("B") == 0}, p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    ap.add_argument("--datasets", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--tok", default="ouro", choices=("ouro",) + tuple(sorted(TOK_VARIANTS)),
                    help="tokenizer family; a non-Ouro family writes items_<task>__<tok>.json")
    a = ap.parse_args(argv)
    lc = os.path.join(a.root, "loopedcot")
    if lc not in sys.path:
        sys.path.insert(0, lc)
    from prod.tasks import build_prompts, prefix_keys, rows as task_rows, task_cfg
    from transformers import AutoTokenizer
    V = TOK_VARIANTS.get(a.tok)
    if V is None:
        toks = {m: AutoTokenizer.from_pretrained(rp, revision=rv, trust_remote_code=True, local_files_only=True)
                for m, (rp, rv) in REPOS.items()}
        tok = toks["ouro_1_4b_base"]
        bos = None
    else:
        tok = AutoTokenizer.from_pretrained(V["repo"], revision=V["revision"], local_files_only=True,
                                            trust_remote_code=bool(V.get("trust")))
        toks, bos = None, V["bos"]
    chat = bool(V and V.get("chat"))

    def plain(t):
        return tok(t, add_special_tokens=False)["input_ids"]

    def enc(t):                                       # prompt-level ids: the BOS of the family, then the text
        return ([bos] if bos is not None else []) + plain(t)

    def enc26(t):
        return toks["ouro_2_6b_base"](t, add_special_tokens=False)["input_ids"]
    cue = plain(CUE)
    assert len(cue) == 1, cue
    os.makedirs(DATA, exist_ok=True)
    src_sha, out_sha = {}, {}
    for task in a.datasets:
        c = task_cfg(task)
        fmt = TASK_FORMAT[task]
        labels = {l: (l.upper() if fmt == "lines" else l) for l in LETTERS}
        letter_ids = {l: plain(labels[l]) for l in LETTERS}
        assert all(len(v) == 1 for v in letter_ids.values()), letter_ids
        rws = task_rows(task)
        _, suffix_ids, _, _, extra = build_prompts(tok, task, rws[:1], chat_template=chat)
        tail_ids = list(suffix_ids) + cue
        if V is None:
            cells = {m: load_cells(a.root, m, task) for m in REPOS}
            grid_main, grid_26 = "ouro_1_4b_base", "ouro_2_6b_base"
        else:
            cells = {V["grid"]: load_cells(a.root, V["grid"], task, V["k"])}
            grid_main, grid_26 = V["grid"], None
        for _, p in cells.values():
            src_sha[os.path.relpath(p, a.root).replace(os.sep, "/")] = sha256_file(p)
        per_row = bool(c.get("per_row_prefix"))
        pfiles = ["prompt_%s.txt" % k for k in prefix_keys(task)] if per_row else [c["prompt_file"]]
        for fn in [c["rows_file"]] + pfiles:
            p = os.path.join(lc, "prod", "tasks", "data", fn)
            src_sha[os.path.relpath(p, a.root).replace(os.sep, "/")] = sha256_file(p)
        heads, hkey = {}, {}
        ap = None if chat else c["a_prefix"]              # the text after the row input (chat: the template tail)
        for row in rws:
            pr = build_prompts(tok, task, [row], chat_template=chat)[0][0]
            if chat:
                i = pr.rfind(row["input"])
                h, t_ = pr[:i], pr[i + len(row["input"]):]
                assert ap is None or ap == t_, "template tail differs between rows (task %s)" % task
                ap = t_
            else:
                h = pr[:len(pr) - len(row["input"]) - len(c["a_prefix"])]
            assert h + row["input"] + ap == pr
            k = row.get("prefix_key") if per_row else task
            assert heads.setdefault(k, h) == h, "exemplar head differs inside one key (task %s)" % task
            hkey[row["idx"]] = k
        order = sorted(range(len(rws)), key=lambda i: "%s:%d" % (task, rws[i]["idx"]))
        random.Random(ORDER_SEED).shuffle(order)
        counts = {}
        items = [None] * len(rws)
        for rank, i in enumerate(order):
            row = rws[i]
            head = heads[hkey[row["idx"]]]
            it = build_item(task, row, head, ap, tail_ids, enc, cells[grid_main][0].get(row["idx"]),
                            counts, fmt)
            it["order"] = rank
            if per_row:
                it["head"] = hkey[row["idx"]]
            g26 = cells[grid_26][0].get(row["idx"]) if grid_26 else None
            it["grid26_correct_k4_b0"] = None if g26 is None else bool(g26["correct_v2"])
            if it["drop"] is None and V is None:
                for t in (head + it["input_A"] + c["a_prefix"], head + it["input_B"] + c["a_prefix"]):
                    if enc26(t) != enc(t):
                        it["drop"] = "tokenizer_2_6b_differs"
                        counts[it["lB"]] -= 1
                        break
            items[i] = it
        drops = {}
        for it in items:
            if it["drop"]:
                drops[it["drop"]] = drops.get(it["drop"], 0) + 1
        kept = [it for it in items if not it["drop"]]
        meta = {"task": task, "n_rows": len(rws), "n_built": len(kept), "drops": drops,
                "a_prefix": ap,
                "suffix_text": extra["suffix_text"], "suffix_ids": list(suffix_ids), "cue": CUE, "cue_ids": cue,
                "letter_ids": {l: v[0] for l, v in letter_ids.items()}, "order_seed": ORDER_SEED,
                "lA_counts": {l: sum(1 for it in kept if it["lA"] == l) for l in LETTERS[:5]},
                "lB_counts": {l: sum(1 for it in kept if it["lB"] == l) for l in LETTERS[:5]},
                "n_first_partner": sum(1 for it in kept if it["first_partner"]),
                "grid_k4_b0_correct_A_built": sum(1 for it in kept if it.get("grid_correct_k4_b0")),
                "grid26_k4_b0_correct_A_built": sum(1 for it in kept if it.get("grid26_correct_k4_b0")),
                "rule": "one transposition of the gold option with the least-used other letter (seeded ties); "
                        "T, the block boundaries and every id outside the block unchanged"}
        if fmt == "lines":
            meta["labels"] = labels
        if V is not None:
            meta.update({"tok": a.tok, "bos_id": bos, "grid_model": V["grid"], "grid_k": V["k"],
                         "tokenizer": {"repo": V["repo"], "revision": V["revision"]}})
        if chat:
            pre = list(suffix_ids[:len(suffix_ids) - len(plain(extra["suffix_text"]))])
            assert pre == plain("</think>") == [extra["end_think_id"]], (pre, extra.get("end_think_id"))
            meta.update({"chat_template": True, "suffix_pre_text": "</think>",
                         "head_file": "prefix_%s__%s.txt" % (task, a.tok)})
        if per_row:
            meta["heads"] = {k: sha256_bytes(h.encode("utf-8")) for k, h in sorted(heads.items())}
            hfiles = {k: "prefix_%s__%s.txt" % (task, k) for k in heads}
        else:
            meta["head_sha256"] = sha256_bytes(heads[task].encode("utf-8"))
            hfiles = {task: meta.get("head_file", "prefix_%s.txt" % task)}
        for k, fn in hfiles.items():
            with open(os.path.join(DATA, fn), "w", encoding="utf-8", newline="") as f:
                f.write(heads[k])
            out_sha[fn] = sha256_file(os.path.join(DATA, fn))
        iname = "items_%s.json" % task if V is None else "items_%s__%s.json" % (task, a.tok)
        p = os.path.join(DATA, iname)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"meta": meta, "items": items}, f, indent=0, sort_keys=True)
        out_sha[iname] = sha256_file(p)
        print(json.dumps({k: meta[k] for k in ("task", "n_rows", "n_built", "drops", "lA_counts", "lB_counts",
                                               "n_first_partner", "grid_k4_b0_correct_A_built",
                                               "grid26_k4_b0_correct_A_built")}))
    sp = os.path.join(DATA, "sources.json")
    old = json.load(open(sp)) if os.path.exists(sp) else {"outputs_sha256": {}, "sources_sha256": {}}
    old["outputs_sha256"].update(out_sha)
    old["sources_sha256"].update(src_sha)
    old["tokenizers"] = {m: {"repo": rp, "revision": rv} for m, (rp, rv) in REPOS.items()}
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(old, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
