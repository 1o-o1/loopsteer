"""PP2 tasks: loaders, prompts, suffixes, stop rules, parsers, the calibration/evaluation split.

Every prompt, suffix, stop string, own-answer marker and parser here is copied verbatim from the
spike that measured it. Sources, per task:

  gsm8k     s9_grid/s9a/scripts/s9a_common.py  (4-shot from openai/gsm8k train rows 0-3,
            "Question: %s\\nAnswer: %s\\n\\n"; suffix "\\nFinal Answer:"; stops
            ["\\n\\nQuestion:", "Final Answer:"]; own marker "####"; 12 answer tokens)
  math500   s13_box_grid/scripts/s13_common.py (4-shot "Problem/Solution/Final Answer" prefix,
            FROZEN from s13_box_grid/artifacts/math500_prompt.json; suffix "\\nFinal Answer:";
            stops ["\\n\\nProblem:", "Final Answer:"]; own marker "\\boxed"; 32 answer tokens;
            last_boxed + math_verify)
  svamp     s28_transfer_tasks (S9a's GSM8K prompt; base suffix " Final Answer:" -- with a newline
            the base model writes " The final answer is \\boxed{" and the 8 answer tokens run out,
            measured in S28's smoke: forced parse 0.70 against 1.00)
  aqua      s28_transfer_tasks (Wei et al. layout, suffix "\\nThe answer is")
  csqa      s28_transfer_tasks (Wei et al. layout, suffix "\\nSo the answer is")
  arc       s28_transfer_tasks taskpack (three exemplars in the BBH style, written for this package
            and not from a published prompt set, as data/README_prompts.md records; suffix
            "\\nSo the answer is")
  bbh_*     s26_bbh/scripts/s26_common.py (official BBH cot prompt with the canary line and the
            "-----" separator dropped; "\\n\\nQ: ... \\nA: Let's think step by step.";
            suffix "\\nSo the answer is"; stops ["\\n\\nQ:", "\\nQ:"]; letter A-F or yes/no)

The frozen row files and prompt files live in `data/`; `data/hashes.json` records the sha256 of each
one and `freeze.py` writes them. No loader touches the network.
"""
import json
import os
import re

import numpy as np

from .. import config as cfgmod
from ..common import DATA, N_CAL, SPLIT_SEED, load_json

LETTERS = "ABCDE"

# Ten evaluation sets. Two structural changes the pooled sets force, both used by BBH and by MMLU:
#   * PER-ROW PROMPT PREFIX. A row may carry `prefix_key`; its exemplar block is
#     `data/prompt_<prefix_key>.txt`. BBH is ONE task slot of 2,500 rows over ten subtasks, each
#     with its own official 3-shot CoT prompt; MMLU is 2,000 rows over 57 subjects grouped into four
#     exemplar files.
#   * PER-ROW ANSWER KIND. A row may carry `kind`, overriding the task's. Pooled BBH mixes letter,
#     yes/no, boolean (`boolean_expressions`) and free-form (`word_sorting`) answers, and a parser
#     chosen per task would be wrong for six of its ten subtasks.
# `subtask` (BBH, for the appendix) and `subject` (MMLU) are carried on the row and stored per cell.

# ------------------------------------------------------------------ registry
# kind      : numeric | letter | yesno | math
# n_answer  : forced read-out length in tokens (the spike's value)
# q_prefix / a_prefix : the question layout inside the prompt
# suffix / suffix_think : the forced read-out suffix for a base / chat-template checkpoint
# stops     : base-model stop strings
# own_marker: the string after which the model's own answer is read inside the cut trace
# n_full    : the full-split N of record
TASKS = {
    "gsm8k": {
        "kind": "numeric", "n_answer": 12, "n_full": cfgmod.DATASETS["gsm8k"],
        "prompt_file": "prompt_gsm8k.txt", "rows_file": "rows_gsm8k.jsonl",
        "sep": "", "q_prefix": "Question: ", "a_prefix": "\nAnswer:",
        "suffix": "\nFinal Answer:", "suffix_think": "\n\nFinal Answer:",
        "stops": ["\n\nQuestion:", "Final Answer:"], "own_marker": "####",
        "chance": 0.0, "forced": True,
    },
    "math500": {
        "kind": "math", "n_answer": 32, "n_full": cfgmod.DATASETS["math500"],
        "prompt_file": "prompt_math500.txt", "rows_file": "rows_math500.jsonl",
        "sep": "", "q_prefix": "Problem: ", "a_prefix": "\nSolution:",
        "suffix": "\nFinal Answer:", "suffix_think": "\n\nFinal Answer:",
        "stops": ["\n\nProblem:", "Final Answer:"], "own_marker": "\\boxed",
        "chance": 0.0, "forced": True,
    },
    "svamp": {
        # SVAMP is sometimes quoted as "test 1000". ChilleD/SVAMP's TEST split is 300 rows; the
        # 1000 is the whole dataset (700 train + 300 test). N of record = the full test split, 300,
        # which is exactly what S28 measured and what gate G1 validates.
        "kind": "numeric", "n_answer": 8, "n_full": cfgmod.DATASETS["svamp"],
        "prompt_file": "prompt_svamp.txt", "rows_file": "rows_svamp.jsonl",
        "sep": "", "q_prefix": "Question: ", "a_prefix": "\nAnswer:",
        "suffix": " Final Answer:", "suffix_think": "\n\nFinal Answer:",
        "stops": ["\n\nQuestion:", "Final Answer:"], "own_marker": "####",
        "chance": 0.0, "forced": True,
    },
    "aqua": {
        "kind": "letter", "n_answer": 8, "n_full": cfgmod.DATASETS["aqua"],
        "prompt_file": "prompt_aqua.txt", "rows_file": "rows_aqua.jsonl",
        "sep": "\n\n", "q_prefix": "Q: ", "a_prefix": "\nA:",
        "suffix": "\nThe answer is", "suffix_think": "\nThe answer is",
        "stops": ["\n\nQ:", "\nQ:"], "own_marker": "The answer is",
        "chance": 0.2, "forced": True,
    },
    "csqa": {
        "kind": "letter", "n_answer": 8, "n_full": cfgmod.DATASETS["csqa"],
        "prompt_file": "prompt_csqa.txt", "rows_file": "rows_csqa.jsonl",
        "sep": "\n\n", "q_prefix": "Q: ", "a_prefix": "\nA:",
        "suffix": "\nSo the answer is", "suffix_think": "\nSo the answer is",
        "stops": ["\n\nQ:", "\nQ:"], "own_marker": "So the answer is",
        "chance": 0.2, "forced": True,
    },
    "arc": {
        "kind": "letter", "n_answer": 8, "n_full": cfgmod.DATASETS["arc"],
        "prompt_file": "prompt_arc.txt", "rows_file": "rows_arc.jsonl",
        "sep": "\n\n", "q_prefix": "Q: ", "a_prefix": "\nA:",
        "suffix": "\nSo the answer is", "suffix_think": "\nSo the answer is",
        "stops": ["\n\nQ:", "\nQ:"], "own_marker": "So the answer is",
        "chance": 0.25, "forced": True,
    },
}

# ---------------------------------------------------------------- the four remaining slots
#: the ten BBH subtasks pooled into ONE slot, with the answer kind and the chance rate of each.
BBH_SUBTASK_KIND = {
    "date_understanding": ("letter", 0.1720),
    "logical_deduction_five_objects": ("letter", 0.2000),
    "tracking_shuffled_objects_three_objects": ("letter", 0.3333),
    "sports_understanding": ("yesno", 0.5000),
    "boolean_expressions": ("boolean", 0.5000),
    "multistep_arithmetic_two": ("numeric", 0.0000),
    "word_sorting": ("freeform", 0.0000),
    "navigate": ("yesno", 0.5000),
    "causal_judgement": ("yesno", 0.5000),
    "temporal_sequences": ("letter", 0.2500),
}
BBH_SUBTASKS = list(cfgmod.BBH_SUBTASKS)

#: the three-shot / five-shot CoT layout of S26, reused verbatim by every PP3 slot so that the
#: prompt surface the gates validated on BBH is the one MMLU, HellaSwag and StrategyQA also use.
_COT_LAYOUT = {
    "sep": "\n\n", "q_prefix": "Q: ", "a_prefix": "\nA: Let's think step by step.",
    "suffix": "\nSo the answer is", "suffix_think": "\nSo the answer is",
    "stops": ["\n\nQ:", "\nQ:"], "own_marker": "So the answer is", "forced": True,
}

TASKS["bbh"] = dict(_COT_LAYOUT, **{
    # word_sorting writes a list, so the forced read-out is longer than the 8 tokens a letter needs;
    # the per-row value below overrides this one where the kind asks for it.
    "kind": "letter", "n_answer": 8, "n_full": cfgmod.DATASETS["bbh"],
    "prompt_file": None, "rows_file": "rows_bbh.jsonl",
    "per_row_prefix": True, "per_row_kind": True,
    "chance": 0.28,        # the mean of the ten subtasks' chance rates, for the cards only
    # a flat 100-row draw over the pooled file spreads the calibration set at about 10 rows per
    # subtask by luck of the permutation; `stratify_cal_by` makes it exactly 10 per subtask by
    # seed, so the allocator's per-cell calibration accuracy is not starved on any one subtask.
    # Nothing else about the split (still cal/eval, still n_cal=100) changes.
    "stratify_cal_by": "subtask",
})

TASKS["mmlu"] = dict(_COT_LAYOUT, **{
    "kind": "letter", "n_answer": 8, "n_full": cfgmod.DATASETS["mmlu"],
    "prompt_file": None, "rows_file": "rows_mmlu.jsonl",
    "per_row_prefix": True, "chance": 0.25,
})

TASKS["hellaswag"] = dict(_COT_LAYOUT, **{
    "kind": "letter", "n_answer": 8, "n_full": cfgmod.DATASETS["hellaswag"],
    "prompt_file": "prompt_hellaswag.txt", "rows_file": "rows_hellaswag.jsonl",
    "chance": 0.25,
})

TASKS["strategyqa"] = dict(_COT_LAYOUT, **{
    "kind": "yesno", "n_answer": 8, "n_full": cfgmod.DATASETS["strategyqa"],
    "prompt_file": "prompt_strategyqa.txt", "rows_file": "rows_strategyqa.jsonl",
    "chance": 0.50,
})

#: forced read-out length by answer kind, where the kind needs more than the task's default.
N_ANSWER_BY_KIND = {"freeform": 48, "math": 32}

TASK_ORDER = list(cfgmod.TASK_ORDER)
FORCED_TASKS = [t for t in TASK_ORDER if TASKS[t]["forced"]]


def task_cfg(task):
    if task not in TASKS:
        raise KeyError("unknown task %r; known: %s" % (task, ", ".join(TASK_ORDER)))
    return TASKS[task]


# ------------------------------------------------------------------ rows and prompts
def rows(task):
    """The task's rows in the order of record: {idx, input, target, options, gold_text}."""
    p = os.path.join(DATA, task_cfg(task)["rows_file"])
    if not os.path.exists(p):
        raise FileNotFoundError(
            "%s missing. Run `python -m prod.tasks.freeze --copy` then, on a host with the "
            "datasets cached, `python -m prod.tasks.freeze --extend`." % p)
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


_PREFIX_CACHE = {}


def prompt_prefix(task, row=None):
    """The exemplar block, exactly as it enters the prompt (trailing separator included).

    A row may name its own exemplar file through `prefix_key` (pooled BBH, MMLU). The task's
    `prompt_file` is the fallback and is the only path the six original tasks ever take.
    """
    c = task_cfg(task)
    key = (row or {}).get("prefix_key") if c.get("per_row_prefix") else None
    name = ("prompt_%s.txt" % key) if key else c["prompt_file"]
    if name is None:
        raise KeyError("task %s needs a per-row prefix_key; row=%r" % (task, row))
    if name in _PREFIX_CACHE:
        return _PREFIX_CACHE[name]
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        _PREFIX_CACHE[name] = f.read()
    return _PREFIX_CACHE[name]


def row_kind(task, row=None):
    """The answer kind of one row: its own `kind` where the task allows it, else the task's."""
    c = task_cfg(task)
    if c.get("per_row_kind") and (row or {}).get("kind"):
        return row["kind"]
    return c["kind"]


def row_n_answer(task, row=None):
    """The forced read-out length for one row (word_sorting writes a list, a letter is 8 tokens)."""
    c = task_cfg(task)
    return int(N_ANSWER_BY_KIND.get(row_kind(task, row), c["n_answer"]))


def prefix_keys(task):
    """Every exemplar file the task's rows name, in first-appearance order."""
    c = task_cfg(task)
    if not c.get("per_row_prefix"):
        return [c["prompt_file"]]
    seen = []
    for r in rows(task):
        k = r.get("prefix_key")
        if k and k not in seen:
            seen.append(k)
    return seen


def data_hashes():
    return load_json(os.path.join(DATA, "hashes.json"), {})


# ------------------------------------------------------------------ split of record
def split_labels(task, n=None, seed=None, n_cal=None):
    """['eval'|'cal'] per row index, from `numpy.random.default_rng(20260908)`.

    Calibration is a fixed 100 questions per task by seed 20260908 (AQuA 100 of 254), evaluation is
    the rest. The draw: a permutation of the row indices, the first 100 are calibration.

    A task whose config names `stratify_cal_by` (pooled BBH, by `subtask`) draws its calibration
    rows stratified over that field instead -- 10 per BBH subtask at n_cal=100 -- so the pooled
    task is not spread thin by the luck of one flat permutation. The split field is not stored in
    `prod/tasks/data/`; it is computed here, same as every other task's.
    """
    n = len(rows(task)) if n is None else int(n)
    seed = SPLIT_SEED if seed is None else seed
    n_cal = N_CAL if n_cal is None else int(n_cal)
    key = task_cfg(task).get("stratify_cal_by")
    if key:
        return _stratified_split_labels(task, n, seed, n_cal, key)
    perm = np.random.default_rng(seed).permutation(n)
    cal = set(int(i) for i in perm[:min(n_cal, n)])
    return ["cal" if i in cal else "eval" for i in range(n)]


def _stratified_split_labels(task, n, seed, n_cal, key):
    """`n_cal` calibration rows spread evenly over `row[key]` (e.g. BBH's `subtask`), by one RNG
    seeded once and drawn from in a fixed (sorted-group) order, so the draw is reproducible.

    `n_cal` need not divide evenly by the number of groups: the remainder goes to the first groups
    in sorted order, one each, so every group still gets within one row of an equal share.
    """
    rws = rows(task)[:n]
    groups = {}
    for i, r in enumerate(rws):
        groups.setdefault(r.get(key), []).append(i)
    gkeys = sorted(groups.keys(), key=lambda x: (x is None, x))
    n_groups = len(gkeys) or 1
    per_group, remainder = divmod(min(n_cal, n), n_groups)
    rng = np.random.default_rng(seed)
    cal = set()
    for gi, gk in enumerate(gkeys):
        idxs = groups[gk]
        take = per_group + (1 if gi < remainder else 0)
        perm = rng.permutation(len(idxs))
        cal.update(int(idxs[j]) for j in perm[:min(take, len(idxs))])
    return ["cal" if i in cal else "eval" for i in range(n)]


# ------------------------------------------------------------------ prompt construction
def build_prompts(tok, task, task_rows, chat_template=False, suffix_text=None,
                  think_tag_is_stop=True):
    """(prompts, suffix_ids, stop_strings, eos_ids, extra).

    `chat_template=False` is the base branch of every spike. `chat_template=True` is the Ouro
    Thinking branch of S26/S28/S13: the exemplars go in the user turn, the forced suffix is
    prefixed with </think>, there are no stop strings and </think> joins the eos set.

    `think_tag_is_stop=False` (config `think_tag_is_stop`) takes the closing tag OUT of the eos set
    and gives the chain the task's stop strings instead, so generation runs past the reasoning block
    to the answer the model writes after it. With the tag in the eos set -- the default, and what
    every grid of record was generated under -- the natural-stop cut lands ON the tag and that
    answer is discarded (ouro_2_6b_think GSM8K natural k=4: 21.7 against 56.8 forced, 98% of rows
    cut at the tag). The forced read-out suffix still opens with the tag either way: closing the
    block is how the read-out is asked for.
    """
    c = task_cfg(task)
    # the exemplar block can differ per row (pooled BBH, MMLU), so the head is built per row.
    heads = [prompt_prefix(task, r) + c["sep"] + c["q_prefix"] for r in task_rows]
    prefix = prompt_prefix(task, task_rows[0] if task_rows else None)
    head = prefix + c["sep"] + c["q_prefix"]
    if not chat_template:
        stext = c["suffix"] if suffix_text is None else suffix_text
        prompts = [h + r["input"] + c["a_prefix"] for h, r in zip(heads, task_rows)]
        suffix_ids = tok(stext, add_special_tokens=False)["input_ids"]
        stops = list(c["stops"])
        eos_ids = [tok.eos_token_id]
        extra = {"prefix_tokens": len(tok(prefix, add_special_tokens=False)["input_ids"]),
                 "prefix_keys": sorted({(r.get("prefix_key") or c["prompt_file"])
                                        for r in task_rows})}
    else:
        stext = c["suffix_think"] if suffix_text is None else suffix_text
        user = [h + r["input"] for h, r in zip(heads, task_rows)]
        prompts = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
                   for u in user]
        end_think = tok.convert_tokens_to_ids("</think>")
        suffix_ids = [end_think] + tok(stext, add_special_tokens=False)["input_ids"]
        stops = None if think_tag_is_stop else list(c["stops"])
        eos_ids = [tok.eos_token_id, end_think] if think_tag_is_stop else [tok.eos_token_id]
        extra = {"end_think_id": int(end_think)}
    extra.update({"think_tag_is_stop": bool(think_tag_is_stop),
                  "suffix_text": stext, "chat_template": bool(chat_template),
                  "own_marker": c["own_marker"], "kind": c["kind"],
                  "per_row_kind": bool(c.get("per_row_kind")),
                  "n_answer_tokens": c["n_answer"]})
    return prompts, suffix_ids, stops, eos_ids, extra


# ------------------------------------------------------------------ natural stop
def make_find_cut(tok, stop_strings, eos_ids, chat_template=False, eos_cut=True,
                  think_tag_is_stop=True):
    """s26_common.make_find_cut / s28_common.make_find_cut, verbatim (both are S9a's).

    Returns f(ids) -> (cut_length, marker_or_None): decode once, find the earliest stop string,
    bisect on the decoded length; an eos before that wins.

    `eos_cut=False` drops the eos branch, an older cut rule that the S9f (McLeish) and S9c (Huginn)
    paths follow: there, a trace that ends at eos with no stop string is cut at `len(ids)`, eos
    token included. Gate G1 uses `eos_cut=False` on those two paths to show that this is the only
    difference from the S26/S28/S32 rule the production package uses everywhere.

    `think_tag_is_stop=False` on a chat-template path uses the BASE cut rule, so the cut is the
    task's first stop string (or the eos), past the closing think tag rather than on it. The tag is
    then not in `eos_ids` either (build_prompts), so nothing cuts the chain at the end of the
    reasoning block.
    """
    def find_cut_base(ids):
        full = tok.decode(ids, clean_up_tokenization_spaces=False)
        hits = [full.find(m) for m in stop_strings]
        hits = [h for h in hits if h >= 0]
        eosp = None
        if eos_cut:
            for j, v in enumerate(ids):
                if v in eos_ids:
                    eosp = j
                    break
        if not hits:
            return (len(ids), None) if eosp is None else (eosp, "eos")
        pos = min(hits)
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tok.decode(ids[:mid], clean_up_tokenization_spaces=False)) > pos:
                hi = mid
            else:
                lo = mid + 1
        c = max(0, lo - 1)
        if eosp is not None and eosp < c:
            return eosp, "eos"
        return c, full[pos:pos + 16]

    def find_cut_think(ids):
        for j, v in enumerate(ids):
            if v in eos_ids:
                return j, tok.decode([v])
        return len(ids), None

    return find_cut_think if (chat_template and think_tag_is_stop) else find_cut_base


# ------------------------------------------------------------------ parsers
# numeric: s9a_common.parse_final_answer / trace_own_answer, verbatim
RE_INT = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_final_answer(text):
    if text is None:
        return None
    if "Final Answer:" in text:
        tail = text.rsplit("Final Answer:", 1)[1]
    else:
        tail = text
    tail = tail.split("\n")[0] if "\n" in tail else tail
    m = RE_INT.findall(tail.replace("$", ""))
    if not m:
        m = RE_INT.findall(text.replace("$", ""))
        if not m:
            return None
    v = m[-1].replace(",", "")
    if v.endswith("."):
        v = v[:-1]
    return v


def num_eq(a, b):
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:                                         # noqa: BLE001
        return a == b


def trace_own_number(cut_text):
    """First number after the LAST '####' in the cut trace (s9a_common, verbatim)."""
    if not cut_text or "####" not in cut_text:
        return None
    m = RE_INT.findall(cut_text.rsplit("####", 1)[1].replace("$", ""))
    return m[0].replace(",", "") if m else None


# letter: s26_common (A-F) widened to lowercase by s28_common (the exemplars write "(a)")
RE_PAREN = re.compile(r"\(([A-Fa-f])\)")
RE_BARE = re.compile(r"^[^A-Za-z0-9]*([A-Fa-f])\b")
RE_YESNO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_letter(text, options=None):
    if text is None:
        return None
    m = RE_PAREN.search(text)
    if m:
        return "(%s)" % m.group(1).upper()
    m = RE_BARE.match(text)
    if m and (not options or m.group(1).upper() in options):
        return "(%s)" % m.group(1).upper()
    return None


def parse_yesno(text):
    if text is None:
        return None
    m = RE_YESNO.search(text)
    return m.group(1).lower() if m else None


# math: s13_common.last_boxed / _norm / math_eq
RE_BOXED = re.compile(r"\\boxed\s*{")


def last_boxed(s):
    if s is None:
        return None
    ms = list(RE_BOXED.finditer(s))
    if not ms:
        return None
    i = ms[-1].end()
    depth, out = 1, []
    while i < len(s) and depth:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() or None


_NORM_DROP = ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", "\\$", " ",
              "\\text{ }", "^{\\circ}", "^\\circ"]


def norm_math(a):
    """s13_common._norm, verbatim in behaviour."""
    if a is None:
        return None
    s = str(a).strip()
    for d in _NORM_DROP:
        s = s.replace(d, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    return s or None


_MV = {}


def math_eq(pred, gold):
    """math_verify when importable, the normalised string comparison otherwise (s13_common)."""
    if pred is None:
        return False
    if "fn" not in _MV:
        try:
            from math_verify import parse as mv_parse, verify as mv_verify
            _MV["fn"] = (mv_parse, mv_verify)
        except Exception:                                     # noqa: BLE001
            _MV["fn"] = None
    if _MV["fn"] is not None:
        mv_parse, mv_verify = _MV["fn"]
        try:
            return bool(mv_verify(mv_parse("$%s$" % gold), mv_parse("$%s$" % pred)))
        except Exception:                                     # noqa: BLE001
            pass
    return norm_math(pred) == norm_math(gold)


def math_verify_available():
    if "fn" not in _MV:
        math_eq("1", "1")
    return _MV["fn"] is not None


# two answer kinds the pooled BBH slot brings in.
RE_BOOL = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def parse_boolean(text):
    """`boolean_expressions`: the first True/False token, normalised to the BBH target casing."""
    if text is None:
        return None
    m = RE_BOOL.search(text)
    return ("True" if m.group(1).lower() == "true" else "False") if m else None


def parse_freeform(text):
    """`word_sorting`: the first line of the read-out, whitespace-collapsed.

    BBH's own scorer compares the answer string exactly; the only normalisation here is the one the
    read-out format forces (a leading space after "So the answer is", a trailing period, and runs of
    whitespace), which is what s26_common does for its letter answers too.
    """
    if text is None:
        return None
    t = text.strip().split("\n")[0].strip()
    t = t.rstrip(".").strip()
    t = " ".join(t.split())
    return t or None


def norm_freeform(a):
    return " ".join(str(a).strip().rstrip(".").split()).lower() if a is not None else None


def parse_forced(text, task, options=None, kind=None):
    """The forced read-out parse: `text` is the decoded forced continuation."""
    kind = kind or task_cfg(task)["kind"]
    if kind == "boolean":
        return parse_boolean(text)
    if kind == "freeform":
        return parse_freeform(text)
    if kind == "numeric":
        return parse_final_answer(text)
    if kind == "letter":
        return parse_letter(text, options)
    if kind == "yesno":
        return parse_yesno(text)
    # math: the last \boxed{...}, else the first line (s32_common.parse_forced)
    if text is None:
        return None
    b = last_boxed(text)
    if b is not None:
        return b
    t = text.strip().split("\n")[0].strip()
    return t or None


THINK_CLOSE = "</think>"


def parse_own(cut_text, task, options=None, kind=None, think_tag_is_stop=True):
    """The model's own answer inside the cut trace, after the LAST own marker.

    `think_tag_is_stop=False` reads THROUGH the closing think tag: with the tag no longer a stop the
    cut trace holds the reasoning block AND the answer written after it, and that answer is the own
    answer. What follows the last tag is parsed first; the whole trace is still parsed when nothing
    after the tag gives an answer, so the switch can never turn an answer into no answer.
    """
    if not think_tag_is_stop and cut_text and THINK_CLOSE in cut_text:
        tail = cut_text.split(THINK_CLOSE)[-1]
        if tail.strip():
            after = _parse_own(tail, task, options, kind)
            if after is not None:
                return after
    return _parse_own(cut_text, task, options, kind)


def _parse_own(cut_text, task, options=None, kind=None):
    c = task_cfg(task)
    kind = kind or c["kind"]
    if not cut_text:
        return None
    if kind == "math":
        return last_boxed(cut_text)
    if kind == "numeric" and c["own_marker"] == "####":
        return trace_own_number(cut_text)
    mk = c["own_marker"]
    if mk not in cut_text:
        return None
    tail = cut_text.rsplit(mk, 1)[1].split("\n")[0]
    if kind == "letter":
        return parse_letter(tail, options)
    if kind == "yesno":
        return parse_yesno(tail)
    if kind == "boolean":
        return parse_boolean(tail)
    if kind == "numeric":
        return parse_final_answer(tail)
    return parse_freeform(tail)


def ans_eq(pred, gold, task, kind=None):
    if pred is None:
        return False
    kind = kind or task_cfg(task)["kind"]
    if kind == "numeric":
        return bool(num_eq(pred, gold))
    if kind == "math":
        return bool(math_eq(pred, gold))
    if kind == "yesno":
        return pred.strip().lower() == str(gold).strip().lower()
    if kind == "boolean":
        return pred.strip().lower() == str(gold).strip().lower()
    if kind == "freeform":
        return norm_freeform(pred) == norm_freeform(gold)
    # letters: gold is a bare letter in most row files and "(X)" in the BBH ones; the parses
    # return "(X)". Strip the parentheses on both sides and compare.
    return pred.strip().upper().strip("()") == str(gold).strip().upper().strip("()")
