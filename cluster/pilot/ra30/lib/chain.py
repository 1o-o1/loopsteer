"""RA-30 chain logic (pure Python, CPU-testable; plan_ra30.md sections 4 to 6).

The clean greedy chain of prompt A is parsed into equations e_1..e_m: the calculator annotations "<<expr=res>>" the
GSM8K exemplars teach (GSM8K and ASDiv both use the GSM8K prompt). The derived value is the result of the first
non-final equation whose result changes under value_A -> value_B (RA brief update; r_1 under the original rule),
when it is not a number of the prompt. derived_B is that equation re-evaluated with value_B in place of value_A. The item is consistent when
e_2..e_m, re-evaluated with r_1 replaced by derived_B (and every later result that used it replaced in turn, the
prompt numbers unchanged: the edit is carried by the chain only), give gold_B. Every drop has one reason code.
"""
import ast
import math
import operator
import re

ANN_RE = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
NUM_RE = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\d])")
OWN_MARKER = "####"

DROP_REASONS = (
    "clean_wrong",            # screen (c): the clean chain's own answer is not gold_A
    "no_equation",            # no calculator annotation before the answer marker
    "one_step",               # m = 1
    "chain_arithmetic_error",  # an annotation's written result is not the value of its expression
    "last_result_not_answer",  # r_m is not gold_A (the chain's answer is not its last equation)
    "no_intermediate",        # r_1 is a number of prompt A
    "ambiguous_value",        # value_A occurs more than once in question A
    "no_changed_step",        # no non-final equation whose result changes under value_A -> value_B (rule of record)
    "e1_independent",         # mode r1 only: e_1 does not change under value_A -> value_B
    "expr_parse",             # an expression does not evaluate under the arithmetic grammar
    "derived_B_format",       # derived_B is negative, not finite, or not integral when derived_A is
    "token_count",            # derived_A and derived_B differ in character (digit) count
    "ambiguous_operand",      # a later operand equals both a prompt number and a changed earlier result
    "inconsistent",           # the chain-carried re-evaluation does not give gold_B
    "value_not_in_chain",     # screen (a): derived_A is not written before the answer marker
    "value_format",           # the first written occurrence of derived_A is not its plain digits
    "token_boundary",         # the tokens covering derived_A decode to more than its digits
    "token_count_ids",        # the token count of derived_B differs from that of derived_A (tokenizer)
    "length_mismatch",        # T2: prompt A + prefix_A and prompt A + prefix_B differ in length or last index
    "value_earlier",          # derived_A is written before its own result slot, or the slot found is not e_i's
    "plen_mismatch",          # prompt A or B does not rebuild to the pool's plen, or the span bounds differ
)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def numval(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def feq(a, b):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)), abs(float(b)))


def clean_expr(expr):
    """Calculator expression to Python arithmetic: x X * multiply, the division sign divides, $ and thousands
    commas removed. Anything outside digits, . + - * / ( ) and spaces makes it unparsable."""
    e = expr.replace("$", "").replace("×", "*").replace("÷", "/")
    e = re.sub(r"(?<=\d),(?=\d{3}\b)", "", e)
    e = re.sub(r"(?<=[\d)\s])[xX](?=[\d(\s])", "*", e)
    if not re.fullmatch(r"[\d.+\-*/()\s]+", e or ""):
        return None
    return e


def safe_eval(expr):
    """Value of an arithmetic expression (+ - * / unary minus, parentheses), float; None when it does not parse."""
    e = clean_expr(expr)
    if e is None:
        return None
    try:
        tree = ast.parse(e.strip(), mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div) and b == 0:
                raise ZeroDivisionError
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        raise ValueError("node %s" % type(n).__name__)
    try:
        v = ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def operands(expr):
    """Number literals of an expression: [(start, end, value)]."""
    out = []
    for m in NUM_RE.finditer(expr):
        v = numval(m.group(0))
        if v is not None:
            out.append((m.start(), m.end(), v))
    return out


def substitute(expr, repl):
    """Replace each operand literal whose value maps in `repl` [(old_value, new_value)] (first match wins)."""
    pieces, last = [], 0
    for a, b, v in operands(expr):
        for old, new in repl:
            if feq(v, old):
                pieces.append(expr[last:a])
                pieces.append(fmt_num(new))
                last = b
                break
    pieces.append(expr[last:])
    return "".join(pieces)


def fmt_num(v):
    """Plain decimal string: integral values without a point, others with trailing zeros removed."""
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return ("%.10f" % v).rstrip("0").rstrip(".")


def body(text):
    """The chain before the last own-answer marker (the equations live here)."""
    if text is None:
        return ""
    return text.rsplit(OWN_MARKER, 1)[0] if OWN_MARKER in text else text


def equations(text):
    """[{expr, res, res_val, val, span}] of the annotations before the answer marker, in order."""
    out = []
    for m in ANN_RE.finditer(body(text)):
        out.append({"expr": m.group(1), "res": m.group(2).strip(), "res_val": numval(m.group(2)),
                    "val": safe_eval(m.group(1)), "span": [m.start(), m.end()]})
    return out


def prompt_numbers(question):
    return [numval(m.group(0)) for m in NUM_RE.finditer(question or "")]


def analyze(chain_text, question_A, value_A, value_B, gold_A, gold_B, mode="first_changed"):
    """CPU screen of one item from the clean chain's cut text. Returns a dict with `drop` (None = kept) and, when
    kept, derived_A / derived_B strings, the character span of derived_A's first written occurrence and the
    chain-carried answer. Screen (c) (own answer = gold_A) is the caller's, from the Table 2 parser.
    mode "first_changed" (rule of record, RA brief update 2026-10-12): the derived value is the result of the first
    equation e_i, i < m, whose result changes when value_A is replaced by value_B and is not the final answer.
    mode "r1" (the original brief rule, information only): the derived value is r_1."""
    rec = {"drop": None, "mode": mode}
    eqs = equations(chain_text)
    rec["m"] = len(eqs)
    if not eqs:
        rec["drop"] = "no_equation"
        return rec
    if len(eqs) == 1:
        rec["drop"] = "one_step"
        return rec
    for e in eqs:
        if e["val"] is None:
            rec["drop"] = "expr_parse"
            return rec
        if e["res_val"] is None or not feq(e["val"], e["res_val"]):
            rec["drop"] = "chain_arithmetic_error"
            return rec
    if not feq(eqs[-1]["res_val"], numval(gold_A)):
        rec["drop"] = "last_result_not_answer"
        return rec
    qn = prompt_numbers(question_A)
    vA, vB = float(value_A), float(value_B)
    if mode == "first_changed":
        i0 = None
        for i, e in enumerate(eqs[:-1]):
            if not any(feq(v, vA) for _a, _b, v in operands(e["expr"])) or feq(e["res_val"], numval(gold_A)):
                continue
            v = safe_eval(substitute(e["expr"], [(vA, vB)]))
            if v is not None and not feq(v, e["res_val"]):
                i0 = i
                break
        if i0 is None:
            rec["drop"] = "no_changed_step"
            return rec
    elif mode == "r1":
        i0 = 0
    else:
        raise ValueError("unknown mode %r" % mode)
    rec["i0"] = i0
    r1 = eqs[i0]["res_val"]
    if any(feq(r1, q) for q in qn if q is not None):
        rec["drop"] = "no_intermediate"
        return rec
    if sum(1 for q in qn if feq(q, vA)) != 1:
        rec["drop"] = "ambiguous_value"
        return rec
    if not any(feq(v, vA) for _a, _b, v in operands(eqs[i0]["expr"])):
        rec["drop"] = "e1_independent"
        return rec
    dB = safe_eval(substitute(eqs[i0]["expr"], [(vA, vB)]))
    if dB is None:
        rec["drop"] = "expr_parse"
        return rec
    dA_str = fmt_num(r1)
    if dB < 0 or (("." not in dA_str) and abs(dB - round(dB)) > 1e-9):
        rec["drop"] = "derived_B_format"
        return rec
    dB_str = fmt_num(dB)
    if feq(dB, r1):
        rec["drop"] = "e1_independent"
        return rec
    if len(dB_str) != len(dA_str):
        rec["drop"] = "token_count"
        return rec
    # chain-carried re-evaluation: results r_j -> r'_j propagate through equal-valued operands; prompt numbers stay
    new = [e["res_val"] for e in eqs[:i0]] + [dB]
    for i in range(i0 + 1, len(eqs)):
        repl = []
        for j in range(i0, i):
            if not feq(new[j], eqs[j]["res_val"]):
                repl.append((eqs[j]["res_val"], new[j]))
        for _a, _b, v in operands(eqs[i]["expr"]):
            hit = [r for r in repl if feq(v, r[0])]
            if hit and any(feq(v, q) for q in qn if q is not None):
                rec["drop"] = "ambiguous_operand"
                return rec
            if len({fmt_num(r[1]) for r in hit}) > 1:
                rec["drop"] = "ambiguous_operand"
                return rec
        v = safe_eval(substitute(eqs[i]["expr"], repl))
        if v is None:
            rec["drop"] = "expr_parse"
            return rec
        new.append(v)
    rec["answer_edit"] = fmt_num(new[-1])
    # information only: the same propagation with value_A -> value_B also substituted in every equation
    full = []
    for i in range(len(eqs)):
        repl = [(vA, vB)] + [(eqs[j]["res_val"], full[j]) for j in range(i) if not feq(full[j], eqs[j]["res_val"])]
        v = safe_eval(substitute(eqs[i]["expr"], repl))
        full.append(v if v is not None else float("nan"))
    rec["answer_full_subst"] = fmt_num(full[-1]) if math.isfinite(full[-1]) else None
    if not feq(new[-1], numval(gold_B)):
        rec["drop"] = "inconsistent"
        return rec
    # screen (a): the first written occurrence of derived_A before the answer marker
    occ = None
    start = eqs[i0 - 1]["span"][1] if i0 > 0 else 0
    for mt in NUM_RE.finditer(body(chain_text), start):
        if feq(numval(mt.group(0)), r1):
            occ = mt
            break
    if occ is None:
        rec["drop"] = "value_not_in_chain"
        return rec
    if occ.group(0) != dA_str:
        rec["drop"] = "value_format"
        return rec
    # review B4: an earlier written copy of the value (left unedited) or a slot past e_i's annotation
    if any(feq(numval(m.group(0)), r1) for m in NUM_RE.finditer(body(chain_text)[:occ.start()]))             or occ.start() >= eqs[i0]["span"][1]:
        rec["drop"] = "value_earlier"
        return rec
    rec.update({"derived_A": dA_str, "derived_B": dB_str, "char_span": [occ.start(), occ.end()],
                "e1": eqs[i0]["expr"], "derived_B_in_prompt": any(feq(dB, q) for q in qn if q is not None),
                # review N8: answer_B already written in prompt A or in prefix_A (a coincidence, not a leak)
                "gold_B_in_prompt_or_prefix": any(feq(numval(gold_B), q) for q in qn if q is not None)
                or any(feq(numval(m.group(0)), numval(gold_B)) for m in NUM_RE.finditer(chain_text[:occ.end()])),
                # review N9: leading digits equal in derived_A and derived_B (positions of P with no real change)
                "n_lead_same": next((i for i, (x, y) in enumerate(zip(dA_str, dB_str)) if x != y), len(dA_str))})
    return rec


# ------------------------------------------------------------------ token geometry (tokenizer passed in)
def char_offsets(tok, ids):
    """End character offset of each token in tok.decode(ids) (prefix decodes; exact for the decode the parser
    reads)."""
    return [len(tok.decode(ids[:i + 1], clean_up_tokenization_spaces=False)) for i in range(len(ids))]


def token_geometry(tok, prompt_ids, chain_ids, rec):
    """prefix_A / prefix_B ids and the absolute derived-value positions P = [p0, p1) in prompt + prefix. Adds
    `drop` on a token-level failure (token_boundary, token_count_ids, length_mismatch)."""
    c0, c1 = rec["char_span"]
    ends = char_offsets(tok, chain_ids)
    starts = [0] + ends[:-1]
    sel = [i for i in range(len(chain_ids)) if ends[i] > c0 and starts[i] < c1 and ends[i] > starts[i]]
    if not sel:
        return dict(rec, drop="token_boundary")
    t0, t1 = sel[0], sel[-1] + 1
    if tok.decode(chain_ids[t0:t1], clean_up_tokenization_spaces=False) != rec["derived_A"]:
        return dict(rec, drop="token_boundary")
    b_ids = tok(rec["derived_B"], add_special_tokens=False)["input_ids"]
    if len(b_ids) != t1 - t0:
        return dict(rec, drop="token_count_ids")
    pre_A = list(chain_ids[:t1])
    pre_B = list(chain_ids[:t0]) + list(b_ids)
    seq_A, seq_B = list(prompt_ids) + pre_A, list(prompt_ids) + pre_B
    if len(seq_A) != len(seq_B):
        return dict(rec, drop="length_mismatch")
    plen = len(prompt_ids)
    out = dict(rec, prefix_A=pre_A, prefix_B=pre_B, P=[plen + t0, plen + t1], prefix_len=t1,
               tokens_A=[tok.decode([i]) for i in pre_A[t0:t1]], tokens_B=[tok.decode([i]) for i in pre_B[t0:t1]],
               canonical_B=tok(tok.decode(pre_B, clean_up_tokenization_spaces=False),
                               add_special_tokens=False)["input_ids"] == pre_B)
    return out


# ------------------------------------------------------------------ readouts
def outcome(own, gold_A, gold_B):
    """steer_gen.outcome: B, A or neither (no parse counts as neither)."""
    if own is None:
        return "neither"
    if feq(numval(own), numval(gold_B)):
        return "B"
    if feq(numval(own), numval(gold_A)):
        return "A"
    return "neither"


def carry(cont_text, derived_A, derived_B):
    """The first complete annotation of the continuation with an operand equal to derived_A or derived_B decides:
    'B' (the edited value is carried), 'A', or 'none' (no later step uses either)."""
    a, b = numval(derived_A), numval(derived_B)
    for m in ANN_RE.finditer(body(cont_text)):
        for _s, _e, v in operands(m.group(1)):
            if feq(v, b):
                return "B"
            if feq(v, a):
                return "A"
    return "none"
