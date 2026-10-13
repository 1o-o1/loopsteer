"""RA-30 Thinking cells (plan_ra30.md section 6e): gold-intermediate matching instead of equation parsing.

The expected intermediates come from the gold solution of the original problem A and of the changed problem B:
- GSM8K: the GSM8K test solution of problem A (calculator annotations "<<expr=res>>", data/gold_gsm8k.json); B's
  values are the same equations with value_A -> value_B substituted in the literals and every changed earlier result
  propagated; the last result must equal gold_A and, re-evaluated, gold_B.
- ASDiv: the pair's equation_A and equation_B (the same expression tree with the changed number); the intermediates
  are the values of the inner operation nodes (the root is the final answer); the roots must equal gold_A and gold_B.
A candidate is an intermediate whose value changes from A to B, is not a number of question A and is not the final
answer. In the clean Thinking chain of prompt A, P is the first number span whose value equals a candidate's A value;
derived_B is that candidate's B value. The chain must end in the correct answer (the caller's screen).
"""
import ast
import math
import operator

import chain as CH

THINK_DROPS = ("plen_mismatch", "clean_wrong", "no_gold_solution", "gold_parse", "gold_inconsistent",
               "no_changed_intermediate", "value_not_in_chain", "value_format", "derived_B_format", "token_count",
               "token_boundary", "token_count_ids", "length_mismatch")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def gsm8k_pairs(solution, value_A, value_B, gold_A, gold_B):
    """[(r_A, r_B)] for every equation of the gold solution (in order) or a drop reason."""
    if not solution:
        return None, "no_gold_solution"
    eqs = CH.equations(solution)
    if not eqs:
        return None, "gold_parse"
    for e in eqs:
        if e["val"] is None or e["res_val"] is None or not CH.feq(e["val"], e["res_val"]):
            return None, "gold_parse"
    if not CH.feq(eqs[-1]["res_val"], CH.numval(gold_A)):
        return None, "gold_inconsistent"
    vA, vB = float(value_A), float(value_B)
    new = []
    for i, e in enumerate(eqs):
        repl = [(vA, vB)] + [(eqs[j]["res_val"], new[j]) for j in range(i) if not CH.feq(new[j], eqs[j]["res_val"])]
        v = CH.safe_eval(CH.substitute(e["expr"], repl))
        if v is None:
            return None, "gold_parse"
        new.append(v)
    if not CH.feq(new[-1], CH.numval(gold_B)):
        return None, "gold_inconsistent"
    return [(e["res_val"], b) for e, b in zip(eqs, new)], None


def _inner_values(expr):
    """Values of the inner operation nodes (post-order) and the root value of an arithmetic expression."""
    tree = ast.parse(expr.strip(), mode="eval").body
    inner = []

    def ev(n, root=False):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div) and b == 0:
                raise ZeroDivisionError
            v = _OPS[type(n.op)](a, b)
            if not root:
                inner.append(v)
            return v
        raise ValueError(type(n).__name__)
    root = ev(tree, root=True)
    return inner, root


def asdiv_pairs(equation_A, equation_B, gold_A, gold_B):
    try:
        ia, ra = _inner_values(equation_A)
        ib, rb = _inner_values(equation_B)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None, "gold_parse"
    if len(ia) != len(ib):
        return None, "gold_parse"
    if not (CH.feq(ra, CH.numval(gold_A)) and CH.feq(rb, CH.numval(gold_B))):
        return None, "gold_inconsistent"
    return list(zip(ia, ib)) + [(ra, rb)], None


def candidates(pairs, question_A, gold_A):
    """Changed, non-final intermediates whose A value is not a number of question A (first occurrence order)."""
    qn = [q for q in CH.prompt_numbers(question_A) if q is not None]
    out = []
    for a, b in pairs[:-1]:
        if CH.feq(a, b) or any(CH.feq(a, q) for q in qn) or CH.feq(a, CH.numval(gold_A)):
            continue
        if not any(CH.feq(a, x[0]) for x in out):
            out.append((a, b))
    return out


def fmt_like(v, written):
    """B's value written in the style of A's occurrence: plain digits, or thousands commas."""
    if "," in written:
        if abs(v - round(v)) > 1e-9:
            return None
        return format(int(round(v)), ",")
    return CH.fmt_num(v)


def match(chain_text, cands):
    """The first number span of the chain whose value is a candidate's A value. Returns a record with drop None, or
    the drop reason."""
    rec = {"drop": None}
    if not cands:
        rec["drop"] = "no_changed_intermediate"
        return rec
    for m in CH.NUM_RE.finditer(chain_text or ""):
        v = CH.numval(m.group(0))
        hit = next(((a, b) for a, b in cands if CH.feq(v, a)), None)
        if hit is None:
            continue
        a, b = hit
        w = m.group(0)
        plain_or_comma = w == CH.fmt_num(a) or (("," in w) and abs(a - round(a)) < 1e-9 and w == format(int(round(a)), ","))
        if not plain_or_comma:
            rec["drop"] = "value_format"
            return rec
        if b < 0 or (abs(a - round(a)) < 1e-9 and abs(b - round(b)) > 1e-9) or not math.isfinite(b):
            rec["drop"] = "derived_B_format"
            return rec
        wb = fmt_like(b, w)
        if wb is None:
            rec["drop"] = "derived_B_format"
            return rec
        if len(wb) != len(w):
            rec["drop"] = "token_count"
            return rec
        rec.update({"derived_A": w, "derived_B": wb, "char_span": [m.start(), m.end()], "value_A": a, "value_B": b,
                    "n_candidates": len(cands), "e1": None})
        return rec
    rec["drop"] = "value_not_in_chain"
    return rec


def carry_prose(cont_text, derived_A, derived_B):
    """Prose carry for Thinking chains (no annotations): the first number of the continuation equal to derived_A or
    derived_B decides (B, A, none)."""
    a, b = CH.numval(derived_A), CH.numval(derived_B)
    for m in CH.NUM_RE.finditer(cont_text or ""):
        v = CH.numval(m.group(0))
        if CH.feq(v, b):
            return "B"
        if CH.feq(v, a):
            return "A"
    return "none"
