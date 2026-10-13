"""RA-30 operand flag and audit wording notes at summary time (plan_ra30.md section 6d; RA decision on the operand items).

The Ouro runs (1.4B and 2.6B, including those frozen in the bundle at plan 3cf3fc98) keep their run-time rules. This
step re-parses each scored item's stored clean chain (screen.jsonl clean_ids) with the run's own parser and rules,
checks that it reproduces the stored derived value and equation, and flags value_is_operand: the derived value
equals an operand of its own equation, so its first written copy can be the operand ("4 / 2 = 2") and the edited
positions P can sit on it. It writes operand_flags.json and summary_operand.md beside the run's summary: CA of every
arm on all items and without the flagged items (90 percent bootstrap, 10,000 resamples, seed 0). It changes no run
file and no frozen code. McLeish drops these items at run time (section 6c) and needs no flag.

    python ra30_flags.py --out OUT/<stage>/<ds>                       (Ouro-1.4B; tokenizer from the host cache)
    python ra30_flags.py --out OUT/ouro_2_6b_base/<stage>/<ds>
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "lib"), os.path.join(HERE, "tree")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PROD_DEV", "cpu")
os.environ.setdefault("K17_DATASET", "gsm8k")

import arms as AR  # noqa: E402
import chain as CH  # noqa: E402
import ra30 as R0  # noqa: E402
import ra30_data as RD  # noqa: E402


def operand_flag(e1, derived_A):
    return any(CH.feq(v, CH.numval(derived_A)) for _s, _e, v in CH.operands(e1))


def flags(out, tok=None):
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model, ds = cfg.get("model", RD.DEFAULT_MODEL), cfg["ds"]
    if tok is None:
        from prod import steer_gen as S
        tok = S.load_tokenizer(model)
    pool = RD.load_pool(ds) if model == RD.DEFAULT_MODEL else RD.load_pool(ds, model)
    by_item = {int(it["idx"]): it for it in pool["items"]}
    scr = {int(r["idx"]): r for r in R0.read_jsonl(os.path.join(out, "screen.jsonl"))}
    rows = R0.read_jsonl(os.path.join(out, "rows.jsonl"))
    by = {}
    for r in rows:
        by.setdefault(int(r["idx"]), {})[r["arm"]] = r
    items = sorted(by)
    flagged, mism = [], []
    for i in items:
        sc, it = scr[i], by_item[i]
        _c, txt, _own = RD.cut_parse(tok, ds, sc["clean_ids"])
        a = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"])
        if a["drop"] is not None or a["derived_A"] != sc["derived_A"] or a["e1"] != sc["e1"]:
            mism.append(i)
            continue
        if operand_flag(a["e1"], a["derived_A"]):
            flagged.append(i)
    keep = [i for i in items if i not in flagged]
    arms = [a for a in AR.ARM_NAMES if any(a in by[i] for i in items)]
    tab = {}
    for a in arms:
        ia = [i for i in items if a in by[i]]
        ca = lambda its: R0.boot_mean([1.0 if by[i][a]["outcome"] == "B" else 0.0 for i in its])
        tab[a] = {"all": ca(ia), "without_flagged": ca([i for i in ia if i in keep])}
    res = {"model": model, "dataset": ds, "n_items": len(items), "flagged": flagged, "n_flagged": len(flagged),
           "reparse_mismatch": mism, "CA": tab,
           "rule": "value_is_operand: the derived value equals an operand of its own equation (summary-time flag)"}
    R0.write_json(os.path.join(out, "operand_flags.json"), res)
    fmt = lambda v: "-" if v is None else "%.3f [%.3f, %.3f] (n %d)" % tuple(v)
    lines = ["# RA-30 operand flag: %s %s" % (model, ds), "",
             "Items %d; flagged value_is_operand %d %s; re-parse mismatches %d." % (len(items), len(flagged),
                                                                                   flagged, len(mism)), "",
             "| arm | CA, all items | CA, without flagged items |", "|---|---|---|"]
    for a in arms:
        lines.append("| %s | %s | %s |" % (a, fmt(tab[a]["all"]), fmt(tab[a]["without_flagged"])))
    with open(os.path.join(out, "summary_operand.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if mism:
        R0.die(R0.EXIT_IDENTITY, "re-parse does not reproduce the stored derived value for items %s" % mism[:10])
    R0.log("operand flags %s %s: %d of %d items flagged" % (model, ds, len(flagged), len(items)))
    return res


def notes(out, rows=None, scr=None):
    """Wording limits of the RA-30 audit (notes/ra_audit_ra30_gsm8k.md W2, W10), for every RA-30 summary: never an
    identity. Writes summary_notes.md and notes.json beside the run's summary (no tokenizer, no run file changed):
    - cvP against cv1: "cvP equals cv1 token for token on X of N chains (CA difference d [90% interval])";
    - kv4 against cv3: the token-identical share on the gate items, with "kv4 ran on the G gate items only";
    - the carried-value floor: the items whose derived_B is a number of prompt A, and the carry-B share that this
      alone explains in an unedited arm."""
    if rows is None:
        rows = R0.read_jsonl(os.path.join(out, "rows.jsonl"))
    if scr is None:
        scr = R0.read_jsonl(os.path.join(out, "screen.jsonl"))
    scr = {int(r["idx"]): r for r in scr}
    by = {}
    for r in rows:
        by.setdefault(int(r["idx"]), {})[r["arm"]] = r
    items = sorted(by)
    res, lines = {"n_items": len(items)}, ["# RA-30 summary notes (audit wording limits)", ""]
    both = [i for i in items if "cvP" in by[i] and "cv1" in by[i]]
    if both:
        same = sum(1 for i in both if by[i]["cvP"]["chain_ids"] == by[i]["cv1"]["chain_ids"])
        d = R0.boot_diff([1.0 if by[i]["cvP"]["outcome"] == "B" else 0.0 for i in both],
                         [1.0 if by[i]["cv1"]["outcome"] == "B" else 0.0 for i in both])
        res["cvP_cv1"] = {"token_identical": same, "n": len(both), "CA_difference": d}
        lines.append("cvP equals cv1 token for token on %d of %d chains (CA difference %+.3f, 90%% interval %.3f to "
                     "%.3f). This is not an identity." % (same, len(both), d[0], d[1], d[2]))
    gp = os.path.join(out, "gates.json")
    g_items = []
    if os.path.exists(gp):
        with open(gp, encoding="utf-8") as f:
            g_items = [g for g in json.load(f).get("items", []) if "tok_identical_kv4_cv3" in g]
    if g_items:
        k = sum(1 for g in g_items if g["tok_identical_kv4_cv3"].get("identical"))
        res["kv4_cv3"] = {"token_identical": k, "gate_items": len(g_items)}
        lines.append("kv4 equals cv3 token for token on %d of %d gate items; kv4 ran on the %d gate items only "
                     "(no CA)." % (k, len(g_items), len(g_items)))
    else:
        lines.append("kv4 was not run in this cell.")
    flo = [i for i in items if scr.get(i, {}).get("derived_B_in_prompt")]
    share = len(flo) / float(len(items)) if items else 0.0
    unedited = [a for a in ("floor", "clean", "base") if any(a in by[i] for i in items)]
    obs = {}
    for a in unedited:
        ia = [i for i in items if a in by[i]]
        obs[a] = round(sum(1 for i in ia if by[i][a].get("carry") == "B") / float(len(ia)), 4)
    res["carried_value_floor"] = {"items_derived_B_in_prompt_A": len(flo), "idx": flo,
                                  "carry_B_share_explained": round(share, 4), "carry_B_unedited_arms": obs}
    lines.append("Carried-value floor: derived_B is a number of prompt A on %d of %d items, which alone explains a "
                 "carry-B share of %.3f in unedited arms (observed %s). Prefer CA over carry-B." % (
                     len(flo), len(items), share, json.dumps(obs)))
    R0.write_json(os.path.join(out, "notes.json"), res)
    with open(os.path.join(out, "summary_notes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes_only", action="store_true")
    a = ap.parse_args(argv)
    if not a.notes_only:
        flags(a.out)
    notes(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
