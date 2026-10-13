"""RA-30 data build (CPU, laptop, once, before any GPU step; plan_ra30.md section 3).

    python make_data.py            pools, precompute splits, sources.json
    python make_data.py --prescreen   also the CPU prescreen on the Table 2 single-card clean chains (estimates)
    python make_data.py --model ouro_2_6b_base   only the 2.6B GSM8K pool, order and sources file (plan 6b); the
                                                 1.4B files are not touched

Pools = the scored problems of the Table 2 generation run of setting a1 (Ouro-1.4B base) on the single card
(steer_gen select, every problem with own answer gold_A on prompt A and gold_B on prompt B). The cluster lists of
Table 2 (391 GSM8K, 1,034 ASDiv) are not on this machine; the single-card lists are (395, 1,036). SVAMP and MAWPS are
dropped (RA brief update 2026-10-12: too few multi-step chains).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), os.path.join(HERE, "tree")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PROD_DEV", "cpu")

import ra30_data as RD  # noqa: E402

LF = os.path.dirname(os.path.dirname(HERE))           # loop-flow
CL = os.path.join("artifacts_spark", "cluster_local")
SOURCES = {
    "gsm8k": {"items": os.path.join(CL, "ra20_g2", "A", "steer_gen", "a1_gsm8k", "items.json"),
              "chains": os.path.join(CL, "ra20_g2", "A", "steer_gen", "a1_gsm8k", "r00",
                                     "chains_ouro_1_4b_base_gsm8k_k4.jsonl")},
    "asdiv": {"items": os.path.join(CL, "ra20_tail2", "A", "steer_gen_math", "a1_asdiv", "items.json"),
              "chains": os.path.join(CL, "ra20_tail2", "A", "steer_gen_math", "a1_asdiv", "screen_A",
                                     "chains_ouro_1_4b_base_asdiv_k4.jsonl")},
}
SOURCES_MODEL = {"ouro_2_6b_base": {
    "gsm8k": os.path.join(CL, "ra20_g2", "A", "steer_gen", "a2_gsm8k", "items.json"),
    "asdiv": os.path.join(CL, "ra20_tail2", "A", "steer_gen_math", "a2_asdiv", "items.json")},
    # McLeish r32 (plan 6c): the steer_gen_raven scored list (237; the same file as the cluster bundle's
    # pilot/data/mcleish_setting/items.json); spans and plen count McLeish's BOS
    "mcleish_llama32_r32": {"gsm8k": os.path.join(CL, "g2_gen", "mcleish", "items.json")}}
FIELDS = ("idx", "value_A", "value_B", "gold_A", "gold_B", "span", "plen", "var_name")


def rel(p):
    return p.replace(os.sep, "/")


def build(ds):
    pairs = {}
    for r in RD.read_jsonl(RD.pairs_path(ds)):
        pairs.setdefault(int(r["idx"]), []).append(r)
    src = SOURCES[ds]
    sel = json.load(open(os.path.join(LF, src["items"]), encoding="utf-8"))
    items = [{k: it[k] for k in FIELDS} for it in sel["scored"]]
    source = {"kind": "Table 2 single-card scored list (setting a1)", "path": rel(src["items"]),
              "sha256": RD.sha256_file(os.path.join(LF, src["items"]))}
    for it in items:
        cand = [r for r in pairs[int(it["idx"])] if r["var_name"] == it["var_name"]]
        if len(cand) != 1:
            raise SystemExit("%s idx %s: %d pairs with var_name %s" % (ds, it["idx"], len(cand), it["var_name"]))
        it["question_A"], it["question_B"] = cand[0]["question_A"], cand[0]["question_B"]
    source["pairs"] = {"path": rel(os.path.relpath(RD.pairs_path(ds), HERE)), "sha256": RD.sha256_file(RD.pairs_path(ds))}
    out = {"dataset": ds, "n": len(items), "source": source, "items": items}
    with open(os.path.join(RD.DATA, "pool_%s.json" % ds), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    split = {"dataset": ds, "key": RD.SPLIT_KEY % ds, "n_split": RD.SPLIT_N,
             "rule": "split = the first n_split[stage] eligible items of `order`; scored = the eligible items after them",
             "order": RD.draw_order([it["idx"] for it in items], ds)}
    with open(os.path.join(RD.DATA, "split_%s.json" % ds), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    return out


def load_tok():
    from transformers import AutoTokenizer
    from prod import config as cfgmod
    tok = AutoTokenizer.from_pretrained("ByteDance/Ouro-1.4B", trust_remote_code=True,
                                        revision=cfgmod.revision_for("ouro_1_4b_base"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def prescreen(ds, pool, tok, n_show=3):
    """The RA-30 screen run on the Table 2 single-card clean chains (G.main, batch 16): an ESTIMATE of the kept
    counts per drop reason, and the T2 printout (derived-value tokens of A and B) for the first n_show kept items.
    The run screens its own batch-1 chains; nothing here selects an item."""
    import chain as CH
    src = SOURCES[ds]["chains"]
    if not src:
        return None
    chains = {int(r["idx"]): r for r in RD.read_jsonl(os.path.join(LF, src))}
    counts, shown, n_seen = {}, [], 0
    for it in pool["items"]:
        cr = chains.get(int(it["idx"]))
        if cr is None:
            counts["no_chain"] = counts.get("no_chain", 0) + 1
            continue
        n_seen += 1
        _c, txt, own = RD.cut_parse(tok, ds, cr["chain_ids"])
        if CH.outcome(own, it["gold_A"], it["gold_B"]) != "A":
            counts["clean_wrong"] = counts.get("clean_wrong", 0) + 1
            continue
        rec = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"])
        if rec["drop"] is None:
            pids = RD.prompt_ids(tok, ds, it["question_A"])
            if len(pids) != int(it["plen"]):
                rec["drop"] = "plen_mismatch"
            else:
                rec = CH.token_geometry(tok, pids, cr["chain_ids"], rec)
        key = rec["drop"] or "kept"
        counts[key] = counts.get(key, 0) + 1
        alt = CH.analyze(txt, it["question_A"], it["value_A"], it["value_B"], it["gold_A"], it["gold_B"],
                         mode="r1")
        if alt["drop"] is None:
            alt = CH.token_geometry(tok, RD.prompt_ids(tok, ds, it["question_A"]), cr["chain_ids"], alt)
        if alt["drop"] is None:
            counts["kept_r1_info"] = counts.get("kept_r1_info", 0) + 1
        if key == "kept" and len(shown) < n_show:
            shown.append({"idx": it["idx"], "P": rec["P"], "tokens_A": rec["tokens_A"], "tokens_B": rec["tokens_B"],
                          "derived_A": rec["derived_A"], "derived_B": rec["derived_B"], "e1": rec["e1"],
                          "answer_edit": rec["answer_edit"], "gold_B": it["gold_B"],
                          "prefix_tail": tok.decode(rec["prefix_A"][-12:])})
    out = {"dataset": ds, "chains": rel(src), "n_pool": pool["n"], "n_with_chain": n_seen, "counts": counts,
           "kept": counts.get("kept", 0), "shown": shown,
           "note": "estimate on the Table 2 batch-16 chains; the run re-screens its own batch-1 chains"}
    with open(os.path.join(RD.DATA, "prescreen_%s.json" % ds), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def build_model(ds, model):
    """The Table 2 single-card scored list of another model (setting a2 = Ouro-2.6B base): pool, order, sources.
    Writes only data/pool_<ds>_<model>.json and split_<ds>_<model>.json; returns the sources entry."""
    pairs = {}
    for r in RD.read_jsonl(RD.pairs_path(ds)):
        pairs.setdefault(int(r["idx"]), []).append(r)
    src = SOURCES_MODEL[model][ds]
    sel = json.load(open(os.path.join(LF, src), encoding="utf-8"))
    items = [{k: it[k] for k in FIELDS} for it in sel["scored"]]
    for it in items:
        cand = [r for r in pairs[int(it["idx"])] if r["var_name"] == it["var_name"]]
        if len(cand) != 1:
            raise SystemExit("%s idx %s: %d pairs with var_name %s" % (ds, it["idx"], len(cand), it["var_name"]))
        it["question_A"], it["question_B"] = cand[0]["question_A"], cand[0]["question_B"]
    sfx = RD.model_suffix(model)
    kind = ("Table 2 single-card scored list (setting a2, %s)" % model if model == "ouro_2_6b_base"
            else "steer_gen_raven single-card scored list (%s)" % model)
    source = {"kind": kind, "path": rel(src),
              "sha256": RD.sha256_file(os.path.join(LF, src)),
              "pairs": {"path": rel(os.path.relpath(RD.pairs_path(ds), HERE)),
                        "sha256": RD.sha256_file(RD.pairs_path(ds))}}
    pool = {"dataset": ds, "model": model, "n": len(items), "source": source, "items": items}
    pf, sf = "pool_%s%s.json" % (ds, sfx), "split_%s%s.json" % (ds, sfx)
    with open(os.path.join(RD.DATA, pf), "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=1)
    split = {"dataset": ds, "model": model, "key": RD.order_key(ds, model), "n_split": RD.SPLIT_N,
             "rule": "split = the first n_split[stage] eligible items of `order` (none while the neuron arms are off)",
             "order": RD.draw_order([it["idx"] for it in items], ds, model)}
    with open(os.path.join(RD.DATA, sf), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    return {"files": {f: RD.sha256_file(os.path.join(RD.DATA, f)) for f in (pf, sf)}, "source": source,
            "n": len(items)}


def build_gold(parquet):
    """data/gold_gsm8k.json (plan 6e): the GSM8K test solutions (openai/gsm8k 740312a, main/test), {row index:
    answer text}, and data/sources_think.json (the gold file and the two pairs files the Thinking cells read)."""
    import pandas as pd
    df = pd.read_parquet(parquet)
    rows = RD.read_jsonl(os.path.join(HERE, "tree", "prod", "tasks", "data", "rows_gsm8k.jsonl"))
    if len(df) != 1319 or any(df.iloc[int(r["idx"])]["question"] != r["input"] for r in rows):
        raise SystemExit("the GSM8K parquet does not match the task rows")
    gold = {str(i): str(a) for i, a in enumerate(df["answer"])}
    gp = os.path.join(RD.DATA, "gold_gsm8k.json")
    with open(gp, "w", encoding="utf-8") as f:
        json.dump({"source": "openai/gsm8k 740312add88f781978c0658806c59bc2815b9866 main/test",
                   "parquet_sha256": RD.sha256_file(parquet), "n": len(gold), "answers": gold}, f, indent=0)
    src = {"files": {"gold_gsm8k.json": RD.sha256_file(gp)},
           "pairs": {ds: {"path": rel(os.path.relpath(RD.pairs_path(ds), HERE)), "sha256": RD.sha256_file(RD.pairs_path(ds))}
                     for ds in ("gsm8k", "asdiv")}}
    with open(os.path.join(RD.DATA, "sources_think.json"), "w", encoding="utf-8") as f:
        json.dump(src, f, indent=1, sort_keys=True)
    print(json.dumps({"gold_n": len(gold)}))
    return 0


def main(argv):
    if "--gold" in argv:
        return build_gold(argv[argv.index("--gold") + 1])
    if "--model" in argv:
        model = argv[argv.index("--model") + 1]
        if model not in SOURCES_MODEL:
            raise SystemExit("no pool source for %s" % model)
        out = {"files": {}, "pool_sources": {}, "pool_n": {}, "model": model}
        for ds in sorted(SOURCES_MODEL[model]):
            b = build_model(ds, model)
            out["files"].update(b["files"])
            out["pool_sources"][ds] = b["source"]
            out["pool_n"][ds] = b["n"]
        with open(os.path.join(RD.DATA, RD.sources_name(model)), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print(json.dumps({"model": model, "pool_n": out["pool_n"]}))
        return 0
    os.makedirs(RD.DATA, exist_ok=True)
    pools = {ds: build(ds) for ds in RD.DATASETS}
    if "--prescreen" in argv:
        tok = load_tok()
        for ds in RD.DATASETS:
            r = prescreen(ds, pools[ds], tok)
            if r:
                print(ds, "pool", r["n_pool"], "kept (estimate)", r["kept"], json.dumps(r["counts"], sort_keys=True))
    files = sorted(f for f in os.listdir(RD.DATA) if f.endswith(".json") and f != "sources.json")
    src = {"files": {f: RD.sha256_file(os.path.join(RD.DATA, f)) for f in files},
           "pool_sources": {ds: pools[ds]["source"] for ds in RD.DATASETS},
           "pool_n": {ds: pools[ds]["n"] for ds in RD.DATASETS}}
    with open(os.path.join(RD.DATA, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(src, f, indent=1, sort_keys=True)
    print(json.dumps(src["pool_n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
