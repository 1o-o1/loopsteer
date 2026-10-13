"""RA-30 Thinking smoke on a small local card (plan 6e; not a run of record): Ouro-1.4B-Thinking, a few GSM8K
problems of data/pool_gsm8k.json, a short horizon. Screens the items (clean chain, gold-intermediate match, token
count and consistency rules), runs the gates and every arm on the first eligible items, and writes
out/smoke_think_laptop/smoke.json and smoke.md.

    python smoke_think.py --think_dir DIR [--n 8] [--n_arms 2] [--horizon 768] [--max_batch 2]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "lib"), os.path.join(HERE, "tree"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import ra30 as R0  # noqa: E402
import ra30_data as RD  # noqa: E402
import ra30_think as T  # noqa: E402


def mark(text, span):
    a, b = span
    return text[:a] + "[[" + text[a:b] + "]]" + text[b:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n_arms", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=768)
    ap.add_argument("--max_batch", type=int, default=2)
    ap.add_argument("--mem_fraction", type=float, default=0.92)
    a = ap.parse_args(argv)
    out = os.path.join(HERE, "out", "smoke_think_laptop")
    os.makedirs(out, exist_ok=True)
    T.HORIZON = R0.HORIZON = int(a.horizon)
    import transformers
    dev = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__, "transformers": transformers.__version__, "horizon": a.horizon,
           "max_batch": a.max_batch, "model": "ouro_1_4b_think", "revision": T.MODELS["ouro_1_4b_think"]["revision"]}
    t0 = time.time()
    ctx = T.load_ctx("ouro_1_4b_think", "gsm8k", a.think_dir, mem_fraction=a.mem_fraction)
    dev["load_seconds"] = round(time.time() - t0, 1)
    items, rec = T.load_pool("ouro_1_4b_think", "gsm8k", pool_local=True)
    by = {int(it["idx"]): it for it in items}
    order = RD.draw_order(list(by), "gsm8k", "ouro_1_4b_think")[:a.n]
    screens, drops = [], {}
    for i in order:
        r = T.screen_item(ctx, by[i])
        T.clear_item(ctx)
        screens.append(r)
        k = r["drop"] or "eligible"
        drops[k] = drops.get(k, 0) + 1
        R0.log("screen %d: %s (%s s, %s tokens)" % (i, k, r.get("sec_clean"), r.get("n_clean")))
    elig = [r for r in screens if r["drop"] is None]
    examples = []
    for r in screens:
        if r.get("clean_ids") is None:
            continue
        _ids, _c, txt, own = T.cut_parse(ctx, r["clean_ids"])
        ex = {"idx": r["idx"], "drop": r["drop"], "own": own, "gold_A": by[r["idx"]]["gold_A"],
              "n_clean": r.get("n_clean")}
        if r["drop"] is None:
            pre = ctx.tok.decode(r["prefix_A"], clean_up_tokenization_spaces=False)
            ex.update(derived_A=r["derived_A"], derived_B=r["derived_B"], P=r["P"], tokens_A=r["tokens_A"],
                      tokens_B=r["tokens_B"], chain_marked=mark(txt, (len(pre) - len(r["derived_A"]), len(pre))))
        else:
            ex["chain_head"] = txt[:600]
        examples.append(ex)
    armed, gates = [], []
    for r in elig[:a.n_arms]:
        it = by[r["idx"]]
        gates.append(T.gates_item(ctx, it, r))
        t1 = time.time()
        rows, meta = T.run_item(ctx, it, r, max_batch=a.max_batch)
        T.clear_item(ctx)
        ref = {x["arm"]: x["chain_ids"] for x in rows}
        armed.append({"idx": r["idx"], "seconds": round(time.time() - t1, 1),
                      "arms": {x["arm"]: {"outcome": x["outcome"], "carry": x["carry"], "own": x["own"],
                                          "n_chain": x["n_chain"], "div_from_ref": x["div_from_ref"]} for x in rows},
                      "identity_cvP_cv1_tokens": ref["cvP"] == ref["cv1"],
                      "text_tail": next(x["text"][-300:] for x in rows if x["arm"] == "text")})
        R0.log("arms %d: %s" % (r["idx"], {x["arm"]: x["outcome"] for x in rows}))
    res = {"device": dev, "n_screened": len(screens), "drops": drops,
           "match_rate": round(len(elig) / max(1, len(screens)), 3), "examples": examples, "gates": gates,
           "armed": armed, "seconds_total": round(time.time() - t0, 1),
           "sec_clean_mean": round(sum(r["sec_clean"] for r in screens if r.get("sec_clean")) /
                                   max(1, sum(1 for r in screens if r.get("sec_clean"))), 1)}
    R0.write_json(os.path.join(out, "smoke.json"), res)
    lines = ["# RA-30 Thinking smoke (laptop card, not a run of record)", "", "Device: %s" % json.dumps(dev), "",
             "Screened %d, eligible %d (match rate %.2f); drops %s" % (len(screens), len(elig), res["match_rate"],
                                                                        json.dumps(drops)), ""]
    for g in gates:
        lines.append("Gates idx %d: G1 %s, G2 %s, G3 %s, G4 %s, G5 %s, pass %s" % (
            g["idx"], g["G1"], g["G2"], g["G3"], g["G4"], g["G5_patch_lands"], g["pass"]))
    for x in armed:
        lines.append("Arms idx %d (%s s): %s; cvP = cv1 tokens %s" % (
            x["idx"], x["seconds"], json.dumps({k: v["outcome"] for k, v in x["arms"].items()}),
            x["identity_cvP_cv1_tokens"]))
    for ex in examples:
        if ex["drop"] is None:
            lines += ["", "Example idx %d: derived_A %s -> derived_B %s, P %s" % (ex["idx"], ex["derived_A"],
                                                                                   ex["derived_B"], ex["P"]),
                      "```", ex["chain_marked"][:1500], "```"]
    with open(os.path.join(out, "smoke.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
