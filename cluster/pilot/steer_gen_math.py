"""steer_gen_math: generation placement (steer_gen arm (a), run-1 rows) on the ASDiv and MAWPS pairs (RA-23 item 4).

prod/steer_gen.py and cluster/pilot/steer_gen_raven.py are imported UNCHANGED. Their arm (a) assumes a SETTLE task
(the A prompts are SETTLE task rows found by index, and a SETTLE cells file pre-filters the A side). ASDiv and MAWPS
have no SETTLE rows and no SETTLE cells, so this file changes only the data layer, in process:
  * a task named after the dataset is registered in prod.tasks with SVAMP's prompt, suffix, stops and parsers (the
    pairs were built in the SVAMP format) and rows = the pool's A problems (input = question_A, target = gold_A,
    idx = the pair's idx); the rows file is written into the setting folder (--rows-dir), bytes fixed by the pool,
    only when missing or different, through a temp file and a rename (write_rows_once);
  * steer_gen's dataset globals point at that task and at data_steer/<dataset>/pairs_<dataset>.jsonl;
  * the pool (core_pool) keeps one pair per problem whose prompts have equal length and differ only in the span on
    the four tokenizers (the builder's filters; there is no SETTLE text to match, so no text_exact rule);
  * prep keeps every pool problem in steer_gen's seeded shuffle order (no SETTLE pre-filter); the clean A side is
    screened by the model (screen_A_from_settle false), as on McLeish;
  * select (Ouro: mode select_full) scores every problem that passes both screens, as cluster/steer_select.py does.
Everything else (hooks, rows r00-r21, sentinels, shift rows, G.main generation, analyze) is steer_gen's or
steer_gen_raven's own code.

    python steer_gen_math.py --dataset asdiv --rows-dir OUT ouro    <prod.steer_gen arguments>
    python steer_gen_math.py --dataset asdiv --rows-dir OUT ouro    select_full --setting a1 --out OUT --config-json CFG
    python steer_gen_math.py --dataset asdiv --rows-dir OUT mcleish <steer_gen_raven arguments>   (--setting hug: Huginn)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOOPEDCOT = os.path.join(HERE, "tree")          # the bundle's prod tree: steer_gen and every prod module it imports
if LOOPEDCOT not in sys.path:
    sys.path.insert(0, LOOPEDCOT)

DATASETS = ("asdiv", "mawps")
TOKS = ("ouro", "huginn", "mcleish", "ifm")


def _args(argv):
    a = list(argv)
    ds = rows_dir = None
    while a and a[0].startswith("--"):
        k, v = a.pop(0), a.pop(0)
        if k == "--dataset":
            ds = v
        elif k == "--rows-dir":
            rows_dir = v
        else:
            raise SystemExit("unknown option %s" % k)
    if ds not in DATASETS or not rows_dir or not a or a[0] not in ("ouro", "mcleish"):
        raise SystemExit(__doc__)
    return ds, rows_dir, a[0], a[1:]


def write_rows_once(path, data):
    """The task rows file (bytes = S.write_jsonl's, fixed by the pool). Many row tasks start together and read it:
    an existing identical file is never rewritten; a missing (or differing) file is written to a per-process temp
    file and renamed into place, so a reader never sees a partial file. Returns True when it wrote."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            if f.read() == data:
                return False
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return True


def install(ds, rows_dir):
    """Point steer_gen at the dataset (task rows, pool, prep). Returns the steer_gen module."""
    os.environ["K17_DATASET"] = "svamp"          # import guard: steer_gen's own table knows gsm8k and svamp
    from prod import steer_gen as S
    pool_path = os.path.join(S.ROOT, "data_steer", ds, "pairs_%s.jsonl" % ds)
    rows = S.read_jsonl(pool_path)
    task_rows = [{"input": r["question_A"], "target": str(r["gold_A"]), "options": [], "gold_text": "",
                  "idx": int(r["idx"]), "src_idx": r.get("src_id", r["idx"])}
                 for r in sorted(rows, key=lambda r: int(r["idx"]))]
    os.makedirs(rows_dir, exist_ok=True)
    rows_path = os.path.join(rows_dir, "task_rows_%s.jsonl" % ds)
    write_rows_once(rows_path, "".join(json.dumps(r) + "\n" for r in task_rows).encode("utf-8"))
    S.register_task(ds, "svamp", rows_path)
    S._DATASETS[ds] = {"pool": (ds, "pairs_%s.jsonl" % ds), "task": ds, "splits": None}
    S.DATASET, S.TASK_A, S.POOL_DEFAULT = ds, ds, pool_path

    def core_pool(path=pool_path):
        rs = S.read_jsonl(path)
        core = [r for r in rs if all(r["tok"][t]["equal_len"] and r["tok"][t]["outside_equal"] for t in TOKS)]
        by = {}
        for r in core:
            by.setdefault(int(r["idx"]), []).append(r)
        return [sorted(v, key=lambda r: r["var_name"])[0] for _k, v in sorted(by.items())], len(core)
    S.core_pool = core_pool
    S.check_items.__defaults__ = (4, pool_path)
    S.prep_a = prep_nosettle(S, pool_path)
    return S


def prep_nosettle(S, pool_path):
    def prep_a(cfg, out_dir, settle_dir=None, think_dir=None, pool_path=pool_path):
        """steer_gen.prep_a without the SETTLE pre-filter: every pool problem in the seeded shuffle order."""
        from prod.tasks import rows as task_rows
        pool, n_core_rows = S.core_pool(pool_path)
        srows = {int(r["idx"]): r for r in task_rows(S.TASK_A)}
        tok = S.load_tokenizer(cfg["model"], think_dir)
        tok_base = tok if not cfg["chat"] else S.load_tokenizer("ouro_1_4b_base")
        by = {int(r["idx"]): r for r in pool}
        order = S.shuffled(by)
        items, bad = [], {}
        for idx in order:
            pr = by[idx]
            g = S.item_geometry(tok, pr, cfg["chat"], srows)
            if not cfg["chat"]:
                pool_span = pr["tok"]["ouro"]["span_A"]
                g["ok"]["matches_pool_span"] = list(g["span"]) == list(pool_span) and g["plen"] == pr["tok"]["ouro"]["len_A"]
            else:
                gb = S.item_geometry(tok_base, pr, False, srows)
                g["ok"]["base_method_matches_pool"] = list(gb["span"]) == list(pr["tok"]["ouro"]["span_A"])
            fails = [k for k, v in g["ok"].items() if not v]
            if fails:
                bad[idx] = fails
                continue
            items.append(item_record(pr, g))
        prep = {"setting": cfg["name"], "dataset": S.DATASET, "n_pool_rows": n_core_rows, "n_pool_problems": len(pool),
                "n_settle_correct": None, "settle_prefilter": "none (no SETTLE cells for this dataset)",
                "n_geometry_ok": len(items), "geometry_fail": bad, "pool_sha256": S.sha256_file(pool_path),
                "settle_file": None, "items": items}
        os.makedirs(out_dir, exist_ok=True)
        S.write_json(os.path.join(out_dir, "prep.json"), prep)
        return prep
    return prep_a


def item_record(pr, g):
    s0, s1 = g["span"]
    return {"idx": int(pr["idx"]), "var_name": pr["var_name"], "value_A": pr["value_A"], "value_B": pr["value_B"],
            "gold_A": str(pr["gold_A"]), "gold_B": str(pr["gold_B"]), "span": [s0, s1], "plen": g["plen"],
            "n_span_tokens": s1 - s0, "offset_from_end": g["plen"] - s1,
            "value_first_digit_differs": bool(pr["value_first_digit_differs"]),
            "first_digit_differs": bool(pr["first_digit_differs"]),
            "gold_B_chain_status": pr["gold_B_chain_status"], "n_digits": pr["n_digits"]}


def install_mcleish(S):
    """steer_gen_raven with the dataset installed: its prep without the SETTLE pre-filter, its hash line naming the
    dataset's pool. Both raven settings (mcl: McLeish, hug: Huginn) take the dataset; the pool span is read for the
    active setting's tokenizer."""
    import steer_gen_raven as R

    def prep(cfg, out_dir, settle_dir=None):
        from prod.tasks import rows as task_rows
        pool, n_core_rows = S.core_pool()
        srows = {int(r["idx"]): r for r in task_rows(S.TASK_A)}
        tok = R.load_tokenizer()
        by = {int(r["idx"]): r for r in pool}
        items, bad = [], {}
        for idx in S.shuffled(by):
            pr = by[idx]
            g = R.geometry(tok, pr, srows)
            ps = pr["tok"][R.spec()["pool_tok"]]
            g["ok"]["matches_pool_span"] = list(g["span"]) == list(ps["span_A"]) and g["plen"] == ps["len_A"]
            fails = [kk for kk, v in g["ok"].items() if not v]
            if fails:
                bad[idx] = fails
                continue
            items.append(item_record(pr, g))
        res = {"setting": cfg["name"], "dataset": S.DATASET, "n_pool_rows": n_core_rows, "n_pool_problems": len(pool),
               "n_settle_correct": None, "settle_prefilter": "none (no SETTLE cells for this dataset)",
               "n_geometry_ok": len(items), "geometry_fail": bad, "pool_sha256": S.sha256_file(S.POOL_DEFAULT),
               "settle_file": None, "items": items}
        os.makedirs(out_dir, exist_ok=True)
        S.write_json(os.path.join(out_dir, "prep.json"), res)
        return res

    def hash_line(cfg, root, settle_dir=None):
        import hashlib
        import time
        plan = os.environ.get("K17_PLAN")
        d = {"plan_sha256": S.sha256_file(plan) if plan and os.path.exists(plan) else None,
             "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest(),
             "code_sha256": {"steer_gen_math.py": S.sha256_file(os.path.abspath(__file__)),
                             "steer_gen_raven.py": S.sha256_file(R.__file__), "steer_gen.py": S.sha256_file(S.__file__)},
             "data_sha256": {os.path.basename(S.POOL_DEFAULT): S.sha256_file(S.POOL_DEFAULT)},
             "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        p = os.path.join(root, "hash_line.json")
        if not os.path.exists(p):
            S.write_json(p, d)
        return json.load(open(p, encoding="utf-8"))
    R.prep, R.hash_line = prep, hash_line
    for st in ("mcl", "hug"):
        R.S.SETTINGS[st]["settle"] = None
        R.S.SETTINGS[st]["dataset"] = S.DATASET
    return R


def select_full(S, argv):
    """cluster/steer_select.py in process: steer_gen select_a with n_scored unbounded; select_full_pool.json; fails
    below --min_scored passing problems."""
    import argparse
    ap = argparse.ArgumentParser(prog="steer_gen_math select_full")
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-json", default="")
    ap.add_argument("--min_scored", type=int, default=40)
    ap.add_argument("--max_scored", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config_json, encoding="utf-8")) if a.config_json else {}
    cfg = dict(cfg, n_scored=a.max_scored if a.max_scored > 0 else 100000)
    with open(os.path.join(a.out, "select_cfg_final.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
    sel = S.select_a(S.setting_cfg(a.setting, cfg), a.out)
    rec = {"n_scored": len(sel["scored"]), "filter_fails": sel.get("filter_fails") or {},
           "n_sentinels": len(sel.get("sentinels") or []), "n_shift": len(sel.get("shift") or []),
           "n_sent_config": cfg.get("n_sent"), "n_shift_config": cfg.get("n_shift"), "max_scored": a.max_scored}
    json.dump(rec, open(os.path.join(a.out, "select_full_pool.json"), "w"), indent=1)
    print(json.dumps(rec))
    if rec["n_scored"] < a.min_scored:
        raise SystemExit("only %d problems pass both filters, fewer than %d" % (rec["n_scored"], a.min_scored))
    return 0


def main(argv=None):
    ds, rows_dir, fam, rest = _args(sys.argv[1:] if argv is None else argv)
    S = install(ds, rows_dir)
    if fam == "ouro":
        if rest and rest[0] == "select_full":
            return select_full(S, rest[1:])
        if rest and rest[0] in ("anschain", "presence", "tune", "tf", "trainrows"):
            raise SystemExit("%s is not part of the ASDiv / MAWPS run" % rest[0])
        return S.main(rest)
    R = install_mcleish(S)
    return R.main(rest)


if __name__ == "__main__":
    sys.exit(main())
