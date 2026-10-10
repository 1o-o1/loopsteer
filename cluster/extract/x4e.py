"""Extractor for k17_x4e.py output (results.json from --summarize, items_D1.jsonl, items_D2.jsonl).

Rows written (card carried_state, grid C):
  results.cells.<facts|addition>["<split>[:<stratum>]|<base>|<rep>"].<retention|dm_over_D>
    = {mean, ci90, ci95, n_items, n_defined, n_clusters[, dyadic{ci90, ci95, n_subjects}]}
  arm = "replace_<rep>" (mean = dev-pool mean state, don = three dev donors averaged per item);
  site/loop from base: le<j> -> r|Llast loop j, src<j> -> src|Llast loop j, mid4 -> r|Lmid loop 4.
  item_set: split all -> own, test -> test, all:<s> -> stratum:<s>, test:<s> -> test_stratum:<s>.
  facts (all|, test| keys): the dyadic interval is primary (stored_dyadic, n_clusters = n_subjects); the A-side
  cluster interval is written beside it as endpoint a_side_cluster.
  results.datasets.<name>.pc_final_e and .sz_e as arms pc_final (r|Llast loop 4) and sz (src|Llast loop 4).
n_items = n_defined. Positive control = datasets.<name>.pc_final_e mean. No random control in this script.
Strength 1.0 (full replacement write).
"""
import os

from . import common, resum

SCRIPT = "k17_x4e.py"
DSNAME = {"facts": "facts", "addition": "addition"}
FILE = {"facts": "D1", "addition": "D2"}
OURO_MODEL = {"ByteDance/Ouro-1.4B": "ouro", "ByteDance/Ouro-2.6B": "ouro26"}
METRIC = {"retention": "retention", "dm_over_D": "e"}


def _src(src_py):
    return src_py or os.path.join(os.environ.get("LOOP_FLOW_SRC", ""), SCRIPT)


def _site(base):
    if base.startswith("le"):
        return "r|Llast", int(base[2:])
    if base.startswith("src"):
        return "src|Llast", int(base[3:])
    if base == "mid4":
        return "r|Lmid", 4
    return "", ""


def _iset(split):
    if ":" in split:
        sp, s = split.split(":", 1)
        return ("stratum:%s" % s) if sp == "all" else ("%s_stratum:%s" % (sp, s))
    return "own" if split == "all" else split


def _stat_rows(s, key, base, ep_name, dyadic_primary):
    out = []
    if dyadic_primary and "dyadic" in s:
        dy = s["dyadic"]
        out.append(dict(endpoint=ep_name, n_items=s["n_defined"], n_clusters=dy["n_subjects"], estimate=s["mean"],
                        ci90_lo=dy["ci90"][0], ci90_hi=dy["ci90"][1], ci90_method="stored_dyadic",
                        ci95_lo=dy["ci95"][0], ci95_hi=dy["ci95"][1], source_key=key + ".dyadic", **base))
        ep_name = "a_side_cluster"
    out.append(dict(endpoint=ep_name, n_items=s["n_defined"], n_clusters=s["n_clusters"], estimate=s["mean"],
                    ci90_lo=s["ci90"][0], ci90_hi=s["ci90"][1], ci90_method="stored", ci95_lo=s["ci95"][0],
                    ci95_hi=s["ci95"][1], source_key=key, **base))
    return out


def extract(job_dir, src_py=None):
    res = common.load_json(os.path.join(job_dir, "results.json"))
    model = OURO_MODEL.get(res["config"]["model"], res["config"]["model"])
    rows = []
    for name, cells in res["cells"].items():
        dsinfo = res["datasets"][name]
        pc = (dsinfo.get("pc_final_e") or {}).get("mean")
        common_kw = dict(grid="C", card="carried_state", script=SCRIPT, model=model, dataset=DSNAME.get(name, name),
                         strength=1.0, positive_control_value=pc)
        for ck, c in cells.items():
            split, b, rep = ck.split("|")
            site, loop = _site(b)
            for ep in ("retention", "dm_over_D"):
                s = c.get(ep)
                if not isinstance(s, dict):
                    continue
                base = dict(arm="replace_%s" % rep, site=site, loop=loop, metric=METRIC[ep], item_set=_iset(split),
                            **common_kw)
                for kw in _stat_rows(s, "results.cells.%s.%s.%s" % (name, ck, ep), base, ep, True):
                    rows.append(common.row(**kw))
        for arm, key, site in (("pc_final", "pc_final_e", "r|Llast"), ("sz", "sz_e", "src|Llast")):
            s = dsinfo.get(key)
            if not isinstance(s, dict):
                continue
            base = dict(arm=arm, site=site, loop=4, metric="e", item_set="own", **common_kw)
            for kw in _stat_rows(s, "results.datasets.%s.%s" % (name, key), base, "dm_over_D", True):
                rows.append(common.row(**kw))
    return rows


def selection(job_dir, sel_dir=None):
    """{D: planned keys} from the selection files the run was given (--sel_dir = the branch-freeze full dir). The
    job dir holds no copy; the file is accepted only when its sha256 equals config.selection_sha256[D]. Looked up in
    sel_dir, then job_dir, then the sibling <job_dir>/../../k17_freeze/full. Missing or mismatched -> not returned."""
    import hashlib
    cfg = common.load_json(os.path.join(job_dir, "config.json"))
    want = cfg.get("selection_sha256") or {}
    dirs = [d for d in (sel_dir, job_dir, os.path.join(job_dir, "..", "..", "k17_freeze", "full")) if d]
    out = {}
    for ds, sha in want.items():
        for d in dirs:
            p = os.path.join(d, "selection_%s.json" % ds)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() == sha:
                        out[ds] = {it["key"] for it in common.load_json(p)["items"]}
                        break
    return out


def item_sets(job_dir, sel_dir=None):
    """{(model, dataset): (planned keys, done keys)}. planned = selection(job_dir, sel_dir) (sha-checked; empty set
    when the selection file is not found); done = keys with a record in items_<D>.jsonl. Sets collapse the one
    duplicated D2 key."""
    res = common.load_json(os.path.join(job_dir, "results.json"))
    model = OURO_MODEL.get(res["config"]["model"], res["config"]["model"])
    sel = selection(job_dir, sel_dir)
    out = {}
    for name, ds in FILE.items():
        p = os.path.join(job_dir, "items_%s.jsonl" % ds)
        if not os.path.exists(p):
            continue
        done = {r["key"] for r in common.load_jsonl(p)}
        out[(model, DSNAME[name])] = (sel.get(ds, set()), done)
    return out


def _walk(res, path):
    # cell keys contain '|' and ':' but no '.', so a plain split works
    cur = res
    for p in path.split("."):
        cur = cur[p]
    return cur


def _stored(res, key):
    path = key.split(".", 1)[1]
    if path.endswith(".dyadic"):
        s = _walk(res, path[:-len(".dyadic")])
        dy = s["dyadic"]
        return s["mean"], dy["ci90"], dy["ci95"], s["n_defined"], dy["n_subjects"]
    s = _walk(res, path)
    return s["mean"], s["ci90"], s["ci95"], s["n_defined"], s["n_clusters"]


def _f(x):
    return float(x) if x not in ("", None) else None


def verify(job_dir, src_py=None):
    res = common.load_json(os.path.join(job_dir, "results.json"))
    rows = extract(job_dir, src_py)
    n = eq = 0
    bad = []

    def cmp(lab, a, b):
        nonlocal n, eq
        n += 1
        if a == b:
            eq += 1
        elif len(bad) < 10:
            bad.append((lab, a, b))

    for r in rows:
        k = r["source_key"]
        est, c90, c95, ni, nc = _stored(res, k)
        cmp(k + ":estimate", _f(r["estimate"]), float(est))
        cmp(k + ":ci90", [_f(r["ci90_lo"]), _f(r["ci90_hi"])], [float(x) for x in c90])
        cmp(k + ":ci95", [_f(r["ci95_lo"]), _f(r["ci95_hi"])], [float(x) for x in c95])
        cmp(k + ":n_items", int(r["n_items"]), int(ni))
        cmp(k + ":n_clusters", int(r["n_clusters"]), int(nc))
    # recompute with the script's own cell() (item_value + stat -> k17_ko240 boot_dist / q / tost_twoway)
    X = resum.load_module(_src(src_py), "k17x4e_x")
    KO = X.KOmod()
    recomputed = []
    for name, ds in FILE.items():
        p = os.path.join(job_dir, "items_%s.jsonl" % ds)
        if not os.path.exists(p) or name not in res["cells"]:
            continue
        recs = common.load_jsonl(p)
        for ck in ("all|le1|don", "test|le2|mean"):
            split, b, rep = ck.split("|")
            rs = recs if split == "all" else [r for r in recs if not r["dev"]]
            got = X.cell(KO, rs, b, rep, ds == "D1")
            st = res["cells"][name][ck]
            for ep in ("retention", "dm_over_D"):
                cmp("recompute:%s.%s.%s" % (name, ck, ep), got[ep], st[ep])
            recomputed.append("cells.%s.%s" % (name, ck))
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "recomputed": recomputed}
