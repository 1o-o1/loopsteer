"""Extractor for k17_public.py and k17_parcae.py output (res_<runner>_<ds>.json records + --summarize checks.json).

The two scripts share one output format (e1_cells, e5_cells, sc_cells keyed "<runner>|<ds>"), so one module serves
both; FORMATS names the differences. Stored intervals are 95 percent ([mean, lo, hi] in each cell, n_items and
n_clusters beside it). The 90 percent interval comes from the script's own summarize re-run on a copy of the
records with the quantile read at (0.05, 0.95) (extract/resum.py): public bootc, 1000 resamples; Parcae boot,
2000 resamples; both seed 0, dyadic on two-unit clusters.

Rows:
  card edit_site_map:  e1_cells[key][site][src_O1 | ans_O1 | ans_O1_a0.25 | src_RND | ans_RND | src_O2 | ans_O2]
                       endpoint e_LO for GSM8K-copy source cells (the scripts' primary), else e.
  card input_refresh:  e5_cells[key][label] for labels that are cells (IN-single(j), IN-persist(j), VE-*, ...).
  card state_ceiling:  sc_cells[key][Llast@j] (public raven models only).
Random control: the same-site, same-position RND cell. Positive control: public, the state ceiling at the same
loop (SC_same_loop, raven j < K only); Parcae, src_informative_pc.value (source rows only). Both are what the
scripts label as their informative positive control; the bit-exact harness controls (pc_final, sz) are booleans.
"""
import glob
import json
import os
import re

from . import common, resum

DS = {"D1": "facts", "CF": "counterfact", "D2": "addition", "D4": "code", "D5": "twostep", "VB": "gsm8k_copy"}
FORMATS = {"k17_public.py": {"mod": "k17pub_x"}, "k17_parcae.py": {"mod": "k17par_x"}}
CELLS = ("src_O1", "ans_O1", "ans_O1_a0.25", "src_RND", "ans_RND", "src_O2", "ans_O2")
LOOP_RE = re.compile(r"\((\d+)\)")


def run_summarize(src_py, job_dir, ninety):
    script = os.path.basename(src_py)
    mod = resum.load_module(src_py, FORMATS.get(script, {"mod": "k17x"})["mod"])
    tmp = resum.copy_records(job_dir, ["res_*.json", "meta_*.json", "inputs_*.json", "fd_*.json"])
    for f in glob.glob(os.path.join(tmp, "res_*.shard*.json")):
        os.remove(f)
    hits = {}
    with resum.quiet():
        if ninety:
            with resum.ninety(hits):
                mod.summarize(tmp)
        else:
            mod.summarize(tmp)
    return common.load_json(os.path.join(tmp, "checks.json")), hits


def _primary(c, ds, pos):
    if ds == "VB" and pos == "src" and c.get("e_LO"):
        return "e_LO", c["e_LO"], c.get("n_LO", c["n_items"])
    return "e", c["e"], c["n_items"]


def _site(name):
    if "@" in name:
        a, b = name.split("@", 1)
        return a, int(b) if b.isdigit() else b
    return name, ""


def extract(job_dir, src_py, ck90=None):
    script = os.path.basename(src_py)
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    if ck90 is None:
        ck90, _ = run_summarize(src_py, job_dir, True)
    rows = []
    for key, sites in (ck.get("e1_cells") or {}).items():
        runner, ds = key.split("|")
        for sname, d in sites.items():
            if not isinstance(d, dict):
                continue
            layer, loop = _site(sname)
            for cn in CELLS:
                c = d.get(cn)
                if not isinstance(c, dict) or not c.get("e"):
                    continue
                pos, op = cn.split("_")[0], cn.split("_")[1]
                strength = cn.split("_a")[1] if "_a" in cn else 1
                ep, p, n = _primary(c, ds, pos)
                ep90, p90, _ = _primary(ck90["e1_cells"][key][sname][cn], ds, pos)
                rnd = d.get("%s_RND" % pos) if op != "RND" else None
                rp = _primary(rnd, ds, pos)[1] if isinstance(rnd, dict) and rnd.get("e") else None
                if script == "k17_parcae.py":
                    pcv = (d.get("src_informative_pc") or {}).get("value") if pos == "src" else None
                    pcv = pcv[0] if pcv else None
                else:
                    pcv = d.get("SC_same_loop")
                rows.append(common.row(
                    grid="B", card="edit_site_map", script=script, model=runner, dataset=DS.get(ds, ds), arm=op,
                    site="%s|%s" % (pos, layer), loop=loop, strength=strength, metric="e", endpoint=ep, item_set="own",
                    n_items=n, n_clusters=c["n_clusters"], estimate=p[0], ci90_lo=p90[1], ci90_hi=p90[2],
                    ci90_method="recomputed", ci95_lo=p[1], ci95_hi=p[2], random_control_estimate=rp[0] if rp else None,
                    positive_control_value=pcv, source_key="checks.json:" + json.dumps(["e1_cells", key, sname, cn, ep])))
    for tab, card in (("e5_cells", "input_refresh"), ("sc_cells", "state_ceiling")):
        for key, cells in (ck.get(tab) or {}).items():
            runner, ds = key.split("|")
            for lab, c in cells.items():
                if not isinstance(c, dict) or not c.get("e") or "n_items" not in c:
                    continue
                c90 = ck90[tab][key][lab]
                m = LOOP_RE.search(lab)
                loop = int(m.group(1)) if m else (_site(lab)[1] if "@" in lab else "")
                rows.append(common.row(
                    grid="C" if card == "input_refresh" else "B", card=card, script=script, model=runner,
                    dataset=DS.get(ds, ds), arm=lab, site=lab.split("(")[0] if "(" in lab else _site(lab)[0], loop=loop,
                    strength=1, metric="e", endpoint="e", item_set="own", n_items=c["n_items"],
                    n_clusters=c["n_clusters"], estimate=c["e"][0], ci90_lo=c90["e"][1], ci90_hi=c90["e"][2],
                    ci90_method="recomputed", ci95_lo=c["e"][1], ci95_hi=c["e"][2],
                    source_key="checks.json:" + json.dumps([tab, key, lab, "e"])))
    return rows


def item_sets(job_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(job_dir, "res_*.json"))):
        if ".shard" in f:
            continue
        r = common.load_json(f)
        runner, ds = r["model"], r["ds"]
        planned = set()
        for pid, st in r.get("scan") or []:
            # k17_public scans runners jointly: an item is planned when every runner of the group kept it
            ok = all(v == "kept" for v in st.values()) if isinstance(st, dict) else st == "kept"
            if ok:
                planned.add(pid)
        skipped = (r.get("meta") or {}).get("skipped")
        done = {it["pair_id"] for it in r.get("items") or []}
        out[(runner, DS.get(ds, ds))] = (planned if not skipped else set(), done)
    return out


def _get(d, parts):
    cur = d
    for p in parts:
        cur = cur[p]
    return cur


def verify(job_dir, src_py):
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    ck95, _ = run_summarize(src_py, job_dir, False)
    ck90, hits = run_summarize(src_py, job_dir, True)
    rows = extract(job_dir, src_py, ck90)
    n = eq = 0
    bad = []
    for r in rows:
        parts = json.loads(r["source_key"].split(":", 1)[1])
        path, ep = parts[:-1], parts[-1]
        st = _get(ck, path)
        rr = _get(ck95, path)
        r9 = _get(ck90, path)
        stv = st[ep]
        ext = [float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"])]
        for lab, a, b in (("extracted", ext, stv), ("rerun95", rr[ep], stv), ("est90", r9[ep][0], stv[0]),
                          ("n", int(r["n_clusters"]), st["n_clusters"])):
            n += 1
            if a == b:
                eq += 1
            elif len(bad) < 10:
                bad.append((lab, "/".join(path), a, b))
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "quantile_reads_patched": hits}
