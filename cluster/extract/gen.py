"""Extractor for loopedcot/prod/steer_gen.py arm-a output (generation placement, grid A).

Stored: analysis.json rows.<row>.<gold_B | gold_A | neither> = [mean, lo95, hi95, n_items, n_clusters] (boot_cluster,
1000 resamples, seed 0, cluster = problem). The 90 percent interval is recomputed from the per-item cells
(<row>/cells_*.jsonl, trace_answer) with steer_gen's own load_cells, outcome and boot_cluster, the quantile read
moved to (0.05, 0.95) (extract/resum.py). Random control: the random row at the same site (table below).
Positive control: r01 (clean B prompt, the swap ceiling).
"""
import json
import os
import sys

from . import common, resum

SCRIPT = "steer_gen.py"
# row -> (arm, site, loop, strength, random row at the same site); from rows_arm_a(k=4) (recon of run 1)
ROWS = {
    "r00": ("clean_A", "none", "", "", None), "r01": ("swap_B", "none", "", "", None),
    "r02": ("emb_patch", "emb|span", 1, 1.0, None), "r03": ("patch", "post|span", 1, 1.0, "r08"),
    "r19": ("patch", "post|span", 2, 1.0, None), "r04": ("patch", "post|span", 3, 1.0, "r14"),
    "r05": ("patch", "post|span", "1-4", 1.0, "r15"), "r06": ("patch", "post|last_prompt", 1, 1.0, "r16"),
    "r07": ("patch", "post|last_prompt", 4, 1.0, None), "r08": ("random_s0", "post|span", 1, "norm-matched", None),
    "r17": ("random_s1", "post|span", 1, "norm-matched", None), "r18": ("random_s2", "post|span", 1, "norm-matched", None),
    "r14": ("random_s0", "post|span", 3, "norm-matched", None), "r15": ("random_s0", "post|span", "1-4", "norm-matched", None),
    "r16": ("random_s0", "post|last_prompt", 1, "norm-matched", None),
    "r09": ("complete_patch", "post+kv|span_to_end", 1, 1.0, None), "r11": ("complete_patch", "post+kv|span_to_end", "1-3", 1.0, None),
    "r10": ("complete_patch", "post+kv|span_to_end", "1-4", 1.0, None), "r12": ("patch", "post|span_to_end", 1, 1.0, None),
    "r13": ("kv_patch", "kv|span", 1, 1.0, None), "r20": ("dup_clean_shifted", "all", "1-4", 0.0, None),
    "r21": ("self_donor", "post|span", 1, 1.0, None),
}
METRICS = ("gold_B", "gold_A", "neither")


def _sg(src_py):
    root = os.path.dirname(os.path.dirname(os.path.abspath(src_py)))     # .../loopedcot (holds prod/)
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    import importlib
    return importlib.import_module("prod.steer_gen")


def recompute(job_dir, src_py, ninety, keep=None):
    """Per row and metric [mean, lo, hi, n_items, n_clusters] from the per-item cells of the scored problems (keep:
    only the scored problems whose idx is in it, the shared-set readout of extract/shared.py)."""
    SG = _sg(src_py)
    sel = common.load_json(os.path.join(job_dir, "items.json"))
    scored = {int(it["idx"]): it for it in sel["scored"]}
    if keep is not None:
        keep = {int(i) for i in keep}
        scored = {i: it for i, it in scored.items() if i in keep}
    out = {}
    hits = {}
    for r in ROWS:
        d = os.path.join(job_dir, r)
        if not os.path.isdir(d):
            continue
        c = SG.load_cells(d)
        oc = {}
        for idx, it in scored.items():
            row = c.get(idx)
            if row is not None:
                oc[idx] = SG.outcome(row.get("trace_answer"), it["gold_A"], it["gold_B"])
        ids = sorted(oc)
        res = {}
        for mname, lab in (("gold_B", "B"), ("gold_A", "A"), ("neither", "neither")):
            xs = [int(oc[i] == lab) for i in ids]
            if ninety:
                with resum.ninety(hits):
                    res[mname] = SG.boot_cluster(xs, ids)
            else:
                res[mname] = SG.boot_cluster(xs, ids)
        out[r] = res
    return out, hits


def extract(job_dir, src_py, r90=None):
    an = common.load_json(os.path.join(job_dir, "analysis.json"))
    st = common.load_json(os.path.join(job_dir, "setting.json")) if os.path.exists(os.path.join(job_dir, "setting.json")) else {}
    model = st.get("model") or an.get("setting")
    dataset = st.get("dataset") or "gsm8k"
    if r90 is None:
        r90, _ = recompute(job_dir, src_py, True)
    pc = ((an["rows"].get("r01") or {}).get("gold_B") or [None])[0]
    rows = []
    for r, cell in an["rows"].items():
        if r not in ROWS:
            continue
        arm, site, loop, strength, rnd = ROWS[r]
        for mname in METRICS:
            v = cell.get(mname)
            if not v:
                continue
            v90 = r90[r][mname]
            rv = ((an["rows"].get(rnd) or {}).get(mname) or [None])[0] if rnd else None
            nr = (cell.get("neither") or [None])[0]
            rows.append(common.row(
                grid="A", card="generation placement", script=SCRIPT, model=model, dataset=dataset,
                arm="%s:%s" % (r, arm), site=site, loop=loop, strength=strength, metric=mname, endpoint="parsed_answer",
                item_set="own", n_items=v[3], n_clusters=v[4], estimate=v[0], neither_rate=nr, ci90_lo=v90[1], ci90_hi=v90[2],
                ci90_method="recomputed", ci95_lo=v[1], ci95_hi=v[2], random_control_estimate=rv,
                positive_control_value=pc if mname == "gold_B" else None,
                source_key="analysis.json:" + json.dumps(["rows", r, mname])))
    return rows


def item_sets(job_dir):
    sel = common.load_json(os.path.join(job_dir, "items.json"))
    planned = {int(it["idx"]) for it in sel["scored"]}
    out = {}
    for r in ROWS:
        d = os.path.join(job_dir, r)
        if not os.path.isdir(d):
            out[("row", r)] = (planned, set())
            continue
        done = set()
        for f in os.listdir(d):
            if f.startswith("cells_") and f.endswith(".jsonl"):
                for x in common.load_jsonl(os.path.join(d, f)):
                    if not x.get("_header") and x.get("row_idx") is not None:
                        done.add(int(x["row_idx"]))
        out[("row", r)] = (planned, done & planned)
    return out


def verify(job_dir, src_py):
    an = common.load_json(os.path.join(job_dir, "analysis.json"))
    r95, _ = recompute(job_dir, src_py, False)
    r90, hits = recompute(job_dir, src_py, True)
    rows = extract(job_dir, src_py, r90)
    n = eq = 0
    bad = []
    for x in rows:
        path = json.loads(x["source_key"].split(":", 1)[1])
        stv = an[path[0]][path[1]][path[2]]
        ext = [float(x["estimate"]), float(x["ci95_lo"]), float(x["ci95_hi"]), int(x["n_items"]), int(x["n_clusters"])]
        r = path[1]
        for lab, a, b in (("extracted", ext, stv), ("rerun95", r95[r][path[2]], stv), ("est90", r90[r][path[2]][0], stv[0])):
            n += 1
            if a == b:
                eq += 1
            elif len(bad) < 10:
                bad.append((lab, path, a, b))
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "quantile_reads_patched": hits}
