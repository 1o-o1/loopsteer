"""Extractor for k17_varbench.py output (vb_<model>.json records + the --summarize checks.json).

Rows written:
  card edit_site_map, dataset gsm8k_copy:    checks.placement.<m>."<op>|<pos>|<site>"   (op O1, RND, O2)
  card edit_site_map, dataset gsm8k_derived: checks.placement_vbd.<m>."<op>|<pos>|<site>"
  card site_selection, every dataset:        checks.carry_gain.<m>.<ds>."a=<s>".test_dm_over_D.<selector>
Estimate, n and the 95 percent interval are the stored values; the list is [mean, lo, hi, n_items, n_units].
The 90 percent interval comes from the script's own summarize re-run on a copy of the records with the
quantile read at (0.05, 0.95) (extract/resum.py). Random control: the same-site RND cell (rnd_pair.rnd_primary)
for O1 cells, the random_site selector for carry gain. Positive control: pc_final.<m>.e (VarBench), placement_vbd
pc_final.e (derived), carry_gain pc_final.e (carry gain).
"""
import glob
import os
import re

from . import common, resum

SCRIPT = "k17_varbench.py"
DS_NAME = {"VarBench": "gsm8k_copy", "VBD": "gsm8k_derived", "D1": "facts_vb", "D2": "addition_vb",
           "D4": "templates_vb", "D4/retrieval": "templates_vb_retrieval", "D4/computed": "templates_vb_computed"}
SITE_RE = re.compile(r"^(L0|Lmid|Llast)@(\d+)$")


def run_summarize(src_py, job_dir, ninety):
    mod = resum.load_module(src_py, "k17vb_x")
    tmp = resum.copy_records(job_dir, ["vb_*.json", "fd_*.json", "bf16_*.json", "window_*.json", "states_*.pt",
                                       "proc_*.json"])
    hits = {}
    with resum.quiet():
        if ninety:
            with resum.ninety(hits):
                mod.summarize(tmp)
        else:
            mod.summarize(tmp)
    ck = common.load_json(os.path.join(tmp, "checks.json"))
    return ck, hits, tmp


def _site(name):
    m = SITE_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else (name, "")


def _cells(tab):
    for key, c in tab.items():
        if isinstance(c, dict) and "primary" in c and key.count("|") == 2:
            yield key, c


def extract(job_dir, src_py, ck90=None):
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    if ck90 is None:
        ck90, _, _ = run_summarize(src_py, job_dir, True)
    rows = []
    for tabname, ds in (("placement", "gsm8k_copy"), ("placement_vbd", "gsm8k_derived")):
        for m, tab in (ck.get(tabname) or {}).items():
            pc = (ck["pc_final"][m] if tabname == "placement" else tab.get("pc_final") or {}).get("e")
            for key, c in _cells(tab):
                op, pos, sname = key.split("|")
                layer, loop = _site(sname)
                p = c["primary"]
                p90 = ck90[tabname][m][key]["primary"]
                rnd = (c.get("rnd_pair") or {}).get("rnd_primary") if op == "O1" else None
                rows.append(common.row(
                    grid="B", card="edit_site_map", script=SCRIPT, model=m, dataset=ds, arm=op, site="%s|%s" % (pos, layer),
                    loop=loop, strength=1, metric="e", endpoint=c["endpoint"], item_set="own", n_items=p[3], n_clusters=p[4],
                    estimate=p[0], ci90_lo=p90[1], ci90_hi=p90[2], ci90_method="recomputed", ci95_lo=p[1], ci95_hi=p[2],
                    random_control_estimate=rnd[0] if rnd else None, positive_control_value=pc[0] if pc else None,
                    source_key="checks.%s.%s.%s.primary" % (tabname, m, key)))
    for m, per_ds in (ck.get("carry_gain") or {}).items():
        for dsk, d in per_ds.items():
            if not isinstance(d, dict):
                continue
            pc = (d.get("pc_final") or {}).get("e")
            for a in ("a=1", "a=0.25"):
                t = ((d.get(a) or {}).get("test_dm_over_D")) or {}
                t90 = ((ck90["carry_gain"][m][dsk].get(a) or {}).get("test_dm_over_D")) or {}
                rnd = t.get("random_site")
                for sel, v in t.items():
                    if not isinstance(v, list) or len(v) < 5:
                        continue
                    v90 = t90[sel]
                    rows.append(common.row(
                        grid="D", card="site_selection", script=SCRIPT, model=m, dataset=DS_NAME.get(dsk, dsk), arm=sel,
                        site="selected", loop="", strength=a.split("=")[1], metric="dm_over_D", endpoint="test_split", item_set="own",
                        n_items=v[3], n_clusters=v[4], estimate=v[0], ci90_lo=v90[1], ci90_hi=v90[2],
                        ci90_method="recomputed", ci95_lo=v[1], ci95_hi=v[2],
                        random_control_estimate=rnd[0] if (rnd and not sel.startswith("predicted_minus")) else None,
                        positive_control_value=pc[0] if pc else None,
                        source_key="checks.carry_gain.%s.%s.%s.test_dm_over_D.%s" % (m, dsk, a, sel)))
    return rows


def item_sets(job_dir):
    """{(model, dataset): (planned ids, done ids)} from the records. Planned = the scan's clean-pass set (VarBench,
    VBD) or the selected list (D1, D2, D4); done = items with a finished record."""
    out = {}
    for f in sorted(glob.glob(os.path.join(job_dir, "vb_*.json"))):
        d = common.load_json(f)
        m = d["model"]
        vb = d.get("varbench") or {}
        planned = {k for k, v in (vb.get("scan") or {}).items() if v.get("A_pass") and v.get("B_pass")}
        done = {k for k, v in (vb.get("items") or {}).items() if v.get("done_M")}
        out[(m, "gsm8k_copy")] = (planned, done)
        vd = d.get("vbd") or {}
        if vd.get("items") is not None:
            planned = {k for k, v in (vd.get("scan") or {}).items() if v.get("A_pass") and v.get("B_pass")}
            n_t = (vd.get("config") or {}).get("vbd_n")
            done = {k for k, v in vd["items"].items() if v.get("done")}
            out[(m, "gsm8k_derived")] = (planned if not n_t else set(vd["items"].keys()), done)
        for ds, a in (d.get("a2") or {}).items():
            out[(m, DS_NAME.get(ds, ds))] = (set((a.get("items") or {}).keys()),
                                             {k for k, v in (a.get("items") or {}).items() if v.get("done")})
    return out


def verify(job_dir, src_py):
    """Stored values against (a) the extracted rows and (b) the script's own summarize re-run at 95 percent.
    Returns (n_compared, n_equal, first mismatches)."""
    ck = common.load_json(os.path.join(job_dir, "checks.json"))
    ck95, _, _ = run_summarize(src_py, job_dir, False)
    ck90, hits, _ = run_summarize(src_py, job_dir, True)
    rows = extract(job_dir, src_py, ck90)
    n = eq = 0
    bad = []
    for r in rows:
        path = r["source_key"].split(".", 1)[1]
        stored = _get(ck, path)
        rerun = _get(ck95, path)
        r90 = _get(ck90, path)
        ext = [float(r["estimate"]), float(r["ci95_lo"]) if r["ci95_lo"] else None,
               float(r["ci95_hi"]) if r["ci95_hi"] else None, int(r["n_items"]), int(r["n_clusters"])]
        for lab, a, b in (("extracted", ext, stored), ("rerun95", rerun, stored), ("est90", r90[0], stored[0]),
                          ("n90", r90[3:5], stored[3:5])):
            n += 1
            if a == b:
                eq += 1
            elif len(bad) < 10:
                bad.append((lab, path, a, b))
    return {"rows": len(rows), "compared": n, "equal": eq, "mismatches": bad, "quantile_reads_patched": hits}


def _get(d, path):
    # path segments are separated by '.', but cell keys contain '.' only inside 'a=0.25' and 'D4/...': walk greedily
    parts = path.split(".")
    cur, i = d, 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if isinstance(cur, dict) and k in cur:
                cur, i = cur[k], j
                break
        else:
            raise KeyError(path)
    return cur
