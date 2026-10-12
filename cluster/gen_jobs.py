"""RA-20 Amendment 2 item 2: the array generator for the cluster run.

Reads cluster/manifest.yaml (the manifest, transcribed) and cluster/cells_skipped.json (time, ability and design
skips per cell), routes every run/scan cell to the pilot script that loads its (model, dataset), and writes:

  jobs/joblist.tsv           one line per job, in queue order: id, group, stage, script, model, datasets, shard,
                             out, depends_on, est_card_seconds, rate_source, cmd
  jobs/<group>.<stage>.tsv   the task list of one Slurm array (one cmd per line, line i = array task i)
  jobs/array_<group>.sbatch  the array template (placeholders: partition, account, $REPO, $WEIGHTS_ROOT, $HF_HOME,
                             $OUT_ROOT, $VENV); it runs line $SLURM_ARRAY_TASK_ID of the task list it is given
  jobs/submit_00_xllm.sh     submits only the xllm build-and-import job (group g0_xllm)
  jobs/submit_10_rest.sh     refuses to run unless that job left its done marker, then submits every other array in
                             queue order with afterok dependencies between stages
  jobs/planned_n.csv, .md    the planned n per cell (manifest pool, script cap, latest kept count in the pilot
                             artifacts); every value is PLANNED, not measured
  jobs/cells.csv             every manifest cell with its final word: run (job ids), done, scan (job ids), skip
                             with reason, scope with reason, or pending with what blocks it
  jobs/card_hours.md, .json  card-hours per queue group at 30 cards, one process per card, from measured
                             per-item rates (seconds per item on the fast host; rates measured on shared GPUs marked)

No host, user or home string is written: paths are $-placeholders resolved inside the job. The pilot scripts run
from the bundle copies in cluster/pilot/ (paths from the environment, pinned revisions, role label instead of the
host name); their inputs are in cluster/pilot/data/ ($DATA_ROOT); the raven branch-freeze imports the frozen prod
tree cluster/pilot/tree/. The planned n per cell reads the pilot artifacts named by $PILOT_ARTIFACTS (optional;
without it planned_n falls back to the pools and quotas).

    PILOT_ARTIFACTS=<pilot artifact folder> python cluster/gen_jobs.py [--cards 30] [--shard_hours 1.5]
"""
import argparse
import collections
import csv
import glob
import json
import math
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")
ART = os.environ.get("PILOT_ARTIFACTS", "")
WEIGHTS = json.load(open(os.path.join(HERE, "weights.json"), encoding="utf-8"))
LOC = {c["key"]: c["location"] for c in WEIGHTS["checkpoints"]}
THINK14, THINK26 = LOC["ouro_1_4b_think"], LOC["ouro_2_6b_think"]     # hub snapshot folders (pinned revisions)

# ----------------------------------------------------------------------------------------------- names per script
VB_KEY = {"ouro_1_4b_base": "ouro", "ouro_2_6b_base": "ouro26", "ouro_1_4b_think": "ouro_think",
          "ouro_2_6b_think": "ouro26_think", "ifm_p1_ouro": "ifm_p1_ouro", "ifm_p1_ouro_inj": "ifm_p1_ouro_inj",
          "ifm_d112": "ifm_d112"}
PUB_GROUP = {"mcleish_llama32_r32": "mcl_nr", "llama_nr": "mcl_nr", "huginn_0125": "huginn",
             "ouro_1_4b_think": "ouro_think"}
PUB_RUNNER = {"mcleish_llama32_r32": "mcleish", "llama_nr": "llama_nr", "huginn_0125": "huginn",
              "ouro_1_4b_think": "ouro_think"}
PUB_DS = {"mcl_nr": ("D1", "CF", "D2", "D4", "D5", "VB"), "huginn": ("D1", "CF", "D2", "D4", "D5", "VB"),
          "ouro_think": ("D1", "D2", "D4")}
PAR_KEY = {"parcae_370m": "parcae-370m", "parcae_1_3b": "parcae-1.3b"}
K15F = {"ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_d112"}     # k15_full.py: D1, D2, CF in one process (its summarize
#                                                            keeps MODEL_ORDER models only, so not ifm_d28: time skip)
K15P = {"ifm_p1_mid", "ifm_p1_mid_inj", "ifm_p2m_r5", "ifm_p2m_diag", "ifm_p2m_ortho"}   # k15_pairs.py: D1
K16O = {"ouro_1_4b_base": "ouro", "ouro_2_6b_base": "ouro26", "ouro_1_4b_think": "ouro_think"}
K16I = {"ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_p1_mid", "ifm_p1_mid_inj", "ifm_d112", "ifm_p2m_diag"}
GEN = {"ouro_1_4b_base": "a1", "ouro_2_6b_base": "a2", "ouro_1_4b_think": "a3", "ouro_2_6b_think": "a4"}
THINK_OF = {"a3": THINK14, "a4": THINK26}          # Thinking settings: --think-dir and K17_THINK_DIR (hub folder)
# cluster extension (2026-10-11): pipelines added while the run was queued get their own queue groups, so the stage
# lists of the running groups keep every line where it is (array task i = line i)
GEN_GROUP = {"ouro_2_6b_think": "g11_gen26t"}
# B-EXT2 item 5: the IFM k17_varbench pipelines rerun with the VBD V1 cells (--vbd_v1) in their own group and folder
VB_V1 = {"ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_d112"}
VB_V1_GROUP = "g12_ifmv1"
# B-EXT2 item 1: k16_path_ouro (Y1 attention block, Y3 source restore, W2 last-read window) on the Thinking models
PATH_CARD = "pathway checks (restore, read window, attention block)"
K16T = {"ouro_1_4b_think": "ouro_think", "ouro_2_6b_think": "ouro26_think"}
# B-EXT2 item 3: edit-site map cells whose k16_path_ifm run had no Y1 arm; Y1 reruns into <model>_<ds>_y1
Y1_CELLS = {("ifm_p1_ouro_inj", "D4"), ("ifm_p1_mid", "D2"), ("ifm_p1_mid", "D4"), ("ifm_p1_mid_inj", "D2"),
            ("ifm_p1_mid_inj", "D4"), ("ifm_p1_mid_inj", "D5"), ("ifm_p2m_diag", "D2")}
FIRST_RUN_CARD = "IFM pathway rows (first run)"     # item 1: setting a4, GSM8K and SVAMP (steer_gen), ASDiv and MAWPS
RAVEN_FREEZE = {"mcleish_llama32_r32": "mcleish", "huginn_0125": "huginn"}
# RA-23 additions (2026-10-09)
FRZ_OURO = {"ouro_1_4b_think": "ouro_think", "ouro_2_6b_think": "ouro26_think"}   # k17_freeze --model <id>; value =
#                                                                                   extractor model name (freeze.py)
FRZ_IFM = {"ifm_p1_ouro"}           # k17_freeze_ifm.py: the RA-3 rows on the IFM code path (no-injection model)
FRZ_DS = ("D1", "D2", "D4")         # RA-23 item 1 and 2 scope: facts, addition, code
GENR = {"mcleish_llama32_r32": "mcleish", "huginn_0125": "huginn"}   # steer_gen_raven.py (GSM8K pairs, rows r00-r21, K = 32)
GENR_DIR = "A/steer_gen_raven/mcleish"
GENR_HUG_PREP, GENR_HUG_N = 151, 124   # B-EXT2 item 4: Huginn prep items (shipped prep.json), planned scored


def genr_dir(model):
    """Output folder of a steer_gen_raven model: McLeish's, or A/steer_gen_raven/huginn (B-EXT2 item 4)."""
    return GENR_DIR if model == "mcleish_llama32_r32" else "A/steer_gen_raven/" + GENR[model]
GENM_DS = {"ASDIV_gen": "asdiv", "MAWPS_gen": "mawps"}       # steer_gen_math.py --dataset
GENM_POOL = {"asdiv": 1114, "mawps": 2082}                  # pairs = problems (one pair per problem)
GENM_PASS = 0.6     # planning only: share of the pool passing both screens, assumed (no full screen yet)         # setting folder: items.json from the shared-host run's select, one folder per row
GENR_ROWS = ["r00", "r01", "r02", "r20", "r21", "r10", "r03", "r04", "r05", "r06", "r07", "r08", "r09", "r11", "r12",
             "r13", "r19", "r14", "r15", "r16", "r17", "r18"]           # steer_gen_raven SETTINGS["mcl"]["rows"]
GEN_ROWS = ["r00", "r01", "r02", "r10", "r20", "r21", "r03", "r04", "r05", "r06", "r07", "r08", "r09", "r11", "r12",
            "r13", "r19", "r14", "r15", "r16", "r17", "r18"]          # run_a1.sh order


def route(card, model, ds):
    """(script, unit) of the pilot script that produces this cell, or None. Edit-site map routing follows the
    coordinator's 2026-10-08 decision: k15_full.py (FULL-A design) where it loads the model, k15_pairs.py for the
    FULL-B pairs, k17_public / k17_parcae for raven and Parcae; the k16 path scripts only for the pathway rows the
    manifest asks for (two-step math on Ouro, code and two-step math on IFM), as cells_skipped.json assumed."""
    if card == "GSM8K copy and derived":
        if model in VB_KEY:
            return ("k17_varbench.py", "vb")
        if ds == "VB" and model in PUB_GROUP and model != "ouro_1_4b_think":
            return ("k17_public.py", "pub")
        if ds == "VB" and model in PAR_KEY:
            return ("k17_parcae.py", "par")
        return None
    if card == "generation placement":
        if model in GENR and ds == "GSM8K_gen":
            return ("steer_gen_raven.py", "genr")
        if model in GEN and ds in GENM_DS:          # RA-23 item 4: ASDiv and MAWPS through steer_gen_math.py
            return ("steer_gen_math.py", "genm")
        return ("steer_gen.py", "gen") if model in GEN and ds in ("GSM8K_gen", "SVAMP_gen") else None
    if card == PATH_CARD:
        return ("k16_path_ouro.py", "k16t") if model in K16T and ds in ("D1", "D2", "D4", "D5") else None
    if card == FIRST_RUN_CARD:          # the first k16_path_ifm folders of the Y1 cells: read by the merge, no job
        return ("k16_path_ifm.py", "k16i0") if (model, ds) in Y1_CELLS else None
    if card == "edit-site map" and ds == "LRE":       # RA-23 item 5: the k17_public facts path on the LRE pool files
        return ("k17_public.py", "publre") if model in PUB_GROUP else None
    if card == "edit-site map":
        if model in PUB_GROUP and ds in PUB_DS[PUB_GROUP[model]]:
            return ("k17_public.py", "pub")
        if model in PAR_KEY and ds in ("D1", "VB"):
            return ("k17_parcae.py", "par")
        if model in K15F and ds in ("D1", "D2", "CF"):
            return ("k15_full.py", "k15f")
        if model in K15P and ds == "D1":
            return ("k15_pairs.py", "k15p")
        if model in K16O and ds == "D5":
            return ("k16_path_ouro.py", "k16o")
        if model in K16I and ds in ("D2", "D4", "D5") and not (model == "ifm_p2m_diag" and ds != "D2") \
                and not (model in K15F and ds == "D2"):
            return ("k16_path_ifm.py", "k16i")
        return None
    if card == "branch-freeze" and model in RAVEN_FREEZE and ds in ("D1", "D2"):
        return ("k17_freeze_raven.py", "frz")
    if card == "branch-freeze" and model in FRZ_OURO and ds in FRZ_DS:
        return ("k17_freeze.py", "frzo")
    if card == "branch-freeze" and model in FRZ_IFM and ds in FRZ_DS:
        return ("k17_freeze_ifm.py", "frzi")
    return None


def vb_dir(model):
    """Output folder name of a k17_varbench model: <key>_v1 for the IFM reruns with the VBD V1 cells (item 5)."""
    return VB_KEY[model] + ("_v1" if model in VB_V1 else "")


DS_NAME = {"D1": "facts", "CF": "counterfact", "D2": "addition", "D4": "code", "D5": "twostep", "VB": "gsm8k_copy",
           "VBD": "gsm8k_derived"}


def cell_target(card, model, ds):
    """Where a routed cell's records land and how its extractor names it: (unit, out dir under $OUT_ROOT,
    model key or None for any, dataset key). merge.py uses this to find a cell's results and item sets."""
    r = route(card, model, ds)
    if r is None:
        return None
    u = r[1]
    if u == "vb":
        return (u, "B/k17_varbench/%s" % vb_dir(model), VB_KEY[model], DS_NAME[ds])
    if u == "pub":
        return (u, "B/k17_public/%s_%s" % (PUB_GROUP[model], ds), PUB_RUNNER[model], DS_NAME[ds])
    if u == "par":
        return (u, "B/k17_parcae/%s" % PAR_KEY[model], PAR_KEY[model], DS_NAME[ds])
    if u == "gen":
        return (u, "A/steer_gen/%s_%s" % (GEN[model], "gsm8k" if ds == "GSM8K_gen" else "svamp"), "row", None)
    if u == "k15f":
        return (u, "B/k15_full/%s" % model, model, DS_NAME[ds])
    if u == "k15p":
        return (u, "B/k15_pairs/%s" % model, model, DS_NAME[ds])
    if u == "k16o":
        return (u, "B/k16_path_ouro/%s_D5" % K16O[model], K16O[model], DS_NAME[ds])
    if u == "k16i":
        return (u, "B/k16_path_ifm/%s_%s%s" % (model, ds, "_y1" if (model, ds) in Y1_CELLS else ""), model, DS_NAME[ds])
    if u == "k16i0":
        return (u, "B/k16_path_ifm/%s_%s" % (model, ds), model, DS_NAME[ds])
    if u == "k16t":
        return (u, "C/k16_path_ouro/%s_%s" % (K16T[model], ds), K16T[model], DS_NAME[ds])
    if u == "frz":
        return (u, "C/k17_freeze_raven/%s_%s" % (RAVEN_FREEZE[model], ds), None, DS_NAME[ds])
    if u == "frzo":
        return (u, "C/k17_freeze/%s_%s" % (FRZ_OURO[model], ds), FRZ_OURO[model], DS_NAME[ds])
    if u == "frzi":
        return (u, "C/k17_freeze_ifm/%s_%s" % (model, ds), model, DS_NAME[ds])
    if u == "genr":
        return (u, genr_dir(model), "row", None)
    if u == "genm":
        return (u, "A/steer_gen_math/%s_%s" % (GEN[model], GENM_DS[ds]), "row", None)
    if u == "publre":        # the records say D1 (facts); the LRE cell is its own output folder
        return (u, "B/k17_public/%s_LRE" % PUB_GROUP[model], PUB_RUNNER[model], "facts")
    return None


# ----------------------------------------------------------------------------------------------- manifest
def load_manifest(path):
    m = yaml.safe_load(open(path, encoding="utf-8"))
    order = [x["id"] for x in m["models"]]
    last = {x["id"] for x in m["models"] if x.get("last")}
    cells = []
    for e in m["cells"]:
        for mod in (e.get("models") or [e["model"]]):
            for ds, v in e["cells"].items():
                st, note = (v[0], v[1]) if isinstance(v, list) else (v, "")
                cells.append({"grid": e["grid"], "card": e["card"], "model": mod, "dataset": ds, "status": st,
                              "note": note})
    return m, order, last, cells


def load_skips(path):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for c in d["cells"]:
        for mod in [x.strip() for x in c["model"].split(",")]:
            out[(c["card"], mod, c["dataset"])] = c
    return out


# ----------------------------------------------------------------------------------------------- measured rates
# Seconds per item on one card, from the pilot artifacts (recon 2026-10-08). "fast" = the fast host card measured
# alone; "shared" = measured on a GPU shared with other jobs (an upper bound for a dedicated card).
RATES = {
    # k17_varbench phase M per VarBench item / per VBD item (per-item seconds_M, seconds); phase W per subset item
    ("vb", "ouro"): (12.3, 10.2, "shared", "k17_varbench full: ouro_think on the fast host (co-tenants in every snapshot); same architecture"),
    ("vb", "ouro26"): (21.8, 21.5, "shared", "k17_varbench full: ouro26, fast host, co-tenants"),
    ("vb", "ouro_think"): (12.3, 10.2, "shared", "k17_varbench full: ouro_think, fast host, co-tenants"),
    ("vb", "ouro26_think"): (27.5, 16.6, "shared", "k17_varbench full: ouro26_think, fast host, co-tenants"),
    ("vb", "ifm_p1_ouro"): (18.5, 10.1, "other", "k17_varbench full: ifm_p1_ouro_inj on the second host, no co-tenant seen; same architecture"),
    ("vb", "ifm_p1_ouro_inj"): (18.5, 10.1, "other", "k17_varbench full: ifm_p1_ouro_inj, second host, no co-tenant seen"),
    ("vb", "ifm_d112"): (18.6, 10.1, "other", "k17_varbench full: ifm_d112, second host, no co-tenant seen"),
    # steer_gen run 1 rows, seconds per item per row at batch 16, horizon 512
    ("gen", "a1"): (0.62, None, "fast", "k17_gen a1: six rows that ran alone, 0.53-0.70 s/item"),
    ("gen", "a2"): (1.45, None, "shared", "k17_gen a2: 1.3-1.6 s/item, card shared with run b2"),
    ("gen", "a3"): (4.96, None, "assumed", "no Thinking placement run exists: a1 rate x 8 (horizon 4096 against 512), not measured"),
    ("gen", "a4"): (11.6, None, "assumed", "no Ouro-2.6B-Thinking placement run: a2 rate x 8 (horizon 4096 against 512), not measured"),
    # k17_public seconds per item (t_item mean) per runner group and dataset; scan seconds per candidate separate
    ("pub", "mcl_nr", "D1"): (171.0, "shared", "k17_public full: mcleish 153 + llama_nr 18 s/item (E8 full search on; off here)"),
    ("pub", "mcl_nr", "D2"): (84.7, "shared", "k17_public full: mcleish 81.2 + llama_nr 3.5"),
    ("pub", "mcl_nr", "D4"): (33.6, "shared", "k17_public full: mcleish 30.7 + llama_nr 2.9"),
    ("pub", "mcl_nr", "VB"): (89.2, "shared", "k17_public full: mcleish 80.6 + llama_nr 8.6"),
    ("pub", "mcl_nr", "CF"): (171.0, "assumed", "no CF run: the D1 rate (same prompt type)"),
    ("pub", "mcl_nr", "D5"): (84.7, "assumed", "no D5 run: the D2 rate (same prompt length class)"),
    ("pub", "huginn", "D1"): (270.7, "shared", "k17_public full: huginn D1, shared GPU"),
    ("pub", "huginn", "D2"): (101.7, "fast", "k17_public full: huginn D2, GPU alone"),
    ("pub", "huginn", "D4"): (101.7, "assumed", "no D4 run: the D2 rate"),
    ("pub", "huginn", "D5"): (101.7, "assumed", "no D5 run: the D2 rate"),
    ("pub", "huginn", "CF"): (270.7, "assumed", "no CF run: the D1 rate"),
    ("pub", "huginn", "VB"): (270.7, "assumed", "no VB run in the artifact: the D1 rate"),
    ("pub", "ouro_think", "D1"): (12.3, "assumed", "no k17_public Ouro run: the k17_varbench Ouro per-item rate"),
    ("pub", "ouro_think", "D2"): (12.3, "assumed", "same"),
    ("pub", "ouro_think", "D4"): (12.3, "assumed", "same"),
    # k17_parcae t_item
    ("par", "parcae-370m", "D1"): (4.9, "shared", "k17_parcae full_final"),
    ("par", "parcae-370m", "VB"): (12.2, "shared", "k17_parcae full_final"),
    ("par", "parcae-1.3b", "D1"): (11.0, "shared", "k17_parcae full_final"),
    ("par", "parcae-1.3b", "VB"): (149.6, "shared", "k17_parcae full_final (process unfinished)"),
    # whole-job seconds (one process per model, all datasets) from k15_fullA / k15_fullB
    ("k15f", "ifm_p1_ouro"): (3.8, "shared", "k15_fullA k15f_ifm_p1_ouro: 1778.2 s for 467 pairs (D1 80, D2 80, CF 307)"),
    ("k15f", "ifm_p1_ouro_inj"): (11.3, "shared", "k15_fullA: CF 11.3 s/pair on ifm_p1_ouro_inj (upper bound for D1, D2)"),
    ("k15f", "ifm_d112"): (8.9, "shared", "k15_fullA: CF 8.9 s/pair on ifm_d112 (upper bound for D1, D2)"),
    ("k15p", "*"): (0.9, "shared", "k15_fullB: 5.98 GPU-minutes for five models of about 80 pairs"),
    # k16 path scripts, seconds per item
    ("k16o", "ouro"): (8.55, "fast", "k16_path_ouro D5 on ouro, fast host (another job ran at the same time)"),
    ("k16o", "ouro26"): (12.7, "fast", "k16_path_ouro ouro26, fast host, shared with the D4/D5 run"),
    ("k16o", "ouro_think"): (6.24, "fast", "k16_path_ouro ouro_think D5, fast host"),
    ("k16i", "*"): (10.6, "shared", "k16_path_ifm: highest per-item rate (Y1, Y3, W2 models), shared host"),
    # k17_freeze_raven: no artifact; k17_freeze on Ouro 1.4B is 0.5 s/item on the fast card; raven forwards cost
    # about 3x (McLeish clean forward 0.34 s against about 0.1 s for Ouro): assumed
    ("frz", "*"): (1.5, "assumed", "no raven freeze run: k17_freeze 0.5 s/item x 3, not measured"),
    # RA-23. k17_freeze RA-3 full run: 490.6 s for 850 items on the fast card (0.58 s/item, scan included)
    ("frzo", "ouro_think"): (0.6, "fast", "k17_freeze RA-3 full run on Ouro-1.4B base: 490.6 s for 850 items (same architecture)"),
    ("frzo", "ouro26_think"): (1.2, "assumed", "the 1.4B rate x 2 (twice the layers), not measured"),
    ("frzi", "*"): (4.4, "shared", "k17_freeze_ifm smoke on the shared host: 4.3-4.6 s/item (3 items per dataset)"),
    ("genr", "huginn"): (25.0, "shared", "steer_gen_raven Huginn smoke screen B on the shared GB10: 74.6 s for 3 problems at batch 3"),
    ("genr", "mcleish"): (19.5, "shared", "steer_gen_raven row r00 on the shared host: 4616 s for 237 scored + 22 sentinel problems at batch 16"),
}


# ----------------------------------------------------------------------------------------------- latest kept counts
def latest_kept():
    """(script unit, script model key, dataset) -> (kept, scanned or pool, source) from the pilot artifacts."""
    out = {}

    def jl(p):
        if not ART:
            return None
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    ck = jl(os.path.join(ART, "k17_varbench", "full", "checks.json")) or {}
    for m, it in (ck.get("items") or {}).items():
        out[("vb", m, "VB")] = (it.get("kept"), it.get("scanned"), "k17_varbench/full checks.items (s200 pool)")
        v = it.get("vbd") or {}
        if v:
            out[("vb", m, "VBD")] = (v.get("kept"), None, "k17_varbench/full checks.items.vbd (vbd_n cap)")
    for art, unit in (("k17_public/full", "pub"), ("k17_parcae/full_final", "par")):
        for f in (glob.glob(os.path.join(ART, art, "res_*.json")) if ART else []):
            r = jl(f)
            if not r:
                continue
            st = (r.get("meta") or {}).get("scan_stats") or {}
            kept = st.get("joint_kept", st.get("kept", len(r.get("items") or [])))
            out[(unit, r["model"], r["ds"])] = (kept, st.get("scanned"), art + " " + os.path.basename(f))
    # quota scripts: (kept, scanned, source, candidates in the pool) from the pilot records; a truncated scan
    # (scanned missing) takes the pass rate of the same dataset on the untruncated model
    for f in (sorted(glob.glob(os.path.join(ART, "k15_fullA", "k15f_*.json"))) if ART else []):
        d = jl(f) or {}
        for ds, v in (d.get("datasets") or {}).items():
            n = len(v.get("items") or [])
            sc = v.get("scanned")
            out[("k15f", d.get("model"), ds)] = (n, sc, "k15_fullA " + os.path.basename(f), v.get("candidates"))
    for (u, m, ds), v in list(out.items()):
        if u == "k15f" and not v[1]:
            ref = [w for (u2, m2, d2), w in out.items() if u2 == "k15f" and d2 == ds and w[1]]
            if ref:
                out[(u, m, ds)] = (ref[0][0], ref[0][1], v[2] + " (scan truncated: pass rate of %s)" % ref[0][2], v[3])
    for f in (sorted(glob.glob(os.path.join(ART, "k15_fullB", "k15p_*.json"))) if ART else []):
        d = jl(f) or {}
        sc = (d.get("d1_scan") or {})
        for ds, v in (d.get("datasets") or {}).items():
            scanned = v.get("scanned")
            scanned = sum(scanned.values()) if isinstance(scanned, dict) else scanned
            out[("k15p", d.get("model"), ds)] = (len(v.get("items") or []), scanned, "k15_fullB " + os.path.basename(f),
                                                 sc.get("n") if ds == "D1" else None)
    for f in (sorted(glob.glob(os.path.join(ART, "k16_path_ouro", "k16p_*.json"))) if ART else []):
        d = jl(f) or {}
        for ds, v in (d.get("datasets") or {}).items():
            out[("k16o", d.get("model"), ds)] = (len(v.get("items") or []), v.get("scanned"),
                                                 "k16_path_ouro " + os.path.basename(f), v.get("candidates"))
    for f in (sorted(glob.glob(os.path.join(ART, "k16_path_ifm", "k16_*.json")) +
                     glob.glob(os.path.join(ART, "k16_path_ifm", "d45", "k16_*.json"))) if ART else []):
        d = jl(f) or {}
        for ds, v in (d.get("datasets") or {}).items():
            out[("k16i", d.get("model"), ds)] = (v.get("n"), v.get("scanned"), "k16_path_ifm " + os.path.basename(f),
                                                 v.get("candidates"))
    for st, m in (("a1", "a1"), ("a2", "a2")):
        it = jl(os.path.join(ART, "k17_gen", st, "items.json"))
        pr = jl(os.path.join(ART, "k17_gen", st, "prep.json"))
        if it:
            ff = it.get("filter_fails") or {}
            n_scr = it.get("n_screened_B") or 240
            passed = n_scr - ff.get("A_wrong", 0) - ff.get("B_wrong", 0)
            n_prep = len((pr or {}).get("items") or [])
            out[("gen", m, "GSM8K_gen")] = (it.get("n_scored"), n_prep, "k17_gen/%s items.json; B pass %d of %d screened" % (st, passed, n_scr))
    return out


# ----------------------------------------------------------------------------------------------- jobs
class Jobs(object):
    def __init__(self, done=()):
        self.rows = []
        self.done = {(d["script"], d["model"], d["out"], d["stage"], d["shard"]): d for d in done}

    def add(self, group, stage, script, model, datasets, shard, out, deps, est, rate_src, cmd):
        jid = "%s.%s.%03d" % (group, stage, sum(1 for r in self.rows if r["group"] == group and r["stage"] == stage) + 1)
        row = {"id": jid, "group": group, "stage": stage, "script": script, "model": model,
               "datasets": ",".join(datasets), "shard": shard, "out": out, "depends_on": ",".join(deps),
               "est_card_seconds": int(round(est)), "rate_source": rate_src, "cmd": cmd}
        d = self.done.get((script, model, out, stage, shard))
        if d is not None:            # ran on our cards (done_local.json): keeps its id, leaves the task lists
            if d.get("id") and d["id"] != jid:
                raise SystemExit("done_local.json: %s is %s in this job list (ids moved)" % (d["id"], jid))
            row["done"] = d
        self.rows.append(row)
        return jid

    def split_done(self):
        """Remove the done jobs (returned) from the list and from every dependency."""
        done = [r for r in self.rows if r.get("done")]
        ids = {r["id"] for r in done}
        self.rows = [r for r in self.rows if not r.get("done")]
        for r in self.rows:
            r["depends_on"] = ",".join(x for x in r["depends_on"].split(",") if x and x not in ids)
        return done


def n_shards(total_s, shard_s, cap=16):
    return max(1, min(cap, int(math.ceil(total_s / shard_s))))


PY = "python -u"
LC = "$REPO/loopedcot"
LF = LC + "/cluster/pilot"          # pilot scripts: the bundle copies (cluster/pilot/), hashed in jobs/scripts_sha256.txt
TREE = LF + "/tree"                 # the bundle's prod tree: steer_gen.py, every prod module and data file it reads, and
                                    # data_steer/ (the pair files); generation lines run `python -m prod.steer_gen` there
DATA = "$DATA_ROOT"                 # pilot inputs: cluster/pilot/data/ unless set otherwise
GEN_CFG = os.path.join(JOBS, "gen_cfg")      # steer_gen --config-json files (it takes a file path)
SVAMP_SETTLE = {"a1": "cells_ouro_1_4b_base_svamp_natural_k4.jsonl", "a2": "cells_ouro_2_6b_base_svamp_natural_k4.jsonl",
                "a3": "cells_ouro_1_4b_think_svamp_natural2_k4.jsonl",
                "a4": "cells_ouro_2_6b_think_svamp_natural2_k4.jsonl"}     # file names inside $SETTLE


def build(cells, order, last, skips, kept, shard_s, done=()):
    """Turn routed run/scan cells into jobs. Returns (Jobs, cell words, conflicts)."""
    J = Jobs(done)
    words = {}
    conflicts = []
    want = collections.defaultdict(set)          # (unit, model) -> datasets
    for c in cells:
        key = (c["card"], c["model"], c["dataset"])
        sk = skips.get(key)
        if c["status"] in ("done", "skip", "scope"):
            words[key] = (c["status"], c["note"])
            continue
        r = route(*key)
        if r is None:
            if sk:
                words[key] = ("skip", "%s: %s" % (sk["reason"], sk["why"]))
            else:
                words[key] = ("pending", "no pilot script route and no cells_skipped.json entry")
            continue
        if sk:
            conflicts.append((key, r[0], sk["reason"], sk["why"]))
        want[(r[1], c["model"])].add(c["dataset"])
        words[key] = (c["status"], "job: %s" % r[0])

    def mkey(m):
        return (m in last, order.index(m))

    # ---- group 0: the xllm build and import check (cluster architecture); every IFM job waits for it (afterok)
    ifm_dep = []
    if any(m.startswith("ifm_") for (u, m) in want):
        ifm_dep = [J.add("g0_xllm", "s1_build", "build_xllm.sh", "xllm", ["-"], "", "$OUT_ROOT/_xllm", [], 3600,
                         "preflight (environment record, import and config checks of the four family venvs), then the "
                         "xllm-loop build (CUDA extension) and import check; assumed 1 h",
                         "bash %s/cluster/env/preflight.sh" % LC)]

    def grp(g, m):
        """IFM jobs form their own wave (group g6_ifm) after the xllm build; the rest keep their queue group."""
        return "g6_ifm" if m.startswith("ifm_") else g

    def deps(m):
        return list(ifm_dep) if m.startswith("ifm_") else []

    # ---- group 1: GSM8K copy and derived (k17_varbench sharded; McLeish copy through k17_public in group 1)
    # Plan protocol (k17_varbench_plan.md sections 9-10): phase W, then the W slots are appended to section 13 of a
    # copy of the plan and the copy is re-hashed (cluster/vb_slots.py), then phase M (scan with the shard plan, rows
    # per shard, merge) runs on that copy and its sha256. A full run refuses phase all.
    for m in sorted([m for (u, m) in want if u == "vb"], key=mkey):
        # B-EXT2 item 5: the IFM models keep their first pipeline (group g6_ifm, ids in place, done) and add the V1
        # rerun (group g12_ifmv1, folder <key>_v1, --vbd_v1); the merge reads the _v1 folder (cell_target)
        for v1 in ((False, True) if m in VB_V1 else (False,)):
            g = VB_V1_GROUP if v1 else grp("g1_copy", m)
            k = VB_KEY[m]
            it_s, vbd_s, role, src = RATES[("vb", k)]
            kp = kept.get(("vb", k, "VB"))
            rate_pass = (kp[0] / float(kp[1])) if kp and kp[1] else 0.9
            n_vb = int(round(VB_POOL * rate_pass))
            n_vbd = int(round(VBD_POOL * rate_pass))
            rows_s = n_vb * it_s + n_vbd * vbd_s
            out = "$OUT_ROOT/B/k17_varbench/%s" % (vb_dir(m) if v1 else k)
            base = "--models %s --data %s/varbench --out %s --role cluster" % (k, DATA, out)
            tail = " --pool pairs --n_vb 100000 --vbd_n %s=100000 --datasets VBD --no_gradient_rank" % k
            if v1:
                tail += " --vbd_v1 %s=1" % k
            if k.endswith("_think"):
                tail += " --model_dirs %s=%s" % (k, THINK14 if k == "ouro_think" else THINK26)
            cw = base + " --plan $VB_PLAN --plan_sha $VB_PLAN_SHA" + tail
            plan_m = "%s/plan_M_%s.md" % (out, k)
            cm = base + " --plan %s --plan_sha $(sha256sum %s | cut -c1-64)" % (plan_m, plan_m) + tail
            # no FD and bf16 gradient checks: the gradient score is not computed on the cluster (--no_gradient_rank)
            win = J.add(g, "s2_window", "k17_varbench.py", m, ["VB", "VBD"], "", out, deps(m), 40 * it_s,
                        "phase W on the 40-item subset (assumed: the per-item M rate)",
                        "%s %s/k17_varbench.py %s --phase W --tag W_%s && %s %s/cluster/vb_slots.py --plan $VB_PLAN --out %s --model %s" % (
                            PY, LF, cw, k, PY, LC, out, k))
            scan = J.add(g, "s3_scan", "k17_varbench.py", m, ["VB", "VBD"], "", out, [win],
                         VB_POOL * 1.0 + VBD_POOL * 1.0, "clean-pass scan of %d pairs and %d VBD items (about 1 s per candidate, two forwards)" % (VB_POOL, VBD_POOL),
                         "%s %s/k17_varbench.py %s --phase M --stage scan --tag M_%s" % (PY, LF, cm, k))
            n = n_shards(rows_s, shard_s)
            sh = []
            for i in range(n):
                sh.append(J.add(g, "s4_rows", "k17_varbench.py", m, ["VB", "VBD"], "%d/%d" % (i, n), out, [scan],
                                rows_s / n, "%s (%s): %s" % (src, role, "%.1f s/VarBench item, %.1f s/VBD item; planned %d + %d items" % (it_s, vbd_s, n_vb, n_vbd)),
                                "%s %s/k17_varbench.py %s --phase M --stage rows --shard %d/%d --tag M_%s" % (PY, LF, cm, i, n, k)))
            # after a good merge the shard files move to shards/ (the summarize globs vb_*.json in the output folder)
            mg = J.add(g, "s5_merge", "k17_varbench.py", m, ["VB", "VBD"], "", out, sh, 1800, "merge: dev-tuned passes on the merged items (assumed 30 min)",
                       "%s %s/k17_varbench.py %s --phase M --stage merge --tag M_%s && mkdir -p %s/shards && mv %s/*.shard*of*.* %s/shards/" % (
                           PY, LF, cm, k, out, out, out))
            J.add(g, "s6_summarize", "k17_varbench.py", m, ["VB", "VBD"], "", out, [mg], 300, "CPU summarize",
                  "%s %s/k17_varbench.py --summarize --out %s --plan %s && %s $REPO/loopedcot/cluster/extract/run.py --script k17_varbench.py --src %s/k17_varbench.py --job %s" % (PY, LF, out, plan_m, PY, LF, out))

    # ---- k17_public job groups (scan, rows shards, merge per dataset; summarize per group)
    def public(group, gk, datasets, lre=False):
        # one --out per dataset: k17_public names its meta, inputs and log files by group and stage only, so two
        # datasets of one group in one folder would overwrite each other's stage files. lre: the RA-23 LRE pool files
        # (data folder public_lre, run as the facts dataset D1, output folder <group>_LRE)
        for ds in datasets:
            out = "$OUT_ROOT/B/k17_public/%s_%s" % (gk, "LRE" if lre else ds)
            c = ("--model %s --data %s/%s --out %s --plan $PUB_PLAN --datasets %s --targets %s=100000 "
                 "--budget_min 100000000 --full_search_cap 0 --no_reduced --no_seeds --no_gradient_rank" % (
                     gk, DATA, "public_lre" if lre else "public", out, ds, ds))
            kp = kept.get(("pub", PUB_RUNNER_OF[gk], ds))
            rate, role, src = RATES[("pub", gk, ds)]
            n_cand = POOL_N.get(("publre" if lre else "pub", PUB_FAM[gk], ds), 0)
            ratio = (kp[0] / float(kp[1])) if kp and kp[1] else PASS_FALLBACK[ds][0]
            # one pool file per dataset (every candidate of the family's file is scanned, no cap): the kept rate of
            # the last scan applied to the whole candidate file
            if ds == "CF":       # the shared CF pool: no earlier scan of it on this family; production pass rate
                ratio = PASS_FALLBACK["CF"][0]
            if lre:              # no scan of the LRE pool yet: pass rate assumed
                ratio, src = LRE_PASS, src + "; LRE pool, pass rate %.2f assumed" % LRE_PASS
            n_plan = int(round(n_cand * min(1.0, ratio)))
            rows_s = n_plan * rate
            sc = J.add(group, "s1_scan", "k17_public.py", gk, [ds], "", out, [], 2.0 * n_cand,
                       "scan: about 1-2.3 s per candidate in k17_public full (scan_seconds / scanned); %d candidates" % n_cand,
                       "%s %s/k17_public.py --run %s --stage scan" % (PY, LF, c))
            n = n_shards(rows_s, shard_s)
            sh = [J.add(group, "s2_rows", "k17_public.py", gk, [ds], "%d/%d" % (i, n), out, [sc], rows_s / n,
                        "%s (%s): %.1f s/item, planned %d items" % (src, role, rate, n_plan),
                        "%s %s/k17_public.py --run %s --stage rows --shard %d/%d" % (PY, LF, c, i, n)) for i in range(n)]
            mg = J.add(group, "s3_merge", "k17_public.py", gk, [ds], "", out, sh, 900,
                       "merge: dev-tuned pass (assumed 15 min)",
                       "%s %s/k17_public.py --run %s --stage merge" % (PY, LF, c))
            J.add(group, "s4_summarize", "k17_public.py", gk, [ds], "", out, [mg], 300, "CPU summarize",
                  "%s %s/k17_public.py --summarize --out %s --plan $PUB_PLAN && %s $REPO/loopedcot/cluster/extract/run.py --script k17_public.py --src %s/k17_public.py --job %s" % (PY, LF, out, PY, LF, out))

    pub_want = collections.defaultdict(set)
    for (u, m), dss in want.items():
        if u == "pub":
            pub_want[PUB_GROUP[m]] |= dss
    if "VB" in pub_want.get("mcl_nr", set()):
        public("g1_copy", "mcl_nr", ["VB"])

    # ---- group 2: generation (steer_gen, one job per (model, row); GSM8K then SVAMP). Every problem that passes the
    # clean filter is scored (RA-20 item 9, correction of 2026-10-09); sentinels come from the prep items left over.
    g = "g2_gen"
    for dsname in ("GSM8K_gen", "SVAMP_gen"):
        for m in sorted([m for (u, m), d in want.items() if u == "gen" and dsname in d], key=mkey):
            st = GEN[m]
            gg = GEN_GROUP.get(m, g)
            rate, _, role, src = RATES[("gen", st)]
            n_items = PLAN_GEN.get((st, dsname), 400) + 22 + 8
            dsenv = "gsm8k" if dsname == "GSM8K_gen" else "svamp"
            out = "$OUT_ROOT/A/steer_gen/%s_%s" % (st, dsenv)
            cfg = {"n_screen": 100000}
            if dsenv == "svamp":
                cfg.update({"dataset": "svamp", "settle": SVAMP_SETTLE[st]})
            if st in THINK_OF:
                cfg.update({"rows": GEN_ROWS, "n_sent": 22, "n_shift": 8})
            cfg_name = "%s_%s.json" % (st, dsenv)
            GEN_CFGS[cfg_name] = cfg
            env = "K17_DATASET=%s PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs" % (dsenv, out, out)
            if st in THINK_OF:
                env += " K17_THINK_DIR=%s" % THINK_OF[st]       # select reads the Thinking folder from the environment only
            base = "cd %s && %s %s -m prod.steer_gen {mode} --setting %s --out %s --mem-fraction $GEN_MEM_FRACTION" % (TREE, env, PY, st, out)
            if st in THINK_OF:
                base += " --think-dir %s" % THINK_OF[st]
            cj = "--config-json %s/cluster/jobs/gen_cfg/%s" % (LC, cfg_name)
            chk = J.add(gg, "s1_checks", "steer_gen.py", m, [dsname], "", out, [], 240,
                        "checks mode: 26-99 s in run 1 and run 5 on 4 items; 12 items with G3 on all 12 (cluster D5m)",
                        base.format(mode="checks").replace("--out %s" % out, "--out %s/chk" % out) + " %s %s && %s %s/gate.py %s/chk/checks_%s.json" % (G3_CLUSTER, cj, PY, LF, out, st))
            prep = J.add(gg, "s2_prep_screen", "steer_gen.py", m, [dsname], "", out, [chk], 480 * rate + 60,
                         "prep (CPU, 3 s) + screen of every SETTLE-correct problem on the B side at the row rate",
                         "%s --settle-dir $SETTLE %s && %s %s" % (base.format(mode="prep"), cj, base.format(mode="screen"), cj))
            sel = J.add(gg, "s3_select", "steer_gen.py", m, [dsname], "", out, [prep], 30, "select (CPU, 3 s)",
                        "cd %s && %s %s %s/cluster/steer_select.py --setting %s --out %s --settle-dir $SETTLE %s%s" % (
                            TREE, env, PY, LC, st, out, cj, " --think-dir %s" % THINK_OF[st] if st in THINK_OF else ""))
            rows = [J.add(gg, "s4_row", "steer_gen.py", m, [dsname], r, out, [sel], n_items * rate + 60,
                          "%s (%s): %.2f s/item at batch 16; planned %d scored + up to 30 sentinel and shift items" % (src, role, rate, n_items - 30),
                          "%s --row %s %s" % (base.format(mode="run"), r, cj)) for r in GEN_ROWS]
            rows.append(J.add(gg, "s4_row", "steer_gen.py", m, [dsname], "bw1_r00", out, [sel], 20 * 7.2 * rate / 0.62,
                              "bw1 check: 7.2 s/item at batch 1 on 20 items (run 1)", "%s --row bw1_r00 --n 20 %s" % (base.format(mode="run"), cj)))
            post = []
            if dsenv == "gsm8k":
                post.append(J.add(gg, "s5_anschain_presence", "steer_gen.py", m, [dsname], "", out, rows, (832 + 93) * (n_items / 172.0) * rate / 0.62,
                                  "anschain 832 s + presence 93 s for 172 items in run 1, scaled",
                                  "%s %s && %s %s" % (base.format(mode="anschain"), cj, base.format(mode="presence"), cj)))
            J.add(gg, "s6_analyze", "steer_gen.py", m, [dsname], "", out, post or rows, 60, "analyze (CPU)",
                  "%s %s && %s $REPO/loopedcot/cluster/extract/run.py --script steer_gen.py --src %s/prod/steer_gen.py --job %s" % (
                      base.format(mode="analyze"), cj, PY, TREE, out))

    # McLeish GSM8K generation (steer_gen_raven.py): rows r00-r21 on the 237 problems the shared-host run selected. The
    # selection ships in the bundle (cluster/pilot/data/mcleish_setting/items.json); each row line puts it into the
    # output folder unless a staged copy is already there (atomic: temp file, then rename)
    for m in sorted([m for (u, m) in want if u == "genr" and m == "mcleish_llama32_r32"], key=mkey):
        rate, role, src = RATES[("genr", GENR[m])]
        out = "$OUT_ROOT/" + GENR_DIR
        env = "K17_DATASET=gsm8k PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs" % (out, out)
        guard = ('mkdir -p %s && { test -f %s/items.json || { cp %s/mcleish_setting/items.json %s/items.json.$$.tmp && '
                 'mv -f %s/items.json.$$.tmp %s/items.json; }; }' % (out, out, DATA, out, out, out))
        base = "%s && cd %s && %s %s %s/steer_gen_raven.py" % (guard, TREE, env, PY, LF)
        rows = [J.add(g, "s4_row", "steer_gen_raven.py", m, ["GSM8K_gen"], r, out, [], (GENR_N + 22) * rate + 120,
                      "%s (%s): %.1f s/problem per row at batch 16; %d scored + 22 sentinels" % (src, role, rate, GENR_N),
                      "%s run --out %s --row %s --mem-fraction $GEN_MEM_FRACTION" % (base, out, r)) for r in GENR_ROWS]
        J.add(g, "s6_analyze", "steer_gen_raven.py", m, ["GSM8K_gen"], "", out, rows, 120, "analyze (CPU)",
              "%s analyze --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script steer_gen_raven.py --src %s/steer_gen_raven.py --job %s" % (
                  base, out, PY, LF, out))

    # ---- group 3: edit-site map at full pools (Huginn and Parcae last); IFM scripts in the g6_ifm wave. The quota
    # scripts run with --full_pool (k15_full, k16_path_*) or --n_half 100000 (k15_pairs): every candidate of the
    # dataset's pool file is scanned and every item that passes the clean filter is kept (no per-model cap).
    g = "g3_map"
    for m in sorted([m for (u, m) in want if u == "k15f"], key=mkey):
        rate, role, src = RATES[("k15f", m)]
        n_plan = sum(planned_quota("k15f", m, d, kept) for d in want[("k15f", m)])
        out = "$OUT_ROOT/B/k15_full/%s" % m
        J.add(grp(g, m), "s1_run", "k15_full.py", m, sorted(want[("k15f", m)]), "", out, deps(m), n_plan * rate + 900,
              "%s (%s): %.1f s/pair, planned %d pairs" % (src, role, rate, n_plan),
              "%s %s/k15_full.py --models %s --out %s --cf %s/k15_full/cf_records.json --cf_pool %s/cf_pool/cf_pool.json --tag %s --pair_budget_min 0 --full_pool && %s %s/k15_full.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script k15_full.py --src %s/k15_full.py --job %s" % (
                  PY, LF, m, out, DATA, DATA, m, PY, LF, out, PY, LF, out))
    for m in sorted([m for (u, m) in want if u == "k15p"], key=mkey):
        rate, role, src = RATES[("k15p", "*")]
        n_plan = planned_quota("k15p", m, "D1", kept)
        out = "$OUT_ROOT/B/k15_pairs/%s" % m
        J.add(grp(g, m), "s1_run", "k15_pairs.py", m, ["D1"], "", out, deps(m), n_plan * rate + 300,
              "%s (%s): %.1f s/pair, planned %d pairs" % (src, role, rate, n_plan),
              "%s %s/k15_pairs.py --models %s --out %s --tag %s --n_half 100000 && %s %s/k15_pairs.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script k15_pairs.py --src %s/k15_pairs.py --job %s" % (PY, LF, m, out, m, PY, LF, out, PY, LF, out))
    for m in sorted([m for (u, m) in want if u == "k16o"], key=mkey):
        k = K16O[m]
        rate, role, src = RATES[("k16o", k)]
        n_plan = planned_quota("k16o", k, "D5", kept)
        out = "$OUT_ROOT/B/k16_path_ouro/%s_D5" % k
        extra = " --model_dir %s --arms w2 --min_items 40" % THINK14 if k == "ouro_think" else ""
        J.add(g, "s1_run", "k16_path_ouro.py", m, ["D5"], "", out, [], n_plan * rate + 300, "%s (%s): %.2f s/item, planned %d items" % (src, role, rate, n_plan),
              "%s %s/k16_path_ouro.py --models %s --datasets D5 --out %s --tag %s_D5 --ftag _D5 --t3n 0 --full_pool%s && %s %s/k16_path_ouro.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script k16_path_ouro.py --src %s/k16_path_ouro.py --job %s" % (
                  PY, LF, k, out, k, extra, PY, LF, out, PY, LF, out))
    for m in sorted([m for (u, m) in want if u == "k16i"], key=mkey):
        rate, role, src = RATES[("k16i", "*")]
        for ds in sorted(want[("k16i", m)]):
            n_plan = planned_quota("k16i", m, ds, kept)
            out = "$OUT_ROOT/B/k16_path_ifm/%s_%s" % (m, ds)
            J.add(grp(g, m), "s1_run", "k16_path_ifm.py", m, [ds], "", out, deps(m), n_plan * rate + 300,
                  "%s (%s): %.1f s/item, planned %d items" % (src, role, rate, n_plan),
                  "%s %s/k16_path_ifm.py --models %s --datasets %s --out %s --tag %s_%s --data_dir %s/k16_path_ifm --guard 100000000 --full_pool && %s %s/k16_path_ifm.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script k16_path_ifm.py --src %s/k16_path_ifm.py --job %s" % (
                      PY, LF, m, ds, out, m, ds, DATA, PY, LF, out, PY, LF, out))
    mapped = sorted(pub_want.get("mcl_nr", set()) - {"VB"})
    if mapped:
        public(g, "mcl_nr", [d for d in ("D1", "CF", "D2", "D4", "D5") if d in mapped])
    if pub_want.get("ouro_think"):
        public(g, "ouro_think", [d for d in ("D1", "D2", "D4") if d in pub_want["ouro_think"]])
    for m in sorted([m for (u, m) in want if u == "par"], key=mkey):
        k = PAR_KEY[m]
        dss = [d for d in ("D1", "VB") if d in want[("par", m)]]
        est = 0
        src_all = []
        for d in dss:
            rate, role, src = RATES[("par", k, d)]
            n_plan = planned_par(k, d, kept)
            est += n_plan * rate
            src_all.append("%s %s (%s): %.1f s/item, planned %d" % (d, src, role, rate, n_plan))
        out = "$OUT_ROOT/B/k17_parcae/%s" % k
        caps = ",".join("%s=100000" % d for d in dss)
        J.add(g, "s1_run", "k17_parcae.py", m, dss, "", out, [], est + 600, "; ".join(src_all),
              "%s %s/k17_parcae.py --run --model %s --data %s/parcae --out %s --tok %s --datasets %s --targets %s --budget_min 100000000 --plan $PAR_PLAN --no_gradient_rank && %s %s/k17_parcae.py --summarize --out %s --plan $PAR_PLAN && %s $REPO/loopedcot/cluster/extract/run.py --script k17_parcae.py --src %s/k17_parcae.py --job %s" % (
                  PY, LF, k, DATA, out, PARCAE_TOK, ",".join(dss), caps, PY, LF, out, PY, LF, out))

    # ---- group 4: C and D scripts on the models they load (raven branch-freeze; Huginn last)
    g = "g4_cd"
    for m in sorted([m for (u, m) in want if u == "frz"], key=mkey):
        k = RAVEN_FREEZE[m]
        rate, role, src = RATES[("frz", "*")]
        for ds in sorted(want[("frz", m)]):
            n_plan = planned_frz(m, ds, kept)
            out = "$OUT_ROOT/C/k17_freeze_raven/%s_%s" % (k, ds)
            J.add(g, "s1_run", "k17_freeze_raven.py", m, [ds], "", out, [], n_plan * rate * 5 + 600, "%s (%s); about 5x the planned %d items (every clean-passing candidate of the pool file)" % (src, role, n_plan),
                  "%s %s/k17_freeze_raven.py --run --model %s --tree %s/tree --data %s/public --out %s --plan $FRZ_PLAN --plan_sha256 $FRZ_PLAN_SHA --datasets %s && %s %s/k17_freeze_raven.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script k17_freeze_raven.py --src %s/k17_freeze_raven.py --job %s" % (
                      PY, LF, k, LF, DATA, out, ds, PY, LF, out, PY, LF, out))

    # ---- RA-23 additions: group g7_ra23 (Ouro Thinking branch-freeze, LRE facts), the IFM branch-freeze in the IFM
    # wave (after the xllm build), the Huginn LRE cell in group 5 (Huginn last)
    def freeze2(unit, m, script, group, extra, dep):
        k = FRZ_OURO.get(m, m)
        rate, role, src = RATES[(unit, k if unit == "frzo" else "*")]
        for ds in [d for d in FRZ_DS if d in want[(unit, m)]]:
            n_plan = planned_frz2(unit, ds)
            out = "$OUT_ROOT/C/%s/%s_%s" % (script[:-3], k, ds)
            J.add(group, "s1_run", script, m, [ds], "", out, dep, n_plan * rate + FRZ_POOL[ds] * rate / 7.0 + 300,
                  "%s (%s): %.1f s/item, planned %d items; scan of %d candidates at 2 forwards each" % (src, role, rate, n_plan, FRZ_POOL[ds]),
                  "%s %s/%s --run --model %s%s --out %s --plan $FRZO_PLAN --plan_sha256 $FRZO_PLAN_SHA --items_d1 %s/k17_freeze/items_d1.json --datasets %s && %s %s/k17_freeze.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script %s --src %s/%s --job %s" % (
                      PY, LF, script, m, extra, out, DATA, ds, PY, LF, out, PY, script, LF, script, out))
    for m in sorted([m for (u, m) in want if u == "frzo"], key=mkey):
        freeze2("frzo", m, "k17_freeze.py", "g7_ra23", " --full_pool --tree %s/tree --mem_fraction 0.3" % LF, [])
    lre_want = sorted({PUB_GROUP[m] for (u, m) in want if u == "publre"}, key=lambda gk: ("huginn" == gk, gk))
    for gk in lre_want:
        if gk != "huginn":
            public("g7_ra23", gk, ["D1"], lre=True)
    # ASDiv and MAWPS generation placement (RA-23 item 4): steer_gen_math.py (steer_gen unchanged, the dataset layer
    # installed in process); the pipeline of group 2 without the GSM8K-only anschain and presence steps and without
    # the bw1 check; both screens run on the model (no SETTLE cells exist for these datasets)
    for dsname in ("ASDIV_gen", "MAWPS_gen"):
        for m in sorted([m for (u, m), d in want.items() if u == "genm" and dsname in d], key=mkey):
            st, dsk = GEN[m], GENM_DS[dsname]
            rate, _, role, src = RATES[("gen", st)]
            n_pool = GENM_POOL[dsk]
            n_items = int(round(n_pool * GENM_PASS)) + 22 + 8
            out = "$OUT_ROOT/A/steer_gen_math/%s_%s" % (st, dsk)
            cfg = {"dataset": dsk, "n_screen": 100000, "screen_A_from_settle": False}
            if st in THINK_OF:
                cfg.update({"rows": GEN_ROWS, "n_sent": 22, "n_shift": 8})
            cfg_name = "%s_%s.json" % (st, dsk)
            GEN_CFGS[cfg_name] = cfg
            env = "PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs" % (out, out)
            if st in THINK_OF:
                env += " K17_THINK_DIR=%s" % THINK_OF[st]
            w = "cd %s && %s %s %s/steer_gen_math.py --dataset %s --rows-dir %s ouro" % (TREE, env, PY, LF, dsk, out)
            tail = " --setting %s --config-json %s/cluster/jobs/gen_cfg/%s" % (st, LC, cfg_name)
            if st in THINK_OF:
                tail += " --think-dir %s" % THINK_OF[st]
            g = GEN_GROUP.get(m, "g7_ra23")
            chk = J.add(g, "s1_checks", "steer_gen_math.py", m, [dsname], "", out, [], 600, "checks mode: 496 s in the ASDiv smoke (Ouro-1.4B, shared card)",
                        "%s checks --out %s/chk --n 4%s && %s %s/gate.py %s/chk/checks_%s.json" % (w, out, tail, PY, LF, out, st))
            prep = J.add(g, "s2_prep_screen", "steer_gen_math.py", m, [dsname], "", out, [chk], 2 * n_pool * rate + 60,
                         "prep (CPU) + both screens of all %d problems at the row rate (%s)" % (n_pool, src),
                         "%s prep --out %s%s && %s screen --out %s%s --mem-fraction $GEN_MEM_FRACTION" % (w, out, tail, w, out, tail))
            sel = J.add(g, "s3_select", "steer_gen_math.py", m, [dsname], "", out, [prep], 30, "select (CPU)",
                        "%s select_full --out %s --setting %s --config-json %s/cluster/jobs/gen_cfg/%s" % (w, out, st, LC, cfg_name))
            rows = [J.add(g, "s4_row", "steer_gen_math.py", m, [dsname], r, out, [sel], n_items * rate + 60,
                          "%s (%s): %.2f s/item at batch 16; planned %d scored (pass rate %.1f assumed) + up to 30 sentinel and shift items" % (
                              src, role, rate, n_items - 30, GENM_PASS),
                          "%s run --out %s --row %s%s --mem-fraction $GEN_MEM_FRACTION" % (w, out, r, tail)) for r in GEN_ROWS]
            J.add(g, "s6_analyze", "steer_gen_math.py", m, [dsname], "", out, rows, 60, "analyze (CPU)",
                  "%s analyze --out %s%s && %s $REPO/loopedcot/cluster/extract/run.py --script steer_gen_math.py --src %s/prod/steer_gen.py --job %s" % (
                      w, out, tail, PY, TREE, out))
    for m in sorted([m for (u, m) in want if u == "frzi"], key=mkey):
        freeze2("frzi", m, "k17_freeze_ifm.py", grp("g7_ra23", m), "", deps(m))

    # ---- group 5: Huginn edit-site and copy cells
    if pub_want.get("huginn"):
        public("g5_huginn", "huginn", [d for d in ("D1", "CF", "D2", "D4", "D5", "VB") if d in pub_want["huginn"]])
    if "huginn" in lre_want:
        public("g5_huginn", "huginn", ["D1"], lre=True)

    # ---- cluster extension B-EXT2 item 1: k16_path_ouro on the Thinking models (Y1, Y3, W2 at full pools), own group
    for m in sorted([m for (u, m) in want if u == "k16t"], key=mkey):
        k = K16T[m]
        rate = 16.2 if k == "ouro_think" else 16.2 * 234 / 138.0 * 2     # s/item: cluster ouro_think D5 run (W2 on)
        for ds in [d for d in ("D1", "D2", "D4", "D5") if d in want[("k16t", m)]]:
            n_plan = K16T_PLAN[ds]
            out = "$OUT_ROOT/C/k16_path_ouro/%s_%s" % (k, ds)
            J.add("g13_paththink", "s1_run", "k16_path_ouro.py", m, [ds], "", out, [], n_plan * rate + 600,
                  "assumed: %.1f s/item (the cluster ouro_think D5 run, 16.2 s with W2 and the scan%s), %d items assumed" % (
                      rate, "" if k == "ouro_think" else ", x 234/138 forwards x 2 layers for 2.6B", n_plan),
                  "%s %s/k16_path_ouro.py --models %s --datasets %s --out %s --tag %s_%s --ftag _%s --t3n 0 --full_pool "
                  "--model_dir %s --arms w2 --min_items 40 && %s %s/k16_path_ouro.py --summarize --out %s && %s "
                  "$REPO/loopedcot/cluster/extract/run.py --script k16_path_ouro.py --src %s/k16_path_ouro.py --job %s" % (
                      PY, LF, k, ds, out, k, ds, ds, THINK14 if k == "ouro_think" else THINK26, PY, LF, out, PY, LF, out))
    # ---- cluster extension B-EXT2 item 3: the Y1 arm (edit-site rows) on the seven cells that ran without it
    for m, ds in sorted(Y1_CELLS, key=lambda x: (order.index(x[0]), x[1])):
        if ds not in want.get(("k16i", m), set()):
            continue
        rate, role, src = RATES[("k16i", "*")]
        n_plan = planned_quota("k16i", m, ds, kept)
        out = "$OUT_ROOT/B/k16_path_ifm/%s_%s_y1" % (m, ds)
        J.add("g14_ifmy1", "s1_run", "k16_path_ifm.py", m, [ds], "", out, deps(m), n_plan * rate + 300,
              "%s (%s): %.1f s/item (all arms; Y1 alone is less), planned %d items" % (src, role, rate, n_plan),
              "%s %s/k16_path_ifm.py --models %s --datasets %s --out %s --tag %s_%s_y1 --data_dir %s/k16_path_ifm --guard 100000000 --full_pool --arms Y1 && %s %s/k16_path_ifm.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script k16_path_ifm.py --src %s/k16_path_ifm.py --job %s" % (
                  PY, LF, m, ds, out, m, ds, DATA, PY, LF, out, PY, LF, out))

    # ---- cluster extension B-EXT2 item 4: Huginn GSM8K generation (steer_gen_raven.py --setting hug), own group;
    # the prep ships (cluster/pilot/data/huginn_setting/prep.json), then screen A and B, select, 22 rows, analyze
    for m in sorted([m for (u, m) in want if u == "genr" and m == "huginn_0125"], key=mkey):
        rate, role, src = RATES[("genr", GENR[m])]
        o = "$OUT_ROOT/" + genr_dir(m)
        env = "K17_DATASET=gsm8k PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs" % (o, o)
        guard = ('mkdir -p %s && { test -f %s/prep.json || { cp %s/huginn_setting/prep.json %s/prep.json.$$.tmp && '
                 'mv -f %s/prep.json.$$.tmp %s/prep.json; }; }' % (o, o, DATA, o, o, o))
        base = "%s && cd %s && %s %s %s/steer_gen_raven.py" % (guard, TREE, env, PY, LF)
        mf = " --mem-fraction $GEN_MEM_FRACTION"
        gh = "g15_hgen"
        scr = [J.add(gh, "s2_screen", "steer_gen_raven.py", m, ["GSM8K_gen"], side, o, [], GENR_HUG_PREP * rate + 120,
                     "%s (%s): %.1f s/problem, %d prep problems" % (src, role, rate, GENR_HUG_PREP),
                     "%s screen --setting hug --out %s --side %s%s" % (base, o, side, mf)) for side in ("A", "B")]
        sel = J.add(gh, "s3_select", "steer_gen_raven.py", m, ["GSM8K_gen"], "", o, scr, 120, "select (CPU)",
                    "%s select --setting hug --root %s --out %s" % (base, o, o))
        rows = [J.add(gh, "s4_row", "steer_gen_raven.py", m, ["GSM8K_gen"], r, o, [sel], (GENR_HUG_N + 22) * rate + 120,
                      "%s (%s): %.1f s/problem per row; %d scored planned + 22 sentinels" % (src, role, rate, GENR_HUG_N),
                      "%s run --setting hug --out %s --row %s%s" % (base, o, r, mf)) for r in GENR_ROWS]
        J.add(gh, "s6_analyze", "steer_gen_raven.py", m, ["GSM8K_gen"], "", o, rows, 120, "analyze (CPU)",
              "%s analyze --setting hug --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script steer_gen_raven.py --src %s/steer_gen_raven.py --job %s" % (
                  base, o, PY, LF, o))

    # ---- every task line activates its own venv (the raven scripts need Python >= 3.10)
    for r in J.rows:
        v = venv_of(r["script"], r["model"])
        if v:
            r["cmd"] = "source $VENV_%s/bin/activate && %s" % (v, r["cmd"])

    # ---- merge and summarize per grid (CPU), after every job of every group
    J.add("g9_merge", "s1_merge", "merge.py", "all", ["all"], "", "$OUT_ROOT", [], 600, "CPU",
          "%s $REPO/loopedcot/cluster/merge.py --out_root $OUT_ROOT --manifest $REPO/loopedcot/cluster/manifest.yaml "
          "--skipped $REPO/loopedcot/cluster/cells_skipped.json --joblist $REPO/loopedcot/cluster/jobs/joblist.tsv" % PY)
    J.rows[-1]["cmd"] = "source $VENV_OURO/bin/activate && " + J.rows[-1]["cmd"]
    return J, words, conflicts


# submission order of submit_10_rest.sh (coordinator 2026-10-09): groups one to four, the IFM wave, the RA-23
# additions, Huginn last, then the merge
GROUP_ORDER = ["g0_xllm", "g1_copy", "g2_gen", "g3_map", "g4_cd", "g6_ifm", "g12_ifmv1", "g14_ifmy1", "g7_ra23", "g11_gen26t", "g13_paththink", "g15_hgen",
               "g5_huginn",
               "g9_merge"]
GROUP_LABEL = {"g0_xllm": "preflight, xllm build and import check", "g1_copy": "GSM8K copy and derived",
               "g2_gen": "generation, GSM8K then SVAMP, McLeish GSM8K rows", "g3_map": "edit-site map at full pools",
               "g4_cd": "C and D scripts (raven branch-freeze)", "g6_ifm": "IFM wave incl. the RA-23 IFM branch-freeze",
               "g7_ra23": "RA-23: Ouro Thinking branch-freeze, LRE facts",
               "g15_hgen": "extension item 4: Huginn GSM8K generation",
               "g13_paththink": "extension: restore, read window and attention block on the Ouro Thinking models",
               "g14_ifmy1": "extension: the Y1 arm (edit-site rows) on seven IFM cells",
               "g12_ifmv1": "extension item 5: IFM k17_varbench rerun with the VBD V1 cells (--vbd_v1)",
               "g11_gen26t": "extension item 1: generation on Ouro-2.6B-Thinking (setting a4; GSM8K, SVAMP, ASDiv, MAWPS)",
               "g5_huginn": "Huginn incl. the RA-23 LRE cell",
               "g9_merge": "merge and grid tables"}


def venv_of(script, model):
    """OURO, RAVEN, IFM or PARCAE (the family venvs of cluster/env/setup_cluster.sh); None for the preflight and xllm
    build job (it names its venvs)."""
    if script == "build_xllm.sh":
        return None
    if script in ("k17_public.py", "k17_freeze_raven.py", "steer_gen_raven.py"):
        return "RAVEN"
    if script == "k17_parcae.py":
        return "PARCAE"
    if script in ("k15_full.py", "k15_pairs.py", "k16_path_ifm.py") or model.startswith("ifm_"):
        return "IFM"
    return "OURO"


PUB_RUNNER_OF = {"mcl_nr": "mcleish", "huginn": "huginn", "ouro_think": "ouro_think"}
LRE_PASS = 0.5      # planning only: clean-pass rate assumed for the LRE pool (no scan of it yet)
PUB_FAM = {"mcl_nr": "llama", "huginn": "huginn", "ouro_think": "ouro"}
GEN_CFGS = {}     # steer_gen config files written to jobs/gen_cfg/ (name -> dict)
# Parcae tokenizer (k17_parcae --tok is read by its build step only; the run takes the flag and ignores it)
PARCAE_TOK = ("$HF_HOME/hub/models--SandyResearch--parcae-tokenizer/snapshots/6247b5d0592876b73660f34c2dd16c9db9e9045c/"
              "tokenizer.json")
VB_POOL, VBD_POOL = 913, 848        # VarBench pairs pool (s200 build) and the uncapped derived pool (--pool pairs)
# Candidate counts of the pool files (one file per dataset and tokenizer family, every candidate scanned): k17_public
# and k17_parcae from cluster/pilot/data (read at generation); the k15 / k16 pools from the pilot records (candidates
# field: the dataset's pool for that tokenizer).
POOL_N = {}


def load_pool_counts(data_dir):
    for f in glob.glob(os.path.join(data_dir, "public", "cands_*_*.json")):
        fam, ds = os.path.basename(f)[6:-5].split("_")
        POOL_N[("pub", fam, ds)] = len(json.load(open(f, encoding="utf-8"))["cands"])
    for f in glob.glob(os.path.join(data_dir, "public_lre", "cands_*_D1.json")):
        POOL_N[("publre", os.path.basename(f)[6:-8], "D1")] = len(json.load(open(f, encoding="utf-8"))["cands"])
    for f in glob.glob(os.path.join(data_dir, "parcae", "cands_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        POOL_N[("par", "*", os.path.basename(f)[6:-5])] = len(d["cands"] if isinstance(d, dict) and "cands" in d else d)


# Pass rate assumed where the family has no earlier scan of the dataset (planning only): CF from k15_full on
# ifm_p1_ouro (307 kept of 8906 scanned), D5 from k16_path_ouro on Ouro 1.4B (80 of 208), D4 from k17_public on
# McLeish (60 of 64), D1 and D2 from the k15 scans (80 of 85, 80 of 80), VB from k17_public McLeish (60 of 67).
PASS_FALLBACK = {"CF": (783 / 3741.0, "the Ouro-1.4B base production scan of the CF pool (783 of 3,741)"), "D5": (80 / 208.0, "k16_path_ouro ouro D5 scan"),
                 "D4": (60 / 64.0, "k17_public mcleish D4 scan"), "D1": (80 / 85.0, "k15 D1 scans"),
                 "D2": (1.0, "k15 D2 scans"), "VB": (60 / 67.0, "k17_public mcleish VB scan")}


# CF: one pool for every model (cluster/pilot/data/cf_pool/cf_pool.json, 3,741 pairs); the pairs well formed on the
# IFM tokenizer (cluster/pilot/cf_pool.py pool_candidates, counted at generation from the pool and the IFM tokenizer
# when it is at hand, else the count below, measured 2026-10-09 on data_steer/_cache/tok_ifm/tokenizer.json)
CF_POOL_IFM = 1764
_cfp = os.path.join(HERE, "pilot", "data", "cf_pool", "cf_pool.json")
CF_NOTE = json.load(open(_cfp, encoding="utf-8"))["header"]["note"] if os.path.exists(_cfp) else "pool file missing"


def planned_quota(unit, m, ds, kept):
    """Planned n of a quota script run with the quota lifted: the pool's candidates times the pass rate of the last
    scan (kept / scanned of the pilot run). PLANNED, not measured; the subject and template caps of the selectors
    (at most 2 pairs per subject, 1 per D4/D5 template) still apply and lower it."""
    if unit == "k15f" and ds == "CF":           # the shared CF pool (no subject cap), production pass rate
        return int(round(CF_POOL_IFM * PASS_FALLBACK["CF"][0]))
    kp = kept.get((unit, m, ds))
    if not kp or not kp[1] or not kp[3]:          # no record for this model: the same dataset on another model
        ref = [v for (u, m2, d), v in sorted(kept.items()) if u == unit and d == ds and v[1] and v[3]]
        if not ref:
            return 80
        kp = ref[0]
    return int(round(kp[3] * min(1.0, kp[0] / float(kp[1]))))


def planned_frz(m, ds, kept):
    """k17_freeze_raven keeps every clean-passing candidate of the k17_public pool file (no cap; D2 one cap per
    operand cluster): the candidates times the kept rate of the k17_public scan of that model."""
    kp = kept.get(("pub", PUB_RUNNER[m], ds))
    n = POOL_N.get(("pub", "llama" if m == "mcleish_llama32_r32" else "huginn", ds), 0)
    ratio = (kp[0] / float(kp[1])) if kp and kp[1] else PASS_FALLBACK[ds][0]
    return int(round(n * min(1.0, ratio)))


FRZ_POOL = {"D1": 267, "D2": 592, "D4": 1311}    # the RA-3 identity list; D2 candidates after the dedupe (cap 4 per
#                                                  operand value: at most 263 kept on Ouro); D4 every candidate under
#                                                  the cap of 2 per cluster side (counted on the D4 file of either tokenizer)
FRZ_KEPT = {"D1": 267, "D2": 263, "D4": 1311}
FRZ_IFM_D1 = 231          # items of the RA-3 list that exist on the IFM tokenizer (smoke 2026-10-09)
K16T_PLAN = {"D1": 280, "D2": 600, "D4": 1100, "D5": 650}   # items assumed (pilot pass rates x pools; no scan)
GENR_N = 237              # McLeish problems scored by the shared-host run's select (rows r00, r01 done on them)


def planned_frz2(unit, ds):
    """Thinking and IFM branch-freeze: the pool's ceiling times the clean-pass fallback of the dataset (PLANNED)."""
    top = FRZ_IFM_D1 if (unit == "frzi" and ds == "D1") else FRZ_KEPT[ds]
    return int(round(top * PASS_FALLBACK[ds][0]))


def planned_par(k, d, kept):
    kp = kept.get(("par", k, d))
    n = POOL_N.get(("par", "*", d), 0)
    if not kp or not kp[1]:
        return PAR_PLAN[(k, d)]
    return int(round(n * min(1.0, kp[0] / float(kp[1]))))


# Parcae fallback when no earlier scan is read: D1 ends at the selector caps (62 and 61 kept); VB 513 candidates
# times the clean-pass rate seen
PAR_PLAN = {("parcae-370m", "D1"): 61, ("parcae-1.3b", "D1"): 62,
            ("parcae-370m", "VB"): int(round(513 * 60 / 134.0)), ("parcae-1.3b", "VB"): int(round(513 * 60 / 94.0))}
# generation, scored = every problem that passes both clean filters: SETTLE-correct problems x the B-pass rate of
# run 1 / run 5 (226 and 223 of 240 screened). SVAMP: the SETTLE k=4 pre-filter keeps 252 problems on 1.4B and on
# 2.6B (RA-18 build, data_steer/svamp); the GSM8K B-pass rates are applied (assumed, no SVAMP screen yet).
# Thinking: no SETTLE count read, 300 assumed.
PLAN_GEN = {("a1", "GSM8K_gen"): int(round(428 * 226 / 240.0)), ("a2", "GSM8K_gen"): int(round(451 * 223 / 240.0)),
            ("a3", "GSM8K_gen"): 300, ("a1", "SVAMP_gen"): int(round(252 * 226 / 240.0)),
            ("a2", "SVAMP_gen"): int(round(252 * 223 / 240.0)), ("a3", "SVAMP_gen"): 300,
            ("a4", "GSM8K_gen"): int(round(462 * 223 / 240.0)), ("a4", "SVAMP_gen"): 300}
# a4 GSM8K: 462 SETTLE-correct problems (prep of the item 1 smoke) x the 2.6B base B-pass rate (assumed)


# ----------------------------------------------------------------------------------------------- outputs
SBATCH = """#!/bin/bash
# Task template for queue group {group} (RA-20 Amendment 2). Generated by cluster/gen_jobs.py; do not edit by hand.
# The header keeps placeholders: jobs/submit_chain.sh, jobs/submit_00_xllm.sh and jobs/submit_10_rest.sh pass
# --partition, --account, --output and the node restriction ($SLURM_RESTRICT) on the sbatch command line, and those
# override the header. By hand:
#   sbatch --partition=$PARTITION --account=$ACCOUNT $SLURM_RESTRICT --output=$LOG_ROOT/ra20_{group}_%A_%a.out \
#          --array=1-<N>%<cards> jobs/task_{group}.sbatch jobs/{group}.<stage>.tsv
#SBATCH --job-name=ra20_{group}
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time={time}
#SBATCH --output=logs/ra20_{group}_%A_%a.out
#SBATCH --requeue
# Placeholders (all exported by the cluster.env that cluster/env/setup_cluster.sh writes; source it first):
#   $REPO          folder holding loopedcot/ (this bundle: cluster/pilot/ holds the pilot script copies)
#   $WEIGHTS_ROOT  IFM, d28, d112 and 819M checkpoint folders, see cluster/weights.json (IFM_CKPT_ROOT overrides it)
#   $XLLM_ROOT     the xllm-loop clone the IFM loader imports (IFM jobs only)
#   $HF_HOME       Hugging Face cache with the pinned hub checkpoints (HF_HUB_OFFLINE=1), Ouro Thinking included
#   $DATA_ROOT     pilot inputs; default $REPO/loopedcot/cluster/pilot/data
#   $OUT_ROOT      output tree; every job writes under $OUT_ROOT/<grid>/<script>/<model>
#   $VENV_OURO     Ouro family venv (Ouro scripts, steer_gen, steer_gen_math, merge)
#   $VENV_RAVEN    raven family venv, Python >= 3.10 (k17_public, k17_freeze_raven, steer_gen_raven)
#   $VENV_IFM      IFM family venv, Python >= 3.10, xllm built by the g0_xllm job (IFM scripts)
#   $VENV_PARCAE   Parcae family venv, Python >= 3.11 (k17_parcae); every task line activates its own venv
#   (the four venvs: cluster/env/setup_cluster.sh, system torch through --system-site-packages; cluster/env/CLUSTER.md)
#   $RUN_ROLE      label written into records instead of a host name (default cluster)
#   $MODULES       optional: environment modules loaded first (module load $MODULES)
# Inputs named in the task lines (set them next to the ones above): $VB_PLAN $VB_PLAN_SHA (k17_varbench plan and its
#   sha256), $PUB_PLAN (k17_public), $PAR_PLAN (k17_parcae), $FRZ_PLAN $FRZ_PLAN_SHA (k17_freeze_raven), $FRZO_PLAN
#   $FRZO_PLAN_SHA (k17_freeze and k17_freeze_ifm: k17_freeze_plan.md, the RA-3 plan, and its sha256), $SETTLE
#   (folder of the steer_gen SETTLE cell files), $GEN_MEM_FRACTION (steer_gen per-process memory cap;
#   at least 36 GiB of the card for the 2.6B rows that patch keys and values, else the batch halves).
set -euo pipefail
LIST="$1"
CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$LIST" | cut -f2-)
ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$LIST" | cut -f1)
: "${{OUT_ROOT:?set OUT_ROOT (source the cluster.env that setup_cluster.sh wrote)}}"
# on any failure $OUT_ROOT/_diag/<task id>.txt records the exit code, the failing check, the host role, the GPU, free
# memory and the last 50 lines of this task's output ($OUT_ROOT/_tasklog/<task id>.log); jobs/status.sh prints it
mkdir -p "$OUT_ROOT/_done" "$OUT_ROOT/_diag" "$OUT_ROOT/_tasklog"
TL="$OUT_ROOT/_tasklog/$ID.log"
: > "$TL"
CHECK="start"
diag() {{
  {{ echo "task: $ID (list $(basename "$LIST") line ${{SLURM_ARRAY_TASK_ID}})"
    echo "exit code: $1"
    echo "failing check: $CHECK"
    echo "host role: ${{RUN_ROLE:-cluster}}"
    echo "slurm: job ${{SLURM_ARRAY_JOB_ID:-${{SLURM_JOB_ID:-none}}}}_${{SLURM_ARRAY_TASK_ID}} node ${{SLURMD_NODENAME:-none}}"
    echo "gpu (name, free, total): $(nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2> /dev/null | paste -sd';' - || true)"
    echo "host memory: $(free -g 2> /dev/null | awk '/^Mem:/{{print $7 " GiB available of " $2 " GiB"}}' || true)"
    echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--- last 50 lines of $TL"
    tail -n 50 "$TL" 2> /dev/null || true; }} > "$OUT_ROOT/_diag/$ID.txt" 2>&1 || true
}}
trap 'rc=$?; if [ $rc -ne 0 ]; then diag $rc; fi' EXIT
trap 'CHECK="$CHECK (killed by SIGTERM: time limit, scancel or node failure)"; exit 143' TERM
# a node whose GPU cannot start (nvidia-smi or the task output shows "No devices were found" or "CUDA unknown error"):
# it goes to $OUT_ROOT/bad_nodes.txt, the diagnosis is written, and the task is requeued once (same job id, so the
# stages waiting on it keep waiting) with that node excluded; exit 75 marks it. A second failure is not requeued.
NODE="${{SLURMD_NODENAME:-$(hostname -s 2> /dev/null || hostname)}}"
bad_node() {{
  CHECK="GPU start on node $NODE: $1 (node added to $OUT_ROOT/bad_nodes.txt)"
  echo "$NODE" >> "$OUT_ROOT/bad_nodes.txt"
  local jid="${{SLURM_ARRAY_JOB_ID:+${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}}}" exc
  jid="${{jid:-${{SLURM_JOB_ID:-}}}}"
  exc=$( {{ echo "${{SLURM_RESTRICT:-}}" | tr ' ' '\\n' | sed -n 's/^--exclude=//p' | tr ',' '\\n'; cat "$OUT_ROOT/bad_nodes.txt"; }} | grep . | sort -u | paste -sd, -)
  if [ -n "$jid" ] && [ "${{SLURM_RESTART_COUNT:-0}}" -eq 0 ] && command -v scontrol > /dev/null 2>&1; then
    CHECK="$CHECK; requeued once with --exclude=$exc"
    diag 75
    scontrol update JobId="$jid" ExcNodeList="$exc" && scontrol requeue "$jid"
  else
    CHECK="$CHECK; not requeued (it was requeued once already, or this is not a Slurm job)"
  fi
  exit 75
}}
CHECK="GPU visible (nvidia-smi -L)"
GPUS=$(nvidia-smi -L 2>&1 || true)
echo "node=$NODE" >> "$TL"
echo "$GPUS" >> "$TL"
case "$GPUS" in *"No devices were found"*|*"CUDA unknown error"*) bad_node "$(echo "$GPUS" | head -1)";; esac
CHECK="previous chain step: no $OUT_ROOT/_chain/setup_check.fail"
if [ -f "$OUT_ROOT/_chain/setup_check.fail" ]; then echo "the setup check failed: see $OUT_ROOT/_chain/setup_check.fail" | tee -a "$TL"; exit 4; fi
if [ -n "${{MODULES:-}}" ]; then
  CHECK="module load $MODULES"
  set +u
  type module > /dev/null 2>&1 || . /etc/profile > /dev/null 2>&1 || true
  module load $MODULES >> "$TL" 2>&1 || {{ set -u; exit 5; }}
  set -u
fi
# the cluster runs the pinned laptop copies only: every script must match jobs/scripts_sha256.txt
CHECK="pinned code hashes (sha256sum -c loopedcot/cluster/jobs/scripts_sha256.txt)"
(cd "$REPO" && sha256sum -c --quiet loopedcot/cluster/jobs/scripts_sha256.txt) 2>&1 | tee -a "$TL"
CHECK="environment and home-directory guard"
export DATA_ROOT="${{DATA_ROOT:-$REPO/loopedcot/cluster/pilot/data}}" RUN_ROLE="${{RUN_ROLE:-cluster}}"
# the five plan files are in cluster/plans/ and the SETTLE cell files in cluster/data/ (cluster.env may name others)
export VB_PLAN="${{VB_PLAN:-$REPO/loopedcot/cluster/plans/k17_varbench_plan.md}}"
export PUB_PLAN="${{PUB_PLAN:-$REPO/loopedcot/cluster/plans/k17_public_plan.md}}"
export PAR_PLAN="${{PAR_PLAN:-$REPO/loopedcot/cluster/plans/k17_parcae_plan.md}}"
export FRZ_PLAN="${{FRZ_PLAN:-$REPO/loopedcot/cluster/plans/k17_freeze_raven_plan.md}}"
export FRZO_PLAN="${{FRZO_PLAN:-$REPO/loopedcot/cluster/plans/k17_freeze_plan.md}}"
export SETTLE="${{SETTLE:-$REPO/loopedcot/cluster/data}}"
# RA-20 item 6: records carry argv and paths (steer_gen), so no folder a job reads or writes may sit in a home directory
for v in REPO OUT_ROOT DATA_ROOT HF_HOME SETTLE WEIGHTS_ROOT IFM_CKPT_ROOT XLLM_ROOT; do
  p="${{!v:-}}"; [ -n "$p" ] || continue
  case "$(readlink -f "$p")/" in
    "$(readlink -f "$HOME")"/*|/home/*|/Users/*|/root/*)
      if [ "${{ALLOW_HOME:-0}}" = 1 ]; then HOME_RUN=1
      else echo "$v=$p is inside a home directory; use a project or scratch folder (or ALLOW_HOME=1 when the cluster has no scratch)" | tee -a "$TL"; exit 2; fi;;
  esac
done
[ -z "${{HOME_RUN:-}}" ] || echo "running inside a home directory; records will be scrubbed at merge" | tee -a "$TL"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export SETUPTOOLS_USE_DISTUTILS=stdlib      # setuptools' own distutils clashes with the stdlib one on venvs over a conda base
if [ -f "$OUT_ROOT/_done/$ID" ]; then echo "done already: $ID"; exit 0; fi
echo "task $ID" | tee -a "$TL"
CHECK="task command (exit code of the task line)"
(eval "$CMD") 2>&1 | tee -a "$TL" || {{
  rc=$?
  if grep -q "CUDA unknown error\\|No devices were found" "$TL"; then bad_node "task output shows a GPU start failure"; fi
  exit $rc
}}
CHECK="done marker"
touch "$OUT_ROOT/_done/$ID"
rm -f "$OUT_ROOT/_diag/$ID.txt"
"""


DONE_SKIP = 'if [ -f "$OUT_ROOT/_done/$ID" ]; then echo "done already: $ID"; exit 0; fi\n'
MERGE_RERUN = ('# the merge reruns every round (merge.py is idempotent): its done marker does not skip it\n'
               'if [ -f "$OUT_ROOT/_done/$ID" ]; then echo "merge ran before: $ID; running it again"; fi\n')


# checks at the top of both submit scripts: the environment file, the smoke marker, and the node restriction of
# cluster.env (SLURM_RESTRICT, plus the nodes in $OUT_ROOT/bad_nodes.txt) passed on every sbatch command line
def gates(env_rel):
    """Checks at the top of the generated submit scripts. env_rel: cluster/env seen from the jobs folder (the
    cluster.env of the first-push setup lives there; setup now writes $OUT_ROOT/cluster.env)."""
    return [
        "# cluster.env: the one already sourced, else $OUT_ROOT/cluster.env, else the first-push location",
        'if [ -z "${PARTITION:-}" ] || [ -z "${REPO:-}" ]; then',
        '  for f in "${OUT_ROOT:+$OUT_ROOT/cluster.env}" "%s/cluster.env"; do' % env_rel,
        '    if [ -n "$f" ] && [ -f "$f" ]; then echo "reading $f"; . "$f"; break; fi',
        "  done",
        "fi",
        ': "${OUT_ROOT:?set OUT_ROOT (source the cluster.env that setup_cluster.sh wrote)}"',
        # any failure of a submit step: $OUT_ROOT/_diag/<script name>.txt (exit code, failing check, last 50 lines)
        'SN=$(basename "$0" .sh); mkdir -p "$OUT_ROOT/_diag" "$OUT_ROOT/_tasklog"; TL="$OUT_ROOT/_tasklog/$SN.log"; : > "$TL"',
        'exec > >(tee -a "$TL") 2>&1',
        'CHECK="start"',
        "diag() { sleep 1; { echo \"step: $SN\"; echo \"exit code: $1\"; echo \"failing check: $CHECK\";"
        " echo \"host role: ${RUN_ROLE:-cluster} (submit host, no GPU)\";"
        " echo \"host memory: $(free -g 2> /dev/null | awk '/^Mem:/{print $7 \" GiB available of \" $2 \" GiB\"}' || true)\";"
        " echo \"time: $(date '+%Y-%m-%d %H:%M:%S')\"; echo \"--- last 50 lines of $TL\"; tail -n 50 \"$TL\";"
        " } > \"$OUT_ROOT/_diag/$SN.txt\" 2>&1 || true; }",
        "trap 'rc=$?; if [ $rc -ne 0 ]; then diag $rc; else rm -f \"$OUT_ROOT/_diag/$SN.txt\"; fi' EXIT",
        'CHECK="cluster.env values PARTITION ACCOUNT LOG_ROOT SLURM_RESTRICT (the node restriction)"',
        "for v in PARTITION ACCOUNT LOG_ROOT SLURM_RESTRICT; do",
        '  case "${!v:-<>}" in *"<"*">"*) echo "$v is not set or still a placeholder in cluster.env (setup_cluster.sh)"; exit 7;; esac',
        "done",
        "# nodes found bad (submit_chain.sh probes, or a task that could not start its GPU) join the exclude list",
        'BAD=$( { sort -u "$OUT_ROOT/bad_nodes.txt" 2> /dev/null || true; } | { grep . || true; } | paste -sd, -)',
        'RESTRICT="$SLURM_RESTRICT"',
        'if [ -n "$BAD" ]; then case "$RESTRICT" in *--exclude=*) RESTRICT=$(echo "$RESTRICT" | sed "s/--exclude=\\([^ ]*\\)/--exclude=\\1,$BAD/");;'
        ' *) RESTRICT="$RESTRICT --exclude=$BAD";; esac; fi',
        'SB="--partition=$PARTITION --account=$ACCOUNT $RESTRICT"',
        'echo "node restriction: $RESTRICT (nodes ${NODELIST:-not recorded}; bad nodes: ${BAD:-none})"',
        'CHECK="smoke marker $OUT_ROOT/_smoke.ok"',
        'if [ ! -f "$OUT_ROOT/_smoke.ok" ]; then',
        '  echo "no $OUT_ROOT/_smoke.ok: run cluster/env/smoke_cluster.sh first (it must pass)"; exit 6',
        "fi"]



def write_outputs(J, words, conflicts, cells, kept, cards, out_dir, staged=True, pins=True):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "g*.s*.tsv")) + glob.glob(os.path.join(out_dir, "smoke_*.tsv")) + glob.glob(os.path.join(out_dir, "task_*.sbatch")) +             glob.glob(os.path.join(out_dir, "gen_cfg", "*.json")):
        os.remove(f)                     # stage lists, templates and configs of an earlier generation
    cols = ["id", "group", "stage", "script", "model", "datasets", "shard", "out", "depends_on", "est_card_seconds",
            "rate_source", "cmd"]
    with open(os.path.join(out_dir, "joblist.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in J.rows:
            w.writerow(r)
    stages = collections.OrderedDict()
    for r in J.rows:
        stages.setdefault((r["group"], r["stage"]), []).append(r)
    groups = list(collections.OrderedDict((g, 1) for g, _ in stages))
    for (g, s), rs in stages.items():
        with open(os.path.join(out_dir, "%s.%s.tsv" % (g, s)), "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write("%s\t%s\n" % (r["id"], r["cmd"]))
    for g in groups:
        mx = max(r["est_card_seconds"] for r in J.rows if r["group"] == g)
        hours = min(48, max(2, int(math.ceil(mx * 2 / 3600.0))))
        txt = SBATCH.format(group=g, time="%d:00:00" % hours)
        if g == "g9_merge":          # B-EXT2 item 0: the merge reruns every round (idempotent), done marker or not
            assert txt.count(DONE_SKIP) == 1
            txt = txt.replace(DONE_SKIP, MERGE_RERUN)
        open(os.path.join(out_dir, "task_%s.sbatch" % g), "w", encoding="utf-8", newline="\n").write(txt)
    build = [r for r in J.rows if r["group"] == "g0_xllm"]
    gsum = {g: (sum(1 for r in J.rows if r["group"] == g), sum(r["est_card_seconds"] for r in J.rows if r["group"] == g) / 3600.0)
            for g in groups}

    def say(g):
        return 'echo "group %s: %d jobs, %.1f card-hours (%s)"' % (g, gsum[g][0], gsum[g][1], GROUP_LABEL.get(g, g))
    head = ["#!/bin/bash", "set -euo pipefail", "cd \"$(dirname \"$0\")\"", "CARDS=${CARDS:-%d}" % cards]
    first = head[:1] + ["# Step 1 of 2: the preflight job alone (group g0_xllm): environment record, import and config checks of",
                        "# the four family venvs, the xllm build and its import check ($OUT_ROOT/_preflight.json). When it has",
                        "# finished OK it leaves $OUT_ROOT/_done/%s; then run jobs/submit_10_rest.sh." % (build[0]["id"] if build else "-"),
                        "# Needs $OUT_ROOT/_smoke.ok (cluster/env/smoke_cluster.sh). Usage: source <OUT_ROOT>/cluster.env && bash jobs/submit_00_xllm.sh"] + head[1:]
    env_rel = os.path.relpath(os.path.join(HERE, "env"), out_dir).replace(os.sep, "/")
    first += gates(env_rel)
    if build:
        first.append(say("g0_xllm"))
        first.append('CHECK="sbatch of task_g0_xllm.sbatch"')
        first.append("sbatch --parsable $SB --output=$LOG_ROOT/ra20_g0_xllm_%A_%a.out --array=1 task_g0_xllm.sbatch g0_xllm.s1_build.tsv")
    lines = head[:1] + [
        "# Step 2 of 2: every other job in queue order, after the xllm build-and-import job finished OK. One pipeline",
        "# = one (group, script, model, output folder); its stages chain with afterok, so a failing config blocks only",
        "# its own pipeline (RA-20 item 4). Pipelines and groups queue at once; the order below is the queue order for",
        "# a throttled queue (item 8). Each sbatch call names the task indices of one stage list. The merge job waits",
        "# for every pipeline (afterany). Usage: source <OUT_ROOT>/cluster.env && bash jobs/submit_10_rest.sh",
        "# (jobs/submit_chain.sh runs it as its last step.) It stops without $OUT_ROOT/_smoke.ok, without the preflight",
        "# done marker, and when cluster.env lacks PARTITION, ACCOUNT, LOG_ROOT or the node restriction SLURM_RESTRICT."]
    if staged:
        lines += ["# Run jobs/stage_done.sh first: it copies the done folders (done_local.json) into $OUT_ROOT and leaves",
                  "# $OUT_ROOT/_staged.ok; without that marker this script stops."]
    else:
        lines += ["# These lists come from gen_jobs.py --no-local-done: every job runs on the cluster, nothing is staged."]
    lines += head[1:]
    lines += gates(env_rel)
    if build:
        lines += ['CHECK="preflight and xllm build done marker $OUT_ROOT/_done/%s"' % build[0]["id"],
                  'if [ ! -f "$OUT_ROOT/_done/%s" ]; then' % build[0]["id"],
                  '  echo "the xllm build-and-import job %s has not finished OK (no $OUT_ROOT/_done/%s): run'
                  ' submit_00_xllm.sh and wait"; exit 3' % (build[0]["id"], build[0]["id"]),
                  "fi"]
    if staged:
        lines += ['CHECK="staged done folders $OUT_ROOT/_staged.ok"',
                  'if [ ! -f "$OUT_ROOT/_staged.ok" ]; then',
                  '  echo "the done folders are not staged (no $OUT_ROOT/_staged.ok): run'
                  ' DONE_SRC=<pilot artifact folder> OUT_ROOT=$OUT_ROOT bash jobs/stage_done.sh"; exit 5',
                  "fi"]
    lines += [
        "# held units: a unit (ouro, gen, raven, mcleish, parcae, ifm) whose preflight or smoke failed",
        "# ($OUT_ROOT/_held/<unit>.held, written by env/preflight.sh and env/smoke_cluster.sh) is submitted held",
        "# (sbatch --hold); its job ids go to $OUT_ROOT/_held/<unit>.jobs, and jobs/release_family.sh <unit> reruns its",
        "# preflight and smoke and releases them. The merge job waits for them (afterany) and is never held.",
        'mkdir -p "$OUT_ROOT/_held"',
        'H() { if [ -f "$OUT_ROOT/_held/$1.held" ]; then echo "--hold"; fi; }',
        'K() { if [ -f "$OUT_ROOT/_held/$1.held" ]; then echo "$2" >> "$OUT_ROOT/_held/$1.jobs"; fi; }',
        "# every submission: Slurm job id, stage list, task indices (jobs/resubmit_pipeline.sh matches the queue to tasks)",
        'mkdir -p "$OUT_ROOT/_chain"',
        'S() { echo "$1 $2 $3" >> "$OUT_ROOT/_chain/submitted.tsv"; }',
        'for f in ouro gen raven mcleish parcae ifm; do',
        '  rm -f "$OUT_ROOT/_held/$f.jobs" "$OUT_ROOT/_held/$f.released"',
        '  if [ -f "$OUT_ROOT/_held/$f.held" ]; then echo "family $f held ($(head -1 "$OUT_ROOT/_held/$f.held")); its jobs are'
        ' submitted held (release after the fix: bash jobs/release_family.sh $f)"; fi',
        "done"]
    lines.append('CHECK="sbatch of the stage arrays (the last printed line is the call that failed)"')
    lines.append('ALL=""')
    pos = {}
    for (g, s_), rs in stages.items():
        for i, r in enumerate(rs):
            pos[r["id"]] = i + 1
    pipes = collections.OrderedDict()
    for r in sorted(J.rows, key=lambda r: GROUP_ORDER.index(r["group"])):      # stable: queue order inside a group
        pipes.setdefault((r["group"], r["script"], r["model"], r["out"]), collections.OrderedDict()).setdefault(r["stage"], []).append(r)
    k = 0
    var_of = {}
    said = set()
    for (g, scr, mod, out), st in pipes.items():
        if g == "g0_xllm":
            continue                     # submitted by submit_00_xllm.sh; submit_10_rest.sh checks its marker
        if g not in said:
            lines.append(say(g))
            said.add(g)
        prev = ""
        for s_, rs in st.items():
            k += 1
            var = "J%d" % k
            idx = ",".join(str(pos[r["id"]]) for r in rs)
            dep = "--dependency=afterok:$%s " % prev if prev else ""
            if not prev:                 # first stage: jobs of other pipelines it waits for (the xllm build)
                ext = sorted(set(var_of[d] for r in rs for d in r["depends_on"].split(",") if d in var_of))
                if ext:
                    dep = "--dependency=afterok:%s " % ":".join("$" + v for v in ext)
            if g == "g9_merge":
                dep = "--dependency=afterany${ALL} "
            fam = "none" if g == "g9_merge" else family_of(rs[0]["cmd"])
            lines.append("%s=$(sbatch --parsable $SB $(H %s) --output=$LOG_ROOT/ra20_%s_%%A_%%a.out %s--array=%s%%$CARDS task_%s.sbatch %s.%s.tsv)   # %s %s %s" % (
                var, fam, g, dep, idx, g, g, s_, scr, mod, s_))
            lines.append("K %s $%s" % (fam, var))
            lines.append("S $%s %s.%s.tsv %s" % (var, g, s_, idx))
            if g != "g9_merge":
                lines.append('ALL="$ALL:$%s"' % var)
            for r in rs:
                var_of[r["id"]] = var
            prev = var
    os.makedirs(os.path.join(out_dir, "gen_cfg"), exist_ok=True)
    write_smoke(out_dir)
    for name, cfg in sorted(GEN_CFGS.items()):
        with open(os.path.join(out_dir, "gen_cfg", name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, indent=1, sort_keys=True)
            f.write("\n")
    lines.append('echo "submitted: %d jobs, %.1f card-hours (the xllm build excluded)"' % (
        sum(v[0] for g, v in gsum.items() if g != "g0_xllm"), sum(v[1] for g, v in gsum.items() if g != "g0_xllm")))
    if pins:
        write_hashes(out_dir)
    for f in ("submit_00_xllm.sh", "submit_10_rest.sh"):
        if os.path.exists(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
    open(os.path.join(out_dir, "submit_00_xllm.sh"), "w", encoding="utf-8", newline="\n").write("\n".join(first) + "\n")
    open(os.path.join(out_dir, "submit_10_rest.sh"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

    # cells.csv: the final word per manifest cell
    jobs_of = collections.defaultdict(list)
    with open(os.path.join(out_dir, "cells.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["grid", "card", "model", "dataset", "manifest_status", "word", "detail"])
        for c in cells:
            key = (c["card"], c["model"], c["dataset"])
            word, det = words[key]
            w.writerow([c["grid"], c["card"], c["model"], c["dataset"], c["status"], word, det])
    with open(os.path.join(out_dir, "skip_conflicts.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["card", "model", "dataset", "routed_script", "cells_skipped_reason", "cells_skipped_why"])
        for key, scr, rsn, why in conflicts:
            w.writerow(list(key) + [scr, rsn, why])
    return groups, stages


# ----------------------------------------------------------------------------------------------- the cluster smoke
SMOKE_CFG = {"n_screen": 12, "n_sent": 3, "n_shift": 3}     # the GPU smoke of 2026-10-08 (SMOKE_HASHES.txt, out_new)


def smoke_lists():
    """Task lists of cluster/env/smoke_cluster.sh: two pipelines at 3 items, built from the same command strings as the
    job list (task-line form, venv activation included): k17_varbench on Ouro-1.4B (window, scan, two row shards,
    merge, summarize; the g1_copy template) and steer_gen setting a1 on GSM8K (checks, prep and screen, select, row
    r00, analyze; the g2_gen template), plus smoke_fam.tsv: one three-item line per further family (raven k17_public
    mcl_nr D1, Parcae k17_parcae parcae-370m D1, IFM k17_freeze_ifm ifm_p1_ouro D1, McLeish generation checks and row
    r00 on 3 problems of the shipped selection). The smoke writes under its own
    output tree, never the run's."""
    k, out = "ouro", "$OUT_ROOT/B/k17_varbench/ouro"
    base = "--models %s --data %s/varbench --out %s --role cluster" % (k, DATA, out)
    tail = " --pool pairs --n_vb 3 --vbd_n %s=3 --datasets VBD --no_gradient_rank" % k
    cw = base + " --plan $VB_PLAN --plan_sha $VB_PLAN_SHA" + tail
    plan_m = "%s/plan_M_%s.md" % (out, k)
    cm = base + " --plan %s --plan_sha $(sha256sum %s | cut -c1-64)" % (plan_m, plan_m) + tail
    vb = [("s2_window", "%s %s/k17_varbench.py %s --phase W --tag W_%s && %s %s/cluster/vb_slots.py --plan $VB_PLAN --out %s --model %s" % (
              PY, LF, cw, k, PY, LC, out, k)),
          ("s3_scan", "%s %s/k17_varbench.py %s --phase M --stage scan --tag M_%s" % (PY, LF, cm, k))]
    vb += [("s4_rows", "%s %s/k17_varbench.py %s --phase M --stage rows --shard %d/2 --tag M_%s" % (PY, LF, cm, i, k)) for i in (0, 1)]
    vb += [("s5_merge", "%s %s/k17_varbench.py %s --phase M --stage merge --tag M_%s && mkdir -p %s/shards && mv %s/*.shard*of*.* %s/shards/" % (
               PY, LF, cm, k, out, out, out)),
           ("s6_summarize", "%s %s/k17_varbench.py --summarize --out %s --plan %s && %s $REPO/loopedcot/cluster/extract/run.py --script k17_varbench.py --src %s/k17_varbench.py --job %s" % (
               PY, LF, out, plan_m, PY, LF, out))]
    st, out = "a1", "$OUT_ROOT/A/steer_gen/a1_gsm8k"
    env = "K17_DATASET=gsm8k PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs" % (out, out)
    base = "cd %s && %s %s -m prod.steer_gen {mode} --setting %s --out %s --mem-fraction $GEN_MEM_FRACTION" % (TREE, env, PY, st, out)
    cj = "--config-json %s/cluster/jobs/gen_cfg/smoke_a1_gsm8k.json" % LC
    gen = [("s1_checks", base.format(mode="checks").replace("--out %s" % out, "--out %s/chk" % out) + " %s %s && %s %s/gate.py %s/chk/checks_%s.json" % (G3_CLUSTER, cj, PY, LF, out, st)),
           ("s2_prep_screen", "%s --settle-dir $SETTLE %s && %s %s" % (base.format(mode="prep"), cj, base.format(mode="screen"), cj)),
           ("s3_select", "cd %s && %s %s %s/cluster/steer_select.py --setting %s --out %s --settle-dir $SETTLE %s --min_scored 3 --max_scored 3" % (
               TREE, env, PY, LC, st, out, cj)),
           ("s4_row", "%s --row r00 %s" % (base.format(mode="run"), cj)),
           ("s6_analyze", "%s %s && %s $REPO/loopedcot/cluster/extract/run.py --script steer_gen.py --src %s/prod/steer_gen.py --job %s" % (
               base.format(mode="analyze"), cj, PY, TREE, out))]
    # one three-item line per further family, the job-list command with --debug 3 (3 kept items; in-run identities are
    # gated by cluster/env/smoke_check.py; SMOKE_VALUES.json has no reference values for them, so values are reported)
    ptok = "$HF_HOME/hub/models--SandyResearch--parcae-tokenizer/snapshots/6247b5d0592876b73660f34c2dd16c9db9e9045c/tokenizer.json"
    o_r, o_p, o_i = "$OUT_ROOT/B/k17_public/mcl_nr_D1", "$OUT_ROOT/B/k17_parcae/parcae-370m", "$OUT_ROOT/C/k17_freeze_ifm/ifm_p1_ouro_D1"
    o_m = "$OUT_ROOT/A/steer_gen_raven/mcleish_smoke"
    menv = "K17_DATASET=gsm8k PROD_ART=%s/prod_art PROD_LOGS=%s/prod_logs" % (o_m, o_m)
    fam_m = ("f_mcleish", "source $VENV_RAVEN/bin/activate && %s %s/cluster/mcl_subset.py %s/mcleish_setting/items.json "
                          "%s/items.json --n 3 && cd %s && %s %s %s/steer_gen_raven.py checks --out %s/chk --n 3 "
                          "--mem-fraction $GEN_MEM_FRACTION && %s %s %s/steer_gen_raven.py run --out %s --row r00 "
                          "--mem-fraction $GEN_MEM_FRACTION" % (PY, LC, DATA, o_m, TREE, menv, PY, LF, o_m, menv, PY, LF, o_m))
    fam = [("f_raven", "source $VENV_RAVEN/bin/activate && %s %s/k17_public.py --run --model mcl_nr --data %s/public --out %s "
                       "--plan $PUB_PLAN --datasets D1 --debug 3 --budget_min 100000000 --full_search_cap 0 --no_reduced "
                       "--no_seeds --no_gradient_rank --stage all && %s %s/k17_public.py --summarize --out %s --plan $PUB_PLAN && "
                       "%s $REPO/loopedcot/cluster/extract/run.py --script k17_public.py --src %s/k17_public.py --job %s" % (
                           PY, LF, DATA, o_r, PY, LF, o_r, PY, LF, o_r)),
           ("f_parcae", "source $VENV_PARCAE/bin/activate && %s %s/k17_parcae.py --run --model parcae-370m --data %s/parcae --out %s "
                        "--tok %s --datasets D1 --debug 3 --budget_min 100000000 --plan $PAR_PLAN --no_gradient_rank && "
                        "%s %s/k17_parcae.py --summarize --out %s --plan $PAR_PLAN && %s $REPO/loopedcot/cluster/extract/run.py "
                        "--script k17_parcae.py --src %s/k17_parcae.py --job %s" % (PY, LF, DATA, o_p, ptok, PY, LF, o_p, PY, LF, o_p)),
           ("f_ifm", "source $VENV_IFM/bin/activate && %s %s/k17_freeze_ifm.py --run --model ifm_p1_ouro --out %s --plan $FRZO_PLAN "
                     "--plan_sha256 $FRZO_PLAN_SHA --items_d1 %s/k17_freeze/items_d1.json --datasets D1 --debug 3 && "
                     "%s %s/k17_freeze.py --summarize --out %s && %s $REPO/loopedcot/cluster/extract/run.py --script "
                     "k17_freeze_ifm.py --src %s/k17_freeze_ifm.py --job %s" % (PY, LF, o_i, DATA, PY, LF, o_i, PY, LF, o_i)),
           fam_m]
    lists = {}
    for g, rows in (("g1_copy", vb), ("g2_gen", gen), ("fam", fam)):
        n = collections.Counter()
        lists[g] = []
        for stage, cmd in rows:
            n[stage] += 1
            lists[g].append(("smoke.%s.%s.%03d" % (g, stage, n[stage]), cmd if cmd.startswith("source ") else
                             "source $VENV_OURO/bin/activate && " + cmd))
    return lists


def write_smoke(out_dir):
    for g, rows in smoke_lists().items():
        with open(os.path.join(out_dir, "smoke_%s.tsv" % g), "w", encoding="utf-8", newline=chr(10)) as f:
            for jid, cmd in rows:
                f.write("%s" % jid + chr(9) + cmd + chr(10))
    GEN_CFGS["smoke_a1_gsm8k.json"] = dict(SMOKE_CFG)


def handover_files():
    """Files of the bundle that jobs run, import or read, relative to the folder above loopedcot/ ($REPO)."""
    lc = os.path.dirname(HERE)
    out = []
    for d in ("cluster/pilot", "cluster/extract", "cluster/plans", "cluster/data", "cluster/reference_local"):
        for root, dirs, files in os.walk(os.path.join(lc, d)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for f in sorted(files):
                if not f.endswith(".pyc"):
                    out.append(os.path.relpath(os.path.join(root, f), os.path.dirname(lc)).replace(os.sep, "/"))
    for f in ("cluster/gen_jobs.py", "cluster/merge.py", "cluster/steer_select.py", "cluster/vb_slots.py",
              "cluster/build_cf_pool.py", "cluster/build_lre_pool.py", "cluster/done_local.json", "cluster/smoke_hash.py", "cluster/mcl_subset.py",
              "cluster/SMOKE_HASHES.txt", "cluster/SMOKE_VALUES.json",
              "cluster/manifest.yaml", "cluster/cells_skipped.json", "cluster/weights.json",
              "cluster/env/build_xllm.sh", "cluster/env/check_xllm.py", "cluster/env/preflight.sh",
              "cluster/env/preflight.py", "cluster/env/setup_cluster.sh", "cluster/env/smoke_cluster.sh",
              "cluster/env/smoke_check.py", "cluster/env/requirements_cluster_ouro.txt",
              "cluster/env/requirements_cluster_raven.txt", "cluster/env/requirements_cluster_parcae.txt",
              "cluster/env/requirements_cluster_ifm.txt", "cluster/jobs/submit_chain.sh", "cluster/jobs/status.sh",
              "cluster/jobs/stage_done.sh", "cluster/jobs/release_family.sh",
              "cluster/jobs/resubmit_pipeline.sh"):
        out.append("loopedcot/" + f)
    return sorted(set(out))


# steer_gen.py, the prod modules it imports and the files they read (prod/tasks/data: prompts, rows, revisions), as
# copies under cluster/pilot/tree/ (byte-identical to loopedcot/prod and loopedcot/data_steer; pinned through the
# cluster/pilot walk above). The committed set is loopedcot/cluster/ only.
STEER_GEN_FILES = ["prod/steer_gen.py", "prod/__init__.py", "prod/common.py", "prod/config.py", "prod/config.yaml",
                   "prod/generate.py", "prod/models/__init__.py", "prod/models/base.py", "prod/models/huginn.py",
                   "prod/models/mcleish.py", "prod/models/ouro.py", "prod/models/raven.py", "prod/tasks/__init__.py",
                   "prod/tasks/data/model_revisions.json", "prod/tasks/data/hashes.json",
                   "prod/tasks/data/prompt_gsm8k.txt", "prod/tasks/data/prompt_svamp.txt",
                   "prod/tasks/data/rows_gsm8k.jsonl", "prod/tasks/data/rows_svamp.jsonl",
                   "data_steer/varbench/pairs_gsm8k.jsonl", "data_steer/svamp/pairs_svamp.jsonl",
                   # cluster/pilot/steer_gen_math.py (RA-23 item 4) reads data_steer/<ds>/pairs_<ds>.jsonl
                   "data_steer/asdiv/pairs_asdiv.jsonl", "data_steer/asdiv/MANIFEST.json", "data_steer/asdiv/checks.json",
                   "data_steer/mawps/pairs_mawps.jsonl", "data_steer/mawps/MANIFEST.json", "data_steer/mawps/checks.json"]


# steer_gen checks on the cluster (smoke and every setting): 12 items, gate G3 on all 12 with the n - n/32 rule; when
# only the agreement count misses and every disagreement sits at a recompute margin < 1 logit, G3 passes as Deviation
# D5m and checks_<setting>.json records the measured agreement (the generation code has no recompute decode path)
G3_CLUSTER = "--n 12 --g3-items 12 --g3-measured-rate"


def family_of(cmd):
    """The hold unit of a task line: mcleish (steer_gen_raven), gen (the Ouro generation scripts, steer_gen and
    steer_gen_math), else the venv it activates (ouro, raven, parcae, ifm); none: no venv, never held. A unit whose
    preflight (its venv) or smoke failed is held; the others run."""
    if "steer_gen_raven" in cmd:
        return "mcleish"
    if "VENV_OURO" in cmd and "steer_gen" in cmd:
        return "gen"
    for v, f in (("VENV_OURO", "ouro"), ("VENV_RAVEN", "raven"), ("VENV_PARCAE", "parcae"), ("VENV_IFM", "ifm")):
        if v in cmd:
            return f
    return "none"


def write_hashes(out_dir):
    """jobs/scripts_sha256.txt: sha256 of every bundle file a job runs, imports or reads (sha256sum -c format, paths
    relative to $REPO). The array template refuses to start when any of them differs."""
    import hashlib
    top = os.path.dirname(os.path.dirname(HERE))
    with open(os.path.join(out_dir, "scripts_sha256.txt"), "w", encoding="utf-8", newline=chr(10)) as f:
        for rel in handover_files():
            pth = os.path.join(top, rel)
            f.write("%s  %s" % (hashlib.sha256(open(pth, "rb").read()).hexdigest(), rel) + chr(10))


def planned_table(cells, words, kept, out_dir):
    """Planned n per routed cell: the manifest pool, the script's cap, the latest kept count (planned, not measured)."""
    rows = []
    for c in cells:
        key = (c["card"], c["model"], c["dataset"])
        if words[key][0] not in ("run", "scan"):
            continue
        r = route(*key)
        unit = r[1]
        m, ds = c["model"], c["dataset"]
        pool, cap, last_k, plan, note = "", "", "", "", ""
        if unit == "vb":
            k = VB_KEY[m]
            kp = kept.get(("vb", k, "VB"))
            ratio = (kp[0] / float(kp[1])) if kp and kp[1] else None
            if ds == "VB":
                pool, cap = VB_POOL, "none (--pool pairs, --n_vb 100000)"
                last_k = "%s of %s scanned (s200)" % (kp[0], kp[1]) if kp else "none"
                plan = int(round(VB_POOL * ratio)) if ratio else "<= %d" % VB_POOL
            else:
                kd = kept.get(("vb", k, "VBD"))
                pool, cap = VBD_POOL, "none (--pool pairs, --vbd_n 100000)"
                last_k = "%s (vbd_n cap)" % kd[0] if kd else "none"
                plan = int(round(VBD_POOL * ratio)) if ratio else "<= %d" % VBD_POOL
            note = "clean-pass rate of the s200 scan applied to the pool" if ratio else "no earlier run"
        elif unit == "pub":
            gk = PUB_GROUP[m]
            kp = kept.get(("pub", PUB_RUNNER[m], ds))
            pool = POOL_N.get(("pub", PUB_FAM[gk], ds), "")
            cap = ("none (--targets CF=100000; the shared CF pool, no subject cap)" if ds == "CF" else
                   "none (--targets %s=100000; selector caps per subject or problem as in every family)" % ds)
            last_k = "%s of %s scanned" % (kp[0], kp[1]) if kp else "none"
            if kp and kp[1] and pool and ds != "CF":
                plan = int(round(pool * min(1.0, kp[0] / float(kp[1]))))
                note = "candidates of the family's pool file x joint kept rate of the last scan"
            else:
                plan = int(round((pool or 0) * PASS_FALLBACK[ds][0]))
                note = "no earlier scan of this family: pass rate assumed from %s" % PASS_FALLBACK[ds][1]
        elif unit == "par":
            kp = kept.get(("par", PAR_KEY[m], ds))
            pool = POOL_N.get(("par", "*", ds), "")
            cap = "none (--targets %s=100000; selector caps per subject)" % ds
            last_k = "%s of %s scanned" % (kp[0], kp[1]) if kp else "none"
            plan = planned_par(PAR_KEY[m], ds, kept)
            note = "candidates x kept rate of the last scan"
        elif unit == "gen":
            st = GEN[m]
            kp = kept.get(("gen", st, ds))
            pool = 522 if ds == "GSM8K_gen" else 995
            cap = "none: every problem that passes both clean filters is scored (sentinels from the prep items left over)"
            last_k = "%s scored (cap 150); %s" % (kp[0], kp[2].split("; ")[-1]) if kp else "none"
            plan = PLAN_GEN.get((st, ds))
            note = ("SETTLE-correct (GSM8K 428 / 451; SVAMP 252 / 252) x B-pass rate of run 1 / run 5"
                    if st in ("a1", "a2") else "no SETTLE count read; assumed")
        elif unit in ("k15f", "k15p", "k16o", "k16i"):
            mk = K16O.get(m, m) if unit == "k16o" else m
            kp = kept.get((unit, mk, ds))
            pool = kp[3] if kp and len(kp) > 3 and kp[3] else ""
            cap = ("none (--full_pool; at most 2 pairs per subject, 1 per D4/D5 template)" if unit != "k15p"
                   else "none (--n_half 100000; at most 2 pairs per subject)")
            last_k = "%s of %s scanned (quota run)" % (kp[0], kp[1]) if kp else "none"
            plan = planned_quota(unit, mk, ds, kept)
            note = "pool candidates x pass rate of the pilot scan (quota-limited scan; rough)"
            if unit == "k15f" and ds == "CF":
                pool = "%d of the 3,741-pair CF pool well formed on the IFM tokenizer" % CF_POOL_IFM
                cap = "none (--full_pool --cf_pool: the shared CF pool, no quota and no subject cap)"
                last_k = "none on the shared pool (the pilot scanned another CF pool)"
                note = "pool pairs well formed on the tokenizer x the production pass rate of the pool (783 of 3,741)"
        elif unit in ("frzo", "frzi"):
            pool = FRZ_IFM_D1 if (unit == "frzi" and ds == "D1") else FRZ_POOL[ds]
            cap = {"D1": "none (the RA-3 identity list, then the model's clean filter)",
                   "D2": "none (cap 4 per edited operand value, as RA-3)",
                   "D4": "none (quota lifted; cap 2 per cluster side)"}[ds]
            last_k = "RA-3 on Ouro-1.4B base: %d" % {"D1": 267, "D2": 263, "D4": 160}[ds]
            plan = planned_frz2(unit, ds)
            note = "ceiling x the clean-pass fallback of %s (%s)" % (ds, PASS_FALLBACK[ds][1])
        elif unit == "genm":
            pool = GENM_POOL[GENM_DS[ds]]
            cap = "none: every problem that passes both screens is scored"
            last_k = "none (smoke only)"
            plan = int(round(pool * GENM_PASS))
            note = "pool x pass rate %.1f assumed" % GENM_PASS
        elif unit == "genr" and m == "huginn_0125":
            pool, cap, last_k, plan = 522, "none (every prep problem that passes both screens)", "none", GENR_HUG_N
            note = "prep %d (SETTLE k32 pre-filter, shipped prep.json); planned scored assumed" % GENR_HUG_PREP
        elif unit == "genr":
            pool, cap, last_k, plan = 522, "none (every problem that passed both screens of the shared-host run)", "237 selected", GENR_N
            note = "the shared-host run's select (rows r00, r01 done on the same problems)"
        elif unit == "publre":
            gk = PUB_GROUP[m]
            pool = POOL_N.get(("publre", PUB_FAM[gk], "D1"), "")
            cap = "the LRE pool file (cluster/build_lre_pool.py: per-relation cap, 2 pairs per subject); --targets D1=100000"
            last_k = "none"
            plan = int(round((pool or 0) * LRE_PASS))
            note = "pool x pass rate %.2f assumed" % LRE_PASS
        elif unit == "frz":
            kp = kept.get(("pub", PUB_RUNNER[m], ds))
            pool = POOL_N.get(("pub", "llama" if m == "mcleish_llama32_r32" else "huginn", ds), "")
            cap = "none (every clean-passing candidate of the k17_public pool file; D2 one per operand cluster)"
            last_k = "%s of %s scanned (k17_public)" % (kp[0], kp[1]) if kp else "none"
            plan = planned_frz(m, ds, kept)
            note = "candidates x kept rate of the k17_public scan of the model"
        rows.append([c["grid"], c["card"], m, ds, c["status"], r[0], pool, cap, last_k, plan, note])
    hdr = ["grid", "card", "model", "dataset", "manifest_status", "script", "pool", "cap", "latest_kept", "planned_n", "basis"]
    with open(os.path.join(out_dir, "planned_n.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(hdr)
        w.writerows(rows)
    md = ["# Planned n per cell (PLANNED, not measured)", "",
          "Generated by cluster/gen_jobs.py. One pool file per dataset for every model (RA-20 item 9): every candidate",
          "of the pool is scanned and every item that passes the model's clean filter is kept; no per-model cap.",
          "latest_kept is the last pilot run's kept count (those runs had quotas); planned_n is the pool times the",
          "pass rate of that run. The run itself decides every n.", "",
          "Generation (grid A): scored = every problem that passes both clean filters. This week's runs (GSM8K run 1",
          "and run 5, SVAMP) scored 150: prep_a orders the pool problems that SETTLE answers correctly by a seeded",
          "shuffle (random.Random(\"k17_gen:20261007\") over the sorted problem ids, steer_gen.shuffled), screens the",
          "first n_screen of that order (240, extended by 80) on the B side (A side from the SETTLE cell), and select_a",
          "keeps the first 150 in that shuffled order that pass both; the 22 sentinels are the prep items left over",
          "nearest the prompt-length quantiles of the scored set, the 8 shift rows the shortest GSM8K eval prompts",
          "outside both sets. Not by id and not by length: by the seeded shuffle order.", "",
          "CF (CounterFact): " + CF_NOTE, "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    open(os.path.join(out_dir, "planned_n.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return rows


def card_hours(J, groups, cards, out_dir):
    out = {}
    for g in groups:
        rs = [r for r in J.rows if r["group"] == g]
        tot = sum(r["est_card_seconds"] for r in rs) / 3600.0
        # critical path: the longest chain of stages (stage maxima summed) bounds the wall time
        by_stage = collections.OrderedDict()
        for r in rs:
            by_stage.setdefault((r["model"], r["stage"]), []).append(r["est_card_seconds"])
        chains = collections.defaultdict(float)
        for (m, s), v in by_stage.items():
            chains[m] += max(v) / 3600.0
        shared = sum(r["est_card_seconds"] for r in rs if "(shared" in r["rate_source"]) / 3600.0
        assumed = sum(r["est_card_seconds"] for r in rs if "assumed" in r["rate_source"]) / 3600.0
        out[g] = {"jobs": len(rs), "card_hours": round(tot, 1), "card_hours_from_shared_gpu_rates": round(shared, 1),
                  "card_hours_from_assumed_rates": round(assumed, 1),
                  "wall_hours_at_%d_cards" % cards: round(max(tot / cards, max(chains.values()) if chains else 0), 1),
                  "longest_chain_hours": round(max(chains.values()) if chains else 0, 1)}
    tot = round(sum(v["card_hours"] for v in out.values()), 1)
    json.dump({"cards": cards, "groups": out, "card_hours_total": tot}, open(os.path.join(out_dir, "card_hours.json"), "w"), indent=1)
    md = ["# Card-hours per queue group (estimate)", "",
          "Per-item rates are seconds per item measured in the pilot artifacts on the fast host, one process per card,",
          "except where marked: rates measured on a GPU shared with other jobs are upper bounds for a dedicated card;",
          "assumed rates have no measurement behind them (see rate_source in joblist.tsv). %d cards, one process each." % cards, "",
          "| group | jobs | card-hours | from shared-GPU rates | from assumed rates | longest chain (h) | wall at %d cards (h) |" % cards,
          "|---|---|---|---|---|---|---|"]
    for g, v in out.items():
        md.append("| %s | %d | %.1f | %.1f | %.1f | %.1f | %.1f |" % (g, v["jobs"], v["card_hours"], v["card_hours_from_shared_gpu_rates"],
                                                                 v["card_hours_from_assumed_rates"], v["longest_chain_hours"], v["wall_hours_at_%d_cards" % cards]))
    md.append("| total | %d | %.1f | | | | |" % (len(J.rows), tot))
    open(os.path.join(out_dir, "card_hours.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return out, tot


def load_done(path):
    """done_local.json entries; with $PILOT_ARTIFACTS set, every evidence file must exist and results.csv must have the
    recorded sha256 (else the generator stops)."""
    if not path or not os.path.exists(path):
        return []
    d = json.load(open(path, encoding="utf-8"))["jobs"]
    if ART:
        import hashlib
        for e in d:
            for f in e.get("evidence", []):
                if not os.path.exists(os.path.join(ART, f)):
                    raise SystemExit("done_local.json: evidence missing under PILOT_ARTIFACTS: %s" % f)
            if e.get("results_csv"):
                h = hashlib.sha256(open(os.path.join(ART, e["results_csv"]), "rb").read()).hexdigest()
                if h != e["results_sha256"]:
                    raise SystemExit("done_local.json: %s sha256 %s, recorded %s" % (e["results_csv"], h, e["results_sha256"]))
    return d


def mark_done_words(cells, words, J, done_rows):
    """Cell words after the done jobs left the list: done when no job of the cell's output folder is left, else the
    job word with the done jobs named."""
    left = collections.Counter(r["out"] for r in J.rows)
    gone = collections.defaultdict(list)
    for r in done_rows:
        gone[r["out"]].append(r["shard"] or r["id"])
    for c in cells:
        key = (c["card"], c["model"], c["dataset"])
        tgt = cell_target(*key) if words.get(key, ("",))[0] in ("run", "scan") else None
        if tgt is None:
            continue
        out = "$OUT_ROOT/" + tgt[1]
        if out in gone:
            if not left.get(out):
                words[key] = ("done", "ran on our cards before the launch (done_local.json): %s" % ",".join(gone[out]))
            else:
                words[key] = (words[key][0], words[key][1] + "; done on our cards: %s" % ",".join(gone[out]))


def write_done(done_rows, out_dir):
    with open(os.path.join(out_dir, "done_local.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "group", "stage", "script", "model", "shard", "out", "copy_from_pilot_artifacts", "ledger"])
        for r in done_rows:
            w.writerow([r["id"], r["group"], r["stage"], r["script"], r["model"], r["shard"], r["out"],
                        r["done"].get("pulled", ""), r["done"].get("ledger", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.yaml"))
    ap.add_argument("--skipped", default=os.path.join(HERE, "cells_skipped.json"))
    ap.add_argument("--out", default=JOBS)
    ap.add_argument("--cards", type=int, default=48)     # 6 nodes x 8 GPUs, the run's share of the cluster
    ap.add_argument("--shard_hours", type=float, default=1.5)
    ap.add_argument("--done", default=os.path.join(HERE, "done_local.json"))
    ap.add_argument("--no-local-done", dest="no_local_done", action="store_true",
                    help="every cell of done_local.json runs on the cluster (same job ids: done jobs keep the ids they "
                         "have in jobs/done_local.tsv); default output jobs/nolocal/; submit_chain.sh DONE_SRC=none uses it")
    a = ap.parse_args()
    if a.no_local_done:
        a.done = ""
        if a.out == JOBS:
            a.out = os.path.join(JOBS, "nolocal")
    os.makedirs(a.out, exist_ok=True)
    m, order, last, cells = load_manifest(a.manifest)
    skips = load_skips(a.skipped)
    load_pool_counts(os.path.join(HERE, "pilot", "data"))
    kept = latest_kept()
    done = load_done(a.done)
    J, words, conflicts = build(cells, order, last, skips, kept, a.shard_hours * 3600, done)
    done_rows = J.split_done()
    mark_done_words(cells, words, J, done_rows)
    write_done(done_rows, a.out)
    groups, stages = write_outputs(J, words, conflicts, cells, kept, a.cards, a.out, staged=bool(done_rows),
                                   pins=not a.no_local_done)
    planned_table(cells, words, kept, a.out)
    per, tot = card_hours(J, groups, a.cards, a.out)
    cnt = collections.Counter(w for w, _ in words.values())
    print(json.dumps({"manifest_cells": len(cells), "jobs": len(J.rows),
                      "jobs_per_group": {g: sum(1 for r in J.rows if r["group"] == g) for g in groups},
                      "cell_words": cnt, "skip_conflicts": len(conflicts), "done_local_jobs": len(done_rows),
                      "card_hours": {g: v["card_hours"] for g, v in per.items()},
                      "card_hours_total": tot}, indent=1))


if __name__ == "__main__":
    main()
