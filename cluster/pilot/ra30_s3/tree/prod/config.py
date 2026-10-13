"""The ONE configuration file, loaded from YAML.

Loop counts per model, caps, horizon, datasets, batch width (and its per-job overrides),
forced-N and the calibration seed live in `prod/config.yaml`; this module is the LOADER. Every
default here is overridable on the command line of `prod.generate`, `prod.manifest` and
`prod.launcher`, in this order: the YAML, then `--config-json`, then the explicit flags. The
effective config is printed into every cells file's header row and into the manifest, so a number
read months later says which protocol produced it.

  python -m prod.config --print                       # the defaults, as JSON
  python -m prod.config --print --caps=0,64,512 --horizon=1024
  PROD_CONFIG_YAML=/path/to/other.yaml python -m prod.config --print   # a different YAML file

Nothing here imports torch, transformers or any model: the launcher must size jobs without loading a
checkpoint, and the manifest must be buildable on a laptop. PyYAML is the one new dependency
(`pip install pyyaml`; see README.md).
"""
import argparse
import hashlib
import json
import os

import yaml

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
#: the YAML file every default below is read from. `PROD_CONFIG_YAML` overrides it (tests, and a
#: deliberately different protocol file); the packaged default lives next to this module.
CONFIG_YAML = os.environ.get("PROD_CONFIG_YAML") or os.path.join(_PKG_DIR, "config.yaml")


def _read_yaml(path=None):
    with open(path or CONFIG_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


_RAW = _read_yaml()

# ------------------------------------------------------------------ protocol
#: caps of record. The extra caps {48, 96, 192, 384} are not run.
CAPS = list(_RAW["caps"])
CAPS_EXTRA = list(_RAW["caps_extra"])
#: natural-stop horizon: generation stops naturally, or at 4096 tokens.
HORIZON = int(_RAW["horizon"])
#: the forced-continuation protocol's own horizon ("Wait" injection, S13)
FORCED_HORIZON = int(_RAW["forced_horizon"])
#: budgets scored on a forced-continuation trace
FORCED_BUDGETS = list(_RAW["forced_budgets"])
#: is the closing think tag a natural stop on a chat-template Thinking checkpoint? True is what
#: every grid was generated under; False lets the chain run past the reasoning block to the task's
#: stop strings or the eos token (see config.yaml and prod/tasks/build_prompts).
THINK_TAG_IS_STOP = bool(_RAW.get("think_tag_is_stop", True))

#: the forced-continuation block: a SEPARATE protocol block, limited by default to
#: GSM8K and MATH500 on the four Ouro checkpoints at full N, switchable by config.
FORCED_BLOCK = json.loads(json.dumps(_RAW["forced_block"]))

#: production batch width, PINNED; stored per row.
BATCH_WIDTH = int(_RAW["batch_width"])
#: hard cap on the decode batch when the width is adaptive (`--batch-width=0`)
BATCH_CAP = int(_RAW["batch_cap"])

# ------------------------------------------------------------------ splits
SPLIT_SEED = int(_RAW["split_seed"])
N_CAL = int(_RAW["n_cal"])

# ------------------------------------------------------------------ datasets
#: the ten evaluation sets of record, with the N of record each.
DATASETS = dict(_RAW["datasets"])
TASK_ORDER = list(DATASETS.keys())

#: the ten BBH subtasks pooled into the `bbh` slot, in file order.
BBH_SUBTASKS = list(_RAW["bbh_subtasks"])

# ------------------------------------------------------------------ models: shapes, depths and
# per-job batch-width overrides, all read from the SAME `models:` block so they cannot drift apart.
MODEL_ORDER = list(_RAW["model_order"])
MODEL_SHAPES = {}
DEPTHS = {}
DEPTHS_EXTRA = {}
BATCH_WIDTH_OVERRIDES = {}
for _name, _m in _RAW["models"].items():
    MODEL_SHAPES[_name] = {k: v for k, v in _m.items()
                           if k not in ("depths", "depths_extra", "batch_width_overrides")}
    DEPTHS[_name] = [int(x) for x in (_m.get("depths") or [])]
    if _m.get("depths_extra"):
        DEPTHS_EXTRA[_name] = {t: [int(x) for x in ks] for t, ks in _m["depths_extra"].items()}
    if _m.get("batch_width_overrides"):
        BATCH_WIDTH_OVERRIDES[_name] = {int(k): int(v) for k, v in
                                        _m["batch_width_overrides"].items()}

#: fraction of a device's memory a GPU slot's running sum may occupy (the Spark rule)
MEM_UTIL = float(_RAW["mem_util"])
#: prompt tokens the KV estimate allows on top of the horizon (mean prompt + suffix + answer)
PROMPT_ALLOWANCE = int(_RAW["prompt_allowance"])
PLACEMENT_HORIZON = _RAW.get("placement_horizon")      # None = the run horizon (worst case)
#: workers per GPU. 1 runs exactly one job per GPU.
WORKERS_PER_GPU = int(_RAW["workers_per_gpu"])
#: assumed cluster/GB10 throughput factor until the first timed job on the cluster
CLUSTER_FACTOR = float(_RAW["cluster_factor"])
#: fractions of the default cost `prod.live_check --budget-fraction` accepts
BUDGET_FRACTIONS = [float(x) for x in _RAW["budget_fractions"]]
VERSION = str(_RAW.get("version", "PP3"))


# ------------------------------------------------------------------ the effective config
def defaults():
    return {
        "caps": list(CAPS),
        "caps_extra": list(CAPS_EXTRA),
        "horizon": HORIZON,
        "forced_horizon": FORCED_HORIZON,
        "forced_budgets": list(FORCED_BUDGETS),
        "think_tag_is_stop": THINK_TAG_IS_STOP,
        "forced_block": json.loads(json.dumps(FORCED_BLOCK)),
        "batch_width": BATCH_WIDTH,
        "batch_cap": BATCH_CAP,
        # per (model, k) overrides of `batch_width`; keys are ints once loaded.
        "batch_width_overrides": {m: dict(d) for m, d in BATCH_WIDTH_OVERRIDES.items()},
        "split_seed": SPLIT_SEED,
        "n_cal": N_CAL,
        "datasets": dict(DATASETS),
        "depths": {k: list(v) for k, v in DEPTHS.items()},
        "depths_extra": json.loads(json.dumps(DEPTHS_EXTRA)),
        "bbh_subtasks": list(BBH_SUBTASKS),
        "mem_util": MEM_UTIL,
        "prompt_allowance": PROMPT_ALLOWANCE,
        "placement_horizon": PLACEMENT_HORIZON,
        "workers_per_gpu": WORKERS_PER_GPU,
        "cluster_factor": CLUSTER_FACTOR,
        "budget_fractions": list(BUDGET_FRACTIONS),
        "version": VERSION,
    }


def _ints(s):
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def _floats(s):
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def _strs(s):
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def add_args(p):
    """The SAME override flags on prod.generate, prod.manifest and prod.launcher."""
    g = p.add_argument_group("config (prod/config.py; every default is overridable here)")
    g.add_argument("--caps", default=None, help="e.g. 0,16,64,512,4096")
    g.add_argument("--horizon", type=int, default=None)
    g.add_argument("--forced-horizon", dest="forced_horizon", type=int, default=None)
    g.add_argument("--forced-budgets", dest="forced_budgets", default=None)
    g.add_argument("--think-tag-is-stop", dest="think_tag_is_stop", action="store_true",
                   default=None,
                   help="the closing think tag ends a chat-template Thinking chain (the default, "
                        "and what every grid was generated under)")
    g.add_argument("--no-think-tag-is-stop", dest="think_tag_is_stop", action="store_false",
                   help="the closing think tag is an ordinary token: the chain runs on to the "
                        "task's stop strings or the eos token and the own answer is read through "
                        "the tag. Every chat-template Thinking natural-stop grid must be "
                        "REGENERATED to be comparable under this")
    g.add_argument("--batch-width", dest="batch_width", type=int, default=None,
                   help="0 = adaptive KV-ceiling width (gates only); default %d" % BATCH_WIDTH)
    g.add_argument("--batch-cap", dest="batch_cap", type=int, default=None)
    g.add_argument("--batch-width-overrides", dest="batch_width_overrides", default=None,
                   help="model:k=width,k=width;model2:k=width  merges into the config.yaml table, "
                        "e.g. huginn_0125:16=8,32=4")
    g.add_argument("--split-seed", dest="split_seed", type=int, default=None)
    g.add_argument("--n-cal", dest="n_cal", type=int, default=None)
    g.add_argument("--datasets", default=None,
                   help="task list, or task=N pairs, e.g. gsm8k=200,bbh")
    g.add_argument("--depths", default=None,
                   help="model:k,k;model:k,k  e.g. huginn_0125:1,4;ouro_1_4b_base:1,2,3,4")
    g.add_argument("--forced-models", dest="forced_models", default=None)
    g.add_argument("--forced-tasks", dest="forced_tasks", default=None)
    g.add_argument("--forced-n", dest="forced_n", default=None,
                   help="'full', an int, or task=N pairs")
    g.add_argument("--forced-ks", dest="forced_ks", default=None)
    g.add_argument("--no-forced-block", dest="no_forced_block", action="store_true")
    g.add_argument("--mem-util", dest="mem_util", type=float, default=None)
    g.add_argument("--prompt-allowance", dest="prompt_allowance", type=int, default=None)
    g.add_argument("--placement-horizon", dest="placement_horizon", type=int, default=None,
                   help="tokens per row the launcher's memory estimate assumes (default: the horizon)")
    g.add_argument("--cluster-factor", dest="cluster_factor", type=float, default=None)
    g.add_argument("--budget-fractions", dest="budget_fractions", default=None,
                   help="e.g. 0.25,0.5,1.0")
    g.add_argument("--config-json", dest="config_json", default=None,
                   help="a JSON file whose keys override the defaults (applied before the flags)")
    return p


def _deep_update(dst, src):
    """Merge `src` into `dst` key by key, recursing into nested dicts, so a partial override such as
    {"forced_block": {"enabled": true}} keeps the other forced_block keys from config.yaml."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def from_args(a):
    """The effective config: defaults (from config.yaml), then --config-json, then the flags."""
    cfg = defaults()
    if getattr(a, "config_json", None):
        with open(a.config_json, encoding="utf-8") as f:
            _deep_update(cfg, json.load(f))
    v = lambda n: getattr(a, n, None)                                        # noqa: E731
    if v("caps") is not None:
        cfg["caps"] = sorted(set(_ints(a.caps)))
    if v("horizon") is not None:
        cfg["horizon"] = int(a.horizon)
    if v("forced_horizon") is not None:
        cfg["forced_horizon"] = int(a.forced_horizon)
    if v("forced_budgets") is not None:
        cfg["forced_budgets"] = sorted(set(_ints(a.forced_budgets)))
    if v("think_tag_is_stop") is not None:
        cfg["think_tag_is_stop"] = bool(a.think_tag_is_stop)
    if v("batch_width") is not None:
        cfg["batch_width"] = int(a.batch_width)
    if v("batch_cap") is not None:
        cfg["batch_cap"] = int(a.batch_cap)
    if v("batch_width_overrides") is not None:
        for part in str(a.batch_width_overrides).split(";"):
            if not part.strip():
                continue
            m, kvs = part.split(":", 1)
            d = cfg["batch_width_overrides"].setdefault(m.strip(), {})
            for tok in kvs.split(","):
                if not tok.strip():
                    continue
                k, w = tok.split("=", 1)
                d[int(k)] = int(w)
    if v("split_seed") is not None:
        cfg["split_seed"] = int(a.split_seed)
    if v("n_cal") is not None:
        cfg["n_cal"] = int(a.n_cal)
    if v("datasets") is not None:
        ds = {}
        for tok in _strs(a.datasets):
            if "=" in tok:
                k, n = tok.split("=", 1)
                ds[k.strip()] = int(n)
            else:
                ds[tok] = DATASETS.get(tok)
        cfg["datasets"] = ds
    if v("depths") is not None:
        for part in str(a.depths).split(";"):
            if not part.strip():
                continue
            m, ks = part.split(":", 1)
            cfg["depths"][m.strip()] = _ints(ks)
    fb = cfg["forced_block"]
    if v("forced_models") is not None:
        fb["models"] = _strs(a.forced_models)
    if v("forced_tasks") is not None:
        fb["tasks"] = _strs(a.forced_tasks)
    if v("forced_ks") is not None:
        fb["ks"] = _ints(a.forced_ks)
    if v("forced_n") is not None:
        s = str(a.forced_n)
        if s == "full":
            fb["n"] = None
        elif "=" in s:
            fb["n"] = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in _strs(s)}
        else:
            fb["n"] = int(s)
    if getattr(a, "no_forced_block", False):
        fb["enabled"] = False
    if v("mem_util") is not None:
        cfg["mem_util"] = float(a.mem_util)
    if v("prompt_allowance") is not None:
        cfg["prompt_allowance"] = int(a.prompt_allowance)
    if v("placement_horizon") is not None:
        cfg["placement_horizon"] = int(a.placement_horizon)
    if v("cluster_factor") is not None:
        cfg["cluster_factor"] = float(a.cluster_factor)
    if v("budget_fractions") is not None:
        cfg["budget_fractions"] = _floats(a.budget_fractions)
    return cfg


def digest(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def header_row(cfg, tag, extra=None):
    """The first line of every cells file."""
    import time
    row = {"_header": True, "version": cfg.get("version", "PP3"), "tag": tag,
           "config": cfg, "config_sha256": digest(cfg), "written_at": time.strftime("%FT%T")}
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------ depth set of record
def depths_for(model, task, cfg=None):
    cfg = cfg or defaults()
    ks = list(cfg["depths"].get(model, []))
    ks += list((cfg.get("depths_extra") or {}).get(model, {}).get(task, []))
    return tuple(sorted(set(ks)))


# ------------------------------------------------------------------ batch width of record: the
# pinned width, replaced per (model, k) where the config.yaml table (or --batch-width-overrides)
# names one.
def batch_width_for(model, k, cfg=None):
    cfg = cfg or defaults()
    ov = (cfg.get("batch_width_overrides") or {}).get(model) or {}
    ov = {int(kk): int(vv) for kk, vv in ov.items()}
    return int(ov.get(int(k), cfg["batch_width"]))


# ------------------------------------------------------------------ memory estimate
def _shapes_file():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks", "data",
                        "model_revisions.json")


def shapes_for(model):
    """The installed shapes if install_models.py has written them, else the static table."""
    p = _shapes_file()
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            s = (d.get("models") or {}).get(model, {}).get("shapes")
            if s:
                out = dict(MODEL_SHAPES.get(model, {}))
                out.update(s)
                return out
        except Exception:                                     # noqa: BLE001
            pass
    return dict(MODEL_SHAPES[model])


def revision_for(model):
    """The pinned Hub revision of a checkpoint, or None before `prod.install_models` has run.

    Every adapter passes this to `from_pretrained`, so a run loads the revision that was installed
    and not whatever `main` points at today.
    """
    p = _shapes_file()
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return (d.get("models") or {}).get(model, {}).get("revision")
    except Exception:                                         # noqa: BLE001
        return None


def cache_entries(model, k):
    """Cached (K, V) layer entries per token at depth k."""
    s = shapes_for(model)
    if s.get("entries") == "ouro":
        return int(s["layers"]) * int(k)
    return int(s["prelude"]) + int(s["core"]) * int(k) + int(s["coda"])


def kv_bytes_per_token(model, k):
    """2 (K and V) x kv_heads x head_dim x 2 bytes (bf16) x cached layer entries."""
    s = shapes_for(model)
    return int(2 * int(s["kv_heads"]) * int(s["head_dim"]) * int(s.get("dtype_bytes", 2))
               * cache_entries(model, k))


def weight_bytes(model):
    s = shapes_for(model)
    return int(int(s["params"]) * int(s.get("dtype_bytes", 2)))


def job_bytes(model, k, batch_width, horizon, prompt_allowance=None, cfg=None):
    """The per-job memory estimate:

        weights (bf16 bytes from the parameter count)
      + batch_width x horizon x KV bytes per token

    `horizon` is widened by `prompt_allowance` because the cache holds prompt tokens too.
    """
    cfg = cfg or defaults()
    pa = cfg["prompt_allowance"] if prompt_allowance is None else int(prompt_allowance)
    width = int(batch_width) or int(cfg["batch_cap"])
    hz = int(cfg.get("placement_horizon") or horizon)
    return int(weight_bytes(model)
               + width * (hz + pa) * kv_bytes_per_token(model, k))


def main(argv=None):
    p = argparse.ArgumentParser(prog="prod.config")
    p.add_argument("--print", dest="show", action="store_true")
    p.add_argument("--sha", action="store_true")
    add_args(p)
    a = p.parse_args(argv)
    cfg = from_args(a)
    if a.sha:
        print(digest(cfg))
    else:
        print(json.dumps(cfg, indent=2, sort_keys=True))
    return cfg


if __name__ == "__main__":
    main()
