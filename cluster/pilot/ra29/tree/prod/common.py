"""PP2 production package: paths, environment, checkpoint helpers, hashing.

Nothing in this package imports from outside it. Every ported line names its source file in a
comment so the cluster install is one self-contained directory.

Environment variables honoured (all optional, all also `run.sh` arguments):
  PROD_ROOT        package root (default: the parent of this file's directory)
  PROD_ART         artifacts directory (default: $PROD_ROOT/artifacts)
  PROD_MEM_FRACTION   torch.cuda.set_per_process_memory_fraction (default 0.85, Spark rule)
  PROD_BATCH_CAP   hard cap on the decode batch (default 32)
  PROD_RESERVE_GB  headroom left outside the measured KV ceiling (default 2.0)
  HF_HOME, HF_HUB_OFFLINE  passed through untouched
"""
import hashlib
import json
import os
import subprocess
import sys
import time

# Caps, horizon, seed, batch width and the forced block live in ONE place, prod/config.py. The
# names below are kept as re-exports so every gate still imports what it always did; they are NOT
# a second copy -- change prod/config.py.
from . import config as _cfg

ROOT = os.environ.get("PROD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("PROD_ART") or os.path.join(ROOT, "artifacts")
LOGS = os.environ.get("PROD_LOGS") or os.path.join(ROOT, "logs")
DATA = os.path.join(PKG, "tasks", "data")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MEM_FRACTION = float(os.environ.get("PROD_MEM_FRACTION", str(_cfg.MEM_UTIL)))
BATCH_CAP = int(os.environ.get("PROD_BATCH_CAP", str(_cfg.BATCH_CAP)))
RESERVE_GB = float(os.environ.get("PROD_RESERVE_GB", "2.0"))

CAPS_STANDARD = list(_cfg.CAPS)
CAPS_EXTRA = list(_cfg.CAPS_EXTRA)
CAPS_ALL = sorted(set(CAPS_STANDARD) | set(CAPS_EXTRA))
HORIZON = _cfg.HORIZON
FORCED_HORIZON = _cfg.FORCED_HORIZON
FORCED_BUDGETS = list(_cfg.FORCED_BUDGETS)
SPLIT_SEED = _cfg.SPLIT_SEED
N_CAL = _cfg.N_CAL

# Production batch width, PINNED: every production run decodes at width 16 and every row records
# the width it was produced at, because per-problem labels carry 6-10% bf16 batch-width noise
# (measured) and a number is only comparable to another number taken at the same width.
# `--batch-width=0` restores the adaptive width and is for the gates only.
BATCH_WIDTH = int(os.environ.get("PROD_BATCH_WIDTH", str(_cfg.BATCH_WIDTH)))

# The forced block runs at FULL N by default and is a separate, configurable protocol block
# (prod/config.py FORCED_BLOCK), switched on in config.yaml (`forced_block.enabled`) and priced
# separately. `FORCED_N` survives only as an explicit override table of forced-continuation counts:
# empty means "full N", which is the default.
FORCED_N = {}

for _d in (ART, LOGS):
    os.makedirs(_d, exist_ok=True)


# ------------------------------------------------------------------ hashing
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def package_hashes():
    """sha256 of every .py and .sh in the package, for the review pack and checks.json."""
    out = {}
    for base, _dirs, files in os.walk(PKG):
        for fn in sorted(files):
            if fn.endswith((".py", ".sh")):
                p = os.path.join(base, fn)
                out[os.path.relpath(p, PKG).replace("\\", "/")] = sha256_file(p)
    return out


# ------------------------------------------------------------------ gpu
def nvsmi():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:                                    # noqa: BLE001
        return "nvidia-smi failed: %r" % (e,)


def gpu_procs():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=60).stdout.strip().replace("\n", " | ")
    except Exception as e:                                    # noqa: BLE001
        return "nvidia-smi failed: %r" % (e,)


def env_report():
    d = {"python": sys.version.split()[0], "argv": list(sys.argv),
         "hf_home": os.environ.get("HF_HOME"), "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
         "mem_fraction": MEM_FRACTION, "batch_cap": BATCH_CAP, "time": time.strftime("%FT%T")}
    try:
        import torch
        import transformers
        d.update({"torch": torch.__version__, "transformers": transformers.__version__,
                  "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())})
        if torch.cuda.is_available():
            d["device"] = torch.cuda.get_device_name(0)
            d["nvidia_smi"] = nvsmi()
    except Exception as e:                                    # noqa: BLE001
        d["torch_import_error"] = repr(e)
    return d


# ------------------------------------------------------------------ jsonl checkpoints
# s28_common.load_ckpt / Appender, verbatim (they are S9a's).
def load_ckpt(path, keyfn):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:                             # noqa: BLE001
                    continue                                  # truncated last line after a kill
                if r.get("_header"):
                    continue     # the effective config row, not a cell
                done[keyfn(r)] = r
    return done


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except Exception:                             # noqa: BLE001
                    continue
                if isinstance(r, dict) and r.get("_header"):
                    continue     # the effective config row, not a cell
                out.append(r)
    return out


class Appender(object):
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def write(self, row):
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        try:
            self.f.close()
        except Exception:                                     # noqa: BLE001
            pass


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def load_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_header(path):
    """The header row of a cells file: the effective config that produced it."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                                 # noqa: BLE001
                return None
            return r if isinstance(r, dict) and r.get("_header") else None
    return None


def count_cells(path):
    """Rows in a cells file, header excluded."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('{"_header"'):
                continue
            n += 1
    return n
