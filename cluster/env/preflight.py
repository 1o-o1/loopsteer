"""Preflight check of one family venv (run by cluster/env/preflight.sh inside the g0_xllm job, one call per family).

Records the interpreter, torch (it must come from the node image, not from the venv), the CUDA device, the pinned
package versions of cluster/env/requirements_cluster_<family>.txt, imports the packages and the bundle scripts the
family runs, and loads one model configuration (no weights): Ouro-1.4B (ouro), Huginn (raven), Parcae 370m (parcae),
the IFM no-injection checkpoint (ifm; xllm and its CUDA extension must import). Writes one JSON file and exits 1 when
any check fails.

    $VENV_<F>/bin/python cluster/env/preflight.py --family ouro|raven|parcae|ifm --out <file.json>
"""
import argparse
import importlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LC = os.path.dirname(os.path.dirname(HERE))                  # .../loopedcot
PILOT = os.path.join(LC, "cluster", "pilot")
PYMIN = {"ouro": (3, 9), "raven": (3, 10), "parcae": (3, 11), "ifm": (3, 10)}
PACKAGES = {"ouro": ["transformers", "tokenizers", "safetensors", "huggingface_hub", "accelerate", "peft", "numpy",
                     "pandas", "scipy", "yaml"],
            "raven": ["transformers", "tokenizers", "safetensors", "huggingface_hub", "accelerate", "peft", "einops",
                      "numpy", "pandas", "scipy", "yaml"],
            "parcae": ["transformers", "tokenizers", "safetensors", "huggingface_hub", "einops", "sentencepiece",
                       "numpy", "parcae_lm.models.parcae.parcae", "parcae_lm.models.parcae.config"],
            "ifm": ["transformers", "tokenizers", "safetensors", "huggingface_hub", "einops", "fla", "tiktoken",
                    "sentencepiece", "typeguard", "numpy", "pandas"]}
# bundle scripts per family (cluster/gen_jobs.py venv_of); imported, not run
SCRIPTS = {"ouro": ["prod.steer_gen", "k16_path_ouro", "k17_freeze", "k17_varbench", "steer_gen_math"],
           "raven": ["k17_public", "k17_freeze_raven", "steer_gen_raven"],
           "parcae": ["k17_parcae"],
           "ifm": ["k10_ifm", "k15_full", "k15_pairs", "k16_path_ifm", "k17_freeze_ifm", "k17_varbench"]}
CONFIG_OF = {"ouro": "ouro_1_4b_base", "raven": "huginn_0125", "parcae": "parcae_370m", "ifm": "ifm_p1_ouro"}


def location(key):
    w = json.load(open(os.path.join(LC, "cluster", "weights.json"), encoding="utf-8"))
    for c in w["checkpoints"]:
        if c["key"] == key:
            return (c["location"].replace("$HF_HOME", os.environ.get("HF_HOME", ""))
                    .replace("$WEIGHTS_ROOT", os.environ.get("IFM_CKPT_ROOT") or os.environ.get("WEIGHTS_ROOT", "")))
    raise KeyError(key)


def pins(family):
    """name -> version of the pinned lines that apply to this interpreter (pip's vendored markers)."""
    from pip._vendor.packaging.requirements import Requirement
    out = {}
    for ln in open(os.path.join(HERE, "requirements_cluster_%s.txt" % family), encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        r = Requirement(ln)
        if r.marker is not None and not r.marker.evaluate():
            continue
        spec = [s.version for s in r.specifier if s.operator == "=="]
        out[r.name] = spec[0] if spec else None
    return out


def check_config(family, rec):
    d = location(CONFIG_OF[family])
    rec["config_dir"] = d
    if family in ("ouro", "raven"):
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
        ref = cfg.auto_map["AutoModelForCausalLM"]
        cls = get_class_from_dynamic_module(ref, d)          # imports the remote modeling file (no weights)
        rec["config"] = {"model_type": cfg.model_type, "modeling_class": cls.__name__}
    elif family == "parcae":
        from parcae_lm.models.parcae.config import ParcaeConfig
        cfgd = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        assert cfgd.pop("_class_name") == "ParcaeConfig"
        cfgd.pop("rope_settings", None)                       # as k17_parcae.load_model
        cfg = ParcaeConfig(**cfgd)
        rec["config"] = {"class": type(cfg).__name__, "fields": len(cfgd)}
    else:
        from xllm.config import ModelConf
        raw = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        cfg = ModelConf.from_dict(dict(raw["model"]))
        tok = os.path.realpath(os.path.join(d, raw["tokenizer"]["path"]))
        assert os.path.exists(tok), "tokenizer path missing: %s" % tok
        rec["config"] = {"arch": cfg.arch, "tokenizer": os.path.basename(tok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(PYMIN))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = a.family
    rec = {"family": f, "t": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
           "executable": sys.executable, "prefix": sys.prefix, "failures": []}

    def step(name, fn):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            rec["failures"].append("%s: %s" % (name, repr(ex)[:400]))
            rec.setdefault("tracebacks", {})[name] = traceback.format_exc()[-2000:]

    if sys.version_info[:2] < PYMIN[f]:
        rec["failures"].append("python %s < %d.%d" % ((rec["python"],) + PYMIN[f]))

    def torch_check():
        import torch
        rec["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "file": torch.__file__,
                        "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()}
        assert not os.path.realpath(torch.__file__).startswith(os.path.realpath(sys.prefix) + os.sep), \
            "torch is installed inside the venv; it must come from the node image"
        assert torch.version.cuda, "torch is not a CUDA build"
        assert torch.cuda.is_available(), "no GPU visible"
        p = torch.cuda.get_device_properties(0)
        rec["torch"]["gpu"] = {"name": p.name, "memory_mib": int(p.total_memory // 2**20),
                               "capability": "%d.%d" % (p.major, p.minor), "count": torch.cuda.device_count()}
    step("torch", torch_check)

    def pins_check():
        from importlib import metadata
        got, bad = {}, []
        for name, ver in pins(f).items():
            try:
                v = metadata.version(name)
            except metadata.PackageNotFoundError:
                v = None
            got[name] = v
            if v is None or (ver is not None and v != ver):
                bad.append("%s want %s got %s" % (name, ver, v))
        rec["packages"] = got
        assert not bad, "; ".join(bad)
    step("pins", pins_check)

    if f == "ifm":
        xr = os.environ.get("XLLM_ROOT", "")
        sys.path.insert(0, xr)
        sys.path.append(os.path.join(PILOT, "stubs"))        # as k10_ifm.load_ifm: FlashAttention 2 stand-in

        def xllm_check():
            import xllm               # noqa: F401
            import xllm_extension     # noqa: F401  (built by the g0_xllm job)
            rec["xllm"] = {"file": xllm.__file__, "extension": xllm_extension.__file__}
            import subprocess
            rec["xllm"]["commit"] = subprocess.check_output(["git", "-C", xr, "rev-parse", "HEAD"]).decode().strip()
        step("xllm", xllm_check)

    for m in PACKAGES[f]:
        step("import " + m, lambda m=m: importlib.import_module(m))
    sys.path[:0] = [os.path.join(PILOT, "tree"), PILOT]   # the bundle's prod tree first: prod.steer_gen and its modules
    for m in SCRIPTS[f]:
        step("import " + m, lambda m=m: importlib.import_module(m))
    step("config " + CONFIG_OF[f], lambda: check_config(f, rec))
    rec["ok"] = not rec["failures"]
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"family": f, "ok": rec["ok"], "failures": rec["failures"]}))
    sys.exit(0 if rec["ok"] else 1)


if __name__ == "__main__":
    main()
