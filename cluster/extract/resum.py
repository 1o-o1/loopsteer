"""Re-run a pilot script's own summarize step on a copy of a job's records, with its percentile bootstrap read
at the 90 percent quantiles instead of the 95 percent ones.

Why this is faithful: every pilot script draws its cluster-bootstrap resamples with a fixed seed and then reads
two quantiles of the resampled statistic (torch.quantile, or numpy quantile/percentile). Only the read changes:
(0.025, 0.975) becomes (0.05, 0.95). The resamples, the seed, the cluster unit, the item filter and the statistic
stay the script's own. Calls that already ask for other quantiles (for example the scripts' TOST, which reads
(0.05, 0.95)) are untouched.

The records are copied to a scratch directory first, so the job's own files are never rewritten; the copy is
removed when the process exits.
"""
import atexit
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile

Q95 = (0.025, 0.975)
Q90 = (0.05, 0.95)


def load_module(path, name=None):
    """Import a pilot script by path without running its main()."""
    sys.dont_write_bytecode = True
    name = name or "pilot_" + os.path.splitext(os.path.basename(path))[0]
    src_dir = os.path.dirname(os.path.abspath(path))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)          # pilot scripts import their siblings (k10_ifm, k15_smoke, ...)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is95(q):
    try:
        vals = [float(x) for x in (q.tolist() if hasattr(q, "tolist") else q)]
    except TypeError:
        return False
    return len(vals) == 2 and abs(vals[0] - Q95[0]) < 1e-12 and abs(vals[1] - Q95[1]) < 1e-12


def _is95_pct(q):
    try:
        vals = [float(x) for x in (q.tolist() if hasattr(q, "tolist") else q)]
    except TypeError:
        return False
    return len(vals) == 2 and abs(vals[0] - 2.5) < 1e-9 and abs(vals[1] - 97.5) < 1e-9


@contextlib.contextmanager
def ninety(counter=None):
    """Inside this block every two-sided 95 percent quantile read becomes the 90 percent read."""
    import numpy as np
    import torch
    t_q, n_q, n_p = torch.quantile, np.quantile, np.percentile
    hits = counter if counter is not None else {}
    hits.setdefault("torch", 0), hits.setdefault("numpy", 0)

    def tq(inp, q, *a, **k):
        if torch.is_tensor(q) and _is95(q):
            hits["torch"] += 1
            q = torch.tensor(list(Q90), dtype=q.dtype, device=q.device)
        elif isinstance(q, (list, tuple)) and _is95(q):
            hits["torch"] += 1
            q = type(q)(Q90)
        return t_q(inp, q, *a, **k)

    def nq(a, q, *args, **k):
        if _is95(q):
            hits["numpy"] += 1
            q = list(Q90)
        return n_q(a, q, *args, **k)

    def npct(a, q, *args, **k):
        if _is95_pct(q):
            hits["numpy"] += 1
            q = [5.0, 95.0]
        return n_p(a, q, *args, **k)

    torch.quantile, np.quantile, np.percentile = tq, nq, npct
    try:
        yield hits
    finally:
        torch.quantile, np.quantile, np.percentile = t_q, n_q, n_p


def copy_records(src_dir, patterns=None, skip_dirs=()):
    """Copy a job's record files into a fresh scratch directory and return its path."""
    dst = tempfile.mkdtemp(prefix="resum_")
    atexit.register(shutil.rmtree, dst, True)            # scratch copies never outlive the process
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        rel = os.path.relpath(root, src_dir)
        for f in files:
            if patterns is not None and not any(_match(f, p) for p in patterns):
                continue
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            shutil.copy2(os.path.join(root, f), os.path.join(dst, rel, f))
    return dst


def _match(name, pat):
    import fnmatch
    return fnmatch.fnmatch(name, pat)


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield
