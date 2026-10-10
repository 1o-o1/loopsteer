"""Extractor for cluster/pilot/steer_gen_raven.py (McLeish r32 generation placement, grid A).

The rows are the script's own results_rows (its row map at K = 32: loop 3 of run 1 is loop K-1 = 31, loop 4 is K), the
90 percent intervals recomputed from the per-item cells as in extract/gen.py (steer_gen's load_cells, outcome and
boot_cluster). Item sets: extract/gen.py (scored problems of items.json against the cells of every row folder r00-r21).
"""
import importlib.util
import os
import sys

from . import common, gen


def _mod(src_py):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(src_py))))    # the loopedcot folder
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("steer_gen_raven", src_py)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def extract(job_dir, src_py):
    M = _mod(src_py)
    an = common.load_json(os.path.join(job_dir, "analysis.json"))
    r90, _ = gen.recompute(job_dir, M.S.__file__, True)
    return M.results_rows(an, r90=r90)


def item_sets(job_dir):
    return gen.item_sets(job_dir)
