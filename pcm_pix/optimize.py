"""
Facade module for optimization utilities.

We keep `optimize_pso.py` as the implementation (historical name), but expose a stable
import path: `from pcm_pix.optimize import ...`.

По-русски: это "витрина" функций оптимизации, чтобы в ноутбуке были короткие импорты.
"""

from __future__ import annotations

# Re-export everything needed by notebooks/scripts.
from .optimize_pso import (  # noqa: F401
    PSOResult,
    f_vec,
    load_solution,
    make_init_ar_from_pos,
    make_linear_constraint_nm,
    make_targets,
    run_de,
    run_de_full,
    run_hybrid_pso_de,
    run_pso,
    run_pso_until,
    run_pso_until as run_pso_until_cost,  # legacy alias if needed
    save_solution,
)

