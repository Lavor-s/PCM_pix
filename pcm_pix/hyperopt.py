from __future__ import annotations
# pyright: reportMissingImports=false

"""
hyperopt.py — подбор гиперпараметров для оптимизаторов.

Здесь реализованы два независимых "hyperopt":
- PSO-hyperopt: подбираем (c1, c2, w) через быстрые прогоны PSO (random → refine)
- DE-hyperopt:  подбираем (mutation, recombination, popsize) через быстрые прогоны DE

Результаты сохраняются в `run.results` как CSV, чтобы:
- можно было переиспользовать найденные параметры без пересчёта
- можно было анализировать таблицы отдельно (pandas/Excel)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PSOHyperParams:
    c1: float
    c2: float
    w: float
    source: str | None = None


def load_best_pso_hyperparams(results_dir: str | Path) -> PSOHyperParams | None:
    """
    Загружает лучшие гиперпараметры PSO из ранее сохранённых таблиц hyperopt.

    Expected files:
    - pso_refine.csv (preferred)
    - pso_random_search.csv (fallback)
    """
    results_dir = Path(results_dir)
    refine = results_dir / "pso_refine.csv"
    rs = results_dir / "pso_random_search.csv"

    path = refine if refine.exists() else rs if rs.exists() else None
    if path is None:
        return None

    df = pd.read_csv(path).sort_values("cost_median").reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return PSOHyperParams(
        c1=float(best["c1"]),
        c2=float(best["c2"]),
        w=float(best["w"]),
        source=path.name,
    )


def apply_pso_hyperparams(cfg: dict[str, Any], hp: PSOHyperParams) -> dict[str, Any]:
    """Применяет найденные гиперпараметры PSO в словарь CFG."""
    cfg["pso_c1"] = float(hp.c1)
    cfg["pso_c2"] = float(hp.c2)
    cfg["pso_w"] = float(hp.w)
    return cfg


def run_pso_hyperopt(
    sur0,
    sur1,
    cfg: dict[str, Any],
    run,
) -> pd.DataFrame:
    """
    Подбор гиперпараметров PSO (двухэтапный: random search → refine).

    Сохраняет результаты в run.results:
    - pso_random_search.csv
    - pso_refine.csv

    Сделано максимально просто, чтобы запускать из Jupyter.
    """
    from itertools import product

    # import here to avoid circular import at module load time
    from .optimize import run_pso_until

    rng = np.random.default_rng(int(cfg.get("hyperopt_seed", 42)))

    # --- stage A: random search ---
    n_trials = int(cfg.get("hyperopt_pso_trials", 80))
    repeats = int(cfg.get("hyperopt_pso_repeats", 2))

    fast_particles = int(cfg.get("hyperopt_pso_n_particles", 250))
    fast_iters = int(cfg.get("hyperopt_pso_iters", 60))

    # constrain restarts/threshold in hyperopt mode (we want speed + comparability)
    base = dict(cfg)
    base["pso_threshold"] = float(cfg.get("hyperopt_pso_threshold", 999999))
    base["pso_max_restarts"] = int(cfg.get("hyperopt_pso_max_restarts", 0))

    def cost_once(c1: float, c2: float, w: float, seed: int) -> float:
        local = dict(base)
        local["pso_c1"] = float(c1)
        local["pso_c2"] = float(c2)
        local["pso_w"] = float(w)
        local["pso_n_particles"] = fast_particles
        local["pso_iters"] = fast_iters

        np.random.seed(seed)
        # В hyperopt-цикле отключаем запись best_pos/best_cost, чтобы не перетирать финальные артефакты.
        best = run_pso_until(sur0, sur1, local, run, save_artifacts=False)
        return float(best.cost)

    rows: list[dict[str, Any]] = []
    for t in range(n_trials):
        c1 = float(rng.uniform(0.1, 1.5))
        c2 = float(rng.uniform(0.1, 1.5))
        w = float(rng.uniform(0.3, 0.95))

        costs = [cost_once(c1, c2, w, seed=1000 * t + r) for r in range(repeats)]
        rows.append(
            {
                "c1": c1,
                "c2": c2,
                "w": w,
                "cost_median": float(np.median(costs)),
                "cost_mean": float(np.mean(costs)),
                "cost_min": float(np.min(costs)),
            }
        )

        if hasattr(run, "logger") and (t + 1) % 10 == 0:
            run.logger.info("hyperopt PSO random %s/%s done", t + 1, n_trials)

    df_rs = pd.DataFrame(rows).sort_values("cost_median").reset_index(drop=True)
    (Path(run.results) / "pso_random_search.csv").write_text(df_rs.to_csv(index=False), encoding="utf-8")

    # --- stage B: refine around top-k ---
    top_k = int(cfg.get("hyperopt_pso_refine_topk", 3))
    repeats2 = int(cfg.get("hyperopt_pso_refine_repeats", 3))

    dc1 = float(cfg.get("hyperopt_refine_dc1", 0.2))
    dc2 = float(cfg.get("hyperopt_refine_dc2", 0.2))
    dw = float(cfg.get("hyperopt_refine_dw", 0.1))

    def refine_grid_around(row: pd.Series):
        c1, c2, w = float(row.c1), float(row.c2), float(row.w)
        grid_c1 = [c1 - dc1, c1 - dc1 / 2, c1, c1 + dc1 / 2, c1 + dc1]
        grid_c2 = [c2 - dc2, c2 - dc2 / 2, c2, c2 + dc2 / 2, c2 + dc2]
        grid_w = [w - dw, w - dw / 2, w, w + dw / 2, w + dw]

        grid_c1 = [float(np.clip(x, 0.05, 2.5)) for x in grid_c1]
        grid_c2 = [float(np.clip(x, 0.05, 2.5)) for x in grid_c2]
        grid_w = [float(np.clip(x, 0.10, 0.99)) for x in grid_w]
        return grid_c1, grid_c2, grid_w

    rows2: list[dict[str, Any]] = []
    top = df_rs.head(top_k)
    for i in range(len(top)):
        grid_c1, grid_c2, grid_w = refine_grid_around(top.iloc[i])

        for c1, c2, w in product(grid_c1, grid_c2, grid_w):
            costs = [cost_once(c1, c2, w, seed=900000 + 1000 * i + r) for r in range(repeats2)]
            rows2.append(
                {
                    "base_rank": i,
                    "c1": float(c1),
                    "c2": float(c2),
                    "w": float(w),
                    "cost_median": float(np.median(costs)),
                    "cost_mean": float(np.mean(costs)),
                    "cost_min": float(np.min(costs)),
                }
            )

        if hasattr(run, "logger"):
            run.logger.info("hyperopt PSO refine around rank=%s done", i)

    df_ref = pd.DataFrame(rows2).sort_values("cost_median").reset_index(drop=True)
    (Path(run.results) / "pso_refine.csv").write_text(df_ref.to_csv(index=False), encoding="utf-8")

    return df_ref


# ----------------------------
# Differential Evolution (DE)
# ----------------------------


@dataclass(frozen=True)
class DEHyperParams:
    """
    Гиперпараметры differential_evolution, которые мы подбираем.
    """

    mutation: float
    recombination: float
    popsize: int
    source: str | None = None


def load_best_de_hyperparams(results_dir: str | Path) -> DEHyperParams | None:
    """
    Загружает лучшие гиперпараметры DE из ранее сохранённых таблиц hyperopt.

    Expected files:
    - de_refine.csv (preferred)
    - de_random_search.csv (fallback)
    """
    results_dir = Path(results_dir)
    refine = results_dir / "de_refine.csv"
    rs = results_dir / "de_random_search.csv"

    path = refine if refine.exists() else rs if rs.exists() else None
    if path is None:
        return None

    df = pd.read_csv(path).sort_values("cost_median").reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return DEHyperParams(
        mutation=float(best["mutation"]),
        recombination=float(best["recombination"]),
        popsize=int(best["popsize"]),
        source=path.name,
    )


def apply_de_hyperparams(cfg: dict[str, Any], hp: DEHyperParams) -> dict[str, Any]:
    """Применяет найденные гиперпараметры DE в словарь CFG."""
    cfg["de_mutation"] = float(hp.mutation)
    cfg["de_recombination"] = float(hp.recombination)
    cfg["de_popsize"] = int(hp.popsize)
    return cfg


def run_de_hyperopt(
    sur0,
    sur1,
    cfg: dict[str, Any],
    run,
    pos0: np.ndarray,
) -> pd.DataFrame:
    """
    Подбор гиперпараметров Differential Evolution (двухэтапный: random search → refine).

    Что подбираем:
    - de_mutation
    - de_recombination
    - de_popsize

    Важно:
    - Подбор делается на "быстром" режиме (маленький maxiter, polish=False),
      иначе это будет очень долго.
    - Целевая функция ровно та же, что и в основной оптимизации (через f_de/f_vec).

    Сохраняет:
    - de_random_search.csv
    - de_refine.csv
    """
    from itertools import product
    from scipy.optimize import Bounds, differential_evolution

    # Чтобы цель совпадала 1-в-1 с основным пайплайном, используем те же помощники.
    from .optimize import _make_bounds_arrays, f_de

    rng = np.random.default_rng(int(cfg.get("hyperopt_de_seed", 42)))

    # --- быстрый режим для hyperopt (можно переопределить из CFG) ---
    fast_maxiter = int(cfg.get("hyperopt_de_maxiter", 50))
    fast_polish = bool(cfg.get("hyperopt_de_polish", False))
    updating = cfg.get("de_updating", "deferred")

    lower, upper = _make_bounds_arrays(cfg)

    # SciPy принимает Bounds; keep_feasible=True как в твоих ноутбуках
    bounds = Bounds(lower, upper, keep_feasible=True)

    # differential_evolution(vectorized=True) подаёт X в виде (dims, npop)
    cfg_base = dict(cfg)
    cfg_base["dims"] = int(len(lower))

    pos0 = np.array(pos0, dtype=float).ravel()

    def cost_once(mutation: float, recombination: float, popsize: int, seed: int) -> float:
        local = dict(cfg_base)
        local["de_mutation"] = float(mutation)
        local["de_recombination"] = float(recombination)
        local["de_popsize"] = int(popsize)

        def obj(X):
            return f_de(X, sur0, sur1, local)

        # seed влияет на рандом в SciPy (инициализация популяции и т.п.)
        np.random.seed(seed)
        res = differential_evolution(
            obj,
            bounds=bounds,
            x0=pos0,
            init="latinhypercube",
            mutation=float(mutation),
            recombination=float(recombination),
            maxiter=fast_maxiter,
            popsize=int(popsize),
            tol=float(local.get("de_tol", 1e-12)),
            atol=float(local.get("de_atol", 1e-12)),
            polish=fast_polish,
            updating=updating,
            vectorized=True,
            seed=seed,
        )
        return float(res.fun)

    # --- stage A: random search ---
    n_trials = int(cfg.get("hyperopt_de_trials", 60))
    repeats = int(cfg.get("hyperopt_de_repeats", 2))

    rows: list[dict[str, Any]] = []
    for t in range(n_trials):
        mutation = float(rng.uniform(0.2, 1.5))
        recombination = float(rng.uniform(0.05, 0.95))
        popsize = int(rng.integers(10, 60))

        costs = [cost_once(mutation, recombination, popsize, seed=2000 * t + r) for r in range(repeats)]
        rows.append(
            {
                "mutation": mutation,
                "recombination": recombination,
                "popsize": popsize,
                "cost_median": float(np.median(costs)),
                "cost_mean": float(np.mean(costs)),
                "cost_min": float(np.min(costs)),
            }
        )

        if hasattr(run, "logger") and (t + 1) % 10 == 0:
            run.logger.info("hyperopt DE random %s/%s done", t + 1, n_trials)

    df_rs = pd.DataFrame(rows).sort_values("cost_median").reset_index(drop=True)
    (Path(run.results) / "de_random_search.csv").write_text(df_rs.to_csv(index=False), encoding="utf-8")

    # --- stage B: refine around top-k ---
    top_k = int(cfg.get("hyperopt_de_refine_topk", 3))
    repeats2 = int(cfg.get("hyperopt_de_refine_repeats", 3))

    dmut = float(cfg.get("hyperopt_de_refine_dmut", 0.15))
    drec = float(cfg.get("hyperopt_de_refine_drec", 0.15))

    def refine_grid_around(row: pd.Series):
        m = float(row.mutation)
        r = float(row.recombination)
        p = int(row.popsize)

        grid_mut = [m - dmut, m - dmut / 2, m, m + dmut / 2, m + dmut]
        grid_rec = [r - drec, r - drec / 2, r, r + drec / 2, r + drec]
        grid_pop = sorted(set([max(5, p - 10), p, p + 10]))

        grid_mut = [float(np.clip(x, 0.05, 1.99)) for x in grid_mut]
        grid_rec = [float(np.clip(x, 0.01, 0.99)) for x in grid_rec]
        grid_pop = [int(np.clip(x, 5, 100)) for x in grid_pop]
        return grid_mut, grid_rec, grid_pop

    rows2: list[dict[str, Any]] = []
    top = df_rs.head(top_k)
    for i in range(len(top)):
        grid_mut, grid_rec, grid_pop = refine_grid_around(top.iloc[i])

        for mutation, recombination, popsize in product(grid_mut, grid_rec, grid_pop):
            costs = [cost_once(mutation, recombination, popsize, seed=910000 + 1000 * i + r) for r in range(repeats2)]
            rows2.append(
                {
                    "base_rank": i,
                    "mutation": float(mutation),
                    "recombination": float(recombination),
                    "popsize": int(popsize),
                    "cost_median": float(np.median(costs)),
                    "cost_mean": float(np.mean(costs)),
                    "cost_min": float(np.min(costs)),
                }
            )

        if hasattr(run, "logger"):
            run.logger.info("hyperopt DE refine around rank=%s done", i)

    df_ref = pd.DataFrame(rows2).sort_values("cost_median").reset_index(drop=True)
    (Path(run.results) / "de_refine.csv").write_text(df_ref.to_csv(index=False), encoding="utf-8")

    return df_ref

