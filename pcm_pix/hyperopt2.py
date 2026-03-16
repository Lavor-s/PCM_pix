from __future__ import annotations
# pyright: reportMissingImports=false

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PSO_SPECS = {
    "c1": (0.1, 1.5, 0.2),
    "c2": (0.1, 1.5, 0.2),
    "w": (0.3, 0.95, 0.1),
    "n_particles": (50, 5000, 200),#250
    "iters": (20, 500, 20), #60
}

DE_SPECS = {
    "mutation": (0.2, 1.5, 0.15),
    "recombination": (0.05, 0.95, 0.15),
    "popsize": (10, 60, 10),
    "init_N": (100, 5000, 100, 1000),
    "init_spread": (0.0, 0.20, 0.02),
 
    # Ниже фиксированные параметры (4-е значение) — можно легко "разфиксировать" позже.
    "tol": (1e-16, 1e-3, 1e-6, 1e-12),
    "atol": (1e-16, 1e-3, 1e-6, 1e-12),
    "maxiter": (5, 5000, 50, 100),
    "polish": (0, 1, 1, False),
    "use_linear_constraint": (0, 1, 1, True),
    "updating": (0, 0, 0, "deferred"),
    "init_mode": (0, 0, 0, "init_ar"),
}


Spec = tuple[Any, Any, Any] | tuple[Any, Any, Any, Any]
SpecMap = dict[str, Spec]


def _is_int_param(name: str, low: float, high: float, cfg: dict[str, Any], prefix: str) -> bool:
    key = f"{prefix}_{name}"
    if key in cfg:
        return isinstance(cfg[key], int) and not isinstance(cfg[key], bool)
    return float(low).is_integer() and float(high).is_integer()


def _sample_params(specs: SpecMap, rng: np.random.Generator, cfg: dict[str, Any], prefix: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, spec in specs.items():
        if len(spec) == 4:
            _, _, _, fixed = spec
            values[name] = fixed
            continue

        low, high, _ = spec
        if _is_int_param(name, low, high, cfg, prefix):
            values[name] = int(rng.integers(int(low), int(high) + 1))
        else:
            values[name] = float(rng.uniform(float(low), float(high)))
    return values


def _apply_params(cfg: dict[str, Any], prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    for name, value in values.items():
        cfg[f"{prefix}_{name}"] = value
    return cfg


def _refine_grid(row: pd.Series, specs: SpecMap, cfg: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    from itertools import product

    names = list(specs.keys())
    grids: list[list[Any]] = []

    for name in names:
        spec = specs[name]

        if len(spec) == 4:
            _, _, _, fixed = spec
            grids.append([fixed])
            continue

        low, high, delta = spec
        center = row[name]
        vals = [center - delta, center - delta / 2, center, center + delta / 2, center + delta]

        if _is_int_param(name, low, high, cfg, prefix):
            vals = [int(np.clip(round(float(v)), low, high)) for v in vals]
            vals = sorted(set(vals))
        else:
            vals = [float(np.clip(float(v), low, high)) for v in vals]

        grids.append(vals)

    return [dict(zip(names, xs)) for xs in product(*grids)]


def _normalize_values(values: dict[str, Any], specs: SpecMap, cfg: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        if len(spec) == 4:
            out[name] = spec[3]
            continue
        low, high, _ = spec
        raw = values[name]
        if _is_int_param(name, low, high, cfg, prefix):
            out[name] = int(round(float(raw)))
        else:
            out[name] = float(raw)
    return out


def _pick_best(df: pd.DataFrame, specs: SpecMap, cfg: dict[str, Any], prefix: str) -> tuple[dict[str, Any], dict[str, float]]:
    if df.empty:
        raise ValueError("Hyperopt produced no candidates")
    row = df.sort_values("cost_median").iloc[0]
    values = {name: row[name] for name in specs.keys()}
    best_values = _normalize_values(values, specs, cfg, prefix)
    metrics = {
        "cost_median": float(row["cost_median"]),
        "cost_mean": float(row["cost_mean"]),
        "cost_min": float(row["cost_min"]),
    }
    return best_values, metrics


def _save_final_hyperparams(results_dir: str | Path, engine: str, params: dict[str, Any], metrics: dict[str, float]) -> None:
    path = Path(results_dir) / "hyperparams_final.json"
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload[engine] = {
        "params": params,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def run_pso_hyperopt(
    sur0,
    sur1,
    cfg: dict[str, Any],
    run,
) -> tuple[dict[str, Any], dict[str, float]]:
    from .optimize import run_pso

    rng = np.random.default_rng(int(cfg.get("hyperopt_seed", 42)))

    n_trials = int(cfg.get("hyperopt_pso_trials", 80))
    repeats = int(cfg.get("hyperopt_pso_repeats", 2))
    base = dict(cfg)

    def cost_once(values: dict[str, Any], seed: int) -> float:
        local = dict(base)
        _apply_params(local, "pso", values)

        np.random.seed(seed)
        best = run_pso(sur0, sur1, local, run, save_artifacts=False)
        return float(best.cost)

    rows: list[dict[str, Any]] = []
    for t in range(n_trials):
        values = _sample_params(PSO_SPECS, rng, cfg, "pso")
        costs = [cost_once(values, seed=1000 * t + r) for r in range(repeats)]

        rows.append(
            {
                **values,
                "cost_median": float(np.median(costs)),
                "cost_mean": float(np.mean(costs)),
                "cost_min": float(np.min(costs)),
            }
        )

        if hasattr(run, "logger") and (t + 1) % 10 == 0:
            run.logger.info("hyperopt PSO random %s/%s done", t + 1, n_trials)

    df_rs = pd.DataFrame(rows).sort_values("cost_median").reset_index(drop=True)

    top_k = int(cfg.get("hyperopt_pso_refine_topk", 3))
    repeats2 = int(cfg.get("hyperopt_pso_refine_repeats", 3))

    rows2: list[dict[str, Any]] = []
    top = df_rs.head(top_k)

    for i in range(len(top)):
        for values in _refine_grid(top.iloc[i], PSO_SPECS, cfg, "pso"):
            costs = [cost_once(values, seed=900000 + 1000 * i + r) for r in range(repeats2)]
            rows2.append(
                {
                    "base_rank": i,
                    **values,
                    "cost_median": float(np.median(costs)),
                    "cost_mean": float(np.mean(costs)),
                    "cost_min": float(np.min(costs)),
                }
            )

        if hasattr(run, "logger"):
            run.logger.info("hyperopt PSO refine around rank=%s done", i)

    df_ref = pd.DataFrame(rows2).sort_values("cost_median").reset_index(drop=True)
    df_best = df_ref if not df_ref.empty else df_rs
    return _pick_best(df_best, PSO_SPECS, cfg, "pso")


def run_de_hyperopt(
    sur0,
    sur1,
    cfg: dict[str, Any],
    run,
    pos0: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    from .optimize import run_de_full

    rng = np.random.default_rng(int(cfg.get("hyperopt_de_seed", 42)))
    pos0 = np.array(pos0, dtype=float).ravel()

    def cost_once(values: dict[str, Any], seed: int) -> float:
        local = dict(cfg)
        _apply_params(local, "de", values)
        local["de_seed"] = seed
        local["de_init_seed"] = seed

        best = run_de_full(
            sur0,
            sur1,
            local,
            run,
            pos=pos0,
            save_artifacts=False,
            enable_callback=False,
            log_start_done=False,
            write_progress=False,
        )
        return float(best.cost)

    n_trials = int(cfg.get("hyperopt_de_trials", 60))
    repeats = int(cfg.get("hyperopt_de_repeats", 2))

    rows: list[dict[str, Any]] = []
    for t in range(n_trials):
        values = _sample_params(DE_SPECS, rng, cfg, "de")
        costs = [cost_once(values, seed=2000 * t + r) for r in range(repeats)]

        rows.append(
            {
                **values,
                "cost_median": float(np.median(costs)),
                "cost_mean": float(np.mean(costs)),
                "cost_min": float(np.min(costs)),
            }
        )

        if hasattr(run, "logger") and (t + 1) % 10 == 0:
            run.logger.info("hyperopt DE random %s/%s done", t + 1, n_trials)

    df_rs = pd.DataFrame(rows).sort_values("cost_median").reset_index(drop=True)

    top_k = int(cfg.get("hyperopt_de_refine_topk", 3))
    repeats2 = int(cfg.get("hyperopt_de_refine_repeats", 3))

    rows2: list[dict[str, Any]] = []
    top = df_rs.head(top_k)

    for i in range(len(top)):
        for values in _refine_grid(top.iloc[i], DE_SPECS, cfg, "de"):
            costs = [cost_once(values, seed=910000 + 1000 * i + r) for r in range(repeats2)]
            rows2.append(
                {
                    "base_rank": i,
                    **values,
                    "cost_median": float(np.median(costs)),
                    "cost_mean": float(np.mean(costs)),
                    "cost_min": float(np.min(costs)),
                }
            )

        if hasattr(run, "logger"):
            run.logger.info("hyperopt DE refine around rank=%s done", i)

    df_ref = pd.DataFrame(rows2).sort_values("cost_median").reset_index(drop=True)
    df_best = df_ref if not df_ref.empty else df_rs
    return _pick_best(df_best, DE_SPECS, cfg, "de")












def resolve_hyperparams(
    engine: str,
    sur0,
    sur1,
    cfg: dict[str, Any],
    run,
    logger=None,
    **kwargs,
):
    mode = cfg.get(f"{engine}_hyperopt_mode", cfg.get("hyperopt_mode", "run"))
    if mode != "run":
        if logger is not None:
            logger.info("Skip %s hyperopt (mode=%s)", engine, mode)
        return None

    if engine == "pso":
        best_params, metrics = run_pso_hyperopt(sur0, sur1, cfg, run)
        _apply_params(cfg, "pso", best_params)
        _save_final_hyperparams(run.results, "pso", best_params, metrics)
        if logger is not None:
            logger.info(
                "PSO hyperparams optimized: c1=%s c2=%s w=%s (median=%s)",
                best_params.get("c1"),
                best_params.get("c2"),
                best_params.get("w"),
                metrics["cost_median"],
            )
        return {"params": best_params, "metrics": metrics}

    if engine == "de":
        pos0 = kwargs["pos0"]
        best_params, metrics = run_de_hyperopt(sur0, sur1, cfg, run, pos0=pos0)
        _apply_params(cfg, "de", best_params)
        _save_final_hyperparams(run.results, "de", best_params, metrics)
        if logger is not None:
            logger.info(
                "DE hyperparams optimized: mutation=%s recombination=%s popsize=%s (median=%s)",
                best_params.get("mutation"),
                best_params.get("recombination"),
                best_params.get("popsize"),
                metrics["cost_median"],
            )
        return {"params": best_params, "metrics": metrics}

    raise ValueError(f"Unknown engine: {engine}")
