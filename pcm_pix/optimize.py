from __future__ import annotations

"""
optimize.py — целевая функция и оптимизация (PSO / DE / гибрид).

Здесь собрана логика из старого ноутбука, но оформленная как функции:
- run_pso(...) — запуск PSO (pyswarms)
- load_hyperopt_params(...) — загрузка pso/de гиперпараметров из adb_data_dir
- run_de_full(...) — differential_evolution с init_ar/constraints/callback, максимально 1-в-1
- run_hybrid_pso_de(...) — последовательный PSO → DE

"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pyswarms as ps

from pcm_pix.run import append_text
from pcm_pix.solution import f_vec


@dataclass(frozen=True)
class OPTResult:
    cost: float
    pos: np.ndarray


def _write_best_result(run, *, cost: float, pos: np.ndarray, prefix: str = "best") -> None:
    cost_path = run.results / f"{prefix}_cost.txt"
    pos_path = run.results / f"{prefix}_pos.txt"
    cost_path.write_text(str(float(cost)) + "\n", encoding="utf-8")
    pos_path.write_text(np.array2string(np.asarray(pos, dtype=float), separator=", ") + "\n", encoding="utf-8")

def _make_bounds_arrays(cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    Nn = int(cfg.get("Nn", 11))
    lower = np.array(
        [100*1e-9] * Nn + [100*1e-9] * Nn + [0] * Nn + [0] + [0] + [0.8] + [0.8],
        dtype=float,
    )
    upper = np.array(
        [1000*1e-9] * Nn + [1000*1e-9] * Nn + [1000*1e-9] * Nn + [2 * np.pi] + [2 * np.pi] + [1] + [1],
        dtype=float,
    )
    return lower, upper

def make_linear_constraint(Nn: int):
    """
    Ровно как в ноутбуке:
    a_i - d_i >= 50*1e-9
    d_i - b_i >= 100*1e-9
    (всё в nm, т.к. оптимизация идёт в nm)
    """
    from scipy.optimize import LinearConstraint
    import numpy as np

    M = np.zeros((2 * Nn, 3 * Nn + 4))
    lb = np.zeros(2 * Nn)
    ub = np.zeros(2 * Nn)

    for i in range(Nn):
        M[i, i] = 1.0
        M[i, Nn + i] = -1.0

        M[Nn + i, Nn + i] = 1.0
        M[Nn + i, 2 * Nn + i] = -1.0

        lb[i] = 50*1e-9
        lb[Nn + i] = 100*1e-9
        ub[i] = np.inf
        ub[Nn + i] = np.inf

    return LinearConstraint(M, lb, ub, keep_feasible=True)

#==============================================================
# PSO
#==============================================================


def run_pso(sur0, sur1, cfg: Dict[str, Any], run, save_artifacts: bool = True) -> OPTResult:
    """
    Один прогон PSO.

    По умолчанию сохраняет артефакты в `run.results`:
    - best_cost.txt
    - best_pos.txt

    Для hyperopt имеет смысл отключать запись (save_artifacts=False), чтобы:
    - не перетирать финальные best_* файлами от промежуточных прогонов
    - не тратить время на лишний IO в цикле подбора
    """
    Nn = int(cfg.get("Nn", 11))
    n_particles = int(cfg.get("pso_n_particles", 3000))
    iters = int(cfg.get("pso_iters", 500))

    lower, upper = _make_bounds_arrays(cfg)
    bounds = (lower, upper)

    options = {
        "c1": float(cfg.get("pso_c1", 0.5)),
        "c2": float(cfg.get("pso_c2", 0.3)),
        "w": float(cfg.get("pso_w", 0.9)),
    }

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=int(len(lower)),
        options=options,
        bounds=bounds,
    )

    run.logger.info("PSO start: particles=%s iters=%s dims=%s", n_particles, iters, len(lower))

    cost, pos = optimizer.optimize(
        lambda X: f_vec(X, sur0, sur1, cfg),
        iters=iters,
        verbose=True,
    )

    run.logger.info("PSO done: cost=%s", cost)

    if save_artifacts:
        _write_best_result(run, cost=cost, pos=pos, prefix="best")

    return OPTResult(cost=float(cost), pos=np.array(pos))










#==============================================================
# DE
#==============================================================



def make_init_ar_from_pos(
    pos: np.ndarray,
    N: int = 1000,
    seed: int | None = None,
    cfg: Dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Генерация облака стартовых точек вокруг pos.

    Разброс задаётся через cfg["de_init_spread"] (по умолчанию 0.05),
    что соответствует диапазону [1 - spread, 1 + spread].
    """
    import numpy as np

    spread = 0.05
    if cfg is not None:
        spread = float(cfg.get("de_init_spread", spread))

    rng = np.random.default_rng(seed)
    pos = np.array(pos, dtype=float).ravel()
    # rand in [0,1) → (rand*2-1) in [-1,1) → масштаб в [1-spread, 1+spread)
    B = (rng.random((N, pos.size)) * 2 - 1) * spread + 1.0
    init_ar = np.tile(pos, (N, 1)) * B
    return init_ar




def f_de(X: np.ndarray, sur0, sur1, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Адаптер f(...): (dims, n_pop) приводим к (n_pop, dims)
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    dims = int(cfg.get("dims", X.shape[0]))

    if X.ndim == 1:
        return f_vec(X.reshape(1, -1), sur0, sur1, cfg)

    # если пришло (dims, n_pop)
    if X.shape[0] == dims:
        return f_vec(X.T, sur0, sur1, cfg)

    # если вдруг пришло (n_pop, dims)
    return f_vec(X, sur0, sur1, cfg)


def run_de_full(
    sur0,
    sur1,
    cfg: Dict[str, Any],
    run,
    pos: np.ndarray,
    *,
    save_artifacts: bool = True,
    enable_callback: bool = True,
    log_start_done: bool = True,
    write_progress: bool = True,
) -> OPTResult:
    """
    Полное соответствие to_server_arch:
    - Bounds(... keep_feasible=True)
    - init=init_ar (или constraints=LinearConstraint)
    - callback как в ноутбуке
    - vectorized=True
    """
    import numpy as np
    import time
    from scipy.optimize import differential_evolution, Bounds

    lower, upper = _make_bounds_arrays(cfg)
    cfg["dims"] = int(len(lower))

    # Bounds как в ноутбуке
    bounds = Bounds(lower, upper, keep_feasible=True)

    # параметры DE (ставь как в ноутбуке, если хочешь 1-в-1)
    mutation = cfg.get("de_mutation", 0.99)
    recombination = float(cfg.get("de_recombination", 0.1))
    maxiter = int(cfg.get("de_maxiter", 1000000))
    popsize = int(cfg.get("de_popsize", 5000))  # в ноутбуке встречается 5000
    tol = float(cfg.get("de_tol", 1e-12))
    atol = float(cfg.get("de_atol", 1e-12))
    polish = bool(cfg.get("de_polish", True))
    seed = cfg.get("de_seed", None)
    updating = cfg.get("de_updating", "deferred")

    # init_ar как в ноутбуке
    init_mode = cfg.get("de_init_mode", "init_ar")  # "init_ar" | "x0"
    init_N = int(cfg.get("de_init_N", 1000))
    init_seed = cfg.get("de_init_seed", None)

    if init_mode == "init_ar":
        init = make_init_ar_from_pos(pos, N=init_N, seed=init_seed, cfg=cfg)
        x0 = None
    else:
        init = "latinhypercube"
        x0 = np.array(pos, dtype=float)

    # constraints как опция (в ноутбуке это было закомментировано, но присутствует)
    use_lc = bool(cfg.get("de_use_linear_constraint", False))
    constraints = make_linear_constraint(int(cfg.get("Nn", 11))) if use_lc else ()

    # callback как в ноутбуке (только вместо OUTPUT=open — логгер + файл)
    callback_every = int(cfg.get("de_callback_every", 1000))
    start_time = time.time()
    state = {"iter": -1}

    def callbackF(xk, convergence):
        state["iter"] += 1
        it = state["iter"]

        if it % callback_every == 0 or it == 0:
            delt = f_de(np.array(xk, dtype=float).reshape(-1, 1), sur0, sur1, cfg)
            val = float(delt[0])

            s = "[" + ", ".join(str(float(v)) for v in xk) + "]"
            run.logger.info("Conv %s %.6f", it, val)
            run.logger.info("%s", s)

            if write_progress:
                # файл прогресса как аналог старого OUTPUT/NAME
                p = run.results / "de_progress.txt"
                append_text(p, f"{it:4d}  {val: .6f}\n{s}\n")

            runtime = time.time() - start_time
            run.logger.info("Runtime %.1f min", runtime / 60)

    if log_start_done:
        run.logger.info("DE start (full): maxiter=%s popsize=%s init_mode=%s lc=%s", maxiter, popsize, init_mode, use_lc)

    result = differential_evolution(
        lambda X: f_de(X, sur0, sur1, cfg),  # vectorized!
        bounds=bounds,
        init=init,
        x0=x0,
        mutation=mutation,
        recombination=recombination,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        atol=atol,
        polish=polish,
        seed=seed,
        updating=updating,
        callback=callbackF if enable_callback else None,
        vectorized=True,
        constraints=constraints,
    )

    cost = float(result.fun)
    x = np.array(result.x, dtype=float)

    if log_start_done:
        run.logger.info("DE done (full): cost=%s", cost)
    if save_artifacts:
        _write_best_result(run, cost=cost, pos=x, prefix="best_de")

    return OPTResult(cost=cost, pos=x)




#==============================================================
# Hybrid PSO + DE
#==============================================================


def run_hybrid_pso_de(sur0, sur1, cfg: Dict[str, Any], run) -> OPTResult:
    """
    1) PSO -> 2) DE (с x0 = pos из PSO)
    """
    best_pso = run_pso(sur0, sur1, cfg, run)
    best_de = run_de_full(sur0, sur1, cfg, run, pos=best_pso.pos)

    # финально считаем лучшим DE (обычно он улучшает)
    _write_best_result(run, cost=best_de.cost, pos=best_de.pos, prefix="best")

    return best_de



#==============================================================
# Hyperparams
#==============================================================


def load_hyperopt_params(
    cfg: Dict[str, Any],
    run=None,
    filename: str = "hyperparams_final.json",
) -> bool:
    """
    Если в cfg["adb_data_dir"] есть JSON с итоговыми гиперпараметрами, применяет их в cfg.

    Ожидаемый формат (из hyperopt2.py):
    {
      "pso": {"params": {"c1": ..., "c2": ..., "w": ...}, ...},
      "de":  {"params": {"mutation": ..., "recombination": ..., ...}, ...}
    }

    Также поддерживается упрощённый формат:
    {
      "pso": {"c1": ..., "c2": ..., "w": ...},
      "de":  {"mutation": ..., "recombination": ..., ...}
    }
    """
    import json
    from pathlib import Path

    adb_dir = Path(str(cfg.get("adb_data_dir", "data")))
    hp_path = adb_dir / filename
    if not hp_path.exists():
        if run is not None and hasattr(run, "logger"):
            run.logger.info("No hyperparams file found: %s", hp_path)
        return False

    data = json.loads(hp_path.read_text(encoding="utf-8"))
    applied_keys: list[str] = []

    for engine in ("pso", "de"):
        block = data.get(engine)
        if not isinstance(block, dict):
            continue

        params = block.get("params")
        if not isinstance(params, dict):
            # fallback: считаем, что в block уже лежат параметры
            params = block

        for name, value in params.items():
            cfg_key = f"{engine}_{name}"
            cfg[cfg_key] = value
            applied_keys.append(cfg_key)

    if run is not None and hasattr(run, "logger"):
        if applied_keys:
            run.logger.info("Loaded hyperparams from %s (%s keys)", hp_path, len(applied_keys))
        else:
            run.logger.info("Hyperparams file %s has no usable pso/de params", hp_path)

    return bool(applied_keys)
