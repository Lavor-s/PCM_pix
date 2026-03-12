from __future__ import annotations

"""
optimize_pso.py — целевая функция и оптимизация (PSO / DE / гибрид).

Здесь собрана логика из старого ноутбука, но оформленная как функции:
- f_vec(...) — целевая функция (векторизованная по частицам PSO)
- run_pso(...) — запуск PSO (pyswarms)
- run_pso_until(...) — PSO с перезапусками до достижения порога (как to_server_arch)
- run_de_full(...) — differential_evolution с init_ar/constraints/callback, максимально 1-в-1
- run_hybrid_pso_de(...) — последовательный PSO → DE

Единицы:
- оптимизация идёт в nm для a/d/b, а внутри f_vec мы переводим в метры (*1e-9),
  чтобы совпасть с обучением суррогата.
"""

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pyswarms as ps


@dataclass(frozen=True)
class PSOResult:
    cost: float
    pos: np.ndarray


def make_targets(Nn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 1-в-1 как у тебя в ноутбуке (ам/кр варианты)
    R_th_0 = np.abs(np.linspace(-1, 1, Nn)) ** 2
    R_th_0 = R_th_0 **2
    phi_R_th_0 = np.linspace(-1, 1, Nn) * 0

    R_th_1 = np.abs(np.linspace(-1, 1, Nn))
    R_th_1 = R_th_1 **2
    phi_R_th_1 = np.pi / 2 * np.sign(np.linspace(-1, 1, Nn))
    return R_th_0, phi_R_th_0, R_th_1, phi_R_th_1


def f_vec(X: np.ndarray, sur0, sur1, cfg: Dict[str, Any]) -> np.ndarray:
    # X shape: (n_particles, dimensions)
    Nn = int(cfg.get("Nn", 11))
    b_min = float(cfg.get("b_min_m", 50e-9))

    R_th_0, phi_R_th_0, R_th_1, phi_R_th_1 = make_targets(Nn)

    a = X[:, 0:Nn] * 1e-9
    d = X[:, Nn:2 * Nn] * 1e-9
    b = X[:, 2 * Nn:3 * Nn] * 1e-9

    b[b < b_min] = 0.0

    z = X[:, -4].reshape(-1, 1)
    z_pump = X[:, -3].reshape(-1, 1)
    # sc_1/sc_2 пока просто читаем (как в ноутбуке), но не используем
    sc_1 = X[:, -2].reshape(-1, 1)
    sc_2 = X[:, -1].reshape(-1, 1)

    a_flat, d_flat, b_flat = a.flatten(), d.flatten(), b.flatten()

    pred_0 = sur0.predict(a_flat, d_flat, b_flat).reshape(X.shape[0], Nn, 4)
    pred_1 = sur1.predict(a_flat, d_flat, b_flat).reshape(X.shape[0], Nn, 4)

    R_th_0 = np.full((X.shape[0], Nn), R_th_0)
    phi_R_th_0 = np.full((X.shape[0], Nn), phi_R_th_0)
    R_th_1 = np.full((X.shape[0], Nn), R_th_1)
    phi_R_th_1 = np.full((X.shape[0], Nn), phi_R_th_1)

    RCa = sc_1 * R_th_0 * np.cos(phi_R_th_0 + z)
    RSa = sc_1 * R_th_0 * np.sin(phi_R_th_0 + z)
    RCc = sc_2 * R_th_1 * np.cos(phi_R_th_1 + z_pump)
    RSc = sc_2 * R_th_1 * np.sin(phi_R_th_1 + z_pump)

    d1 = (pred_0[:, :, 0] - RCa) ** 2
    d2 = (pred_0[:, :, 1] - RSa) ** 2
    d3 = (pred_1[:, :, 0] - RCc) ** 2
    d4 = (pred_1[:, :, 1] - RSc) ** 2

    # penalties как у тебя (перенесено 1-в-1 из старых ноутбуков).
    #
    # ВНИМАНИЕ: формулы ниже *исторически* выглядят немного странно:
    # - условие в комментариях/LinearConstraint: a_i - d_i >= 50nm и d_i - b_i >= 100nm
    # - в penalty для constr1 используется (d - a + 50e-9), но модуль берётся от (d - a + 100e-9)
    #   (то есть "50" и "100" смешаны).
    #
    # Это может быть опечаткой в исходном ноутбуке, но мы оставляем как есть, чтобы
    # результат совпадал со старой версией. Если решим исправлять — лучше сделать это
    # отдельным осознанным шагом и сравнить влияние на найденные решения.
    constr1 = (np.sign(d - a + 50e-9) + 1) * np.abs(d - a + 100e-9) * 1e9
    constr2 = (np.sign(b - d + 100e-9) + 1) * np.abs(b - d + 100e-9) * 1e9
    penalty = constr1 + constr2

    return np.sum(d1 + d2 + d3 + d4 + penalty, axis=1)


def run_pso(sur0, sur1, cfg: Dict[str, Any], run, save_artifacts: bool = True) -> PSOResult:
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

    lower = np.array([100] * Nn + [100] * Nn + [0] * Nn + [0] + [0] + [0.8] + [0.8], dtype=float)
    upper = np.array([1000] * Nn + [1000] * Nn + [1000] * Nn + [2*np.pi] + [2*np.pi] + [1] + [1], dtype=float)
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
        # сохраним результат
        (run.results / "best_cost.txt").write_text(str(cost) + "\n", encoding="utf-8")
        (run.results / "best_pos.txt").write_text(np.array2string(pos, separator=", ") + "\n", encoding="utf-8")

    return PSOResult(cost=float(cost), pos=np.array(pos))



import json
from pathlib import Path


def save_solution(run, name: str, pos: np.ndarray, cost: float | None = None, meta: dict | None = None) -> Path:
    """
    Сохраняет решение в run.results/solutions/<name>.json
    """
    out_dir = run.results / "solutions"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": name,
        "cost": None if cost is None else float(cost),
        "pos": [float(x) for x in np.array(pos).ravel()],
        "meta": meta or {},
    }

    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run.logger.info("saved solution %s", path.name)
    return path


def load_solution(path: str | Path) -> tuple[np.ndarray, float | None, dict]:
    """
    Загружает решение из json, возвращает (pos, cost, meta)
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    pos = np.array(payload["pos"], dtype=float)
    cost = payload.get("cost", None)
    meta = payload.get("meta", {}) or {}
    return pos, cost, meta



def _make_bounds_arrays(cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    Nn = int(cfg.get("Nn", 11))
    lower = np.array([100] * Nn + [100] * Nn + [0] * Nn + [0] + [0] + [0.8] + [0.8], dtype=float)
    upper = np.array([1000] * Nn + [1000] * Nn + [1000] * Nn + [2*np.pi] + [2*np.pi] + [1] + [1], dtype=float)
    return lower, upper


def run_de(sur0, sur1, cfg: Dict[str, Any], run, x0: np.ndarray | None = None) -> PSOResult:
    """
    Differential Evolution дорабатывает решение.
    x0: стартовая точка (например, результат PSO или preset pos)
    """
    from scipy.optimize import differential_evolution

    lower, upper = _make_bounds_arrays(cfg)

    # SciPy принимает Bounds или список пар; список пар самый совместимый:
    bounds = list(zip(lower.tolist(), upper.tolist()))

    maxiter = int(cfg.get("de_maxiter", 200))
    popsize = int(cfg.get("de_popsize", 30))
    mutation = cfg.get("de_mutation", (0.5, 1.0))  # можно float или (min,max)
    recombination = float(cfg.get("de_recombination", 0.7))
    polish = bool(cfg.get("de_polish", True))
    seed = cfg.get("de_seed", None)

    run.logger.info(
        "DE start: maxiter=%s popsize=%s mutation=%s recomb=%s polish=%s",
        maxiter, popsize, mutation, recombination, polish
    )

    def obj(x: np.ndarray) -> float:
        # x shape: (dims,)
        return float(f_vec(x.reshape(1, -1), sur0, sur1, cfg)[0])

    kwargs = dict(
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        seed=seed,
        updating="deferred",
    )
    if x0 is not None:
        kwargs["x0"] = np.array(x0, dtype=float)

    result = differential_evolution(obj, **kwargs)

    cost = float(result.fun)
    pos = np.array(result.x, dtype=float)

    run.logger.info("DE done: cost=%s", cost)

    (run.results / "best_de_cost.txt").write_text(str(cost) + "\n", encoding="utf-8")
    (run.results / "best_de_pos.txt").write_text(np.array2string(pos, separator=", ") + "\n", encoding="utf-8")

    return PSOResult(cost=cost, pos=pos)


def run_hybrid_pso_de(sur0, sur1, cfg: Dict[str, Any], run) -> PSOResult:
    """
    1) PSO -> 2) DE (с x0 = pos из PSO)
    """
    best_pso = run_pso(sur0, sur1, cfg, run)
    best_de = run_de_full(sur0, sur1, cfg, run, pos=best_pso.pos)

    # финально считаем лучшим DE (обычно он улучшает)
    (run.results / "best_cost.txt").write_text(str(best_de.cost) + "\n", encoding="utf-8")
    (run.results / "best_pos.txt").write_text(np.array2string(best_de.pos, separator=", ") + "\n", encoding="utf-8")
    return best_de



def make_linear_constraint_nm(Nn: int):
    """
    Ровно как в ноутбуке:
    a_i - d_i >= 50
    d_i - b_i >= 100
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

        lb[i] = 50
        lb[Nn + i] = 100
        ub[i] = np.inf
        ub[Nn + i] = np.inf

    return LinearConstraint(M, lb, ub, keep_feasible=True)


def make_init_ar_from_pos(pos: np.ndarray, N: int = 1000, seed: int | None = None) -> np.ndarray:
    """
    Ровно как в ноутбуке:
    B = (rand*2-1)/10/2 + 1  -> множитель в [0.95..1.05]
    init_ar = tile(pos, (N,1)) * B
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    pos = np.array(pos, dtype=float).ravel()
    B = (rng.random((N, pos.size)) * 2 - 1) / 10 / 2 + 1
    init_ar = np.tile(pos, (N, 1)) * B
    return init_ar


def f_de(X: np.ndarray, sur0, sur1, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Аналог f(...) из ноутбука для differential_evolution(vectorized=True).
    SciPy подаёт X формы (dims, n_pop). Мы приводим к (n_pop, dims) и зовём f_vec.
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


def run_pso_until(sur0, sur1, cfg: Dict[str, Any], run, save_artifacts: bool = True) -> PSOResult:
    """
    Полное соответствие to_server_arch:
    повторяем PSO с reset(), пока cost не станет <= порога (или пока не исчерпаем рестарты).
    """
    import numpy as np

    threshold = float(cfg.get("pso_threshold", np.inf))  # в ноутбуке было 4
    max_restarts = int(cfg.get("pso_max_restarts", 0))   # сколько раз можно reset()

    # используем твою существующую run_pso, но добавляем цикл
    best = run_pso(sur0, sur1, cfg, run, save_artifacts=save_artifacts)

    restarts = 0
    while best.cost > threshold and restarts < max_restarts:
        restarts += 1
        run.logger.info("PSO restart %s/%s (cost=%s > %s)", restarts, max_restarts, best.cost, threshold)

        # пересоздадим оптимизатор тем же кодом, что в run_pso (самый простой способ)
        best = run_pso(sur0, sur1, cfg, run, save_artifacts=save_artifacts)

    return best


def run_de_full(sur0, sur1, cfg: Dict[str, Any], run, pos: np.ndarray) -> PSOResult:
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
    updating = cfg.get("de_updating", "deferred")

    # init_ar как в ноутбуке
    init_mode = cfg.get("de_init_mode", "init_ar")  # "init_ar" | "x0"
    init_N = int(cfg.get("de_init_N", 1000))
    init_seed = cfg.get("de_init_seed", None)

    if init_mode == "init_ar":
        init = make_init_ar_from_pos(pos, N=init_N, seed=init_seed)
        x0 = None
    else:
        init = "latinhypercube"
        x0 = np.array(pos, dtype=float)

    # constraints как опция (в ноутбуке это было закомментировано, но присутствует)
    use_lc = bool(cfg.get("de_use_linear_constraint", False))
    constraints = make_linear_constraint_nm(int(cfg.get("Nn", 11))) if use_lc else ()

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

            # файл прогресса как аналог старого OUTPUT/NAME
            p = run.results / "de_progress.txt"
            with p.open("a", encoding="utf-8") as f:
                f.write(f"{it:4d}  {val: .6f}\n")
                f.write(s + "\n")

            runtime = time.time() - start_time
            run.logger.info("Runtime %.1f min", runtime / 60)

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
        updating=updating,
        callback=callbackF,
        vectorized=True,
        constraints=constraints,
    )

    cost = float(result.fun)
    x = np.array(result.x, dtype=float)

    run.logger.info("DE done (full): cost=%s", cost)
    (run.results / "best_de_cost.txt").write_text(str(cost) + "\n", encoding="utf-8")
    (run.results / "best_de_pos.txt").write_text(np.array2string(x, separator=", ") + "\n", encoding="utf-8")
    return PSOResult(cost=cost, pos=x)