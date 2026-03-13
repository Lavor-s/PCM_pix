from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


import json
import numpy as np

@dataclass
class Solution:
    name: str
    pos: np.ndarray                      # в m
    cost: float | None
    reflection: dict[str, np.ndarray]    # {"am": ..., "cr": ...}
    transmission: dict[str, np.ndarray]  # {"am": ..., "cr": ...}
    phase_shift_reflection: dict[str, np.ndarray]
    phase_shift_transmission: dict[str, np.ndarray]
    meta: dict[str, Any]

    # --- базовые конструкторы ---

    @classmethod
    def from_json(cls, path: str | Path) -> Solution:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        raw_pos = np.array(payload["pos"], dtype=float)
        wl = float(payload.get("wl", payload.get("meta", {}).get("wl", 1.55e-6)))
        pos = cls._ensure_meters(raw_pos, wl)

        cost = payload.get("cost", None)

        def as_array_dict(d: dict | None) -> dict[str, np.ndarray]:
            d = d or {}
            return {k: np.array(v, dtype=float) for k, v in d.items()}

        reflection = as_array_dict(payload.get("reflection"))
        transmission = as_array_dict(payload.get("transmission"))
        phase_R = as_array_dict(payload.get("phase_shift_reflection"))
        phase_T = as_array_dict(payload.get("phase_shift_transmission"))
        meta = payload.get("meta", {}) or {}

        return cls(
            name=payload.get("name", path.stem),
            pos=pos,
            cost=cost,
            reflection=reflection,
            transmission=transmission,
            phase_shift_reflection=phase_R,
            phase_shift_transmission=phase_T,
            meta=meta,
        )


    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_payload()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


    def to_json_payload(self) -> dict[str, Any]:
        def to_list_dict(d: dict[str, np.ndarray]) -> dict[str, list[float]]:
            return {k: [float(x) for x in np.asarray(v).ravel()] for k, v in d.items()}

        return {
            "name": self.name,
            "pos": [float(x) for x in np.asarray(self.pos).ravel()],
            "cost": None if self.cost is None else float(self.cost),
            "reflection": to_list_dict(self.reflection),
            "transmission": to_list_dict(self.transmission),
            "phase_shift_reflection": to_list_dict(self.phase_shift_reflection),
            "phase_shift_transmission": to_list_dict(self.phase_shift_transmission),
            "meta": self.meta or {},
        }


    @classmethod
    def from_pos_and_surrogates(
        cls,
        name: str,
        pos: np.ndarray,
        *,
        cfg: dict,
        sur0,
        sur1,
        meta: dict | None = None,
    ) -> Solution:
        """Построение Solution из pos + sur0/sur1 + CFG"""

        wl = float(cfg.get("wl", 1.55e-6))
        pos = np.asarray(pos, dtype=float).ravel()
        pos = cls._ensure_meters(pos, wl)

        fields = cls.predict_fields_from_pos(
            pos,
            cfg=cfg,
            sur0=sur0,
            sur1=sur1,
        )

        cost_val = float(f_vec(pos.reshape(1, -1), sur0, sur1, cfg)[0])

        return cls(
            name=name,
            pos=pos,
            cost=cost_val,
            reflection=fields["reflection"],
            transmission=fields["transmission"],
            phase_shift_reflection=fields["phase_shift_reflection"],
            phase_shift_transmission=fields["phase_shift_transmission"],
            meta=meta or {},
        )


    @staticmethod
    def _ensure_meters(x: np.ndarray, wl: float, factor: float = 5.0) -> np.ndarray:
        """
        Интерпретирует x как длины:
        - если типичный размер >= factor * wl -> считаем, что x в нм и переводим в м (x * 1e-9)
        - иначе считаем, что x уже в м.
        """
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x

        s = float(np.nanmedian(np.abs(x)))
        if s >= factor * wl:
            # значения выглядят как nm, интерпретированные как "метры" -> переводим
            x[-4:] = x[-4:] * 1e9
            return x * 1e-9

        return x

    @staticmethod
    def rtphi_to_cos_sin(R, T, phi_R, phi_T):
        Rcos = R * np.cos(phi_R)
        Rsin = R * np.sin(phi_R)
        Tcos = T * np.cos(phi_T)
        Tsin = T * np.sin(phi_T)
        return Rcos, Rsin, Tcos, Tsin

    @staticmethod
    def cos_sin_to_rtphi(Rcos, Rsin, Tcos, Tsin):
        R = np.sqrt(Rcos**2 + Rsin**2)
        T = np.sqrt(Tcos**2 + Tsin**2)
        phi_R = np.arctan2(Rsin, Rcos)
        phi_T = np.arctan2(Tsin, Tcos)
        return R, T, phi_R, phi_T

    @staticmethod
    def split_to_adb(
        x_m: np.ndarray,
        Nn: int,
        b_min_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Универсально режет x на a,d,b (в метрах) и применяет порог b_min_m.

        x_m может быть:
        - shape (dims,)
        - shape (n, dims)

        Возвращает a,d,b:
        - если x_m 1D: (Nn,)
        - если x_m 2D: (n, Nn)
        """
        x = np.asarray(x_m, dtype=float)

        if x.ndim == 1:
            if x.shape[0] < 3 * Nn:
                raise ValueError(f"x too short: got {x.shape[0]} for Nn={Nn} (expected >= {3*Nn})")
            a = x[0:Nn]
            d = x[Nn:2 * Nn]
            b = x[2 * Nn:3 * Nn].copy()
            b[b < b_min_m] = 0.0
            return a, d, b

        if x.ndim == 2:
            if x.shape[1] < 3 * Nn:
                raise ValueError(f"x too short: got dims={x.shape[1]} for Nn={Nn} (expected >= {3*Nn})")
            a = x[:, 0:Nn]
            d = x[:, Nn:2 * Nn]
            b = x[:, 2 * Nn:3 * Nn].copy()
            b[b < b_min_m] = 0.0
            return a, d, b

        raise ValueError(f"Expected x_m with ndim 1 or 2, got ndim={x.ndim}")
    

    # --- работа с решениями внутри run (outputs/<run>/results/solutions) ---

    def save_to_run(self, run, name: str) -> Path:
        """
        Сохраняет Solution в outputs/<run>/results/solutions/<name>.json.

        Формат:
        - name, pos, cost, meta (как раньше)
        - reflection / transmission / phase_shift_* пишутся, если есть.
        """
        out_dir = run.results / "solutions"
        out_dir.mkdir(parents=True, exist_ok=True)

        path = self.save_json(out_dir / f"{name}.json")
        run.logger.info("saved solution %s", path.name)
        return path

    @classmethod
    def from_run_file(cls, path: str | Path) -> "Solution":
        """
        Загружает Solution из файла outputs/<run>/results/solutions/<name>.json.
        """
        return cls.from_json(path)


    @staticmethod
    def pred_to_rtphi(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred = np.asarray(pred, dtype=float)
        if pred.ndim != 2 or pred.shape[1] != 4:
            raise ValueError(f"Expected pred with shape (n, 4), got {pred.shape}")
        return Solution.cos_sin_to_rtphi(
            pred[:, 0],
            pred[:, 1],
            pred[:, 2],
            pred[:, 3],
        )

    @staticmethod
    def predict_rtphi(sur, a, d, b) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred = sur.predict(a, d, b)
        return Solution.pred_to_rtphi(pred)

    @staticmethod
    def wrap_to_pi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x + np.pi) % (2 * np.pi) - np.pi
    @staticmethod
    def wrap_to_0_2pi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.mod(x, 2 * np.pi)
    @staticmethod
    def align_phase_to_target(phi: np.ndarray, target: float = 0.0) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        return Solution.wrap_to_pi(phi - float(phi[0]) + target)


    @classmethod
    def predict_fields_from_pos(
        cls,
        pos: np.ndarray,
        *,
        cfg: dict,
        sur0,
        sur1,
    ) -> dict[str, dict[str, np.ndarray]]:
        wl = float(cfg.get("wl", 1.55e-6))
        Nn = int(cfg.get("Nn", 11))
        b_min = float(cfg.get("b_min_m", 50e-9))

        pos = np.asarray(pos, dtype=float).ravel()
        pos = cls._ensure_meters(pos, wl)

        a, d, b = cls.split_to_adb(pos, Nn, b_min)

        R_0, T_0, phi_R_0, phi_T_0 = cls.predict_rtphi(sur0, a, d, b)
        R_1, T_1, phi_R_1, phi_T_1 = cls.predict_rtphi(sur1, a, d, b)

        return {
            "reflection": {"am": R_0, "cr": R_1},
            "transmission": {"am": T_0, "cr": T_1},
            "phase_shift_reflection": {"am": phi_R_0, "cr": phi_R_1},
            "phase_shift_transmission": {"am": phi_T_0, "cr": phi_T_1},
        }


def save_preset(
    name: str,
    pos: np.ndarray,
    *,
    cfg: dict,
    sur0,
    sur1,
    preset_dir: str | Path,
    meta: dict | None = None,
) -> Solution:
    """
    Строит Solution из pos + суррогатов и сохраняет в preset_dir/<name>.json (pos в метрах).
    Возвращает объект Solution.
    """
    sol = Solution.from_pos_and_surrogates(
        name=name,
        pos=pos,
        cfg=cfg,
        sur0=sur0,
        sur1=sur1,
        meta=meta,
    )
    preset_dir = Path(preset_dir)
    preset_dir.mkdir(parents=True, exist_ok=True)
    sol.save_json(preset_dir / f"{name}.json")
    return sol


def load_preset(
    name: str,
    preset_dir: str | Path,
) -> Solution:
    """
    Загружает Solution из preset_dir/<name>.json (pos приводится к метрам).
    """
    path = Path(preset_dir) / f"{name}.json"
    return Solution.from_json(path)



def check_and_fix_presets(

    cfg: dict | None = None,
    sur0=None,
    sur1=None,
) -> dict[str, Any]:
    """
    Проходит по preset_dir/*.json, проверяет и при возможности дополняет пресеты.

    Делает:
    - загружает каждый файл через Solution.from_json (pos -> метры);
    - если нет reflection/transmission/phase_* и переданы cfg,sur0,sur1,
      пересобирает Solution через from_pos_and_surrogates и перезаписывает файл;
    - собирает дубликаты по pos (по округлённым значениям).

    Возвращает dict-отчёт:
    {
      "checked": int,
      "fixed": [Path, ...],
      "skipped_no_surrogates": [Path, ...],
      "duplicates": [[Path, Path, ...], ...],
    }
    """
    preset_dir = Path(cfg.get("preset_dir"))
    report: dict[str, Any] = {
        "checked": 0,
        "fixed": [],
        "skipped_no_surrogates": [],
        "duplicates": [],
    }

    files = sorted(preset_dir.glob("*.json"))
    pos_map: dict[str, list[Path]] = {}

    for path in files:
        report["checked"] += 1

        # Загружаем Solution (pos -> метры, поля если есть читаются как есть)
        sol = Solution.from_json(path)

        # ключ по pos для поиска дублей
        pos_flat = np.asarray(sol.pos, dtype=float).ravel()
        if pos_flat.size:
            key = ",".join(f"{x:.9e}" for x in pos_flat)
            pos_map.setdefault(key, []).append(path)

        has_all_fields = bool(sol.reflection and sol.transmission
                              and sol.phase_shift_reflection and sol.phase_shift_transmission)

        if has_all_fields:
            continue

        if cfg is None or sur0 is None or sur1 is None or pos_flat.size == 0:
            report["skipped_no_surrogates"].append(path)
            continue

        # Достраиваем решение из pos + суррогатов + cfg
        sol_fixed = Solution.from_pos_and_surrogates(
            name=sol.name,
            pos=sol.pos,
            cfg=cfg,
            sur0=sur0,
            sur1=sur1,
            meta=sol.meta,
        )
        sol_fixed.save_json(path)
        report["fixed"].append(path)

    # Собираем дубликаты по pos
    for paths in pos_map.values():
        if len(paths) > 1:
            report["duplicates"].append(paths)

    return report



def save_solution(
    run,
    name: str,
    pos: np.ndarray,
    cost: float | None = None,
    meta: dict | None = None,
    *,
    reflection: dict | None = None,
    transmission: dict | None = None,
    phase_shift_reflection: dict | None = None,
    phase_shift_transmission: dict | None = None,
) -> Path:
    """
    Унифицированная версия save_solution: строит Solution и сохраняет в run.
    """
    sol = Solution(
        name=name,
        pos=np.array(pos, dtype=float),
        cost=None if cost is None else float(cost),
        reflection={k: np.asarray(v, dtype=float) for k, v in (reflection or {}).items()},
        transmission={k: np.asarray(v, dtype=float) for k, v in (transmission or {}).items()},
        phase_shift_reflection={k: np.asarray(v, dtype=float) for k, v in (phase_shift_reflection or {}).items()},
        phase_shift_transmission={k: np.asarray(v, dtype=float) for k, v in (phase_shift_transmission or {}).items()},
        meta=meta or {},
    )
    return sol.save_to_run(run, name)


def load_solution(path: str | Path) -> tuple[np.ndarray, float | None, dict]:
    """
    Унифицированная версия load_solution: читает Solution и возвращает (pos, cost, meta),
    где в meta дополнительно прокинуты reflection / transmission / phase_shift_*.
    """
    sol = Solution.from_run_file(path)

    meta: dict[str, Any] = dict(sol.meta)
    for key in ("reflection", "transmission", "phase_shift_reflection", "phase_shift_transmission"):
        val = getattr(sol, key, None)
        if val and key not in meta:
            # приводим ndarray -> list для совместимости со старым форматом
            meta[key] = {k: np.asarray(v).tolist() for k, v in val.items()}

    return np.asarray(sol.pos, dtype=float), sol.cost, meta








def make_targets(Nn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R_th_0 = np.abs(np.linspace(-1, 1, Nn)) ** 2
    R_th_0 = R_th_0 **2 # Чтобы оптимизировать не по значению интенсивности, а по значению поля
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

    a, d, b = Solution.split_to_adb(X, Nn, b_min)

    z = X[:, -4].reshape(-1, 1)
    z_pump = X[:, -3].reshape(-1, 1)
    sc_1 = X[:, -2].reshape(-1, 1)
    sc_2 = X[:, -1].reshape(-1, 1)

    a_flat, d_flat, b_flat = a.flatten(), d.flatten(), b.flatten()

    pred_0 = sur0.predict(a_flat, d_flat, b_flat).reshape(X.shape[0], Nn, 4)
    pred_1 = sur1.predict(a_flat, d_flat, b_flat).reshape(X.shape[0], Nn, 4)

    R_th_0 = np.full((X.shape[0], Nn), R_th_0)
    phi_R_th_0 = np.full((X.shape[0], Nn), phi_R_th_0)
    R_th_1 = np.full((X.shape[0], Nn), R_th_1)
    phi_R_th_1 = np.full((X.shape[0], Nn), phi_R_th_1)

    # амплитуда и фаза для am-таргета
    RCa, RSa, _, _ = Solution.rtphi_to_cos_sin(
        R=R_th_0 * sc_1,
        T=np.zeros_like(R_th_0),
        phi_R=phi_R_th_0 + z,
        phi_T=np.zeros_like(phi_R_th_0),
    )
    # амплитуда и фаза для cr-таргета (pump)
    RCc, RSc, _, _ = Solution.rtphi_to_cos_sin(
        R=R_th_1 * sc_2,
        T=np.zeros_like(R_th_1),
        phi_R=phi_R_th_1 + z_pump,
        phi_T=np.zeros_like(phi_R_th_1),
    )

    d1 = (pred_0[:, :, 0] - RCa) ** 2
    d2 = (pred_0[:, :, 1] - RSa) ** 2
    d3 = (pred_1[:, :, 0] - RCc) ** 2
    d4 = (pred_1[:, :, 1] - RSc) ** 2

    # penalties
    constr1 = (np.sign(d - a + 100e-9) + 1) * np.abs(d - a + 100e-9) * 1e9
    constr2 = (np.sign(b - d + 100e-9) + 1) * np.abs(b - d + 100e-9) * 1e9
    penalty = constr1 + constr2

    return np.sum(d1 + d2 + d3 + d4 + penalty, axis=1)

