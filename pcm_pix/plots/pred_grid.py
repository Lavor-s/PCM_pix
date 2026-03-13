from __future__ import annotations

"""
pred_grid.py — построение таблиц предсказаний суррогата на регулярной сетке.

В старом `3rd art_PCM_bagel_2025.ipynb` создавался `data_pred`:
- выбирались диапазоны a,d (в метрах), фиксировался b
- строилась сетка (meshgrid)
- прогонялись два суррогата (am/cr)
- считались амплитуды R_0/R_1 и фазы phi_R_0/phi_R_1

Здесь делаем то же самое как чистую функцию, чтобы plots.ipynb был тонким.
"""

from dataclasses import dataclass
from typing import Any
from pcm_pix.solution import Solution

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredGridCfg:
    """Параметры сетки для data_pred."""

    a_min_m: float = 0.2e-6
    a_max_m: float = 1.0e-6
    d_min_m: float = 0.2e-6
    d_max_m: float = 1.0e-6
    b_value_m: float = 0.0
    n: int = 400  # в старом ноутбуке встречается 1000; 400 обычно достаточно и быстрее

    filter_a_gt_d: bool = True
    # (иногда в ноутбуке ещё фильтровали d>b; оставляем опцией на будущее)
    filter_d_gt_b: bool = False

def make_data_pred_grid(
    sur0,
    sur1,
    cfg: PredGridCfg,
) -> pd.DataFrame:
    """
    Возвращает DataFrame `data_pred` со столбцами:
    - a, d, b (в метрах)
    - R_0, phi_R_0 (am surrogate)
    - R_1, phi_R_1 (cr surrogate)
    - dR = R_0 - R_1
    """
    a_vals = np.linspace(float(cfg.a_min_m), float(cfg.a_max_m), int(cfg.n))
    d_vals = np.linspace(float(cfg.d_min_m), float(cfg.d_max_m), int(cfg.n))
    b_vals = np.array([float(cfg.b_value_m)], dtype=float)

    a_grid, d_grid, b_grid = np.meshgrid(a_vals, d_vals, b_vals, indexing="xy")

    a_flat = a_grid.ravel()
    d_flat = d_grid.ravel()
    b_flat = b_grid.ravel()

    R_0, _, phi_R_0, _ = Solution.predict_rtphi(sur0, a_flat, d_flat, b_flat)
    R_1, _, phi_R_1, _ = Solution.predict_rtphi(sur1, a_flat, d_flat, b_flat)

    # как в ноутбуке: фаза в [0..2π)
    phi_R_0 = Solution.wrap_to_0_2pi(phi_R_0)
    phi_R_1 = Solution.wrap_to_0_2pi(phi_R_1)

    df = pd.DataFrame(
        {
            "a": a_flat,
            "d": d_flat,
            "b": b_flat,
            "R_0": R_0,
            "R_1": R_1,
            "phi_R_0": phi_R_0,
            "phi_R_1": phi_R_1,
        }
    )

    # как в ноутбуке: отсечь некорректные области
    if cfg.filter_a_gt_d:
        df = df[df["a"] > df["d"]]
    if cfg.filter_d_gt_b:
        df = df[df["d"] > df["b"]]

    df = df.assign(dR=df["R_0"] - df["R_1"])
    df = df.reset_index(drop=True)
    return df

