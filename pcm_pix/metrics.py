from __future__ import annotations

"""
metrics.py — оценка качества суррогатных моделей.

На вход: таблица ds.data_0 / ds.data_1 и объект Surrogate.
На выход: MAE/RMSE для амплитуд R/T и для фаз (ошибка фазы считается с wrap в [-π, π]).
"""

from typing import Any

import numpy as np


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Нормализует (wrap) массив углов в диапазон [-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def evaluate_surrogate(
    ds_df,
    sur,
    n: int | None = 5000,
    seed: int = 42,
    label: str = "am",
    return_pred: bool = False,
) -> dict[str, Any]:
    """
    Оценивает качество суррогата на подвыборке `ds.data_0` / `ds.data_1`.

    Ожидаемые колонки `ds_df`: `a`, `d`, `b`, `R`, `T1`, `phi_R`, `phi_T1`.

    Контракт суррогата:
    - `sur.predict(a, d, b) -> ndarray(shape=(N, 4))`
    - столбцы: `[Rcos, Rsin, Tcos, Tsin]` уже в "физическом" масштабе
      (после inverse_transform скейлера).

    Как считаем метрики:
    - амплитуды: $R = \\sqrt{Rcos^2 + Rsin^2}$, $T = \\sqrt{Tcos^2 + Tsin^2}$
    - фазы: `atan2(sin, cos)`; ошибка фазы считается как wrap в [-π, π], чтобы
      не было ложных больших ошибок на границе 2π.
    """
    rng = np.random.default_rng(seed)

    if n is None or n >= len(ds_df):
        idx = np.arange(len(ds_df))
    else:
        idx = rng.choice(len(ds_df), size=int(n), replace=False)

    df = ds_df.iloc[idx]

    # true
    R_true = df["R"].to_numpy()
    T_true = df["T1"].to_numpy()
    phiR_true = df["phi_R"].to_numpy()
    phiT_true = df["phi_T1"].to_numpy()

    pred = sur.predict(df["a"].to_list(), df["d"].to_list(), df["b"].to_list())
    Rcos_p, Rsin_p, Tcos_p, Tsin_p = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

    R_pred = np.sqrt(Rcos_p**2 + Rsin_p**2)
    T_pred = np.sqrt(Tcos_p**2 + Tsin_p**2)
    phiR_pred = np.arctan2(Rsin_p, Rcos_p)
    phiT_pred = np.arctan2(Tsin_p, Tcos_p)

    # amplitude errors
    R_mae = float(np.mean(np.abs(R_pred - R_true)))
    R_rmse = float(np.sqrt(np.mean((R_pred - R_true) ** 2)))
    T_mae = float(np.mean(np.abs(T_pred - T_true)))
    T_rmse = float(np.sqrt(np.mean((T_pred - T_true) ** 2)))

    # phase errors, wrapped
    dphiR = wrap_to_pi(phiR_pred - phiR_true)
    dphiT = wrap_to_pi(phiT_pred - phiT_true)
    phiR_mae = float(np.mean(np.abs(dphiR)))
    phiR_rmse = float(np.sqrt(np.mean(dphiR**2)))
    phiT_mae = float(np.mean(np.abs(dphiT)))
    phiT_rmse = float(np.sqrt(np.mean(dphiT**2)))

    # sanity
    R_out_of_01 = int(np.sum((R_pred < 0) | (R_pred > 1)))
    T_out_of_01 = int(np.sum((T_pred < 0) | (T_pred > 1)))

    out: dict[str, Any] = {
        "label": label,
        "n": int(len(df)),
        "R_mae": R_mae,
        "R_rmse": R_rmse,
        "T_mae": T_mae,
        "T_rmse": T_rmse,
        "phiR_mae": phiR_mae,
        "phiR_rmse": phiR_rmse,
        "phiT_mae": phiT_mae,
        "phiT_rmse": phiT_rmse,
        "R_out_of_01": R_out_of_01,
        "T_out_of_01": T_out_of_01,
    }

    if return_pred:
        out["pred"] = pred
        out["idx"] = idx

    return out

