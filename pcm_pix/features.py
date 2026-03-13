from __future__ import annotations

"""
features.py — подготовка датасета для суррогатных нейросетей.

Берём "mesh tables" (таблицы из симулятора) и приводим их к виду:
- признаки X: [a, d, b] (геометрия)
- цели y: [Rcos, Rsin, Tcos, Tsin] (амплитуда+фаза через cos/sin)

Также считаем A = 1 - (R + T1) как в исходном ноутбуке.
"""

from pathlib import Path
from typing import Sequence, Dict, Any, Tuple
from pcm_pix.solution import Solution

import numpy as np
import pandas as pd


def load_mesh_tables(cfg: Dict[str, Any], base_dir: str | Path = "data") -> list[pd.DataFrame]:
    """Забираем датасеты в tuple"""
    base_dir = Path(base_dir)
    names = cfg.get("adb_data")
    cols = ["a", "d", "b", "wl", "R", "T1", "T2", "phi_R", "phi_T1", "phi_T2"]
    # чуть устойчивее, чем sep=" " (если много пробелов/табов)
    tables = [
        pd.read_csv(base_dir / name, sep=r"\s+", engine="python", header=1, names=cols)
        for name in names
    ]
    return tables

def make_nn_dataset(
    mesh_tables: Sequence[pd.DataFrame], wl: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит полный датасет для суррогатных моделей.

    Возвращает:
    - df      : полный DataFrame
    - data_0  : подтаблица для N=0 при заданной wl
    - data_1  : подтаблица для N=1 при заданной wl
    - X_0, y_0: признаки (a, d, b) и цели (Rcos, Rsin, Tcos, Tsin) для N=0
    - X_1, y_1: признаки (a, d, b) и цели (Rcos, Rsin, Tcos, Tsin) для N=1
    """
    t0 = clean_mesh_table(mesh_tables[0])
    t1 = clean_mesh_table(mesh_tables[1])

    t0 = t0.copy()
    t0["N"] = 0

    t1 = t1.copy()
    t1["N"] = 1

    df = pd.concat([t0, t1], ignore_index=True)

    Rcos, Rsin, Tcos, Tsin = Solution.rtphi_to_cos_sin(
        R=df["R"].to_numpy(),
        T=df["T1"].to_numpy(),
        phi_R=df["phi_R"].to_numpy(),
        phi_T=df["phi_T1"].to_numpy(),
    )

    df["Rcos"] = Rcos
    df["Rsin"] = Rsin
    df["Tcos"] = Tcos
    df["Tsin"] = Tsin

    
    df = df[
        [
            "a",
            "d",
            "b",
            "wl",
            "N",
            "R",
            "T1",
            "phi_R",
            "phi_T1",
            "A",
            "Rcos",
            "Rsin",
            "Tcos",
            "Tsin",
        ]
    ]

    data_0 = df[(df.wl == wl) & (df.N == 0)].copy()
    data_1 = df[(df.wl == wl) & (df.N == 1)].copy()

    X_0 = data_0[["a", "d", "b"]].values
    y_0 = data_0[["Rcos", "Rsin", "Tcos", "Tsin"]].values

    X_1 = data_1[["a", "d", "b"]].values
    y_1 = data_1[["Rcos", "Rsin", "Tcos", "Tsin"]].values

    return df, data_0, data_1, X_0, y_0, X_1, y_1




def _str_to_complex_angle(col: pd.Series) -> pd.Series:
    """Конвертирует колонку вида 'a+bi' в фазу angle(x) в диапазоне [0..2π)."""
    col = col.apply(lambda x: complex(str(x).replace("i", "j")))
    col[col == (1 + 1j)] = np.nan
    col = col.apply(lambda x: np.angle(x))
    col = col.apply(lambda x: x + 2 * np.pi if x < 0 else x)
    return col


def clean_mesh_table(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка/нормализация колонок таблицы — максимально близко к исходному ноутбуку."""
    df = df.copy()

    # как в ноутбуке: привести R/T в [0..1]
    df["T1"] = df["T1"].apply(lambda x: max(min(abs(x), 1), 0))
    df["T2"] = df["T2"].apply(lambda x: max(min(abs(x), 1), 0))
    df["R"] = df["R"].apply(lambda x: max(min(abs(x), 1), 0))

    # фазы: строка "a+bi" -> угол, [0..2pi)
    df["phi_R"] = _str_to_complex_angle(df["phi_R"])
    df["phi_T1"] = _str_to_complex_angle(df["phi_T1"])
    df["phi_T2"] = _str_to_complex_angle(df["phi_T2"])

    # в ноутбуке: phi_T2 удаляется
    df.drop(["phi_T2"], axis=1, inplace=True)

    # выкинуть nan после конвертаций
    df.dropna(inplace=True)

    # A = 1 - (R + T1), затем ограничить [0..1]
    df["A"] = 1 - (df["R"] + df["T1"])
    df["A"] = df["A"].apply(lambda x: max(min(x, 1), 0))

    return df
