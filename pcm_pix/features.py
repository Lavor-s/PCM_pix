from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetaDataset:
    df: pd.DataFrame
    data_0: pd.DataFrame
    data_1: pd.DataFrame
    X_0: np.ndarray
    y_0: np.ndarray
    X_1: np.ndarray
    y_1: np.ndarray


def load_mesh_tables(cfg: Dict[str, Any], base_dir: str | Path = "data") -> list[pd.DataFrame]:
    base_dir = Path(base_dir)
    names = cfg.get("mesh_names", [
        "Sb2Se3 - amorphous_mesh_sbse_2025.txt",
        "Sb2Se3 - crystal_mesh12_sbse_2025.txt",
    ])

    cols = ["a", "d", "b", "wl", "R", "T1", "T2", "phi_R", "phi_T1", "phi_T2"]

    # чуть устойчивее, чем sep=" " (если много пробелов/табов)
    tables = [
        pd.read_csv(base_dir / name, sep=r"\s+", engine="python", header=1, names=cols)
        for name in names
    ]
    return tables


def _str_to_complex_angle(col: pd.Series) -> pd.Series:
    col = col.apply(lambda x: complex(str(x).replace("i", "j")))
    col[col == (1 + 1j)] = np.nan
    col = col.apply(lambda x: np.angle(x))
    col = col.apply(lambda x: x + 2 * np.pi if x < 0 else x)
    return col


def clean_mesh_table(df: pd.DataFrame) -> pd.DataFrame:
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


def make_nn_dataset(mesh_tables: Sequence[pd.DataFrame], wl: float = 1.55e-6) -> MetaDataset:
    t0 = clean_mesh_table(mesh_tables[0])
    t1 = clean_mesh_table(mesh_tables[1])

    t0 = t0.copy()
    t0["N"] = 0

    t1 = t1.copy()
    t1["N"] = 1

    df = pd.concat([t0, t1], ignore_index=True)

    df["Rcos"] = df["R"] * np.cos(df["phi_R"])
    df["Rsin"] = df["R"] * np.sin(df["phi_R"])
    df["Tcos"] = df["T1"] * np.cos(df["phi_T1"])
    df["Tsin"] = df["T1"] * np.sin(df["phi_T1"])

    df = df[["a", "d", "b", "wl", "N", "R", "T1", "phi_R", "phi_T1", "A", "Rcos", "Rsin", "Tcos", "Tsin"]]

    data_0 = df[(df.wl == wl) & (df.N == 0)].copy()
    data_1 = df[(df.wl == wl) & (df.N == 1)].copy()

    X_0 = data_0[["a", "d", "b"]].values
    y_0 = data_0[["Rcos", "Rsin", "Tcos", "Tsin"]].values

    X_1 = data_1[["a", "d", "b"]].values
    y_1 = data_1[["Rcos", "Rsin", "Tcos", "Tsin"]].values

    return MetaDataset(df=df, data_0=data_0, data_1=data_1, X_0=X_0, y_0=y_0, X_1=X_1, y_1=y_1)