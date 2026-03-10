from __future__ import annotations

"""
io.py — загрузка исходных таблиц материалов.

Здесь нет "магии": просто читаем файлы в pandas.DataFrame и возвращаем удобный контейнер.
Пути можно задавать через cfg, чтобы в ноутбуке не было абсолютных путей.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import pandas as pd


@dataclass(frozen=True)
class Materials:
    sb2se3_am: pd.DataFrame
    sb2se3_cr: pd.DataFrame
    gst_am: pd.DataFrame
    gst_cr: pd.DataFrame


def load_materials(cfg: Dict[str, Any], base_dir: str | Path = ".") -> Materials:
    """
    Загружает 4 таблицы материалов (как в твоём ноутбуке), но аккуратно:
    - пути задаются в cfg, либо берутся дефолтные имена файлов
    - base_dir позволяет хранить данные рядом с ноутбуком/в папке data/
    """
    base_dir = Path(base_dir)

    sb2se3_am_path = base_dir / cfg.get("sb2se3_am_path", "Sb2Se3_am.txt")
    sb2se3_cr_path = base_dir / cfg.get("sb2se3_cr_path", "Sb2Se3_cr.txt")
    gst_am_path = base_dir / cfg.get("gst_am_path", "Frantz-amorphous.csv")
    gst_cr_path = base_dir / cfg.get("gst_cr_path", "Frantz-crystal.csv")

    sb2se3_am = pd.read_csv(sb2se3_am_path, header=0, decimal=",", sep="\t")
    sb2se3_cr = pd.read_csv(sb2se3_cr_path, header=0, decimal=",", sep="\t")

    gst_am = pd.read_csv(gst_am_path, names=["wl", "n", "k"], decimal=".", sep=r"\s+")
    gst_cr = pd.read_csv(gst_cr_path, names=["wl", "n", "k"], decimal=".", sep=r"\s+")

    return Materials(
        sb2se3_am=sb2se3_am,
        sb2se3_cr=sb2se3_cr,
        gst_am=gst_am,
        gst_cr=gst_cr,
    )