from __future__ import annotations

"""
multipole_fields.py — графики разложения по мультиполям + карты полей (|E|^2, |H|^2).

Переносим один из "важных" составных графиков из `3rd art_PCM_bagel_2025.ipynb`:
- слева большой график "Multipole decomposition" (ED/EQ/MD/MQ)
- справа 4 карты: |E|^2 и |H|^2 в XY и YZ сечениях

Файлы в исходном ноутбуке:
- multi_EQ / multi_EQMQ: таблица с колонками ["wl","ED","MD","EQ","MQ","SUM"]
- field_EQ / field_EQMQ: таблица с колонками ["x","y","z","f","Ex","Ey","Ez","Hx","Hy","Hz"]

Важно:
- поля в файлах часто записаны как строки комплексных чисел с "i" — конвертируем в python complex
- wl для поля восстанавливаем как wl = c / f
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import patches


def load_multipole_table(path: str | Path) -> pd.DataFrame:
    """
    Загружает multipole таблицу (EQ/EQMQ) из txt.
    Ожидает колонки: wl, ED, MD, EQ, MQ, SUM.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        names=["wl", "ED", "MD", "EQ", "MQ", "SUM"],
        index_col=False,
    )
    return df


def _to_complex_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()

    # "i" -> "j"
    s2 = s2.str.replace("i", "j", regex=False)

    # часто после замены получается "1.0 + 2.0 j" -> complex не любит пробел перед j
    s2 = s2.str.replace(r"\s+", "", regex=True)

    def parse_one(x: str):
        xl = str(x).strip()
        if xl.lower() in ("nan", "none", ""):
            return complex("nan")

        s = xl.replace("i", "j", 1)
        s = s.replace("D", "E").replace("d", "E")  # если научная нотация встречается
        s = s.replace(",", ".")                    # на всякий случай
        s = "".join(s.split())                    # убираем пробелы

        # ключевое: python complex не любит "+-" и "-+"
        while "+-" in s:
            s = s.replace("+-", "-")
        while "-+" in s:
            s = s.replace("-+", "-")

        return complex(s)

    return s2.map(parse_one)


def load_field_table(path: str | Path) -> pd.DataFrame:
    """
    Загружает таблицу поля EHforV (EQ/EQMQ).

    Формат как в ноутбуке:
    cols = ["x","y","z","f","Ex","Ey","Ez","Hx","Hy","Hz"]
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        names=["x", "y", "z", "f", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        index_col=False,
    )

    for col in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        df[col] = _to_complex_series(df[col])

    # восстановим длину волны (в метрах), как в старом ноутбуке
    c = 3e8
    df["wl"] = c / df["f"].astype(float)
    return df


def select_field_by_wavelength_nearest(field: pd.DataFrame, wl_target_m: float) -> tuple[pd.DataFrame, float]:
    """
    Выбирает из `field` одну длину волны, ближайшую к wl_target_m.

    В старом ноутбуке делалось примерно так:
    - nearest_index = abs(field["wl"] - 850e-9).argsort()[0]
    - field = field[field["wl"] == field["wl"][nearest_index]]

    Возвращает:
    - отфильтрованный df (только одна wl)
    - выбранное значение wl (в метрах)
    """
    if "wl" not in field.columns:
        raise KeyError("field must have column 'wl' (meters). Call load_field_table() first.")

    wl_target_m = float(wl_target_m)
    wl_arr = field["wl"].to_numpy(dtype=float)
    if wl_arr.size == 0:
        raise ValueError("field is empty")

    idx = int(np.argmin(np.abs(wl_arr - wl_target_m)))
    wl_sel = float(wl_arr[idx])

    # обычно wl повторяется точь-в-точь, т.к. считается из дискретного f; используем isclose на всякий случай
    mask = np.isclose(wl_arr, wl_sel, rtol=0.0, atol=0.0)
    out = field[mask]
    out = out.reset_index(drop=True)
    return out, wl_sel


def _norm01(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = float(np.max(np.max(z)))
    return z / m


def _slice_xy(df: pd.DataFrame, *, z0: float, z_rel: float = 0.01) -> pd.DataFrame:
    """Сечение XY: берём точки с z в диапазоне [z0*(1-z_rel), z0*(1+z_rel)]."""
    z = df["z"].astype(float)
    mask = (z >= z0 * (1 - z_rel)) & (z <= z0 * (1 + z_rel))
    return df[mask]


def _slice_yz(df: pd.DataFrame, *, x_eps: float = 1e-12) -> pd.DataFrame:
    """Сечение YZ: берём точки около x=0 в диапазоне [-x_eps, x_eps]."""
    x = df["x"].astype(float)
    mask = (x >= -x_eps) & (x <= x_eps)
    return df[mask]


def plot_multipole_and_fields(
    multi: pd.DataFrame,
    field: pd.DataFrame,
    *,
    field_wl_target_nm: float | None = 1550.0,
    z_xy: float = 76.1538e-9,
    z_rel: float = 0.02,
    x_eps: float = 1e-12,
    wl_marker_nm: float = 1550.0,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.9, 4.0),
    dpi: int | None = None,
) -> plt.Figure:
    """
    Строит составной график:
    - multipole decomposition (нормированный на max(SUM))
    - |E|^2 и |H|^2 в XY и YZ плоскостях
    """
    diameter = 0.327
    inner_diameter = 0.074
    height = 0.22

    fig, axs = plt.subplots(2, 4, figsize=figsize, dpi=dpi)

    # удалим 4 оси слева и сделаем одну большую
    gs = axs[1, 1].get_gridspec()
    axs[0, 0].remove()
    axs[1, 0].remove()
    axs[0, 1].remove()
    axs[1, 1].remove()
    axbig = fig.add_subplot(gs[:2, :2])

    # --- multipole decomposition ---
    wl_nm = multi["wl"].to_numpy(dtype=float) * 1e9
    smax = float(np.max(multi["SUM"].to_numpy(dtype=float)))
    denom = smax if smax > 0 else 1.0

    axbig.plot(wl_nm, multi["ED"] / denom, "r-", label="ED", linewidth=1)
    axbig.plot(wl_nm, multi["EQ"] / denom, "r--", label="EQ", linewidth=1)
    axbig.plot(wl_nm, multi["MD"] / denom, "b-", label="MD", linewidth=1)
    axbig.plot(wl_nm, multi["MQ"] / denom, "b--", label="MQ", linewidth=1)

    axbig.set(
        title="Multipole decomposition",
        xlabel=r"$\lambda$, nm",
        ylabel=r"$\sigma_{scat}$",
        xlim=[1400, 1700],
        ylim=[0, 1.1],
    )
    axbig.legend(loc=1)
    axbig.grid(True)
    axbig.plot([wl_marker_nm, wl_marker_nm], [-1, 5], "k--", linewidth=0.7)

    # --- IMPORTANT: select ONE wavelength for field maps (as in the original notebook) ---
    if field_wl_target_nm is not None:
        field, wl_sel = select_field_by_wavelength_nearest(field, float(field_wl_target_nm) * 1e-9)
        # небольшая подпись для диагностики
        #axbig.text(802, 0.93, f"field wl={wl_sel*1e9:.2f} nm", fontsize=8)

    # --- XY plane ---
    df_xy = _slice_xy(field, z0=z_xy, z_rel=z_rel)
    X = df_xy["x"].to_numpy(dtype=float) * 1e6
    Y = df_xy["y"].to_numpy(dtype=float) * 1e6

    Ex = df_xy["Ex"].to_numpy()
    Ey = df_xy["Ey"].to_numpy()
    Ez = df_xy["Ez"].to_numpy()
    Z_E_xy = _norm01(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)

    Hx = df_xy["Hx"].to_numpy()
    Hy = df_xy["Hy"].to_numpy()
    Hz = df_xy["Hz"].to_numpy()
    Z_H_xy = _norm01(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)

    plot = axs[0, 2].tricontourf(X, Y, Z_E_xy, levels=np.linspace(0, 1, 200), cmap=cm.jet)
    cbar = fig.colorbar(plot, ax=axs[0, 2], ticks=[0, 1], pad=0.05, fraction=0.046)
    cbar.set_ticklabels(["min", "max"])

    plot = axs[0, 3].tricontourf(X, Y, Z_H_xy, levels=np.linspace(0, 1, 200), cmap=cm.jet)
    cbar = fig.colorbar(plot, ax=axs[0, 3], ticks=[0, 1], pad=0.05, fraction=0.046)
    cbar.set_ticklabels(["min", "max"])

    # --- YZ plane (x ~ 0) ---
    df_yz = _slice_yz(field, x_eps=x_eps)
    X2 = df_yz["y"].to_numpy(dtype=float) * 1e6
    Y2 = df_yz["z"].to_numpy(dtype=float) * 1e6

    Ex = df_yz["Ex"].to_numpy()
    Ey = df_yz["Ey"].to_numpy()
    Ez = df_yz["Ez"].to_numpy()
    Z_E_yz = _norm01(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)

    Hx = df_yz["Hx"].to_numpy()
    Hy = df_yz["Hy"].to_numpy()
    Hz = df_yz["Hz"].to_numpy()
    Z_H_yz = _norm01(np.abs(Hx) ** 2 + np.abs(Hy) ** 2 + np.abs(Hz) ** 2)

    plot = axs[1, 2].tricontourf(X2, Y2, Z_E_yz, levels=np.linspace(0, 1, 200), cmap=cm.jet)
    cbar = fig.colorbar(plot, ax=axs[1, 2], ticks=[0, 1], pad=0.05, fraction=0.046)
    cbar.set_ticklabels(["min", "max"])

    plot = axs[1, 3].tricontourf(X2, Y2, Z_H_yz, levels=np.linspace(0, 1, 200), cmap=cm.jet)
    cbar = fig.colorbar(plot, ax=axs[1, 3], ticks=[0, 1], pad=0.05, fraction=0.046)
    cbar.set_ticklabels(["min", "max"])

    # подписи осей/лимиты как в ноутбуке
    axs[0, 2].set(title=r"$|E|^2$ in XY plane", xlabel=r"x, $\mu$m", ylabel=r"y, $\mu$m", xlim=[-0.2, 0.2], ylim=[-0.2, 0.2])
    axs[1, 2].set(title=r"$|E|^2$ in YZ plane", xlabel=r"y, $\mu$m", ylabel=r"z, $\mu$m", xlim=[-0.2, 0.2], ylim=[-0.1, 0.3])
    axs[0, 3].set(title=r"$|H|^2$ in XY plane", xlabel=r"x, $\mu$m", ylabel=r"y, $\mu$m", xlim=[-0.2, 0.2], ylim=[-0.2, 0.2])
    axs[1, 3].set(title=r"$|H|^2$ in YZ plane", xlabel=r"y, $\mu$m", ylabel=r"z, $\mu$m", xlim=[-0.2, 0.2], ylim=[-0.1, 0.3])


    # окружности/прямоугольники (как в исходном)
    for ax in (axs[0, 2], axs[0, 3]):
        ax.add_artist(plt.Circle((0, 0), diameter / 2, fill=False, linewidth=0.7, color="r", linestyle="--"))
        ax.add_artist(plt.Circle((0, 0), inner_diameter / 2, fill=False, linewidth=0.7, color="r", linestyle="--"))

    for ax in (axs[1, 2], axs[1, 3]):
        ax.add_artist(patches.Rectangle((-diameter / 2, 0), (diameter - inner_diameter) / 2, height, fill=False, linewidth=0.7, color="r", linestyle="--"))
        ax.add_artist(patches.Rectangle((-diameter / 2 + (diameter + inner_diameter) / 2, 0), (diameter - inner_diameter) / 2, height, fill=False, linewidth=0.7, color="r", linestyle="--"))

    # стрелочки белые (как в ноутбуке)
    for ax in (axs[0, 2], axs[0, 3]):
        ax.arrow(0.17, 0.1, 0, 0.05, color="w", width=0.004)
        ax.arrow(0.17, 0.15, 0, -0.05, color="w", width=0.004)
    for ax in (axs[1, 2], axs[1, 3]):
        ax.arrow(0.1, 0.25, 0.05, 0, color="w", width=0.004)
        ax.arrow(0.15, 0.25, -0.05, 0, color="w", width=0.004)

    # пропорции и (a)-(e) подписи
    axs[0, 2].set_box_aspect(1)
    axs[0, 3].set_box_aspect(1)
    axs[1, 2].set_box_aspect(1)
    axs[1, 3].set_box_aspect(1)

    #axbig.text(802, 1.025, "(a)")
    axs[0, 2].text(-0.19, 0.14, "(b)", color="w")
    axs[1, 2].text(-0.19, 0.24, "(c)", color="w")
    axs[0, 3].text(-0.19, 0.14, "(d)", color="w")
    axs[1, 3].text(-0.19, 0.24, "(e)", color="w")

    plt.tight_layout(w_pad=0, h_pad=0)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    return fig

