from __future__ import annotations

"""
theory_pix_lum.py — "накладываемый" график (2×2) из 3rd art ноутбука:

- сравнение "теории" (гладкие функции T/T_p/phi) и "пиксельной" модели (T_pix/phi_pix)
- поверх: поправочные данные (lum) несколькими наборами точек на twiny-осях

Важно:
- `pos` берём из пресета (CFG["preset_name"]) так же, как в `main_clean.ipynb`.
- Для удобства дальнейшего расширения (добавится ещё что-то) предусмотрены flags для
  включения/выключения отдельных слоёв (overlays).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def T(x: np.ndarray, D: float) -> np.ndarray:
    """Теоретическая амплитуда отражения (как в старом ноутбуке)."""
    return 4 * x**2 / D**2


def T_p(x: np.ndarray, D: float) -> np.ndarray:
    """Теоретическая амплитуда отражения 'с накачкой' (как в старом ноутбуке)."""
    return 2 * np.abs(x) / D


def phi(x: np.ndarray, D: float) -> np.ndarray:
    """Теоретический фазовый сдвиг (как в старом ноутбуке)."""
    _ = D
    return np.sign(x) * np.pi / 2


def T_pix(u: float, L: float) -> float:
    """Пиксельная аппроксимация T: по номеру пикселя N -> N/5 (1-в-1 как в 3rd art)."""
    u = u * 0.999
    if abs(u) < L / 2:
        N = 0
    else:
        N = (abs(u) - L / 2) // L + 1
    return float(N) / 5.0


def phi_pix(u: float, L: float) -> float:
    """Пиксельная аппроксимация фазы: 0 в центральном пикселе, ±π/2 по знаку (1-в-1)."""
    if abs(u) < L / 2:
        _N = 0
    else:
        _N = (abs(u) - L / 2) // L + 1

    if (-L / 2) < u < (L / 2):
        return 0.0
    if u > 0:
        return float(np.pi / 2)
    return float(-np.pi / 2)


def _pos_to_pred_lum(pos: np.ndarray, sur0, sur1, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Пересчёт "lum" из суррогатов (как в старом ноутбуке: data_pred_0.R_0/R_1/phi_R_0/phi_R_1).

    Важные детали 1-в-1:
    - a/d/b в pos хранятся в nm
    - b < b_min -> b=0
    - R = sqrt(RC^2 + RS^2), phi = atan2(RS, RC)
    """
    pos = np.array(pos, dtype=float).ravel()
    Nn = int(cfg.get("Nn", 11))
    b_min = float(cfg.get("b_min_m", 50e-9))

    if len(pos) < 3 * Nn + 4:
        raise ValueError(f"pos length too small: got {len(pos)} for Nn={Nn} (expected >= {3*Nn+4})")

    a = pos[0:Nn] * 1e-9
    d = pos[Nn : 2 * Nn] * 1e-9
    b = pos[2 * Nn : 3 * Nn] * 1e-9
    b[b < b_min] = 0.0

    pred_0 = sur0.predict(a, d, b).reshape(Nn, 4)
    pred_1 = sur1.predict(a, d, b).reshape(Nn, 4)

    R_0 = np.sqrt(pred_0[:, 0] ** 2 + pred_0[:, 1] ** 2)
    R_1 = np.sqrt(pred_1[:, 0] ** 2 + pred_1[:, 1] ** 2)
    phi_R_0 = np.arctan2(pred_0[:, 1], pred_0[:, 0])
    phi_R_1 = np.arctan2(pred_1[:, 1], pred_1[:, 0])

    return {"R_0": R_0, "R_1": R_1, "phi_R_0": phi_R_0, "phi_R_1": phi_R_1}


@dataclass(frozen=True)
class TheoryPixLumLayers:
    """Переключатели слоёв. Если в будущем появятся новые — просто добавим поля."""

    show_theory: bool = True
    show_pix: bool = True
    show_lum_ann: bool = True  # набор "1r" (в старом ноутбуке — из data_pred_0)
    show_lum_exp: bool = True  # набор "2r" (жёстко заданные массивы)
    show_lum_cones: bool = True  # набор "3g" (жёстко заданные массивы)
    show_panel_labels: bool = True
    show_grid: bool = True


def plot_theory_pix_lum_overlay_2x2(
    *,
    pos: np.ndarray,
    sur0,
    sur1,
    cfg: dict[str, Any],
    out_path: str | Path | None = None,
    layers: TheoryPixLumLayers | None = None,
    figsize: tuple[float, float] = (3.5, 3.3),
    dpi: int | None = None,
    D: float | None = None,
) -> plt.Figure:
    """
    Строит график из `3rd art ...` вокруг строк ~5377..5492:
    2×2:
    - (a) Reflection: T(x)*c1 vs pixel T_pix^2*c1 + lum точки
    - (c) Reflection: T_p(x)*c2 vs pixel T_pix*c2 + lum точки
    - (b) Phase shift: 0-линия + lum точки (phi_lum - phi_lum[0])
    - (d) Phase shift: -phi(x) + lum точки (phi_p - phi_p[0] + pi/2)
    """
    layers = layers or TheoryPixLumLayers()
    pos = np.array(pos, dtype=float).ravel()

    Nn = int(cfg.get("Nn", 11))
    if Nn != 11:
        # В исходном графике оси pixel number и массивы "lum" рассчитаны на 11 пикселей.
        raise ValueError(f"this plot expects Nn=11 (as in old notebook), got Nn={Nn}")

    if len(pos) < 3 * Nn + 4:
        raise ValueError(f"pos length too small: got {len(pos)} for Nn={Nn} (expected >= {3*Nn+4})")

    # В старом ноутбуке c1/c2 брались из X[-2], X[-1] (фактически sc_1/sc_2).
    c1 = float(pos[-2])
    c2 = float(pos[-1])

    # Геометрия для графика (в старом ноутбуке D=220nm, 11 пикселей).
    D = float(D if D is not None else cfg.get("plots_theory_pix_lum_D_m", 220e-9))
    L = D / 11.0
    x = np.linspace(-D / 2, D / 2, 201)

    # Данные "lum" (набор 1): из суррогатов по pos.
    pred = _pos_to_pred_lum(pos, sur0, sur1, cfg)
    T_lum = pred["R_0"]
    T_p_lum = pred["R_1"]
    phi_lum = pred["phi_R_0"]
    phi_p_lum = pred["phi_R_1"]

    # Данные "lum" (набор 2): как в старом ноутбуке (пересчёт люма).
    T_lum_2 = np.array([0.77, 0.58, 0.3, 0.03, 0.05, 0.0, 0.07, 0.12, 0.13, 0.6, 0.95], dtype=float)
    T_p_lum_2 = np.array([0.91, 0.88, 0.74, 0.48, 0.14, 0.0, 0.23, 0.45, 0.63, 0.99, 0.93], dtype=float)
    phi_lum_2 = np.array([-1.3, -1.4, -1.3, -1.2, -0.2, -0.1, -1.0, -1.4, -1.6, -1.4, -1.3], dtype=float)
    phi_p_lum_2 = np.array([2.2, 2.2, 2.2, 2.1, 2.8, 2.8, -0.7, -1.0, -1.0, -0.8, -0.8], dtype=float)

    # Данные "lum" (набор 3): для конусов.
    R_lum_cr_3 = np.array([0.85, 0.83, 0.71, 0.51, 0.10, 0.00, 0.32, 0.40, 0.65, 0.93, 0.97], dtype=float)
    R_lum_am_3 = np.array([0.97, 1.00, 0.97, 0.53, 0.09, 0.02, 0.15, 0.23, 0.06, 0.33, 0.77], dtype=float)
    phi_lum_cr_3 = np.array([2.1, 2.2, 2.2, 2.0, 2.0, -0.2, -1.4, -1.1, -1.6, -1.5, -1.3], dtype=float)
    phi_lum_am_3 = np.array([-0.6, -0.5, -0.8, -1.0, -0.8, -0.4, -1.0, -1.4, -2.4, -2.1, -1.8], dtype=float)

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    # (a) Reflection
    if layers.show_theory:
        axs[0, 0].plot(x * 1e6, T(x, D) * 1.1, "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[0, 0].plot(x * 1e6, [T_pix(float(i), L) ** 2 * c1 for i in x], "r-", linewidth=0.5, markersize=2)
    ax2_00 = axs[0, 0].twiny()
    axs[0, 0].set(
        title="",
        xlabel=r"x, $\mu$m",
        ylabel=r"Reflection",
        xlim=[-D * 1e6 / 2, D * 1e6 / 2],
        ylim=[-0.1, 1.1],
        yticks=[0, 0.5, 1],
    )
    ax2_00.set(title="", xlabel=r"Pixel number", ylabel="", xlim=[0.5, 11.5], xticks=[1, 3, 5, 7, 9, 11])

    # (c) Reflection (pump)
    if layers.show_theory:
        axs[1, 0].plot(x * 1e6, T_p(x, D) * 0.9, "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[1, 0].plot(x * 1e6, [T_pix(float(i), L) * c2 for i in x], "r-", linewidth=0.5, markersize=2)
    ax2_10 = axs[1, 0].twiny()
    axs[1, 0].set(
        title="",
        xlabel=r"x, $\mu$m",
        ylabel=r"Reflection",
        xlim=[-D * 1e6 / 2, D * 1e6 / 2],
        ylim=[-0.1, 1.1],
        yticks=[0, 0.5, 1],
    )
    ax2_10.set(title="", xlabel=r"Pixel number", ylabel="", xlim=[0.5, 11.5], xticks=[1, 3, 5, 7, 9, 11])

    # (b) Phase shift (am)
    if layers.show_theory:
        axs[0, 1].plot(x * 1e6, phi(x, D) * 0, "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[0, 1].plot(x * 1e6, [phi_pix(float(i), L) * 0 for i in x], "r-", linewidth=0.5, markersize=2)
    ax2_01 = axs[0, 1].twiny()
    axs[0, 1].set(
        title="",
        xlabel=r"x, $\mu$m",
        ylabel=r"Phase shift",
        xlim=[-D * 1e6 / 2, D * 1e6 / 2],
        ylim=[-np.pi, np.pi],
        yticks=[0, np.pi, -np.pi],
        yticklabels=["0", r"$\pi$", r"-$\pi$"],
    )
    ax2_01.set(title="", xlabel=r"Pixel number", ylabel="", xlim=[0.5, 11.5], xticks=[1, 3, 5, 7, 9, 11])

    # (d) Phase shift (cr)
    if layers.show_theory:
        axs[1, 1].plot(x * 1e6, phi(x, D) * -1, "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[1, 1].plot(x * 1e6, [phi_pix(float(i), L) * -1 for i in x], "r-", linewidth=0.5, markersize=2)
    ax2_11 = axs[1, 1].twiny()
    axs[1, 1].set(
        title="",
        xlabel=r"x, $\mu$m",
        ylabel=r"Phase shift",
        xlim=[-D * 1e6 / 2, D * 1e6 / 2],
        ylim=[-np.pi, np.pi],
        yticks=[0, np.pi, -np.pi],
        yticklabels=["0", r"$\pi$", r"-$\pi$"],
    )
    ax2_11.set(title="", xlabel=r"Pixel number", ylabel="", xlim=[0.5, 11.5], xticks=[1, 3, 5, 7, 9, 11])

    # Панельные метки (a)-(d)
    if layers.show_panel_labels:
        axs[0, 0].text(-0.11, 0, "(a)", color="k", horizontalalignment="left", verticalalignment="baseline")
        axs[0, 1].text(-0.11, -np.pi * 0.875, "(b)", color="k", horizontalalignment="left", verticalalignment="baseline")
        axs[1, 0].text(-0.11, 0, "(c)", color="k", horizontalalignment="left", verticalalignment="baseline")
        axs[1, 1].text(-0.11, -np.pi * 0.875, "(d)", color="k", horizontalalignment="left", verticalalignment="baseline")

    # --- Поправочные данные из люма ---
    pixel_x = np.linspace(1, 11, 11)

    if layers.show_lum_ann:
        phi_lum_plot = np.array(phi_lum, dtype=float) - float(phi_lum[0])
        phi_p_lum_plot = np.array(phi_p_lum, dtype=float) - float(phi_p_lum[0]) + np.pi / 2
        ax2_00.plot(pixel_x, T_lum, "1r", markersize=5, mec="k")
        ax2_01.plot(pixel_x, phi_lum_plot, "1r", markersize=5, mec="k")
        ax2_10.plot(pixel_x, T_p_lum, "1r", markersize=5, mec="k")
        ax2_11.plot(pixel_x, phi_p_lum_plot, "1r", markersize=5, mec="k")

    if layers.show_lum_exp:
        ax2_00.plot(pixel_x, T_lum_2, "2r", markersize=5)
        ax2_01.plot(pixel_x, phi_lum_2 - phi_lum_2[0], "2r", markersize=5)
        ax2_10.plot(pixel_x, T_p_lum_2, "2r", markersize=5)
        ax2_11.plot(pixel_x, phi_p_lum_2 - phi_p_lum_2[0] + np.pi / 2, "2r", markersize=5)

    if layers.show_lum_cones:
        ax2_00.plot(pixel_x, R_lum_am_3, "3g", markersize=5)
        ax2_01.plot(pixel_x, phi_lum_am_3 - phi_lum_am_3[0], "3g", markersize=5)
        ax2_10.plot(pixel_x, R_lum_cr_3, "3g", markersize=5)
        ax2_11.plot(pixel_x, phi_lum_cr_3 - phi_lum_cr_3[0] + np.pi / 2, "3g", markersize=5)

    if layers.show_grid:
        for ax in axs.ravel():
            ax.grid(True)

    plt.tight_layout(pad=1)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    return fig

