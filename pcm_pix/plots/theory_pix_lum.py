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

from ..lum_tables import (
    R_lum_am_2,
    R_lum_cr_2,
    phi_lum_am_2,
    phi_lum_cr_2,
    R_lum_cr_3,
    R_lum_am_3,
    phi_lum_cr_3,
    phi_lum_am_3,
)

# пишу новые функции для обобщения и унификации:


def g_theory(u, *, state: str):
    'Теоретическая передаточная функция'
    u = np.asarray(u, dtype=float)
    if state == "am":
        return u**2 + 0j
    if state == "cr":
        return 1j * u
    raise ValueError(f"unknown state={state!r}")

# модуль и фаза функции + нормировка
def amp_phase_from_g(u, *, state: str, D: float):
    'Модуль и фаза теоретической передаточной функции + нормировка'
    _ = D
    u = np.asarray(u, dtype=float)
    g = g_theory(u, state=state)
    amp = np.abs(g)
    gmax = np.amax(amp)
    if gmax > 0:
        amp = amp / gmax
    phase = np.angle(g)
    return amp, phase

def pixel_edges(D: float, Npix: int) -> np.ndarray:
    # Npix=11 -> 12 границ: [-5.5L, ..., +5.5L]
    L = D / Npix
    half = Npix / 2.0
    return np.linspace(-half * L, +half * L, Npix + 1)

def pixel_centers(D: float, Npix: int) -> np.ndarray:
    # центры: [-5L, -4L, ..., 0, ..., +5L]
    L = D / Npix
    return (np.arange(Npix) - (Npix // 2)) * L

def f_step(x, x1, y1):
    """
    Ступенчатая функция по границам интервалов.

    Пусть `x1` — массив границ (edges) длины n+1, а `y1` — значения на интервалах длины n:
      y1[i] действует на [x1[i], x1[i+1]) для i=0..n-1.
    Вне диапазона:
      x < x1[0]   -> 0
      x >= x1[-1] -> 0
    """
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1)
    x  = np.asarray(x, dtype=float)
    if x1.ndim != 1 or y1.ndim != 1:
        raise ValueError("x1 и y1 должны быть одномерными массивами")
    if len(x1) < 2:
        raise ValueError("x1 должен содержать хотя бы 2 границы (n+1)")
    n = len(x1) - 1
    if len(y1) != n:
        raise ValueError("ожидается len(y1)=len(x1)-1 (значения на интервалах)")
    if not np.all(np.diff(x1) > 0):
        raise ValueError("x1 должен быть строго возрастающим массивом границ")

    idx = np.searchsorted(x1, x, side="right") - 1
    out_lo = idx < 0
    out_hi = idx >= n
    idx_safe = np.clip(idx, 0, n - 1)
    result = np.where(out_lo | out_hi, 0.0, y1[idx_safe])
    return result
    

def T_phi_pix(u, v, state: str, D: float, Npix: int, f_step):
    _ = v
    edges = pixel_edges(D, Npix)
    centers = pixel_centers(D, Npix)
    amp_c, phi_c = amp_phase_from_g(centers, state=state, D=D)  # теоретические в центрах
    Tpix = f_step(u, edges, amp_c)
    phipix = f_step(u, edges, phi_c)
    return Tpix, phipix






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
    Собственно график
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
    D = float(D if D is not None else cfg.get("plots_theory_pix_lum_D_m", 220e-6))
    L = D / Nn
    x = np.linspace(-D / 2, D / 2, 201)

    # Данные "lum" (набор 1): из суррогатов по pos.
    pred = _pos_to_pred_lum(pos, sur0, sur1, cfg)
    T_lum = pred["R_0"]
    T_p_lum = pred["R_1"]
    phi_lum = pred["phi_R_0"]
    phi_p_lum = pred["phi_R_1"]

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    # (a) Reflection
    if layers.show_theory:
        axs[0, 0].plot(x * 1e6, amp_phase_from_g(x, state = "am", D = D)[0] * 1.1, "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[0, 0].plot(x * 1e6, T_phi_pix(x, x, state="am", D=D, Npix=Nn, f_step = f_step)[0] * 0.95, "r-", linewidth=0.5, markersize=2)
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
        axs[1, 0].plot(x * 1e6, amp_phase_from_g(x, state = "cr", D = D)[0] * 0.9, "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[1, 0].plot(x * 1e6,  T_phi_pix(x, x, state="cr", D=D, Npix=Nn, f_step = f_step)[0] * 0.85, "r-", linewidth=0.5, markersize=2)
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
        axs[0, 1].plot(x * 1e6, amp_phase_from_g(x, state = "am", D = D)[1], "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[0, 1].plot(x * 1e6, T_phi_pix(x, x, state="am", D=D, Npix=Nn, f_step = f_step)[1], "r-", linewidth=0.5, markersize=2)
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
        axs[1, 1].plot(x * 1e6, amp_phase_from_g(x, state = "cr", D = D)[1] * -1, "b-", linewidth=0.9, markersize=2)
    if layers.show_pix:
        axs[1, 1].plot(x * 1e6, T_phi_pix(x, x, state="cr", D=D, Npix=Nn, f_step = f_step)[1]*-1, "r-", linewidth=0.5, markersize=2)
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
        ax2_00.plot(pixel_x, R_lum_am_2, "2r", markersize=5)
        ax2_01.plot(pixel_x, phi_lum_am_2 - phi_lum_am_2[0], "2r", markersize=5)
        ax2_10.plot(pixel_x, R_lum_cr_2, "2r", markersize=5)
        ax2_11.plot(pixel_x, phi_lum_cr_2 - phi_lum_cr_2[0] + np.pi / 2, "2r", markersize=5)
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

