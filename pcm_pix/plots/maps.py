from __future__ import annotations

"""
maps.py — вспомогательные функции для "карт" (contour / tricontourf) по (a,d).

В старом ноутбуке много раз повторяется один и тот же шаблон:
- X = df.a * 1e6, Y = df.d * 1e6
- tricontourf(...)
- fig.colorbar(..., pad=0.05, fraction=0.046)
- set_box_aspect(1)
- одинаковые лимиты осей

Здесь держим эту логику в одном месте.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _apply_common_axis(ax, *, xlim=(0.20, 1.00), ylim=(0.20, 1.00)) -> None:
    ax.set(
        xlim=xlim,
        ylim=ylim,
        xticks=[0.2, 0.4, 0.6, 0.8, 1.0],
        yticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    ax.set_box_aspect(1)


def _apply_axis_custom(ax, *, xlim, ylim) -> None:
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_box_aspect(1)


def plot_data_pred_2x2(
    data_pred: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int | None = None,
) -> plt.Figure:
    """
    Блок `3rd art ... (1-67)`:
    2×2 карты по data_pred:
    - R_0
    - R_1
    - phi_R_0
    - phi_R_1
    """
    if len(data_pred) < 3:
        raise ValueError("tricontourf requires at least 3 points; got len(data_pred) < 3")

    fig, ax = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    X = data_pred["a"].to_numpy() * 1e6
    Y = data_pred["d"].to_numpy() * 1e6

    # R_0
    Z = data_pred["R_0"].to_numpy()
    plot = ax[0, 0].tricontourf(X, Y, Z, levels=np.linspace(0, 1, 50))
    plot.set_clim(0.0, 1.0)
    cbar = fig.colorbar(plot, ax=ax[0, 0], ticks=[0, 0.5, 1], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels(["0", r"0.5", r"1"])
    ax[0, 0].set(title=r"$R_{am}^{ANN}$", xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
    _apply_common_axis(ax[0, 0])

    # R_1
    Z = data_pred["R_1"].to_numpy()
    plot = ax[1, 0].tricontourf(X, Y, Z, levels=np.linspace(0, 1, 50))
    plot.set_clim(0.0, 1.0)
    cbar = fig.colorbar(plot, ax=ax[1, 0], ticks=[0, 0.5, 1], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels(["0", r"0.5", r"1"])
    ax[1, 0].set(title=r"$R_{cr}^{ANN}$", xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
    _apply_common_axis(ax[1, 0])

    # phi_R_0
    Z = data_pred["phi_R_0"].to_numpy()
    plot = ax[0, 1].tricontourf(X, Y, Z, levels=np.linspace(0, 2 * np.pi, 50))
    plot.set_clim(0.0, 2 * np.pi)
    cbar = fig.colorbar(plot, ax=ax[0, 1], ticks=[0, np.pi, 2 * np.pi], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax[0, 1].set(title=r"$\Delta\varphi_{am}^{ANN}$", xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
    _apply_common_axis(ax[0, 1])

    # phi_R_1
    Z = data_pred["phi_R_1"].to_numpy()
    plot = ax[1, 1].tricontourf(X, Y, Z, levels=np.linspace(0, 2 * np.pi, 50))
    plot.set_clim(0.0, 2 * np.pi)
    cbar = fig.colorbar(plot, ax=ax[1, 1], ticks=[0, np.pi, 2 * np.pi], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax[1, 1].set(title=r"$\Delta\varphi_{am}^{ANN}$", xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
    _apply_common_axis(ax[1, 1])

    plt.tight_layout(h_pad=-4, w_pad=1)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    return fig


def plot_data_pred_diffs_2x2(
    data_pred: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int | None = None,
) -> plt.Figure:
    """
    Блок `3rd art ... (1-73)`:
    2×2 карты по data_pred:
    - R_0
    - phi_R_0
    - (R_1 - R_0)
    - (phi_R_1 - phi_R_0)

    Для разностей используется diverging colormap (seismic), как в старом ноутбуке.
    """
    if len(data_pred) < 3:
        raise ValueError("tricontourf requires at least 3 points; got len(data_pred) < 3")

    fig, ax = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    X = data_pred["a"].to_numpy() * 1e6
    Y = data_pred["d"].to_numpy() * 1e6

    # R_0
    Z = data_pred["R_0"].to_numpy()
    plot = ax[0, 0].tricontourf(X, Y, Z, levels=np.linspace(0, 1, 50))
    plot.set_clim(0.0, 1.0)
    cbar = fig.colorbar(plot, ax=ax[0, 0], ticks=[0, 0.5, 1], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels(["0", r"0.5", r"1"])
    ax[0, 0].set(title=r"$R_{am}^{ANN}$", xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
    _apply_common_axis(ax[0, 0])

    # phi_R_0
    Z = data_pred["phi_R_0"].to_numpy()
    plot = ax[0, 1].tricontourf(X, Y, Z, levels=np.linspace(0, 2 * np.pi, 50))
    plot.set_clim(0.0, 2 * np.pi)
    cbar = fig.colorbar(plot, ax=ax[0, 1], ticks=[0, np.pi, 2 * np.pi], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax[0, 1].set(title=r"$\Delta\varphi_{am}^{ANN}$", xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
    _apply_common_axis(ax[0, 1])

    # dR = R_1 - R_0
    Z = (data_pred["R_1"] - data_pred["R_0"]).to_numpy()
    plot = ax[1, 0].tricontourf(X, Y, Z, levels=np.linspace(-1, 1, 500), cmap="seismic")
    plot.set_clim(-1.0, 1.0)
    cbar = fig.colorbar(plot, ax=ax[1, 0], ticks=[-1, 0, 1], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels([r"-1", r"0", r"1"])
    ax[1, 0].set(title=r"$R_{cr}^{ANN} - R_{am}^{ANN}$", xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
    _apply_common_axis(ax[1, 0])

    # dphi = phi_R_1 - phi_R_0
    Z = (data_pred["phi_R_1"] - data_pred["phi_R_0"]).to_numpy()
    plot = ax[1, 1].tricontourf(X, Y, Z, levels=np.linspace(-2 * np.pi, 2 * np.pi, 500), cmap="seismic")
    plot.set_clim(-2 * np.pi, 2 * np.pi)
    cbar = fig.colorbar(plot, ax=ax[1, 1], ticks=[-2 * np.pi, 0, 2 * np.pi], pad=0.05, fraction=0.046)
    cbar.ax.set_yticklabels([r"$-2\pi$", r"0", r"$2\pi$"])
    ax[1, 1].set(
        title=r"$\Delta\varphi_{cr}^{ANN} - \Delta\varphi_{am}^{ANN}$",
        xlabel=r"a, $\mu$m",
        ylabel=r"d, $\mu$m",
    )
    _apply_common_axis(ax[1, 1])

    plt.tight_layout(h_pad=-4, w_pad=1)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    return fig


def plot_true_vs_pred_2x4(
    data: pd.DataFrame,
    data_pred: pd.DataFrame,
    *,
    wl: float = 1.55e-6,
    b_fix: float = 0.0,
    wl_rtol: float = 0.0,
    wl_atol: float = 0.0,
    b_rtol: float = 0.0,
    b_atol: float = 0.0,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (7.0, 3.0),
    dpi: int | None = None,
    xlim_c: tuple[float, float] = (0.25, 1.0),
    ylim_c: tuple[float, float] | None = None,
    dots_um: tuple[list[float], list[float]] | None = None,
) -> plt.Figure:
    """
    Блок `3rd art ... (1-170)`:
    2×4 карты, сравнение истинных mesh-данных и предсказаний суррогатов.

    Колонки, которые ожидаем:
    - data: a,d,b,wl,N,R,phi_R
    - data_pred: a,d,b,R_0,phi_R_0,R_1,phi_R_1

    Оси:
    - X=a*1e6, Y=d*1e6 (в мкм)
    """
    if ylim_c is None:
        ylim_c = xlim_c

    if dots_um is None:
        # как в старом ноутбуке
        dots_um = (
            [0.35, 0.5, 0.9, 0.7, 0.85, 0.8, 0.9, 0.9, 0.95],
            [0.3, 0.3, 0.3, 0.6, 0.6, 0.65, 0.75, 0.85, 0.9],
        )

    # фильтр по wl/b_fix (в старом ноутбуке b_fix=0).
    # Важно: сравнение float через == может иногда давать пустой df, поэтому поддерживаем isclose.
    wl = float(wl)
    b_fix = float(b_fix)

    wl_mask = np.isclose(data["wl"].to_numpy(dtype=float), wl, rtol=wl_rtol, atol=wl_atol)
    b_mask = np.isclose(data["b"].to_numpy(dtype=float), b_fix, rtol=b_rtol, atol=b_atol)
    d_am = data[wl_mask & b_mask & (data["N"] == 0)]
    d_cr = data[wl_mask & b_mask & (data["N"] == 1)]

    if len(d_am) < 3 or len(d_cr) < 3:
        raise ValueError(
            "Not enough points for tricontourf after filtering. "
            f"Got len(am)={len(d_am)} len(cr)={len(d_cr)} for wl={wl} b_fix={b_fix}. "
            "Try relaxing tolerances: wl_atol/wl_rtol and b_atol/b_rtol."
        )

    fig, ax = plt.subplots(2, 4, figsize=figsize, dpi=dpi)

    def _tric(ax_, df, z_col, *, levels, clim, ticks, ticklabels, title):
        X = df["a"].to_numpy() * 1e6
        Y = df["d"].to_numpy() * 1e6
        Z = df[z_col].to_numpy()
        plot = ax_.tricontourf(X, Y, Z, levels=levels)
        plot.set_clim(*clim)
        cbar = fig.colorbar(plot, ax=ax_, ticks=ticks, pad=0.05, fraction=0.046)
        cbar.ax.set_yticklabels(ticklabels)
        ax_.set(title=title, xlabel=r"a, $\mu$m", ylabel=r"d, $\mu$m")
        ax_.set_box_aspect(1)
        return plot

    # --- TRUE: amorphous ---
    _tric(
        ax[0, 0],
        d_am,
        "R",
        levels=np.linspace(0, 1, 50),
        clim=(0.0, 1.0),
        ticks=[0, 0.5, 1],
        ticklabels=["0.0", "0.5", "1"],
        title=r"$R_{am}$",
    )
    _tric(
        ax[1, 0],
        d_am,
        "phi_R",
        levels=np.linspace(0, 2 * np.pi, 50),
        clim=(0.0, 2 * np.pi),
        ticks=[0, np.pi, 2 * np.pi],
        ticklabels=["0", r"$\pi$", r"$2\pi$"],
        title=r"$\Delta\varphi_{am}$",
    )

    # --- TRUE: crystal ---
    _tric(
        ax[0, 1],
        d_cr,
        "R",
        levels=np.linspace(0, 1, 50),
        clim=(0.0, 1.0),
        ticks=[0, 0.5, 1],
        ticklabels=["0.0", "0.5", "1"],
        title=r"$R_{cr}$",
    )
    _tric(
        ax[1, 1],
        d_cr,
        "phi_R",
        levels=np.linspace(0, 2 * np.pi, 50),
        clim=(0.0, 2 * np.pi),
        ticks=[0, np.pi, 2 * np.pi],
        ticklabels=["0", r"$\pi$", r"$2\pi$"],
        title=r"$\Delta\varphi_{cr}$",
    )

    # --- PRED: amorphous surrogate ---
    _tric(
        ax[0, 2],
        data_pred,
        "R_0",
        levels=np.linspace(0, 1, 50),
        clim=(0.0, 1.0),
        ticks=[0, 0.5, 1],
        ticklabels=["0.0", "0.5", "1"],
        title=r"$R_{am}^{pred}$",
    )
    _tric(
        ax[1, 2],
        data_pred,
        "phi_R_0",
        levels=np.linspace(0, 2 * np.pi, 50),
        clim=(0.0, 2 * np.pi),
        ticks=[0, np.pi, 2 * np.pi],
        ticklabels=["0", r"$\pi$", r"$2\pi$"],
        title=r"$\Delta\varphi_{am}^{pred}$",
    )

    # --- PRED: crystal surrogate ---
    _tric(
        ax[0, 3],
        data_pred,
        "R_1",
        levels=np.linspace(0, 1, 50),
        clim=(0.0, 1.0),
        ticks=[0, 0.5, 1],
        ticklabels=["0.0", "0.5", "1"],
        title=r"$R_{cr}^{pred}$",
    )
    _tric(
        ax[1, 3],
        data_pred,
        "phi_R_1",
        levels=np.linspace(0, 2 * np.pi, 50),
        clim=(0.0, 2 * np.pi),
        ticks=[0, np.pi, 2 * np.pi],
        ticklabels=["0", r"$\pi$", r"$2\pi$"],
        title=r"$\Delta\varphi_{cr}^{pred}$",
    )

    # лимиты как в старом ноутбуке (в мкм)
    for r in range(2):
        for c in range(4):
            _apply_axis_custom(ax[r, c], xlim=xlim_c, ylim=ylim_c)

    # красные точки (в мкм)
    for r in range(2):
        for c in range(4):
            ax[r, c].plot(dots_um[0], dots_um[1], ".r", markersize=1)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    return fig

