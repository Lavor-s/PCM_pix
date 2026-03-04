from __future__ import annotations

"""
materials.py — графики по материалам (n/k от длины волны).

Переносим 1-в-1 первый график из `3rd art_PCM_bagel_2025.ipynb` (пример 2×2):
- слева полный диапазон λ
- справа zoom на 1400..1600 nm
- выделение диапазона на левом графике + ConnectionPatch стрелки
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import patches


@dataclass(frozen=True)
class MaterialsNK:
    """Минимальный контейнер для построения графика n/k."""

    sb2se3_am: Any
    sb2se3_cr: Any
    gst_am: Any
    gst_cr: Any


def plot_materials_nk_zoom(
    m: MaterialsNK,
    *,
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
    dpi: int | None = None,
    title: str = r"$Sb_2Se_3$ vs $Ge_2Sb_2Te_5$",
) -> plt.Figure:
    """
    Строит график материалов n/k как в старом ноутбуке.

    Важно про единицы:
    - Sb2Se3 таблицы у тебя в nm (`sb2se3_*.wl`)
    - GST в файлах похоже в μm, поэтому в ноутбуке использовалось `gst.wl * 1000` -> nm.
      Здесь сохраняем это поведение 1-в-1.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    lw = 0.8
    lw2 = 0.5

    sb2se3_am, sb2se3_cr = m.sb2se3_am, m.sb2se3_cr
    gst_am, gst_cr = m.gst_am, m.gst_cr

    # левый столбец — широкий диапазон
    axs[0, 0].plot(sb2se3_am.wl, sb2se3_am.n, "b-", linewidth=lw, markersize=5)
    axs[0, 0].plot(sb2se3_cr.wl, sb2se3_cr.n, "r-", linewidth=lw, markersize=5)
    axs[1, 0].plot(sb2se3_am.wl, sb2se3_am.k, "b-", linewidth=lw, markersize=5)
    axs[1, 0].plot(sb2se3_cr.wl, sb2se3_cr.k, "r-", linewidth=lw, markersize=5)

    axs[0, 0].plot(gst_am.wl * 1000, gst_am.n, "b--", linewidth=lw2, markersize=5)
    axs[0, 0].plot(gst_cr.wl * 1000, gst_cr.n, "r--", linewidth=lw2, markersize=5)
    axs[1, 0].plot(gst_am.wl * 1000, gst_am.k, "b--", linewidth=lw2, markersize=5)
    axs[1, 0].plot(gst_cr.wl * 1000, gst_cr.k, "r--", linewidth=lw2, markersize=5)

    # правый столбец — zoom
    axs[0, 1].plot(sb2se3_am.wl, sb2se3_am.n, "b-", linewidth=1, markersize=5)
    axs[0, 1].plot(sb2se3_cr.wl, sb2se3_cr.n, "r-", linewidth=1, markersize=5)
    axs[1, 1].plot(sb2se3_am.wl, sb2se3_am.k, "b-", linewidth=1, markersize=5)
    axs[1, 1].plot(sb2se3_cr.wl, sb2se3_cr.k, "r-", linewidth=1, markersize=5)

    axs[0, 1].plot(gst_am.wl * 1000, gst_am.n, "b--", linewidth=lw2, markersize=5)
    axs[0, 1].plot(gst_cr.wl * 1000, gst_cr.n, "r--", linewidth=lw2, markersize=5)
    axs[1, 1].plot(gst_am.wl * 1000, gst_am.k, "b--", linewidth=lw2, markersize=5)
    axs[1, 1].plot(gst_cr.wl * 1000, gst_cr.k, "r--", linewidth=lw2, markersize=5)

    axs[0, 0].set(title="Refractive index", xlabel=r"$\lambda$, nm", ylabel="n", xlim=[0, 2000], ylim=[2, 8])
    axs[1, 0].set(title="Extinction index", xlabel=r"$\lambda$, nm", ylabel="k", xlim=[0, 2000], ylim=[0, 4])
    axs[0, 1].set(title="Refractive index", xlabel=r"$\lambda$, nm", ylabel="n", xlim=[1400, 1600], ylim=[2, 8])
    axs[1, 1].set(title="Extinction index", xlabel=r"$\lambda$, nm", ylabel="k", xlim=[1400, 1600], ylim=[-0.1, 2])

    fig.suptitle(title)

    # выделение диапазона + вертикали на левых графиках
    for ax in (axs[0, 0], axs[1, 0]):
        ax.plot([1400, 1400], [-50, 50], "k-", linewidth=0.3, markersize=5)
        ax.plot([1600, 1600], [-50, 50], "k-", linewidth=0.3, markersize=5)
        ax.fill_between([1400, 1600], [-50, -50], [50, 50], color="k", alpha=0.1)

    # стрелки связи (как в ноутбуке)
    arrow = patches.ConnectionPatch(
        [1500, 7],
        [1500, 7],
        coordsA=axs[0, 0].transData,
        coordsB=axs[0, 1].transData,
        color="black",
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.5,
    )
    fig.patches.append(arrow)

    arrow2 = patches.ConnectionPatch(
        [1500, 3.5],
        [1500, 1.75],
        coordsA=axs[1, 0].transData,
        coordsB=axs[1, 1].transData,
        color="black",
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.5,
    )
    fig.patches.append(arrow2)

    axs[0, 0].plot([1500], [7], "k.", markersize=2)
    axs[1, 0].plot([1500], [3.5], "k.", markersize=2)

    for ax in axs.ravel():
        ax.grid(True)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    return fig

