from __future__ import annotations

"""
style.py — единый стиль графиков.

В старом большом ноутбуке стиль (plt.rc/rcParams) задавался кусками перед каждым графиком.
В рефакторинге держим стиль в одном месте, чтобы:
- все картинки выглядели одинаково
- можно было менять dpi/шрифты/размеры централизованно
"""

from typing import Any

import matplotlib as mpl


def apply_style(cfg: dict[str, Any] | None = None, *, preset: str = "paper") -> None:
    """
    Применяет единые rcParams.

    preset:
    - "paper": плотные картинки (dpi выше), удобны для отчётов
    - "screen": чуть крупнее шрифт, комфортно смотреть в ноутбуке
    """
    cfg = cfg or {}

    if preset not in ("paper", "screen"):
        raise ValueError(f"Unknown preset={preset!r}, expected 'paper' or 'screen'")

    # базовые размеры как в 3rd art (SMALL/MEDIUM/BIGGER)
    # Поддерживаем legacy-ключи из старых ноутбуков: SMALL_SIZE/MEDIUM_SIZE/BIGGER_SIZE.
    if preset == "paper":
        default_small, default_medium, default_big = 8, 10, 12
    else:
        default_small, default_medium, default_big = 10, 12, 14

    small = int(cfg.get("SMALL_SIZE", default_small))
    medium = int(cfg.get("MEDIUM_SIZE", default_medium))
    big = int(cfg.get("BIGGER_SIZE", default_big))

    mpl.rcParams.update(
        {
            "font.size": small,
            "axes.titlesize": small,
            "axes.labelsize": small,
            "xtick.labelsize": small,
            "ytick.labelsize": small,
            "legend.fontsize": small,
            "figure.titlesize": big,
            # мелкие улучшения читаемости
            "axes.grid": False,  # сетку лучше включать явно на графиках
            "savefig.bbox": "tight",
        }
    )


def get_plot_size(
    cfg: dict[str, Any] | None,
    plot_name: str,
    *,
    default_figsize: tuple[float, float],
    default_dpi: int | None,
) -> tuple[tuple[float, float], int | None]:
    """
    Достаёт figsize/dpi для конкретного графика из CFG.

    Поддерживаем форматы (в порядке приоритета):
    1) Legacy (максимально похоже на старые ноутбуки, без префиксов):
       - <plot_name>_figsize: [w, h]
       - <plot_name>_dpi: 300

    2) Рекомендуемый (вложенный словарь, удобно хранить в config.json):
       CFG["plots"][plot_name] = {"figsize": [w, h], "dpi": 300}

    3) Back-compat (старый формат из первых итераций рефакторинга):
       - plot_<plot_name>_figsize: [w, h]
       - plot_<plot_name>_dpi: 300
    """
    cfg = cfg or {}
    plot_name = str(plot_name)

    # nested format (recommended)
    nested = (cfg.get("plots", {}) or {}).get(plot_name, {}) if isinstance(cfg.get("plots", {}), dict) else {}
    nested_figsize = nested.get("figsize", None) if isinstance(nested, dict) else None
    nested_dpi = nested.get("dpi", None) if isinstance(nested, dict) else None

    # legacy flat format (no prefix)
    legacy_figsize = cfg.get(f"{plot_name}_figsize", None)
    legacy_dpi = cfg.get(f"{plot_name}_dpi", None)

    # back-compat flat format (with "plot_" prefix)
    flat_figsize = cfg.get(f"plot_{plot_name}_figsize", None)
    flat_dpi = cfg.get(f"plot_{plot_name}_dpi", None)

    raw_figsize = (
        legacy_figsize
        if legacy_figsize is not None
        else nested_figsize
        if nested_figsize is not None
        else flat_figsize
    )
    raw_dpi = (
        legacy_dpi
        if legacy_dpi is not None
        else nested_dpi
        if nested_dpi is not None
        else flat_dpi
    )

    figsize = default_figsize
    if raw_figsize is not None:
        if isinstance(raw_figsize, (list, tuple)) and len(raw_figsize) == 2:
            figsize = (float(raw_figsize[0]), float(raw_figsize[1]))
        elif isinstance(raw_figsize, str):
            # формат "7,3" или "7 3"
            parts = raw_figsize.replace(",", " ").split()
            if len(parts) == 2:
                figsize = (float(parts[0]), float(parts[1]))

    dpi = default_dpi
    if raw_dpi is not None:
        try:
            dpi = int(raw_dpi)
        except Exception:
            dpi = default_dpi

    return figsize, dpi

