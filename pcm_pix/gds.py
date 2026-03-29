from __future__ import annotations

"""
gds.py — экспорт GDS для фабрикации (перенос из `3rd art_PCM_bagel_2025.ipynb`).

Этот модуль специально отделён от графиков:
- в `plots.ipynb` мы строим картинки
- здесь мы строим *геометрию* и пишем артефакты в `outputs/<run>/gds/`

Переносим базовый сценарий из блока "##FOR FABRICATION" (ячейка ~238) + подготовку
`edge/Number_x` (ячейка ~234):
- массивы `a, d, b` берутся из оптимизированного `pos` (a/d/b в метрах)
- строится набор вертикальных "полос" по X, каждая полоса заполнена решёткой колец
  с периодом `a[i]`, радиусом `d[i]/2`, внутренним радиусом `b[i]/2`
- ширина каждой полосы ≈ `l` (по умолчанию 20 µm), количество колонок `Number_x[i]`
  вычисляется как int((edge[i+1]-edge[i])/a[i]) с подстройкой границ edge
- высота шаблона задаётся `L` (в 3rd art: `L = 11*l`)

Важно:
- `gdspy` считается опциональной зависимостью. Если её нет — выдаём понятную ошибку.
- Опционально: вторая ячейка (по умолчанию `CELL_1`) с покрывающим кольцом Al2O3:
  внешний радиус +r_m, внутренний полурадиус max(0, b_in - r_m), как в 3rd art.
- Нумерация GDS-layer: `layer_all_zero=True` (по умолчанию) — обе ячейки: layer=0, datatype=0;
  иначе layer растёт на 1 каждые `layer_step_n` дисков (один счётчик по сетке колец Si).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class GdspyNotInstalled(RuntimeError):
    pass


def _import_gdspy():
    try:
        import gdspy  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GdspyNotInstalled(
            "Package `gdspy` is required for GDS export. "
            "Install it (e.g. `pip install gdspy`) and re-run."
        ) from e
    return gdspy


@dataclass(frozen=True)
class GDSFabCfg:
    """Настройки экспорта GDS (минимальный набор для перенесённого сценария)."""

    name: str = "FOR_FAB_110mum"
    multipl: float = 1e6  # масштаб: метры -> микрометры (как в ноутбуке)
    margin_m: float = 50e-9  # (L - margin) / period
    l_m: float = 20e-6  # ширина каждой полосы по X
    L_m: float | None = None  # высота по Y; если None -> 11*l
    ring_limit: int = 1000  # устар.; раньше: lum + шаг num_lay; см. layer_all_zero / layer_step_n
    layer_mode: str = "fab"  # устар. для нумерации слоёв; см. layer_all_zero / layer_step_n
    layer0: tuple[int, int] = (0, 0)  # при layer_all_zero=False: layer = disk//step_n; datatype — для Si и Al2O3
    tolerance: float = 0.001
    # Нумерация GDS-layer для обеих ячеек (Si и Al2O3):
    # - layer_all_zero=True (по умолчанию): всегда layer=0, datatype=0;
    # - layer_all_zero=False: layer += 1 каждые layer_step_n дисков (один счётчик на позицию в сетке).
    layer_all_zero: bool = True
    layer_step_n: int = 1000
    # Покрывающий слой Al2O3 (вторая ячейка в той же библиотеке)
    al2o3_cover_enable: bool = False
    al2o3_r_m: float = 100e-9  # +r к внешнему радиусу, −r к внутреннему (половина b)
    al2o3_cell_name: str = "CELL_1"


def compute_edge_and_number_x(a_m: np.ndarray, *, l_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Перенос логики из ячейки ~234 (edge/Number_x) 1-в-1.

    a_m: периоды по полосам (метры), длины Nn.
    l_m: целевая ширина каждой полосы по X.

    Возвращает:
    - edge: длины Nn+1 (границы полос)
    - number_x: длины Nn (число колонок в каждой полосе)
    """
    a_m = np.array(a_m, dtype=float).ravel()
    Nn = len(a_m)

    edge = [l_m * i for i in range(Nn + 1)]

    def num_reroll(a: float, x_start: float, x_fin: float) -> int:
        return int((x_fin - x_start) / a)

    number_x = np.array([num_reroll(a_m[i], edge[i], edge[i + 1]) for i in range(Nn)], dtype=int)

    # подстройка границ как в ноутбуке
    for i in range(Nn - 1):
        d_edge = (edge[i + 1] - edge[i]) - a_m[i] * number_x[i]
        if d_edge < a_m[i] / 2:
            edge[i + 1] -= (d_edge) * 0.999
        else:
            edge[i + 1] += (a_m[i] - d_edge) * 1.001

        number_x[i] = num_reroll(a_m[i], edge[i], edge[i + 1])
        number_x[i + 1] = num_reroll(a_m[i + 1], edge[i + 1], edge[i + 2])

    return np.array(edge, dtype=float), np.array(number_x, dtype=int)


def export_fabrication_gds(
    *,
    a_m: np.ndarray,
    d_m: np.ndarray,
    b_m: np.ndarray,
    out_dir: str | Path,
    cfg: GDSFabCfg,
    edge_m: np.ndarray | None = None,
    number_x: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """
    Экспортирует:
    - `<name>.gds`
    - `<name>.txt` (лог/таблица a/d/b)

    Возвращает (gds_path, txt_path).
    """
    gdspy = _import_gdspy()

    a_m = np.array(a_m, dtype=float).ravel()
    d_m = np.array(d_m, dtype=float).ravel()
    b_m = np.array(b_m, dtype=float).ravel()

    if not (len(a_m) == len(d_m) == len(b_m)):
        raise ValueError("a/d/b must have same length")

    Nn = len(a_m)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L_m = float(cfg.L_m if cfg.L_m is not None else 11.0 * cfg.l_m)

    if edge_m is None or number_x is None:
        edge_m, number_x = compute_edge_and_number_x(a_m, l_m=cfg.l_m)
    edge_m = np.array(edge_m, dtype=float).ravel()
    number_x = np.array(number_x, dtype=int).ravel()

    if len(edge_m) != Nn + 1:
        raise ValueError(f"edge_m must have length Nn+1={Nn+1}, got {len(edge_m)}")
    if len(number_x) != Nn:
        raise ValueError(f"number_x must have length Nn={Nn}, got {len(number_x)}")

    step_n = int(cfg.layer_step_n)
    if not cfg.layer_all_zero and step_n < 1:
        raise ValueError(f"layer_step_n must be >= 1, got {step_n}")

    # --- init library ---
    gdspy.current_library = gdspy.GdsLibrary()
    lib = gdspy.GdsLibrary()
    cell_si = lib.new_cell("CELL_0")
    cell_al2o3 = lib.new_cell(cfg.al2o3_cell_name) if cfg.al2o3_cover_enable else None

    r_al = float(cfg.al2o3_r_m)
    disk_i = 0

    for i in range(Nn):
        rad = float(d_m[i]) / 2.0
        per = float(a_m[i])
        b_in = float(b_m[i]) / 2.0

        x0 = float(edge_m[i])
        y0 = 0.0

        j_col = int(number_x[i])
        k_col = int((L_m - cfg.margin_m) / per)

        for k in range(k_col):
            for j in range(j_col):
                x_center = x0 + per * j + per / 2
                y_center = y0 + per * k + per / 2

                if cfg.layer_all_zero:
                    layer_si = {"layer": 0, "datatype": 0}
                    layer_al = {"layer": 0, "datatype": 0}
                else:
                    lay = disk_i // step_n
                    dt = int(cfg.layer0[1])
                    layer_si = {"layer": int(lay), "datatype": dt}
                    layer_al = {"layer": int(lay), "datatype": dt}

                cell_si.add(
                    gdspy.Round(
                        (x_center * cfg.multipl, y_center * cfg.multipl),
                        rad * cfg.multipl,
                        inner_radius=b_in * cfg.multipl,
                        initial_angle=-np.pi,
                        final_angle=np.pi,
                        tolerance=cfg.tolerance,
                        **layer_si,
                    )
                )

                if cell_al2o3 is not None:
                    rad_al = rad + r_al
                    b_in_al = max(0.0, b_in - r_al)
                    cell_al2o3.add(
                        gdspy.Round(
                            (x_center * cfg.multipl, y_center * cfg.multipl),
                            rad_al * cfg.multipl,
                            inner_radius=b_in_al * cfg.multipl,
                            initial_angle=-np.pi,
                            final_angle=np.pi,
                            tolerance=cfg.tolerance,
                            **layer_al,
                        )
                    )

                disk_i += 1

    # --- write files ---
    gds_path = out_dir / f"{cfg.name}.gds"
    txt_path = out_dir / f"{cfg.name}.txt"

    # .gds
    lib.write_gds(str(gds_path))

    # .txt log
    lines = []
    lines.append("Текстовое описание одноименного файла .gds\n")
    lines.append(f"Дата создания {datetime.now().date()}\n")
    lines.append(f"Время создания {datetime.now().time()}\n")
    lines.append("a, nm\td, nm\tb, nm\n")
    for i in range(Nn):
        A = round(a_m[i] * 1e9, 0)
        D = round(d_m[i] * 1e9, 0)
        B = round(b_m[i] * 1e9, 0)
        lines.append(f"{A}\t{D}\t{B}\n")

    if cfg.al2o3_cover_enable:
        lines.append(
            f"# Al2O3 cover: cell={cfg.al2o3_cell_name!r}, r_m={cfg.al2o3_r_m}\n"
        )
    lines.append(
        f"# layers: all_zero={cfg.layer_all_zero}, step_n={cfg.layer_step_n}\n"
    )

    if meta:
        lines.append("\n# meta\n")
        for k, v in meta.items():
            lines.append(f"# {k}: {v}\n")

    txt_path.write_text("".join(lines), encoding="utf-8")
    return gds_path, txt_path

