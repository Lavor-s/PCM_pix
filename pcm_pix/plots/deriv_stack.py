from __future__ import annotations

"""
deriv_stack.py — перенос "всей подноготной" для графика из `3rd art_PCM_bagel_2025.ipynb`
(блок вокруг построения `ZXY_array`, `conv_corr_first/conv_corr_sec`, и финального графика
с наложениями).

Что делает этот модуль:
- Загружает внешние карты (ZEMAX-подобные таблицы) из txt (utf16 + skiprows=15)
- Считает "корреляционные" карты через FFT: `conv_corr_first` и `conv_corr_sec`
  (перенос 1-в-1 исходных формул)
- Формирует `ZXY_array` (для каждого уровня i и для каждой панели j=0/1)
- Вычисляет теоретические/симуляционные экстремумы и ошибки (d1p/d3p и т.д.)
- Строит итоговый график 1×2 с множеством наложений (слоёв), которые можно
  включать/выключать через flags

Важно про расширяемость:
- график "ещё не закончен" и будет пополняться: поэтому ключевая идея —
  расчёт отделён от рисования, а рисование — от отдельных overlays.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# I/O helpers (ZEMAX-like tables)
# ----------------------------


def load_txt_table_utf16(path: str | Path, *, skiprows: int = 15) -> np.ndarray:
    """
    Загружает таблицу из txt (как в старом ноутбуке через pandas.read_csv(..., encoding='utf16')).

    Формат: табличные числа, разделитель обычно TAB.
    Возвращаем 2D numpy array.
    """
    path = Path(path)
    # genfromtxt устойчив к "рваным" строкам, но может дать NaN на мусоре — это ок.
    arr = np.genfromtxt(
        path,
        delimiter="\t",
        skip_header=int(skiprows),
        dtype=float,
        encoding="utf-16",
    )
    # если файл "одна колонка" — приведём к 2D
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def zemax_to_line(
    x_or_y: int,
    val_cross: float,
    N_dots: int,
    N_fin: int,
    x_width: float,
    y_width: float,
    Z_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Перенос 1-в-1 `ZEMAX_to_line` из 3rd art.

    x_or_y:
    - 0: вернуть сечение Z[:, ind] как функция x_vals
    - 1: вернуть сечение Z[ind, :] как функция y_vals

    x_width/y_width — в тех единицах, как в старом ноутбуке (обычно µm или mm — зависит от файла).
    """
    x_min = -x_width / 2 * N_fin / N_dots
    x_max = x_width / 2 * N_fin / N_dots
    y_min = -y_width / 2 * N_fin / N_dots
    y_max = y_width / 2 * N_fin / N_dots
    x_vals = np.linspace(x_min, x_max, N_fin)
    y_vals = np.linspace(y_min, y_max, N_fin)

    Z = np.array(Z_data)[
        N_dots // 2 - N_fin // 2 : -(N_dots // 2 - N_fin // 2),
        N_dots // 2 - N_fin // 2 : -(N_dots // 2 - N_fin // 2),
    ]
    Z = Z / np.max(Z)

    if int(x_or_y) == 0:
        ind, _ = min(enumerate(y_vals), key=lambda x: abs(x[1] - val_cross))
        return x_vals, Z[:, ind]
    ind, _ = min(enumerate(x_vals), key=lambda x: abs(x[1] - val_cross))
    return y_vals, Z[ind, :]


# ----------------------------
# "LUM" correction (from old notebook)
# ----------------------------


@dataclass(frozen=True)
class LumTables:
    """
    Табличные поправки из "люма".
    В 3rd art эти массивы несколько раз переопределялись; здесь держим финальную
    версию (для конусов), но даём возможность переопределить через CFG.
    """

    R_lum_cr: np.ndarray
    R_lum_am: np.ndarray
    phi_lum_cr: np.ndarray
    phi_lum_am: np.ndarray


def default_lum_tables_cones() -> LumTables:
    # финальные значения из ячейки 148 (после переопределения "для конусов")
    return LumTables(
        R_lum_cr=np.array([0.85, 0.83, 0.71, 0.51, 0.10, 0.00, 0.32, 0.40, 0.65, 0.93, 0.97], dtype=float),
        R_lum_am=np.array([0.97, 1.00, 0.97, 0.53, 0.09, 0.02, 0.15, 0.23, 0.06, 0.33, 0.77], dtype=float),
        phi_lum_cr=np.array([2.1, 2.2, 2.2, 2.0, 2.0, -0.2, -1.4, -1.1, -1.6, -1.5, -1.3], dtype=float),
        phi_lum_am=np.array([-0.6, -0.5, -0.8, -1.0, -0.8, -0.4, -1.0, -1.4, -2.4, -2.1, -1.8], dtype=float),
    )


def R_lum_fun(u: float, v: float, *, state: str, L: float, S: float, tables: LumTables) -> float:
    """Перенос 1-в-1 `R_lum_fun`."""
    if abs(u) >= S / 2 or abs(v) >= S / 2:
        return 0.0
    # номер пикселя с 0 до 10
    N = int((u - L / 2) // L + 1 + 5)
    if state == "cr":
        return float(tables.R_lum_cr[N])
    return float(tables.R_lum_am[N])


def phi_lum_fun(u: float, v: float, *, state: str, L: float, S: float, tables: LumTables) -> float:
    """Перенос 1-в-1 `phi_lum_fun`."""
    if abs(u) >= S / 2 or abs(v) >= S / 2:
        return 0.0
    # номер пикселя с 0 до 10
    N = int((u - L / 2) // L + 1 + 5)
    if state == "cr":
        return float(tables.phi_lum_cr[N])
    return float(tables.phi_lum_am[N])


# ----------------------------
# Core model: H_func + conv_corr_* (FFT)
# ----------------------------


def H_func_gauss(x: float, y: float, *, s: float, phi: float = 0.0, shift_x: float = 0.0, shift_y: float = 0.0) -> float:
    """
    Та версия `H_func`, которая реально используется в блоке conv_corr_* в 3rd art:
    простая гауссиана exp(-(x^2+y^2)/s^2).

    Параметры phi/shift_* оставлены для будущих расширений (как в исходной сигнатуре).
    """
    _ = (phi, shift_x, shift_y)
    return float(np.exp(-((x**2 + y**2) / s / s)))


def conv_corr_first(
    *,
    s: float,
    N: int,
    f_gl: float,
    wl_gl: float,
    L_gl: float,
    S_gl: float,
    tables: LumTables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Перенос 1-в-1 `conv_corr_first` (ячейка 162):
    - строим тестовую "картинку" из H_func на сетке [-20..20] (в условных единицах)
    - FFT -> спектр -> домножаем на lum-коррекции (cr)
    - IFFT -> RES, плюс V_x/V_y (оси)
    """
    ref_im = np.zeros([N, N])
    test_im = np.zeros([N, N])

    X_ar = np.linspace(-20, 20, N)
    Y_ar = np.linspace(-20, 20, N)

    for i in range(N):
        for j in range(N):
            test_im[i, j] = H_func_gauss(X_ar[j], Y_ar[i], s=s, phi=np.pi / 2)
            # ref_im не используется в текущей версии (как и в 3rd art)
            _ = ref_im

    RES = np.fft.fftshift(np.fft.fft2(test_im))
    U_x = np.fft.fftshift(np.fft.fftfreq(np.shape(RES)[0], d=1e-3 * abs(X_ar[1] - X_ar[0]))) * f_gl * wl_gl
    U_y = np.fft.fftshift(np.fft.fftfreq(np.shape(RES)[1], d=1e-3 * abs(Y_ar[1] - Y_ar[0]))) * f_gl * wl_gl

    Nu, Nv = np.shape(RES)
    for i in range(Nu):
        for j in range(Nv):
            RES[i, j] *= R_lum_fun(U_x[i], U_y[j], state="cr", L=L_gl, S=S_gl, tables=tables) * np.exp(
                1j * phi_lum_fun(U_x[i], U_y[j], state="cr", L=L_gl, S=S_gl, tables=tables)
            )

    RES = np.fft.ifft2(RES)
    V_x = np.fft.ifftshift(np.fft.fftfreq(np.shape(RES)[0], d=abs(U_x[1] - U_x[0]) / f_gl / wl_gl))
    V_y = np.fft.ifftshift(np.fft.fftfreq(np.shape(RES)[1], d=abs(U_y[1] - U_y[0]) / f_gl / wl_gl))
    return RES, V_x, V_y


def conv_corr_sec(
    *,
    s: float,
    N: int,
    f_gl: float,
    wl_gl: float,
    L_gl: float,
    S_gl: float,
    tables: LumTables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Перенос 1-в-1 `conv_corr_sec` (ячейка 162):
    то же самое, но умножение на lum-коррекции делается в режиме state='am'.
    """
    ref_im = np.zeros([N, N])
    test_im = np.zeros([N, N])

    X_ar = np.linspace(-20, 20, N)
    Y_ar = np.linspace(-20, 20, N)

    for i in range(N):
        for j in range(N):
            test_im[i, j] = H_func_gauss(X_ar[j], Y_ar[i], s=s, phi=np.pi / 2)
            _ = ref_im

    RES = np.fft.fftshift(np.fft.fft2(test_im))
    U_x = np.fft.fftshift(np.fft.fftfreq(np.shape(RES)[0], d=1e-3 * abs(X_ar[1] - X_ar[0]))) * f_gl * wl_gl
    U_y = np.fft.fftshift(np.fft.fftfreq(np.shape(RES)[1], d=1e-3 * abs(Y_ar[1] - Y_ar[0]))) * f_gl * wl_gl

    Nu, Nv = np.shape(RES)
    for i in range(Nu):
        for j in range(Nv):
            RES[i, j] *= R_lum_fun(U_x[i], U_y[j], state="am", L=L_gl, S=S_gl, tables=tables) * np.exp(
                1j * phi_lum_fun(U_x[i], U_y[j], state="am", L=L_gl, S=S_gl, tables=tables)
            )

    RES = np.fft.ifft2(RES)
    V_x = np.fft.ifftshift(np.fft.fftfreq(np.shape(RES)[0], d=abs(U_x[1] - U_x[0]) / f_gl / wl_gl))
    V_y = np.fft.ifftshift(np.fft.fftfreq(np.shape(RES)[1], d=abs(U_y[1] - U_y[0]) / f_gl / wl_gl))
    return RES, V_x, V_y


# ----------------------------
# ZXY_array builder + extrema metrics
# ----------------------------


@dataclass(frozen=True)
class ZXYItem:
    panel_j: int
    level_i: int
    Z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    X_cut: np.ndarray
    Z_cut: np.ndarray


def build_s_list_from_alpha(alpha: np.ndarray, *, f_gl: float, wl_gl: float, S_gl: float) -> np.ndarray:
    """s_list = 2*f*wl/pi/(alpha*1e-3)/S (1-в-1)."""
    alpha = np.array(alpha, dtype=float).ravel()
    return 2 * f_gl * wl_gl / np.pi / (alpha * 1e-3) / S_gl


def build_zxy_array(
    *,
    alpha: np.ndarray,
    f_gl: float,
    wl_gl: float,
    L_gl: float,
    S_gl: float,
    conv_N: int = 500,
    tables: LumTables | None = None,
    y0_tol: float = 0.0,
) -> tuple[np.ndarray, list[ZXYItem]]:
    """
    Перенос блока (ячейка 167): строим ZXY_array для всех уровней alpha (через s_list).

    y0_tol:
    - 0.0: строго как в ноутбуке (element == 0)
    - >0: использовать |y|<=tol, если из-за float-шума не нашлось ровно нуля
    """
    tables = tables or default_lum_tables_cones()
    s_list = build_s_list_from_alpha(alpha, f_gl=f_gl, wl_gl=wl_gl, S_gl=S_gl)

    items: list[ZXYItem] = []
    for i in range(len(s_list)):
        for j in (0, 1):
            if j == 0:
                Z, X, Y = conv_corr_sec(s=float(s_list[i]), N=int(conv_N), f_gl=f_gl, wl_gl=wl_gl, L_gl=L_gl, S_gl=S_gl, tables=tables)
            else:
                Z, X, Y = conv_corr_first(s=float(s_list[i]), N=int(conv_N), f_gl=f_gl, wl_gl=wl_gl, L_gl=L_gl, S_gl=S_gl, tables=tables)

            # 1-в-1 перестановка как в ноутбуке
            X, Y = np.meshgrid(Y, X)
            Z = np.abs(Z.T)
            Z = Z / np.amax(Z)

            # Сечение по y=0
            X_cut_list: list[float] = []
            Z_cut_list: list[float] = []
            for row_index, row in enumerate(Y):
                for col_index, element in enumerate(row):
                    if y0_tol == 0.0:
                        ok = (element == 0)
                    else:
                        ok = (abs(float(element)) <= y0_tol)
                    if ok:
                        X_cut_list.append(float(X[row_index, col_index]))
                        Z_cut_list.append(float(Z[row_index, col_index]))

            X_cut = np.array(X_cut_list, dtype=float) * S_gl / wl_gl / f_gl
            Z_cut = np.array(Z_cut_list, dtype=float)

            items.append(ZXYItem(panel_j=int(j), level_i=int(i), Z=Z, X=X, Y=Y, X_cut=X_cut, Z_cut=Z_cut))

    return s_list, items


def min_by_cond(arr_y: np.ndarray, arr_x: np.ndarray, cond_func_x) -> float:
    arr_x = np.array(arr_x, dtype=float)
    arr_y = np.array(arr_y, dtype=float)
    ind = np.argmin(arr_y[cond_func_x(arr_x)])
    ind_res = np.where(arr_y == arr_y[cond_func_x(arr_x)][ind])[0][0]
    return float(arr_x[ind_res])


def max_by_cond(arr_y: np.ndarray, arr_x: np.ndarray, cond_func_x) -> float:
    arr_x = np.array(arr_x, dtype=float)
    arr_y = np.array(arr_y, dtype=float)
    ind = np.argmax(arr_y[cond_func_x(arr_x)])
    ind_res = np.where(arr_y == arr_y[cond_func_x(arr_x)][ind])[0][0]
    return float(arr_x[ind_res])


@dataclass(frozen=True)
class ExtremaMetrics:
    MAX_1st_x_pos_th: np.ndarray
    MAX_1st_x_neg_th: np.ndarray
    MAX_2nd_x_pos_th: np.ndarray
    MAX_2nd_x_neg_th: np.ndarray

    MAX_1st_x_pos_sim: np.ndarray
    MAX_1st_x_neg_sim: np.ndarray
    MAX_2nd_x_pos_sim: np.ndarray
    MAX_2nd_x_neg_sim: np.ndarray

    d1p: np.ndarray
    d1n: np.ndarray
    d3p: np.ndarray
    d3n: np.ndarray


@dataclass(frozen=True)
class NeonOverlayData:
    """Данные для neon-наложения (экспериментальные кривые) уже в координатах графика."""

    x_am: np.ndarray
    z_am: np.ndarray
    x_cr: np.ndarray
    z_cr: np.ndarray


@dataclass(frozen=True)
class DerivStackData:
    """
    Полный набор расчётных данных для графика.

    Это главный объект, который позволяет разделить:
    - тяжёлый расчёт (FFT/поиск экстремумов) -> DerivStackData
    - лёгкую отрисовку (много вариантов слоёв/стилей) -> plot_*_from_data
    """

    alpha: np.ndarray
    s_list: np.ndarray
    items: list[ZXYItem]
    metrics: ExtremaMetrics
    neon: NeonOverlayData | None
    cfg_meta: dict[str, Any]


def compute_extrema_metrics(
    *,
    s_list: np.ndarray,
    items: list[ZXYItem],
    S_gl: float,
    wl_gl: float,
    f_gl: float,
) -> ExtremaMetrics:
    """
    Перенос логики экстремумов (ячейки 173..):
    - теоретические позиции (через s_list и формулы)
    - симуляционные позиции из Z_cut(X_cut)
    - относительные ошибки
    """
    MAX_1st_x_pos_th = []
    MAX_1st_x_neg_th = []
    MAX_2nd_x_pos_th = []
    MAX_2nd_x_neg_th = []

    for i in range(len(s_list)):
        p1 = float(s_list[i] / np.sqrt(2) * 1e-3 * S_gl / wl_gl / f_gl)
        p2 = float(s_list[i] / np.sqrt(2 / 3) * 1e-3 * S_gl / wl_gl / f_gl)
        MAX_1st_x_pos_th.append(p1)
        MAX_1st_x_neg_th.append(-p1)
        MAX_2nd_x_pos_th.append(p2)
        MAX_2nd_x_neg_th.append(-p2)

    MAX_1st_x_pos_sim = []
    MAX_1st_x_neg_sim = []
    MAX_2nd_x_pos_sim = []
    MAX_2nd_x_neg_sim = []

    # как в ноутбуке: для j==1 берём 1-й максимум, для j==0 — 2-й максимум после первой зоны
    for it in items:
        j, i = it.panel_j, it.level_i
        if j == 1:
            MAX_1st_x_pos_sim.append(max_by_cond(it.Z_cut, it.X_cut, lambda x: x > 0))
            MAX_1st_x_neg_sim.append(max_by_cond(it.Z_cut, it.X_cut, lambda x: x < 0))
        else:
            # первый максимум здесь не используется; используем MAX_2nd по условиям из ноутбука
            MAX_2nd_x_pos_sim.append(max_by_cond(it.Z_cut, it.X_cut, lambda x: x > float(MAX_1st_x_pos_th[i])))
            MAX_2nd_x_neg_sim.append(max_by_cond(it.Z_cut, it.X_cut, lambda x: x < float(MAX_1st_x_neg_th[i])))

    MAX_1st_x_pos_th = np.array(MAX_1st_x_pos_th, dtype=float)
    MAX_1st_x_neg_th = np.array(MAX_1st_x_neg_th, dtype=float)
    MAX_2nd_x_pos_th = np.array(MAX_2nd_x_pos_th, dtype=float)
    MAX_2nd_x_neg_th = np.array(MAX_2nd_x_neg_th, dtype=float)

    MAX_1st_x_pos_sim = np.array(MAX_1st_x_pos_sim, dtype=float)
    MAX_1st_x_neg_sim = np.array(MAX_1st_x_neg_sim, dtype=float)
    MAX_2nd_x_pos_sim = np.array(MAX_2nd_x_pos_sim, dtype=float)
    MAX_2nd_x_neg_sim = np.array(MAX_2nd_x_neg_sim, dtype=float)

    d1p = np.abs(MAX_1st_x_pos_th - MAX_1st_x_pos_sim) / np.abs(MAX_1st_x_pos_th)
    d1n = np.abs(MAX_1st_x_neg_th - MAX_1st_x_neg_sim) / np.abs(MAX_1st_x_neg_th)
    d3p = np.abs(MAX_2nd_x_pos_th - MAX_2nd_x_pos_sim) / np.abs(MAX_2nd_x_pos_th)
    d3n = np.abs(MAX_2nd_x_neg_th - MAX_2nd_x_neg_sim) / np.abs(MAX_2nd_x_neg_th)

    return ExtremaMetrics(
        MAX_1st_x_pos_th=MAX_1st_x_pos_th,
        MAX_1st_x_neg_th=MAX_1st_x_neg_th,
        MAX_2nd_x_pos_th=MAX_2nd_x_pos_th,
        MAX_2nd_x_neg_th=MAX_2nd_x_neg_th,
        MAX_1st_x_pos_sim=MAX_1st_x_pos_sim,
        MAX_1st_x_neg_sim=MAX_1st_x_neg_sim,
        MAX_2nd_x_pos_sim=MAX_2nd_x_pos_sim,
        MAX_2nd_x_neg_sim=MAX_2nd_x_neg_sim,
        d1p=d1p,
        d1n=d1n,
        d3p=d3p,
        d3n=d3n,
    )


# ----------------------------
# Plotting (layers / overlays)
# ----------------------------


@dataclass(frozen=True)
class DerivStackLayers:
    show_profiles: bool = True
    show_theory: bool = True
    show_extrema_lines: bool = True
    show_extrema_points: bool = True
    show_level_guides: bool = True
    show_neon_experiment: bool = True
    show_suptitle: bool = True


def neon_plot(x: np.ndarray, y: np.ndarray, *, ax=None, color: str = "red") -> Any:
    """Упрощённый перенос neon_plot из 3rd art (многослойная толстая линия)."""
    if ax is None:
        ax = plt.gca()
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    for cont in range(5, 1, -1):
        ax.plot(x, y, lw=cont * 0.4, color=color, zorder=5, alpha=0.2 * 0.08 * cont**2)
    for cont in range(5, 1, -1):
        ax.plot(x, y, lw=cont / 2 * 0.25, color="white", zorder=5, alpha=0.2 * 0.04 * cont**2)
    return ax


def compute_neon_overlay_data(cfg: dict[str, Any], *, dy: float) -> NeonOverlayData | None:
    """
    Загружает gauss_* файлы и подготавливает координаты для neon-наложения.

    Возвращает None, если пути не заданы.
    """
    gauss_am_path = cfg.get("gauss_am_165_path", None)
    gauss_cr_path = cfg.get("gauss_cr_165_path", None)
    if not gauss_am_path or not gauss_cr_path:
        return None

    base = Path(cfg.get("data_dir", "data"))
    gauss_am = load_txt_table_utf16(base / gauss_am_path)
    gauss_cr = load_txt_table_utf16(base / gauss_cr_path)

    # параметры из ноутбука
    N_dots = int(cfg.get("plots_deriv_stack_gauss_N_dots", 512))
    N_fin = int(cfg.get("plots_deriv_stack_gauss_N_fin", 512 // 2))
    x_width_am = float(cfg.get("plots_deriv_stack_gauss_x_width_am", 3.5555e01))
    y_width_am = float(cfg.get("plots_deriv_stack_gauss_y_width_am", 3.5555e01))
    x_width_cr = float(cfg.get("plots_deriv_stack_gauss_x_width_cr", 3.5557e01))
    y_width_cr = float(cfg.get("plots_deriv_stack_gauss_y_width_cr", 3.5557e01))

    X_cr, Z_cr = zemax_to_line(1, 0, N_dots, N_fin, x_width_cr, y_width_cr, gauss_cr)
    X_am, Z_am = zemax_to_line(1, 0, N_dots, N_fin, x_width_am, y_width_am, gauss_am)

    # масштабирование оси X как в ноутбуке
    exp_div = float(cfg.get("plots_deriv_stack_exp_div", 125.0))
    exp_L = float(cfg.get("plots_deriv_stack_exp_L_m", 165e-6))
    exp_wl = float(cfg.get("plots_deriv_stack_exp_wl_m", 1550e-9))
    level_idx = int(cfg.get("plots_deriv_stack_exp_level_i", 3))

    return NeonOverlayData(
        x_am=(X_am + 0.0) / exp_div * exp_L / exp_wl,
        z_am=np.array(Z_am, dtype=float) + dy * level_idx,
        x_cr=X_cr / exp_div * exp_L / exp_wl,
        z_cr=np.array(Z_cr, dtype=float) + dy * level_idx,
    )


def compute_deriv_stack_data(
    cfg: dict[str, Any],
    *,
    layers: DerivStackLayers | None = None,
) -> DerivStackData:
    """
    Тяжёлая часть: считает FFT/карты/экстремумы и (опционально) данные для neon overlay.

    layers используется только чтобы понять, надо ли готовить neon-данные.
    """
    layers = layers or DerivStackLayers()

    wl_gl = float(cfg.get("wl_gl", 1550e-9))
    f_gl = float(cfg.get("f_gl", 125e-3))
    L_gl = float(cfg.get("L_gl", 20e-6))
    S_gl = float(cfg.get("S_gl", 220e-6))

    alpha = cfg.get("alpha", cfg.get("alpha_list", [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]))
    alpha = np.array(alpha, dtype=float)
    conv_N = int(cfg.get("plots_deriv_stack_conv_N", cfg.get("conv_N", 500)))

    dy = float(cfg.get("plots_deriv_stack_dy", 1.0))

    s_list, items = build_zxy_array(alpha=alpha, f_gl=f_gl, wl_gl=wl_gl, L_gl=L_gl, S_gl=S_gl, conv_N=conv_N)
    metrics = compute_extrema_metrics(s_list=s_list, items=items, S_gl=S_gl, wl_gl=wl_gl, f_gl=f_gl)

    neon = compute_neon_overlay_data(cfg, dy=dy) if layers.show_neon_experiment else None

    cfg_meta = {
        "wl_gl": wl_gl,
        "f_gl": f_gl,
        "L_gl": L_gl,
        "S_gl": S_gl,
        "conv_N": conv_N,
        "dy": dy,
    }
    return DerivStackData(alpha=alpha, s_list=s_list, items=items, metrics=metrics, neon=neon, cfg_meta=cfg_meta)


def plot_deriv_stack_1x2_from_data(
    data: DerivStackData,
    *,
    cfg: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
    layers: DerivStackLayers | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    dpi: int | None = None,
) -> plt.Figure:
    """
    Лёгкая часть: отрисовка из заранее посчитанного DerivStackData.
    cfg здесь нужен только для подписей/стиля (wl_gl/f_gl/S_gl и т.п.), но по умолчанию
    мы берём их из data.cfg_meta.
    """
    layers = layers or DerivStackLayers()

    cfg = cfg or {}
    wl_gl = float(cfg.get("wl_gl", data.cfg_meta["wl_gl"]))
    f_gl = float(cfg.get("f_gl", data.cfg_meta["f_gl"]))
    S_gl = float(cfg.get("S_gl", data.cfg_meta["S_gl"]))

    dy = float(cfg.get("plots_deriv_stack_dy", data.cfg_meta["dy"]))
    lw = float(cfg.get("plots_deriv_stack_lw", 0.5))
    lw2 = float(cfg.get("plots_deriv_stack_lw2", lw / 2))

    alpha = np.array(cfg.get("alpha", data.alpha), dtype=float)
    s_list = data.s_list
    items = data.items
    m = data.metrics

    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    if layers.show_profiles or layers.show_theory or layers.show_extrema_lines or layers.show_extrema_points:
        for it in items:
            j, i = it.panel_j, it.level_i
            if layers.show_profiles:
                ax[j].plot(it.X_cut, it.Z_cut + dy * i, lw=lw)

            if layers.show_theory:
                X_g = np.linspace(-10, 10, 500)
                Z_g = np.array([H_func_gauss(float(k), 0.0, s=float(s_list[i])) for k in X_g], dtype=float)
                Z_g = Z_g / np.amax(Z_g) + dy * i
                ax[j].plot(X_g * 1e-3 * S_gl / wl_gl / f_gl, Z_g, "k--", lw=lw2, alpha=1)

            if layers.show_extrema_lines:
                ax[j].plot([m.MAX_1st_x_pos_th[i], m.MAX_1st_x_pos_th[i]], [dy * i, dy * (i + 1)], "k--", lw=lw2, alpha=1)
                ax[j].plot([m.MAX_1st_x_neg_th[i], m.MAX_1st_x_neg_th[i]], [dy * i, dy * (i + 1)], "k--", lw=lw2, alpha=1)
                ax[j].plot([m.MAX_2nd_x_pos_th[i], m.MAX_2nd_x_pos_th[i]], [dy * i, dy * (i + 1)], "b--", lw=lw2, alpha=1)
                ax[j].plot([m.MAX_2nd_x_neg_th[i], m.MAX_2nd_x_neg_th[i]], [dy * i, dy * (i + 1)], "b--", lw=lw2, alpha=1)

            if layers.show_extrema_points:
                if j == 1:
                    sc_color = "r" if float(m.d1p[i]) > 0.1 else "k"
                    ax[j].scatter(m.MAX_1st_x_pos_sim[i], it.Z_cut[np.where(it.X_cut == m.MAX_1st_x_pos_sim[i])[0].item()] + dy * i, s=1, color=sc_color)
                    sc_color = "r" if float(m.d1n[i]) > 0.1 else "k"
                    ax[j].scatter(m.MAX_1st_x_neg_sim[i], it.Z_cut[np.where(it.X_cut == m.MAX_1st_x_neg_sim[i])[0].item()] + dy * i, s=1, color=sc_color)
                else:
                    sc_color = "r" if float(m.d3p[i]) > 0.1 else "k"
                    ax[j].scatter(m.MAX_2nd_x_pos_sim[i], it.Z_cut[np.where(it.X_cut == m.MAX_2nd_x_pos_sim[i])[0].item()] + dy * i, s=1, color=sc_color)
                    sc_color = "r" if float(m.d3n[i]) > 0.1 else "k"
                    ax[j].scatter(m.MAX_2nd_x_neg_sim[i], it.Z_cut[np.where(it.X_cut == m.MAX_2nd_x_neg_sim[i])[0].item()] + dy * i, s=1, color=sc_color)

    if layers.show_level_guides:
        for i in range(len(s_list)):
            ax[0].plot([-100, 100], [dy * i, dy * i], "k--", lw=lw2, alpha=0.3)
            ax[1].plot([-100, 100], [dy * i, dy * i], "k--", lw=lw2, alpha=0.3)
        ax[0].plot([0, 0], [0, dy * (len(s_list) + 1)], "k--", lw=lw2, alpha=0.3)

    if layers.show_neon_experiment and data.neon is not None:
        neon_plot(data.neon.x_am, data.neon.z_am, ax=ax[0], color="red")
        neon_plot(data.neon.x_cr, data.neon.z_cr, ax=ax[1], color="red")

    ax[0].set(title="Amorphous", ylabel=r"$\alpha = \dfrac{2w_0}{L_{meta}}$", xlim=[-5, 5], xlabel=r"$\dfrac{S}{\lambda f}\cdot x$")
    ax[1].set(title="Crystal", ylabel="", xlim=[-5, 5], xlabel=r"$\dfrac{S}{\lambda f}\cdot x$")
    ax[0].set(ylim=[-dy, (len(s_list) + 2) * dy], yticks=np.linspace(0, (len(s_list) - 1) * dy, len(s_list)), yticklabels=np.round(alpha, 2))
    ax[1].set(ylim=[-dy, (len(s_list) + 2) * dy], yticks=np.linspace(0, (len(s_list) - 1) * dy, len(s_list)), yticklabels=np.round(alpha, 2))

    if layers.show_suptitle:
        sup1 = r"$\lambda = $" + str(round(wl_gl * 1e9)) + " nm, f = " + str(round(float(cfg.get("f_gl", data.cfg_meta["f_gl"])) * 1e3)) + " mm, S = " + str(round(S_gl * 1e6)) + r"$\mu m$" + "\n"
        alpha_opt = float(cfg.get("plots_deriv_stack_alpha_opt", 0.5))
        HWHM = 2 * float(cfg.get("f_gl", data.cfg_meta["f_gl"])) * wl_gl / np.pi / (alpha_opt * 1e-3) / S_gl
        sup2 = r"Optimim: $\alpha=0.5$, HWHM = " + str(round(HWHM, 2)) + "mm"
        fig.suptitle(sup1 + sup2)

    plt.tight_layout(pad=2)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    return fig


def plot_deriv_stack_1x2(
    *,
    cfg: dict[str, Any],
    out_path: str | Path | None = None,
    layers: DerivStackLayers | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    dpi: int | None = None,
) -> plt.Figure:
    """
    Итоговый перенос графика (ячейка 179 + экспериментальное наложение).

    Входы берём из CFG:
    - wl_gl/f_gl/L_gl/S_gl
    - alpha (или alpha_list)
    - conv_N
    - gauss_am_path / gauss_cr_path (для neon overlay)
    """
    layers = layers or DerivStackLayers()
    data = compute_deriv_stack_data(cfg, layers=layers)
    return plot_deriv_stack_1x2_from_data(
        data,
        cfg=cfg,
        out_path=out_path,
        layers=layers,
        figsize=figsize,
        dpi=dpi,
    )

