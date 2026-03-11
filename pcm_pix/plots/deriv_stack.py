from __future__ import annotations

r"""
deriv_stack.py — перенос Фурье-оптики из `3rd art_PCM_bagel_2025.ipynb`

Что делает этот модуль:

1) `load_txt_table_utf16` — загрузка табличных данных из UTF‑16 txt (TAB, skiprows) в 2D `numpy` 
2) `zemax_to_line` — извлечение 1D-сечения из 2D карты (по x или y) в стиле ZEMAX  
3) `R_lum_fun` — LUM‑амплитудная поправка \(R\) для точки (u,v) для состояния `cr`/`am`  
4) `phi_lum_fun` — LUM‑фазовая поправка \(\phi\) для точки (u,v) для состояния `cr`/`am`  

5) `H_func_gauss` — 2D гауссиана (ядро \(H\)) для дальнейшего FFT‑расчёта  
6) `conv_corr_first` — FFT→умножение на LUM (state=`cr`)→IFFT; возвращает (RES, Vx, Vy)  
7) `conv_corr_sec` — как `conv_corr_first`, но LUM в режиме state=`am`  
8) `build_s_list_from_alpha` — перевод списка \(\alpha\) в список \(s\) по формуле из ноутбука  
9) `build_zxy_array` — строит набор `ZXYItem` (карты Z, сетки X/Y и сечения) для всех уровней и панелей  
10) `min_by_cond` — находит `x`, где `y` минимален при условии на `x`  
11) `max_by_cond` — находит `x`, где `y` максимален при условии на `x`  
12) `compute_extrema_metrics` — считает теорию/симуляцию положений экстремумов и ошибки (d1p/d1n/d3p/d3n)  
13) `neon_plot` — рисует “неоновую” линию (многослойный толстый контур)  
14) `compute_neon_overlay_data` — загружает экспериментальные gauss_* и приводит их к координатам графика для overlay  
15) `compute_deriv_stack_data` — “тяжёлый” расчёт: FFT‑карты, сечения, экстремумы, (опц.) neon; возвращает `DerivStackData`  

16) `plot_deriv_stack_1x2_from_data` — “лёгкая” отрисовка 1×2 из готового `DerivStackData`  
17) `plot_deriv_stack_1x2` — точка входа: посчитать данные + нарисовать 1×2 график по `cfg`

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from numpy.typing import NDArray
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ..lum_tables import R_lum_cr_2, R_lum_am_2, phi_lum_cr_2, phi_lum_am_2 # Для люма
from ..lum_tables import R_lum_cr_3, R_lum_am_3, phi_lum_cr_3, phi_lum_am_3 # Для конусов
from .theory_pix_lum import f_step, pixel_edges, amp_phase_from_g, f_step, T_phi_pix

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

def zemax_to_line(x_or_y: int, val_cross: float, N_dots: int, N_fin: int, x_width: float, y_width: float, Z_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    x_or_y:
    - 0: вернуть сечение Z[:, ind] как функция x_vals
    - 1: вернуть сечение Z[ind, :] как функция y_vals
    val_cross:
    - по какой величине отсчитывается сечение    

    x_width/y_width:
    - в тех единицах, как в старом ноутбуке (обычно µm или mm — зависит от файла).
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


def Transfer_function(u: float, v: float, mode: str, state: str, S: float, Npix: int=11):
    """Передаточная функция. mode: theory, ideal_pixel, dataset_*, transparent"""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    mask = (np.abs(u) < S / 2) & (np.abs(v) < S / 2)

    match mode:
        case "transparent":
            return np.where(mask, 1.0 + 0.0j, 0.0 + 0.0j)

        case "theory":
            amp, phase = amp_phase_from_g(u, state=state, D=S)
            H = amp * np.exp(1j * phase)
            return np.where(mask, H, 0.0 + 0.0j)

        case "ideal_pixel":
            amp, phase = T_phi_pix(u, v, state=state, D=S, Npix=Npix, f_step=f_step)
            H = amp * np.exp(1j * phase)
            return np.where(mask, H, 0.0 + 0.0j)

        case "dataset_2":
            # 1) массивы амплитуды и фазы из LUM-таблиц
            if state == "cr":
                amp_vals = R_lum_cr_2
                phi_vals = phi_lum_cr_2
            elif state == "am":
                amp_vals = R_lum_am_2
                phi_vals = phi_lum_am_2
            else:
                raise ValueError(f"unknown state={state!r}")

            # 2) геометрия пикселей вдоль u
            Npix = len(amp_vals)
            edges = pixel_edges(S, Npix)

            # 3) ступенчатые функции по этим значениям
            amp = f_step(u, edges, amp_vals)
            phase = f_step(u, edges, phi_vals)
            H = amp * np.exp(1j * phase)
            return np.where(mask, H, 0.0 + 0.0j)


        case "dataset_3":
            # 1) массивы амплитуды и фазы из LUM-таблиц
            if state == "cr":
                amp_vals = R_lum_cr_3
                phi_vals = phi_lum_cr_3
            elif state == "am":
                amp_vals = R_lum_am_3
                phi_vals = phi_lum_am_3
            else:
                raise ValueError(f"unknown state={state!r}")

            # 2) геометрия пикселей вдоль u
            Npix = len(amp_vals)
            edges = pixel_edges(S, Npix)

            # 3) ступенчатые функции по этим значениям
            amp = f_step(u, edges, amp_vals)
            phase = f_step(u, edges, phi_vals)
            H = amp * np.exp(1j * phase)
            return np.where(mask, H, 0.0 + 0.0j)



def conv_corr(
    *,
    N: int, # кол-во точек
    width: float,      # ширина входного окна в м
    field_fun:  Callable[..., np.ndarray],
    field_kwargs: dict[str, Any] | None = None,
    filt_fun: Callable[..., np.ndarray],
    filt_kwargs: dict[str, Any] | None = None,
    f_gl: float, #фокальное расстояние в м
    wl_gl: float, #длина волны в м
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    "сворачиваем field_fun и filt_fun"
    field_kwargs = {} if field_kwargs is None else field_kwargs
    filt_kwargs  = {} if filt_kwargs  is None else filt_kwargs

    # 1) Сетка (как X_ar/Y_ar)
    X_ar = np.linspace(-width/2, width/2, N)
    Y_ar = np.linspace(-width/2, width/2, N)
    XX, YY = np.meshgrid(X_ar, Y_ar, indexing="xy")

    # 2) Поле в пространстве
    test_im = np.asarray(field_fun(XX, YY, **field_kwargs))
    if test_im.shape != (N, N):
        raise ValueError(f"field_fun must return shape {(N,N)}, got {test_im.shape}")

    # 3) FFT и частотный сдвиг
    RES = np.fft.fftshift(np.fft.fft2(test_im))

    # 4) Оси U_x/U_y
    #dx = 1e-3 * abs(X_ar[1] - X_ar[0])
    #dy = 1e-3 * abs(Y_ar[1] - Y_ar[0])
    dx = abs(X_ar[1] - X_ar[0])
    dy = abs(Y_ar[1] - Y_ar[0])
    U_x = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * f_gl * wl_gl
    U_y = np.fft.fftshift(np.fft.fftfreq(N, d=dy)) * f_gl * wl_gl
    UX, UY = np.meshgrid(U_x, U_y, indexing="xy")

    # 5) Домножение на фильтр в (u,v)
    H = np.asarray(filt_fun(UX, UY, **filt_kwargs))
    if H.shape != (N, N):
        raise ValueError(f"filt_fun must return shape {(N,N)}, got {H.shape}")
    RES *= H

    # 6) IFFT (важно: убрать shift перед ifft2)
    RES = np.fft.ifft2(np.fft.ifftshift(RES))

    # 7) Оси V_x/V_y как у тебя
    V_x = np.fft.ifftshift(np.fft.fftfreq(N, d=abs(U_x[1] - U_x[0]) / f_gl / wl_gl))
    V_y = np.fft.ifftshift(np.fft.fftfreq(N, d=abs(U_y[1] - U_y[0]) / f_gl / wl_gl))
    return RES, V_x, V_y








def plot_transfer_conv_stack(
    *,
    N: int = 512,
    width: float = 50e-3,
    f_gl: float = 125e-3,
    wl_gl: float = 1550e-9,
    S_gl: float = 165e-6,
    s: float = 1.0e-3,
    Npix: int = 11,
    xy_display: float | None = None,
    uv_display: float | None = None,
    figsize: Tuple[float, float] = (16.0, 14.0),
    dpi: int | None = None,
) -> plt.Figure:
    """
    Строит 7×8 стек субплотов для демонстрации работы Фурье‑фильтрации.

    Численные параметры:
      - N:     количество точек по x и y (квадратная сетка N×N)
      - width: ширина входного окна по x и y в метрах
      - f_gl:  фокусное расстояние (м)
      - wl_gl: длина волны (м)
      - S_gl:  размер апертуры/пиксельного окна в метрах (используется в Transfer_function)
      - s:     параметр ширины входного гаусса (в тех же единицах, что x,y)
      - Npix:  число пикселей для режимов "ideal_pixel" (передаётся в Transfer_function)

    Параметры отображения:
      - xy_display: половина видимого диапазона по x,y ([-xy_display, +xy_display]);
                    по умолчанию width/2
      - uv_display: половина видимого диапазона по u,v ([-uv_display, +uv_display]);
                    по умолчанию S_gl/2

    Макет субплотов: 7 строк (T=7), 8 столбцов (t=8).

    Строки (по вертикали):
      1) |E_in(x,y)|      — модуль входного поля
      2) |FFT(E_in)|      — модуль Фурье‑образа
      3) arg(FFT(E_in))   — фаза Фурье‑образа
      4) |H(u,v)|         — модуль передаточной функции
      5) arg(H(u,v))      — фаза передаточной функции
      6) |E_out(x',y')|   — модуль выхода conv_corr
      7) arg(E_out)       — фаза выхода conv_corr

    Столбцы (по горизонтали) — комбинации (state, mode) для Transfer_function:
      1) state="am", mode="theory"
      2) state="cr", mode="theory"
      3) state="am", mode="ideal_pixel"
      4) state="cr", mode="ideal_pixel"
      5) state="am", mode="dataset_2"
      6) state="cr", mode="dataset_2"
      7) state="am", mode="dataset_3"
      8) state="cr", mode="dataset_3"

    Нормировка:
      - все модули (|…|) нормируются на максимум своего массива → [0, 1]
      - фазы приводятся к диапазону [0, 2π) через (angle + 2π) % (2π)

    Визуализация:
      - все изображения рисуются через contourf с 50 уровнями
      - без colorbar’ов и подписей осей/заголовков
    """
    if xy_display is None:
        xy_display = width / 2.0
    if uv_display is None:
        uv_display = S_gl / 2.0

    # Векторизованный гаусс для массивов (x, y)
    def gauss2d(x: NDArray, y: NDArray, *, s: float, phi: float = np.pi / 2.0,
                shift_x: float = 0.0, shift_y: float = 0.0) -> NDArray:
        _ = phi
        return np.exp(-(((x - shift_x) ** 2 + (y - shift_y) ** 2) / (s ** 2)))

    # 1) Сетка в пространстве x,y
    X_ar = np.linspace(-width / 2, width / 2, N)
    Y_ar = np.linspace(-width / 2, width / 2, N)
    XX, YY = np.meshgrid(X_ar, Y_ar, indexing="xy")

    # Входное поле
    field_im = gauss2d(XX, YY, s=s, phi=np.pi / 2)

    # 2) Фурье‑образ входного поля
    RES_freq = np.fft.fftshift(np.fft.fft2(field_im))

    # 3) Частотные оси u,v (как в conv_corr)
    dx = abs(X_ar[1] - X_ar[0])
    dy = abs(Y_ar[1] - Y_ar[0])
    U_x = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * f_gl * wl_gl
    U_y = np.fft.fftshift(np.fft.fftfreq(N, d=dy)) * f_gl * wl_gl
    UX, UY = np.meshgrid(U_x, U_y, indexing="xy")

    # Конфигурации столбцов: (state, mode)
    cols_cfg = [
        ("am", "theory"),
        ("cr", "theory"),
        ("am", "ideal_pixel"),
        ("cr", "ideal_pixel"),
        ("am", "dataset_2"),
        ("cr", "dataset_2"),
        ("am", "dataset_3"),
        ("cr", "dataset_3"),
    ]

    # Вспомогательные функции нормировки
    def norm_abs(z: NDArray) -> NDArray:
        z = np.asarray(z)
        m = np.max(np.abs(z))
        return np.zeros_like(z, dtype=float) if m == 0 else np.abs(z) / m

    def phase_with_threshold(z: NDArray, amp_norm: NDArray, thr: float = 1e-1) -> NDArray:
        """
        Фаза ∈ [0, 2π), но зануляется там, где amp_norm < thr.
        """
        phase = (np.angle(z) + 2.0 * np.pi) % (2.0 * np.pi)
        phase = np.where(amp_norm < thr, 0, phase)
        return phase

    fig, axs = plt.subplots(7, len(cols_cfg), figsize=figsize, dpi=dpi, squeeze=False)

    # Данные, общие для всех столбцов: вход и его FFT
    abs_field = norm_abs(field_im)
    phase_field = phase_with_threshold(field_im, abs_field)
    abs_fft   = norm_abs(RES_freq)
    phase_fft = phase_with_threshold(RES_freq, abs_fft)


    # 1‑я строка — входное поле (одинаковое для всех столбцов)
    for j in range(len(cols_cfg)):
        ax = axs[0, j]
        cs = ax.contourf(X_ar, Y_ar, abs_field, levels=50, vmin=0.0, vmax=1.0)
        ax.set_xlim(-xy_display, xy_display)
        ax.set_ylim(-xy_display, xy_display)

    # 2‑я строка — модуль FFT входа
    for j in range(len(cols_cfg)):
        ax = axs[1, j]
        cs = ax.contourf(U_x, U_y, abs_fft, levels=50, vmin=0.0, vmax=1.0)
        ax.set_xlim(-uv_display, uv_display)
        ax.set_ylim(-uv_display, uv_display)

    # 3‑я строка — фаза FFT входа
    for j in range(len(cols_cfg)):
        ax = axs[2, j]
        cs = ax.contourf(U_x, U_y, phase_fft, levels=50, vmin=0.0, vmax=2.0 * np.pi)
        ax.set_xlim(-uv_display, uv_display)
        ax.set_ylim(-uv_display, uv_display)

    # Далее по столбцам считаем H и conv_corr
    for j, (state, mode) in enumerate(cols_cfg):
        # 4) Передаточная функция H(u,v) для данного столбца
        H = Transfer_function(UX, UY, mode=mode, state=state, S=S_gl, Npix=Npix)

        abs_H = norm_abs(H)
        phase_H = phase_with_threshold(H, abs_H, thr=-1e-1)


        # 4‑я строка — модуль H(u,v)
        ax4 = axs[3, j]
        cs4 = ax4.contourf(U_x, U_y, abs_H, levels=50, vmin=0.0, vmax=1.0)
        ax4.set_xlim(-uv_display, uv_display)
        ax4.set_ylim(-uv_display, uv_display)

        # 5‑я строка — фаза H(u,v)
        ax5 = axs[4, j]
        cs5 = ax5.contourf(U_x, U_y, phase_H, levels=50, vmin=0.0, vmax=2.0 * np.pi)
        ax5.set_xlim(-uv_display, uv_display)
        ax5.set_ylim(-uv_display, uv_display)

        # 6–7 строки — результат conv_corr для данного фильтра
        RES_spatial, V_x, V_y = conv_corr(
            N=N,
            width=width,
            field_fun=lambda X, Y: gauss2d(X, Y, s=s, phi=np.pi / 2),
            field_kwargs={},
            filt_fun=Transfer_function,
            filt_kwargs={
                "mode": mode,
                "state": state,
                "S": S_gl,
                "Npix": Npix,
            },
            f_gl=f_gl,
            wl_gl=wl_gl,
        )

        abs_out = norm_abs(RES_spatial)
        phase_out = phase_with_threshold(RES_spatial, abs_out)

        # 6‑я строка — модуль выхода
        ax6 = axs[5, j]
        cs6 = ax6.contourf(V_x, V_y, abs_out, levels=50, vmin=0.0, vmax=1.0)
        ax6.set_xlim(-xy_display, xy_display)
        ax6.set_ylim(-xy_display, xy_display)

        # 7‑я строка — фаза выхода
        ax7 = axs[6, j]
        cs7 = ax7.contourf(V_x, V_y, phase_out, levels=50, vmin=0.0, vmax=2.0 * np.pi)
        ax7.set_xlim(-xy_display, xy_display)
        ax7.set_ylim(-xy_display, xy_display)

    #убрать подписи осей, колорбары, подписи строк и столбцов
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        #ax.set_colorbar(False)
        #ax.set_title('')


    plt.tight_layout()
    return fig


def plot_output_slices_1x2(
    *,
    N: int = 128,
    width: float = 50e-3,
    f_gl: float = 125e-3,
    wl_gl: float = 1550e-9,
    S_gl: float = 165e-6,
    Npix: int = 11,
    xy_display: float = 10.0,
    mode: str = "theory",
    alpha_list: list[float] | np.ndarray = (0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
    exp_files: dict[tuple[float, str], str] | None = None,
) -> plt.Figure:
    """
    Рисует 1×2 субплота с сечениями квадрата выходного поля для двух состояний:
    слева state="am", справа state="cr".

    По оси x — нормированная координата:
        x_norm = S_gl * x / (wl_gl * f_gl)

    По оси y каждый субплот поделен на len(s_list) полос; в полосе idx
    (нижняя граница y = idx) рисуется график |E_out|^2 (сечение по V_y≈0),
    нормированный на [0,1] и заполняющий высоту [idx, idx+1].

    Риска по оси y для s_list[idx] совпадает с низом полосы (y = idx).
    """

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    rc = {
        "font.size": SMALL_SIZE,
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "figure.figsize": (7, 4),
        "figure.dpi": 300,
    }
    with plt.rc_context(rc):

        if exp_files is None:
            exp_files = {}
            
        alpha_arr = np.array(alpha_list, dtype=float)
        if alpha_arr.size == 0:
            raise ValueError("alpha_list must contain at least one value")

        s_arr = 2.0 / S_gl * wl_gl * f_gl / (np.pi * alpha_arr)
        n_slices = alpha_arr.size
        states = ["am", "cr"]
        
        fig, axes = plt.subplots(1, 2, squeeze=False)
        axes = axes[0]

        # Вспомогательная гауссиана (как в plot_transfer_conv_stack)
        def gauss2d(x: NDArray, y: NDArray, *, s: float, phi: float = np.pi / 2.0,
                    shift_x: float = 0.0, shift_y: float = 0.0) -> NDArray:
            _ = phi
            return np.exp(-(((x - shift_x) ** 2 + (y - shift_y) ** 2) / (s ** 2)))

        #Чтение и подготовка экспериментальной кривой
        def load_exp_curve_1d(
            path: str | Path,
            *,
            x_or_y: int,
            val_cross: float,
            N_dots: int,
            N_fin: int,
            x_width: float,
            y_width: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Читает файл LUM-формата и возвращает (x_vals, Z_line),
            где x_vals — физическая координата (в тех же единицах, что x_width/y_width),
            Z_line — нормированная (0..1) интенсивность.
            """
            Z_data = load_txt_table_utf16(path)      # 2D массив
            # Z_data имеет форму (N_dots, N_dots) — вы зададите N_dots и ширины из своего эксперимента
            x_vals, Z_line = zemax_to_line(
                x_or_y=x_or_y,
                val_cross=val_cross,
                N_dots=N_dots,
                N_fin=N_fin,
                x_width=x_width,
                y_width=y_width,
                Z_data=Z_data,
            )
            # нормировка на максимум
            Z_line = np.asarray(Z_line, dtype=float)
            m = np.max(np.abs(Z_line))
            if m > 0:
                Z_line = Z_line / m
            else:
                Z_line = np.zeros_like(Z_line)
            return x_vals, Z_line




        for ax, state in zip(axes, states):
            x_norm_global_min = None
            x_norm_global_max = None

            for idx, s in enumerate(s_arr):
                RES_spatial, V_x, V_y = conv_corr(
                    N=N,
                    width=width,
                    field_fun=lambda X, Y, s_local=s: gauss2d(X, Y, s=s_local, phi=np.pi / 2),
                    field_kwargs={},
                    filt_fun=Transfer_function,
                    filt_kwargs={
                        "mode": mode,
                        "state": state,
                        "S": S_gl,
                        "Npix": Npix,
                    },
                    f_gl=f_gl,
                    wl_gl=wl_gl,
                )

                alpha_val = float(alpha_arr[idx])

                # индекс по V_y, ближайший к 0
                ind_y0 = int(np.argmin(np.abs(V_y)))
                line = RES_spatial[ind_y0, :]
                intensity = np.abs(line) ** 2

                # Нормированная координата (по x' == V_x)
                x_norm = S_gl * V_x / (wl_gl * f_gl)

                # Запомним полный диапазон x_norm, чтобы при xy_display<=0
                # можно было аккуратно выставить xlim
                cur_min = float(np.min(x_norm))
                cur_max = float(np.max(x_norm))
                x_norm_global_min = cur_min if x_norm_global_min is None else min(x_norm_global_min, cur_min)
                x_norm_global_max = cur_max if x_norm_global_max is None else max(x_norm_global_max, cur_max)

                # Ограничение по xy_display, если он > 0
                if xy_display > 0:
                    mask = np.abs(x_norm) <= xy_display
                else:
                    mask = np.ones_like(x_norm, dtype=bool)

                x_plot = x_norm[mask]
                y_val = intensity[mask]
                if y_val.size == 0:
                    continue

                max_y = np.max(y_val)
                y_norm = np.zeros_like(y_val) if max_y == 0 else y_val / max_y

                # Полоса: [idx, idx+1]; низ полосы = idx
                y_bottom = float(idx)
                y_top = float(idx + 0.9)
                y_line = y_bottom + y_norm * (y_top - y_bottom)



                x_phys = x_plot * wl_gl * f_gl / S_gl  # из x_norm → x
                gauss_in = (np.exp(-(x_phys**2 / s**2))) ** 2
                gauss_norm = gauss_in / np.max(gauss_in)
                y_gauss = y_bottom + gauss_norm * (y_top - y_bottom)

                ax.plot(x_plot, y_line, 'k', lw=0.5)
                ax.plot(x_plot, np.ones_like(x_plot) * y_bottom, 'k--', lw=0.5)
                ax.plot(x_plot, y_gauss, 'k--', lw=0.5)



                key = (alpha_val, state)
                # ищем ближайший ключ к alpha_val
                if exp_files:
                    # ищем подходящий ключ по alpha и state
                    candidates = [
                        (a, st, path)
                        for (a, st), path in exp_files.items()
                        if st == state and np.isclose(a, alpha_val, atol=1e-3)
                    ]
                    if candidates:
                        _, _, fname = candidates[0]
                        fname = Path(fname)
                        Z_data = load_txt_table_utf16(fname)
                        N_dots = Z_data.shape[0]
                        x_width = width
                        y_width = width
                        x_vals_exp, Z_line_exp = zemax_to_line(
                            x_or_y=1,
                            val_cross=0.0,
                            N_dots=N_dots,
                            N_fin=N_dots // 2,
                            x_width=x_width,
                            y_width=y_width,
                            Z_data=Z_data,
                        )
                        x_vals_exp_m = np.asarray(x_vals_exp, dtype=float)
                        x_norm_exp = S_gl * x_vals_exp_m / (wl_gl * f_gl)
                        if xy_display > 0:
                            mask_exp = np.abs(x_norm_exp) <= xy_display
                        else:
                            mask_exp = np.ones_like(x_norm_exp, dtype=bool)
                        x_plot_exp = x_norm_exp[mask_exp]
                        y_exp = np.asarray(Z_line_exp[mask_exp], dtype=float)
                        if y_exp.size > 0:
                            m_exp = np.max(np.abs(y_exp))
                            y_exp_norm = y_exp / m_exp if m_exp > 0 else np.zeros_like(y_exp)
                            y_exp_line = y_bottom + y_exp_norm * (y_top - y_bottom)
                            ax.plot(x_plot_exp, y_exp_line, color="tab:red", lw=0.8)


            # Настройка xlim
            if xy_display > 0:
                ax.set_xlim(-xy_display, xy_display)
            else:
                # если xy_display<=0, используем фактический диапазон
                if x_norm_global_min is not None and x_norm_global_max is not None:
                    ax.set_xlim(x_norm_global_min, x_norm_global_max)

            # Настройка ylim
            ax.set_ylim(0.0, float(n_slices)*1.0)

            # Левая ось Y — α
            y_ticks = [float(i) for i in range(n_slices)]
            alpha_labels = [f"{a_val:.1f}" for a_val in alpha_arr]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(alpha_labels)
            ax.set_ylabel(r"$\alpha$")
            # Правая ось Y — s
            ax_right = ax.twinx()
            ax_right.set_ylim(ax.get_ylim())
            ax_right.set_yticks(y_ticks)
            s_labels = [f"{s_val*1e3:.2f} mm" for s_val in s_arr]  # s в мм
            ax_right.set_yticklabels(s_labels)
            ax_right.set_ylabel("w (мм)")
            ax.set_xlabel(r"$\dfrac{D x}{\lambda f}$")
            #ax.set_title(f"state = {state!r}, mode = {mode!r}")

        fig.suptitle(
            f"$\\lambda = {wl_gl*1e9:.0f}$ nm, "
            f"$f = {f_gl*1e3:.0f}$ mm, "
            f"$D = {S_gl*1e6:.0f}$ $\\mu m$ " "\n"
            r"$\alpha = \dfrac{2w_0}{D} = \dfrac{8 f \lambda}{\pi D w}$"
        )
        fig.tight_layout()
        return fig

