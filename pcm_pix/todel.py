import numpy as np

def build_square_lattices(L, a_pix, d_pix, b_pix, axis="x", origin=(0.0, 0.0)):
    a_pix = np.asarray(a_pix, dtype=float)  # период (и размер ячейки-квадрата)
    d_pix = np.asarray(d_pix, dtype=float)  # внешний диаметр
    b_pix = np.asarray(b_pix, dtype=float)  # внутренний диаметр

    if not (a_pix.shape == d_pix.shape == b_pix.shape) or a_pix.ndim != 1:
        raise ValueError("a_pix, d_pix, b_pix должны быть 1D массивами одной длины (N)")
    if L <= 0:
        raise ValueError("L должно быть > 0")
    if axis not in ("x", "y"):
        raise ValueError("axis должен быть 'x' или 'y'")

    N = len(a_pix)
    ox, oy = map(float, origin)
    x0, x1 = ox - L / 2.0, ox + L / 2.0
    y0, y1 = oy - L / 2.0, oy + L / 2.0
    stripe_w = L / N

    x_d_list, y_d_list = [], []
    a_d_list, d_d_list, b_d_list = [], [], []

    prev_touch = None  # край последней ячейки предыдущей полосы (x или y)

    def grid_1d(vmin, vmax, step):
        if vmax < vmin:
            return np.array([])
        n = int(np.floor((vmax - vmin) / step + 1e-12)) + 1
        return vmin + step * np.arange(n)

    for i in range(N):
        a = float(a_pix[i])
        d = float(d_pix[i])
        b = float(b_pix[i])
        half = a / 2.0

        if a <= 0 or d <= 0:
            continue
        if b < 0 or b > d:
            raise ValueError(f"Некорректные диаметры в полосе i={i}: b={b}, d={d}")
        if d > a + 1e-15:
            raise ValueError(f"В полосе i={i} диск не помещается в ячейку: d={d} > a={a}")

        if axis == "x":
            sx0_nom = x0 + i * stripe_w
            sx1_nom = x0 + (i + 1) * stripe_w
            sx0 = sx0_nom if (prev_touch is None) else min(sx0_nom, prev_touch)

            x_min, x_max = sx0 + half, sx1_nom - half
            y_min, y_max = y0 + half, y1 - half

            xs = grid_1d(x_min, x_max, a)
            ys = grid_1d(y_min, y_max, a)
            if xs.size == 0 or ys.size == 0:
                continue

            prev_touch = float(xs[-1] + half)

        else:
            sy0_nom = y0 + i * stripe_w
            sy1_nom = y0 + (i + 1) * stripe_w
            sy0 = sy0_nom if (prev_touch is None) else min(sy0_nom, prev_touch)

            y_min, y_max = sy0 + half, sy1_nom - half
            x_min, x_max = x0 + half, x1 - half

            xs = grid_1d(x_min, x_max, a)
            ys = grid_1d(y_min, y_max, a)
            if xs.size == 0 or ys.size == 0:
                continue

            prev_touch = float(ys[-1] + half)

        X, Y = np.meshgrid(xs, ys, indexing="xy")
        k = X.size

        x_d_list.append(X.ravel())
        y_d_list.append(Y.ravel())
        a_d_list.append(np.full(k, a))
        d_d_list.append(np.full(k, d))
        b_d_list.append(np.full(k, b))

    if not x_d_list:
        empty = np.array([])
        return empty, empty, empty, empty, empty

    x_d = np.concatenate(x_d_list)
    y_d = np.concatenate(y_d_list)
    a_d = np.concatenate(a_d_list)
    d_d = np.concatenate(d_d_list)
    b_d = np.concatenate(b_d_list)

    return x_d, y_d, a_d, d_d, b_d




from scipy.special import k1


def write_disks_gds_forlum(x, y, d, b, gds_path, cell_name="DISKS", datatype=0, points=64):
    """
    По 1D массивам x, y, d, b (одинаковой длины K) строит для каждой позиции кольцо (диск с отверстием):
    - центр: (x[k], y[k])
    - внешний диаметр: d[k]
    - внутренний диаметр: b[k] (0 => сплошной диск)
    Пишет результат в gds_path.
    """
    x = np.asarray(x); y = np.asarray(y); d = np.asarray(d); b = np.asarray(b)
    if not (x.shape == y.shape == d.shape == b.shape):
        raise ValueError("x, y, d, b должны иметь одинаковую форму (1D)")
    if x.ndim != 1:
        raise ValueError("x, y, d, b должны быть 1D массивами")

    gdspy.current_library = gdspy.GdsLibrary()
    lib = gdspy.GdsLibrary()
    cell = lib.new_cell(cell_name)
    
    layer = 1

    for k in range(len(x)):
        cx, cy = float(x[k]), float(y[k])
        do, bi = float(d[k]), float(b[k])

        if k % 1000 == 0:
            layer += 1

        if do <= 0:
            continue
        if bi < 0 or bi > do:
            raise ValueError(f"Некорректные диаметры в k={k}: b={bi}, d={do}")

        r_out = do / 2.0
        r_in  = bi / 2.0

        if r_in == 0:
            shape = gdspy.Round((cx, cy), r_out,
                                number_of_points=points, layer=layer, datatype=datatype)
        else:
            shape = gdspy.Round((cx, cy), r_out, inner_radius=r_in,
                                number_of_points=points, layer=layer, datatype=datatype)

        cell.add(shape)

    lib.write_gds(gds_path)
    return cell