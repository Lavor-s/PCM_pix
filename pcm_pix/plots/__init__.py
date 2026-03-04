"""
pcm_pix.plots — набор функций для построения и сохранения графиков.

Принцип:
- расчёты/оптимизация живут в `main_clean.ipynb` и пишут артефакты в `outputs/<run_name>/`
- этот пакет отвечает за воспроизводимые графики, которые можно строить отдельно (например, из `plots.ipynb`)

Каждая функция plot_* должна:
- принимать входные данные явно (без глобальных переменных)
- уметь сохранять картинку в заданный путь (обычно `outputs/<run_name>/plots/...`)
- возвращать matplotlib.figure.Figure (и/или axes), чтобы в ноутбуке можно было дополнительно дооформить.
"""

# re-export a few common helpers for convenience
from .style import apply_style  # noqa: F401

# plots with heavy calculation/overlays
from .deriv_stack import DerivStackLayers, plot_deriv_stack_1x2  # noqa: F401

