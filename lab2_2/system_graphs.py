import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import system_functions as sf

# Цвета методов одинаковые на обоих графиках и не совпадают с цветами кривых
METHOD_COLORS = {"Простая итерация": "tab:green", "Ньютон": "tab:purple"}


# Кривые f1 = 0 и f2 = 0; решение системы — точка их пересечения
# bounds — область G (прямоугольник), x0 — начальное приближение,
# paths — список пар (подпись, history) для траекторий методов
def plot_curves(ax, bounds=None, x0=None, paths=()):
    # lg x2 определён только при x2 > 0, поэтому сетка по x2 начинается чуть выше нуля
    X1, X2 = np.meshgrid(np.linspace(-2.5, 2.5, 500), np.linspace(0.05, 4, 500))
    ax.contour(X1, X2, sf.f1(X1, X2), levels=[0], colors="tab:blue")
    ax.contour(X1, X2, sf.f2(X1, X2), levels=[0], colors="tab:orange")

    if bounds is not None:
        (low1, high1), (low2, high2) = bounds
        ax.add_patch(Rectangle((low1, low2), high1 - low1, high2 - low2,
                               fill=False, linestyle="--", color="gray", label="область G"))
    if x0 is not None:
        ax.plot(x0[0], x0[1], "ko", label="x⁰")
    for label, history in paths:
        points = np.array([x0] + [x for x, _ in history])
        ax.plot(points[:, 0], points[:, 1], ".-", color=METHOD_COLORS[label], label=label)

    # contour не попадает в легенду сам, поэтому подписи для кривых делаем вручную
    curve_handles = [
        Line2D([], [], color="tab:blue", label="f1 = 0"),
        Line2D([], [], color="tab:orange", label="f2 = 0"),
    ]
    ax.legend(handles=curve_handles + ax.get_legend_handles_labels()[0], loc="upper left")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


# Погрешность ||x^(k) - x^(k-1)|| по номеру итерации, ось y логарифмическая
def plot_errors(ax, history_simple, history_newton, eps):
    for history, label in ((history_simple, "Простая итерация"), (history_newton, "Ньютон")):
        errors = [error for _, error in history]
        ax.semilogy(range(1, len(errors) + 1), errors, "o-", color=METHOD_COLORS[label], label=label)
    ax.axhline(eps, color="red", linestyle="--", label=f"ε = {eps:g}")
    ax.grid(True)
    ax.set_xlabel("k")
    ax.set_ylabel("||x^(k) - x^(k-1)||")
    ax.legend()
