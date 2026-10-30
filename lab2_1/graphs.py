import matplotlib.pyplot as plt
import numpy as np

import functions


# Общая настройка осей: оси через ноль, сетка, диапазоны
# Деления не задаём вручную: matplotlib сам подбирает их под масштаб при приближении
def setup_axes(ax, xlim, ylim):
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend()


# Способ 1: график заданной функции и ось Ox
def plot_f(ax):
    x = np.linspace(-1, 2, 400)
    ax.plot(x, functions.f(x), label='y = x^4 - 2x - 1')
    setup_axes(ax, (-1, 2), (-3, 5))


# Способ 2: пересечение y = x^4 и y = 2x + 1
def plot_intersection(ax):
    x = np.linspace(-1, 2, 400)
    ax.plot(x, x**4, label='y = x^4')
    ax.plot(x, 2*x + 1, label='y = 2x + 1')
    setup_axes(ax, (-1, 2), (-3, 5))


# Метод простых итераций: пересечение y = fi(x) и y = x
# fi определена при 2x + 1 >= 0, поэтому x берём от -0.5
def plot_fi(ax):
    x = np.linspace(-0.5, 2, 400)
    ax.plot(x, functions.fi_MSI(x), label='y = (2x + 1)^(1/4)')
    ax.plot(x, x, label='y = x')
    setup_axes(ax, (-0.5, 2), (-0.5, 2))


# Погрешность |x_k - x_(k-1)| по номеру итерации, ось y логарифмическая
def plot_errors(ax, history_simple, history_newton, eps):
    for history, label in ((history_simple, 'Простая итерация'), (history_newton, 'Ньютон')):
        errors = [error for _, error in history]
        ax.semilogy(range(1, len(errors) + 1), errors, 'o-', label=label)
    ax.axhline(eps, color='red', linestyle='--', label=f'ε = {eps:g}')
    ax.grid(True)
    ax.set_xlabel('k')
    ax.set_ylabel('|x_k - x_(k-1)|')
    ax.legend()


def graphs():
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plot_f(ax1)
    plot_intersection(ax2)
    plot_fi(ax3)
    plt.show()
