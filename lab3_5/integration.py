import numpy as np


# Узлы с шагом h
def grid(a, b, h):
    count = round((b - a) / h)
    if count < 1 or abs(count * h - (b - a)) > 1e-9:
        raise ValueError(f"Шаг h = {h:g} не укладывается в [{a:g}, {b:g}] целое число раз.")
    return np.linspace(a, b, count + 1)


# Левые прямоугольники
def left_rectangles(f, a, b, h):
    return h * np.sum(f(grid(a, b, h)[:-1]))


# Правые прямоугольники
def right_rectangles(f, a, b, h):
    return h * np.sum(f(grid(a, b, h)[1:]))


# Средние прямоугольники
def middle_rectangles(f, a, b, h):
    x = grid(a, b, h)
    return h * np.sum(f(x[:-1] + h / 2))


# Метод трапеций
def trapezoids(f, a, b, h):
    y = f(grid(a, b, h))
    return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)


# Метод Симпсона (N должно быть чётным)
def simpson(f, a, b, h):
    y = f(grid(a, b, h))
    if (len(y) - 1) % 2 != 0:
        raise ValueError(f"Для метода Симпсона число отрезков должно быть чётным, а при h = {h:g} оно "
                         f"равно {len(y) - 1}.")
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])


# Метод Рунге-Ромберга
def runge_romberg(F_h, F_kh, k, p):
    correction = (F_h - F_kh) / (k**p - 1)
    return F_h + correction, abs(correction)
