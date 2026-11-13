import os
import sys

import numpy as np

# Прогонка из ЛР 1.2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab1_2"))
from tridiagonale import tridiagonale


# Кубический сплайн
def cubic_spline(X, Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    n = len(X) - 1
    h = np.diff(X)  # шаги

    # Система для c
    m = n - 1
    lower, diag, upper, rhs = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)
    for k in range(m):
        j = k + 1
        lower[k] = h[j - 1] if k > 0 else 0.0
        diag[k] = 2 * (h[j - 1] + h[j])
        upper[k] = h[j] if k < m - 1 else 0.0
        rhs[k] = 3 * ((Y[j + 1] - Y[j]) / h[j] - (Y[j] - Y[j - 1]) / h[j - 1])

    _, _, c_inner = tridiagonale(lower, diag, upper, rhs)
    c = np.concatenate(([0.0], c_inner, [0.0]))  # на концах c = 0

    a = Y[:-1]
    b = (Y[1:] - Y[:-1]) / h - h * (c[1:] + 2 * c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)
    return a, b, c[:-1], d, (lower, diag, upper, rhs)


# Номер отрезка, где лежит x
def find_segment(X, x):
    if not X[0] <= x <= X[-1]:
        raise ValueError(f"Точка {x:g} вне отрезка интерполяции [{X[0]:g}, {X[-1]:g}].")
    for j in range(len(X) - 1):
        if x <= X[j + 1]:
            return j
    return len(X) - 2


# Значение сплайна S(x)
def spline_value(X, coefficients, x):
    a, b, c, d = coefficients
    j = find_segment(X, x)
    t = x - X[j]
    return a[j] + b[j] * t + c[j] * t**2 + d[j] * t**3


# Вторая производная сплайна
def spline_second_derivative(X, coefficients, x):
    _, _, c, d = coefficients
    j = find_segment(X, x)
    return 2 * c[j] + 6 * d[j] * (x - X[j])
