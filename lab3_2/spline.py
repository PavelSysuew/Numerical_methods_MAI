import os
import sys

import numpy as np

# Система для коэффициентов c трёхдиагональная — решаем прогонкой из ЛР 1.2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab1_2"))
from tridiagonale import tridiagonale


# Кубический сплайн с нулевой кривизной на концах: S''(x_0) = S''(x_n) = 0
# На отрезке [x_(j), x_(j+1)], j = 0, ..., n − 1:
#   S(x) = a_j + b_j (x − x_j) + c_j (x − x_j)² + d_j (x − x_j)³
# Возвращает (a, b, c, d) — массивы длины n — и систему для c (lower, diag, upper, rhs)
def cubic_spline(X, Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    n = len(X) - 1
    h = np.diff(X)  # h[j] = x_(j+1) − x_j

    # Уравнения для c_1, ..., c_(n−1) (c_0 = 0 — нулевая кривизна слева), по одному на внутренний узел x_j:
    #   h_(j−1) c_(j−1) + 2(h_(j−1) + h_j) c_j + h_j c_(j+1) = 3[(f_(j+1) − f_j)/h_j − (f_j − f_(j−1))/h_(j−1)]
    m = n - 1
    lower, diag, upper, rhs = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)
    for k in range(m):
        j = k + 1
        lower[k] = h[j - 1] if k > 0 else 0.0     # при c_0 = 0 слагаемое пропадает
        diag[k] = 2 * (h[j - 1] + h[j])
        upper[k] = h[j] if k < m - 1 else 0.0     # при c_n = 0 слагаемое пропадает
        rhs[k] = 3 * ((Y[j + 1] - Y[j]) / h[j] - (Y[j] - Y[j - 1]) / h[j - 1])

    _, _, c_inner = tridiagonale(lower, diag, upper, rhs)
    c = np.concatenate(([0.0], c_inner, [0.0]))  # c_0 = 0 и c_n = 0 (нулевая кривизна справа)

    a = Y[:-1]
    b = (Y[1:] - Y[:-1]) / h - h * (c[1:] + 2 * c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)
    return a, b, c[:-1], d, (lower, diag, upper, rhs)


# Номер отрезка [x_j, x_(j+1)], на котором лежит x
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


# Вторая производная сплайна S''(x) = 2c_j + 6d_j (x − x_j) — для проверки нулевой кривизны
def spline_second_derivative(X, coefficients, x):
    _, _, c, d = coefficients
    j = find_segment(X, x)
    return 2 * c[j] + 6 * d[j] * (x - X[j])
