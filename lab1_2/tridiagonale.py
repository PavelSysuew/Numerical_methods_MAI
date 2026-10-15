"""Задание 1.2. Метод прогонки для СЛАУ с трёхдиагональной матрицей.

Матрица целиком не хранится: только три диагонали и правая часть.
Формулы — из конспекта ЛК_1, раздел «Метод прогонки».
"""

import numpy as np


def tridiagonale(a, diag, c, d):
    """Решает трёхдиагональную систему методом прогонки.

    i-е уравнение системы:  a[i]*x[i-1] + diag[i]*x[i] + c[i]*x[i+1] = d[i]
      a    — поддиагональ, коэффициенты при x[i-1] (a[0] = 0);
      diag — главная диагональ, как она стоит в матрице;
      c    — наддиагональ, коэффициенты при x[i+1] (c[n-1] = 0);
      d    — правая часть.
    Все четыре массива длины n.

    Возвращает (P, Q, x): прогоночные коэффициенты P, Q (длины n + 1,
    P[k] в коде — это P_(k+1) в лекции) и решение x (длины n).
    """
    a = np.array(a, dtype=float)
    diag = np.array(diag, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    n = len(diag)  # Размерность системы
    if not (len(a) == len(c) == len(d) == n):
        raise ValueError("Массивы a, диагональ, c и d должны быть одной длины.")
    if a[0] != 0 or c[n - 1] != 0:
        raise ValueError("В первом уравнении нет x₀, в последнем нет xₙ₊₁: a₁ и cₙ должны быть равны 0.")

    # В лекции канонический вид a_i x_{i-1} − b_i x_i + c_i x_{i+1} = d_i,
    # то есть на диагонали стоит −b_i
    b = -diag

    # P[0] = Q[0] = 0 — стартовые значения (P₁ = Q₁ = 0 в лекции)
    P = np.zeros(n + 1)
    Q = np.zeros(n + 1)

    # Прямой ход: P_{i+1} = c_i / (b_i − a_i P_i),  Q_{i+1} = (a_i Q_i − d_i) / (b_i − a_i P_i)
    for i in range(n):
        denominator = b[i] - a[i] * P[i]
        if abs(denominator) <= 1e-12:  # Условие корректности: b_i − a_i P_i ≠ 0
            raise ValueError(f"Знаменатель в уравнении {i + 1} равен нулю, метод прогонки неприменим.")
        P[i + 1] = c[i] / denominator
        Q[i + 1] = (a[i] * Q[i] - d[i]) / denominator

    # Обратный ход: x_n = Q_{n+1},  x_i = P_{i+1} x_{i+1} + Q_{i+1}
    x = np.zeros(n)
    x[n - 1] = Q[n]
    for i in range(n - 2, -1, -1):
        x[i] = P[i + 1] * x[i + 1] + Q[i + 1]

    return P, Q, x


def has_diagonal_dominance(a, diag, c):
    """Проверяет достаточное условие корректности и устойчивости прогонки:
    |diag_i| >= |a_i| + |c_i| для всех i, причём хотя бы для одного i строго.
    """
    a, diag, c = np.abs(a), np.abs(diag), np.abs(c)
    return bool(np.all(diag >= a + c) and np.any(diag > a + c))


def residual(a, diag, c, d, x):
    """Невязка r_i = a_i x_{i-1} + diag_i x_i + c_i x_{i+1} − d_i — без построения полной матрицы."""
    n = len(x)
    r = np.zeros(n)
    for i in range(n):
        r[i] = diag[i] * x[i] - d[i]
        if i > 0:
            r[i] += a[i] * x[i - 1]
        if i < n - 1:
            r[i] += c[i] * x[i + 1]
    return r
