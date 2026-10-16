import numpy as np

# Решает трёхдиагональную систему методом прогонки
def tridiagonale(a, diag, c, d):
    a = np.array(a, dtype=float)
    diag = np.array(diag, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    n = len(diag)  # Размерность системы
    if not (len(a) == len(c) == len(d) == n):
        raise ValueError("Массивы a, диагональ, c и d должны быть одной длины.")
    if a[0] != 0 or c[n - 1] != 0:
        raise ValueError("В первом уравнении нет x₀, в последнем нет xₙ₊₁: a₁ и cₙ должны быть равны 0.")

    b = -diag

    # P[0] = Q[0] = 0 — стартовые значения 
    P = np.zeros(n + 1)
    Q = np.zeros(n + 1)

    # Прямой ход:
    for i in range(n):
        denominator = b[i] - a[i] * P[i]
        if abs(denominator) <= 1e-12:  # Условие корректности: b_i − a_i P_i ≠ 0
            raise ValueError(f"Знаменатель в уравнении {i + 1} равен нулю, метод прогонки неприменим.")
        P[i + 1] = c[i] / denominator
        Q[i + 1] = (a[i] * Q[i] - d[i]) / denominator

    x = np.zeros(n)
    x[n - 1] = Q[n]
    for i in range(n - 2, -1, -1):
        x[i] = P[i + 1] * x[i + 1] + Q[i + 1]

    return P, Q, x


# Проверяет достаточное условие корректности и устойчивости прогонки
def has_diagonal_dominance(a, diag, c):
    a, diag, c = np.abs(a), np.abs(diag), np.abs(c)
    return bool(np.all(diag >= a + c) and np.any(diag > a + c))


# Невязка
def residual(a, diag, c, d, x):
    n = len(x)
    r = np.zeros(n)
    for i in range(n):
        r[i] = diag[i] * x[i] - d[i]
        if i > 0:
            r[i] += a[i] * x[i - 1]
        if i < n - 1:
            r[i] += c[i] * x[i + 1]
    return r
