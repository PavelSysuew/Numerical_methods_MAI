"""Задание 1.4. Метод вращений Якоби для симметрических матриц.

В конспектах лекций этого метода нет; формулы — из учебника
Пантелеев А. В., Киселёв И. А. «Численные методы в примерах и задачах» (МАИ).

Идея: симметрическую матрицу A приводят к диагональному виду
последовательностью поворотов A⁽ᵏ⁺¹⁾ = Uₖᵀ A⁽ᵏ⁾ Uₖ. Каждый поворот обнуляет
максимальный по модулю внедиагональный элемент. На диагонали в итоге стоят
собственные значения, а столбцы матрицы V = U₀U₁…Uₖ — собственные векторы.
"""

import math

import numpy as np


def off_diagonal_norm(A):
    """Критерий остановки t(A) = sqrt(Σ_{i<j} a_ij²) — «размер» внедиагональной части."""
    n = A.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += A[i, j] ** 2
    return math.sqrt(total)


def max_off_diagonal(A):
    """Возвращает (i, j) максимального по модулю элемента выше главной диагонали."""
    n = A.shape[0]
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > abs(A[best_i, best_j]):
                best_i, best_j = i, j
    return best_i, best_j


def rotation_matrix(A, i, j):
    """Матрица вращения U, обнуляющая элемент a_ij в Uᵀ A U.

    Угол: φ = ½·arctg(2a_ij / (a_ii − a_jj)); при a_ii = a_jj φ = π/4.
    U — единичная матрица, у которой
      u_ii = u_jj = cos φ,  u_ij = −sin φ,  u_ji = sin φ.
    """
    if abs(A[i, i] - A[j, j]) <= 1e-12:
        phi = math.pi / 4
    else:
        phi = 0.5 * math.atan(2 * A[i, j] / (A[i, i] - A[j, j]))

    n = A.shape[0]
    U = np.eye(n)
    U[i, i] = math.cos(phi)
    U[j, j] = math.cos(phi)
    U[i, j] = -math.sin(phi)
    U[j, i] = math.sin(phi)
    return U


def jacobi_rotations(A, eps, max_iter=10000):
    """Находит собственные значения и векторы симметрической матрицы A.

    Итерации продолжаются, пока t(A⁽ᵏ⁾) > ε.
    Возвращает (eigenvalues, V, history):
      eigenvalues — диагональ последней A⁽ᵏ⁾;
      V           — матрица собственных векторов (вектор λ_i — столбец V[:, i]);
      history     — список t(A⁽ᵏ⁾) после каждой итерации, число итераций — len(history).
    """
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, а её размер {A.shape}.")
    if not np.allclose(A, A.T):
        raise ValueError("Метод вращений применим только к симметрической матрице: должно быть a_ij = a_ji.")

    n = A.shape[0]
    V = np.eye(n)
    history = []

    while n > 1 and off_diagonal_norm(A) > eps:
        if len(history) >= max_iter:
            raise ValueError(f"Метод вращений не сошёлся за {max_iter} итераций.")
        i, j = max_off_diagonal(A)
        U = rotation_matrix(A, i, j)
        A = U.T @ A @ U  # поворот: a_ij и a_ji становятся нулями
        V = V @ U        # накапливаем произведение матриц вращения
        history.append(off_diagonal_norm(A))

    eigenvalues = np.array([A[k, k] for k in range(n)])
    return eigenvalues, V, history
