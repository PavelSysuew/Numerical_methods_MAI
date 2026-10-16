

import cmath
import math

import numpy as np


def householder_qr(A):
    # QR-разложение матрицы A методом Хаусхолдера. Возвращает (Q, R)
    R = np.array(A, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, а её размер {R.shape}.")

    n = R.shape[0]
    Q = np.eye(n)
    for k in range(n - 1):
        column_norm = math.sqrt(sum(R[i, k] ** 2 for i in range(k, n)))
        if column_norm <= 1e-15:
            continue  # столбец под диагональю уже нулевой

        sign = 1.0 if R[k, k] >= 0 else -1.0  # знак выбираем так, чтобы не вычитать близкие числа
        v = np.zeros(n)
        v[k] = R[k, k] + sign * column_norm
        for i in range(k + 1, n):
            v[i] = R[i, k]

        H = np.eye(n) - 2 * np.outer(v, v) / (v @ v)
        R = H @ R
        Q = Q @ H  # H симметрична и ортогональна: H⁻¹ = H
    return Q, R


def below_norm(A, row, column):
    """sqrt(Σ a_i,column²) по строкам i ≥ row — норма части столбца под диагональю."""
    n = A.shape[0]
    return math.sqrt(sum(A[i, column] ** 2 for i in range(row, n)))


def block_eigenvalues(A, j):
    """Корни характеристического уравнения блока 2×2 в строках и столбцах j, j+1:
    (a_jj − λ)(a_{j+1,j+1} − λ) − a_{j,j+1}·a_{j+1,j} = 0, то есть
    λ² − (a_jj + a_{j+1,j+1})·λ + (a_jj·a_{j+1,j+1} − a_{j,j+1}·a_{j+1,j}) = 0.
    """
    trace = A[j, j] + A[j + 1, j + 1]
    det = A[j, j] * A[j + 1, j + 1] - A[j, j + 1] * A[j + 1, j]
    root = cmath.sqrt(trace ** 2 - 4 * det)
    return (trace + root) / 2, (trace - root) / 2


def current_eigenvalues(A, eps):
    """Собственные значения, которые «читаются» из текущей матрицы A⁽ᵏ⁾.

    Идём по диагонали: если столбец j под диагональю мал (≤ ε) — λ = a_jj
    вещественное; иначе берём блок 2×2 и его пару корней.
    Возвращает (eigenvalues, structure_ok): structure_ok = True, если под
    каждым диагональным элементом и под каждым блоком всё меньше ε.
    """
    n = A.shape[0]
    eigenvalues = []
    structure_ok = True
    j = 0
    while j < n:
        if j == n - 1 or below_norm(A, j + 1, j) <= eps:
            eigenvalues.append(complex(A[j, j]))
            j += 1
        else:
            eigenvalues.extend(block_eigenvalues(A, j))
            # Под блоком 2×2 (в столбцах j и j+1) тоже должны быть почти нули
            if j + 2 < n and (below_norm(A, j + 2, j) > eps or below_norm(A, j + 2, j + 1) > eps):
                structure_ok = False
            j += 2
    return eigenvalues, structure_ok


def qr_algorithm(A, eps, max_iter=10000):
    """QR-алгоритм: A⁽ᵏ⁺¹⁾ = R⁽ᵏ⁾Q⁽ᵏ⁾, где A⁽ᵏ⁾ = Q⁽ᵏ⁾R⁽ᵏ⁾.

    Остановка, когда матрица разделилась на вещественные элементы и блоки 2×2
    (всё под ними ≤ ε) и собственные значения перестали меняться:
    |λ_i⁽ᵏ⁾ − λ_i⁽ᵏ⁻¹⁾| ≤ ε для всех i. Второе условие нужно для комплексных
    пар: блок 2×2 не становится треугольным, но его корни сходятся.

    Возвращает (eigenvalues, A_k, iterations).
    """
    A_k = np.array(A, dtype=float)
    previous = None
    for iteration in range(1, max_iter + 1):
        Q, R = householder_qr(A_k)
        A_k = R @ Q
        eigenvalues, structure_ok = current_eigenvalues(A_k, eps)
        if structure_ok and previous is not None and len(previous) == len(eigenvalues):
            change = max(abs(new - old) for new, old in zip(eigenvalues, previous))
            if change <= eps:
                return eigenvalues, A_k, iteration
        previous = eigenvalues
    raise ValueError(f"QR-алгоритм не сошёлся за {max_iter} итераций.")
