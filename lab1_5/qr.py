import cmath
import math

import numpy as np


# QR-разложение методом Хаусхолдера, A = QR
def householder_qr(A):
    R = np.array(A, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, а её размер {R.shape}.")

    n = R.shape[0]
    Q = np.eye(n)
    for k in range(n - 1):
        column_norm = math.sqrt(sum(R[i, k] ** 2 for i in range(k, n)))
        if column_norm <= 1e-15:
            continue  # под диагональю уже нули

        # Вектор v: v_k = a_kk + sign(a_kk) * ||столбец||, ниже — элементы столбца
        sign = 1.0 if R[k, k] >= 0 else -1.0
        v = np.zeros(n)
        v[k] = R[k, k] + sign * column_norm
        for i in range(k + 1, n):
            v[i] = R[i, k]

        H = np.eye(n) - 2 * np.outer(v, v) / (v @ v)  # H = E - 2vv^T / (v^T v)
        R = H @ R
        Q = Q @ H  # H^(-1) = H, поэтому Q = H_1 H_2 ... H_(n-1)
    return Q, R


# Норма части столбца column начиная со строки row
def below_norm(A, row, column):
    n = A.shape[0]
    return math.sqrt(sum(A[i, column] ** 2 for i in range(row, n)))


# Корни блока 2x2: lambda^2 - (след) * lambda + det = 0
def block_eigenvalues(A, j):
    trace = A[j, j] + A[j + 1, j + 1]
    det = A[j, j] * A[j + 1, j + 1] - A[j, j + 1] * A[j + 1, j]
    root = cmath.sqrt(trace ** 2 - 4 * det)  # cmath — чтобы получить комплексные корни
    return (trace + root) / 2, (trace - root) / 2


# Собственные значения текущей матрицы: a_jj или корни блока 2x2
def current_eigenvalues(A, eps):
    n = A.shape[0]
    eigenvalues = []
    structure_ok = True  # всё ли под диагональю и под блоками меньше eps
    j = 0
    while j < n:
        if j == n - 1 or below_norm(A, j + 1, j) <= eps:
            eigenvalues.append(complex(A[j, j]))  # вещественное
            j += 1
        else:
            eigenvalues.extend(block_eigenvalues(A, j))  # пара из блока 2x2
            if j + 2 < n and (below_norm(A, j + 2, j) > eps or below_norm(A, j + 2, j + 1) > eps):
                structure_ok = False
            j += 2
    return eigenvalues, structure_ok


# QR-алгоритм: A_k = Q_k R_k, A_(k+1) = R_k Q_k
def qr_algorithm(A, eps, max_iter=10000):
    A_k = np.array(A, dtype=float)
    previous = None
    for iteration in range(1, max_iter + 1):
        Q, R = householder_qr(A_k)
        A_k = R @ Q
        eigenvalues, structure_ok = current_eigenvalues(A_k, eps)
        # Стоп: под блоками нули и собственные значения почти не меняются
        if structure_ok and previous is not None and len(previous) == len(eigenvalues):
            change = max(abs(new - old) for new, old in zip(eigenvalues, previous))
            if change <= eps:
                return eigenvalues, A_k, iteration
        previous = eigenvalues
    raise ValueError(f"QR-алгоритм не сошёлся за {max_iter} итераций.")
