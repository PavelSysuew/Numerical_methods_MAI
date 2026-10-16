import math

import numpy as np


# Критерий остановки: t(A) = sqrt(сумма a_ij^2 над диагональю)
def off_diagonal_norm(A):
    n = A.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += A[i, j] ** 2
    return math.sqrt(total)


# Поиск максимального по модулю элемента над диагональю
def max_off_diagonal(A):
    n = A.shape[0]
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > abs(A[best_i, best_j]):
                best_i, best_j = i, j
    return best_i, best_j


# Матрица вращения U, которая обнуляет a_ij
def rotation_matrix(A, i, j):
    # phi = 1/2 * arctg(2a_ij / (a_ii - a_jj)), при a_ii = a_jj phi = pi/4
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


# Метод вращений: собственные значения и векторы симметрической матрицы
def jacobi_rotations(A, eps, max_iter=10000):
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, а её размер {A.shape}.")
    if not np.allclose(A, A.T):
        raise ValueError("Метод вращений применим только к симметрической матрице: должно быть a_ij = a_ji.")

    n = A.shape[0]
    V = np.eye(n)  # сюда собираем произведение матриц вращения
    history = []   # t(A) после каждой итерации

    while n > 1 and off_diagonal_norm(A) > eps:
        if len(history) >= max_iter:
            raise ValueError(f"Метод вращений не сошёлся за {max_iter} итераций.")
        i, j = max_off_diagonal(A)
        U = rotation_matrix(A, i, j)
        A = U.T @ A @ U  # после поворота a_ij = a_ji = 0
        V = V @ U
        history.append(off_diagonal_norm(A))

    # Собственные значения — диагональ, собственные векторы — столбцы V
    eigenvalues = np.array([A[k, k] for k in range(n)])
    return eigenvalues, V, history
