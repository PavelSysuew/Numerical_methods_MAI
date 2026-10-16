import numpy as np

# Если норма приближения больше, считаем, что итерации расходятся
DIVERGENCE_LIMIT = 1e100


# Норма матрицы: максимум сумм модулей по строкам
def matrix_norm(M):
    return max(np.sum(np.abs(row)) for row in M)


# Норма вектора: максимум модулей (согласована с нормой матрицы)
def vector_norm(v):
    return np.max(np.abs(v))


# Приведение Ax = b к виду x = alpha*x + beta
def to_iteration_form(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, а её размер {A.shape}.")

    n = A.shape[0]
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        if abs(A[i, i]) <= 1e-12:
            raise ValueError(
                f"Диагональный элемент a{i + 1}{i + 1} равен нулю: уравнение {i + 1} "
                f"нельзя разрешить относительно x{i + 1}. Переставьте уравнения."
            )
        beta[i] = b[i] / A[i, i]  # beta_i = b_i / a_ii
        for j in range(n):
            if j != i:
                alpha[i, j] = -A[i, j] / A[i, i]  # alpha_ij = -a_ij / a_ii, alpha_ii = 0
    return alpha, beta


# Множитель в оценке погрешности: norm_part / (1 - ||alpha||)
def stop_coefficient(norm_part, norm_alpha):
    if norm_alpha < 1:
        return norm_part / (1 - norm_alpha)
    return 1.0  # при ||alpha|| >= 1 оценка не работает, останавливаемся по ||x_k - x_(k-1)||


# Метод простых итераций: x_(k+1) = alpha*x_k + beta
def simple_iteration(alpha, beta, eps, max_iter=10000):
    norm_alpha = matrix_norm(alpha)
    coefficient = stop_coefficient(norm_alpha, norm_alpha)

    x = beta.copy()  # начальное приближение x_0 = beta
    history = []     # (приближение, оценка погрешности) на каждой итерации
    for _ in range(max_iter):
        x_new = alpha @ x + beta
        if vector_norm(x_new) > DIVERGENCE_LIMIT:
            break  # расходится
        error = coefficient * vector_norm(x_new - x)
        history.append((x_new, error))
        x = x_new
        if error <= eps:
            return x, history
    raise ValueError(f"Метод простых итераций не сошёлся (выполнено итераций: {len(history)}).")


# Метод Зейделя: x_(k+1) = B*x_(k+1) + C*x_k + beta
def seidel(alpha, beta, eps, max_iter=10000):
    n = len(beta)
    C = np.triu(alpha)  # C — элементы на диагонали и выше, B — ниже диагонали
    # Для Зейделя в оценке погрешности ||C|| вместо ||alpha||
    coefficient = stop_coefficient(matrix_norm(C), matrix_norm(alpha))

    x = beta.copy()
    history = []
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            # Первая сумма — B*x_(k+1) (уже новые x), вторая — C*x_k (старые x)
            x_new[i] = beta[i] + alpha[i, :i] @ x_new[:i] + alpha[i, i:] @ x[i:]
        if vector_norm(x_new) > DIVERGENCE_LIMIT:
            break
        error = coefficient * vector_norm(x_new - x)
        history.append((x_new, error))
        x = x_new
        if error <= eps:
            return x, history
    raise ValueError(f"Метод Зейделя не сошёлся (выполнено итераций: {len(history)}).")
