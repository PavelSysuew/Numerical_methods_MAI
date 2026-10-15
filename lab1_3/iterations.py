"""Задание 1.3. Метод простых итераций и метод Зейделя.

Формулы МПИ — из конспекта ЛК_1, раздел «Метод простых итераций»;
метод Зейделя — из docs/зейдель.md.

Нормы: матрицы — максимум сумм модулей по строкам (ЛК_1, норма ‖A‖ₑ),
вектора — согласованная с ней максимум модулей компонент.
"""

import numpy as np

# Если приближение стало больше по норме, считаем, что итерации расходятся
DIVERGENCE_LIMIT = 1e100


def matrix_norm(M):
    """Норма матрицы: максимальная среди сумм модулей элементов строк."""
    return max(np.sum(np.abs(row)) for row in M)


def vector_norm(v):
    """Норма вектора, согласованная с matrix_norm: максимальный модуль компоненты."""
    return np.max(np.abs(v))


def to_iteration_form(A, b):
    """Приводит систему Ax = b к виду x = αx + β выделением диагональных элементов.

    Из i-го уравнения выражается x_i:
      β_i = b_i / a_ii,   α_ij = −a_ij / a_ii (i ≠ j),   α_ii = 0.
    Возвращает (alpha, beta).
    """
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
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if j != i:
                alpha[i, j] = -A[i, j] / A[i, i]
    return alpha, beta


def stop_coefficient(norm_part, norm_alpha):
    """Множитель в оценке погрешности ‖x* − x⁽ᵏ⁾‖ ≤ коэффициент · ‖x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾‖.

    При ‖α‖ ≥ 1 оценка неприменима, и останавливаемся просто по ‖x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾‖ ≤ ε.
    """
    if norm_alpha < 1:
        return norm_part / (1 - norm_alpha)
    return 1.0


def simple_iteration(alpha, beta, eps, max_iter=10000):
    """Метод простых итераций: x⁽ᵏ⁺¹⁾ = αx⁽ᵏ⁾ + β, начальное приближение x⁽⁰⁾ = β.

    Остановка, когда оценка погрешности ‖α‖/(1 − ‖α‖) · ‖x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾‖ ≤ ε (ЛК_1).
    Возвращает (x, history), где history[k] = (x⁽ᵏ⁺¹⁾, оценка погрешности);
    число итераций — len(history).
    """
    norm_alpha = matrix_norm(alpha)
    coefficient = stop_coefficient(norm_alpha, norm_alpha)

    x = beta.copy()
    history = []
    for _ in range(max_iter):
        x_new = alpha @ x + beta
        if vector_norm(x_new) > DIVERGENCE_LIMIT:
            break  # приближения неограниченно растут — процесс расходится
        error = coefficient * vector_norm(x_new - x)
        history.append((x_new, error))
        x = x_new
        if error <= eps:
            return x, history
    raise ValueError(f"Метод простых итераций не сошёлся (выполнено итераций: {len(history)}).")


def seidel(alpha, beta, eps, max_iter=10000):
    """Метод Зейделя: x⁽ᵏ⁺¹⁾ = Bx⁽ᵏ⁺¹⁾ + Cx⁽ᵏ⁾ + β, начальное приближение x⁽⁰⁾ = β.

    B — элементы α ниже главной диагонали, C — на диагонали и выше.
    Формулу x⁽ᵏ⁺¹⁾ = (E − B)⁻¹Cx⁽ᵏ⁾ + (E − B)⁻¹β считаем покомпонентно:
    x_i⁽ᵏ⁺¹⁾ = β_i + Σ_{j<i} α_ij x_j⁽ᵏ⁺¹⁾ + Σ_{j≥i} α_ij x_j⁽ᵏ⁾ —
    это прямой ход для нижней треугольной матрицы E − B, обращать её не нужно.

    Остановка, когда ‖C‖/(1 − ‖α‖) · ‖x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾‖ ≤ ε. Этой оценки нет
    в docs/зейдель.md, она из учебника (Пантелеев, «Численные методы в
    примерах и задачах»).
    Возвращает (x, history) так же, как simple_iteration.
    """
    n = len(beta)
    C = np.triu(alpha)  # элементы на диагонали и выше (диагональ α нулевая)
    coefficient = stop_coefficient(matrix_norm(C), matrix_norm(alpha))

    x = beta.copy()
    history = []
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            # B x⁽ᵏ⁺¹⁾: уже найденные на этой итерации x_1..x_{i-1}
            # C x⁽ᵏ⁾:   старые значения x_i..x_n
            x_new[i] = beta[i] + alpha[i, :i] @ x_new[:i] + alpha[i, i:] @ x[i:]
        if vector_norm(x_new) > DIVERGENCE_LIMIT:
            break
        error = coefficient * vector_norm(x_new - x)
        history.append((x_new, error))
        x = x_new
        if error <= eps:
            return x, history
    raise ValueError(f"Метод Зейделя не сошёлся (выполнено итераций: {len(history)}).")
