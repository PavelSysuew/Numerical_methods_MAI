import os
import sys

import numpy as np

# Линейную систему с матрицей Якоби решаем своим LU-разложением из ЛР 1.1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab1_1"))
from lu import LU, solve_lu

# Защита от бесконечного цикла, если метод не сходится
MAX_ITERATIONS = 1000


# Норма вектора: максимум модулей
def vector_norm(v):
    return np.max(np.abs(v))


# Норма матрицы: максимум сумм модулей по строкам (согласована с нормой вектора)
def matrix_norm(M):
    return max(np.sum(np.abs(row)) for row in M)


# Лежит ли точка x в области G; bounds — список пар (нижняя, верхняя граница) по каждой переменной
def in_region(x, bounds):
    return all(low <= xi <= high for xi, (low, high) in zip(x, bounds))


# Метод простой итерации: x^(k+1) = phi(x^(k)), остановка при ||x^(k+1) - x^(k)|| < eps
# Возвращает (x, history): x — найденное решение,
# history — список пар (x^(k), ||x^(k) - x^(k-1)||) для k = 1, 2, ...
def simple_iteration(phi, x0, eps, bounds):
    history = []
    x = np.array(x0, dtype=float)
    for _ in range(MAX_ITERATIONS):
        x_new = phi(x)
        # Вне G условие ||phi'(x)|| < 1 не проверено, а lg x2 может быть не определён
        if not in_region(x_new, bounds):
            raise ValueError(f"Простая итерация: приближение {x_new} вышло из области G.")

        error = vector_norm(x_new - x)
        history.append((x_new, error))
        if error < eps:
            return x_new, history
        x = x_new

    raise ValueError(f"Простая итерация не сошлась за {MAX_ITERATIONS} итераций.")


# Метод Ньютона: x^(k+1) = x^(k) - J^(-1)(x^(k)) F(x^(k))
# J^(-1) не вычисляем: решаем J(x^(k)) dx = -F(x^(k)), тогда x^(k+1) = x^(k) + dx
# Возвращает то же, что simple_iteration
def newton(F, J, x0, eps, bounds):
    history = []
    x = np.array(x0, dtype=float)
    for _ in range(MAX_ITERATIONS):
        try:
            L, U, perm, _ = LU(J(x))
        except ValueError:
            raise ValueError(f"Метод Ньютона: матрица Якоби вырождена в точке {x}.") from None
        dx = solve_lu(L, U, perm, -F(x))
        x_new = x + dx
        if not in_region(x_new, bounds):
            raise ValueError(f"Метод Ньютона: приближение {x_new} вышло из области G.")

        error = vector_norm(dx)
        history.append((x_new, error))
        if error < eps:
            return x_new, history
        x = x_new

    raise ValueError(f"Метод Ньютона не сошёлся за {MAX_ITERATIONS} итераций.")
