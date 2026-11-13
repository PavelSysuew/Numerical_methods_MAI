import os
import sys

import numpy as np

# Нормальную систему решаем своим LU-разложением из ЛР 1.1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab1_1"))
from lu import LU, solve_lu


# Нормальная система МНК для многочлена F(x) = a_0 + a_1 x + ... + a_m x^m:
#   Σ_j a_j · Σ_k x_k^(i+j) = Σ_k y_k · x_k^i,   i = 0, ..., m
def normal_system(X, Y, degree):
    size = degree + 1
    A = np.array([[np.sum(X ** (i + j)) for j in range(size)] for i in range(size)])
    b = np.array([np.sum(Y * X ** i) for i in range(size)])
    return A, b


# Коэффициенты приближающего многочлена a_0, ..., a_m и нормальная система (для вывода)
def least_squares(X, Y, degree):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    A, b = normal_system(X, Y, degree)
    L, U, perm, _ = LU(A)
    return solve_lu(L, U, perm, b), A, b


# Значение многочлена F(x) = Σ a_j x^j (x — число или массив)
def polynomial(coefficients, x):
    return sum(a * x ** j for j, a in enumerate(coefficients))


# Сумма квадратов ошибок Φ = Σ (F(x_k) − y_k)²
def squared_error_sum(coefficients, X, Y):
    return np.sum((polynomial(coefficients, np.array(X)) - np.array(Y)) ** 2)


# Среднеквадратичная ошибка MSE = Φ / N — среднее квадрата ошибки по всем N точкам
def mean_squared_error(coefficients, X, Y):
    return squared_error_sum(coefficients, X, Y) / len(X)
