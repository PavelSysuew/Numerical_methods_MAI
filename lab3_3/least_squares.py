import os
import sys

import numpy as np

# LU-разложение из ЛР 1.1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab1_1"))
from lu import LU, solve_lu


# Нормальная система МНК
def normal_system(X, Y, degree):
    size = degree + 1
    A = np.array([[np.sum(X ** (i + j)) for j in range(size)] for i in range(size)])
    b = np.array([np.sum(Y * X ** i) for i in range(size)])
    return A, b


# Коэффициенты многочлена по МНК
def least_squares(X, Y, degree):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    A, b = normal_system(X, Y, degree)
    L, U, perm, _ = LU(A)
    return solve_lu(L, U, perm, b), A, b


# Значение многочлена
def polynomial(coefficients, x):
    return sum(a * x ** j for j, a in enumerate(coefficients))


# Сумма квадратов ошибок
def squared_error_sum(coefficients, X, Y):
    return np.sum((polynomial(coefficients, np.array(X)) - np.array(Y)) ** 2)


# Среднеквадратичная ошибка MSE
def mean_squared_error(coefficients, X, Y):
    return squared_error_sum(coefficients, X, Y) / len(X)
