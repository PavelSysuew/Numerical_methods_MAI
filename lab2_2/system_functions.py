import numpy as np

# Система, вариант 19, параметр a = 1:
#   f1(x1, x2) = x1^2 - 2 lg x2 - 1 = 0
#   f2(x1, x2) = x1^2 - a x1 x2 + a = 0
A = 1
N = 2  # число уравнений и неизвестных
LN10 = np.log(10)  # (lg x)' = 1 / (x ln 10)

# Система для вывода в окне
DESCRIPTION = [
    "⎧ x1² − 2·lg x2 − 1 = 0",
    f"⎩ x1² − a·x1·x2 + a = 0,      a = {A}",
]


# Уравнения по отдельности: работают и с числами, и с сетками numpy (для графика)
def f1(x1, x2):
    return x1**2 - 2*np.log10(x2) - 1


def f2(x1, x2):
    return x1**2 - A*x1*x2 + A


# Вектор F(x) = (f1, f2)
def F(x):
    return np.array([f1(x[0], x[1]), f2(x[0], x[1])])


# Матрица Якоби J(x): в строке i — частные производные f_i по x1 и x2
def jacobian(x):
    x1, x2 = x
    return np.array([
        [2*x1,          -2 / (x2 * LN10)],
        [2*x1 - A*x2,   -A*x1],
    ])


# Вид x = phi(x) для метода простой итерации:
#   из f1: x1 = sqrt(1 + 2 lg x2)
#   из f2: a x1 x2 = x1^2 + a  =>  x2 = x1 / a + 1 / x1
def phi(x):
    x1, x2 = x
    return np.array([np.sqrt(1 + 2*np.log10(x2)), x1 / A + 1 / x1])


# Матрица производных phi: по ней проверяется условие сходимости ||phi'(x)|| < 1
def phi_jacobian(x):
    x1, x2 = x
    return np.array([
        [0.0,                 1 / (x2 * LN10 * np.sqrt(1 + 2*np.log10(x2)))],
        [1 / A - 1 / x1**2,   0.0],
    ])
