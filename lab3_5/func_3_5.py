import numpy as np

# Функция из варианта
DESCRIPTION = "y = x² / (625 − x⁴)"


def f(x):
    return x**2 / (625 - x**4)


# Точный интеграл через первообразную (для проверки)
def antiderivative(x):
    return np.log((5 + x) / (5 - x)) / 20 - np.arctan(x / 5) / 10


def exact_integral(a, b):
    return antiderivative(b) - antiderivative(a)
