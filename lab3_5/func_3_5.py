import numpy as np

# Подынтегральная функция варианта 19
DESCRIPTION = "y = x² / (625 − x⁴)"


def f(x):
    return x**2 / (625 - x**4)


# Точное значение интеграла — только для проверки численных методов.
# x² / (625 − x⁴) = ½ [1/(25 − x²) − 1/(25 + x²)], отсюда первообразная
#   F(x) = (1/20) ln((5 + x)/(5 − x)) − (1/10) arctg(x/5),  |x| < 5
def antiderivative(x):
    return np.log((5 + x) / (5 - x)) / 20 - np.arctan(x / 5) / 10


def exact_integral(a, b):
    return antiderivative(b) - antiderivative(a)
