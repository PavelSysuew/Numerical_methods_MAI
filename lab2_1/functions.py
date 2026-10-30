# Уравнение варианта 19 — для вывода в окне
EQUATION = "x⁴ − 2x − 1 = 0"


# Заданная функция
def f(x):
    return x**4 - 2*x - 1


# Первая и вторая производные f
def derivative_f(x):
    return 4*x**3 - 2


def derivative_2_f(x):
    return 12*x**2


# Функция фи для метода простых итераций: x = (2x + 1)^(1/4)
def fi_MSI(x):
    return (2*x + 1) ** (1 / 4)


# Производная фи, для проверки условия сходимости |fi'(x)| < 1
def derivative_fi_MSI(x):
    return (1/4) * (2*x + 1)**(-3/4) * 2


# Производная фи для метода Ньютона: fi'(x) = f(x) * f''(x) / f'(x)^2
def derivative_fi_newton(x):
    return f(x) * derivative_2_f(x) / (derivative_f(x) ** 2)
