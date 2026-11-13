import numpy as np


# Узлы x_0 = a, x_1 = a + h, ..., x_N = b; шаг должен укладываться в [a, b] целое число раз
def grid(a, b, h):
    count = round((b - a) / h)
    if count < 1 or abs(count * h - (b - a)) > 1e-9:
        raise ValueError(f"Шаг h = {h:g} не укладывается в [{a:g}, {b:g}] целое число раз.")
    return np.linspace(a, b, count + 1)


# Метод левых прямоугольников: высота прямоугольника — значение в левом конце отрезка
#   F ≈ h · (f_0 + f_1 + ... + f_(N−1))
def left_rectangles(f, a, b, h):
    return h * np.sum(f(grid(a, b, h)[:-1]))


# Метод правых прямоугольников: значение в правом конце отрезка
#   F ≈ h · (f_1 + f_2 + ... + f_N)
def right_rectangles(f, a, b, h):
    return h * np.sum(f(grid(a, b, h)[1:]))


# Метод средних прямоугольников: значение в середине отрезка
#   F ≈ h · Σ f(x_(i−1) + h/2)
def middle_rectangles(f, a, b, h):
    x = grid(a, b, h)
    return h * np.sum(f(x[:-1] + h / 2))


# Метод трапеций: F ≈ h · (f_0/2 + f_1 + ... + f_(N−1) + f_N/2)
def trapezoids(f, a, b, h):
    y = f(grid(a, b, h))
    return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)


# Метод Симпсона: F ≈ h/3 · (f_0 + 4f_1 + 2f_2 + 4f_3 + ... + 4f_(N−1) + f_N), N чётное
def simpson(f, a, b, h):
    y = f(grid(a, b, h))
    if (len(y) - 1) % 2 != 0:
        raise ValueError(f"Для метода Симпсона число отрезков должно быть чётным, а при h = {h:g} оно "
                         f"равно {len(y) - 1}.")
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])


# Метод Рунге–Ромберга: уточнение по двум расчётам с шагами h и kh, p — порядок точности метода
#   F ≈ F_h + (F_h − F_kh) / (k^p − 1),   оценка погрешности F_h: |F_h − F_kh| / (k^p − 1)
def runge_romberg(F_h, F_kh, k, p):
    correction = (F_h - F_kh) / (k**p - 1)
    return F_h + correction, abs(correction)
