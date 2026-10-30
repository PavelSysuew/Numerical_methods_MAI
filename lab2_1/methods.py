# Защита от бесконечного цикла, если метод не сходится
MAX_ITERATIONS = 1000


# Метод простой итерации
def simple_iteration(fi, x0, eps, a, b):
    history = []
    x = x0
    for _ in range(MAX_ITERATIONS):
        x_new = fi(x)
        if not a <= x_new <= b:
            raise ValueError(f"Простая итерация: приближение x = {x_new} вышло за отрезок [{a}, {b}].")

        error = abs(x_new - x)
        history.append((x_new, error))
        if error < eps:
            return x_new, history
        x = x_new

    raise ValueError(f"Простая итерация не сошлась за {MAX_ITERATIONS} итераций.")


# Метод Ньютона
def newton(f, df, x0, eps, a, b):
    history = []
    x = x0
    for _ in range(MAX_ITERATIONS):
        # Касательная горизонтальна — шаг сделать нельзя
        if df(x) == 0:
            raise ValueError(f"Метод Ньютона: f'(x) = 0 в точке x = {x}.")
        x_new = x - f(x) / df(x)
        # Пологая касательная может «выстрелить» за пределы отрезка
        if not a <= x_new <= b:
            raise ValueError(f"Метод Ньютона: приближение x = {x_new} вышло за отрезок [{a}, {b}].")

        error = abs(x_new - x)
        history.append((x_new, error))
        if error < eps:
            return x_new, history
        x = x_new

    raise ValueError(f"Метод Ньютона не сошёлся за {MAX_ITERATIONS} итераций.")
