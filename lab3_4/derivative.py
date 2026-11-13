# Наклон хорды
def slope(X, Y, i):
    return (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])


# Первая производная, первый порядок
def first_order(X, Y, i):
    return slope(X, Y, i)


# Парабола через три узла (для графика)
def parabola(X, Y, i, x):
    second = (slope(X, Y, i + 1) - slope(X, Y, i)) / (X[i + 2] - X[i])
    return Y[i] + slope(X, Y, i) * (x - X[i]) + second * (x - X[i]) * (x - X[i + 1])


# Первая производная, второй порядок
def second_order(X, Y, i, x):
    return slope(X, Y, i) + (slope(X, Y, i + 1) - slope(X, Y, i)) / (X[i + 2] - X[i]) * (2 * x - X[i] - X[i + 1])


# Вторая производная
def second_derivative(X, Y, i):
    return 2 * (slope(X, Y, i + 1) - slope(X, Y, i)) / (X[i + 2] - X[i])


# Все производные в точке x
def derivatives_at(X, Y, x):
    n = len(X) - 1
    if n < 2:
        raise ValueError("Нужно хотя бы 3 узла: вторая производная берётся по параболе.")
    if not X[0] <= x <= X[-1]:
        raise ValueError(f"Точка {x:g} вне таблицы [{X[0]:g}, {X[-1]:g}].")

    result = {}
    node = next((k for k in range(n + 1) if abs(X[k] - x) < 1e-12), None)
    if node is not None:
        # x совпадает с узлом
        if node > 0:
            result["left"] = first_order(X, Y, node - 1)
        if node < n:
            result["right"] = first_order(X, Y, node)
        start = node - 1
    else:
        # x между узлами
        start = next(j for j in range(n) if x < X[j + 1])
        result["segment"] = first_order(X, Y, start)

    # Парабола не должна выходить за таблицу
    start = min(max(start, 0), n - 2)
    result["start"] = start
    result["first"] = second_order(X, Y, start, x)
    result["second"] = second_derivative(X, Y, start)
    return result
