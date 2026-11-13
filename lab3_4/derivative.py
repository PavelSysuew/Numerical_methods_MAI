# Численное дифференцирование таблично заданной функции.
# Производные берутся от интерполяционного многочлена: первой степени (по 2 узлам)
# или второй степени (по 3 узлам x_i, x_(i+1), x_(i+2)).


# Разделённая разность (наклон хорды) на отрезке [x_i, x_(i+1)]
def slope(X, Y, i):
    return (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])


# Первая производная с первым порядком точности: y' ≈ (y_(i+1) − y_i) / (x_(i+1) − x_i)
def first_order(X, Y, i):
    return slope(X, Y, i)


# Парабола через узлы x_i, x_(i+1), x_(i+2) (многочлен Ньютона 2-й степени) — для графика
def parabola(X, Y, i, x):
    second = (slope(X, Y, i + 1) - slope(X, Y, i)) / (X[i + 2] - X[i])
    return Y[i] + slope(X, Y, i) * (x - X[i]) + second * (x - X[i]) * (x - X[i + 1])


# Первая производная со вторым порядком точности — производная параболы:
#   y' ≈ Δ_i + (Δ_(i+1) − Δ_i) / (x_(i+2) − x_i) · (2x − x_i − x_(i+1)),  Δ_i — наклон на [x_i, x_(i+1)]
def second_order(X, Y, i, x):
    return slope(X, Y, i) + (slope(X, Y, i + 1) - slope(X, Y, i)) / (X[i + 2] - X[i]) * (2 * x - X[i] - X[i + 1])


# Вторая производная — вторая производная параболы (постоянна):
#   y'' ≈ 2 (Δ_(i+1) − Δ_i) / (x_(i+2) − x_i)
def second_derivative(X, Y, i):
    return 2 * (slope(X, Y, i + 1) - slope(X, Y, i)) / (X[i + 2] - X[i])


# Все производные в точке x. Возвращает словарь:
#   "left", "right" — односторонние (если x — узел), "segment" — по отрезку с x (если x не узел),
#   "start" — номер i первого узла параболы, "first" — y' 2-го порядка, "second" — y''
def derivatives_at(X, Y, x):
    n = len(X) - 1
    if n < 2:
        raise ValueError("Нужно хотя бы 3 узла: вторая производная берётся по параболе.")
    if not X[0] <= x <= X[-1]:
        raise ValueError(f"Точка {x:g} вне таблицы [{X[0]:g}, {X[-1]:g}].")

    result = {}
    node = next((k for k in range(n + 1) if abs(X[k] - x) < 1e-12), None)
    if node is not None:
        # x совпадает с узлом: левая и правая разности, парабола по узлу и двум соседям
        if node > 0:
            result["left"] = first_order(X, Y, node - 1)
        if node < n:
            result["right"] = first_order(X, Y, node)
        start = node - 1
    else:
        # x внутри отрезка [x_j, x_(j+1)]: парабола по этому отрезку и следующему узлу
        start = next(j for j in range(n) if x < X[j + 1])
        result["segment"] = first_order(X, Y, start)

    # Парабола должна целиком лежать в таблице: узлы start, start + 1, start + 2
    start = min(max(start, 0), n - 2)
    result["start"] = start
    result["first"] = second_order(X, Y, start, x)
    result["second"] = second_derivative(X, Y, start)
    return result
