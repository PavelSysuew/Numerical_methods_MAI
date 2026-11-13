# Все функции работают и с числом x, и с массивом numpy (для графика)


# ω'(x_i) = произведение (x_i − x_j) по всем j ≠ i
def omega_derivative(X, i):
    result = 1.0
    for j in range(len(X)):
        if j != i:
            result *= X[i] - X[j]
    return result


# Многочлен Лагранжа: L(x) = Σ f_i · Π_(j≠i) (x − x_j) / (x_i − x_j)
def lagrange(X, Y, x):
    total = 0.0
    for i in range(len(X)):
        term = Y[i]
        for j in range(len(X)):
            if j != i:
                term = term * (x - X[j]) / (X[i] - X[j])
        total = total + term
    return total


# Таблица разделённых разностей: table[k][i] = f(x_i, x_(i+1), ..., x_(i+k))
# table[0] — сами значения f_i
def divided_differences(X, Y):
    table = [list(Y)]
    for k in range(1, len(X)):
        previous = table[-1]
        table.append([(previous[i + 1] - previous[i]) / (X[i + k] - X[i]) for i in range(len(X) - k)])
    return table


# Многочлен Ньютона: P(x) = f(x_0) + f(x_0, x_1)(x − x_0) + f(x_0, x_1, x_2)(x − x_0)(x − x_1) + ...
# Коэффициенты — верхняя строка таблицы разделённых разностей: table[k][0]
def newton(X, table, x):
    total = 0.0
    product = 1.0  # (x − x_0)(x − x_1)...(x − x_(k−1))
    for k in range(len(X)):
        total = total + table[k][0] * product
        product = product * (x - X[k])
    return total
