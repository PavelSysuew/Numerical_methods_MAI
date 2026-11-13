# Производная ω в узле
def omega_derivative(X, i):
    result = 1.0
    for j in range(len(X)):
        if j != i:
            result *= X[i] - X[j]
    return result


# Многочлен Лагранжа
def lagrange(X, Y, x):
    total = 0.0
    for i in range(len(X)):
        term = Y[i]
        for j in range(len(X)):
            if j != i:
                term = term * (x - X[j]) / (X[i] - X[j])
        total = total + term
    return total


# Таблица разделённых разностей
def divided_differences(X, Y):
    table = [list(Y)]
    for k in range(1, len(X)):
        previous = table[-1]
        table.append([(previous[i + 1] - previous[i]) / (X[i + k] - X[i]) for i in range(len(X) - k)])
    return table


# Многочлен Ньютона
def newton(X, table, x):
    total = 0.0
    product = 1.0
    for k in range(len(X)):
        total = total + table[k][0] * product
        product = product * (x - X[k])
    return total
