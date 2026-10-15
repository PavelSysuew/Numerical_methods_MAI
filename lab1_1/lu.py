import numpy as np


# Функция LU разложения матрицы A на нижнюю и верхнюю треугольные матрицы L и U
def LU(A: np.ndarray):
    # LU-разложение определено только для квадратной матрицы
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, а её размер {A.shape}.")

    n = A.shape[0] # Размерность матрицы A
    L = np.zeros((A.shape[0], A.shape[1]))
    U = np.array(A, dtype=float) # Копируем матрицу A в U
    perm = np.arange(A.shape[0]) # Массив индексов перестановок

    s = 0 # Число перестановок строк

    for k in range(n): # k-ый шаг разложения. Бежим по столбцам с номером k
        # Находим индекс максимального элемента в столбце i
        max_index = np.argmax(np.abs(U[k:n, k])) + k

        # Если максимальный элемент не на диагонали, меняем строки местами
        if max_index != k:
            U[[k, max_index], :] = U[[max_index, k], :]
            L[[k, max_index], :] = L[[max_index, k], :]
            perm[[k, max_index]] = perm[[max_index, k]]
            s += 1 # Увеличиваем счетчик перестановок

        if abs(U[k, k]) <= 1e-12:  # Проверяем, не является ли максимальный элемент слишком малым    
            raise ValueError("Матрица вырождена, LU разложение невозможно.")
    
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k] # Вычисляем коэффициент для L
            for j in range(k, n):
                U[i, j] = U[i, j] - L[i, k] * U[k, j] # Обновляем элементы U

    for i in range(n):
                L[i, i] = 1.0 # Заполняем диагональ L единицами

    return L, U, perm, s

def solve_lu(L: np.ndarray, U: np.ndarray, perm: np.ndarray, b: np.ndarray):
    n = L.shape[0]
    # Применяем перестановки к вектору b
    b_permuted = b[perm]

    # Решаем систему Lz = Pb
    z = np.zeros(n)
    for i in range(n):
        z[i] = b_permuted[i] - np.dot(L[i, :i], z[:i])

    # Решаем систему Ux = z
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def determinant(U: np.ndarray, s: int):
    det = 1
    for i in range(U.shape[0]):
        if abs(U[i, i]) <= 1e-12:  # Проверяем, не является ли диагональный элемент слишком малым
            return 0.0  # Если да, то определитель равен нулю
        det *= U[i, i]
    if s % 2 != 0:  # Если число перестановок нечетное, меняем знак определителя
        det = -det  
    return det

def inverse(L: np.ndarray, U: np.ndarray, perm: np.ndarray):
    n = L.shape[0]
    inv_A = np.zeros((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        inv_A[:, i] = solve_lu(L, U, perm, e)

    return inv_A


