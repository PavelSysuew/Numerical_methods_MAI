"""Задание 1.2. Окно программы для метода прогонки.

Запуск из корня проекта:  .venv/bin/python lab1_2/tridiagonale_app.py

Здесь только интерфейс: ввод системы (вручную или из файла) и вывод
результатов. Все вычисления — в tridiagonale.py.

Матрица вводится и записывается в файле целиком, как на листочке, но
программа берёт из неё только три диагонали: в памяти хранятся массивы
a, диагональ, c и d длины n, а полная матрица n x n не создаётся.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np

from tridiagonale import tridiagonale, has_diagonal_dominance, residual


def parse_number(text):
    """Превращает строку в число; запятая допускается вместо точки."""
    return float(text.strip().replace(",", "."))


def is_tridiagonal_cell(i, j):
    """Лежит ли элемент (i, j) на главной диагонали или на одной из соседних."""
    return abs(i - j) <= 1


def read_system_from_file(path):
    """Читает систему из файла и возвращает (a, diag, c, d).

    Формат как на листочке: n строк по n + 1 числу — строка матрицы и в конце
    элемент правой части d. Файл читается построчно, и из каждой строки
    берутся только три элемента у диагонали и d; остальные элементы
    проверяются на равенство нулю и не сохраняются.
    """
    a, diag, c, d = [], [], [], []
    n = None
    i = 0  # номер текущей строки матрицы

    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = [parse_number(part) for part in line.split()]

            if n is None:
                n = len(row) - 1  # размерность берём из длины первой строки
            if len(row) != n + 1:
                raise ValueError(f"В строке {i + 1} должно быть {n + 1} чисел, а их {len(row)}.")
            if i >= n:
                raise ValueError(f"В файле больше {n} строк матрицы.")

            for j in range(n):
                if not is_tridiagonal_cell(i, j) and row[j] != 0:
                    raise ValueError(
                        f"Элемент в строке {i + 1}, столбце {j + 1} равен {row[j]:g}, "
                        f"а матрица должна быть трёхдиагональной."
                    )

            a.append(row[i - 1] if i > 0 else 0.0)
            diag.append(row[i])
            c.append(row[i + 1] if i < n - 1 else 0.0)
            d.append(row[n])
            i += 1

    if n is None or i != n:
        raise ValueError(f"В файле должно быть {n} строк матрицы, а их {i}.")
    return np.array(a), np.array(diag), np.array(c), np.array(d)


def format_number(value):
    # "+ 0.0" превращает -0.0 в 0.0, чтобы не печатать "-0.0000"
    return f"{round(value, 6) + 0.0:12.6f}"


def format_array(values):
    return "(" + ", ".join(f"{v + 0.0:g}" for v in values) + ")"


class TridiagonalApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 1.2 — Метод прогонки")

        # Верхняя панель: размерность и кнопки
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)

        tk.Label(top, text="Размерность n:").pack(side="left")
        self.n_var = tk.StringVar(value="3")
        tk.Spinbox(top, from_=1, to=20, width=4, textvariable=self.n_var).pack(side="left", padx=4)
        tk.Button(top, text="Создать таблицу", command=self.on_resize).pack(side="left", padx=4)
        tk.Button(top, text="Загрузить из файла…", command=self.on_load).pack(side="left", padx=4)
        tk.Button(top, text="Решить", command=self.on_solve).pack(side="left", padx=4)

        # Таблица ввода: матрица и столбец d, как на листочке
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(padx=8, pady=6)
        self.matrix_entries = {}  # matrix_entries[(i, j)] — поле элемента (i, j), только на трёх диагоналях
        self.d_entries = []       # d_entries[i] — поле элемента правой части d[i]

        # Поле вывода результатов
        self.output = scrolledtext.ScrolledText(root, width=80, height=26, font=("Courier", 10))
        self.output.pack(fill="both", expand=True, padx=8, pady=6)

        self.build_grid(3)

    def build_grid(self, n):
        """Пересоздаёт таблицу под систему размера n.

        Поля ввода есть только на трёх диагоналях; в остальных клетках
        стоит неизменяемый ноль, потому что матрица трёхдиагональная.
        """
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.matrix_entries = {}
        self.d_entries = []

        tk.Label(self.grid_frame, text="A").grid(row=0, column=0, columnspan=n)
        tk.Label(self.grid_frame, text="d").grid(row=0, column=n + 1)

        for i in range(n):
            for j in range(n):
                if is_tridiagonal_cell(i, j):
                    entry = tk.Entry(self.grid_frame, width=7, justify="right")
                    entry.grid(row=i + 1, column=j, padx=1, pady=1)
                    self.matrix_entries[(i, j)] = entry
                else:
                    tk.Label(self.grid_frame, text="0", width=7, fg="gray").grid(row=i + 1, column=j)

            # Столбец n — разделитель между матрицей и d
            tk.Label(self.grid_frame, text="|").grid(row=i + 1, column=n)
            entry = tk.Entry(self.grid_frame, width=7, justify="right")
            entry.grid(row=i + 1, column=n + 1, padx=1, pady=1)
            self.d_entries.append(entry)

    def fill_grid(self, a, diag, c, d):
        """Записывает числа системы в поля ввода."""
        n = len(diag)
        self.n_var.set(str(n))
        self.build_grid(n)
        for i in range(n):
            self.matrix_entries[(i, i)].insert(0, f"{diag[i]:g}")
            if i > 0:
                self.matrix_entries[(i, i - 1)].insert(0, f"{a[i]:g}")
            if i < n - 1:
                self.matrix_entries[(i, i + 1)].insert(0, f"{c[i]:g}")
            self.d_entries[i].insert(0, f"{d[i]:g}")

    def read_grid(self):
        """Считывает из таблицы только три диагонали и d, возвращает (a, diag, c, d)."""
        n = len(self.d_entries)
        a, diag, c, d = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            diag[i] = parse_number(self.matrix_entries[(i, i)].get())
            if i > 0:
                a[i] = parse_number(self.matrix_entries[(i, i - 1)].get())  # под диагональю
            if i < n - 1:
                c[i] = parse_number(self.matrix_entries[(i, i + 1)].get())  # над диагональю
            d[i] = parse_number(self.d_entries[i].get())
        return a, diag, c, d

    def write(self, text=""):
        self.output.insert("end", text + "\n")

    def on_resize(self):
        try:
            n = int(self.n_var.get())
            if n < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Размерность должна быть целым положительным числом.")
            return
        self.build_grid(n)

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Файл с системой",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*")],
        )
        if not path:
            return
        try:
            a, diag, c, d = read_system_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.fill_grid(a, diag, c, d)

    def on_solve(self):
        try:
            a, diag, c, d = self.read_grid()
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля должны быть заполнены числами.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(a, diag, c, d)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    def show_results(self, a, diag, c, d):
        """Выполняет прогонку и печатает коэффициенты, решение и массивы a, b, c, d."""
        n = len(diag)

        if has_diagonal_dominance(a, diag, c):
            self.write("Диагональное преобладание есть: прогонка корректна и устойчива.")
        else:
            self.write("ВНИМАНИЕ: диагонального преобладания нет, метод может быть неустойчив.")

        P, Q, x = tridiagonale(a, diag, c, d)

        # Нумерация как в лекции: P₁..Pₙ₊₁, Q₁..Qₙ₊₁
        self.write("\nПрогоночные коэффициенты:")
        self.write(f"{'i':>3}{'P_i':>14}{'Q_i':>14}")
        for k in range(n + 1):
            self.write(f"{k + 1:>3}{format_number(P[k]):>14}{format_number(Q[k]):>14}")

        self.write("\nРешение системы:")
        for i in range(n):
            self.write(f"  x{i + 1} = {format_number(x[i])}")

        r = residual(a, diag, c, d, x)
        self.write(f"\nПроверка: максимальная невязка |A·x − d| = {np.max(np.abs(r)):.3e}")

        # Массивы в каноническом виде лекции: a_i x_{i-1} − b_i x_i + c_i x_{i+1} = d_i
        self.write("\nМассивы, которые хранит программа (вид a·xᵢ₋₁ − b·xᵢ + c·xᵢ₊₁ = d):")
        self.write(f"  a = {format_array(a)}")
        self.write(f"  b = {format_array(-diag)}   (диагональ со знаком минус)")
        self.write(f"  c = {format_array(c)}")
        self.write(f"  d = {format_array(d)}")


if __name__ == "__main__":
    root = tk.Tk()
    TridiagonalApp(root)
    root.mainloop()
