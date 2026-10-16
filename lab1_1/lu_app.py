import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np

from lu import LU, solve_lu, determinant, inverse


# Строку в число (можно с запятой)
def parse_number(text):
    return float(text.strip().replace(",", "."))


# Чтение системы из файла
def read_system_from_file(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append([parse_number(part) for part in line.split()])

    n = len(rows)
    for row in rows:
        if len(row) != n + 1:
            raise ValueError(
                f"В файле {n} строк, значит в каждой должно быть {n + 1} чисел "
                f"(строка матрицы и элемент b), а встретилось {len(row)}."
            )

    data = np.array(rows)
    A = data[:, :n]  # первые n столбцов — матрица A
    b = data[:, n]   # последний столбец — правая часть b
    return A, b


# Вывод числа
def format_number(value):
    # + 0.0 убирает -0.0
    return f"{round(value, 4) + 0.0:10.4f}"


def format_matrix(M):
    return "\n".join("".join(format_number(v) for v in row) for row in M)


def format_vector(v):
    return "".join(format_number(x) for x in v)


class LuApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 1.1 — LU-разложение")

        # Кнопки сверху
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)

        tk.Label(top, text="Размерность n:").pack(side="left")
        self.n_var = tk.StringVar(value="3")
        tk.Spinbox(top, from_=1, to=20, width=4, textvariable=self.n_var).pack(side="left", padx=4)
        tk.Button(top, text="Создать таблицу", command=self.on_resize).pack(side="left", padx=4)
        tk.Button(top, text="Загрузить из файла…", command=self.on_load).pack(side="left", padx=4)
        tk.Button(top, text="Решить", command=self.on_solve).pack(side="left", padx=4)

        # Таблица для ввода
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(padx=8, pady=6)
        self.a_entries = []
        self.b_entries = []

        # Поле вывода
        self.output = scrolledtext.ScrolledText(root, width=90, height=30, font=("Courier", 10))
        self.output.pack(fill="both", expand=True, padx=8, pady=6)

        self.build_grid(3)

    # Создание таблицы для ввода
    def build_grid(self, n):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.a_entries = []
        self.b_entries = []

        tk.Label(self.grid_frame, text="A").grid(row=0, column=0, columnspan=n)
        tk.Label(self.grid_frame, text="b").grid(row=0, column=n + 1)

        for i in range(n):
            row_entries = []
            for j in range(n):
                entry = tk.Entry(self.grid_frame, width=8, justify="right")
                entry.grid(row=i + 1, column=j, padx=1, pady=1)
                row_entries.append(entry)
            self.a_entries.append(row_entries)

            # Разделитель
            tk.Label(self.grid_frame, text="|").grid(row=i + 1, column=n)
            entry = tk.Entry(self.grid_frame, width=8, justify="right")
            entry.grid(row=i + 1, column=n + 1, padx=1, pady=1)
            self.b_entries.append(entry)

    # Заполнение таблицы числами
    def fill_grid(self, A, b):
        n = len(A)
        self.n_var.set(str(n))
        self.build_grid(n)
        for i in range(n):
            for j in range(n):
                self.a_entries[i][j].insert(0, f"{A[i][j]:g}")
            self.b_entries[i].insert(0, f"{b[i]:g}")

    # Чтение чисел из таблицы
    def read_grid(self):
        A = np.array([[parse_number(entry.get()) for entry in row] for row in self.a_entries])
        b = np.array([parse_number(entry.get()) for entry in self.b_entries])
        return A, b

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
            A, b = read_system_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.fill_grid(A, b)

    def on_solve(self):
        try:
            A, b = self.read_grid()
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля должны быть заполнены числами.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(A, b)
        except ValueError as error:
            # Вырожденная матрица
            messagebox.showerror("Ошибка", str(error))

    # Решение и вывод результатов
    def show_results(self, A, b):
        n = len(A)

        self.write("Исходная матрица A:")
        self.write(format_matrix(A))
        self.write("\nПравая часть b:")
        self.write(format_vector(b))

        L, U, perm, s = LU(A)
        self.write("\nМатрица L:")
        self.write(format_matrix(L))
        self.write("\nМатрица U:")
        self.write(format_matrix(U))

        self.write("\nL * U:")
        self.write(format_matrix(L @ U))

        # Матрица перестановок
        P = np.eye(n)[perm]
        self.write(f"\nМатрица перестановок P (перестановок строк: {s}):")
        self.write(format_matrix(P))
        self.write("\nP * A:")
        self.write(format_matrix(P @ A))
        self.write(f"Проверка L * U = P * A: {np.allclose(L @ U, P @ A)}")

        x = solve_lu(L, U, perm, b)
        self.write("\nРешение системы x:")
        self.write(format_vector(x))
        self.write(f"Проверка A * x = b: {np.allclose(A @ x, b)}")

        self.write(f"\nОпределитель det A = {determinant(U, s):.6g}")

        A_inv = inverse(L, U, perm)
        self.write("\nОбратная матрица A^(-1):")
        self.write(format_matrix(A_inv))

        self.write("\nA * A^(-1):")
        self.write(format_matrix(A @ A_inv))
        self.write(f"Проверка A * A^(-1) = E: {np.allclose(A @ A_inv, np.eye(n))}")


if __name__ == "__main__":
    root = tk.Tk()
    LuApp(root)
    root.mainloop()
