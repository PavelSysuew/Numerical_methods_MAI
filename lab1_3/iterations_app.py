import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np

from iterations import matrix_norm, vector_norm, to_iteration_form, simple_iteration, seidel

# Точности для анализа числа итераций
ANALYSIS_EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]


# Строку в число (можно с запятой)
def parse_number(text):
    return float(text.strip().replace(",", "."))


# Чтение системы из файла
def read_system_from_file(path):
    lines = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                lines.append([parse_number(part) for part in line.split()])

    if not lines or len(lines[0]) != 1:
        raise ValueError("В первой строке файла должно быть одно число — точность ε.")
    eps = lines[0][0]

    rows = lines[1:]
    n = len(rows)
    for row in rows:
        if len(row) != n + 1:
            raise ValueError(
                f"После строки с ε идут {n} строк, значит в каждой должно быть {n + 1} чисел "
                f"(строка матрицы и элемент b), а встретилось {len(row)}."
            )

    data = np.array(rows)
    return eps, data[:, :n], data[:, n]


# Вывод числа
def format_number(value):
    # + 0.0 убирает -0.0
    return f"{round(value, 6) + 0.0:12.6f}"


def format_matrix(M):
    return "\n".join("".join(format_number(v) for v in row) for row in M)


def format_vector(v):
    return "".join(format_number(x) for x in v)


class IterationsApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 1.3 — Метод простых итераций и метод Зейделя")

        # Кнопки сверху
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)

        tk.Label(top, text="Размерность n:").pack(side="left")
        self.n_var = tk.StringVar(value="3")
        tk.Spinbox(top, from_=1, to=20, width=4, textvariable=self.n_var).pack(side="left", padx=4)
        tk.Button(top, text="Создать таблицу", command=self.on_resize).pack(side="left", padx=4)

        tk.Label(top, text="Точность ε:").pack(side="left", padx=(12, 0))
        self.eps_var = tk.StringVar(value="0.01")
        tk.Entry(top, width=10, textvariable=self.eps_var).pack(side="left", padx=4)

        tk.Button(top, text="Загрузить из файла…", command=self.on_load).pack(side="left", padx=4)
        tk.Button(top, text="Решить", command=self.on_solve).pack(side="left", padx=4)

        # Таблица для ввода
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(padx=8, pady=6)
        self.a_entries = []
        self.b_entries = []

        # Поле вывода
        self.output = scrolledtext.ScrolledText(root, width=100, height=32, font=("Courier", 10))
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
            eps, A, b = read_system_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.eps_var.set(f"{eps:g}")
        self.fill_grid(A, b)

    def on_solve(self):
        try:
            A, b = self.read_grid()
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля матрицы и правой части должны быть заполнены числами.")
            return
        try:
            eps = parse_number(self.eps_var.get())
            if eps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Точность ε должна быть положительным числом.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(A, b, eps)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    # Вывод таблицы итераций
    def write_iterations(self, title, history):
        self.write(f"\n{title}:")
        self.write(f"{'k':>5}{'x⁽ᵏ⁾':>{12 * len(history[0][0])}}{'оценка погрешности':>22}")
        for k, (x, error) in enumerate(history, start=1):
            self.write(f"{k:>5}{format_vector(x)}{error:>22.3e}")

    # Решение и вывод результатов
    def show_results(self, A, b, eps):
        self.write(f"Заданная точность ε = {eps:g}")

        alpha, beta = to_iteration_form(A, b)
        norm_alpha = matrix_norm(alpha)
        self.write("\nПриведение к виду x = αx + β:")
        self.write("Матрица α:")
        self.write(format_matrix(alpha))
        self.write("Вектор β:")
        self.write(format_vector(beta))

        self.write(f"\n‖α‖ = {norm_alpha:.6g}")
        if norm_alpha < 1:
            self.write("‖α‖ < 1: достаточное условие сходимости выполнено, оба метода сходятся при любом x⁽⁰⁾.")
        else:
            self.write(
                "ВНИМАНИЕ: ‖α‖ ≥ 1, достаточное условие сходимости не выполнено. Методы могут "
                "не сойтись;\nостановка — по ‖x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾‖ ≤ ε, оценка погрешности неприменима."
            )

        x_simple, history_simple = simple_iteration(alpha, beta, eps)
        x_seidel, history_seidel = seidel(alpha, beta, eps)
        self.write_iterations("Метод простых итераций", history_simple)
        self.write_iterations("Метод Зейделя", history_seidel)

        # Эталонное решение через numpy
        x_exact = np.linalg.solve(A, b)
        self.write("\nРезультаты:")
        self.write(f"{'':<26}{'x':<{12 * len(b)}}{'итераций':>10}{'‖x − x*‖':>14}")
        self.write(f"{'Эталонное решение x*':<26}{format_vector(x_exact)}")
        self.write(
            f"{'Метод простых итераций':<26}{format_vector(x_simple)}"
            f"{len(history_simple):>10}{vector_norm(x_simple - x_exact):>14.3e}"
        )
        self.write(
            f"{'Метод Зейделя':<26}{format_vector(x_seidel)}"
            f"{len(history_seidel):>10}{vector_norm(x_seidel - x_exact):>14.3e}"
        )

        self.write("\nСравнение по скорости сходимости:")
        if len(history_seidel) < len(history_simple):
            self.write(
                f"Метод Зейделя сошёлся быстрее. Итераций: {len(history_seidel)} против "
                f"{len(history_simple)} у МПИ — в {len(history_simple) / len(history_seidel):.2f} раза меньше."
            )
        elif len(history_seidel) > len(history_simple):
            self.write(
                f"Метод простых итераций сошёлся быстрее. Итераций: {len(history_simple)} против "
                f"{len(history_seidel)} у метода Зейделя."
            )
        else:
            self.write(f"Оба метода сделали одинаковое число итераций: {len(history_simple)}.")

        self.write("\nЧисло итераций в зависимости от точности:")
        self.write(f"{'ε':>10}{'МПИ':>8}{'Зейдель':>10}")
        for analysis_eps in ANALYSIS_EPS:
            try:
                count_simple = len(simple_iteration(alpha, beta, analysis_eps)[1])
                count_seidel = len(seidel(alpha, beta, analysis_eps)[1])
            except ValueError:
                self.write(f"{analysis_eps:>10.0e}{'не сошёлся':>18}")
                continue
            self.write(f"{analysis_eps:>10.0e}{count_simple:>8}{count_seidel:>10}")


if __name__ == "__main__":
    root = tk.Tk()
    IterationsApp(root)
    root.mainloop()
