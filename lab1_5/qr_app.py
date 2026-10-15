"""Задание 1.5. Окно программы для QR-разложения и QR-алгоритма.

Запуск из корня проекта:  .venv/bin/python lab1_5/qr_app.py

Здесь только интерфейс: ввод матрицы и точности (вручную или из файла)
и вывод результатов. Все вычисления — в qr.py.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np

from qr import householder_qr, qr_algorithm


def parse_number(text):
    """Превращает строку в число; запятая допускается вместо точки."""
    return float(text.strip().replace(",", "."))


def read_matrix_from_file(path):
    """Читает из файла точность и матрицу, возвращает (eps, A).

    Формат: первая строка — точность ε; затем n строк по n чисел — матрица A.
    Пустые строки пропускаются.
    """
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
        if len(row) != n:
            raise ValueError(f"Матрица из {n} строк должна иметь по {n} чисел в строке, а встретилось {len(row)}.")
    return eps, np.array(rows)


def format_number(value):
    # "+ 0.0" превращает -0.0 в 0.0, чтобы не печатать "-0.0000"
    return f"{round(value, 6) + 0.0:12.6f}"


def format_matrix(M):
    return "\n".join("".join(format_number(v) for v in row) for row in M)


def format_complex(z, eps):
    """Вещественное число печатается без мнимой части, комплексное — как a ± bi."""
    if abs(z.imag) <= eps:
        return f"{z.real + 0.0:.6f}"
    sign = "+" if z.imag > 0 else "−"
    return f"{z.real + 0.0:.6f} {sign} {abs(z.imag):.6f}i"


def sort_key(z):
    return (round(z.real, 6), z.imag)


class QrApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 1.5 — QR-разложение и QR-алгоритм")

        # Верхняя панель: размерность, точность и кнопки
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

        # Таблица полей ввода матрицы
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(padx=8, pady=6)
        self.entries = []  # entries[i][j] — поле ввода элемента A[i][j]

        # Поле вывода результатов
        self.output = scrolledtext.ScrolledText(root, width=90, height=32, font=("Courier", 10))
        self.output.pack(fill="both", expand=True, padx=8, pady=6)

        self.build_grid(3)

    def build_grid(self, n):
        """Пересоздаёт таблицу полей ввода под матрицу размера n."""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.entries = []

        tk.Label(self.grid_frame, text="A").grid(row=0, column=0, columnspan=n)
        for i in range(n):
            row_entries = []
            for j in range(n):
                entry = tk.Entry(self.grid_frame, width=8, justify="right")
                entry.grid(row=i + 1, column=j, padx=1, pady=1)
                row_entries.append(entry)
            self.entries.append(row_entries)

    def fill_grid(self, A):
        """Записывает элементы матрицы в поля ввода."""
        n = len(A)
        self.n_var.set(str(n))
        self.build_grid(n)
        for i in range(n):
            for j in range(n):
                self.entries[i][j].insert(0, f"{A[i][j]:g}")

    def read_grid(self):
        """Считывает матрицу из полей ввода."""
        return np.array([[parse_number(entry.get()) for entry in row] for row in self.entries])

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
            title="Файл с матрицей",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*")],
        )
        if not path:
            return
        try:
            eps, A = read_matrix_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.eps_var.set(f"{eps:g}")
        self.fill_grid(A)

    def on_solve(self):
        try:
            A = self.read_grid()
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля матрицы должны быть заполнены числами.")
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
            self.show_results(A, eps)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    def show_results(self, A, eps):
        """Выполняет расчёты задания 1.5 и печатает их по порядку."""
        n = len(A)
        self.write(f"Заданная точность ε = {eps:g}")
        self.write("\nИсходная матрица A:")
        self.write(format_matrix(A))

        Q, R = householder_qr(A)
        self.write("\nQR-разложение (метод Хаусхолдера):")
        self.write("Матрица Q:")
        self.write(format_matrix(Q))
        self.write("Матрица R:")
        self.write(format_matrix(R))
        self.write("Q·R:")
        self.write(format_matrix(Q @ R))
        self.write(f"Проверка Q·R = A: max |Q·R − A| = {np.max(np.abs(Q @ R - A)):.3e}")
        self.write(f"Проверка ортогональности Q: max |Qᵀ·Q − E| = {np.max(np.abs(Q.T @ Q - np.eye(n))):.3e}")

        eigenvalues, A_k, iterations = qr_algorithm(A, eps)
        self.write(f"\nQR-алгоритм: число итераций {iterations}")
        self.write("Матрица A⁽ᵏ⁾ на последней итерации:")
        self.write(format_matrix(A_k))

        self.write("\nНайденные собственные значения:")
        for i, z in enumerate(eigenvalues, start=1):
            self.write(f"  λ{i} = {format_complex(z, eps)}")

        has_complex = any(abs(z.imag) > eps for z in eigenvalues)
        if has_complex:
            self.write("Есть комплексно-сопряжённая пара: ей соответствует блок 2×2 на диагонали A⁽ᵏ⁾,\n"
                       "который не становится треугольным — его элемент под диагональю не стремится к нулю.")

        # Проверка через NumPy — разрешена в этом задании для комплексно-сопряжённых пар
        reference = sorted(np.linalg.eigvals(A), key=sort_key)
        found = sorted(eigenvalues, key=sort_key)
        self.write("\nПроверка (np.linalg.eigvals):")
        for z_found, z_ref in zip(found, reference):
            self.write(f"  {format_complex(z_found, eps):>28}   эталон {format_complex(z_ref, eps):>28}"
                       f"   |разница| = {abs(z_found - z_ref):.2e}")


if __name__ == "__main__":
    root = tk.Tk()
    QrApp(root)
    root.mainloop()
