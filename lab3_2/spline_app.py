import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from spline import cubic_spline, spline_value, spline_second_derivative


# Строку «-0.4 -0.1 0.2» в список чисел (можно с запятой вместо точки)
def parse_numbers(text):
    return [float(part.replace(",", ".")) for part in text.split()]


# Чтение из файла: строка 1 — xᵢ, строка 2 — fᵢ, строка 3 — X*
def read_input_from_file(path):
    with open(path, encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    if len(lines) != 3:
        raise ValueError("В файле должно быть 3 строки: xᵢ, fᵢ, X*.")
    return parse_numbers(lines[0]), parse_numbers(lines[1]), parse_numbers(lines[2])[0]


class SplineApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 3.2 — Кубический сплайн")

        tk.Label(root, text="Вариант 19.  Кубический сплайн с нулевой кривизной на концах: "
                            "S''(x₀) = S''(xₙ) = 0.  Значение в точке X*.",
                 font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 0))

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.vars = {}
        for name, default, width in (("xᵢ", "-0.4 -0.1 0.2 0.5 0.8", 26),
                                     ("fᵢ", "-0.81152 -0.20017 0.40136 1.0236 1.7273", 36),
                                     ("X*", "0.1", 8)):
            tk.Label(top, text=f"{name}:").pack(side="left", padx=(8, 0))
            self.vars[name] = tk.StringVar(value=default)
            tk.Entry(top, width=width, textvariable=self.vars[name]).pack(side="left", padx=4)
        tk.Button(top, text="Загрузить из файла…", command=self.on_load).pack(side="left", padx=(12, 4))
        tk.Button(top, text="Решить", command=self.on_solve).pack(side="left", padx=4)

        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=8, pady=6)
        self.output = scrolledtext.ScrolledText(tabs, font=("Courier", 10))
        tabs.add(self.output, text="Результаты")
        graphs_tab = tk.Frame(tabs)
        tabs.add(graphs_tab, text="Графики")

        self.figure = Figure(figsize=(10, 5))
        self.ax = self.figure.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=graphs_tab)
        NavigationToolbar2Tk(self.canvas, graphs_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def write(self, text=""):
        self.output.insert("end", text + "\n")

    def on_load(self):
        path = filedialog.askopenfilename(title="Файл с входными данными",
                                          filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*")])
        if not path:
            return
        try:
            X, Y, x_star = read_input_from_file(path)
        except (OSError, ValueError, IndexError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.vars["xᵢ"].set(" ".join(f"{x:g}" for x in X))
        self.vars["fᵢ"].set(" ".join(f"{y:g}" for y in Y))
        self.vars["X*"].set(f"{x_star:g}")

    def on_solve(self):
        try:
            X = np.array(parse_numbers(self.vars["xᵢ"].get()))
            Y = np.array(parse_numbers(self.vars["fᵢ"].get()))
            x_star = parse_numbers(self.vars["X*"].get())[0]
        except (ValueError, IndexError):
            messagebox.showerror("Ошибка", "Все поля должны быть числами через пробел.")
            return
        if len(X) != len(Y):
            messagebox.showerror("Ошибка", f"Узлов xᵢ: {len(X)}, значений fᵢ: {len(Y)} — должно быть поровну.")
            return
        if len(X) < 3 or np.any(np.diff(X) <= 0):
            messagebox.showerror("Ошибка", "Нужно хотя бы 3 узла, xᵢ должны строго возрастать.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(X, Y, x_star)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    def show_results(self, X, Y, x_star):
        n = len(X) - 1
        a, b, c, d, (lower, diag, upper, rhs) = cubic_spline(X, Y)
        coefficients = (a, b, c, d)
        h = np.diff(X)

        self.write(f"Узлов: {n + 1}, отрезков: n = {n}")
        self.write("\nИсходные узлы и значения функции:")
        self.write(f"{'i':>3}{'xᵢ':>10}{'fᵢ':>14}")
        for i in range(n + 1):
            self.write(f"{i:>3}{X[i]:>10g}{Y[i]:>14.6f}")
        self.write("\nШаги hⱼ = xⱼ₊₁ − xⱼ: " + ", ".join(f"{v:g}" for v in h))

        self.write("\nСистема для c₁ … c₍ₙ₋₁₎ (c₀ = cₙ = 0), решается прогонкой:")
        self.write(f"{'':>6}{'под диаг.':>12}{'диагональ':>12}{'над диаг.':>12}{'правая часть':>16}")
        for k in range(n - 1):
            self.write(f"{f'c{k + 1}':>6}{lower[k]:>12.4f}{diag[k]:>12.4f}{upper[k]:>12.4f}{rhs[k]:>16.6f}")

        self.write("\nКоэффициенты сплайна  S(x) = aⱼ + bⱼ(x − xⱼ) + cⱼ(x − xⱼ)² + dⱼ(x − xⱼ)³:")
        self.write(f"{'j':>3}{'интервал':>18}{'a':>14}{'b':>14}{'c':>14}{'d':>14}")
        for j in range(n):
            segment = f"[{X[j]:g}, {X[j + 1]:g}]"
            self.write(f"{j:>3}{segment:>18}{a[j]:>14.6f}{b[j]:>14.6f}{c[j]:>14.6f}{d[j]:>14.6f}")

        j = next(j for j in range(n) if x_star <= X[j + 1])
        self.write(f"\nX* = {x_star:g} лежит в интервале № {j}: [{X[j]:g}, {X[j + 1]:g}]")
        self.write(f"S(X*) = {spline_value(X, coefficients, x_star):.10f}")

        # Проверка: сплайн проходит через узлы
        self.write("\nПроверка прохождения через узлы:")
        self.write(f"{'i':>3}{'xᵢ':>10}{'fᵢ':>14}{'S(xᵢ)':>16}{'|S(xᵢ) − fᵢ|':>16}")
        for i in range(n + 1):
            s = spline_value(X, coefficients, X[i])
            self.write(f"{i:>3}{X[i]:>10g}{Y[i]:>14.6f}{s:>16.10f}{abs(s - Y[i]):>16.2e}")

        # Проверка граничных условий: нулевая кривизна на концах
        self.write("\nПроверка граничных условий (должно быть 0):")
        self.write(f"  S''(x₀) = S''({X[0]:g}) = {spline_second_derivative(X, coefficients, X[0]):.3e}")
        self.write(f"  S''(xₙ) = S''({X[-1]:g}) = {spline_second_derivative(X, coefficients, X[-1]):.3e}")

        self.draw_graphs(X, Y, coefficients, x_star)

    # Каждый участок сплайна своим цветом, узлы и точка (X*, S(X*))
    def draw_graphs(self, X, Y, coefficients, x_star):
        self.ax.clear()
        for j in range(len(X) - 1):
            x = np.linspace(X[j], X[j + 1], 100)
            self.ax.plot(x, [spline_value(X, coefficients, t) for t in x], linewidth=2,
                         label=f"участок {j}: [{X[j]:g}, {X[j + 1]:g}]")
        self.ax.plot(X, Y, "ko", label="узлы (xᵢ, fᵢ)")
        self.ax.plot(x_star, spline_value(X, coefficients, x_star), "r*", markersize=14,
                     label=f"(X*, S(X*)), X* = {x_star:g}")
        self.ax.set_title("Кубический сплайн")
        self.ax.grid(True)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    SplineApp(root)
    root.mainloop()
