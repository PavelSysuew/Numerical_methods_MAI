import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from least_squares import least_squares, polynomial, squared_error_sum, mean_squared_error

# Степени приближающих многочленов по требованиям: 1-я, 2-я, 3-я
DEGREES = (1, 2, 3)
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉"


# Строку «-0.7 -0.4 -0.1» в список чисел (можно с запятой вместо точки)
def parse_numbers(text):
    return [float(part.replace(",", ".")) for part in text.split()]


# Чтение из файла: строка 1 — xᵢ, строка 2 — yᵢ
def read_input_from_file(path):
    with open(path, encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    if len(lines) != 2:
        raise ValueError("В файле должно быть 2 строки: xᵢ и yᵢ.")
    return parse_numbers(lines[0]), parse_numbers(lines[1])


# Многочлен в строку: 0.005526 + 2.106704·x − ...
def polynomial_to_string(coefficients):
    text = f"{coefficients[0]:.6f}"
    for j, a in enumerate(coefficients[1:], start=1):
        power = "x" if j == 1 else f"x^{j}"
        text += f" {'−' if a < 0 else '+'} {abs(a):.6f}·{power}"
    return text


class LeastSquaresApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 3.3 — Метод наименьших квадратов")

        tk.Label(root, text="Вариант 19.  МНК: приближающие многочлены 1-й, 2-й и 3-й степени, "
                            "сравнение по MSE.",
                 font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 0))

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.vars = {}
        for name, default, width in (("xᵢ", "-0.7 -0.4 -0.1 0.2 0.5 0.8", 30),
                                     ("yᵢ", "-1.4754 -0.81152 -0.20017 0.40136 1.0236 1.7273", 44)):
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
            X, Y = read_input_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.vars["xᵢ"].set(" ".join(f"{x:g}" for x in X))
        self.vars["yᵢ"].set(" ".join(f"{y:g}" for y in Y))

    def on_solve(self):
        try:
            X = np.array(parse_numbers(self.vars["xᵢ"].get()))
            Y = np.array(parse_numbers(self.vars["yᵢ"].get()))
        except ValueError:
            messagebox.showerror("Ошибка", "Поля xᵢ и yᵢ должны быть числами через пробел.")
            return
        if len(X) != len(Y):
            messagebox.showerror("Ошибка", f"Точек xᵢ: {len(X)}, значений yᵢ: {len(Y)} — должно быть поровну.")
            return
        if len(set(X)) < max(DEGREES) + 1:
            messagebox.showerror("Ошибка", f"Нужно хотя бы {max(DEGREES) + 1} различных точки xᵢ.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(X, Y)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    def show_results(self, X, Y):
        self.write(f"Точек: N = {len(X)}")
        results, errors = {}, {}
        for degree in DEGREES:
            coefficients, A, b = least_squares(X, Y, degree)
            results[degree] = coefficients
            errors[degree] = mean_squared_error(coefficients, X, Y)
            name = f"P{SUBSCRIPTS[degree]}"

            self.write(f"\n{'=' * 25} многочлен степени {degree} {'=' * 25}")
            self.write("Нормальная система  Σⱼ aⱼ·Σ xₖ^(i+j) = Σ yₖ·xₖ^i:")
            for i in range(degree + 1):
                row = "".join(f"{v:>12.6f}" for v in A[i])
                self.write(f"  {row}  | {b[i]:>12.6f}")
            self.write("Коэффициенты (LU-разложение): "
                       + ", ".join(f"a{SUBSCRIPTS[j]} = {a:.6f}" for j, a in enumerate(coefficients)))
            self.write(f"{name}(x) = {polynomial_to_string(coefficients)}")
            self.write(f"Сумма квадратов ошибок Φ = Σ ({name}(xₖ) − yₖ)² = {squared_error_sum(coefficients, X, Y):.6e}")
            self.write(f"MSE{degree} = Φ / N = {errors[degree]:.6e}")

        self.write("\nЗначения в исходных точках:")
        header = "".join(f"{f'P{SUBSCRIPTS[d]}(x)':>12}" for d in DEGREES)
        self.write(f"{'x':>8}{'y':>12}{header}")
        for x, y in zip(X, Y):
            self.write(f"{x:>8g}{y:>12.6f}" + "".join(f"{polynomial(results[d], x):>12.6f}" for d in DEGREES))

        # Сравнение ошибок: чем меньше MSE, тем лучше многочлен приближает данные
        self.write("\nСравнение ошибок:")
        for degree in DEGREES:
            self.write(f"  MSE{degree} = {errors[degree]:.6e}")
        best = min(DEGREES, key=lambda d: errors[d])
        self.write(f"Лучше всего приближает данные P{SUBSCRIPTS[best]}(x) (степень {best}): у него наименьшая MSE.")
        self.write("С ростом степени MSE не увеличивается: многочлен меньшей степени — частный случай "
                   "многочлена большей степени.")

        self.draw_graph(X, Y, results)

    def draw_graph(self, X, Y, results):
        margin = 0.1 * (X.max() - X.min())
        x = np.linspace(X.min() - margin, X.max() + margin, 400)
        self.ax.clear()
        self.ax.plot(X, Y, "ko", label="исходные данные")
        for degree, coefficients in results.items():
            self.ax.plot(x, polynomial(coefficients, x), color=COLORS[degree],
                         label=f"P{SUBSCRIPTS[degree]}(x), степень {degree}")
        self.ax.set_title("Приближение методом наименьших квадратов")
        self.ax.grid(True)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    LeastSquaresApp(root)
    root.mainloop()
