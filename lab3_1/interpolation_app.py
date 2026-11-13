import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import func_3_1
from interpolation import omega_derivative, lagrange, divided_differences, newton


# Строку в список чисел
def parse_numbers(text):
    return [float(part.replace(",", ".")) for part in text.split()]


# Чтение данных из файла
def read_input_from_file(path):
    with open(path, encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    if len(lines) != 3:
        raise ValueError("В файле должно быть 3 строки: узлы а), узлы б), X*.")
    x_star = parse_numbers(lines[2])
    if len(x_star) != 1:
        raise ValueError("В третьей строке должно быть одно число — X*.")
    return parse_numbers(lines[0]), parse_numbers(lines[1]), x_star[0]


# Множитель для вывода многочлена
def factor(a):
    if a == 0:
        return "x"
    return f"(x {'−' if a > 0 else '+'} {abs(a):g})"


# Число со знаком для вывода
def signed(value, first=False):
    if first:
        return f"{value:.6f}"
    return f" {'−' if value < 0 else '+'} {abs(value):.6f}"


class InterpolationApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 3.1 — Интерполяционные многочлены Лагранжа и Ньютона")

        # Условие задачи
        tk.Label(root, text=f"Вариант 19.  {func_3_1.DESCRIPTION}.  Многочлены Лагранжа и Ньютона, "
                            "погрешность интерполяции в точке X*.",
                 font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 0))

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.vars = {}
        for name, default, width in (("Xᵢ а)", "-0.4 -0.1 0.2 0.5", 22),
                                     ("Xᵢ б)", "-0.4 0 0.2 0.5", 22),
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
        self.ax_set_a, self.ax_set_b = self.figure.subplots(1, 2)
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
            nodes_a, nodes_b, x_star = read_input_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.vars["Xᵢ а)"].set(" ".join(f"{x:g}" for x in nodes_a))
        self.vars["Xᵢ б)"].set(" ".join(f"{x:g}" for x in nodes_b))
        self.vars["X*"].set(f"{x_star:g}")

    def on_solve(self):
        try:
            nodes_a = parse_numbers(self.vars["Xᵢ а)"].get())
            nodes_b = parse_numbers(self.vars["Xᵢ б)"].get())
            x_star = parse_numbers(self.vars["X*"].get())[0]
        except (ValueError, IndexError):
            messagebox.showerror("Ошибка", "Узлы и X* должны быть числами через пробел.")
            return
        for nodes in (nodes_a, nodes_b):
            if len(nodes) < 2 or len(set(nodes)) != len(nodes):
                messagebox.showerror("Ошибка", "Нужно хотя бы 2 узла, и все узлы должны быть различны.")
                return

        self.output.delete("1.0", "end")
        try:
            results = [self.show_set("а", np.array(nodes_a), x_star),
                       self.show_set("б", np.array(nodes_b), x_star)]
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))
            return
        self.draw_graphs(results, x_star)

    # Решение для одного набора узлов
    def show_set(self, name, X, x_star):
        Y = func_3_1.f(X)
        if np.any(np.isnan(Y)):
            raise ValueError(f"Набор {name}): функция не определена в некоторых узлах (нужно |x| ≤ 1).")
        n = len(X) - 1
        self.write(f"{'=' * 30} Набор {name}) {'=' * 30}")
        self.write(f"Узлов: {n + 1}, многочлены степени n = {n}")
        self.write("\nИсходные узлы интерполяции:")
        self.write(f"{'i':>3}{'xᵢ':>10}{'yᵢ = f(xᵢ)':>16}")
        for i in range(len(X)):
            self.write(f"{i:>3}{X[i]:>10g}{Y[i]:>16.10f}")

        # Многочлен Лагранжа
        self.write("\nМногочлен Лагранжа  L(x) = Σ fᵢ / ω'(xᵢ) · ω(x) / (x − xᵢ)")
        self.write(f"{'i':>3}{'xᵢ':>10}{'fᵢ':>14}{'ω′(xᵢ)':>14}{'fᵢ/ω′(xᵢ)':>14}{'X* − xᵢ':>12}")
        coefficients = []
        for i in range(len(X)):
            w = omega_derivative(X, i)
            coefficients.append(Y[i] / w)
            self.write(f"{i:>3}{X[i]:>10g}{Y[i]:>14.6f}{w:>14.6f}{Y[i] / w:>14.6f}{x_star - X[i]:>12g}")

        terms = []
        for i in range(len(X)):
            others = "".join(factor(X[j]) for j in range(len(X)) if j != i)
            terms.append(f"{signed(coefficients[i], first=(i == 0))}·{others}")
        self.write("L(x) = " + "\n       ".join(terms))
        value_lagrange = lagrange(X, Y, x_star)

        # Многочлен Ньютона
        table = divided_differences(X, Y)
        self.write("\nМногочлен Ньютона, разделённые разности:")
        self.write(f"{'i':>3}{'xᵢ':>10}" + "".join(f"{f'порядок {k}':>14}" for k in range(len(X))))
        for i in range(len(X)):
            row = "".join(f"{table[k][i]:>14.6f}" for k in range(len(X) - i))
            self.write(f"{i:>3}{X[i]:>10g}{row}")

        terms, product = [], ""
        for k in range(len(X)):
            terms.append(f"{signed(table[k][0], first=(k == 0))}{'·' + product if product else ''}")
            product += factor(X[k])
        self.write("P(x) = " + "\n       ".join(terms))
        value_newton = newton(X, table, x_star)

        # Погрешность
        exact = func_3_1.f(x_star)
        self.write(f"\nЗначения в точке X* = {x_star:g}:")
        self.write(f"  L(X*)      = {value_lagrange:.10f}")
        self.write(f"  P(X*)      = {value_newton:.10f}")
        self.write(f"  f(X*)      = {exact:.10f}   (точное значение функции)")
        self.write("Абсолютная погрешность интерполяции:")
        self.write(f"  |L(X*) − f(X*)| = {abs(value_lagrange - exact):.6e}")
        self.write(f"  |P(X*) − f(X*)| = {abs(value_newton - exact):.6e}")
        self.write(f"Разность методов: |L(X*) − P(X*)| = {abs(value_lagrange - value_newton):.3e}  "
                   "(Лагранж и Ньютон — один и тот же многочлен)\n")
        return name, X, Y, table

    # Графики для обоих наборов
    def draw_graphs(self, results, x_star):
        all_nodes = np.concatenate([X for _, X, _, _ in results])
        x = np.linspace(max(-1, all_nodes.min() - 0.1), min(1, all_nodes.max() + 0.1), 400)

        for ax, (name, X, Y, table) in zip((self.ax_set_a, self.ax_set_b), results):
            ax.clear()
            ax.plot(x, func_3_1.f(x), color="black", linewidth=3, alpha=0.4, label=f"f(x): {func_3_1.DESCRIPTION}")
            ax.plot(x, lagrange(X, Y, x), color="tab:blue", label="многочлен Лагранжа L(x)")
            # Ньютон совпадает с Лагранжем, рисуем пунктиром
            ax.plot(x, newton(X, table, x), "--", color="tab:orange", label="многочлен Ньютона P(x)")
            ax.plot(X, Y, "ko", label="узлы интерполяции")
            ax.plot(x_star, lagrange(X, Y, x_star), "r*", markersize=14, label=f"точка X* = {x_star:g}")
            ax.set_title(f"Набор {name})")
            ax.grid(True)
            ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    InterpolationApp(root)
    root.mainloop()
