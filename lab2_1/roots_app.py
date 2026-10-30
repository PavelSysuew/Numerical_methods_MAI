import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import functions
import graphs
from methods import simple_iteration, newton

# Точности для анализа числа итераций
ANALYSIS_EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]

# Число точек на [a, b] для оценки max|fi'(x)| и знака f'(x)
CHECK_POINTS = 1000


# Строку в число (можно с запятой)
def parse_number(text):
    return float(text.strip().replace(",", "."))


# Чтение из файла: ε, затем a и b, затем x⁰ (переносы строк не важны)
def read_input_from_file(path):
    numbers = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            numbers.extend(parse_number(part) for part in line.split())

    if len(numbers) != 4:
        raise ValueError(f"В файле должно быть 4 числа: ε, a, b, x⁰. Найдено: {len(numbers)}.")
    return numbers


# Наибольший модуль функции на [a, b], оценка по сетке точек
def max_abs_on_segment(func, a, b):
    x = np.linspace(a, b, CHECK_POINTS)
    return np.max(np.abs(func(x)))


class RootsApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 2.1 — Метод простой итерации и метод Ньютона")

        # Решаемое уравнение — видно всегда, ещё до нажатия «Решить»
        tk.Label(root, text=f"Вариант 19.  Уравнение:  {functions.EQUATION}.  Ищем положительный корень.",
                 font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 0))

        # Поля ввода и кнопки сверху
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)

        self.vars = {}
        for name, default in (("ε", "0.0001"), ("a", "1.3"), ("b", "1.4"), ("x⁰", "1.4")):
            tk.Label(top, text=f"{name}:").pack(side="left", padx=(8, 0))
            self.vars[name] = tk.StringVar(value=default)
            tk.Entry(top, width=10, textvariable=self.vars[name]).pack(side="left", padx=4)

        tk.Button(top, text="Загрузить из файла…", command=self.on_load).pack(side="left", padx=(12, 4))
        tk.Button(top, text="Решить", command=self.on_solve).pack(side="left", padx=4)

        # Две вкладки: текстовые результаты и графики
        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=8, pady=6)

        self.output = scrolledtext.ScrolledText(tabs, font=("Courier", 10))
        tabs.add(self.output, text="Результаты")

        graphs_tab = tk.Frame(tabs)
        tabs.add(graphs_tab, text="Графики")

        # Figure — фигура matplotlib, которую можно встроить в окно tkinter
        self.figure = Figure(figsize=(10, 7))
        axes = self.figure.subplots(2, 2)
        graphs.plot_f(axes[0][0])
        graphs.plot_intersection(axes[0][1])
        graphs.plot_fi(axes[1][0])
        self.errors_ax = axes[1][1]
        self.errors_ax.set_title("Погрешность появится после «Решить»")
        self.figure.tight_layout()

        # Холст с фигурой и панель инструментов (лупа, перемещение, сохранение)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graphs_tab)
        NavigationToolbar2Tk(self.canvas, graphs_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def write(self, text=""):
        self.output.insert("end", text + "\n")

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Файл с входными данными",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*")],
        )
        if not path:
            return
        try:
            values = read_input_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        for name, value in zip(("ε", "a", "b", "x⁰"), values):
            self.vars[name].set(f"{value:g}")

    def on_solve(self):
        try:
            eps, a, b, x0 = (parse_number(self.vars[name].get()) for name in ("ε", "a", "b", "x⁰"))
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля должны быть заполнены числами.")
            return
        if eps <= 0:
            messagebox.showerror("Ошибка", "Точность ε должна быть положительным числом.")
            return
        if not a < b:
            messagebox.showerror("Ошибка", "Должно быть a < b.")
            return
        if not a <= x0 <= b:
            messagebox.showerror("Ошибка", "Начальное приближение x⁰ должно лежать на [a, b].")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(eps, a, b, x0)
        except (ValueError, NotImplementedError) as error:
            messagebox.showerror("Ошибка", str(error))

    # Вывод таблицы итераций
    def write_iterations(self, title, history):
        self.write(f"\n{title}:")
        self.write(f"{'k':>5}{'xᵏ':>18}{'|xᵏ − xᵏ⁻¹|':>16}")
        for k, (x, error) in enumerate(history, start=1):
            self.write(f"{k:>5}{x:>18.12f}{error:>16.3e}")

    # Проверка условий и решение
    def show_results(self, eps, a, b, x0):
        self.write(f"Уравнение (вариант 19): {functions.EQUATION}")
        self.write(f"\nЗаданная точность ε = {eps:g}")
        self.write(f"Отрезок [a, b] = [{a:g}, {b:g}], начальное приближение x⁰ = {x0:g}")

        # Теорема 2: f(a)·f(b) < 0 и f'(x) сохраняет знак на [a, b]
        self.write("\nОтделение корня (теорема 2):")
        fa, fb = functions.f(a), functions.f(b)
        self.write(f"f(a) = {fa:.6g},  f(b) = {fb:.6g},  f(a)·f(b) = {fa * fb:.6g}")
        df = functions.derivative_f(np.linspace(a, b, CHECK_POINTS))
        self.write(f"f'(x) на [a, b]: от {df.min():.6g} до {df.max():.6g}")
        if fa * fb < 0 and (df.min() > 0 or df.max() < 0):
            self.write("Условия выполнены: на [a, b] ровно один корень.")
        else:
            self.write("ВНИМАНИЕ: условия теоремы 2 не выполнены, корень на [a, b] может быть не единственным или отсутствовать.")

        # Условие сходимости |fi'(x)| < 1 для обоих методов
        self.write("\nУсловие сходимости |φ'(x)| < 1 на [a, b] (максимум по сетке точек):")
        q_simple = max_abs_on_segment(functions.derivative_fi_MSI, a, b)
        q_newton = max_abs_on_segment(functions.derivative_fi_newton, a, b)
        for name, q in (("Простая итерация, φ(x) = (2x + 1)^(1/4)", q_simple),
                        ("Ньютон, φ'(x) = f(x)·f''(x) / f'(x)²", q_newton)):
            verdict = "выполнено" if q < 1 else "НЕ выполнено"
            self.write(f"{name}:  max|φ'| = {q:.6g} — {verdict}")

        x_simple, history_simple = simple_iteration(functions.fi_MSI, x0, eps, a, b)
        x_newton, history_newton = newton(functions.f, functions.derivative_f, x0, eps, a, b)
        self.write_iterations("Метод простой итерации", history_simple)
        self.write_iterations("Метод Ньютона", history_newton)

        # Невязка |f(x)| — насколько найденное x не удовлетворяет уравнению
        self.write("\nРезультаты:")
        self.write(f"{'':<24}{'x':>18}{'итераций':>10}{'|f(x)|':>12}")
        for name, x, history in (("Простая итерация", x_simple, history_simple),
                                 ("Ньютон", x_newton, history_newton)):
            self.write(f"{name:<24}{x:>18.12f}{len(history):>10}{abs(functions.f(x)):>12.3e}")

        self.write("\nЧисло итераций в зависимости от точности:")
        self.write(f"{'ε':>10}{'Простая итерация':>20}{'Ньютон':>10}")
        for analysis_eps in ANALYSIS_EPS:
            counts = []
            for method, args in ((simple_iteration, (functions.fi_MSI,)),
                                 (newton, (functions.f, functions.derivative_f))):
                try:
                    counts.append(str(len(method(*args, x0, analysis_eps, a, b)[1])))
                except ValueError:
                    counts.append("не сошёлся")
            self.write(f"{analysis_eps:>10.0e}{counts[0]:>20}{counts[1]:>10}")

        # График погрешности на вкладке «Графики»
        self.errors_ax.clear()
        graphs.plot_errors(self.errors_ax, history_simple, history_newton, eps)
        self.errors_ax.set_title("Погрешность по итерациям")
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    RootsApp(root)
    root.mainloop()
