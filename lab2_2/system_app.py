import itertools
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import system_functions as sf
import system_graphs
from system_methods import simple_iteration, newton, vector_norm, matrix_norm
from lu import LU, determinant  # путь к lab1_1 добавляет system_methods

# Точности для анализа числа итераций
ANALYSIS_EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]

# Число точек сетки по каждой переменной для проверки условий сходимости в G
CHECK_POINTS = 101


# Строку в число (можно с запятой)
def parse_number(text):
    return float(text.strip().replace(",", "."))


# Чтение из файла: ε; затем x⁰ (n чисел); затем n строк «нижняя верхняя» — границы области G
def read_input_from_file(path):
    lines = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                lines.append([parse_number(part) for part in line.split()])

    n = sf.N
    if len(lines) != n + 2 or len(lines[0]) != 1 or len(lines[1]) != n or any(len(row) != 2 for row in lines[2:]):
        raise ValueError(
            f"Формат файла: 1-я строка — ε; 2-я — x⁰ ({n} числа); "
            f"затем {n} строки по 2 числа — границы области G по каждой переменной."
        )
    return lines[0][0], lines[1], [tuple(row) for row in lines[2:]]


# Вектор в строку
def format_vector(v):
    return "".join(f"{x:16.10f}" for x in v)


# Все точки сетки в прямоугольной области G
def grid_points(bounds):
    axes = [np.linspace(low, high, CHECK_POINTS) for low, high in bounds]
    return [np.array(point) for point in itertools.product(*axes)]


class SystemApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 2.2 — Системы нелинейных уравнений")

        # Решаемая система — видна всегда, ещё до нажатия «Решить»
        header = tk.Frame(root)
        header.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(header, text="Вариант 19.  Система:", font=("TkDefaultFont", 13, "bold")).pack(side="left", anchor="n")
        tk.Label(header, text="\n".join(sf.DESCRIPTION), font=("TkDefaultFont", 13, "bold"),
                 justify="left").pack(side="left", padx=8)
        tk.Label(header, text="Ищем решение с положительными x1, x2.",
                 font=("TkDefaultFont", 11)).pack(side="left", anchor="s", padx=8)

        # Первая строка: ε и кнопки
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(top, text="Точность ε:").pack(side="left")
        self.eps_var = tk.StringVar(value="0.0001")
        tk.Entry(top, width=10, textvariable=self.eps_var).pack(side="left", padx=4)
        tk.Button(top, text="Загрузить из файла…", command=self.on_load).pack(side="left", padx=(12, 4))
        tk.Button(top, text="Решить", command=self.on_solve).pack(side="left", padx=4)

        # Таблица: для каждой переменной x⁰ и границы области G
        table = tk.Frame(root)
        table.pack(fill="x", padx=8, pady=6)
        for column, text in enumerate(("", "x⁰", "G: от", "до")):
            tk.Label(table, text=text).grid(row=0, column=column, padx=4)

        # Значения по умолчанию — из data/variant_19.txt: (x⁰, от, до) для каждой переменной
        defaults = [("1.3", "1.2", "1.4"), ("2.1", "2.0", "2.2")]
        self.x0_vars, self.low_vars, self.high_vars = [], [], []
        for i in range(sf.N):
            tk.Label(table, text=f"x{i + 1}").grid(row=i + 1, column=0, padx=4)
            for column, var_list, default in zip((1, 2, 3), (self.x0_vars, self.low_vars, self.high_vars),
                                                 defaults[i]):
                var = tk.StringVar(value=default)
                tk.Entry(table, width=10, textvariable=var).grid(row=i + 1, column=column, padx=4, pady=1)
                var_list.append(var)

        # Вкладки: текстовые результаты и графики
        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=8, pady=6)

        self.output = scrolledtext.ScrolledText(tabs, font=("Courier", 10))
        tabs.add(self.output, text="Результаты")

        graphs_tab = tk.Frame(tabs)
        tabs.add(graphs_tab, text="Графики")

        self.figure = Figure(figsize=(10, 5))
        self.curves_ax, self.errors_ax = self.figure.subplots(1, 2)
        system_graphs.plot_curves(self.curves_ax)
        self.curves_ax.set_title("Графическое отделение решения")
        self.errors_ax.set_title("Погрешность появится после «Решить»")
        self.figure.tight_layout()

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
            eps, x0, bounds = read_input_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        self.eps_var.set(f"{eps:g}")
        for i in range(sf.N):
            self.x0_vars[i].set(f"{x0[i]:g}")
            self.low_vars[i].set(f"{bounds[i][0]:g}")
            self.high_vars[i].set(f"{bounds[i][1]:g}")

    def on_solve(self):
        try:
            eps = parse_number(self.eps_var.get())
            x0 = np.array([parse_number(var.get()) for var in self.x0_vars])
            bounds = [(parse_number(low.get()), parse_number(high.get()))
                      for low, high in zip(self.low_vars, self.high_vars)]
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля должны быть заполнены числами.")
            return
        if eps <= 0:
            messagebox.showerror("Ошибка", "Точность ε должна быть положительным числом.")
            return
        if any(low >= high for low, high in bounds):
            messagebox.showerror("Ошибка", "Для каждой переменной нижняя граница G должна быть меньше верхней.")
            return
        if not all(low <= xi <= high for xi, (low, high) in zip(x0, bounds)):
            messagebox.showerror("Ошибка", "Начальное приближение x⁰ должно лежать в области G.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(eps, x0, bounds)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    # Вывод таблицы итераций
    def write_iterations(self, title, history):
        self.write(f"\n{title}:")
        header = "".join(f"{f'x{i + 1}':>16}" for i in range(sf.N))
        self.write(f"{'k':>5}{header}{'||x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾||':>22}")
        for k, (x, error) in enumerate(history, start=1):
            self.write(f"{k:>5}{format_vector(x)}{error:>22.3e}")

    # Проверка условий сходимости простой итерации в области G (по сетке точек)
    def check_simple_iteration(self, bounds):
        points = grid_points(bounds)
        q = max(matrix_norm(sf.phi_jacobian(x)) for x in points)
        images = np.array([sf.phi(x) for x in points])

        self.write("\nМетод простой итерации: x = φ(x)")
        self.write("φ1(x) = sqrt(1 + 2·lg x2),  φ2(x) = x1 / a + 1 / x1")
        self.write(f"q = max ||φ'(x)|| по G = {q:.6g} — {'< 1, выполнено' if q < 1 else '≥ 1, НЕ выполнено'}")

        inside = True
        for i, (low, high) in enumerate(bounds):
            low_image, high_image = images[:, i].min(), images[:, i].max()
            self.write(f"φ{i + 1}(G) ⊂ [{low_image:.6g}, {high_image:.6g}], граница G: [{low:g}, {high:g}]")
            inside = inside and low <= low_image and high_image <= high
        self.write("φ(G) ⊂ G — выполнено" if inside else "φ(G) ⊂ G — НЕ выполнено")

    # Проверка условий и решение
    def show_results(self, eps, x0, bounds):
        self.write("Система (вариант 19):")
        for line in sf.DESCRIPTION:
            self.write("  " + line)
        self.write(f"\nЗаданная точность ε = {eps:g}")
        self.write(f"Начальное приближение x⁰ = ({', '.join(f'{x:g}' for x in x0)})")
        self.write("Область G: " + " × ".join(f"[{low:g}, {high:g}]" for low, high in bounds))

        self.check_simple_iteration(bounds)

        # Ньютону нужна невырожденная матрица Якоби; определитель — через LU из ЛР 1.1
        self.write("\nМетод Ньютона:")
        self.write(f"F(x⁰) = ({', '.join(f'{v:.6g}' for v in sf.F(x0))})")
        _, U, _, swaps = LU(sf.jacobian(x0))
        self.write(f"det J(x⁰) = {determinant(U, swaps):.6g} ≠ 0")

        x_simple, history_simple = simple_iteration(sf.phi, x0, eps, bounds)
        x_newton, history_newton = newton(sf.F, sf.jacobian, x0, eps, bounds)
        self.write_iterations("Метод простой итерации", history_simple)
        self.write_iterations("Метод Ньютона", history_newton)

        # Невязка ||F(x)|| — насколько найденное x не удовлетворяет системе
        self.write("\nРезультаты:")
        header = "".join(f"{f'x{i + 1}':>16}" for i in range(sf.N))
        self.write(f"{'':<20}{header}{'итераций':>10}{'||F(x)||':>12}")
        for name, x, history in (("Простая итерация", x_simple, history_simple),
                                 ("Ньютон", x_newton, history_newton)):
            self.write(f"{name:<20}{format_vector(x)}{len(history):>10}{vector_norm(sf.F(x)):>12.3e}")

        self.write("\nЧисло итераций в зависимости от точности:")
        self.write(f"{'ε':>10}{'Простая итерация':>20}{'Ньютон':>10}")
        for analysis_eps in ANALYSIS_EPS:
            counts = []
            for method, function in ((simple_iteration, (sf.phi,)), (newton, (sf.F, sf.jacobian))):
                try:
                    counts.append(str(len(method(*function, x0, analysis_eps, bounds)[1])))
                except ValueError:
                    counts.append("не сошёлся")
            self.write(f"{analysis_eps:>10.0e}{counts[0]:>20}{counts[1]:>10}")

        # Графики: кривые с траекториями методов и погрешность по итерациям
        self.curves_ax.clear()
        system_graphs.plot_curves(self.curves_ax, bounds, x0,
                                  [("Простая итерация", history_simple), ("Ньютон", history_newton)])
        self.curves_ax.set_title("Кривые f1 = 0, f2 = 0 и траектории методов")
        self.errors_ax.clear()
        system_graphs.plot_errors(self.errors_ax, history_simple, history_newton, eps)
        self.errors_ax.set_title("Погрешность по итерациям")
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    SystemApp(root)
    root.mainloop()
