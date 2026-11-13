import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from derivative import derivatives_at, parabola, slope


# Строку в список чисел
def parse_numbers(text):
    return [float(part.replace(",", ".")) for part in text.split()]


# Чтение данных из файла
def read_input_from_file(path):
    with open(path, encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    if len(lines) != 3:
        raise ValueError("В файле должно быть 3 строки: xᵢ, yᵢ, X*.")
    return parse_numbers(lines[0]), parse_numbers(lines[1]), parse_numbers(lines[2])[0]


class DerivativeApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 3.4 — Численное дифференцирование")

        tk.Label(root, text="Вариант 19.  Первая и вторая производные таблично заданной функции в точке X*.",
                 font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 0))

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.vars = {}
        for name, default, width in (("xᵢ", "-1 0 1 2 3", 22),
                                     ("yᵢ", "-1.7854 0 1.7854 3.1071 4.249", 34),
                                     ("X*", "1", 8)):
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
        self.vars["yᵢ"].set(" ".join(f"{y:g}" for y in Y))
        self.vars["X*"].set(f"{x_star:g}")

    def on_solve(self):
        try:
            X = np.array(parse_numbers(self.vars["xᵢ"].get()))
            Y = np.array(parse_numbers(self.vars["yᵢ"].get()))
            x_star = parse_numbers(self.vars["X*"].get())[0]
        except (ValueError, IndexError):
            messagebox.showerror("Ошибка", "Все поля должны быть числами через пробел.")
            return
        if len(X) != len(Y):
            messagebox.showerror("Ошибка", f"Узлов xᵢ: {len(X)}, значений yᵢ: {len(Y)} — должно быть поровну.")
            return
        if np.any(np.diff(X) <= 0):
            messagebox.showerror("Ошибка", "Узлы xᵢ должны строго возрастать.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(X, Y, x_star)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    def show_results(self, X, Y, x_star):
        result = derivatives_at(X, Y, x_star)
        i = result["start"]

        self.write("Таблица и наклоны хорд Δᵢ = (yᵢ₊₁ − yᵢ) / (xᵢ₊₁ − xᵢ):")
        self.write(f"{'i':>3}{'xᵢ':>8}{'yᵢ':>12}{'Δᵢ':>14}")
        for k in range(len(X)):
            delta = f"{slope(X, Y, k):>14.6f}" if k < len(X) - 1 else ""
            self.write(f"{k:>3}{X[k]:>8g}{Y[k]:>12.6f}{delta}")

        self.write(f"\nТочка вычисления: X* = {x_star:g}")

        # Номер узла, который совпадает с X*
        node = next((k for k in range(len(X)) if abs(X[k] - x_star) < 1e-12), None)

        def nodes_text(*indices):
            return ", ".join(f"x{k} = {X[k]:g}" for k in indices)

        self.write("\nПервая производная, первый порядок точности (по двум узлам):")
        if "left" in result:
            self.write(f"  левая  разность y'(X*) ≈ {result['left']:.6f}    узлы: {nodes_text(node - 1, node)}")
        if "right" in result:
            self.write(f"  правая разность y'(X*) ≈ {result['right']:.6f}    узлы: {nodes_text(node, node + 1)}")
        if "segment" in result:
            j = next(j for j in range(len(X) - 1) if x_star < X[j + 1])
            self.write(f"  по отрезку с X*: y'(X*) ≈ {result['segment']:.6f}    узлы: {nodes_text(j, j + 1)}")

        self.write(f"\nВторой порядок точности, узлы: {nodes_text(i, i + 1, i + 2)} (парабола через них)")
        self.write(f"  y'(X*)  ≈ Δ{i} + (Δ{i + 1} − Δ{i}) / (x{i + 2} − x{i}) · (2X* − x{i} − x{i + 1}) "
                   f"= {result['first']:.6f}")
        self.write(f"  y''(X*) ≈ 2(Δ{i + 1} − Δ{i}) / (x{i + 2} − x{i}) = {result['second']:.6f}")

        self.draw_graph(X, Y, x_star, result)

    # График
    def draw_graph(self, X, Y, x_star, result):
        i = result["start"]
        x = np.linspace(X[0], X[-1], 400)
        y_star = parabola(X, Y, i, x_star)
        width = 0.4 * (X[-1] - X[0])
        x_line = np.linspace(x_star - width / 2, x_star + width / 2, 2)

        self.ax.clear()
        self.ax.plot(X, Y, "ko-", linewidth=0.8, label="таблица (ломаная)")
        self.ax.plot(x, parabola(X, Y, i, x), color="tab:orange",
                     label=f"парабола по узлам {i}, {i + 1}, {i + 2}")
        self.ax.plot(x_line, y_star + result["first"] * (x_line - x_star), color="tab:red",
                     label=f"касательная, y' ≈ {result['first']:.4f}")
        for key, color in (("left", "tab:green"), ("right", "tab:purple"), ("segment", "tab:green")):
            if key in result:
                self.ax.plot(x_line, y_star + result[key] * (x_line - x_star), "--", color=color,
                             label=f"{'левая' if key == 'left' else 'правая' if key == 'right' else 'хорда'}: "
                                   f"{result[key]:.4f}")
        self.ax.plot(x_star, y_star, "r*", markersize=12)
        self.ax.set_title(f"Производная в точке X* = {x_star:g}")
        self.ax.grid(True)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    DerivativeApp(root)
    root.mainloop()
