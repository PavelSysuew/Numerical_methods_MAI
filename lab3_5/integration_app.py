import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import func_3_5
from integration import (grid, left_rectangles, right_rectangles, middle_rectangles,
                         trapezoids, simpson, runge_romberg)

# Методы и их порядок точности p
METHODS = [
    ("Левые прямоуг.", left_rectangles, 1),
    ("Правые прямоуг.", right_rectangles, 1),
    ("Средние прямоуг.", middle_rectangles, 2),
    ("Трапеции", trapezoids, 2),
    ("Симпсон", simpson, 4),
]
FIELDS = ("X₀", "Xₖ", "h₁", "h₂")


# Строку в число
def parse_number(text):
    return float(text.strip().replace(",", "."))


# Чтение данных из файла
def read_input_from_file(path):
    numbers = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            numbers.extend(parse_number(part) for part in line.split())
    if len(numbers) != 4:
        raise ValueError(f"В файле должно быть 4 числа: X₀, Xₖ, h₁, h₂. Найдено: {len(numbers)}.")
    return numbers


class IntegrationApp:
    def __init__(self, root):
        self.root = root
        root.winfo_toplevel().title("ЛР 3.5 — Численное интегрирование")

        tk.Label(root, text=f"Вариант 19.  F = ∫ y dx от X₀ до Xₖ,  {func_3_5.DESCRIPTION}.  "
                            "Прямоугольники, трапеции, Симпсон, Рунге–Ромберг.",
                 font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 0))

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.vars = {}
        for name, default in zip(FIELDS, ("0", "4", "1.0", "0.5")):
            tk.Label(top, text=f"{name}:").pack(side="left", padx=(8, 0))
            self.vars[name] = tk.StringVar(value=default)
            tk.Entry(top, width=8, textvariable=self.vars[name]).pack(side="left", padx=4)
        tk.Button(top, text="Загрузить из файла…", command=self.on_load).pack(side="left", padx=(12, 4))
        tk.Button(top, text="Решить", command=self.on_solve).pack(side="left", padx=4)

        # Поле для вывода
        self.output = scrolledtext.ScrolledText(root, font=("Courier", 10))
        self.output.pack(fill="both", expand=True, padx=8, pady=6)

    def write(self, text=""):
        self.output.insert("end", text + "\n")

    def on_load(self):
        path = filedialog.askopenfilename(title="Файл с входными данными",
                                          filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*")])
        if not path:
            return
        try:
            values = read_input_from_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка чтения файла", str(error))
            return
        for name, value in zip(FIELDS, values):
            self.vars[name].set(f"{value:g}")

    def on_solve(self):
        try:
            a, b, h1, h2 = (parse_number(self.vars[name].get()) for name in FIELDS)
        except ValueError:
            messagebox.showerror("Ошибка", "Все поля должны быть заполнены числами.")
            return
        if not a < b:
            messagebox.showerror("Ошибка", "Должно быть X₀ < Xₖ.")
            return
        if not 0 < h2 < h1:
            messagebox.showerror("Ошибка", "Шаги должны быть положительными, h₂ < h₁.")
            return

        self.output.delete("1.0", "end")
        try:
            self.show_results(a, b, h1, h2)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))

    def show_results(self, a, b, h1, h2):
        k = h1 / h2
        self.write(f"F = ∫ от {a:g} до {b:g},  {func_3_5.DESCRIPTION}")
        self.write(f"Шаги: h₁ = {h1:g}, h₂ = {h2:g},  k = h₁ / h₂ = {k:g}")

        for h in (h1, h2):
            x = grid(a, b, h)
            self.write(f"\nУзлы при h = {h:g} (N = {len(x) - 1}):")
            self.write(f"{'i':>3}{'xᵢ':>10}{'yᵢ':>16}")
            for i, xi in enumerate(x):
                self.write(f"{i:>3}{xi:>10g}{func_3_5.f(xi):>16.10f}")

        self.write("\nРезультаты (оценка погрешности по Рунге–Ромбергу: |F_h₂ − F_h₁| / (kᵖ − 1)):")
        self.write(f"{'Метод':<18}{f'h₁ = {h1:g}':>15}{f'h₂ = {h2:g}':>15}{'p':>4}"
                   f"{'Рунге–Ромберг':>16}{'оценка погр. h₂':>17}")
        results = {}
        for name, method, p in METHODS:
            F_h1 = method(func_3_5.f, a, b, h1)
            F_h2 = method(func_3_5.f, a, b, h2)
            refined, estimate = runge_romberg(F_h2, F_h1, k, p)
            results[name] = (F_h1, F_h2, refined)
            self.write(f"{name:<18}{F_h1:>15.10f}{F_h2:>15.10f}{p:>4}{refined:>16.10f}{estimate:>17.3e}")

        # Сравнение с точным значением
        exact = func_3_5.exact_integral(a, b)
        self.write(f"\nТочное значение (по первообразной): F = {exact:.10f}")
        self.write("Абсолютная погрешность |F_численное − F|:")
        self.write(f"{'Метод':<18}{f'h₁ = {h1:g}':>15}{f'h₂ = {h2:g}':>15}{'Рунге–Ромберг':>16}")
        for name, values in results.items():
            self.write(f"{name:<18}" + "".join(f"{abs(v - exact):>15.3e}" for v in values[:2])
                       + f"{abs(values[2] - exact):>16.3e}")

        best_h2 = min(results, key=lambda name: abs(results[name][1] - exact))
        best_refined = min(results, key=lambda name: abs(results[name][2] - exact))
        self.write(f"\nНаиболее точный метод с шагом h₂ = {h2:g}: {best_h2} "
                   f"(погрешность {abs(results[best_h2][1] - exact):.3e})")
        self.write(f"Наиболее точный результат после уточнения Рунге–Ромберга: {best_refined} "
                   f"(погрешность {abs(results[best_refined][2] - exact):.3e})")


if __name__ == "__main__":
    root = tk.Tk()
    IntegrationApp(root)
    root.mainloop()
