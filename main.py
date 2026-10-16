"""Лабораторная работа № 1 по численным методам — общее окно для всех заданий.

Запуск из корня проекта:  .venv/bin/python main.py

Слева меню с заданиями, справа — окно выбранного задания. Каждое задание
по-прежнему можно запустить и отдельно, например .venv/bin/python lab1_1/lu_app.py.
"""

import os
import sys
import tkinter as tk

# Папки заданий добавляем в пути поиска модулей, чтобы работали их импорты
# вида "from lu import ...": каждое задание лежит в своей папке.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for folder in ("lab1_1", "lab1_2", "lab1_3", "lab1_4", "lab1_5"):
    sys.path.insert(0, os.path.join(PROJECT_DIR, folder))

from lu_app import LuApp
from tridiagonale_app import TridiagonalApp
from iterations_app import IterationsApp
from jacobi_app import JacobiApp
from qr_app import QrApp

# (номер, название, класс окна задания)
LABS = [
    ("1.1", "LU-разложение", LuApp),
    ("1.2", "Метод прогонки", TridiagonalApp),
    ("1.3", "Метод простых итераций\nи метод Зейделя", IterationsApp),
    ("1.4", "Метод вращений Якоби", JacobiApp),
    ("1.5", "QR-разложение\nи QR-алгоритм", QrApp),
]


class MainApp:
    def __init__(self, root):
        self.root = root
        root.title("Численные методы — ЛР № 1")
        root.geometry("1250x820")

        # Левое меню
        menu = tk.Frame(root, width=230, bg="#e8e8e8")
        menu.pack(side="left", fill="y")
        menu.pack_propagate(False)  # ширина меню не зависит от длины надписей

        tk.Label(menu, text="Лабораторная работа № 1", bg="#e8e8e8",
                 font=("TkDefaultFont", 11, "bold")).pack(pady=(12, 8))

        self.buttons = []
        for index, (number, name, _) in enumerate(LABS):
            button = tk.Button(
                menu, text=f"{number}  {name}", anchor="w", justify="left",
                relief="flat", padx=10, pady=6,
                command=lambda i=index: self.show_lab(i),
            )
            button.pack(fill="x", padx=8, pady=2)
            self.buttons.append(button)

        # Правая часть: сюда помещается окно выбранного задания
        self.content = tk.Frame(root)
        self.content.pack(side="left", fill="both", expand=True)

        # Окна заданий создаются при первом открытии и потом только
        # скрываются и показываются — введённые данные не теряются
        self.frames = {}  # frames[i] — рамка, в которой построено окно задания i
        self.apps = {}    # apps[i] — объект окна задания i (LuApp, QrApp, ...)
        self.current = None

        self.show_lab(0)

    def show_lab(self, index):
        """Показывает задание с номером index из списка LABS."""
        if self.current is not None:
            self.frames[self.current].pack_forget()
            self.buttons[self.current].config(relief="flat", bg=self.root.cget("bg"))

        if index not in self.frames:
            frame = tk.Frame(self.content)
            app_class = LABS[index][2]
            self.apps[index] = app_class(frame)  # окно задания строится внутри frame
            self.frames[index] = frame

        self.frames[index].pack(fill="both", expand=True)
        self.buttons[index].config(relief="sunken", bg="#c8d8f0")
        self.current = index

        # Каждое окно задания при создании ставит свой заголовок; при
        # повторном открытии ставим его здесь
        number, name, _ = LABS[index]
        self.root.title(f"Численные методы — ЛР {number}. {name.replace(chr(10), ' ')}")


if __name__ == "__main__":
    root = tk.Tk()
    MainApp(root)
    root.mainloop()
