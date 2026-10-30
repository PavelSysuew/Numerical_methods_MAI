import os
import sys
import tkinter as tk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
for folder in ("lab2_1", "lab2_2"):
    sys.path.insert(0, os.path.join(PROJECT_DIR, folder))

from roots_app import RootsApp
from system_app import SystemApp

# Номер, название и окно задания
LABS = [
    ("2.1", "Метод простой итерации\nи метод Ньютона", RootsApp),
    ("2.2", "Системы нелинейных\nуравнений", SystemApp),
]


# Главное окно: слева меню, справа выбранное задание
class MainApp:
    def __init__(self, root):
        self.root = root
        root.title("Численные методы — ЛР № 2")
        root.geometry("1250x820")

        # Меню слева
        menu = tk.Frame(root, width=230, bg="#e8e8e8")
        menu.pack(side="left", fill="y")
        menu.pack_propagate(False)  # фиксированная ширина

        tk.Label(menu, text="Лабораторная работа № 2", bg="#e8e8e8",
                 font=("TkDefaultFont", 11, "bold")).pack(pady=(12, 8))

        self.buttons = []
        for index, (number, name, _) in enumerate(LABS):
            button = tk.Button(
                menu, text=f"{number}  {name}", anchor="w", justify="left",
                relief="flat", padx=10, pady=6,
                command=lambda i=index: self.show_lab(i),  # i=index, чтобы кнопка запомнила свой номер
            )
            button.pack(fill="x", padx=8, pady=2)
            self.buttons.append(button)

        # Справа — окно задания
        self.content = tk.Frame(root)
        self.content.pack(side="left", fill="both", expand=True)

        self.frames = {}  # рамки заданий, создаются один раз
        self.apps = {}    # объекты окон заданий
        self.current = None

        self.show_lab(0)

    # Переключение на задание index
    def show_lab(self, index):
        if self.current is not None:
            self.frames[self.current].pack_forget()  # прячем, а не удаляем — данные сохраняются
            self.buttons[self.current].config(relief="flat", bg=self.root.cget("bg"))

        if index not in self.frames:
            frame = tk.Frame(self.content)
            app_class = LABS[index][2]
            self.apps[index] = app_class(frame)
            self.frames[index] = frame

        self.frames[index].pack(fill="both", expand=True)
        self.buttons[index].config(relief="sunken", bg="#c8d8f0")
        self.current = index

        number, name, _ = LABS[index]
        self.root.title(f"Численные методы — ЛР {number}. {name.replace(chr(10), ' ')}")


if __name__ == "__main__":
    root = tk.Tk()
    MainApp(root)
    root.mainloop()
