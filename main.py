import NumbersNN
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory

class App(tk.Tk):

    def __init__(self):

        super().__init__()          # tk.Tk.__init__(self)

        self.title("NumbersNN")
        self.resizable(False, False)

        self.__cell_data = [[0] * 28 for _ in range(28)]
        self.__cell_size = 15

        self.__ai_logic = NumbersNN.NumbersNN()

        self.createMenu()
        self.createWidgets()


    def createMenu(self):
        
        self.__menu = tk.Menu()
        self.configure(menu=self.__menu)
        self.__file_menu = tk.Menu(self.__menu)
        self.__menu.add_cascade(label="File", menu=self.__file_menu)
        self.__file_menu.add_command(label="Загрузить модель", command=lambda: self.__ai_logic.load_model(askopenfilename(filetypes=[("ML model", "*.pth")])))
        self.__file_menu.add_command(label="Выгрузить модель", command=self.__ai_logic.unload_model)
        self.__file_menu.add_command(label="Создать пустую модель", command=lambda: self.__ai_logic.create_model(input("Введите название модели: ")))
        self.__file_menu.add_command(label="Обучить модель", command=lambda: self.__ai_logic.train_model(askopenfilename(filetypes=[("CSV file", "*.csv")]),
                                                                                                 askdirectory(),
                                                                                                 askopenfilename(filetypes=[("ML model", "*.pth")]), 10))


    def createWidgets(self):

        self.__frame1 = tk.Frame(self)
        self.__frame2 = tk.Frame(self)

        self.__frame1.pack(side='left')
        self.__frame2.pack(side='left')

        self.__canvas = tk.Canvas(self.__frame1, width=self.__cell_size*28, height=self.__cell_size*28, bg='black')

        self.__canvas.pack()

        self.__progress_bars_list = []
        for i in range(10):
            self.__progress_bars_list.append(ttk.Progressbar(self.__frame2))
            tk.Label(self.__frame2, text=i).grid(column=0, row=i, pady=4)
            self.__progress_bars_list[i].grid(column=1, row=i, pady=4)

        self.__canvas.bind("<B1-Motion>", lambda e : self.draw(e, False))
        self.__canvas.bind("<B3-Motion>", lambda e : self.draw(e, True))


    def draw(self, event, is_black: bool):

        color, n = ('black', 0) if is_black else ('white', 255)

        try:
            self.__cell_data[event.y // self.__cell_size ][event.x // self.__cell_size ] = n
            self.__cell_data[event.y // self.__cell_size ][event.x // self.__cell_size + 1] = n
            self.__cell_data[event.y // self.__cell_size + 1][event.x // self.__cell_size ] = n
            self.__cell_data[event.y // self.__cell_size + 1][event.x // self.__cell_size + 1] = n
        except:
            pass

        x1 = event.x // self.__cell_size * self.__cell_size
        y1 = event.y // self.__cell_size * self.__cell_size

        x2 = event.x // self.__cell_size * self.__cell_size + self.__cell_size * 2
        y2 = event.y // self.__cell_size * self.__cell_size + self.__cell_size * 2

        self.__canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

        self.__ai_logic.check_number([[self.__cell_data]], self.__progress_bars_list)


if __name__ == '__main__':
    root = App()
    root.mainloop()