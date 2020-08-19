from detection import Detection
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt


class GUI(Detection):
    def __init__(self, type_task, data, file_name, directory, mask):
        super().__init__(type_task, data, file_name, directory, mask)
        self.threshold_scores = 0.5
        self.size_acc_changes = 100

    def visualize_data(self):
        self.counter = -1
        self.index_image = self.index
        self.size = self.size_dataset

        self.master = tk.Toplevel()
        self.master.title("Visualize data for {}".format(self.type_task))

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.next(self._visualize_data, self.info)

        self.movement_buttons(self._visualize_data, self.info)

    def set_threshold(self):
        self.threshold_scores = float(self.message.get())
        self.counter -= 1

        self.next(self._visualize_data, self.info)

    def info(self):
        name = self.index_image[self.counter]
        text_name = name
        if len(text_name) > 28:
            text_name = text_name[:28] + '\n' + text_name[28:]

        self.frame_top = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_top.grid(row=0, column=0, padx=10, pady=10)

        self.frame_bot = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_bot.grid(row=1, column=0, padx=10, pady=10)

        tk.Label(self.frame_top, height=2, width=45, text="picture name: {}".format(text_name)).pack(side=tk.TOP,
                                                                                                     pady=10)
        tk.Label(self.frame_top, height=1, width=30, text="threshold score").pack(side=tk.TOP, pady=10)

        message_button = tk.Button(self.frame_bot, text="Change value", command=self.set_threshold)
        message_button.pack(side=tk.LEFT, pady=10, padx=25)

        self.message = tk.StringVar()
        message_entry = tk.Entry(self.frame_bot, textvariable=self.message, width=10)
        message_entry.insert(tk.END, str(self.threshold_scores))
        message_entry.pack(side=tk.LEFT, pady=10)

    def top_n(self):
        self.master = tk.Toplevel()
        self.master.geometry("300x150+500+500")
        self.size = 10

        self.frame_bot = tk.Frame(self.master, padx=25, pady=2)
        self.frame_bot.pack(anchor="n")
        tk.Label(self.frame_bot, height=1, width=15, text="N").pack(side=tk.LEFT, pady=5)

        self.message_n = tk.StringVar()
        message_entry = tk.Entry(self.frame_bot, textvariable=self.message_n)
        message_entry.insert(tk.END, str(self.size))
        message_entry.pack(side=tk.LEFT, pady=10, padx=2)

        tk.Button(self.master, text="GO", command=self.top_n_solver).pack()

    def top_n_solver(self):
        self.size = int(self.message_n.get())
        self.master.destroy()

        self.counter = -1

        self.master = tk.Toplevel()
        self.master.title("Top N for {}".format(self.type_task))

        self.index_image = self._top_n(n=self.size)

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.next(self._visualize_data, self.info)

        self.movement_buttons(self._visualize_data, self.info)

    def metrics(self):
        self.counter = -1
        self.master = tk.Toplevel()
        self.master.title("Metric for {}".format(self.type_task))

        self.frame_top = tk.Frame(self.master, padx=15, pady=2)
        self.frame_top.pack(anchor="n")

        self.frame_bot = tk.Frame(self.master, padx=15, pady=2)
        self.frame_bot.pack(anchor="n")

        self.variable = tk.StringVar(self.frame_top)
        self.variable.set("average precision changes")

        menu = tk.OptionMenu(self.frame_top, self.variable, "average precision changes")
        menu.config(height=1, width=25)
        menu.pack(side=tk.LEFT, pady=10)

        tk.Button(master=self.frame_top, text="plot", command=lambda: self.treatment(canvas, ax),
                  height=1, width=10).pack(side=tk.LEFT, pady=10)

        self.message = tk.StringVar()
        tk.Label(master=self.frame_bot, height=1, width=15, text="window size = ").pack(side=tk.LEFT, pady=10)
        tk.Entry(master=self.frame_bot, textvariable=self.message, width=7).pack(side=tk.LEFT, pady=10)

        self.frame_plot = tk.Frame(self.master, width=800, height=800)
        self.frame_plot.pack(fill='both', expand=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.frame_plot)
        toolbar.update()

    def treatment(self, canvas, ax):
        option = {"average precision changes": self.plot_average_precision_changes, }
        if self.variable.get() == "average precision changes":
            try:
                self.size_acc_changes = int(self.message.get())
            except ValueError:
                pass

        option[self.variable.get()](canvas, ax)

    def plot_average_precision_changes(self, canvas, ax):
        ax.clear()
        self._plot_average_precision_changes(ax, k=self.size_acc_changes)

        canvas.draw()

    def multiple_visualize_data(self, set_task):
        self.counter = -1
        self.index_image = self.index
        self.size = self.size_dataset

        self.set_task = set_task

        self.master = tk.Toplevel()
        self.master.title("Visualize data for {}".format(self.type_task))

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.next(self._multiple_visualize_data, self.multiple_info)

        self.movement_buttons(self._multiple_visualize_data, self.multiple_info)

    def multiple_info(self):
        name = self.index_image[self.counter]
        text_name = name
        if len(text_name) > 28:
            text_name = text_name[:28] + '\n' + text_name[28:]

        self.frame_top = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_top.grid(row=0, column=0, padx=10, pady=10)

        self.frame_bot = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_bot.grid(row=1, column=0, padx=10, pady=10)

        tk.Label(self.frame_top, height=2, width=45, text="picture name: {}".format(text_name)).pack(side=tk.TOP,
                                                                                                     pady=10)
        tk.Label(self.frame_top, height=1, width=30, text="threshold score").pack(side=tk.TOP, pady=10)

        message_button = tk.Button(self.frame_bot, text="Change value", command=self.set_threshold)
        message_button.pack(side=tk.LEFT, pady=10, padx=25)

        self.message = tk.StringVar()
        message_entry = tk.Entry(self.frame_bot, textvariable=self.message, width=10)
        message_entry.insert(tk.END, str(self.threshold_scores))
        message_entry.pack(side=tk.LEFT, pady=10)

    def multiple_set_threshold(self):
        self.threshold_scores = float(self.message.get())
        self.counter -= 1
        self.next(self._multiple_visualize_data, self.multiple_info)

    def multiple_top_n(self, set_task):
        self.master = tk.Toplevel()
        self.master.geometry("300x150+500+500")
        self.set_task = set_task
        self.size = 10

        self.frame_top = tk.Frame(self.master, padx=25, pady=2)
        self.frame_top.pack(anchor="n")

        self.frame_bot = tk.Frame(self.master, padx=25, pady=2)
        self.frame_bot.pack(anchor="n")

        tk.Label(self.frame_top, height=1, width=15, text="threshold score").pack(side=tk.LEFT, pady=5)

        self.message_tresh = tk.StringVar()
        message_entry = tk.Entry(self.frame_top, textvariable=self.message_tresh)
        message_entry.insert(tk.END, str(self.threshold_scores))
        message_entry.pack(side=tk.LEFT, pady=10, padx=2)

        tk.Label(self.frame_bot, height=1, width=15, text="N").pack(side=tk.LEFT, pady=5)

        self.message_n = tk.StringVar()
        message_entry = tk.Entry(self.frame_bot, textvariable=self.message_n)
        message_entry.insert(tk.END, str(self.size))
        message_entry.pack(side=tk.LEFT, pady=10, padx=2)

        b_prev = tk.Button(self.master, text="GO", command=self.multiple_top_n_solver)
        b_prev.pack()

    def multiple_top_n_solver(self):
        self.threshold_scores = float(self.message_tresh.get())
        self.size = int(self.message_n.get())
        self.master.destroy()

        self.counter = -1

        self.master = tk.Toplevel()
        self.master.title("Visualize data for {}".format(self.type_task))

        self.index_image = self.set_task[0]._multiple_top_n(self.set_task, self.threshold_scores, n=self.size)

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.next(self._multiple_visualize_data, self.multiple_info)

        self.movement_buttons(self._multiple_visualize_data, self.multiple_info)
