from segmentation import Segmentation
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt


class GUI(Segmentation):
    def __init__(self, type_task, data, file_name, directory, mask, true_mask):
        super().__init__(type_task, data, file_name, directory, mask, true_mask)
        self.flag_prediction = False
        self.flag_annotation = False

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

    def applied_masks(self, visualize_data, info):
        self.counter -= 1
        self.flag_prediction = self.prediction_state.get()
        self.flag_annotation = self.annotation_state.get()
        self.next(visualize_data, info)

    def info(self):
        name = self.index_image[self.counter % self.size]
        text_name = name
        if len(text_name) > 13:
            text_name = text_name[:13] + '\n' + text_name[13:]

        self.frame_box = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_box.grid(row=0, column=0, padx=10, pady=10)

        self.frame_top = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_top.grid(row=1, column=0, padx=10, pady=10)

        self.frame_mid = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_mid.grid(row=2, column=0, padx=10, pady=10)

        self.frame_bot = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_bot.grid(row=3, column=0, padx=10, pady=10)

        self.annotation_state = tk.BooleanVar()
        self.annotation_state.set(self.flag_annotation)
        chk = tk.Checkbutton(self.frame_box, text='annotation box', var=self.annotation_state)
        chk.pack(side=tk.LEFT, pady=10)

        self.prediction_state = tk.BooleanVar()
        self.prediction_state.set(self.flag_prediction)
        chk = tk.Checkbutton(self.frame_box, text='prediction box', var=self.prediction_state)
        chk.pack(side=tk.LEFT, pady=10)

        tk.Button(self.frame_top, text="Change applied mask",
                  command=lambda: self.applied_masks(self._visualize_data, self.info)).pack()

        tk.Label(self.frame_mid, height=2, width=30, text="picture name: {}".format(text_name)).pack(side=tk.TOP,
                                                                                                     pady=10)
        self.square_colors = []
        for i, color in enumerate(self.segmentation_colors):
            square = np.zeros((7,7,3))
            square[:,:] = color
            square = square.astype(np.uint8)
            square = cv2.cvtColor(square, cv2.COLOR_RGB2BGR)

            square = Image.fromarray(square)
            self.square_colors.append(ImageTk.PhotoImage(image=square))
            tk.Label(self.frame_bot, image=self.square_colors[i]).grid(row=int(i/2), column=2*(i%2), padx=10, pady=3)
            tk.Label(self.frame_bot, text=self.label_map[str(i)]).grid(row=int(i/2), column=2*(i%2)+1, padx=10, pady=3)

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

        self.frame_top = tk.Frame(self.master, padx=100, pady=10)
        self.frame_top.pack(anchor="n")

        self.variable = tk.StringVar(self.frame_top)
        self.variable.set("confusion matrix")
        menu = tk.OptionMenu(self.frame_top, self.variable, "confusion matrix")
        menu.config(height=1, width=20)
        menu.grid(row=0, column=0, padx=10, pady=10)

        tk.Button(master=self.frame_top, text="plot", command=lambda: self.treatment(canvas, ax),
                  height=1, width=10).grid(row=0, column=1, padx=10, pady=10)

        self.frame_image = tk.Frame(self.master, width=800, height=800)
        self.frame_image.pack(fill='both', expand=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        canvas = FigureCanvasTkAgg(fig, master=self.frame_image)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.frame_image)
        toolbar.update()

    def treatment(self, canvas, ax):
        option = {"confusion matrix": self.plot_confusion_matrix}
        if self.variable.get() == "accuracy changes":
            try:
                self.size_acc_changes = int(self.message.get())
            except ValueError:
                pass

        option[self.variable.get()](canvas, ax)

    def plot_confusion_matrix(self, canvas, ax):
        ax.clear()
        self._plot_confusion_matrix(ax)

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
        name = self.index_image[self.counter % self.size]
        text_name = name
        if len(text_name) > 13:
            text_name = text_name[:13] + '\n' + text_name[13:]

        self.frame_box = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_box.grid(row=0, column=0, padx=10, pady=10)

        self.frame_top = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_top.grid(row=1, column=0, padx=10, pady=10)

        self.frame_mid = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_mid.grid(row=2, column=0, padx=10, pady=10)

        self.frame_bot = tk.Frame(self.frame_for_info, padx=25, pady=2)
        self.frame_bot.grid(row=3, column=0, padx=10, pady=10)

        self.annotation_state = tk.BooleanVar()
        self.annotation_state.set(self.flag_annotation)
        chk = tk.Checkbutton(self.frame_box, text='annotation box', var=self.annotation_state)
        chk.pack(side=tk.LEFT, pady=10)

        self.prediction_state = tk.BooleanVar()
        self.prediction_state.set(self.flag_prediction)
        chk = tk.Checkbutton(self.frame_box, text='prediction box', var=self.prediction_state)
        chk.pack(side=tk.LEFT, pady=10)

        tk.Button(self.frame_top, text="Change applied mask",
                  command=lambda: self.applied_masks(self._multiple_visualize_data, self.multiple_info)).pack()

        tk.Label(self.frame_mid, height=2, width=30, text="picture name: {}".format(text_name)).pack(side=tk.TOP,
                                                                                                     pady=10)
        self.square_colors = []
        for i, color in enumerate(self.segmentation_colors):
            square = np.zeros((7,7,3))
            square[:,:] = color
            square = square.astype(np.uint8)
            square = cv2.cvtColor(square, cv2.COLOR_RGB2BGR)

            square = Image.fromarray(square)
            self.square_colors.append(ImageTk.PhotoImage(image=square))
            tk.Label(self.frame_bot, image=self.square_colors[i]).grid(row=int(i/2), column=2*(i%2), padx=10, pady=3)
            tk.Label(self.frame_bot, text=self.label_map[str(i)]).grid(row=int(i/2), column=2*(i%2)+1, padx=10, pady=3)

    def multiple_top_n(self, set_task):
        self.master = tk.Toplevel()
        self.master.geometry("300x150+500+500")
        self.set_task = set_task
        self.size = 10

        self.frame_bot = tk.Frame(self.master, padx=25, pady=2)
        self.frame_bot.pack(anchor="n")

        tk.Label(self.frame_bot, height=1, width=15, text="N").pack(side=tk.LEFT, pady=5)

        self.message_n = tk.StringVar()
        message_entry = tk.Entry(self.frame_bot, textvariable=self.message_n)
        message_entry.insert(tk.END, str(self.size))
        message_entry.pack(side=tk.LEFT, pady=10, padx=2)

        tk.Button(self.master, text="GO", command=self.multiple_top_n_solver).pack()

    def multiple_top_n_solver(self):
        self.size = int(self.message_n.get())
        self.master.destroy()

        self.counter = -1

        self.master = tk.Toplevel()
        self.master.title("top n for {}".format(self.type_task))

        self.index_image = self.set_task[0]._multiple_top_n(self.set_task, n=self.size)

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.next(self._multiple_visualize_data, self.multiple_info)

        self.movement_buttons(self._multiple_visualize_data, self.multiple_info)
