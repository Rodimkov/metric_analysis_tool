from classification import Classification
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt


class GUI(Classification):
    def __init__(self, type_task, data, file_name, directory, mask):
        super().__init__(type_task, data, file_name, directory, mask)
        self.size_acc_changes = 100

    def visualize_data(self):
        self.counter = -1
        self.index_image = self.index
        self.master = tk.Toplevel()
        self.size = self.size_dataset
        self.master.title("Visualize data for {}".format(self.type_task))

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.next()

        b_prev = tk.Button(self.frame_for_button, text="prev", command=self.prev)
        b_prev.grid(row=0, column=0, padx=10, pady=10)
        b_next = tk.Button(self.frame_for_button, text="next", command=self.next)
        b_next.grid(row=0, column=1, padx=10, pady=10)
        b_open = tk.Button(self.frame_for_button, text="open", command=self.open_image)
        b_open.grid(row=0, column=3, padx=10, pady=10)

    def info(self):
        name = self.index_image[self.counter]

        pred_label = self.label_map.get(str(self.prediction_label[name]))
        true_label = self.label_map.get(str(self.annotation_label[name]))
        value = np.max(self.prediction_scores[name])

        pred_label = "prediction class: {}".format(pred_label)
        true_label = "annotation class: {}".format(true_label)
        value_label = "prediction score: {}".format(np.around(value, 3))

        tk.Label(self.frame_for_info, height=1, width=45, text="picture name: {}".format(name)).grid(row=0, column=0,
                                                                                                     padx=10, pady=10)
        tk.Label(self.frame_for_info, height=1, width=45, text=pred_label).grid(row=1, column=0, padx=10, pady=10)
        tk.Label(self.frame_for_info, height=1, width=45, text=true_label).grid(row=2, column=0, padx=10, pady=10)
        tk.Label(self.frame_for_info, height=1, width=45, text=value_label).grid(row=3, column=0, padx=10, pady=10)

    def open_image(self):
        import os
        from tkinter import filedialog as fd
        file_name = fd.askopenfilename(title="Select file")
        file_name = os.path.basename(file_name)
        self.counter = np.where(np.array(self.index) == file_name)[0][0] - 1
        self.next()

    def top_n(self):
        self.counter = -1
        self.master = tk.Toplevel()
        self.n = 10
        self.index_image = self._top_n(n=self.n)
        self.size = self.n
        self.master.title("Top N for {}".format(self.type_task))

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.next()

        b_prev = tk.Button(self.frame_for_button, text="prev", command=self.prev)
        b_prev.grid(row=0, column=0, padx=10, pady=10)
        b_next = tk.Button(self.frame_for_button, text="next", command=self.next)
        b_next.grid(row=0, column=1, padx=10, pady=10)
        b_open = tk.Button(self.frame_for_button, text="open", command=self.open_image)
        b_open.grid(row=0, column=3, padx=10, pady=10)

    def next(self):
        self.counter += 1

        image = cv2.imread(self.picture_directory + (self.index_image[self.counter % self.size]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        self.info()

    def prev(self):
        self.counter -= 1

        image = cv2.imread(self.picture_directory + (self.index_image[self.counter % self.size]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        self.info()

    def metrics(self):
        self.counter = -1
        self.master = tk.Toplevel()
        self.master.title("Metric for {}".format(self.type_task))

        self.frame_top = tk.Frame(self.master, padx=100, pady=10)
        self.frame_top.pack(anchor="n")

        self.variable = tk.StringVar(self.frame_top)
        self.variable.set("accuracy changes")
        menu = tk.OptionMenu(self.frame_top, self.variable, "accuracy changes", "confusion matrix")
        menu.config(height=1, width=10)
        menu.grid(row=0, column=0, padx=10, pady=10)

        tk.Button(master=self.frame_top, text="plot", command=lambda: self.treatment(canvas, ax),
                  height=1, width=10).grid(row=0, column=1, padx=10, pady=10)

        self.message = tk.StringVar()

        tk.Entry(master=self.frame_top, textvariable=self.message).grid(row=0, column=2, padx=10, pady=10)

        self.frame_image = tk.Frame(self.master, width=800, height=800)
        self.frame_image.pack(fill='both', expand=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        canvas = FigureCanvasTkAgg(fig, master=self.frame_image)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.frame_image)
        toolbar.update()

    def treatment(self, canvas, ax):
        option = {"accuracy changes": self.plot_accuracy_changes, "confusion matrix": self.plot_confusion_matrix}
        try:
            self.size_acc_changes = int(self.message.get())
        except ValueError:
            pass

        option[self.variable.get()](canvas, ax)

    def plot_accuracy_changes(self, canvas, ax):
        ax.clear()
        self._plot_accuracy_changes(ax, k=self.size_acc_changes)

        canvas.draw()

    def plot_confusion_matrix(self, canvas, ax):
        ax.clear()
        self._plot_confusion_matrix(ax)

        canvas.draw()

    def multiple_visualize_data(self, set_task):
        self.counter = -1
        self.set_task = set_task
        self.index_image = self.index
        self.master = tk.Toplevel()
        self.size = self.size_dataset
        self.master.title("Visualize data for {}".format(self.type_task))

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.multiple_next()

        b_prev = tk.Button(self.frame_for_button, text="prev", command=self.multiple_prev)
        b_prev.grid(row=0, column=0, padx=10, pady=10)
        b_next = tk.Button(self.frame_for_button, text="next", command=self.multiple_next)
        b_next.grid(row=0, column=1, padx=10, pady=10)
        b_open = tk.Button(self.frame_for_button, text="open", command=self.open_image)
        b_open.grid(row=0, column=3, padx=10, pady=10)

    def multiple_next(self):
        self.counter += 1
        image = cv2.imread(self.picture_directory + (self.index_image[self.counter % self.size]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        self.multiple_info()

    def multiple_prev(self):
        self.counter -= 1

        image = cv2.imread(self.picture_directory + (self.index_image[self.counter % self.size]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        self.multiple_info()

    def multiple_info(self):
        name = self.index_image[self.counter]

        pred_label = self.label_map.get(str(self.set_task[1].prediction_label[name]))
        true_label = self.label_map.get(str(self.set_task[0].prediction_label[name]))
        pred_value = np.max(self.set_task[1].prediction_scores[name])
        true_value = np.max(self.set_task[0].prediction_scores[name])

        pred_label = "prediction class in 1 file: {}".format(pred_label)
        true_label = "prediction class in 2 file: {}".format(true_label)
        value_label_pred = "prediction score in 1 file: {}".format(np.around(pred_value, 3))
        value_label_true = "prediction score in 1 file: {}".format(np.around(true_value, 3))

        tk.Label(self.frame_for_info, height=1, width=45, text="picture name: {}".format(name)).grid(row=0, column=0,
                                                                                                     padx=10, pady=10)
        tk.Label(self.frame_for_info, height=1, width=45, text=pred_label).grid(row=1, column=0, padx=10, pady=10)
        tk.Label(self.frame_for_info, height=1, width=45, text=true_label).grid(row=2, column=0, padx=10, pady=10)
        tk.Label(self.frame_for_info, height=1, width=45, text=value_label_pred).grid(row=3, column=0, padx=10, pady=10)
        tk.Label(self.frame_for_info, height=1, width=45, text=value_label_true).grid(row=4, column=0, padx=10, pady=10)

    def multiple_top_n(self, set_task):
        self.n = 10
        self.counter = -1
        self.set_task = set_task
        self.index_image = self.set_task[0]._multiple_top_n(set_task)
        self.master = tk.Toplevel()
        self.size = self.n
        self.master.title("Visualize data for {}".format(self.type_task))

        self.frame_for_image = tk.Frame(self.master)
        self.frame_for_image.grid(row=0, column=0)
        self.frame_for_button = tk.Frame(self.master)
        self.frame_for_button.grid(row=1, column=0)
        self.frame_for_info = tk.Frame(self.master)
        self.frame_for_info.grid(row=0, column=1)

        self.multiple_next()

        b_prev = tk.Button(self.frame_for_button, text="prev", command=self.multiple_prev)
        b_prev.grid(row=0, column=0, padx=10, pady=10)
        b_next = tk.Button(self.frame_for_button, text="next", command=self.multiple_next)
        b_next.grid(row=0, column=1, padx=10, pady=10)
        b_open = tk.Button(self.frame_for_button, text="open", command=self.open_image)
        b_open.grid(row=0, column=3, padx=10, pady=10)
