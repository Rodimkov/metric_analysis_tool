from segmentation import Segmentation
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt


class GUI(Segmentation):
    def __init__(self, type_task, data, file_name, directory, mask):
        super().__init__(type_task, data, file_name, directory, mask)

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

        self.next()

        b_prev = tk.Button(self.frame_for_button, text="prev", command=self.prev)
        b_prev.grid(row=0, column=0, padx=10, pady=10)
        b_next = tk.Button(self.frame_for_button, text="next", command=self.next)
        b_next.grid(row=0, column=1, padx=10, pady=10)
        b_open = tk.Button(self.frame_for_button, text="open", command=self.open_image)
        b_open.grid(row=0, column=3, padx=10, pady=10)

    def info(self):
        name = self.index_image[self.counter]
        pass

    def open_image(self):
        import os
        from tkinter import filedialog as fd
        file_name = fd.askopenfilename(title="Select file")
        file_name = os.path.basename(file_name)
        self.counter = np.where(np.array(self.index) == file_name)[0][0] - 1
        self.next()

    def top_n(self):
        self.counter = -1
        self.size = 10

        self.master = tk.Toplevel()
        self.master.title("Top N for {}".format(self.type_task))

        self.index_image = self._top_n(n=self.size)

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

        mask = np.load(self.mask + self.predicted_mask[self.index_image[self.counter % self.size]], allow_pickle=True)
        image = cv2.imread(self.picture_directory + (self.index_image[self.counter % self.size]))
        image = self._visualize_data(image, mask)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        self.info()

    def prev(self):
        self.counter -= 1

        mask = np.load(self.mask + self.predicted_mask[self.index_image[self.counter % self.size]], allow_pickle=True)
        image = cv2.imread(self.picture_directory + (self.index_image[self.counter % self.size]))
        image = self._visualize_data(image, mask)
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
        self.variable.set("confusion matrix")
        menu = tk.OptionMenu(self.frame_top, self.variable, "confusion matrix")
        menu.config(height=1, width=10)
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

        self.multiple_next()

        b_prev = tk.Button(self.frame_for_button, text="prev", command=self.multiple_prev)
        b_prev.grid(row=0, column=0, padx=10, pady=10)
        b_next = tk.Button(self.frame_for_button, text="next", command=self.multiple_next)
        b_next.grid(row=0, column=1, padx=10, pady=10)
        b_open = tk.Button(self.frame_for_button, text="open", command=self.open_image)
        b_open.grid(row=0, column=3, padx=10, pady=10)

    def multiple_next(self):
        self.counter += 1

        images = self._multiple_visualize_data(self.set_task, self.index_image[self.counter % self.size])

        self.image = []
        for i, image in enumerate(images):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (512, 512))

            image = Image.fromarray(image)
            self.image.append(ImageTk.PhotoImage(image=image))
            tk.Label(self.frame_for_image, image=self.image[i]).grid(row=0, column=i, padx=50, pady=10)

        self.multiple_info()

    def multiple_prev(self):
        self.counter -= 1

        image = self._multiple_visualize_data(self.set_task, self.index_image[self.counter % self.size])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        self.multiple_info()

    def multiple_info(self):
        name = self.index_image[self.counter]
        pass

    def multiple_top_n(self, set_task):
        self.set_task = set_task

        self.size = 10
        self.counter = -1

        self.master = tk.Toplevel()
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
