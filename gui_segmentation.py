from  segmentation import Segmentation
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


class GUI(Segmentation):
    def __init__(self, type_task, data, file_name, directory, mask):
        super().__init__(type_task, data, file_name, directory, mask)

    def visualize_data(self):
        self.counter = -1
        self.master = tk.Toplevel()
        self.master.title("Visualize data for {}".format(self.type_task))
        self.buttonframe0 = tk.Frame(self.master)
        self.buttonframe0.grid(row=0, column=1)

        buttonframe1 = tk.Frame(self.master)
        buttonframe1.grid(row=1, column=0)

        self.next()

        self.b_prev = tk.Button(buttonframe1, text="prev", command=self.prev)
        self.b_prev.grid(row=0, column=0, padx=10, pady=10)
        self.b_next = tk.Button(buttonframe1, text="next", command=self.next)
        self.b_next.grid(row=0, column=1, padx=10, pady=10)

        self.b_next = tk.Button(buttonframe1, text="open", command=self.extractText)
        self.b_next.grid(row=0, column=3, padx=10, pady=10)

    def extractText(self):
        import os
        from tkinter import filedialog as fd
        file_name = fd.askopenfilename(title="Select file")
        file_name = os.path.basename(file_name)
        self.counter = np.where(np.array(self.index) == file_name)[0][0] - 1
        self.next()

    def info(self):
        name = self.index[self.counter]
        w0 = tk.Label(self.buttonframe0, height=1, width=45, text="picture name: {}".format(name))
        w0.grid(row=0, column=0, padx=10, pady=10)


    def next(self):
        self.counter += 1
        image = cv2.imread(self.picture_directory + (self.index[self.counter % self.size_dataset]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        im = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=im)
        self.imageLabel2 = tk.Label(self.master, image=self.image)
        self.imageLabel2.grid(row=0, column=0, padx=50, pady=10)
        self.info()

    def prev(self):
        self.counter -= 1
        image = cv2.imread(self.picture_directory + (self.index[self.counter % self.size_dataset]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        im = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=im)
        self.imageLabel2 = tk.Label(self.master, image=self.image)
        self.imageLabel2.grid(row=0, column=0, padx=50, pady=10)
        self.info()