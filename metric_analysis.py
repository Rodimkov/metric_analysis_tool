from abc import ABC, abstractmethod
from collections import OrderedDict
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class MetricAnalysis(ABC):

    def __init__(self, type_task, data, file_name, directory, mask):
        self.picture_directory = directory
        self.mask = mask
        self.file = file_name
        self.type_task = type_task

        self.data = data

        self.task_info = None
        self.dataset_meta = None
        self.reports = None
        self.report_type = None
        self.label_map = None

    def parser(self):
        self.processing_info = self.data.get("processing_info")
        self.dataset_meta = self.data.get("dataset_meta")
        self.reports = self.data.get("report")
        self.report_type = self.data.get("report_type")
        self.label_map = OrderedDict()

        for name in sorted(self.dataset_meta.get("label_map").keys()):
            self.label_map[name] = self.dataset_meta.get("label_map").get(name)

        #for i in range(1000):
        #   self.label_map[str(i)] = str(i)

        self.size_dataset = len(self.reports)

    def validate(self):
        report_error = []

        if not 'processing_info' in self.data:
            report_error.append('processing info')
        if not 'dataset_meta' in self.data:
            report_error.append('dataset meta')
        if not 'report' in self.data:
            report_error.append('report')
        if not 'report_type' in self.data:
            report_error.append('report type')

        #if not 'label_map' in self.data.get("dataset_meta"):
        #    report_error.append('label map')

        if report_error:
            report_error = ', '.join(report_error)
            raise KeyError("there are no keys in the file <json>: {}".format(report_error))

    def movement_buttons(self, visual_method, info_method):
        b_prev = tk.Button(self.frame_for_button, text="prev",
                           command=lambda: self.prev(visual_method, info_method))
        b_prev.grid(row=0, column=0, padx=10, pady=10)
        b_next = tk.Button(self.frame_for_button, text="next",
                           command=lambda: self.next(visual_method, info_method))
        b_next.grid(row=0, column=1, padx=10, pady=10)
        b_open = tk.Button(self.frame_for_button, text="open",
                           command=lambda: self.open_image(visual_method, info_method))
        b_open.grid(row=0, column=3, padx=10, pady=10)

    def open_image(self, visual_method, info_method):
        import os
        from tkinter import filedialog as fd

        file_name = fd.askopenfilename(title="Select file")
        file_name = os.path.basename(file_name)
        self.counter = np.where(np.array(self.index) == file_name)[0][0] - 1
        self.next(visual_method, info_method)

    def next(self, visual_method, info_method):
        self.counter += 1

        image = visual_method((self.index_image[self.counter % self.size]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        info_method()

    def prev(self, visual_method, info_method):
        self.counter -= 1

        image = visual_method((self.index_image[self.counter % self.size]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))

        image = Image.fromarray(image)
        self.image = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(self.frame_for_image, image=self.image)
        image_label.grid(row=0, column=0, padx=50, pady=10)

        info_method()

    @abstractmethod
    def metrics(self):
        pass

    @abstractmethod
    def visualize_data(self):
        pass

    @abstractmethod
    def top_n(self, n):
        pass
