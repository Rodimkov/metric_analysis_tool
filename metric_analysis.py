from abc import ABC, abstractmethod
from collections import OrderedDict
import cv2
import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import numpy as np
import os
import warnings


class MetricAnalysis(ABC):

    def __init__(self, type_task, data, file_name, directory, mask, true_mask):
        self.picture_directory = directory
        self.mask = mask
        self.file = file_name
        self.type_task = type_task
        self.directory_true_mask = true_mask

        self.data = data

        self.task_info = None
        self.dataset_meta = None
        self.reports = None
        self.report_type = None
        self.label_map = None

    def set_winwow_size(self, width, height):
        self.winwow_width = int(width / 1.8)
        self.winwow_height = int(height / 1.5)

    def parser(self):
        self.processing_info = self.data.get("processing_info")
        self.dataset_meta = self.data.get("dataset_meta")
        self.reports = self.data.get("report")
        self.report_type = self.data.get("report_type")
        self.label_map = OrderedDict()

        for name in sorted(self.dataset_meta.get("label_map").keys()):
            self.label_map[name] = self.dataset_meta.get("label_map").get(name)

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

        if not 'label_map' in self.data.get("dataset_meta"):
            report_error.append('label map')

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

        file_name = fd.askopenfilename(title="Select file")
        file_name = file_name[file_name.find(self.picture_directory):]
        file_name = file_name.replace(self.picture_directory,"")

        if self.identifier.get(file_name) is None:
            warnings.warn("""We could not find such a picture in the sent file. Perhaps you have chosen the wrong 
                          picture, perhaps we are not working correctly with relative paths (if not the names of the 
                          pictures, but relative paths are specified in the * .json file, then the script may not 
                          work correctly)""")
        else:
            self.counter = np.where(np.array(self.index) == file_name)[0][0] - 1
            self.next(visual_method, info_method)

    def image_resize(self, image, coeff=1):
        im_scale = min(self.winwow_height / image.shape[0], self.winwow_width / image.shape[1]) / coeff

        if im_scale > 5:
            im_scale = 5
        image = cv2.resize(image, (int(image.shape[1] * im_scale), int(image.shape[0] * im_scale)))
        return image

    def next(self, visual_method, info_method):
        self.counter += 1

        image = visual_method((self.index_image[self.counter % self.size]))
        if isinstance(image, list):
            self.image = []
            images = image
            for i, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = self.image_resize(image, coeff=2)

                image = Image.fromarray(image)
                self.image.append(ImageTk.PhotoImage(image=image))
                image_label = tk.Label(self.frame_for_image, bg='white', image=self.image[i],
                                       width=int(self.winwow_width / 2),
                                       height=int(self.winwow_height / 2))
                image_label.grid(row=0, column=i, padx=50, pady=10)

        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.image_resize(image)

            image = Image.fromarray(image)
            self.image = ImageTk.PhotoImage(image=image)
            image_label = tk.Label(self.frame_for_image, image=self.image, bg='white', width=self.winwow_width,
                                   height=self.winwow_height)
            image_label.grid(row=0, column=0, padx=50, pady=10)

        info_method()

    def prev(self, visual_method, info_method):
        self.counter -= 1

        image = visual_method((self.index_image[self.counter % self.size]))
        if isinstance(image, list):
            self.image = []
            images = image
            for i, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = self.image_resize(image, coeff=2)

                image = Image.fromarray(image)
                self.image.append(ImageTk.PhotoImage(image=image))
                image_label = tk.Label(self.frame_for_image, bg='white', image=self.image[i],
                                       width=int(self.winwow_width / 2),
                                       height=int(self.winwow_height / 2))

                image_label.grid(row=0, column=i, padx=50, pady=10)

        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.image_resize(image)

            image = Image.fromarray(image)
            self.image = ImageTk.PhotoImage(image=image)
            image_label = tk.Label(self.frame_for_image, image=self.image, bg='white', width=self.winwow_width,
                                   height=self.winwow_height)
            image_label.grid(row=0, column=0, padx=50, pady=10)

        info_method()

    def log_top_n(self, names):
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logging.info("Top {} picture: {}".format(len(names), " ".join(names)))

    @abstractmethod
    def _visualize_data(self, name):
        pass

    @abstractmethod
    def _top_n(self, n):
        pass

    @abstractmethod
    def _multiple_visualize_data(self, name):
        pass

    @staticmethod
    @abstractmethod
    def _multiple_top_n(set_task, n=10):
        pass
