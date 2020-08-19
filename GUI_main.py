import tkinter as tk
import gui_classification
import gui_detection
import gui_segmentation
import json


def read_data(json_file):
    with open(json_file, "r") as read_file:
        data = json.load(read_file)

    if not 'report_type' in data:
        raise KeyError("no key report type in file <json>")

    type_task = data.get("report_type")

    return data, type_task


class MainMenu(object):
    def __init__(self, file_name, directory, mask):
        self.master = tk.Tk()
        self.master.title("Tool for visualizing")
        self.master.geometry("300x220+300+300")
        self.master.resizable(width=False, height=False)

        self.frame = tk.Frame(self.master)
        self.frame.pack(anchor="n")

        if len(file_name) == 1:
            data, name = read_data(file_name[0])
            self.normal_mode(name, data, file_name[0], directory, mask[0])
        if len(file_name) == 2:
            task = []
            for i in range(len(file_name)):
                data, name = read_data(file_name[i])
                task.append(self.create_task(name, data, file_name[i], directory, mask[i]))
            self.compare_mode(task)

    def normal_mode(self, name, data, file_name, directory, mask):
        label_task = "type task: {}".format(name)
        tk.Label(self.frame, text=label_task).grid(row=0, column=0, padx=10, pady=10)

        tk.Button(self.frame, text="visualize data", width=12,
                  command=self.visualize_data).grid(row=1, column=0, padx=10, pady=10)

        tk.Button(self.frame, text="top n", width=12,
                  command=self.top_n).grid(row=2, column=0, padx=10, pady=10)

        tk.Button(self.frame, text="metric", width=12,
                  command=self.metric).grid(row=3, column=0, padx=10, pady=10)

        self.task = self.create_task(name, data, file_name, directory, mask)

        tk.mainloop()

    def compare_mode(self, task):
        label_task = "type task: {}".format(task[0].type_task)
        tk.Label(self.frame, text=label_task).grid(row=0, column=0, padx=10, pady=10)

        tk.Button(self.frame, text="visualize data",
                  command=self.multiple_visualize_data).grid(row=1, column=0, padx=10, pady=10)

        tk.Button(self.frame, text="top n",
                  command=self.multiple_top_n).grid(row=2, column=0, padx=10, pady=10)

        self.set_task = task
        tk.mainloop()

    @staticmethod
    def create_task(name, data, file_name, directory, mask):
        type_task = {
            "classification": gui_classification.GUI,
            "detection": gui_detection.GUI,
            "segmentation": gui_segmentation.GUI
        }
        return type_task[name](name, data, file_name, directory, mask)

    def visualize_data(self):
        obj = self.task.visualize_data()

    def top_n(self):
        obj = self.task.top_n()

    def metric(self):
        obj = self.task.metrics()

    def multiple_visualize_data(self):
        obj = self.set_task[0].multiple_visualize_data(self.set_task)

    def multiple_top_n(self):
        obj = self.set_task[0].multiple_top_n(self.set_task)
