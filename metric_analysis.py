from abc import ABC, abstractmethod
from collections import OrderedDict


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
        self.task_info = self.data.get("processing_info")
        self.dataset_meta = self.data.get("dataset_meta")
        self.reports = self.data.get("report")
        self.report_type = self.dataset_meta.get("report_type")
        self.label_map = OrderedDict()

        for name in sorted(self.dataset_meta.get("label_map").keys()):
            self.label_map[name] = self.dataset_meta.get("label_map").get(name)

        # for i in range(1000):
        #    self.label_map[str(i)] = str(i)

        self.size_dataset = len(self.reports)

    def validate(self):
        report_error = ""
        report_obj = self.data.get("report")[0]

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

    @abstractmethod
    def metrics(self):
        pass

    @abstractmethod
    def visualize_data(self):
        pass

    @abstractmethod
    def top_n(self, n):
        pass
