from abc import ABC, abstractmethod
import json


class MetricAnalysis(ABC):

    def __init__(self, data, directory, mask):
        self.picture_directory = directory
        self.mask = mask

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
        self.label_map = {}

        for name in sorted(self.dataset_meta.get("label_map").keys()):
            self.label_map[name] = self.dataset_meta.get("label_map").get(name)

    def validate(self):
        try:
            if not 'processing_info' in self.data:
                raise Exception('processing_info')
            if not 'dataset_meta' in self.data:
                raise Exception('dataset_meta')
            if not 'report' in self.data:
                raise Exception('report')
            if not 'report_type' in self.data:
                raise Exception('report_type')

            if not 'label_map' in self.data.get("dataset_meta"):
                raise Exception('label_map')

        except Exception as e:
            print("no key '{}' in file <json>".format(e))
            raise SystemExit(1)

    @abstractmethod
    def metrics(self):
        pass

    @abstractmethod
    def visualize_data(self):
        pass

    @abstractmethod
    def top_n(self, n):
        pass
