import numpy as np
import cv2
import os
from metric_analysis import MetricAnalysis
from collections import OrderedDict
import warnings


class Segmentation(MetricAnalysis):

    def __init__(self, type_task, data, file_name, directory, mask):
        super().__init__(type_task, data, file_name, directory, mask)

        self.identifier = OrderedDict()
        self.index = []
        self.confusion_matrix = OrderedDict()
        self.predicted_mask = OrderedDict()
        self.segmentation_colors = None
        self.argmax_data = False

        self.validate()
        self.parser()

    def parser(self):
        super().parser()
        for i, report in enumerate(self.reports):
            self.identifier[report["identifier"]] = report["identifier"]
            self.index.append(report["identifier"])

            name = os.path.basename(report["predicted_mask"])
            self.predicted_mask[report["identifier"]] = name
            self.confusion_matrix[report["identifier"]] = report["confusion_matrix"]

        if 'segmentation_colors' in self.dataset_meta.keys():
            self.argmax_data = True
            self.segmentation_colors = np.array(self.data.get("dataset_meta").get('segmentation_colors'))
        elif len(self.label_map) > 20:
            self.segmentation_colors = np.random.randint(180, size=(len(self.label_map), 3))
        else:
            self.segmentation_colors = np.zeros((len(self.label_map), 1, 3), np.uint8)
            for i in range(len(self.label_map)):
                self.segmentation_colors[i] = np.array([i * int(180 / (len(self.label_map) + 1)), 255, 255])

            self.segmentation_colors = cv2.cvtColor(self.segmentation_colors, cv2.COLOR_HSV2BGR)
            self.segmentation_colors = np.resize(self.segmentation_colors, new_shape=(len(self.label_map), 3))

    def validate(self):
        super().validate()

        report_error = ""
        report_obj = self.data.get("report")[0]

        if not 'identifier' in report_obj:
            report_error.append('identifier')

        if not 'confusion_matrix' in report_obj:
            report_error.append('confusion_matrix')

        if report_error:
            report_error = ', '.join(report_error)
            raise Exception(report_error)

        if report_error:
            report_error = ', '.join(report_error)
            raise KeyError("there are no keys in the file <json>: {}".format(report_error))

    def _plot_confusion_matrix(self, ax):
        import seaborn as sns

        cm_all = np.array(list(self.confusion_matrix.values()))

        cm = np.sum(cm_all, axis=0)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_prob = cm / cm_sum.astype(float) * 100

        text = np.zeros(cm.shape, dtype=object)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[0]):
                text[i, j] = "{0:.2f}".format(cm_prob[i, j])

        sns.heatmap(cm_prob, cmap="YlGnBu", annot=text, fmt='', ax=ax)

        ax.set_title("Confusion matrix")

        ax.set_xlabel("prediction label")
        ax.set_ylabel("annotation label")
        return ax

    def plot_image(self, image, mask):
        if not self.argmax_data:
            mask = np.argmax(mask, axis=0).astype(np.uint8)
        mask = mask.astype(np.uint8)
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for temp in range((self.segmentation_colors.shape[0])):
            color_mask[:, :, 0] = np.where((mask == temp),
                                           self.segmentation_colors[temp][0], color_mask[:, :, 0])
            color_mask[:, :, 1] = np.where((mask == temp),
                                           self.segmentation_colors[temp][1], color_mask[:, :, 1])
            color_mask[:, :, 2] = np.where((mask == temp),
                                           self.segmentation_colors[temp][2], color_mask[:, :, 2])

        image = cv2.addWeighted(image, 1.0, color_mask, 0.7, 0)

        return image

    def _visualize_data(self, image, mask):

        image = self.plot_image(image, mask)
        return image

    def _top_n(self, n=10):
        if n > self.size_dataset:
            warnings.warn("""value n is greater than the size of the dataset,
                             it will be reduced to the size of the dataset""")
            n = self.size_dataset

        cm_info = OrderedDict()

        for name, cm in self.confusion_matrix.items():
            cm_info[name] = np.sum(cm) - np.sum(np.diag(cm))

        sort_key = sorted(cm_info, key=lambda k: (cm_info[k]), reverse=True)[:n]

        return sort_key

    @staticmethod
    def _multiple_visualize_data(set_task, name):
        prediction = []

        if not set_task[1].identifier.get(name):
            warnings.warn("in file {} no image {}".format(set_task[1].file, name))
        else:
            for i, task in enumerate(set_task):
                mask = np.load(task.mask + task.predicted_mask[name], allow_pickle=True)
                image = cv2.imread(task.picture_directory + name)
                prediction.append(image.copy())
                prediction[i] = task.plot_image(prediction[i], mask)

        return prediction

    @staticmethod
    def _multiple_top_n(set_task, n=10):

        dist = OrderedDict()

        for name in set_task[0].identifier:

            if not set_task[1].identifier.get(name, []):
                warnings.warn("in file {} no image {}".format(set_task[1].file, name))
            else:

                dist[name] = np.linalg.norm(
                    np.diag(set_task[0].confusion_matrix[name]) - np.diag(set_task[1].confusion_matrix[name]))

        if n > len(dist):
            warnings.warn("""the n value is greater than the number of identical objects in both files, 
                            it will be reduced to the size of the dataset""")
            n = len(dist)

        sort_key = sorted(dist, key=lambda k: (dist[k]), reverse=True)[:n]

        return sort_key
