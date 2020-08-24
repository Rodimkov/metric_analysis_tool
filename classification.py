from collections import OrderedDict
import warnings
import cv2
import numpy as np
from metric_analysis import MetricAnalysis


class Classification(MetricAnalysis):

    def __init__(self, type_task, data, file_name, directory, mask, true_mask):
        super().__init__(type_task, data, file_name, directory, mask, true_mask)
        self.identifier = OrderedDict()
        self.index = []
        self.prediction_label = OrderedDict()
        self.annotation_label = OrderedDict()
        self.prediction_scores = OrderedDict()
        self.accuracy_result = OrderedDict()

        self.set_task = None

        self.validate()
        self.parser()

    def validate(self):
        super().validate()

        report_error = []
        report_obj = self.data.get("report")[0]

        if not 'identifier' in report_obj:
            report_error.append('identifier')

        if not 'prediction_label' in report_obj:
            report_error.append('prediction_label')

        if not 'annotation_label' in report_obj:
            report_error.append('annotation_label')

        if not 'prediction_scores' in report_obj:
            report_error.append('prediction_scores')

        if not 'accuracy_result' in report_obj:
            report_error.append('accuracy_result')

        if report_error:
            report_error = ', '.join(report_error)
            raise KeyError("there are no keys in the file <json>: {}".format(report_error))

        report_error = []
        report_obj = self.data.get("report")[0]

        value = report_obj['identifier']

        if not isinstance(value, str):
            report_error.append('identifier')

        value = report_obj['prediction_label']
        if not isinstance(value, int):
            report_error.append('prediction_label')

        value = report_obj['annotation_label']
        if not isinstance(value, int):
            report_error.append('annotation_label')

        value = report_obj['prediction_scores']

        if not isinstance(value, list):
            report_error.append('prediction_scores')

        value = report_obj['accuracy_result']
        if not isinstance(value, float):
            report_error.append('accuracy_result')

        if report_error:
            report_error = ', '.join(report_error)
            raise KeyError("there are no keys in the file <json>: {}".format(report_error))

    def parser(self):
        super().parser()

        for report in self.reports:
            self.identifier[report["identifier"]] = report["identifier"]
            self.index.append(report["identifier"])
            self.prediction_label[report["identifier"]] = report["prediction_label"]
            self.annotation_label[report["identifier"]] = report["annotation_label"]
            self.prediction_scores[report["identifier"]] = report["prediction_scores"]
            self.accuracy_result[report["identifier"]] = report["accuracy_result"]

    def _visualize_data(self, name):

        image = cv2.imread(self.picture_directory + name)
        if image is None:
            raise KeyError("in directory {} no image {}".format(self.picture_directory, name))
        return image

    def _top_n(self, n=10):
        if n > self.size_dataset:
            warnings.warn("""value n is greater than the size of the dataset,
                             it will be reduced to the size of the dataset""")
            n = self.size_dataset

        position = OrderedDict()
        scores = OrderedDict()

        for name in self.identifier:
            sort_index = np.argsort(self.prediction_scores[name])[::-1]
            array_with_index = np.where(sort_index == self.annotation_label[name])
            value_with_index = array_with_index[0][0]
            position_prediction = -value_with_index
            position[name] = position_prediction
            scores[name] = self.prediction_scores[name][self.annotation_label[name]]

        sort_key = sorted(self.identifier, key=lambda k: (position[k], scores[k]))[:n]
        return sort_key

    def _plot_accuracy_changes(self, ax, k=100):
        accuracy_change = []

        accuracy = list(self.accuracy_result.values())

        for i in range(int(k / 2), int(len(accuracy) - int(k / 2))):
            value = np.mean(accuracy[i:(i + k)])
            accuracy_change.append(value)
        x_range = range(int(k / 2), len(accuracy_change) + int(k / 2))

        ax.set_title("Change accuracy in the process of predicting results")

        ax.set_xlabel("image number")
        ax.set_ylabel("accuracy")

        ax.plot(x_range, accuracy_change)
        return ax

    def _plot_confusion_matrix(self, ax):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(list(self.prediction_label.values()), list(self.annotation_label.values()),
                              labels=np.unique(list(self.annotation_label.values())))

        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_prob = cm / cm_sum.astype(float) * 100

        text = np.zeros(cm.shape, dtype=object)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[0]):
                text[i, j] = "{0:.2f}%\n{1} / {2}".format(cm_prob[i, j], cm[i, j], cm_sum[i][0])

        sns.heatmap(cm, cmap="YlGnBu", annot=text, fmt='', ax=ax, cbar=False)

        ax.set_title("Confusion matrix")

        ax.set_xlabel("prediction label")
        ax.set_ylabel("annotation label")
        return ax

    def _multiple_visualize_data(self, name):
        image = cv2.imread(self.set_task[0].picture_directory + name)

        if image is None:
            raise KeyError("in directory {} no image {}".format(self.picture_directory, name))
        return image

    @staticmethod
    def _multiple_top_n(set_task, n=10):
        position = OrderedDict()
        scores = OrderedDict()

        for name in set_task[0].identifier:
            if not set_task[1].identifier.get(name, []):
                warnings.warn("in file {} no image {}".format(set_task[1].file, name))
            else:
                sort_index = np.argsort(set_task[1].prediction_scores[name])[::-1]

                array_with_index = np.where(sort_index == set_task[0].prediction_label[name])
                value_with_index = array_with_index[0][0]
                position_prediction = -value_with_index
                position[name] = position_prediction
                scores[name] = set_task[1].prediction_scores[name][set_task[0].prediction_label[name]]

        if n > len(position):
            warnings.warn("""the n value is greater than the number of identical objects in both files,
                            it will be reduced to the size of the dataset""")
            n = len(position)

        sort_key = sorted(set_task[0].identifier, key=lambda k: (position[k], scores[k]))[:n]

        return sort_key
