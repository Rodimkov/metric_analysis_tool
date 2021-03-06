import cv2
import numpy as np
import matplotlib.pyplot as plt
from metric_analysis import MetricAnalysis
from collections import OrderedDict
import warnings


class Classification(MetricAnalysis):

    def __init__(self, type_task, data, file_name, directory, mask):
        super().__init__(type_task, data, file_name, directory, mask)

        self.identifier = OrderedDict()
        self.index = []
        self.prediction_label = OrderedDict()
        self.annotation_label = OrderedDict()
        self.prediction_scores = OrderedDict()
        self.accuracy_result = OrderedDict()

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

        for i, report in enumerate(self.reports):
            self.identifier[report["identifier"]] = report["identifier"]
            self.index.append(report["identifier"])
            self.prediction_label[report["identifier"]] = report["prediction_label"]
            self.annotation_label[report["identifier"]] = report["annotation_label"]
            self.prediction_scores[report["identifier"]] = report["prediction_scores"]
            #self.accuracy_result[report["identifier"]] = report["accuracy@top1_result"]
            self.accuracy_result[report["identifier"]] = report["accuracy_result"]

    def top_n(self, n=10):
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

        for name in sort_key:
            pred_label = self.label_map.get(str(self.prediction_label[name]))
            true_label = self.label_map.get(str(self.annotation_label[name]))
            print("image name:", self.identifier[name],
                  "\nprediction label:", pred_label,
                  "annotation label:", true_label)
            image = cv2.imread(self.picture_directory + self.identifier[name])
            cv2.imshow(self.identifier[name], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    def simple_metric(self):
        from sklearn.metrics import classification_report

        print(classification_report(list(self.annotation_label.values()), list(self.prediction_label.values()),
                                    target_names=self.label_map.values()))

    def plot_accuracy_changes(self, k=100):
        accuracy_change = []

        accuracy = list(self.accuracy_result.values())

        for i in range(int(k / 2), int(len(accuracy) - int(k / 2))):
            value = np.mean(accuracy[i:(i + k)])
            accuracy_change.append(value)
        x_range = range(int(k / 2), len(accuracy_change) + int(k / 2))

        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Change accuracy in the process of predicting results")

        ax.set_xlabel("image number")
        ax.set_ylabel("accuracy")

        ax.plot(x_range, accuracy_change)
        plt.show()

    def plot_confusion_matrix(self):
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

        _, ax = plt.subplots(figsize=(8, 8))

        sns.heatmap(cm, cmap="YlGnBu", annot=text, fmt='', ax=ax)

        ax.set_title("Confusion matrix")

        ax.set_xlabel("prediction label")
        ax.set_ylabel("annotation label")
        plt.show()

    def metrics(self):
        self.simple_metric()
        self.plot_accuracy_changes()
        self.plot_confusion_matrix()

    def visualize_data(self):
        for name in self.identifier:

            pred_label = self.label_map.get(str(self.prediction_label[name]))
            true_label = self.label_map.get(str(self.annotation_label[name]))

            print("image name:", self.identifier[name],
                  "\nprediction label:", pred_label,
                  "annotation label:", true_label,
                  "prediction scores:", np.max(self.prediction_scores[name]))

            image = cv2.imread(self.picture_directory + name)
            cv2.imshow(name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    @staticmethod
    def multiple_visualize_data(set_task):
        for name in list(set_task[0].identifier.keys()):

            if not set_task[1].identifier.get(name, []):
                warnings.warn("in file {} no image {}".format(set_task[1].file, name))
            else:
                print("image name:", name)

                for task in set_task:
                    label = task.label_map.get(str(task.prediction_label[name]))
                    print("prediction label:", label,
                          "prediction scores:", np.max(task.prediction_scores[name]))

                image = cv2.imread(set_task[0].picture_directory + name)
                cv2.imshow(name, image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:
                    break

    @staticmethod
    def multiple_top_n(set_task, n=10):
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

        for name in sort_key:
            print("image name:", name)

            label_0 = set_task[0].label_map.get(str(set_task[0].prediction_label[name]))
            label_1 = set_task[1].label_map.get(str(set_task[1].prediction_label[name]))
            print("prediction label 0:", label_0,
                  "prediction label 1:", label_1,
                  "prediction scores:", np.max(set_task[0].prediction_scores[name]))

            image = cv2.imread(set_task[0].picture_directory + name)
            cv2.imshow(name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break
