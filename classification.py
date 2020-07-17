import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from metric_analysis import MetricAnalysis


class Classification(MetricAnalysis):

    def __init__(self, file_json, directory):
        self.file = file_json
        self.picture_directory = directory

        self.data = self.read_data()

        self.task_info = None
        self.dataset_meta = None
        self.reports = None

        self.label_map = None

        self.identifier = []
        self.prediction_label = []
        self.annotation_label = []
        self.prediction_scores = []
        self.accuracy_result = []

        self.validate()
        self.parser()

    def read_data(self):
        try:
            with open(self.file, "r") as read_file:
                return json.load(read_file)

        except FileNotFoundError:
            print('file {} does not exist'.format(self.file))
            raise SystemExit(1)

        except IsADirectoryError:
            print("file was expected but this is a directory")
            raise SystemExit(1)

    def validate(self):
        try:
            if not 'processing_info' in self.data:
                raise Exception('processing_info')
            if not 'dataset_meta' in self.data:
                raise Exception('dataset_meta')
            if not 'report' in self.data:
                raise Exception('report')

            if not 'label_map' in self.data.get("dataset_meta"):
                raise Exception('label_map')

            if not 'identifier' in self.data.get("report")[0]:
                raise Exception('identifier')

            if not 'prediction_label' in self.data.get("report")[0]:
                raise Exception('prediction_label')

            if not 'annotation_label' in self.data.get("report")[0]:
                raise Exception('annotation_label')

            if not 'prediction_scores' in self.data.get("report")[0]:
                raise Exception('prediction_scores')

        except Exception as e:
            print("no key '{}' in file <json>".format(e))
            raise SystemExit(1)

    def parser(self):
        self.task_info = self.data.get("processing_info")
        self.dataset_meta = self.data.get("dataset_meta")
        self.reports = self.data.get("report")

        self.label_map = self.dataset_meta.get("label_map")

        for report in self.reports:
            self.identifier.append(report.get("identifier"))
            self.prediction_label.append(report.get("prediction_label"))
            self.annotation_label.append(report.get("annotation_label"))
            self.prediction_scores.append(report.get("prediction_scores"))
            self.accuracy_result.append(report.get("accuracy_result"))

    def simple_metric(self):
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
        
        print(accuracy_score(self.prediction_label, self.annotation_label))
        print(recall_score(self.prediction_label, self.annotation_label, average='macro'))
        print(precision_score(self.prediction_label, self.annotation_label, average='macro'))
        print(f1_score(self.prediction_label, self.annotation_label, average='macro'))

    def plot_accuracy_changes(self):
        accuracy_change = []
        value = 0.0

        for i, accuracy in enumerate(self.accuracy_result):
            value += accuracy
            accuracy_change.append(value / (i + 1))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Change in accuracy in the process of predicting results")

        ax.set_xlabel("image number")
        ax.set_ylabel("accuracy")

        ax.plot(range(len(self.reports)), accuracy_change)
        plt.show()

    def plot_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(self.prediction_label, self.annotation_label,
                              labels=np.unique(self.annotation_label))

        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_prob = cm / cm_sum.astype(float) * 100

        text = np.zeros(cm.shape, dtype=object)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[0]):
                text[i, j] = "{0:.2f}%\n{1} / {2}".format(cm_prob[i, j], cm[i, j], cm_sum[i][0])

        fig, ax = plt.subplots(figsize=(8, 8))

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

        for i, report in enumerate(self.reports):

            pred_label = self.label_map.get(str(self.prediction_label[i]))
            true_label = self.label_map.get(str(self.prediction_label[i]))

            print("image name:", self.identifier[i],
                  "\nprediction label:", pred_label,
                  "annotation label:", true_label,
                  "prediction scores:", np.max(self.prediction_scores[i]))

            image = cv2.imread(self.picture_directory + self.identifier[i])
            cv2.imshow(self.identifier[i], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break
