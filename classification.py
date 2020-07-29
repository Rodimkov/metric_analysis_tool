import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from metric_analysis import MetricAnalysis


class Classification(MetricAnalysis):

    def __init__(self, data, directory, mask):
        super(Classification, self).__init__(data, directory, mask)

        self.identifier = []
        self.prediction_label = []
        self.annotation_label = []
        self.prediction_scores = []
        self.accuracy_result = []

        self.validate()
        self.parser()

    def validate(self):
        super(Classification, self).validate()

        try:
            report_error = ""
            report_obj = self.data.get("report")[0]

            if not 'identifier' in report_obj:
                report_error += ' identifier'

            if not 'prediction_label' in report_obj:
                report_error += ' prediction_label'

            if not 'annotation_label' in report_obj:
                report_error += ' annotation_label'

            if not 'prediction_scores' in report_obj:
                report_error += ' prediction_scores'

            if not 'accuracy_result' in report_obj:
                report_error += ' accuracy_result'

            if report_error:
                report_error = report_error[1:].replace(' ', ", ")
                raise Exception(report_error)

        except Exception as e:
            print("there are no keys in the file <json>: {}".format(e))
            raise SystemExit(1)

    def parser(self):
        super(Classification, self).parser()

        for report in self.reports:
            self.identifier.append(report.get("identifier"))
            self.prediction_label.append(report.get("prediction_label"))
            self.annotation_label.append(report.get("annotation_label"))
            self.prediction_scores.append(report.get("prediction_scores"))
            self.accuracy_result.append(report.get("accuracy_result"))

    def top_n(self, n=10):
        position = []
        scores = []

        for i in range(len(self.reports)):
            position.append(-np.where(np.argsort(self.prediction_scores[i])[::-1] == self.annotation_label[i])[0][0])
            scores.append(self.prediction_scores[i][self.annotation_label[i]])

        sort_index = np.lexsort((scores, position))[:n]

        for i in sort_index:
            pred_label = self.label_map.get(str(self.prediction_label[i]))
            true_label = self.label_map.get(str(self.annotation_label[i]))
            print("image name:", self.identifier[i],
                  "\nprediction label:", pred_label,
                  "annotation label:", true_label)
            image = cv2.imread(self.picture_directory + self.identifier[i])
            cv2.imshow(self.identifier[i], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    def simple_metric(self):
        from sklearn.metrics import classification_report

        print(classification_report(self.annotation_label, self.prediction_label,
                                    target_names=self.label_map.values()))

    def plot_accuracy_changes(self):
        k = 100
        accuracy_change = []

        for i in range(int(k / 2), int(len(self.accuracy_result) - int(k / 2))):
            value = np.mean(self.accuracy_result[i:(i + k)])
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

        cm = confusion_matrix(self.prediction_label, self.annotation_label,
                              labels=np.unique(self.annotation_label))

        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_prob = cm / cm_sum.astype(float) * 100

        text = np.zeros(cm.shape, dtype=object)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[0]):
                text[i, j] = "{0:.2f}%\n{1} / {2}".format(cm_prob[i, j], cm[i, j], cm_sum[i][0])

        _, ax = plt.subplots(figsize=(8, 8))

        df_cm = pd.DataFrame(cm, index=self.label_map.values(),
                             columns=self.label_map.values())

        sns.heatmap(df_cm, cmap="YlGnBu", annot=text, fmt='', ax=ax)

        ax.set_title("Confusion matrix")

        ax.set_xlabel("prediction label")
        ax.set_ylabel("annotation label")
        plt.show()

    def metrics(self):
        self.simple_metric()
        self.plot_accuracy_changes()
        self.plot_confusion_matrix()

    def visualize_data(self):

        for i in range(len(self.reports)):

            pred_label = self.label_map.get(str(self.prediction_label[i]))
            true_label = self.label_map.get(str(self.annotation_label[i]))

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
