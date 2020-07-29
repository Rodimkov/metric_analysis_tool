import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from metric_analysis import MetricAnalysis


class Segmentation(MetricAnalysis):

    def __init__(self, data, directory, mask):
        super(Segmentation, self).__init__(data, directory, mask)

        self.identifier = []
        self.confusion_matrix = []
        self.predicted_mask = []
        self.segmentation_colors = None

        self.validate()
        self.parser()

    def parser(self):
        super(Segmentation, self).parser()

        for report in self.reports:
            self.identifier.append(report.get("identifier"))

            name = os.path.basename(report.get("predicted_mask"))
            self.predicted_mask.append(name)
            self.confusion_matrix.append((report.get("confusion_matrix")))

        if 'segmentation_colors' in self.dataset_meta.keys():
            self.segmentation_colors = np.array(self.data.get("dataset_meta").get('segmentation_colors'))
        elif len(self.label_map) > 10:
            self.segmentation_colors = np.random.randint(180, size=(len(self.label_map), 3))
        else:
            self.segmentation_colors = np.ones((len(self.label_map), 3))
            self.segmentation_colors = np.zeros((len(self.label_map), 1, 3), np.uint8)
            for i in range(len(self.label_map)):
                self.segmentation_colors[i] = np.array([i * int(180 / (len(self.label_map))), 255, 255])

            self.segmentation_colors = cv2.cvtColor(self.segmentation_colors, cv2.COLOR_HSV2BGR)
            self.segmentation_colors = np.resize(self.segmentation_colors, new_shape=(len(self.label_map), 3))

    def validate(self):
        super(Segmentation, self).validate()

        try:
            report_error = ""
            report_obj = self.data.get("report")[0]

            if not 'identifier' in report_obj:
                report_error += ' identifier'

            if not 'confusion_matrix' in report_obj:
                report_error += ' confusion_matrix'

            if report_error:
                raise Exception(report_error)

        except Exception as e:
            print("there are no keys in the file <json>: {}".format(e))
            raise SystemExit(1)

    def top_n(self, n=10):
        cm_info = []
        cm_all = np.array(self.confusion_matrix)

        for k, cm in enumerate(cm_all):
            value = 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i != j:
                        value += cm[i, j]
            cm_info.append(value)

        cm_info = np.array(cm_info)

        sort_index = np.argsort(cm_info)[-n:][::-1]

        for i in sort_index:
            print(i)
            mask = np.load(self.mask + self.predicted_mask[i], allow_pickle=True)
            image = cv2.imread(self.picture_directory + self.identifier[i])

            image = self.plot_image(image, mask)

            cv2.imshow(self.identifier[i], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    def plot_confusion_matrix(self):
        import seaborn as sns

        cm_all = np.array(self.confusion_matrix)

        cm = np.sum(cm_all, axis=0)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_prob = cm / cm_sum.astype(float) * 100

        text = np.zeros(cm.shape, dtype=object)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[0]):
                text[i, j] = "{0:.2f}".format(cm_prob[i, j])

        _, ax = plt.subplots(figsize=(8, 8))

        sns.heatmap(cm_prob, cmap="YlGnBu", annot=text, fmt='', ax=ax)

        ax.set_title("Confusion matrix")

        ax.set_xlabel("prediction label")
        ax.set_ylabel("annotation label")
        plt.show()

    def metrics(self):
        self.plot_confusion_matrix()

    def visualize_data(self):

        for i in range(len(self.reports)):
            mask = np.load(self.mask + self.predicted_mask[i], allow_pickle=True)
            image = cv2.imread(self.picture_directory + self.identifier[i])

            image = self.plot_image(image, mask)

            cv2.imshow("image", image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    def plot_image(self, image, mask):
        if len(mask.shape) == 3:
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
