import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from metric_analysis import MetricAnalysis


class Segmentation(MetricAnalysis):

    def __init__(self, data, directory, mask):
        super(Segmentation, self).__init__(data, directory, mask)

        self.identifier = []
        self.confusion_matrix = []
        self.predicted_mask = []

        self.validate()
        self.parser()

    def parser(self):
        super(Segmentation, self).parser()

        for report in self.reports:
            self.identifier.append(report.get("identifier"))

            name = os.path.basename(report.get("predicted_mask"))
            self.predicted_mask.append(name)
            self.confusion_matrix.append((report.get("confusion_matrix")))

    def validate(self):
        super(Segmentation, self).validate()

        try:
            if not 'identifier' in self.data.get("report")[0]:
                raise Exception('identifier')

            if not 'confusion_matrix' in self.data.get("report")[0]:
                raise Exception('confusion_matrix')

        except Exception as e:
            print("no key '{}' in file <json>".format(e))
            raise SystemExit(1)

    def top_n(self):
        cm_info = []
        n = 15
        cm_all = np.array(self.confusion_matrix)

        for k, cm in enumerate(cm_all):
            value = 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i != j:
                        value += cm[i, j]
            cm_info.append([self.identifier[k], self.predicted_mask[k], value])

        df_top_n = pd.DataFrame(cm_info).sort_values(by=[2], ascending=False)[:n]

        flag = bool(self.data.get("dataset_meta").get('segmentation_colors'))
        if not flag:
            for i in range(df_top_n.shape[0]):
                mask = np.load(self.mask + df_top_n.iloc[i][1], allow_pickle=True)
                image = cv2.imread(self.picture_directory + df_top_n.iloc[i][0])

                image = self.plot_image(image, mask)

                cv2.imshow(df_top_n.iloc[i][1], image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:
                    break
        else:
            for i in range(df_top_n.shape[0]):
                mask = np.load(self.mask + df_top_n.iloc[i][1], allow_pickle=True)
                image = cv2.imread(self.picture_directory + df_top_n.iloc[i][0])

                image = self.plot_image_arg(image, mask)

                cv2.imshow(df_top_n.iloc[i][1], image)
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
        flag = bool(self.data.get("dataset_meta").get('segmentation_colors'))

        if not flag:

            for i in range(len(self.reports)):
                mask = np.load(self.mask + self.predicted_mask[i], allow_pickle=True)
                image = cv2.imread(self.picture_directory + self.identifier[i])

                image = self.plot_image(image, mask)

                cv2.imshow("image", image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:
                    break

        else:
            for i in range(len(self.reports)):
                mask = np.load(self.mask + self.predicted_mask[i], allow_pickle=True)
                image = cv2.imread(self.picture_directory + self.identifier[i])

                image = self.plot_image_arg(image, mask)

                cv2.imshow("save/" + str(self.identifier[i]), image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:
                    break

    def plot_image_arg(self, image, mask):
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        segmentation_colors = np.array(self.data.get("dataset_meta").get('segmentation_colors'))

        class_mask = mask.astype(np.uint8)

        color_mask = cv2.cvtColor(class_mask, cv2.COLOR_GRAY2BGR)

        for temp in range((segmentation_colors.shape[0])):
            color_mask[:, :, 0] = np.where((class_mask == temp),
                                           segmentation_colors[temp][0], color_mask[:, :, 0])
            color_mask[:, :, 1] = np.where((class_mask == temp),
                                           segmentation_colors[temp][1], color_mask[:, :, 1])
            color_mask[:, :, 2] = np.where((class_mask == temp),
                                           segmentation_colors[temp][2], color_mask[:, :, 2])

        image = cv2.addWeighted(image, 1.0, color_mask, 0.7, 0)

        return image

    def plot_image(self, image, mask):
        image = cv2.resize(image, (mask.shape[2], mask.shape[1]))

        class_mask = np.argmax(mask, axis=0).astype(np.uint8)

        bg_class = np.where(class_mask == 0, 0, 1)

        class_mask = cv2.cvtColor(class_mask, cv2.COLOR_GRAY2BGR)
        class_mask = cv2.cvtColor(class_mask, cv2.COLOR_BGR2HSV)

        class_mask[:, :, 2] = class_mask[:, :, 2] * int(180 / (mask.shape[0] - 1))

        for l in range(1, mask.shape[0]):
            r_value = np.random.randint(180)
            class_mask[:, :, 0] = np.where(class_mask[:, :, 2] == l, r_value, class_mask[:, :, 0])

        class_mask[:, :, 0] = class_mask[:, :, 2]
        class_mask[:, :, 1] = 255
        class_mask[:, :, 2] = 255 * bg_class

        res_mask = cv2.cvtColor(class_mask, cv2.COLOR_HSV2BGR)

        image = cv2.addWeighted(image, 0.95, res_mask, 0.7, 0)

        return image


""""
TODO 

весьма сильно ограничеваем в плане что если у нас арг то должна быть и цветовая маска

"""
