import cv2
from metric_analysis import MetricAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Detection(MetricAnalysis):

    def __init__(self, data, directory, mask):
        super(Detection, self).__init__(data, directory, mask)

        self.identifier = []
        self.per_class_result = []
        self.prediction_boxes = []
        self.annotation_boxes = []
        self.prediction_scores = []
        self.average_precision = []
        self.average_precision_picture = []

        self.validate()
        self.parser()

    def validate(self):
        super(Detection, self).validate()

        try:
            if not 'identifier' in self.data.get("report")[0]:
                raise Exception('identifier')

            if not 'per_class_result' in self.data.get("report")[0]:
                raise Exception('prediction_label')

            key_class = list(self.data.get("report")[0].get("per_class_result").keys())[0]

            if not 'prediction_boxes' in self.data.get("report")[0] \
                    .get("per_class_result").get(key_class):
                raise Exception('prediction_boxes')

            if not 'annotation_boxes' in self.data.get("report")[0] \
                    .get("per_class_result").get(key_class):
                raise Exception('annotation_boxes')

            if not 'prediction_scores' in self.data.get("report")[0] \
                    .get("per_class_result").get(key_class):
                raise Exception('prediction_scores')

            if not 'average_precision' in self.data.get("report")[0] \
                    .get("per_class_result").get(key_class):
                raise Exception('average_precision')

        except Exception as e:
            print("no key '{}' in file <json>".format(e))
            raise SystemExit(1)

    def parser(self):
        super(Detection, self).parser()

        for report in self.reports:
            self.identifier.append(report.get("identifier"))
            self.per_class_result.append(report.get("per_class_result"))

        for info_class in self.per_class_result:
            pb = {}
            ab = {}
            ps = {}
            ap = {}
            value = []

            for key in info_class.keys():
                pb[key] = info_class.get(key).get("prediction_boxes")
                ab[key] = info_class.get(key).get("annotation_boxes")
                ps[key] = info_class.get(key).get("prediction_scores")
                ap[key] = info_class.get(key).get("average_precision")
                value.append(info_class.get(key).get("average_precision"))

            self.prediction_boxes.append(pb)
            self.annotation_boxes.append(ab)
            self.prediction_scores.append(ps)
            self.average_precision.append(ap)
            if not all(np.isnan(value)):
                self.average_precision_picture.append(np.nanmean(value))
            else:
                self.average_precision_picture.append(np.nan)

    def top_n(self):
        """Highlighting the top worst objects in terms of predictions
        This tool show four different types of "bad" objects:

        1) Without correct answer: those for which there are annotation, but there is no right answer. Ratio of
        the number of elements in the annotation to the predicted ones is considered as a metric. Predicting false
        box two for two annotations is better than predicting two hundred false box for two annotations

        2) Without annotation: those for which there is no annotation. Not sorted.

        3) Precision: ratio of the number of correctly predicted to all predictions. the smallest values are
        selected. Not stable if there are two correct answers and a lot of erroneous assumptions (such objects will
        be the worst, despite the low probability of most of the predicted elements). Objects without annotation or
        objects with no correct predictions are not taken into account.

        4) average precision: is averaged for the picture for each class and the minimum. Objects without annotation
        or objects with no correct predictions are not taken into account.

        """
        without_correct_answers = []
        without_annotation = []
        precision = []
        average_precision_score = []
        n = 50

        for i, report in enumerate(self.per_class_result):
            pm = []
            am = []
            flag_annotation = False
            flag_prediction = False

            for key in report.keys():
                pm = pm + report.get(key).get('prediction_matches')
                am = am + report.get(key).get('annotation_matches')

                if report.get(key).get('annotation_matches'):
                    flag_annotation = True

                if np.sum(report.get(key).get('prediction_matches')) != 0:
                    flag_prediction = True

            if (not flag_prediction) and flag_annotation:
                mark = len(am) / len(pm)
                without_correct_answers.append([i, self.identifier[i], mark])
            elif not flag_annotation:
                without_annotation.append([i, self.identifier[i]])
            else:
                tp = np.sum(pm)
                fp = np.sum(np.logical_not(pm))

                precision.append([i, self.identifier[i], tp / (tp + fp)])
                average_precision_score.append([i, self.identifier[i], self.average_precision_picture[i]])

        top_precision = pd.DataFrame(precision).sort_values(by=[2])[:n]
        top_average_precision_score = pd.DataFrame(average_precision_score).sort_values(by=[2])[:n]
        top_without_annotation = pd.DataFrame(without_annotation)

        if without_correct_answers:
            top_without_correct_answers = pd.DataFrame(without_correct_answers).sort_values(by=[2])[:n]
            for image_name, index in zip(top_without_correct_answers[1], top_without_correct_answers[0]):
                image = cv2.imread(self.picture_directory + image_name)
                image = self.marking(image, index, self.per_class_result[index], 0.3)
                cv2.imshow(image_name, image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:
                    break
        else:
            print('no objects without correct answers')

        for image_name, index in zip(top_precision[1], top_precision[0]):
            image = cv2.imread(self.picture_directory + image_name)
            image = self.marking(image, index, self.per_class_result[index], 0.3)
            cv2.imshow(image_name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

        for image_name, index in zip(top_average_precision_score[1], top_average_precision_score[0]):
            image = cv2.imread(self.picture_directory + image_name)
            image = self.marking(image, index, self.per_class_result[index], 0.3)
            cv2.imshow(image_name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

        for image_name, index in zip(top_without_annotation[1], top_without_annotation[0]):
            image = cv2.imread(self.picture_directory + image_name)
            image = self.marking(image, index, self.per_class_result[index], 0.3)
            cv2.imshow(image_name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    def plot_average_precision_changes(self):
        k = 100
        accuracy_change = []

        for i in range(int(k / 2), int(len(self.average_precision) - int(k / 2))):
            value = np.nanmean(self.average_precision_picture[i:(i + k)])
            accuracy_change.append(value)
        x_range = range(int(k / 2), len(accuracy_change) + int(k / 2))

        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Change average precision in the process of predicting results")

        ax.set_xlabel("image number")
        ax.set_ylabel("accuracy")

        ax.plot(x_range, accuracy_change)
        plt.show()

    def metrics(self):
        self.plot_average_precision_changes()

    @staticmethod
    def draw_boxes(image, boxes, label_class, color=(255, 255, 255), thickness=2):

        for i, box in enumerate(boxes):
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, label_class[i], (int(box[0] + 6), int(box[3]) - 6),
                        font, 0.7, color, 1)

            image = cv2.rectangle(image, start_point, end_point, color, thickness,
                                  cv2.FILLED)

        return image

    def marking(self, image, i, info_class, threshold_scores):
        for key in info_class:

            label_class = []
            predict_boxes = []

            for box, score in zip(self.prediction_boxes[i].get(key, []),
                                  self.prediction_scores[i].get(key, [])):
                if score > threshold_scores:
                    predict_boxes.append(box)
                    name_class = self.label_map.get(key)
                    label_class.append('{} {:.3f}'.format(name_class, score))

            image = self.draw_boxes(image, predict_boxes, label_class, color=(255, 0, 0))

            label_class = [self.label_map.get(key)] * len(self.annotation_boxes[i].get(key))
            image = self.draw_boxes(image, self.annotation_boxes[i].get(key, []), label_class,
                                    color=(0, 0, 255))

        return image

    def visualize_data(self):
        threshold_scores = 0.5
        for i, info_class in enumerate(self.per_class_result):
            image = cv2.imread(self.picture_directory + self.identifier[i])

            self.marking(image, i, info_class, threshold_scores)

            cv2.imshow(self.identifier[i], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break
