import cv2
from metric_analysis import MetricAnalysis
import matplotlib.pyplot as plt
import numpy as np


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
            report_error = []
            report_obj = self.data.get("report")[0]

            if not 'identifier' in report_obj:
                report_error.append('identifier')

            if not 'per_class_result' in report_obj:
                report_error.append('prediction_label')

            if report_error:
                report_error = ', '.join(report_error)
                raise Exception(report_error)

            key_class = list(self.data.get("report")[0].get("per_class_result").keys())[0]

            class_result_error = []
            class_result_obj = self.data.get("report")[0].get("per_class_result").get(key_class)

            if not 'prediction_boxes' in class_result_obj:
                class_result_error.append('prediction_boxes')

            if not 'annotation_boxes' in class_result_obj:
                class_result_error.append('annotation_boxes')

            if not 'prediction_scores' in class_result_obj:
                class_result_error.append('prediction_scores')

            if not 'average_precision' in class_result_obj:
                class_result_error.append('average_precision')

            if report_error:
                class_result_error = ', '.join(class_result_error)
                raise Exception(class_result_error)

        except Exception as e:
            print("there are no keys in the file <json>: {}".format(e))
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
            mean_ap = []

            for key, value in info_class.items():
                pb[key] = value.get("prediction_boxes")
                ab[key] = value.get("annotation_boxes")
                ps[key] = value.get("prediction_scores")
                ap[key] = value.get("average_precision")
                mean_ap.append(value.get("average_precision"))

            self.prediction_boxes.append(pb)
            self.annotation_boxes.append(ab)
            self.prediction_scores.append(ps)
            self.average_precision.append(ap)

            if not all(np.isnan(mean_ap)):
                self.average_precision_picture.append(np.nanmean(mean_ap))
            else:
                self.average_precision_picture.append(np.nan)

    def top_n(self, n=10):
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
                without_annotation.append(i)
            else:
                tp = np.sum(pm)
                fp = np.sum(np.logical_not(pm))

                precision.append([i, tp / (tp + fp)])
                average_precision_score.append([i, self.average_precision_picture[i]])

        precision = np.array(precision)
        average_precision_score = np.array(average_precision_score)

        precision = precision[np.argsort(precision[:, 1])][:n][:, 0].astype(int)
        average_precision_score = average_precision_score[np.argsort(average_precision_score[:, 1])][:n][:, 0].astype(
            int)

        if without_correct_answers:
            without_correct_answers = np.array(without_correct_answers)
            without_correct_answers = without_correct_answers[np.argsort(without_correct_answers[:, 2])][:n]
            without_correct_answers = without_correct_answers[:, 0].astype(int)
            print("pictures without the correct answer, but annotated")
            for i in without_correct_answers:
                image = cv2.imread(self.picture_directory + self.identifier[i])
                image = self.marking(image, i, self.per_class_result[i], 0.3)
                print("picture name: ", self.identifier[i])
                cv2.imshow(self.identifier[i], image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:
                    break
        else:
            print('no objects without correct answers')

        print("worst based on the metric: 'precision'")
        for i in precision:
            image = cv2.imread(self.picture_directory + self.identifier[i])
            image = self.marking(image, i, self.per_class_result[i], 0.3)
            print("picture name: ", self.identifier[i])
            cv2.imshow(self.identifier[i], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

        print("worst based on the metric: 'average precision'")
        for i in average_precision_score:
            image = cv2.imread(self.picture_directory + self.identifier[i])
            image = self.marking(image, i, self.per_class_result[i], 0.3)
            print("picture name: ", self.identifier[i])
            cv2.imshow(self.identifier[i], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

        print("pictures without annotation")
        for i in without_annotation:
            image = cv2.imread(self.picture_directory + self.identifier[i])
            image = self.marking(image, i, self.per_class_result[i], 0.3)
            print("picture name: ", self.identifier[i])
            cv2.imshow(self.identifier[i], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    def plot_average_precision_changes(self, k=100):
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
