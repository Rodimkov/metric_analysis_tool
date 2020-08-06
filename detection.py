import cv2
from metric_analysis import MetricAnalysis
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

class Detection(MetricAnalysis):

    def __init__(self, data, directory, mask):
        super(Detection, self).__init__(data, directory, mask)

        self.identifier = OrderedDict()
        self.index = []
        self.per_class_result = OrderedDict()
        self.prediction_boxes = OrderedDict()
        self.annotation_boxes = OrderedDict()
        self.prediction_scores = OrderedDict()
        self.average_precision = OrderedDict()
        self.average_precision_picture = OrderedDict()

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

        for i, report in enumerate(self.reports):
            self.identifier[report.get("identifier")] = report.get("identifier")
            self.index.append(report.get("identifier"))
            self.per_class_result[report.get("identifier")] = report.get("per_class_result")

        for name, info_class in self.per_class_result.items():
            pb = OrderedDict()
            ab = OrderedDict()
            ps = OrderedDict()
            ap = OrderedDict()
            mean_ap = []

            for key, value in info_class.items():
                pb[key] = value.get("prediction_boxes")
                ab[key] = value.get("annotation_boxes")
                ps[key] = value.get("prediction_scores")
                ap[key] = value.get("average_precision")
                mean_ap.append(value.get("average_precision"))

            self.prediction_boxes[name] = pb
            self.annotation_boxes[name] = ab
            self.prediction_scores[name] = ps
            self.average_precision[name] = ap

            if not all(np.isnan(mean_ap)):
                self.average_precision_picture[name] = np.nanmean(mean_ap)
            else:
                self.average_precision_picture[name] = np.nan

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

        if n > self.size_dataset:
            print("value n is greater than the size of the dataset, it will be reduced to the size of the dataset")
            n = self.size_dataset

        without_correct_answers = OrderedDict()
        without_annotation = OrderedDict()
        precision = OrderedDict()
        average_precision_score = OrderedDict()

        for name, report in self.per_class_result.items():
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
                without_correct_answers[name] = mark
            elif not flag_annotation:
                without_annotation[name] = name
            else:
                tp = np.sum(pm)
                fp = np.sum(np.logical_not(pm))

                precision[name] = tp / (tp + fp)
                average_precision_score[name] = self.average_precision_picture[name]

        sort_precision = sorted(precision, key=lambda k: (precision[k]))[:n]
        sort_average_precision = sorted(average_precision_score, key=lambda k: (average_precision_score[k]))[:n]

        if without_correct_answers:
            print("pictures without the correct answer, but annotated")
            for name in without_correct_answers:
                image = cv2.imread(self.picture_directory + self.identifier[name])

                self.marking_predition(image, name, (255, 0, 0), 0.3)
                self.marking_annotation(image, name, (0, 0, 255))

                print("picture name: ", self.identifier[name])
                cv2.imshow(self.identifier[name], image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:
                    break
        else:
            print('no objects without correct answers')

        print("worst based on the metric: 'precision'")
        for name in sort_precision:
            image = cv2.imread(self.picture_directory + self.identifier[name])

            self.marking_predition(image, name, (255, 0, 0), 0.3)
            self.marking_annotation(image, name, (0, 0, 255))

            print("picture name: ", self.identifier[name])
            cv2.imshow(self.identifier[name], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

        print("worst based on the metric: 'average precision'")
        for name in sort_average_precision:
            image = cv2.imread(self.picture_directory + self.identifier[name])

            self.marking_predition(image, name, (255, 0, 0), 0.3)
            self.marking_annotation(image, name, (0, 0, 255))

            print("picture name: ", self.identifier[name])
            cv2.imshow(self.identifier[name], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

        print("pictures without annotation")
        for name in without_annotation:
            image = cv2.imread(self.picture_directory + self.identifier[name])

            self.marking_predition(image, name, (255, 0, 0), 0.3)
            self.marking_annotation(image, name, (0, 0, 255))

            print("picture name: ", self.identifier[name])
            cv2.imshow(self.identifier[name], image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    def plot_average_precision_changes(self, k=100):
        precision_change = []

        precision = list(self.average_precision_picture.values())

        for i in range(int(k / 2), int(len(precision) - int(k / 2))):
            value = np.nanmean(precision[i:(i + k)])
            precision_change.append(value)
        x_range = range(int(k / 2), len(precision_change) + int(k / 2))

        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Change average precision in the process of predicting results")

        ax.set_xlabel("image number")
        ax.set_ylabel("average precision")

        ax.plot(x_range, precision_change)
        plt.show()

    def metrics(self):
        self.plot_average_precision_changes()

    @staticmethod
    def draw_boxes(image, boxes, label_class, color=(255, 255, 255), thickness=2):
        color = [int(x) for x in color]
        for i, box in enumerate(boxes):
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, label_class[i], (int(box[0] + 6), int(box[3]) - 6),
                        font, 0.7, color, 1)

            image = cv2.rectangle(image, start_point, end_point, color, thickness,
                                  cv2.FILLED)

        return image

    def marking_predition(self, image, name, color, threshold_scores):
        info = self.per_class_result[name]
        for key in info:

            label_class = []
            predict_boxes = []

            for box, score in zip(self.prediction_boxes[name].get(key),
                                  self.prediction_scores[name].get(key)):
                if score > threshold_scores:
                    predict_boxes.append(box)
                    name_class = self.label_map.get(key)
                    label_class.append('{} {:.3f}'.format(name_class, score))

            image = self.draw_boxes(image, predict_boxes, label_class, color=color)

        return image

    def marking_annotation(self, image, name, color):
        info = self.per_class_result[name]
        for key in info:
            label_class = [self.label_map.get(key)] * len(self.annotation_boxes[name].get(key))
            image = self.draw_boxes(image, self.annotation_boxes[name].get(key, []), label_class,
                                    color=color)

        return image

    def visualize_data(self):
        threshold_scores = 0.5
        for name in self.identifier:
            image = cv2.imread(self.picture_directory + name)

            self.marking_predition(image, name, (255,0,0), threshold_scores)
            self.marking_annotation(image, name, (0,0,255))

            cv2.imshow(name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    @staticmethod
    def multiple_visualize_data(objs):
        threshold_scores = 0.5
        color = np.zeros((len(objs), 1, 3), np.uint8)

        for i in range(len(objs)):
            color[i] = np.array([i * int(180 / (len(objs))), 255, 255])

        color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        color = np.resize(color, new_shape=(len(objs), 3))

        for name in objs[0].identifier:
            image = cv2.imread(objs[0].picture_directory + name)
            for i, obj in enumerate(objs):
                obj.marking_predition(image, name, tuple(color[i]), threshold_scores)

            #objs[0].marking_annotation(image, name, (0,0,255))

            cv2.imshow(name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

    @staticmethod
    def intersections(prediction_box, annotation_boxes):
        px_min, py_min, px_max, py_max = prediction_box
        ax_mins, ay_mins, ax_maxs, ay_maxs = annotation_boxes

        x_mins = np.maximum(ax_mins, px_min)
        y_mins = np.maximum(ay_mins, py_min)
        x_maxs = np.minimum(ax_maxs, px_max)
        y_maxs = np.minimum(ay_maxs, py_max)

        return x_mins, y_mins, np.maximum(x_mins, x_maxs), np.maximum(y_mins, y_maxs)

    @staticmethod
    def area(box):
        x0, y0, x1, y1 = box
        return (x1 - x0) * (y1 - y0)

    def calculate_similarity_matrix(self, set_a, set_b):
        similarity = np.zeros([len(set_a), len(set_b)], dtype=np.float32)
        for i, box_a in enumerate(set_a):
            for j, box_b in enumerate(set_b):
                similarity[i, j] = self.evaluate(box_a, box_b)

        return similarity

    def evaluate(self, prediction_box, annotation_boxes):
        intersections_area = self.area(self.intersections(prediction_box, annotation_boxes))
        unions = self.area(prediction_box) + self.area(annotation_boxes) - intersections_area
        return np.divide(
            intersections_area, unions, out=np.zeros_like(intersections_area, dtype=float), where=unions != 0
        )

    @staticmethod
    def multiple_top_n(objs, n=10):
        result = OrderedDict()
        without_answers = OrderedDict()
        for name, report in list(objs[0].per_class_result.items()):
            flag = False
            value = []

            for key in report.keys():
                box = []
                for obj in objs:
                    count = np.sum(np.array(obj.prediction_scores[name][key]) > 0.3)
                    box.append(obj.prediction_boxes[name][key][:count])

                similarity_matrix = objs[0].calculate_similarity_matrix(box[0], box[1])

                if similarity_matrix.size != 0:
                    flag = True

                count_match = np.sum(np.sum(similarity_matrix > 0.5, axis=0) >= 1.0)
                value.append(np.divide(count_match, len(box[0]), out=np.zeros(1), where=(len(box[0]) != 0)))

            if flag:
                result[name] = np.mean(value)
            else:
                without_answers[name] = name

        sort_key = sorted(result, key=lambda k: (result[k]), reverse=False)[:n]

        color = np.zeros((len(objs), 1, 3), np.uint8)

        for i in range(len(objs)):
            color[i] = np.array([i * int(180 / (len(objs))), 255, 255])

        color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        color = np.resize(color, new_shape=(len(objs), 3))

        for name in without_answers:
            image = cv2.imread(objs[0].picture_directory + name)

            cv2.imshow(name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break

        for name in sort_key:
            image = cv2.imread(objs[0].picture_directory + name)

            for i, obj in enumerate(objs):
                obj.marking_predition(image, name, tuple(color[i]), 0.3)

            cv2.imshow(name, image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break




