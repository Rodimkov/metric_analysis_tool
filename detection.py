import cv2
import numpy as np
from collections import OrderedDict
import warnings
from metric_analysis import MetricAnalysis



class Detection(MetricAnalysis):

    def __init__(self, type_task, data, file_name, directory, mask, true_mask):
        super().__init__(type_task, data, file_name, directory, mask, true_mask)

        self.identifier = OrderedDict()
        self.index = []
        self.per_class_result = OrderedDict()
        self.prediction_boxes = OrderedDict()
        self.annotation_boxes = OrderedDict()
        self.prediction_scores = OrderedDict()
        self.average_precision = OrderedDict()
        self.average_precision_picture = OrderedDict()

        self.set_task = None
        self.threshold_scores = 0.5
        self.validate()
        self.parser()

    def validate(self):
        super().validate()

        report_error = []
        report_obj = self.data.get("report")[0]

        if not 'identifier' in report_obj:
            report_error.append('identifier')

        if not 'per_class_result' in report_obj:
            report_error.append('prediction_label')

        if report_error:
            report_error = ', '.join(report_error)
            raise KeyError("there are no keys in the file <json>: {}".format(report_error))

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
            report_error = ', '.join(report_error)
            raise KeyError("there are no keys in the file <json>: {}".format(report_error))

    def parser(self):
        super().parser()

        for report in self.reports:
            self.identifier[report["identifier"]] = report["identifier"]
            self.index.append(report["identifier"])
            self.per_class_result[report["identifier"]] = report["per_class_result"]

        for name, info_class in self.per_class_result.items():
            pb = OrderedDict()
            ab = OrderedDict()
            ps = OrderedDict()
            ap = OrderedDict()
            mean_ap = []

            for key, value in info_class.items():
                pb[key] = value["prediction_boxes"]
                ab[key] = value["annotation_boxes"]
                ps[key] = value["prediction_scores"]
                ap[key] = value["average_precision"]
                mean_ap.append(value["average_precision"])

            self.prediction_boxes[name] = pb
            self.annotation_boxes[name] = ab
            self.prediction_scores[name] = ps
            self.average_precision[name] = ap

            if not all(np.isnan(mean_ap)):
                self.average_precision_picture[name] = np.nanmean(mean_ap)
            else:
                self.average_precision_picture[name] = np.nan

    def _top_n(self, n=10):
        if n > self.size_dataset:
            warnings.warn("""value n is greater than the size of the dataset,
                             it will be reduced to the size of the dataset""")
            n = self.size_dataset

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

            if flag_prediction and flag_annotation:
                average_precision_score[name] = self.average_precision_picture[name]

        sort_average_precision = sorted(average_precision_score, key=lambda k: (average_precision_score[k]))[:n]

        return sort_average_precision

    def _plot_average_precision_changes(self, ax, k=100):
        precision_change = []
        precision = list(self.average_precision_picture.values())

        for i in range(int(k / 2), int(len(precision) - int(k / 2))):
            value = np.nanmean(precision[i:(i + k)])
            precision_change.append(value)
        x_range = range(int(k / 2), len(precision_change) + int(k / 2))

        ax.set_title("Change average precision in the process of predicting results")

        ax.set_xlabel("image number")
        ax.set_ylabel("average precision")

        ax.plot(x_range, precision_change)
        return ax

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

    def marking_predict(self, image, name, color, threshold_scores):
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

    def _visualize_data(self, name):
        image = cv2.imread(self.picture_directory + name)

        if image is None:
            raise KeyError("in directory {} no image {}".format(self.picture_directory, name))

        if self.flag_prediction:
            self.marking_predict(image, name, (255, 0, 0), self.threshold_scores)
        if self.flag_annotation:
            self.marking_annotation(image, name, (0, 0, 255))

        return image

    def _multiple_visualize_data(self, name):
        image = cv2.imread(self.set_task[0].picture_directory + name)

        if image is None:
            raise KeyError("in directory {} no image {}".format(self.picture_directory, name))

        color = np.zeros((2, 3), np.uint8)

        color[0] = np.array([0, 255, 0])
        color[1] = np.array([255, 0, 0])

        if not self.set_task[1].identifier.get(name, []):
            warnings.warn("in file {} no image {}".format(self.set_task[1].file, name))
        else:
            if self.flag_prediction:
                for i, task in enumerate(self.set_task):
                    task.marking_predict(image, name, tuple(color[i]), self.set_task[0].threshold_scores)
            if self.flag_annotation:
                self.marking_annotation(image, name, (0, 0, 255))

        return image

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
    def _multiple_top_n(set_task, n=10):
        result = OrderedDict()
        without_answers = OrderedDict()
        for name, report in list(set_task[0].per_class_result.items()):
            flag = False
            value = []

            if not set_task[1].identifier.get(name, []):
                warnings.warn("in file {} no image {}".format(set_task[1].file, name))
            else:
                for key in report.keys():
                    box = []
                    for task in set_task:
                        count = np.sum(np.array(task.prediction_scores[name][key]) > set_task[0].threshold_scores)
                        box.append(task.prediction_boxes[name][key][:count])

                    similarity_matrix = set_task[0].calculate_similarity_matrix(box[0], box[1])

                    if similarity_matrix.size != 0:
                        flag = True

                    count_match = np.sum(np.sum(similarity_matrix > 0.5, axis=0) >= 1.0)
                    value.append(np.divide(count_match, len(box[0]), out=np.zeros(1), where=(len(box[0]) != 0)))

                if flag:
                    result[name] = np.mean(value)
                else:
                    without_answers[name] = name

        if n > len(result):
            warnings.warn("""the n value is greater than the number of identical objects in both files, 
                            it will be reduced to the size of the dataset""")
            n = len(result)

        sort_key = sorted(result, key=lambda k: (result[k]), reverse=False)[:n]

        return sort_key
