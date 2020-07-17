import cv2
import numpy as np


def visual(data, picture_directory):
    dataset_meta = data.get("dataset_meta")
    reports = data.get("report", [])

    for report in reports:
        identifier = report.get("identifier")

        prediction_label = dataset_meta.get("label_map").get(str(report.get("prediction_label")))
        annotation_label = dataset_meta.get("label_map").get(str(report.get("annotation_label")))

        print("image name:", identifier,
              "\nprediction label:", prediction_label,
              "annotation label:", annotation_label,
              "prediction scores:", np.max(report.get("prediction_scores")))

        image = cv2.imread(picture_directory + identifier)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            break
        cv2.destroyAllWindows()
