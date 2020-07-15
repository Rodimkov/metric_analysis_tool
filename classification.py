import cv2
import numpy as np


def visual(data, picture_directory):
    dataset_meta = data.get("dataset_meta")

    for report in data.get("report"):
        identifier = report.get("identifier")

        prediction_label = dataset_meta.get("label_map").get(str(report.get("prediction_label")))
        annotation_label = dataset_meta.get("label_map").get(str(report.get("annotation_label")))

        print("prediction label:", prediction_label,
              "annotation label:", annotation_label,
              "prediction scores:", np.max(report.get("prediction_scores")))

        image = cv2.imread(picture_directory + identifier)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        cv2.destroyAllWindows()
