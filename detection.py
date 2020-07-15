import cv2


def draw_boxes(image, boxes, label_class, color=(255, 255, 255), thickness=2):
    for i, box in enumerate(boxes):
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, label_class[i], (int(box[0] + 6), int(box[3]) - 6), font, 0.4, color, 1)
        image = cv2.rectangle(image, start_point, end_point, color, thickness, cv2.FILLED)

    return image


def draw_info_class(image, class_result, dataset_meta):
    threshold_scores = 0.5
    for id_class in class_result:

        prediction_boxes = class_result.get(id_class).get("prediction_boxes", [])
        annotation_boxes = class_result.get(id_class).get("annotation_boxes", [])
        prediction_scores = class_result.get(id_class).get("prediction_scores", [])

        prediction_boxes = [box for box, score in zip(prediction_boxes, prediction_scores)
                            if score > threshold_scores]

        label_class = []
        for score in prediction_scores:
            if score > threshold_scores:
                name_class = dataset_meta.get("label_map").get(id_class)
                label_class.append('{} {:.3f}'.format(name_class, score))

        image = draw_boxes(image, prediction_boxes, label_class, color=(255, 0, 0))

        label_class = [dataset_meta.get("label_map").get(id_class)] * len(annotation_boxes)
        image = draw_boxes(image, annotation_boxes, label_class, color=(0, 0, 255))


def visual(data, picture_directory):
    dataset_meta = data.get("dataset_meta")
    reports = data.get("report", [])

    for report in reports:
        identifier = report.get("identifier")
        class_result = report.get("per_class_result", [])

        image = cv2.imread(picture_directory + identifier)

        draw_info_class(image, class_result, dataset_meta)

        cv2.imshow(identifier, image)
        key = cv2.waitKey(0)
        if key == 27:
            break
        cv2.destroyAllWindows()
