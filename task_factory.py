import classification
import detection
import segmentation


class MetricAnalysisFactory:
    @staticmethod
    def create_task(name, data, file, directory, mask):
        type_task = {
            "classification": classification.Classification,
            "detection": detection.Detection,
            "segmentation": segmentation.Segmentation
        }
        return type_task[name](data, file, directory, mask)
