import classification
import detection
import segmentation


class MetricAnalysisFactory:
    @staticmethod
    def create_task(name, data, file_name, directory, mask):
        type_task = {
            "classification": classification.Classification,
            "detection": detection.Detection,
            "segmentation": segmentation.Segmentation
        }
        return type_task[name](name, data, file_name, directory, mask)
