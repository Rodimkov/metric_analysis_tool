import classification
import detection
import segmentation
import gui_classification

class MetricAnalysisFactory:
    @staticmethod
    def create_task(name, data, file_name, directory, mask):
        type_task = {
            "classification": gui_classification.ChCl,
            "detection": detection.Detection,
            "segmentation": segmentation.Segmentation
        }
        return type_task[name](name, data, file_name, directory, mask)
