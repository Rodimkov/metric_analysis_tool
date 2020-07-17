import classification
import detection


class MetricAnalysisFactory:
    @staticmethod
    def create_task(name, file_json, directory):
        type_task = {
            "Classification": classification.Classification,
            "Detection": detection.Detection
        }
        return type_task[name](file_json, directory)
