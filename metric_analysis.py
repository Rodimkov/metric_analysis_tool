from abc import ABC, abstractmethod


class MetricAnalysis(ABC):

    @abstractmethod
    def parser(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def visualize_data(self):
        pass
