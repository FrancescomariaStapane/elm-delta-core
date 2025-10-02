import numpy as np


class RepeatedMeasurement:
    def __init__(self, name, measurements):
        self.measurements = measurements
        self.name = name
    def get_mean(self):
        return np.mean(self.measurements)
    def get_std(self):
        return np.std(self.measurements)
    def get_median(self):
        return np.median(self.measurements)
    def get_n_of_measurements(self):
        return len(self.measurements)
    def get_name(self):
        return self.name




class ExperimentInfo:
    def __init__(self, name, value,):
        self.value = value
        self.name = name



class Experiment:
    def __init__(self, name: str, experiment_infos: list[ExperimentInfo], repeated_measurement: list[RepeatedMeasurement]):
        self.name = name
        self.experiment_infos = experiment_infos
        self.repeated_measurements = repeated_measurement

    def get_repeated_measurement(self, name: str):
        for measurement in self.repeated_measurements:
            if name == measurement.name:
                return measurement
        return None

    def get_info(self, name: str):
        for info in self.experiment_infos:
            if name == info.name:
                return info
        return None