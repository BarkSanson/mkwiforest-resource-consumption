from typing import Union, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

from monitor import Monitor
from monitor_factory import MonitorFactory
from online_outlier_detection.drift import MannKendallWilcoxonDriftDetector
from online_outlier_detection.pipelines.base.batch_detector_pipeline import BatchDetectorPipeline


class MKWIForestBatchPipeline(BatchDetectorPipeline):
    SAMPLES = 0

    def __init__(self,
                 monitor_factory: MonitorFactory,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)
        self.monitor_factory = monitor_factory

        self.model = IsolationForest()
        self.drift_detector = MannKendallWilcoxonDriftDetector(alpha, slope_threshold, self.monitor_factory)

        self.samples = 0

    def update(self, x) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        self.window.append(x)

        if not self.window.is_full():
            return None

        if not self.warm:
            return self._first_training()

        MKWIForestBatchPipeline.SAMPLES += 1

        if self.drift_detector.detect_drift(self.window.get(), self.reference_window):
            monitor = self.monitor_factory.create_monitor("Retraining")
            monitor.start()
            self._retrain()
            monitor.stop()
            monitor.join()

            monitor = self.monitor_factory.create_monitor("Scoring")
            monitor.start()
            scores = np.abs(self.model.score_samples(self.reference_window.reshape(-1, 1)))
            labels = np.where(scores > self.score_threshold, 1, 0)
            monitor.stop()
            monitor.join()

            self.window.clear()
            return scores, labels

        monitor = self.monitor_factory.create_monitor("Scoring")
        monitor.start()
        scores = np.abs(self.model.score_samples(self.window.get().reshape(-1, 1)))
        labels = np.where(scores > self.score_threshold, 1, 0)
        monitor.stop()
        monitor.join()

        self.window.clear()
        return scores, labels
