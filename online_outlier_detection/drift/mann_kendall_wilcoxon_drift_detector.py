import numpy as np
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon

from monitor_factory import MonitorFactory
from .base_drift_detector import BaseDriftDetector


class MannKendallWilcoxonDriftDetector(BaseDriftDetector):
    def __init__(self, alpha: float, slope_threshold: float, monitor_factory: MonitorFactory):
        self.alpha = alpha
        self.slope_threshold = slope_threshold
        self.monitor_factory = monitor_factory

    def detect_drift(self, x: np.ndarray, y: np.ndarray) -> bool:
        monitor = self.monitor_factory.create_monitor("Mann-Kendall-Wilcoxon")
        monitor.start()

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(x)
        d = np.around(x - y, decimals=3)
        stat, p_value = wilcoxon(d, zero_method='zsplit')

        monitor.stop()
        monitor.join()

        return (h and slope > self.slope_threshold) or p_value < self.alpha
