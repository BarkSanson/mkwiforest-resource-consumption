import numpy as np
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon

from .base_drift_detector import BaseDriftDetector
from monitor import Monitor


class MannKendallWilcoxonDriftDetector(BaseDriftDetector):
    def __init__(self, monitor: Monitor, alpha: float, slope_threshold: float):
        self.alpha = alpha
        self.slope_threshold = slope_threshold
        self.monitor = monitor

    def detect_drift(self, x: np.ndarray, y: np.ndarray) -> bool:
        self.monitor.pre_measure()
        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(x)
        d = np.around(x - y, decimals=3)
        stat, p_value = wilcoxon(d, zero_method='zsplit')
        self.monitor.post_measure('Mann-Kendall-Wilcoxon')

        return (h and slope > self.slope_threshold) or p_value < self.alpha
