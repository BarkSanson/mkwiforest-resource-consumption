from typing import Union, Tuple
import numpy as np

from online_outlier_detection.pipelines.base.base_detector_pipeline import BaseDetectorPipeline
from online_outlier_detection.window.sliding_window import SlidingWindow


class SlidingDetectorPipeline(BaseDetectorPipeline):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int,
                 step_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)

        self.window = SlidingWindow(self.window_size)
        self.step_size = step_size
        self.current_step = 1

    def update(self, x) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        pass
