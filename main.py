import sys

import numpy as np

import pulse
from online_outlier_detection.pipelines import MKWIForestBatchPipeline

SAMPLES = 10

def main():
    pulse.generate_trigger_pulse()
    window_size = int(sys.argv[1])

    blocks = 2 * [np.random.normal(0.5, 0.1, window_size)]

    for _ in range(SAMPLES):
        model = MKWIForestBatchPipeline(
            score_threshold=0.8,
            alpha=0.05,
            slope_threshold=0.1,
            window_size=window_size)

        pulse.generate_trigger_pulse()
        for b in blocks:
            for x in b:
                _ = model.update(x)

    pulse.clean()


if __name__ == '__main__':
    main()
