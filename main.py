import sys

import numpy.random

import pulse
from online_outlier_detection.pipelines import MKWIForestBatchPipeline

MAX_SAMPLES = 10

BLOCK = numpy.random.normal(0.5, 0.5, 64)


def main():
    window_size = int(sys.argv[1])

    for _ in range(MAX_SAMPLES):
        model = MKWIForestBatchPipeline(
            score_threshold=0.8,
            alpha=0.05,
            slope_threshold=0.1,
            window_size=window_size)

        for _, x in BLOCK:
            _ = model.update(x)

    pulse.clean()


if __name__ == '__main__':
    main()
