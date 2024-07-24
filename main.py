import os

import pandas as pd

import tracemalloc

from monitor import Monitor
from online_outlier_detection.pipelines import MKWIForestBatchPipeline

INTERVAL = 0.01
MAX_SAMPLES = 100


def merge_data(date_dir):
    df = pd.DataFrame()
    for file in os.listdir(date_dir):
        new_df = pd.read_csv(f"{date_dir}/{file}")
        df = pd.concat([df, new_df])

    df = df.set_index(pd.to_datetime(df['dateTime']))
    df = df.drop(columns=['dateTime'])

    return df


def get_performance(data_list, data_path, window_size):
    pid = os.getpid()
    monitor = Monitor(pid)

    for station in data_list:
        path = f"{data_path}/{station}"

        for date in os.listdir(path):
            model = MKWIForestBatchPipeline(
                monitor=monitor,
                score_threshold=0.8,
                alpha=0.05,
                slope_threshold=0.1,
                window_size=window_size)
            df = merge_data(f"{path}/{date}")

            for _, x in df.iterrows():
                if MKWIForestBatchPipeline.SAMPLES >= MAX_SAMPLES:
                    MKWIForestBatchPipeline.SAMPLES = 0

                    monitor.write(window_size)

                    return

                _ = model.update(x['value'])


def main():
    tracemalloc.start()
    window_sizes = [32, 64, 128, 256]

    data_path = "./labeled_data"

    data_list = [x for x in os.listdir(data_path) if os.path.isdir(f"{data_path}/{x}")]
    for window_size in window_sizes:
        print("MEASURING WINDOW SIZE: ", window_size)
        get_performance(data_list, data_path, window_size)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 50 ]")
    for stat in top_stats[:50]:
        print(stat)


if __name__ == '__main__':
    main()
