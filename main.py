import os

import pandas as pd

from monitor import Monitor
from online_outlier_detection.pipelines import MKWIForestBatchPipeline

INTERVAL = 0.1
MAX_SAMPLES = 30


def merge_data(date_dir):
    df = pd.DataFrame()
    for file in os.listdir(date_dir):
        new_df = pd.read_csv(f"{date_dir}/{file}")
        df = pd.concat([df, new_df])

    df = df.set_index(pd.to_datetime(df['dateTime']))
    df = df.drop(columns=['dateTime'])

    return df


def main():
    window_sizes = [32, 64, 128, 256]

    data_path = "./labeled_data"

    data_list = [x for x in os.listdir(data_path) if os.path.isdir(f"{data_path}/{x}")]
    for window_size in window_sizes:
        samples = 0
        for station in data_list:
            path = f"{data_path}/{station}"

            for date in os.listdir(path):
                model = MKWIForestBatchPipeline(
                    score_threshold=0.8,
                    alpha=0.05,
                    slope_threshold=0.1,
                    window_size=window_size)
                df = merge_data(f"{path}/{date}")

                current_size = 0
                for _, x in df.iterrows():
                    if samples >= MAX_SAMPLES:
                        break
                    current_size += 1
                    if current_size == window_size:
                        current_size = 0
                        pid = os.getpid()
                        monitor = Monitor(pid, 0.01)
                        monitor.start()

                        _ = model.update(x['value'])

                        monitor.stop()
                        monitor.join()

                        samples += 1
                    else:
                        _ = model.update(x['value'])


if __name__ == '__main__':
    main()
