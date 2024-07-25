import os
import sys

import pandas as pd

from monitor import Monitor
from online_outlier_detection.pipelines import MKWIForestBatchPipeline

INTERVAL = 0.5
MAX_SAMPLES = 100


def merge_data(date_dir):
    df = pd.DataFrame()
    for file in os.listdir(date_dir):
        new_df = pd.read_csv(f"{date_dir}/{file}")
        df = pd.concat([df, new_df])

    df = df.set_index(pd.to_datetime(df['dateTime']))
    df = df.drop(columns=['dateTime'])

    return df


def main():
    window_size = int(sys.argv[1])

    data_path = "./labeled_data"

    data_list = [x for x in os.listdir(data_path) if os.path.isdir(f"{data_path}/{x}")]
    pid = os.getpid()

    monitor = Monitor(pid, INTERVAL)

    for station in data_list:
        path = f"{data_path}/{station}"

        for date in os.listdir(path):
            model = MKWIForestBatchPipeline(
                score_threshold=0.8,
                alpha=0.05,
                slope_threshold=0.1,
                window_size=window_size)
            monitor.add_pipeline(model)
            if not monitor.is_alive():
                monitor.start()
            df = merge_data(f"{path}/{date}")

            for _, x in df.iterrows():
                if MKWIForestBatchPipeline.SAMPLES >= MAX_SAMPLES:
                    MKWIForestBatchPipeline.SAMPLES = 0

                    monitor.stop()
                    monitor.join()
                    monitor.write(window_size)

                    return

                _ = model.update(x['value'])


if __name__ == '__main__':
    main()
