import os

import pandas as pd

from monitor import Monitor
from online_outlier_detection.pipelines import MKWIForestBatchPipeline

INTERVAL = 0.05
MAX_SAMPLES = 100


def merge_data(date_dir):
    df = pd.DataFrame()
    for file in os.listdir(date_dir):
        new_df = pd.read_csv(f"{date_dir}/{file}")
        df = pd.concat([df, new_df])

    df = df.set_index(pd.to_datetime(df['dateTime']))
    df = df.drop(columns=['dateTime'])

    return df


def get_performance(data_list, data_path, window_size, pid):
    samples = 0

    performance = pd.DataFrame(columns=['cpu_percent', 'memory_mib'])

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
                    performance = performance[performance['cpu_percent'] != 0]
                    cpu_max = performance['cpu_percent'].max()
                    cpu_mean = performance['cpu_percent'].mean()
                    memory_max = performance['memory_mib'].max()

                    # Dump results into a log file
                    pd.DataFrame({
                        'cpu_max': [cpu_max],
                        'cpu_mean': [cpu_mean],
                        'memory_max': [memory_max]
                    }).to_csv(os.path.join(os.getcwd(), f"./{window_size}_performance.csv"), index=False)

                    return

                if not model.warm:
                    _ = model.update(x['value'])
                    continue

                current_size += 1
                if current_size == window_size:
                    current_size = 0
                    monitor = Monitor(pid, INTERVAL, performance)
                    monitor.start()

                    _ = model.update(x['value'])

                    monitor.stop()
                    monitor.join()

                    samples += 1
                else:
                    _ = model.update(x['value'])


def main():
    window_sizes = [32, 64, 128, 256]
    pid = os.getpid()

    data_path = "./labeled_data"

    data_list = [x for x in os.listdir(data_path) if os.path.isdir(f"{data_path}/{x}")]
    for window_size in window_sizes:
        print("MEASURING WINDOW SIZE: ", window_size)
        get_performance(data_list, data_path, window_size, pid)
        input("Press Enter to continue...")


if __name__ == '__main__':
    main()
