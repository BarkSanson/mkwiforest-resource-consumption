import os

import pandas as pd

from monitor_factory import MonitorFactory
from online_outlier_detection.pipelines import MKWIForestBatchPipeline

INTERVAL = 0.01
MAX_SAMPLES = 20


def merge_data(date_dir):
    df = pd.DataFrame()
    for file in os.listdir(date_dir):
        new_df = pd.read_csv(f"{date_dir}/{file}")
        df = pd.concat([df, new_df])

    df = df.set_index(pd.to_datetime(df['dateTime']))
    df = df.drop(columns=['dateTime'])

    return df


def get_performance(data_list, data_path, window_size):
    performance = pd.DataFrame(columns=['element', 'cpu_percent', 'memory_mib'])
    pid = os.getpid()
    factory = MonitorFactory(pid, INTERVAL, performance)

    for station in data_list:
        path = f"{data_path}/{station}"

        for date in os.listdir(path):
            model = MKWIForestBatchPipeline(
                monitor_factory=factory,
                score_threshold=0.8,
                alpha=0.05,
                slope_threshold=0.1,
                window_size=window_size)
            df = merge_data(f"{path}/{date}")

            for _, x in df.iterrows():
                if MKWIForestBatchPipeline.SAMPLES >= MAX_SAMPLES:
                    MKWIForestBatchPipeline.SAMPLES = 0
                    drift = performance[performance['element'] == 'Mann-Kendall-Wilcoxon']
                    scoring = performance[performance['element'] == 'Scoring']
                    retraining = performance[performance['element'] == 'Retraining']

                    # Dump results into a log file
                    pd.DataFrame({
                        'element': ['Mann-Kendall-Wilcoxon', 'Scoring', 'Retraining'],
                        'cpu_max': [drift['cpu_percent'].max(), scoring['cpu_percent'].max(), retraining['cpu_percent'].max()],
                        'cpu_percent_mean': [drift['cpu_percent'].mean(), scoring['cpu_percent'].mean(), retraining['cpu_percent'].mean()],
                        'cpu_std_dev': [drift['cpu_percent'].std(), scoring['cpu_percent'].std(), retraining['cpu_percent'].std()],
                        'memory_mib': [drift['memory_mib'].mean(), scoring['memory_mib'].mean(), retraining['memory_mib'].mean()],
                        'max_memory_mib': [drift['memory_mib'].max(), scoring['memory_mib'].max(), retraining['memory_mib'].max()]
                    }).to_csv(os.path.join(os.getcwd(), f"./{window_size}_performance.csv"), index=False)

                    return

                _ = model.update(x['value'])


def main():
    window_sizes = [32, 64, 128, 256]

    data_path = "./labeled_data"

    data_list = [x for x in os.listdir(data_path) if os.path.isdir(f"{data_path}/{x}")]
    for window_size in window_sizes:
        print("MEASURING WINDOW SIZE: ", window_size)
        get_performance(data_list, data_path, window_size)


if __name__ == '__main__':
    main()
