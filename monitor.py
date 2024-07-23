import os
import sys

import pandas as pd
import psutil


class Monitor:
    def __init__(self, pid):
        super().__init__()
        self._process = psutil.Process(pid)
        self._performance_df = pd.DataFrame(columns=['element', 'cpu_percent', 'memory_mib'])

        self._last_cpu_measure = self._process.cpu_percent()
        self._base_memory = self._process.memory_info().rss / 1024 / 1024
        print(f"Base memory: {self._base_memory} MiB")

    def pre_measure(self):
        self._last_cpu_measure = self._process.cpu_percent()

    def post_measure(self, name):
        cpu_percent = self._process.cpu_percent()
        memory_mib = self._process.memory_info().rss / 1024 / 1024 - self._base_memory - sys.getsizeof(self._performance_df) / 1024 / 1024
        self._performance_df.loc[len(self._performance_df)] = [name, cpu_percent, memory_mib]

    def write(self, window_size):
        drift = self._performance_df[self._performance_df['element'] == 'Mann-Kendall-Wilcoxon']
        scoring_ref = self._performance_df[self._performance_df['element'] == 'Scoring reference']
        scoring_online = self._performance_df[self._performance_df['element'] == 'Scoring online']
        retraining = self._performance_df[self._performance_df['element'] == 'Retraining']

        # Dump results into a log file
        pd.DataFrame({
            'element': ['Mann-Kendall-Wilcoxon', 'Scoring reference', 'Scoring online', 'Retraining'],
            'cpu_max': [drift['cpu_percent'].max(), scoring_ref['cpu_percent'].max(),
                        scoring_online['cpu_percent'].max(), retraining['cpu_percent'].max()],
            'cpu_percent_mean': [drift['cpu_percent'].mean(), scoring_ref['cpu_percent'].mean(),
                                 scoring_online['cpu_percent'].mean(), retraining['cpu_percent'].mean()],
            'cpu_std_dev': [drift['cpu_percent'].std(), scoring_ref['cpu_percent'].std(),
                            scoring_online['cpu_percent'].std(), retraining['cpu_percent'].std()],
            'memory_mib': [drift['memory_mib'].mean(), scoring_ref['memory_mib'].mean(),
                           scoring_online['memory_mib'].mean(), retraining['memory_mib'].mean()],
            'max_memory_mib': [drift['memory_mib'].max(), scoring_ref['memory_mib'].max(),
                               scoring_online['memory_mib'].max(), retraining['memory_mib'].max()]
        }).to_csv(os.path.join(os.getcwd(), f"./{window_size}_performance.csv"), index=False)

        drift.to_csv(os.path.join(os.getcwd(), f"./{window_size}_drift.csv"), index=False)
        scoring_ref.to_csv(os.path.join(os.getcwd(), f"./{window_size}_scoring_ref.csv"), index=False)
        scoring_online.to_csv(os.path.join(os.getcwd(), f"./{window_size}_scoring_online.csv"), index=False)
        retraining.to_csv(os.path.join(os.getcwd(), f"./{window_size}_retraining.csv"), index=False)

