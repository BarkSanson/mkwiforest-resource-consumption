import os
import sys

import pandas as pd
import psutil
from pympler import asizeof


class Monitor:
    def __init__(self, pid):
        super().__init__()
        self._process = psutil.Process(pid)
        self._cpu_df = pd.DataFrame(columns=['element', 'cpu_percent'])
        self._memory_df = pd.DataFrame(columns=['window', 'iforest', 'memory_mib'])
        self._pipeline = None

        self._last_cpu_measure = self._process.cpu_percent()
        self._base_memory = self._process.memory_info().rss / 1024 / 1024
        print(f"Base memory: {self._base_memory} MiB")

    def add_pipeline(self, pipeline):
        self._pipeline = pipeline

    def pre_measure(self):
        self._last_cpu_measure = self._process.cpu_percent()

    def post_measure(self, name):
        cpu_percent = self._process.cpu_percent()

        self._cpu_df.loc[len(self._cpu_df)] = [name, cpu_percent]
        window_size_mib = asizeof.asizeof(self._pipeline.window) / 1024 / 1024
        iforest_mib = asizeof.asizeof(self._pipeline.model) / 1024 / 1024
        memory_mib = self._process.memory_info().rss / 1024 / 1024 - sys.getsizeof(
            self._cpu_df) / 1024 / 1024 - sys.getsizeof(self._memory_df) / 1024 / 1024

        self._memory_df.loc[len(self._memory_df)] = [window_size_mib, iforest_mib, memory_mib]

    def write(self, window_size):
        drift = self._cpu_df[self._cpu_df['element'] == 'Mann-Kendall-Wilcoxon']
        scoring_ref = self._cpu_df[self._cpu_df['element'] == 'Scoring reference']
        scoring_online = self._cpu_df[self._cpu_df['element'] == 'Scoring online']
        retraining = self._cpu_df[self._cpu_df['element'] == 'Retraining']

        # Dump results into a log file
        pd.DataFrame({
            'element': ['Mann-Kendall-Wilcoxon', 'Scoring reference', 'Scoring online', 'Retraining'],
            'cpu_max': [drift['cpu_percent'].max(), scoring_ref['cpu_percent'].max(),
                        scoring_online['cpu_percent'].max(), retraining['cpu_percent'].max()],
            'cpu_percent_mean': [drift['cpu_percent'].mean(), scoring_ref['cpu_percent'].mean(),
                                 scoring_online['cpu_percent'].mean(), retraining['cpu_percent'].mean()],
            'cpu_std_dev': [drift['cpu_percent'].std(), scoring_ref['cpu_percent'].std(),
                            scoring_online['cpu_percent'].std(), retraining['cpu_percent'].std()],
        }).to_csv(os.path.join(os.getcwd(), f"./{window_size}_performance.csv"), index=False)

        drift.to_csv(os.path.join(os.getcwd(), f"./{window_size}_drift.csv"), index=False)
        scoring_ref.to_csv(os.path.join(os.getcwd(), f"./{window_size}_scoring_ref.csv"), index=False)
        scoring_online.to_csv(os.path.join(os.getcwd(), f"./{window_size}_scoring_online.csv"), index=False)
        retraining.to_csv(os.path.join(os.getcwd(), f"./{window_size}_retraining.csv"), index=False)

        pd.DataFrame({
            'element': ['Window', 'iForest', 'Program'],
            'memory_mib_mean': [self._memory_df[:]['window'].mean(),
                                self._memory_df[:]['iforest'].mean(),
                                self._memory_df[:]['memory_mib'].mean()],
            'memory_mib_max': [self._memory_df[:]['window'].max(),
                               self._memory_df[:]['iforest'].max(),
                               self._memory_df[:]['memory_mib'].max()],
        }).to_csv(os.path.join(os.getcwd(), f"./{window_size}_memory.csv"), index=False)
