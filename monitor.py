import os
import threading
import time

import pandas as pd
import psutil
from pympler import asizeof


class Monitor(threading.Thread):
    def __init__(self, pid, interval):
        super().__init__()
        self._process = psutil.Process(pid)
        self._memory_df = pd.DataFrame(columns=[
            'program', 'window', 'reference_window', 'total_window', 'iforest'])
        self._pipeline = None
        self._running = True
        self._interval = interval

    def add_pipeline(self, pipeline):
        self._pipeline = pipeline

    def run(self):
        while self._running:
            online_window_size_kib = asizeof.asizeof(self._pipeline.window) / 1024
            reference_window_size_kib = asizeof.asizeof(self._pipeline.reference_window) / 1024
            iforest_kib = asizeof.asizeof(self._pipeline.model) / 1024
            memory_kib = self._process.memory_info().rss / 1024 - asizeof.asizeof(self._memory_df) / 1024

            self._memory_df.loc[len(self._memory_df)] = [
                memory_kib,
                online_window_size_kib,
                reference_window_size_kib,
                online_window_size_kib + reference_window_size_kib,
                iforest_kib]

            time.sleep(self._interval)

    def stop(self):
        self._running = False

    def write(self, window_size):
        pd.DataFrame({
            'element': ['Program', 'Online window', 'Reference window', 'Total window', 'iForest'],
            'memory_kib_mean': [
                self._memory_df[:]['program'].mean(),
                self._memory_df[:]['window'].mean(),
                self._memory_df[:]['reference_window'].mean(),
                self._memory_df[:]['total_window'].mean(),
                self._memory_df[:]['iforest'].mean()],
            'memory_kib_max': [
                self._memory_df[:]['program'].max(),
                self._memory_df[:]['window'].max(),
                self._memory_df[:]['reference_window'].max(),
                self._memory_df[:]['total_window'].max(),
                self._memory_df[:]['iforest'].max()]
        }).to_csv(os.path.join(os.getcwd(), f"./{window_size}_memory.csv"), index=False)
