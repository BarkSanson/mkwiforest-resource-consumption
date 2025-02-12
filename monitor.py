import threading
import time

import psutil


class Monitor(threading.Thread):
    def __init__(self, name, pid, interval, performance_df):
        super().__init__()
        self.name = name

        self._process = psutil.Process(pid)
        self._interval = interval
        self._stop_event = threading.Event()
        self._performance_df = performance_df

    def run(self):
        while not self._stop_event.is_set():
            cpu_percent = self._process.cpu_percent()
            memory_mib = self._process.memory_info().rss / 1024 / 1024
            self._performance_df.loc[len(self._performance_df)] = [self.name, cpu_percent, memory_mib]
            time.sleep(self._interval)

        self._stop_event.clear()

    def stop(self):
        self._stop_event.set()
