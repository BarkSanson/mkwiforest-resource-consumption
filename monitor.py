import os
import time
import threading
import psutil


class Monitor(threading.Thread):
    def __init__(self, pid, interval):
        super().__init__()
        self._process = psutil.Process(pid)
        self._interval = interval
        self._results = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            cpu_percent = self._process.cpu_percent()
            memory_mib = self._process.memory_info().rss / 1024 / 1024
            print(f"CPU: {cpu_percent:.2f}%, Memory: {memory_mib:.2f} MiB")
            self._results.append((cpu_percent, memory_mib))
            time.sleep(self._interval)

    def stop(self):
        self._stop_event.set()
