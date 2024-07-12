from monitor import Monitor


class MonitorFactory:
    def __init__(self, pid, interval, performance_df):
        self.pid = pid
        self.interval = interval
        self.performance_df = performance_df

    def create_monitor(self, name):
        return Monitor(name, self.pid, self.interval, self.performance_df)