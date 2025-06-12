class MetricsCollector:
    def __init__(self):
        self.metrics = {}

    def record(self, name: str, value: float):
        self.metrics[name] = value
