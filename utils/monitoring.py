"""System monitoring utilities."""

import psutil
from typing import Dict, Any
from prometheus_client import Counter, Gauge, Histogram

class MetricsCollector:
    """Collects and exposes system metrics."""
    
    def __init__(self):
        # Counters
        self.error_counter = Counter('system_errors_total', 'Total number of system errors')
        self.trade_counter = Counter('trades_total', 'Total number of trades')
        
        # Gauges
        self.cpu_gauge = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_gauge = Gauge('memory_usage_percent', 'Memory usage percentage')
        
        # Histograms
        self.latency_histogram = Histogram('api_latency_seconds', 'API latency in seconds')
        self.trade_duration_histogram = Histogram('trade_duration_seconds', 'Trade duration in seconds')

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counter.inc()

    def record_trade(self):
        """Record a trade occurrence."""
        self.trade_counter.inc()

    def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics."""
        if 'cpu_percent' in metrics:
            self.cpu_gauge.set(metrics['cpu_percent'])
        if 'memory_percent' in metrics:
            self.memory_gauge.set(metrics['memory_percent'])

    def record_latency(self, seconds: float):
        """Record API latency."""
        self.latency_histogram.observe(seconds)

    def record_trade_duration(self, seconds: float):
        """Record trade duration."""
        self.trade_duration_histogram.observe(seconds)

    def collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available
            }
            
            self.update_system_metrics(metrics)
            return metrics
        except Exception as e:
            self.record_error("system_metrics_collection_error")
            raise e

# Global metrics collector instance
_metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector