"""Benchmark tests to track system performance metrics over time"""
import pytest
import json
import os
import time
import datetime
from pathlib import Path
from typing import Dict, Any

BENCHMARK_FILE = "benchmark_history.json"

class BenchmarkTracker:
    """Tracks and persists benchmark results"""
    
    def __init__(self, benchmark_file: str = BENCHMARK_FILE):
        self.benchmark_file = Path("tests/performance/history") / benchmark_file
        self.benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Load existing benchmark history"""
        if self.benchmark_file.exists():
            with open(self.benchmark_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save_benchmark(self, name: str, metrics: Dict[str, Any]):
        """Save new benchmark results"""
        benchmark_data = {
            'name': name,
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics
        }
        self.history.append(benchmark_data)
        
        with open(self.benchmark_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_previous_benchmark(self, name: str) -> Dict[str, Any]:
        """Get most recent benchmark results for comparison"""
        matching = [b for b in self.history if b['name'] == name]
        return matching[-1] if matching else None

@pytest.mark.benchmark
class TestSystemBenchmarks:
    """System benchmark tests"""
    
    @pytest.fixture(autouse=True)
    def setup_tracker(self):
        """Setup benchmark tracker"""
        self.tracker = BenchmarkTracker()
        
    def compare_with_baseline(self, name: str, current_metrics: Dict[str, float], 
                            max_regression: float = 0.1):
        """Compare current metrics with baseline"""
        previous = self.tracker.get_previous_benchmark(name)
        if not previous:
            return True  # No baseline yet
            
        prev_metrics = previous['metrics']
        for key in ['median', 'mean']:
            if key in current_metrics and key in prev_metrics:
                regression = (current_metrics[key] - prev_metrics[key]) / prev_metrics[key]
                assert regression <= max_regression, \
                    f"Performance regression of {regression:.1%} detected in {key}"

    @pytest.mark.benchmark
    def test_market_data_benchmark(self, test_credentials):
        """Benchmark market data retrieval performance"""
        from python_libs.bybit_v5_connector import BybitV5Connector
        
        connector = BybitV5Connector(**test_credentials)
        symbol = "BTCUSDT"
        
        # Measure baseline latency
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            connector.get_market_data(symbol)
            latencies.append(time.perf_counter() - start)
            
        metrics = {
            'min': min(latencies),
            'max': max(latencies),
            'mean': sum(latencies) / len(latencies),
            'median': sorted(latencies)[len(latencies)//2]
        }
        
        # Compare with previous benchmark
        self.compare_with_baseline('market_data_latency', metrics)
        
        # Save new benchmark
        self.tracker.save_benchmark('market_data_latency', metrics)
        
    @pytest.mark.benchmark
    def test_data_processing_benchmark(self):
        """Benchmark data processing pipeline performance"""
        import pandas as pd
        import numpy as np
        from data.data.data_preprocessing import preprocess_pipeline
        
        # Generate consistent test dataset
        np.random.seed(42)
        rows = 10000
        dates = pd.date_range("2023-01-01", periods=rows, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, rows),
            "high": np.random.uniform(110, 120, rows),
            "low": np.random.uniform(90, 100, rows),
            "close": np.random.uniform(100, 115, rows),
            "volume": np.random.randint(1000, 2000, rows),
        })
        
        processing_times = []
        iterations = 5
        
        for _ in range(iterations):
            start = time.perf_counter()
            preprocess_pipeline(df.copy(), price_col="close", fill_method="ffill")
            processing_times.append(time.perf_counter() - start)
            
        metrics = {
            'min': min(processing_times),
            'max': max(processing_times),
            'mean': sum(processing_times) / len(processing_times),
            'median': sorted(processing_times)[len(processing_times)//2],
            'rows_per_second': rows / (sum(processing_times) / len(processing_times))
        }
        
        # Compare with previous benchmark
        self.compare_with_baseline('data_processing', metrics)
        
        # Save new benchmark
        self.tracker.save_benchmark('data_processing', metrics)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])