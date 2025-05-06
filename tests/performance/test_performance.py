"""Performance tests for critical system components"""
import pytest
import time
import statistics
from typing import List, Dict, Any
from contextlib import contextmanager

@contextmanager
def measure_time():
    """Context manager to measure execution time"""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    return elapsed

@pytest.mark.performance
class TestSystemPerformance:
    """Performance tests for critical operations"""
    
    def collect_metrics(self, iterations: int, func: callable, *args, **kwargs) -> Dict[str, float]:
        """Collect performance metrics for a function"""
        times: List[float] = []
        for _ in range(iterations):
            with measure_time() as elapsed:
                func(*args, **kwargs)
            times.append(elapsed)
            
        return {
            'min': min(times),
            'max': max(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stddev': statistics.stdev(times) if len(times) > 1 else 0
        }

    @pytest.mark.performance
    def test_market_data_latency(self, test_credentials):
        """Test market data retrieval latency"""
        from python_libs.bybit_v5_connector import BybitV5Connector
        
        connector = BybitV5Connector(**test_credentials)
        symbol = "BTCUSDT"
        
        metrics = self.collect_metrics(
            iterations=10,
            func=connector.get_market_data,
            symbol=symbol
        )
        
        # Latency thresholds (in seconds)
        assert metrics['median'] < 1.0, f"High median latency: {metrics['median']:.3f}s"
        assert metrics['max'] < 2.0, f"High max latency: {metrics['max']:.3f}s"
        
    @pytest.mark.performance
    def test_data_processing_performance(self):
        """Test data processing pipeline performance"""
        import pandas as pd
        import numpy as np
        from data.data.data_preprocessing import preprocess_pipeline
        
        # Generate test data
        dates = pd.date_range("2023-01-01", periods=1000, freq="5min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, 1000),
            "high": np.random.uniform(110, 120, 1000),
            "low": np.random.uniform(90, 100, 1000),
            "close": np.random.uniform(100, 115, 1000),
            "volume": np.random.randint(1000, 2000, 1000),
        })
        
        metrics = self.collect_metrics(
            iterations=5,
            func=preprocess_pipeline,
            df=df.copy(),
            price_col="close",
            fill_method="ffill"
        )
        
        # Processing time thresholds for 1000 rows
        assert metrics['median'] < 0.5, f"Slow processing: {metrics['median']:.3f}s for 1000 rows"
        assert metrics['max'] < 1.0, f"Processing spike: {metrics['max']:.3f}s"
        
    @pytest.mark.performance
    def test_model_inference_speed(self):
        """Test AI model inference performance"""
        from ai_models.sentiment_ai import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        test_text = "Bitcoin price showing bullish momentum as trading volume increases"
        
        metrics = self.collect_metrics(
            iterations=50,
            func=analyzer.analyze,
            text=test_text
        )
        
        # Inference time thresholds
        assert metrics['median'] < 0.1, f"Slow inference: {metrics['median']:.3f}s per prediction"
        assert metrics['stddev'] < 0.05, f"High inference time variance: {metrics['stddev']:.3f}s"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])