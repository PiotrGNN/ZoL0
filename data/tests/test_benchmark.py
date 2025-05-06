"""
test_benchmark.py
--------------
Performance benchmarking tests for critical system components.
"""

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any, List
from datetime import datetime

from data.tests import BaseTestCase
from data.data.data_preprocessing import preprocess_pipeline
from data.ai_models.model_loader import ModelLoader
from data.execution.order_execution import OrderExecution
from data.risk_management.position_sizing import calculate_position_size

class TestBenchmark(BaseTestCase):
    """Performance benchmarking tests."""

    def setUp(self):
        """Initialize benchmark resources."""
        super().setUp()
        self.benchmark_dir = Path("benchmarks")
        self.benchmark_dir.mkdir(exist_ok=True)
        
        self.benchmark_file = self.benchmark_dir / "benchmark_results.json"
        
        # Initialize test data
        self.small_dataset = self.generate_test_data(periods=100)
        self.medium_dataset = self.generate_test_data(periods=1000)
        self.large_dataset = self.generate_test_data(periods=10000)

    def test_data_processing_benchmark(self):
        """Benchmark data preprocessing performance."""
        datasets = {
            "small": self.small_dataset,
            "medium": self.medium_dataset,
            "large": self.large_dataset
        }
        
        results = {}
        for name, data in datasets.items():
            times = []
            for _ in range(5):  # Run multiple times for stable results
                start_time = time.time()
                processed_df = preprocess_pipeline(
                    data,
                    price_col="close",
                    fill_method="ffill",
                    outlier_threshold=2.5
                )
                execution_time = time.time() - start_time
                times.append(execution_time)
            
            avg_time = np.mean(times)
            results[name] = {
                "avg_time": avg_time,
                "records_per_second": len(data) / avg_time,
                "dataset_size": len(data)
            }
            
            # Verify performance meets requirements
            max_allowed_time = {
                "small": 0.1,    # 100ms
                "medium": 0.5,   # 500ms
                "large": 2.0     # 2s
            }
            
            self.assertLess(
                avg_time,
                max_allowed_time[name],
                f"Data processing for {name} dataset too slow: "
                f"{avg_time:.3f}s > {max_allowed_time[name]}s"
            )
        
        self._save_benchmark_results("data_processing", results)

    def test_model_inference_benchmark(self):
        """Benchmark model inference performance."""
        model_loader = ModelLoader()
        models = model_loader.load_available_models()
        batch_sizes = [1, 10, 100, 1000]
        
        results = {}
        for model_name, model in models.items():
            if not hasattr(model, "predict"):
                continue
                
            model_results = {}
            for batch_size in batch_sizes:
                # Prepare feature data
                features = pd.DataFrame(
                    np.random.random((batch_size, 10)),
                    columns=[f"feature_{i}" for i in range(10)]
                )
                
                times = []
                for _ in range(10):  # Multiple runs for stability
                    start_time = time.time()
                    _ = model.predict(features)
                    execution_time = time.time() - start_time
                    times.append(execution_time)
                
                avg_time = np.mean(times)
                model_results[f"batch_{batch_size}"] = {
                    "avg_time": avg_time,
                    "predictions_per_second": batch_size / avg_time
                }
                
                # Verify latency requirements
                max_latency = {
                    1: 0.01,     # 10ms for single prediction
                    10: 0.02,    # 20ms for small batch
                    100: 0.05,   # 50ms for medium batch
                    1000: 0.2    # 200ms for large batch
                }
                
                self.assertLess(
                    avg_time,
                    max_latency[batch_size],
                    f"Model {model_name} inference too slow for batch size {batch_size}: "
                    f"{avg_time:.3f}s > {max_latency[batch_size]}s"
                )
            
            results[model_name] = model_results
        
        self._save_benchmark_results("model_inference", results)

    def test_order_execution_benchmark(self):
        """Benchmark order execution performance."""
        execution = OrderExecution()
        num_orders = [1, 10, 50, 100]
        
        results = {}
        for num in num_orders:
            orders = [
                {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": 0.001
                }
                for _ in range(num)
            ]
            
            times = []
            for _ in range(5):  # Multiple runs
                start_time = time.time()
                for order in orders:
                    _ = execution.send_order(**order)
                execution_time = time.time() - start_time
                times.append(execution_time)
            
            avg_time = np.mean(times)
            results[f"orders_{num}"] = {
                "avg_time": avg_time,
                "orders_per_second": num / avg_time,
                "avg_latency": avg_time / num
            }
            
            # Verify execution speed
            max_batch_time = {
                1: 0.05,    # 50ms for single order
                10: 0.3,    # 300ms for 10 orders
                50: 1.0,    # 1s for 50 orders
                100: 1.5    # 1.5s for 100 orders
            }
            
            self.assertLess(
                avg_time,
                max_batch_time[num],
                f"Order execution too slow for {num} orders: "
                f"{avg_time:.3f}s > {max_batch_time[num]}s"
            )
        
        self._save_benchmark_results("order_execution", results)

    def test_risk_calculation_benchmark(self):
        """Benchmark risk calculation performance."""
        num_calculations = [100, 1000, 10000]
        
        results = {}
        for num in num_calculations:
            # Generate random scenarios
            scenarios = [
                {
                    "balance": np.random.uniform(1000, 100000),
                    "risk": np.random.uniform(0.01, 0.05),
                    "entry": np.random.uniform(10000, 60000),
                    "stop": np.random.uniform(9000, 59000)
                }
                for _ in range(num)
            ]
            
            times = []
            for _ in range(5):  # Multiple runs
                start_time = time.time()
                for scenario in scenarios:
                    _ = calculate_position_size(
                        account_balance=scenario["balance"],
                        risk_per_trade=scenario["risk"],
                        entry_price=scenario["entry"],
                        stop_loss_price=scenario["stop"]
                    )
                execution_time = time.time() - start_time
                times.append(execution_time)
            
            avg_time = np.mean(times)
            results[f"calcs_{num}"] = {
                "avg_time": avg_time,
                "calculations_per_second": num / avg_time
            }
            
            # Verify calculation speed
            max_calc_time = {
                100: 0.01,    # 10ms for 100 calculations
                1000: 0.05,   # 50ms for 1000 calculations
                10000: 0.3    # 300ms for 10000 calculations
            }
            
            self.assertLess(
                avg_time,
                max_calc_time[num],
                f"Risk calculations too slow for {num} scenarios: "
                f"{avg_time:.3f}s > {max_calc_time[num]}s"
            )
        
        self._save_benchmark_results("risk_calculation", results)

    def test_memory_usage_benchmark(self):
        """Benchmark memory usage of critical operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        results = {}
        
        # Test data processing memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        _ = preprocess_pipeline(self.large_dataset)
        final_memory = process.memory_info().rss / 1024 / 1024
        
        results["data_processing"] = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - initial_memory
        }
        
        # Verify memory usage
        max_memory_increase = 500  # 500MB
        self.assertLess(
            final_memory - initial_memory,
            max_memory_increase,
            f"Memory usage too high: {final_memory - initial_memory:.1f}MB"
        )
        
        self._save_benchmark_results("memory_usage", results)

    def _save_benchmark_results(self, benchmark_type: str, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        if self.benchmark_file.exists():
            with open(self.benchmark_file) as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Update results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if benchmark_type not in all_results:
            all_results[benchmark_type] = {}
        
        all_results[benchmark_type][timestamp] = results
        
        # Save updated results
        with open(self.benchmark_file, "w") as f:
            json.dump(all_results, f, indent=2)

    def _analyze_historical_performance(
        self,
        benchmark_type: str,
        metric: str,
        threshold: float
    ) -> bool:
        """Analyze historical benchmark results."""
        if not self.benchmark_file.exists():
            return True
            
        with open(self.benchmark_file) as f:
            all_results = json.load(f)
        
        if benchmark_type not in all_results:
            return True
            
        # Get historical values for metric
        historical_values = []
        for timestamp, results in all_results[benchmark_type].items():
            if metric in results:
                historical_values.append(results[metric])
        
        if not historical_values:
            return True
            
        # Calculate trend
        trend = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
        return trend <= threshold  # Return True if performance is stable or improving

    def tearDown(self):
        """Clean up benchmark resources."""
        super().tearDown()
        # Keep benchmark results for historical analysis