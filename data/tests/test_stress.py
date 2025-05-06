"""
test_stress.py
------------
Stress tests to verify system behavior under heavy load.
"""

import time
import threading
import multiprocessing
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from data.tests import BaseTestCase
from data.execution.order_execution import OrderExecution
from data.data.data_preprocessing import preprocess_pipeline
from data.ai_models.model_loader import ModelLoader
from data.monitoring.system_monitor import SystemMonitor

class TestStress(BaseTestCase):
    """Test system behavior under stress."""

    def setUp(self):
        """Initialize stress test resources."""
        super().setUp()
        self.monitor = SystemMonitor()
        self.execution = OrderExecution()
        self.model_loader = ModelLoader()
        
        # Initialize test data
        self.test_data = self.generate_test_data(periods=1000)
        self.models = self.model_loader.load_available_models()

    @pytest.mark.stress
    def test_concurrent_order_execution(self):
        """Test system under concurrent order execution load."""
        num_threads = 10
        orders_per_thread = 100
        max_time = 30  # 30 seconds maximum
        
        def execute_orders(orders: List[Dict[str, Any]], results: List[Dict[str, Any]]):
            """Execute a batch of orders."""
            for order in orders:
                try:
                    result = self.execution.send_order(**order)
                    results.append(result)
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
        
        # Generate test orders
        test_orders = [
            {
                "symbol": "BTCUSDT",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "type": "MARKET",
                "quantity": 0.001
            }
            for i in range(num_threads * orders_per_thread)
        ]
        
        # Split orders into batches
        order_batches = np.array_split(test_orders, num_threads)
        results = []
        
        # Execute orders concurrently
        start_time = time.time()
        threads = []
        
        for batch in order_batches:
            thread = threading.Thread(
                target=execute_orders,
                args=(batch, results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        execution_time = time.time() - start_time
        
        # Verify results
        self.assertLess(execution_time, max_time)
        self.assertEqual(len(results), len(test_orders))
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        self.assertGreater(success_rate, 0.95)  # 95% success rate minimum

    @pytest.mark.stress
    def test_concurrent_data_processing(self):
        """Test concurrent data processing performance."""
        num_processes = multiprocessing.cpu_count()
        chunks_per_process = 5
        total_chunks = num_processes * chunks_per_process
        
        # Split data into chunks
        chunk_size = len(self.test_data) // total_chunks
        data_chunks = [
            self.test_data[i:i + chunk_size]
            for i in range(0, len(self.test_data), chunk_size)
        ]
        
        def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
            """Process a single data chunk."""
            return preprocess_pipeline(
                df,
                price_col="close",
                fill_method="ffill",
                outlier_threshold=2.5
            )
        
        start_time = time.time()
        processed_chunks = []
        
        # Process data concurrently
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            processed_chunks = list(executor.map(process_chunk, data_chunks))
        
        processing_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(len(processed_chunks), len(data_chunks))
        for chunk in processed_chunks:
            self.assertFalse(chunk.isnull().any().any())
        
        # Verify processing speed
        max_processing_time = 5.0  # 5 seconds maximum
        self.assertLess(
            processing_time,
            max_processing_time,
            f"Concurrent processing took {processing_time:.2f}s"
        )

    @pytest.mark.stress
    def test_model_inference_under_load(self):
        """Test model inference performance under load."""
        num_threads = 4
        predictions_per_thread = 1000
        
        def run_predictions(
            model,
            num_predictions: int,
            results: List[float],
            latencies: List[float]
        ):
            """Run model predictions."""
            for _ in range(num_predictions):
                features = pd.DataFrame(
                    np.random.random((1, 10)),
                    columns=[f"feature_{i}" for i in range(10)]
                )
                
                start_time = time.time()
                prediction = model.predict(features)
                latency = time.time() - start_time
                
                results.append(prediction[0])
                latencies.append(latency)
        
        for model_name, model in self.models.items():
            if not hasattr(model, "predict"):
                continue
                
            results = []
            latencies = []
            threads = []
            
            # Run predictions concurrently
            start_time = time.time()
            
            for _ in range(num_threads):
                thread = threading.Thread(
                    target=run_predictions,
                    args=(model, predictions_per_thread, results, latencies)
                )
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Verify results
            total_predictions = num_threads * predictions_per_thread
            self.assertEqual(len(results), total_predictions)
            
            # Analyze latencies
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            max_latency = max(latencies)
            
            # Verify performance metrics
            self.assertLess(avg_latency, 0.01)  # 10ms average
            self.assertLess(p95_latency, 0.02)  # 20ms 95th percentile
            self.assertLess(max_latency, 0.05)  # 50ms maximum
            
            throughput = total_predictions / total_time
            min_throughput = 1000  # 1000 predictions per second minimum
            self.assertGreater(throughput, min_throughput)

    @pytest.mark.stress
    def test_system_stability_under_load(self):
        """Test overall system stability under heavy load."""
        duration = 60  # 1 minute test
        interval = 1   # Check metrics every second
        
        def generate_load():
            """Generate system load."""
            while time.time() - start_time < duration:
                # Process data
                _ = preprocess_pipeline(self.test_data)
                
                # Run model inference
                for model in self.models.values():
                    if hasattr(model, "predict"):
                        features = pd.DataFrame(
                            np.random.random((100, 10)),
                            columns=[f"feature_{i}" for i in range(10)]
                        )
                        _ = model.predict(features)
                
                # Execute orders
                _ = self.execution.send_order(
                    symbol="BTCUSDT",
                    side="BUY",
                    type="MARKET",
                    quantity=0.001
                )
        
        # Start load generation in background
        start_time = time.time()
        load_thread = threading.Thread(target=generate_load)
        load_thread.start()
        
        # Monitor system metrics
        metrics_history = []
        while time.time() - start_time < duration:
            metrics = self.monitor.collect_metrics()
            metrics_history.append(metrics)
            time.sleep(interval)
        
        load_thread.join()
        
        # Analyze metrics
        cpu_usage = [m["cpu_usage"] for m in metrics_history]
        memory_usage = [m["memory_usage"] for m in metrics_history]
        
        # Verify system stability
        max_cpu = max(cpu_usage)
        max_memory = max(memory_usage)
        
        self.assertLess(max_cpu, 80)  # CPU usage below 80%
        self.assertLess(max_memory, 80)  # Memory usage below 80%
        
        # Check for metric stability
        cpu_std = np.std(cpu_usage)
        memory_std = np.std(memory_usage)
        
        self.assertLess(cpu_std, 20)  # CPU usage should be stable
        self.assertLess(memory_std, 20)  # Memory usage should be stable

    @pytest.mark.stress
    def test_error_handling_under_load(self):
        """Test error handling and recovery under stress."""
        num_threads = 10
        operations_per_thread = 100
        error_injection_rate = 0.1  # 10% of operations will have errors
        
        def run_operations(results: List[Dict[str, Any]]):
            """Run mixed operations with error injection."""
            for _ in range(operations_per_thread):
                try:
                    if np.random.random() < error_injection_rate:
                        # Inject error
                        raise ValueError("Simulated error")
                    
                    # Random operation
                    op_type = np.random.choice([
                        "data_processing",
                        "model_inference",
                        "order_execution"
                    ])
                    
                    if op_type == "data_processing":
                        _ = preprocess_pipeline(
                            self.test_data.sample(n=100)
                        )
                    elif op_type == "model_inference":
                        model = np.random.choice(list(self.models.values()))
                        if hasattr(model, "predict"):
                            features = pd.DataFrame(
                                np.random.random((1, 10)),
                                columns=[f"feature_{i}" for i in range(10)]
                            )
                            _ = model.predict(features)
                    else:  # order_execution
                        _ = self.execution.send_order(
                            symbol="BTCUSDT",
                            side="BUY",
                            type="MARKET",
                            quantity=0.001
                        )
                    
                    results.append({"success": True})
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e)
                    })
        
        # Run concurrent operations
        results = []
        threads = []
        
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(
                target=run_operations,
                args=(results,)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Analyze results
        total_operations = num_threads * operations_per_thread
        succeeded = sum(1 for r in results if r["success"])
        failed = sum(1 for r in results if not r["success"])
        
        # Verify error handling
        self.assertEqual(len(results), total_operations)
        self.assertGreater(
            succeeded / total_operations,
            0.85  # At least 85% success rate
        )
        self.assertLess(
            failed / total_operations,
            0.15  # Less than 15% failure rate
        )