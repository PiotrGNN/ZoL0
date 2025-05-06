"""Performance benchmarks and optimization tests."""

import pytest
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import concurrent.futures
from datetime import datetime, timedelta
from .base_test import BaseTestCase

class TestPerformanceBenchmarks(BaseTestCase):
    """Performance benchmark test suite."""
    
    def setUp(self):
        """Set up performance test resources."""
        super().setUp()
        self.large_df = self.generate_test_data(periods=10000)
        np.random.seed(42)
        self.prediction_latencies = []
        
    @pytest.mark.benchmark
    def test_model_inference_latency(self):
        """Benchmark model inference latency."""
        from ai_models.model_loader import ModelLoader
        loader = ModelLoader()
        model = loader.load_model("trading_strategy")
        
        batch_sizes = [1, 10, 100, 1000]
        samples = 100
        
        for batch_size in batch_sizes:
            latencies = []
            for _ in range(samples):
                data = self.large_df.sample(batch_size)
                start_time = time.perf_counter()
                _ = model.predict(data)
                latency = time.perf_counter() - start_time
                latencies.append(latency)
                
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            self.assertLess(
                avg_latency,
                0.1,  # 100ms max average latency
                f"Average inference latency too high for batch size {batch_size}"
            )
            self.assertLess(
                p99_latency,
                0.5,  # 500ms max P99 latency
                f"P99 inference latency too high for batch size {batch_size}"
            )
            
    @pytest.mark.benchmark
    def test_concurrent_prediction_throughput(self):
        """Test prediction throughput under concurrent load."""
        from ai_models.model_loader import ModelLoader
        loader = ModelLoader()
        model = loader.load_model("trading_strategy")
        
        num_requests = 100
        concurrent_users = [1, 5, 10, 20]
        batch_size = 10
        
        for num_users in concurrent_users:
            start_time = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = []
                for _ in range(num_requests):
                    data = self.large_df.sample(batch_size)
                    futures.append(executor.submit(model.predict, data))
                
                # Wait for all predictions
                results = [f.result() for f in futures]
            
            total_time = time.perf_counter() - start_time
            throughput = num_requests / total_time
            
            self.assertGreater(
                throughput,
                50.0,  # Minimum 50 predictions per second
                f"Throughput too low with {num_users} concurrent users"
            )
            
    @pytest.mark.benchmark
    def test_data_processing_performance(self):
        """Benchmark data processing pipeline performance."""
        from ai_models.model_utils import preprocess_pipeline
        
        data_sizes = [1000, 5000, 10000]
        max_processing_times = {
            1000: 0.1,   # 100ms for 1K records
            5000: 0.3,   # 300ms for 5K records
            10000: 0.6   # 600ms for 10K records
        }
        
        for size in data_sizes:
            test_data = self.large_df[:size].copy()
            
            start_time = time.perf_counter()
            processed_data = preprocess_pipeline(
                test_data,
                fill_method="ffill",
                normalize=True,
                remove_outliers=True
            )
            processing_time = time.perf_counter() - start_time
            
            self.assertLess(
                processing_time,
                max_processing_times[size],
                f"Processing {size} records took too long"
            )
            
            # Verify output quality
            self.assertFalse(processed_data.isnull().any().any())
            self.assertTrue((processed_data.select_dtypes(include=[np.number]) != 0).any().any())
            
    @pytest.mark.benchmark
    def test_memory_usage_optimization(self):
        """Test memory usage optimization."""
        import psutil
        import os
        
        def get_process_memory():
            """Get current process memory usage."""
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        
        # Baseline memory
        baseline_memory = get_process_memory()
        
        # Test memory usage with large dataset
        from ai_models.model_loader import ModelLoader
        loader = ModelLoader()
        model = loader.load_model("trading_strategy")
        
        # Process in chunks to optimize memory
        chunk_size = 1000
        num_chunks = len(self.large_df) // chunk_size
        
        max_memory_increase = 100  # Max 100MB increase
        
        for i in range(num_chunks):
            chunk = self.large_df[i*chunk_size:(i+1)*chunk_size]
            _ = model.predict(chunk)
            
            current_memory = get_process_memory()
            memory_increase = current_memory - baseline_memory
            
            self.assertLess(
                memory_increase,
                max_memory_increase,
                f"Memory usage increased by {memory_increase:.1f}MB, exceeding limit"
            )
            
    @pytest.mark.benchmark
    def test_real_time_performance(self):
        """Test real-time processing performance."""
        from ai_models.model_loader import ModelLoader
        loader = ModelLoader()
        model = loader.load_model("trading_strategy")
        
        # Simulate real-time data feed
        window_size = 100
        update_interval = 0.1  # 100ms
        num_updates = 100
        
        latencies = []
        start_time = time.perf_counter()
        
        for i in range(num_updates):
            # Simulate new data arrival
            new_data = self.generate_test_data(periods=1)
            window_data = self.large_df[i:i+window_size]
            
            # Process and predict
            process_start = time.perf_counter()
            _ = model.predict(window_data)
            latency = time.perf_counter() - process_start
            latencies.append(latency)
            
            # Ensure we maintain update interval
            elapsed = time.perf_counter() - start_time
            if elapsed < (i + 1) * update_interval:
                time.sleep((i + 1) * update_interval - elapsed)
        
        # Verify real-time performance
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        missed_updates = sum(1 for lat in latencies if lat > update_interval)
        
        self.assertLess(avg_latency, update_interval * 0.5)
        self.assertLess(max_latency, update_interval * 0.9)
        self.assertEqual(missed_updates, 0, "Missed real-time updates")