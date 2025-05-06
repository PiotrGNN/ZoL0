"""Load testing scenarios for high-throughput and stress testing"""
import pytest
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
from datetime import datetime, timedelta

@pytest.mark.load
class TestLoadScenarios:
    """Load and stress testing scenarios"""
    
    @pytest.fixture(autouse=True)
    def setup(self, test_credentials):
        """Setup test environment"""
        from python_libs.bybit_v5_connector import BybitV5Connector
        self.connector = BybitV5Connector(**test_credentials)
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        self.test_duration = 60  # seconds
        
    @pytest.mark.load
    async def test_concurrent_trades(self):
        """Test system under concurrent trading load"""
        max_concurrent = 50
        trades_completed = 0
        errors = []
        
        async def execute_trade(symbol: str) -> bool:
            try:
                # Place test order
                order = await self.connector.place_order_async(
                    symbol=symbol,
                    side="Buy",
                    order_type="Limit",
                    qty=0.001,
                    price=await self.get_safe_price(symbol)
                )
                
                if order.get("order_id"):
                    await self.connector.cancel_order_async(order["order_id"])
                return True
            except Exception as e:
                errors.append(str(e))
                return False
                
        async def get_safe_price(symbol: str) -> float:
            """Get price with safety margin"""
            market_data = await self.connector.get_market_data_async(symbol)
            return float(market_data["last_price"]) * 0.95  # 5% below market
                
        # Execute concurrent trades
        start_time = datetime.now()
        tasks = []
        async with aiohttp.ClientSession() as session:
            while (datetime.now() - start_time).seconds < self.test_duration:
                while len(tasks) < max_concurrent:
                    symbol = np.random.choice(self.symbols)
                    task = asyncio.create_task(execute_trade(symbol))
                    tasks.append(task)
                    
                done, pending = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    if task.result():
                        trades_completed += 1
                tasks = list(pending)
                
        success_rate = trades_completed / (trades_completed + len(errors))
        assert success_rate >= 0.95, f"High error rate: {1-success_rate:.1%} errors"
        assert trades_completed >= 1000, f"Low throughput: {trades_completed} trades"
        
    @pytest.mark.load
    def test_high_frequency_data(self):
        """Test high-frequency market data processing"""
        with ProcessPoolExecutor() as executor:
            # Simulate multiple data streams
            futures = []
            for symbol in self.symbols:
                futures.append(
                    executor.submit(
                        self._process_market_data_stream,
                        symbol,
                        self.test_duration
                    )
                )
                
            # Collect results
            total_updates = 0
            errors = []
            for future in futures:
                result = future.result()
                total_updates += result["updates"]
                if result["errors"]:
                    errors.extend(result["errors"])
                    
        updates_per_sec = total_updates / self.test_duration
        assert updates_per_sec >= 100, f"Low update rate: {updates_per_sec:.1f} updates/sec"
        assert len(errors) == 0, f"Errors during data processing: {errors}"
        
    def _process_market_data_stream(self, symbol: str, duration: int) -> Dict[str, Any]:
        """Process market data stream for a symbol"""
        from data.data.data_preprocessing import preprocess_pipeline
        
        updates = 0
        errors = []
        start_time = datetime.now()
        data_buffer = []
        
        while (datetime.now() - start_time).seconds < duration:
            try:
                # Get latest data
                data = self.connector.get_market_data(symbol)
                data_buffer.append(data)
                updates += 1
                
                # Process in batches
                if len(data_buffer) >= 100:
                    df = pd.DataFrame(data_buffer)
                    preprocess_pipeline(df.copy())
                    data_buffer = []
                    
            except Exception as e:
                errors.append(str(e))
                
        return {
            "updates": updates,
            "errors": errors
        }
        
    @pytest.mark.load
    def test_multiple_symbols(self):
        """Test handling multiple trading symbols simultaneously"""
        with ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
            futures = []
            for symbol in self.symbols:
                futures.append(
                    executor.submit(
                        self._run_symbol_operations,
                        symbol,
                        self.test_duration
                    )
                )
                
            # Collect results
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
                    
        # Verify results
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        assert success_rate >= 0.95, f"Symbol operations failed: {1-success_rate:.1%} error rate"
        
    def _run_symbol_operations(self, symbol: str, duration: int) -> Dict[str, Any]:
        """Run various operations for a symbol"""
        try:
            start_time = datetime.now()
            operations_count = 0
            
            while (datetime.now() - start_time).seconds < duration:
                # Get market data
                market_data = self.connector.get_market_data(symbol)
                assert market_data is not None
                
                # Get order book
                order_book = self.connector.get_order_book(symbol)
                assert order_book is not None
                
                # Get recent trades
                trades = self.connector.get_recent_trades(symbol)
                assert trades is not None
                
                operations_count += 3
                
            return {
                "success": True,
                "operations": operations_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    @pytest.mark.load
    def test_peak_load(self):
        """Test system behavior under peak load conditions"""
        # Simulate peak load with multiple components
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # 1. High-frequency data processing
            futures.append(executor.submit(self._run_peak_data_processing))
            
            # 2. Concurrent trading operations
            futures.append(executor.submit(self._run_peak_trading))
            
            # 3. Model inference
            futures.append(executor.submit(self._run_peak_inference))
            
            # Collect results
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
                    
        # Verify system stability
        assert all(r["success"] for r in results), "System unstable under peak load"
        
    def _run_peak_data_processing(self) -> Dict[str, Any]:
        """Run peak data processing load"""
        try:
            from data.data.data_preprocessing import preprocess_pipeline
            
            processed_count = 0
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < self.test_duration:
                # Generate large dataset
                dates = pd.date_range("2023-01-01", periods=10000, freq="1min")
                df = pd.DataFrame({
                    "timestamp": dates,
                    "open": np.random.uniform(100, 110, 10000),
                    "high": np.random.uniform(110, 120, 10000),
                    "low": np.random.uniform(90, 100, 10000),
                    "close": np.random.uniform(100, 115, 10000),
                    "volume": np.random.randint(1000, 2000, 10000),
                })
                
                # Process data
                preprocess_pipeline(df.copy())
                processed_count += len(df)
                
            return {
                "success": True,
                "processed_rows": processed_count
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _run_peak_trading(self) -> Dict[str, Any]:
        """Run peak trading operations"""
        try:
            trades_completed = 0
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < self.test_duration:
                # Place multiple orders rapidly
                for symbol in self.symbols:
                    market_data = self.connector.get_market_data(symbol)
                    price = float(market_data["last_price"]) * 0.95
                    
                    order = self.connector.place_order(
                        symbol=symbol,
                        side="Buy",
                        order_type="Limit",
                        qty=0.001,
                        price=price
                    )
                    
                    if order.get("order_id"):
                        self.connector.cancel_order(order["order_id"])
                        trades_completed += 1
                        
            return {
                "success": True,
                "trades_completed": trades_completed
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _run_peak_inference(self) -> Dict[str, Any]:
        """Run peak model inference load"""
        try:
            from ai_models.model_recognition import ModelRecognizer
            from ai_models.sentiment_ai import SentimentAnalyzer
            
            recognizer = ModelRecognizer()
            analyzer = SentimentAnalyzer()
            inferences = 0
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < self.test_duration:
                # Generate test data
                dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
                df = pd.DataFrame({
                    "timestamp": dates,
                    "close": np.random.uniform(100, 115, 1000)
                })
                
                # Run pattern recognition
                recognizer.identify_model_type(df)
                
                # Run sentiment analysis
                analyzer.analyze("BTC showing strong momentum with high volume")
                
                inferences += 2
                
            return {
                "success": True,
                "inferences": inferences
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])