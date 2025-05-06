"""
test_integration.py
----------------
Integration tests for component interactions.
"""

import os
import time
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from data.tests import BaseTestCase
from data.execution.order_execution import OrderExecution
from data.execution.trade_executor import TradeExecutor
from data.risk_management.position_sizing import calculate_position_size
from data.risk_management.stop_loss_manager import calculate_stop_loss
from data.data.data_preprocessing import preprocess_pipeline
from data.ai_models.model_loader import ModelLoader
from data.data.historical_data import HistoricalDataManager

class TestSystemIntegration(BaseTestCase):
    """Test interactions between system components."""

    def setUp(self):
        """Initialize test resources."""
        super().setUp()
        # Create test paths
        self.test_data_dir = Path(self.create_temp_file()).parent
        self.db_path = self.test_data_dir / "test.db"
        self.csv_path = self.test_data_dir / "market_data.csv"
        
        # Generate and save test data
        self.market_data = self.generate_test_data(periods=200)
        self.market_data.to_csv(self.csv_path, index=False)
        
        # Initialize components
        self.historical_data = HistoricalDataManager(
            csv_path=str(self.csv_path),
            db_path=str(self.db_path)
        )
        self.model_loader = ModelLoader()
        self.order_execution = OrderExecution()
        self.trade_executor = TradeExecutor(self.order_execution)

    @pytest.mark.integration
    def test_data_pipeline_to_model(self):
        """Test data preprocessing to model prediction pipeline."""
        try:
            # Load and preprocess data
            raw_data = self.historical_data.load_from_csv()
            processed_data = preprocess_pipeline(
                raw_data,
                price_col="close",
                fill_method="ffill",
                outlier_threshold=2.5
            )
            
            # Load model and make predictions
            model = self.model_loader.load_model("pattern_recognition")
            predictions = model.predict(processed_data)
            
            # Verify results
            self.assertEqual(len(predictions), len(processed_data))
            self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
            self.assertTrue(all(-1 <= p <= 1 for p in predictions))
            
            # Test data persistence
            self.historical_data.save_predictions(predictions, "pattern_signals")
            loaded_preds = self.historical_data.load_predictions("pattern_signals")
            np.testing.assert_array_almost_equal(predictions, loaded_preds)
            
        except Exception as e:
            self.fail(f"Data pipeline to model integration failed: {str(e)}")

    @pytest.mark.integration
    def test_model_to_trading_pipeline(self):
        """Test model prediction to trade execution pipeline."""
        try:
            # Load model and get predictions
            model = self.model_loader.load_model("trading_signals")
            data = preprocess_pipeline(self.market_data)
            predictions = model.predict(data)
            
            # Convert predictions to trade signals
            signals = self._convert_predictions_to_signals(predictions)
            
            # Execute trades based on signals
            account_balance = 10000.0
            executed_trades = []
            
            for idx, signal in enumerate(signals):
                if signal != 0:  # 1 for buy, -1 for sell
                    current_price = data.iloc[idx]["close"]
                    stop_loss = calculate_stop_loss(
                        entry_price=current_price,
                        risk_percent=0.02,
                        position_type="long" if signal > 0 else "short"
                    )
                    
                    # Calculate position size
                    position_size = calculate_position_size(
                        account_balance=account_balance,
                        risk_per_trade=0.02,
                        entry_price=current_price,
                        stop_loss_price=stop_loss
                    )
                    
                    # Execute trade
                    trade = self.trade_executor.execute_trade(
                        symbol="BTCUSDT",
                        side="BUY" if signal > 0 else "SELL",
                        quantity=position_size,
                        stop_loss=stop_loss
                    )
                    executed_trades.append(trade)
            
            # Verify trade execution
            self.assertGreater(len(executed_trades), 0)
            for trade in executed_trades:
                self.assertTrue(trade["success"])
                self.assertIn("orderId", trade)
                self.assertIn("entry_price", trade)
                
        except Exception as e:
            self.fail(f"Model to trading integration failed: {str(e)}")

    @pytest.mark.integration
    def test_risk_management_integration(self):
        """Test risk management integration with trading."""
        try:
            initial_balance = 10000.0
            risk_per_trade = 0.02
            entry_price = 50000.0
            
            # Calculate position size with risk management
            stop_loss = calculate_stop_loss(
                entry_price=entry_price,
                risk_percent=risk_per_trade,
                position_type="long"
            )
            
            position_size = calculate_position_size(
                account_balance=initial_balance,
                risk_per_trade=risk_per_trade,
                entry_price=entry_price,
                stop_loss_price=stop_loss
            )
            
            # Verify position sizing
            max_loss = (entry_price - stop_loss) * position_size
            self.assertLessEqual(
                max_loss,
                initial_balance * risk_per_trade,
                "Position size exceeds risk limit"
            )
            
            # Test trade execution with risk parameters
            trade = self.trade_executor.execute_trade_with_position_sizing(
                symbol="BTCUSDT",
                side="BUY",
                account_balance=initial_balance,
                risk_per_trade=risk_per_trade,
                stop_loss_pct=0.02
            )
            
            self.assertTrue(trade["success"])
            self.assertIn("stop_loss_price", trade)
            self.assertLess(
                trade["stop_loss_price"],
                trade["entry_price"],
                "Stop loss should be below entry for long position"
            )
            
        except Exception as e:
            self.fail(f"Risk management integration failed: {str(e)}")

    @pytest.mark.integration
    def test_historical_data_integration(self):
        """Test historical data management integration."""
        try:
            # Test data import
            self.historical_data.load_to_database(self.market_data)
            
            # Query recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            recent_data = self.historical_data.query_database(
                f"""
                SELECT *
                FROM historical_data
                WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY timestamp DESC
                """
            )
            
            # Process queried data
            processed_data = preprocess_pipeline(recent_data)
            
            # Make predictions
            model = self.model_loader.load_model("market_prediction")
            predictions = model.predict(processed_data)
            
            # Store predictions
            self.historical_data.save_predictions(
                predictions,
                model_name="market_prediction"
            )
            
            # Verify data retrieval
            stored_preds = self.historical_data.load_predictions(
                model_name="market_prediction"
            )
            np.testing.assert_array_almost_equal(predictions, stored_preds)
            
        except Exception as e:
            self.fail(f"Historical data integration failed: {str(e)}")

    @pytest.mark.integration
    def test_full_trading_cycle(self):
        """Test complete trading cycle integration."""
        try:
            # 1. Load and process historical data
            raw_data = self.historical_data.load_from_csv()
            processed_data = preprocess_pipeline(raw_data)
            
            # 2. Generate trading signals
            model = self.model_loader.load_model("trading_strategy")
            signals = model.predict(processed_data)
            
            # 3. Execute trades with risk management
            account_balance = 10000.0
            executed_trades = []
            
            for idx, signal in enumerate(signals[-5:]):  # Test last 5 signals
                if signal != 0:
                    # Calculate risk parameters
                    current_price = processed_data.iloc[idx]["close"]
                    stop_loss = calculate_stop_loss(
                        entry_price=current_price,
                        risk_percent=0.02,
                        position_type="long" if signal > 0 else "short"
                    )
                    
                    # Calculate position size
                    position_size = calculate_position_size(
                        account_balance=account_balance,
                        risk_per_trade=0.02,
                        entry_price=current_price,
                        stop_loss_price=stop_loss
                    )
                    
                    # Execute trade
                    trade = self.trade_executor.execute_trade(
                        symbol="BTCUSDT",
                        side="BUY" if signal > 0 else "SELL",
                        quantity=position_size,
                        stop_loss=stop_loss,
                        take_profit=current_price * (1.02 if signal > 0 else 0.98)
                    )
                    executed_trades.append(trade)
                    
                    # Update account balance based on trade result
                    if trade["success"]:
                        profit_loss = trade.get("realized_pnl", 0)
                        account_balance += profit_loss
            
            # Verify trading cycle results
            self.assertGreater(len(executed_trades), 0)
            for trade in executed_trades:
                self.assertTrue(trade["success"])
                self.assertIn("orderId", trade)
                self.assertIn("entry_price", trade)
                self.assertIn("stop_loss_price", trade)
            
            # Verify account balance maintained risk limits
            self.assertGreater(
                account_balance,
                initial_balance * 0.9,  # No more than 10% drawdown
                "Trading cycle exceeded maximum drawdown"
            )
            
        except Exception as e:
            self.fail(f"Full trading cycle integration failed: {str(e)}")

    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        # Clean up database
        if self.db_path.exists():
            self.db_path.unlink()
        # Clean up CSV file
        if self.csv_path.exists():
            self.csv_path.unlink()

    def _convert_predictions_to_signals(self, predictions: np.ndarray) -> np.ndarray:
        """Convert model predictions to trading signals."""
        signals = np.zeros_like(predictions)
        signals[predictions > 0.5] = 1    # Buy signals
        signals[predictions < -0.5] = -1  # Sell signals
        return signals

"""Integration tests and dependency health checks."""

import pytest
import subprocess
import pkg_resources
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path
from .base_test import BaseTestCase

class TestIntegration(BaseTestCase):
    """Integration test suite."""
    
    def setUp(self):
        """Set up integration test resources."""
        super().setUp()
        self.data = self.generate_test_data(periods=100)
        
    def test_dependency_versions(self):
        """Verify all dependencies are at required versions."""
        required = {
            'numpy': '>=1.20.0',
            'pandas': '>=1.3.0',
            'scikit-learn': '>=1.0.0',
            'pytest': '>=6.0.0',
            'requests': '>=2.25.0'
        }
        
        for package, version_req in required.items():
            try:
                pkg_version = pkg_resources.get_distribution(package).version
                self.assertTrue(
                    pkg_resources.parse_version(pkg_version) >= 
                    pkg_resources.parse_version(version_req.replace('>=', '')),
                    f"{package} version {pkg_version} does not meet requirement {version_req}"
                )
            except pkg_resources.DistributionNotFound:
                self.fail(f"Required package {package} not installed")
                
    def test_dependency_conflicts(self):
        """Check for dependency conflicts."""
        import pipdeptree
        from io import StringIO
        import sys
        
        # Capture pipdeptree output
        stdout = StringIO()
        sys.stdout = stdout
        pipdeptree.main()
        sys.stdout = sys.__stdout__
        
        # Check for conflicts
        output = stdout.getvalue()
        self.assertNotIn("! Conflicts", output)
        
    @pytest.mark.integration
    def test_model_pipeline_integration(self):
        """Test full model pipeline integration."""
        from ai_models.model_loader import ModelLoader
        from ai_models.model_training import ModelTrainer
        from ai_models.model_evaluation import ModelEvaluator
        
        # 1. Load and prepare data
        loader = ModelLoader()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        
        # 2. Train model
        model = trainer.train_model(
            features=self.data[self.sample_features],
            target=self.data["target"]
        )
        self.assertIsNotNone(model)
        
        # 3. Evaluate model
        metrics = evaluator.evaluate_model(
            model,
            self.data[self.sample_features],
            self.data["target"]
        )
        
        required_metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
            
        # 4. Save and reload model
        model_path = self.create_temp_file(suffix=".pkl")
        trainer.save_model(model, model_path)
        
        loaded_model = loader.load_model(model_path)
        self.assertIsNotNone(loaded_model)
        
        # 5. Verify predictions match
        orig_preds = model.predict(self.data[self.sample_features])
        loaded_preds = loaded_model.predict(self.data[self.sample_features])
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)
        
    @pytest.mark.integration
    def test_api_integration(self):
        """Test API endpoint integration."""
        from dashboard_api import app
        client = app.test_client()
        
        # Test data ingestion endpoint
        response = client.post(
            "/api/data/ingest",
            json=self.data.to_dict(orient="records")
        )
        self.assertEqual(response.status_code, 200)
        
        # Test model prediction endpoint
        response = client.post(
            "/api/models/predict",
            json={"data": self.data.to_dict(orient="records")}
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn("predictions", result)
        
        # Test model metrics endpoint
        response = client.get("/api/models/metrics")
        self.assertEqual(response.status_code, 200)
        metrics = response.get_json()
        self.assertIn("accuracy", metrics)
        
    @pytest.mark.integration
    def test_database_integration(self):
        """Test database integration."""
        import sqlite3
        from contextlib import closing
        
        # Create test database
        db_path = self.create_temp_file(suffix=".db")
        
        with closing(sqlite3.connect(db_path)) as conn:
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_data (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    value REAL
                )
            """)
            
            # Insert test data
            test_data = [
                (1, "2025-01-01", 100.0),
                (2, "2025-01-02", 101.0)
            ]
            cursor.executemany(
                "INSERT INTO test_data VALUES (?, ?, ?)",
                test_data
            )
            conn.commit()
            
            # Verify data
            cursor.execute("SELECT * FROM test_data")
            rows = cursor.fetchall()
            self.assertEqual(len(rows), 2)
            
    def test_logging_integration(self):
        """Test logging system integration."""
        log_file = self.create_temp_file(suffix=".log")
        
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        
        # Generate some log messages
        logging.info("Test info message")
        logging.warning("Test warning message")
        logging.error("Test error message")
        
        # Verify log file
        with open(log_file) as f:
            content = f.read()
            self.assertIn("Test info message", content)
            self.assertIn("Test warning message", content)
            self.assertIn("Test error message", content)
            
    @pytest.mark.integration
    def test_cache_integration(self):
        """Test caching system integration."""
        from ai_models.model_utils import cache_result
        
        @cache_result(timeout=60)
        def expensive_calculation(x: int) -> int:
            self.calculation_count += 1
            return x * x
        
        self.calculation_count = 0
        
        # First call should calculate
        result1 = expensive_calculation(5)
        self.assertEqual(result1, 25)
        self.assertEqual(self.calculation_count, 1)
        
        # Second call should use cache
        result2 = expensive_calculation(5)
        self.assertEqual(result2, 25)
        self.assertEqual(self.calculation_count, 1)  # Still 1
        
        # Different input should calculate
        result3 = expensive_calculation(6)
        self.assertEqual(result3, 36)
        self.assertEqual(self.calculation_count, 2)