"""
test_data_processing.py
-----------------------
Tests for data preprocessing and historical data management modules.
"""

import logging
import os
import sqlite3
from typing import Any, List, Dict
import numpy as np
import pandas as pd

from data.tests import MockTestCase
from data.data.data_preprocessing import (
    clean_data,
    compute_log_returns,
    detect_outliers,
    preprocess_pipeline,
    winsorize_series,
)
from data.data.historical_data import HistoricalDataManager

class TestDataProcessing(MockTestCase):
    """Test data preprocessing functionality."""

    def setUp(self):
        """Initialize test resources."""
        super().setUp()
        self.df = self.generate_test_data()
        # Add test cases with known patterns
        self.df.loc[10, "close"] = np.nan  # Missing value
        self.df.loc[20, "close"] = self.df["close"].max() * 2.0  # Outlier
        self.df.loc[30, "volume"] = 0  # Zero volume
        self.df.loc[40, ["high", "low", "close"]] = np.nan  # Multiple missing values
        
        # Create severe outliers for testing
        self.df.loc[50, "close"] = self.df["close"].mean() + self.df["close"].std() * 5
        self.df.loc[60, "close"] = self.df["close"].mean() - self.df["close"].std() * 5
        
        # Add some price jumps for testing
        self.df.loc[70:75, "close"] *= 1.5  # Price surge
        self.df.loc[80:85, "close"] *= 0.7  # Price drop

        # Notify test setup
        self.mock_notifier.send_notification("Test data prepared", "debug")

    def test_clean_data(self):
        """Test data cleaning with different methods."""
        methods = ["ffill", "bfill", "median", "linear"]
        
        for method in methods:
            with self.subTest(fill_method=method):
                try:
                    df_clean = clean_data(self.df, fill_method=method)
                    self.assertFalse(
                        df_clean.isnull().values.any(),
                        f"Data should be clean with fill_method='{method}'"
                    )
                    self.mock_notifier.send_notification(
                        f"Clean data test passed for method: {method}", 
                        "info"
                    )
                    
                    # Verify data integrity
                    self.assertTrue((df_clean["high"] >= df_clean["low"]).all())
                    self.assertTrue((df_clean["volume"] >= 0).all())
                    
                    # Check interpolation results
                    if method == "linear":
                        self.assertTrue(
                            df_clean.loc[40, "close"] >= df_clean.loc[40, "low"],
                            "Interpolated close should be >= low"
                        )
                        
                except Exception as e:
                    self.mock_notifier.send_alert(
                        f"Clean data test failed for method {method}: {str(e)}", 
                        "error"
                    )
                    raise

        # Test volume handling
        df_clean = clean_data(self.df, fill_method="median", handle_zero_volume=True)
        self.assertTrue(
            (df_clean["volume"] > 0).all(),
            "All volume values should be positive"
        )
        
        # Test error handling
        with self.assertRaises(ValueError):
            clean_data(self.df, fill_method="invalid_method")

    def test_compute_log_returns(self) -> None:
        """Test log return calculations."""
        try:
            df_clean = clean_data(self.df, fill_method="median")
            
            # Test for close prices
            log_returns = compute_log_returns(df_clean, price_col="close")
            self.assertIsInstance(log_returns, pd.Series)
            self.assertEqual(len(log_returns), len(df_clean) - 1)
            self.assertTrue(np.isfinite(log_returns).all(), "Returns should be finite")

            # Test different price columns
            for col in ["open", "high", "low"]:
                with self.subTest(price_column=col):
                    returns = compute_log_returns(df_clean, price_col=col)
                    self.assertEqual(len(returns), len(df_clean) - 1)
                    # Verify log returns are reasonable
                    self.assertTrue((-1 < returns).all(), "Log returns should be > -1")
                    self.assertTrue(
                        (abs(returns) < 1).mean() > 0.9,
                        "Most log returns should be within ±1"
                    )

            # Test error handling
            with self.assertRaises(KeyError):
                compute_log_returns(df_clean, price_col="nonexistent")
                
            self.mock_notifier.send_notification("Log returns tests passed", "info")
        except Exception as e:
            self.mock_notifier.send_alert(f"Log returns test failed: {str(e)}", "error")
            raise

    def test_detect_outliers(self) -> None:
        """Test outlier detection."""
        try:
            df_clean = clean_data(self.df, fill_method="median")

            # Test with different thresholds
            thresholds = [1.5, 2.0, 2.5, 3.0]
            for threshold in thresholds:
                with self.subTest(threshold=threshold):
                    df_out = detect_outliers(df_clean, column="close", threshold=threshold)
                    self.assertIn("is_outlier", df_out.columns)
                    self.assertTrue(isinstance(df_out["is_outlier"].iloc[20], bool))
                    
                    # Verify severe outliers are detected
                    self.assertTrue(
                        df_out.loc[50, "is_outlier"],
                        f"Severe positive outlier not detected with threshold {threshold}"
                    )
                    self.assertTrue(
                        df_out.loc[60, "is_outlier"],
                        f"Severe negative outlier not detected with threshold {threshold}"
                    )
                    
                    # Count outliers - should decrease with higher threshold
                    if threshold > 1.5:
                        n_outliers = df_out["is_outlier"].sum()
                        prev_n_outliers = detect_outliers(
                            df_clean, 
                            column="close",
                            threshold=threshold-0.5
                        )["is_outlier"].sum()
                        self.assertLess(
                            n_outliers,
                            prev_n_outliers,
                            "Higher threshold should detect fewer outliers"
                        )

            # Test multiple columns
            cols_to_check = ["close", "volume"]
            df_out = detect_outliers(df_clean, column=cols_to_check, threshold=2.5)
            for col in cols_to_check:
                self.assertIn(
                    f"{col}_is_outlier",
                    df_out.columns,
                    f"Missing outlier column for {col}"
                )
            
            self.mock_notifier.send_notification("Outlier detection tests passed", "info")
        except Exception as e:
            self.mock_notifier.send_alert(f"Outlier detection test failed: {str(e)}", "error")
            raise

    def test_winsorize_series(self) -> None:
        """Test winsorization of time series."""
        try:
            series = self.df["close"]
            
            # Test different limits
            limit_scenarios = [
                (0.01, 0.01),  # Light winsorization
                (0.05, 0.05),  # Moderate winsorization
                (0.1, 0.1),    # Heavy winsorization
                (0.05, 0.1),   # Asymmetric winsorization
            ]
            
            for limits in limit_scenarios:
                with self.subTest(limits=limits):
                    winsorized = winsorize_series(series, limits=limits)
                    lower_limit, upper_limit = series.quantile([limits[0], 1 - limits[1]])
                    
                    # Check bounds
                    self.assertGreaterEqual(winsorized.min(), lower_limit)
                    self.assertLessEqual(winsorized.max(), upper_limit)
                    
                    # Check distribution preservation
                    self.assertAlmostEqual(
                        winsorized.mean(),
                        series.mean(),
                        delta=series.std() * 0.5,
                        msg="Winsorization shouldn't drastically change mean"
                    )

            # Test edge cases and error handling
            with self.assertRaises(ValueError):
                winsorize_series(series, limits=(0.5, 0.6))  # Invalid limits
            with self.assertRaises(ValueError):
                winsorize_series(series, limits=(-0.1, 0.1))  # Negative limit
            with self.assertRaises(ValueError):
                winsorize_series(pd.Series([]))  # Empty series
            
            self.mock_notifier.send_notification("Winsorization tests passed", "info")
        except Exception as e:
            self.mock_notifier.send_alert(f"Winsorization test failed: {str(e)}", "error")
            raise

    def test_preprocess_pipeline(self) -> None:
        """Test full preprocessing pipeline."""
        try:
            # Test basic pipeline
            df_processed = preprocess_pipeline(
                self.df,
                price_col="close",
                fill_method="median",
                outlier_threshold=2.5,
                winsorize_limits=(0.05, 0.05)
            )

            # Check required columns exist
            required_cols = ["log_return", "volatility", "is_outlier"]
            for col in required_cols:
                self.assertIn(col, df_processed.columns)

            # Verify data quality
            self.assertFalse(df_processed.isnull().values.any(), "No null values")
            self.assertTrue((df_processed["volume"] > 0).all(), "Volume > 0")
            self.assertTrue(
                np.isfinite(df_processed["log_return"]).all(), 
                "Returns are finite"
            )
            
            # Test pipeline with different parameters
            parameter_scenarios = [
                {
                    "fill_method": "ffill",
                    "outlier_threshold": 3.0,
                    "winsorize_limits": (0.01, 0.01)
                },
                {
                    "fill_method": "linear",
                    "outlier_threshold": 2.0,
                    "winsorize_limits": (0.1, 0.1)
                }
            ]
            
            for params in parameter_scenarios:
                with self.subTest(**params):
                    df_test = preprocess_pipeline(self.df, price_col="close", **params)
                    self.assertFalse(df_test.isnull().values.any())
                    self.assertTrue(np.isfinite(df_test["volatility"]).all())
            
            # Cache results for potential reuse
            self.mock_cache.set("last_processed_data", df_processed)
            self.mock_notifier.send_notification("Preprocessing pipeline tests passed", "info")
        except Exception as e:
            self.mock_notifier.send_alert(
                f"Preprocessing pipeline test failed: {str(e)}", 
                "error"
            )
            raise

class TestDataStorageAndHistoricalData(MockTestCase):
    """Test data storage and retrieval operations."""

    def setUp(self) -> None:
        """Initialize test resources."""
        super().setUp()
        self.test_csv = self.create_temp_file(suffix=".csv")
        self.test_db = self.create_temp_file(suffix=".db")
        
        self.df = self.generate_test_data(periods=10)
        self.df.to_csv(self.test_csv, index=False)
        
        self.historical_manager = HistoricalDataManager(
            csv_path=self.test_csv,
            db_path=self.test_db
        )
        self.mock_notifier.send_notification("Historical data test setup complete", "debug")

    def test_load_from_csv(self) -> None:
        """Test CSV data loading."""
        try:
            df_loaded = self.historical_manager.load_from_csv()
            self.assert_df_valid(df_loaded)
            self.assertEqual(len(df_loaded), len(self.df))
            
            # Test data types
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_loaded["timestamp"]))
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                self.assertTrue(pd.api.types.is_numeric_dtype(df_loaded[col]))
            
            # Test error handling for malformed CSV
            with open(self.test_csv, 'a') as f:
                f.write("invalid,data,format\n")
            with self.assertRaises(pd.errors.EmptyDataError):
                self.historical_manager.load_from_csv()
            
            self.mock_notifier.send_notification("CSV load test passed", "info")
        except Exception as e:
            self.mock_notifier.send_alert(f"CSV load test failed: {str(e)}", "error")
            raise

    def test_database_operations(self) -> None:
        """Test database operations."""
        try:
            # Test initial load
            self.historical_manager.load_to_database(self.df)
            
            # Test basic query
            query = "SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT 1"
            result = self.historical_manager.query_database(query)
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
            
            # Test data integrity
            query = "SELECT MIN(close) as min_close, MAX(close) as max_close FROM historical_data"
            result = self.historical_manager.query_database(query)
            self.assertEqual(
                result.iloc[0]["min_close"],
                self.df["close"].min()
            )
            self.assertEqual(
                result.iloc[0]["max_close"],
                self.df["close"].max()
            )
            
            # Test transaction rollback
            with self.assertRaises(sqlite3.IntegrityError):
                self.historical_manager.load_to_database(self.df)  # Duplicate data
            
            # Verify data wasn't duplicated
            query = "SELECT COUNT(*) as count FROM historical_data"
            result = self.historical_manager.query_database(query)
            self.assertEqual(result.iloc[0]["count"], len(self.df))
            
            # Cache query for monitoring
            self.mock_cache.set("last_db_query", query)
            self.mock_notifier.send_notification("Database operation tests passed", "info")
        except Exception as e:
            self.mock_notifier.send_alert(
                f"Database operation test failed: {str(e)}", 
                "error"
            )
            raise

    def tearDown(self) -> None:
        """Clean up test resources."""
        try:
            self.historical_manager.close_connection()
        except Exception as e:
            self.mock_notifier.send_alert(f"Error closing database connection: {str(e)}", "error")
        super().tearDown()
