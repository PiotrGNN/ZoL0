"""
test_data_consistency.py
---------------------
Tests for data consistency and integrity across system operations.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from data.tests import BaseTestCase
from data.data.data_preprocessing import preprocess_pipeline
from data.data.historical_data import HistoricalDataManager
from data.ai_models.model_loader import ModelLoader

class TestDataConsistency(BaseTestCase):
    """Test data consistency and integrity."""

    def setUp(self):
        """Initialize test resources."""
        super().setUp()
        self.data_dir = Path(self.create_temp_dir())
        self.db_path = self.data_dir / "test.db"
        self.csv_path = self.data_dir / "market_data.csv"
        
        # Initialize components
        self.historical_data = HistoricalDataManager(
            csv_path=str(self.csv_path),
            db_path=str(self.db_path)
        )
        self.model_loader = ModelLoader()
        
        # Generate test data
        self.market_data = self.generate_test_data(periods=1000)
        self.market_data.to_csv(self.csv_path, index=False)

    def test_data_persistence(self):
        """Test data consistency during persistence operations."""
        # Calculate original data hash
        original_hash = self._calculate_dataframe_hash(self.market_data)
        
        # Test CSV persistence
        self.market_data.to_csv(self.csv_path, index=False)
        loaded_from_csv = pd.read_csv(self.csv_path)
        csv_hash = self._calculate_dataframe_hash(loaded_from_csv)
        
        self.assertEqual(
            original_hash,
            csv_hash,
            "Data corruption detected in CSV persistence"
        )
        
        # Test database persistence
        self.historical_data.load_to_database(self.market_data)
        loaded_from_db = self.historical_data.query_database(
            "SELECT * FROM historical_data"
        )
        db_hash = self._calculate_dataframe_hash(loaded_from_db)
        
        self.assertEqual(
            original_hash,
            db_hash,
            "Data corruption detected in database persistence"
        )

    def test_data_transformation_consistency(self):
        """Test data consistency through transformations."""
        # Original data hash
        original_data = self.market_data.copy()
        original_hash = self._calculate_dataframe_hash(original_data)
        
        # Apply transformations
        processed_data = preprocess_pipeline(
            original_data,
            price_col="close",
            fill_method="ffill",
            outlier_threshold=2.5
        )
        
        # Reverse transformations where possible
        reversed_data = self._reverse_transformations(processed_data)
        reversed_hash = self._calculate_dataframe_hash(reversed_data)
        
        # Compare critical columns
        critical_columns = ["open", "high", "low", "close", "volume"]
        for col in critical_columns:
            if col in original_data and col in reversed_data:
                pd.testing.assert_series_equal(
                    original_data[col],
                    reversed_data[col],
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-3  # Allow 0.1% difference due to floating point
                )

    def test_model_prediction_consistency(self):
        """Test consistency of model predictions."""
        models = self.model_loader.load_available_models()
        test_features = pd.DataFrame(
            np.random.random((100, 10)),
            columns=[f"feature_{i}" for i in range(10)]
        )
        
        for model_name, model in models.items():
            if not hasattr(model, "predict"):
                continue
            
            # Make predictions multiple times
            predictions = []
            for _ in range(10):
                pred = model.predict(test_features)
                predictions.append(pred)
            
            # Verify prediction consistency
            for i in range(1, len(predictions)):
                np.testing.assert_array_almost_equal(
                    predictions[0],
                    predictions[i],
                    decimal=6,
                    err_msg=f"Inconsistent predictions from {model_name}"
                )

    def test_time_series_consistency(self):
        """Test time series data consistency."""
        # Generate time series data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='1min'
        )
        
        time_series = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.random(len(dates))
        })
        
        # Test resampling consistency
        frequencies = ['5min', '15min', '1H', '4H', '1D']
        
        for freq in frequencies:
            resampled = time_series.resample(freq, on='timestamp').mean()
            reconstructed = resampled.resample('1min').ffill()
            
            # Verify value conservation
            original_sum = time_series['value'].sum()
            resampled_sum = resampled['value'].sum() * (
                pd.Timedelta(freq) / pd.Timedelta('1min')
            )
            
            np.testing.assert_almost_equal(
                original_sum,
                resampled_sum,
                decimal=2,
                err_msg=f"Value conservation failed for {freq} resampling"
            )

    def test_data_synchronization(self):
        """Test data synchronization across different sources."""
        # Create test data in different formats
        test_data = {
            'timestamp': [
                datetime.now() - timedelta(minutes=i)
                for i in range(100)
            ],
            'value': np.random.random(100)
        }
        
        # Save to different sources
        df = pd.DataFrame(test_data)
        df.to_csv(self.csv_path, index=False)
        
        conn = sqlite3.connect(str(self.db_path))
        df.to_sql('test_data', conn, if_exists='replace', index=False)
        conn.close()
        
        # Load from different sources
        csv_data = pd.read_csv(
            self.csv_path,
            parse_dates=['timestamp']
        )
        
        conn = sqlite3.connect(str(self.db_path))
        db_data = pd.read_sql(
            'SELECT * FROM test_data',
            conn,
            parse_dates=['timestamp']
        )
        conn.close()
        
        # Verify data consistency
        pd.testing.assert_frame_equal(
            csv_data,
            db_data,
            check_dtype=False  # SQLite might change some dtypes
        )

    def test_data_schema_consistency(self):
        """Test data schema consistency."""
        # Define expected schema
        expected_schema = {
            'timestamp': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        }
        
        # Test CSV schema
        csv_data = pd.read_csv(
            self.csv_path,
            parse_dates=['timestamp']
        )
        for column, dtype in expected_schema.items():
            self.assertEqual(
                str(csv_data[column].dtype),
                dtype,
                f"CSV column {column} has incorrect dtype"
            )
        
        # Test database schema
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get database schema
        cursor.execute("""
            SELECT sql 
            FROM sqlite_master 
            WHERE type='table' AND name='historical_data'
        """)
        
        schema = cursor.fetchone()[0]
        conn.close()
        
        # Verify required columns in schema
        for column in expected_schema:
            self.assertIn(
                column,
                schema,
                f"Database schema missing column {column}"
            )

    def test_data_boundaries(self):
        """Test data value boundaries and constraints."""
        # Load test data
        data = pd.read_csv(self.csv_path)
        
        # Define value constraints
        constraints = {
            'open': {'min': 0, 'max': 1e6},
            'high': {'min': 0, 'max': 1e6},
            'low': {'min': 0, 'max': 1e6},
            'close': {'min': 0, 'max': 1e6},
            'volume': {'min': 0, 'max': 1e9}
        }
        
        # Verify constraints
        for column, bounds in constraints.items():
            self.assertTrue(
                (data[column] >= bounds['min']).all(),
                f"{column} contains values below minimum"
            )
            self.assertTrue(
                (data[column] <= bounds['max']).all(),
                f"{column} contains values above maximum"
            )
        
        # Verify OHLC relationships
        self.assertTrue(
            (data['high'] >= data['low']).all(),
            "High prices must be >= low prices"
        )
        self.assertTrue(
            (data['high'] >= data['open']).all(),
            "High prices must be >= open prices"
        )
        self.assertTrue(
            (data['high'] >= data['close']).all(),
            "High prices must be >= close prices"
        )
        self.assertTrue(
            (data['low'] <= data['open']).all(),
            "Low prices must be <= open prices"
        )
        self.assertTrue(
            (data['low'] <= data['close']).all(),
            "Low prices must be <= close prices"
        )

    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for consistency checking."""
        # Convert to string representation and hash
        return hashlib.sha256(
            df.to_string().encode()
        ).hexdigest()

    def _reverse_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse data transformations where possible."""
        # Implement reverse transformations based on preprocessing steps
        reversed_df = df.copy()
        
        # Reverse any scaling
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'inverse_transform'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            reversed_df[numeric_cols] = self.scaler.inverse_transform(
                df[numeric_cols]
            )
        
        return reversed_df

    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        # Clean up data directory
        for file in self.data_dir.glob("*"):
            file.unlink()
        self.data_dir.rmdir()