"""Base test case classes and utilities."""

import os
import unittest
import logging
import tempfile
import numpy as np
import pandas as pd
from typing import List

class BaseTestCase(unittest.TestCase):
    """Base test case with common utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test-wide resources."""
        np.random.seed(42)  # Ensure reproducible tests
        cls.logger = logging.getLogger(cls.__name__)
        cls._old_maxDiff = cls.maxDiff
        cls.maxDiff = None  # Enable full diff output
    
    def setUp(self):
        """Set up test case resources."""
        self.temp_files: List[str] = []
        self.addTypeEqualityFunc(pd.DataFrame, self.assert_dataframe_equal)
        self.addTypeEqualityFunc(np.ndarray, self.assert_array_equal)
        
    def tearDown(self):
        """Clean up test resources."""
        for file in self.temp_files:
            if os.path.exists(file):
                os.remove(file)

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level resources."""
        cls.maxDiff = cls._old_maxDiff
    
    def generate_test_data(self, periods: int = 100) -> pd.DataFrame:
        """Generate standard test data frame."""
        dates = pd.date_range("2023-01-01", periods=periods, freq="D")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, periods),
            "high": np.random.uniform(110, 120, periods),
            "low": np.random.uniform(90, 100, periods),
            "close": np.random.uniform(100, 115, periods),
            "volume": np.random.randint(1000, 2000, periods),
        })
        self.assert_df_valid(df)
        return df
    
    def assert_df_valid(self, df: pd.DataFrame, check_nulls: bool = True):
        """Validate DataFrame structure and content."""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
        if check_nulls:
            self.assertFalse(df.isnull().values.any(), "DataFrame contains null values")
        self.assertTrue((df["high"] >= df["low"]).all(), "High prices must be >= low prices")
        self.assertTrue((df["volume"] >= 0).all(), "Volume must be non-negative")
        
    def create_temp_file(self, suffix: str = "") -> str:
        """Create and track a temporary file."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            self.temp_files.append(tmp.name)
            return tmp.name

    def assert_dataframe_equal(self, first: pd.DataFrame, second: pd.DataFrame, msg=None):
        """Custom DataFrame equality assertion with better error messages."""
        try:
            pd.testing.assert_frame_equal(first, second)
        except AssertionError as e:
            raise self.failureException(msg or str(e))
            
    def assert_array_equal(self, first: np.ndarray, second: np.ndarray, msg=None):
        """Custom numpy array equality assertion with better error messages."""
        try:
            np.testing.assert_array_equal(first, second)
        except AssertionError as e:
            raise self.failureException(msg or str(e))
            
    def assert_json_equal(self, first: dict, second: dict, msg=None):
        """Assert two JSON-like objects are equal, with clear diff output."""
        import json
        try:
            self.assertEqual(
                json.dumps(first, sort_keys=True, indent=2),
                json.dumps(second, sort_keys=True, indent=2),
                msg=msg
            )
        except AssertionError:
            self.assertEqual(first, second, msg=msg)