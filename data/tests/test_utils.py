"""Test utilities and mock classes."""

import os
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from .base_test import BaseTestCase

class MockDB:
    """Mock database for testing."""
    def __init__(self):
        self.connected = False
        self.queries = []
        self.data = {}
        
    def connect(self) -> bool:
        self.connected = True
        return True
        
    def disconnect(self) -> bool:
        self.connected = False
        return True
        
    def is_connected(self) -> bool:
        return self.connected
        
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Any:
        self.queries.append((query, params))
        return "mock_result"
        
    def get_last_query(self) -> Union[tuple, None]:
        return self.queries[-1] if self.queries else None

class MockCache:
    """Mock cache for testing."""
    def __init__(self):
        self.data = {}
        self._valid = {}
        
    def get(self, key: str) -> Any:
        return self.data.get(key)
        
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._valid[key] = True
        
    def delete(self, key: str) -> None:
        self.data.pop(key, None)
        self._valid.pop(key, None)
        
    def clear(self) -> None:
        self.data.clear()
        self._valid.clear()
        
    def is_valid(self, key: str) -> bool:
        return self._valid.get(key, False)

class MockNotifier:
    """Mock notification system for testing."""
    def __init__(self):
        self.notifications = []
        self.alerts = []
        
    def send_notification(self, message: str, level: str = "info") -> Dict[str, Any]:
        self.notifications.append({"message": message, "level": level})
        return {"status": "sent", "id": len(self.notifications)}
        
    def send_alert(self, message: str, level: str = "warning") -> bool:
        self.alerts.append({"message": message, "level": level})
        return True
        
    def get_notifications(self) -> List[Dict[str, Any]]:
        return self.notifications
        
    def get_alerts(self) -> List[Dict[str, Any]]:
        return self.alerts

class MockTestCase(BaseTestCase):
    """Base test case with common mock objects."""
    
    def setUp(self):
        """Set up test case resources including mock objects."""
        super().setUp()
        self.mock_db = MockDB()
        self.mock_cache = MockCache()
        self.mock_notifier = MockNotifier()
        
    def tearDown(self):
        """Clean up test resources including mock connections."""
        if hasattr(self.mock_db, 'disconnect') and self.mock_db.is_connected():
            self.mock_db.disconnect()
        super().tearDown()

def create_test_dataframe(
    periods: int = 100,
    frequency: str = "D",
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """Create a DataFrame with test data."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    dates = pd.date_range("2023-01-01", periods=periods, freq=frequency)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, periods),
        "high": np.random.uniform(110, 120, periods),
        "low": np.random.uniform(90, 100, periods),
        "close": np.random.uniform(100, 115, periods),
        "volume": np.random.randint(1000, 2000, periods),
    })
    return df

def with_temp_file(suffix: str = ""):
    """Decorator to create a temporary file for test."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                try:
                    kwargs["temp_path"] = tmp.name
                    return func(*args, **kwargs)
                finally:
                    if os.path.exists(tmp.name):
                        os.remove(tmp.name)
        return wrapper
    return decorator

def assert_dataframe_valid(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    check_nulls: bool = True
) -> None:
    """Validate DataFrame structure and content."""
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise AssertionError(f"Missing required columns: {missing}")
    
    if check_nulls and df.isnull().values.any():
        raise AssertionError("DataFrame contains null values")
