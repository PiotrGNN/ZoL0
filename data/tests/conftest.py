"""Basic pytest configuration and fixtures."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
import logging
from typing import Dict, Any, Generator
from pathlib import Path

def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Fixture providing test data directory."""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data."""
    periods = 100
    dates = pd.date_range("2025-01-01", periods=periods, freq="H")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, periods),
        "high": np.random.uniform(110, 120, periods),
        "low": np.random.uniform(90, 100, periods),
        "close": np.random.uniform(100, 115, periods),
        "volume": np.random.randint(1000, 2000, periods),
    })
    return df

@pytest.fixture
def mock_config():
    """Fixture providing test configuration."""
    return {
        "environment": {
            "mode": "test",
            "debug": True
        },
        "trading": {
            "max_positions": 5,
            "risk_per_trade": 0.02,
            "max_leverage": 3.0,
            "allowed_symbols": ["BTCUSDT", "ETHUSDT"]
        }
    }

@pytest.fixture
def temp_file() -> Generator[str, None, None]:
    """Create a temporary file and clean it up after use."""
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def mock_db(test_data_dir) -> Generator[str, None, None]:
    """Create a temporary SQLite database for testing."""
    db_path = test_data_dir / "test.db"
    yield str(db_path)
    if db_path.exists():
        db_path.unlink()

@pytest.fixture
def mock_cache_dir(test_data_dir) -> Path:
    """Create a temporary cache directory."""
    cache_dir = test_data_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

@pytest.fixture
def mock_model_dir(test_data_dir) -> Path:
    """Create a temporary directory for model files."""
    model_dir = test_data_dir / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir

@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Configure environment for all tests."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("MOCK_EXCHANGE", "true")

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("EXCHANGE_API_KEY", "test_api_key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "test_api_secret")
    monkeypatch.setenv("TRADING_MODE", "test")

@pytest.fixture
def assert_log_contains(caplog):
    """Helper fixture to check log contents."""
    def _assert_log_contains(message: str, level: int = logging.INFO):
        records = [r for r in caplog.records if r.levelno == level]
        assert any(message in r.getMessage() for r in records), \
            f"Log message '{message}' not found in logs"
    return _assert_log_contains

@pytest.fixture
def mock_time(freezer):
    """Fixture to control time in tests."""
    freezer.move_to("2025-01-01")
    return freezer

@pytest.fixture
def sample_trade_data() -> Dict[str, Any]:
    """Generate sample trade data for testing."""
    return {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "LIMIT",
        "quantity": 0.001,
        "price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 52000.0,
        "leverage": 1.0
    }

@pytest.fixture
def large_test_data():
    """Generate large test dataset."""
    import pandas as pd
    import numpy as np
    
    periods = 10000
    dates = pd.date_range("2023-01-01", periods=periods, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, periods),
        "high": np.random.uniform(110, 120, periods),
        "low": np.random.uniform(90, 100, periods),
        "close": np.random.uniform(100, 115, periods),
        "volume": np.random.randint(1000, 2000, periods),
    })
    return df

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "critical: mark test as critical for trading functionality"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Always run critical tests first
    items.sort(key=lambda x: 1 if "critical" in x.keywords else 2)

    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)