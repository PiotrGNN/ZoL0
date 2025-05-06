"""
PyTest configuration and fixtures
"""
import os
import sys
import pytest
import yaml
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def config() -> Dict[str, Any]:
    """Load test configuration"""
    with open(os.path.join(os.path.dirname(__file__), 'test_config.yaml')) as f:
        return yaml.safe_load(f)

@pytest.fixture
def test_credentials() -> Dict[str, str]:
    """Provide test API credentials"""
    return {
        'api_key': os.getenv('BYBIT_TEST_API_KEY', 'test'),
        'api_secret': os.getenv('BYBIT_TEST_API_SECRET', 'test'),
        'testnet': True
    }

@pytest.fixture
def mock_market_env():
    """Create a mock market environment for testing"""
    class MockMarketEnv:
        def __init__(self):
            self.reset()
        
        def reset(self):
            self.balance = 10000.0
            self.position = 0.0
            return self._get_state()
            
        def _get_state(self):
            return {
                'balance': self.balance,
                'position': self.position
            }
    return MockMarketEnv()

@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate sample market data for testing"""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="5min")
    data = {
        'timestamp': dates,
        'open': np.random.uniform(45000, 50000, 100),
        'high': np.random.uniform(45500, 50500, 100),
        'low': np.random.uniform(44500, 49500, 100),
        'close': np.random.uniform(45000, 50000, 100),
        'volume': np.random.uniform(1, 100, 100)
    }
    # Ensure high/low are properly set
    data['high'] = np.maximum(data['high'], data['open'])
    data['low'] = np.minimum(data['low'], data['open'])
    return pd.DataFrame(data)

@pytest.fixture
def mock_order_book():
    """Generate mock order book data"""
    return {
        'bids': [[45000.0, 1.5], [44900.0, 2.0], [44800.0, 2.5]],
        'asks': [[45100.0, 1.2], [45200.0, 1.8], [45300.0, 2.2]]
    }

@pytest.fixture
def mock_trade_history():
    """Generate mock trade history"""
    return [
        {'timestamp': datetime.now() - timedelta(minutes=i),
         'side': 'buy' if i % 2 == 0 else 'sell',
         'price': 45000 + i * 10,
         'amount': 0.1,
         'fee': 0.001} 
        for i in range(10)
    ]

@pytest.fixture
def mock_sentiment_data():
    """Generate mock sentiment analysis data"""
    return {
        'positive': ['bullish momentum', 'strong support', 'breakout confirmed'],
        'negative': ['bearish divergence', 'resistance hit', 'volume declining'],
        'neutral': ['consolidation phase', 'sideways movement', 'range bound']
    }

@pytest.fixture
def mock_market_patterns():
    """Generate mock market pattern data"""
    return {
        'bull_flag': {'confidence': 0.92, 'duration': '2h'},
        'head_shoulders': {'confidence': 0.85, 'duration': '4h'},
        'triangle': {'confidence': 0.78, 'duration': '1h'}
    }

@pytest.fixture(autouse=True)
def setup_logging(request):
    """Setup test logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=f'logs/test_{request.node.name}.log'
    )
    return logging.getLogger(request.node.name)

@pytest.fixture
def cleanup_test_data(request):
    """Cleanup test data after each test"""
    def finalizer():
        test_files = [
            'temp_test_data.csv',
            'temp_historical_data.db',
            'test_models.pkl'
        ]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    request.addfinalizer(finalizer)

@pytest.fixture
def mock_api_response():
    """Mock API response data"""
    return {
        'success': True,
        'message': 'OK',
        'data': {
            'balance': {'BTC': 1.0, 'USDT': 50000.0},
            'positions': [],
            'orders': []
        }
    }