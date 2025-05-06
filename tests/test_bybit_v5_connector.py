"""
Tests for Bybit V5 connector
"""
import pytest
from unittest.mock import MagicMock, patch
from python_libs.bybit_v5_connector import BybitV5Connector
from utils.error_handling import TradingSystemError
from pybit.exceptions import InvalidRequestError, FailedRequestError

@pytest.fixture
def connector():
    """Create test connector with mocked initialization"""
    with patch('python_libs.bybit_v5_connector.HTTP') as mock_http:
        connector = BybitV5Connector(testnet=True)
        connector._initialized = False  # Reset initialization for testing
        return connector

def test_initialization(connector):
    """Test connector initialization"""
    assert connector.testnet is True
    assert connector.initialized is False
    
    # Test initialization
    connector.initialize()
    assert connector.initialized is True

@patch('python_libs.bybit_v5_connector.HTTP')
def test_get_market_data(mock_http, connector):
    mock_response = {
        "result": {
            "symbol": "BTCUSDT",
            "lastPrice": "30000",
            "volume24h": "1000"
        }
    }
    mock_http.return_value.get_tickers.return_value = mock_response
    connector.initialize()  # Initialize first
    connector.client = mock_http.return_value  # Set mocked client
    
    result = connector.get_market_data("BTCUSDT")
    assert result == mock_response["result"]
    mock_http.return_value.get_tickers.assert_called_with(
        category="linear",
        symbol="BTCUSDT"
    )

@patch('python_libs.bybit_v5_connector.HTTP')
def test_market_data_error(mock_http, connector):
    """Test market data error handling"""
    mock_error = InvalidRequestError(
        message="Test error",
        status_code=400,
        time="2025-05-01T12:00:00Z",
        resp_headers={"Content-Type": "application/json"},
        request={"method": "GET", "url": "test-url"}
    )
    mock_http.return_value.get_tickers.side_effect = mock_error
    connector.initialize()
    connector.client = mock_http.return_value
    
    with pytest.raises(InvalidRequestError):
        connector.get_market_data("BTCUSDT")

@patch('python_libs.bybit_v5_connector.HTTP')
def test_place_order(mock_http, connector):
    mock_response = {
        "result": {
            "orderId": "123456",
            "symbol": "BTCUSDT",
            "status": "NEW"
        }
    }
    mock_http.return_value.place_order.return_value = mock_response
    connector.initialize()
    connector.client = mock_http.return_value
    
    result = connector.place_order(
        symbol="BTCUSDT",
        side="Buy",
        order_type="Market",
        qty=0.001
    )
    assert result == mock_response["result"]

@patch('python_libs.bybit_v5_connector.HTTP')
def test_order_operations(mock_http, connector):
    """Test order placement and management"""
    mock_order_response = {
        "result": {
            "orderId": "123456",
            "symbol": "BTCUSDT",
            "status": "NEW"
        }
    }
    mock_modify_response = {
        "result": {
            "orderId": "123456",
            "symbol": "BTCUSDT",
            "status": "MODIFIED"
        }
    }
    mock_cancel_response = {
        "result": {
            "orderId": "123456"
        }
    }
    
    mock_http.return_value.place_order.return_value = mock_order_response
    mock_http.return_value.amend_order.return_value = mock_modify_response
    mock_http.return_value.cancel_order.return_value = mock_cancel_response
    
    connector.initialize()
    connector.client = mock_http.return_value
    
    # Place test order
    order = connector.place_order(
        symbol="BTCUSDT",
        side="Buy",
        order_type="Limit",
        qty=0.001,
        price=25000.0
    )
    assert isinstance(order, dict)
    assert order["orderId"] == "123456"
    
    if order.get("orderId"):
        # Test order modification
        modified = connector.modify_order(
            symbol="BTCUSDT",
            order_id=order["orderId"],
            price=24000.0
        )
        assert isinstance(modified, dict)
        assert modified["status"] == "MODIFIED"
        
        # Test order cancellation
        cancelled = connector.cancel_order(
            symbol="BTCUSDT",
            order_id=order["orderId"]
        )
        assert isinstance(cancelled, dict)
        assert cancelled["orderId"] == "123456"

@patch('python_libs.bybit_v5_connector.HTTP')
def test_position_management(mock_http, connector):
    """Test position management"""
    mock_positions_response = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "size": "0.001",
                    "side": "Buy"
                }
            ]
        }
    }
    mock_balances_response = {
        "result": {
            "list": [
                {
                    "coin": {
                        "USDT": {
                            "walletBalance": "1000",
                            "availableToWithdraw": "900"
                        }
                    }
                }
            ]
        }
    }
    
    mock_http.return_value.get_positions.return_value = mock_positions_response
    mock_http.return_value.get_wallet_balance.return_value = mock_balances_response
    
    connector.initialize()
    connector.client = mock_http.return_value
    
    # Get positions
    positions = connector.get_positions("BTCUSDT")
    assert isinstance(positions, list)
    assert positions[0]["symbol"] == "BTCUSDT"
    
    # Get account balances
    balances = connector.get_balances()
    assert isinstance(balances, dict)
    assert "USDT" in balances

def test_error_handling():
    """Test error handling"""
    connector = BybitV5Connector(testnet=True)
    connector._initialized = False  # Force uninitialized state
    
    with pytest.raises(TradingSystemError):
        connector.get_market_data("BTCUSDT")

@pytest.mark.parametrize("response_data", [
    {"retCode": 0, "result": {"list": [{"lastPrice": "50000"}]}},
    {"retCode": -1, "result": None},
    {}
])
def test_market_data_response_handling(connector, response_data):
    """Test handling different market data response formats"""
    connector._initialized = True
    connector.client.get_tickers = MagicMock(return_value=response_data)
    
    if response_data.get("retCode") == 0:
        result = connector.get_market_data("BTCUSDT")
        assert isinstance(result, dict)
    else:
        result = connector.get_market_data("BTCUSDT")
        assert result == {}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])