"""Tests for BybitConnector"""
import pytest
from unittest.mock import MagicMock, patch
from data.execution.bybit_connector import BybitConnector
import requests

@pytest.fixture
def mock_bybit_connector():
    """Create mocked BybitConnector"""
    mock_get = MagicMock()
    with patch('requests.Session.get', mock_get):
        connector = BybitConnector(
            api_key="test_key",
            api_secret="test_secret",
            use_testnet=True
        )
        connector.session = requests.Session()
        yield connector, mock_get

def test_successful_ticker_response(mock_bybit_connector):
    connector, mock_get = mock_bybit_connector

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "retCode": 0,
        "result": {
            "list": [{
                "symbol": "BTCUSDT",
                "bid1Price": "30000.50",
                "ask1Price": "30001.50",
                "lastPrice": "30000.75",
                "volume24h": "1000.5"
            }]
        }
    }
    mock_get.return_value = mock_response

    result = connector.get_ticker("BTCUSDT")
    assert result["success"] is True
    assert "data" in result
    assert result["data"]["list"][0]["symbol"] == "BTCUSDT"

def test_rate_limit_handling(mock_bybit_connector):
    connector, mock_get = mock_bybit_connector

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "retCode": 10006,
        "retMsg": "Too many visits!"
    }
    mock_get.return_value = mock_response

    result = connector.get_ticker("BTCUSDT")
    assert result["success"] is False
    assert "Rate limit exceeded" in result["error"]

def test_error_response_handling(mock_bybit_connector):
    connector, mock_get = mock_bybit_connector

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "retCode": 10001,
        "retMsg": "Invalid API key"
    }
    mock_get.return_value = mock_response

    result = connector.get_ticker("BTCUSDT")
    assert result["success"] is False
    assert "HTTP Error: 400" in result["error"]

def test_invalid_response_format(mock_bybit_connector):
    connector, mock_get = mock_bybit_connector

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}  # Empty response
    mock_get.return_value = mock_response

    result = connector.get_ticker("BTCUSDT")
    assert result["success"] is False
    assert "Invalid response format" in result["error"]