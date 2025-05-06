"""
test_api_environment.py
---------------------
Tests for ApiClient and EnvironmentManager
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import time
from datetime import datetime

from data.utils.api_client import ApiClient
from data.utils.environment_manager import EnvironmentManager

@pytest.fixture
def api_client():
    """Fixture providing an ApiClient instance"""
    return ApiClient(
        api_key="test_key",
        api_secret="test_secret",
        use_testnet=True
    )

@pytest.fixture
def env_manager(api_client):
    """Fixture providing an EnvironmentManager instance"""
    return EnvironmentManager(api_client=api_client)

class TestApiClient:
    def test_init_defaults_to_testnet(self):
        """Test that ApiClient defaults to testnet environment"""
        client = ApiClient()
        assert client.use_testnet is True
        assert "testnet" in client.base_url

    def test_rate_limiting(self, api_client):
        """Test rate limiting behavior"""
        # Make multiple requests in quick succession
        for _ in range(3):
            response = api_client.get_server_time()
            assert response["success"] is True
            
        # Check that timestamps are properly spaced
        timestamps = api_client.request_timestamps[-3:]
        for i in range(1, len(timestamps)):
            assert timestamps[i] - timestamps[i-1] >= api_client.min_request_interval

    @patch('requests.request')
    def test_request_signing(self, mock_request, api_client):
        """Test request signing mechanism"""
        mock_request.return_value.status_code = 200
        mock_request.return_value.json.return_value = {"retCode": 0, "result": {}}

        api_client.request("GET", "/test", {"param": "value"})
        
        # Verify headers
        call_args = mock_request.call_args
        headers = call_args[1]["headers"]
        assert "X-BAPI-API-KEY" in headers
        assert "X-BAPI-SIGN" in headers
        assert "X-BAPI-TIMESTAMP" in headers

    def test_production_validation(self):
        """Test production environment validation"""
        with pytest.raises(ValueError) as exc_info:
            ApiClient(use_testnet=False)
        assert "Production API access not explicitly confirmed" in str(exc_info.value)

class TestEnvironmentManager:
    @pytest.mark.asyncio
    async def test_validate_switch(self, env_manager):
        """Test environment switch validation"""
        # Test rapid switching prevention
        env_manager.state.last_switch_time = time.time()
        validation = await env_manager.validate_switch(True)
        assert validation["can_switch"] is False
        assert "wait at least 60 seconds" in validation["reason"]

        # Test switch to production validation
        validation = await env_manager.validate_switch(False)
        assert validation["can_switch"] is False
        assert "Production environment not properly confirmed" in validation["reason"]

    @pytest.mark.asyncio
    async def test_switch_environment(self, env_manager):
        """Test environment switching"""
        env_manager.state.last_switch_time = 0  # Allow immediate switch
        
        # Switch to testnet (already in testnet, should succeed)
        result = await env_manager.switch_environment(True)
        assert result["success"] is True
        assert result["environment"] == "testnet"
        
        # Try switching to production (should fail without proper config)
        result = await env_manager.switch_environment(False)
        assert result["success"] is False
        assert "Production environment not properly confirmed" in result["error"]

    def test_environment_status(self, env_manager):
        """Test environment status reporting"""
        status = env_manager.get_environment_status()
        assert "environment" in status
        assert "initialized" in status
        assert "switch_in_progress" in status
        assert isinstance(status["production_enabled"], bool)
        assert isinstance(status["production_confirmed"], bool)

    def test_load_environment(self):
        """Test environment loading from env vars"""
        with patch.dict(os.environ, {
            "BYBIT_TESTNET": "true"
        }):
            manager = EnvironmentManager()
            assert manager.state.is_testnet is True

        with patch.dict(os.environ, {
            "BYBIT_TESTNET": "false",
            "BYBIT_PRODUCTION_CONFIRMED": "true",
            "BYBIT_PRODUCTION_ENABLED": "true"
        }):
            manager = EnvironmentManager()
            assert manager.state.is_testnet is False