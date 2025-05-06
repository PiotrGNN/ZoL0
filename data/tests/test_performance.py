"""Performance tests for critical components."""

import unittest
import pytest
from data.tests.base_test import BaseTestCase
from data.execution.bybit_connector import BybitConnector

class TestPerformance(BaseTestCase):
    """Test performance of critical operations."""
    
    def setUp(self):
        """Set up test resources."""
        super().setUp()
        self.connector = BybitConnector(
            api_key="test_key",
            api_secret="test_secret",
            use_testnet=True
        )

    @pytest.mark.benchmark
    def test_connector_initialization(self):
        """Test BybitConnector initialization performance."""
        self.assertIsNotNone(self.connector)
        self.assertTrue(self.connector.use_testnet)
        
    @pytest.mark.benchmark
    def test_get_server_time(self):
        """Test server time retrieval performance."""
        result = self.connector.get_server_time()
        self.assertTrue(result["success"])
        self.assertIn("time_ms", result)
        
    @pytest.mark.benchmark
    def test_get_klines(self):
        """Test klines data retrieval performance."""
        klines = self.connector.get_klines(
            symbol="BTCUSDT",
            interval="15",
            limit=100
        )
        self.assertGreater(len(klines), 0)
        self.assertTrue(all(isinstance(k, dict) for k in klines))
        
    @pytest.mark.benchmark
    def test_get_order_book(self):
        """Test order book retrieval performance."""
        order_book = self.connector.get_order_book(
            symbol="BTCUSDT",
            limit=5
        )
        self.assertTrue(order_book["success"])
        self.assertIn("bids", order_book)
        self.assertIn("asks", order_book)
        
    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()