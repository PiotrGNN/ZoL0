"""
Test połączenia z giełdą Bybit
"""
import unittest
from data.execution.bybit_connector import BybitConnector

class TestBybitConnection(unittest.TestCase):
    def setUp(self):
        self.connector = BybitConnector(
            api_key="test_key",
            api_secret="test_secret",
            use_testnet=True
        )

    def test_server_time(self):
        """Test pobierania czasu serwera"""
        result = self.connector.get_server_time()
        self.assertTrue('success' in result)
        self.assertTrue('time_ms' in result)
        self.assertTrue(isinstance(result['time_ms'], int))

    def test_get_klines(self):
        """Test pobierania świec"""
        klines = self.connector.get_klines(symbol="BTCUSDT", interval="15m", limit=10)
        self.assertTrue(isinstance(klines, list))
        self.assertEqual(len(klines), 10)
        if klines:
            self.assertTrue('timestamp' in klines[0])
            self.assertTrue('open' in klines[0])

    def test_get_order_book(self):
        """Test pobierania książki zleceń"""
        order_book = self.connector.get_order_book(symbol="BTCUSDT", limit=5)
        self.assertTrue('bids' in order_book)
        self.assertTrue('asks' in order_book)
        self.assertEqual(len(order_book['bids']), 5)
        self.assertEqual(len(order_book['asks']), 5)

if __name__ == '__main__':
    unittest.main()