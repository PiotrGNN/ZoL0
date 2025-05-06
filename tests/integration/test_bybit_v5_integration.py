import pytest
from python_libs.bybit_v5_connector import BybitV5Connector
import os

@pytest.mark.integration
class TestBybitV5Integration:
    """Integration tests for Bybit V5 connector"""

    @pytest.fixture(autouse=True)
    def setup(self, test_credentials):
        """Setup test environment"""
        self.symbol = "BTCUSDT"
        self.connector = BybitV5Connector(
            api_key=test_credentials['api_key'],
            api_secret=test_credentials['api_secret'],
            testnet=test_credentials['testnet']
        )
        # Ensure we have valid credentials before running tests
        if not test_credentials['api_key'] or not test_credentials['api_secret']:
            pytest.skip("Skipping integration tests: No valid API credentials")
            
        self.connector.initialize()
        yield
        # Cleanup active orders in teardown
        try:
            self.connector.cancel_all_orders(self.symbol)
        except Exception as e:
            pytest.warn(f"Cleanup failed: {e}")

    @pytest.mark.order(1)
    def test_market_data_access(self):
        """Test market data endpoints"""
        data = self.connector.get_market_data(self.symbol)
        assert data is not None
        assert "last_price" in data
        assert float(data["last_price"]) > 0

    @pytest.mark.order(2) 
    def test_full_trading_cycle(self):
        """Test complete trading cycle: market data -> order -> position -> close"""
        try:
            # 1. Get market data
            market_data = self.connector.get_market_data(self.symbol)
            assert market_data is not None
            assert "last_price" in market_data

            # 2. Place market order
            order = self.connector.place_order(
                symbol=self.symbol,
                side="Buy",
                order_type="Market",
                qty=0.001,
                reduce_only=False
            )
            assert order is not None
            assert order.get("success") is True

            # 3. Check position
            position = self.connector.get_positions(self.symbol)
            assert position is not None
            assert position.get("size", 0) > 0
            
            # 4. Close position
            close_order = self.connector.place_order(
                symbol=self.symbol,
                side="Sell",
                order_type="Market", 
                qty=0.001,
                reduce_only=True
            )
            assert close_order is not None
            assert close_order.get("success") is True

        except Exception as e:
            pytest.fail(f"Trading cycle failed: {e}")

    @pytest.mark.order(3)
    @pytest.mark.parametrize("interval", ["1", "5", "15", "30", "60", "240", "D"])
    def test_ohlcv_data(self, interval):
        """Test getting OHLCV data with different intervals"""
        try:
            data = self.connector.get_klines(
                symbol=self.symbol,
                interval=interval,
                limit=100
            )
            assert data is not None
            assert len(data) > 0
            
            # Verify data structure
            for candle in data[:5]:
                required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
                assert all(key in candle for key in required_fields)
                assert all(isinstance(candle[key], (int, float, str)) for key in required_fields)
                
        except Exception as e:
            pytest.fail(f"OHLCV data test failed for interval {interval}: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])