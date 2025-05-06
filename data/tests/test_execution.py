"""
test_execution.py
-----------------
Tests for exchange connectivity and trade execution modules.
"""

from typing import Any, Dict, List
from datetime import datetime
import json
from data.tests import BaseTestCase
from data.execution.exchange_connector import ExchangeConnector
from data.execution.order_execution import OrderExecution
from data.execution.trade_executor import TradeExecutor

class MockExchangeConnector(ExchangeConnector):
    """Mock connector for testing."""

    def __init__(self) -> None:
        """Initialize mock connector."""
        self.connected = False
        self.last_request = None
        self.order_book = {}
        self.market_data = self._generate_mock_market_data()
        self.error_simulation = None

    def _generate_mock_market_data(self) -> List[Dict[str, Any]]:
        """Generate realistic mock market data."""
        base_price = 50000.0
        data = []
        for i in range(100):
            price = base_price * (1 + 0.0001 * i)  # Small upward trend
            data.append({
                "timestamp": datetime.now().timestamp() - i * 60,
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 10.0 + i * 0.1
            })
        return list(reversed(data))

    def connect(self) -> bool:
        """Simulate connection."""
        self.connected = True
        return True

    def get_market_data(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[Dict[str, Any]]:
        """Return mock market data."""
        if self.error_simulation == "market_data":
            raise ConnectionError("Simulated market data error")
        self.last_request = {
            "method": "get_market_data",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self.market_data[:limit]

    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place mock order with proper validation."""
        if self.error_simulation == "order":
            raise ConnectionError("Simulated order error")

        required_fields = ["symbol", "side", "type", "quantity"]
        if not all(field in order for field in required_fields):
            raise ValueError("Missing required order fields")

        if order["type"] == "LIMIT" and "price" not in order:
            raise ValueError("Limit order requires price")

        order_id = f"test_order_{len(self.order_book)}"
        self.order_book[order_id] = {
            **order,
            "status": "FILLED",
            "filled_qty": order["quantity"],
            "avg_price": order.get("price", self.market_data[0]["close"])
        }
        
        self.last_request = {
            "method": "place_order",
            "order": order
        }
        
        return {
            "orderId": order_id,
            "status": "FILLED",
            "executedQty": order["quantity"],
            "avgPrice": order.get("price", self.market_data[0]["close"])
        }

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get mock order status with validation."""
        if order_id not in self.order_book:
            raise ValueError(f"Order {order_id} not found")
        return self.order_book[order_id]

    def get_last_request(self) -> Dict[str, Any]:
        """Get last API request details for validation."""
        return self.last_request

    def simulate_error(self, error_type: str) -> None:
        """Set error simulation for testing error handling."""
        self.error_simulation = error_type

class MockOrderExecution(OrderExecution):
    """Mock order execution for testing."""

    def __init__(self, connector: MockExchangeConnector):
        """Initialize with mock connector."""
        super().__init__(connector)
        self.executed_orders = []

    def send_order(self, **kwargs) -> Dict[str, Any]:
        """Send order with tracking."""
        result = super().send_order(**kwargs)
        self.executed_orders.append(result)
        return result

class TestExecutionModules(BaseTestCase):
    """Test execution module functionality."""

    def setUp(self):
        """Initialize test resources."""
        super().setUp()
        self.connector = MockExchangeConnector()
        self.order_exec = MockOrderExecution(self.connector)
        self.trade_exec = TradeExecutor(self.order_exec)

    def test_market_data_retrieval(self):
        """Test market data API with validation."""
        test_cases = [
            {"symbol": "BTCUSDT", "interval": "1m", "limit": 5},
            {"symbol": "ETHUSDT", "interval": "5m", "limit": 10},
            {"symbol": "BNBUSDT", "interval": "15m", "limit": 15}
        ]

        for case in test_cases:
            with self.subTest(**case):
                data = self.connector.get_market_data(**case)
                
                # Verify data structure
                self.assertEqual(len(data), case["limit"])
                self.assertTrue(all(isinstance(d, dict) for d in data))
                
                # Verify required fields
                required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
                self.assertTrue(all(k in data[0] for k in required_fields))
                
                # Verify data integrity
                for candle in data:
                    self.assertTrue(candle["high"] >= candle["low"])
                    self.assertTrue(candle["volume"] > 0)
                
                # Verify request parameters
                last_request = self.connector.get_last_request()
                self.assertEqual(last_request["method"], "get_market_data")
                self.assertEqual(last_request["symbol"], case["symbol"])

        # Test error handling
        self.connector.simulate_error("market_data")
        with self.assertRaises(ConnectionError):
            self.connector.get_market_data("BTCUSDT")
        self.connector.simulate_error(None)

    def test_order_execution(self):
        """Test order execution flow."""
        # Test market orders
        market_orders = [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": 0.001
            },
            {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "MARKET",
                "quantity": 0.01
            }
        ]
        
        for order in market_orders:
            with self.subTest(**order):
                result = self.order_exec.send_order(**order)
                self.assertIn("orderId", result)
                self.assertEqual(result["status"], "FILLED")
                self.assertEqual(float(result["executedQty"]), order["quantity"])

        # Test limit orders
        limit_orders = [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": 0.001,
                "price": 50000.0
            },
            {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "quantity": 0.01,
                "price": 3000.0
            }
        ]
        
        for order in limit_orders:
            with self.subTest(**order):
                result = self.order_exec.send_order(**order)
                self.assertIn("orderId", result)
                self.assertEqual(result["status"], "FILLED")
                self.assertEqual(float(result["avgPrice"]), order["price"])
                
                # Verify order tracking
                self.assertIn(
                    result["orderId"],
                    [o["orderId"] for o in self.order_exec.executed_orders]
                )

        # Test error handling
        invalid_orders = [
            {},  # Empty order
            {"symbol": "", "side": "", "type": "", "quantity": 0},  # Invalid fields
            {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT"}  # Missing price
        ]
        
        for order in invalid_orders:
            with self.subTest(**order):
                with self.assertRaises((ValueError, KeyError)):
                    self.order_exec.send_order(**order)

        # Test connection errors
        self.connector.simulate_error("order")
        with self.assertRaises(ConnectionError):
            self.order_exec.send_order(**market_orders[0])
        self.connector.simulate_error(None)

    def test_trade_executor(self):
        """Test trade execution strategies."""
        # Test basic trade execution
        trade_params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
            "take_profit": 52000.0,
            "stop_loss": 48000.0
        }
        
        result = self.trade_exec.execute_trade(**trade_params)
        self.assertTrue(result["success"])
        self.assertIn("orderId", result)
        self.assertIn("entry_price", result)
        
        # Test trade with position sizing
        sized_trade = self.trade_exec.execute_trade_with_position_sizing(
            symbol="BTCUSDT",
            side="BUY",
            account_balance=100000.0,
            risk_per_trade=0.01,
            stop_loss_pct=0.02
        )
        self.assertTrue(sized_trade["success"])
        self.assertGreater(sized_trade["quantity"], 0)
        
        # Verify risk calculations
        max_loss = abs(
            sized_trade["entry_price"] -
            sized_trade["stop_loss_price"]
        ) * sized_trade["quantity"]
        self.assertLessEqual(max_loss, 100000.0 * 0.01)

        # Test error handling
        with self.assertRaises(ValueError):
            self.trade_exec.execute_trade(
                symbol="BTCUSDT",
                side="INVALID",
                quantity=0.001
            )

    def test_error_handling(self):
        """Test error handling in order execution."""
        # Test invalid parameters
        invalid_params = [
            {"symbol": "", "side": "BUY", "quantity": 0.001},  # Empty symbol
            {"symbol": "BTCUSDT", "side": "BUY", "quantity": -1},  # Negative quantity
            {"symbol": "BTCUSDT", "side": "INVALID", "quantity": 0.001}  # Invalid side
        ]
        
        for params in invalid_params:
            with self.subTest(**params):
                with self.assertRaises(ValueError):
                    self.order_exec.send_order(**params)

        # Test connection handling
        self.connector.connected = False
        with self.assertRaises(ConnectionError):
            self.order_exec.send_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.001
            )

        # Test order validation
        with self.assertRaises(ValueError):
            self.order_exec.send_order(
                symbol="BTCUSDT",
                side="BUY",
                type="LIMIT",
                quantity=0.001
                # Missing price for limit order
            )

    def test_trade_monitoring(self):
        """Test trade monitoring and status updates."""
        # Place test order
        order = self.order_exec.send_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            type="MARKET"
        )
        
        # Test order status retrieval
        status = self.connector.get_order_status(order["orderId"])
        self.assertEqual(status["status"], "FILLED")
        self.assertEqual(float(status["filled_qty"]), 0.001)
        
        # Test error handling for invalid order ID
        with self.assertRaises(ValueError):
            self.connector.get_order_status("invalid_id")
