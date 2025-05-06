"""
mock_order_execution.py
----------------------
Mock implementation of order execution for testing.
"""

import time
import random
from typing import Dict, Any


class MockOrderExecution:
    """Mock version of OrderExecution for testing."""
    
    def __init__(self):
        """Initialize mock executor."""
        self.orders = {}
    
    def send_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Mock order sending with realistic latency simulation."""
        # Simulate network latency (5-20ms)
        time.sleep(random.uniform(0.005, 0.02))
        
        order_id = f"TEST-{int(time.time())}-{random.randint(1000, 9999)}"
        order = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price,
            "status": "FILLED",
            "timestamp": int(time.time() * 1000)
        }
        self.orders[order_id] = order
        return order
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get mock order details."""
        return self.orders.get(order_id, {"status": "NOT_FOUND"})