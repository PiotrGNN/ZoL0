"""Minimal stub of Bybit V5 connector used for tests."""
from typing import Any, Dict

class HTTP:
    """Dummy HTTP client for mocking."""
    def get_tickers(self, category: str, symbol: str) -> Dict[str, Any]:
        return {"result": {"symbol": symbol}}
    def place_order(self, **kwargs) -> Dict[str, Any]:
        return {"result": kwargs}
    def amend_order(self, **kwargs) -> Dict[str, Any]:
        return {"result": kwargs}
    def cancel_order(self, **kwargs) -> Dict[str, Any]:
        return {"result": {"orderId": kwargs.get("order_id", "")}}
    def get_positions(self, **kwargs) -> Dict[str, Any]:
        return {"result": {"list": [{"symbol": kwargs.get("symbol", "BTCUSDT"), "size": "0.0", "side": "Buy"}]}}
    def get_wallet_balance(self) -> Dict[str, Any]:
        return {"result": {"list": [{"coin": {"USDT": {"walletBalance": "0", "availableToWithdraw": "0"}}}]}}

class BybitV5Connector:
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        self.client = HTTP()
        self._initialized = True

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        if not self._initialized:
            raise ValueError("Connector not initialized")
        data = self.client.get_tickers(category="linear", symbol=symbol)
        return data.get("result", {})

    def place_order(self, **kwargs) -> Dict[str, Any]:
        return self.client.place_order(**kwargs).get("result", {})

    def modify_order(self, **kwargs) -> Dict[str, Any]:
        return self.client.amend_order(**kwargs).get("result", {})

    def cancel_order(self, **kwargs) -> Dict[str, Any]:
        return self.client.cancel_order(**kwargs).get("result", {})

    def get_positions(self, symbol: str) -> Any:
        return self.client.get_positions(symbol=symbol).get("result", {}).get("list", [])

    def get_balances(self) -> Dict[str, Any]:
        data = self.client.get_wallet_balance().get("result", {}).get("list", [])
        if data and isinstance(data[0], dict):
            return {"USDT": data[0].get("coin", {}).get("USDT", {}).get("walletBalance", "0")}
        return {}

    def cancel_all_orders(self, symbol: str) -> None:
        pass
