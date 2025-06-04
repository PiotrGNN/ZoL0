class BybitV5Connector:
    """Minimal stub for Bybit V5 connector used in tests."""
    def __init__(self, api_key="", api_secret="", testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    def initialize(self):
        self._initialized = True

    # Synchronous methods
    def get_market_data(self, symbol):
        return {"symbol": symbol, "last_price": "0", "volume24h": "0"}

    def place_order(self, symbol, side, order_type, qty, price=None):
        return {"order_id": "123456", "symbol": symbol, "status": "NEW"}

    def modify_order(self, symbol, order_id, price=None):
        return {"orderId": order_id, "symbol": symbol, "status": "MODIFIED"}

    def cancel_order(self, order_id):
        return {"orderId": order_id}

    def get_positions(self, symbol):
        return [{"symbol": symbol, "size": "0", "side": "Buy"}]

    def get_balances(self):
        return {"USDT": 0}

    def get_order_book(self, symbol):
        return {"bids": [], "asks": []}

    def get_recent_trades(self, symbol):
        return []

    # Async stubs
    async def get_market_data_async(self, symbol):
        return {"symbol": symbol, "last_price": "0"}

    async def place_order_async(self, symbol, side, order_type, qty, price=None):
        return {"order_id": "123"}

    async def cancel_order_async(self, order_id):
        return {"order_id": order_id}
