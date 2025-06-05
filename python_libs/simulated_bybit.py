class SimulatedBybitConnector:
    """Minimal Bybit connector stub used for tests."""
    def __init__(self, *args, **kwargs):
        self.initialized = True

    def get_market_data(self, symbol):
        return {"symbol": symbol, "price": 0}
