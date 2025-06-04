class SimulatedBybitConnector:
    """Simplified simulated Bybit connector."""
    def __init__(self, api_key="", api_secret="", use_testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet

    def create_order(self, symbol, order_type, side, quantity, price=None):
        return {"success": True, "order_id": "SIM123", "status": "NEW"}

    def cancel_order(self, order_id):
        return {"success": True, "order_id": order_id}
