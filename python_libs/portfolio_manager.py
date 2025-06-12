class PortfolioManager:
    """Minimal portfolio manager used for tests."""
    def __init__(self, db_manager=None, config=None, initial_balance=0.0, currency="USDT", mode="simulated"):
        self.db_manager = db_manager
        self.config = config or {}
        self.balance = float(initial_balance)
        self.currency = currency
        self.mode = mode

    def get_balance(self):
        return {"balance": self.balance, "currency": self.currency}
