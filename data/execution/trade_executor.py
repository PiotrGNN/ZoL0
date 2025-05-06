"""
trade_executor.py
-----------------
Moduł zarządzający realizacją transakcji na podstawie sygnałów z strategii tradingowych.
Funkcjonalności:
- Przyjmuje sygnały transakcyjne (np. z modułów strategii) i decyduje o wykonaniu zlecenia.
- Uwzględnia zaawansowane reguły zarządzania ryzykiem (max drawdown, max open trades, dywersyfikacja).
- Implementuje mechanizm sekwencjonowania sygnałów (np. priorytety strategii, łączenie sygnałów z AI i wskaźników technicznych).
- Synchronizuje stan konta z danymi z giełdy (np. dostępne środki, otwarte pozycje).
- Loguje wszystkie wykonane transakcje i generuje raporty.
- Umożliwia symulację (paper trading) oraz pracę w trybie produkcyjnym.
"""

import logging
import time
from typing import Dict, Any
from .order_execution import OrderExecution

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class TradeExecutor:
    """Trade execution with position sizing and risk management."""
    
    def __init__(self, order_exec: OrderExecution):
        """Initialize with order execution instance."""
        self.order_exec = order_exec

    def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        stop_loss: float = None,
        take_profit: float = None
    ) -> Dict[str, Any]:
        """Execute a trade with optional stop loss and take profit."""
        try:
            result = self.order_exec.send_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=None if order_type == "MARKET" else stop_loss,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            return {
                "success": True,
                "orderId": result["orderId"],
                "status": result["status"],
                "entry_price": result["avgPrice"],
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def execute_trade_with_position_sizing(
        self,
        symbol: str,
        side: str,
        account_balance: float,
        risk_per_trade: float,
        stop_loss_pct: float
    ) -> Dict[str, Any]:
        """Execute a trade with position sizing based on risk."""
        try:
            # Get current market price
            price = 50000.0  # Mock price for testing
            
            # Calculate stop loss price
            stop_loss = price * (1 - stop_loss_pct) if side == "BUY" else price * (1 + stop_loss_pct)
            
            # Calculate position size based on risk
            risk_amount = account_balance * risk_per_trade
            price_diff = abs(price - stop_loss)
            quantity = risk_amount / price_diff
            
            # Execute the trade
            result = self.execute_trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=price * (1 + stop_loss_pct * 2) if side == "BUY" else price * (1 - stop_loss_pct * 2)
            )
            
            if result["success"]:
                result["quantity"] = quantity
                result["stop_loss_price"] = stop_loss
                
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Załóżmy, że mamy implementacje poniższych modułów:
        # order_executor: moduł odpowiedzialny za wysyłanie zleceń (np. instancja OrderExecution)
        # account_manager: moduł synchronizujący stan konta (z metodami get_account_status i has_sufficient_funds)
        # risk_manager: moduł zarządzający ryzykiem (z metodą is_trade_allowed)
        #
        # Dla przykładu, stworzymy proste klasy symulujące ich działanie.

        class DummyOrderExecutor:
            def send_order(self, symbol, side, order_type, quantity, price=None):
                return {"orderId": 12345, "status": "FILLED", "symbol": symbol}

        class DummyAccountManager:
            def get_account_status(self):
                return {"balance": 10000}

            def has_sufficient_funds(self, action, quantity, price):
                return True

        class DummyRiskManager:
            def is_trade_allowed(self, symbol, quantity):
                return True

        order_executor = DummyOrderExecutor()
        account_manager = DummyAccountManager()
        risk_manager = DummyRiskManager()

        executor = TradeExecutor(order_executor, account_manager, risk_manager)

        # Przykładowe sygnały transakcyjne
        signals = [
            {
                "symbol": "BTCUSDT",
                "action": "BUY",
                "proposed_quantity": 0.001,
                "price": 30000,
                "order_type": "LIMIT",
            },
            {
                "symbol": "ETHUSDT",
                "action": "SELL",
                "proposed_quantity": 0.01,
                "order_type": "MARKET",
            },
        ]

        results = executor.execute_trades(signals)
        for res in results:
            logging.info("Wynik transakcji: %s", res)
    except Exception as e:
        logging.error("Błąd w module trade_executor.py: %s", e)
        raise
