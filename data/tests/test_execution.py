"""
test_execution.py
-----------------
Testy jednostkowe dla modułów:
- exchange_connector.py
- order_execution.py
- trade_executor.py

Testy weryfikują poprawność komunikacji z giełdą (przy użyciu mocków lub sandbox), obsługę błędów, wysyłanie zleceń oraz synchronizację między modułami.
"""

import os
import unittest
import logging
import numpy as np
import pandas as pd

# Konfiguracja logowania do testów
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Zakładamy, że moduły znajdują się w odpowiednich folderach
from exchange_connector import ExchangeConnector
from order_execution import OrderExecution
from trade_executor import TradeExecutor

class DummyExchangeConnector(ExchangeConnector):
    """Dummy connector symulujący interakcję z API giełdowym."""
    def __init__(self):
        # Używamy przykładowych kluczy, ale nie wykonujemy rzeczywistych zapytań
        super().__init__(exchange="binance", api_key="dummy_key", api_secret="dummy_secret", base_url="https://api.binance.com")
    
    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        # Zwracamy przykładowe dane dla testów
        if endpoint == "/api/v3/klines":
            # Symulujemy dane świecowe
            return [[1620000000000, "30000", "31000", "29500", "30500", "1000", "1620003599999", "30500000", "100", "500", "15250000", "0"]]
        elif endpoint == "/api/v3/order":
            # Symulujemy odpowiedź zlecenia
            return {"orderId": 12345, "status": "FILLED", "symbol": params.get("symbol", "UNKNOWN")}
        elif endpoint == "/api/v3/order" and method == "GET":
            # Symulujemy status zlecenia
            return {"orderId": params.get("orderId"), "status": "FILLED"}
        else:
            return {}

class DummyOrderExecution(OrderExecution):
    """Dummy OrderExecution wykorzystujący DummyExchangeConnector."""
    def __init__(self, connector):
        super().__init__(connector, max_retries=1, retry_delay=0.1)

class DummyAccountManager:
    """Dummy manager konta do symulacji stanu konta."""
    def get_account_status(self):
        return {"balance": 10000}
    def has_sufficient_funds(self, action, quantity, price):
        return True

class DummyRiskManager:
    """Dummy risk manager symulujący akceptację transakcji."""
    def is_trade_allowed(self, symbol, quantity):
        return True

class DummyTradeExecutor(TradeExecutor):
    """Dummy TradeExecutor używający dummy komponentów."""
    def __init__(self):
        dummy_connector = DummyExchangeConnector()
        dummy_order_exec = DummyOrderExecution(dummy_connector)
        dummy_account = DummyAccountManager()
        dummy_risk = DummyRiskManager()
        super().__init__(dummy_order_exec, dummy_account, dummy_risk)

class TestExecutionModules(unittest.TestCase):
    def setUp(self):
        self.connector = DummyExchangeConnector()
        self.order_exec = DummyOrderExecution(self.connector)
        self.trade_exec = DummyTradeExecutor()

    def test_exchange_connector_get_market_data(self):
        data = self.connector.get_market_data("BTCUSDT", interval="1m", limit=5)
        self.assertIsInstance(data, list, "Dane z giełdy powinny być listą.")
        self.assertGreaterEqual(len(data), 1, "Lista danych nie powinna być pusta.")

    def test_order_execution_send_order(self):
        # Testujemy wysyłkę zlecenia MARKET
        order_response = self.order_exec.send_order("BTCUSDT", side="BUY", order_type="MARKET", quantity=0.001)
        self.assertIn("orderId", order_response, "Odpowiedź zlecenia powinna zawierać orderId.")

    def test_trade_executor_execute_trade(self):
        # Testujemy wykonanie transakcji przy użyciu dummy sygnałów
        signal = {"symbol": "BTCUSDT", "action": "BUY", "proposed_quantity": 0.001, "price": 30500, "order_type": "MARKET"}
        result = self.trade_exec.execute_trade(signal)
        self.assertIn("status", result, "Wynik transakcji powinien zawierać status.")
        self.assertEqual(result["status"], "executed", "Transakcja powinna być wykonana (executed).")

    def test_trade_executor_execute_trades(self):
        signals = [
            {"symbol": "BTCUSDT", "action": "BUY", "proposed_quantity": 0.001, "price": 30500, "order_type": "MARKET"},
            {"symbol": "ETHUSDT", "action": "SELL", "proposed_quantity": 0.01, "order_type": "MARKET"}
        ]
        results = self.trade_exec.execute_trades(signals)
        self.assertEqual(len(results), 2, "Powinny być przetworzone dwa sygnały transakcyjne.")

if __name__ == "__main__":
    unittest.main()
