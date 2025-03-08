"""
test_execution.py
-----------------
Testy jednostkowe dla modułów:
  - exchange_connector.py,
  - order_execution.py,
  - trade_executor.py.

Weryfikują komunikację z giełdą (symulowaną przez dummy connector),
obsługę błędów oraz poprawną synchronizację między modułami.
"""

import logging
import unittest
from typing import Any, Dict

from data.execution.exchange_connector import ExchangeConnector
from data.execution.order_execution import OrderExecution
from data.execution.trade_executor import TradeExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DummyExchangeConnector(ExchangeConnector):
    """
    Dummy connector symulujący interakcję z API giełdowym.
    Zwraca z góry ustalone dane w metodzie _request, zamiast wywoływać prawdziwe REST API.
    """

    def __init__(self) -> None:
        super().__init__(
            exchange="binance",
            api_key="dummy_key",
            api_secret="dummy_secret",
            base_url="https://api.binance.com"
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        Metoda symulująca odpowiedź z API giełdowego.
        Zwraca uproszczone dane klines lub wynik zlecenia.
        """
        if endpoint == "/api/v3/klines":
            return [{
                0: 1620000000000,
                1: "30000",
                2: "31000",
                3: "29500",
                4: "30500",
                5: "1000",
                6: "1620003599999",
                7: "30500000",
                8: "100",
                9: "500",
                10: "15250000",
                11: "0",
            }]
        elif endpoint == "/api/v3/order":
            return {
                "orderId": 12345,
                "status": "FILLED",
                "symbol": params.get("symbol", "UNKNOWN"),
            }
        return {}


class DummyOrderExecution(OrderExecution):
    """
    Dummy OrderExecution wykorzystujący DummyExchangeConnector.
    Nie wywołuje realnego API, a jedynie symuluje proces wysyłania zleceń.
    """

    def __init__(self, connector: ExchangeConnector) -> None:
        super().__init__(connector, max_retries=1, retry_delay=0.1)


class DummyTradeExecutor(TradeExecutor):
    """
    Dummy TradeExecutor używający dummy komponentów:
      - DummyExchangeConnector (symuluje API),
      - DummyAccountManager (stan konta, sprawdzenie środków),
      - DummyRiskManager (zezwolenie na trade).
    """

    def __init__(self) -> None:
        dummy_connector = DummyExchangeConnector()
        dummy_order_exec = DummyOrderExecution(dummy_connector)

        # Klasa managera konta z minimalną funkcjonalnością (balans, sprawdzenie środków).
        dummy_account_manager = type(
            "DummyAccountManager",
            (),
            {
                "get_account_status": lambda self: {"balance": 10000},
                "has_sufficient_funds": lambda self, symbol, quantity, price=None: True
            }
        )()

        # Klasa risk_manager z minimalną funkcjonalnością: is_trade_allowed zawsze zwraca True.
        dummy_risk_manager = type(
            "DummyRiskManager",
            (),
            {
                "is_trade_allowed": lambda self, symbol, quantity: True
            }
        )()

        super().__init__(dummy_order_exec, dummy_account_manager, dummy_risk_manager)


class TestExecutionModules(unittest.TestCase):
    """
    Testy modułów wykonawczych:
      - ExchangeConnector (DummyExchangeConnector),
      - OrderExecution (DummyOrderExecution),
      - TradeExecutor (DummyTradeExecutor).
    """

    def setUp(self) -> None:
        """Inicjalizacja dummy connectora, order execution i trade executora."""
        self.connector = DummyExchangeConnector()
        self.order_exec = DummyOrderExecution(self.connector)
        self.trade_exec = DummyTradeExecutor()

    def test_exchange_connector_get_market_data(self) -> None:
        """
        Test pobierania danych rynkowych z metody get_market_data.
        Oczekuje listy z co najmniej jednym elementem.
        """
        data: Any = self.connector.get_market_data("BTCUSDT", interval="1m", limit=5)
        self.assertIsInstance(data, list, "Metoda powinna zwrócić listę danych.")
        self.assertGreaterEqual(len(data), 1, "Lista danych nie powinna być pusta.")

    def test_order_execution_send_order(self) -> None:
        """
        Test wysyłania zlecenia przez metodę send_order.
        Sprawdza, czy odpowiedź zawiera 'orderId'.
        """
        order_response: Dict[str, Any] = self.order_exec.send_order(
            "BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.001
        )
        self.assertIn("orderId", order_response, "Odpowiedź powinna zawierać 'orderId'.")

    def test_trade_executor_execute_trade(self) -> None:
        """
        Test wykonania pojedynczej transakcji przez TradeExecutor.
        Oczekujemy statusu 'executed' przy braku błędów
        (DummyAccountManager i DummyRiskManager są zawsze pozytywne).
        """
        signal = {
            "symbol": "BTCUSDT",
            "action": "BUY",
            "proposed_quantity": 0.001,
            "price": 30500,
            "order_type": "MARKET",
        }
        result: Dict[str, Any] = self.trade_exec.execute_trade(signal)
        self.assertIn("status", result, "Wynik powinien zawierać klucz 'status'.")
        self.assertEqual(
            result["status"], "executed",
            "Powinniśmy uzyskać status 'executed' zamiast 'error'."
        )

    def test_trade_executor_execute_trades(self) -> None:
        """
        Test wykonania listy transakcji przez TradeExecutor.
        Oczekujemy, że obie transakcje zwrócą prawidłowy status w liście wyników.
        """
        signals = [
            {
                "symbol": "BTCUSDT",
                "action": "BUY",
                "proposed_quantity": 0.001,
                "price": 30500,
                "order_type": "MARKET",
            },
            {
                "symbol": "ETHUSDT",
                "action": "SELL",
                "proposed_quantity": 0.01,
                "order_type": "MARKET",
            },
        ]
        results = self.trade_exec.execute_trades(signals)
        self.assertEqual(len(results), 2, "Powinny zostać przetworzone 2 transakcje.")


if __name__ == "__main__":
    unittest.main()
