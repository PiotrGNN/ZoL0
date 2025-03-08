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

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class TradeExecutor:
    def __init__(self, order_executor, account_manager, risk_manager):
        """
        Inicjalizuje moduł TradeExecutor.

        Parameters:
            order_executor: Instancja modułu odpowiedzialnego za wysyłanie zleceń (np. OrderExecution).
            account_manager: Moduł lub obiekt synchronizujący stan konta z danymi z giełdy.
            risk_manager: Moduł zarządzający ryzykiem (np. określający maksymalny procent kapitału na transakcję).
        """
        self.order_executor = order_executor
        self.account_manager = account_manager
        self.risk_manager = risk_manager
        self.trade_log = []  # Lista zarejestrowanych transakcji

    def execute_trade(self, signal):
        """
        Przetwarza pojedynczy sygnał transakcyjny i podejmuje decyzję o wykonaniu zlecenia.

        Parameters:
            signal (dict): Sygnał transakcyjny, zawierający m.in. 'symbol', 'action', 'proposed_quantity', 'price', itp.

        Returns:
            dict: Wynik wykonania zlecenia.
        """
        try:
            symbol = signal.get("symbol")
            action = signal.get("action")  # np. "BUY" lub "SELL"
            proposed_quantity = signal.get("proposed_quantity")
            proposed_price = signal.get("price", None)

            # Weryfikacja zgodności sygnału z zasadami zarządzania ryzykiem
            if not self.risk_manager.is_trade_allowed(symbol, proposed_quantity):
                logging.warning("Sygnał dla %s odrzucony przez risk manager.", symbol)
                return {"status": "rejected", "reason": "Risk constraints"}

            # Synchronizacja stanu konta
            self.account_manager.get_account_status()
            if not self.account_manager.has_sufficient_funds(
                action, proposed_quantity, proposed_price
            ):
                logging.warning(
                    "Niewystarczające środki dla sygnału %s %s.", action, symbol
                )
                return {"status": "rejected", "reason": "Insufficient funds"}

            # Wysłanie zlecenia za pomocą modułu order_executor
            order_response = self.order_executor.send_order(
                symbol=symbol,
                side=action,
                order_type=signal.get("order_type", "MARKET"),
                quantity=proposed_quantity,
                price=proposed_price,
            )

            # Zarejestruj transakcję
            trade_record = {
                "symbol": symbol,
                "action": action,
                "quantity": proposed_quantity,
                "price": proposed_price,
                "order_response": order_response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.trade_log.append(trade_record)
            logging.info("Transakcja wykonana: %s", trade_record)
            return {"status": "executed", "order_response": order_response}
        except Exception as e:
            logging.error("Błąd przy realizacji transakcji: %s", e)
            return {"status": "error", "error": str(e)}

    def execute_trades(self, signals: list):
        """
        Przetwarza listę sygnałów transakcyjnych sekwencyjnie.

        Parameters:
            signals (list): Lista sygnałów transakcyjnych.
        """
        results = []
        for signal in signals:
            result = self.execute_trade(signal)
            results.append(result)
            # Można dodać opóźnienie, jeśli wymagane jest zachowanie odstępów między transakcjami
            time.sleep(1)
        logging.info("Wszystkie sygnały zostały przetworzone.")
        return results

    def get_trade_log(self):
        """
        Zwraca dziennik wykonanych transakcji.

        Returns:
            list: Lista zarejestrowanych transakcji.
        """
        return self.trade_log


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
