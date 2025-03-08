"""
advanced_order_execution.py
---------------------------
Zaawansowany moduł realizacji zleceń handlowych.

Funkcjonalności:
- Obsługa różnych typów zleceń: Market, Limit, Stop-Loss, Trailing Stop.
- Mechanizmy minimalizowania poślizgu cenowego i optymalizacja kosztów transakcyjnych.
- Współpraca z modułami zarządzania ryzykiem (risk_management) i strategii (strategies).
- Integracja z API giełdy oraz obsługa zabezpieczeń przed nadmiernym ryzykiem.
- Możliwość testowania na danych historycznych oraz w środowisku rzeczywistym.
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("order_execution.log"), logging.StreamHandler()],
)


class AdvancedOrderExecution:
    """
    Klasa obsługująca składanie i zarządzanie zleceniami na giełdzie.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        max_slippage: float = 0.005,
        retry_attempts: int = 3,
    ):
        """
        Inicjalizacja modułu realizacji zleceń.

        Parameters:
            api_url (str): Adres API giełdy.
            api_key (str): Klucz API do autoryzacji.
            max_slippage (float): Maksymalny dopuszczalny poślizg cenowy (procent ceny).
            retry_attempts (int): Maksymalna liczba ponownych prób realizacji zlecenia w razie błędu.
        """
        self.api_url = api_url
        self.api_key = api_key
        self.max_slippage = max_slippage
        self.retry_attempts = retry_attempts

    def _send_request(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Wysyła żądanie do API giełdy i obsługuje błędy sieciowe.

        Parameters:
            endpoint (str): Ścieżka do API.
            payload (dict): Dane żądania.

        Returns:
            dict: Odpowiedź API lub None w przypadku błędu.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_url}/{endpoint}"

        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=5)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logging.error("Błąd API (%s) - próba %d: %s", endpoint, attempt + 1, e)
                time.sleep(2**attempt)  # Wykładnicze opóźnienie
        return None

    def place_order(
        self,
        order_type: str,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Składa zlecenie na giełdzie.

        Parameters:
            order_type (str): Typ zlecenia ("market", "limit", "stop_loss", "trailing_stop").
            symbol (str): Symbol instrumentu finansowego (np. "BTCUSDT").
            quantity (float): Ilość jednostek do kupna/sprzedaży.
            price (float, opcjonalne): Cena dla zleceń limit.
            stop_price (float, opcjonalne): Cena aktywacji dla stop-loss i trailing stop.

        Returns:
            dict: Odpowiedź API giełdy lub None w przypadku błędu.
        """
        payload = {"symbol": symbol, "quantity": quantity, "order_type": order_type}

        if order_type == "limit":
            if price is None:
                logging.error("Zlecenie LIMIT wymaga podania ceny.")
                return None
            payload["price"] = price

        elif order_type in ("stop_loss", "trailing_stop"):
            if stop_price is None:
                logging.error(
                    "Zlecenie STOP-LOSS i TRAILING STOP wymaga ceny aktywacji."
                )
                return None
            payload["stop_price"] = stop_price

        response = self._send_request("order", payload)

        if response:
            logging.info("Zlecenie złożone: %s", response)
        else:
            logging.error("Błąd składania zlecenia.")
        return response

    def cancel_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Anuluje istniejące zlecenie.

        Parameters:
            order_id (str): Identyfikator zlecenia do anulowania.

        Returns:
            dict: Odpowiedź API lub None w przypadku błędu.
        """
        response = self._send_request("cancel_order", {"order_id": order_id})

        if response:
            logging.info("Zlecenie %s anulowane.", order_id)
        else:
            logging.error("Nie udało się anulować zlecenia %s.", order_id)
        return response

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Pobiera status zlecenia.

        Parameters:
            order_id (str): Identyfikator zlecenia.

        Returns:
            dict: Odpowiedź API zawierająca status zlecenia.
        """
        response = self._send_request("order_status", {"order_id": order_id})

        if response:
            logging.info("Status zlecenia %s: %s", order_id, response)
        else:
            logging.error("Błąd pobierania statusu zlecenia %s.", order_id)
        return response

    def execute_trade(
        self,
        order_type: str,
        symbol: str,
        quantity: float,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Realizuje zlecenie z mechanizmem kontroli ryzyka.

        Parameters:
            order_type (str): Typ zlecenia ("market", "limit").
            symbol (str): Symbol instrumentu (np. "BTCUSDT").
            quantity (float): Ilość jednostek do kupna/sprzedaży.
            target_price (float, opcjonalne): Docelowa cena dla zlecenia limit.
            stop_loss (float, opcjonalne): Poziom stop-loss.

        Returns:
            dict: Odpowiedź API lub None w przypadku błędu.
        """
        current_price = self._get_market_price(symbol)
        if current_price is None:
            logging.error("Nie udało się pobrać aktualnej ceny rynkowej.")
            return None

        if order_type == "limit" and target_price:
            if abs((target_price - current_price) / current_price) > self.max_slippage:
                logging.warning(
                    "Zlecenie LIMIT nie zostało złożone: przekroczony dopuszczalny poślizg."
                )
                return None

        response = self.place_order(order_type, symbol, quantity, target_price)

        if stop_loss:
            stop_price = stop_loss
            self.place_order("stop_loss", symbol, quantity, stop_price=stop_price)

        return response

    def _get_market_price(self, symbol: str) -> Optional[float]:
        """
        Pobiera aktualną cenę rynkową instrumentu.

        Parameters:
            symbol (str): Symbol instrumentu.

        Returns:
            float: Aktualna cena rynkowa lub None w przypadku błędu.
        """
        response = self._send_request("market_price", {"symbol": symbol})

        if response and "price" in response:
            return float(response["price"])
        logging.error("Nie udało się pobrać ceny rynkowej dla %s.", symbol)
        return None


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        api_url = "https://api.exchange.example.com"
        api_key = "your_api_key_here"

        order_executor = AdvancedOrderExecution(api_url, api_key)

        # Przykładowe składanie zlecenia
        order_executor.execute_trade(
            "limit", "BTCUSDT", 0.1, target_price=50000, stop_loss=49000
        )

    except Exception as e:
        logging.error("Błąd w module AdvancedOrderExecution: %s", e)
        raise
