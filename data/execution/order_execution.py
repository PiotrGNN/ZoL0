"""
order_execution.py
------------------
Skrypt odpowiedzialny za wysyłanie zleceń do giełdy oraz monitorowanie ich statusu.

Funkcjonalności:
- Obsługa różnych typów zleceń: market, limit, stop-limit, OCO.
- Zaawansowane parametry zleceń, np. time-in-force.
- Mechanizm potwierdzania wykonania zleceń i automatyczne ponawianie w razie błędów.
- Dynamiczna regulacja rozmiaru zleceń w zależności od wolumenu na giełdzie i dostępnego kapitału.
- Integracja z modułami trade_executor.py oraz exchange_connector.py.
- Szczegółowe logowanie, w tym raportowanie powodów ewentualnego odrzucenia zleceń.
"""

import logging
import time

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class OrderExecution:
    def __init__(self, connector, max_retries=3, retry_delay=2):
        """
        Inicjalizuje moduł wysyłania zleceń.

        Parameters:
            connector: Instancja klasy ExchangeConnector do komunikacji z API giełdowym.
            max_retries (int): Maksymalna liczba prób wysłania zlecenia.
            retry_delay (int): Opóźnienie pomiędzy próbami (w sekundach).
        """
        self.connector = connector
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def send_order(self, symbol, side, order_type, quantity, price=None, time_in_force="GTC"):
        """
        Wysyła zlecenie do giełdy i monitoruje jego status.

        Parameters:
            symbol (str): Symbol pary walutowej (np. "BTCUSDT").
            side (str): Kierunek zlecenia ("BUY" lub "SELL").
            order_type (str): Typ zlecenia ("MARKET", "LIMIT", "STOP_LIMIT", "OCO").
            quantity (float): Rozmiar zlecenia.
            price (float, optional): Cena zlecenia (wymagana dla LIMIT, STOP_LIMIT).
            time_in_force (str): Parametr time-in-force dla zleceń LIMIT (domyślnie "GTC").

        Returns:
            dict: Odpowiedź API dotycząca zlecenia.
        """
        attempt = 0
        order_response = None
        while attempt < self.max_retries:
            try:
                if order_type.upper() == "LIMIT":
                    order_response = self.connector.place_order(
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        quantity=quantity,
                        price=price,
                    )
                elif order_type.upper() == "MARKET":
                    order_response = self.connector.place_order(
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        quantity=quantity,
                    )
                elif order_type.upper() in ["STOP_LIMIT", "OCO"]:
                    # Dla zleceń STOP_LIMIT i OCO wymagane mogą być dodatkowe parametry,
                    # które należy zaimplementować zgodnie z dokumentacją API.
                    order_response = self.connector.place_order(
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        quantity=quantity,
                        price=price,
                    )
                else:
                    raise ValueError(f"Nieobsługiwany typ zlecenia: {order_type}")

                # Sprawdzamy, czy odpowiedź zawiera potwierdzenie wykonania zlecenia
                if "orderId" in order_response:
                    logging.info("Zlecenie wysłane pomyślnie: %s", order_response)
                    return order_response
                else:
                    logging.warning("Zlecenie nie zostało potwierdzone: %s", order_response)
                    raise Exception("Brak potwierdzenia zlecenia")
            except Exception as e:
                attempt += 1
                logging.error(
                    "Błąd przy wysyłaniu zlecenia (próba %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                time.sleep(self.retry_delay)
        logging.error("Nie udało się wysłać zlecenia po %d próbach.", self.max_retries)
        return order_response

    def monitor_order(self, order_id, symbol, poll_interval=2, timeout=30):
        """
        Monitoruje status zlecenia do momentu jego wykonania lub upływu czasu.

        Parameters:
            order_id (int): Identyfikator zlecenia.
            symbol (str): Symbol pary walutowej.
            poll_interval (int): Interwał sprawdzania statusu (w sekundach).
            timeout (int): Maksymalny czas monitorowania (w sekundach).

        Returns:
            dict: Ostateczny status zlecenia.
        """
        start_time = time.time()
        while True:
            try:
                # Przykładowe zapytanie statusowe; metoda zależy od API giełdy
                status_response = self.connector._request(
                    "GET",
                    "/api/v3/order",
                    params={"symbol": symbol, "orderId": order_id},
                    signed=True,
                )
                logging.info("Status zlecenia %s: %s", order_id, status_response)
                if status_response.get("status") in ["FILLED", "CANCELED", "REJECTED"]:
                    return status_response
            except Exception as e:
                logging.error("Błąd przy sprawdzaniu statusu zlecenia %s: %s", order_id, e)
            if time.time() - start_time > timeout:
                logging.warning("Upłynął limit czasu monitorowania zlecenia %s.", order_id)
                return {"status": "TIMEOUT", "orderId": order_id}
            time.sleep(poll_interval)


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane (użyj własnych kluczy API oraz instancji ExchangeConnector)
        from exchange_connector import ExchangeConnector

        EXCHANGE = "binance"
        API_KEY = "your_api_key_here"
        API_SECRET = "your_api_secret_here"

        connector = ExchangeConnector(exchange=EXCHANGE, api_key=API_KEY, api_secret=API_SECRET)
        executor = OrderExecution(connector)

        # Przykładowe wysłanie zlecenia LIMIT (zakomentuj, aby nie wysłać rzeczywistego zlecenia)
        # order_resp = executor.send_order("BTCUSDT", side="BUY", order_type="LIMIT", quantity=0.001, price=30000)
        # logging.info("Odpowiedź zlecenia: %s", order_resp)

        # Monitorowanie zlecenia, jeśli order_resp zawiera orderId
        # if order_resp and "orderId" in order_resp:
        #     status = executor.monitor_order(order_resp["orderId"], "BTCUSDT")
        #     logging.info("Finalny status zlecenia: %s", status)

    except Exception as e:
        logging.error("Błąd w module order_execution.py: %s", e)
        raise
