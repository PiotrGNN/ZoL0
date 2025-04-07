"""
WebSocket Manager - moduł odpowiedzialny za utrzymanie połączenia WebSocket
z giełdą Bybit, obsługę reconnect i logowania komunikatów.
"""

import json
import logging
import threading
import time
from typing import Callable, Dict, Optional

import websocket

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/websocket.log"), logging.StreamHandler()],
)
logger = logging.getLogger("WebSocketManager")


class WebSocketManager:
    """
    Klasa zarządzająca połączeniem WebSocket z giełdą.

    Obsługuje:
    - Automatyczne reconnect po utracie połączenia
    - Uwierzytelnianie
    - Subskrypcje na różne tematy (kursów, orderbook, etc.)
    - Callback dla przychodzących wiadomości
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        reconnect_attempts: int = 5,
        reconnect_delay: int = 5,
    ):
        """
        Inicjalizacja WebSocket Managera.

        Args:
            url: URL endpointu WebSocket giełdy
            api_key: Klucz API (opcjonalny dla publicznych endpointów)
            api_secret: Sekret API (opcjonalny dla publicznych endpointów)
            ping_interval: Interwał pingowania w sekundach
            ping_timeout: Timeout dla pinga w sekundach
            reconnect_attempts: Maksymalna liczba prób reconnectu
            reconnect_delay: Opóźnienie między próbami reconnectu w sekundach
        """
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.ws = None
        self.is_connected = False
        self.should_reconnect = True
        self.reconnect_count = 0

        self.subscriptions = set()
        self.message_callbacks = []
        self.ws_thread = None

        logger.info(f"WebSocketManager zainicjalizowany dla {url}")

    def connect(self) -> bool:
        """
        Nawiązuje połączenie WebSocket i uruchamia wątek nasłuchujący.

        Returns:
            bool: True jeśli połączenie zostało nawiązane, False w przeciwnym razie
        """
        try:
            websocket.enableTrace(False)
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_ping=self._on_ping,
                on_pong=self._on_pong,
            )

            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={
                    "ping_interval": self.ping_interval,
                    "ping_timeout": self.ping_timeout,
                },
                daemon=True,
            )
            self.ws_thread.start()

            # Czekamy na nawiązanie połączenia
            timeout = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)

            return self.is_connected
        except Exception as e:
            logger.error(f"Błąd podczas nawiązywania połączenia WebSocket: {e}")
            return False

    def disconnect(self) -> None:
        """Zamyka połączenie WebSocket."""
        self.should_reconnect = False
        if self.ws:
            self.ws.close()
        logger.info("Połączenie WebSocket zostało zamknięte")

    def subscribe(self, topics: list) -> None:
        """
        Subskrybuje tematy WebSocket.

        Args:
            topics: Lista tematów do subskrypcji
        """
        if not self.is_connected:
            logger.warning("Próba subskrypcji bez aktywnego połączenia")
            self.subscriptions.update(topics)
            return

        try:
            for topic in topics:
                if topic not in self.subscriptions:
                    subscribe_message = {
                        "op": "subscribe",
                        "args": [topic],
                    }
                    self.ws.send(json.dumps(subscribe_message))
                    self.subscriptions.add(topic)
                    logger.info(f"Zasubskrybowano temat: {topic}")
        except Exception as e:
            logger.error(f"Błąd podczas subskrypcji tematów: {e}")

    def unsubscribe(self, topics: list) -> None:
        """
        Anuluje subskrypcję tematów WebSocket.

        Args:
            topics: Lista tematów do anulowania subskrypcji
        """
        if not self.is_connected:
            logger.warning("Próba anulowania subskrypcji bez aktywnego połączenia")
            self.subscriptions -= set(topics)
            return

        try:
            for topic in topics:
                if topic in self.subscriptions:
                    unsubscribe_message = {
                        "op": "unsubscribe",
                        "args": [topic],
                    }
                    self.ws.send(json.dumps(unsubscribe_message))
                    self.subscriptions.remove(topic)
                    logger.info(f"Anulowano subskrypcję tematu: {topic}")
        except Exception as e:
            logger.error(f"Błąd podczas anulowania subskrypcji tematów: {e}")

    def add_message_callback(self, callback: Callable[[dict], None]) -> None:
        """
        Dodaje callback dla przychodzących wiadomości.

        Args:
            callback: Funkcja wywołana dla każdej przychodzącej wiadomości
        """
        self.message_callbacks.append(callback)
        logger.debug(f"Dodano nowy callback dla wiadomości (total: {len(self.message_callbacks)})")

    def send_message(self, message: Dict) -> bool:
        """
        Wysyła wiadomość przez WebSocket.

        Args:
            message: Wiadomość do wysłania (zostanie skonwertowana do JSON)

        Returns:
            bool: True jeśli wiadomość została wysłana, False w przeciwnym razie
        """
        if not self.is_connected:
            logger.warning("Próba wysłania wiadomości bez aktywnego połączenia")
            return False

        try:
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania wiadomości: {e}")
            return False

    def _on_open(self, ws) -> None:
        """Wywołane po nawiązaniu połączenia WebSocket."""
        self.is_connected = True
        self.reconnect_count = 0
        logger.info("Połączenie WebSocket nawiązane")

        # Jeśli mamy zapisane subskrypcje, ponownie je subskrybujemy
        if self.subscriptions:
            self.subscribe(list(self.subscriptions))

        # Jeśli mamy klucze API, uwierzytelniamy się
        if self.api_key and self.api_secret:
            self._authenticate()

    def _on_message(self, ws, message) -> None:
        """
        Wywołane po otrzymaniu wiadomości.

        Args:
            ws: Obiekt WebSocket
            message: Otrzymana wiadomość
        """
        try:
            data = json.loads(message)

            # Jeśli to wiadomość typu pong, ignorujemy ją w logach
            if "op" in data and data["op"] == "pong":
                return

            logger.debug(f"Odebrano wiadomość: {message[:100]}...")

            # Wywołujemy wszystkie zarejestrowane callbacki
            for callback in self.message_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Błąd w callbacku wiadomości: {e}")
        except json.JSONDecodeError:
            logger.warning(f"Otrzymano nieprawidłową wiadomość JSON: {message[:100]}...")
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania wiadomości: {e}")

    def _on_error(self, ws, error) -> None:
        """
        Wywołane po wystąpieniu błędu.

        Args:
            ws: Obiekt WebSocket
            error: Błąd
        """
        logger.error(f"Błąd WebSocket: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """
        Wywołane po zamknięciu połączenia.

        Args:
            ws: Obiekt WebSocket
            close_status_code: Kod statusu zamknięcia
            close_msg: Wiadomość zamknięcia
        """
        self.is_connected = False
        logger.info(f"Połączenie WebSocket zamknięte. Kod: {close_status_code}, Komunikat: {close_msg}")

        # Jeśli powinniśmy ponownie połączyć i nie przekroczyliśmy limitu prób
        if self.should_reconnect and self.reconnect_count < self.reconnect_attempts:
            self.reconnect_count += 1
            reconnect_delay = self.reconnect_delay * self.reconnect_count  # Exponential backoff
            logger.info(f"Próba ponownego połączenia za {reconnect_delay}s (próba {self.reconnect_count}/{self.reconnect_attempts})")

            time.sleep(reconnect_delay)
            self.connect()

    def _on_ping(self, ws, message) -> None:
        """
        Wywołane po otrzymaniu ping.

        Args:
            ws: Obiekt WebSocket
            message: Wiadomość ping
        """
        logger.debug("Otrzymano ping od serwera")

    def _on_pong(self, ws, message) -> None:
        """
        Wywołane po otrzymaniu pong.

        Args:
            ws: Obiekt WebSocket
            message: Wiadomość pong
        """
        logger.debug("Otrzymano pong od serwera")

    def _authenticate(self) -> None:
        """Uwierzytelnia połączenie z użyciem kluczy API."""
        try:
            # Implementacja uwierzytelniania zależy od giełdy
            # Przykład dla Bybit - używa innego formatu uwierzytelniania
            # To jest uproszczona wersja, w rzeczywistości wymaga generowania podpisu
            auth_message = {
                "op": "auth",
                "args": [self.api_key, int(time.time() * 1000), "signature_placeholder"],
            }
            self.ws.send(json.dumps(auth_message))
            logger.info("Wysłano żądanie uwierzytelnienia")
        except Exception as e:
            logger.error(f"Błąd podczas uwierzytelniania: {e}")


# Przykład użycia
if __name__ == "__main__":
    # URL dla testnet Bybit
    ws_url = "wss://stream-testnet.bybit.com/v5/public/spot"

    # Inicjalizacja WebSocket Managera
    ws_manager = WebSocketManager(ws_url)

    # Przykładowy callback dla wiadomości
    def handle_message(data):
        print(f"Otrzymano dane: {data}")

    # Dodajemy callback
    ws_manager.add_message_callback(handle_message)

    # Nawiązujemy połączenie
    if ws_manager.connect():
        # Subskrybujemy kanał ticker dla BTC/USDT
        ws_manager.subscribe(["tickers.BTCUSDT"])

        # Czekamy przez 60 sekund
        time.sleep(60)

        # Zamykamy połączenie
        ws_manager.disconnect()
    else:
        print("Nie udało się nawiązać połączenia WebSocket")