
import websocket
import threading
import time
import json
import logging
import random
from typing import Dict, List, Callable, Optional, Any, Union
from queue import Queue

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/websocket.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("websocket")

class WebSocketManager:
    """
    Zaawansowany menedżer połączeń WebSocket 
    z automatycznym reconnect, obsługą błędów i kolejkami wiadomości
    """
    
    def __init__(self, url: str, name: str = "default"):
        """
        Inicjalizacja menedżera WebSocket
        
        Args:
            url (str): Adres URL WebSocket
            name (str): Nazwa dla identyfikacji połączenia w logach
        """
        self.url = url
        self.name = name
        self.ws = None
        self.is_connected = False
        self.should_reconnect = True
        
        # Kolejka wiadomości
        self.queue = Queue()
        self.message_processors = []
        
        # Parametry reconnecta
        self.reconnect_count = 0
        self.max_reconnect = 10
        self.reconnect_interval = 5  # Początkowy interwał w sekundach
        self.max_reconnect_interval = 300  # Maksymalny interwał (5 minut)
        
        # Subskrypcje do odnowienia po reconnect
        self.subscriptions = set()
        
        # Thread management
        self.ws_thread = None
        self.processor_thread = None
        self.is_processor_running = False
        
        # Statystyki
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'errors': 0,
            'reconnects': 0,
            'last_message_time': None,
            'connection_time': None
        }
        
        self.lock = threading.Lock()
        
        logger.info(f"WebSocketManager '{name}' zainicjalizowany z URL: {url}")
    
    def _on_message(self, ws, message):
        """Obsługa otrzymanej wiadomości"""
        try:
            with self.lock:
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = time.time()
                
            # Dodanie wiadomości do kolejki
            self.queue.put(message)
            
        except Exception as e:
            logger.error(f"Błąd w obsłudze wiadomości WebSocket '{self.name}': {e}")
    
    def _on_error(self, ws, error):
        """Obsługa błędów WebSocket"""
        with self.lock:
            self.stats['errors'] += 1
            
        logger.error(f"Błąd WebSocket '{self.name}': {error}")
    
    def _on_close(self, ws, close_status_code, close_reason):
        """Obsługa zamknięcia połączenia"""
        self.is_connected = False
        logger.warning(f"WebSocket '{self.name}' zamknięty: kod={close_status_code}, powód={close_reason}")
        
        # Automatyczny reconnect
        if self.should_reconnect:
            self._schedule_reconnect()
    
    def _on_open(self, ws):
        """Obsługa otwarcia połączenia"""
        self.is_connected = True
        self.reconnect_count = 0
        
        with self.lock:
            self.stats['connection_time'] = time.time()
            
        logger.info(f"WebSocket '{self.name}' połączony")
        
        # Odnów subskrypcje
        self._renew_subscriptions()
    
    def _renew_subscriptions(self):
        """Odnawia wszystkie subskrypcje po reconnect"""
        if not self.subscriptions:
            return
            
        logger.info(f"Odnawianie {len(self.subscriptions)} subskrypcji dla '{self.name}'")
        
        for subscription in self.subscriptions:
            self.send(subscription)
    
    def _schedule_reconnect(self):
        """Planuje automatyczny reconnect z wykładniczym backoff"""
        if self.reconnect_count >= self.max_reconnect:
            logger.error(f"Przekroczono maksymalną liczbę prób połączenia ({self.max_reconnect}) dla '{self.name}'")
            return
            
        # Oblicz interwał z jitter (losowe wahanie)
        interval = min(self.reconnect_interval * (2 ** self.reconnect_count), self.max_reconnect_interval)
        jitter = random.uniform(0.8, 1.2)
        reconnect_time = interval * jitter
        
        self.reconnect_count += 1
        
        with self.lock:
            self.stats['reconnects'] += 1
            
        logger.info(f"Próba ponownego połączenia '{self.name}' za {reconnect_time:.1f}s (próba {self.reconnect_count}/{self.max_reconnect})")
        
        # Zaplanuj reconnect
        threading.Timer(reconnect_time, self.connect).start()
    
    def connect(self):
        """Nawiązuje połączenie WebSocket"""
        if self.is_connected:
            logger.warning(f"WebSocket '{self.name}' jest już połączony")
            return True
            
        try:
            # Zamknij istniejące połączenie
            if self.ws is not None:
                self.ws.close()
                
            # Utwórz nowe połączenie
            websocket.enableTrace(False)
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Uruchom WebSocket w osobnym wątku
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Uruchom procesor wiadomości, jeśli nie działa
            if not self.is_processor_running:
                self._start_message_processor()
                
            # Poczekaj na połączenie
            timeout = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Błąd podczas łączenia z WebSocket '{self.name}': {e}")
            return False
    
    def _start_message_processor(self):
        """Uruchamia wątek przetwarzający wiadomości z kolejki"""
        def processor_job():
            self.is_processor_running = True
            logger.info(f"Uruchomiono procesor wiadomości dla '{self.name}'")
            
            while self.should_reconnect:  # Używamy tej samej flagi co do reconnect
                try:
                    # Pobierz wiadomość z kolejki (z timeoutem)
                    try:
                        message = self.queue.get(timeout=1)
                    except:
                        continue
                        
                    # Przetwórz wiadomość przez wszystkie zarejestrowane procesory
                    for processor in self.message_processors:
                        try:
                            processor(message)
                        except Exception as e:
                            logger.error(f"Błąd w procesorze wiadomości '{self.name}': {e}")
                            
                    # Oznacz zadanie jako wykonane
                    self.queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Błąd w procesorze wiadomości '{self.name}': {e}")
                    time.sleep(1)
                    
            logger.info(f"Zatrzymano procesor wiadomości dla '{self.name}'")
            self.is_processor_running = False
            
        self.processor_thread = threading.Thread(target=processor_job)
        self.processor_thread.daemon = True
        self.processor_thread.start()
    
    def add_message_processor(self, processor: Callable[[str], None]):
        """
        Dodaje funkcję przetwarzającą wiadomości
        
        Args:
            processor (callable): Funkcja przyjmująca wiadomość jako argument
        """
        self.message_processors.append(processor)
    
    def send(self, message: Union[str, Dict]) -> bool:
        """
        Wysyła wiadomość przez WebSocket
        
        Args:
            message (str | dict): Wiadomość do wysłania (str lub dict konwertowany do JSON)
            
        Returns:
            bool: Status wysłania
        """
        if not self.is_connected:
            logger.warning(f"Nie można wysłać wiadomości - WebSocket '{self.name}' nie jest połączony")
            return False
            
        try:
            # Jeśli wiadomość jest słownikiem, konwertuj do JSON
            if isinstance(message, dict):
                message = json.dumps(message)
                
            self.ws.send(message)
            
            with self.lock:
                self.stats['messages_sent'] += 1
                
            return True
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania wiadomości przez WebSocket '{self.name}': {e}")
            return False
    
    def subscribe(self, subscription):
        """
        Dodaje subskrypcję do listy (do odnowienia po reconnect)
        i wysyła ją
        
        Args:
            subscription (str | dict): Wiadomość subskrypcji
        
        Returns:
            bool: Status subskrypcji
        """
        # Dodaj do listy subskrypcji do odnowienia
        original_subscription = subscription
        
        # Konwertuj do JSON, jeśli to słownik
        if isinstance(subscription, dict):
            subscription = json.dumps(subscription)
            
        # Dodaj do listy subskrypcji
        self.subscriptions.add(subscription)
        
        # Wyślij, jeśli połączony
        if self.is_connected:
            return self.send(original_subscription)
        else:
            logger.info(f"WebSocket '{self.name}' nie jest połączony. Subskrypcja zostanie wysłana po połączeniu.")
            return self.connect()
    
    def unsubscribe(self, subscription):
        """
        Usuwa subskrypcję z listy i wysyła wiadomość
        
        Args:
            subscription (str | dict): Wiadomość anulowania subskrypcji
        
        Returns:
            bool: Status anulowania subskrypcji
        """
        # Usuń z listy subskrypcji
        if isinstance(subscription, dict):
            subscription_str = json.dumps(subscription)
        else:
            subscription_str = subscription
            
        # Usuń subskrypcję (jeśli istnieje)
        if subscription_str in self.subscriptions:
            self.subscriptions.remove(subscription_str)
            
        # Wyślij anulowanie subskrypcji
        if self.is_connected:
            return self.send(subscription)
        else:
            logger.warning(f"WebSocket '{self.name}' nie jest połączony. Nie można anulować subskrypcji.")
            return False
    
    def disconnect(self):
        """Rozłącza WebSocket i zatrzymuje wątki"""
        self.should_reconnect = False
        
        if self.ws is not None:
            self.ws.close()
            
        self.is_connected = False
        
        # Poczekaj na zakończenie wątków
        if self.ws_thread is not None:
            self.ws_thread.join(timeout=1)
            
        if self.processor_thread is not None:
            self.processor_thread.join(timeout=1)
            
        logger.info(f"WebSocket '{self.name}' rozłączony")
    
    def get_stats(self):
        """Zwraca statystyki WebSocket"""
        with self.lock:
            return self.stats.copy()
    
    def is_healthy(self, max_time_without_message=60):
        """
        Sprawdza, czy połączenie WebSocket jest zdrowe
        
        Args:
            max_time_without_message (int): Maksymalny czas bez wiadomości (w sekundach)
            
        Returns:
            bool: Status połączenia
        """
        if not self.is_connected:
            return False
            
        with self.lock:
            last_msg_time = self.stats['last_message_time']
            
        if last_msg_time is None:
            # Jeśli nie otrzymano żadnej wiadomości, sprawdź czas połączenia
            with self.lock:
                conn_time = self.stats['connection_time']
                
            if conn_time is None:
                return False
                
            # Jeśli połączono niedawno, uznaj za zdrowe
            return time.time() - conn_time < max_time_without_message
            
        # Sprawdź, czy otrzymano wiadomość w ostatnim czasie
        return time.time() - last_msg_time < max_time_without_message

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja menedżera WebSocket
    ws_manager = WebSocketManager(
        url="wss://stream.bybit.com/v5/public/spot",
        name="bybit_spot"
    )
    
    # Funkcja przetwarzająca wiadomości
    def process_message(message):
        try:
            data = json.loads(message)
            print(f"Otrzymano: {json.dumps(data, indent=2)}")
        except Exception as e:
            print(f"Błąd przetwarzania: {e}")
    
    # Dodanie procesora wiadomości
    ws_manager.add_message_processor(process_message)
    
    # Połączenie
    if ws_manager.connect():
        print("Połączono!")
        
        # Subskrypcja orderbooka BTC/USDT
        subscription = {
            "op": "subscribe",
            "args": ["orderbook.50.BTCUSDT"]
        }
        ws_manager.subscribe(subscription)
        
        # Działaj przez 30 sekund
        time.sleep(30)
        
        # Anuluj subskrypcję
        unsubscription = {
            "op": "unsubscribe",
            "args": ["orderbook.50.BTCUSDT"]
        }
        ws_manager.unsubscribe(unsubscription)
        
        # Rozłącz
        ws_manager.disconnect()
        
        print("Rozłączono!")
    else:
        print("Nie udało się połączyć!")
