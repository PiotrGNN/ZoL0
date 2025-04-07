
import time
import hmac
import hashlib
import json
import logging
import requests
import websocket
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union, Any
from queue import Queue
from urllib.parse import urlencode

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bybit_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bybit_api")

# Importowanie modułu config_loader
try:
    from config.config_loader import config_loader
except ImportError:
    logging.error("Nie można zaimportować config_loader. Sprawdź konfigurację.")
    raise

class BytbitRateLimiter:
    """
    Klasa zarządzająca ograniczeniami zapytań (rate limity) dla Bybit API
    """
    def __init__(self, max_requests_per_sec=5, max_requests_per_min=60):
        self.max_requests_per_sec = max_requests_per_sec
        self.max_requests_per_min = max_requests_per_min
        self.second_requests = []
        self.minute_requests = []
        self.lock = threading.Lock()
    
    def can_request(self) -> bool:
        """Sprawdza, czy można wykonać kolejne zapytanie"""
        with self.lock:
            current_time = time.time()
            
            # Usuwanie starych zapytań
            self.second_requests = [t for t in self.second_requests if current_time - t < 1]
            self.minute_requests = [t for t in self.minute_requests if current_time - t < 60]
            
            # Sprawdzanie limitów
            if (len(self.second_requests) >= self.max_requests_per_sec or
                len(self.minute_requests) >= self.max_requests_per_min):
                return False
            
            # Dodawanie nowego zapytania
            self.second_requests.append(current_time)
            self.minute_requests.append(current_time)
            return True
    
    def wait_if_needed(self):
        """Czeka, jeśli przekroczono limit zapytań"""
        while not self.can_request():
            time.sleep(0.1)


class BybitConnector:
    """
    Klasa do komunikacji z Bybit API (REST i WebSocket)
    """
    def __init__(self, api_key=None, api_secret=None, test_mode=None):
        """
        Inicjalizacja konektora Bybit
        
        Args:
            api_key (str, optional): Klucz API
            api_secret (str, optional): Sekret API
            test_mode (bool, optional): Tryb testowy (testnet)
        """
        # Jeśli nie podano parametrów, pobierz z konfiguracji
        if api_key is None or api_secret is None:
            api_keys = config_loader.get_api_keys('bybit')
            if api_keys:
                api_key = api_keys.get('api_key')
                api_secret = api_keys.get('api_secret')
            else:
                logger.warning("Nie znaleziono kluczy API Bybit w konfiguracji")
        
        # Tryb testowy
        if test_mode is None:
            test_mode = config_loader.get_test_mode()
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
        
        # Pobierz endpointy z konfiguracji
        endpoints = config_loader.get_bybit_endpoints()
        self.rest_endpoint = endpoints.get('rest_api')
        self.ws_endpoint = endpoints.get('websocket')
        
        # Ogranicznik przepustowości (rate limiter)
        self.rate_limiter = BytbitRateLimiter()
        
        # Dla WebSocket
        self.ws = None
        self.ws_connected = False
        self.ws_subscriptions = set()
        self.ws_callbacks = {}
        self.ws_reconnect_count = 0
        self.ws_max_reconnect = 10
        self.ws_reconnect_delay = 5  # sekundy
        
        # Informacje diagnostyczne
        self.last_response_time = 0
        self.request_stats = {
            'success': 0,
            'error': 0,
            'retry': 0,
            'avg_response_time': 0
        }
        
        logger.info(f"BybitConnector zainicjalizowany w trybie {'TESTOWYM' if self.test_mode else 'PRODUKCYJNYM'}")

    def _generate_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """
        Generuje podpis do autoryzacji z Bybit API
        
        Args:
            params (dict): Parametry zapytania
            timestamp (int): Timestamp (ms)
            
        Returns:
            str: Wygenerowany podpis HMAC
        """
        param_str = str(timestamp) + self.api_key
        
        # Dla zapytań GET
        if isinstance(params, dict):
            params_sorted = dict(sorted(params.items()))
            if params_sorted:
                param_str += urlencode(params_sorted)
        
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(param_str, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def _make_request(self, method: str, endpoint: str, params: Dict = None, retry_count: int = 3) -> Dict:
        """
        Wykonuje zapytanie do Bybit API z obsługą błędów i ponownych prób
        
        Args:
            method (str): Metoda HTTP (GET, POST, DELETE)
            endpoint (str): Endpoint API
            params (dict, optional): Parametry zapytania
            retry_count (int): Liczba ponownych prób w przypadku błędów
            
        Returns:
            dict: Odpowiedź API
        """
        if params is None:
            params = {}
        
        # Czekaj, jeśli przekroczono limit zapytań
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.rest_endpoint}{endpoint}"
        headers = {}
        
        # Dodaj nagłówki autoryzacji, jeśli podano klucze API
        if self.api_key and self.api_secret:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(params, timestamp)
            
            headers = {
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-SIGN-TYPE': '2',
                'X-BAPI-TIMESTAMP': str(timestamp),
                'X-BAPI-RECV-WINDOW': '5000'
            }
        
        # Wykonaj zapytanie z pomiarem czasu
        start_time = time.time()
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, data=json.dumps(params))
            elif method == 'DELETE':
                headers['Content-Type'] = 'application/json'
                response = requests.delete(url, headers=headers, data=json.dumps(params))
            else:
                raise ValueError(f"Nieobsługiwana metoda HTTP: {method}")
            
            # Oblicz czas odpowiedzi
            response_time = time.time() - start_time
            self.last_response_time = response_time
            
            # Aktualizuj statystyki
            total_requests = self.request_stats['success'] + self.request_stats['error']
            if total_requests > 0:
                self.request_stats['avg_response_time'] = (
                    (self.request_stats['avg_response_time'] * total_requests + response_time) / 
                    (total_requests + 1)
                )
            else:
                self.request_stats['avg_response_time'] = response_time
            
            # Sprawdź kod odpowiedzi
            response.raise_for_status()
            
            # Parsuj JSON
            result = response.json()
            
            # Sprawdź kod wyniku Bybit
            if result.get('retCode') != 0:
                error_code = result.get('retCode')
                error_msg = result.get('retMsg', 'Unknown error')
                
                # Obsługa znanych błędów (np. rate limit, maintenance)
                if error_code in [10006, 10018]:  # Rate limit
                    logger.warning(f"Rate limit przekroczony: {error_msg}")
                    time.sleep(1)  # Odczekaj przed ponowną próbą
                    self.request_stats['retry'] += 1
                    
                    if retry_count > 0:
                        logger.info(f"Ponowna próba zapytania ({retry_count}): {endpoint}")
                        return self._make_request(method, endpoint, params, retry_count - 1)
                
                # Raportuj błąd
                logger.error(f"Błąd API Bybit: {error_code} - {error_msg}")
                self.request_stats['error'] += 1
                
                return {
                    'success': False,
                    'error_code': error_code,
                    'error_message': error_msg,
                    'data': result.get('result', {})
                }
            
            # Sukces
            self.request_stats['success'] += 1
            
            return {
                'success': True,
                'data': result.get('result', {}),
                'response_time': response_time
            }
            
        except requests.exceptions.RequestException as e:
            # Obsługa błędów sieciowych
            logger.error(f"Błąd sieciowy: {e}")
            self.request_stats['error'] += 1
            
            # Ponowna próba dla błędów sieciowych
            if retry_count > 0:
                retry_delay = self.ws_reconnect_delay * (4 - retry_count)  # Zwiększaj opóźnienie przy kolejnych próbach
                logger.info(f"Ponowna próba za {retry_delay}s ({retry_count}): {endpoint}")
                time.sleep(retry_delay)
                return self._make_request(method, endpoint, params, retry_count - 1)
            
            return {
                'success': False,
                'error_code': 'NETWORK_ERROR',
                'error_message': str(e),
                'data': {}
            }
            
        except Exception as e:
            # Inne błędy
            logger.error(f"Nieoczekiwany błąd: {e}")
            self.request_stats['error'] += 1
            
            return {
                'success': False,
                'error_code': 'UNKNOWN_ERROR',
                'error_message': str(e),
                'data': {}
            }

    # Funkcje do komunikacji z REST API Bybit
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200, 
                   start_time: Optional[int] = None, end_time: Optional[int] = None) -> Dict:
        """
        Pobiera dane świecowe (kline/candles) dla danego symbolu
        
        Args:
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT')
            interval (str): Interwał czasowy ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
            limit (int, optional): Liczba świec do pobrania (max 200)
            start_time (int, optional): Czas początkowy (timestamp ms)
            end_time (int, optional): Czas końcowy (timestamp ms)
            
        Returns:
            dict: Dane świecowe
        """
        endpoint = '/v5/market/kline'
        params = {
            'category': 'spot',  # lub 'linear', 'inverse' dla futures
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time
            
        return self._make_request('GET', endpoint, params)
    
    def get_tickers(self, symbol: Optional[str] = None) -> Dict:
        """
        Pobiera tickery dla wszystkich lub określonego symbolu
        
        Args:
            symbol (str, optional): Symbol pary handlowej (np. 'BTCUSDT')
            
        Returns:
            dict: Dane tickera
        """
        endpoint = '/v5/market/tickers'
        params = {'category': 'spot'}  # lub 'linear', 'inverse' dla futures
        
        if symbol:
            params['symbol'] = symbol
            
        return self._make_request('GET', endpoint, params)
    
    def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        """
        Pobiera książkę zleceń (orderbook) dla danego symbolu
        
        Args:
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT')
            limit (int, optional): Głębokość orderbooka (1, 50, 200, 500)
            
        Returns:
            dict: Dane orderbooka
        """
        endpoint = '/v5/market/orderbook'
        params = {
            'category': 'spot',  # lub 'linear', 'inverse' dla futures
            'symbol': symbol,
            'limit': limit
        }
            
        return self._make_request('GET', endpoint, params)
    
    def get_wallet_balance(self, coin: Optional[str] = None) -> Dict:
        """
        Pobiera balans portfela
        
        Args:
            coin (str, optional): Symbol waluty (np. 'BTC', 'USDT')
            
        Returns:
            dict: Balans portfela
        """
        endpoint = '/v5/account/wallet-balance'
        params = {'accountType': 'UNIFIED'}  # lub 'CONTRACT', 'SPOT'
        
        if coin:
            params['coin'] = coin
            
        return self._make_request('GET', endpoint, params)
    
    def get_positions(self, symbol: Optional[str] = None) -> Dict:
        """
        Pobiera otwarte pozycje
        
        Args:
            symbol (str, optional): Symbol pary handlowej (np. 'BTCUSDT')
            
        Returns:
            dict: Otwarte pozycje
        """
        endpoint = '/v5/position/list'
        params = {'category': 'linear'}  # tylko dla futures
        
        if symbol:
            params['symbol'] = symbol
            
        return self._make_request('GET', endpoint, params)
    
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, time_in_force: str = 'GTC',
                   take_profit: Optional[float] = None, stop_loss: Optional[float] = None,
                   reduce_only: bool = False, close_on_trigger: bool = False) -> Dict:
        """
        Składa zlecenie
        
        Args:
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT')
            side (str): Strona ('Buy' lub 'Sell')
            order_type (str): Typ zlecenia ('Limit', 'Market')
            qty (float): Ilość
            price (float, optional): Cena (wymagana dla 'Limit')
            time_in_force (str): Ważność zlecenia ('GTC', 'IOC', 'FOK', 'PostOnly')
            take_profit (float, optional): Poziom Take Profit
            stop_loss (float, optional): Poziom Stop Loss
            reduce_only (bool): Czy zlecenie ma tylko zmniejszać pozycję
            close_on_trigger (bool): Czy zlecenie ma zamknąć pozycję po aktywacji
            
        Returns:
            dict: Status złożenia zlecenia
        """
        endpoint = '/v5/order/create'
        params = {
            'category': 'spot',  # lub 'linear', 'inverse' dla futures
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'qty': str(qty),
            'timeInForce': time_in_force
        }
        
        if order_type == 'Limit' and price is not None:
            params['price'] = str(price)
            
        if take_profit is not None:
            params['takeProfit'] = str(take_profit)
            
        if stop_loss is not None:
            params['stopLoss'] = str(stop_loss)
            
        if reduce_only:
            params['reduceOnly'] = True
            
        if close_on_trigger:
            params['closeOnTrigger'] = True
            
        return self._make_request('POST', endpoint, params)
    
    def cancel_order(self, symbol: str, order_id: Optional[str] = None, 
                    order_link_id: Optional[str] = None) -> Dict:
        """
        Anuluje zlecenie
        
        Args:
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT')
            order_id (str, optional): ID zlecenia
            order_link_id (str, optional): Własne ID zlecenia
            
        Returns:
            dict: Status anulowania zlecenia
        """
        endpoint = '/v5/order/cancel'
        params = {
            'category': 'spot',  # lub 'linear', 'inverse' dla futures
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif order_link_id:
            params['orderLinkId'] = order_link_id
        else:
            raise ValueError("Wymagane jest order_id lub order_link_id")
            
        return self._make_request('POST', endpoint, params)
    
    def get_active_orders(self, symbol: Optional[str] = None, order_id: Optional[str] = None,
                         order_link_id: Optional[str] = None) -> Dict:
        """
        Pobiera aktywne zlecenia
        
        Args:
            symbol (str, optional): Symbol pary handlowej (np. 'BTCUSDT')
            order_id (str, optional): ID zlecenia
            order_link_id (str, optional): Własne ID zlecenia
            
        Returns:
            dict: Aktywne zlecenia
        """
        endpoint = '/v5/order/realtime'
        params = {'category': 'spot'}  # lub 'linear', 'inverse' dla futures
        
        if symbol:
            params['symbol'] = symbol
        if order_id:
            params['orderId'] = order_id
        if order_link_id:
            params['orderLinkId'] = order_link_id
            
        return self._make_request('GET', endpoint, params)
    
    def get_order_history(self, symbol: Optional[str] = None, 
                        start_time: Optional[int] = None, 
                        end_time: Optional[int] = None,
                        limit: int = 50) -> Dict:
        """
        Pobiera historię zleceń
        
        Args:
            symbol (str, optional): Symbol pary handlowej (np. 'BTCUSDT')
            start_time (int, optional): Czas początkowy (timestamp ms)
            end_time (int, optional): Czas końcowy (timestamp ms)
            limit (int, optional): Liczba rekordów (max 50)
            
        Returns:
            dict: Historia zleceń
        """
        endpoint = '/v5/order/history'
        params = {
            'category': 'spot',  # lub 'linear', 'inverse' dla futures
            'limit': limit
        }
        
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return self._make_request('GET', endpoint, params)
    
    # WebSocket API
    
    def _on_ws_message(self, ws, message):
        """Obsługa wiadomości z WebSocketa"""
        try:
            data = json.loads(message)
            
            # Obsługa pingu
            if 'op' in data and data['op'] == 'ping':
                pong_msg = json.dumps({"op": "pong"})
                ws.send(pong_msg)
                return
                
            # Obsługa potwierdzenia subskrypcji
            if 'op' in data and data['op'] == 'subscribe' and 'success' in data:
                if data['success']:
                    logger.info(f"Subskrypcja udana: {data.get('req_id')}")
                else:
                    logger.error(f"Błąd subskrypcji: {data.get('ret_msg')}")
                return
                
            # Obsługa danych
            if 'topic' in data:
                topic = data['topic']
                
                # Wywołaj zarejestrowany callback dla danego topicu
                if topic in self.ws_callbacks:
                    for callback in self.ws_callbacks[topic]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Błąd w callbacku dla {topic}: {e}")
            
        except json.JSONDecodeError:
            logger.error(f"Nieprawidłowy format JSON: {message}")
        except Exception as e:
            logger.error(f"Błąd w obsłudze wiadomości WebSocket: {e}")
    
    def _on_ws_error(self, ws, error):
        """Obsługa błędów WebSocketa"""
        logger.error(f"Błąd WebSocket: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_reason):
        """Obsługa zamknięcia połączenia WebSocket"""
        logger.warning(f"Połączenie WebSocket zamknięte: kod={close_status_code}, powód={close_reason}")
        
        # Automatyczne ponowne połączenie
        self.ws_connected = False
        if self.ws_reconnect_count < self.ws_max_reconnect:
            self.ws_reconnect_count += 1
            reconnect_delay = self.ws_reconnect_delay * self.ws_reconnect_count
            logger.info(f"Ponowne połączenie za {reconnect_delay} sekund (próba {self.ws_reconnect_count}/{self.ws_max_reconnect})")
            
            threading.Timer(reconnect_delay, self.connect_websocket).start()
    
    def _on_ws_open(self, ws):
        """Obsługa otwarcia połączenia WebSocket"""
        logger.info("Połączenie WebSocket ustanowione")
        self.ws_connected = True
        self.ws_reconnect_count = 0
        
        # Ponowna subskrypcja wszystkich tematów
        for topic in self.ws_subscriptions:
            self._send_subscription(topic)
    
    def connect_websocket(self):
        """Nawiązuje połączenie z WebSocket API"""
        # Zamknij istniejące połączenie jeśli istnieje
        if self.ws is not None:
            self.ws.close()
        
        # Utwórz nowe połączenie
        endpoint = f"{self.ws_endpoint}/v5/public/spot"
        
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            endpoint,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            on_open=self._on_ws_open
        )
        
        # Uruchom WebSocket w osobnym wątku
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Poczekaj na połączenie
        timeout = 10
        start_time = time.time()
        while not self.ws_connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.ws_connected:
            logger.error("Nie udało się nawiązać połączenia WebSocket w określonym czasie")
            return False
            
        return True
    
    def _send_subscription(self, topic):
        """Wysyła subskrypcję do określonego tematu"""
        if not self.ws_connected:
            logger.warning(f"Nie można subskrybować {topic} - brak połączenia WebSocket")
            return False
            
        subscribe_message = {
            "op": "subscribe",
            "args": [topic]
        }
        
        try:
            self.ws.send(json.dumps(subscribe_message))
            return True
        except Exception as e:
            logger.error(f"Błąd podczas subskrypcji {topic}: {e}")
            return False
    
    def subscribe(self, topic, callback):
        """
        Subskrybuje temat WebSocket i rejestruje callback
        
        Args:
            topic (str): Temat do subskrypcji (np. 'orderbook.50.BTCUSDT')
            callback (callable): Funkcja wywoływana po otrzymaniu danych
            
        Returns:
            bool: Status subskrypcji
        """
        # Dodaj temat do listy subskrypcji
        self.ws_subscriptions.add(topic)
        
        # Zarejestruj callback
        if topic not in self.ws_callbacks:
            self.ws_callbacks[topic] = []
        self.ws_callbacks[topic].append(callback)
        
        # Jeśli połączenie jest aktywne, wyślij subskrypcję
        if self.ws_connected:
            return self._send_subscription(topic)
        else:
            # Automatycznie połącz jeśli nie ma połączenia
            logger.info("Automatyczne nawiązywanie połączenia WebSocket...")
            return self.connect_websocket()
    
    def unsubscribe(self, topic):
        """
        Anuluje subskrypcję tematu WebSocket
        
        Args:
            topic (str): Temat do anulowania subskrypcji
            
        Returns:
            bool: Status anulowania subskrypcji
        """
        if not self.ws_connected:
            logger.warning(f"Nie można anulować subskrypcji {topic} - brak połączenia WebSocket")
            return False
            
        # Usuń temat z listy subskrypcji
        if topic in self.ws_subscriptions:
            self.ws_subscriptions.remove(topic)
            
        # Usuń callbacki
        if topic in self.ws_callbacks:
            del self.ws_callbacks[topic]
            
        # Wyślij anulowanie subskrypcji
        unsubscribe_message = {
            "op": "unsubscribe",
            "args": [topic]
        }
        
        try:
            self.ws.send(json.dumps(unsubscribe_message))
            return True
        except Exception as e:
            logger.error(f"Błąd podczas anulowania subskrypcji {topic}: {e}")
            return False
    
    def disconnect_websocket(self):
        """Zamyka połączenie WebSocket"""
        if self.ws is not None:
            self.ws.close()
            self.ws_connected = False
            logger.info("Połączenie WebSocket zamknięte")
            
    def get_connection_status(self):
        """Zwraca status połączenia i statystyki"""
        return {
            'rest_endpoint': self.rest_endpoint,
            'websocket_connected': self.ws_connected,
            'test_mode': self.test_mode,
            'last_response_time': self.last_response_time,
            'stats': self.request_stats
        }

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja konfiguracji
    from config.config_loader import initialize_configuration
    config = initialize_configuration()
    
    # Utwórz instancję konektora
    bybit = BybitConnector()
    
    # Przykład zapytania REST
    ticker = bybit.get_tickers("BTCUSDT")
    print(f"Ticker BTCUSDT: {ticker}")
    
    # Przykład użycia WebSocket
    def process_orderbook(data):
        print(f"Odebrano dane orderbooka: {data}")
    
    # Połącz z WebSocket i subskrybuj orderbook
    bybit.connect_websocket()
    bybit.subscribe("orderbook.50.BTCUSDT", process_orderbook)
    
    # Poczekaj na dane
    time.sleep(30)
    
    # Zamknij połączenie
    bybit.disconnect_websocket()
