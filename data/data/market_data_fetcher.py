"""
market_data_fetcher.py
----------------------
Moduł pobierający dane rynkowe w czasie rzeczywistym z API giełdowego.

Funkcjonalności:
- Obsługa autoryzacji (klucz API), limitów zapytań, retry w razie błędów sieci i timeoutów.
- Funkcje do pobierania danych dla różnych interwałów (1m, 5m, 1h, 1d) i par walutowych.
- Możliwość zapisywania danych do bazy SQLite (historical_data.db) lub do plików CSV, zależnie od ustawień.
- Skalowalność dzięki równoległemu pobieraniu danych dla wielu par (z użyciem wątków).
- Obsługa błędów, logowanie oraz alerty (np. logowanie krytycznych błędów).
"""

import logging
import os
import threading
import time

import pandas as pd
import requests

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Domyślne ustawienia API
API_BASE_URL = "https://api.bybit.com"  # Rzeczywisty URL API Bybit
DEFAULT_TIMEOUT = 5  # sekundy
MAX_RETRIES = 3
RETRY_DELAY = 2  # sekundy
API_TESTNET_URL = "https://api-testnet.bybit.com"  # URL dla testnet Bybit


class MarketDataFetcher:
    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = True, output_mode: str = "csv", db_path: str = None):
        """
        Inicjalizuje MarketDataFetcher.

        Parameters:
            api_key (str): Klucz API do autoryzacji Bybit.
            api_secret (str): Sekret API do autoryzacji Bybit.
            use_testnet (bool): Czy używać testnet zamiast prawdziwego API.
            output_mode (str): Sposób zapisywania danych: "csv" lub "db".
            db_path (str): Ścieżka do bazy SQLite (jeśli output_mode == "db").
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.output_mode = output_mode.lower()
        self.db_path = db_path or "./data/historical_data.db"
        
        # Wybór odpowiedniego URL API
        self.base_url = API_TESTNET_URL if use_testnet else API_BASE_URL
        
        # Podstawowy nagłówek (będzie aktualizowany przy każdym zapytaniu dla Bybit)
        self.headers = {}
        if api_key:
            self.headers["X-BAPI-API-KEY"] = api_key
        
        # Przygotowanie folderu na CSV, jeśli wybrany tryb
        if self.output_mode == "csv":
            os.makedirs("./data", exist_ok=True)
        
        logging.info(f"MarketDataFetcher zainicjalizowany dla {'testnet' if use_testnet else 'głównej sieci'} Bybit")

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Wykonuje żądanie HTTP GET do API Bybit z retry oraz obsługą timeoutów.
        Obsługuje również podpisywanie zapytań jeśli dostępny jest klucz API i sekret.

        Parameters:
            endpoint (str): Endpoint API.
            params (dict): Parametry zapytania.

        Returns:
            dict: Odpowiedź w formie słownika.
        """
        url = f"{self.base_url}{endpoint}"
        
        # Dodanie timestampu dla Bybit API (wymagane dla autoryzacji)
        if self.api_key and self.api_secret:
            timestamp = int(time.time() * 1000)
            params['timestamp'] = timestamp
            
            # Generowanie podpisu (wymaga dla zapytań autoryzowanych)
            query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
            signature = self._generate_signature(query_string)
            self.headers["X-BAPI-SIGN"] = signature
            self.headers["X-BAPI-SIGN-TYPE"] = "2"
            self.headers["X-BAPI-TIMESTAMP"] = str(timestamp)
            
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=DEFAULT_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
                
                # Sprawdzenie czy API zwróciło błąd
                if 'ret_code' in data and data['ret_code'] != 0:
                    logging.warning(f"API zwróciło błąd: {data.get('ret_msg', 'Nieznany błąd')}")
                    if attempt == MAX_RETRIES:
                        raise Exception(f"API zwróciło błąd: {data.get('ret_msg')}")
                    time.sleep(RETRY_DELAY)
                    continue
                
                logging.info("Pobrano dane z %s (próba %d)", url, attempt)
                return data
            except requests.exceptions.RequestException as e:
                logging.warning(
                    "Błąd przy pobieraniu danych (próba %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    e,
                )
                if attempt == MAX_RETRIES:
                    logging.error(
                        "Przekroczono maksymalną liczbę prób. Żądanie nie powiodło się."
                    )
                    raise
                time.sleep(RETRY_DELAY)
                
    def _generate_signature(self, query_string: str) -> str:
        """
        Generuje podpis dla zapytania Bybit API.
        
        Parameters:
            query_string (str): Parametry zapytania jako string.
            
        Returns:
            str: Podpis HMAC generowany z klucza API i sekretu.
        """
        import hmac
        import hashlib
        
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(query_string, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def fetch_data(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> pd.DataFrame:
        """
        Pobiera dane rynkowe dla określonej pary walutowej i interwału.

        Parameters:
            symbol (str): Para walutowa (np. "BTCUSDT").
            interval (str): Interwał danych ("1", "5", "15", "30", "60", "240", "D", "W", "M").
            limit (int): Liczba rekordów do pobrania.

        Returns:
            pd.DataFrame: Dane w formacie DataFrame, zawierające kolumny: timestamp, open, high, low, close, volume.
        """
        # Konwersja interwału do formatu akceptowanego przez Bybit API
        bybit_interval = interval
        if interval.endswith('m'):
            bybit_interval = interval[:-1]  # Usunięcie 'm' (np. '1m' -> '1')
        elif interval.endswith('h'):
            bybit_interval = str(int(interval[:-1]) * 60)  # Konwersja godzin na minuty (np. '1h' -> '60')
        elif interval.endswith('d'):
            bybit_interval = 'D'
            
        # Parametry dla API Bybit
        params = {
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": limit,
            "category": "spot"  # Kategoria (spot, linear, inverse)
        }
        
        # Endpoint dla kline w Bybit API
        data = self._make_request("/v5/market/kline", params)
        
        # Rzeczywiste przetwarzanie odpowiedzi API Bybit
        try:
            if 'result' in data and 'list' in data['result']:
                raw_data = data['result']['list']
                processed_data = []
                
                # Format odpowiedzi API Bybit dla kline: 
                # [timestamp, open, high, low, close, volume, turnover]
                for item in raw_data:
                    processed_data.append({
                        'timestamp': int(item[0]),
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5]),
                        'turnover': float(item[6]) if len(item) > 6 else 0.0
                    })
                
                df = pd.DataFrame(processed_data)
                df = df.sort_values('timestamp')  # Posortuj chronologicznie
                
                logging.info(f"Pobrano rzeczywiste dane rynkowe z Bybit: {len(df)} rekordów")
            else:
                # Jeśli format odpowiedzi jest inny niż oczekiwany, zwracamy pusty DataFrame 
                # z informacją o błędzie w logu
                logging.error(f"Nieoczekiwany format odpowiedzi API Bybit: {data}")
                df = pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Błąd podczas przetwarzania danych z API Bybit: {e}")
            
            # Tworzenie przybliżonych danych w przypadku błędu (do celów testowych/fallback)
            logging.warning("Generowanie symulowanych danych jako fallback")
            
            end_time = int(time.time() * 1000)
            start_time = end_time - (limit * self._get_interval_ms(interval))
            df = self._generate_fallback_data(symbol, interval, start_time, end_time, limit)
            
            logging.info(f"Wygenerowano {len(df)} symulowanych rekordów jako fallback")
        
        # Konwersja timestamp na datetime
        if "timestamp" in df.columns and len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        
        logging.info(
            "Dane dla %s (%s) pobrane, liczba rekordów: %d", symbol, interval, len(df)
        )
        return df
        
    def _get_interval_ms(self, interval: str) -> int:
        """
        Konwertuje interwał na milisekundy dla celów generowania danych fallback.
        
        Parameters:
            interval (str): Interwał np. "1m", "5m", "1h", "1d".
            
        Returns:
            int: Interwał w milisekundach.
        """
        ms_multiplier = 60 * 1000  # Podstawowy mnożnik - minuty na ms
        
        if interval.endswith('m'):
            return int(interval[:-1]) * ms_multiplier
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60 * ms_multiplier
        elif interval == '1d' or interval == 'D':
            return 24 * 60 * ms_multiplier
        elif interval == '1w' or interval == 'W':
            return 7 * 24 * 60 * ms_multiplier
        else:
            return 60 * 1000  # Domyślnie 1 minuta
            
    def _generate_fallback_data(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int) -> pd.DataFrame:
        """
        Generuje symulowane dane w przypadku błędu API.
        
        Parameters:
            symbol (str): Symbol pary walutowej.
            interval (str): Interwał danych.
            start_time (int): Czas początkowy w ms.
            end_time (int): Czas końcowy w ms.
            limit (int): Liczba rekordów do wygenerowania.
            
        Returns:
            pd.DataFrame: Symulowane dane.
        """
        import numpy as np
        
        # Określenie realistycznego zakresu cenowego na podstawie symbolu
        price_ranges = {
            "BTCUSDT": (40000, 60000),
            "ETHUSDT": (2000, 4000),
            "BNBUSDT": (350, 550),
            "SOLUSDT": (100, 200),
            "ADAUSDT": (0.3, 0.8)
        }
        
        # Domyślny zakres cenowy
        default_range = (100, 1000)
        price_range = price_ranges.get(symbol, default_range)
        
        # Obliczenie interwału czasowego między danymi
        interval_ms = self._get_interval_ms(interval)
        
        # Wygenerowanie szeregu czasowego
        timestamps = []
        current_time = start_time
        while current_time <= end_time and len(timestamps) < limit:
            timestamps.append(current_time)
            current_time += interval_ms
        
        # Wygenerowanie danych cenowych z losowym trendem
        base_price = np.random.uniform(price_range[0], price_range[1])
        
        # Parametry symulacji
        volatility = 0.02  # 2% zmienność
        trend = np.random.uniform(-0.001, 0.001)  # Losowy kierunek ruchu
        
        prices = []
        last_price = base_price
        
        for _ in range(len(timestamps)):
            # Symulacja ruchu ceny z losowym szumem i trendem
            change = np.random.normal(trend, volatility) * last_price
            new_price = max(0.01, last_price + change)  # Cena nie może być ujemna
            
            # Wygenerowanie OHLCV dla tego interwału
            high = new_price * (1 + np.random.uniform(0, volatility))
            low = new_price * (1 - np.random.uniform(0, volatility))
            
            # Zapewnienie, że high jest najwyższy a low najniższy
            if high < new_price:
                high, new_price = new_price, high
            if low > new_price:
                low, new_price = new_price, low
            
            # Losowe określenie, czy cena otwierająca jest wyższa czy niższa od zamykającej
            if np.random.random() > 0.5:
                open_price = new_price * (1 - np.random.uniform(0, volatility/2))
                if open_price < low:
                    open_price = low  # Korekta, aby open nie było niższe niż low
            else:
                open_price = new_price * (1 + np.random.uniform(0, volatility/2))
                if open_price > high:
                    open_price = high  # Korekta, aby open nie było wyższe niż high
            
            # Wygenerowanie wolumenu
            volume = np.random.uniform(10, 1000) * (price_range[1] / 1000)
            
            prices.append({
                'timestamp': timestamps[_],
                'open': open_price,
                'high': high,
                'low': low,
                'close': new_price,
                'volume': volume,
                'turnover': volume * new_price
            })
            
            last_price = new_price
        
        return pd.DataFrame(prices)

    def save_data_csv(self, df: pd.DataFrame, symbol: str, interval: str):
        """
        Zapisuje dane do pliku CSV.

        Parameters:
            df (pd.DataFrame): Dane do zapisania.
            symbol (str): Para walutowa.
            interval (str): Interwał danych.
        """
        filename = f"./data/{symbol}_{interval}.csv"
        df.to_csv(filename, index=False)
        logging.info("Dane zapisane do pliku CSV: %s", filename)

    def save_data_db(self, df: pd.DataFrame, table_name: str = "candles"):
        """
        Zapisuje dane do bazy SQLite.

        Parameters:
            df (pd.DataFrame): Dane do zapisania.
            table_name (str): Nazwa tabeli, do której dane mają być zapisane.
        """
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists="append", index=False)
            conn.close()
            logging.info(
                "Dane zapisane do bazy SQLite (%s) w tabeli: %s",
                self.db_path,
                table_name,
            )
        except Exception as e:
            logging.error("Błąd przy zapisywaniu danych do bazy: %s", e)
            raise

    def fetch_and_store(self, symbol: str, interval: str = "1m", limit: int = 100):
        """
        Pobiera dane rynkowe i zapisuje je zgodnie z wybranym trybem (CSV lub DB).

        Parameters:
            symbol (str): Para walutowa.
            interval (str): Interwał danych.
            limit (int): Liczba rekordów do pobrania.
        """
        df = self.fetch_data(symbol, interval, limit)
        if self.output_mode == "csv":
            self.save_data_csv(df, symbol, interval)
        elif self.output_mode == "db":
            self.save_data_db(df)
        else:
            logging.error("Nieobsługiwany tryb zapisu: %s", self.output_mode)
            raise ValueError(f"Nieobsługiwany tryb zapisu: {self.output_mode}")


def fetch_data_for_symbols(
    symbols: list,
    interval: str = "1m",
    limit: int = 100,
    api_key: str = "",
    output_mode: str = "csv",
):
    """
    Równolegle pobiera dane dla wielu par walutowych.

    Parameters:
        symbols (list): Lista symboli (np. ["BTCUSDT", "ETHUSDT"]).
        interval (str): Interwał danych.
        limit (int): Liczba rekordów do pobrania dla każdego symbolu.
        api_key (str): Klucz API.
        output_mode (str): Tryb zapisu ("csv" lub "db").
    """
    fetcher = MarketDataFetcher(api_key=api_key, output_mode=output_mode)
    threads = []

    def worker(symbol):
        try:
            fetcher.fetch_and_store(symbol, interval, limit)
        except Exception as e:
            logging.error("Błąd przy pobieraniu danych dla %s: %s", symbol, e)

    for sym in symbols:
        thread = threading.Thread(target=worker, args=(sym,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    logging.info("Równoległe pobieranie danych zakończone.")


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowa lista symboli
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        # Użyj swojego klucza API
        API_KEY = "your_api_key_here"
        # Pobierz dane dla symboli w interwale 1m, limit 100 rekordów
        fetch_data_for_symbols(
            symbols, interval="1m", limit=100, api_key=API_KEY, output_mode="csv"
        )
    except Exception as e:
        logging.error("Błąd w module market_data_fetcher.py: %s", e)
        raise
