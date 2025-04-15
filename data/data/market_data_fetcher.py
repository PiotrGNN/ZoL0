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
API_BASE_URL = "https://api.exampleexchange.com"  # Przykładowy URL API
DEFAULT_TIMEOUT = 5  # sekundy
MAX_RETRIES = 3
RETRY_DELAY = 2  # sekundy


class MarketDataFetcher:
    def __init__(self, api_key: str, output_mode: str = "csv", db_path: str = None):
        """
        Inicjalizuje MarketDataFetcher.

        Parameters:
            api_key (str): Klucz API do autoryzacji.
            output_mode (str): Sposób zapisywania danych: "csv" lub "db".
            db_path (str): Ścieżka do bazy SQLite (jeśli output_mode == "db").
        """
        self.api_key = api_key
        self.output_mode = output_mode.lower()
        self.db_path = db_path or "./data/historical_data.db"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        # Przygotowanie folderu na CSV, jeśli wybrany tryb
        if self.output_mode == "csv":
            os.makedirs("./data", exist_ok=True)

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Wykonuje żądanie HTTP GET do API z retry oraz obsługą timeoutów.

        Parameters:
            endpoint (str): Endpoint API.
            params (dict): Parametry zapytania.

        Returns:
            dict: Odpowiedź w formie słownika.
        """
        url = f"{API_BASE_URL}{endpoint}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=DEFAULT_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
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

    def fetch_data(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> pd.DataFrame:
        """
        Pobiera dane rynkowe dla określonej pary walutowej i interwału.

        Parameters:
            symbol (str): Para walutowa (np. "BTCUSDT").
            interval (str): Interwał danych ("1m", "5m", "1h", "1d").
            limit (int): Liczba rekordów do pobrania.

        Returns:
            pd.DataFrame: Dane w formacie DataFrame, zawierające kolumny: timestamp, open, high, low, close, volume.
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = self._make_request("/market_data", params)
        # Rzeczywiste przetwarzanie odpowiedzi API
        try:
            # Przetwarzanie danych z API Bybit
            if 'result' in data and 'list' in data['result']:
                raw_data = data['result']['list']
                processed_data = []
                for item in raw_data:
                    processed_data.append({
                        'timestamp': int(item[0]),
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5]),
                    })
                df = pd.DataFrame(processed_data)
            else:
                # Fallback dla innych formatów API
                df = pd.DataFrame(data)
                
            logging.info(f"Pobrano rzeczywiste dane rynkowe: {len(df)} rekordów")
        except Exception as e:
            logging.error(f"Błąd podczas przetwarzania danych z API: {e}")
            raise
        # Konwersja timestamp na datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        logging.info(
            "Dane dla %s (%s) pobrane, liczba rekordów: %d", symbol, interval, len(df)
        )
        return df

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
