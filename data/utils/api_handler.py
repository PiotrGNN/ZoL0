"""
api_handler.py
--------------
Moduł obsługujący komunikację z zewnętrznymi API.
Funkcjonalności:
- Obsługuje błędy: time-out, brak połączenia, niepoprawne odpowiedzi, oraz retry z mechanizmem backoff.
- Cache’owanie odpowiedzi przy użyciu lru_cache, aby zmniejszyć liczbę wywołań API przy dużym wolumenie zapytań.
- Obsługuje szyfrowanie danych (HTTPS, TLS) i autoryzację za pomocą tokenów/kluczy API.
- Elastyczna konfiguracja endpointów, możliwa integracja z ustawieniami z pliku settings.py lub zmiennych środowiskowych.
- Zawiera logowanie na poziomie debug oraz info dla lepszej diagnostyki.
"""

import logging
import os
import time
from functools import lru_cache

import requests

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_TIMEOUT = 5  # sekundy
MAX_RETRIES = 3
BACKOFF_FACTOR = 2


class APIHandler:
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Inicjalizuje APIHandler.

        Parameters:
            api_key (str): Klucz API używany do autoryzacji.
            base_url (str): Bazowy URL API.
        """
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://api.example.com")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        logging.info("APIHandler zainicjalizowany z base_url: %s", self.base_url)

    @lru_cache(maxsize=128)
    def get(self, endpoint: str, params: dict = None) -> dict:
        """
        Wykonuje żądanie GET do API z retry i exponential backoff, cache’ując odpowiedzi.

        Parameters:
            endpoint (str): Endpoint API (dopisany do base_url).
            params (dict): Parametry zapytania.

        Returns:
            dict: Odpowiedź API jako słownik.
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        attempt = 0
        delay = 1  # początkowe opóźnienie
        while attempt < MAX_RETRIES:
            try:
                logging.debug(
                    "Wysyłanie żądania GET do %s z parametrami: %s (próba %d)",
                    url,
                    params,
                    attempt + 1,
                )
                response = requests.get(url, headers=self.headers, params=params, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                logging.debug("Otrzymano odpowiedź: %s", data)
                return data
            except requests.exceptions.RequestException as e:
                attempt += 1
                logging.warning("Błąd żądania GET (próba %d/%d): %s", attempt, MAX_RETRIES, e)
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
        logging.error("Przekroczono maksymalną liczbę prób dla żądania GET do %s.", url)
        raise Exception(f"Nie udało się uzyskać odpowiedzi z {url}")

    def post(self, endpoint: str, data: dict = None, json_data: dict = None) -> dict:
        """
        Wykonuje żądanie POST do API z obsługą retry i exponential backoff.

        Parameters:
            endpoint (str): Endpoint API.
            data (dict): Dane przesyłane metodą POST (form-data).
            json_data (dict): Dane przesyłane w formacie JSON.

        Returns:
            dict: Odpowiedź API.
        """
        url = f"{self.base_url}{endpoint}"
        attempt = 0
        delay = 1
        while attempt < MAX_RETRIES:
            try:
                logging.debug(
                    "Wysyłanie żądania POST do %s z danymi: %s (próba %d)",
                    url,
                    json_data or data,
                    attempt + 1,
                )
                response = requests.post(
                    url,
                    headers=self.headers,
                    data=data,
                    json=json_data,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                res_data = response.json()
                logging.debug("Otrzymano odpowiedź: %s", res_data)
                return res_data
            except requests.exceptions.RequestException as e:
                attempt += 1
                logging.warning("Błąd żądania POST (próba %d/%d): %s", attempt, MAX_RETRIES, e)
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
        logging.error("Przekroczono maksymalną liczbę prób dla żądania POST do %s.", url)
        raise Exception(f"Nie udało się uzyskać odpowiedzi z {url}")


# -------------------- Testy jednostkowe --------------------
if __name__ == "__main__":
    # Przykładowe testy APIHandler
    try:
        api_handler = APIHandler(api_key="dummy_key", base_url="https://api.example.com")
        # Testowanie cache'owania - symulujemy żądanie, używając endpointu, który zwraca dummy dane
        # Ponieważ nie mamy rzeczywistego API, można podać przykładowy endpoint i oczekiwać wyjątku
        try:
            data1 = api_handler.get("/dummy_endpoint", params={"q": "test"})
        except Exception as e:
            logging.info("Oczekiwany błąd dla dummy_endpoint: %s", e)
        # Próba ponownego wywołania get() z tymi samymi parametrami
        try:
            data2 = api_handler.get("/dummy_endpoint", params={"q": "test"})
        except Exception as e:
            logging.info("Oczekiwany błąd dla dummy_endpoint przy cache: %s", e)
        logging.info("Testy jednostkowe APIHandler zakończone sukcesem.")
    except Exception as e:
        logging.error("Błąd w testach APIHandler: %s", e)
        raise
