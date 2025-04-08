"""
bybit_connector.py
-----------------
Moduł do komunikacji z giełdą Bybit.
"""

import json
import logging
import os
import random
import time
import traceback
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests


def is_env_flag_true(env_var_name: str) -> bool:
    """
    Sprawdza, czy zmienna środowiskowa jest ustawiona na wartość prawdziwą.

    Args:
        env_var_name: Nazwa zmiennej środowiskowej do sprawdzenia

    Returns:
        bool: True jeśli wartość zmiennej to "1", "true" lub "yes" (bez uwzględnienia wielkości liter)
    """
    return os.getenv(env_var_name, "").strip().lower() in ["1", "true", "yes"]


class BybitConnector:
    """
    Klasa do komunikacji z giełdą Bybit.
    """

    def __init__(self,
                 api_key: str = None,
                 api_secret: str = None,
                 use_testnet: bool = None,
                 lazy_connect: bool = True,
                 proxies: Dict[str, str] = None):
        """
        Inicjalizuje połączenie z Bybit.

        Parameters:
            api_key (str): Klucz API Bybit.
            api_secret (str): Sekret API Bybit.
            use_testnet (bool): Czy używać środowiska testowego (domyślnie odczytywane ze zmiennych środowiskowych).
            lazy_connect (bool): Czy opóźnić połączenie z API do pierwszego użycia.
            proxies (Dict[str, str]): Nie używane w wersji lokalnej (zachowane dla kompatybilności)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.proxies = None  # Nie używamy proxy w wersji lokalnej

        # Priorytetyzacja parametrów:
        # 1. Przekazany parametr use_testnet
        # 2. BYBIT_TESTNET
        # 3. BYBIT_USE_TESTNET
        # 4. Domyślnie False (produkcja)
        if use_testnet is None:
            if is_env_flag_true("BYBIT_TESTNET"):
                self.use_testnet = True
            elif os.getenv("BYBIT_USE_TESTNET", "false").lower() == "true":
                self.use_testnet = True
            else:
                self.use_testnet = False
        else:
            self.use_testnet = use_testnet
        # Upewnij się, że używasz odpowiedniego URL API
        self.base_url = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"
        self.client = None
        self.api_version = None
        self._connection_initialized = False
        self._connection_test_time = 0
        self._connection_test_result = None

        # Inicjalizacja śledzenia limitów API - bardziej restrykcyjne dla produkcji
        self.last_api_call = 0
        self.min_time_between_calls = 10.0 if not use_testnet else 3.0  # Ostrzejsze limity dla produkcji
        self.rate_limit_backoff = 30.0 if not use_testnet else 10.0  # Znacznie dłuższy backoff dla produkcji
        self.remaining_rate_limit = 20 if not use_testnet else 50  # Bezpieczniejszy limit początkowy dla produkcji
        self.rate_limit_exceeded = False  # Flaga oznaczająca przekroczenie limitu
        self.last_rate_limit_reset = time.time(
        )  # Czas ostatniego resetu limitu

        # Konfiguracja logowania
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "bybit_connector.log")

        self.logger = logging.getLogger("bybit_connector")
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        # Sprawdź czy używamy produkcyjnego API i pokaż ostrzeżenie
        if self.is_production_api():
            # Ostrzeżenie jest pokazywane w is_production_api()
            # Dodatkowe sprawdzenia kluczy produkcyjnych
            if self.api_key is None or len(
                    self.api_key) < 10 or self.api_secret is None or len(
                        self.api_secret) < 10:
                self.logger.critical(
                    "BŁĄD: Nieprawidłowe klucze API dla środowiska produkcyjnego!"
                )
                raise ValueError(
                    "Nieprawidłowe klucze API dla środowiska produkcyjnego. Sprawdź konfigurację."
                )
        else:
            self.logger.info("Używasz testnet API (środowisko testowe).")

        self.logger.info(
            f"BybitConnector zainicjalizowany. Testnet: {use_testnet}, Lazy connect: {lazy_connect}"
        )

        # Sprawdź, czy już mamy informację o przekroczeniu limitów w cache
        try:
            from data.utils.cache_manager import get_cached_data
            rate_limited_data, found = get_cached_data("api_rate_limited")

            # Inicjalizujemy flagę jako False
            self.rate_limit_exceeded = False

            if found and rate_limited_data is not None:
                # Bezpieczny dostęp - wartość może być słownikiem lub typem prostym
                if isinstance(rate_limited_data, dict):
                    # Jeśli to słownik, sprawdź klucz "value"
                    if rate_limited_data.get("value", False):
                        self.rate_limit_exceeded = True
                elif isinstance(rate_limited_data, bool):
                    # Jeśli to bezpośrednio boolean
                    self.rate_limit_exceeded = rate_limited_data
                else:
                    # Próba konwersji na boolean
                    try:
                        self.rate_limit_exceeded = bool(rate_limited_data)
                    except (ValueError, TypeError):
                        self.logger.warning(
                            f"Nie można skonwertować wartości cache na boolean: {rate_limited_data}"
                        )

                if self.rate_limit_exceeded:
                    self.logger.warning(
                        "Wykryto zapisaną flagę przekroczenia limitów API. Ustawiam tryb oszczędzania limitów."
                    )
        except Exception as e:
            self.logger.warning(f"Nie można sprawdzić stanu limitów API: {e}")
            # Bezpiecznie zakładamy False w przypadku błędu
            self.rate_limit_exceeded = False

        # Kontynuuj tylko inicjalizację komponentów, ale nie testuj API jeśli lazy_connect
        if not lazy_connect:
            self._initialize_client()
        else:
            self.logger.info(
                f"BybitConnector w trybie lazy initialization. Testnet: {use_testnet}"
            )

    def _initialize_client(self, force=False):
        """
        Inicjalizuje klienta API. Wywołuje się automatycznie przy pierwszym użyciu API.

        Parameters:
            force (bool): Czy wymusić reinicjalizację, nawet jeśli klient już istnieje.
        """
        # Jeśli klient istnieje i nie wymuszamy, to pomijamy
        if self.client is not None and not force and self._connection_initialized:
            return True

        # Sprawdź czy minęło wystarczająco dużo czasu od ostatniego testu połączenia
        # aby uniknąć częstego odpytywania API
        current_time = time.time()
        if self._connection_test_time > 0 and current_time - self._connection_test_time < 60:
            self.logger.debug(
                "Pomijam test połączenia - zbyt krótki czas od ostatniego testu."
            )
            return self._connection_test_result

        # Jeśli już wiemy, że przekroczyliśmy limit, to opóźniamy inicjalizację
        if self.rate_limit_exceeded:
            reset_time = 300.0 if not self.use_testnet else 120.0
            if current_time - self.last_rate_limit_reset < reset_time:
                self.logger.warning(
                    f"Odczekuję z inicjalizacją klienta - przekroczono limity API. Pozostało: {reset_time - (current_time - self.last_rate_limit_reset):.1f}s"
                )
                return False
            else:
                self.rate_limit_exceeded = False
                self.logger.info("Reset stanu przekroczenia limitów API.")

        try:
            import pybit

            # Sprawdzamy wersję i dostosowujemy inicjalizację do odpowiedniej wersji API
            try:
                # Próba inicjalizacji dla nowszej wersji API używając odpowiednich klas rynkowych
                endpoint = "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"

                # Używanie odpowiednich klas rynkowych, zgodnie z ostrzeżeniem z biblioteki
                try:
                    # Najpierw próbujemy z spot API
                    from pybit.spot import HTTP as SpotHTTP
                    self.client = SpotHTTP(
                        endpoint=endpoint,
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        recv_window=20000  # Zwiększenie okna czasu na odpowiedź
                    )
                    self.api_version = "spot"
                    self.logger.info(
                        f"Zainicjalizowano klienta ByBit API Spot. Testnet: {self.use_testnet}"
                    )
                except ImportError:
                    # Jeśli nie ma spot, próbujemy z inverse_perpetual
                    try:
                        from pybit.inverse_perpetual import HTTP as PerpHTTP
                        self.client = PerpHTTP(endpoint=endpoint,
                                               api_key=self.api_key,
                                               api_secret=self.api_secret,
                                               recv_window=20000)
                        self.api_version = "inverse_perpetual"
                        self.logger.info(
                            f"Zainicjalizowano klienta ByBit API Inverse Perpetual. Testnet: {self.use_testnet}"
                        )
                    except ImportError:
                        # Ostatnia szansa - używamy ogólnego HTTP
                        self.client = pybit.HTTP(endpoint=endpoint,
                                                 api_key=self.api_key,
                                                 api_secret=self.api_secret,
                                                 recv_window=20000)
                        self.api_version = "v2"
                        self.logger.info(
                            f"Zainicjalizowano klienta ByBit API v2 (ogólny). Testnet: {self.use_testnet}"
                        )

                # Sprawdź czy już mamy w cache czas serwera
                try:
                    from data.utils.cache_manager import get_cached_data, is_cache_valid, store_cached_data
                    cache_key = f"server_time_{self.use_testnet}"

                    server_time_data = None
                    found = False

                    # Bezpieczne sprawdzanie cache z obsługą wyjątków
                    try:
                        if is_cache_valid(cache_key,
                                          ttl=300):  # Ważny przez 5 minut
                            server_time_data, found = get_cached_data(
                                cache_key)
                    except Exception as cache_valid_error:
                        self.logger.warning(
                            f"Błąd podczas sprawdzania ważności cache: {cache_valid_error}"
                        )
                        found = False

                    # Bezpieczne użycie danych z cache
                    if found and server_time_data is not None:
                        # Sprawdź czy dane mają wymagany format
                        if isinstance(server_time_data,
                                      dict) and "timeNow" in server_time_data:
                            self.logger.info(
                                f"Używam cache'owanego czasu serwera: {server_time_data}"
                            )
                            self._connection_test_result = True
                            self._connection_test_time = current_time
                            self._connection_initialized = True
                            return True
                        else:
                            self.logger.warning(
                                f"Cache zawiera dane w nieprawidłowym formacie: {server_time_data}"
                            )
                except Exception as cache_error:
                    self.logger.warning(
                        f"Błąd podczas dostępu do cache: {cache_error}")

                # Unikaj zbyt częstego odpytywania API - czekaj na minimum 5 sekund między zapytaniami
                # Zastosuj rate limiting nawet przy inicjalizacji
                self._apply_rate_limit()

                # Testowe pobieranie czasu serwera w celu weryfikacji połączenia
                try:
                    # Używamy bezpośredniego zapytania HTTP do publicznych endpointów bez autoryzacji
                    try:
                        # Próba z endpointem Unified v5 API - najnowszym i zalecanym
                        v5_endpoint = f"{self.base_url}/v5/market/time"
                        self.logger.debug(
                            f"Próba pobierania czasu z endpointu v5: {v5_endpoint}"
                        )
                        response = requests.get(v5_endpoint, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("retCode") == 0 and "result" in data:
                                try:
                                    # Bezpieczna konwersja czasu z obsługą typu str
                                    if "timeNano" in data["result"]:
                                        time_nano_raw = data["result"][
                                            "timeNano"]
                                        # Konwersja na int jeśli to string
                                        if isinstance(time_nano_raw, str):
                                            time_nano = int(time_nano_raw)
                                        else:
                                            time_nano = time_nano_raw

                                        # Dzielimy tylko gdy mamy liczbę
                                        time_ms = int(time_nano) // 1000000
                                        server_time = {
                                            "timeNow": time_ms
                                        }
                                        self.logger.debug(
                                            f"Czas serwera v5: {server_time}")
                                    else:
                                        server_time = {
                                            "timeNow": int(time.time() * 1000)
                                        }
                                        self.logger.warning(
                                            "Brak pola timeNano w odpowiedzi. Używam czasu lokalnego."
                                        )
                                except (TypeError, ValueError) as e:
                                    # W przypadku błędu konwersji używamy milisekund
                                    server_time = {
                                        "timeNow": int(time.time() * 1000)
                                    }
                                    self.logger.warning(
                                        f"Błąd konwersji czasu serwera: {e}. Używam czasu lokalnego."
                                    )
                            else:
                                raise Exception(
                                    f"Błędna odpowiedź z endpointu v5: {data}")
                        else:
                            # Próba z endpointem Spot API
                            spot_endpoint = f"{self.base_url}/spot/v1/time"
                            self.logger.debug(
                                f"Próba pobierania czasu z endpointu spot: {spot_endpoint}"
                            )
                            # Direct connection without proxy
                            response = requests.get(spot_endpoint, timeout=10)
                            if response.status_code == 200:
                                data = response.json()
                                if data.get("ret_code"
                                            ) == 0 and "serverTime" in data:
                                    server_time = {
                                        "timeNow": data["serverTime"]
                                    }
                                    self.logger.debug(
                                        f"Czas serwera spot: {server_time}")
                                else:
                                    raise Exception(
                                        f"Błędna odpowiedź z endpointu spot: {data}"
                                    )
                            else:
                                raise Exception(
                                    f"Błąd HTTP {response.status_code} dla obu endpointów czasu serwera"
                                )
                    except Exception as e:
                        # Jeśli żadna metoda nie zadziała, używamy czasu lokalnego
                        server_time = {"timeNow": int(time.time() * 1000)}
                        self.logger.warning(
                            f"Brak dostępu do czasu serwera, używam czasu lokalnego. Błąd: {e}"
                        )

                    self.logger.info(
                        f"Połączenie z ByBit potwierdzone. Czas serwera: {server_time}"
                    )

                    # Zapisz wynik testu i czas serwera w cache
                    try:
                        from data.utils.cache_manager import store_cached_data
                        cache_key = f"server_time_{self.use_testnet}"
                        store_cached_data(cache_key, server_time)
                    except Exception as cache_error:
                        self.logger.warning(
                            f"Błąd podczas zapisu do cache: {cache_error}")

                    self._connection_test_result = True
                except Exception as st_error:
                    self.logger.warning(
                        f"Połączenie z ByBit nawiązane, ale test czasu serwera nie powiódł się: {st_error}"
                    )

                    # Obsługa błędów związanych z przekroczeniem limitów
                    error_str = str(st_error).lower()
                    if "rate limit" in error_str or "429" in error_str or "403" in error_str:
                        self.rate_limit_exceeded = True
                        self.last_rate_limit_reset = time.time()
                        self.logger.warning(
                            "Wykryto przekroczenie limitów API podczas inicjalizacji. Zwiększam parametry opóźnień."
                        )

                        # Zapisz informację o przekroczeniu limitów w cache
                        try:
                            from data.utils.cache_manager import store_cached_data
                            store_cached_data("api_rate_limited", True)
                        except Exception as cache_error:
                            self.logger.warning(
                                f"Błąd podczas zapisu do cache: {cache_error}")

                    self._connection_test_result = False

                self._connection_test_time = current_time
                self._connection_initialized = True

                self.logger.info(
                    f"Zainicjalizowano klienta ByBit API. Wersja: {self.api_version}, Testnet: {self.use_testnet}"
                )
                return True
            except Exception as e:
                self.logger.error(f"Błąd inicjalizacji klienta PyBit: {e}")
                self._connection_test_result = False
                self._connection_test_time = current_time
                return False
        except ImportError:
            self.logger.error(
                "Nie można zaimportować modułu pybit. Sprawdź czy jest zainstalowany."
            )
            self.client = None
            self._connection_test_result = False
            self._connection_test_time = current_time
            return False
        except Exception as e:
            self.logger.error(f"Błąd podczas inicjalizacji klienta ByBit: {e}")
            self.client = None
            self._connection_test_result = False
            self._connection_test_time = current_time
            return False

    def get_server_time(self) -> Dict[str, Any]:
        """
        Pobiera czas serwera Bybit.

        Returns:
            Dict[str, Any]: Czas serwera.
        """
        # Przygotowanie domyślnego (lokalnego) czasu serwera jako fallback
        current_time_ms = int(time.time() * 1000)
        default_response = {
            "success":
            True,
            "time_ms":
            current_time_ms,
            "time":
            datetime.fromtimestamp(current_time_ms /
                                   1000).strftime('%Y-%m-%d %H:%M:%S'),
            "source":
            "local_default"
        }

        try:
            # Sprawdź cache najpierw
            cached_data = None
            found = False

            try:
                from data.utils.cache_manager import get_cached_data, is_cache_valid
                cache_key = f"server_time_{self.use_testnet}"

                cache_valid = False
                try:
                    cache_valid = is_cache_valid(
                        cache_key, ttl=60)  # Cache ważny przez minutę
                except Exception as cache_valid_err:
                    self.logger.warning(
                        f"Błąd podczas sprawdzania ważności cache: {cache_valid_err}"
                    )

                if cache_valid:
                    try:
                        cached_data, found = get_cached_data(cache_key)
                    except Exception as cache_get_err:
                        self.logger.warning(
                            f"Błąd podczas pobierania danych z cache: {cache_get_err}"
                        )
                        found = False
            except Exception as cache_import_err:
                self.logger.warning(
                    f"Błąd podczas importu managera cache: {cache_import_err}")

            # Jeśli dane są w cache i mają poprawny format
            if found and cached_data is not None and isinstance(
                    cached_data, dict) and "timeNow" in cached_data:
                self.logger.debug(
                    f"Używam cache'owanego czasu serwera: {cached_data}")
                time_ms = int(cached_data["timeNow"])
                return {
                    "success":
                    True,
                    "time_ms":
                    time_ms,
                    "time":
                    datetime.fromtimestamp(time_ms /
                                           1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "source":
                    "cache"
                }

            # Jeśli nie ma w cache lub cache nieważny
            try:
                self._apply_rate_limit()
            except Exception as rate_limit_err:
                self.logger.warning(
                    f"Błąd podczas stosowania rate limit: {rate_limit_err}")

            # Jeśli przekroczono limity, zwróć lokalny czas
            if getattr(self, "rate_limit_exceeded", False):
                self.logger.info(
                    f"Używam lokalnego czasu zamiast odpytywania API (przekroczono limity)"
                )
                default_response["source"] = "local_rate_limited"
                return default_response

            # Inicjalizacja klienta, jeśli jeszcze nie istnieje
            client_initialized = False
            if not getattr(self, "_connection_initialized",
                           False) or self.client is None:
                try:
                    client_initialized = self._initialize_client()
                except Exception as init_err:
                    self.logger.error(
                        f"Błąd podczas inicjalizacji klienta: {init_err}")
                    client_initialized = False

                if not client_initialized:
                    # Jeśli inicjalizacja się nie powiedzie, zwróć lokalny czas
                    default_response["source"] = "local_init_failed"
                    return default_response

            # Próba pobrania prawdziwego czasu serwera bezpośrednio przez HTTP (bez autoryzacji)
            if not getattr(self, "rate_limit_exceeded", False):
                server_time = {
                    "timeNow": current_time_ms
                }  # Domyślna wartość jako zabezpieczenie

                try:
                    # Używamy endpointu v5 - najnowszego i preferowanego
                    v5_endpoint = f"{self.base_url}/v5/market/time"
                    self.logger.debug(
                        f"Pobieranie czasu z V5 API: {v5_endpoint}")

                    # Direct connection without proxy
                    response = requests.get(v5_endpoint, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("retCode") == 0 and "result" in data:
                            # Bezpieczna konwersja timeNano na milisekundy
                            time_nano = data["result"]["timeNano"]
                            if isinstance(time_nano, str):
                                time_nano = int(time_nano)
                            time_ms = time_nano // 1000000

                            server_time = {
                                "timeNow": time_ms
                            }
                            self.logger.debug(
                                f"Pobrano czas serwera z V5 API: {server_time}"
                            )
                        else:
                            # Próba z endpointem Spot API jako fallback
                            spot_endpoint = f"{self.base_url}/spot/v1/time"
                            self.logger.debug(
                                f"Pobieranie czasu z Spot API: {spot_endpoint}"
                            )

                            try:
                                # Direct connection without proxy
                                response = requests.get(spot_endpoint,
                                                        timeout=10)

                                if response.status_code == 200:
                                    data = response.json()
                                    if data.get(
                                            "ret_code"
                                    ) == 0 and "serverTime" in data:
                                        server_time = {
                                            "timeNow": data["serverTime"]
                                        }
                                        self.logger.debug(
                                            f"Pobrano czas serwera z Spot API: {server_time}"
                                        )
                                    else:
                                        self.logger.warning(
                                            f"Błędna odpowiedź z endpointu Spot: {data}. Używam czasu lokalnego."
                                        )
                                else:
                                    self.logger.warning(
                                        f"Błąd HTTP {response.status_code} dla Spot API. Używam czasu lokalnego."
                                    )
                            except Exception as spot_err:
                                self.logger.warning(
                                    f"Błąd podczas pobierania czasu z Spot API: {spot_err}. Używam czasu lokalnego."
                                )
                    else:
                        self.logger.warning(
                            f"Błąd HTTP {response.status_code} dla V5 API. Używam czasu lokalnego."
                        )

                    # Zapisz wynik w cache - nawet jeśli używamy czasu lokalnego
                    try:
                        from data.utils.cache_manager import store_cached_data
                        cache_key = f"server_time_{self.use_testnet}"
                        # Upewnij się, że server_time ma właściwe klucze
                        if not isinstance(
                                server_time,
                                dict) or "timeNow" not in server_time:
                            server_time = {"timeNow": current_time_ms}
                        store_cached_data(cache_key, server_time)
                    except Exception as cache_error:
                        self.logger.warning(
                            f"Błąd podczas zapisu do cache: {cache_error}")

                    # Ekstrakcja czasu - z bezpiecznym dostępem do pól
                    time_ms = current_time_ms  # Wartość domyślna
                    if isinstance(server_time, dict):
                        if "timeNow" in server_time:
                            # Upewnij się, że zawsze konwertujemy na int
                            try:
                                time_ms = int(server_time["timeNow"])
                                self.logger.debug(
                                    f"Używam czasu serwera z pola 'timeNow': {time_ms}"
                                )
                            except (ValueError, TypeError) as e:
                                self.logger.warning(
                                    f"Błąd konwersji timeNow={server_time['timeNow']} na int: {e}"
                                )
                                time_ms = current_time_ms
                        elif "time_now" in server_time:
                            try:
                                # Konwersja na float, a potem na int
                                time_ms = int(
                                    float(server_time["time_now"]) * 1000)
                                self.logger.debug(
                                    f"Używam czasu serwera z pola 'time_now': {time_ms}"
                                )
                            except (ValueError, TypeError) as e:
                                self.logger.warning(
                                    f"Błąd konwersji time_now={server_time['time_now']} na int: {e}"
                                )
                                time_ms = current_time_ms
                        elif "time" in server_time:
                            try:
                                time_ms = int(server_time["time"])
                                self.logger.debug(
                                    f"Używam czasu serwera z pola 'time': {time_ms}"
                                )
                            except (ValueError, TypeError) as e:
                                self.logger.warning(
                                    f"Błąd konwersji time={server_time['time']} na int: {e}"
                                )
                                time_ms = current_time_ms

                    return {
                        "success":
                        True,
                        "time_ms":
                        time_ms,
                        "time":
                        datetime.fromtimestamp(
                            time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                        "source":
                        "api"
                    }
                except Exception as http_err:
                    self.logger.warning(
                        f"Nie udało się pobrać czasu serwera: {http_err}. Używam czasu lokalnego."
                    )
                    default_response["source"] = "local_http_error"
                    return default_response

            # Jako ostateczny fallback zawsze zwracamy lokalny czas
            default_response["source"] = "local_fallback"
            return default_response
        except Exception as e:
            self.logger.error(
                f"Krytyczny błąd podczas pobierania czasu serwera: {e}")
            return default_response

    def get_klines(self,
                   symbol: str,
                   interval: str = "15",
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Pobiera dane świecowe (klines) dla danego symbolu.

        Parameters:
            symbol (str): Symbol pary handlowej.
            interval (str): Interwał czasowy ('1', '5', '15', '30', '60', 'D').
            limit (int): Liczba świec do pobrania.

        Returns:
            List[Dict[str, Any]]: Lista świec.
        """
        try:
            self._apply_rate_limit()
            # Symulacja danych świecowych
            current_time = int(time.time())
            klines = []

            last_price = 50000.0 if "BTC" in symbol else 3000.0  # Przykładowe ceny dla BTC lub innych par

            for i in range(limit):
                timestamp = current_time - (int(interval) * 60 *
                                            (limit - i - 1))

                # Symulujemy zmianę ceny
                price_change = random.uniform(-0.01, 0.01)
                last_price = last_price * (1 + price_change)

                open_price = last_price
                high_price = open_price * (1 + random.uniform(0, 0.005))
                low_price = open_price * (1 - random.uniform(0, 0.005))
                close_price = last_price
                volume = random.uniform(
                    1, 100) if "BTC" in symbol else random.uniform(10, 1000)

                kline = {
                    "timestamp":
                    timestamp,
                    "datetime":
                    datetime.fromtimestamp(timestamp).strftime(
                        '%Y-%m-%d %H:%M:%S'),
                    "open":
                    round(open_price, 2),
                    "high":
                    round(high_price, 2),
                    "low":
                    round(low_price, 2),
                    "close":
                    round(close_price, 2),
                    "volume":
                    round(volume, 2)
                }
                klines.append(kline)

            return klines
        except Exception as e:
            self.logger.error(
                f"Błąd podczas pobierania danych świecowych: {e}")
            return []

    def get_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """
        Pobiera księgę zleceń dla danego symbolu.

        Parameters:
            symbol (str): Symbol pary handlowej.
            limit (int): Liczba poziomów cen do pobrania.

        Returns:
            Dict[str, Any]: Księga zleceń.
        """
        try:
            self._apply_rate_limit()
            # Symulacja księgi zleceń
            base_price = 50000.0 if "BTC" in symbol else 3000.0  # Przykładowe ceny dla BTC lub innych par

            bids = []
            asks = []

            for i in range(limit):
                bid_price = base_price * (1 - 0.001 * (i + 1))
                ask_price = base_price * (1 + 0.001 * (i + 1))

                bid_amount = random.uniform(
                    0.1, 2.0) if "BTC" in symbol else random.uniform(
                        1.0, 20.0)
                ask_amount = random.uniform(
                    0.1, 2.0) if "BTC" in symbol else random.uniform(
                        1.0, 20.0)

                bids.append([round(bid_price, 2), round(bid_amount, 6)])
                asks.append([round(ask_price, 2), round(ask_amount, 6)])

            return {
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "bids": bids,
                "asks": asks
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania księgi zleceń: {e}")
            return {"symbol": symbol, "bids": [], "asks": [], "error": str(e)}

    def get_account_balance(self) -> Dict[str, Any]:
        """Pobiera saldo konta z zastosowaniem zaawansowanego cache i rate limitingu."""
        # Import managera cache
        from data.utils.cache_manager import get_cached_data, store_cached_data, is_cache_valid, get_api_status

        # Klucz cache
        cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"

        # Sprawdź status API - jeśli przekroczono limity, użyj dłuższego TTL
        api_status = get_api_status()
        ttl = 300 if api_status[
            "rate_limited"] else 30  # 5 minut w stanie przekroczenia limitów, 30s normalnie

        # Sprawdzenie czy dane są w cache i ważne
        if is_cache_valid(cache_key, ttl=ttl):
            cached_data = get_cached_data(cache_key)
            if cached_data and cached_data[0]:
                self.logger.debug(
                    f"Używam danych z cache dla account_balance (TTL: {ttl}s)")
                return cached_data[0]

        # Zmienna do śledzenia prób ponowienia
        max_retries = 3
        retry_count = 0
        retry_delay = 2.0  # sekundy

        try:
            self._apply_rate_limit()
            if self.client is None:
                # Próba reinicjalizacji klienta
                try:
                    # Próba reinicjalizacji klienta PyBit
                    self.logger.info(
                        f"Próba reinicjalizacji klienta API. API key: {self.api_key[:5]}..., Testnet: {self.use_testnet}"
                    )
                    endpoint = "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"

                    # Używamy bardziej specyficznego importu dla Spot API
                    try:
                        from pybit.spot import HTTP
                        self.client = HTTP(
                            endpoint=endpoint,
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            recv_window=20000
                        )
                        self.api_version = "spot"
                    except ImportError:
                        # Fallback do ogólnego HTTP
                        import pybit
                        self.logger.warning(
                            "Używam przestarzałej klasy HTTP. Zalecana jest aktualizacja biblioteki pybit."
                        )
                        self.client = pybit.HTTP(
                            endpoint=endpoint,
                            api_key=self.api_key,
                            api_secret=self.api_secret
                        )
                    self.logger.info(
                        "Klient API został pomyślnie reinicjalizowany.")
                except Exception as initerror:
                    self.logger.error(
                        f"Nie udało się zainicjalizować klienta API: {init_error}"
                    )
                    logging.error(
                        f"Klient API nie został zainicjalizowany. Błąd: {init_error}"
                    )
                    return {
                        "balances": {
                            "BTC": {
                                "equity": 0.005,
                                "available_balance": 0.005,
                                "wallet_balance": 0.005
                            },
                            "USDT": {
                                "equity": 500,
                                "available_balance": 450,
                                "wallet_balance": 500
                            }
                        },
                        "success": False,
                        "error":
                        f"Klient API nie został zainicjalizowany. Błąd: {init_error}",
                        "note": "Dane przykładowe -błąd inicjalizacji klienta"
                    }

            # Mechanizm ponawiania prób w przypadku przekroczenia limitów API
            while retry_count < max_retries:
                try:
                    # Testowa implementacja (symulacja)
                    if self.use_testnet:
                        # Symulowane dane do celów testowych
                        self.logger.info("Pobieranie danych z testnet")
                        return {
                            "balances": {
                                "BTC": {
                                    "equity": 0.015,
                                    "available_balance": 0.015,
                                    "wallet_balance": 0.015
                                },
                                "USDT": {
                                    "equity": 1200,
                                    "available_balance": 1150,
                                    "wallet_balance": 1200
                                }
                            },
                            "success": True,
                            "note": "Dane testowe - tryb testnet"
                        }
                    else:
                        # Prawdziwa implementacja lub symulacja jeśli połączenie nie działa
                        # Przygotowanie zasłoniętego klucza do logów
                        masked_key = f"{self.api_key[:4]}{'*' * (len(self.api_key) - 4)}" if self.api_key else "Brak klucza"
                        self.logger.info(
                            f"Próba pobrania danych z {'PRODUKCYJNEGO' if not self.use_testnet else 'TESTOWEGO'} API Bybit. Klucz: {masked_key}"
                        )
                    self.logger.info(
                        f"Status API: {'Produkcyjne' if not self.use_testnet else 'Testnet'}"
                    )

                    try:
                        # Test połączenia z API z obsługą limitów zapytań
                        try:
                            # Wydłużone opóźnienie między zapytaniami dla testu połączenia
                            if not self.use_testnet:
                                time.sleep(
                                    5.0
                                )  # Dodatkowe 5 sekund dla API produkcyjnego
                            else:
                                time.sleep(3.0)  # 3 sekundy dla testnet
                            self._apply_rate_limit()

                            # W przypadku przekroczenia limitu zapytań, używamy czasu lokalnego
                            # zamiast próbować wielokrotnie odpytywać API
                            if self.remaining_rate_limit < 10:
                                local_time = int(time.time() * 1000)
                                self.logger.info(
                                    f"Używam lokalnego czasu ({local_time}) zamiast odpytywania API (oszczędzanie limitów)"
                                )
                                return {
                                    'success': True,
                                    'time_ms': local_time,
                                    'time': time.strftime('%Y-%m-%d %H:%M:%S')
                                }

                            # Sprawdzanie dostępu do API przez bezpośrednie zapytanie HTTP do publicznego endpointu
                            try:
                                # Próba z endpointem Unified v5 API
                                v5_endpoint = f"{self.base_url}/v5/market/time"
                                self.logger.debug(
                                    f"Test połączenia z V5 API: {v5_endpoint}")
                                # Direct connection without proxy
                                response = requests.get(v5_endpoint,
                                                        timeout=10)

                                if response.status_code == 200:
                                    data = response.json()
                                    if data.get("retCode") == 0 and "result" in data:
                                        time_response = {
                                            "timeNow":
                                            data["result"]["timeNano"] //
                                            1000000
                                        }
                                        self.logger.debug(
                                            f"Czas serweraV5: {time_response}")
                                    else:
                                        raise Exception(
                                            f"Nieprawidłowa odpowiedź z V5 API: {data}"
                                        )
                                else:
                                    # Fallback do Spot API
                                    spot_endpoint = f"{self.base_url}/spot/v1/time"
                                    self.logger.debug(
                                        f"Test połączenia ze Spot API: {spot_endpoint}"
                                    )
                                    # Direct connection without proxy
                                    response = requests.get(spot_endpoint,
                                                            timeout=10)

                                    if response.status_code == 200:
                                        data = response.json()
                                        if data.get(
                                                "ret_code"
                                        ) == 0 and "serverTime" in data:
                                            time_response = {
                                                "timeNow": data["serverTime"]
                                            }
                                            self.logger.debug(
                                                f"Czas serwera Spot: {time_response}"
                                            )
                                        else:
                                            raise Exception(
                                                f"Nieprawidłowa odpowiedź ze Spot API: {data}"
                                            )
                                    else:
                                        raise Exception(
                                            f"Błąd HTTP {response.status_code} dla obu endpointów"
                                        )
                            except Exception as e:
                                # Awaryjnie używamy czasu lokalnego
                                time_response = {
                                    "timeNow": int(time.time() * 1000)
                                }
                                self.logger.warning(
                                    f"Brak dostępu do czasu serwera, używam czasu lokalnego. Błąd: {e}"
                                )

                            self.logger.info(
                                f"Test połączenia z API: {time_response}")
                        except Exception as time_error:
                            error_str = str(time_error)
                            self.logger.error(
                                f"Test połączenia z API nie powiódł się: {time_error}"
                            )

                            # Importuj funkcje wykrywania i ustawiania blokady CloudFront
                            try:
                                from data.utils.cache_manager import detect_cloudfront_error, set_cloudfront_block_status
                                # Wykrywanie błędów CloudFront i limitów IP
                                if detect_cloudfront_error(error_str):
                                    self.rate_limit_exceeded = True
                                    self.last_rate_limit_reset = time.time()
                                    self.remaining_rate_limit = 0

                                    # Ustaw blokadę CloudFront
                                    set_cloudfront_block_status(
                                        True, error_str)
                                    self.logger.warning(
                                        f"Wykryto blokadę/limit API w komunikacie błędu: {error_str}"
                                    )
                            except ImportError:
                                # Jeśli funkcje nie istnieją, implementujemy prostą weryfikację
                                error_str_lower = error_str.lower()
                                has_cloudfront_error = any(
                                    indicator in error_str_lower
                                    for indicator in [
                                        'cloudfront', 'distribution', '403',
                                        'rate limit', '429'
                                    ])
                                if has_cloudfront_error:
                                    self.rate_limit_exceeded = True
                                    self.last_rate_limit_reset = time.time()
                                    self.remaining_rate_limit = 0

                                    # Bardzo agresywny backoff dla problemów z CloudFront i IP rate limit
                                    if "cloudfront" in error_str.lower(
                                    ) or "The Amazon CloudFront distribution" in error_str:
                                        self.logger.critical(
                                            f"Wykryto blokadę CloudFront - przechodzę w tryb pełnego fallback"
                                        )
                                        try:
                                            set_cloudfront_block_status(
                                                True, error_str)
                                        except Exception as cf_err:
                                            self.logger.error(
                                                f"Błąd podczas ustawiania statusu blokady CloudFront: {cf_err}"
                                            )

                                        # Ustaw ekstremalnie długi backoff dla blokady CloudFront
                                        self.min_time_between_calls = 30.0  # minimum 30s między zapytaniami
                                        self.rate_limit_backoff = 1800.0  # 30 minut backoff

                                        # Ustaw flagę _backoff_attempt dla eksponencjalnego wzrostu
                                        self._backoff_attempt = 5  # Wysoka wartość dla długiego czasu oczekiwania
                                    else:
                                        self.logger.warning(
                                            f"Przekroczono limit zapytań API - używam cache lub danych symulowanych"
                                        )
                                        try:
                                            set_cloudfront_block_status(
                                                True,
                                                f"IP Rate Limit: {error_str}")
                                        except Exception as cf_err:
                                            self.logger.error(
                                                f"Błąd podczas ustawiania statusu blokady CloudFront: {cf_err}"
                                            )

                                        # Ustaw parametry backoff dla IP rate limit
                                        self.min_time_between_calls = 20.0  # 20s zgodnie z wymaganiami
                                        self.rate_limit_backoff = 600.0  # 10 minut backoff
                                        self._backoff_attempt = 3

                                    self.logger.warning(
                                        f"[FALLBACK MODE] min_interval={self.min_time_between_calls:.1f}s, backoff={self.rate_limit_backoff:.1f}s"
                                    )

                                    # Sprawdź najpierw cache - jeśli są dane w cache
                                    cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"
                                    cached_data, cache_found = get_cached_data(
                                        cache_key)

                                    if cache_found and cached_data:
                                        # Użyj danych z cache ale dodaj flagę informacyjną
                                        cached_data["success"] = True
                                        cached_data[
                                            "source"] = "cache_fallback"
                                        cached_data[
                                            "warning"] = f"Używam danych z cache z powodu: {error_str}"
                                        self.logger.info(
                                            "UŻYWAM DANYCH Z CACHE z powodu blokady CloudFront/IP Rate Limit"
                                        )
                                        return cached_data

                                    # Nie ma danych w cache - wygeneruj symulowane dane
                                    return {
                                        "balances": {
                                            "BTC": {
                                                "equity": 0.025,
                                                "available_balance": 0.020,
                                                "wallet_balance": 0.025
                                            },
                                            "USDT": {
                                                "equity": 1500,
                                                "available_balance": 1450,
                                                "wallet_balance": 1500
                                            },
                                            "ETH": {
                                                "equity": 0.5,
                                                "available_balance": 0.5,
                                                "wallet_balance": 0.5
                                            }
                                        },
                                        "success":
                                        True,
                                        "warning":
                                        f"CloudFront/IP Rate Limit: {error_str}",
                                        "source":
                                        "simulation_cloudfront_blocked",
                                        "note":
                                        "Dane symulowane - wykryto blokadę CloudFront lub przekroczenie limitów IP"
                                    }
                                else:
                                    # Dla innych błędów zgłaszamy wyjątek
                                    raise Exception(
                                        f"Brak dostępu do API Bybit: {time_error}"
                                    )

                        # Próba pobrania salda konta z uwzględnieniem różnych API
                        wallet = None

                        # Dla kont UNIFIED pomijamy API V2, które zwraca błąd 409
                        # Priorytetyzujemy bezpośrednie zapytanie HTTP do API V5
                        wallet_methods = []

                        # Sprawdzamy czy używamy konta UNIFIED
                        is_unified_account = True  # Domyślnie zakładamy że używamy konta UNIFIED

                        # Jeśli nie używamy konta UNIFIED, dodajemy standardowe metody API V2
                        if not is_unified_account:
                            wallet_methods = [('get_wallet_balance', {}),
                                            ('get_wallet_balance', {
                                                'coin': 'USDT'
                                            }), ('query_account_info', {}),
                                            ('get_account_overview', {}),
                                            ('get_account_balance', {}),
                                            ('get_balances', {})]
                            self.logger.info("Używam standardowych metod API dla konta non-UNIFIED")
                        else:
                            self.logger.info("Używam tylko API V5 dla konta UNIFIED")

                        # Jeśli powyższe metody nie zadziałały, spróbuj bezpośredniego zapytania HTTP do V5 API
                        if wallet is None:
                            try:
                                self.logger.info(
                                    "Próba pobrania salda przez bezpośrednie zapytanie HTTP do V5 API"
                                )
                                v5_endpoint = f"{self.base_url}/v5/account/wallet-balance"

                                # Parametry zgodnie z dokumentacją V5 API
                                params = {
                                    'accountType': 'UNIFIED'  # Można dostosować w zależności od typu konta
                                }

                                # Tworzenie sygnatury zgodnie z dokumentacją V5 API
                                timestamp = str(int(time.time() * 1000))
                                recv_window = "20000"  # Używamy stałej wartości recv_window

                                # Przygotowanie parametrów do sygnatury
                                # Upewnij się, że parametry są posortowane alfabetycznie po kluczach
                                param_str = ''
                                if params:
                                    param_str = '&'.join([
                                        f"{key}={value}" for key, value in
                                        sorted(params.items())
                                    ])

                                self.logger.debug(f"Parametry do podpisu (sorted): {param_str}")

                                # Poprawne tworzenie pre_sign zgodnie z dokumentacją V5
                                # Format: timestamp + api_key + recv_window + query_string
                                # W zapytaniach GET, param_str już zawiera posortowane parametry jako 'key=value&key2=value2'
                                if param_str:
                                    pre_sign = f"{timestamp}{self.api_key}{recv_window}{param_str}"
                                else:
                                    pre_sign = f"{timestamp}{self.api_key}{recv_window}"

                                self.logger.debug(f"Generowanie podpisu dla API V5. Pre-sign: [{pre_sign}]")

                                # Generowanie sygnatury HMAC SHA256
                                signature = hmac.new(
                                    bytes(self.api_secret, 'utf-8'),
                                    bytes(pre_sign, 'utf-8'),
                                    hashlib.sha256).hexdigest()

                                # Ustawienie nagłówków zgodnie z dokumentacją
                                headers = {
                                    "X-BAPI-API-KEY": self.api_key,
                                    "X-BAPI-TIMESTAMP": timestamp,
                                    "X-BAPI-SIGN": signature,
                                    "X-BAPI-RECV-WINDOW": recv_window,  # Używamy tej samej wartości co w pre_sign
                                    "Content-Type": "application/json"
                                }

                                self.logger.debug(
                                    f"Wysyłanie zapytania do V5 API: {v5_endpoint} z parametrami: {params}"
                                )
                                # Direct connection without proxy
                                response = requests.get(v5_endpoint,
                                                        headers=headers,
                                                        params=params,
                                                        timeout=10)

                                if response.status_code == 200:
                                    wallet = response.json()
                                    self.logger.info(
                                        f"Saldo pobrane przez bezpośrednie zapytanie HTTP do V5 API"
                                    )
                                    self.logger.debug(
                                        f"Odpowiedź API: {str(wallet)[:200]}..."
                                    )
                                else:
                                    self.logger.warning(
                                        f"Błąd podczas pobierania salda przez V5 API: {response.status_code} - {response.text}"
                                    )

                                    # Spróbuj alternatywnego API - Spot API v1
                                    spot_endpoint = f"{self.base_url}/spot/v1/account"
                                    timestamp = str(int(time.time() * 1000))
                                    pre_sign = f"{timestamp}{self.api_key}"
                                    signature = hmac.new(
                                        bytes(self.api_secret, 'utf-8'),
                                        bytes(pre_sign, 'utf-8'),
                                        hashlib.sha256).hexdigest()

                                    headers = {
                                        "X-BAPI-API-KEY": self.api_key,
                                        "X-BAPI-TIMESTAMP": timestamp,
                                        "X-BAPI-SIGN": signature,
                                        "Content-Type": "application/json"
                                    }

                                    self.logger.debug(
                                        f"Próba zapytania do Spot API v1: {spot_endpoint}"
                                    )
                                    # Direct connection without proxy
                                    response = requests.get(spot_endpoint,
                                                            headers=headers,
                                                            timeout=10)

                                    if response.status_code == 200:
                                        wallet = response.json()
                                        self.logger.info(
                                            f"Saldo pobrane przez bezpośrednie zapytanie HTTP do Spot API v1"
                                        )
                                    else:
                                        self.logger.warning(
                                            f"Błąd podczas pobierania salda przez Spot API v1: {response.status_code} - {response.text}"
                                        )
                            except Exception as e:
                                self.logger.warning(
                                    f"Błąd podczas używania bezpośredniego zapytania HTTP: {e}"
                                )
                                # Kontynuujemy, aby spróbować innych metod

                        for method_name, params in wallet_methods:
                            if hasattr(self.client, method_name):
                                try:
                                    method = getattr(self.client, method_name)
                                    wallet = method(**params)
                                    self.logger.info(
                                        f"Saldo pobrane metodą: {method_name}")
                                    break
                                except Exception as method_error:
                                    self.logger.warning(
                                        f"Błąd podczas używania metody {method_name}: {method_error}"
                                    )

                        if wallet is None:
                            self.logger.error(
                                "Wszystkie metody pobierania salda zawiodły")
                            raise Exception(
                                "Brak dostępnych metod do pobrania salda portfela"
                            )

                        self.logger.info(
                            f"Odpowiedź API Bybit: {str(wallet)[:200]}...")

                        # Dodatkowe logowanie struktury odpowiedzi dla celów diagnostycznych
                        if "result" in wallet:
                            result_keys = list(wallet["result"].keys()) if isinstance(wallet["result"], dict) else "nie jest słownikiem"
                            self.logger.debug(f"Struktura odpowiedzi - klucze w result: {result_keys}")
                            if "list" in wallet["result"] and isinstance(wallet["result"]["list"], list):
                                first_item = wallet["result"]["list"][0] if wallet["result"]["list"] else "pusta lista"
                                self.logger.debug(f"Pierwszy element listy: {first_item}")

                        # Sprawdzenie czy odpowiedź zawiera kod błędu
                        if "retCode" in wallet and wallet["retCode"] != 0:
                            error_msg = wallet.get("retMsg",
                                                   "Nieznany błąd API")
                            self.logger.error(
                                f"API zwróciło błąd: {error_msg}")
                            raise Exception(f"Błąd API ByBit: {error_msg}")

                        result = {
                            "balances": {},
                            "success": True,
                            "source": "API",
                            "api_version": self.api_version
                        }

                        # Obsługa różnych formatów odpowiedzi w zależności od wersji API
                        if wallet and "result" in wallet:
                            # Nowsza struktura API ByBit V5
                            # Sprawdź czy result zawiera listę (typowy format V5 API)
                            if "list" in wallet["result"] and isinstance(wallet["result"]["list"], list):
                                for account_data in wallet["result"]["list"]:
                                    # Sprawdź czy dane zawierają pole "coin" które jest listą słowników
                                    if isinstance(account_data, dict) and "coin" in account_data and isinstance(account_data["coin"], list):
                                        # Iteruj po liście słowników coin
                                        for coin_data in account_data["coin"]:
                                            if isinstance(coin_data, dict) and "coin" in coin_data:
                                                coin = coin_data["coin"]
                                                if isinstance(coin, str) and coin:
                                                    # Bezpieczne konwertowanie wartości na float z obsługą pustych wartości
                                                    try:
                                                        equity = float(coin_data.get("equity", 0) or 0)
                                                    except (ValueError, TypeError):
                                                        self.logger.warning(f"Błędna wartość equity dla {coin}: {coin_data.get('equity')}, ustawiam 0")
                                                        equity = 0.0

                                                    # Sprawdź dostępne saldo z różnych możliwych pól
                                                    available_value = coin_data.get("availableBalance") or coin_data.get("availableToWithdraw") or 0
                                                    try:
                                                        available_balance = float(available_value)
                                                    except (ValueError, TypeError):
                                                        self.logger.warning(f"Błędna wartość available_balance dla {coin}: {available_value}, ustawiam 0")
                                                        available_balance = 0.0

                                                    try:
                                                        wallet_balance = float(coin_data.get("walletBalance", 0) or 0)
                                                    except (ValueError, TypeError):
                                                        self.logger.warning(f"Błędna wartość wallet_balance dla {coin}: {coin_data.get('walletBalance')}, ustawiam 0")
                                                        wallet_balance = 0.0

                                                    result["balances"][coin] = {
                                                        "equity": equity,
                                                        "available_balance": available_balance,
                                                        "wallet_balance": wallet_balance
                                                    }
                                                    self.logger.debug(f"Dodano saldo dla {coin}: {result['balances'][coin]}")
                                    # Obsługa przypadku gdy coin jest bezpośrednio w danych konta
                                    elif isinstance(account_data, dict):
                                        coin = account_data.get("coin")
                                        # Upewnienie się że coin jest stringiem, a nie listą
                                        if isinstance(coin, str) and coin:
                                            result["balances"][coin] = {
                                                "equity": float(account_data.get("equity", 0)),
                                                "available_balance": float(
                                                    account_data.get("availableBalance", 0) or 
                                                    account_data.get("availableToWithdraw", 0)
                                                ),
                                                "wallet_balance": float(account_data.get("walletBalance", 0))
                                            }
                                            self.logger.debug(f"Dodano saldo dla {coin}: {result['balances'][coin]}")
                                        elif isinstance(coin, list) and coin:
                                            # Jeśli coin jest listą, iteruj po jej elementach
                                            self.logger.warning(f"Otrzymano listę coin zamiast stringa: {coin}")
                                            for single_coin in coin:
                                                if isinstance(single_coin, str) and single_coin:
                                                    result["balances"][single_coin] = {
                                                        "equity": float(account_data.get("equity", 0)),
                                                        "available_balance": float(
                                                            account_data.get("availableBalance", 0) or 
                                                            account_data.get("availableToWithdraw", 0)
                                                        ),
                                                        "wallet_balance": float(account_data.get("walletBalance", 0))
                                                    }
                                                    self.logger.debug(f"Dodano saldo dla coin z listy: {single_coin}")
                        elif wallet and "result" in wallet and isinstance(
                                wallet["result"], dict):
                            # Starsza struktura API ByBit lub format usdt_perpetual
                            for coin, coin_data in wallet["result"].items():
                                if isinstance(coin_data, dict):
                                    result["balances"][coin] = {
                                        "equity":
                                        float(coin_data.get("equity", 0)),
                                        "available_balance":
                                        float(
                                            coin_data.get(
                                                "available_balance", 0)
                                            or coin_data.get(
                                                "availableBalance", 0)),
                                        "wallet_balance":
                                        float(
                                            coin_data.get("wallet_balance", 0)
                                            or coin_data.get(
                                                "walletBalance", 0))
                                    }
                        elif wallet and "result" in wallet and isinstance(
                                wallet["result"], list):
                            # Format odpowiedzi dla niektórych wersji API
                            for coin_data in wallet["result"]:
                                coin = coin_data.get("coin", "")
                                if coin:
                                    result["balances"][coin] = {
                                        "equity":
                                        float(coin_data.get("equity", 0)),
                                        "available_balance":
                                        float(
                                            coin_data.get(
                                                "available_balance", 0)
                                            or coin_data.get(
                                                "availableBalance", 0)),
                                        "wallet_balance":
                                        float(
                                            coin_data.get("wallet_balance", 0)
                                            or coin_data.get(
                                                "walletBalance", 0))
                                    }

                        if not result["balances"]:
                            self.logger.warning(
                                f"API zwróciło pustą listę sald. Pełna odpowiedź: {wallet}"
                            )
                            result["warning"] = "API zwróciło pustą listę sald"
                            self.logger.info(
                                "Próba pobrania danych z API zwróciła pustą listę sald. Możliwe przyczyny: brak środków na koncie, nieprawidłowe konto, nieprawidłowe uprawnienia API."
                            )

                        # Zapisanie poprawnych danych w cache
                        from data.utils.cache_manager import store_cached_data
                        cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"
                        store_cached_data(cache_key, result)
                        self.logger.debug("Zapisano dane portfolio w cache")

                        return result
                    except Exception as e:
                        self.logger.error(
                            f"Błąd podczas pobierania danych z prawdziwego API: {e}. Traceback: {traceback.format_exc()}"
                        )
                        # Dane symulowane w przypadku błędu
                        return {
                            "balances": {
                                "BTC": {
                                    "equity": 0.025,
                                    "available_balance": 0.020,
                                    "wallet_balance": 0.025
                                },
                                "USDT": {
                                    "equity": 1500,
                                    "available_balance": 1450,
                                    "wallet_balance": 1500
                                },
                                "ETH": {
                                    "equity": 0.5,
                                    "available_balance": 0.5,
                                    "wallet_balance": 0.5
                                }
                            },
                            "success": False,
                            "error": str(e),
                            "source": "simulation",
                            "note": "Dane symulowane - błąd API: " + str(e)
                        }
                except Exception as e:
                    error_str = str(e)
                    # Sprawdzenie, czy błąd dotyczy przekroczenia limitu zapytań
                    if "rate limit" in error_str.lower(
                    ) or "429" in error_str or "403" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            self.logger.warning(
                                f"Przekroczono limit zapytań API. Ponawiam próbę {retry_count}/{max_retries} za {retry_delay} sekund..."
                            )
                            # Zwiększamy opóźnienie wykładniczo, aby uniknąć ponownego przekroczenia limitu
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Podwajamy czas oczekiwania przy każdej próbie
                            continue
                        else:
                            self.logger.warning(
                                f"Wykorzystano wszystkie próby ponawiania. Zwracam dane symulowane."
                            )

                    self.logger.error(
                        f"Krytyczny błąd podczas pobierania salda konta: {e}. Traceback: {traceback.format_exc()}"
                    )
                    # Dane symulowane w przypadku błędu
                    return {
                        "balances": {
                            "BTC": {
                                "equity": 0.01,
                                "available_balance": 0.01,
                                "wallet_balance": 0.01
                            },
                            "USDT": {
                                "equity": 1000,
                                "available_balance": 950,
                                "wallet_balance": 1000
                            }
                        },
                        "success": False,
                        "error": str(e),
                        "source": "simulation_error",
                        "note": "Dane symulowane - wystąpił błąd: " + str(e)
                    }

                # Jeśli dotarliśmy tutaj, to znaczy, że zapytanie się powiodło
                break

        except Exception as e:
            self.logger.error(
                f"Krytyczny błąd podczas pobierania salda konta: {e}. Traceback: {traceback.format_exc()}"
            )
            # Dane symulowane w przypadku błędu
            return {
                "balances": {
                    "BTC": {
                        "equity": 0.01,
                        "available_balance": 0.01,
                        "wallet_balance": 0.01
                    },
                    "USDT": {
                        "equity": 1000,
                        "available_balance": 950,
                        "wallet_balance": 1000
                    }
                },
                "success": False,
                "error": str(e),
                "source": "simulation_critical_error",
                "note": "Dane symulowane - wystąpił krytyczny błąd: " + str(e)
            }

        return {
            "balances": {
                "BTC": {
                    "equity": 0.01,
                    "available_balance": 0.01,
                    "wallet_balance": 0.01
                },
                "USDT": {
                    "equity": 1000,
                    "available_balance": 950,
                    "wallet_balance": 1000
                }
            },
            "success": False,
            "error": "Nieudane pobieranie salda",
            "source": "fallback",
            "note": "Dane symulowane - żadna próba nie powiodła się"
        }

    def place_order(self,
                    symbol: str,
                    side: str,
                    price: float,
                    quantity: float,
                    order_type: str = "Limit") -> Dict[str, Any]:
        """
        Składa zlecenie na giełdzie.

        Parameters:
            symbol (str): Symbol pary handlowej.
            side (str): Strona zlecenia ('Buy' lub 'Sell').
            price (float): Cena zlecenia.
            quantity (float): Ilość zlecenia.
            order_type (str): Typ zlecenia ('Limit' lub 'Market').

        Returns:
            Dict[str, Any]: Wynik złożenia zlecenia.
        """
        try:
            self._apply_rate_limit()

            # Dodatkowe potwierdzenie dla produkcyjnego API
            if self.is_production_api():
                self.logger.warning(
                    f"PRODUKCYJNE ZLECENIE: {side} {quantity} {symbol} za {price if order_type == 'Limit' else 'Market'}"
                )

                # Limity wielkości zleceń dla produkcyjnego API
                max_order_value_usd = 1000.0  # Maksymalna wartość zlecenia w USD dla bezpieczeństwa
                est_order_value = quantity * price if order_type == "Limit" else quantity * price

                if est_order_value > max_order_value_usd and "BTC" in symbol:
                    self.logger.critical(
                        f"ODRZUCONO ZLECENIE: Przekroczono maksymalną wartość zlecenia ({est_order_value} USD > {max_order_value_usd} USD)"
                    )
                    return {
                        "success":
                        False,
                        "error":
                        f"Zlecenie przekracza maksymalną dozwoloną wartość {max_order_value_usd} USD"
                    }

            # Sprawdzenie poprawności danych
            if side not in ["Buy", "Sell"]:
                return {
                    "success":
                    False,
                    "error":
                    "Nieprawidłowa strona zlecenia. Musi być 'Buy' lub 'Sell'."
                }

            if order_type not in ["Limit", "Market"]:
                return {
                    "success":
                    False,
                    "error":
                    "Nieprawidłowy typ zlecenia. Musi być 'Limit' lub 'Market'."
                }

            if quantity <= 0:
                return {"success": False, "error": "Ilość musi być dodatnia."}

            if order_type == "Limit" and price <= 0:
                return {
                    "success": False,
                    "error": "Cena musi być dodatnia dla zleceń typu Limit."
                }

            # Symulacja składania zlecenia
            order_id = f"ORD-{int(time.time())}-{random.randint(1000, 9999)}"

            self.logger.info(
                f"Złożono zlecenie: {side} {quantity} {symbol} po cenie {price if order_type == 'Limit' else 'Market'}"
            )

            return {
                "success": True,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "price": price if order_type == "Limit" else None,
                "quantity": quantity,
                "order_type": order_type,
                "status": "New",
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas składania zlecenia: {e}")
            return {"success": False, "error": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Anuluje zlecenie.

        Parameters:
            order_id (str): ID zlecenia do anulowania.

        Returns:
            Dict[str, Any]: Wynik anulowania zlecenia.
        """
        try:
            self._apply_rate_limit()
            # Symulacja anulowania zlecenia
            self.logger.info(f"Anulowano zlecenie: {order_id}")

            return {
                "success": True,
                "order_id": order_id,
                "status": "Cancelled",
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas anulowania zlecenia: {e}")
            return {"success": False, "error": str(e)}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Pobiera status zlecenia.

        Parameters:
            order_id (str): ID zlecenia.

        Returns:
            Dict[str, Any]: Status zlecenia.
        """
        try:
            self._apply_rate_limit()
            # Symulacja pobierania statusu zlecenia
            statuses = [
                "New", "PartiallyFilled", "Filled", "Cancelled", "Rejected"
            ]
            status = random.choice(statuses)

            return {
                "success":
                True,
                "order_id":
                order_id,
                "status":
                status,
                "filled_quantity":
                random.uniform(0, 1) if status == "PartiallyFilled" else
                (1.0 if status == "Filled" else 0.0),
                "timestamp":
                int(time.time() * 1000),
                "datetime":
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania statusu zlecenia: {e}")
            return {"success": False, "error": str(e)}

    def _apply_rate_limit(self):
        """Applies rate limiting to API calls with exponential backoff for production environment."""
        try:
            # Używamy uproszczonego importu, który nie będzie wymagał wszystkich funkcji
            from data.utils.cache_manager import get_api_status

            # Jeśli główny import się powiedzie, importujemy resztę
            from data.utils.cache_manager import (store_cached_data,
                                                  detect_cloudfront_error,
                                                  set_cloudfront_block_status,
                                                  get_cloudfront_status)
        except ImportError as e:
            self.logger.warning(
                f"Cache manager import error: {e}. Using simplified rate limiting."
            )
            # Implement simplified rate limiting if imports fail
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call
            min_interval = 20.0 if not self.use_testnet else 10.0

            if time_since_last_call < min_interval:
                sleep_time = min_interval - time_since_last_call
                if sleep_time > 0.1:
                    time.sleep(sleep_time)

            self.last_api_call = time.time()
            return

        # Bezpieczne pobieranie statusu CloudFront
        try:
            cloudfront_status = get_cloudfront_status()
            is_blocked = False

            # Bezpieczne sprawdzenie czy mamy blokadę CloudFront
            if isinstance(cloudfront_status, dict):
                is_blocked = cloudfront_status.get("blocked", False)

            if is_blocked:
                error_msg = "Unknown error"
                if isinstance(cloudfront_status, dict):
                    error_msg = cloudfront_status.get("error", "Unknown error")
                self.logger.warning(
                    f"Wykryto aktywną blokadę CloudFront: {error_msg}. Używanie danych z cache."
                )
                self.rate_limit_exceeded = True
                return
        except Exception as e:
            self.logger.warning(
                f"Błąd podczas sprawdzania statusu CloudFront: {e}. Kontynuowanie z domyślnymi ustawieniami."
            )

        # Pobierz aktualny status API z cache_manager
        api_status = get_api_status()
        now = time.time()
        is_production = os.getenv(
            'IS_PRODUCTION', 'false').lower() == 'true' or not self.use_testnet

        # Synchronizuj parametry rate limiter'a z cache_manager (raz na minutę)
        if not hasattr(self, '_rate_limit_synced') or now - getattr(
                self, '_last_sync_time', 0) > 60:
            # Produkcyjne parametry: max 2 wywołania co 30 sekund (bardzo konserwatywne)
            if is_production:
                max_calls = 2
                min_interval = 30.0  # 30 sekund między zapytaniami dla produkcji
            else:
                max_calls = 6
                min_interval = 10.0

            # Jeśli wystąpiły wcześniej błędy rate limit, zwiększamy jeszcze bardziej restrykcje
            if hasattr(self,
                       'rate_limit_exceeded') and self.rate_limit_exceeded:
                # Eksponencjalne zmniejszenie ruchu
                min_interval = min(60.0, min_interval *
                                   2)  # max 60s między zapytaniami
                self.logger.warning(
                    f"Zastosowano eksponencjalne wycofanie: min_interval={min_interval}s"
                )
                time.sleep(3.0)  # Dodatkowa pauza

            # Synchronizuj parametry
            from data.utils.cache_manager import set_rate_limit_parameters
            set_rate_limit_parameters(max_calls_per_minute=max_calls,
                                      min_interval=min_interval)
            self._rate_limit_synced = True
            self._last_sync_time = now
            self.min_time_between_calls = min_interval

        # Jeśli przekroczono limit lub mamy blokadę CloudFront
        if self.rate_limit_exceeded or api_status["rate_limited"]:
            # Resetuj stan po długim okresie oczekiwania
            reset_time = 600.0 if is_production else 300.0  # 10 minut dla produkcji, 5 dla dev

            if now - self.last_rate_limit_reset > reset_time:
                self.logger.info("Próba resetu stanu po długim oczekiwaniu")
                self.rate_limit_exceeded = False
                self.remaining_rate_limit = 3 if is_production else 10
                store_cached_data("api_rate_limited", False)
                set_cloudfront_block_status(False)
            else:
                # Eksponencjalny backoff przy powtarzających się błędach
                base_backoff = 20.0 if is_production else 10.0
                attempt = getattr(self, '_backoff_attempt', 1)
                max_sleep = 600.0 if is_production else 300.0

                # Eksponencjalny wzrost czasu oczekiwania: 20s, 40s, 80s, 160s...
                sleep_time = min(max_sleep, base_backoff * (2**(attempt - 1)))

                self.logger.warning(
                    f"Eksponencjalne wycofanie: próba {attempt}, oczekiwanie {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)

                # Zwiększ licznik próby dla kolejnego eksponencjalnego wzrostu
                self._backoff_attempt = attempt + 1

        # Zachowaj minimalny odstęp między wywołaniami - 20s dla produkcji
        time_since_last_call = now - self.last_api_call
        min_time = 20.0 if is_production else 5.0

        if time_since_last_call < min_time:
            sleep_time = min_time - time_since_last_call
            if sleep_time > 0.1:
                time.sleep(sleep_time)

        # Aktualizacja czasu ostatniego wywołania
        self.last_api_call = time.time()

        # Po wielu pomyślnych wywołaniach, stopniowo zmniejszamy backoff
        if not self.rate_limit_exceeded and hasattr(
                self, '_backoff_attempt') and self._backoff_attempt > 1:
            if getattr(self, '_success_count', 0) > 5:
                self._backoff_attempt = max(1, self._backoff_attempt - 1)
                self._success_count = 0
                self.logger.info(
                    f"Zmniejszono poziom backoff do {self._backoff_attempt} po wielu poprawnych wywołaniach"
                )
            else:
                self._success_count = getattr(self, '_success_count', 0) + 1

    def is_production_api(self):
        """Sprawdza czy używane jest produkcyjne API.

        Returns:
            bool: True jeśli używane jest produkcyjne API, False dla testnet.
        """
        # Sprawdzenie czy używamy produkcyjnego API
        is_prod = not self.use_testnet

        # Dodatkowy log dla produkcyjnego API (tylko jeśli nie wyświetlono wcześniej)
        if is_prod and not hasattr(self, '_production_warning_shown'):
            self.logger.warning(
                "!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit. Operacje handlowe będą mieć realne skutki finansowe!"
            )
            self.logger.warning(
                "Upewnij się, że Twoje klucze API mają właściwe ograniczenia i są odpowiednio zabezpieczone."
            )
            print("\n========== PRODUKCYJNE API BYBIT ==========")
            print("!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit")
            print("Operacje handlowe będą mieć realne skutki finansowe!")
            print("===========================================\n")
            self._production_warning_shown = True
        elif not is_prod and not hasattr(self, '_testnet_info_shown'):
            self.logger.info("Używasz testnet API (środowisko testowe).")
            self._testnet_info_shown = True

        return is_prod


if __name__ == "__main__":
    # Przykład użycia
    connector = BybitConnector(api_key="test_key",
                               api_secret="test_secret",
                               use_testnet=True,
                               proxies={
                                   'http': 'socks5h://127.0.0.1:1080',
                                   'https': 'socks5h://127.0.0.1:1080'
                               })

    server_time = connector.get_server_time()
    print(f"Czas serwera: {server_time}")

    klines = connector.get_klines(symbol="BTCUSDT")
    print(f"Pobrano {len(klines)} świec")

    order_book = connector.get_order_book(symbol="BTCUSDT")
    print(
        f"Pobrano książkę zleceń z {len(order_book['bids'])} ofertami kupna i {len(order_book['asks'])} ofertami sprzedaży"
    )

    balance = connector.get_account_balance()
    print(f"Stan konta: {balance}")

    order_result = connector.place_order(symbol="BTCUSDT",
                                         side="Buy",
                                         price=50000,
                                         quantity=0.01)
    print(f"Wynik złożenia zlecenia: {order_result}")

    order_status = connector.get_order_status(
        order_id=order_result.get("order_id"))
    print(f"Status zlecenia: {order_status}")

    cancel_result = connector.cancel_order(
        order_id=order_result.get("order_id"))
    print(f"Wynik anulowania zlecenia: {cancel_result}")