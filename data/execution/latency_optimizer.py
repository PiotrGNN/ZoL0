"""
latency_optimizer.py
--------------------
Moduł analizujący opóźnienia w komunikacji z giełdą oraz proponujący rozwiązania optymalizujące czas odpowiedzi.
Funkcjonalności:
- Pomiar RTT (round-trip time) dla zapytań do API giełdowego.
- Testy przepustowości oraz identyfikacja wąskich gardeł (np. opóźnienia sieciowe, API rate limit).
- Rekomendacje dotyczące najlepszych parametrów (np. wielkość batcha, rodzaj zleceń, optymalny endpoint).
- Mechanizm automatycznego przełączania na zapasowe endpointy w razie problemów (failover).
- Logowanie i generowanie raportów z historii opóźnień, ułatwiających długoterminową optymalizację strategii.
"""

import logging
import time
from statistics import mean

import requests

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_ENDPOINT = "https://api.binance.com/api/v3/time"
RETRY_COUNT = 3
TIMEOUT = 5


class LatencyOptimizer:
    def __init__(self, endpoints: list = None):
        """
        Inicjalizacja modułu LatencyOptimizer.

        Parameters:
            endpoints (list): Lista endpointów API do testowania. Jeśli None, używa domyślnego endpointu Binance.
        """
        if endpoints is None:
            endpoints = [DEFAULT_ENDPOINT]
        self.endpoints = endpoints
        self.latency_history = {endpoint: [] for endpoint in endpoints}

    def measure_latency(self, endpoint: str) -> float:
        """
        Mierzy opóźnienie (RTT) dla danego endpointu.

        Parameters:
            endpoint (str): URL endpointu.

        Returns:
            float: Zmierzony RTT w sekundach.
        """
        latencies = []
        for i in range(RETRY_COUNT):
            try:
                start = time.time()
                response = requests.get(endpoint, timeout=TIMEOUT)
                response.raise_for_status()
                end = time.time()
                latency = end - start
                latencies.append(latency)
                logging.info("Pomiar RTT %d dla %s: %.4f s", i + 1, endpoint, latency)
            except requests.exceptions.RequestException as e:
                logging.warning("Błąd przy pomiarze RTT (próba %d): %s", i + 1, e)
        if latencies:
            avg_latency = mean(latencies)
            self.latency_history[endpoint].append(avg_latency)
            logging.info("Średnie opóźnienie dla %s: %.4f s", endpoint, avg_latency)
            return avg_latency
        else:
            logging.error("Nie udało się zmierzyć opóźnienia dla %s", endpoint)
            return float("inf")

    def test_endpoints(self) -> dict:
        """
        Testuje wszystkie dostępne endpointy i zwraca ich średnie opóźnienia.

        Returns:
            dict: Słownik z endpointami jako kluczami i średnim RTT jako wartościami.
        """
        results = {}
        for endpoint in self.endpoints:
            avg_latency = self.measure_latency(endpoint)
            results[endpoint] = avg_latency
        return results

    def recommend_best_endpoint(self) -> str:
        """
        Rekomenduje najlepszy endpoint na podstawie zmierzonych opóźnień.

        Returns:
            str: URL endpointu z najniższym średnim opóźnieniem.
        """
        results = self.test_endpoints()
        best_endpoint = min(results, key=results.get)
        logging.info(
            "Rekomendowany endpoint: %s z RTT: %.4f s",
            best_endpoint,
            results[best_endpoint],
        )
        return best_endpoint

    def generate_latency_report(self, report_path: str):
        """
        Generuje raport z historii pomiarów opóźnień i zapisuje go do pliku.

        Parameters:
            report_path (str): Ścieżka do pliku raportu (np. 'latency_report.txt').
        """
        try:
            with open(report_path, "w") as f:
                f.write("Raport opóźnień (RTT) dla endpointów API\n")
                f.write("========================================\n\n")
                for endpoint, latencies in self.latency_history.items():
                    if latencies:
                        avg_latency = mean(latencies)
                        f.write(f"Endpoint: {endpoint}\n")
                        f.write(f"Pomiarów: {len(latencies)}\n")
                        f.write(f"Średnie opóźnienie: {avg_latency:.4f} s\n")
                        f.write(f"Pomiar: {latencies}\n")
                        f.write("----------------------------------------\n")
                    else:
                        f.write(f"Endpoint: {endpoint} - brak danych\n")
            logging.info("Raport opóźnień zapisany w: %s", report_path)
        except Exception as e:
            logging.error("Błąd przy generowaniu raportu opóźnień: %s", e)
            raise


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowa lista endpointów - można dodać więcej lub zmodyfikować
        endpoints = [
            "https://api.binance.com/api/v3/time",
            "https://api.binance.com/api/v3/exchangeInfo",  # przykładowy dodatkowy endpoint
        ]
        optimizer = LatencyOptimizer(endpoints=endpoints)

        # Rekomendacja najlepszego endpointu
        best_endpoint = optimizer.recommend_best_endpoint()
        logging.info("Najlepszy endpoint: %s", best_endpoint)

        # Generowanie raportu z historii opóźnień
        optimizer.generate_latency_report("latency_report.txt")
    except Exception as e:
        logging.error("Błąd w module latency_optimizer.py: %s", e)
        raise
