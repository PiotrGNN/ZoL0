"""
real_time_optimizer.py
----------------------
Moduł optymalizujący strategie handlowe w czasie rzeczywistym.
Funkcjonalności:
- Wykorzystanie streamingu danych rynkowych i bieżących wyników strategii do dynamicznego dopasowywania parametrów,
  takich jak progi wejścia/wyjścia.
- Uwzględnienie ograniczeń ryzyka, np. limitu straty dziennej, margin call, oraz aktualnych warunków rynkowych (zmienność, wolumen).
- Zapewnienie szybkiego działania oraz odporności na awarie (fallback do ustawień domyślnych w przypadku błędów).
- Implementacja funkcji logowania i alertów, monitorujących efekty wprowadzanych zmian.
- Skalowalność umożliwiająca obsługę wielu par walutowych jednocześnie.
"""

import logging
import threading
import time

import numpy as np

# Przykładowe ustawienia domyślne
DEFAULT_PARAMS = {
    "entry_threshold": 0.01,
    "exit_threshold": 0.005,
    "daily_loss_limit": 0.1,  # 10% dziennej straty
}

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RealTimeOptimizer:
    def __init__(self, initial_params: dict = None):
        """
        Inicjalizuje optymalizator strategii handlowych w czasie rzeczywistym.

        Parameters:
            initial_params (dict): Słownik początkowych parametrów strategii. Jeśli None, używa DEFAULT_PARAMS.
        """
        self.params = initial_params or DEFAULT_PARAMS.copy()
        self.lock = threading.Lock()
        self.last_update_time = time.time()
        logging.info("RealTimeOptimizer zainicjalizowany z parametrami: %s", self.params)

    def update_parameters(self, market_data: dict, strategy_performance: dict):
        """
        Aktualizuje parametry strategii na podstawie bieżących danych rynkowych oraz wyników strategii.

        Parameters:
            market_data (dict): Aktualne dane rynkowe, np. {"volatility": ..., "volume": ...}
            strategy_performance (dict): Bieżące wyniki strategii, np. {"drawdown": ..., "profit": ...}
        """
        with self.lock:
            # Przykładowa logika: jeśli zmienność jest wysoka, zwiększ próg wejścia, aby unikać fałszywych sygnałów.
            volatility = market_data.get("volatility", 0)
            if volatility > 0.02:
                self.params["entry_threshold"] = min(self.params["entry_threshold"] * 1.1, 0.05)
            else:
                self.params["entry_threshold"] = max(self.params["entry_threshold"] * 0.95, 0.005)

            # Jeśli strategia generuje duże straty (przekracza dzienny limit), zwiększ próg wyjścia.
            current_drawdown = strategy_performance.get("drawdown", 0)
            if current_drawdown > self.params["daily_loss_limit"]:
                self.params["exit_threshold"] = min(self.params["exit_threshold"] * 1.2, 0.05)
            else:
                self.params["exit_threshold"] = max(self.params["exit_threshold"] * 0.9, 0.001)

            self.last_update_time = time.time()
            logging.info("Parametry zaktualizowane: %s", self.params)

    def get_current_parameters(self) -> dict:
        """
        Zwraca aktualnie używane parametry strategii.

        Returns:
            dict: Aktualne parametry.
        """
        with self.lock:
            return self.params.copy()

    def fallback_to_defaults(self):
        """
        Resetuje parametry do ustawień domyślnych w przypadku krytycznych błędów.
        """
        with self.lock:
            self.params = DEFAULT_PARAMS.copy()
            self.last_update_time = time.time()
            logging.warning("Parametry zostały zresetowane do ustawień domyślnych: %s", self.params)

    def monitor_and_optimize(self, market_data_stream, strategy_performance_stream, check_interval: int = 60):
        """
        Monitoruje dane rynkowe oraz wyniki strategii i okresowo aktualizuje parametry.

        Parameters:
            market_data_stream: Funkcja lub generator zwracający bieżące dane rynkowe.
            strategy_performance_stream: Funkcja lub generator zwracający bieżące wyniki strategii.
            check_interval (int): Czas (w sekundach) pomiędzy kolejnymi aktualizacjami.
        """
        while True:
            try:
                market_data = next(market_data_stream)
                performance = next(strategy_performance_stream)
                self.update_parameters(market_data, performance)
            except Exception as e:
                logging.error("Błąd podczas monitorowania i optymalizacji: %s", e)
                self.fallback_to_defaults()
            time.sleep(check_interval)


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe funkcje symulujące strumienie danych
        def market_data_generator():
            while True:
                # Symulacja: losowa zmienność i wolumen
                yield {
                    "volatility": np.random.uniform(0.005, 0.03),
                    "volume": np.random.uniform(1000, 5000),
                }

        def performance_generator():
            while True:
                # Symulacja: losowy drawdown i profit
                yield {
                    "drawdown": np.random.uniform(0, 0.15),
                    "profit": np.random.uniform(-100, 100),
                }

        optimizer = RealTimeOptimizer()
        # Uruchomienie monitoringu w osobnym wątku
        monitor_thread = threading.Thread(
            target=optimizer.monitor_and_optimize,
            args=(market_data_generator(), performance_generator(), 10),
        )
        monitor_thread.daemon = True
        monitor_thread.start()

        # Symulacja działania głównego systemu przez 60 sekund
        for i in range(6):
            current_params = optimizer.get_current_parameters()
            logging.info("Aktualne parametry strategii: %s", current_params)
            time.sleep(10)

        logging.info("Symulacja optymalizacji zakończona.")
    except Exception as e:
        logging.error("Błąd w module real_time_optimizer.py: %s", e)
        raise
