"""
strategy_manager.py
-------------------
Moduł zarządzający zestawem strategii handlowych.
Funkcjonalności:
- Umożliwia aktywację i dezaktywację strategii w czasie rzeczywistym.
- Monitoruje wyniki poszczególnych strategii i pozwala na raportowanie ich skuteczności.
- Implementuje funkcje zarządzania priorytetami strategii oraz limitowania ekspozycji na każdą strategię.
- Integruje się z modułami trade_executor.py i risk_management (np. stop_loss_manager.py), aby koordynować decyzje tradingowe.
- Zawiera mechanizmy testowe, umożliwiające przeprowadzenie scenariuszy stress-test oraz logowanie decyzji.
"""

import logging
import threading
import time

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class StrategyManager:
    def __init__(self, strategies: dict, exposure_limits: dict):
        """
        Inicjalizuje menedżera strategii.

        Parameters:
            strategies (dict): Słownik strategii, gdzie kluczem jest nazwa strategii, a wartością obiekt strategii
                               implementujący metodę `generate_signal(data)` i `evaluate_performance(data)`.
            exposure_limits (dict): Limity ekspozycji dla poszczególnych strategii, np. {"trend_following": 0.5, "mean_reversion": 0.3}.
        """
        self.strategies = strategies
        self.exposure_limits = exposure_limits
        self.active_strategies = {name: True for name in strategies.keys()}
        self.strategy_results = {}
        self.lock = threading.Lock()
        logging.info(
            "StrategyManager zainicjalizowany z strategiami: %s",
            list(strategies.keys()),
        )

    def activate_strategy(self, strategy_name: str):
        with self.lock:
            if strategy_name in self.strategies:
                self.active_strategies[strategy_name] = True
                logging.info("Strategia %s aktywowana.", strategy_name)
            else:
                logging.warning("Strategia %s nie istnieje.", strategy_name)

    def deactivate_strategy(self, strategy_name: str):
        with self.lock:
            if strategy_name in self.strategies:
                self.active_strategies[strategy_name] = False
                logging.info("Strategia %s dezaktywowana.", strategy_name)
            else:
                logging.warning("Strategia %s nie istnieje.", strategy_name)

    def evaluate_strategies(self, market_data: pd.DataFrame):
        """
        Ocena wyników wszystkich aktywnych strategii na podstawie dostarczonych danych rynkowych.

        Parameters:
            market_data (pd.DataFrame): Dane rynkowe wykorzystywane do ewaluacji.
        """
        results = {}
        for name, strategy in self.strategies.items():
            if self.active_strategies.get(name, False):
                try:
                    signal = strategy.generate_signal(market_data)
                    performance = strategy.evaluate_performance(market_data)
                    results[name] = {"signal": signal, "performance": performance}
                    logging.info(
                        "Strategia %s - sygnał: %s, performance: %s",
                        name,
                        signal,
                        performance,
                    )
                except Exception as e:
                    logging.error("Błąd przy ocenie strategii %s: %s", name, e)
            else:
                logging.info("Strategia %s jest dezaktywowana.", name)
        with self.lock:
            self.strategy_results = results
        return results

    def limit_exposure(self, strategy_name: str, proposed_exposure: float) -> float:
        """
        Ogranicza proponowaną ekspozycję dla danej strategii zgodnie z ustalonym limitem.

        Parameters:
            strategy_name (str): Nazwa strategii.
            proposed_exposure (float): Proponowany procent ekspozycji (np. 0.6 oznacza 60% kapitału).

        Returns:
            float: Ograniczona ekspozycja.
        """
        limit = self.exposure_limits.get(strategy_name, 1.0)
        adjusted_exposure = min(proposed_exposure, limit)
        logging.info(
            "Ekspozycja dla strategii %s ograniczona do: %.2f (proponowana: %.2f)",
            strategy_name,
            adjusted_exposure,
            proposed_exposure,
        )
        return adjusted_exposure

    def generate_report(self):
        """
        Generuje raport podsumowujący wyniki strategii.

        Returns:
            dict: Raport zawierający wyniki oceny każdej strategii.
        """
        with self.lock:
            report = self.strategy_results.copy()
        logging.info("Wygenerowano raport strategii: %s", report)
        return report

    def stress_test(self, market_data: pd.DataFrame, test_duration: int = 60):
        """
        Przeprowadza stress-test wszystkich aktywnych strategii przez określony czas.

        Parameters:
            market_data (pd.DataFrame): Dane rynkowe używane w teście.
            test_duration (int): Czas trwania testu w sekundach.
        """
        logging.info(
            "Rozpoczynam stress-test strategii na okres %d sekund.", test_duration
        )
        start_time = time.time()
        while time.time() - start_time < test_duration:
            self.evaluate_strategies(market_data)
            time.sleep(5)
        logging.info("Stress-test zakończony.")


# -------------------- Przykładowe klasy strategii --------------------
class DummyStrategy:
    def __init__(self, name):
        self.name = name

    def generate_signal(self, market_data: pd.DataFrame):
        # Przykładowa logika: zwraca losowy sygnał -1, 0, lub 1
        return np.random.choice([-1, 0, 1])

    def evaluate_performance(self, market_data: pd.DataFrame):
        # Przykładowa logika: zwraca losową wartość performance
        return np.random.uniform(-0.05, 0.05)


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Tworzymy przykładowe strategie
        strategies = {
            "trend_following": DummyStrategy("trend_following"),
            "mean_reversion": DummyStrategy("mean_reversion"),
            "momentum": DummyStrategy("momentum"),
        }
        exposure_limits = {
            "trend_following": 0.5,
            "mean_reversion": 0.3,
            "momentum": 0.4,
        }
        manager = StrategyManager(strategies, exposure_limits)

        # Przykładowe dane rynkowe: symulacja prostego DataFrame
        dates = pd.date_range(start="2023-01-01", periods=100, freq="T")
        market_data = pd.DataFrame(
            {
                "close": np.linspace(100, 105, 100) + np.random.normal(0, 0.5, 100),
                "volume": np.random.randint(1000, 1500, 100),
            },
            index=dates,
        )

        # Ocena strategii
        eval_results = manager.evaluate_strategies(market_data)
        report = manager.generate_report()
        logging.info("Raport strategii: %s", report)

        # Przykładowe ograniczenie ekspozycji
        proposed_exposure = 0.6
        adjusted_exposure = manager.limit_exposure("trend_following", proposed_exposure)
        logging.info(
            "Dla strategii trend_following, proponowana ekspozycja %.2f, ograniczona do %.2f",
            proposed_exposure,
            adjusted_exposure,
        )

        # Opcjonalnie, uruchomienie stress-testu
        # manager.stress_test(market_data, test_duration=30)

    except Exception as e:
        logging.error("Błąd w module strategy_manager.py: %s", e)
        raise
