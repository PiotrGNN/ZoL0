"""
simplified_strategy.py
---------------------
Uproszczony menedżer strategii dla platformy tradingowej.
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class StrategyManager:
    """Menedżer strategii do zarządzania strategiami tradingowymi."""

    def __init__(self, strategies: Dict[str, Dict[str, Any]] = None, exposure_limits: Dict[str, float] = None):
        """
        Inicjalizuje menedżera strategii.

        Parameters:
            strategies (Dict[str, Dict[str, Any]]): Słownik strategii
            exposure_limits (Dict[str, float]): Limity ekspozycji dla strategii
        """
        self.strategies = strategies or {
            "trend_following": {"name": "Trend Following", "enabled": False},
            "mean_reversion": {"name": "Mean Reversion", "enabled": False},
            "breakout": {"name": "Breakout", "enabled": False}
        }

        self.exposure_limits = exposure_limits or {
            "trend_following": 0.5,
            "mean_reversion": 0.3,
            "breakout": 0.4
        }

        self.active_strategies = []
        self.strategy_performance = {}

        # Domyślne definicje strategii
        self.strategy_definitions = {
            "trend_following": self._trend_following_strategy,
            "mean_reversion": self._mean_reversion_strategy,
            "breakout": self._breakout_strategy
        }

        logger.info(f"Zainicjalizowano StrategyManager z {len(self.strategies)} strategiami")

    def activate_strategy(self, strategy_id: str) -> bool:
        """
        Aktywuje strategię.

        Parameters:
            strategy_id (str): ID strategii

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Nieznana strategia: {strategy_id}")
                return False

            self.strategies[strategy_id]["enabled"] = True

            if strategy_id not in self.active_strategies:
                self.active_strategies.append(strategy_id)

            logger.info(f"Aktywowano strategię: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktywacji strategii {strategy_id}: {e}")
            return False

    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Deaktywuje strategię.

        Parameters:
            strategy_id (str): ID strategii

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Nieznana strategia: {strategy_id}")
                return False

            self.strategies[strategy_id]["enabled"] = False

            if strategy_id in self.active_strategies:
                self.active_strategies.remove(strategy_id)

            logger.info(f"Deaktywowano strategię: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas deaktywacji strategii {strategy_id}: {e}")
            return False

    def evaluate_strategies(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ocenia wszystkie aktywne strategie na podstawie danych rynkowych.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wyniki oceny strategii
        """
        try:
            results = {}
            combined_signal = 0.0
            weights_sum = 0.0

            for strategy_id in self.active_strategies:
                if strategy_id in self.strategy_definitions:
                    # Wykonaj strategię
                    strategy_result = self.strategy_definitions[strategy_id](market_data)

                    # Zapisz wynik
                    results[strategy_id] = strategy_result

                    # Dodaj do sygnału łączonego
                    weight = self.exposure_limits.get(strategy_id, 0.0)
                    combined_signal += strategy_result["signal"] * weight
                    weights_sum += weight

            # Normalizuj sygnał łączony
            if weights_sum > 0:
                combined_signal /= weights_sum

            # Określ decyzję
            if combined_signal > 0.5:
                decision = "buy"
            elif combined_signal < -0.5:
                decision = "sell"
            else:
                decision = "hold"

            return {
                "strategy_results": results,
                "combined_signal": combined_signal,
                "decision": decision,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Błąd podczas oceny strategii: {e}")
            return {"error": str(e)}

    def _trend_following_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strategia podążania za trendem.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wynik strategii
        """
        # Symulacja dla celów demonstracyjnych
        trend_strength = random.uniform(-1.0, 1.0)
        signal = trend_strength

        return {
            "signal": signal,
            "indicators": {
                "trend_strength": trend_strength,
                "ma_cross": random.choice([True, False]),
                "adx": random.uniform(10, 40)
            },
            "confidence": abs(signal) * 0.8 + 0.1
        }

    def _mean_reversion_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strategia powrotu do średniej.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wynik strategii
        """
        # Symulacja dla celów demonstracyjnych
        overbought = random.random() > 0.7
        oversold = random.random() > 0.7

        if overbought:
            signal = -random.uniform(0.5, 1.0)
        elif oversold:
            signal = random.uniform(0.5, 1.0)
        else:
            signal = random.uniform(-0.3, 0.3)

        return {
            "signal": signal,
            "indicators": {
                "rsi": random.uniform(0, 100),
                "bollinger_bands": {
                    "upper": random.uniform(30000, 40000),
                    "middle": random.uniform(28000, 35000),
                    "lower": random.uniform(25000, 30000)
                }
            },
            "confidence": abs(signal) * 0.7 + 0.2
        }

    def _breakout_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strategia breakoutu.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wynik strategii
        """
        # Symulacja dla celów demonstracyjnych
        breakout_up = random.random() > 0.8
        breakout_down = random.random() > 0.8

        if breakout_up:
            signal = random.uniform(0.7, 1.0)
        elif breakout_down:
            signal = random.uniform(-1.0, -0.7)
        else:
            signal = random.uniform(-0.2, 0.2)

        return {
            "signal": signal,
            "indicators": {
                "support_resistance": {
                    "support": random.uniform(25000, 30000),
                    "resistance": random.uniform(35000, 40000)
                },
                "volume_increase": random.uniform(0, 200)
            },
            "confidence": abs(signal) * 0.9 + 0.1
        }

    def get_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Zwraca dostępne strategie.

        Returns:
            Dict[str, Dict[str, Any]]: Dostępne strategie
        """
        return self.strategies

    def get_active_strategies(self) -> List[str]:
        """
        Zwraca aktywne strategie.

        Returns:
            List[str]: Aktywne strategie
        """
        return self.active_strategies

    def add_strategy(self, strategy_id: str, strategy_name: str, strategy_function: Callable) -> bool:
        """
        Dodaje nową strategię.

        Parameters:
            strategy_id (str): ID strategii
            strategy_name (str): Nazwa strategii
            strategy_function (Callable): Funkcja implementująca strategię

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id in self.strategies:
                logger.warning(f"Strategia o ID {strategy_id} już istnieje")
                return False

            self.strategies[strategy_id] = {"name": strategy_name, "enabled": False}
            self.strategy_definitions[strategy_id] = strategy_function
            self.exposure_limits[strategy_id] = 0.3  # Domyślny limit ekspozycji

            logger.info(f"Dodano nową strategię: {strategy_id} ({strategy_name})")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas dodawania strategii {strategy_id}: {e}")
            return False

    def update_strategy_exposure(self, strategy_id: str, exposure: float) -> bool:
        """
        Aktualizuje limit ekspozycji dla strategii.

        Parameters:
            strategy_id (str): ID strategii
            exposure (float): Limit ekspozycji

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Nieznana strategia: {strategy_id}")
                return False

            self.exposure_limits[strategy_id] = exposure
            logger.info(f"Zaktualizowano limit ekspozycji dla strategii {strategy_id}: {exposure}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji limitu ekspozycji dla strategii {strategy_id}: {e}")
            return False

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja managera strategii
    strategy_manager = StrategyManager()

    # Aktywacja strategii
    strategy_manager.activate_strategy("trend_following")
    strategy_manager.activate_strategy("breakout")

    # Wyświetlenie aktywnych strategii
    active_strategies = strategy_manager.get_active_strategies()
    print(f"Aktywne strategie: {active_strategies}")

    # Przykładowa analiza danych rynkowych
    market_data = {"symbol": "BTCUSDT", "price": 50000, "volume": 100, "timestamp": time.time()}
    results = strategy_manager.evaluate_strategies(market_data)

    # Wyświetlenie wyników
    print(f"Wyniki analizy: {results}")