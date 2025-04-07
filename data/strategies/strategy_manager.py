"""
strategy_manager.py
------------------
Moduł zarządzający strategiami inwestycyjnymi.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class StrategyManager:
    """
    Klasa zarządzająca strategiami inwestycyjnymi.
    Umożliwia dynamiczne wybieranie strategii w zależności od warunków rynkowych.
    """

    def __init__(self, strategies: Dict[str, Any], exposure_limits: Dict[str, float]):
        """
        Inicjalizacja managera strategii.

        Args:
            strategies: Słownik zawierający strategie (nazwa: obiekt strategii)
            exposure_limits: Limity ekspozycji dla każdej strategii
        """
        self.strategies = strategies or {}  # Zabezpieczenie przed None
        self.exposure_limits = exposure_limits or {}  # Zabezpieczenie przed None
        self.active_strategy = None

        logger.info(f"Zainicjalizowano StrategyManager z {len(self.strategies)} strategiami")

    def add_strategy(self, name: str, strategy: Any, exposure_limit: float = 1.0):
        """
        Dodaje nową strategię do managera.

        Args:
            name: Nazwa strategii
            strategy: Obiekt strategii
            exposure_limit: Limit ekspozycji (0.0-1.0)
        """
        self.strategies[name] = strategy
        self.exposure_limits[name] = exposure_limit
        logger.info(f"Dodano strategię {name} z limitem ekspozycji {exposure_limit}")

    def activate_strategy(self, name: str) -> bool:
        """
        Aktywuje wybraną strategię.

        Args:
            name: Nazwa strategii do aktywowania

        Returns:
            bool: True jeśli aktywacja się powiodła, False w przeciwnym wypadku
        """
        if name in self.strategies:
            self.active_strategy = name
            logger.info(f"Aktywowano strategię {name}")
            return True
        else:
            logger.warning(f"Próba aktywacji nieistniejącej strategii: {name}")
            return False

    def get_active_strategy(self) -> Optional[Any]:
        """
        Zwraca aktualnie aktywną strategię.

        Returns:
            Obiekt aktywnej strategii lub None jeśli żadna nie jest aktywna
        """
        if self.active_strategy and self.active_strategy in self.strategies:
            return self.strategies[self.active_strategy]
        return None

    def get_exposure_limit(self) -> float:
        """
        Zwraca limit ekspozycji dla aktualnie aktywnej strategii.

        Returns:
            float: Limit ekspozycji lub 0.0 jeśli żadna strategia nie jest aktywna
        """
        if self.active_strategy and self.active_strategy in self.exposure_limits:
            return self.exposure_limits[self.active_strategy]
        return 0.0

    def list_strategies(self) -> List[str]:
        """
        Zwraca listę dostępnych strategii.

        Returns:
            List[str]: Lista nazw strategii
        """
        return list(self.strategies.keys())

    def update_exposure_limit(self, name: str, new_limit: float) -> bool:
        """
        Aktualizuje limit ekspozycji dla danej strategii.

        Args:
            name: Nazwa strategii
            new_limit: Nowy limit ekspozycji (0.0-1.0)

        Returns:
            bool: True jeśli aktualizacja się powiodła, False w przeciwnym wypadku
        """
        if name in self.strategies:
            self.exposure_limits[name] = max(0.0, min(1.0, new_limit))  # Ograniczenie do zakresu 0.0-1.0
            logger.info(f"Zaktualizowano limit ekspozycji dla strategii {name} na {self.exposure_limits[name]}")
            return True
        else:
            logger.warning(f"Próba aktualizacji limitu dla nieistniejącej strategii: {name}")
            return False

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
    import numpy as np
    import pandas as pd
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

        # Aktywacja strategii i generowanie sygnału
        manager.activate_strategy("trend_following")
        active_strategy = manager.get_active_strategy()
        if active_strategy:
            signal = active_strategy.generate_signal(market_data)
            logger.info(f"Sygnał z trend_following: {signal}")

        # Wyświetlenie listy strategii
        logger.info(f"Dostępne strategie: {manager.list_strategies()}")

        # Aktualizacja limitu ekspozycji
        manager.update_exposure_limit("mean_reversion", 0.6)


    except Exception as e:
        logger.error("Błąd w module strategy_manager.py: %s", e)
        raise