
"""
simplified_strategy.py
----------------------
Uproszczony moduł strategii handlowej, kompatybilny zarówno z lokalnym środowiskiem, jak i Replit.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Uproszczony manager strategii handlowych.
    """
    
    def __init__(self, strategies: Dict[str, Dict[str, Any]], exposure_limits: Dict[str, float] = None):
        """
        Inicjalizacja managera strategii.
        
        Args:
            strategies: Słownik strategii w formacie {id_strategii: {name: nazwa, enabled: True/False}}
            exposure_limits: Maksymalna ekspozycja dla każdej strategii {id_strategii: wartość}
        """
        self.strategies = strategies
        self.exposure_limits = exposure_limits or {}
        self.active_strategies = []
        logger.info(f"Zainicjalizowano StrategyManager z {len(strategies)} strategiami")
        
    def activate_strategy(self, strategy_id: str) -> bool:
        """
        Aktywuje strategię o podanym ID.
        
        Args:
            strategy_id: ID strategii do aktywacji
            
        Returns:
            bool: True jeśli aktywacja się powiodła, False w przeciwnym razie
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Nie można aktywować strategii {strategy_id} - nie istnieje")
            return False
            
        if not self.strategies[strategy_id].get("enabled", False):
            logger.warning(f"Strategia {strategy_id} jest wyłączona i nie może być aktywowana")
            return False
            
        if strategy_id not in self.active_strategies:
            self.active_strategies.append(strategy_id)
            logger.info(f"Aktywowano strategię: {strategy_id}")
        return True
        
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Deaktywuje strategię o podanym ID.
        
        Args:
            strategy_id: ID strategii do deaktywacji
            
        Returns:
            bool: True jeśli deaktywacja się powiodła, False w przeciwnym razie
        """
        if strategy_id in self.active_strategies:
            self.active_strategies.remove(strategy_id)
            logger.info(f"Deaktywowano strategię: {strategy_id}")
            return True
        return False
        
    def get_active_strategies(self) -> List[str]:
        """
        Zwraca listę aktywnych strategii.
        
        Returns:
            List[str]: Lista ID aktywnych strategii
        """
        return self.active_strategies
        
    def get_strategy_exposure(self, strategy_id: str) -> float:
        """
        Zwraca maksymalną ekspozycję dla danej strategii.
        
        Args:
            strategy_id: ID strategii
            
        Returns:
            float: Maksymalna ekspozycja (0.0 - 1.0)
        """
        return self.exposure_limits.get(strategy_id, 0.0)
        
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizuje dane rynkowe za pomocą aktywnych strategii.
        
        Args:
            market_data: Dane rynkowe do analizy
            
        Returns:
            Dict: Zagregowane sygnały z wszystkich aktywnych strategii
        """
        results = {}
        
        for strategy_id in self.active_strategies:
            try:
                # W rzeczywistej implementacji tutaj byłoby wywołanie konkretnej strategii
                # Dla uproszczenia zwracamy podstawowe sygnały
                if strategy_id == "trend_following":
                    results[strategy_id] = {"signal": 1, "strength": 0.7, "timestamp": datetime.now().isoformat()}
                elif strategy_id == "mean_reversion":
                    results[strategy_id] = {"signal": -1, "strength": 0.5, "timestamp": datetime.now().isoformat()}
                elif strategy_id == "breakout":
                    results[strategy_id] = {"signal": 0, "strength": 0.3, "timestamp": datetime.now().isoformat()}
                else:
                    results[strategy_id] = {"signal": 0, "strength": 0.0, "timestamp": datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"Błąd podczas analizy strategii {strategy_id}: {e}")
                results[strategy_id] = {"signal": 0, "strength": 0.0, "error": str(e), "timestamp": datetime.now().isoformat()}
        
        return results

# Przykład użycia
if __name__ == "__main__":
    # Przykładowa definicja strategii
    strategies = {
        "trend_following": {"name": "Trend Following", "enabled": True},
        "mean_reversion": {"name": "Mean Reversion", "enabled": False},
        "breakout": {"name": "Breakout", "enabled": True}
    }
    
    # Przykładowe limity ekspozycji
    exposure_limits = {
        "trend_following": 0.5,
        "mean_reversion": 0.3,
        "breakout": 0.4
    }
    
    # Inicjalizacja manager strategii
    strategy_manager = StrategyManager(strategies, exposure_limits)
    
    # Aktywacja strategii
    strategy_manager.activate_strategy("trend_following")
    strategy_manager.activate_strategy("breakout")
    
    # Wyświetlenie aktywnych strategii
    active_strategies = strategy_manager.get_active_strategies()
    print(f"Aktywne strategie: {active_strategies}")
    
    # Przykładowa analiza danych rynkowych
    market_data = {"symbol": "BTCUSDT", "price": 50000, "volume": 100, "timestamp": datetime.now().isoformat()}
    results = strategy_manager.analyze(market_data)
    
    # Wyświetlenie wyników
    for strategy_id, result in results.items():
        print(f"Strategia {strategy_id}: {result}")
