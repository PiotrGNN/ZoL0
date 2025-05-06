"""
simplified_strategy.py
-------------------
Module for managing trading strategies.
"""

import logging
from typing import Dict, Any, List

class StrategyManager:
    def __init__(self, strategies: Dict[str, Dict[str, Any]], exposure_limits: Dict[str, float]):
        """Initialize strategy manager."""
        self.strategies = strategies
        self.exposure_limits = exposure_limits
        self.active_strategies = []
        logging.info(f"Initialized StrategyManager with {len(strategies)} strategies")
        
    def activate_strategy(self, strategy_name: str) -> bool:
        """Activate a specific strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name]["enabled"] = True
            if strategy_name not in self.active_strategies:
                self.active_strategies.append(strategy_name)
            logging.info(f"Strategia {strategy_name} została włączona")
            return True
        logging.warning(f"Próba aktywacji nieistniejącej strategii: {strategy_name}")
        return False
        
    def deactivate_strategy(self, strategy_name: str) -> bool:
        """Deactivate a specific strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name]["enabled"] = False
            if strategy_name in self.active_strategies:
                self.active_strategies.remove(strategy_name)
            logging.info(f"Strategia {strategy_name} została wyłączona")
            return True
        logging.warning(f"Próba deaktywacji nieistniejącej strategii: {strategy_name}")
        return False
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategies."""
        return [name for name, data in self.strategies.items() if data["enabled"]]
    
    def get_strategy_exposure_limit(self, strategy_name: str) -> float:
        """Get exposure limit for a strategy."""
        return self.exposure_limits.get(strategy_name, 0.0)
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed information about a strategy."""
        if strategy_name in self.strategies:
            info = self.strategies[strategy_name].copy()
            info["exposure_limit"] = self.exposure_limits.get(strategy_name, 0.0)
            info["is_active"] = strategy_name in self.active_strategies
            return info
        return {}
    
    def get_all_strategies_info(self) -> List[Dict[str, Any]]:
        """Get information about all strategies."""
        return [
            {
                "name": name,
                **data,
                "exposure_limit": self.exposure_limits.get(name, 0.0),
                "is_active": name in self.active_strategies
            }
            for name, data in self.strategies.items()
        ]
