"""
drawdown_control.py
-----------------
Moduł do kontroli i zarządzania drawdown w portfelu.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class DrawdownController:
    """Klasa do monitorowania i kontrolowania drawdown portfela."""
    
    def __init__(self, max_drawdown: float = 0.1, recovery_threshold: float = 0.05):
        """
        Inicjalizacja kontrolera drawdown.

        Parameters:
            max_drawdown (float): Maksymalny dopuszczalny drawdown (domyślnie 10%)
            recovery_threshold (float): Próg odzyskania kapitału przed wznowieniem tradingu (domyślnie 5%)
        """
        self.max_drawdown = max_drawdown
        self.recovery_threshold = recovery_threshold
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.trading_suspended = False
        self.drawdown_history = []
        
        logging.info(f"Zainicjalizowano DrawdownController (max_drawdown: {max_drawdown:.1%}, recovery_threshold: {recovery_threshold:.1%})")

    def update(self, current_value: float) -> Dict[str, Any]:
        """
        Aktualizuje stan drawdown na podstawie aktualnej wartości portfela.

        Parameters:
            current_value (float): Aktualna wartość portfela

        Returns:
            Dict[str, Any]: Stan kontrolera zawierający informacje o drawdown i zalecenia
        """
        if current_value <= 0:
            raise ValueError("Wartość portfela musi być dodatnia")

        # Aktualizacja peak value
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Obliczenie aktualnego drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

        self.drawdown_history.append({
            'value': current_value,
            'peak': self.peak_value,
            'drawdown': self.current_drawdown
        })

        # Sprawdzenie czy przekroczono maksymalny drawdown
        if self.current_drawdown >= self.max_drawdown and not self.trading_suspended:
            self.trading_suspended = True
            logging.warning(f"Przekroczono maksymalny drawdown ({self.current_drawdown:.1%})")
        
        # Sprawdzenie czy można wznowić trading
        elif self.trading_suspended and self.current_drawdown <= self.recovery_threshold:
            self.trading_suspended = False
            logging.info("Drawdown spadł poniżej progu recovery, wznawianie tradingu")

        return {
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
            'trading_suspended': self.trading_suspended,
            'max_drawdown_exceeded': self.current_drawdown >= self.max_drawdown
        }

    def get_drawdown_stats(self) -> Dict[str, float]:
        """
        Zwraca statystyki drawdown.

        Returns:
            Dict[str, float]: Statystyki zawierające maksymalny, średni i aktualny drawdown
        """
        if not self.drawdown_history:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'current_drawdown': 0.0
            }

        drawdowns = [entry['drawdown'] for entry in self.drawdown_history]
        return {
            'max_drawdown': max(drawdowns),
            'avg_drawdown': np.mean(drawdowns),
            'current_drawdown': self.current_drawdown
        }

    def can_trade(self) -> bool:
        """
        Sprawdza czy można handlować (drawdown nie przekracza limitu).

        Returns:
            bool: True jeśli można handlować, False w przeciwnym razie
        """
        return not self.trading_suspended

    def reset(self) -> None:
        """Resetuje stan kontrolera drawdown."""
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.trading_suspended = False
        self.drawdown_history = []
        logging.info("Zresetowano DrawdownController")

    def get_recovery_info(self) -> Dict[str, float]:
        """
        Oblicza informacje o potrzebnym odzyskaniu kapitału.

        Returns:
            Dict[str, float]: Informacje o wymaganym zwrocie do odzyskania peak value
        """
        if self.current_drawdown <= 0:
            return {
                'required_return': 0.0,
                'recovery_target': self.peak_value
            }

        recovery_target = self.peak_value
        current_value = self.peak_value * (1 - self.current_drawdown)
        required_return = (recovery_target - current_value) / current_value

        return {
            'required_return': required_return,
            'recovery_target': recovery_target
        }