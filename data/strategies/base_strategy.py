"""
base_strategy.py
---------------
Moduł definiujący podstawową klasę dla wszystkich strategii tradingowych.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

class BaseStrategy:
    """Bazowa klasa dla wszystkich strategii tradingowych."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Inicjalizacja bazowej strategii.
        
        Args:
            name: Nazwa strategii
            params: Parametry konfiguracyjne strategii
        """
        self.name = name
        self.params = params or {}
        self.logger = logging.getLogger(f"strategy.{name}")
        
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generuje sygnał handlowy na podstawie danych rynkowych.
        
        Args:
            market_data: DataFrame z danymi rynkowymi
            
        Returns:
            Dict zawierający sygnał i dodatkowe informacje
        """
        raise NotImplementedError("Metoda generate_signal musi być zaimplementowana w klasie pochodnej")
        
    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """
        Aktualizuje parametry strategii.
        
        Args:
            new_params: Nowe parametry do zaktualizowania
        """
        self.params.update(new_params)
        self.logger.info(f"Zaktualizowano parametry strategii {self.name}: {new_params}")
        
    def validate_data(self, market_data: pd.DataFrame) -> bool:
        """
        Sprawdza poprawność danych wejściowych.
        
        Args:
            market_data: DataFrame z danymi rynkowymi
            
        Returns:
            bool: Czy dane są poprawne
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in market_data.columns for col in required_columns)