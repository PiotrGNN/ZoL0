"""
Katalog z uproszczonymi modułami dla systemu tradingowego, 
kompatybilnymi zarówno z lokalnym środowiskiem, jak i Replit.
"""

import sys
import os
import logging
import random
from datetime import datetime

# Dodanie ścieżki do bibliotek Pythona
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print("Dodano katalog python_libs do ścieżki Pythona.")
"""
Pakiet python_libs zawiera lokalne biblioteki i moduły pomocnicze.
"""
import sys
import os

# Dodaj ścieżkę bieżącego katalogu do sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

"""Python libraries for trading system."""

from .db_manager import DatabaseManager
from .risk_manager import RiskManager
from .trade_executor import TradeExecutor
from .strategy_manager import StrategyManager
from .market_data_manager import MarketDataManager
from .portfolio_manager import PortfolioManager
from .monitoring import MonitoringSystem
from .notification_system import NotificationSystem
from .system_state import SystemState
from .performance_tracker import PerformanceTracker

__all__ = [
    'DatabaseManager',
    'RiskManager',
    'TradeExecutor',
    'StrategyManager',
    'MarketDataManager',
    'PortfolioManager',
    'MonitoringSystem',
    'NotificationSystem',
    'SystemState',
    'PerformanceTracker'
]
