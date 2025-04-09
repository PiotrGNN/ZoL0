
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_live_trading.py
------------------
Skrypt do przeprowadzenia tradingu na rzeczywistych danych przez określony czas.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

# Dodanie ścieżek do modułów
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("python_libs")

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_trading")

def setup_trading_engine():
    """Inicjalizacja silnika tradingowego i jego komponentów."""
    try:
        # Import wymaganych modułów
        from python_libs.simplified_risk_manager import SimplifiedRiskManager
        from python_libs.simplified_strategy import StrategyManager
        from python_libs.simplified_trading_engine import SimplifiedTradingEngine
        from data.execution.bybit_connector import BybitConnector
        
        # Inicjalizacja risk managera
        logger.info("Inicjalizacja zarządzania ryzykiem...")
        risk_manager = SimplifiedRiskManager(
            max_risk=0.02,  # Max 2% ryzyka na transakcję
            max_position_size=0.1,  # Max 10% kapitału na pozycję
            max_drawdown=0.05  # Max 5% drawdown
        )
        
        # Inicjalizacja strategii
        logger.info("Inicjalizacja strategii tradingowych...")
        strategies = {
            "trend_following": {"name": "Trend Following", "enabled": True},
            "mean_reversion": {"name": "Mean Reversion", "enabled": False},
            "breakout": {"name": "Breakout", "enabled": True}
        }
        
        exposure_limits = {
            "trend_following": 0.5,
            "mean_reversion": 0.3,
            "breakout": 0.4
        }
        
        strategy_manager = StrategyManager(strategies, exposure_limits)
        
        # Aktywacja wybranych strategii
        strategy_manager.activate_strategy("trend_following")
        strategy_manager.activate_strategy("breakout")
        
        # Inicjalizacja konektora do giełdy (jeśli dostępny)
        logger.info("Inicjalizacja połączenia z giełdą...")
        try:
            bybit_connector = BybitConnector()
            logger.info("Połączenie z Bybit nawiązane pomyślnie")
        except Exception as e:
            logger.warning(f"Nie udało się połączyć z giełdą Bybit: {str(e)}")
            bybit_connector = None
            logger.info("Trading będzie prowadzony w trybie symulacji")
        
        # Inicjalizacja silnika handlowego
        logger.info("Inicjalizacja silnika tradingowego...")
        trading_engine = SimplifiedTradingEngine(
            risk_manager=risk_manager,
            strategy_manager=strategy_manager,
            exchange_connector=bybit_connector
        )
        
        logger.info("Silnik tradingowy zainicjalizowany pomyślnie")
        return trading_engine
    
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji silnika tradingowego: {str(e)}")
        return None

def run_trading_for_duration(duration_minutes=60, symbols=None):
    """
    Uruchamia trading na podanych symbolach przez określony czas.
    
    Args:
        duration_minutes: Czas trwania sesji tradingowej w minutach
        symbols: Lista symboli do tradingu (par walutowych)
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]  # Domyślne symbole
    
    # Inicjalizacja silnika
    trading_engine = setup_trading_engine()
    if not trading_engine:
        logger.error("Nie udało się zainicjalizować silnika tradingowego. Przerywam.")
        return
    
    # Rozpoczęcie tradingu
    try:
        logger.info(f"Rozpoczynam sesję tradingową na {symbols} na {duration_minutes} minut...")
        
        # Uruchomienie tradingu
        result = trading_engine.start_trading(symbols)
        if not result:
            logger.error("Nie udało się uruchomić tradingu")
            return
        
        logger.info(f"Trading uruchomiony. Czas trwania: {duration_minutes} minut.")
        
        # Obliczenie czasu zakończenia
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Główna pętla tradingowa
        while datetime.now() < end_time:
            # Pobranie aktualnego statusu
            status = trading_engine.get_status()
            logger.info(f"Status silnika: {status}")
            
            # Pobranie aktualnych pozycji
            positions = trading_engine.get_positions() if hasattr(trading_engine, 'get_positions') else {}
            logger.info(f"Aktualne pozycje: {positions}")
            
            # Pobranie historii transakcji
            trades = trading_engine.get_trade_history() if hasattr(trading_engine, 'get_trade_history') else []
            if trades:
                logger.info(f"Ostatnie transakcje: {trades[-5:] if len(trades) > 5 else trades}")
            
            # Obliczenie pozostałego czasu
            remaining = (end_time - datetime.now()).total_seconds() / 60
            logger.info(f"Pozostały czas: {remaining:.2f} minut")
            
            # Pauza między aktualizacjami
            time.sleep(60)  # Aktualizacja co minutę
    
    except KeyboardInterrupt:
        logger.info("Trading przerwany przez użytkownika")
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas sesji tradingowej: {str(e)}")
    finally:
        # Zatrzymanie tradingu
        logger.info("Zatrzymuję trading...")
        if trading_engine and hasattr(trading_engine, 'stop_trading'):
            trading_engine.stop_trading()
        
        # Generowanie raportu
        logger.info("Generuję raport z sesji tradingowej...")
        generate_trading_report(trading_engine)
        
        logger.info("Sesja tradingowa zakończona")

def generate_trading_report(trading_engine):
    """Generuje raport z sesji tradingowej"""
    try:
        if not trading_engine:
            return
        
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": "1h",
            "positions": trading_engine.get_positions() if hasattr(trading_engine, 'get_positions') else {},
            "trades": trading_engine.get_trade_history() if hasattr(trading_engine, 'get_trade_history') else [],
            "performance": trading_engine.get_performance() if hasattr(trading_engine, 'get_performance') else {}
        }
        
        # Zapis raportu do pliku
        import json
        os.makedirs("reports", exist_ok=True)
        report_file = f"reports/trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Raport zapisany do pliku: {report_file}")
    
    except Exception as e:
        logger.error(f"Błąd podczas generowania raportu: {str(e)}")

if __name__ == "__main__":
    # Utworzenie katalogu na logi
    os.makedirs("logs", exist_ok=True)
    
    # Parametry sesji tradingowej
    duration = 60  # 60 minut (1 godzina)
    trading_symbols = ["BTCUSDT", "ETHUSDT"]  # Pary walutowe do tradingu
    
    logger.info(f"Uruchamiam godzinny test tradingowy na symbolach: {trading_symbols}")
    run_trading_for_duration(duration, trading_symbols)
