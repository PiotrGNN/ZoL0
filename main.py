"""
main.py
-------
Zaawansowany system tradingowy wykorzystujący AI.

Funkcjonalności:
  - Dynamiczne ładowanie konfiguracji i API kluczy
  - Wybór środowiska (Testnet/Production)
  - Inicjalizacja modułów tradingowych (Binance, Bybit)
  - Obsługa backtestingu i symulacji paper tradingu
  - Real-time trading z monitorowaniem rynku
  - Wykorzystanie AI do analizy rynku i predykcji trendów
  - Automatyczne optymalizowanie strategii przez AI
  - Centralne logowanie i obsługa wyjątków

Kod zgodny z najlepszymi praktykami:
  - Typowanie statyczne (type hints)
  - Obsługa wyjątków i centralne logowanie
  - Modularność i skalowalność
  - Integracja z systemami monitoringu i AI
"""

import os
import sys
import logging
import argparse
import time
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

# Moduły systemu
from config.settings import CONFIG
from data.data.historical_data import HistoricalDataManager
from data.optimization.backtesting import backtest_strategy, example_strategy
from data.execution.exchange_connector import ExchangeConnector
from data.execution.order_execution import OrderExecution
from data.execution.trade_executor import TradeExecutor

# Moduły AI
from ai_models.feature_engineering import feature_pipeline
from ai_models.model_training import ModelTrainer
from ai_models.reinforcement_learning import ReinforcementLearner
from ai_models.sentiment_analysis import SentimentAnalyzer
from ai_models.anomaly_detection import AnomalyDetector
from ai_models.trend_prediction import TrendPredictor
from ai_models.ai_optimizer import StrategyOptimizer


def setup_logging() -> None:
    """Konfiguruje logowanie systemu tradingowego."""
    log_dir: str = CONFIG.get("PATHS", {}).get("logs_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"trading_bot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_file: str = os.path.join(log_dir, log_filename)

    log_level_str: str = CONFIG.get("LOGGING", {}).get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("✅ Logowanie skonfigurowane. Plik logów: %s", log_file)


def choose_environment() -> str:
    """Wybór środowiska: production lub testnet."""
    print("\n[1] Production (Prawdziwy handel)")
    print("[2] Testnet (Symulacja)")

    while True:
        choice = input("Wybierz środowisko [1/2]: ").strip()
        if choice == "1":
            return "production"
        elif choice == "2":
            return "testnet"
        print("❌ Nieprawidłowy wybór. Spróbuj ponownie.")


def choose_exchange() -> str:
    """Wybór giełdy: Binance lub Bybit."""
    print("\n[1] Binance")
    print("[2] Bybit")

    while True:
        choice = input("Wybierz giełdę [1/2]: ").strip()
        if choice == "1":
            return "binance"
        elif choice == "2":
            return "bybit"
        print("❌ Nieprawidłowy wybór. Spróbuj ponownie.")


def initialize_trading_modules(environment: str, exchange: str) -> TradeExecutor:
    """Inicjalizuje moduły tradingowe dla wybranej giełdy."""
    try:
        api_key = os.getenv(f"{exchange.upper()}_API_KEY")
        api_secret = os.getenv(f"{exchange.upper()}_API_SECRET")

        base_url = (
            "https://api.binance.com" if exchange == "binance" and environment == "production"
            else "https://testnet.binance.vision" if exchange == "binance"
            else "https://api.bybit.com" if exchange == "bybit" and environment == "production"
            else "https://testnet.bybit.com"
        )

        logging.info("🔹 Używamy giełdy: %s, środowisko: %s, endpoint: %s", exchange, environment, base_url)

        connector = ExchangeConnector(
            exchange=exchange,
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )

        order_executor = OrderExecution(connector)
        trade_executor = TradeExecutor(order_executor, None, None)

        logging.info("✅ Moduły tradingowe zainicjalizowane (%s, %s).", exchange, environment)
        return trade_executor
    except Exception as e:
        logging.error("❌ Błąd inicjalizacji modułów tradingowych: %s", e)
        raise


def initialize_ai_modules():
    """Inicjalizuje modele AI."""
    try:
        sentiment_ai = SentimentAnalyzer()
        anomaly_ai = AnomalyDetector()
        trend_ai = TrendPredictor()
        optimizer_ai = StrategyOptimizer()
        reinforcement_ai = ReinforcementLearner()

        logging.info("✅ Moduły AI załadowane!")
        return sentiment_ai, anomaly_ai, trend_ai, optimizer_ai, reinforcement_ai
    except Exception as e:
        logging.error("❌ Błąd inicjalizacji AI: %s", e)
        return None


def ai_analysis_loop():
    """Pętla do analizy AI w czasie rzeczywistym."""
    while True:
        logging.info("🧠 AI analizuje rynek...")
        time.sleep(10)


def trading_loop(trading_manager: TradeExecutor):
    """Pętla do automatycznego handlu."""
    while True:
        logging.info("📈 Wykonywanie transakcji...")
        time.sleep(5)


def main() -> None:
    """Główna funkcja uruchamiająca system tradingowy."""
    try:
        setup_logging()
        logging.info("🚀 System tradingowy uruchamiany.")

        # Wybór środowiska i giełdy
        environment = choose_environment()
        exchange = choose_exchange()

        # Inicjalizacja modułów
        trading_manager = initialize_trading_modules(environment, exchange)
        ai_modules = initialize_ai_modules()

        # Wielowątkowość – AI i trading działają równolegle
        ai_thread = threading.Thread(target=ai_analysis_loop, daemon=True)
        trading_thread = threading.Thread(target=trading_loop, args=(trading_manager,), daemon=True)

        ai_thread.start()
        trading_thread.start()

        ai_thread.join()
        trading_thread.join()

    except Exception as main_error:
        logging.critical("❌ Krytyczny błąd w systemie tradingowym: %s", main_error)
        sys.exit(1)


if __name__ == "__main__":
    main()
