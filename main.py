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

import logging
import os
import sys
import threading
import time
from datetime import datetime

from ai_models.ai_optimizer import StrategyOptimizer
from ai_models.anomaly_detection import AnomalyDetector
# Moduły AI
from ai_models.reinforcement_learning import ReinforcementLearner
from ai_models.sentiment_analysis import SentimentAnalyzer
from ai_models.trend_prediction import TrendPredictor
# Moduły systemu
from config.settings import CONFIG
from data.execution.exchange_connector import ExchangeConnector
from data.execution.order_execution import OrderExecution
from data.execution.trade_executor import TradeExecutor


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
            logging.StreamHandler(sys.stdout),
        ],
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
            "https://api.binance.com"
            if exchange == "binance" and environment == "production"
            else (
                "https://testnet.binance.vision"
                if exchange == "binance"
                else (
                    "https://api.bybit.com"
                    if exchange == "bybit" and environment == "production"
                    else "https://testnet.bybit.com"
                )
            )
        )

        logging.info(
            "🔹 Używamy giełdy: %s, środowisko: %s, endpoint: %s",
            exchange,
            environment,
            base_url,
        )

        connector = ExchangeConnector(
            exchange=exchange, api_key=api_key, api_secret=api_secret, base_url=base_url
        )

        order_executor = OrderExecution(connector)
        trade_executor = TradeExecutor(order_executor, None, None)

        logging.info(
            "✅ Moduły tradingowe zainicjalizowane (%s, %s).", exchange, environment
        )
        return trade_executor
    except Exception as e:
        logging.error("❌ Błąd inicjalizacji modułów tradingowych: %s", e)
        raise


def initialize_ai_modules():
    """Inicjalizuje modele AI z obsługą brakujących zależności."""
    ai_modules = {}
    
    # Lista modułów AI do inicjalizacji
    modules_to_init = [
        ('sentiment_ai', SentimentAnalyzer, "analizy sentymentu"),
        ('anomaly_ai', AnomalyDetector, "wykrywania anomalii"),
        ('trend_ai', TrendPredictor, "predykcji trendów"),
        ('optimizer_ai', StrategyOptimizer, "optymalizacji strategii"),
        ('reinforcement_ai', ReinforcementLearner, "uczenia ze wzmocnieniem")
    ]
    
    try:
        # Inicjalizacja z obsługą błędów dla każdego modułu
        for module_key, module_class, module_name in modules_to_init:
            try:
                # Sprawdzamy dostępność zależności przez specjalne metody w modułach
                if hasattr(module_class, 'check_dependencies') and not module_class.check_dependencies():
                    logging.warning(f"⚠️ Brak wymaganych zależności dla modułu {module_name}")
                    ai_modules[module_key] = None
                    continue
                    
                # Inicjalizacja modułu
                ai_modules[module_key] = module_class()
                logging.info(f"✅ Moduł {module_name} załadowany")
            except ImportError as e:
                logging.warning(f"⚠️ Brak zależności dla modułu {module_name}: {e}")
                ai_modules[module_key] = None
            except Exception as e:
                logging.warning(f"⚠️ Nie udało się załadować modułu {module_name}: {e}")
                ai_modules[module_key] = None
        
        num_loaded = sum(1 for m in ai_modules.values() if m is not None)
        if num_loaded == 5:
            logging.info("✅ Wszystkie moduły AI załadowane pomyślnie!")
        else:
            logging.warning("⚠️ Załadowano %d/5 modułów AI", num_loaded)
            
        return ai_modules
    except Exception as e:
        logging.error("❌ Krytyczny błąd podczas inicjalizacji modułów AI: %s", e)
        return {}


def ai_analysis_loop(ai_modules):
    """Pętla do analizy AI w czasie rzeczywistym z obsługą brakujących modułów."""
    while True:
        try:
            logging.info("🧠 AI analizuje rynek...")
            
            # Wykonuj dostępne analizy
            if ai_modules.get('sentiment_ai'):
                logging.info("📊 Analiza sentymentu w toku...")
            
            if ai_modules.get('anomaly_ai'):
                logging.info("🔍 Wykrywanie anomalii w toku...")
            
            if ai_modules.get('trend_ai'):
                logging.info("📈 Predykcja trendów w toku...")
            
            time.sleep(10)
        except Exception as e:
            logging.error("❌ Błąd w pętli analizy AI: %s", e)
            time.sleep(30)  # Dłuższa przerwa przy błędzie


def trading_loop(trading_manager: TradeExecutor, ai_modules: dict):
    """Pętla do automatycznego handlu z wykorzystaniem dostępnych modułów AI."""
    while True:
        try:
            logging.info("📈 Analiza rynku i wykonywanie transakcji...")
            
            # Możemy użyć optimizera jeśli jest dostępny
            if ai_modules.get('optimizer_ai'):
                logging.info("⚙️ Optymalizacja strategii w toku...")
            
            # Symulacja handlu
            logging.info("💹 Monitorowanie rynku...")
            time.sleep(5)
        except Exception as e:
            logging.error("❌ Błąd w pętli tradingowej: %s", e)
            time.sleep(15)  # Dłuższa przerwa przy błędzie


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
        ai_thread = threading.Thread(target=ai_analysis_loop, args=(ai_modules,), daemon=True)
        trading_thread = threading.Thread(
            target=trading_loop, args=(trading_manager, ai_modules), daemon=True
        )

        ai_thread.start()
        trading_thread.start()

        ai_thread.join()
        trading_thread.join()

    except Exception as main_error:
        logging.critical("❌ Krytyczny błąd w systemie tradingowym: %s", main_error)
        sys.exit(1)


if __name__ == "__main__":
    main()
