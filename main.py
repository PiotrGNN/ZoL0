#!/usr/bin/env python3
"""
Main.py - Główny plik projektu Trading Bot System Analityczny
Zawiera funkcje inicjalizujące system i uruchamiające jego komponenty
"""

import os
import sys
import logging
from pathlib import Path

# Ustal ścieżkę do katalogu projektu
PROJECT_ROOT = Path(__file__).resolve().parent

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "app.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_config():
    """Ładuje konfigurację systemu z plików."""
    try:
        logger.info("Ładowanie konfiguracji systemu...")
        # Tu można dodać faktyczne ładowanie konfiguracji
        return {}
    except Exception as e:
        logger.error(f"Błąd podczas ładowania konfiguracji: {e}")
        return None


def initialize_components():
    """Inicjalizuje niezbędne komponenty systemu."""
    try:
        logger.info("Inicjalizacja komponentów systemu...")

        # Import modułu wykrywania anomalii
        from ai_models.anomaly_detection import AnomalyDetector
        anomaly_detector = AnomalyDetector(method="isolation_forest")

        # Tutaj można dodać inicjalizację innych komponentów
        # na razie tylko symulacja
        components = {
            "anomaly_detector": anomaly_detector
        }

        # Sprawdź, czy możliwe jest zainicjalizowanie managera strategii
        try:
            from data.strategies.strategy_manager import StrategyManager
            components["strategy_manager"] = StrategyManager()
            logger.info("Zainicjalizowano StrategyManager")
        except ImportError:
            logger.warning("Nie można zainicjalizować StrategyManager - kontynuowanie bez tego komponentu")

        return components
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji komponentów: {e}")
        raise


def setup_environment():
    """Konfiguracja środowiska aplikacji."""
    # Upewnij się, że folder logs istnieje
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    logger.info("Środowisko zostało skonfigurowane")


def display_welcome_message():
    """Wyświetla informację powitalną w konsoli"""
    welcome_message = """
    =================================================================
    🤖 Trading Bot - System Analityczny
    =================================================================
    Projekt gotowy do działania w środowisku Replit

    Dostępne tryby pracy:
    - Symulacja (domyślna)
    - Analiza
    - Testowanie

    Więcej informacji w README.md
    =================================================================
    """
    print(welcome_message)


def main():
    """Główna funkcja uruchamiająca system."""
    try:
        display_welcome_message()
        config = load_config()
        if not config:
            logger.warning("Nie udało się załadować konfiguracji. Używam wartości domyślnych.")

        components = initialize_components()
        if not components:
            logger.error("Nie udało się zainicjalizować komponentów. Kończenie pracy.")
            return

        setup_environment()
        logger.info("Uruchamianie systemu tradingowego...")

        # Przykładowe wykorzystanie modułu detekcji anomalii
        detector = components.get("anomaly_detector")
        if detector:
            detector.info()

        logger.info("System tradingowy uruchomiony pomyślnie")
    except Exception as e:
        logger.error(f"Wystąpił nieoczekiwany błąd: {e}")


if __name__ == "__main__":
    main()