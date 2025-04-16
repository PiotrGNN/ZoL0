#!/usr/bin/env python3
"""
run_dashboard.py - Skrypt uruchamiający dashboard i API
"""

import os
import sys
import logging
import subprocess
import threading
import time
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/dashboard.log")
    ]
)
logger = logging.getLogger(__name__)

# Upewnij się, że potrzebne katalogi istnieją
for directory in ["logs", "data/cache", "static/img"]:
    os.makedirs(directory, exist_ok=True)

def run_api_server():
    """Uruchamia serwer API Flask."""
    try:
        logger.info("Uruchamianie serwera API...")
        # Importuj i uruchom dashboard_api.py
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dashboard_api

        # Uruchom serwer Flask
        dashboard_api.app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania serwera API: {e}")
        sys.exit(1)

def run_streamlit_dashboard():
    """Uruchamia dashboard Streamlit."""
    try:
        logger.info("Uruchamianie dashboardu Streamlit...")

        # Przygotuj ścieżkę do pliku dashboard.py
        dashboard_path = Path(__file__).parent / "dashboard.py"

        # Uruchom komendę streamlit run
        cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", "8501", "--server.address", "0.0.0.0"]
        logger.info(f"Uruchamiam komendę: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Czytaj logi z procesu
        for line in process.stdout:
            logger.info(f"Streamlit: {line.strip()}")

        # Jeśli proces zakończył się, sprawdź błędy
        for line in process.stderr:
            logger.error(f"Streamlit error: {line.strip()}")

        # Zwróć kod wyjścia
        return process.returncode

    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania dashboardu Streamlit: {e}")
        return 1

def main():
    """Funkcja główna."""
    logger.info("Uruchamianie dashboardu tradingowego...")

    # Uruchom serwer API w osobnym wątku
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # Daj serwerowi API czas na uruchomienie
    logger.info("Czekam 3 sekundy na uruchomienie serwera API...")
    time.sleep(3)

    # Uruchom dashboard Streamlit
    exit_code = run_streamlit_dashboard()

    # Zakończ z odpowiednim kodem wyjścia
    sys.exit(exit_code)

if __name__ == "__main__":
    main()