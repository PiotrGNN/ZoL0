
#!/usr/bin/env python3
"""
run_dashboard.py - Skrypt uruchamiający dashboard i API
"""

import os
import sys
import logging
import subprocess
import time
import threading
import signal

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

# Upewnij się, że katalogi istnieją
os.makedirs("logs", exist_ok=True)

def start_api_server():
    """Uruchamia serwer API na porcie 5000"""
    logger.info("Uruchamianie serwera API...")
    
    # Uruchom API w trybie symulacji
    env = os.environ.copy()
    env["APP_ENV"] = "simulation"
    
    try:
        # Użyj subprocess do uruchomienia API w tle
        api_process = subprocess.Popen(
            [sys.executable, "dashboard_api.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Poczekaj chwilę na uruchomienie serwera
        time.sleep(2)
        
        # Sprawdź, czy proces działa
        if api_process.poll() is None:
            logger.info("Serwer API uruchomiony pomyślnie.")
            return api_process
        else:
            stdout, stderr = api_process.communicate()
            logger.error(f"Błąd podczas uruchamiania API: {stderr.decode()}")
            return None
    except Exception as e:
        logger.error(f"Nie udało się uruchomić serwera API: {e}")
        return None

def start_streamlit():
    """Uruchamia aplikację Streamlit"""
    logger.info("Uruchamianie dashboardu Streamlit...")
    
    try:
        # Uruchom Streamlit
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Poczekaj chwilę na uruchomienie
        time.sleep(3)
        
        # Sprawdź, czy proces działa
        if streamlit_process.poll() is None:
            logger.info("Dashboard Streamlit uruchomiony pomyślnie.")
            return streamlit_process
        else:
            stdout, stderr = streamlit_process.communicate()
            logger.error(f"Błąd podczas uruchamiania Streamlit: {stderr.decode()}")
            return None
    except Exception as e:
        logger.error(f"Nie udało się uruchomić Streamlit: {e}")
        return None

def main():
    """Główna funkcja uruchamiająca dashboard i API"""
    logger.info("Uruchamianie dashboardu tradingowego...")
    
    # Uruchom serwer API
    api_process = start_api_server()
    
    # Uruchom Streamlit
    streamlit_process = start_streamlit()
    
    if api_process is None or streamlit_process is None:
        logger.error("Nie udało się uruchomić wszystkich komponentów.")
        # Zakończ wszystkie procesy
        if api_process:
            api_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
        return 1
    
    # Obsługa sygnałów, aby poprawnie zamknąć procesy
    def signal_handler(sig, frame):
        logger.info("Otrzymano sygnał zakończenia. Zamykanie procesów...")
        api_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Monitoruj procesy
    try:
        while True:
            # Sprawdź, czy procesy działają
            if api_process.poll() is not None:
                logger.error("Serwer API zakończył działanie. Ponowne uruchamianie...")
                api_process = start_api_server()
                
            if streamlit_process.poll() is not None:
                logger.error("Dashboard Streamlit zakończył działanie. Ponowne uruchamianie...")
                streamlit_process = start_streamlit()
                
            time.sleep(5)
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")
    finally:
        # Zakończ procesy
        if api_process:
            api_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
