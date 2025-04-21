#!/usr/bin/env python3
"""
run_dashboard.py - Skrypt uruchamiający dashboard i wszystkie komponenty
"""

import os
import sys
import logging
import threading
import time
import signal
import requests
from subprocess import Popen
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Globalne zmienne dla procesów
api_server = None
streamlit_process = None
is_running = True

def ensure_directories():
    """Upewnia się, że wszystkie wymagane katalogi istnieją."""
    directories = [
        'logs',
        'data',
        'data/cache',
        'static',
        'static/img',
        'models',
        'saved_models'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Sprawdzono katalog: {directory}")

def check_port_availability(port):
    """Sprawdza, czy port jest dostępny."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except socket.error:
            return False

def wait_for_api_server(timeout=30):
    """Czeka na uruchomienie serwera API."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get('http://localhost:5000/api/health')
            if response.status_code == 200:
                logger.info("Serwer API jest gotowy")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    return False

def start_api_server():
    """Uruchamia serwer API Flask."""
    if not check_port_availability(5000):
        logger.error("Port 5000 jest już zajęty")
        return False

    try:
        from dashboard_api import app
        import threading
        
        def run_flask():
            app.run(host='0.0.0.0', port=5000)
        
        thread = threading.Thread(target=run_flask)
        thread.daemon = True
        thread.start()
        
        if not wait_for_api_server():
            logger.error("Nie można uruchomić serwera API")
            return False
            
        logger.info("Uruchomiono serwer API na porcie 5000")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania serwera API: {e}")
        return False

def start_streamlit():
    """Uruchamia dashboard Streamlit."""
    streamlit_port = 8502  # Zmiana portu z 8501 na 8502
    
    if not check_port_availability(streamlit_port):
        logger.error(f"Port {streamlit_port} jest już zajęty")
        return None

    try:
        streamlit_command = [
            'streamlit', 'run', 
            'dashboard.py',
            '--server.port', str(streamlit_port),
            '--server.address', '0.0.0.0'
        ]
        process = Popen(streamlit_command)
        logger.info(f"Uruchomiono dashboard Streamlit na porcie {streamlit_port}")
        return process
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania dashboardu: {e}")
        return None

def initialize_ai_models():
    """Inicjalizuje modele AI."""
    try:
        from ai_models.sentiment_ai import SentimentAnalyzer
        from ai_models.anomaly_detection import AnomalyDetector
        from ai_models.model_recognition import ModelRecognizer
        
        sentiment_analyzer = SentimentAnalyzer()
        anomaly_detector = AnomalyDetector()
        model_recognizer = ModelRecognizer()
        
        logger.info("Zainicjalizowano wszystkie modele AI")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji modeli AI: {e}")
        return False

def cleanup(signum=None, frame=None):
    """Funkcja czyszcząca przy zamykaniu."""
    global is_running, streamlit_process
    logger.info("Zatrzymywanie dashboardu...")
    is_running = False
    
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
        logger.info("Dashboard Streamlit zatrzymany")
    
    sys.exit(0)

def monitor_components():
    """Monitoruje stan komponentów."""
    while is_running:
        try:
            response = requests.get('http://localhost:5000/api/component-status')
            if response.status_code == 200:
                components = response.json().get('components', {})
                for component, status in components.items():
                    if status == 'offline':
                        logger.warning(f"Komponent {component} jest offline")
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas monitorowania komponentów: {e}")
        time.sleep(30)  # Sprawdzaj co 30 sekund

def main():
    """Główna funkcja uruchamiająca dashboard."""
    # Rejestracja handlera dla sygnałów
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Sprawdź i utwórz wymagane katalogi
    ensure_directories()
    
    # Inicjalizuj modele AI
    if not initialize_ai_models():
        logger.error("Nie można kontynuować bez modeli AI")
        return
    
    # Uruchom serwer API
    if not start_api_server():
        logger.error("Nie można uruchomić serwera API")
        return
    
    # Uruchom dashboard Streamlit
    global streamlit_process
    streamlit_process = start_streamlit()
    if not streamlit_process:
        logger.error("Nie można uruchomić dashboardu")
        return
    
    # Uruchom monitoring komponentów w osobnym wątku
    monitor_thread = threading.Thread(target=monitor_components)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        while is_running:
            time.sleep(1)
            # Sprawdź, czy procesy nadal działają
            if streamlit_process.poll() is not None:
                logger.error("Dashboard Streamlit nieoczekiwanie się zatrzymał")
                cleanup()
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()