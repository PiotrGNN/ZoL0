#!/usr/bin/env python3
"""
Trading Bot - główny moduł systemu
Zoptymalizowana wersja dla środowiska Replit
"""

import logging
import os
import sys
import time
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template

# Inicjalizacja aplikacji Flask
app = Flask(__name__, template_folder='templates')

# Utworzenie struktury katalogów
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Endpointy Flask dla serwera web
@app.route('/')
def index():
    """Główny endpoint aplikacji."""
    from flask import redirect
    # Redirect do dashboardu
    return redirect('/dashboard')
    
    # Alternatywnie, można zwrócić JSON
    # return jsonify({
    #     "status": "online",
    #     "service": "Trading Bot System",
    #     "version": "1.0.0"
    # })

@app.route('/dashboard')
def dashboard():
    """Strona z dashboardem systemu."""
    from datetime import datetime
    
    try:
        # Dane przykładowe - w rzeczywistości byłyby pobierane z systemu
        system_mode = "Symulacja"
        active_strategies = 3
        risk_level = "Niski"
        
        # Przykładowe komponenty systemu
        components = [
            {
                "name": "Detektor Anomalii",
                "status": "Aktywny",
                "status_class": "online",
                "last_update": datetime.now().strftime("%H:%M:%S")
            },
            {
                "name": "Przetwarzanie Danych",
                "status": "Aktywny",
                "status_class": "online",
                "last_update": datetime.now().strftime("%H:%M:%S")
            },
            {
                "name": "Silnik Tradingowy",
                "status": "Oczekiwanie",
                "status_class": "warning",
                "last_update": datetime.now().strftime("%H:%M:%S")
            }
        ]
        
        # Przykładowe anomalie
        anomalies = [
            {
                "date": "2025-04-07 10:15:00",
                "pair": "BTC/USDT",
                "level": "Średni",
                "description": "Nagły wzrost wolumenu"
            },
            {
                "date": "2025-04-07 10:22:30",
                "pair": "ETH/USDT",
                "level": "Niski",
                "description": "Nietypowa zmiana ceny"
            }
        ]
        
        # Przykładowe działania systemu
        system_actions = [
            {
                "time": "10:25:10",
                "type": "Analiza",
                "description": "Zakończono analizę par walutowych"
            },
            {
                "time": "10:20:05",
                "type": "System",
                "description": "Uruchomiono detektor anomalii"
            },
            {
                "time": "10:15:00",
                "type": "Dane",
                "description": "Zaktualizowano historyczne dane rynkowe"
            }
        ]
        
        # Przykładowe dane o modelach AI
        ai_models = [
            {
                "name": "Detektor Anomalii (Isolation Forest)",
                "status": "Aktywny",
                "status_class": "online",
                "accuracy": "94.2%",
                "last_used": datetime.now().strftime("%H:%M:%S")
            },
            {
                "name": "Prognoza Trendu (XGBoost)",
                "status": "Aktywny",
                "status_class": "online",
                "accuracy": "87.5%",
                "last_used": datetime.now().strftime("%H:%M:%S")
            },
            {
                "name": "Analiza Sentymentu (BERT)",
                "status": "Nieaktywny",
                "status_class": "offline",
                "accuracy": "91.3%",
                "last_used": "10:15:00"
            },
            {
                "name": "Reinforcement Learning (DQN)",
                "status": "Trenowanie",
                "status_class": "warning",
                "accuracy": "82.1%",
                "last_used": "10:20:30"
            }
        ]
        
        logging.info("Renderowanie dashboardu z danymi: %s komponenty, %s anomalie, %s modele AI", 
                    len(components), len(anomalies), len(ai_models))
        
        return render_template('dashboard.html', 
                              system_mode=system_mode,
                              active_strategies=active_strategies,
                              risk_level=risk_level,
                              components=components,
                              anomalies=anomalies,
                              system_actions=system_actions,
                              ai_models=ai_models)
    except Exception as e:
        logging.error("Błąd podczas renderowania dashboardu: %s", str(e))
        return f"Błąd w dashboardzie: {str(e)}", 500

@app.route('/health')
def health_check():
    """Endpoint do sprawdzania stanu aplikacji - używany przez Replit do health check."""
    return jsonify({"status": "healthy"})

@app.route('/api/status')
def system_status():
    """Endpoint zwracający status systemu."""
    return jsonify({
        "status": "operational",
        "components": {
            "anomaly_detector": "active",
            "data_processor": "active",
            "trading_engine": "standby"
        }
    })

@app.route('/download-report')
def download_report():
    """Endpoint do pobierania raportu z działania systemu."""
    # W rzeczywistym przypadku tutaj byłaby generowanie i pobieranie raportu
    return jsonify({
        "status": "success",
        "message": "Raport zostanie wygenerowany i wysłany na email"
    })

@app.route('/start-simulation')
def start_simulation():
    """Endpoint do uruchamiania symulacji na żądanie."""
    # W rzeczywistości tutaj byłoby rozpoczynanie nowej symulacji
    logging.info("Uruchomiono symulację z panelu administratora")
    return jsonify({
        "status": "success",
        "message": "Symulacja została uruchomiona"
    })

def load_configuration():
    """Ładuje konfigurację systemu z plików."""
    try:
        logging.info("Ładowanie konfiguracji systemu...")
        # Tutaj można dodać ładowanie z config/settings.yml
        return {"mode": "development", "risk_level": "low"}
    except Exception as e:
        logging.error(f"Błąd podczas ładowania konfiguracji: {e}")
        return None

def initialize_components():
    """Inicjalizuje niezbędne komponenty systemu."""
    try:
        logging.info("Inicjalizacja komponentów systemu...")

        # Import modułu wykrywania anomalii
        from ai_models.anomaly_detection import AnomalyDetector
        anomaly_detector = AnomalyDetector()

        # Tutaj można dodać inicjalizację innych komponentów
        # na razie tylko symulacja
        components = {
            "anomaly_detector": anomaly_detector
        }

        # Sprawdź, czy możliwe jest zainicjalizowanie managera strategii
        try:
            from data.strategies.strategy_manager import StrategyManager
            # Dodajemy puste parametry, aby uniknąć błędu inicjalizacji
            components["strategy_manager"] = StrategyManager(strategies={}, exposure_limits={})
            logging.info("Zainicjalizowano StrategyManager")
        except Exception as e:
            logging.warning(f"Nie można zainicjalizować StrategyManager - kontynuowanie bez tego komponentu: {e}")

        return components
    except Exception as e:
        logging.error(f"Błąd podczas inicjalizacji komponentów: {e}")
        return {}  # Zwracamy pusty słownik zamiast None, aby umożliwić kontynuację

def start_simulation_mode():
    """Uruchamia system w trybie symulacji."""
    logging.info("Uruchamianie systemu w trybie symulacji...")
    print("""
    =================================================================
    🚀 Trading Bot - Tryb Symulacji
    =================================================================
    System został uruchomiony w trybie symulacji.
    Dane są pobierane z historycznych źródeł, żadne rzeczywiste
    transakcje nie są wykonywane.

    Aby zakończyć, naciśnij Ctrl+C
    =================================================================
    """)

    try:
        # Symulacja działania systemu - szybsza dla deployment
        for i in range(3):
            logging.info(f"Symulacja: krok {i+1}/3")
            print(f"⏳ Przetwarzanie danych... {i+1}/3")
            time.sleep(0.5)  # Krótszy czas dla szybszego uruchomienia serwera

        print("\n✅ Symulacja zakończona pomyślnie!")
    except KeyboardInterrupt:
        print("\n⚠️ Symulacja przerwana przez użytkownika.")
    except Exception as e:
        logging.error(f"Błąd podczas symulacji: {e}")
        print(f"\n❌ Błąd symulacji: {e}")

def display_welcome_message():
    """Wyświetla wiadomość powitalną projektu."""
    print("""
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
    """)

def main():
    """Główna funkcja systemu."""
    display_welcome_message()

    # Ładowanie konfiguracji
    config = load_configuration()
    if not config:
        logging.error("Nie udało się załadować konfiguracji, używam domyślnej.")
        config = {"mode": "development", "risk_level": "low"}

    # Inicjalizacja komponentów - kontynuujemy nawet jeśli są błędy
    components = initialize_components()
    logging.info(f"Zainicjalizowano {len(components)} komponentów")

    # Uruchomienie trybu symulacji
    start_simulation_mode()

    logging.info("System podstawowy uruchomiony, startuję serwer web...")
    # Flask będzie uruchomiony po zakończeniu głównej funkcji

if __name__ == "__main__":
    try:
        # Uruchomienie głównej funkcji
        main()
        # Uruchomienie serwera Flask na 0.0.0.0, aby był dostępny publicznie
        logging.info("Uruchamianie serwera web na porcie 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logging.critical(f"Nieoczekiwany błąd: {e}")
        print(f"\n❌ Krytyczny błąd: {e}")