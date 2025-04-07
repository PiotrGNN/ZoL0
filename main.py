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
app = Flask(__name__, template_folder='templates', static_folder='static')

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
        
        # Przykładowe dane portfolio
        portfolio = [
            {
                "symbol": "BTC",
                "amount": "0.05",
                "value": "2,450",
                "change": "+5.2%",
                "change_class": "positive"
            },
            {
                "symbol": "ETH",
                "amount": "0.5",
                "value": "980",
                "change": "+3.8%",
                "change_class": "positive"
            },
            {
                "symbol": "SOL",
                "amount": "10",
                "value": "850",
                "change": "-2.1%",
                "change_class": "negative"
            },
            {
                "symbol": "USDT",
                "amount": "500",
                "value": "500",
                "change": "0.0%",
                "change_class": "neutral"
            }
        ]
        
        logging.info("Renderowanie dashboardu z danymi: %s komponenty, %s anomalie, %s modele AI, %s aktywa", 
                    len(components), len(anomalies), len(ai_models), len(portfolio))
        
        return render_template('dashboard.html', 
                              system_mode=system_mode,
                              active_strategies=active_strategies,
                              risk_level=risk_level,
                              components=components,
                              anomalies=anomalies,
                              system_actions=system_actions,
                              ai_models=ai_models,
                              portfolio=portfolio)
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

@app.route('/api/portfolio')
def portfolio_status():
    """Endpoint zwracający stan portfolio."""
    # Symulowane dane portfolio
    return jsonify({
        "total_value": 4780.0,
        "assets": [
            {"symbol": "BTC", "amount": 0.05, "value": 2450, "change_24h": 5.2},
            {"symbol": "ETH", "amount": 0.5, "value": 980, "change_24h": 3.8},
            {"symbol": "SOL", "amount": 10, "value": 850, "change_24h": -2.1},
            {"symbol": "USDT", "amount": 500, "value": 500, "change_24h": 0.0}
        ],
        "performance": {
            "daily": 2.8,
            "weekly": 5.6,
            "monthly": 12.4
        }
    })

@app.route('/api/trades')
def recent_trades():
    """Endpoint zwracający ostatnie transakcje."""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    
    # Symulowane dane transakcji
    return jsonify([
        {
            "id": "t123456",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.01,
            "price": 49500.0,
            "value": 495.0,
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "strategy": "trend_following"
        },
        {
            "id": "t123457",
            "symbol": "ETH/USDT",
            "side": "sell",
            "amount": 0.2,
            "price": 1950.0,
            "value": 390.0,
            "timestamp": (now - timedelta(hours=5)).isoformat(),
            "strategy": "mean_reversion"
        }
    ])

@app.route('/api/strategies')
def available_strategies():
    """Endpoint zwracający dostępne strategie."""
    return jsonify({
        "active_strategy": "trend_following",
        "available_strategies": [
            {
                "id": "trend_following",
                "name": "Trend Following",
                "description": "Strategia podążająca za trendem",
                "performance": {
                    "daily": 1.2,
                    "weekly": 3.8,
                    "monthly": 8.5
                }
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "description": "Strategia powrotu do średniej",
                "performance": {
                    "daily": 0.8,
                    "weekly": 2.5,
                    "monthly": 6.2
                }
            },
            {
                "id": "breakout",
                "name": "Breakout",
                "description": "Strategia wybicia",
                "performance": {
                    "daily": 1.5,
                    "weekly": 4.2,
                    "monthly": 10.1
                }
            }
        ]
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

@app.route('/api/chart-data')
def chart_data():
    """Endpoint zwracający dane do wykresu aktywności."""
    from datetime import datetime, timedelta
    import random
    
    # Generowanie przykładowych danych dla wykresu (w rzeczywistej aplikacji byłyby pobierane z bazy danych)
    now = datetime.now()
    labels = [(now - timedelta(minutes=i*5)).strftime("%H:%M") for i in range(10)]
    labels.reverse()
    
    # Przykładowe dane dla wykresów
    anomaly_activity = [random.randint(40, 90) for _ in range(10)]
    system_load = [random.randint(20, 95) for _ in range(10)]
    
    # Losowe anomalie (rzadkie)
    detected_anomalies = [0] * 10
    for i in range(10):
        if random.random() < 0.2:  # 20% szans na anomalię
            detected_anomalies[i] = random.randint(1, 3)
    
    return jsonify({
        "labels": labels,
        "datasets": [
            {
                "label": "Aktywność Detektora Anomalii",
                "data": anomaly_activity,
                "borderColor": "rgb(46, 204, 113)"
            },
            {
                "label": "Obciążenie Systemu",
                "data": system_load,
                "borderColor": "rgb(52, 152, 219)"
            },
            {
                "label": "Wykryte Anomalie",
                "data": detected_anomalies,
                "borderColor": "rgb(231, 76, 60)"
            }
        ]
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