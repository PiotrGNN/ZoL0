
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Inicjalizacja komponentów systemu (import wewnątrz funkcji dla uniknięcia cyklicznych importów)
def initialize_system():
    try:
        from data.utils.notification_system import NotificationSystem
        from data.indicators.sentiment_analysis import SentimentAnalyzer
        from ai_models.anomaly_detection import AnomalyDetector
        
        global notification_system, sentiment_analyzer, anomaly_detector
        
        notification_system = NotificationSystem()
        sentiment_analyzer = SentimentAnalyzer()
        anomaly_detector = AnomalyDetector()
        
        logging.info("System zainicjalizowany poprawnie")
        return True
    except Exception as e:
        logging.error(f"Błąd podczas inicjalizacji systemu: {e}")
        return False

# Trasy aplikacji
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# API endpoints
@app.route('/api/dashboard/data')
def get_dashboard_data():
    try:
        # Symulowane dane dla demonstracji
        return jsonify({
            'success': True,
            'balance': 10250.75,
            'profit_loss': 325.50,
            'open_positions': 3,
            'total_trades': 48,
            'win_rate': 68.5,
            'max_drawdown': 8.2,
            'market_sentiment': sentiment_analyzer.get_current_sentiment() if 'sentiment_analyzer' in globals() else 'Neutralny',
            'anomalies': anomaly_detector.get_detected_anomalies() if 'anomaly_detector' in globals() else [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania danych dashboardu: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/status')
def get_system_status():
    return jsonify({
        'success': True,
        'components': {
            'trading_engine': 'active',
            'data_fetcher': 'active',
            'risk_management': 'active',
            'ml_prediction': 'active',
            'order_execution': 'active',
            'notification_system': 'active' if 'notification_system' in globals() else 'inactive',
        },
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    try:
        # Tu byłaby logika uruchamiania systemu tradingowego
        return jsonify({'success': True, 'message': 'Trading automatyczny uruchomiony'})
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania tradingu: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    try:
        # Tu byłaby logika zatrzymywania systemu tradingowego
        return jsonify({'success': True, 'message': 'Trading automatyczny zatrzymany'})
    except Exception as e:
        logging.error(f"Błąd podczas zatrzymywania tradingu: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/reset', methods=['POST'])
def reset_system():
    try:
        # Tu byłaby logika resetowania systemu
        return jsonify({'success': True, 'message': 'System zresetowany'})
    except Exception as e:
        logging.error(f"Błąd podczas resetowania systemu: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Uruchomienie aplikacji
if __name__ == "__main__":
    # Tworzenie katalogu logs jeśli nie istnieje
    os.makedirs("logs", exist_ok=True)
    
    # Inicjalizacja systemu
    initialize_system()
    
    # Uruchomienie aplikacji
    app.run(host='0.0.0.0', port=5000, debug=True)
