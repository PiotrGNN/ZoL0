
import logging
import os
from datetime import datetime, timedelta

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
    # Przykładowe ustawienia dla szablonu
    default_settings = {
        'risk_level': 'medium',
        'max_position_size': 10,
        'enable_auto_trading': False
    }
    
    # Przykładowe dane AI modeli
    ai_models = [
        {
            'name': 'Trend Predictor',
            'type': 'LSTM',
            'accuracy': 78.5,
            'status': 'Active',
            'last_used': '2025-04-07 10:15'
        },
        {
            'name': 'Sentiment Analyzer',
            'type': 'BERT',
            'accuracy': 82.3,
            'status': 'Active',
            'last_used': '2025-04-07 10:30'
        },
        {
            'name': 'Volatility Predictor',
            'type': 'XGBoost',
            'accuracy': 75.1,
            'status': 'Active',
            'last_used': '2025-04-07 09:45'
        }
    ]
    
    # Przykładowe dane strategii
    strategies = [
        {
            'id': 1,
            'name': 'Trend Following',
            'description': 'Strategia podążająca za trendem z wykorzystaniem SMA i EMA',
            'enabled': True,
            'win_rate': 65.8,
            'profit_factor': 1.75
        },
        {
            'id': 2,
            'name': 'Mean Reversion',
            'description': 'Strategia wykorzystująca powrót ceny do średniej',
            'enabled': False,
            'win_rate': 58.2,
            'profit_factor': 1.35
        },
        {
            'id': 3,
            'name': 'Breakout',
            'description': 'Strategia bazująca na przełamaniach poziomów wsparcia/oporu',
            'enabled': True,
            'win_rate': 62.0,
            'profit_factor': 1.65
        }
    ]
    
    return render_template(
        'dashboard.html',
        settings=default_settings,
        ai_models=ai_models,
        strategies=strategies,
        trades=[],
        alerts=[],
        sentiment_data=None,
        anomalies=[]
    )

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

@app.route('/api/chart-data')
def get_chart_data():
    """Endpoint dostarczający dane do wykresów"""
    try:
        # Symulowane dane do wykresu
        days = 30
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        dates.reverse()
        
        # Symulacja wartości portfela
        import random
        initial_value = 10000
        values = [initial_value]
        for i in range(1, days):
            change = random.uniform(-200, 300)
            values.append(round(values[-1] + change, 2))
        
        return jsonify({
            'success': True,
            'data': {
                'labels': dates,
                'datasets': [
                    {
                        'label': 'Wartość Portfela',
                        'data': values,
                        'borderColor': '#4CAF50',
                        'backgroundColor': 'rgba(76, 175, 80, 0.1)'
                    }
                ]
            }
        })
    except Exception as e:
        logging.error(f"Błąd podczas generowania danych wykresu: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications')
def get_notifications():
    """Endpoint do pobierania powiadomień systemowych"""
    try:
        # Przykładowe powiadomienia
        notifications = [
            {
                'id': 1,
                'type': 'info',
                'message': 'System został uruchomiony poprawnie.',
                'timestamp': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 2,
                'type': 'warning',
                'message': 'Wykryto zwiększoną zmienność na rynku BTC/USDT.',
                'timestamp': (datetime.now() - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 3,
                'type': 'success',
                'message': 'Transakcja kupna ETH/USDT zakończona powodzeniem.',
                'timestamp': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        
        return jsonify({
            'success': True,
            'notifications': notifications
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania powiadomień: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/component-status')
def get_component_status():
    """Endpoint do pobierania statusu komponentów systemu"""
    try:
        components = [
            {
                'id': 'trading_engine',
                'name': 'Silnik tradingowy', 
                'status': 'online',
                'last_updated': (datetime.now() - timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 'data_fetcher',
                'name': 'Pobieranie danych', 
                'status': 'online',
                'last_updated': (datetime.now() - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 'risk_management',
                'name': 'Zarządzanie ryzykiem', 
                'status': 'online',
                'last_updated': (datetime.now() - timedelta(minutes=3)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 'ml_prediction',
                'name': 'Predykcje AI', 
                'status': 'online',
                'last_updated': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 'order_execution',
                'name': 'Wykonywanie zleceń', 
                'status': 'online',
                'last_updated': (datetime.now() - timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        
        return jsonify({
            'success': True,
            'components': components,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu komponentów: {e}")
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

@app.route('/api/trading-stats')
def get_trading_stats():
    """Endpoint do pobierania statystyk tradingowych"""
    try:
        # Symulowane statystyki dla demonstracji
        return jsonify({
            'success': True,
            'stats': {
                'total_trades': 48,
                'winning_trades': 32,
                'losing_trades': 16,
                'win_rate': 66.7,
                'avg_win': 42.35,
                'avg_loss': 28.75,
                'profit_factor': 1.85,
                'sharpe_ratio': 1.92,
                'max_drawdown': 8.2,
                'avg_trade_duration': '3h 45m',
                'most_profitable_pair': 'ETH/USDT',
                'most_losing_pair': 'SOL/USDT'
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statystyk tradingowych: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recent-trades')
def get_recent_trades():
    """Endpoint do pobierania ostatnich transakcji"""
    try:
        # Symulowane transakcje dla demonstracji
        trades = [
            {
                'id': 'T123456',
                'symbol': 'BTC/USDT',
                'type': 'buy',
                'entry_price': 62450.50,
                'exit_price': 62890.25,
                'size': 0.05,
                'profit_loss': '+2.15%',
                'status': 'closed',
                'entry_time': (datetime.now() - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': (datetime.now() - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'),
                'strategy': 'Trend Following'
            },
            {
                'id': 'T123457',
                'symbol': 'ETH/USDT',
                'type': 'sell',
                'entry_price': 3245.75,
                'exit_price': 3180.25,
                'size': 0.8,
                'profit_loss': '+1.98%',
                'status': 'closed',
                'entry_time': (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'strategy': 'Breakout'
            },
            {
                'id': 'T123458',
                'symbol': 'SOL/USDT',
                'type': 'buy',
                'entry_price': 124.85,
                'size': 10,
                'profit_loss': '-0.75%',
                'status': 'open',
                'entry_time': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                'strategy': 'Mean Reversion'
            }
        ]
        
        return jsonify({
            'success': True,
            'trades': trades,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania ostatnich transakcji: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alerts')
def get_alerts():
    """Endpoint do pobierania alertów systemowych"""
    try:
        # Symulowane alerty dla demonstracji
        alerts = [
            {
                'id': 'A1001',
                'type': 'warning',
                'message': 'Zwiększona zmienność na parze BTC/USDT',
                'timestamp': (datetime.now() - timedelta(minutes=25)).strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'active'
            },
            {
                'id': 'A1002',
                'type': 'info',
                'message': 'Wykryto wzorzec świecowy "młot" na ETH/USDT',
                'timestamp': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'active'
            },
            {
                'id': 'A1003',
                'type': 'danger',
                'message': 'Stop loss uruchomiony dla pozycji SOL/USDT',
                'timestamp': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'active'
            }
        ]
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania alertów: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai-models-status')
def get_ai_models_status():
    """Endpoint do pobierania statusu modeli AI"""
    try:
        # Symulowane dane modeli AI dla demonstracji
        models = [
            {
                'name': 'Trend Predictor',
                'type': 'LSTM',
                'accuracy': 78.5,
                'status': 'active',
                'last_trained': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
                'last_prediction': (datetime.now() - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M')
            },
            {
                'name': 'Sentiment Analyzer',
                'type': 'BERT',
                'accuracy': 82.3,
                'status': 'active',
                'last_trained': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M'),
                'last_prediction': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M')
            },
            {
                'name': 'Volatility Predictor',
                'type': 'XGBoost',
                'accuracy': 75.1,
                'status': 'active',
                'last_trained': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M'),
                'last_prediction': (datetime.now() - timedelta(minutes=8)).strftime('%Y-%m-%d %H:%M')
            }
        ]
        
        return jsonify({
            'success': True,
            'models': models,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu modeli AI: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Uruchomienie aplikacji
if __name__ == "__main__":
    # Tworzenie katalogu logs jeśli nie istnieje
    os.makedirs("logs", exist_ok=True)
    
    # Inicjalizacja systemu
    initialize_system()
    
    # Uruchomienie aplikacji
    app.run(host='0.0.0.0', port=5000, debug=True)
