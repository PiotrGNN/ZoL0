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
logger = logging.getLogger(__name__)

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Import klienta ByBit
from data.execution.bybit_connector import BybitConnector

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Deklaracja zmiennych globalnych
global bybit_client, notification_system, sentiment_analyzer, anomaly_detector
bybit_client = None
notification_system = None
sentiment_analyzer = None
anomaly_detector = None

# Inicjalizacja komponentów systemu (import wewnątrz funkcji dla uniknięcia cyklicznych importów)
def initialize_system():
    try:
        from data.utils.notification_system import NotificationSystem
        from data.indicators.sentiment_analysis import SentimentAnalyzer
        from ai_models.anomaly_detection import AnomalyDetector
        from data.utils.import_manager import exclude_modules_from_auto_import

        # Wykluczamy podpakiet tests z automatycznego importu
        exclude_modules_from_auto_import(["tests"])
        logging.info("Wykluczam podpakiet 'tests' z automatycznego importu.")

        global notification_system, sentiment_analyzer, anomaly_detector, bybit_client

        notification_system = NotificationSystem()
        logging.info("Zainicjalizowano system powiadomień z 2 kanałami.")

        sentiment_analyzer = SentimentAnalyzer()

        anomaly_detector = AnomalyDetector()

        # Inicjalizacja klienta ByBit (tylko raz podczas startu aplikacji)
        try:
            api_key = os.getenv("BYBIT_API_KEY")
            api_secret = os.getenv("BYBIT_API_SECRET")
            if not api_key or not api_secret:
                logging.warning("Brak kluczy API ByBit w zmiennych środowiskowych. Sprawdź plik .env")
                # Ustawiamy hardcoded klucze jako fallback - NIE ROBIC TEGO W PRODUKCJI!
                api_key = "9VJok4aTxM7RRHJWCW"
                api_secret = "EpQuWQiHwgVNQeKHqedA7i69Qv6mVFasxv2F"

            bybit_client = BybitConnector(
                api_key=api_key,
                api_secret=api_secret,
                use_testnet=os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"
            )
            server_time = bybit_client.get_server_time()
            logger.info(f"Klient API ByBit zainicjalizowany pomyślnie. Czas serwera: {server_time}")
        except Exception as e:
            logger.error(f"Błąd inicjalizacji klienta ByBit: {e}")
            bybit_client = None


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

    # Pobieranie stanu portfela
    portfolio = None
    if bybit_client:
        try:
            portfolio = bybit_client.get_account_balance()
            logging.info(f"Pobrano portfolio: {portfolio}")
        except Exception as e:
            logging.error(f"Błąd podczas pobierania portfolio: {e}")
            portfolio = {"error": str(e)}

    return render_template(
        'dashboard.html',
        settings=default_settings,
        ai_models=ai_models,
        strategies=strategies,
        trades=[],
        alerts=[],
        sentiment_data=None,
        anomalies=[],
        portfolio=portfolio
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
            'market_sentiment': sentiment_analyzer.analyze()["analysis"] if 'sentiment_analyzer' in globals() else 'Neutralny',
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

@app.route('/api/recent-trades')
def get_recent_trades():
    """Endpoint do pobierania ostatnich transakcji"""
    try:
        # Przykładowe dane transakcji
        trades = [
            {
                'symbol': 'BTC/USDT',
                'type': 'Buy',
                'time': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'profit': 3.2
            },
            {
                'symbol': 'ETH/USDT',
                'type': 'Sell',
                'time': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                'profit': -1.5
            },
            {
                'symbol': 'SOL/USDT',
                'type': 'Buy',
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'profit': 2.7
            }
        ]
        return jsonify({'trades': trades})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania ostatnich transakcji: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Endpoint do pobierania alertów systemowych"""
    try:
        # Przykładowe alerty
        alerts = [
            {
                'level': 'warning',
                'level_class': 'warning',
                'time': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                'message': 'Wysoka zmienność rynkowa'
            },
            {
                'level': 'info',
                'level_class': 'online',
                'time': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
                'message': 'Nowa strategia aktywowana'
            }
        ]
        return jsonify({'alerts': alerts})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania alertów: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading-stats')
def get_trading_stats():
    """Endpoint do pobierania statystyk tradingowych"""
    try:
        stats = {
            'profit': '$356.42',
            'trades_count': 28,
            'win_rate': '67.8%',
            'max_drawdown': '8.2%'
        }
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statystyk tradingowych: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/component-status')
def get_component_status():
    """Endpoint do pobierania statusu komponentów systemu"""
    try:
        components = [
            {
                'id': 'api-connector',
                'status': 'online'
            },
            {
                'id': 'data-processor',
                'status': 'online'
            },
            {
                'id': 'trading-engine',
                'status': 'warning'
            },
            {
                'id': 'risk-manager',
                'status': 'online'
            }
        ]
        return jsonify({'components': components})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu komponentów: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-models-status')
def get_ai_models_status():
    """Endpoint do pobierania statusu modeli AI"""
    try:
        models = [
            {
                'name': 'Trend Predictor',
                'type': 'LSTM',
                'accuracy': 78.5,
                'status': 'Active',
                'last_used': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'name': 'Sentiment Analyzer',
                'type': 'BERT',
                'accuracy': 82.3,
                'status': 'Active',
                'last_used': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'name': 'Volatility Predictor',
                'type': 'XGBoost',
                'accuracy': 75.1,
                'status': 'Active',
                'last_used': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        return jsonify({'models': models})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu modeli AI: {e}")
        return jsonify({'error': str(e)}), 500

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
            'bybit_api':'active' if bybit_client else 'inactive'
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

@app.route("/api/bybit/server-time", methods=["GET"])
def get_bybit_server_time():
    """Endpoint zwracający czas serwera ByBit."""
    if not bybit_client:
        return jsonify({"error": "Klient ByBit nie jest zainicjalizowany"}), 500

    try:
        server_time = bybit_client.get_server_time()
        return jsonify(server_time)
    except Exception as e:
        logger.error(f"Błąd podczas pobierania czasu serwera ByBit: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/bybit/market-data/<symbol>", methods=["GET"])
def get_bybit_market_data(symbol):
    """Endpoint zwracający dane rynkowe dla określonego symbolu z ByBit."""
    if not bybit_client:
        return jsonify({"error": "Klient ByBit nie jest zainicjalizowany"}), 500

    try:
        # Pobieranie świec (klines)
        klines = bybit_client.get_klines(symbol=symbol, interval="15", limit=10)

        # Pobieranie księgi zleceń
        order_book = bybit_client.get_order_book(symbol=symbol, limit=5)

        return jsonify({
            "klines": klines,
            "orderBook": order_book
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych rynkowych z ByBit: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/bybit/account-balance", methods=["GET"])
def get_bybit_account_balance():
    """Endpoint zwracający stan konta ByBit."""
    if not bybit_client:
        return jsonify({"error": "Klient ByBit nie jest zainicjalizowany"}), 500

    try:
        balance = bybit_client.get_account_balance()
        return jsonify(balance)
    except Exception as e:
        logger.error(f"Błąd podczas pobierania stanu konta ByBit: {e}")
        return jsonify({"error": str(e)}), 500


# Uruchomienie aplikacji
if __name__ == "__main__":
    # Tworzenie katalogu logs jeśli nie istnieje
    os.makedirs("logs", exist_ok=True)

    # Inicjalizacja systemu
    initialize_system()

    # Uruchomienie aplikacji - zawsze używamy 0.0.0.0 i portu 5000 w Replit
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)