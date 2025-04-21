#!/usr/bin/env python3
"""
dashboard_api.py - API servera dla dashboardu tradingowego

Ten moduł zapewnia API RESTowe dla dashboardu, obsługujące:
- Ładowanie i przetwarzanie danych rynkowych
- Generowanie sygnałów
- Zarządzanie modelami
- Wizualizację wyników

Integruje również autonomiczny system AI poprzez dashboard_api_auto.py.
"""
# Istniejące importy
import os
import sys
import json
import logging
import random
import traceback
import sqlite3
import jwt
import time
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, current_app
from flask_cors import CORS
from functools import wraps
from ai_models.sentiment_ai import SentimentAnalyzer
from ai_models.anomaly_detection import AnomalyDetector
from ai_models.model_recognition import ModelRecognizer
from ai_models.model_utils import get_model_performance

# Dodajemy brakujące importy
try:
    from ai_models.enhanced_backtester import EnhancedBacktester
    from ai_models.strategy_runner import StrategyBacktestRunner
except ImportError:
    EnhancedBacktester = None
    StrategyBacktestRunner = None

# Próba importu klas zarządzania ryzykiem
try:
    from portfolio.risk_manager import RiskManager
except ImportError:
    class RiskManager:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_risk_metrics_history(self, *args, **kwargs):
            return {}
        
        def update_risk_limits(self, *args, **kwargs):
            return False
        
        def calculate_position_size(self, *args, **kwargs):
            return {}

# Próba importu analizatora technicznego
try:
    from data.technical_analysis import TechnicalAnalyzer
except ImportError:
    class TechnicalAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        
        def run_backtest(self, *args, **kwargs):
            return {}
        
        def plot_backtest_results(self, *args, **kwargs):
            pass
        
        def get_custom_indicators(self, *args, **kwargs):
            return []
        
        def add_custom_indicator(self, *args, **kwargs):
            return False
        
        def optimize_strategy(self, *args, **kwargs):
            return {}

from werkzeug.security import generate_password_hash, check_password_hash
from data.database import verify_user, update_last_login, create_user
from data.portfolio_analytics import PortfolioAnalytics
from notifications.advanced_alerts import NotificationManager

# Ścieżka do bazy danych
DB_PATH = 'users.db'

# Dodanie funkcji pomocniczych
def get_jwt_identity():
    """Zwraca identyfikator użytkownika z tokenu JWT"""
    if hasattr(request, 'username'):
        return request.username
    return None

def get_active_alerts(user_id):
    """Pobiera aktywne alerty użytkownika"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
            SELECT id, type, symbol, condition, value, created_at, last_triggered, active
            FROM alerts
            WHERE user_id = ? AND active = 1
        ''', (user_id,))
        
        columns = ['id', 'type', 'symbol', 'condition', 'value', 'created_at', 'last_triggered', 'active']
        alerts = [dict(zip(columns, row)) for row in c.fetchall()]
        conn.close()
        return alerts
    except Exception as e:
        logging.error(f"Błąd podczas pobierania alertów: {e}")
        return []

def add_alert(user_id, alert_data):
    """Dodaje nowy alert dla użytkownika"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO alerts (user_id, type, symbol, condition, value, created_at, active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        ''', (
            user_id,
            alert_data['type'],
            alert_data['symbol'],
            alert_data['condition'],
            alert_data['value'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Błąd podczas dodawania alertu: {e}")
        return False

# Dodanie lokalnych bibliotek do ścieżki
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/dashboard_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Upewnij się, że potrzebne katalogi istnieją
for directory in ["logs", "data/cache", "static/img"]:
    os.makedirs(directory, exist_ok=True)

# Inicjalizacja Flask
app = Flask(__name__)
# Konfiguracja CORS - zezwól na wszystkie źródła
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type"]
    }
})

# Konfiguracja Flask
app.config.update(
    PROPAGATE_EXCEPTIONS=True,
    JSONIFY_PRETTYPRINT_REGULAR=False,
    JSON_SORT_KEYS=False,
    CORS_HEADERS='Content-Type'
)

# Dodanie konfiguracji JWT
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-here')  # W produkcji użyj bezpiecznego klucza
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)

def get_jwt_token(username: str) -> str:
    """Generuje JWT token dla użytkownika"""
    expires = datetime.utcnow() + JWT_ACCESS_TOKEN_EXPIRES
    return jwt.encode(
        {
            "username": username,
            "exp": expires
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

def jwt_required(f):
    """Dekorator wymagający ważnego tokenu JWT"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({"message": "Token is missing"}), 401
        
        if not token:
            return jsonify({"message": "Token is missing"}), 401

        try:
            decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            request.username = decoded["username"]
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated_function

# Obsługa błędów
@app.errorhandler(500)
def handle_500_error(error):
    logger.error(f"Internal Server Error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(error),
        'status': 500
    }), 500

# Konfiguracja retry mechanism dla błędów serwera
MAX_RETRIES = 3
RETRY_DELAY = 1  # sekund
RETRY_ERRORS = {502, 503, 504}  # kody błędów dla których próbujemy ponownie

@app.errorhandler(502)
def handle_502_error(error):
    """Rozszerzona obsługa błędu 502 Bad Gateway"""
    request_id = request.headers.get('X-Request-ID', 'unknown')
    error_details = {
        'error': 'Bad Gateway',
        'status': 502,
        'request_id': request_id,
        'message': 'Nie można połączyć się z jednym z komponentów systemu',
        'timestamp': datetime.now().isoformat(),
        'retry_after': RETRY_DELAY,
        'component': 'unknown'  # będzie aktualizowane w zależności od kontekstu
    }
    
    # Próba identyfikacji komponentu, który nie odpowiada
    try:
        current_route = request.endpoint
        if current_route:
            if 'portfolio' in current_route:
                error_details['component'] = 'Portfolio Manager'
            elif 'trading' in current_route:
                error_details['component'] = 'Trading Engine'
            elif 'ai' in current_route:
                error_details['component'] = 'AI Models'
            elif 'simulation' in current_route:
                error_details['component'] = 'Simulation Engine'
    except Exception as e:
        logger.error(f"Błąd podczas identyfikacji komponentu: {e}")

    # Szczegółowe logowanie
    logger.error(
        "Bad Gateway Error: %s, Request ID: %s, Component: %s",
        str(error),
        request_id,
        error_details['component']
    )

    # Dodaj informację o retry do nagłówków odpowiedzi
    response = jsonify(error_details)
    response.headers['Retry-After'] = str(RETRY_DELAY)
    
    return response, 502

def retry_on_server_error(f):
    """Dekorator do automatycznych prób ponownego połączenia dla określonych błędów"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        attempts = 0
        last_error = None
        
        while attempts < MAX_RETRIES:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                last_error = e
                if hasattr(e, 'code') and e.code in RETRY_ERRORS:
                    attempts += 1
                    if attempts < MAX_RETRIES:
                        logger.warning(
                            f"Próba {attempts}/{MAX_RETRIES} dla {request.endpoint} nie powiodła się. Ponowna próba za {RETRY_DELAY}s"
                        )
                        time.sleep(RETRY_DELAY)
                        continue
                raise
        
        logger.error(f"Wszystkie próby połączenia nie powiodły się: {last_error}")
        return handle_502_error(last_error)
    
    return wrapper

@app.errorhandler(404)
def handle_404_error(error):
    logger.error(f"Not Found Error: {error}")
    return jsonify({
        'error': 'Not Found',
        'message': 'Nie znaleziono żądanego zasobu',
        'status': 404
    }), 404

# Inicjalizacja modeli AI z obsługą błędów
try:
    sentiment_analyzer = SentimentAnalyzer()
    anomaly_detector = AnomalyDetector()
    model_recognizer = ModelRecognizer()
    logger.info("Zainicjalizowano wszystkie modele AI")
except Exception as e:
    logger.error(f"Błąd podczas inicjalizacji modeli AI: {e}")
    sentiment_analyzer = None
    anomaly_detector = None
    model_recognizer = None

# Symulowane dane dla dashboardu
SIMULATED_DATA = {
    "portfolio": {
        "balance": {"USDT": 10000, "BTC": 0.05, "ETH": 0.5},
        "positions": [
            {
                "symbol": "BTCUSDT", 
                "side": "LONG", 
                "size": 0.01, 
                "entry_price": 48500, 
                "mark_price": 50200, 
                "unrealized_pnl": 17,
                "roe": 3.5
            },
            {
                "symbol": "ETHUSDT", 
                "side": "SHORT", 
                "size": 0.2, 
                "entry_price": 3200, 
                "mark_price": 3150, 
                "unrealized_pnl": 10,
                "roe": 1.5
            }
        ],
        "equity": 10350,
        "available": 9600,
        "unrealized_pnl": 27
    },
    "trades": {
        "trades": [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "entry_price": 48500,
                "exit_price": 49000,
                "profit_loss": 5,
                "profit_loss_percent": 1.03,
                "status": "CLOSED",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "quantity": 0.1,
                "entry_price": 3250,
                "exit_price": 3150,
                "profit_loss": 10,
                "profit_loss_percent": 3.08,
                "status": "CLOSED",
                "timestamp": (datetime.now() - timedelta(hours=5)).isoformat()
            },
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.005,
                "entry_price": 47800,
                "status": "OPEN",
                "timestamp": (datetime.now() - timedelta(hours=8)).isoformat()
            }
        ]
    },
    "component-status": {
        "api_status": "online",
        "trading_status": "online"
    },
    "chart-data": {
        # Generuj przykładowe dane świecowe za ostatnie 24h z 1h interwałem
        "candles": [
            {
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "open": 50000 - i * 10 + random.randint(-100, 100),
                "high": 50100 - i * 10 + random.randint(-50, 150),
                "low": 49900 - i * 10 + random.randint(-150, 50),
                "close": 50000 - i * 10 + random.randint(-100, 100),
                "volume": 100 + random.randint(0, 50)
            }
            for i in range(24)
        ],
        "indicators": {
            "SMA_20": [
                {
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "value": 50000 - i * 8 + random.randint(-50, 50)
                }
                for i in range(24)
            ],
            "EMA_50": [
                {
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "value": 49800 - i * 7 + random.randint(-50, 50)
                }
                for i in range(24)
            ]
        }
    },
    "sentiment": {
        "value": 0.32,
        "analysis": "Umiarkowanie pozytywny",
        "sources": {
            "twitter": 0.45,
            "news": 0.28,
            "forum": 0.22
        }
    },
    "market-analyze": {
        "signal": "BUY",
        "confidence": 0.78,
        "indicators": {
            "RSI": 62.5,
            "MACD": 0.0025,
            "Bollinger": 0.85,
            "ADX": 28.3,
            "ATR": 0.0125,
            "Volume_Change": 0.35
        }
    },
    "ai-models-status": {
        "models": [
            {
                "name": "TrendPredictor",
                "type": "XGBoost",
                "status": "active",
                "accuracy": 78,
                "last_active": (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                "name": "PatternRecognizer",
                "type": "CNN",
                "status": "active",
                "accuracy": 81,
                "last_active": (datetime.now() - timedelta(minutes=15)).isoformat()
            },
            {
                "name": "SentimentAnalyzer",
                "type": "BERT",
                "status": "active",
                "accuracy": 75,
                "last_active": (datetime.now() - timedelta(minutes=10)).isoformat()
            },
            {
                "name": "AnomalyDetector",
                "type": "IsolationForest",
                "status": "active",
                "accuracy": 68,
                "last_active": (datetime.now() - timedelta(minutes=3)).isoformat()
            },
            {
                "name": "MarketRegime",
                "type": "HMM",
                "status": "inactive",
                "accuracy": 62,
                "last_active": (datetime.now() - timedelta(hours=2)).isoformat()
            }
        ]
    },
    "ai-learning-status": {
        "models_in_training": 1,
        "last_trained": (datetime.now() - timedelta(hours=3)).isoformat(),
        "best_accuracy": 85,
        "training_progress": 67
    },
    "ai-thoughts": {
        "thoughts": [
            "Wykryto silny trend wzrostowy na BTCUSDT w czasie 4h z prawdopodobieństwem 87%.",
            "Analiza sentymentu wskazuje na umiarkowany optymizm, ale wolumen transakcji jest poniżej średniej.",
            "Wzorce cenowe wskazują na potencjalną kontynuację trendu, zalecane ustawienie stop-loss na poziomie 48200.",
            "Wykryto potencjalny sygnał Wyckoff Accumulation na ETHUSDT, wczesna faza C."
        ]
    },
    "simulation-results": {
        "results": [
            {
                "id": "sim_btc_trend_20250415",
                "initial_capital": 10000,
                "final_capital": 10850,
                "profit": 850,
                "profit_percentage": 8.5,
                "max_drawdown": 2.3,
                "total_trades": 24,
                "winning_trades": 16,
                "losing_trades": 8,
                "win_rate": 0.67,
                "chart_path": "static/img/simulation_chart.png",
                "trades": [
                    {
                        "timestamp": "2025-04-14T12:00:00",
                        "action": "BUY",
                        "price": 49200,
                        "size": 0.02,
                        "capital": 10000
                    },
                    {
                        "timestamp": "2025-04-14T14:30:00",
                        "action": "SELL",
                        "price": 49600,
                        "size": 0.02,
                        "pnl": 8,
                        "capital": 10008
                    }
                ]
            },
            {
                "id": "sim_eth_breakout_20250415",
                "initial_capital": 10000,
                "final_capital": 10620,
                "profit": 620,
                "profit_percentage": 6.2,
                "max_drawdown": 1.8,
                "total_trades": 18,
                "winning_trades": 12,
                "losing_trades": 6,
                "win_rate": 0.67,
                "chart_path": "static/img/simulation_chart.png",
                "trades": []
            }
        ]
    },
    "logs": {
        "logs": [
            {
                "level": "INFO",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "message": "System uruchomiony pomyślnie",
                "details": None
            },
            {
                "level": "INFO",
                "timestamp": (datetime.now() - timedelta(minutes=4)).isoformat(),
                "message": "Połączono z API giełdy",
                "details": None
            },
            {
                "level": "WARNING",
                "timestamp": (datetime.now() - timedelta(minutes=3)).isoformat(),
                "message": "Wysoka latencja API",
                "details": "Opóźnienie przekracza 500ms"
            },
            {
                "level": "ERROR",
                "timestamp": (datetime.now() - timedelta(minutes=2)).isoformat(),
                "message": "Nie można przetworzyć danych rynkowych",
                "details": "Timeout during API call for SOLUSDT"
            }
        ]
    },
    "status": {
        "status": "online",
        "components": {
            "bybit_api": "online",
            "binance_api": "online",
            "ccxt": "online",
            "strategy_manager": "online",
            "model_recognizer": "online",
            "anomaly_detector": "online",
            "sentiment_analyzer": "offline"
        }
    }
}

# Dodanie importu dla obsługi handlu w czasie rzeczywistym
try:
    from python_libs.dashboard_realtime_connector import get_dashboard_connector
except ImportError:
    logging.error("Nie można zaimportować DashboardRealtimeConnector")
    get_dashboard_connector = None

# Endpoints API
@app.route('/api/portfolio', methods=['GET'])
@retry_on_server_error
@jwt_required
def get_portfolio():
    logger.info("Zapytanie o dane portfela")
    return jsonify(SIMULATED_DATA["portfolio"])

@app.route('/api/trades', methods=['GET'])
@jwt_required
def get_trades():
    logger.info("Zapytanie o historię transakcji")
    return jsonify(SIMULATED_DATA["trades"])

@app.route('/api/component-status', methods=['GET'])
@jwt_required
def get_component_status():
    """Endpoint API zwracający status wszystkich komponentów."""
    try:
        components = {
            "sentiment_analyzer": "online" if sentiment_analyzer.active else "offline",
            "anomaly_detector": "online" if hasattr(anomaly_detector, 'detect') else "offline",
            "model_recognizer": "online" if hasattr(model_recognizer, 'identify_model_type') else "offline",
            "trading_engine": "online",
            "data_processor": "online"
        }
        
        return jsonify({
            "status": "success",
            "components": components,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania statusu komponentów: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    symbol = request.args.get('symbol', 'BTCUSDT')
    timeframe = request.args.get('timeframe', '1h')
    logger.info(f"Zapytanie o dane wykresu dla {symbol} ({timeframe})")
    return jsonify(SIMULATED_DATA["chart-data"])

@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    logger.info("Zapytanie o dane sentymentu")
    return jsonify(SIMULATED_DATA["sentiment"])

@app.route('/api/market/analyze', methods=['GET'])
def get_market_analysis():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '1h')
    strategy = request.args.get('strategy', 'trend_following')
    logger.info(f"Zapytanie o analizę rynku dla {symbol} ({interval}, {strategy})")
    return jsonify(SIMULATED_DATA["market-analyze"])

@app.route('/api/ai-models-status', methods=['GET'])
def get_ai_models_status():
    logger.info("Zapytanie o status modeli AI")
    return jsonify(SIMULATED_DATA["ai-models-status"])

@app.route('/api/ai/learning-status', methods=['GET'])
def get_ai_learning_status():
    logger.info("Zapytanie o status uczenia AI")
    return jsonify(SIMULATED_DATA["ai-learning-status"])

@app.route('/api/ai/thoughts', methods=['GET'])
def get_ai_thoughts():
    logger.info("Zapytanie o przemyślenia AI")
    return jsonify(SIMULATED_DATA["ai-thoughts"])

@app.route('/api/simulation-results', methods=['GET'])
def get_simulation_results():
    logger.info("Zapytanie o wyniki symulacji")
    return jsonify(SIMULATED_DATA["simulation-results"])

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logger.info("Zapytanie o logi systemowe")
    return jsonify(SIMULATED_DATA["logs"])

@app.route('/api/status', methods=['GET'])
def get_status():
    logger.info("Zapytanie o status systemu")
    return jsonify(SIMULATED_DATA["status"])

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

# Dodanie endpointów autoryzacji
@app.route('/api/auth/login', methods=['POST'])
def login():
    """Endpoint logowania zwracający JWT token"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"message": "Missing username or password"}), 400
        
    if verify_user(username, password):
        update_last_login(username)  # Aktualizuj czas ostatniego logowania
        token = get_jwt_token(username)
        return jsonify({
            "token": token,
            "user": username,
            "expires_in": JWT_ACCESS_TOKEN_EXPIRES.total_seconds()
        })
    
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/api/auth/verify', methods=['GET'])
@jwt_required
def verify_token():
    """Endpoint weryfikacji tokenu JWT"""
    return jsonify({
        "valid": True,
        "username": request.username
    })

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Endpoint rejestracji nowego użytkownika"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not username or not password:
        return jsonify({"message": "Missing username or password"}), 400
        
    if create_user(username, password, email):
        token = get_jwt_token(username)
        return jsonify({
            "message": "User created successfully",
            "token": token,
            "user": username
        })
    
    return jsonify({"message": "Username already exists"}), 409

@app.route('/api/portfolio/history', methods=['GET'])
@jwt_required
def get_portfolio_history():
    """Zwraca historię zmian wartości portfela"""
    user_id = get_jwt_identity()
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Pobierz ostatnie 100 zapisów historii portfela
        c.execute('''
            SELECT timestamp, total_equity, available_balance, 
                   used_margin, unrealized_pnl, realized_pnl
            FROM portfolio_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''', (user_id,))
        
        columns = ['timestamp', 'total_equity', 'available_balance', 
                  'used_margin', 'unrealized_pnl', 'realized_pnl']
        history = [dict(zip(columns, row)) for row in c.fetchall()]
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania historii portfela: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        conn.close()

@app.route('/api/trades/history', methods=['GET'])
@jwt_required
def get_trade_history():
    """Zwraca historię transakcji"""
    user_id = get_jwt_identity()
    
    # Parametry filtrowania
    symbol = request.args.get('symbol')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        query = '''
            SELECT * FROM trades 
            WHERE user_id = ?
        '''
        params = [user_id]
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
            
        if start_date:
            query += ' AND executed_at >= ?'
            params.append(start_date)
            
        if end_date:
            query += ' AND executed_at <= ?'
            params.append(end_date)
            
        query += ' ORDER BY executed_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        c.execute(query, params)
        
        columns = [description[0] for description in c.description]
        trades = [dict(zip(columns, row)) for row in c.fetchall()]
        
        # Pobierz statystyki
        c.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(realized_pnl) as total_pnl,
                SUM(commission) as total_commission
            FROM trades 
            WHERE user_id = ?
        ''', (user_id,))
        
        stats = dict(zip(['total_trades', 'winning_trades', 'total_pnl', 'total_commission'], 
                        c.fetchone()))
        
        if stats['total_trades'] > 0:
            stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
        else:
            stats['win_rate'] = 0
            
        return jsonify({
            'success': True,
            'trades': trades,
            'stats': stats,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': stats['total_trades']
            }
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania historii transakcji: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        conn.close()

@app.route('/api/alerts', methods=['GET', 'POST'])
@jwt_required
def manage_alerts():
    """Zarządzanie alertami cenowymi"""
    user_id = get_jwt_identity()
    
    if request.method == 'GET':
        try:
            alerts = get_active_alerts(user_id)
            return jsonify({
                'success': True,
                'alerts': alerts
            })
        except Exception as e:
            logger.error(f"Błąd podczas pobierania alertów: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    elif request.method == 'POST':
        data = request.get_json()
        
        required_fields = ['type', 'symbol', 'condition', 'value']
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': 'Brak wymaganych pól'
            }), 400
            
        try:
            success = add_alert(user_id, data)
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Alert został dodany'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Nie udało się dodać alertu'
                }), 500
        except Exception as e:
            logger.error(f"Błąd podczas dodawania alertu: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

@app.route('/api/alerts/<int:alert_id>', methods=['DELETE'])
@jwt_required
def delete_alert(alert_id):
    """Usuwa alert"""
    user_id = get_jwt_identity()
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute('''
            DELETE FROM alerts 
            WHERE id = ? AND user_id = ?
        ''', (alert_id, user_id))
        
        conn.commit()
        
        if c.rowcount > 0:
            return jsonify({
                'success': True,
                'message': 'Alert został usunięty'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Alert nie został znaleziony'
            }), 404
            
    except Exception as e:
        logger.error(f"Błąd podczas usuwania alertu: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        conn.close()

@app.route('/api/risk/metrics', methods=['GET'])
@jwt_required
def get_risk_metrics():
    """Zwraca metryki ryzyka dla portfela"""
    try:
        user_id = get_jwt_identity()
        days = request.args.get('days', 30, type=int)
        
        risk_manager = RiskManager()
        metrics = risk_manager.get_risk_metrics_history(user_id, days)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania metryk ryzyka: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/risk/limits', methods=['GET', 'POST'])
@jwt_required
def manage_risk_limits():
    """Zarządza limitami ryzyka"""
    try:
        user_id = get_jwt_identity()
        risk_manager = RiskManager()
        
        if request.method == 'GET':
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT * FROM risk_limits WHERE user_id = ?', (user_id,))
            limits = dict(zip([col[0] for col in c.description], c.fetchone() or []))
            conn.close()
            
            return jsonify({
                'success': True,
                'limits': limits
            })
            
        elif request.method == 'POST':
            data = request.get_json()
            success = risk_manager.update_risk_limits(user_id, data)
            
            return jsonify({
                'success': success,
                'message': 'Limity ryzyka zaktualizowane' if success else 'Błąd aktualizacji limitów'
            })
            
    except Exception as e:
        logger.error(f"Błąd podczas zarządzania limitami ryzyka: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/indicators/custom', methods=['GET', 'POST'])
@jwt_required
def manage_custom_indicators():
    """Zarządza własnymi wskaźnikami technicznymi"""
    try:
        user_id = get_jwt_identity()
        analyzer = TechnicalAnalyzer()
        
        if request.method == 'GET':
            indicators = analyzer.get_custom_indicators(user_id)
            return jsonify({
                'success': True,
                'indicators': indicators
            })
            
        elif request.method == 'POST':
            data = request.get_json()
            success = analyzer.add_custom_indicator(user_id, data)
            
            return jsonify({
                'success': success,
                'message': 'Wskaźnik dodany' if success else 'Błąd dodawania wskaźnika'
            })
            
    except Exception as e:
        logger.error(f"Błąd podczas zarządzania wskaźnikami: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest/run', methods=['POST'])
@jwt_required
def run_backtest():
    """Uruchamia backtest strategii"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Pobierz dane historyczne
        symbol = data['symbol']
        timeframe = data['timeframe']
        start_date = data.get('start_date', (datetime.now() - timedelta(days=30)).isoformat())
        end_date = data.get('end_date', datetime.now().isoformat())
        
        # Pobierz dane historyczne z bazy lub API
        historical_data = pd.DataFrame(SIMULATED_DATA["chart-data"]["candles"])
        
        # Uruchom backtest
        analyzer = TechnicalAnalyzer()
        results = analyzer.run_backtest(
            data['strategy_config'],
            historical_data,
            data.get('initial_balance', 10000.0)
        )
        
        if not results:
            return jsonify({
                'success': False,
                'error': 'Błąd podczas backtestingu'
            }), 500
            
        # Zapisz wyniki w bazie danych
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO backtest_results (
            user_id, strategy_id, start_date, end_date,
            initial_balance, final_balance, total_trades,
            winning_trades, losing_trades, sharpe_ratio,
            sortino_ratio, max_drawdown, parameters
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            data['strategy_config'].get('id', 'unknown'),
            start_date,
            end_date,
            data.get('initial_balance', 10000.0),
            results['equity_curve'][-1],
            results['metrics']['total_trades'],
            results['metrics']['winning_trades'],
            results['metrics']['total_trades'] - results['metrics']['winning_trades'],
            results['metrics']['sharpe_ratio'],
            results['metrics'].get('sortino_ratio', 0),
            results['metrics']['max_drawdown'],
            json.dumps(data['strategy_config'])
        ))
        
        conn.commit()
        conn.close()
        
        # Wygeneruj wykresy
        plot_path = f'static/img/backtest_{user_id}_{int(datetime.now().timestamp())}.png'
        analyzer.plot_backtest_results(results, f'static/{plot_path}')
        
        return jsonify({
            'success': True,
            'results': results,
            'plot_url': plot_path
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas backtestingu: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest/optimize', methods=['POST'])
@jwt_required
def optimize_strategy():
    """Optymalizuje parametry strategii"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Pobierz dane historyczne
        symbol = data['symbol']
        timeframe = data['timeframe']
        historical_data = pd.DataFrame(SIMULATED_DATA["chart-data"]["candles"])
        
        # Uruchom optymalizację
        analyzer = TechnicalAnalyzer()
        results = analyzer.optimize_strategy(
            data['strategy_config'],
            historical_data,
            data['param_grid']
        )
        
        if not results:
            return jsonify({
                'success': False,
                'error': 'Błąd podczas optymalizacji'
            }), 500
            
        return jsonify({
            'success': True,
            'best_parameters': results['best_result']['parameters'],
            'best_metrics': results['best_result']['metrics'],
            'all_results': results['all_results']
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas optymalizacji strategii: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/risk/position-calculator', methods=['POST'])
@jwt_required
def calculate_position():
    """Kalkulator wielkości pozycji na podstawie ryzyka"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        risk_manager = RiskManager()
        result = risk_manager.calculate_position_size(
            symbol=data['symbol'],
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            risk_per_trade=data.get('risk_per_trade'),
            current_capital=data.get('current_capital')
        )
        
        return jsonify({
            'success': True,
            'calculation': result
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas obliczania wielkości pozycji: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/analytics/allocation', methods=['GET'])
@jwt_required
def get_allocation_history():
    """Zwraca historię alokacji portfela"""
    try:
        user_id = get_jwt_identity()
        days = request.args.get('days', 30, type=int)
        
        analytics = PortfolioAnalytics(user_id)
        allocation_history = analytics.get_allocation_history(days)
        
        return jsonify({
            'success': True,
            'data': allocation_history.to_dict() if not allocation_history.empty else {}
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania historii alokacji: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/analytics/diversification', methods=['GET'])
@jwt_required
def get_diversification_metrics():
    """Zwraca metryki dywersyfikacji portfela"""
    try:
        user_id = get_jwt_identity()
        analytics = PortfolioAnalytics(user_id)
        metrics = analytics.calculate_diversification_metrics()
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Błąd podczas obliczania metryk dywersyfikacji: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/analytics/risk', methods=['GET'])
@jwt_required
def get_risk_exposure():
    """Zwraca analizę ekspozycji na ryzyko"""
    try:
        user_id = get_jwt_identity()
        analytics = PortfolioAnalytics(user_id)
        risk_metrics = analytics.analyze_risk_exposure()
        
        return jsonify({
            'success': True,
            'risk_metrics': risk_metrics
        })
    except Exception as e:
        logger.error(f"Błąd podczas analizy ryzyka: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/analytics/turnover', methods=['GET'])
@jwt_required
def get_turnover_metrics():
    """Zwraca metryki rotacji kapitału"""
    try:
        user_id = get_jwt_identity()
        period_days = request.args.get('days', 30, type=int)
        
        analytics = PortfolioAnalytics(user_id)
        turnover_metrics = analytics.calculate_turnover_metrics(period_days)
        
        return jsonify({
            'success': True,
            'metrics': turnover_metrics
        })
    except Exception as e:
        logger.error(f"Błąd podczas obliczania metryk rotacji: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/optimize', methods=['POST'])
@jwt_required
def optimize_portfolio():
    """Optymalizuje alokację portfela"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        risk_tolerance = data.get('risk_tolerance', 0.02)
        
        analytics = PortfolioAnalytics(user_id)
        optimal_allocation = analytics.optimize_portfolio(risk_tolerance)
        
        return jsonify({
            'success': True,
            'allocation': optimal_allocation
        })
    except Exception as e:
        logger.error(f"Błąd podczas optymalizacji portfela: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alerts/pattern', methods=['POST'])
@jwt_required
def add_pattern_alert():
    """Dodaje nowy alert wzorca cenowego"""
    try:
        user_id = get_jwt_identity()
        alert_data = request.get_json()
        
        notification_manager = NotificationManager(user_id)
        success = notification_manager.add_pattern_alert(alert_data)
        
        return jsonify({
            'success': success,
            'message': 'Alert wzorca został dodany' if success else 'Błąd podczas dodawania alertu'
        })
    except Exception as e:
        logger.error(f"Błąd podczas dodawania alertu wzorca: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alerts/anomaly', methods=['POST'])
@jwt_required
def add_anomaly_alert():
    """Dodaje nowy alert anomalii AI"""
    try:
        user_id = get_jwt_identity()
        alert_data = request.get_json()
        
        notification_manager = NotificationManager(user_id)
        success = notification_manager.add_ai_anomaly_alert(alert_data)
        
        return jsonify({
            'success': success,
            'message': 'Alert anomalii został dodany' if success else 'Błąd podczas dodawania alertu'
        })
    except Exception as e:
        logger.error(f"Błąd podczas dodawania alertu anomalii: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications/channels', methods=['POST'])
@jwt_required
def configure_notification_channel():
    """Konfiguruje nowy kanał powiadomień"""
    try:
        user_id = get_jwt_identity()
        channel_data = request.get_json()
        
        notification_manager = NotificationManager(user_id)
        success = notification_manager.configure_notification_channel(channel_data)
        
        return jsonify({
            'success': success,
            'message': 'Kanał powiadomień został skonfigurowany' if success else 'Błąd podczas konfiguracji kanału'
        })
    except Exception as e:
        logger.error(f"Błąd podczas konfiguracji kanału powiadomień: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alerts/active', methods=['GET'])
@jwt_required
def get_active_alerts():
    """Pobiera wszystkie aktywne alerty"""
    try:
        user_id = get_jwt_identity()
        notification_manager = NotificationManager(user_id)
        alerts = notification_manager.get_active_alerts()
        
        return jsonify({
            'success': True,
            'alerts': alerts
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania aktywnych alertów: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications/settings', methods=['GET', 'POST'])
@jwt_required
def manage_notification_settings():
    """Zarządza ustawieniami powiadomień użytkownika"""
    try:
        user_id = get_jwt_identity()
        notification_manager = NotificationManager()
        
        if request.method == 'GET':
            settings = notification_manager.get_channel_settings(user_id)
            return jsonify({
                'success': True,
                'settings': settings
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            channel_type = data.get('channel_type')
            settings = data.get('settings')
            
            success = notification_manager.update_channel_settings(
                user_id, channel_type, settings
            )
            
            return jsonify({
                'success': success,
                'message': 'Ustawienia zaktualizowane' if success else 'Błąd aktualizacji'
            })
            
    except Exception as e:
        logger.error(f"Błąd podczas zarządzania ustawieniami powiadomień: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications/history', methods=['GET'])
@jwt_required
def get_notifications_history():
    """Pobiera historię powiadomień użytkownika"""
    try:
        user_id = get_jwt_identity()
        limit = request.args.get('limit', 50, type=int)
        
        notification_manager = NotificationManager()
        history = notification_manager.get_notification_history(user_id, limit)
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania historii powiadomień: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications/test', methods=['POST'])
@jwt_required
def test_notification():
    """Wysyła testowe powiadomienie przez wybrany kanał"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        channel_type = data.get('channel_type')
        
        notification_manager = NotificationManager()
        message = f"Test powiadomienia przez kanał {channel_type} - {datetime.now()}"
        
        success = notification_manager.send_notification(
            user_id=user_id,
            notification_type='test',
            message=message,
            priority=1
        )
        
        return jsonify({
            'success': success,
            'message': 'Powiadomienie testowe wysłane' if success else 'Błąd wysyłania'
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas testu powiadomienia: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Nowe endpointy dla rozszerzonego backtestingu
@app.route('/api/backtest/enhanced/run', methods=['POST'])
@jwt_required
def run_enhanced_backtest():
    """Uruchamia rozszerzony backtest z dodatkową analizą"""
    try:
        data = request.get_json()
        user_id = request.username
        
        # Inicjalizacja backtestera
        backtester = EnhancedBacktester(
            initial_capital=data.get('initial_capital', 10000.0),
            commission=data.get('commission', 0.001),
            spread=data.get('spread', 0.0005),
            slippage=data.get('slippage', 0.0005),
            risk_free_rate=data.get('risk_free_rate', 0.02),
            position_sizing_method=data.get('position_sizing', "volatility")
        )
        
        # Pobieranie danych historycznych
        historical_data = pd.DataFrame(data['historical_data'])
        
        # Uruchomienie backtestów dla wybranych strategii
        runner = StrategyBacktestRunner(
            initial_capital=data.get('initial_capital', 10000.0),
            position_sizing_method=data.get('position_sizing', "volatility")
        )
        
        results = runner.run_strategy_comparison(
            historical_data,
            strategies_to_test=data['strategies'],
            params_dict=data.get('strategy_params', {})
        )
        
        # Generowanie wykresów
        plot_path = f'static/img/backtest_{user_id}_{int(datetime.now().timestamp())}.png'
        runner.plot_strategy_comparison(results, save_path=plot_path)
        
        # Przygotowanie szczegółowego raportu
        detailed_results = {}
        for strategy_name, strategy_results in results.items():
            # Dodanie analizy Monte Carlo
            mc_results = backtester.run_monte_carlo_simulation(
                pd.Series(strategy_results['equity_curve']).pct_change()
            )
            
            # Dodanie wyników walk-forward analysis
            walk_forward = strategy_results['walk_forward']
            
            detailed_results[strategy_name] = {
                'performance': strategy_results['performance'],
                'monte_carlo': mc_results,
                'walk_forward': walk_forward,
                'trades': strategy_results['trades']
            }
        
        return jsonify({
            'success': True,
            'results': detailed_results,
            'plot_url': plot_path
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas rozszerzonego backtestingu: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest/enhanced/compare', methods=['POST'])
@jwt_required
def compare_strategies():
    """Porównuje wyniki różnych strategii"""
    try:
        data = request.get_json()
        user_id = request.username
        
        runner = StrategyBacktestRunner(
            initial_capital=data.get('initial_capital', 10000.0),
            position_sizing_method=data.get('position_sizing', "volatility")
        )
        
        # Porównanie strategii
        comparison_results = runner.run_strategy_comparison(
            pd.DataFrame(data['historical_data']),
            strategies_to_test=data['strategies'],
            params_dict=data.get('strategy_params', {})
        )
        
        # Generowanie wykresów porównawczych
        plot_path = f'static/img/strategy_comparison_{user_id}_{int(datetime.now().timestamp())}.png'
        runner.plot_strategy_comparison(comparison_results, save_path=plot_path)
        
        return jsonify({
            'success': True,
            'comparison': comparison_results,
            'plot_url': plot_path
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas porównywania strategii: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest/enhanced/monte-carlo', methods=['POST'])
@jwt_required
def run_monte_carlo():
    """Wykonuje symulację Monte Carlo dla wybranej strategii"""
    try:
        data = request.get_json()
        
        backtester = EnhancedBacktester(
            initial_capital=data.get('initial_capital', 10000.0),
            commission=data.get('commission', 0.001),
            spread=data.get('spread', 0.0005),
            slippage=data.get('slippage', 0.0005)
        )
        
        # Przeprowadzenie symulacji Monte Carlo
        returns = pd.Series(data['returns'])
        mc_results = backtester.run_monte_carlo_simulation(
            returns,
            n_simulations=data.get('n_simulations', 1000),
            n_days=data.get('n_days', 252)
        )
        
        return jsonify({
            'success': True,
            'monte_carlo_results': mc_results
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas symulacji Monte Carlo: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest/enhanced/walk-forward', methods=['POST'])
@jwt_required
def run_walk_forward():
    """Wykonuje analizę walk-forward dla wybranej strategii"""
    try:
        data = request.get_json()
        
        backtester = EnhancedBacktester(
            initial_capital=data.get('initial_capital', 10000.0),
            commission=data.get('commission', 0.001),
            spread=data.get('spread', 0.0005),
            slippage=data.get('slippage', 0.0005)
        )
        
        # Wykonanie analizy walk-forward
        historical_data = pd.DataFrame(data['historical_data'])
        
        def strategy_wrapper(data):
            runner = StrategyBacktestRunner()
            strategy_func = runner.strategies[data['strategy']]
            return strategy_func(data, **data.get('params', {}))
        
        results = backtester.run_walk_forward_analysis(
            historical_data,
            strategy_wrapper,
            train_ratio=data.get('train_ratio', 0.7),
            n_splits=data.get('n_splits', 5)
        )
        
        return jsonify({
            'success': True,
            'walk_forward_results': results
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy walk-forward: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Dodanie nowych endpointów dla handlu w czasie rzeczywistym
@app.route('/api/realtime/status', methods=['GET'])
@jwt_required
def get_realtime_status():
    """Zwraca status połączenia z API w czasie rzeczywistym"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        connector = get_dashboard_connector()
        dashboard_data = connector.get_dashboard_data()
        
        return jsonify({
            'success': True,
            'status': dashboard_data['status'],
            'error': dashboard_data['error'],
            'stats': dashboard_data.get('stats', {})
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statusu tradingu w czasie rzeczywistym: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/realtime/connect', methods=['POST'])
@jwt_required
def connect_realtime():
    """Inicjalizuje połączenie z API ByBit w czasie rzeczywistym"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        # Pobierz parametry połączenia z żądania
        data = request.get_json()
        
        connector = get_dashboard_connector()
        success = connector.initialize_connector(
            symbol=data.get('symbol', 'BTCUSDT'),
            timeframe=data.get('timeframe', '15m'),
            initial_capital=data.get('initial_capital', 10000.0),
            use_testnet=data.get('use_testnet', True),
            api_key=data.get('api_key'),
            api_secret=data.get('api_secret')
        )
        
        if success:
            # Uruchom aktualizację danych w tle
            connector.start_updating()
            
            return jsonify({
                'success': True,
                'status': connector.get_dashboard_data()['status'],
                'message': 'Połączono z API ByBit'
            })
        else:
            return jsonify({
                'success': False,
                'error': connector.get_dashboard_data()['error'] or 'Nie udało się połączyć z API ByBit'
            }), 500
            
    except Exception as e:
        logger.error(f"Błąd podczas łączenia z API w czasie rzeczywistym: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/realtime/disconnect', methods=['POST'])
@jwt_required
def disconnect_realtime():
    """Rozłącza połączenie z API w czasie rzeczywistym"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        connector = get_dashboard_connector()
        success = connector.disconnect()
        
        return jsonify({
            'success': success,
            'message': 'Rozłączono z API ByBit',
            'status': connector.get_dashboard_data()['status']
        })
    except Exception as e:
        logger.error(f"Błąd podczas rozłączania z API w czasie rzeczywistym: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/realtime/data', methods=['GET'])
@jwt_required
def get_realtime_data():
    """Pobiera aktualne dane z tradingu w czasie rzeczywistym"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        connector = get_dashboard_connector()
        dashboard_data = connector.get_dashboard_data()
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych z tradingu w czasie rzeczywistym: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/realtime/order-book/<symbol>', methods=['GET'])
@jwt_required
def get_realtime_order_book(symbol):
    """Pobiera aktualną księgę zleceń dla wybranego symbolu"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        connector = get_dashboard_connector()
        order_book = connector.get_order_book_snapshot(symbol)
        
        return jsonify({
            'success': True,
            'order_book': order_book
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania księgi zleceń dla {symbol}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/realtime/watched-symbols', methods=['GET', 'POST', 'DELETE'])
@jwt_required
def manage_watched_symbols():
    """Zarządzanie obserwowanymi symbolami w tradingu w czasie rzeczywistym"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        connector = get_dashboard_connector()
        
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'symbols': connector.watched_symbols
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            symbols = data.get('symbols', [])
            
            if isinstance(symbols, list):
                result = connector.set_watched_symbols(symbols)
                return jsonify({
                    'success': result,
                    'symbols': connector.watched_symbols
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Nieprawidłowy format danych'
                }), 400
        
        elif request.method == 'DELETE':
            data = request.get_json()
            symbol = data.get('symbol', '')
            
            if symbol:
                result = connector.remove_watched_symbol(symbol)
                return jsonify({
                    'success': result,
                    'symbols': connector.watched_symbols
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Nie podano symbolu'
                }), 400
                
    except Exception as e:
        logger.error(f"Błąd podczas zarządzania obserwowanymi symbolami: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/realtime/trade', methods=['POST'])
@jwt_required
def execute_realtime_trade():
    """Wykonuje operację handlową w czasie rzeczywistym"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        data = request.get_json()
        action = data.get('action')
        position_size = data.get('position_size')
        
        if not action:
            return jsonify({
                'success': False,
                'error': 'Nie podano akcji handlowej'
            }), 400
            
        connector = get_dashboard_connector()
        result = connector.execute_trade(action, position_size)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'result': result,
                'message': f'Wykonano operację: {action}'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Błąd podczas wykonywania operacji handlowej'),
                'result': result
            }), 500
            
    except Exception as e:
        logger.error(f"Błąd podczas wykonywania operacji handlowej: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/realtime/reset', methods=['POST'])
@jwt_required
def reset_realtime():
    """Resetuje stan konektora w czasie rzeczywistym"""
    try:
        if get_dashboard_connector is None:
            return jsonify({
                'success': False,
                'error': 'Moduł tradingu w czasie rzeczywistym nie jest dostępny'
            }), 500
        
        connector = get_dashboard_connector()
        success = connector.reset()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Zresetowano stan konektora',
                'status': connector.get_dashboard_data()['status']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Nie udało się zresetować konektora',
                'status': connector.get_dashboard_data()['status']
            }), 500
            
    except Exception as e:
        logger.error(f"Błąd podczas resetowania konektora: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Brakujące endpointy API

@app.route('/api/portfolio/allocation', methods=['GET', 'POST'])
@jwt_required
def manage_portfolio_allocation():
    """
    Pobiera lub aktualizuje alokację portfela.
    GET: Zwraca aktualną alokację portfela
    POST: Aktualizuje alokację portfela
    """
    user_id = get_jwt_identity()
    
    if request.method == 'GET':
        try:
            # Pobierz dane alokacji z bazy danych
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            c.execute('''
                SELECT asset, percentage
                FROM portfolio_allocation
                WHERE user_id = ?
            ''', (user_id,))
            
            allocation = {}
            for row in c.fetchall():
                asset, percentage = row
                allocation[asset] = float(percentage)
            
            conn.close()
            
            # Jeśli nie ma zapisanej alokacji, zwracamy domyślną
            if not allocation:
                allocation = {
                    "BTC": 40.0,
                    "ETH": 30.0,
                    "SOL": 10.0,
                    "ADA": 5.0,
                    "USDT": 15.0
                }
            
            return jsonify({
                "success": True,
                "allocation": allocation
            })
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania alokacji portfela: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if not isinstance(data.get('allocation'), dict):
                return jsonify({
                    "success": False,
                    "error": "Nieprawidłowy format danych alokacji"
                }), 400
            
            allocation = data['allocation']
            
            # Sprawdź, czy suma wartości jest równa 100%
            total = sum(allocation.values())
            if abs(total - 100.0) > 0.01:
                return jsonify({
                    "success": False,
                    "error": f"Suma alokacji musi wynosić 100% (aktualna suma: {total}%)"
                }), 400
            
            # Zapisz dane alokacji do bazy danych
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # Usuń istniejącą alokację
            c.execute('DELETE FROM portfolio_allocation WHERE user_id = ?', (user_id,))
            
            # Dodaj nową alokację
            for asset, percentage in allocation.items():
                c.execute('''
                    INSERT INTO portfolio_allocation (user_id, asset, percentage)
                    VALUES (?, ?, ?)
                ''', (user_id, asset, percentage))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                "success": True,
                "message": "Alokacja portfela została zaktualizowana"
            })
            
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji alokacji portfela: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

@app.route('/api/portfolio/correlation', methods=['GET'])
@jwt_required
def get_portfolio_correlation():
    """Zwraca macierz korelacji aktywów w portfelu."""
    user_id = get_jwt_identity()
    
    try:
        analytics = PortfolioAnalytics(user_id)
        correlation_data = analytics.calculate_asset_correlation()
        
        # Znajdź pary aktywów o wysokiej korelacji (powyżej 0.7)
        high_correlation_pairs = []
        
        for i, symbol1 in enumerate(correlation_data.index):
            for j, symbol2 in enumerate(correlation_data.columns):
                if i < j and abs(correlation_data.loc[symbol1, symbol2]) > 0.7:
                    high_correlation_pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation_data.loc[symbol1, symbol2]
                    })
        
        return jsonify({
            "success": True,
            "correlation_matrix": correlation_data.to_dict(),
            "high_correlation_pairs": high_correlation_pairs
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas obliczania korelacji portfela: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/trading/statistics', methods=['GET'])
@jwt_required
def get_trading_statistics():
    """Zwraca statystyki tradingowe dla wybranego okresu."""
    user_id = get_jwt_identity()
    days = int(request.args.get('days', 30))
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Ustal datę graniczną
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Pobierz transakcje z wybranego okresu
        c.execute('''
            SELECT symbol, side, entry_price, exit_price, quantity, 
                   realized_pnl, executed_at, closed_at, duration_hours
            FROM trades
            WHERE user_id = ? AND executed_at >= ?
            ORDER BY executed_at DESC
        ''', (user_id, cutoff_date))
        
        trades = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
        
        # Oblicz statystyki
        total_trades = len(trades)
        
        if total_trades == 0:
            # Jeśli nie ma transakcji, zwróć domyślne wartości
            return jsonify({
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'avg_holding_time_hours': 0,
                'largest_profit': 0,
                'largest_loss': 0,
                'profit_factor': 0,
                'best_trades': [],
                'worst_trades': []
            })
        
        # Oblicz podstawowe statystyki
        winning_trades = [t for t in trades if t['realized_pnl'] is not None and t['realized_pnl'] > 0]
        losing_trades = [t for t in trades if t['realized_pnl'] is not None and t['realized_pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Oblicz średnie zyski i straty
        avg_profit = sum(t['realized_pnl'] for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(t['realized_pnl'] for t in losing_trades) / loss_count if loss_count > 0 else 0
        
        # Oblicz średni czas trzymania
        holding_times = [t['duration_hours'] for t in trades if t['duration_hours'] is not None]
        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
        
        # Znajdź największe zyski i straty
        sorted_by_pnl = sorted(trades, key=lambda x: x['realized_pnl'] if x['realized_pnl'] is not None else 0)
        best_trades = sorted_by_pnl[-5:] if len(sorted_by_pnl) >= 5 else sorted_by_pnl
        worst_trades = sorted_by_pnl[:5] if len(sorted_by_pnl) >= 5 else sorted_by_pnl
        
        # Oblicz profit factor
        total_profit = sum(t['realized_pnl'] for t in winning_trades)
        total_loss = abs(sum(t['realized_pnl'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Przygotuj dane do zwrócenia
        largest_profit = max((t['realized_pnl'] for t in trades if t['realized_pnl'] is not None), default=0)
        largest_loss = min((t['realized_pnl'] for t in trades if t['realized_pnl'] is not None), default=0)
        
        best_trades_formatted = [
            {
                'symbol': t['symbol'],
                'pnl': t['realized_pnl'],
                'date': t['executed_at']
            }
            for t in reversed(best_trades[-3:])
        ]
        
        worst_trades_formatted = [
            {
                'symbol': t['symbol'],
                'pnl': t['realized_pnl'],
                'date': t['executed_at']
            }
            for t in worst_trades[:3]
        ]
        
        conn.close()
        
        return jsonify({
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_holding_time_hours': avg_holding_time,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'best_trades': best_trades_formatted,
            'worst_trades': worst_trades_formatted
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statystyk tradingowych: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/portfolio/optimal-allocation', methods=['GET'])
@jwt_required
def get_optimal_allocation():
    """Zwraca optymalną alokację portfela na podstawie profilu ryzyka."""
    user_id = get_jwt_identity()
    risk_profile = request.args.get('risk_profile', 'umiarkowany')
    
    try:
        analytics = PortfolioAnalytics(user_id)
        optimal_allocation = analytics.calculate_optimal_allocation(risk_profile)
        
        return jsonify({
            "success": True,
            "allocation": optimal_allocation,
            "risk_profile": risk_profile
        })
        
    except Exception as e:
        logger.error(f"Błąd podczas obliczania optymalnej alokacji portfela: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Główna trasa dla testowania API
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "API działa poprawnie",
        "endpoints": [
            "/api/portfolio",
            "/api/trades",
            "/api/component-status",
            "/api/chart-data",
            "/api/sentiment",
            "/api/market/analyze",
            "/api/ai-models-status",
            "/api/ai/learning-status",
            "/api/ai/thoughts",
            "/api/simulation-results",
            "/api/logs",
            "/api/status",
            "/api/health"
        ]
    })

# Import autonomicznego API
from dashboard_api_auto import auto_api

# Rejestracja Blueprint autonomicznego API
app.register_blueprint(auto_api)

@app.route('/api/v1/status', methods=['GET'])
def api_status():
    """Zwraca status API."""
    return jsonify({
        'status': 'online',
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'features': [
            'data_loading',
            'signal_generation',
            'model_management',
            'autonomous_trading'  # Dodano nową funkcję
        ]
    })

# Importy modułów autonomicznych
try:
    from ai_models.autonomous.autonomous_trading import AutonomousTrading
    from ai_models.autonomous.meta_agent import MetaAgent
    from ai_models.autonomous.risk_manager import AutonomousRiskManager
    autonomous_system_available = True
except ImportError as e:
    logger.warning(f"Nie można zaimportować modułów autonomicznego systemu: {e}")
    autonomous_system_available = False

# Inicjalizacja autonomicznego systemu
autonomous_trading_system = None
if autonomous_system_available:
    try:
        autonomous_trading_system = AutonomousTrading()
        autonomous_trading_system.initialize()
        logger.info("Zainicjalizowano autonomiczny system handlowy")
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji autonomicznego systemu handlowego: {e}")
        logger.error(traceback.format_exc())

# Endpointy API dla systemu autonomicznego
@app.route('/api/autonomous/status', methods=['GET'])
@jwt_required
def get_autonomous_status():
    """Endpoint zwracający status autonomicznego systemu."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "autonomous_mode": False,
                "models": {},
                "error": "Autonomiczny system handlowy nie jest dostępny"
            })
        
        status = autonomous_trading_system.get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statusu autonomicznego systemu: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "autonomous_mode": False,
            "error": str(e)
        })

@app.route('/api/autonomous/decisions', methods=['GET'])
@jwt_required
def get_autonomous_decisions():
    """Endpoint zwracający ostatnie decyzje autonomicznego systemu."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "decisions": [],
                "error": "Autonomiczny system handlowy nie jest dostępny"
            })
        
        meta_agent = autonomous_trading_system.meta_agent
        if not meta_agent:
            return jsonify({
                "decisions": [],
                "error": "Meta-Agent nie jest dostępny"
            })
        
        # Pobierz ostatnie decyzje
        recent_decisions = meta_agent.recent_decisions[-10:] if hasattr(meta_agent, 'recent_decisions') else []
        
        # Transformuj decyzje do odpowiedniego formatu
        formatted_decisions = []
        for decision in recent_decisions:
            formatted_decision = {
                "decision": decision.get('decision', 'UNKNOWN'),
                "confidence": decision.get('confidence', 0.0),
                "timestamp": decision.get('timestamp', ''),
                "symbol": decision.get('symbol', 'UNKNOWN'),
                "explanation": "Brak szczegółowych wyjaśnień"
            }
            
            # Dodaj wyjaśnienie, jeśli istnieje
            if 'details' in decision and 'explanation' in decision['details']:
                formatted_decision["explanation"] = decision['details']['explanation']
                
            formatted_decisions.append(formatted_decision)
            
        return jsonify({"decisions": formatted_decisions})
    except Exception as e:
        logger.error(f"Błąd podczas pobierania decyzji autonomicznego systemu: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "decisions": [],
            "error": str(e)
        })

@app.route('/api/autonomous/model-weights', methods=['GET'])
@jwt_required
def get_model_weights():
    """Endpoint zwracający wagi modeli w autonomicznym systemie."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "weights": {},
                "error": "Autonomiczny system handlowy nie jest dostępny"
            })
        
        meta_agent = autonomous_trading_system.meta_agent
        if not meta_agent:
            return jsonify({
                "weights": {},
                "error": "Meta-Agent nie jest dostępny"
            })
        
        # Pobierz wagi modeli
        weights = {}
        for model_name, model_state in meta_agent.models_state.items():
            weights[model_name] = model_state.get('weight', 1.0)
            
        return jsonify({"weights": weights})
    except Exception as e:
        logger.error(f"Błąd podczas pobierania wag modeli: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "weights": {},
            "error": str(e)
        })

@app.route('/api/autonomous/model-weights', methods=['POST'])
@jwt_required
def update_model_weight():
    """Endpoint do aktualizacji wagi modelu w autonomicznym systemie."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "success": False,
                "error": "Autonomiczny system handlowy nie jest dostępny"
            }), 503
        
        meta_agent = autonomous_trading_system.meta_agent
        if not meta_agent:
            return jsonify({
                "success": False,
                "error": "Meta-Agent nie jest dostępny"
            }), 503
        
        data = request.json
        if not data or "model" not in data or "weight" not in data:
            return jsonify({
                "success": False,
                "error": "Nieprawidłowe dane. Wymagane pola: 'model' i 'weight'"
            }), 400
        
        model_name = data["model"]
        weight = float(data["weight"])
        
        # Sprawdź, czy model istnieje
        if model_name not in meta_agent.models_state:
            return jsonify({
                "success": False,
                "error": f"Model '{model_name}' nie istnieje"
            }), 404
            
        # Zaktualizuj wagę modelu
        meta_agent.models_state[model_name]['weight'] = weight
        logger.info(f"Zaktualizowano wagę modelu {model_name} na {weight}")
            
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji wagi modelu: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/autonomous/risk-parameters', methods=['GET'])
@jwt_required
def get_risk_parameters():
    """Endpoint zwracający parametry ryzyka autonomicznego systemu."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "parameters": {},
                "error": "Autonomiczny system handlowy nie jest dostępny"
            })
        
        risk_manager = autonomous_trading_system.risk_manager
        if not risk_manager:
            return jsonify({
                "parameters": {},
                "error": "Zarządca ryzyka nie jest dostępny"
            })
        
        # Pobierz parametry ryzyka
        parameters = risk_manager.get_risk_parameters()
        return jsonify({"parameters": parameters})
    except Exception as e:
        logger.error(f"Błąd podczas pobierania parametrów ryzyka: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "parameters": {},
            "error": str(e)
        })

@app.route('/api/autonomous/risk-parameters', methods=['POST'])
@jwt_required
def update_risk_parameters():
    """Endpoint do aktualizacji parametrów ryzyka autonomicznego systemu."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "success": False,
                "error": "Autonomiczny system handlowy nie jest dostępny"
            }), 503
        
        risk_manager = autonomous_trading_system.risk_manager
        if not risk_manager:
            return jsonify({
                "success": False,
                "error": "Zarządca ryzyka nie jest dostępny"
            }), 503
        
        data = request.json
        if not data:
            return jsonify({
                "success": False,
                "error": "Nieprawidłowe dane"
            }), 400
        
        # Aktualizuj parametry ryzyka
        for key, value in data.items():
            if key in risk_manager.config:
                risk_manager.config[key] = float(value)
                logger.info(f"Zaktualizowano parametr ryzyka {key} na {value}")
        
        # Wywołaj funkcję adaptacji parametrów ryzyka
        risk_manager._adapt_risk_parameters()
            
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji parametrów ryzyka: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/autonomous/learning-status', methods=['GET'])
@jwt_required
def get_learning_status():
    """Endpoint zwracający status systemu uczenia."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "status": {},
                "error": "Autonomiczny system handlowy nie jest dostępny"
            })
        
        # Zbierz dane o statusie uczenia
        status = {
            "samples_in_queue": 0,
            "last_update": "Brak",
            "best_model": "Brak",
            "models_in_training": []
        }
        
        # W rzeczywistej implementacji, dane te byłyby pobierane z odpowiednich komponentów
        # Na potrzeby demonstracji zwracamy przykładowe dane
        meta_agent = autonomous_trading_system.meta_agent
        if meta_agent:
            # Pobierz najlepszy model (o najwyższej wadze)
            best_model = max(meta_agent.models_state.items(), key=lambda x: x[1]['weight'])[0] if meta_agent.models_state else "Brak"
            status["best_model"] = best_model
            
            # Symuluj ostatnią aktualizację
            status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Symuluj próbki w kolejce
            status["samples_in_queue"] = random.randint(10, 100)
            
            # Symuluj modele w treningu
            status["models_in_training"] = [
                {"model": model_name, "progress": random.randint(0, 100)}
                for model_name in list(meta_agent.models_state.keys())[:2]  # Pierwsze 2 modele
            ]
        
        return jsonify({"status": status})
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statusu uczenia: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": {},
            "error": str(e)
        })

@app.route('/api/autonomous/model-performance', methods=['GET'])
@jwt_required
def get_model_performance():
    """Endpoint zwracający dane o wydajności modeli."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "performance": [],
                "error": "Autonomiczny system handlowy nie jest dostępny"
            })
        
        meta_agent = autonomous_trading_system.meta_agent
        if not meta_agent:
            return jsonify({
                "performance": [],
                "error": "Meta-Agent nie jest dostępny"
            })
        
        # W rzeczywistej implementacji dane byłyby pobierane z systemu
        # Na potrzeby demonstracji generujemy przykładowe dane
        performance_data = []
        
        # Generuj dane historyczne dla każdego modelu
        for model_name, model_state in meta_agent.models_state.items():
            # Utwórz 20 punktów danych dla każdego modelu, co 12 godzin od teraz
            for i in range(20):
                timestamp = (datetime.now() - timedelta(hours=12 * i)).isoformat()
                
                # Podstawowa wartość zależna od wagi modelu
                base_score = model_state.get('weight', 1.0) * 50
                
                # Dodaj losową wariację
                score = base_score + random.uniform(-10, 10)
                
                performance_data.append({
                    "model": model_name,
                    "timestamp": timestamp,
                    "score": score
                })
        
        # Sortuj dane według daty (od najstarszej do najnowszej)
        performance_data.sort(key=lambda x: x["timestamp"])
        
        return jsonify({"performance": performance_data})
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych o wydajności modeli: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "performance": [],
            "error": str(e)
        })

@app.route('/api/autonomous/mode', methods=['POST'])
@jwt_required
def update_autonomous_mode():
    """Endpoint do włączania/wyłączania trybu autonomicznego."""
    try:
        if not autonomous_trading_system:
            return jsonify({
                "success": False,
                "error": "Autonomiczny system handlowy nie jest dostępny"
            }), 503
        
        data = request.json
        if not data or "enabled" not in data:
            return jsonify({
                "success": False,
                "error": "Nieprawidłowe dane. Wymagane pole: 'enabled'"
            }), 400
        
        enabled = bool(data["enabled"])
        
        # Zaktualizuj tryb autonomiczny
        if hasattr(autonomous_trading_system, 'config') and isinstance(autonomous_trading_system.config, dict):
            autonomous_trading_system.config['autonomous_mode'] = enabled
            logger.info(f"Zaktualizowano tryb autonomiczny na: {enabled}")
        
        return jsonify({"success": True, "autonomous_mode": enabled})
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji trybu autonomicznego: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)