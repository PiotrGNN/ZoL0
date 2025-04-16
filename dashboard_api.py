#!/usr/bin/env python3
"""
dashboard_api.py - API servera dla dashboardu tradingowego
"""

import os
import sys
import json
import logging
import random
from datetime import datetime, timedelta
from flask import Flask, jsonify, request

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

# Inicjalizacja Flaska
app = Flask(__name__)

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

# Endpoints API
@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    logger.info("Zapytanie o dane portfela")
    return jsonify(SIMULATED_DATA["portfolio"])

@app.route('/api/trades', methods=['GET'])
def get_trades():
    logger.info("Zapytanie o historię transakcji")
    return jsonify(SIMULATED_DATA["trades"])

@app.route('/api/component-status', methods=['GET'])
def get_component_status():
    logger.info("Zapytanie o status komponentów")
    return jsonify(SIMULATED_DATA["component-status"])

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)