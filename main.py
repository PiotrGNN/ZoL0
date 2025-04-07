import os
import sys
import logging
from datetime import datetime
import flask
from flask import Flask, jsonify, render_template, request

# Próba importu dotenv z obsługą błędu
try:
    from dotenv import load_dotenv
    load_dotenv()  # ładujemy zmienne środowiskowe
except ImportError:
    logging.warning("Moduł dotenv nie jest zainstalowany. Zmienne środowiskowe mogą nie być dostępne.")

# Konfiguracja logowania
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Inicjalizacja Flask
app = Flask(__name__)

# Trasy API dla dashboardu
@app.route("/")
def index():
    # Przykładowe dane dla szablonu
    settings = {
        "risk_level": "low",
        "max_position_size": 10.0,
        "enable_auto_trading": False
    }
    
    # Przykładowe dane dla modeli AI
    ai_models = [
        {"name": "Trend Predictor", "type": "XGBoost", "accuracy": 78, "status": "Active", "last_used": "2025-04-07 10:15:22"},
        {"name": "Volatility Model", "type": "LSTM", "accuracy": 82, "status": "Active", "last_used": "2025-04-07 09:45:10"},
        {"name": "Sentiment Analyzer", "type": "Transformer", "accuracy": 65, "status": "Inactive", "last_used": "2025-04-06 18:30:45"}
    ]
    
    # Przykładowe dane dla strategii
    strategies = [
        {"id": 1, "name": "Trend Following", "description": "Podąża za trendem rynkowym", "enabled": True, "win_rate": 68, "profit_factor": 1.8},
        {"id": 2, "name": "Mean Reversion", "description": "Wykorzystuje powroty do średniej", "enabled": False, "win_rate": 55, "profit_factor": 1.3},
        {"id": 3, "name": "Breakout", "description": "Wykrywa i wykorzystuje wybicia", "enabled": True, "win_rate": 62, "profit_factor": 1.5}
    ]
    
    # Przykładowe dane dla alertów
    alerts = [
        {"level_class": "warning", "time": "10:15", "message": "Wysoka zmienność na BTC/USDT"},
        {"level_class": "offline", "time": "09:30", "message": "Utracono połączenie z API"}
    ]
    
    # Przykładowe dane dla transakcji
    trades = [
        {"symbol": "BTC/USDT", "type": "BUY", "time": "10:05", "profit": 2.5},
        {"symbol": "ETH/USDT", "type": "SELL", "time": "09:45", "profit": -1.2},
        {"symbol": "SOL/USDT", "type": "BUY", "time": "09:15", "profit": 3.8}
    ]
    
    # Przykładowe dane dla sentymentu
    sentiment_data = {
        "overall_score": 0.25,
        "analysis": "Umiarkowanie pozytywny",
        "sources": {
            "Twitter": {"score": 0.35, "volume": 1250},
            "Reddit": {"score": 0.18, "volume": 850},
            "News": {"score": 0.22, "volume": 320}
        },
        "timeframe": "24h",
        "timestamp": "2025-04-07 08:30:00"
    }
    
    # Przykładowe dane dla anomalii
    anomalies = [
        {"timestamp": "10:05", "type": "Spike Detection", "description": "Nagły wzrost wolumenu BTC", "score": 0.85},
        {"timestamp": "09:30", "type": "Price Pattern", "description": "Nietypowy wzór cenowy na ETH", "score": 0.72}
    ]
    
    return render_template(
        "dashboard.html",
        settings=settings,
        ai_models=ai_models,
        strategies=strategies,
        alerts=alerts,
        trades=trades,
        sentiment_data=sentiment_data,
        anomalies=anomalies
    )

@app.route("/api/dashboard/data", methods=["GET"])
def get_dashboard_data():
    # Symulowane dane dla dashboardu
    return jsonify({"success": True, "data": {"balance": 10000.00, "open_positions": 2}})

@app.route("/api/trading-stats", methods=["GET"])
def get_trading_stats():
    # Symulowane statystyki handlowe
    return jsonify({
        "success": True,
        "data": {
            "total_trades": 120,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "daily_pnl": 125.50
        }
    })

@app.route("/api/recent-trades", methods=["GET"])
def get_recent_trades():
    # Symulowane ostatnie transakcje
    return jsonify({
        "success": True,
        "data": [
            {"id": 1, "symbol": "BTCUSDT", "side": "BUY", "price": 65432.10, "time": "2025-04-07 11:30:45"},
            {"id": 2, "symbol": "ETHUSDT", "side": "SELL", "price": 3210.50, "time": "2025-04-07 11:15:22"},
            {"id": 3, "symbol": "SOLUSDT", "side": "BUY", "price": 180.75, "time": "2025-04-07 10:55:18"}
        ]
    })

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    # Symulowane alerty
    return jsonify({
        "success": True,
        "data": [
            {"id": 1, "type": "INFO", "message": "Połączono z API Bybit", "time": "2025-04-07 10:00:00"},
            {"id": 2, "type": "WARNING", "message": "Wysoka zmienność na BTC", "time": "2025-04-07 11:30:00"}
        ]
    })

@app.route("/api/ai-models-status", methods=["GET"])
def get_ai_status():
    # Symulowany status modeli AI
    return jsonify({
        "success": True,
        "data": [
            {"name": "Trend Predictor", "status": "active", "accuracy": 0.78, "last_update": "2025-04-07 11:45:00"},
            {"name": "Volatility Model", "status": "active", "accuracy": 0.82, "last_update": "2025-04-07 11:40:00"},
            {"name": "Sentiment Analyzer", "status": "inactive", "accuracy": 0.0, "last_update": "2025-04-07 09:30:00"}
        ]
    })

@app.route("/api/component-status", methods=["GET"])
def get_component_status():
    # Symulowany status komponentów systemu
    return jsonify({
        "success": True,
        "components": [
            {"id": "api_connection", "name": "API Connection", "status": "operational"},
            {"id": "websocket_feed", "name": "WebSocket Feed", "status": "degraded"},
            {"id": "ai_prediction", "name": "AI Prediction Engine", "status": "operational"},
            {"id": "trade_execution", "name": "Trade Execution", "status": "operational"},
            {"id": "risk_management", "name": "Risk Management", "status": "operational"}
        ]
    })

@app.route("/api/notifications", methods=["GET"])
def get_notifications():
    # Symulowane powiadomienia
    return jsonify({
        "success": True,
        "data": [
            {"id": 1, "type": "success", "message": "Zlecenie BUY BTCUSDT wykonane", "time": "2025-04-07 11:30:45"},
            {"id": 2, "type": "warning", "message": "Wysoka zmienność rynku", "time": "2025-04-07 10:15:22"}
        ]
    })

@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    # Symulowane dane portfela
    return jsonify({
        "success": True,
        "data": {
            "total_value": 15240.75,
            "assets": [
                {"symbol": "BTC", "amount": 0.24, "value_usd": 8760.50, "allocation": 57.5, "pnl_24h": 3.8},
                {"symbol": "ETH", "amount": 1.85, "value_usd": 3700.25, "allocation": 24.3, "pnl_24h": -1.2},
                {"symbol": "SOL", "amount": 18.5, "value_usd": 1680.00, "allocation": 11.0, "pnl_24h": 5.4},
                {"symbol": "USDT", "amount": 1100.00, "value_usd": 1100.00, "allocation": 7.2, "pnl_24h": 0.0}
            ],
            "pnl_total": 1240.75,
            "pnl_percentage": 8.85,
            "last_updated": "2025-04-07 12:45:30"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Wiadomość startowa
    logging.info(f"Uruchamianie aplikacji na porcie {port}")
    print(f"Uruchamianie aplikacji na porcie {port}")

    # Uruchomienie serwera Flask
    app.run(host="0.0.0.0", port=port, debug=True)