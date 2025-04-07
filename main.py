import os
import sys
import logging
from datetime import datetime
import flask
from flask import Flask, jsonify, render_template, request

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
    return render_template("dashboard.html")

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Wiadomość startowa
    logging.info(f"Uruchamianie aplikacji na porcie {port}")
    print(f"Uruchamianie aplikacji na porcie {port}")

    # Uruchomienie serwera Flask
    app.run(host="0.0.0.0", port=port, debug=True)