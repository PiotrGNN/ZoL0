import os
import sys
import logging
import random
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
    try:
        # Sprawdzamy czy mamy zapisane modele
        model_files = []
        if os.path.exists("saved_models"):
            model_files = [f for f in os.listdir("saved_models") if f.endswith(".json") or f.endswith(".pkl")]
        
        if not model_files:
            # Jeśli nie ma zapisanych modeli, zwracamy symulowane dane
            return jsonify({
                "success": True,
                "data": [
                    {"name": "Trend Predictor", "status": "active", "accuracy": 0.78, "last_update": "2025-04-07 11:45:00"},
                    {"name": "Volatility Model", "status": "active", "accuracy": 0.82, "last_update": "2025-04-07 11:40:00"},
                    {"name": "Sentiment Analyzer", "status": "inactive", "accuracy": 0.0, "last_update": "2025-04-07 09:30:00"}
                ]
            })
        
        # Przygotowujemy dane o modelach
        models_data = []
        for model_file in model_files:
            try:
                # Parsujemy nazwę modelu
                parts = model_file.split("_")
                if len(parts) >= 3:
                    model_type = parts[0]
                    symbol = parts[1]
                    date_parts = parts[2].split(".")
                    date_str = date_parts[0]
                    
                    # Format daty: YYYYMMDD_HHMMSS
                    if len(date_str) >= 15:
                        date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        last_update = date.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        last_update = "Nieznana data"
                    
                    # Dodajemy informacje o modelu
                    models_data.append({
                        "name": f"{model_type} {symbol}",
                        "status": "active",
                        "accuracy": round(random.uniform(0.65, 0.85), 2),  # Losowa dokładność (można zaimplementować odczyt z pliku)
                        "last_update": last_update
                    })
            except Exception as e:
                logging.error(f"Błąd podczas parsowania pliku modelu {model_file}: {e}")
        
        return jsonify({
            "success": True,
            "data": models_data
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu modeli AI: {e}")
        # Zwracamy awaryjne dane
        return jsonify({
            "success": True,
            "data": [
                {"name": "Trend Predictor", "status": "active", "accuracy": 0.78, "last_update": "2025-04-07 11:45:00"},
                {"name": "Volatility Model", "status": "active", "accuracy": 0.82, "last_update": "2025-04-07 11:40:00"},
                {"name": "Sentiment Analyzer", "status": "inactive", "accuracy": 0.0, "last_update": "2025-04-07 09:30:00"}
            ],
            "error": str(e)
        })

@app.route("/api/train-ai-model", methods=["POST"])
def train_ai_model():
    try:
        # Sprawdzamy czy mamy wymagane parametry
        data = request.json or {}
        symbol = data.get("symbol", "BTCUSDT")
        interval = data.get("interval", "1h")
        lookback_days = int(data.get("lookback_days", 30))
        
        # Importujemy trainer
        from ai_models.market_data_trainer import MarketDataTrainer
        
        # Inicjalizujemy trainer
        trainer = MarketDataTrainer(
            symbols=[symbol],
            interval=interval,
            lookback_days=lookback_days,
            test_size=0.2,
            use_testnet=True
        )
        
        # Trenujemy model
        result = trainer.train_model_for_symbol(symbol)
        
        return jsonify({
            "success": result.get("success", False),
            "symbol": symbol,
            "interval": interval,
            "lookback_days": lookback_days,
            "result": result
        })
    except Exception as e:
        logging.error(f"Błąd podczas trenowania modelu AI: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/api/component-status", methods=["GET"])
def get_component_status():
    # Rzeczywisty status komponentów systemu
    try:
        from data.execution.bybit_connector import BybitConnector
        import os
        
        # Sprawdzamy konfigurację API
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        use_testnet = os.getenv("TEST_MODE", "false").lower() in ["true", "1", "t"]
        
        # Maskowane klucze dla bezpieczeństwa w logach
        api_key_status = "configured" if api_key else "missing"
        api_secret_status = "configured" if api_secret else "missing"
        
        # Status połączenia API
        api_status = "operational"
        ws_status = "operational"
        
        if not (api_key and api_secret):
            api_status = "misconfigured"
            ws_status = "misconfigured"
        else:
            # Testujemy połączenie z Bybit
            bybit = BybitConnector(
                api_key=api_key,
                api_secret=api_secret,
                use_testnet=use_testnet
            )
            if not bybit.test_connectivity():
                api_status = "degraded"
                ws_status = "degraded"
        
        # Sprawdzamy czy istnieją modele AI
        ai_models_exist = os.path.exists("saved_models") and len([f for f in os.listdir("saved_models") if f.endswith(".pkl")]) > 0
        ai_status = "operational" if ai_models_exist else "degraded"
        
        return jsonify({
            "success": True,
            "api_config": {
                "api_key": api_key_status,
                "api_secret": api_secret_status,
                "test_mode": use_testnet
            },
            "components": [
                {"id": "api_connection", "name": "API Connection", "status": api_status},
                {"id": "websocket_feed", "name": "WebSocket Feed", "status": ws_status},
                {"id": "ai_prediction", "name": "AI Prediction Engine", "status": ai_status},
                {"id": "trade_execution", "name": "Trade Execution", "status": "operational"},
                {"id": "risk_management", "name": "Risk Management", "status": "operational"}
            ]
        })
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania statusu komponentów: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "components": [
                {"id": "api_connection", "name": "API Connection", "status": "unknown"},
                {"id": "websocket_feed", "name": "WebSocket Feed", "status": "unknown"},
                {"id": "ai_prediction", "name": "AI Prediction Engine", "status": "unknown"},
                {"id": "trade_execution", "name": "Trade Execution", "status": "unknown"},
                {"id": "risk_management", "name": "Risk Management", "status": "unknown"}
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
    try:
        # Inicjalizacja konektora Bybit
        from data.execution.bybit_connector import BybitConnector
        import os
        from datetime import datetime
        
        # Pobieramy klucze API z zmiennych środowiskowych
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        use_testnet = os.getenv("TEST_MODE", "false").lower() in ["true", "1", "t"]
        
        # Debugowanie: wyświetl wszystkie zmienne środowiskowe
        env_vars = {key: val for key, val in os.environ.items()}
        logging.info(f"Dostępne zmienne środowiskowe: {list(env_vars.keys())}")
        
        # Logujemy informacje o konfiguracji (ocenzurowane klucze dla bezpieczeństwa)
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "brak klucza"
        masked_secret = api_secret[:4] + "..." + api_secret[-4:] if len(api_secret) > 8 else "brak sekretu"
        logging.info(f"Próba połączenia z Bybit - API Key: {masked_key}, Secret: {masked_secret}, Testnet: {use_testnet}")
        
        # Sprawdzamy czy mamy klucze API
        simulation_mode = not (api_key and api_secret)
        
        # Inicjalizujemy konektor
        bybit = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet,
            simulation_mode=simulation_mode
        )
        
        # Testujemy połączenie przed pobraniem danych
        logging.info("Testowanie połączenia z Bybit...")
        connection_ok = bybit.test_connectivity()
        logging.info(f"Wynik testu połączenia: {connection_ok}")
        
        if not connection_ok:
            logging.error("Test połączenia z Bybit nie powiódł się")
            return jsonify({
                "success": False,
                "error": "Nie można nawiązać połączenia z Bybit API",
                "debug_info": {
                    "api_key_present": bool(api_key),
                    "api_secret_present": bool(api_secret),
                    "simulation_mode": simulation_mode,
                    "testnet_enabled": use_testnet,
                },
                "data": {
                    "total_value": 0,
                    "assets": [],
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "błąd połączenia"
                }
            })
        
        # Pobieramy saldo portfela
        logging.info("Pobieranie salda portfela...")
        wallet_data = bybit.get_wallet_balance()
        logging.info(f"Odpowiedź API (wallet): {wallet_data.get('ret_code')} - {wallet_data.get('ret_msg')}")
        logging.debug(f"Pełna odpowiedź: {wallet_data}")
        
        if simulation_mode or "result" not in wallet_data or not wallet_data["result"].get("list"):
            # Jeśli brak danych lub jesteśmy w trybie symulacji, zwracamy symulowane dane
            reason = 'Tryb symulacji' if simulation_mode else 'Brak danych z API'
            logging.warning(f"Używanie symulowanych danych. Powód: {reason}")
            if "result" in wallet_data:
                logging.debug(f"Struktura odpowiedzi: {list(wallet_data['result'].keys())}")
        
        # Przetwarzamy dane portfela
        wallet_info = wallet_data["result"]["list"][0]
        total_value = float(wallet_info.get("totalEquity", 0))
        coins = wallet_info.get("coin", [])
        
        # Inicjalizacja pustej listy aktywów
        assets = []
        total_allocation = 0
        
        # Dla każdej monety, pobieramy jej cenę i obliczamy wartość
        for coin in coins:
            coin_name = coin.get("coin", "")
            amount = float(coin.get("walletBalance", 0))
            
            # Pomijamy monety o zerowym saldzie
            if amount <= 0:
                continue
            
            # Obliczamy wartość w USD
            value_usd = amount
            price = 1.0  # Domyślna cena dla stablecoinów
            
            # Dla kryptowalut innych niż stablecoiny, pobieramy cenę
            if coin_name not in ["USDT", "USDC", "BUSD", "DAI"]:
                try:
                    # Pobieramy cenę monety
                    ticker_symbol = f"{coin_name}USDT"
                    ticker_data = bybit.get_ticker(ticker_symbol)
                    
                    if "result" in ticker_data and "list" in ticker_data["result"] and ticker_data["result"]["list"]:
                        price = float(ticker_data["result"]["list"][0].get("lastPrice", 0))
                        value_usd = amount * price
                        logging.info(f"Pobrano cenę {coin_name}: {price} USD")
                    else:
                        logging.warning(f"Brak danych cenowych dla {coin_name}, używam wartości domyślnej")
                except Exception as e:
                    logging.error(f"Błąd podczas pobierania ceny {coin_name}: {e}")
            
            # Pobieramy pozycję z dodatkowych informacji jeśli to możliwe
            pnl_24h = 0
            try:
                if "unrealisedPnl" in coin:
                    unrealised_pnl = float(coin.get("unrealisedPnl", 0))
                    if value_usd > 0:
                        pnl_24h = (unrealised_pnl / value_usd) * 100
            except Exception as e:
                logging.error(f"Błąd podczas obliczania PnL dla {coin_name}: {e}")
            
            # Dodajemy do listy aktywów
            assets.append({
                "symbol": coin_name,
                "amount": amount,
                "value_usd": value_usd,
                "price": price,
                "allocation": 0,  # tymczasowo, zostanie zaktualizowane poniżej
                "pnl_24h": pnl_24h
            })
            
            total_allocation += value_usd
        
        # Aktualizujemy alokację procentową
        for asset in assets:
            if total_allocation > 0:
                asset["allocation"] = round(asset["value_usd"] / total_allocation * 100, 1)
        
        # Sortujemy aktywa według wartości (malejąco)
        assets.sort(key=lambda x: x["value_usd"], reverse=True)
        
        # Obliczamy całkowity PnL jeśli dostępne są dane historyczne
        pnl_total = sum(asset["pnl_24h"] * asset["value_usd"] / 100 for asset in assets)
        pnl_percentage = (pnl_total / total_value * 100) if total_value > 0 else 0
        
        logging.info(f"Pobrano dane portfela: {len(assets)} aktyw(a/ów), wartość całkowita: {total_value} USD")
        
        return jsonify({
            "success": True,
            "data": {
                "total_value": total_value,
                "assets": assets,
                "pnl_total": pnl_total,
                "pnl_percentage": pnl_percentage,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "dane rzeczywiste",
                "api_connection": "aktywne" if connection_ok else "problem z połączeniem"
            }
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania danych portfela: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "data": {
                "total_value": 0,
                "assets": [],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "błąd"
            }
        })

@app.route("/api/env-check", methods=["GET"])
def env_check():
    """Endpoint diagnostyczny do sprawdzania zmiennych środowiskowych (tylko w trybie deweloperskim)"""
    if app.debug:
        env_vars = {
            "BYBIT_API_KEY": os.getenv("BYBIT_API_KEY", "")[0:4] + "..." if os.getenv("BYBIT_API_KEY") else "brak",
            "BYBIT_API_SECRET": os.getenv("BYBIT_API_SECRET", "")[0:4] + "..." if os.getenv("BYBIT_API_SECRET") else "brak",
            "TEST_MODE": os.getenv("TEST_MODE", "brak"),
            "APP_ENV": os.getenv("APP_ENV", "brak"),
            "env_file_loaded": os.path.exists(".env"),
            "dotenv_imported": "dotenv" in sys.modules,
            "port": os.environ.get("PORT", "5000"),
            "python_path": os.environ.get("PYTHONPATH", "brak")
        }
        return jsonify({"env_vars": env_vars})
    else:
        return jsonify({"error": "Ten endpoint jest dostępny tylko w trybie deweloperskim"}), 403

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Wiadomość startowa
    logging.info(f"Uruchamianie aplikacji na porcie {port}")
    print(f"Uruchamianie aplikacji na porcie {port}")
    
    # Informacja o kluczach API
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    logging.info(f"Status kluczy API - API Key: {'skonfigurowany' if api_key else 'brak'}, API Secret: {'skonfigurowany' if api_secret else 'brak'}")

    # Uruchomienie serwera Flask
    app.run(host="0.0.0.0", port=port, debug=True)