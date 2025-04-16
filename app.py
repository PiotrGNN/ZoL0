
"""
app.py - Główny plik aplikacji FastAPI dla dashboardu tradingowego
"""

import os
import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import pandas as pd
import numpy as np
import uvicorn

# Importy lokalne
from data.execution.bybit_connector import BybitConnector
from data.execution.exchange_connector import ExchangeConnector
from data.utils.cache_manager import store_cached_data, get_cached_data, is_cache_valid
from data.risk_management.advanced_risk_manager import AdvancedRiskManager
from data.strategies.strategy_manager import StrategyManager

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/api.log")]
)
logger = logging.getLogger(__name__)

# Inicjalizacja aplikacji FastAPI
app = FastAPI(
    title="Trading Dashboard API",
    description="API dla dashboardu tradingowego z danymi w czasie rzeczywistym",
    version="1.0.0"
)

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji należy ograniczyć do konkretnych domen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfiguracja statycznych plików
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dodaj obsługę plików HTML
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serwuj główny dashboard."""
    return templates.TemplateResponse("trading-dashboard.html", {"request": request})

# Klasy modeli Pydantic
class OrderRequest(BaseModel):
    symbol: str
    side: str  # "buy" lub "sell"
    order_type: str  # "market" lub "limit"
    quantity: float
    price: Optional[float] = None  # Wymagane tylko dla zleceń limit

class OrderResponse(BaseModel):
    success: bool
    order_id: Optional[str] = None
    message: Optional[str] = None
    timestamp: str

# Inicjalizacja połączenia z giełdą
try:
    # Załaduj klucze API z pliku .env
    from dotenv import load_dotenv
    load_dotenv()
    
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
    BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    
    exchange = BybitConnector(
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET,
        use_testnet=BYBIT_TESTNET,
        market_type="spot"
    )
    logger.info(f"Połączono z giełdą Bybit (testnet: {BYBIT_TESTNET})")
except Exception as e:
    logger.error(f"Błąd podczas inicjalizacji połączenia z giełdą: {e}")
    exchange = None

# Inicjalizacja menedżera ryzyka
risk_manager = AdvancedRiskManager(
    max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.02")),
    max_drawdown=0.15,
    volatility_factor=1.2,
    min_stop_loss=0.005,
    max_stop_loss=0.03,
    take_profit_factor=2.5,
    leverage_adjustment=True
)
risk_manager.initialize(initial_capital=10000)

# Klasa do zarządzania klientami WebSocketów
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def broadcast_json(self, data: Dict):
        json_data = json.dumps(data)
        await self.broadcast(json_data)

manager = ConnectionManager()

# Zadanie w tle do aktualizacji danych w czasie rzeczywistym
async def update_market_data():
    while True:
        try:
            if exchange is not None:
                # Pobierz ticker BTC/USDT
                btc_ticker = exchange.get_ticker("BTCUSDT")
                
                # Pobierz ticker ETH/USDT
                eth_ticker = exchange.get_ticker("ETHUSDT")
                
                # Przygotuj dane do wysłania
                market_data = {
                    "type": "market_update",
                    "data": {
                        "btc": {
                            "price": btc_ticker.get("last_price", 0),
                            "bid": btc_ticker.get("bid", 0),
                            "ask": btc_ticker.get("ask", 0),
                            "volume": btc_ticker.get("volume_24h", 0),
                            "timestamp": datetime.now().isoformat()
                        },
                        "eth": {
                            "price": eth_ticker.get("last_price", 0),
                            "bid": eth_ticker.get("bid", 0),
                            "ask": eth_ticker.get("ask", 0),
                            "volume": eth_ticker.get("volume_24h", 0),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                }
                
                # Wyślij dane do wszystkich klientów
                await manager.broadcast_json(market_data)
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji danych rynkowych: {e}")
        
        # Poczekaj przed kolejną aktualizacją
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    """Inicjalizacja zadań uruchamianych przy starcie aplikacji."""
    asyncio.create_task(update_market_data())

@app.get("/api/health")
async def health_check():
    """Endpoint sprawdzający stan aplikacji."""
    return {"status": "online", "timestamp": datetime.now().isoformat()}

@app.get("/api/balance")
async def get_balance():
    """Pobierz saldo portfela z giełdy."""
    try:
        if exchange is None:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "Brak połączenia z giełdą"}
            )
        
        # Pobierz saldo z pamięci podręcznej, jeśli dostępne
        cache_key = "account_balance"
        if is_cache_valid(cache_key, ttl=30):  # 30 sekund TTL
            cached_data, found = get_cached_data(cache_key)
            if found and cached_data:
                return cached_data
        
        balance_data = exchange.get_account_balance()
        
        # Zapisz dane w pamięci podręcznej
        store_cached_data(cache_key, balance_data)
        
        return balance_data
    except Exception as e:
        logger.error(f"Błąd podczas pobierania salda: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/ticker/{symbol}")
async def get_ticker(symbol: str):
    """Pobierz ticker dla danego symbolu."""
    try:
        if exchange is None:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "Brak połączenia z giełdą"}
            )
        
        # Pobierz ticker z pamięci podręcznej, jeśli dostępny
        cache_key = f"ticker_{symbol}"
        if is_cache_valid(cache_key, ttl=5):  # 5 sekund TTL
            cached_data, found = get_cached_data(cache_key)
            if found and cached_data:
                return cached_data
        
        ticker_data = exchange.get_ticker(symbol)
        
        # Zapisz dane w pamięci podręcznej
        store_cached_data(cache_key, ticker_data)
        
        return ticker_data
    except Exception as e:
        logger.error(f"Błąd podczas pobierania tickera dla {symbol}: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/klines/{symbol}")
async def get_klines(symbol: str, interval: str = "15m", limit: int = 200):
    """Pobierz dane świecowe dla danego symbolu i interwału."""
    try:
        if exchange is None:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "Brak połączenia z giełdą"}
            )
        
        # Pobierz dane z pamięci podręcznej, jeśli dostępne
        cache_key = f"klines_{symbol}_{interval}_{limit}"
        if is_cache_valid(cache_key, ttl=60):  # 60 sekund TTL
            cached_data, found = get_cached_data(cache_key)
            if found and cached_data:
                return cached_data
        
        klines_data = exchange.get_klines(symbol, interval, limit)
        
        # Zapisz dane w pamięci podręcznej
        store_cached_data(cache_key, klines_data)
        
        return {"success": True, "data": klines_data}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych świecowych dla {symbol}: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/order")
async def place_order(order: OrderRequest):
    """Złóż zlecenie na giełdzie."""
    try:
        if exchange is None:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "Brak połączenia z giełdą"}
            )
        
        # Sprawdź ryzyko zlecenia
        ticker_data = exchange.get_ticker(order.symbol)
        current_price = ticker_data.get("last_price", 0)
        
        # Oblicz wartość zlecenia
        order_value = order.quantity * (order.price if order.price else current_price)
        
        # Pobierz saldo
        balance_data = exchange.get_account_balance()
        
        # Pobierz dostępne saldo USDT
        available_balance = 0
        if "balances" in balance_data and "USDT" in balance_data["balances"]:
            available_balance = balance_data["balances"]["USDT"].get("available_balance", 0)
        
        # Sprawdź czy mamy wystarczającą ilość środków
        if order_value > available_balance and order.side.lower() == "buy":
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Niewystarczające środki. Wymagane: {order_value} USDT, dostępne: {available_balance} USDT"
                }
            )
        
        # W przypadku prawdziwego API wykonaj zlecenie
        # response = exchange.place_order(
        #     symbol=order.symbol,
        #     side=order.side,
        #     price=order.price,
        #     quantity=order.quantity,
        #     order_type=order.order_type
        # )
        
        # Na potrzeby demonstracji zwróć symulowane dane
        order_id = f"SIM_ORDER_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return OrderResponse(
            success=True,
            order_id=order_id,
            message=f"Złożono zlecenie {order.side} {order.quantity} {order.symbol}",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Błąd podczas składania zlecenia: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/market_summary")
async def get_market_summary():
    """Pobierz podsumowanie rynku dla głównych kryptowalut."""
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
        results = {}
        
        for symbol in symbols:
            cache_key = f"ticker_{symbol}"
            if is_cache_valid(cache_key, ttl=30):  # 30 sekund TTL
                cached_data, found = get_cached_data(cache_key)
                if found and cached_data:
                    results[symbol] = cached_data
                    continue
            
            if exchange is not None:
                ticker_data = exchange.get_ticker(symbol)
                store_cached_data(cache_key, ticker_data)
                results[symbol] = ticker_data
        
        return {"success": True, "data": results}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania podsumowania rynku: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/portfolio_performance")
async def get_portfolio_performance():
    """Pobierz dane dotyczące wydajności portfela."""
    try:
        # Na potrzeby demonstracji zwróć symulowane dane
        dates = pd.date_range(start='2025-01-01', periods=90)
        
        # Symulacja wartości portfela
        np.random.seed(42)
        initial_value = 10000
        daily_returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_values = initial_value * np.cumprod(1 + daily_returns)
        
        # Oblicz metrics
        total_return = (portfolio_values[-1] / initial_value - 1) * 100
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values) * 100
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        # Przygotuj odpowiedź
        performance_data = {
            "portfolio_history": [
                {"date": date.strftime('%Y-%m-%d'), "value": value}
                for date, value in zip(dates, portfolio_values)
            ],
            "metrics": {
                "total_return": round(total_return, 2),
                "max_drawdown": round(max_drawdown, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "win_rate": 68.5,  # Symulowane
                "avg_win": 2.4,  # Symulowane
                "avg_loss": -1.2,  # Symulowane
                "best_trade": 12.8,  # Symulowane
                "worst_trade": -5.3  # Symulowane
            }
        }
        
        return {"success": True, "data": performance_data}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych wydajności: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/strategies")
async def get_strategies():
    """Pobierz listę dostępnych strategii."""
    try:
        # Przykładowe strategie
        strategies = [
            {
                "id": "trend_following",
                "name": "Trend Following",
                "description": "Strategia podążająca za trendem z wykorzystaniem wskaźników MACD i RSI",
                "status": "active",
                "performance": {
                    "win_rate": 68,
                    "profit_factor": 1.85,
                    "sharpe_ratio": 1.35
                }
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "description": "Strategia powrotu do średniej wykorzystująca Bollinger Bands",
                "status": "inactive",
                "performance": {
                    "win_rate": 72,
                    "profit_factor": 1.62,
                    "sharpe_ratio": 1.18
                }
            },
            {
                "id": "breakout",
                "name": "Breakout",
                "description": "Strategia przełamania wykorzystująca wsparcia i opory",
                "status": "active",
                "performance": {
                    "win_rate": 55,
                    "profit_factor": 2.1,
                    "sharpe_ratio": 1.42
                }
            }
        ]
        
        return {"success": True, "data": strategies}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania strategii: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket dla aktualizacji w czasie rzeczywistym."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo z powrotem - w przyszłości można dodać przetwarzanie komend
            await manager.send_personal_message(f"Otrzymano: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
