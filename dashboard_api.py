
import os
import logging
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# FastAPI
from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Websockets i asyncio
import asyncio
import websockets
from starlette.websockets import WebSocketDisconnect

# Baza danych
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Integracje giełdowe
import ccxt
try:
    import pybit
except ImportError:
    print("Pybit nie został zainstalowany, używane będzie tylko CCXT i symulowane dane")
    pybit = None

try:
    from binance.client import Client as BinanceClient
    from binance.websockets import BinanceSocketManager
    binance_available = True
except ImportError:
    print("Binance API nie został zainstalowany, używane będzie tylko CCXT i symulowane dane")
    binance_available = False

# Analiza techniczna
import ta
try:
    import pandas_ta as pta
except ImportError:
    print("pandas_ta nie został zainstalowany, używana będzie tylko standardowa biblioteka ta")
    pta = None

# Plotly do wykresów
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Wczytanie lokalnych modułów projektu
from dotenv import load_dotenv
import sys

# Dodanie katalogu projektu do ścieżki Pythona
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Dodaj katalog z własnymi bibliotekami
LOCAL_LIBS_DIR = "python_libs"
if os.path.exists(LOCAL_LIBS_DIR):
    sys.path.insert(0, LOCAL_LIBS_DIR)
    print(f"Dodano katalog {LOCAL_LIBS_DIR} do ścieżki Pythona.")

# Importy z istniejącego projektu
try:
    from ai_models.model_recognition import ModelRecognizer
    from ai_models.anomaly_detection import AnomalyDetector
    from ai_models.sentiment_ai import SentimentAnalyzer
    from data.data.market_data_fetcher import MarketDataFetcher
    from python_libs.simplified_trading_engine import SimplifiedTradingEngine
    from python_libs.simplified_strategy import StrategyManager
    from python_libs.simplified_risk_manager import SimplifiedRiskManager
except ImportError as e:
    print(f"Błąd podczas importowania modułów projektu: {e}")
    print("Używane będą uproszczone implementacje")

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/dashboard_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Wczytanie zmiennych środowiskowych
load_dotenv()

# Tworzenie katalogów dla danych
os.makedirs("data/cache", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)

# Konfiguracja bazy danych
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/trading_dashboard.db")
Base = declarative_base()

# Modele bazy danych
class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # BUY or SELL
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float)
    entry_time = Column(DateTime, default=datetime.now)
    exit_time = Column(DateTime, nullable=True)
    profit_loss = Column(Float, nullable=True)
    profit_loss_percent = Column(Float, nullable=True)
    status = Column(String, default="OPEN")  # OPEN, CLOSED
    strategy = Column(String)
    exchange = Column(String)
    order_id = Column(String, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

class SignalEvent(Base):
    __tablename__ = "signal_events"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    signal_type = Column(String)  # BUY, SELL, NEUTRAL
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    strategy = Column(String)
    indicator_values = Column(Text)  # JSON string of indicator values
    confidence = Column(Float)
    executed = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    interval = Column(String)
    exchange = Column(String)
    
class SystemLog(Base):
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)
    component = Column(String, nullable=True)
    details = Column(Text, nullable=True)

# Tworzenie silnika bazy danych
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Funkcja pomocnicza dla pozyskiwania sesji DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Modele Pydantic dla API
class TradeCreate(BaseModel):
    symbol: str
    side: str
    entry_price: float
    quantity: float
    strategy: str
    exchange: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None

class TradeUpdate(BaseModel):
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit_loss: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    status: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None

class SignalCreate(BaseModel):
    symbol: str
    signal_type: str
    price: float
    strategy: str
    indicator_values: Dict[str, Any]
    confidence: float
    notes: Optional[str] = None

class StrategyConfig(BaseModel):
    strategy_name: str
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    initial_capital: float = 10000.0
    parameters: Dict[str, Any] = Field(default_factory=dict)

# Inicjalizacja FastAPI
app = FastAPI(
    title="Trading Dashboard API",
    description="API dla zaawansowanego dashboardu tradingowego z obsługą danych z giełd i sygnałów AI",
    version="1.0.0"
)

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Klasa zarządzająca połączeniami WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "market_data": [],
            "signals": [],
            "trades": [],
            "system_logs": []
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)
            logger.info(f"Nowe połączenie WebSocket na kanale: {channel}")

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections and websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)
            logger.info(f"Rozłączono WebSocket z kanału: {channel}")

    async def broadcast(self, message: dict, channel: str):
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Błąd podczas wysyłania wiadomości WebSocket: {e}")
                    # Automatyczne odłączenie klienta przy błędzie
                    await connection.close()
                    self.active_connections[channel].remove(connection)

manager = ConnectionManager()

# Klasa dla pobierania danych rynkowych
class MarketDataService:
    def __init__(self):
        # Konfiguracja dostępu do giełd
        self.bybit_api_key = os.getenv("BYBIT_API_KEY")
        self.bybit_api_secret = os.getenv("BYBIT_API_SECRET")
        self.bybit_testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
        
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_api_secret = os.getenv("BINANCE_API_SECRET")
        self.binance_testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        # Inicjalizacja klientów giełdowych
        self.bybit_client = None
        self.binance_client = None
        self.ccxt_clients = {}
        
        # Cache dla danych rynkowych
        self.market_data_cache = {}
        self.last_update_time = {}
        
        # Inicjalizacja połączeń
        self.initialize_connections()
        
        # Pobierz instancję MarketDataFetcher z projektu
        try:
            self.data_fetcher = MarketDataFetcher(
                api_key=self.bybit_api_key, 
                api_secret=self.bybit_api_secret,
                use_testnet=self.bybit_testnet,
                output_mode="csv"
            )
            logger.info("Zainicjalizowano MarketDataFetcher")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji MarketDataFetcher: {e}")
            self.data_fetcher = None
    
    def initialize_connections(self):
        # Inicjalizacja klienta Bybit
        if pybit and self.bybit_api_key and self.bybit_api_secret:
            try:
                from pybit import usdt_perpetual
                
                base_url = "https://api-testnet.bybit.com" if self.bybit_testnet else "https://api.bybit.com"
                self.bybit_client = usdt_perpetual.HTTP(
                    endpoint=base_url,
                    api_key=self.bybit_api_key,
                    api_secret=self.bybit_api_secret
                )
                logger.info("Połączono z API Bybit")
            except Exception as e:
                logger.error(f"Błąd podczas inicjalizacji klienta Bybit: {e}")
        
        # Inicjalizacja klienta Binance
        if binance_available and self.binance_api_key and self.binance_api_secret:
            try:
                base_url = "https://testnet.binance.vision" if self.binance_testnet else None
                self.binance_client = BinanceClient(
                    api_key=self.binance_api_key,
                    api_secret=self.binance_api_secret,
                    testnet=self.binance_testnet
                )
                logger.info("Połączono z API Binance")
            except Exception as e:
                logger.error(f"Błąd podczas inicjalizacji klienta Binance: {e}")
        
        # Inicjalizacja klientów CCXT
        try:
            # Inicjalizacja klienta Bybit przez CCXT
            bybit_ccxt_config = {
                'apiKey': self.bybit_api_key,
                'secret': self.bybit_api_secret,
                'enableRateLimit': True
            }
            if self.bybit_testnet:
                bybit_ccxt_config['testnet'] = True
            
            self.ccxt_clients['bybit'] = ccxt.bybit(bybit_ccxt_config)
            
            # Inicjalizacja klienta Binance przez CCXT
            binance_ccxt_config = {
                'apiKey': self.binance_api_key,
                'secret': self.binance_api_secret,
                'enableRateLimit': True
            }
            if self.binance_testnet:
                binance_ccxt_config['testnet'] = True
            
            self.ccxt_clients['binance'] = ccxt.binance(binance_ccxt_config)
            
            logger.info("Zainicjalizowano klientów CCXT")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji klientów CCXT: {e}")
    
    async def fetch_ohlcv(self, symbol: str, interval: str = '1h', limit: int = 100, exchange: str = 'bybit'):
        """Pobiera dane OHLCV dla danego symbolu"""
        cache_key = f"{exchange}_{symbol}_{interval}"
        current_time = time.time()
        
        # Sprawdź czy dane są w cache i aktualne
        if cache_key in self.market_data_cache and cache_key in self.last_update_time:
            last_update = self.last_update_time[cache_key]
            # Określ interwał odświeżania na podstawie interwału danych
            refresh_interval = 60  # domyślnie 60 sekund
            if interval.endswith('m'):
                refresh_interval = int(interval[:-1]) * 60  # minuty na sekundy
            elif interval.endswith('h'):
                refresh_interval = int(interval[:-1]) * 3600  # godziny na sekundy
            elif interval.endswith('d'):
                refresh_interval = 86400  # dzień w sekundach
            
            # Jeśli dane są aktualne, zwróć z cache
            if current_time - last_update < refresh_interval / 2:
                return self.market_data_cache[cache_key]
        
        try:
            # Konwersja interwałów do formatu CCXT
            ccxt_interval = {
                '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
                '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
            }.get(interval, '1h')
            
            data = None
            
            # Spróbuj użyć MarketDataFetcher z projektu
            if self.data_fetcher is not None and exchange == 'bybit':
                try:
                    df = self.data_fetcher.fetch_data(symbol=symbol, interval=interval, limit=limit)
                    if not df.empty:
                        # Konwersja DataFrame do formatu OHLCV
                        data = []
                        for _, row in df.iterrows():
                            # Zamień timestamp z pandas datetime na timestamp unix w ms
                            timestamp = int(row['timestamp'].timestamp() * 1000) if isinstance(row['timestamp'], pd.Timestamp) else row['timestamp']
                            data.append([
                                timestamp,
                                float(row['open']),
                                float(row['high']),
                                float(row['low']),
                                float(row['close']),
                                float(row['volume'])
                            ])
                        logger.info(f"Pobrano dane za pomocą MarketDataFetcher: {symbol} {interval} ({len(data)} rekordów)")
                except Exception as e:
                    logger.error(f"Błąd podczas pobierania danych z MarketDataFetcher: {e}")
                    data = None
            
            # Jeśli nie udało się pobrać danych z MarketDataFetcher, użyj CCXT
            if data is None:
                if exchange in self.ccxt_clients:
                    try:
                        client = self.ccxt_clients[exchange]
                        if not client.has['fetchOHLCV']:
                            raise Exception(f"Exchange {exchange} nie obsługuje fetchOHLCV")
                        
                        data = await asyncio.to_thread(
                            client.fetch_ohlcv,
                            symbol=symbol,
                            timeframe=ccxt_interval,
                            limit=limit
                        )
                        logger.info(f"Pobrano dane za pomocą CCXT ({exchange}): {symbol} {interval} ({len(data)} rekordów)")
                    except Exception as e:
                        logger.error(f"Błąd podczas pobierania danych OHLCV poprzez CCXT ({exchange}): {e}")
                        data = None
            
            # Fallback - generuj dane symulowane jeśli wszystkie metody zawiodły
            if data is None:
                logger.warning(f"Generowanie symulowanych danych dla {symbol} {interval}")
                data = self._generate_mock_data(symbol, interval, limit)
            
            # Aktualizuj cache
            self.market_data_cache[cache_key] = data
            self.last_update_time[cache_key] = current_time
            
            # Zapisz również do bazy danych
            await self._save_ohlcv_to_db(symbol, data, interval, exchange)
            
            return data
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania danych OHLCV: {e}")
            # W przypadku błędu generuj dane symulowane
            mock_data = self._generate_mock_data(symbol, interval, limit)
            return mock_data
    
    def _generate_mock_data(self, symbol: str, interval: str, limit: int = 100):
        """Generuje symulowane dane OHLCV"""
        end_time = int(time.time() * 1000)
        
        # Określ wielkość interwału w milisekundach
        if interval.endswith('m'):
            interval_ms = int(interval[:-1]) * 60 * 1000
        elif interval.endswith('h'):
            interval_ms = int(interval[:-1]) * 60 * 60 * 1000
        elif interval.endswith('d'):
            interval_ms = 24 * 60 * 60 * 1000
        else:
            interval_ms = 60 * 60 * 1000  # domyślnie 1h
        
        # Określ bazową cenę bazując na symbolu
        base_price = 0
        if 'BTC' in symbol:
            base_price = 50000 + np.random.normal(0, 100)
        elif 'ETH' in symbol:
            base_price = 3000 + np.random.normal(0, 50)
        elif 'BNB' in symbol:
            base_price = 500 + np.random.normal(0, 10)
        else:
            base_price = 100 + np.random.normal(0, 5)
        
        data = []
        price = base_price
        
        for i in range(limit):
            timestamp = end_time - (limit - i) * interval_ms
            
            # Dodaj losowy ruch ceny
            change = np.random.normal(0, price * 0.01)  # 1% odchylenie standardowe
            price += change
            
            # Generuj OHLCV
            open_price = price
            close_price = price + np.random.normal(0, price * 0.005)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, price * 0.003))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, price * 0.003))
            volume = np.random.uniform(10, 100) * (price / 100)
            
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            price = close_price
        
        return data
    
    async def _save_ohlcv_to_db(self, symbol: str, data: List, interval: str, exchange: str):
        """Zapisuje dane OHLCV do bazy danych"""
        try:
            db = SessionLocal()
            
            # Ogranicz zapis do bazy danych do np. 10 ostatnich rekordów
            for candle in data[-10:]:
                timestamp, open_price, high, low, close, volume = candle
                
                # Konwertuj timestamp na datetime
                candle_time = datetime.fromtimestamp(timestamp / 1000)
                
                # Sprawdź czy rekord już istnieje
                existing = db.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp == candle_time,
                    MarketData.interval == interval,
                    MarketData.exchange == exchange
                ).first()
                
                if not existing:
                    # Dodaj nowy rekord
                    db_candle = MarketData(
                        symbol=symbol,
                        timestamp=candle_time,
                        open=open_price,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        interval=interval,
                        exchange=exchange
                    )
                    db.add(db_candle)
            
            db.commit()
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania danych OHLCV do bazy danych: {e}")
        finally:
            db.close()
    
    async def get_ticker(self, symbol: str, exchange: str = 'bybit'):
        """Pobiera aktualną cenę i informacje o tickerze"""
        try:
            if exchange in self.ccxt_clients:
                client = self.ccxt_clients[exchange]
                ticker = await asyncio.to_thread(client.fetch_ticker, symbol)
                return ticker
            else:
                raise Exception(f"Brak klienta dla giełdy {exchange}")
        except Exception as e:
            logger.error(f"Błąd podczas pobierania danych tickera: {e}")
            # Fallback do danych symulowanych
            last_price = 0
            if 'BTC' in symbol:
                last_price = 50000 + np.random.normal(0, 100)
            elif 'ETH' in symbol:
                last_price = 3000 + np.random.normal(0, 50)
            elif 'BNB' in symbol:
                last_price = 500 + np.random.normal(0, 10)
            else:
                last_price = 100 + np.random.normal(0, 5)
                
            return {
                'symbol': symbol,
                'last': last_price,
                'bid': last_price * 0.999,
                'ask': last_price * 1.001,
                'high': last_price * 1.02,
                'low': last_price * 0.98,
                'volume': np.random.uniform(100, 1000),
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat()
            }
    
    async def subscribe_to_websocket(self, symbol: str, exchange: str = 'bybit'):
        """
        Tworzy subskrypcję WebSocket do danych rynkowych
        To jest przykładowa implementacja
        """
        # Ta metoda będzie implementowana w przyszłości dla danych czasu rzeczywistego
        pass

# Klasa dla zarządzania strategiami tradingowymi
class TradingStrategyService:
    def __init__(self):
        # Inicjalizujemy StrategyManager z projektu
        try:
            self.strategy_manager = StrategyManager(
                strategies={
                    "trend_following": {"name": "Trend Following", "enabled": True},
                    "mean_reversion": {"name": "Mean Reversion", "enabled": False},
                    "breakout": {"name": "Breakout", "enabled": True}
                },
                exposure_limits={
                    "trend_following": 0.5,
                    "mean_reversion": 0.3,
                    "breakout": 0.4
                }
            )
            logger.info("Zainicjalizowano StrategyManager")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji StrategyManager: {e}")
            self.strategy_manager = None
        
        # Inicjalizacja modeli AI z projektu
        try:
            self.model_recognizer = ModelRecognizer()
            logger.info("Zainicjalizowano ModelRecognizer")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji ModelRecognizer: {e}")
            self.model_recognizer = None
        
        try:
            self.anomaly_detector = AnomalyDetector()
            logger.info("Zainicjalizowano AnomalyDetector")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji AnomalyDetector: {e}")
            self.anomaly_detector = None
        
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("Zainicjalizowano SentimentAnalyzer")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji SentimentAnalyzer: {e}")
            self.sentiment_analyzer = None
    
    def generate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generuje wskaźniki techniczne dla danych świecowych"""
        try:
            # Upewnij się, że DataFrame ma odpowiednie kolumny
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                # Jeśli brakuje kolumn, spróbuj przekształcić format
                if len(df.columns) >= 5:
                    df.columns = required_columns[:len(df.columns)]
                else:
                    raise ValueError(f"DataFrame nie zawiera wymaganych kolumn: {required_columns}")
            
            # Wskaźniki trendu
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bollinger_mavg'] = bollinger.bollinger_mavg()
            df['bollinger_high'] = bollinger.bollinger_hband()
            df['bollinger_low'] = bollinger.bollinger_lband()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ATR (Average True Range)
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Objętość
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # Dodatkowe wskaźniki z pandas_ta jeśli dostępne
            if pta:
                # Dodaj wskaźniki z pandas_ta
                try:
                    # Ichimoku Cloud
                    ichimoku = pta.ichimoku(df['high'], df['low'], df['close'])
                    df = pd.concat([df, ichimoku], axis=1)
                    
                    # Hull Moving Average
                    df['hma_20'] = pta.hma(df['close'], length=20)
                    
                    # Supertrend
                    supertrend = pta.supertrend(df['high'], df['low'], df['close'])
                    df = pd.concat([df, supertrend], axis=1)
                except Exception as e:
                    logger.error(f"Błąd podczas generowania wskaźników pandas_ta: {e}")
            
            return df
        
        except Exception as e:
            logger.error(f"Błąd podczas generowania wskaźników technicznych: {e}")
            return df
    
    def analyze_market(self, df: pd.DataFrame, strategy: str = 'trend_following') -> Dict:
        """Analizuje rynek i generuje sygnały na podstawie wybranej strategii"""
        try:
            # Dodaj wskaźniki techniczne
            df = self.generate_technical_indicators(df)
            
            # Inicjalizuj wynik
            result = {
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'indicators': {},
                'price': df['close'].iloc[-1],
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
            
            # Zbierz wartości wskaźników
            for col in df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    try:
                        result['indicators'][col] = float(df[col].iloc[-1])
                    except:
                        pass
            
            # Zastosuj strategię
            if strategy == 'trend_following':
                # Prosty sygnał oparty na przecięciu SMA
                if df['close'].iloc[-1] > df['sma_50'].iloc[-1] and df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
                    result['signal'] = 'BUY'
                    result['confidence'] = min(0.9, 0.5 + 0.1 * ((df['close'].iloc[-1] / df['sma_50'].iloc[-1]) - 1) * 10)
                elif df['close'].iloc[-1] < df['sma_50'].iloc[-1] and df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1]:
                    result['signal'] = 'SELL'
                    result['confidence'] = min(0.9, 0.5 + 0.1 * ((df['sma_50'].iloc[-1] / df['close'].iloc[-1]) - 1) * 10)
            
            elif strategy == 'mean_reversion':
                # Strategia powrotu do średniej z RSI
                if df['rsi'].iloc[-1] < 30:
                    result['signal'] = 'BUY'
                    result['confidence'] = 0.5 + (30 - df['rsi'].iloc[-1]) / 100
                elif df['rsi'].iloc[-1] > 70:
                    result['signal'] = 'SELL'
                    result['confidence'] = 0.5 + (df['rsi'].iloc[-1] - 70) / 100
            
            elif strategy == 'breakout':
                # Strategia przełamania na podstawie Bollinger Bands
                if df['close'].iloc[-1] > df['bollinger_high'].iloc[-2]:
                    result['signal'] = 'BUY'
                    result['confidence'] = 0.5 + min(0.4, ((df['close'].iloc[-1] / df['bollinger_high'].iloc[-2]) - 1) * 5)
                elif df['close'].iloc[-1] < df['bollinger_low'].iloc[-2]:
                    result['signal'] = 'SELL'
                    result['confidence'] = 0.5 + min(0.4, ((df['bollinger_low'].iloc[-2] / df['close'].iloc[-1]) - 1) * 5)
            
            # Dodaj analizę z modeli AI
            try:
                # Dodaj rozpoznanie wzorca
                if self.model_recognizer:
                    pattern = self.model_recognizer.identify_model_type(df)
                    if pattern and isinstance(pattern, dict):
                        result['pattern'] = pattern
                        # Dostosuj pewność sygnału na podstawie rozpoznanego wzorca
                        if 'confidence' in pattern and pattern.get('type') == 'bullish' and result['signal'] == 'BUY':
                            result['confidence'] = min(0.95, result['confidence'] + 0.1)
                        elif 'confidence' in pattern and pattern.get('type') == 'bearish' and result['signal'] == 'SELL':
                            result['confidence'] = min(0.95, result['confidence'] + 0.1)
                
                # Dodaj wykrywanie anomalii
                if self.anomaly_detector and hasattr(self.anomaly_detector, 'predict'):
                    # Przygotuj dane dla detektora anomalii
                    features = df[['close', 'volume']].values if 'volume' in df.columns else df[['close']].values
                    anomaly_score = self.anomaly_detector.predict(features)
                    result['anomaly_score'] = float(anomaly_score[-1]) if isinstance(anomaly_score, (list, np.ndarray)) else float(anomaly_score)
                    
                    # Jeśli wykryto anomalię, obniż pewność
                    if result['anomaly_score'] > 0.7:
                        result['confidence'] = max(0.3, result['confidence'] - 0.2)
                        result['anomaly_detected'] = True
                    else:
                        result['anomaly_detected'] = False
                
                # Dodaj analizę sentymentu
                if self.sentiment_analyzer:
                    sentiment = self.sentiment_analyzer.analyze()
                    result['sentiment'] = sentiment
                    
                    # Dostosuj pewność sygnału na podstawie sentymentu
                    sentiment_value = sentiment.get('value', 0)
                    if sentiment_value > 0.3 and result['signal'] == 'BUY':
                        result['confidence'] = min(0.95, result['confidence'] + 0.1)
                    elif sentiment_value < -0.3 and result['signal'] == 'SELL':
                        result['confidence'] = min(0.95, result['confidence'] + 0.1)
                    elif abs(sentiment_value) > 0.3 and result['signal'] != 'NEUTRAL':
                        # Sentyment przeciwny do sygnału
                        result['confidence'] = max(0.3, result['confidence'] - 0.1)
            
            except Exception as e:
                logger.error(f"Błąd podczas analizy AI: {e}")
            
            return result
        
        except Exception as e:
            logger.error(f"Błąd podczas analizy rynku: {e}")
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'error': str(e),
                'price': df['close'].iloc[-1] if not df.empty and 'close' in df.columns else 0,
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_backtest(self, request: BacktestRequest, db: Session) -> Dict:
        """Przeprowadza backtesting dla wybranej strategii"""
        try:
            # Pobierz dane historyczne
            market_service = MarketDataService()
            end_date = request.end_date or datetime.now()
            
            # Oblicz start_timestamp i end_timestamp w ms
            start_timestamp = int(request.start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            # Pobierz dane OHLCV
            raw_data = await market_service.fetch_ohlcv(
                symbol=request.symbol,
                interval=request.timeframe,
                limit=1000,  # Pobierz maksymalną ilość danych
                exchange='bybit'
            )
            
            # Filtruj dane według zakresu dat
            filtered_data = [candle for candle in raw_data if start_timestamp <= candle[0] <= end_timestamp]
            
            if not filtered_data:
                return {
                    'success': False,
                    'message': 'Brak danych w wybranym zakresie dat'
                }
            
            # Konwertuj na DataFrame
            df = pd.DataFrame(filtered_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Dodaj wskaźniki techniczne
            df = self.generate_technical_indicators(df)
            
            # Zainicjuj zmienne backtestingu
            initial_capital = request.initial_capital
            current_capital = initial_capital
            position = None
            position_size = 0
            position_entry_price = 0
            trades = []
            
            # Przeprowadź backtest
            for i in range(20, len(df)):  # Rozpocznij od 20, aby wskaźniki miały dane
                current_time = df.index[i]
                current_price = df['close'].iloc[i]
                
                # Twórz DataFrame dla danych tylko do bieżącego momentu
                current_df = df.iloc[:i+1].copy()
                
                # Przeprowadź analizę na podstawie aktualnych danych
                analysis = self.analyze_market(current_df, strategy=request.strategy)
                signal = analysis['signal']
                confidence = analysis['confidence']
                
                # Logika tradingu dla backtestingu
                if position is None:  # Brak pozycji
                    if signal == 'BUY' and confidence > 0.7:
                        # Otwórz pozycję długą
                        position = 'LONG'
                        position_size = (current_capital * 0.95) / current_price  # 95% kapitału
                        position_entry_price = current_price
                        trades.append({
                            'timestamp': current_time,
                            'action': 'LONG',
                            'price': current_price,
                            'size': position_size,
                            'capital': current_capital
                        })
                    elif signal == 'SELL' and confidence > 0.7:
                        # Otwórz pozycję krótką (w prawdziwym backtestingu potrzebne byłyby więcej szczegółów)
                        position = 'SHORT'
                        position_size = (current_capital * 0.95) / current_price  # 95% kapitału
                        position_entry_price = current_price
                        trades.append({
                            'timestamp': current_time,
                            'action': 'SHORT',
                            'price': current_price,
                            'size': position_size,
                            'capital': current_capital
                        })
                else:  # Mamy otwartą pozycję
                    if position == 'LONG':
                        if signal == 'SELL' and confidence > 0.7:
                            # Zamknij pozycję długą
                            pnl = position_size * (current_price - position_entry_price)
                            current_capital += pnl
                            trades.append({
                                'timestamp': current_time,
                                'action': 'CLOSE LONG',
                                'price': current_price,
                                'size': position_size,
                                'pnl': pnl,
                                'capital': current_capital
                            })
                            position = None
                            position_size = 0
                    elif position == 'SHORT':
                        if signal == 'BUY' and confidence > 0.7:
                            # Zamknij pozycję krótką
                            pnl = position_size * (position_entry_price - current_price)
                            current_capital += pnl
                            trades.append({
                                'timestamp': current_time,
                                'action': 'CLOSE SHORT',
                                'price': current_price,
                                'size': position_size,
                                'pnl': pnl,
                                'capital': current_capital
                            })
                            position = None
                            position_size = 0
            
            # Oblicz podsumowanie backtestu
            winning_trades = [trade for trade in trades if 'pnl' in trade and trade['pnl'] > 0]
            losing_trades = [trade for trade in trades if 'pnl' in trade and trade['pnl'] <= 0]
            
            total_trades = len(winning_trades) + len(losing_trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            profit = current_capital - initial_capital
            profit_percentage = (profit / initial_capital) * 100
            
            # Oblicz maksymalny drawdown
            max_capital = initial_capital
            max_drawdown = 0
            for trade in trades:
                capital = trade.get('capital', 0)
                if capital > max_capital:
                    max_capital = capital
                drawdown = (max_capital - capital) / max_capital * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Utwórz wykres dla backtestu
            chart_path = self._create_backtest_chart(df, trades, request.symbol, request.strategy)
            
            # Zapisz wyniki do bazy danych
            backtest_id = f"{request.symbol}_{request.strategy}_{int(time.time())}"
            backtest_path = f"reports/backtest_{backtest_id}.json"
            os.makedirs("reports", exist_ok=True)
            
            backtest_data = {
                'id': backtest_id,
                'symbol': request.symbol,
                'strategy': request.strategy,
                'timeframe': request.timeframe,
                'start_date': request.start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'initial_capital': initial_capital,
                'final_capital': current_capital,
                'profit': profit,
                'profit_percentage': profit_percentage,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'trades': trades,
                'parameters': request.parameters,
                'timestamp': datetime.now().isoformat(),
                'chart_path': chart_path
            }
            
            # Zapisz wyniki do pliku JSON
            with open(backtest_path, 'w') as f:
                json.dump(backtest_data, f, indent=4, default=str)
                
            # Zapisz wyniki do logów systemowych
            system_log = SystemLog(
                level="INFO",
                message=f"Zakończono backtest dla {request.symbol} używając strategii {request.strategy}",
                component="Backtest",
                details=json.dumps({
                    'profit_percentage': profit_percentage,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'total_trades': total_trades
                })
            )
            db.add(system_log)
            db.commit()
            
            return {
                'success': True,
                'summary': {
                    'initial_capital': initial_capital,
                    'final_capital': current_capital,
                    'profit': profit,
                    'profit_percentage': profit_percentage,
                    'total_trades': total_trades,
                    'wins': len(winning_trades),
                    'losses': len(losing_trades),
                    'win_rate': win_rate * 100,
                    'max_drawdown': max_drawdown
                },
                'trades': trades,
                'chart_path': chart_path,
                'backtest_id': backtest_id
            }
        
        except Exception as e:
            logger.error(f"Błąd podczas backtestingu: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_backtest_chart(self, df: pd.DataFrame, trades: List[Dict], symbol: str, strategy: str) -> str:
        """Tworzy wykres dla wyników backtestingu"""
        try:
            # Utwórz wykres
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=('Cena i sygnały', 'Wskaźniki', 'Kapitał')
            )
            
            # Dodaj świece
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Świece'
                ),
                row=1, col=1
            )
            
            # Dodaj średnie ruchome
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_20'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_50'],
                        name='SMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # Dodaj Bollinger Bands
            if 'bollinger_high' in df.columns and 'bollinger_low' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['bollinger_high'],
                        name='BB górne',
                        line=dict(color='rgba(250, 128, 114, 0.5)', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['bollinger_low'],
                        name='BB dolne',
                        line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                        fill='tonexty' if 'bollinger_high' in df.columns else None
                    ),
                    row=1, col=1
                )
            
            # Dodaj sygnały i transakcje
            buy_x = []
            buy_y = []
            sell_x = []
            sell_y = []
            
            capital_x = [df.index[0]]
            capital_y = [trades[0]['capital'] if trades else 0]
            
            for trade in trades:
                timestamp = trade['timestamp']
                price = trade['price']
                
                if trade['action'] == 'LONG' or trade['action'] == 'CLOSE SHORT':
                    buy_x.append(timestamp)
                    buy_y.append(price)
                elif trade['action'] == 'SHORT' or trade['action'] == 'CLOSE LONG':
                    sell_x.append(timestamp)
                    sell_y.append(price)
                
                capital_x.append(timestamp)
                capital_y.append(trade['capital'])
            
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode='markers',
                    name='Kupno',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode='markers',
                    name='Sprzedaż',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
            
            # Dodaj wskaźniki
            if 'rsi' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['rsi'],
                        name='RSI',
                        line=dict(color='purple', width=1)
                    ),
                    row=2, col=1
                )
                
                # Dodaj linie RSI 30 i 70
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[0], df.index[-1]],
                        y=[30, 30],
                        name='RSI 30',
                        line=dict(color='green', width=1, dash='dash')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[0], df.index[-1]],
                        y=[70, 70],
                        name='RSI 70',
                        line=dict(color='red', width=1, dash='dash')
                    ),
                    row=2, col=1
                )
            
            # Dodaj MACD
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['macd'],
                        name='MACD',
                        line=dict(color='blue', width=1)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['macd_signal'],
                        name='MACD Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=2, col=1
                )
            
            # Dodaj wykres kapitału
            fig.add_trace(
                go.Scatter(
                    x=capital_x,
                    y=capital_y,
                    name='Kapitał',
                    fill='tozeroy',
                    line=dict(color='green', width=2)
                ),
                row=3, col=1
            )
            
            # Aktualizuj układ wykresu
            fig.update_layout(
                title=f'Backtest {symbol} - {strategy}',
                xaxis_title='Data',
                yaxis_title='Cena',
                xaxis3_title='Data',
                yaxis3_title='Kapitał',
                height=900,
                template='plotly_dark',
                legend=dict(orientation='h', y=1.05),
                margin=dict(l=50, r=50, t=80, b=50),
            )
            
            fig.update_yaxes(title_text='RSI / MACD', row=2, col=1)
            fig.update_yaxes(title_text='Kapitał', row=3, col=1)
            
            # Zapisz wykres
            chart_path = f"static/charts/backtest_{symbol}_{strategy}_{int(time.time())}.png"
            fig.write_image(chart_path)
            
            return chart_path
        
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia wykresu backtestingu: {e}", exc_info=True)
            return ""

# Usługi i moduły API
market_service = MarketDataService()
trading_service = TradingStrategyService()

# Endpointy API
@app.get("/api/status")
async def get_status():
    """Sprawdza status API i komponentów systemu"""
    try:
        # Sprawdź połączenia z giełdami
        bybit_status = "online" if market_service.bybit_client is not None else "offline"
        binance_status = "online" if market_service.binance_client is not None else "offline"
        ccxt_status = "online" if market_service.ccxt_clients else "offline"
        
        # Sprawdź składniki analityczne
        strategy_manager_status = "online" if trading_service.strategy_manager is not None else "offline"
        model_recognizer_status = "online" if trading_service.model_recognizer is not None else "offline"
        anomaly_detector_status = "online" if trading_service.anomaly_detector is not None else "offline"
        sentiment_analyzer_status = "online" if trading_service.sentiment_analyzer is not None else "offline"
        
        return {
            "status": "online",
            "time": datetime.now().isoformat(),
            "components": {
                "bybit_api": bybit_status,
                "binance_api": binance_status,
                "ccxt": ccxt_status,
                "strategy_manager": strategy_manager_status,
                "model_recognizer": model_recognizer_status,
                "anomaly_detector": anomaly_detector_status,
                "sentiment_analyzer": sentiment_analyzer_status
            }
        }
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania statusu: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/symbols")
async def get_market_symbols():
    """Pobiera dostępne symbole rynkowe"""
    try:
        symbols = []
        
        # Dodaj typowe pary walutowe dla kryptowalut
        default_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
        
        # Spróbuj pobrać pary z CCXT
        try:
            if 'bybit' in market_service.ccxt_clients:
                markets = await asyncio.to_thread(market_service.ccxt_clients['bybit'].fetch_markets)
                symbols = [market['symbol'] for market in markets if 'USDT' in market['symbol']]
                logger.info(f"Pobrano {len(symbols)} symboli z CCXT Bybit")
        except Exception as e:
            logger.error(f"Błąd podczas pobierania symboli z CCXT: {e}")
            symbols = default_symbols
        
        # Jeśli nie udało się pobrać symboli, użyj domyślnych
        if not symbols:
            symbols = default_symbols
        
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania symboli: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/timeframes")
async def get_market_timeframes():
    """Pobiera dostępne timeframe'y"""
    timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"]
    return {"timeframes": timeframes}

@app.get("/api/market/ohlcv")
async def get_market_ohlcv(
    symbol: str = Query(..., description="Symbol rynkowy, np. BTCUSDT"),
    interval: str = Query("1h", description="Interwał danych"),
    limit: int = Query(100, description="Liczba świec"),
    exchange: str = Query("bybit", description="Nazwa giełdy")
):
    """Pobiera dane OHLCV dla wybranego symbolu"""
    try:
        data = await market_service.fetch_ohlcv(symbol, interval, limit, exchange)
        
        # Konwertuj dane na format czytelny dla człowieka
        formatted_data = []
        for candle in data:
            timestamp, open_price, high, low, close, volume = candle
            formatted_data.append({
                "timestamp": datetime.fromtimestamp(timestamp / 1000).isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
        
        return {"data": formatted_data, "symbol": symbol, "interval": interval}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych OHLCV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/ticker")
async def get_market_ticker(
    symbol: str = Query(..., description="Symbol rynkowy, np. BTCUSDT"),
    exchange: str = Query("bybit", description="Nazwa giełdy")
):
    """Pobiera aktualną cenę i informacje o tickerze"""
    try:
        ticker = await market_service.get_ticker(symbol, exchange)
        return ticker
    except Exception as e:
        logger.error(f"Błąd podczas pobierania tickera: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/analyze")
async def analyze_market(
    symbol: str = Query(..., description="Symbol rynkowy, np. BTCUSDT"),
    interval: str = Query("1h", description="Interwał danych"),
    strategy: str = Query("trend_following", description="Strategia analizy"),
    exchange: str = Query("bybit", description="Nazwa giełdy")
):
    """Analizuje rynek i zwraca sygnały dla wybranej strategii"""
    try:
        # Pobierz dane OHLCV
        data = await market_service.fetch_ohlcv(symbol, interval, 100, exchange)
        
        # Konwertuj na DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Analizuj rynek
        result = trading_service.analyze_market(df, strategy)
        
        # Zaktualizuj metadane
        result.update({
            "symbol": symbol,
            "interval": interval,
            "exchange": exchange
        })
        
        return result
    except Exception as e:
        logger.error(f"Błąd podczas analizy rynku: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    """Przeprowadza backtesting dla wybranej strategii"""
    try:
        result = await trading_service.run_backtest(request, db)
        return result
    except Exception as e:
        logger.error(f"Błąd podczas backtestingu: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trades")
async def create_trade(trade: TradeCreate, db: Session = Depends(get_db)):
    """Tworzy nową transakcję"""
    try:
        db_trade = Trade(
            symbol=trade.symbol,
            side=trade.side,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            entry_time=datetime.now(),
            strategy=trade.strategy,
            exchange=trade.exchange,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            notes=trade.notes
        )
        db.add(db_trade)
        db.commit()
        db.refresh(db_trade)
        
        # Powiadom klientów WebSocket
        await manager.broadcast(
            {
                "type": "new_trade",
                "data": {
                    "id": db_trade.id,
                    "symbol": db_trade.symbol,
                    "side": db_trade.side,
                    "entry_price": db_trade.entry_price,
                    "quantity": db_trade.quantity,
                    "entry_time": db_trade.entry_time.isoformat(),
                    "strategy": db_trade.strategy,
                    "exchange": db_trade.exchange,
                    "status": db_trade.status
                }
            },
            "trades"
        )
        
        return db_trade
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia transakcji: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/trades/{trade_id}")
async def update_trade(trade_id: int, trade_update: TradeUpdate, db: Session = Depends(get_db)):
    """Aktualizuje istniejącą transakcję"""
    try:
        db_trade = db.query(Trade).filter(Trade.id == trade_id).first()
        if not db_trade:
            raise HTTPException(status_code=404, detail="Transakcja nie znaleziona")
        
        # Aktualizuj pola
        for key, value in trade_update.dict(exclude_unset=True).items():
            setattr(db_trade, key, value)
        
        # Jeśli status zmieniony na CLOSED, oblicz P/L
        if trade_update.status == "CLOSED" and trade_update.exit_price:
            if db_trade.side == "BUY":
                db_trade.profit_loss = (trade_update.exit_price - db_trade.entry_price) * db_trade.quantity
                db_trade.profit_loss_percent = (trade_update.exit_price / db_trade.entry_price - 1) * 100
            else:  # SELL
                db_trade.profit_loss = (db_trade.entry_price - trade_update.exit_price) * db_trade.quantity
                db_trade.profit_loss_percent = (db_trade.entry_price / trade_update.exit_price - 1) * 100
            
            if not db_trade.exit_time:
                db_trade.exit_time = datetime.now()
        
        db.commit()
        db.refresh(db_trade)
        
        # Powiadom klientów WebSocket
        await manager.broadcast(
            {
                "type": "update_trade",
                "data": {
                    "id": db_trade.id,
                    "symbol": db_trade.symbol,
                    "side": db_trade.side,
                    "entry_price": db_trade.entry_price,
                    "exit_price": db_trade.exit_price,
                    "quantity": db_trade.quantity,
                    "entry_time": db_trade.entry_time.isoformat(),
                    "exit_time": db_trade.exit_time.isoformat() if db_trade.exit_time else None,
                    "profit_loss": db_trade.profit_loss,
                    "profit_loss_percent": db_trade.profit_loss_percent,
                    "status": db_trade.status,
                    "strategy": db_trade.strategy,
                    "exchange": db_trade.exchange
                }
            },
            "trades"
        )
        
        return db_trade
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji transakcji: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades")
async def get_trades(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Pobiera listę transakcji"""
    try:
        query = db.query(Trade)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        if status:
            query = query.filter(Trade.status == status)
        if strategy:
            query = query.filter(Trade.strategy == strategy)
        
        trades = query.order_by(Trade.entry_time.desc()).limit(limit).all()
        return {"trades": trades}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania transakcji: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signals")
async def create_signal(signal: SignalCreate, db: Session = Depends(get_db)):
    """Tworzy nowy sygnał tradingowy"""
    try:
        db_signal = SignalEvent(
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            price=signal.price,
            strategy=signal.strategy,
            indicator_values=json.dumps(signal.indicator_values),
            confidence=signal.confidence,
            notes=signal.notes
        )
        db.add(db_signal)
        db.commit()
        db.refresh(db_signal)
        
        # Powiadom klientów WebSocket
        await manager.broadcast(
            {
                "type": "new_signal",
                "data": {
                    "id": db_signal.id,
                    "symbol": db_signal.symbol,
                    "signal_type": db_signal.signal_type,
                    "price": db_signal.price,
                    "timestamp": db_signal.timestamp.isoformat(),
                    "strategy": db_signal.strategy,
                    "confidence": db_signal.confidence,
                    "executed": db_signal.executed
                }
            },
            "signals"
        )
        
        return db_signal
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia sygnału: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals")
async def get_signals(
    symbol: Optional[str] = None,
    signal_type: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Pobiera listę sygnałów tradingowych"""
    try:
        query = db.query(SignalEvent)
        
        if symbol:
            query = query.filter(SignalEvent.symbol == symbol)
        if signal_type:
            query = query.filter(SignalEvent.signal_type == signal_type)
        if strategy:
            query = query.filter(SignalEvent.strategy == strategy)
        
        signals = query.order_by(SignalEvent.timestamp.desc()).limit(limit).all()
        
        # Konwertuj indicator_values z JSON string
        result = []
        for signal in signals:
            signal_dict = {
                "id": signal.id,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "price": signal.price,
                "timestamp": signal.timestamp.isoformat(),
                "strategy": signal.strategy,
                "confidence": signal.confidence,
                "executed": signal.executed,
                "notes": signal.notes
            }
            
            try:
                signal_dict["indicator_values"] = json.loads(signal.indicator_values)
            except:
                signal_dict["indicator_values"] = {}
            
            result.append(signal_dict)
        
        return {"signals": result}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania sygnałów: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Pobiera logi systemowe"""
    try:
        query = db.query(SystemLog)
        
        if level:
            query = query.filter(SystemLog.level == level)
        if component:
            query = query.filter(SystemLog.component == component)
        
        logs = query.order_by(SystemLog.timestamp.desc()).limit(limit).all()
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania logów: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment")
async def get_sentiment():
    """Pobiera aktualną analizę sentymentu rynkowego"""
    try:
        if trading_service.sentiment_analyzer:
            sentiment = trading_service.sentiment_analyzer.analyze()
            return sentiment
        else:
            return {
                "value": 0,
                "analysis": "Neutralny",
                "sources": {
                    "twitter": 0,
                    "news": 0,
                    "forum": 0
                },
                "error": "Analizator sentymentu nie jest dostępny"
            }
    except Exception as e:
        logger.error(f"Błąd podczas pobierania analizy sentymentu: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies")
async def get_strategies():
    """Pobiera dostępne strategie tradingowe"""
    try:
        strategies = [
            {
                "id": "trend_following",
                "name": "Trend Following",
                "description": "Strategia podążająca za trendem, wykorzystująca SMA, EMA i MACD",
                "indicators": ["SMA", "EMA", "MACD"],
                "timeframes": ["1h", "4h", "1d"]
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "description": "Strategia powrotu do średniej, wykorzystująca RSI, Bollinger Bands",
                "indicators": ["RSI", "Bollinger Bands"],
                "timeframes": ["15m", "1h", "4h"]
            },
            {
                "id": "breakout",
                "name": "Breakout",
                "description": "Strategia przełamania, oparta na wykrywaniu przebić poziomów wsparcia/oporu",
                "indicators": ["Bollinger Bands", "ATR", "Volume"],
                "timeframes": ["1h", "4h", "1d"]
            },
            {
                "id": "momentum",
                "name": "Momentum",
                "description": "Strategia pędu, wykorzystująca wskaźniki momentum i trendu",
                "indicators": ["RSI", "MACD", "Stochastic"],
                "timeframes": ["5m", "15m", "1h", "4h"]
            },
            {
                "id": "volatility_breakout",
                "name": "Volatility Breakout",
                "description": "Strategia przełamania zmienności, używająca ATR do określenia poziomów",
                "indicators": ["ATR", "Bollinger Bands", "Keltner Channels"],
                "timeframes": ["1h", "4h", "1d"]
            }
        ]
        
        # Jeśli dostępny jest StrategyManager, dodaj jego strategie
        if trading_service.strategy_manager:
            try:
                if hasattr(trading_service.strategy_manager, 'get_available_strategies'):
                    sm_strategies = trading_service.strategy_manager.get_available_strategies()
                    for strategy_id, strategy_info in sm_strategies.items():
                        # Sprawdź czy strategia już istnieje w liście
                        if not any(s['id'] == strategy_id for s in strategies):
                            strategies.append({
                                "id": strategy_id,
                                "name": strategy_info.get('name', strategy_id),
                                "description": strategy_info.get('description', ""),
                                "indicators": strategy_info.get('indicators', []),
                                "timeframes": strategy_info.get('timeframes', ["1h", "4h", "1d"])
                            })
            except Exception as e:
                logger.error(f"Błąd podczas pobierania strategii z StrategyManager: {e}")
        
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania strategii: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicators")
async def get_indicators():
    """Pobiera dostępne wskaźniki techniczne"""
    indicators = [
        {
            "id": "sma",
            "name": "Simple Moving Average",
            "description": "Średnia krocząca prosta",
            "parameters": ["window"],
            "category": "trend"
        },
        {
            "id": "ema",
            "name": "Exponential Moving Average",
            "description": "Wykładnicza średnia krocząca",
            "parameters": ["window"],
            "category": "trend"
        },
        {
            "id": "rsi",
            "name": "Relative Strength Index",
            "description": "Wskaźnik siły względnej",
            "parameters": ["window"],
            "category": "momentum"
        },
        {
            "id": "macd",
            "name": "Moving Average Convergence Divergence",
            "description": "Zbieżność/rozbieżność średnich kroczących",
            "parameters": ["fast_window", "slow_window", "signal_window"],
            "category": "trend"
        },
        {
            "id": "bollinger",
            "name": "Bollinger Bands",
            "description": "Wstęgi Bollingera",
            "parameters": ["window", "std_dev"],
            "category": "volatility"
        },
        {
            "id": "atr",
            "name": "Average True Range",
            "description": "Średni rzeczywisty zakres",
            "parameters": ["window"],
            "category": "volatility"
        },
        {
            "id": "stochastic",
            "name": "Stochastic Oscillator",
            "description": "Oscylator stochastyczny",
            "parameters": ["k_window", "d_window", "smooth_window"],
            "category": "momentum"
        },
        {
            "id": "obv",
            "name": "On-Balance Volume",
            "description": "Bilans wolumenu",
            "parameters": [],
            "category": "volume"
        }
    ]
    
    return {"indicators": indicators}

@app.get("/api/performance")
async def get_performance():
    """Pobiera statystyki wydajności systemu"""
    try:
        # Pobierz statystyki wydajności z bazy danych (dobre i złe transakcje)
        db = SessionLocal()
        
        total_trades = db.query(Trade).count()
        closed_trades = db.query(Trade).filter(Trade.status == "CLOSED").count()
        winning_trades = db.query(Trade).filter(Trade.status == "CLOSED", Trade.profit_loss > 0).count()
        losing_trades = db.query(Trade).filter(Trade.status == "CLOSED", Trade.profit_loss <= 0).count()
        
        # Oblicz zwrot
        total_profit = db.query(Trade).filter(
            Trade.status == "CLOSED", 
            Trade.profit_loss != None
        ).with_entities(
            func.sum(Trade.profit_loss).label("total_profit")
        ).scalar() or 0
        
        # Oblicz maksymalny drawdown
        max_drawdown = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        trades = db.query(Trade).filter(Trade.status == "CLOSED").order_by(Trade.exit_time).all()
        peak_equity = 10000  # Początkowy kapitał
        current_equity = peak_equity
        
        for trade in trades:
            if trade.profit_loss:
                current_equity += trade.profit_loss
                
                if trade.profit_loss > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                if current_equity > peak_equity:
                    peak_equity = current_equity
                else:
                    drawdown = (peak_equity - current_equity) / peak_equity * 100
                    max_drawdown = max(max_drawdown, drawdown)
        
        db.close()
        
        return {
            "total_trades": total_trades,
            "closed_trades": closed_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": (winning_trades / closed_trades * 100) if closed_trades > 0 else 0,
            "total_profit": total_profit,
            "max_drawdown": max_drawdown,
            "max_consecutive_losses": max_consecutive_losses,
            "active_strategies": 3  # Przykładowa wartość
        }
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statystyk wydajności: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart/{symbol}/{interval}")
async def get_chart_data(
    symbol: str,
    interval: str,
    indicators: str = Query("sma,ema,bollinger,rsi", description="Lista wskaźników oddzielonych przecinkami"),
    limit: int = 100
):
    """Pobiera dane dla wykresu"""
    try:
        # Pobierz dane OHLCV
        ohlcv_data = await market_service.fetch_ohlcv(symbol, interval, limit, "bybit")
        
        # Konwertuj na DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Dodaj wskaźniki techniczne
        df = trading_service.generate_technical_indicators(df)
        
        # Lista wskaźników do zwrócenia
        indicator_list = [ind.strip() for ind in indicators.split(',')]
        
        # Przygotuj dane do zwrócenia
        result = {
            "candlestick": [],
            "indicators": {}
        }
        
        # Dodaj dane świecowe
        for _, row in df.iterrows():
            result["candlestick"].append({
                "timestamp": row['timestamp'].isoformat(),
                "open": row['open'],
                "high": row['high'],
                "low": row['low'],
                "close": row['close'],
                "volume": row['volume']
            })
        
        # Dodaj wybrane wskaźniki
        for indicator in indicator_list:
            if indicator == 'sma':
                if 'sma_20' in df.columns and 'sma_50' in df.columns and 'sma_200' in df.columns:
                    result["indicators"]["sma_20"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['sma_20']}
                        for _, row in df.iterrows() if not pd.isna(row['sma_20'])
                    ]
                    result["indicators"]["sma_50"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['sma_50']}
                        for _, row in df.iterrows() if not pd.isna(row['sma_50'])
                    ]
                    result["indicators"]["sma_200"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['sma_200']}
                        for _, row in df.iterrows() if not pd.isna(row['sma_200'])
                    ]
            
            elif indicator == 'ema':
                if 'ema_12' in df.columns and 'ema_26' in df.columns:
                    result["indicators"]["ema_12"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['ema_12']}
                        for _, row in df.iterrows() if not pd.isna(row['ema_12'])
                    ]
                    result["indicators"]["ema_26"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['ema_26']}
                        for _, row in df.iterrows() if not pd.isna(row['ema_26'])
                    ]
            
            elif indicator == 'bollinger':
                if 'bollinger_high' in df.columns and 'bollinger_low' in df.columns and 'bollinger_mavg' in df.columns:
                    result["indicators"]["bollinger_high"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['bollinger_high']}
                        for _, row in df.iterrows() if not pd.isna(row['bollinger_high'])
                    ]
                    result["indicators"]["bollinger_low"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['bollinger_low']}
                        for _, row in df.iterrows() if not pd.isna(row['bollinger_low'])
                    ]
                    result["indicators"]["bollinger_mavg"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['bollinger_mavg']}
                        for _, row in df.iterrows() if not pd.isna(row['bollinger_mavg'])
                    ]
            
            elif indicator == 'rsi':
                if 'rsi' in df.columns:
                    result["indicators"]["rsi"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['rsi']}
                        for _, row in df.iterrows() if not pd.isna(row['rsi'])
                    ]
            
            elif indicator == 'macd':
                if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_diff' in df.columns:
                    result["indicators"]["macd"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['macd']}
                        for _, row in df.iterrows() if not pd.isna(row['macd'])
                    ]
                    result["indicators"]["macd_signal"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['macd_signal']}
                        for _, row in df.iterrows() if not pd.isna(row['macd_signal'])
                    ]
                    result["indicators"]["macd_diff"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['macd_diff']}
                        for _, row in df.iterrows() if not pd.isna(row['macd_diff'])
                    ]
            
            elif indicator == 'stochastic':
                if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                    result["indicators"]["stoch_k"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['stoch_k']}
                        for _, row in df.iterrows() if not pd.isna(row['stoch_k'])
                    ]
                    result["indicators"]["stoch_d"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['stoch_d']}
                        for _, row in df.iterrows() if not pd.isna(row['stoch_d'])
                    ]
            
            elif indicator == 'atr':
                if 'atr' in df.columns:
                    result["indicators"]["atr"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['atr']}
                        for _, row in df.iterrows() if not pd.isna(row['atr'])
                    ]
            
            elif indicator == 'obv':
                if 'obv' in df.columns:
                    result["indicators"]["obv"] = [
                        {"timestamp": row['timestamp'].isoformat(), "value": row['obv']}
                        for _, row in df.iterrows() if not pd.isna(row['obv'])
                    ]
        
        return result
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych wykresu: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpointy WebSocket
@app.websocket("/ws/market/{symbol}")
async def websocket_market_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint do przesyłania danych rynkowych w czasie rzeczywistym"""
    await manager.connect(websocket, "market_data")
    try:
        while True:
            # Pobierz aktualne dane tickera
            ticker = await market_service.get_ticker(symbol)
            
            # Wyślij dane do klienta
            await websocket.send_json({
                "type": "ticker",
                "data": ticker
            })
            
            # Poczekaj przed kolejnym wysłaniem
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "market_data")
    except Exception as e:
        logger.error(f"Błąd w websocket market: {e}")
        await websocket.close()
        manager.disconnect(websocket, "market_data")

@app.websocket("/ws/signals")
async def websocket_signals_endpoint(websocket: WebSocket):
    """WebSocket endpoint do przesyłania sygnałów tradingowych w czasie rzeczywistym"""
    await manager.connect(websocket, "signals")
    try:
        # Pobierz początkowe dane - ostatnie 10 sygnałów
        db = SessionLocal()
        signals = db.query(SignalEvent).order_by(SignalEvent.timestamp.desc()).limit(10).all()
        db.close()
        
        signals_data = []
        for signal in signals:
            signals_data.append({
                "id": signal.id,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "price": signal.price,
                "timestamp": signal.timestamp.isoformat(),
                "strategy": signal.strategy,
                "confidence": signal.confidence,
                "executed": signal.executed
            })
        
        # Wyślij początkowe dane
        await websocket.send_json({
            "type": "initial_signals",
            "data": signals_data
        })
        
        # Nasłuchuj nowych wydarzeń
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "signals")
    except Exception as e:
        logger.error(f"Błąd w websocket signals: {e}")
        await websocket.close()
        manager.disconnect(websocket, "signals")

# Funkcja do uruchomienia API w tle
def start_dashboard_api():
    """Uruchamia API dashboardu w tle"""
    import uvicorn
    
    port = int(os.getenv("API_PORT", 5100))
    
    # Uruchom API na innym porcie niż główna aplikacja Flask
    uvicorn.run(app, host="0.0.0.0", port=port)

# Funkcja główna
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 5100))
    
    logger.info(f"Uruchamianie Trading Dashboard API na porcie {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
