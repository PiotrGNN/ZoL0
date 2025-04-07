
"""
market_data_trainer.py
----------------------
Moduł do pobierania rzeczywistych danych rynkowych z Bybit i trenowania modeli AI.
"""

import logging
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# Konfiguracja logowania
logging.basicConfig(
    filename="logs/market_data_trainer.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("MarketDataTrainer")

class MarketDataTrainer:
    """
    Klasa do pobierania danych z Bybit i trenowania modeli AI na rzeczywistych danych.
    """
    
    def __init__(
        self,
        symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval: str = "1h",
        lookback_days: int = 30,
        test_size: float = 0.2,
        use_testnet: bool = True
    ):
        """
        Inicjalizacja trenera danych rynkowych.
        
        Args:
            symbols: Lista symboli do trenowania
            interval: Interwał danych (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            lookback_days: Liczba dni wstecz do pobrania danych
            test_size: Proporcja danych testowych
            use_testnet: Czy używać testnet zamiast mainnet
        """
        self.symbols = symbols
        self.interval = interval
        self.lookback_days = lookback_days
        self.test_size = test_size
        self.use_testnet = use_testnet
        
        # Inicjalizacja konektora Bybit
        try:
            from data.execution.bybit_connector import BybitConnector
            
            # Pobieramy klucze API z zmiennych środowiskowych
            self.api_key = os.getenv("BYBIT_API_KEY", "")
            self.api_secret = os.getenv("BYBIT_API_SECRET", "")
            
            # Sprawdzamy czy mamy klucze API
            self.simulation_mode = not (self.api_key and self.api_secret)
            
            # Inicjalizujemy konektor
            self.bybit = BybitConnector(
                api_key=self.api_key,
                api_secret=self.api_secret,
                use_testnet=self.use_testnet,
                simulation_mode=self.simulation_mode
            )
            
            logger.info(f"MarketDataTrainer zainicjalizowany dla {len(symbols)} symboli, interwał {interval}")
            
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji konektora Bybit: {e}")
            self.bybit = None
    
    def fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Pobiera historyczne dane dla danego symbolu.
        
        Args:
            symbol: Symbol handlowy (np. "BTCUSDT")
            
        Returns:
            pd.DataFrame: DataFrame z danymi historycznymi
        """
        try:
            if not self.bybit:
                logger.error("Konektor Bybit nie jest dostępny")
                return pd.DataFrame()
            
            # Obliczanie przedziału czasowego
            end_time = int(time.time() * 1000)  # teraz w milisekundach
            start_time = end_time - (self.lookback_days * 24 * 60 * 60 * 1000)  # lookback_days dni wstecz
            
            # Lista do przechowywania wszystkich danych
            all_data = []
            
            # Bybit ogranicza liczbę świec w jednym zapytaniu do 200
            # Musimy iteracyjnie pobierać dane w mniejszych fragmentach
            current_end = end_time
            
            while current_end > start_time:
                # Pobieramy dane
                response = self.bybit.get_klines(
                    symbol=symbol,
                    interval=self.interval,
                    limit=200,
                    end_time=current_end
                )
                
                if "result" not in response or "list" not in response["result"]:
                    logger.error(f"Błąd podczas pobierania danych dla {symbol}: {response}")
                    break
                
                klines = response["result"]["list"]
                
                if not klines:
                    logger.info(f"Brak danych dla {symbol} przed {current_end}")
                    break
                
                # Bybit zwraca dane od najnowszych do najstarszych
                # Dodajemy je do naszej listy
                all_data.extend(klines)
                
                # Aktualizujemy current_end do timestamp najstarszej świecy - 1ms
                oldest_timestamp = int(klines[-1][0])
                if oldest_timestamp >= current_end:
                    # Zabezpieczenie przed nieskończoną pętlą
                    logger.warning(f"Otrzymano nieprawidłowy timestamp dla {symbol}: {oldest_timestamp}")
                    break
                
                current_end = oldest_timestamp - 1
                
                # Limit API rate - odpoczywamy między zapytaniami
                time.sleep(0.5)
                
                # Jeśli mamy wystarczającą ilość danych, kończymy
                if oldest_timestamp <= start_time:
                    break
            
            # Jeśli nie mamy danych, zwracamy pusty DataFrame
            if not all_data:
                logger.error(f"Nie udało się pobrać danych dla {symbol}")
                return pd.DataFrame()
            
            # Przetwarzamy dane na DataFrame
            # Format danych z Bybit: [timestamp, open, high, low, close, volume, turnover]
            df = pd.DataFrame(all_data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            
            # Konwertujemy typy danych
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col])
            
            # Sortujemy dane chronologicznie
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Dodajemy kolumnę symbol
            df["symbol"] = symbol
            
            logger.info(f"Pobrano {len(df)} świec dla {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania danych historycznych dla {symbol}: {e}")
            return pd.DataFrame()
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje dodatkowe cechy do danych.
        
        Args:
            df: DataFrame z danymi historycznymi
            
        Returns:
            pd.DataFrame: DataFrame z dodanymi cechami
        """
        if df.empty:
            return df
        
        try:
            # Dodajemy podstawowe wskaźniki techniczne
            
            # Zwroty procentowe
            df["returns"] = df["close"].pct_change()
            
            # Średnie kroczące
            df["ma7"] = df["close"].rolling(window=7).mean()
            df["ma25"] = df["close"].rolling(window=25).mean()
            df["ma99"] = df["close"].rolling(window=99).mean()
            
            # Względna siła cenowa
            df["price_rel_ma7"] = df["close"] / df["ma7"]
            df["price_rel_ma25"] = df["close"] / df["ma25"]
            df["price_rel_ma99"] = df["close"] / df["ma99"]
            
            # Volatility
            df["volatility"] = df["returns"].rolling(window=30).std()
            
            # Wolumen relatywny
            df["vol_rel"] = df["volume"] / df["volume"].rolling(window=20).mean()
            
            # RSI - Relative Strength Index
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # MACD - Moving Average Convergence Divergence
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
            
            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            df["bb_std"] = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
            df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            
            # Usuwamy wiersze z NaN
            df = df.dropna().reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Błąd podczas dodawania cech: {e}")
            return df
    
    def prepare_train_test_data(self, df: pd.DataFrame) -> Dict:
        """
        Przygotowuje dane treningowe i testowe.
        
        Args:
            df: DataFrame z danymi historycznymi i cechami
            
        Returns:
            Dict: Słownik z danymi treningowymi i testowymi
        """
        if df.empty:
            return {"X_train": None, "X_test": None, "y_train": None, "y_test": None}
        
        try:
            # Definiujemy cechy i cel
            # Celem będzie zmiana ceny w następnej świecy
            df["target"] = df["close"].shift(-1) / df["close"] - 1
            
            # Usuwamy ostatni wiersz (bez celu)
            df = df.iloc[:-1]
            
            # Definiujemy cechy
            feature_columns = [
                "returns", "ma7", "ma25", "ma99", "price_rel_ma7", "price_rel_ma25", 
                "price_rel_ma99", "volatility", "vol_rel", "rsi", "macd", "macd_signal", 
                "macd_hist", "bb_width", "bb_pos"
            ]
            
            # Możemy dodać więcej zaawansowanych cech tutaj
            
            # Dzielimy dane na treningowe i testowe
            train_size = int(len(df) * (1 - self.test_size))
            
            # Zbiór treningowy
            X_train = df.iloc[:train_size][feature_columns]
            y_train = df.iloc[:train_size]["target"]
            
            # Zbiór testowy
            X_test = df.iloc[train_size:][feature_columns]
            y_test = df.iloc[train_size:]["target"]
            
            logger.info(f"Przygotowano dane treningowe: {X_train.shape} i testowe: {X_test.shape}")
            
            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "train_dates": df.iloc[:train_size]["timestamp"],
                "test_dates": df.iloc[train_size:]["timestamp"],
                "symbol": df["symbol"].iloc[0]
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas przygotowania danych treningowych: {e}")
            return {"X_train": None, "X_test": None, "y_train": None, "y_test": None}
    
    def train_xgboost_model(self, train_data: Dict) -> Dict:
        """
        Trenuje model XGBoost na danych treningowych.
        
        Args:
            train_data: Słownik z danymi treningowymi
            
        Returns:
            Dict: Słownik z modelem i metrykami
        """
        if (train_data["X_train"] is None or train_data["y_train"] is None or 
            train_data["X_test"] is None or train_data["y_test"] is None):
            logger.error("Brak danych treningowych")
            return {"model": None, "metrics": {}, "feature_importance": {}}
        
        try:
            import xgboost as xgb
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Przygotowujemy dane DMatrix
            dtrain = xgb.DMatrix(train_data["X_train"], label=train_data["y_train"])
            dtest = xgb.DMatrix(train_data["X_test"], label=train_data["y_test"])
            
            # Parametry modelu
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42
            }
            
            # Trenujemy model
            evallist = [(dtrain, "train"), (dtest, "test")]
            
            model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=100, 
                evals=evallist,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Przewidujemy na zbiorze testowym
            y_pred = model.predict(dtest)
            
            # Obliczamy metryki
            mse = mean_squared_error(train_data["y_test"], y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(train_data["y_test"], y_pred)
            
            # Skuteczność kierunkowa
            direction_accuracy = np.mean((train_data["y_test"] > 0) == (y_pred > 0))
            
            # Ważność cech
            feature_importance = model.get_score(importance_type="gain")
            
            # Zapisujemy model
            model_path = f"saved_models/XGB_{train_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            model.save_model(model_path)
            
            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "direction_accuracy": direction_accuracy,
                "model_path": model_path,
                "symbol": train_data["symbol"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"Model XGBoost wytrenowany dla {train_data['symbol']}, dokładność kierunkowa: {direction_accuracy:.4f}")
            
            return {
                "model": model,
                "metrics": metrics,
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu XGBoost: {e}")
            return {"model": None, "metrics": {}, "feature_importance": {}}
    
    def train_model_for_symbol(self, symbol: str) -> Dict:
        """
        Trenuje model dla danego symbolu.
        
        Args:
            symbol: Symbol handlowy (np. "BTCUSDT")
            
        Returns:
            Dict: Wyniki trenowania
        """
        logger.info(f"Rozpoczynanie trenowania modelu dla {symbol}")
        
        # Pobieranie danych
        df = self.fetch_historical_data(symbol)
        
        if df.empty:
            logger.error(f"Nie udało się pobrać danych dla {symbol}")
            return {"success": False, "error": "Brak danych historycznych"}
        
        # Dodajemy cechy
        df_with_features = self.add_features(df)
        
        # Przygotowujemy dane treningowe
        train_data = self.prepare_train_test_data(df_with_features)
        
        # Trenujemy model
        model_results = self.train_xgboost_model(train_data)
        
        if model_results["model"] is None:
            logger.error(f"Nie udało się wytrenować modelu dla {symbol}")
            return {"success": False, "error": "Błąd trenowania modelu"}
        
        return {
            "success": True,
            "symbol": symbol,
            "metrics": model_results["metrics"],
            "feature_importance": model_results["feature_importance"],
            "data_points": len(df),
            "period": f"{self.lookback_days} dni"
        }
    
    def train_all_models(self) -> List[Dict]:
        """
        Trenuje modele dla wszystkich symboli.
        
        Returns:
            List[Dict]: Lista wyników trenowania
        """
        results = []
        
        for symbol in self.symbols:
            result = self.train_model_for_symbol(symbol)
            results.append(result)
        
        return results

# Przykład użycia
if __name__ == "__main__":
    # Tworzymy folder dla logów
    os.makedirs("logs", exist_ok=True)
    
    # Inicjalizujemy trainer
    trainer = MarketDataTrainer(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval="1h",
        lookback_days=30,
        test_size=0.2,
        use_testnet=True
    )
    
    # Trenujemy model dla BTC/USDT
    result = trainer.train_model_for_symbol("BTCUSDT")
    
    if result["success"]:
        print(f"Model wytrenowany pomyślnie dla BTCUSDT")
        print(f"Dokładność kierunkowa: {result['metrics']['direction_accuracy']:.4f}")
    else:
        print(f"Błąd podczas trenowania modelu: {result.get('error', 'Nieznany błąd')}")
