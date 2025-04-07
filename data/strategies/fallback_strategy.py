
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/fallback_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fallback_strategy")

class FallbackStrategy:
    """
    Strategia awaryjna używana w przypadku braku sygnałów z AI 
    lub problemów z główną strategią.
    
    Opiera się na prostych wskaźnikach technicznych i zasadach zarządzania ryzykiem.
    """
    
    def __init__(self, lookback_period: int = 14, volatility_window: int = 20):
        """
        Inicjalizacja strategii awaryjnej
        
        Args:
            lookback_period (int): Okres wsteczny dla wskaźników
            volatility_window (int): Okno do obliczania zmienności
        """
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.last_signal = None
        self.last_signal_time = None
        
    def calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Oblicza podstawowe wskaźniki techniczne
        
        Args:
            data (pd.DataFrame): Dane cenowe z kolumnami: timestamp, open, high, low, close, volume
            
        Returns:
            pd.DataFrame: Dane z dodanymi wskaźnikami
        """
        df = data.copy()
        
        # Upewnij się, że dane są posortowane chronologicznie
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            
        # Średnie kroczące
        df['sma_short'] = df['close'].rolling(window=5).mean()
        df['sma_medium'] = df['close'].rolling(window=20).mean()
        df['sma_long'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.lookback_period).mean()
        avg_loss = loss.rolling(window=self.lookback_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range) - zmienność
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.volatility_window).mean()
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Volatility ratio
        df['volatility_ratio'] = df['atr'] / df['close'] * 100
        
        return df
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generuje sygnał handlowy na podstawie prostych wskaźników
        
        Args:
            data (pd.DataFrame): Dane cenowe z obliczonymi wskaźnikami
            
        Returns:
            Dict: Sygnał handlowy (BUY, SELL, HOLD)
        """
        if len(data) < 50:
            logger.warning(f"Za mało danych do generowania sygnału: {len(data)} punktów")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': 'Niewystarczająca ilość danych'
            }
            
        # Użyj ostatnich danych
        df = self.calculate_basic_indicators(data)
        current = df.iloc[-1]
        
        # Sprawdź, czy mamy wszystkie potrzebne wskaźniki
        if pd.isna(current['sma_short']) or pd.isna(current['sma_medium']) or \
           pd.isna(current['sma_long']) or pd.isna(current['rsi']) or \
           pd.isna(current['atr']):
            logger.warning("Brakujące wskaźniki w danych")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': 'Brakujące wskaźniki'
            }
        
        # Przechowywanie powodów sygnału
        reasons = []
        buy_signals = 0
        sell_signals = 0
        
        # 1. Sygnał z przecięcia średnich kroczących
        if current['sma_short'] > current['sma_medium'] and df.iloc[-2]['sma_short'] <= df.iloc[-2]['sma_medium']:
            buy_signals += 1
            reasons.append("Przecięcie SMA 5/20 w górę")
        elif current['sma_short'] < current['sma_medium'] and df.iloc[-2]['sma_short'] >= df.iloc[-2]['sma_medium']:
            sell_signals += 1
            reasons.append("Przecięcie SMA 5/20 w dół")
            
        # 2. Poziomy RSI
        if current['rsi'] < 30:
            buy_signals += 1
            reasons.append(f"Wyprzedanie (RSI={current['rsi']:.1f})")
        elif current['rsi'] > 70:
            sell_signals += 1
            reasons.append(f"Wykupienie (RSI={current['rsi']:.1f})")
            
        # 3. Trend długoterminowy
        if current['close'] > current['sma_long']:
            buy_signals += 0.5
            reasons.append("Trend długoterminowy wzrostowy")
        elif current['close'] < current['sma_long']:
            sell_signals += 0.5
            reasons.append("Trend długoterminowy spadkowy")
            
        # 4. Momentum
        if current['momentum'] > 0.02:  # 2% momentum
            buy_signals += 0.5
            reasons.append(f"Silny momentum wzrostowy: {current['momentum']*100:.1f}%")
        elif current['momentum'] < -0.02:
            sell_signals += 0.5
            reasons.append(f"Silny momentum spadkowy: {current['momentum']*100:.1f}%")
            
        # Oblicz siłę sygnału i poziom pewności
        total_signals = buy_signals + sell_signals
        max_signals = 3  # Maksymalna liczba sygnałów (1+1+0.5+0.5)
        
        # Wygeneruj akcję
        if buy_signals > sell_signals and buy_signals >= 1:
            action = 'BUY'
            confidence = min(0.85, buy_signals / max_signals)
        elif sell_signals > buy_signals and sell_signals >= 1:
            action = 'SELL'
            confidence = min(0.85, sell_signals / max_signals)
        else:
            action = 'HOLD'
            confidence = 0.5
            reasons.append("Brak wyraźnych sygnałów")
        
        # Oblicz poziomy TP/SL na podstawie ATR
        current_price = current['close']
        atr = current['atr']
        
        if action == 'BUY':
            take_profit = current_price + (atr * 3)  # 3x ATR dla TP
            stop_loss = current_price - (atr * 1.5)  # 1.5x ATR dla SL
        elif action == 'SELL':
            take_profit = current_price - (atr * 3)
            stop_loss = current_price + (atr * 1.5)
        else:
            take_profit = None
            stop_loss = None
        
        # Zapisz ostatni sygnał
        self.last_signal = {
            'action': action,
            'confidence': confidence,
            'reasons': reasons,
            'price': current_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'timestamp': datetime.now(),
            'indicators': {
                'sma_short': current['sma_short'],
                'sma_medium': current['sma_medium'],
                'rsi': current['rsi'],
                'atr': current['atr'],
                'momentum': current['momentum']
            }
        }
        self.last_signal_time = datetime.now()
        
        logger.info(f"Wygenerowano sygnał awaryjny: {action} (pewność: {confidence:.2f}) - {', '.join(reasons)}")
        
        return self.last_signal
    
    def get_last_signal(self) -> Dict:
        """Zwraca ostatni wygenerowany sygnał"""
        return self.last_signal
    
    def should_update_signal(self, min_interval_minutes: int = 15) -> bool:
        """
        Sprawdza, czy należy wygenerować nowy sygnał
        
        Args:
            min_interval_minutes (int): Minimalny interwał między sygnałami w minutach
            
        Returns:
            bool: True, jeśli należy wygenerować nowy sygnał
        """
        if self.last_signal_time is None:
            return True
            
        time_diff = datetime.now() - self.last_signal_time
        return time_diff.total_seconds() / 60 >= min_interval_minutes

# Przykład użycia
if __name__ == "__main__":
    import yfinance as yf
    
    # Pobierz przykładowe dane
    data = yf.download("BTC-USD", period="60d", interval="1h")
    data.reset_index(inplace=True)
    
    # Dostosuj nazwy kolumn
    data.columns = [col.lower() for col in data.columns]
    data.rename(columns={'date': 'timestamp'}, inplace=True)
    
    # Inicjalizuj strategię awaryjną
    fallback = FallbackStrategy()
    
    # Wygeneruj sygnał
    signal = fallback.generate_signal(data)
    
    print(f"Sygnał awaryjny: {signal['action']}")
    print(f"Pewność: {signal['confidence']:.2f}")
    print(f"Powody: {', '.join(signal['reasons'])}")
    print(f"Cena: {signal['price']:.2f}")
    
    if signal['take_profit']:
        print(f"Take Profit: {signal['take_profit']:.2f}")
    if signal['stop_loss']:
        print(f"Stop Loss: {signal['stop_loss']:.2f}")
