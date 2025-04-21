"""
advanced_indicators.py
--------------------
Moduł implementujący zaawansowane wskaźniki techniczne dla analizy rynku.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        self.cached_data = {}
        
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza wszystkie dostępne wskaźniki techniczne."""
        result = df.copy()
        
        # Trendy i momentum
        result = self.add_moving_averages(result)
        result = self.add_momentum_indicators(result)
        result = self.add_volatility_indicators(result)
        result = self.add_volume_indicators(result)
        result = self.add_trend_indicators(result)
        
        return result
        
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje różne typy średnich kroczących."""
        df = df.copy()
        
        # Simple Moving Average (SMA)
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            
        # Exponential Moving Average (EMA)
        for period in [5, 10, 20, 50, 200]:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
        # Weighted Moving Average (WMA)
        for period in [5, 10, 20]:
            weights = np.arange(1, period + 1)
            df[f'WMA_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
            
        # Hull Moving Average (HMA)
        for period in [9, 16, 25]:
            wma1 = df['close'].rolling(period//2).apply(
                lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
            )
            wma2 = df['close'].rolling(period).apply(
                lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
            )
            df[f'HMA_{period}'] = wma1.ewm(span=int(np.sqrt(period))).mean()
            
        return df
        
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje wskaźniki momentum."""
        df = df.copy()
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'Stoch_K_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(window=3).mean()
            
        # ROC (Rate of Change)
        for period in [10, 20, 30]:
            df[f'ROC_{period}'] = (
                (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
            )
            
        return df
        
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje wskaźniki zmienności."""
        df = df.copy()
        
        # Bollinger Bands
        for period in [20, 50]:
            for std in [2.0, 2.5]:
                ma = df['close'].rolling(window=period).mean()
                std_dev = df['close'].rolling(window=period).std()
                df[f'BB_Upper_{period}_{std}'] = ma + (std_dev * std)
                df[f'BB_Lower_{period}_{std}'] = ma - (std_dev * std)
                df[f'BB_Width_{period}_{std}'] = (
                    (df[f'BB_Upper_{period}_{std}'] - df[f'BB_Lower_{period}_{std}']) / ma
                )
                
        # ATR (Average True Range)
        for period in [14, 21]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df[f'ATR_{period}'] = true_range.rolling(period).mean()
            
        # Keltner Channels
        for period in [20, 50]:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df[f'KC_Middle_{period}'] = typical_price.rolling(period).mean()
            atr = df[f'ATR_{14}']
            df[f'KC_Upper_{period}'] = df[f'KC_Middle_{period}'] + (2 * atr)
            df[f'KC_Lower_{period}'] = df[f'KC_Middle_{period}'] - (2 * atr)
            
        return df
        
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje wskaźniki wolumenu."""
        df = df.copy()
        
        # OBV (On Balance Volume)
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume SMA
        for period in [20, 50]:
            df[f'Volume_SMA_{period}'] = df['volume'].rolling(window=period).mean()
            
        # Chaikin Money Flow
        for period in [20, 50]:
            mf_multiplier = (
                ((df['close'] - df['low']) - (df['high'] - df['close']))
                / (df['high'] - df['low'])
            )
            mf_volume = mf_multiplier * df['volume']
            df[f'CMF_{period}'] = (
                mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
            )
            
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (
            (df['high'] + df['low'] + df['close']) / 3 * df['volume']
        ).cumsum() / df['volume'].cumsum()
        
        return df
        
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje wskaźniki trendu."""
        df = df.copy()
        
        # ADX (Average Directional Index)
        for period in [14, 21]:
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            pos_directional = np.where(
                (high_diff > low_diff) & (high_diff > 0),
                high_diff,
                0
            )
            neg_directional = np.where(
                (low_diff > high_diff) & (low_diff > 0),
                low_diff,
                0
            )
            
            tr = pd.DataFrame({
                'hl': df['high'] - df['low'],
                'hc': abs(df['high'] - df['close'].shift(1)),
                'lc': abs(df['low'] - df['close'].shift(1))
            }).max(axis=1)
            
            pdi = 100 * pd.Series(pos_directional).rolling(period).mean() / tr.rolling(period).mean()
            ndi = 100 * pd.Series(neg_directional).rolling(period).mean() / tr.rolling(period).mean()
            df[f'PDI_{period}'] = pdi
            df[f'NDI_{period}'] = ndi
            df[f'ADX_{period}'] = (
                abs(pdi - ndi) / (pdi + ndi) * 100
            ).rolling(period).mean()
            
        # Ichimoku Cloud
        high_val = df['high'].rolling(window=9).max()
        low_val = df['low'].rolling(window=9).min()
        df['Ichimoku_Conversion'] = (high_val + low_val) / 2
        
        high_val = df['high'].rolling(window=26).max()
        low_val = df['low'].rolling(window=26).min()
        df['Ichimoku_Base'] = (high_val + low_val) / 2
        
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        
        high_val = df['high'].rolling(window=52).max()
        low_val = df['low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((high_val + low_val) / 2).shift(26)
        
        df['Ichimoku_Lagging'] = df['close'].shift(-26)
        
        return df
        
    def get_technical_analysis(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generuje kompletną analizę techniczną na podstawie wskaźników.
        
        Returns:
            Dict z analizą zawierającą:
            - sygnały
            - trendy
            - wsparcia/opory
            - poziomy zwrotne
        """
        analysis = {
            'signals': self._generate_signals(df),
            'trends': self._analyze_trends(df),
            'support_resistance': self._find_support_resistance(df),
            'pivot_points': self._calculate_pivot_points(df),
            'pattern_recognition': self._recognize_patterns(df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        return analysis
        
    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generuje sygnały tradingowe na podstawie wskaźników."""
        signals = {}
        
        # Sygnały ze średnich kroczących
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            signals['MA_Cross'] = 'BUY' if sma_20 > sma_50 else 'SELL'
            
        # Sygnały RSI
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].iloc[-1]
            if rsi > 70:
                signals['RSI'] = 'OVERSOLD'
            elif rsi < 30:
                signals['RSI'] = 'OVERSOLD'
            else:
                signals['RSI'] = 'NEUTRAL'
                
        # Sygnały MACD
        if all(x in df.columns for x in ['MACD', 'MACD_Signal']):
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_Signal'].iloc[-1]
            signals['MACD'] = 'BUY' if macd > signal else 'SELL'
            
        # Sygnały Bollinger Bands
        if all(x in df.columns for x in ['BB_Upper_20_2.0', 'BB_Lower_20_2.0']):
            price = df['close'].iloc[-1]
            upper = df['BB_Upper_20_2.0'].iloc[-1]
            lower = df['BB_Lower_20_2.0'].iloc[-1]
            
            if price > upper:
                signals['Bollinger'] = 'OVERBOUGHT'
            elif price < lower:
                signals['Bollinger'] = 'OVERSOLD'
            else:
                signals['Bollinger'] = 'NEUTRAL'
                
        return signals
        
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analizuje trendy na różnych okresach czasowych."""
        trends = {}
        
        # Trend krótkoterminowy (SMA 20)
        if 'SMA_20' in df.columns:
            short_term_slope = (
                df['SMA_20'].iloc[-1] - df['SMA_20'].iloc[-5]
            ) / df['SMA_20'].iloc[-5]
            trends['short_term'] = (
                'UPTREND' if short_term_slope > 0.01
                else 'DOWNTREND' if short_term_slope < -0.01
                else 'SIDEWAYS'
            )
            
        # Trend średnioterminowy (SMA 50)
        if 'SMA_50' in df.columns:
            medium_term_slope = (
                df['SMA_50'].iloc[-1] - df['SMA_50'].iloc[-10]
            ) / df['SMA_50'].iloc[-10]
            trends['medium_term'] = (
                'UPTREND' if medium_term_slope > 0.02
                else 'DOWNTREND' if medium_term_slope < -0.02
                else 'SIDEWAYS'
            )
            
        # Trend długoterminowy (SMA 200)
        if 'SMA_200' in df.columns:
            long_term_slope = (
                df['SMA_200'].iloc[-1] - df['SMA_200'].iloc[-20]
            ) / df['SMA_200'].iloc[-20]
            trends['long_term'] = (
                'UPTREND' if long_term_slope > 0.05
                else 'DOWNTREND' if long_term_slope < -0.05
                else 'SIDEWAYS'
            )
            
        return trends
        
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Znajduje poziomy wsparcia i oporu."""
        support_levels = []
        resistance_levels = []
        
        # Znajdź lokalne minima i maksima
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] <= df['low'].iloc[i-window:i+window]):
                support_levels.append(df['low'].iloc[i])
            if all(df['high'].iloc[i] >= df['high'].iloc[i-window:i+window]):
                resistance_levels.append(df['high'].iloc[i])
                
        # Grupuj podobne poziomy
        support_levels = self._cluster_levels(support_levels)
        resistance_levels = self._cluster_levels(resistance_levels)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
        
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """Oblicza punkty pivot."""
        if len(df) < 1:
            return {}
            
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }
        
    def _recognize_patterns(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Rozpoznaje formacje cenowe."""
        patterns = []
        
        # Doji
        doji = self._find_doji(df)
        if doji:
            patterns.extend(doji)
            
        # Engulfing patterns
        engulfing = self._find_engulfing(df)
        if engulfing:
            patterns.extend(engulfing)
            
        # Hammer/Hanging Man
        hammer = self._find_hammer(df)
        if hammer:
            patterns.extend(hammer)
            
        return {'patterns': patterns}
        
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.002) -> List[float]:
        """Grupuje podobne poziomy cenowe."""
        if not levels:
            return []
            
        levels = sorted(levels)
        clusters = [[levels[0]]]
        
        for level in levels[1:]:
            if abs(level - np.mean(clusters[-1])) / level <= tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])
                
        return [np.mean(cluster) for cluster in clusters]
        
    def _find_doji(self, df: pd.DataFrame) -> List[Dict]:
        """Znajduje świece Doji."""
        dojis = []
        
        for i in range(len(df)):
            body_size = abs(df['open'].iloc[i] - df['close'].iloc[i])
            total_size = df['high'].iloc[i] - df['low'].iloc[i]
            
            if total_size > 0 and body_size / total_size < 0.1:
                dojis.append({
                    'type': 'Doji',
                    'index': i,
                    'price': df['close'].iloc[i],
                    'confidence': 0.8
                })
                
        return dojis
        
    def _find_engulfing(self, df: pd.DataFrame) -> List[Dict]:
        """Znajduje formacje engulfing."""
        engulfing = []
        
        for i in range(1, len(df)):
            prev_body = abs(df['open'].iloc[i-1] - df['close'].iloc[i-1])
            curr_body = abs(df['open'].iloc[i] - df['close'].iloc[i])
            
            if curr_body > prev_body:
                # Bullish Engulfing
                if (df['open'].iloc[i] < df['close'].iloc[i-1] and
                    df['close'].iloc[i] > df['open'].iloc[i-1]):
                    engulfing.append({
                        'type': 'Bullish Engulfing',
                        'index': i,
                        'price': df['close'].iloc[i],
                        'confidence': 0.85
                    })
                # Bearish Engulfing
                elif (df['close'].iloc[i] < df['open'].iloc[i-1] and
                      df['open'].iloc[i] > df['close'].iloc[i-1]):
                    engulfing.append({
                        'type': 'Bearish Engulfing',
                        'index': i,
                        'price': df['close'].iloc[i],
                        'confidence': 0.85
                    })
                    
        return engulfing
        
    def _find_hammer(self, df: pd.DataFrame) -> List[Dict]:
        """Znajduje formacje młota i wisielca."""
        hammers = []
        
        for i in range(len(df)):
            body_size = abs(df['open'].iloc[i] - df['close'].iloc[i])
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            
            if body_size > 0:
                if lower_shadow > 2 * body_size and upper_shadow < body_size:
                    hammers.append({
                        'type': 'Hammer',
                        'index': i,
                        'price': df['close'].iloc[i],
                        'confidence': 0.75
                    })
                elif upper_shadow > 2 * body_size and lower_shadow < body_size:
                    hammers.append({
                        'type': 'Hanging Man',
                        'index': i,
                        'price': df['close'].iloc[i],
                        'confidence': 0.75
                    })
                    
        return hammers

if __name__ == "__main__":
    # Przykład użycia
    import yfinance as yf
    
    # Pobierz dane testowe
    symbol = "BTC-USD"
    df = yf.download(symbol, start="2023-01-01", end="2024-01-01")
    
    # Inicjalizuj analizator
    analyzer = TechnicalIndicators()
    
    # Oblicz wskaźniki
    df_with_indicators = analyzer.calculate_all(df)
    
    # Wykonaj analizę techniczną
    analysis = analyzer.get_technical_analysis(df_with_indicators)
    
    print("Analiza techniczna dla", symbol)
    print("Sygnały:", analysis['signals'])
    print("Trendy:", analysis['trends'])
    print("Wsparcia i opory:", analysis['support_resistance'])
    print("Punkty pivot:", analysis['pivot_points'])
    print("Rozpoznane formacje:", analysis['pattern_recognition'])