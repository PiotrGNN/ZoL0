
"""
Fallback Strategy - strategia awaryjna, aktywowana gdy modele AI 
nie dostarczą sygnału lub mają zbyt niską pewność.

Bazuje na prostych wskaźnikach technicznych i zarządzaniu ryzykiem,
aby zapewnić sensowne decyzje handlowe podczas awarii modeli AI.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/fallback_strategy.log"), logging.StreamHandler()],
)
logger = logging.getLogger("FallbackStrategy")


class FallbackStrategy:
    """
    Strategia awaryjna używająca prostych wskaźników technicznych.
    
    Używana, gdy modele AI:
    - nie dostarczą sygnału
    - mają zbyt niską pewność predykcji 
    - występują problemy z danymi wejściowymi
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        sma_short_period: int = 10,
        sma_long_period: int = 30,
        min_interval_minutes: int = 60,
        risk_reward_ratio: float = 2.0,
    ):
        """
        Inicjalizacja strategii awaryjnej.
        
        Args:
            rsi_period: Okres dla wskaźnika RSI
            rsi_overbought: Próg wykupienia dla RSI
            rsi_oversold: Próg wyprzedania dla RSI
            sma_short_period: Okres dla krótkiej średniej kroczącej
            sma_long_period: Okres dla długiej średniej kroczącej
            min_interval_minutes: Minimalny interwał między sygnałami (minuty)
            risk_reward_ratio: Stosunek zysku do ryzyka
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.sma_short_period = sma_short_period
        self.sma_long_period = sma_long_period
        self.min_interval_minutes = min_interval_minutes
        self.risk_reward_ratio = risk_reward_ratio
        
        # Ostatni wygenerowany sygnał
        self.last_signal_time = None
        self.last_signal = None
        
        logger.info(
            f"Strategia awaryjna zainicjowana: RSI({rsi_period}), SMA({sma_short_period},{sma_long_period}), "
            f"Interwał: {min_interval_minutes}min, RR: {risk_reward_ratio}"
        )

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generuje sygnał handlowy na podstawie prostych wskaźników technicznych.
        
        Args:
            data: DataFrame z danymi cenowymi, musi zawierać kolumny: 
                 open, high, low, close, volume, timestamp
                 
        Returns:
            Dict: Słownik zawierający:
                - 'action': 'BUY', 'SELL' lub 'HOLD'
                - 'confidence': pewność sygnału (0-1)
                - 'price': aktualna cena
                - 'stop_loss': sugerowany poziom stop-loss
                - 'take_profit': sugerowany poziom take-profit
                - 'reasons': powody decyzji (lista stringów)
        """
        try:
            # Upewniamy się, że mamy wystarczająco danych
            if len(data) < max(self.rsi_period, self.sma_long_period) + 10:
                logger.warning(f"Za mało danych: {len(data)} punktów")
                return self._create_hold_signal(
                    data, confidence=0.1, reasons=["Niewystarczająca ilość danych historycznych"]
                )
                
            # Sprawdzamy czy kolumny są w odpowiednim formacie
            required_columns = ["open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"Brak kolumny: {col}")
                    return self._create_hold_signal(
                        data, confidence=0.1, reasons=[f"Brak wymaganych danych (kolumna {col})"]
                    )
            
            # Przygotowujemy kopię danych (nie modyfikujemy oryginału)
            df = data.copy()
            
            # Upewniamy się, że dane są posortowane chronologicznie
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
                
            # Sprawdzamy czy upłynął minimalny interwał między sygnałami
            if self.last_signal_time and not self._min_interval_passed():
                logger.info("Nie upłynął minimalny interwał między sygnałami")
                return self.last_signal
                
            # Obliczamy wskaźniki techniczne
            self._calculate_indicators(df)
            
            # Generujemy sygnał na podstawie wskaźników
            signal = self._generate_signal_from_indicators(df)
            
            # Aktualizujemy ostatni sygnał
            self.last_signal_time = datetime.now()
            self.last_signal = signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania sygnału awaryjnego: {e}", exc_info=True)
            return {
                "action": "HOLD",
                "confidence": 0.1,
                "price": data["close"].iloc[-1] if "close" in data.columns and len(data) > 0 else None,
                "reasons": [f"Błąd podczas analizy: {str(e)}"],
            }

    def _calculate_indicators(self, df: pd.DataFrame) -> None:
        """
        Oblicza wskaźniki techniczne używane przez strategię.
        
        Args:
            df: DataFrame z danymi cenowymi
        """
        # Średnie kroczące
        df["sma_short"] = df["close"].rolling(window=self.sma_short_period).mean()
        df["sma_long"] = df["close"].rolling(window=self.sma_long_period).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Wskaźnik zmienności (ATR)
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        
        df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(window=14).mean()
        
        # Wolumen względny
        df["vol_ma"] = df["volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"]
        
        logger.debug("Obliczono wskaźniki techniczne dla strategii awaryjnej")

    def _generate_signal_from_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Generuje sygnał handlowy na podstawie obliczonych wskaźników.
        
        Args:
            df: DataFrame z danymi i obliczonymi wskaźnikami
            
        Returns:
            Dict: Sygnał handlowy
        """
        # Pobieramy ostatnie wartości wskaźników
        last_idx = -1
        
        rsi = df["rsi"].iloc[last_idx]
        sma_short = df["sma_short"].iloc[last_idx]
        sma_long = df["sma_long"].iloc[last_idx]
        close = df["close"].iloc[last_idx]
        atr = df["atr"].iloc[last_idx]
        vol_ratio = df["vol_ratio"].iloc[last_idx]
        
        # Lista powodów decyzji
        reasons = []
        
        # Sprawdzamy sygnały
        sell_signals = 0
        buy_signals = 0
        
        # 1. RSI
        if rsi > self.rsi_overbought:
            sell_signals += 1
            reasons.append(f"RSI wykupiony ({rsi:.1f})")
        elif rsi < self.rsi_oversold:
            buy_signals += 1
            reasons.append(f"RSI wyprzedany ({rsi:.1f})")
            
        # 2. Przecięcie średnich kroczących
        if sma_short > sma_long:
            buy_signals += 1
            reasons.append("Przecięcie SMA w górę (trend wzrostowy)")
        elif sma_short < sma_long:
            sell_signals += 1
            reasons.append("Przecięcie SMA w dół (trend spadkowy)")
            
        # 3. Analiza wolumenu
        if vol_ratio > 1.5:
            if close > df["close"].iloc[-2]:
                buy_signals += 1
                reasons.append(f"Wysoki wolumen przy wzroście ceny (x{vol_ratio:.1f})")
            else:
                sell_signals += 1
                reasons.append(f"Wysoki wolumen przy spadku ceny (x{vol_ratio:.1f})")
                
        # 4. Ostatnie 3 świece (prosta analiza price action)
        last_3_candles = df.iloc[-3:].copy()
        price_action = self._analyze_price_action(last_3_candles)
        
        if price_action == "bullish":
            buy_signals += 1
            reasons.append("Bycza formacja świecowa")
        elif price_action == "bearish":
            sell_signals += 1
            reasons.append("Niedźwiedzia formacja świecowa")
            
        # Ocena pewności na podstawie liczby sygnałów (max 4 sygnały)
        if buy_signals > sell_signals:
            action = "BUY"
            confidence = min(0.5 + (buy_signals / 8), 0.85)  # max 0.85 pewności dla strategii awaryjnej
            reasons.insert(0, f"{buy_signals} sygnałów kupna vs {sell_signals} sygnałów sprzedaży")
        elif sell_signals > buy_signals:
            action = "SELL"
            confidence = min(0.5 + (sell_signals / 8), 0.85)
            reasons.insert(0, f"{sell_signals} sygnałów sprzedaży vs {buy_signals} sygnałów kupna")
        else:
            action = "HOLD"
            confidence = 0.5
            reasons.insert(0, "Sygnały niejednoznaczne lub sprzeczne")
            
        # Ustawienie poziomów Stop Loss i Take Profit
        sl_distance = atr * 1.5  # 1.5x ATR 
        tp_distance = sl_distance * self.risk_reward_ratio
        
        if action == "BUY":
            stop_loss = close - sl_distance
            take_profit = close + tp_distance
        elif action == "SELL":
            stop_loss = close + sl_distance
            take_profit = close - tp_distance
        else:
            stop_loss = None
            take_profit = None
            
        # Konstruujemy sygnał
        signal = {
            "action": action,
            "confidence": confidence,
            "price": close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasons": reasons,
        }
        
        logger.info(
            f"Wygenerowano sygnał awaryjny: {action} z pewnością {confidence:.2f}, "
            f"Cena: {close:.2f}, SL: {stop_loss:.2f if stop_loss else None}, "
            f"TP: {take_profit:.2f if take_profit else None}"
        )
        
        return signal

    def _analyze_price_action(self, candles: pd.DataFrame) -> str:
        """
        Analizuje formacje świecowe i określa ich charakter.
        
        Args:
            candles: DataFrame z danymi świecowymi
            
        Returns:
            str: "bullish", "bearish" lub "neutral"
        """
        # Sprawdzamy trendy między świecami
        closes = candles["close"].values
        opens = candles["open"].values
        
        # Rozmiary świec (body)
        candle_sizes = abs(closes - opens)
        avg_size = np.mean(candle_sizes)
        
        # Określamy charakter każdej świecy
        candle_chars = []
        for i in range(len(closes)):
            if closes[i] > opens[i]:
                if candle_sizes[i] > avg_size * 1.2:
                    candle_chars.append("strong_bull")
                else:
                    candle_chars.append("bull")
            elif closes[i] < opens[i]:
                if candle_sizes[i] > avg_size * 1.2:
                    candle_chars.append("strong_bear")
                else:
                    candle_chars.append("bear")
            else:
                candle_chars.append("neutral")
                
        # Analizujemy sekwencję świec
        # To bardzo uproszczona analiza - w rzeczywistości analizy formacji świecowych
        # są znacznie bardziej zaawansowane
        
        # Bycze formacje
        if all(c in ["bull", "strong_bull"] for c in candle_chars):
            return "bullish"  # Wszystkie świece wzrostowe
        if candle_chars[-1] == "strong_bull" and candle_chars[-2] in ["bear", "strong_bear"]:
            return "bullish"  # Odbicie po spadku
        if candle_chars[-1] == "bull" and all(c in ["bear", "strong_bear"] for c in candle_chars[:-1]):
            return "bullish"  # Odwrócenie trendu
            
        # Niedźwiedzie formacje
        if all(c in ["bear", "strong_bear"] for c in candle_chars):
            return "bearish"  # Wszystkie świece spadkowe
        if candle_chars[-1] == "strong_bear" and candle_chars[-2] in ["bull", "strong_bull"]:
            return "bearish"  # Załamanie po wzroście
        if candle_chars[-1] == "bear" and all(c in ["bull", "strong_bull"] for c in candle_chars[:-1]):
            return "bearish"  # Odwrócenie trendu
            
        # Neutralne formacje
        return "neutral"

    def _create_hold_signal(self, data: pd.DataFrame, confidence: float = 0.1, reasons: Optional[List[str]] = None) -> Dict:
        """
        Tworzy sygnał HOLD gdy nie możemy wygenerować prawidłowego sygnału.
        
        Args:
            data: DataFrame z danymi cenowymi
            confidence: Pewność sygnału (0-1)
            reasons: Lista powodów decyzji
            
        Returns:
            Dict: Sygnał HOLD
        """
        if reasons is None:
            reasons = ["Niewystarczające dane do analizy"]
            
        try:
            price = data["close"].iloc[-1] if "close" in data.columns and len(data) > 0 else None
        except Exception:
            price = None
            
        return {
            "action": "HOLD",
            "confidence": confidence,
            "price": price,
            "stop_loss": None,
            "take_profit": None,
            "reasons": reasons,
        }

    def _min_interval_passed(self) -> bool:
        """
        Sprawdza, czy upłynął minimalny interwał między sygnałami.
        
        Returns:
            bool: True jeśli upłynął minimalny interwał
        """
        if self.last_signal_time is None:
            return True
            
        time_diff = datetime.now() - self.last_signal_time
        return time_diff.total_seconds() / 60 >= self.min_interval_minutes


# Przykład użycia
if __name__ == "__main__":
    try:
        import yfinance as yf
        
        # Pobierz przykładowe dane
        data = yf.download("BTC-USD", period="60d", interval="1h")
        data.reset_index(inplace=True)
        
        # Dostosuj nazwy kolumn
        data.columns = [col.lower() for col in data.columns]
        data.rename(columns={"date": "timestamp"}, inplace=True)
        
        # Inicjalizuj strategię awaryjną
        fallback = FallbackStrategy()
        
        # Wygeneruj sygnał
        signal = fallback.generate_signal(data)
        
        print(f"Sygnał awaryjny: {signal['action']}")
        print(f"Pewność: {signal['confidence']:.2f}")
        print(f"Powody: {', '.join(signal['reasons'])}")
        print(f"Cena: {signal['price']:.2f}")
        
        if signal["take_profit"]:
            print(f"Take Profit: {signal['take_profit']:.2f}")
        if signal["stop_loss"]:
            print(f"Stop Loss: {signal['stop_loss']:.2f}")
    except ImportError:
        print("Zainstaluj yfinance, aby uruchomić ten przykład")
    except Exception as e:
        print(f"Błąd podczas testowania strategii awaryjnej: {e}")
