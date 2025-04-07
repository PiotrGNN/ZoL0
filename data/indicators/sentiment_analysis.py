"""
sentiment_analysis.py
--------------------
Moduł do analizy sentymentu rynkowego z różnych źródeł danych.
"""

import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analizator sentymentu rynkowego używający różnych źródeł danych.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicjalizacja analizatora sentymentu.

        Args:
            config (dict, optional): Konfiguracja analizatora
        """
        self.config = config or {}
        self.sources = self.config.get("sources", ["twitter", "news", "reddit", "tradingview"])
        self.last_update = None
        self.current_sentiment = None

        # Próba importu NLTK
        try:
            import nltk
            self.nltk_available = True
        except ImportError:
            self.nltk_available = False
            logger.warning("Biblioteka NLTK nie jest dostępna. Używam symulowanych danych sentymentu.")

        logger.info(f"Inicjalizacja analizatora sentymentu z {len(self.sources)} źródłami.")

    def analyze(self, symbol: str = "BTC/USDT", force_update: bool = False) -> Dict[str, Any]:
        """
        Analizuje sentyment dla danego symbolu.

        Args:
            symbol (str): Symbol rynkowy do analizy
            force_update (bool): Czy wymusić aktualizację nawet jeśli dane są aktualne

        Returns:
            dict: Wyniki analizy sentymentu
        """
        # Sprawdzenie, czy potrzebna jest aktualizacja
        current_time = datetime.now()

        if (not force_update and 
            self.last_update and 
            self.current_sentiment and 
            (current_time - self.last_update).total_seconds() < 3600):
            # Używanie pamięci podręcznej jeśli dane są względnie aktualne (godzina)
            return {
                "success": True,
                "cached": True,
                "timestamp": self.last_update.isoformat(),
                "symbol": symbol,
                "analysis": self.current_sentiment
            }

        # Generowanie danych (w rzeczywistej implementacji byłoby pobieranie z API)
        try:
            sentiment_data = self._collect_sentiment_data(symbol)
            overall_score = self._calculate_overall_sentiment(sentiment_data)

            # Kategoryzacja sentymentu
            if overall_score >= 0.5:
                sentiment_category = "Bardzo Pozytywny"
            elif overall_score >= 0.2:
                sentiment_category = "Pozytywny"
            elif overall_score >= -0.2:
                sentiment_category = "Neutralny"
            elif overall_score >= -0.5:
                sentiment_category = "Negatywny"
            else:
                sentiment_category = "Bardzo Negatywny"

            self.current_sentiment = {
                "score": overall_score,
                "category": sentiment_category,
                "sources": sentiment_data,
                "summary": f"Sentyment dla {symbol}: {sentiment_category} ({overall_score:.2f})"
            }

            self.last_update = current_time

            return {
                "success": True,
                "cached": False,
                "timestamp": current_time.isoformat(),
                "symbol": symbol,
                "analysis": self.current_sentiment
            }
        except Exception as e:
            logger.error(f"Błąd podczas analizy sentymentu: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    def _collect_sentiment_data(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Zbiera dane sentymentu z różnych źródeł.

        Args:
            symbol (str): Symbol rynkowy

        Returns:
            dict: Dane sentymentu z różnych źródeł
        """
        result = {}

        for source in self.sources:
            # Symulowane dane dla celów demonstracyjnych
            if source == "twitter":
                score = random.uniform(-0.8, 0.8)
                volume = random.randint(5000, 50000)
                result[source] = {
                    "score": score,
                    "volume": volume,
                    "keywords": ["crypto", symbol.split('/')[0], "bull", "bear"],
                    "confidence": random.uniform(0.6, 0.9)
                }
            elif source == "news":
                score = random.uniform(-0.6, 0.6)
                articles = random.randint(10, 100)
                result[source] = {
                    "score": score,
                    "articles_count": articles,
                    "top_source": random.choice(["Bloomberg", "CoinDesk", "Reuters", "CNBC"]),
                    "confidence": random.uniform(0.7, 0.95)
                }
            elif source == "reddit":
                score = random.uniform(-0.9, 0.9)
                posts = random.randint(50, 500)
                result[source] = {
                    "score": score,
                    "posts_count": posts,
                    "top_subreddits": ["CryptoCurrency", "Bitcoin", "CryptoMarkets"],
                    "confidence": random.uniform(0.5, 0.85)
                }
            elif source == "tradingview":
                score = random.uniform(-0.7, 0.7)
                ideas = random.randint(5, 30)
                result[source] = {
                    "score": score,
                    "ideas_count": ideas,
                    "technical_sentiment": random.choice(["bullish", "bearish", "neutral"]),
                    "confidence": random.uniform(0.6, 0.9)
                }

        # Symulacja opóźnienia sieci
        time.sleep(0.1)

        return result

    def _calculate_overall_sentiment(self, sentiment_data: Dict[str, Dict[str, Any]]) -> float:
        """
        Oblicza ogólny wskaźnik sentymentu na podstawie danych z różnych źródeł.

        Args:
            sentiment_data (dict): Dane sentymentu z różnych źródeł

        Returns:
            float: Ogólny wskaźnik sentymentu od -1 (bardzo negatywny) do 1 (bardzo pozytywny)
        """
        if not sentiment_data:
            return 0.0

        weighted_scores = []
        weights = []

        # Wagi dla różnych źródeł
        source_weights = {
            "twitter": 0.3,
            "news": 0.25,
            "reddit": 0.2,
            "tradingview": 0.25
        }

        for source, data in sentiment_data.items():
            score = data.get("score", 0)
            confidence = data.get("confidence", 0.7)

            # Waga źródła skorygowana o pewność analizy
            weight = source_weights.get(source, 0.1) * confidence

            weighted_scores.append(score * weight)
            weights.append(weight)

        # Obliczenie ważonej średniej
        if sum(weights) > 0:
            overall_score = sum(weighted_scores) / sum(weights)
        else:
            overall_score = 0.0

        # Ograniczenie wyniku do zakresu -1 do 1
        return max(min(overall_score, 1.0), -1.0)

    def get_current_sentiment(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """
        Zwraca bieżący sentyment dla danego symbolu.

        Args:
            symbol (str): Symbol rynkowy

        Returns:
            dict: Bieżący sentyment lub nowy jeśli poprzedni nie istnieje
        """
        if not self.current_sentiment or not self.last_update:
            return self.analyze(symbol=symbol)

        return {
            "success": True,
            "cached": True,
            "timestamp": self.last_update.isoformat(),
            "symbol": symbol,
            "analysis": self.current_sentiment
        }

# Przykładowe użycie
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(symbol="ETH/USD")
    print(result)
    time.sleep(2)
    cached_result = analyzer.get_current_sentiment("ETH/USD")
    print(cached_result)