
"""
sentiment_analysis.py
--------------------
Moduł do analizy sentymentu z różnych źródeł (media społecznościowe, newsy, raporty)
w celu określenia nastrojów rynkowych.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analizator sentymentu dla danych rynkowych i wiadomości.
    W wersji produkcyjnej korzystałby z API (np. Twitter/X, NewsAPI) 
    i modeli NLP do analizy tekstów.
    """

    def __init__(self, use_external_api: bool = False):
        """
        Inicjalizacja analizatora sentymentu.
        
        Args:
            use_external_api: Czy używać zewnętrznych API (np. Twitter/X, NewsAPI)
        """
        self.use_external_api = use_external_api
        self.cache = {}  # Prosty cache ostatnich wyników
        logger.info("Inicjalizacja SentimentAnalyzer, external_api=%s", use_external_api)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analizuje sentyment pojedynczego tekstu.
        
        Args:
            text: Tekst do analizy
            
        Returns:
            Dict zawierający wynik analizy sentymentu
        """
        # W rzeczywistej aplikacji używalibyśmy modelu NLP
        # W wersji symulacyjnej generujemy losowy wynik
        
        # Losowa wartość sentymentu (od -1 do 1)
        sentiment_score = random.uniform(-1, 1)
        
        # Mapowanie wyniku na kategorię
        if sentiment_score > 0.3:
            sentiment = "positive"
        elif sentiment_score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        result = {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "sentiment": sentiment,
            "score": sentiment_score,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug("Analiza tekstu: %s, wynik: %s", text[:30], sentiment)
        return result

    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Pobiera sentyment rynkowy dla danego symbolu.
        
        Args:
            symbol: Symbol rynkowy (np. BTC/USDT)
            
        Returns:
            Dict zawierający zagregowany sentyment rynkowy
        """
        # Sprawdź cache
        if symbol in self.cache and (datetime.now() - self.cache[symbol]["timestamp"]).seconds < 300:
            logger.debug("Używam zbuforowanego sentymentu dla %s", symbol)
            return self.cache[symbol]
            
        # W rzeczywistej aplikacji pobieralibyśmy dane z różnych źródeł
        # W wersji symulacyjnej generujemy losowy wynik
        social_sentiment = random.uniform(-1, 1)
        news_sentiment = random.uniform(-1, 1)
        
        # Wagi dla różnych źródeł
        social_weight = 0.4
        news_weight = 0.6
        
        # Ważona suma
        combined_score = (social_sentiment * social_weight) + (news_sentiment * news_weight)
        
        # Mapowanie wyniku na kategorię
        if combined_score > 0.3:
            sentiment = "positive"
        elif combined_score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        result = {
            "symbol": symbol,
            "sentiment": sentiment,
            "score": combined_score,
            "sources": {
                "social": social_sentiment,
                "news": news_sentiment
            },
            "timestamp": datetime.now()
        }
        
        # Aktualizacja cache
        self.cache[symbol] = result
        
        logger.info("Sentyment rynkowy dla %s: %s (%.2f)", symbol, sentiment, combined_score)
        return result

    def detect_sentiment_shifts(self) -> List[Dict[str, Any]]:
        """
        Wykrywa istotne zmiany sentymentu, które mogą wskazywać na istotne wydarzenia.

        Returns:
            Lista wykrytych zmian sentymentu
        """
        # W wersji symulacyjnej czasami zwracamy alertu
        alerts = []
        if random.random() < 0.1:  # 10% szans na alert
            alerts.append({
                "type": "sentiment_shift",
                "from": random.choice(["positive", "neutral"]),
                "to": "negative",
                "magnitude": round(random.uniform(0.3, 0.9), 2),
                "timestamp": datetime.now().isoformat(),
                "symbol": random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
            })
        return alerts

    def get_sentiment_history(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Zwraca historię sentymentu dla symbolu z określonej liczby dni.
        
        Args:
            symbol: Symbol rynkowy
            days: Liczba dni historii
            
        Returns:
            Lista wyników sentymentu z każdego dnia
        """
        history = []
        now = datetime.now()
        
        # Generowanie symulowanych danych historycznych
        for i in range(days):
            day_offset = i
            date = datetime(now.year, now.month, now.day) - datetime.timedelta(days=day_offset)
            
            # Losowy sentyment z trendem (bardziej pozytywny dla starszych dni)
            trend_factor = i / days  # Im starszy dzień, tym wyższy współczynnik
            score = random.uniform(-0.8, 0.8) + trend_factor * 0.3
            
            # Mapowanie wyniku na kategorię
            if score > 0.3:
                sentiment = "positive"
            elif score < -0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"
                
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "sentiment": sentiment,
                "score": score,
                "volume": random.randint(100, 1000)  # Symulowana ilość wzmianek
            })
            
        return history
    
    def analyze_news_impact(self, symbol: str) -> Dict[str, Any]:
        """
        Analizuje wpływ najnowszych wiadomości na cenę instrumentu.
        
        Args:
            symbol: Symbol rynkowy
            
        Returns:
            Wynik analizy wpływu wiadomości
        """
        # Symulacja wyników
        impact_score = random.uniform(-1, 1)
        
        return {
            "symbol": symbol,
            "impact_score": impact_score,
            "news_count": random.randint(5, 30),
            "top_keywords": ["regulation", "adoption", "technology", "market"],
            "timestamp": datetime.now().isoformat()
        }

# Przykładowe użycie modułu
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Analiza pojedynczego tekstu
    result = analyzer.analyze_text("Bitcoin może osiągnąć nowy ATH w tym roku!")
    print(f"Sentyment: {result['sentiment']}, Score: {result['score']:.2f}")

    # Pobieranie sentymentu rynkowego
    market_sentiment = analyzer.get_market_sentiment("BTC/USDT")
    print(f"Sentyment rynkowy dla BTC/USDT: {market_sentiment['sentiment']}")

    # Wykrywanie zmian sentymentu
    shifts = analyzer.detect_sentiment_shifts()
    if shifts:
        print(f"Wykryto {len(shifts)} istotnych zmian sentymentu!")
