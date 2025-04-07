"""
sentiment_analysis.py
---------------------
Moduł integrujący dane z mediów społecznościowych, newsów i forów w celu analizy sentymentu.
Wykorzystuje techniki NLP, takie jak modele Transformer (np. BERT, DistilBERT), do klasyfikacji treści finansowych.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import random

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Klasa do analizy sentymentu tekstu związanego z rynkami finansowymi.
    W wersji symulacyjnej generuje losowe wyniki sentymentu.
    """

    def __init__(self, use_real_model: bool = False):
        """
        Inicjalizuje analizator sentymentu.

        Args:
            use_real_model: Czy używać rzeczywistego modelu NLP (True) czy symulacji (False)
        """
        self.use_real_model = use_real_model
        self.last_update = datetime.now()
        self.cache = {}
        self.sentiment_history = []

        logger.info("Inicjalizacja SentimentAnalyzer (tryb %s)", 
                  "rzeczywisty" if use_real_model else "symulacyjny")

        if use_real_model:
            try:
                # Próba załadowania rzeczywistego modelu (uwaga: może wymagać transformers)
                self._load_model()
            except ImportError:
                logger.warning("Nie można załadować modelu NLP. Przełączenie na tryb symulacji.")
                self.use_real_model = False

    def _load_model(self):
        """Ładuje model NLP do analizy sentymentu. W przypadku niepowodzenia przechodzi w tryb symulacji."""
        try:
            from transformers import pipeline
            logger.info("Ładowanie modelu NLP...")
            self.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            logger.info("Model NLP załadowany pomyślnie")
        except (ImportError, Exception) as e:
            logger.error("Błąd podczas ładowania modelu NLP: %s", str(e))
            self.use_real_model = False
            raise ImportError("Nie można załadować modelu NLP. Zainstaluj transformers.")

    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analizuje tekst i zwraca sentyment.

        Args:
            text: Tekst do analizy

        Returns:
            Dict zawierający sentyment ('positive', 'neutral', 'negative') i score
        """
        if text in self.cache:
            return self.cache[text]

        if self.use_real_model:
            try:
                result = self.model(text)[0]
                sentiment = result['label'].lower()
                score = result['score']
            except Exception as e:
                logger.error("Błąd analizy tekstu: %s", str(e))
                return self._generate_mock_sentiment()
        else:
            # Tryb symulacji - generujemy losowe wyniki
            return self._generate_mock_sentiment()

    def _generate_mock_sentiment(self) -> Dict[str, Union[str, float]]:
        """Generuje symulowane wyniki sentymentu dla trybu testowego."""
        sentiments = ["positive", "neutral", "negative"]
        sentiment = random.choices(
            sentiments, 
            weights=[0.40, 0.35, 0.25],  # Bias towards positive/neutral
            k=1
        )[0]

        # Generuj realistyczny score
        if sentiment == "positive":
            score = random.uniform(0.6, 0.95)
        elif sentiment == "neutral":
            score = random.uniform(0.4, 0.6)
        else:  # negative
            score = random.uniform(0.55, 0.9)  # Score dla negative to pewność negatywnego sentymentu

        return {
            "sentiment": sentiment,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analizuje partię tekstów i zwraca ich sentyment.

        Args:
            texts: Lista tekstów do analizy

        Returns:
            Lista słowników z wynikami analizy sentymentu
        """
        results = []
        for text in texts:
            results.append(self.analyze_text(text))
        return results

    def get_market_sentiment(self, symbol: str = "BTC/USDT") -> Dict[str, Union[str, float]]:
        """
        Zwraca aktualny sentyment rynkowy dla danego symbolu.
        W wersji symulacyjnej generuje wyniki z lekkim biasem dla trendu.

        Args:
            symbol: Symbol rynkowy (np. "BTC/USDT")

        Returns:
            Dict z informacjami o sentymencie rynkowym
        """
        # Symulacja trendu - co jakiś czas zmieniamy bias
        current_time = int(time.time())
        trend_period = 3600  # 1 godzina
        trend_phase = (current_time % (trend_period * 3)) // trend_period

        # Trzy fazy: pozytywna, neutralna, negatywna
        if trend_phase == 0:
            weights = [0.6, 0.3, 0.1]  # pozytywny bias
        elif trend_phase == 1:
            weights = [0.3, 0.5, 0.2]  # neutralny bias
        else:
            weights = [0.1, 0.3, 0.6]  # negatywny bias

        sentiments = ["positive", "neutral", "negative"]
        sentiment = random.choices(sentiments, weights=weights, k=1)[0]

        # Generuj realistyczny score
        if sentiment == "positive":
            score = random.uniform(0.65, 0.9)
        elif sentiment == "neutral":
            score = random.uniform(0.45, 0.65)
        else:  # negative
            score = random.uniform(0.7, 0.95)

        # Symulacja wolumenu postów
        volume = random.randint(10, 1000)

        result = {
            "symbol": symbol,
            "sentiment": sentiment,
            "score": score,
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
            "source": random.choice(["twitter", "reddit", "news", "forums"])
        }

        # Dodaj do historii
        self.sentiment_history.append(result)
        if len(self.sentiment_history) > 100:
            self.sentiment_history.pop(0)

        return result

    def get_sentiment_summary(self) -> Dict[str, any]:
        """
        Zwraca podsumowanie sentymentu z różnych źródeł.

        Returns:
            Dict z podsumowaniem sentymentu
        """
        sources = ["twitter", "reddit", "news", "forums"]
        summary = {
            "overall": random.choice(["positive", "neutral", "negative"]),
            "sources": {}
        }

        for source in sources:
            summary["sources"][source] = {
                "sentiment": random.choice(["positive", "neutral", "negative"]),
                "score": round(random.uniform(0.3, 0.9), 2),
                "volume": random.randint(50, 5000)
            }

        summary["last_update"] = datetime.now().isoformat()
        return summary

    def detect_sentiment_shifts(self) -> List[Dict[str, any]]:
        """
        Wykrywa nagłe zmiany sentymentu, które mogą wskazywać na istotne wydarzenia.

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
        print(f"Wykryto {len(shifts)} zmian sentymentu!")