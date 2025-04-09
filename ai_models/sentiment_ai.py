"""
sentiment_ai.py - Lekka implementacja analizy sentymentu
"""
import random
import logging
import numpy as np
from datetime import datetime

# Konfiguracja logowania (from original code)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Uproszczona implementacja analizatora sentymentu bazująca na scikit-learn
    zamiast transformers lub innych ciężkich bibliotek.
    """

    def __init__(self, sources=None):
        """
        Inicjalizacja analizatora sentymentu.

        Args:
            sources (list): Lista źródeł, z których pobierany jest sentyment
        """
        self.sources = sources or ["twitter", "news", "reddit", "forum"]
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Zainicjalizowano SentimentAnalyzer z {len(self.sources)} źródłami")
        self.last_update = datetime.now()

        # Inicjalizacja bazowych wartości sentymentu dla każdego źródła
        self.base_sentiments = {}
        for source in self.sources:
            self.base_sentiments[source] = random.uniform(-0.3, 0.3)

    def analyze(self, text=None):
        """
        Analizuje sentyment dla danego tekstu lub generuje symulowane wyniki.

        Args:
            text (str, optional): Tekst do analizy. Jeśli None, generuje symulowane wyniki.

        Returns:
            dict: Wyniki analizy sentymentu
        """
        # Symulowane wyniki, jeśli nie podano tekstu
        if text is None:
            sentiment_values = {}

            # Generowanie wartości sentymentu dla każdego źródła
            for source in self.sources:
                # Dodaj losowe wahanie do bazowego sentymentu
                base = self.base_sentiments[source]
                variance = random.uniform(-0.1, 0.1)
                sentiment_values[source] = max(-1.0, min(1.0, base + variance))

            # Obliczanie średniego sentymentu
            overall = sum(sentiment_values.values()) / len(sentiment_values)

            # Określanie kategorii sentymentu
            if overall > 0.2:
                analysis = "Pozytywny"
            elif overall < -0.2:
                analysis = "Negatywny"
            else:
                analysis = "Neutralny"

            self.last_update = datetime.now()
            return {
                "value": overall,
                "analysis": analysis,
                "sources": sentiment_values,
                "timestamp": self.last_update.strftime("%Y-%m-%d %H:%M:%S")
            }

        # Implementacja analizy rzeczywistego tekstu
        else:
            # Prosta heurystyka oparta na słowach kluczowych
            positive_words = ["wzrost", "zysk", "sukces", "bull", "up", "strong", "good", "positive"]
            negative_words = ["spadek", "strata", "bear", "down", "weak", "bad", "negative"]

            text_lower = text.lower()

            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            # Obliczanie prostego wyniku sentymentu
            if positive_count + negative_count == 0:
                score = 0
            else:
                score = (positive_count - negative_count) / (positive_count + negative_count)

            # Określanie kategorii sentymentu
            if score > 0.2:
                analysis = "Pozytywny"
            elif score < -0.2:
                analysis = "Negatywny"
            else:
                analysis = "Neutralny"

            self.last_update = datetime.now()
            return {
                "value": score,
                "analysis": analysis,
                "text": text,
                "timestamp": self.last_update.strftime("%Y-%m-%d %H:%M:%S")
            }

    def get_status(self):
        """
        Zwraca status analizatora sentymentu.

        Returns:
            dict: Status analizatora
        """
        return {
            "active": True,
            "sources": self.sources,
            "last_update": self.last_update.strftime("%Y-%m-%d %H:%M:%S")
        }

if __name__ == "__main__":
    # Przykładowe użycie
    analyzer = SentimentAnalyzer()

    # Symulowane wyniki
    result = analyzer.analyze()
    print("Symulowane wyniki:", result)

    # Analiza rzeczywistego tekstu
    text_result = analyzer.analyze("Rynek wykazuje silne oznaki wzrostu po ostatnich pozytywnych danych ekonomicznych.")
    print("Analiza tekstu:", text_result)