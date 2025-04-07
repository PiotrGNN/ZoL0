"""
sentiment_analysis.py
---------------------
Moduł integrujący dane z mediów społecznościowych, newsów i forów w celu analizy sentymentu.
Wykorzystuje techniki NLP, takie jak modele Transformer (np. BERT, DistilBERT), do klasyfikacji treści finansowych.
Funkcjonalności:
- Wykrywanie sentymentu w pojedynczych tekstach.
- Agregacja wyników sentymentu w przedziałach czasowych, umożliwiająca ich integrację z danymi cenowymi.
- Wykrywanie kluczowych słów (np. "FUD", "HODL", "moon", "crash") i przypisywanie im wagi w zależności od kontekstu.
- Logowanie oraz generowanie alarmów w przypadku gwałtownych zmian sentymentu (np. masowy negatywny wydźwięk).
- Skalowalność przy przetwarzaniu dużych strumieni danych w czasie rzeczywistym.
"""

import logging
from typing import Dict, List, Optional

try:
    from transformers import Pipeline, pipeline
except ImportError as e:
    raise ImportError(
        "Biblioteka transformers nie jest zainstalowana. Zainstaluj ją używając 'pip install transformers'"
    ) from e

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class SentimentAnalysis:
    def __init__(self, model_name: Optional[str] = None):
        """
        Inicjalizacja modułu sentiment analysis.

        Parameters:
            model_name (str, optional): Nazwa modelu Transformer do analizy sentymentu.
                                        Domyślnie używany jest "distilbert-base-uncased-finetuned-sst-2-english".
        """
        self.model_name = (
            model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        )
        try:
            self.nlp: Pipeline = pipeline(
                "sentiment-analysis", model=self.model_name, tokenizer=self.model_name
            )
            logging.info("Załadowano model sentymentu: %s", self.model_name)
        except Exception as e:
            logging.error("Błąd przy ładowaniu modelu sentymentu: %s", e)
            raise

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analizuje pojedynczy tekst i zwraca wynik sentymentu.

        Parameters:
            text (str): Tekst do analizy.

        Returns:
            dict: Słownik zawierający etykietę (LABEL_0: negatywny, LABEL_1: pozytywny) i przypisane prawdopodobieństwo.
                  Dla lepszej interpretacji przemapowujemy etykiety na "NEGATIVE" i "POSITIVE".
        """
        try:
            result = self.nlp(text)
            if result and isinstance(result, list) and "label" in result[0]:
                label = result[0]["label"]
                score = result[0]["score"]
                mapped = (
                    "POSITIVE"
                    if label.upper() in ["POSITIVE", "LABEL_1"]
                    else "NEGATIVE"
                )
                logging.info(
                    'Analiza tekstu: "%s" -> %s (score: %.4f)', text, mapped, score
                )
                return {mapped: score}
            else:
                logging.warning("Nieoczekiwany format wyniku dla tekstu: %s", text)
                return {}
        except Exception as e:
            logging.error("Błąd podczas analizy tekstu: %s", e)
            return {}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analizuje listę tekstów i zwraca listę wyników sentymentu.

        Parameters:
            texts (List[str]): Lista tekstów.

        Returns:
            List[dict]: Lista wyników dla każdego tekstu.
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results

    def aggregate_sentiment(
        self, texts: List[str], weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Agreguje wyniki sentymentu z listy tekstów. Umożliwia przypisanie wag do poszczególnych tekstów.

        Parameters:
            texts (List[str]): Lista tekstów do analizy.
            weights (List[float], optional): Lista wag odpowiadających tekstom. Jeśli None, wszystkie teksty mają równą wagę.

        Returns:
            dict: Agregowany wynik sentymentu, np. {"NEGATIVE": 0.4, "POSITIVE": 0.6}
        """
        aggregated = {"NEGATIVE": 0.0, "POSITIVE": 0.0}
        results = self.analyze_batch(texts)
        if weights is None or len(weights) != len(results):
            weights = [1.0] * len(results)
        total_weight = sum(weights)
        for result, weight in zip(results, weights):
            for key, score in result.items():
                aggregated[key] += score * weight
        if total_weight > 0:
            aggregated = {key: val / total_weight for key, val in aggregated.items()}
        logging.info("Zagregowany sentyment: %s", aggregated)
        return aggregated

    def alert_on_sentiment_shift(
        self,
        current_sentiment: Dict[str, float],
        previous_sentiment: Dict[str, float],
        shift_threshold: float = 0.2,
    ) -> bool:
        """
        Sprawdza, czy nastąpiła gwałtowna zmiana sentymentu.

        Parameters:
            current_sentiment (dict): Aktualny wynik sentymentu.
            previous_sentiment (dict): Poprzedni wynik sentymentu.
            shift_threshold (float): Próg zmiany, powyżej którego generowany jest alarm.

        Returns:
            bool: True, jeśli zmiana przekracza próg, w przeciwnym razie False.
        """
        try:
            shift = {
                key: abs(current_sentiment.get(key, 0) - previous_sentiment.get(key, 0))
                for key in set(current_sentiment.keys()).union(
                    previous_sentiment.keys()
                )
            }
            max_shift = max(shift.values()) if shift else 0.0
            if max_shift >= shift_threshold:
                logging.warning(
                    "Wykryto gwałtowną zmianę sentymentu: %s (próg: %.2f)",
                    shift,
                    shift_threshold,
                )
                return True
            return False
        except Exception as e:
            logging.error("Błąd przy ocenie zmiany sentymentu: %s", e)
            return False


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        sa = SentimentAnalysis()
        sample_texts = [
            "Investors are bullish after the recent market rally.",
            "There are fears of an upcoming market crash due to economic instability.",
            "The overall sentiment remains neutral with mixed signals from various news sources.",
        ]
        # Analiza pojedynczych tekstów
        for text in sample_texts:
            print(sa.analyze_text(text))

        # Agregacja sentymentu dla partii tekstów
        aggregated = sa.aggregate_sentiment(sample_texts)
        print("Agregowany sentyment:", aggregated)

        # Przykładowa ocena zmiany sentymentu
        previous = {"NEGATIVE": 0.3, "POSITIVE": 0.7}
        current = aggregated
        if sa.alert_on_sentiment_shift(current, previous):
            logging.info("Alarm: Wykryto znaczną zmianę sentymentu!")
        else:
            logging.info("Brak istotnej zmiany sentymentu.")
    except Exception as e:
        logging.error("Błąd w module sentiment_analysis.py: %s", e)
        raise
"""
sentiment_analysis.py
---------------------
Moduł do analizy sentymentu rynkowego oparty na danych tekstowych.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class SentimentIndicator:
    """
    Klasa do obliczania wskaźników sentymentu rynkowego.
    """
    
    def __init__(self, default_window: int = 14):
        """
        Inicjalizacja wskaźnika sentymentu.
        
        Args:
            default_window: Domyślne okno czasowe (w dniach) dla obliczeń
        """
        self.default_window = default_window
        logger.info(f"Inicjalizacja SentimentIndicator z oknem {default_window}")
        
    def calculate_sentiment_score(self, texts: List[str], weights: Optional[List[float]] = None) -> float:
        """
        Oblicza prosty wskaźnik sentymentu na podstawie listy tekstów.
        
        Args:
            texts: Lista tekstów do analizy
            weights: Opcjonalne wagi dla każdego tekstu
            
        Returns:
            float: Wskaźnik sentymentu w zakresie [-1, 1]
        """
        if not texts:
            return 0.0
            
        # W rzeczywistym systemie użylibyśmy tu modelu NLP,
        # na przykład z modułu ai_models.sentiment_ai
        # Na potrzeby demonstracji używamy prostego podejścia losowego
        
        # Symulacja wyników analizy sentymentu (losowe wartości)
        random_scores = np.random.uniform(-1, 1, len(texts))
        
        if weights is None:
            weights = [1.0] * len(texts)
            
        # Obliczenie ważonej średniej
        total_weight = sum(weights)
        weighted_score = sum(score * weight for score, weight in zip(random_scores, weights)) / total_weight
        
        logger.info(f"Obliczono wskaźnik sentymentu: {weighted_score:.4f}")
        return weighted_score
        
    def market_mood_index(self, sentiment_data: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Oblicza indeks nastrojów rynkowych jako średnią kroczącą sentymentu.
        
        Args:
            sentiment_data: Seria pandas zawierająca dane sentymentu
            window: Rozmiar okna dla średniej kroczącej
            
        Returns:
            pd.Series: Indeks nastrojów rynkowych
        """
        if window is None:
            window = self.default_window
            
        mood_index = sentiment_data.rolling(window=window).mean()
        logger.info(f"Obliczono indeks nastrojów rynkowych z oknem {window}")
        return mood_index
        
    def sentiment_oscillator(self, sentiment_data: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Tworzy oscylator sentymentu, pokazujący zmiany nastrojów.
        
        Args:
            sentiment_data: Seria pandas zawierająca dane sentymentu
            window: Rozmiar okna dla oscylatora
            
        Returns:
            pd.Series: Oscylator sentymentu
        """
        if window is None:
            window = self.default_window
        
        # Obliczenie różnicy między bieżącym sentymentem a średnią kroczącą
        mood_index = self.market_mood_index(sentiment_data, window)
        oscillator = sentiment_data - mood_index
        
        logger.info(f"Obliczono oscylator sentymentu z oknem {window}")
        return oscillator
        
    def get_market_sentiment_signals(self, sentiment_data: pd.Series) -> Dict[str, float]:
        """
        Generuje sygnały handlowe na podstawie danych sentymentu.
        
        Args:
            sentiment_data: Seria pandas zawierająca dane sentymentu
            
        Returns:
            Dict: Słownik z sygnałami handlowymi
        """
        if len(sentiment_data) < self.default_window:
            return {"sentiment_signal": 0, "confidence": 0.0, "market_mood": "Neutral"}
            
        recent_sentiment = sentiment_data.iloc[-1]
        mood_index = self.market_mood_index(sentiment_data).iloc[-1]
        oscillator = self.sentiment_oscillator(sentiment_data).iloc[-1]
        
        # Określenie nastroju rynku
        if mood_index > 0.3:
            market_mood = "Byczący"
            sentiment_signal = 1
        elif mood_index < -0.3:
            market_mood = "Niedźwiedzi"
            sentiment_signal = -1
        else:
            market_mood = "Neutralny"
            sentiment_signal = 0
            
        # Obliczenie pewności sygnału
        confidence = min(abs(mood_index) * 2, 1.0)
        
        signals = {
            "sentiment_signal": sentiment_signal,
            "confidence": confidence,
            "market_mood": market_mood,
            "recent_sentiment": recent_sentiment,
            "mood_index": mood_index,
            "oscillator": oscillator
        }
        
        logger.info(f"Wygenerowano sygnały sentymentu: {signals}")
        return signals

# -------------------- Przykładowe testy --------------------

def generate_sample_sentiment_data(days: int = 30) -> pd.Series:
    """Generuje przykładowe dane sentymentu do testów."""
    dates = pd.date_range(end=datetime.datetime.now(), periods=days)
    # Symulacja trendu sentymentu z szumem
    trend = np.linspace(-0.5, 0.5, days)
    noise = np.random.normal(0, 0.2, days)
    sentiment = np.clip(trend + noise, -1, 1)
    return pd.Series(sentiment, index=dates, name="sentiment")

if __name__ == "__main__":
    # Przykładowe użycie
    indicator = SentimentIndicator()
    
    # Przykładowa analiza tekstów
    sample_texts = [
        "Rynek wykazuje silne oznaki wzrostu, inwestorzy są optymistyczni.",
        "Niepewność na rynkach globalnych rośnie, analitycy zalecają ostrożność.",
        "Stabilny wzrost, ale niskie wolumeny transakcji sugerują brak przekonania."
    ]
    sample_weights = [1.2, 1.0, 0.8]
    
    sentiment_score = indicator.calculate_sentiment_score(sample_texts, sample_weights)
    print(f"Wynik analizy sentymentu: {sentiment_score:.4f}")
    
    # Przykładowa analiza historyczna
    sample_data = generate_sample_sentiment_data(60)
    mood_index = indicator.market_mood_index(sample_data)
    oscillator = indicator.sentiment_oscillator(sample_data)
    
    signals = indicator.get_market_sentiment_signals(sample_data)
    print(f"Sygnały sentymentu rynkowego: {signals}")
    
    print("Test modułu sentiment_analysis zakończony pomyślnie.")
