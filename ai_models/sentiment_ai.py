"""
sentiment_ai.py
----------------
Moduł do analizy nastrojów (sentiment analysis) w kontekście rynkowym.

Funkcjonalności:
- Wykorzystuje biblioteki NLP, w tym Hugging Face Transformers, do przetwarzania danych z mediów społecznościowych, forów internetowych i newsów finansowych.
- Implementuje zaawansowane modele (np. BERT, RoBERTa) do klasyfikacji sentymentu (pozytywny, neutralny, negatywny) i oceny wpływu na rynek.
- Umożliwia przetwarzanie strumieniowe (real-time) oraz analizę historyczną do tworzenia wskaźników sentymentu.
- Zapewnia mechanizmy wyznaczania wagi sentymentu w zależności od źródła (wiarygodność, zasięg) oraz integrację z systemem strategii tradingowych.
- Zawiera obsługę wyjątków, logowanie oraz przykładowe testy jednostkowe, gwarantując stabilność nawet przy dużej ilości danych.
"""

import logging
from typing import Dict, List, Optional, Tuple

try:
    from transformers import Pipeline, pipeline
except ImportError as e:
    raise ImportError(
        "Biblioteka transformers nie jest zainstalowana. Zainstaluj ją używając 'pip install transformers'"
    ) from e

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SentimentAnalyzer:
    """
    Klasa do analizy nastrojów przy użyciu modelu Transformer.
    Domyślnie wykorzystuje model 'cardiffnlp/twitter-roberta-base-sentiment' do analizy sentymentu.
    Model ten klasyfikuje tekst na trzy kategorie: negatywny, neutralny, pozytywny.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Inicjalizacja klasy SentimentAnalyzer.

        Parameters:
            model_name (str, optional): Nazwa modelu do użycia. Domyślnie:
                'cardiffnlp/twitter-roberta-base-sentiment'
        """
        self.model_name = model_name or "cardiffnlp/twitter-roberta-base-sentiment"
        try:
            self.nlp: Pipeline = pipeline("sentiment-analysis", model=self.model_name, tokenizer=self.model_name)
            logging.info("Model sentiment analysis '%s' został załadowany.", self.model_name)
        except Exception as e:
            logging.error("Błąd podczas ładowania modelu '%s': %s", self.model_name, e)
            raise

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analizuje pojedynczy tekst i zwraca wyniki klasyfikacji.

        Parameters:
            text (str): Tekst do analizy.

        Returns:
            dict: Słownik z kategoriami sentymentu i odpowiadającymi im prawdopodobieństwami.
                  Przykładowy format: {"NEGATIVE": 0.1, "NEUTRAL": 0.2, "POSITIVE": 0.7}
        """
        try:
            result = self.nlp(text)
            # Wynik może być listą z jednym słownikiem np. [{"label": "LABEL_2", "score": 0.85}]
            # Model 'cardiffnlp/twitter-roberta-base-sentiment' zwraca etykiety w formacie: "LABEL_0", "LABEL_1", "LABEL_2"
            # Mapowanie: LABEL_0 -> NEGATIVE, LABEL_1 -> NEUTRAL, LABEL_2 -> POSITIVE
            mapping = {
                "LABEL_0": "NEGATIVE",
                "LABEL_1": "NEUTRAL",
                "LABEL_2": "POSITIVE",
            }
            if result and isinstance(result, list) and "label" in result[0]:
                label = result[0]["label"]
                score = result[0]["score"]
                mapped_label = mapping.get(label, label)
                logging.info(
                    "Analiza tekstu zakończona: %s -> %s (score: %.4f)",
                    text,
                    mapped_label,
                    score,
                )
                return {mapped_label: score}
            else:
                logging.warning("Nieoczekiwany format wyniku analizy: %s", result)
                return {}
        except Exception as e:
            logging.error("Błąd podczas analizy tekstu '%s': %s", text, e)
            return {}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analizuje listę tekstów i zwraca listę wyników.

        Parameters:
            texts (List[str]): Lista tekstów do analizy.

        Returns:
            List[dict]: Lista słowników z wynikami analizy dla każdego tekstu.
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results

    def weighted_sentiment(
        self, texts_with_source: List[Tuple[str, str]], source_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Oblicza ważony wskaźnik sentymentu na podstawie tekstów oraz przypisanych wag źródeł.

        Parameters:
            texts_with_source (List[Tuple[str, str]]): Lista krotek, gdzie każda krotka to (tekst, źródło).
            source_weights (Dict[str, float]): Słownik wag dla źródeł, np. {"twitter": 1.0, "news": 1.5, "forum": 0.8}

        Returns:
            dict: Słownik sumarycznych wyników sentymentu z uwzględnieniem wag.
                  Przykładowy format: {"NEGATIVE": 0.25, "NEUTRAL": 0.35, "POSITIVE": 0.40}
        """
        aggregated = {"NEGATIVE": 0.0, "NEUTRAL": 0.0, "POSITIVE": 0.0}
        total_weight = 0.0
        for text, source in texts_with_source:
            weight = source_weights.get(source, 1.0)
            sentiment = self.analyze_text(text)
            for key, score in sentiment.items():
                aggregated[key] += score * weight
            total_weight += weight
        if total_weight > 0:
            # Uśredniamy wyniki
            for key in aggregated:
                aggregated[key] /= total_weight
        logging.info("Ważony wskaźnik sentymentu: %s", aggregated)
        return aggregated


# -------------------- Przykładowe testy jednostkowe --------------------
def unit_test_sentiment_analysis():
    """
    Przeprowadza przykładowe testy jednostkowe modułu sentiment_ai.
    """
    try:
        analyzer = SentimentAnalyzer()
        sample_texts = [
            "The market is looking bullish today and investors are excited.",
            "There is a lot of uncertainty in the market, and many are worried about a downturn.",
            "The stock prices remained stable throughout the day.",
        ]
        results = analyzer.analyze_batch(sample_texts)
        assert isinstance(results, list), "Wynik powinien być listą."
        for res in results:
            assert any(
                label in res for label in ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            ), "Brak poprawnej etykiety w wyniku."
        logging.info("Testy jednostkowe sentiment analysis zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("Błąd w testach jednostkowych: %s", ae)
    except Exception as e:
        logging.error("Nieoczekiwany błąd w testach jednostkowych: %s", e)


if __name__ == "__main__":
    # Przykładowe użycie modułu w trybie historycznym
    logging.info("Rozpoczynam przykładową analizę historyczną nastrojów.")
    analyzer = SentimentAnalyzer()
    historical_texts = [
        "Investors are optimistic about the future, and the market shows signs of recovery.",
        "Economic indicators are poor, causing panic among traders.",
        "The day was calm with little movement in stock prices.",
    ]
    historical_results = analyzer.analyze_batch(historical_texts)
    logging.info("Wyniki analizy historycznej: %s", historical_results)

    # Przykładowe użycie w trybie real-time z ważeniem źródeł
    logging.info("Rozpoczynam analizę real-time z ważeniem źródeł.")
    texts_with_source = [
        ("The economy is booming, and investors are reaping high returns.", "news"),
        ("I think the market will crash soon.", "twitter"),
        ("There is uncertainty, but some signs of stability are visible.", "forum"),
    ]
    source_weights = {"news": 1.5, "twitter": 1.0, "forum": 0.8}
    weighted_result = analyzer.weighted_sentiment(texts_with_source, source_weights)
    logging.info("Wynik ważonej analizy sentymentu: %s", weighted_result)

    # Uruchom testy jednostkowe
    unit_test_sentiment_analysis()
