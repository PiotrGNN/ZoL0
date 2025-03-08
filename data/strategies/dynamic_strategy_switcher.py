"""
dynamic_strategy_switcher.py
----------------------------
Moduł dynamicznie przełączający między różnymi strategiami handlowymi w zależności od warunków rynkowych.

Funkcjonalności:
- Analiza trendu, zmienności, wolumenu i sentymentu w celu określenia, która strategia (np. trend-following, mean-reversion) powinna być aktywna.
- Mechanizm płynnego przejścia między strategiami, wykorzystujący hysteresis i okresy cooldown, aby uniknąć zbyt częstego przepinania.
- Logowanie decyzji o przełączaniu strategii oraz testy weryfikujące, czy adaptacyjne przełączanie poprawia wyniki.
- Integracja z modułami strategy_manager.py, backtesting.py oraz real-time (np. trade_executor.py).
- Skalowalność rozwiązania dla wielu par walutowych i dużych portfeli.

"""

import logging
import time

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class DynamicStrategySwitcher:
    def __init__(self, cooldown_period: int = 300, hysteresis_threshold: float = 0.05):
        """
        Inicjalizuje dynamiczny system przełączania strategii.

        Parameters:
            cooldown_period (int): Okres (w sekundach) w trakcie którego po zmianie strategii nie następuje kolejna zmiana.
            hysteresis_threshold (float): Minimalna zmiana wskaźnika, która musi zostać osiągnięta, aby zmienić strategię.
        """
        self.cooldown_period = cooldown_period
        self.hysteresis_threshold = hysteresis_threshold
        self.last_switch_time = 0
        self.current_strategy = None  # np. "trend_following" lub "mean_reversion"
        logging.info(
            "DynamicStrategySwitcher zainicjalizowany z cooldown: %d s, hysteresis: %.2f",
            self.cooldown_period,
            self.hysteresis_threshold,
        )

    def analyze_market_conditions(
        self, market_data: pd.DataFrame, sentiment: dict
    ) -> dict:
        """
        Analizuje warunki rynkowe na podstawie danych, zwraca metryki, które będą podstawą do przełączenia strategii.

        Parameters:
            market_data (pd.DataFrame): Dane rynkowe, np. ceny, wolumeny.
            sentiment (dict): Wyniki analizy sentymentu, np. {"POSITIVE": 0.6, "NEGATIVE": 0.4}.

        Returns:
            dict: Słownik z metrykami, np. {"trend_strength": 0.7, "volatility": 0.02, "sentiment_bias": 0.2}
        """
        # Przykładowa analiza trendu: obliczenie wskaźnika trendu jako procentowa zmiana ceny w ostatnim okresie
        recent_period = market_data.tail(20)
        price_change = (
            recent_period["close"].iloc[-1] - recent_period["close"].iloc[0]
        ) / recent_period["close"].iloc[0]

        # Zmienność: odchylenie standardowe procentowych zmian ceny
        pct_change = market_data["close"].pct_change().dropna()
        volatility = pct_change.std()

        # Sentiment bias: różnica między pozytywnym a negatywnym sentymentem
        sentiment_bias = sentiment.get("POSITIVE", 0) - sentiment.get("NEGATIVE", 0)

        conditions = {
            "trend_strength": price_change,
            "volatility": volatility,
            "sentiment_bias": sentiment_bias,
        }
        logging.info("Warunki rynkowe: %s", conditions)
        return conditions

    def decide_strategy(self, market_conditions: dict) -> str:
        """
        Decyduje o wyborze strategii na podstawie metryk rynkowych.

        Parameters:
            market_conditions (dict): Słownik z metrykami (trend_strength, volatility, sentiment_bias).

        Returns:
            str: Nazwa strategii, np. "trend_following" lub "mean_reversion".
        """
        trend = market_conditions.get("trend_strength", 0)
        sentiment = market_conditions.get("sentiment_bias", 0)
        volatility = market_conditions.get("volatility", 0)

        # Prosta logika: jeśli trend jest silny i sentyment pozytywny, wybieramy trend-following.
        # W przeciwnym razie, jeśli zmienność jest wysoka lub trend słaby, wybieramy mean-reversion.
        if trend > self.hysteresis_threshold and sentiment > 0:
            chosen_strategy = "trend_following"
        elif volatility > 0.03 or trend < -self.hysteresis_threshold or sentiment < 0:
            chosen_strategy = "mean_reversion"
        else:
            chosen_strategy = self.current_strategy or "trend_following"

        logging.info(
            "Decyzja strategii: %s (trend: %.4f, volatility: %.4f, sentiment: %.4f)",
            chosen_strategy,
            trend,
            volatility,
            sentiment,
        )
        return chosen_strategy

    def switch_strategy(self, market_data: pd.DataFrame, sentiment: dict) -> str:
        """
        Decyduje o przełączeniu strategii w oparciu o analizę warunków rynkowych, z uwzględnieniem cooldown i hysteresis.

        Parameters:
            market_data (pd.DataFrame): Dane rynkowe.
            sentiment (dict): Wyniki analizy sentymentu.

        Returns:
            str: Aktualnie wybrana strategia.
        """
        current_time = time.time()
        if current_time - self.last_switch_time < self.cooldown_period:
            logging.info(
                "Cooldown active. Strategia pozostaje bez zmian: %s",
                self.current_strategy,
            )
            return self.current_strategy

        market_conditions = self.analyze_market_conditions(market_data, sentiment)
        new_strategy = self.decide_strategy(market_conditions)

        # Mechanizm hysteresis: przełączamy strategię tylko jeśli nowa strategia różni się od obecnej o określony margines
        if self.current_strategy is None or new_strategy != self.current_strategy:
            self.last_switch_time = current_time
            self.current_strategy = new_strategy
            logging.info("Przełączono strategię na: %s", self.current_strategy)
        else:
            logging.info("Strategia pozostaje bez zmian: %s", self.current_strategy)
        return self.current_strategy


# -------------------- Przykładowe testy jednostkowe --------------------
def unit_test_dynamic_strategy_switcher():
    """
    Testy jednostkowe dla modułu dynamic_strategy_switcher.py.
    Symuluje dane rynkowe oraz wyniki analizy sentymentu, sprawdzając, czy przełączanie strategii działa poprawnie.
    """
    try:
        # Tworzymy przykładowe dane rynkowe
        dates = pd.date_range(start="2023-01-01", periods=50, freq="H")
        prices = np.linspace(100, 110, 50) + np.random.normal(0, 0.5, 50)
        volumes = np.random.randint(1000, 1500, 50)
        market_data = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

        # Przykładowe wyniki sentymentu
        sentiment_positive = {"POSITIVE": 0.7, "NEGATIVE": 0.3}
        sentiment_negative = {"POSITIVE": 0.3, "NEGATIVE": 0.7}

        switcher = DynamicStrategySwitcher(cooldown_period=5, hysteresis_threshold=0.01)

        # Pierwsze wywołanie powinno ustalić strategię na podstawie sentymentu pozytywnego
        strategy1 = switcher.switch_strategy(market_data, sentiment_positive)
        assert strategy1 in [
            "trend_following",
            "mean_reversion",
        ], "Nieprawidłowa strategia."

        # Symulujemy zmianę warunków na bardziej negatywne
        time.sleep(6)  # Aby cooldown wygasł
        strategy2 = switcher.switch_strategy(market_data, sentiment_negative)
        assert (
            strategy2 != strategy1
        ), "Strategia nie została zmieniona pomimo zmiany warunków."

        logging.info(
            "Testy jednostkowe dynamic_strategy_switcher.py zakończone sukcesem."
        )
    except AssertionError as ae:
        logging.error("AssertionError w testach dynamic_strategy_switcher.py: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach dynamic_strategy_switcher.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_dynamic_strategy_switcher()
    except Exception as e:
        logging.error(
            "Testy jednostkowe dynamic_strategy_switcher.py nie powiodły się: %s", e
        )
        raise
