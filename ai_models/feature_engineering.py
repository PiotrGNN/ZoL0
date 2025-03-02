"""
feature_engineering.py
----------------------
Moduł zawierający funkcje do inżynierii cech dla danych finansowych.

Zawiera zarówno proste funkcje (add_rsi, add_macd, feature_pipeline) jak i bardziej
zaawansowane (compute_rsi, compute_macd, compute_bollinger_bands, itp.).

Możesz użyć dowolnego zestawu tych funkcji w swoim pipeline, w zależności od potrzeb.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------- Podstawowe funkcje (z poprzedniego pliku) --------------------
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Dodaje kolumnę 'rsi' do DataFrame z cenami w kolumnie 'close'.
    """
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Dodaje kolumny 'macd' i 'macd_signal' do DataFrame z cenami w kolumnie 'close'.
    """
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    return df


def feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline inżynierii cech: dodaje RSI, MACD, usuwa zbędne kolumny pomocnicze itp.
    """
    df = add_rsi(df, period=14)
    df = add_macd(df, fast=12, slow=26, signal=9)
    # Możesz usunąć kolumny tymczasowe, np. 'ema_fast', 'ema_slow', jeśli nie chcesz ich w finalnym DF
    # df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True)
    return df


# -------------------- Zaawansowane funkcje (nowe) --------------------
def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Oblicza RSI dla serii cen. Bardziej rozbudowana wersja,
    korzystająca z wykładniczych średnich kroczących.
    """
    try:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))
        logging.info("RSI obliczone (okno=%d).", window)
        return rsi
    except Exception as e:
        logging.error("Błąd przy obliczaniu RSI: %s", e)
        raise


def compute_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Oblicza MACD, linię sygnału oraz histogram.
    Zwraca DataFrame z kolumnami 'MACD', 'Signal', 'Histogram'.
    """
    try:
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal

        df_macd = pd.DataFrame({"MACD": macd, "Signal": signal, "Histogram": histogram})
        logging.info(
            "MACD obliczone (fast=%d, slow=%d, signal=%d).",
            fast_period,
            slow_period,
            signal_period,
        )
        return df_macd
    except Exception as e:
        logging.error("Błąd przy obliczaniu MACD: %s", e)
        raise


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Oblicza Bollinger Bands na podstawie serii cen.
    Zwraca DataFrame z kolumnami 'Middle', 'Upper', 'Lower'.
    """
    try:
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std

        bands = pd.DataFrame({"Middle": middle, "Upper": upper, "Lower": lower})
        logging.info("Bollinger Bands obliczone (okno=%d, odch.std=%.2f).", window, num_std)
        return bands
    except Exception as e:
        logging.error("Błąd przy obliczaniu Bollinger Bands: %s", e)
        raise


def log_price_transformation(prices: pd.Series) -> pd.Series:
    """
    Zwraca logarytmiczną transformację cen.
    """
    try:
        log_prices = np.log(prices.replace(0, np.nan)).dropna()
        logging.info("Logarytmiczna transformacja cen wykonana pomyślnie.")
        return log_prices
    except Exception as e:
        logging.error("Błąd przy transformacji logarytmicznej cen: %s", e)
        raise


def create_lag_features(df: pd.DataFrame, column: str, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Tworzy cechy opóźnione (lag features) dla wskazanej kolumny.
    """
    try:
        df_lagged = df.copy()
        for lag in lags:
            new_col = f"{column}_lag_{lag}"
            df_lagged[new_col] = df_lagged[column].shift(lag)
            logging.info("Stworzono cechę lag: %s", new_col)
        return df_lagged
    except Exception as e:
        logging.error("Błąd przy tworzeniu cech lag: %s", e)
        raise


def compute_moving_average(prices: pd.Series, window: int = 10) -> pd.Series:
    """
    Oblicza średnią kroczącą dla serii cen.
    """
    try:
        ma = prices.rolling(window=window).mean()
        logging.info("Średnia krocząca obliczona (okno=%d).", window)
        return ma
    except Exception as e:
        logging.error("Błąd przy obliczaniu średniej kroczącej: %s", e)
        raise


def adaptive_feature_selection(
    df: pd.DataFrame, target_column: str, volatility_threshold: float = 0.02
) -> pd.DataFrame:
    """
    Dobiera cechy dynamicznie w zależności od zmienności rynku.
    Jeżeli zmienność przekracza próg, dodaje cechy lag, MA, RSI, MACD.
    """
    try:
        df_selected = df.copy()
        price_std = df[target_column].pct_change().std()
        logging.info("Zmienność (std) cen: %f", price_std)
        if price_std > volatility_threshold:
            logging.info("Zmienność przekracza próg. Dodawanie cech opóźnionych i wskaźników technicznych.")
            # Dodaj cechy lag
            df_selected = create_lag_features(df_selected, target_column, lags=[1, 2, 3])
            # Dodaj średnią kroczącą
            df_selected[f"{target_column}_ma_10"] = compute_moving_average(df_selected[target_column], window=10)
            # Dodaj RSI
            df_selected[f"{target_column}_rsi"] = compute_rsi(df_selected[target_column], window=14)
            # Dodaj MACD
            macd_df = compute_macd(df_selected[target_column])
            df_selected = pd.concat([df_selected, macd_df], axis=1)
        else:
            logging.info("Zmienność poniżej progu. Używane są tylko podstawowe cechy.")
        return df_selected.dropna()
    except Exception as e:
        logging.error("Błąd w adaptive_feature_selection: %s", e)
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Weryfikuje jakość danych: sprawdza brakujące wartości oraz typy kolumn.
    """
    try:
        if df.isnull().values.any():
            missing = df.isnull().sum().sum()
            logging.error("Wykryto %d brakujących wartości w danych.", missing)
            raise ValueError(f"Dane zawierają {missing} brakujących wartości.")
        if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
            logging.error("Wykryto kolumny z nie-numerycznymi typami danych.")
            raise ValueError("Nie wszystkie kolumny są numeryczne.")
        logging.info("Walidacja danych przebiegła pomyślnie.")
        return True
    except Exception as e:
        logging.error("Błąd podczas walidacji danych: %s", e)
        raise


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = pd.Series(np.random.uniform(50, 150, size=100), index=dates, name="close")
        df_example = pd.DataFrame(prices)

        # Walidacja danych
        validate_data(df_example)

        # Proste wskaźniki RSI i MACD
        df_example = add_rsi(df_example, period=14)
        df_example = add_macd(df_example, fast=12, slow=26, signal=9)

        # Bardziej zaawansowane wskaźniki
        macd_df = compute_macd(df_example["close"])
        df_example = pd.concat([df_example, macd_df], axis=1)
        bollinger_df = compute_bollinger_bands(df_example["close"])
        df_example = pd.concat([df_example, bollinger_df], axis=1)
        df_example["log_close"] = log_price_transformation(df_example["close"])
        df_features = adaptive_feature_selection(df_example, "close", volatility_threshold=0.02)

        logging.info("Przykładowe dane z cechami:\n%s", df_features.head())
    except Exception as e:
        logging.error("Wystąpił błąd w przykładowym użyciu: %s", e)
