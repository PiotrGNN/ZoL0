"""
backtesting.py
--------------
Skrypt do przeprowadzania backtestingu strategii handlowych na danych historycznych.

Funkcjonalności:
- Implementacja walk-forward analysis, dzieląc dane na kolejne okresy treningowe i testowe.
- Obsługa różnych interwałów czasowych oraz równoczesne testowanie wielu strategii.
- Porównywanie wyników strategii i generowanie zbiorczego raportu z metrykami takimi jak
  CAGR, max drawdown, Sharpe Ratio.
- Uwzględnienie prowizji, spreadów i slippage dla realistycznych wyników.
- Logowanie przebiegu backtestingu oraz integracja z modułem trade_logger.py
  dla spójności z realnymi warunkami.
- Skalowalność kodu przy obsłudze dużych zbiorów danych historycznych.

Uwaga: Przykładowa implementacja; strategia tradingowa jest symulowana przez funkcję przykładową.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Import modułu trade_logger (zakładamy, że jest w ścieżce data/logging/)
try:
    from data.logging.trade_logger import TradeLogger
except ImportError:
    TradeLogger = None
    logging.warning("Nie znaleziono modułu trade_logger. Logowanie transakcji będzie ograniczone.")


def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
    """
    Oblicza CAGR (Compound Annual Growth Rate).

    Parameters:
        initial_value (float): Kapitał początkowy.
        final_value (float): Kapitał końcowy.
        years (float): Liczba lat (może być ułamkiem).

    Returns:
        float: Wartość CAGR.
    """
    if initial_value <= 0 or years <= 0:
        return 0.0
    return (final_value / initial_value) ** (1 / years) - 1


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Oblicza maksymalne obsunięcie (max drawdown) na podstawie krzywej kapitału.

    Parameters:
        equity_curve (pd.Series): Krzywa kapitału (wartości w czasie).

    Returns:
        float: Najniższa wartość (ujemna) względem osiągniętego szczytu (np. -0.3 = -30%).
    """
    if equity_curve.empty:
        return 0.0
    cumulative_max = equity_curve.cummax()
    drawdowns = (equity_curve - cumulative_max) / cumulative_max
    return float(drawdowns.min())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Oblicza Sharpe Ratio przy założeniu stopy wolnej od ryzyka (domyślnie 0).

    Parameters:
        returns (pd.Series): Stopy zwrotu w poszczególnych okresach.
        risk_free_rate (float): Stopa wolna od ryzyka.

    Returns:
        float: Wartość Sharpe Ratio.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))


def walk_forward_split(df: pd.DataFrame, n_splits: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Dzieli dane na kolejne okresy walk-forward.
    Zwraca listę krotek (train_data, test_data).

    Parameters:
        df (pd.DataFrame): Dane z indeksem czasowym.
        n_splits (int): Liczba podziałów walk-forward.

    Returns:
        List[tuple]: Lista krotek (train_data, test_data).
    """
    if df.index.size < n_splits * 2:
        logging.warning("Rozmiar danych jest mniejszy niż wymagany do walk-forward. Zmniejsz n_splits.")
    if df.index.size < 2:
        logging.warning("Dane zawierają zbyt mało rekordów do jakiegokolwiek backtestu.")

    total_period = (df.index[-1] - df.index[0]).days
    if total_period <= 0:
        logging.warning("Długość okresu w danych jest niepoprawna (0 lub ujemna).")

    split_period = max(1, total_period // n_splits)
    splits = []
    for i in range(n_splits):
        train_end = df.index[0] + pd.Timedelta(days=(i + 1) * split_period)
        test_end = train_end + pd.Timedelta(days=split_period)

        train_data = df[df.index <= train_end]
        test_data = df[(df.index > train_end) & (df.index <= test_end)]
        if not train_data.empty and not test_data.empty:
            splits.append((train_data, test_data))
        else:
            logging.info("Pominięto fold %d: train_data lub test_data jest puste.", i + 1)
    return splits


def run_strategy(
    strategy_func: Callable[[pd.DataFrame], Dict[Any, int]],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    commission: float = 0.001,
    spread: float = 0.0005,
    slippage: float = 0.0005,
) -> Dict[str, Any]:
    """
    Uruchamia strategię na danych testowych.

    Parameters:
        strategy_func: Funkcja strategii, przyjmująca DataFrame i zwracająca sygnały {timestamp: signal}.
        train_data (pd.DataFrame): Dane treningowe (opcjonalnie do optymalizacji).
        test_data (pd.DataFrame): Dane testowe, na których strategia jest symulowana.
        commission (float): Prowizja od transakcji.
        spread (float): Spread transakcyjny.
        slippage (float): Poślizg cenowy.

    Returns:
        dict: Wyniki strategii (final_capital, CAGR, max_drawdown, Sharpe_ratio, trade_log).
    """
    initial_capital = 10000.0
    capital = initial_capital
    positions = 0  # 1 - long, -1 - short, 0 - brak pozycji
    entry_price = None
    equity_curve = []
    trade_log = []

    # Generujemy sygnały (np. 1 = kupno, -1 = sprzedaż, 0 = brak sygnału)
    signals = strategy_func(test_data)

    for time_point, row in test_data.iterrows():
        price = row.get("close", None)
        if price is None:
            equity_curve.append(capital)
            continue

        signal = signals.get(time_point, 0)
        if signal == 1 and positions == 0:
            # Otwieramy pozycję long
            positions = 1
            entry_price = price * (1 + spread + slippage)
            trade_log.append({"timestamp": time_point, "action": "BUY", "price": entry_price})
        elif signal == -1 and positions == 1:
            # Zamykamy pozycję long
            exit_price = price * (1 - spread - slippage)
            profit = (exit_price - entry_price) * capital / entry_price
            profit -= exit_price * commission
            capital += profit
            positions = 0
            entry_price = None
            trade_log.append(
                {
                    "timestamp": time_point,
                    "action": "SELL",
                    "price": exit_price,
                    "profit": profit,
                }
            )
        equity_curve.append(capital)

    if len(test_data) > 0:
        test_duration_days = (test_data.index[-1] - test_data.index[0]).days
        test_duration_years = test_duration_days / 365.25 if test_duration_days > 0 else 0.0
    else:
        test_duration_years = 0.0

    cagr = calculate_cagr(initial_capital, capital, test_duration_years) if test_duration_years > 0 else 0.0
    equity_series = pd.Series(equity_curve, index=test_data.index[: len(equity_curve)])
    max_dd = calculate_max_drawdown(equity_series)
    returns = equity_series.pct_change().dropna()
    sharpe = calculate_sharpe_ratio(returns)

    return {
        "final_capital": capital,
        "CAGR": cagr,
        "max_drawdown": max_dd,
        "Sharpe_ratio": sharpe,
        "trade_log": trade_log,
    }


def backtest_strategy(
    strategy_func: Callable[[pd.DataFrame], Dict[Any, int]],
    df: pd.DataFrame,
    n_splits: int = 5,
    commission: float = 0.001,
    spread: float = 0.0005,
    slippage: float = 0.0005,
) -> Dict[str, Any]:
    """
    Przeprowadza backtesting strategii za pomocą walk-forward analysis oraz równoległego testowania.

    Parameters:
        strategy_func: Funkcja strategii, przyjmująca DataFrame i zwracająca sygnały {timestamp: signal}.
        df (pd.DataFrame): Dane historyczne (indeks: timestamp, kolumna 'close' wymagana).
        n_splits (int): Liczba podziałów walk-forward.
        commission, spread, slippage: Parametry transakcyjne.

    Returns:
        dict: Raport zawierający:
            - 'average_CAGR'
            - 'average_max_drawdown'
            - 'average_Sharpe_ratio'
            - 'individual_results': lista wyników z poszczególnych foldów.
    """
    splits = walk_forward_split(df, n_splits)
    results_list = []

    def worker(train_test_tuple):
        train_data, test_data = train_test_tuple
        return run_strategy(
            strategy_func,
            train_data,
            test_data,
            commission=commission,
            spread=spread,
            slippage=slippage,
        )

    # Równoległe przetwarzanie przy użyciu ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n_splits, 4)) as executor:
        futures = [executor.submit(worker, split) for split in splits]
        for future in futures:
            results_list.append(future.result())

    if not results_list:
        logging.warning("Brak wyników z backtestingu (być może brak odpowiednich foldów).")
        return {
            "average_CAGR": 0.0,
            "average_max_drawdown": 0.0,
            "average_Sharpe_ratio": 0.0,
            "individual_results": [],
        }

    cagr_list = [res["CAGR"] for res in results_list]
    mdd_list = [res["max_drawdown"] for res in results_list]
    sharpe_list = [res["Sharpe_ratio"] for res in results_list]

    report = {
        "average_CAGR": float(np.mean(cagr_list)),
        "average_max_drawdown": float(np.mean(mdd_list)),
        "average_Sharpe_ratio": float(np.mean(sharpe_list)),
        "individual_results": results_list,
    }

    logging.info("Backtesting zakończony. Raport: %s", report)
    return report


def example_strategy(df: pd.DataFrame) -> Dict[Any, int]:
    """
    Przykładowa strategia tradingowa:
    - Generuje sygnał kupna, jeśli cena zamknięcia przekracza średnią kroczącą z 10 okresów.
    - Generuje sygnał sprzedaży, jeśli cena zamknięcia spada poniżej tej średniej.

    Parameters:
        df (pd.DataFrame): Dane historyczne z kolumną 'close'.

    Returns:
        dict: Słownik sygnałów z kluczami (timestamp), gdzie
              1 oznacza sygnał kupna, -1 sygnał sprzedaży, 0 brak sygnału.
    """
    signals: Dict[Any, int] = {}
    if "close" not in df.columns:
        logging.error("Brak kolumny 'close' w DataFrame. Strategia nie może generować sygnałów.")
        return signals

    sma = df["close"].rolling(window=10, min_periods=1).mean()

    for idx, row in df.iterrows():
        close_price = row.get("close", None)
        if close_price is None:
            signals[idx] = 0
            continue

        sma_value = sma.loc[idx]
        if close_price > sma_value:
            signals[idx] = 1
        elif close_price < sma_value:
            signals[idx] = -1
        else:
            signals[idx] = 0

    return signals


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        data_file = "./data/historical_data.csv"
        if not os.path.exists(data_file):
            logging.error("Plik z danymi historycznymi nie istnieje: %s", data_file)
            raise FileNotFoundError(f"Brak pliku: {data_file}")

        df = pd.read_csv(data_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        report = backtest_strategy(
            example_strategy,
            df,
            n_splits=5,
            commission=0.001,
            spread=0.0005,
            slippage=0.0005,
        )

        # Zapis raportu do pliku JSON
        import json

        os.makedirs("./reports", exist_ok=True)
        report_path = "./reports/backtesting_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4, default=str)

        logging.info("Raport backtestingu zapisany w: %s", report_path)

    except Exception as e:
        logging.error("Błąd w module backtesting.py: %s", e)
        raise
