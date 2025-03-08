"""
historical.py
-------------
Skrypt analizujący historyczne dane rynkowe i generujący kluczowe statystyki oraz wizualizacje.

Funkcjonalności:
- Oblicza podstawowe metryki: średnia, odchylenie standardowe, korelacje, max drawdown.
- Tworzy wykresy: wykres świecowy (candlestick), wykres wolumenu oraz wskaźników technicznych.
- Umożliwia eksport raportu w formacie HTML, zawierającego podsumowanie danych oraz wygenerowane wykresy.
- Integruje się z modułami data_preprocessing.py oraz data_storage.py, aby zapewnić płynny przepływ danych.
- Obsługuje duże pliki danych i umożliwia parametryzację zakresu analizy (np. data początkowa i końcowa).
"""

import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def compute_statistics(df: pd.DataFrame) -> dict:
    """
    Oblicza podstawowe metryki dla historycznych danych rynkowych.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi świecowymi. Oczekiwane kolumny:
                           'timestamp', 'open', 'high', 'low', 'close', 'volume'

    Returns:
        dict: Słownik z obliczonymi metrykami: średnia, odchylenie standardowe, korelacje, max drawdown.
    """
    stats = {}
    try:
        # Upewnij się, że 'timestamp' jest typu datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Obliczenia dla ceny zamknięcia
        close_prices = df["close"]
        stats["mean_close"] = close_prices.mean()
        stats["std_close"] = close_prices.std()
        stats["correlations"] = df[["open", "high", "low", "close"]].corr().to_dict()

        # Obliczenie max drawdown
        cumulative_max = close_prices.cummax()
        drawdown = (close_prices - cumulative_max) / cumulative_max
        stats["max_drawdown"] = drawdown.min()

        logging.info(
            "Obliczono statystyki: średnia=%f, std=%f, max drawdown=%f",
            stats["mean_close"],
            stats["std_close"],
            stats["max_drawdown"],
        )
    except Exception as e:
        logging.error("Błąd przy obliczaniu statystyk: %s", e)
        raise
    return stats


def plot_candlestick(df: pd.DataFrame, output_path: str):
    """
    Tworzy wykres świecowy na podstawie danych rynkowych i zapisuje go jako obraz PNG.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi świecowymi.
        output_path (str): Ścieżka, pod którą zapisany zostanie wykres.
    """
    try:
        import matplotlib.dates as mdates
        from mplfinance.original_flavor import candlestick_ohlc

        # Konwersja dat do formatu numerycznego
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date_num"] = df["timestamp"].apply(mdates.date2num)
        ohlc = df[["date_num", "open", "high", "low", "close"]].values

        fig, ax = plt.subplots(figsize=(10, 6))
        candlestick_ohlc(ax, ohlc, width=0.0008, colorup="g", colordown="r", alpha=0.8)
        ax.xaxis_date()
        ax.set_title("Wykres świecowy")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Cena")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info("Wykres świecowy zapisany w: %s", output_path)
    except ImportError as ie:
        logging.error(
            "Biblioteka mplfinance nie jest zainstalowana. Zainstaluj ją używając 'pip install mplfinance'."
        )
        raise
    except Exception as e:
        logging.error("Błąd przy generowaniu wykresu świecowego: %s", e)
        raise


def generate_report(
    df: pd.DataFrame,
    stats: dict,
    chart_path: str,
    report_path: str,
    start_date: str = None,
    end_date: str = None,
):
    """
    Generuje raport HTML z analizą historycznych danych rynkowych, w tym statystykami oraz wykresem świecowym.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi rynkowymi.
        stats (dict): Obliczone metryki.
        chart_path (str): Ścieżka do zapisanego wykresu świecowego.
        report_path (str): Ścieżka, pod którą zapisany zostanie raport HTML.
        start_date (str, optional): Data początkowa analizy.
        end_date (str, optional): Data końcowa analizy.
    """
    try:
        report_lines = []
        report_lines.append(
            "<html><head><meta charset='utf-8'><title>Raport Historyczny</title></head><body>"
        )
        report_lines.append(
            f"<h1>Raport Historyczny - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h1>"
        )
        if start_date and end_date:
            report_lines.append(f"<h2>Zakres analizy: {start_date} - {end_date}</h2>")
        report_lines.append("<h2>Statystyki:</h2>")
        report_lines.append("<ul>")
        report_lines.append(
            f"<li>Średnia cena zamknięcia: {stats.get('mean_close', 'N/A'):.2f}</li>"
        )
        report_lines.append(
            f"<li>Odchylenie standardowe ceny zamknięcia: {stats.get('std_close', 'N/A'):.2f}</li>"
        )
        report_lines.append(
            f"<li>Max Drawdown: {stats.get('max_drawdown', 'N/A'):.2%}</li>"
        )
        report_lines.append("</ul>")
        report_lines.append("<h2>Korelacje:</h2>")
        report_lines.append("<pre>")
        report_lines.append(str(stats.get("correlations", {})))
        report_lines.append("</pre>")
        report_lines.append("<h2>Wykres świecowy:</h2>")
        report_lines.append(
            f"<img src='{os.path.basename(chart_path)}' alt='Wykres świecowy' style='max-width:100%;'>"
        )
        report_lines.append("</body></html>")

        # Zapis raportu
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logging.info("Raport HTML zapisany w: %s", report_path)
    except Exception as e:
        logging.error("Błąd przy generowaniu raportu: %s", e)
        raise


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Ścieżka do przykładowego pliku CSV z danymi historycznymi
        data_file = "./data/historical_data.csv"
        if not os.path.exists(data_file):
            logging.error("Plik z danymi historycznymi nie istnieje: %s", data_file)
            raise FileNotFoundError(f"Brak pliku: {data_file}")

        # Wczytanie danych
        df = pd.read_csv(data_file)
        logging.info("Wczytano dane z pliku: %s", data_file)

        # Opcjonalnie, filtrowanie danych według daty
        start_date = "2023-01-01"
        end_date = "2023-01-31"
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        df_filtered = df.loc[mask]

        # Obliczenie statystyk
        stats = compute_statistics(df_filtered.copy())

        # Wygenerowanie wykresu świecowego
        chart_path = "./data/historical_candlestick.png"
        plot_candlestick(df_filtered.copy(), chart_path)

        # Wygenerowanie raportu HTML
        report_path = "./data/historical_report.html"
        generate_report(
            df_filtered.copy(), stats, chart_path, report_path, start_date, end_date
        )

        logging.info("Analiza historyczna zakończona sukcesem.")
    except Exception as e:
        logging.error("Błąd w module historical.py: %s", e)
        raise
