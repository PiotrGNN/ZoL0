"""
generate_historical_data.py
---------------------------
Skrypt generujący dane historyczne dla celów testowych.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_crypto_price_data(symbol="BTCUSDT", timeframe="1h", days=365, 
                               start_price=30000.0, volatility=0.02, trend=0.0001):
    """
    Generuje realistyczne dane cenowe dla kryptowalut
    
    Parameters:
        symbol (str): Symbol kryptowaluty
        timeframe (str): Przedział czasowy (np. "1h", "15m")
        days (int): Liczba dni danych historycznych
        start_price (float): Cena początkowa
        volatility (float): Zmienność ceny
        trend (float): Współczynnik trendu (dodatni - trend wzrostowy, ujemny - trend spadkowy)
        
    Returns:
        DataFrame: Wygenerowane dane cenowe
    """
    # Określenie liczby punktów danych w zależności od timeframe
    if timeframe == "1h":
        periods = days * 24
        freq = "H"
    elif timeframe == "15m":
        periods = days * 24 * 4
        freq = "15min"
    elif timeframe == "1d":
        periods = days
        freq = "D"
    else:
        periods = days * 24  # domyślnie godzinowe
        freq = "H"
    
    # Generowanie czasu
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamp = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generowanie cen
    np.random.seed(42)  # Dla powtarzalności
    
    # Trend logarytmiczny dla większego realizmu
    time_factor = np.linspace(0, 1, periods)
    trend_component = np.exp(trend * np.cumsum(np.ones(periods)))
    
    # Komponent losowy
    random_component = np.exp(np.random.normal(0, volatility, periods))
    
    # Sezonowość - dzienny wzór
    seasonality = 1.0
    if timeframe in ["1h", "15m"]:
        hour_of_day = np.array([t.hour for t in timestamp])
        # Więcej aktywności w ciągu dnia, mniej w nocy
        seasonality = 1.0 + 0.005 * np.sin(2 * np.pi * (hour_of_day - 10) / 24)
    
    # Obliczenie cen
    close_prices = start_price * trend_component * random_component * seasonality
    
    # Generowanie OHLC
    open_prices = close_prices * np.exp(np.random.normal(0, 0.01, periods))
    high_prices = np.maximum(close_prices, open_prices) * np.exp(np.random.normal(0.005, 0.01, periods))
    low_prices = np.minimum(close_prices, open_prices) * np.exp(np.random.normal(-0.005, 0.01, periods))
    
    # Wolumen - wyższy przy większych zmianach ceny
    price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    volume = np.random.gamma(shape=2, scale=1, size=periods) * (1 + 5 * price_changes / np.mean(price_changes))
    volume = volume * 1000  # Skalowanie wolumenu
    
    # Tworzenie DataFrame
    df = pd.DataFrame({
        'timestamp': timestamp,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume.astype(int),
        'symbol': symbol,
        'timeframe': timeframe
    })
    
    return df

def save_to_csv(df, filepath):
    """Zapisuje DataFrame do pliku CSV"""
    # Utworzenie katalogów, jeśli nie istnieją
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"Zapisano {len(df)} wierszy danych do {filepath}")

def save_to_sqlite(df, db_path, table_name='price_data'):
    """Zapisuje DataFrame do bazy danych SQLite"""
    import sqlite3
    
    # Utworzenie katalogu, jeśli nie istnieje
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Połączenie z bazą danych
    conn = sqlite3.connect(db_path)
    
    # Zapisanie danych
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    # Zamknięcie połączenia
    conn.close()
    print(f"Zapisano {len(df)} wierszy danych do bazy {db_path}, tabela {table_name}")

def main():
    """Funkcja główna"""
    # Utworzenie katalogów, jeśli nie istnieją
    os.makedirs('data/backtesting', exist_ok=True)
    
    # Generowanie danych dla różnych kryptowalut i timeframe'ów
    print("Generowanie danych historycznych...")
    
    # BTCUSDT 1h
    btc_1h = generate_crypto_price_data(symbol="BTCUSDT", timeframe="1h", days=365, 
                                       start_price=29000.0, volatility=0.02, trend=0.0003)
    save_to_csv(btc_1h, 'data/backtesting/BTCUSDT_1h.csv')
    
    # ETHUSDT 1h
    eth_1h = generate_crypto_price_data(symbol="ETHUSDT", timeframe="1h", days=365, 
                                       start_price=1800.0, volatility=0.025, trend=0.0002)
    save_to_csv(eth_1h, 'data/backtesting/ETHUSDT_1h.csv')
    
    # BTCUSDT 15m
    btc_15m = generate_crypto_price_data(symbol="BTCUSDT", timeframe="15m", days=30, 
                                        start_price=29000.0, volatility=0.015, trend=0.0001)
    save_to_csv(btc_15m, 'data/backtesting/BTCUSDT_15m.csv')
    
    # Połączone dane do bazy SQLite
    all_data = pd.concat([btc_1h, eth_1h, btc_15m])
    save_to_sqlite(all_data, 'data/historical_data.db', 'price_data')
    
    print("Generowanie danych zakończone.")

if __name__ == "__main__":
    main()