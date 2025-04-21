#!/usr/bin/env python3
"""
fix_dashboard_issues.py - Naprawia problemy w dashboardzie ZoL0
"""
import os
import sqlite3
import logging
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("dashboard_fixes.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ścieżka do bazy danych
DB_PATH = 'users.db'

def fix_database_schema():
    """Naprawia schemat bazy danych, dodając brakujące kolumny."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Sprawdź, czy istnieje tabela portfolio_allocation
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_allocation'")
        if not cursor.fetchone():
            logger.info("Tworzenie tabeli portfolio_allocation")
            cursor.execute('''
            CREATE TABLE portfolio_allocation (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                asset_symbol TEXT,
                allocation_percentage REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
        else:
            # Sprawdź, czy tabela ma kolumnę asset_symbol
            cursor.execute("PRAGMA table_info(portfolio_allocation)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'asset' in columns and 'asset_symbol' not in columns:
                logger.info("Zmiana nazwy kolumny 'asset' na 'asset_symbol' w tabeli portfolio_allocation")
                # SQLite nie wspiera bezpośredniej zmiany nazwy kolumny, więc tworzymy nową tabelę
                cursor.execute('''
                CREATE TABLE portfolio_allocation_temp (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    asset_symbol TEXT,
                    allocation_percentage REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                cursor.execute("INSERT INTO portfolio_allocation_temp (id, user_id, asset_symbol, allocation_percentage, timestamp) SELECT id, user_id, asset, allocation_percentage, timestamp FROM portfolio_allocation")
                cursor.execute("DROP TABLE portfolio_allocation")
                cursor.execute("ALTER TABLE portfolio_allocation_temp RENAME TO portfolio_allocation")
            elif 'asset_symbol' not in columns:
                logger.info("Dodawanie kolumny 'asset_symbol' do tabeli portfolio_allocation")
                cursor.execute("ALTER TABLE portfolio_allocation ADD COLUMN asset_symbol TEXT")
        
        # Sprawdź, czy istnieje tabela trades
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if not cursor.fetchone():
            logger.info("Tworzenie tabeli trades")
            cursor.execute('''
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                entry_price REAL,
                exit_price REAL,
                realized_pnl REAL,
                executed_at DATETIME,
                closed_at DATETIME,
                duration_hours REAL,
                commission REAL
            )
            ''')
        else:
            # Sprawdź, czy tabela ma kolumnę entry_price
            cursor.execute("PRAGMA table_info(trades)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'entry_price' not in columns:
                logger.info("Dodawanie kolumny 'entry_price' do tabeli trades")
                cursor.execute("ALTER TABLE trades ADD COLUMN entry_price REAL")
        
        # Dodaj pozostałe brakujące tabele i kolumny, jeśli są potrzebne
        # Tabela asset_allocation_history
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='asset_allocation_history'")
        if not cursor.fetchone():
            logger.info("Tworzenie tabeli asset_allocation_history")
            cursor.execute('''
            CREATE TABLE asset_allocation_history (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                asset_symbol TEXT,
                allocation_percentage REAL,
                market_value REAL,
                asset_class TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')

        # Tabela diversification_metrics
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='diversification_metrics'")
        if not cursor.fetchone():
            logger.info("Tworzenie tabeli diversification_metrics")
            cursor.execute('''
            CREATE TABLE diversification_metrics (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                herfindahl_index REAL,
                effective_n REAL,
                asset_class_diversity REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')

        # Tabela asset_correlations
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='asset_correlations'")
        if not cursor.fetchone():
            logger.info("Tworzenie tabeli asset_correlations")
            cursor.execute('''
            CREATE TABLE asset_correlations (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                asset1 TEXT,
                asset2 TEXT,
                correlation REAL,
                timestamp DATETIME
            )
            ''')

        # Tabela historical_prices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_prices'")
        if not cursor.fetchone():
            logger.info("Tworzenie tabeli historical_prices")
            cursor.execute('''
            CREATE TABLE historical_prices (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            ''')

        # Zapewnij, że są jakieś przykładowe dane
        # Dodaj przykładowe dane do tabeli asset_allocation_history
        cursor.execute("SELECT COUNT(*) FROM asset_allocation_history")
        if cursor.fetchone()[0] == 0:
            logger.info("Dodawanie przykładowych danych do tabeli asset_allocation_history")
            example_data = [
                (1, 'BTC', 40.0, 4000.0, 'Cryptocurrency', '2025-04-01 12:00:00'),
                (1, 'ETH', 30.0, 3000.0, 'Cryptocurrency', '2025-04-01 12:00:00'),
                (1, 'SOL', 20.0, 2000.0, 'Cryptocurrency', '2025-04-01 12:00:00'),
                (1, 'USDT', 10.0, 1000.0, 'Stablecoin', '2025-04-01 12:00:00')
            ]
            cursor.executemany('''
            INSERT INTO asset_allocation_history (user_id, asset_symbol, allocation_percentage, market_value, asset_class, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', example_data)
            
        # Dodaj przykładowe dane do historical_prices
        cursor.execute("SELECT COUNT(*) FROM historical_prices")
        if cursor.fetchone()[0] == 0:
            import random
            from datetime import datetime, timedelta
            logger.info("Dodawanie przykładowych danych do tabeli historical_prices")
            
            # Generuj 90 dni danych dla BTC, ETH, SOL, USDT
            symbols = ['BTC', 'ETH', 'SOL', 'USDT']
            base_prices = {'BTC': 50000, 'ETH': 3000, 'SOL': 100, 'USDT': 1}
            
            example_data = []
            for symbol in symbols:
                base = base_prices[symbol]
                price = base
                for i in range(90):
                    date = (datetime.now() - timedelta(days=90-i)).strftime('%Y-%m-%d %H:%M:%S')
                    # Generuj realistyczne ceny
                    change = random.uniform(-0.03, 0.03)  # -3% do +3%
                    price = price * (1 + change)
                    
                    high = price * (1 + random.uniform(0, 0.01))
                    low = price * (1 - random.uniform(0, 0.01))
                    open_price = price * (1 + random.uniform(-0.005, 0.005))
                    volume = random.uniform(1000, 10000)
                    
                    example_data.append((symbol, date, open_price, high, low, price, volume))
            
            cursor.executemany('''
            INSERT INTO historical_prices (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', example_data)
        
        conn.commit()
        conn.close()
        logger.info("Naprawiono schemat bazy danych")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania schematu bazy danych: {e}")
        return False

def fix_vector_memory():
    """Naprawia problem z KeyError: 'use_faiss' w vector_memory.py"""
    try:
        # Znajdź ścieżkę do pliku vector_memory.py
        vector_memory_path = None
        for root, dirs, files in os.walk('.'):
            if 'vector_memory.py' in files:
                vector_memory_path = os.path.join(root, 'vector_memory.py')
                break
        
        if not vector_memory_path:
            logger.error("Nie znaleziono pliku vector_memory.py")
            return False
        
        # Odczytaj plik
        with open(vector_memory_path, 'r') as file:
            content = file.read()
        
        # Sprawdź czy naprawka już istnieje
        if "# Upewnij się, że klucz 'use_faiss' istnieje w konfiguracji" in content:
            logger.info("Plik vector_memory.py już został naprawiony")
            return True
            
        # Znajdź metodę __init__ i dodaj kod
        import re
        init_pattern = r"def __init__\s*\(\s*self\s*,\s*config\s*=\s*None\s*\)\s*:"
        init_match = re.search(init_pattern, content)
        
        if not init_match:
            logger.error("Nie znaleziono metody __init__ w vector_memory.py")
            return False
        
        # Znajdź początek bloku kodu po definicji metody __init__
        pos = init_match.end()
        lines = content.splitlines()
        line_num = content[:pos].count('\n')
        
        # Znajdź linie do poprawy
        updated_lines = lines.copy()
        insertion_line = line_num + 1  # Dodaj po pierwszej linii w metodzie __init__
        
        updated_lines.insert(insertion_line, "        # Upewnij się, że klucz 'use_faiss' istnieje w konfiguracji")
        updated_lines.insert(insertion_line + 1, "        if config is not None and 'use_faiss' not in config:")
        updated_lines.insert(insertion_line + 2, "            config['use_faiss'] = False")
        updated_lines.insert(insertion_line + 3, "            logger.info(\"Dodano brakujący klucz 'use_faiss' do konfiguracji\")")
        
        # Zapisz poprawiony plik
        with open(vector_memory_path, 'w') as file:
            file.write('\n'.join(updated_lines))
            
        logger.info(f"Naprawiono plik {vector_memory_path}")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania pliku vector_memory.py: {e}")
        return False

def fix_api_portfolio_correlation():
    """Naprawia problem z wywołaniem metody calculate_asset_correlation w dashboard_api.py"""
    try:
        api_file = 'dashboard_api.py'
        if not os.path.exists(api_file):
            logger.error(f"Nie znaleziono pliku {api_file}")
            return False
        
        with open(api_file, 'r') as file:
            content = file.read()
        
        # Sprawdź czy funkcja get_portfolio_correlation używa poprawnej nazwy kolumny
        if "c.execute('SELECT asset, percentage" in content:
            # Zamień 'asset' na 'asset_symbol'
            content = content.replace("c.execute('SELECT asset, percentage", "c.execute('SELECT asset_symbol, percentage")
            content = content.replace("allocation[asset] = float(percentage)", "allocation[asset_symbol] = float(percentage)")
            
            # Zapisz poprawiony plik
            with open(api_file, 'w') as file:
                file.write(content)
            
            logger.info(f"Naprawiono nazwy kolumn w funkcji get_portfolio_correlation w pliku {api_file}")
        
        # Sprawdź, czy wywołanie metody calculate_asset_correlation jest poprawne
        if "correlation_data = analytics.calculate_asset_correlation()" in content:
            logger.info("Wywołanie metody calculate_asset_correlation jest poprawne")
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania pliku {api_file}: {e}")
        return False

def main():
    """Główna funkcja naprawiająca wszystkie problemy."""
    logger.info("Rozpoczynam naprawę problemów w dashboardzie ZoL0...")
    
    # Napraw problem z vector_memory.py
    if fix_vector_memory():
        logger.info("Naprawiono problem z KeyError: 'use_faiss' w vector_memory.py")
    else:
        logger.warning("Nie udało się naprawić problemu z vector_memory.py")
    
    # Napraw schemat bazy danych
    if fix_database_schema():
        logger.info("Naprawiono schemat bazy danych")
    else:
        logger.warning("Nie udało się naprawić schematu bazy danych")
    
    # Napraw wywołanie metody calculate_asset_correlation
    if fix_api_portfolio_correlation():
        logger.info("Naprawiono wywołanie metody calculate_asset_correlation")
    else:
        logger.warning("Nie udało się naprawić wywołania metody calculate_asset_correlation")
    
    logger.info("Zakończono naprawę problemów w dashboardzie ZoL0")
    print("Naprawy zostały zakończone. Sprawdź plik dashboard_fixes.log, aby uzyskać więcej informacji.")

if __name__ == "__main__":
    main()