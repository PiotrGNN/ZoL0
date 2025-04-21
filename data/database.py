"""
pkill -f "python3.*flask.*run" && FLASK_APP=dashboard_api.py FLASK_ENV=development FLASK_DEBUG=1 python3 -m flask runpkill -f "python3.*flask.*run" && FLASK_APP=dashboard_api.py FLASK_ENV=development FLASK_DEBUG=1 python3 -m flask runModuł obsługujący bazę danych dla użytkowników i uwierzytelniania.
"""

import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
import logging
from typing import Dict, Any, List
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = 'users.db'

def init_db():
    """Inicjalizuje bazę danych i tworzy wymagane tabele"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Tworzenie tabeli użytkowników
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
        ''')

        # Tworzenie tabeli transakcji
        c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            order_type TEXT NOT NULL,
            realized_pnl REAL,
            commission REAL,
            strategy_id TEXT,
            order_id TEXT UNIQUE,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli historii portfela
        c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_equity REAL NOT NULL,
            available_balance REAL NOT NULL,
            used_margin REAL NOT NULL,
            unrealized_pnl REAL,
            realized_pnl REAL,
            snapshot_data TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli alertów
        c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT NOT NULL,
            symbol TEXT,
            condition TEXT NOT NULL,
            value REAL,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            triggered_at TIMESTAMP,
            notification_sent BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli wskaźników ryzyka
        c.execute('''
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            max_drawdown_duration INTEGER,
            win_rate REAL,
            profit_factor REAL,
            avg_win_loss_ratio REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli limitów ryzyka
        c.execute('''
        CREATE TABLE IF NOT EXISTS risk_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            portfolio_stop_loss REAL,
            portfolio_take_profit REAL,
            max_position_size REAL,
            max_leverage REAL,
            max_daily_trades INTEGER,
            max_daily_drawdown REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli własnych wskaźników technicznych
        c.execute('''
        CREATE TABLE IF NOT EXISTS custom_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            formula TEXT NOT NULL,
            parameters TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli wyników backtestów
        c.execute('''
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            strategy_id TEXT NOT NULL,
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            initial_balance REAL,
            final_balance REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            parameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli alokacji portfela
        c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_allocation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT NOT NULL,
            target_allocation REAL,
            current_allocation REAL,
            last_rebalance TIMESTAMP,
            next_rebalance TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tworzenie tabeli dla parametrów modeli AI
        c.execute('''
        CREATE TABLE IF NOT EXISTS ai_model_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            parameter_name TEXT NOT NULL,
            parameter_value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Tworzenie tabeli historii cen
        c.execute('''
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
        ''')
        
        # Tworzenie tabeli historii rebalancingu
        c.execute('''
        CREATE TABLE IF NOT EXISTS rebalance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            old_allocation REAL,
            new_allocation REAL,
            transaction_type TEXT,
            amount REAL,
            price REAL,
            status TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # Tworzenie tabeli rekomendacji alokacji
        c.execute('''
        CREATE TABLE IF NOT EXISTS allocation_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_profile TEXT,
            recommendations TEXT,  -- JSON z rekomendowanymi alokacjami
            status TEXT,
            applied BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # Tworzenie tabeli dla szczegółów portfela
        c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_assets REAL,
            total_liabilities REAL,
            margin_used REAL,
            margin_level REAL,
            liquidation_price REAL,
            buying_power REAL,
            maintenance_margin REAL,
            initial_margin REAL,
            positions_json TEXT,  -- JSON z szczegółami pozycji
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela dla rozkładu alokacji aktywów w czasie
        c.execute('''
        CREATE TABLE IF NOT EXISTS asset_allocation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            asset_symbol TEXT NOT NULL,
            allocation_percentage REAL NOT NULL,
            market_value REAL NOT NULL,
            asset_class TEXT NOT NULL,
            risk_contribution REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela metryk dywersyfikacji
        c.execute('''
        CREATE TABLE IF NOT EXISTS diversification_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            herfindahl_index REAL,
            effective_n REAL,
            correlation_score REAL,
            risk_contribution_score REAL,
            asset_class_diversity REAL,
            geographic_diversity REAL,
            sector_diversity REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela ekspozycji na ryzyko
        c.execute('''
        CREATE TABLE IF NOT EXISTS risk_exposure (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            market_risk REAL,
            credit_risk REAL,
            liquidity_risk REAL,
            volatility_risk REAL,
            concentration_risk REAL,
            systematic_risk REAL,
            unsystematic_risk REAL,
            var_95 REAL,
            cvar_95 REAL,
            stress_test_results TEXT,
            scenario_analysis TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela rotacji kapitału
        c.execute('''
        CREATE TABLE IF NOT EXISTS capital_turnover (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            period_start DATE,
            period_end DATE,
            total_buys REAL,
            total_sells REAL,
            turnover_ratio REAL,
            portfolio_value REAL,
            trading_costs REAL,
            tax_impact REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela alertów wzorców cenowych
        c.execute('''
        CREATE TABLE IF NOT EXISTS pattern_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            confidence REAL,
            price_level REAL,
            volume_confirmation BOOLEAN,
            timeframe TEXT,
            additional_indicators TEXT,
            status TEXT DEFAULT 'active',
            triggered_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela alertów anomalii AI
        c.execute('''
        CREATE TABLE IF NOT EXISTS ai_anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            severity REAL,
            description TEXT,
            affected_assets TEXT,
            confidence_score REAL,
            false_positive_probability REAL,
            action_taken TEXT,
            resolution_status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela konfiguracji kanałów powiadomień
        c.execute('''
        CREATE TABLE IF NOT EXISTS notification_channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            channel_type TEXT NOT NULL,
            channel_name TEXT NOT NULL,
            configuration TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            priority INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            last_error TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela dla historii powiadomień
        c.execute('''
        CREATE TABLE IF NOT EXISTS notification_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notification_type TEXT NOT NULL,
            channel TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            delivery_attempts INTEGER DEFAULT 0,
            delivered_at TIMESTAMP,
            metadata TEXT,  -- JSON z dodatkowymi danymi
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela konfiguracji kanałów komunikacji
        c.execute('''
        CREATE TABLE IF NOT EXISTS communication_channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            channel_type TEXT NOT NULL,  -- email, telegram, sms, etc.
            channel_name TEXT NOT NULL,
            config_json TEXT NOT NULL,  -- JSON z konfiguracją (adresy, tokeny, etc.)
            is_active BOOLEAN DEFAULT 1,
            priority INTEGER DEFAULT 1,
            notification_types TEXT,  -- JSON z typami powiadomień dla kanału
            rate_limit INTEGER,  -- limit powiadomień na minutę
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela szablonów powiadomień
        c.execute('''
        CREATE TABLE IF NOT EXISTS notification_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_name TEXT NOT NULL,
            template_type TEXT NOT NULL,
            content TEXT NOT NULL,
            variables TEXT,  -- JSON z opisem zmiennych
            channel_specific_configs TEXT,  -- JSON z konfiguracją per kanał
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Tabela dla modeli AI i ich historii
        c.execute('''
        CREATE TABLE IF NOT EXISTS ai_model_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            training_duration INTEGER,  -- w sekundach
            epochs_completed INTEGER,
            accuracy REAL,
            loss REAL,
            validation_accuracy REAL,
            validation_loss REAL,
            hyperparameters TEXT,  -- JSON z parametrami
            metrics TEXT,  -- JSON z dodatkowymi metrykami
            dataset_version TEXT,
            model_version TEXT,
            notes TEXT
        )
        ''')

        # Tabela konfiguracji backtestów
        c.execute('''
        CREATE TABLE IF NOT EXISTS backtest_configurations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            strategy_type TEXT NOT NULL,
            parameters TEXT NOT NULL,  -- JSON z parametrami
            timeframe TEXT NOT NULL,
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            initial_capital REAL,
            leverage REAL,
            commission_rate REAL,
            risk_settings TEXT,  -- JSON z ustawieniami ryzyka
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_run_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela porównań backtestów
        c.execute('''
        CREATE TABLE IF NOT EXISTS backtest_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            backtest_ids TEXT NOT NULL,  -- JSON z listą ID backtestów
            comparison_metrics TEXT NOT NULL,  -- JSON z metrykami porównawczymi
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela własnych wskaźników technicznych
        c.execute('''
        CREATE TABLE IF NOT EXISTS custom_technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            description TEXT,
            formula TEXT NOT NULL,
            parameters TEXT,  -- JSON z parametrami
            validation_rules TEXT,  -- JSON z regułami walidacji
            visualization_settings TEXT,  -- JSON z ustawieniami wizualizacji
            is_public BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            version INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        # Tabela zapisanych analiz technicznych
        c.execute('''
        CREATE TABLE IF NOT EXISTS saved_technical_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            indicators TEXT NOT NULL,  -- JSON z użytymi wskaźnikami
            drawings TEXT,  -- JSON z elementami rysunkowymi
            annotations TEXT,  -- JSON z adnotacjami
            screenshot_path TEXT,
            is_public BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')

        conn.commit()
        logger.info("Baza danych została zainicjalizowana z nowymi tabelami")
        
        # Dodaj domyślnego użytkownika admin/admin jeśli nie istnieje
        c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        if c.fetchone()[0] == 0:
            create_user('admin', 'admin', 'admin@example.com')
            logger.info("Utworzono domyślnego użytkownika admin")
            
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas inicjalizacji bazy danych: {e}")
    finally:
        conn.close()

def create_user(username: str, password: str, email: str = None) -> bool:
    """
    Tworzy nowego użytkownika w bazie danych.
    
    Args:
        username (str): Nazwa użytkownika
        password (str): Hasło użytkownika
        email (str, optional): Adres email użytkownika
        
    Returns:
        bool: True jeśli utworzenie użytkownika się powiodło, False w przeciwnym razie
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        password_hash = generate_password_hash(password)
        c.execute(
            "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
            (username, password_hash, email)
        )
        
        conn.commit()
        logger.info(f"Utworzono użytkownika: {username}")
        return True
        
    except sqlite3.IntegrityError:
        logger.error(f"Użytkownik {username} już istnieje")
        return False
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas tworzenia użytkownika: {e}")
        return False
    finally:
        conn.close()

def verify_user(username: str, password: str) -> bool:
    """
    Weryfikuje dane logowania użytkownika.
    
    Args:
        username (str): Nazwa użytkownika
        password (str): Hasło użytkownika
        
    Returns:
        bool: True jeśli dane logowania są poprawne, False w przeciwnym razie
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        
        if result is None:
            return False
            
        return check_password_hash(result[0], password)
        
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas weryfikacji użytkownika: {e}")
        return False
    finally:
        conn.close()

def update_last_login(username: str) -> bool:
    """
    Aktualizuje timestamp ostatniego logowania użytkownika.
    
    Args:
        username (str): Nazwa użytkownika
        
    Returns:
        bool: True jeśli aktualizacja się powiodła, False w przeciwnym razie
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
            (username,)
        )
        
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas aktualizacji last_login: {e}")
        return False
    finally:
        conn.close()

def add_trade(user_id: int, trade_data: Dict[str, Any]) -> bool:
    """Dodaje nową transakcję do bazy danych"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO trades (
            user_id, symbol, side, quantity, price, status, 
            order_type, realized_pnl, commission, strategy_id, order_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, trade_data['symbol'], trade_data['side'],
            trade_data['quantity'], trade_data['price'], trade_data['status'],
            trade_data['order_type'], trade_data.get('realized_pnl', 0),
            trade_data.get('commission', 0), trade_data.get('strategy_id'),
            trade_data.get('order_id')
        ))
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas dodawania transakcji: {e}")
        return False
    finally:
        conn.close()

def get_user_trades(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Pobiera historię transakcji użytkownika"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
        SELECT * FROM trades 
        WHERE user_id = ? 
        ORDER BY executed_at DESC 
        LIMIT ?
        ''', (user_id, limit))
        
        columns = [description[0] for description in c.description]
        trades = [dict(zip(columns, row)) for row in c.fetchall()]
        
        return trades
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas pobierania historii transakcji: {e}")
        return []
    finally:
        conn.close()

def update_portfolio_history(user_id: int, portfolio_data: Dict[str, Any]) -> bool:
    """Aktualizuje historię portfela"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO portfolio_history (
            user_id, total_equity, available_balance, used_margin,
            unrealized_pnl, realized_pnl, snapshot_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            portfolio_data['total_equity'],
            portfolio_data['available_balance'],
            portfolio_data['used_margin'],
            portfolio_data.get('unrealized_pnl', 0),
            portfolio_data.get('realized_pnl', 0),
            portfolio_data.get('snapshot_data')
        ))
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas aktualizacji historii portfela: {e}")
        return False
    finally:
        conn.close()

def add_alert(user_id: int, alert_data: Dict[str, Any]) -> bool:
    """Dodaje nowy alert cenowy"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO alerts (
            user_id, type, symbol, condition, value, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            alert_data['type'],
            alert_data['symbol'],
            alert_data['condition'],
            alert_data['value'],
            'active'
        ))
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas dodawania alertu: {e}")
        return False
    finally:
        conn.close()

def get_active_alerts(user_id: int) -> List[Dict[str, Any]]:
    """Pobiera aktywne alerty użytkownika"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
        SELECT * FROM alerts 
        WHERE user_id = ? AND status = 'active' 
        ORDER BY created_at DESC
        ''', (user_id,))
        
        columns = [description[0] for description in c.description]
        alerts = [dict(zip(columns, row)) for row in c.fetchall()]
        
        return alerts
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas pobierania alertów: {e}")
        return []
    finally:
        conn.close()

# Inicjalizacja bazy danych przy pierwszym imporcie modułu
if not os.path.exists(DB_PATH):
    init_db()