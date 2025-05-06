-- Trading system database schema

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    exit_price REAL,
    status TEXT NOT NULL,
    strategy TEXT,
    pnl REAL,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    closed_at INTEGER
);

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);

-- Model predictions table
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    prediction REAL NOT NULL,
    confidence REAL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- System events table
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Portfolio history table
CREATE TABLE IF NOT EXISTS portfolio_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    total_equity REAL NOT NULL,
    available_balance REAL NOT NULL,
    used_margin REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USDT',
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON model_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_history(timestamp);