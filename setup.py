"""
setup.py
--------
Script to initialize and verify the trading system environment.
"""

import os
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/setup.log"),
        logging.StreamHandler()
    ]
)

def create_directories():
    """Create all required directories."""
    directories = [
        "logs",
        "data/cache",
        "reports",
        "saved_models",
        "python_libs",
        "static/img",
        "templates",
        "models",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def create_default_config():
    """Create default configuration files."""
    config = {
        "trading": {
            "mode": "simulated",
            "initial_balance": 1000.0,
            "base_currency": "USDT",
            "trading_pairs": ["BTCUSDT"],
            "risk_management": {
                "max_risk_per_trade": 0.02,
                "max_open_positions": 3,
                "max_daily_drawdown": 0.05
            }
        },
        "strategies": {
            "trend_following": {
                "enabled": True,
                "timeframes": ["1h", "4h"],
                "indicators": ["EMA", "RSI"]
            },
            "breakout": {
                "enabled": True,
                "timeframes": ["1h"],
                "indicators": ["BB", "Volume"]
            },
            "mean_reversion": {
                "enabled": False,
                "timeframes": ["15m", "1h"],
                "indicators": ["BB", "RSI"]
            }
        }
    }
    
    import yaml
    config_path = "config/config.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.info(f"Created default configuration: {config_path}")

def create_env_file():
    """Create .env file if it doesn't exist."""
    if not os.path.exists('.env'):
        env_content = """FLASK_APP=main.py
FLASK_ENV=development
PORT=5000
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=true
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        logging.info("Created .env file with default settings")

def verify_installation():
    """Verify all components are properly installed."""
    checks = {
        "Directories": all(os.path.exists(d) for d in ["logs", "data/cache", "python_libs"]),
        "Config": os.path.exists("config/config.yaml"),
        "Environment": os.path.exists(".env"),
        "Portfolio Manager": os.path.exists("python_libs/portfolio_manager.py")
    }
    
    all_passed = all(checks.values())
    if all_passed:
        logging.info("✅ All installation checks passed")
    else:
        failed = [k for k, v in checks.items() if not v]
        logging.error(f"❌ Failed checks: {failed}")
    
    return all_passed

def main():
    """Main setup function."""
    logging.info("Starting trading system setup...")
    
    create_directories()
    create_default_config()
    create_env_file()
    
    if verify_installation():
        logging.info("Setup completed successfully")
        return True
    else:
        logging.error("Setup completed with errors")
        return False

if __name__ == "__main__":
    main()