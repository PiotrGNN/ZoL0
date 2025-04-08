
#!/usr/bin/env python3
"""
fix_imports.py
-------------
Skrypt testujący poprawność importów i zależności w projekcie.
"""

import os
import sys
import importlib
import logging
import time
from datetime import datetime

# Konfiguracja logowania
log_file = "fix_imports_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def fix_imports():
    """Sprawdza importy i brakujące moduły"""
    start_time = time.time()
    logging.info(f"Rozpoczynam weryfikację importów w {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sprawdzam importy, log w pliku: {log_file}")
    
    # Lista kluczowych importów do sprawdzenia
    import_tests = [
        "import flask",
        "import requests",
        "import pandas as pd",
        "import numpy as np",
        "import dotenv",
        "import pybit",
        "import matplotlib",
        "import sklearn",
        "import psutil",
        "import yaml",
        "import joblib"
    ]
    
    missing_modules = []
    
    try:
        for test in import_tests:
            try:
                exec(test)
                logging.info(f"Successfully imported: {test}")
            except ImportError as e:
                missing_modules.append((test.split("import ")[1].split(",")[0].strip(), str(e)))
                logging.error(f"Import error with '{test}': {e}")

        if missing_modules:
            print("\n==== MISSING MODULES ====")
            print("Run the following commands to install missing dependencies:")
            for module, error in missing_modules:
                print(f"pip install {module}")
            print("===========================\n")
        else:
            print("\n✅ All key imports checked successfully!")

        logging.info("All key imports checked.")
    except Exception as e:
        logging.error(f"Error in import testing: {e}")
        print(f"Error checking imports: {e}")
    
    # Sprawdzenie importów modułów projektu
    modules_to_check = [
        ("data.execution.bybit_connector", "BybitConnector"),
        ("data.utils.cache_manager", "init_cache_manager"),
        ("data.logging.anomaly_detector", "AnomalyDetector"),
        ("data.logging.trade_logger", "TradeLogger"),
        ("data.optimization.hyperparameter_tuner", "HyperparameterTuner"),
        ("data.risk_management.portfolio_risk", "PortfolioRiskManager"),
        ("data.strategies.AI_strategy_generator", "AIStrategyGenerator"),
        ("data.utils.performance_monitor", "PerformanceMonitor")
    ]
    
    print("\n==== CHECKING PROJECT MODULES ====")
    for module_path, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                print(f"✅ Module {module_path} with {class_name} OK")
                logging.info(f"Module {module_path} with {class_name} imported successfully")
            else:
                print(f"❌ Module {module_path} found but {class_name} missing")
                logging.error(f"Module {module_path} found but {class_name} missing")
        except ImportError as e:
            print(f"❌ Failed to import {module_path}: {e}")
            logging.error(f"Failed to import {module_path}: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Import verification completed in {duration:.2f} seconds")
    print(f"\nVerification completed in {duration:.2f} seconds")

if __name__ == "__main__":
    print("Fixing imports...")
    fix_imports()
    print(f"Done! Check {log_file} for more details.")
