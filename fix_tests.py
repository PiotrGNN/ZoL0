
#!/usr/bin/env python3
"""
fix_tests.py
-----------
Skrypt testujący podstawowe funkcjonalności projektu.
"""

import os
import sys
import logging
import importlib
from datetime import datetime

# Konfiguracja logowania
log_file = "test_errors.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def run_basic_tests():
    """Uruchamia podstawowe testy funkcjonalności."""
    print("\n===== BASIC FUNCTIONALITY TESTS =====\n")
    logging.info(f"Rozpoczęcie testów funkcjonalności: {datetime.now()}")
    
    tests = [
        test_cache_manager,
        test_bybit_connector,
        test_rate_limiter,
        test_project_structure
    ]
    
    results = []
    for test_func in tests:
        try:
            name = test_func.__name__.replace('test_', '').replace('_', ' ').capitalize()
            print(f"Running test: {name}...")
            success, message = test_func()
            results.append((name, success, message))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {message}\n")
        except Exception as e:
            results.append((test_func.__name__, False, f"Unexpected error: {e}"))
            logging.exception(f"Error in test {test_func.__name__}")
            print(f"❌ FAIL: Test crashed with error: {e}\n")
    
    # Wyświetl podsumowanie
    print("\n===== TEST SUMMARY =====")
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
    
    logging.info(f"Zakończenie testów funkcjonalności. Passed: {passed}/{total}")
    return passed == total

def test_cache_manager():
    """Test funkcjonalności cache managera."""
    try:
        from data.utils.cache_manager import (
            init_cache_manager, 
            store_cached_data, 
            get_cached_data,
            is_cache_valid
        )
        
        # Inicjalizacja
        init_cache_manager()
        
        # Test zapisu i odczytu
        test_key = "test_key"
        test_data = {"test": True, "value": 42, "nested": {"more": "data"}}
        
        # Zapisz dane
        store_result = store_cached_data(test_key, test_data)
        if not store_result:
            return False, "Failed to store data in cache"
        
        # Odczytaj dane
        data, found = get_cached_data(test_key)
        if not found:
            return False, "Data not found in cache after storing"
        
        if data.get("value") != 42:
            return False, f"Incorrect data retrieved from cache: {data}"
        
        # Sprawdź ważność
        valid = is_cache_valid(test_key)
        if not valid:
            return False, "Cache marked as invalid immediately after storing"
        
        # Test obsługi wartości bool
        bool_key = "bool_test"
        store_cached_data(bool_key, True)
        bool_data, bool_found = get_cached_data(bool_key)
        
        if not bool_found:
            return False, "Boolean data not found in cache"
        
        if not isinstance(bool_data, dict):
            return False, f"Boolean data not properly converted to dict: {type(bool_data)}"
        
        return True, "Cache manager tests passed successfully"
    except Exception as e:
        logging.exception("Error in cache manager test")
        return False, f"Cache manager test failed with error: {e}"

def test_bybit_connector():
    """Test podstawowej funkcjonalności Bybit connectora."""
    try:
        from data.execution.bybit_connector import BybitConnector
        
        # Tworzenie instancji z domyślnymi wartościami
        connector = BybitConnector(
            api_key="test_key", 
            api_secret="test_secret",
            use_testnet=True
        )
        
        # Sprawdź, czy pole base_url zostało ustawione poprawnie
        if "testnet" not in connector.base_url:
            return False, f"Incorrect base URL for testnet: {connector.base_url}"
        
        # Test metody get_server_time
        server_time = connector.get_server_time()
        if not isinstance(server_time, dict) or 'time_ms' not in server_time:
            return False, f"Invalid server time response: {server_time}"
        
        return True, "Bybit connector basic tests passed"
    except Exception as e:
        logging.exception("Error in bybit connector test")
        return False, f"Bybit connector test failed with error: {e}"

def test_rate_limiter():
    """Test funkcjonalności rate limitera."""
    try:
        from data.utils.cache_manager import set_rate_limit_parameters, get_api_status
        
        # Ustaw parametry
        set_rate_limit_parameters(max_calls_per_minute=10, min_interval=0.5)
        
        # Pobierz status i sprawdź, czy nie ma błędów
        status = get_api_status()
        if not isinstance(status, dict):
            return False, f"Invalid API status response: {status}"
        
        return True, "Rate limiter tests passed"
    except Exception as e:
        logging.exception("Error in rate limiter test")
        return False, f"Rate limiter test failed with error: {e}"

def test_project_structure():
    """Test struktury projektu."""
    expected_dirs = [
        "data",
        "data/execution",
        "data/utils",
        "data/logging",
        "data/strategies",
        "data/risk_management",
        "data/optimization",
        "logs"
    ]
    
    missing_dirs = []
    for dir_path in expected_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
    
    expected_files = [
        "main.py",
        "requirements.txt",
        ".env.example",
        "data/execution/bybit_connector.py",
        "data/utils/cache_manager.py",
        "data/logging/trade_logger.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        message = ""
        if missing_dirs:
            message += f"Missing directories: {', '.join(missing_dirs)}. "
        if missing_files:
            message += f"Missing files: {', '.join(missing_files)}"
        return False, message
    
    return True, "Project structure validation passed"

if __name__ == "__main__":
    print("Running tests to verify project functionality...")
    success = run_basic_tests()
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check logs for details.")
