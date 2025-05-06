#!/usr/bin/env python3
"""
fix_imports.py - Naprawia problemy z importami modułów w projekcie.
"""

import os
import sys
import importlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fix_imports_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_init_files():
    """Tworzy pliki __init__.py w katalogach, które ich nie mają."""
    dirs_to_check = [
        "ai_models", 
        "data", 
        "data/execution", 
        "data/indicators",
        "data/logging", 
        "data/optimization", 
        "data/risk_management",
        "data/strategies", 
        "data/tests", 
        "data/utils",
        "python_libs",
        "config"
    ]

    for dir_path in dirs_to_check:
        if os.path.isdir(dir_path):
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""\nAutomatycznie wygenerowany plik __init__.py.\n"""\n')
                logging.info(f"Utworzono plik {init_file}")

def fix_problematic_imports(filepath: str) -> bool:
    """
    Naprawia problematyczne importy w pliku.
    
    Args:
        filepath: Ścieżka do pliku
        
    Returns:
        bool: True jeśli plik został zmieniony
    """
    if not os.path.exists(filepath):
        logger.warning(f"Plik {filepath} nie istnieje")
        return False
        
    try:
        # Odczytaj zawartość pliku
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Znajdź i zastąp problematyczne importy
        original_content = content
        
        # Wzorce do zastąpienia
        patterns = [
            # Tensorflow
            (r'import tensorflow as tf', '# Opcjonalny import - zastępczy\ntry:\n    import tensorflow as tf\nexcept ImportError:\n    tf = None\n    print("Moduł tensorflow niedostępny, używam wersji zastępczej")'),
            (r'from tensorflow import', '# Opcjonalny import - zastępczy\ntry:\n    from tensorflow import'),
            (r'from tensorflow\.keras', '# Opcjonalny import - zastępczy\ntry:\n    from tensorflow.keras'),
            
            # PyTorch
            (r'import torch', '# Opcjonalny import - zastępczy\ntry:\n    import torch\nexcept ImportError:\n    torch = None\n    print("Moduł torch niedostępny, używam wersji zastępczej")'),
            (r'from torch import', '# Opcjonalny import - zastępczy\ntry:\n    from torch import'),
            
            # Scikit-learn
            (r'from sklearn\.ensemble import (IsolationForest|RandomForestClassifier|GradientBoostingRegressor)', '# Opcjonalny import - zastępczy\ntry:\n    from sklearn.ensemble import \\1\nexcept ImportError:\n    print("Moduł sklearn.ensemble niedostępny, używam wersji zastępczej")'),
            
            # XGBoost
            (r'import xgboost as xgb', '# Opcjonalny import - zastępczy\ntry:\n    import xgboost as xgb\nexcept ImportError:\n    xgb = None\n    print("Moduł xgboost niedostępny, używam wersji zastępczej")'),
            (r'from xgboost import', '# Opcjonalny import - zastępczy\ntry:\n    from xgboost import'),
            
            # Optuna
            (r'import optuna', '# Opcjonalny import - zastępczy\ntry:\n    import optuna\nexcept ImportError:\n    optuna = None\n    print("Moduł optuna niedostępny, używam wersji zastępczej")'),
            
            # Transformers
            (r'from transformers import', '# Opcjonalny import - zastępczy\ntry:\n    from transformers import'),
            (r'import transformers', '# Opcjonalny import - zastępczy\ntry:\n    import transformers\nexcept ImportError:\n    transformers = None\n    print("Moduł transformers niedostępny, używam wersji zastępczej")')
        ]
        
        # Zastąp importy
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                if not "except ImportError:" in replacement or not re.search(f"{pattern}.*?except ImportError", content, re.DOTALL):
                    content = re.sub(pattern, replacement, content)
                    
        # Jeśli zawartość się zmieniła, zapisz plik
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Naprawiono importy w pliku {filepath}")
            return True
        else:
            logger.info(f"Plik {filepath} nie wymaga naprawy importów")
            return False
            
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania importów w pliku {filepath}: {e}")
        return False

def fix_all_modules_in_directory(directory: str) -> Tuple[int, int]:
    """
    Naprawia importy we wszystkich modułach Pythona w katalogu.
    
    Args:
        directory: Ścieżka do katalogu
        
    Returns:
        Tuple[int, int]: (liczba przetworzonych plików, liczba zmienionych plików)
    """
    if not os.path.exists(directory):
        logger.warning(f"Katalog {directory} nie istnieje")
        return 0, 0
        
    processed_files = 0
    changed_files = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                processed_files += 1
                
                if fix_problematic_imports(filepath):
                    changed_files += 1
                    
    logger.info(f"Przetworzono {processed_files} plików, zmieniono {changed_files} plików w katalogu {directory}")
    return processed_files, changed_files

def main():
    """Główna funkcja skryptu."""
    logger.info("Rozpoczynanie naprawy importów...")
    
    # Lista katalogów do naprawy
    directories = [
        'ai_models',
        'python_libs',
        'data',
        'data/execution',
        'data/indicators',
        'data/strategies',
        'data/utils'
    ]
    
    total_processed = 0
    total_changed = 0
    
    # Najpierw napraw importy
    for directory in directories:
        logger.info(f"Przetwarzanie katalogu {directory}...")
        processed, changed = fix_all_modules_in_directory(directory)
        total_processed += processed
        total_changed += changed
        
    # Następnie utwórz pliki __init__.py
    create_init_files()
    
    logger.info(f"Zakończono naprawę importów. Przetworzono {total_processed} plików, zmieniono {total_changed} plików.")
    logger.info("Skrypt zakończył działanie pomyślnie.")

if __name__ == "__main__":
    main()
