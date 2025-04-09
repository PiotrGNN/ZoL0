
#!/usr/bin/env python3
"""
test_models.py
-------------
Skrypt do testowania modeli AI w projekcie.
Skanuje folder ai_models/ i testuje modele.
"""

import os
import sys
import argparse
from python_libs.model_tester import ModelTester

def parse_arguments():
    """Parsowanie argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(description='Testowanie modeli AI')
    parser.add_argument(
        '--path', 
        type=str, 
        default='ai_models',
        help='Ścieżka do folderu z modelami (domyślnie: ai_models)'
    )
    parser.add_argument(
        '--log', 
        type=str, 
        default='logs/model_check.log',
        help='Ścieżka do pliku logów (domyślnie: logs/model_check.log)'
    )
    return parser.parse_args()

def main():
    """Główna funkcja skryptu."""
    args = parse_arguments()
    
    # Tworzenie katalogu logów, jeśli nie istnieje
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    print(f"🔍 Rozpoczynam testowanie modeli w {args.path}...")
    
    # Uruchomienie testera
    tester = ModelTester(models_path=args.path, log_path=args.log)
    stats = tester.run_tests()
    
    # Wyświetlenie podsumowania
    print("\n📊 Podsumowanie testów:")
    print(f"- Przeskanowano plików .py: {stats['py_files_scanned']}")
    print(f"- Przeskanowano plików .pkl: {stats['pkl_files_scanned']}")
    print(f"- Logi zapisano do: {args.log}")
    
    print("\n✅ Testowanie zakończone.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Test Models

Skrypt sprawdzający, czy wszystkie modele AI są prawidłowo ładowane.
"""

import os
import sys
import logging
import importlib
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test_models")

def scan_directory(directory="ai_models"):
    """Skanuje wskazany katalog i zwraca listę plików .py"""
    module_files = []
    try:
        for file in Path(directory).glob("*.py"):
            if file.name != "__init__.py" and not file.name.startswith("_"):
                module_files.append(file.stem)
        logger.info(f"Znaleziono {len(module_files)} plików .py w katalogu {directory}")
        return module_files
    except Exception as e:
        logger.error(f"Błąd podczas skanowania katalogu {directory}: {e}")
        return []

def test_model_loader():
    """Testuje moduł ładujący modele AI"""
    try:
        # Sprawdź, czy istnieje moduł model_loader
        from ai_models.model_loader import model_loader
        
        # Sprawdź, czy metoda load_models istnieje
        if hasattr(model_loader, "load_models"):
            model_loader.load_models()
            models = model_loader.get_models_summary()
            logger.info(f"Model loader działa poprawnie. Załadowano {len(models)} modeli")
            for model in models:
                logger.info(f"  - {model.get('name', 'Nieznany')} ({model.get('type', 'Nieznany typ')})")
            return True
        else:
            logger.error("Model loader nie posiada metody load_models")
            return False
    except ImportError as e:
        logger.error(f"Nie można zaimportować model_loader: {e}")
        return False
    except Exception as e:
        logger.error(f"Błąd podczas testowania model_loader: {e}")
        return False

def test_import_modules(modules):
    """Próbuje zaimportować wszystkie moduły z listy"""
    successful = 0
    failed = 0
    
    for module_name in modules:
        try:
            module_path = f"ai_models.{module_name}"
            module = importlib.import_module(module_path)
            logger.info(f"✅ Zaimportowano moduł: {module_path}")
            
            # Sprawdź klasy w module
            classes = [name for name, obj in module.__dict__.items() 
                      if isinstance(obj, type) and not name.startswith("_")]
            
            if classes:
                logger.info(f"   Znalezione klasy: {', '.join(classes)}")
            successful += 1
        except Exception as e:
            logger.error(f"❌ Błąd importu modułu {module_name}: {e}")
            failed += 1
    
    return successful, failed

if __name__ == "__main__":
    print("\n===== TEST MODELI AI =====\n")
    
    # Sprawdź, czy katalog ai_models istnieje
    if not os.path.isdir("ai_models"):
        logger.error("Katalog ai_models nie istnieje!")
        sys.exit(1)
    
    # Skanuj katalog ai_models
    modules = scan_directory("ai_models")
    
    # Testuj model loader
    loader_ok = test_model_loader()
    
    # Testuj import modułów
    successful, failed = test_import_modules(modules)
    
    # Podsumowanie
    print("\n===== PODSUMOWANIE =====")
    print(f"Znaleziono {len(modules)} modułów w katalogu ai_models")
    print(f"Pomyślnie zaimportowano: {successful}")
    print(f"Błędy importu: {failed}")
    print(f"Model loader: {'OK' if loader_ok else 'BŁĄD'}")
    
    if failed > 0 or not loader_ok:
        print("\n⚠️ Wykryto problemy z modelami AI!")
        sys.exit(1)
    else:
        print("\n✅ Wszystkie modele AI są prawidłowo ładowane!")
#!/usr/bin/env python3
"""
test_models.py - Skrypt do testowania modeli AI.

Ten skrypt testuje wszystkie modele AI w projekcie, sprawdzając ich
poprawność i działanie.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Dict, List, Any

# Dodaj katalog główny do ścieżki Pythona
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importuj tester modeli
try:
    from python_libs.model_tester import ModelTester
except ImportError:
    print("❌ Nie znaleziono modułu model_tester w python_libs")
    print("Uruchom najpierw setup_local_packages.py aby zainstalować wymagane pakiety")
    sys.exit(1)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_models.log"),
        logging.StreamHandler()
    ]
)

def generate_test_data() -> tuple:
    """
    Generuje dane do testowania modeli.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Generuj dane do klasyfikacji
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_classes=2, 
        random_state=42
    )
    
    # Podziel na zestaw treningowy i testowy
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def test_models() -> Dict[str, Any]:
    """
    Testuje wszystkie modele w projekcie.
    
    Returns:
        Dict[str, Any]: Wyniki testów
    """
    print("🔍 Rozpoczęcie testowania modeli AI...")
    
    # Inicjalizuj tester modeli
    model_tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')
    
    # Uruchom testy
    results = model_tester.run_tests()
    
    # Pobierz załadowane modele
    models = model_tester.get_loaded_models()
    
    # Wygeneruj dane testowe
    X_train, X_test, y_train, y_test = generate_test_data()
    
    # Testuj modele typu ML z metodami fit/predict
    ml_results = {}
    for model_info in models:
        model_name = model_info['name']
        instance = model_info['instance']
        
        # Sprawdź czy model ma metody fit i predict
        if model_info['has_fit'] and model_info['has_predict']:
            print(f"⏳ Testowanie modelu ML: {model_name}...")
            
            try:
                # Trenuj model
                instance.fit(X_train, y_train)
                
                # Ocena modelu
                evaluation = model_tester.evaluate_model(model_name, X_test, y_test)
                ml_results[model_name] = evaluation
                
                # Zapisz accuracy w informacjach o modelu
                if 'accuracy' in evaluation:
                    print(f"✅ Model {model_name}: accuracy = {evaluation['accuracy']:.4f}")
                else:
                    print(f"⚠️ Model {model_name}: brak metryki accuracy")
            except Exception as e:
                print(f"❌ Błąd podczas testowania modelu {model_name}: {e}")
                ml_results[model_name] = {"error": str(e)}
    
    # Zapisz metadane modeli
    model_tester.save_model_metadata("model_metadata.json")
    
    print("\n📋 Podsumowanie testów modeli:")
    print(f"- Wykryto {results.get('models_detected', 0)} modeli")
    print(f"- Załadowano {results.get('models_loaded', 0)} modeli")
    print(f"- Przetestowano {len(ml_results)} modeli ML")
    
    if results.get('errors', []):
        print("\n⚠️ Problemy podczas testów:")
        for error in results.get('errors', []):
            print(f"- {error}")
            
    return {
        "results": results,
        "ml_results": ml_results,
        "models": [model['name'] for model in models]
    }

def main():
    """
    Funkcja główna.
    """
    try:
        results = test_models()
        
        print("\n✅ Testowanie modeli zakończone pomyślnie!")
        return 0
    except Exception as e:
        print(f"\n❌ Błąd podczas testowania modeli: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
