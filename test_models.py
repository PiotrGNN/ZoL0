
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
