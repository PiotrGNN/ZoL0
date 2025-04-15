
#!/usr/bin/env python3
"""
manage_ai_models.py - Narzędzie do zarządzania modelami AI w środowisku lokalnym
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Dodanie ścieżek do PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
python_libs_dir = os.path.join(current_dir, "python_libs")
sys.path.append(python_libs_dir)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def list_models():
    """Wyświetla listę dostępnych modeli AI."""
    try:
        from ai_models.model_recognition import ModelRecognizer
        recognizer = ModelRecognizer()
        models = recognizer.scan_models()
        
        if not models:
            logger.info("Nie znaleziono żadnych modeli AI.")
            return
        
        logger.info(f"Znaleziono {len(models)} modeli AI:")
        for idx, model in enumerate(models, 1):
            status = model.get("status", "Nieznany")
            status_symbol = "✅" if status == "Active" else "❌" if status == "Error" else "❓"
            logger.info(f"{idx}. {status_symbol} {model['name']} ({model.get('type', 'Nieznany')}) - Dokładność: {model.get('accuracy', 'N/A')}%")
    
    except Exception as e:
        logger.error(f"Błąd podczas wyświetlania modeli: {e}")

def test_model(model_name=None):
    """Testuje określony model AI."""
    try:
        from ai_models.model_recognition import ModelRecognizer
        from python_libs.model_tester import ModelTester
        
        tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')
        
        if model_name:
            logger.info(f"Testowanie modelu: {model_name}")
            result = tester.test_model(model_name)
            if result["success"]:
                logger.info(f"Test zakończony pomyślnie. Dokładność: {result.get('accuracy', 'N/A')}%")
            else:
                logger.error(f"Test nie powiódł się: {result.get('error', 'Nieznany błąd')}")
        else:
            logger.info("Testowanie wszystkich modeli...")
            results = tester.run_tests()
            logger.info(f"Testy zakończone. Wyniki: {len(results)} modeli przetestowano.")
            
            # Wyświetlanie wyników
            for model_name, result in results.items():
                status = "✅ Sukces" if result.get("success", False) else "❌ Błąd"
                logger.info(f"{model_name}: {status} - Dokładność: {result.get('accuracy', 'N/A')}%")
    
    except Exception as e:
        logger.error(f"Błąd podczas testowania modeli: {e}")

def main():
    """Funkcja główna."""
    parser = argparse.ArgumentParser(description="Narzędzie do zarządzania modelami AI")
    parser.add_argument("--list", action="store_true", help="Wyświetl listę dostępnych modeli")
    parser.add_argument("--test", action="store_true", help="Uruchom testy modeli")
    parser.add_argument("--model", type=str, help="Określony model do przetestowania")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.test:
        test_model(args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    logger.info("Narzędzie zarządzania modelami AI")
    main()
#!/usr/bin/env python3
"""
manage_ai_models.py
------------------
Narzędzie CLI do zarządzania modelami AI: przeglądanie, czyszczenie, retrenowanie
i monitorowanie ich wydajności.
"""

import os
import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_models_manager")

try:
    from ai_models.model_utils import (
        list_available_models, 
        save_model, 
        load_model,
        create_model_checkpoint,
        ensure_model_dirs
    )
except ImportError:
    logger.error("Nie można zaimportować modułu model_utils. Sprawdź, czy plik istnieje i czy ścieżka jest poprawna.")
    # Fallback do podstawowych funkcji
    def ensure_model_dirs():
        os.makedirs("models", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)
    
    def list_available_models():
        models = []
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if file.endswith(".pkl"):
                    models.append({
                        "name": file.replace(".pkl", ""),
                        "path": os.path.join("models", file),
                        "size": os.path.getsize(os.path.join("models", file)),
                        "last_modified": datetime.fromtimestamp(os.path.getmtime(os.path.join("models", file))).strftime("%Y-%m-%d %H:%M:%S")
                    })
        return models

def print_models_table(models: List[Dict[str, Any]]) -> None:
    """
    Wyświetla tabelę z informacjami o modelach.
    
    Args:
        models: Lista modeli
    """
    if not models:
        print("Nie znaleziono modeli.")
        return
    
    # Przygotuj dane do wyświetlenia
    headers = ["Nazwa", "Typ", "Rozmiar", "Ostatnia modyfikacja", "Accuracy"]
    rows = []
    
    for model in models:
        # Pobierz typ modelu z metadanych, jeśli dostępne
        model_type = model.get("metadata", {}).get("model_type", "Nieznany")
        
        # Pobierz accuracy z metadanych, jeśli dostępne
        accuracy = model.get("metadata", {}).get("accuracy", "N/A")
        if isinstance(accuracy, (int, float)):
            accuracy = f"{accuracy:.2f}%"
        
        # Sformatuj rozmiar
        size_kb = model["size"] / 1024
        if size_kb < 1024:
            size_str = f"{size_kb:.2f} KB"
        else:
            size_mb = size_kb / 1024
            size_str = f"{size_mb:.2f} MB"
        
        rows.append([
            model["name"],
            model_type,
            size_str,
            model["last_modified"],
            accuracy
        ])
    
    # Znajdź maksymalną szerokość dla każdej kolumny
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    
    # Wyświetl nagłówek
    header_row = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))
    
    # Wyświetl dane
    for row in rows:
        print(" | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(headers))))

def create_model_backups() -> None:
    """Tworzy kopie zapasowe wszystkich modeli."""
    models = list_available_models()
    if not models:
        print("Nie znaleziono modeli do wykonania kopii zapasowej.")
        return
    
    print(f"Tworzenie kopii zapasowych {len(models)} modeli...")
    
    try:
        from python_libs.model_tester import ModelTester
        tester = ModelTester()
        backup_dir = os.path.join("saved_models", f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_dir, exist_ok=True)
        
        for model_info in models:
            model_name = model_info["name"]
            
            try:
                # Załaduj model
                model, metadata, success = load_model(model_name)
                
                if success and model is not None:
                    # Utwórz kopię zapasową
                    import joblib
                    backup_path = os.path.join(backup_dir, f"{model_name}.pkl")
                    joblib.dump(model, backup_path)
                    
                    # Zapisz metadane
                    with open(os.path.join(backup_dir, f"{model_name}_metadata.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"✅ Utworzono kopię zapasową modelu {model_name} w {backup_path}")
                else:
                    print(f"❌ Nie udało się załadować modelu {model_name}")
            except Exception as e:
                print(f"❌ Błąd podczas tworzenia kopii zapasowej modelu {model_name}: {e}")
        
        print(f"Kopie zapasowe zapisane w {backup_dir}")
    except Exception as e:
        print(f"❌ Błąd podczas tworzenia kopii zapasowych: {e}")

def test_all_models() -> None:
    """Testuje wszystkie dostępne modele."""
    try:
        from python_libs.model_tester import ModelTester
        tester = ModelTester()
        results = tester.run_tests()
        
        print("\n📊 Wyniki testów modeli:")
        for model_name, result in results.items():
            status = "✅" if result.get("predict_successful", False) else "❌"
            accuracy = result.get("accuracy")
            accuracy_str = f"{accuracy:.2f}%" if isinstance(accuracy, (int, float)) else "N/A"
            
            print(f"{status} {model_name}: Accuracy: {accuracy_str}")
    except Exception as e:
        print(f"❌ Błąd podczas testowania modeli: {e}")

def retrain_selected_models(models_to_retrain: List[str], force: bool = False) -> None:
    """
    Trenuje wybrane modele.
    
    Args:
        models_to_retrain: Lista nazw modeli do przetrenowania
        force: Czy wymusić przetrenowanie nawet jeśli dane są te same
    """
    try:
        from python_libs.model_tester import ModelTester
        tester = ModelTester()
        
        if not models_to_retrain or models_to_retrain[0].lower() == "all":
            # Trenuj wszystkie modele
            print("Rozpoczynam trening wszystkich modeli...")
            results = tester.run_tests(force_retrain=force)
        else:
            # Trenuj tylko wybrane modele
            results = {}
            for model_name in models_to_retrain:
                print(f"Rozpoczynam trening modelu {model_name}...")
                model_results = tester.test_model_by_name(model_name, force_retrain=force)
                results[model_name] = model_results
        
        print("\n📝 Wyniki treningu:")
        for model_name, result in results.items():
            fit_success = "✅" if result.get("fit_successful", False) else "❌"
            fit_time = result.get("fit_time", 0)
            fit_time_str = f"{fit_time:.2f}s" if isinstance(fit_time, (int, float)) else "N/A"
            
            print(f"{fit_success} {model_name}: Czas treningu: {fit_time_str}")
    except Exception as e:
        print(f"❌ Błąd podczas trenowania modeli: {e}")

def show_model_details(model_name: str) -> None:
    """
    Wyświetla szczegółowe informacje o modelu.
    
    Args:
        model_name: Nazwa modelu
    """
    try:
        # Załaduj model i metadane
        model, metadata, success = load_model(model_name)
        
        if not success or model is None:
            print(f"❌ Nie udało się załadować modelu {model_name}")
            return
        
        print(f"\n📋 Szczegóły modelu {model_name}:")
        print(f"Typ: {type(model).__name__}")
        print(f"Moduł: {type(model).__module__}")
        
        # Wyświetl metadane, jeśli dostępne
        if metadata:
            print("\nMetadane:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("\nBrak metadanych.")
        
        # Sprawdź, czy model ma atrybuty, które mogą być interesujące
        if hasattr(model, 'get_params'):
            print("\nParametry modelu:")
            try:
                params = model.get_params()
                for key, value in params.items():
                    print(f"  {key}: {value}")
            except:
                print("  Nie udało się pobrać parametrów.")
        
        # Sprawdź, czy model ma atrybuty: feature_importances_, coef_, nebo classes_
        for attr in ['feature_importances_', 'coef_', 'classes_']:
            if hasattr(model, attr):
                print(f"\n{attr}:")
                try:
                    value = getattr(model, attr)
                    if hasattr(value, 'shape'):
                        print(f"  Kształt: {value.shape}")
                    if hasattr(value, 'tolist'):
                        if value.size <= 20:  # Wyświetl tylko niewielkie wartości
                            print(f"  Wartości: {value.tolist()}")
                        else:
                            print(f"  Pierwsze 5 wartości: {value.flatten()[:5].tolist()}")
                    else:
                        print(f"  Wartość: {value}")
                except:
                    print(f"  Nie udało się pobrać {attr}.")
    except Exception as e:
        print(f"❌ Błąd podczas wyświetlania szczegółów modelu: {e}")

def main():
    # Upewnij się, że katalogi istnieją
    ensure_model_dirs()
    
    # Parser argumentów
    parser = argparse.ArgumentParser(description="Narzędzie do zarządzania modelami AI")
    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy")
    
    # Komenda: list
    list_parser = subparsers.add_parser("list", help="Wyświetla listę modeli")
    
    # Komenda: backup
    backup_parser = subparsers.add_parser("backup", help="Tworzy kopie zapasowe modeli")
    
    # Komenda: test
    test_parser = subparsers.add_parser("test", help="Testuje modele")
    
    # Komenda: retrain
    retrain_parser = subparsers.add_parser("retrain", help="Trenuje wybrane modele")
    retrain_parser.add_argument("models", nargs="*", default=["all"], help="Modele do przetrenowania (domyślnie: all)")
    retrain_parser.add_argument("--force", action="store_true", help="Wymusza przetrenowanie nawet jeśli dane są te same")
    
    # Komenda: details
    details_parser = subparsers.add_parser("details", help="Wyświetla szczegóły modelu")
    details_parser.add_argument("model", help="Nazwa modelu")
    
    # Parsuj argumenty
    args = parser.parse_args()
    
    # Obsługa komend
    if args.command == "list" or not args.command:
        # Domyślnie wyświetl listę modeli
        models = list_available_models()
        print_models_table(models)
    elif args.command == "backup":
        create_model_backups()
    elif args.command == "test":
        test_all_models()
    elif args.command == "retrain":
        retrain_selected_models(args.models, args.force)
    elif args.command == "details":
        show_model_details(args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
