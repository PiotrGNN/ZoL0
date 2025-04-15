
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
#!/usr/bin/env python
"""
manage_ai_models.py
------------------
Skrypt do zarządzania modelami AI - sprawdzanie statusu, czyszczenie, wymuszanie retreningu.
"""

import os
import sys
import json
import time
import logging
import argparse
import shutil
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_models_management.log'),
        logging.StreamHandler()
    ]
)

# Upewnij się, że katalog logów istnieje
os.makedirs('logs', exist_ok=True)

def list_models():
    """Wyświetla listę dostępnych modeli."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("Katalog modeli nie istnieje!")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
    
    if not model_files:
        print("Brak zapisanych modeli.")
        return
    
    print(f"\nZnaleziono {len(model_files)} modeli:\n")
    
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '')
        model_path = os.path.join(models_dir, model_file)
        
        # Podstawowe informacje o pliku
        size_kb = os.path.getsize(model_path) / 1024
        modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        
        # Sprawdź czy istnieją metadane
        metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        metadata_info = ""
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if isinstance(metadata, dict) and 'accuracy' in metadata:
                    metadata_info = f", Dokładność: {metadata['accuracy']:.2f}"
                elif isinstance(metadata, dict) and 'version' in metadata:
                    metadata_info = f", Wersja: {metadata['version']}"
            except json.JSONDecodeError:
                metadata_info = " (Błąd odczytu metadanych)"
        
        print(f"- {model_name:<20} | {size_kb:6.1f} KB | {modified_time.strftime('%Y-%m-%d %H:%M:%S')}{metadata_info}")

def clean_models(force=False):
    """Czyści uszkodzone i niepełne modele."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("Katalog modeli nie istnieje!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
    metadata_files = [f for f in os.listdir(models_dir) if f.endswith('_metadata.json')]
    
    # Wykrywanie uszkodzonych plików modeli
    removed_count = 0
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        # Sprawdź czy plik jest pusty lub uszkodzony
        if os.path.getsize(model_path) == 0:
            if force or input(f"Model {model_file} jest pusty. Usunąć? [t/N]: ").lower() == 't':
                os.remove(model_path)
                print(f"Usunięto pusty model: {model_file}")
                removed_count += 1
        else:
            # Test wczytania modelu
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    # Próba wczytania tylko nagłówka pliku pickle
                    pickle.load(f)
            except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                if force or input(f"Model {model_file} jest uszkodzony ({str(e)}). Usunąć? [t/N]: ").lower() == 't':
                    os.remove(model_path)
                    print(f"Usunięto uszkodzony model: {model_file}")
                    removed_count += 1
    
    # Sprawdź czy są niepasujące pliki metadanych
    for metadata_file in metadata_files:
        model_name = metadata_file.replace('_metadata.json', '')
        model_file = f"{model_name}_model.pkl"
        
        if model_file not in model_files:
            if force or input(f"Plik metadanych {metadata_file} nie ma odpowiadającego modelu. Usunąć? [t/N]: ").lower() == 't':
                os.remove(os.path.join(models_dir, metadata_file))
                print(f"Usunięto osierocony plik metadanych: {metadata_file}")
                removed_count += 1
    
    print(f"\nCzyszczenie zakończone. Usunięto {removed_count} plików.")

def backup_models():
    """Tworzy kopię zapasową modeli."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("Katalog modeli nie istnieje!")
        return
    
    # Utwórz katalog backupu z timestampem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = f"models_backup_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Kopiuj wszystkie pliki modeli
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('_model.pkl', '_metadata.json'))]
    
    if not model_files:
        print("Brak modeli do backupu!")
        os.rmdir(backup_dir)
        return
    
    for file in model_files:
        src_path = os.path.join(models_dir, file)
        dst_path = os.path.join(backup_dir, file)
        shutil.copy2(src_path, dst_path)
    
    print(f"Utworzono backup {len(model_files)} plików w katalogu {backup_dir}")

def test_models():
    """Testuje wszystkie modele."""
    print("Uruchamianie testów modeli...")
    
    # Importuj tester modeli
    try:
        sys.path.insert(0, os.getcwd())
        from python_libs.model_tester import ModelTester
        
        # Inicjalizacja testera modeli
        tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')
        
        # Ładowanie i testowanie modeli
        loaded_models = tester.load_models()
        
        print(f"\nZaładowano {len(loaded_models)} modeli\n")
        
        # Przeprowadź testy
        success_count = 0
        for model_info in loaded_models:
            model_name = model_info.get('name', 'Nieznany model')
            model_instance = model_info.get('instance')
            
            if model_instance:
                print(f"Testowanie modelu: {model_name}")
                result = tester.test_model(model_instance, model_name)
                if result:
                    success_count += 1
                    print(f"  ✅ Sukces")
                else:
                    print(f"  ❌ Błąd")
            else:
                print(f"Nie udało się załadować modelu: {model_name}")
        
        print(f"\nWyniki testów: {success_count}/{len(loaded_models)} modeli przeszło testy")
        
    except ImportError as e:
        print(f"Nie można zaimportować testera modeli: {e}")
        print("Uruchom test_models.py w celu pełnego testowania.")

def main():
    """Główna funkcja skryptu."""
    parser = argparse.ArgumentParser(description="Zarządzanie modelami AI")
    
    subparsers = parser.add_subparsers(dest='command', help='Polecenie do wykonania')
    
    # Polecenie list
    list_parser = subparsers.add_parser('list', help='Wyświetl listę modeli')
    
    # Polecenie clean
    clean_parser = subparsers.add_parser('clean', help='Usuń uszkodzone modele')
    clean_parser.add_argument('--force', action='store_true', help='Usuń bez potwierdzenia')
    
    # Polecenie backup
    backup_parser = subparsers.add_parser('backup', help='Utwórz kopię zapasową modeli')
    
    # Polecenie test
    test_parser = subparsers.add_parser('test', help='Testuj modele')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_models()
    elif args.command == 'clean':
        clean_models(args.force)
    elif args.command == 'backup':
        backup_models()
    elif args.command == 'test':
        test_models()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
