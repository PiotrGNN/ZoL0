#!/usr/bin/env python3
"""
manage_ai_models.py - Narzędzie do zarządzania modelami AI w środowisku lokalnym
"""

import os
import sys
import argparse
import logging
import time
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional

# Dodanie ścieżek do PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
python_libs_dir = os.path.join(current_dir, "python_libs")
sys.path.append(python_libs_dir)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "ai_models_management.log")) if os.path.exists("logs") else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Upewnij się, że katalog logów istnieje
os.makedirs('logs', exist_ok=True)

def ensure_model_dirs():
    """Tworzy wymagane katalogi dla modeli, jeśli nie istnieją."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

def list_models():
    """Wyświetla listę dostępnych modeli AI."""
    try:
        # Próba importu funkcji z model_utils
        try:
            from ai_models.model_utils import list_available_models
            models = list_available_models()
            if not models:
                logger.info("Nie znaleziono żadnych modeli AI.")
                return

            # Nagłówki tabeli
            headers = ["Nazwa", "Typ", "Rozmiar", "Ostatnia modyfikacja", "Accuracy"]
            rows = []

            for model in models:
                # Pobierz typ modelu z metadanych, jeśli dostępne
                model_type = model.get("model_type", "Nieznany")

                # Pobierz accuracy z metadanych, jeśli dostępne
                accuracy = model.get("accuracy", "N/A")
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

        except ImportError:
            # Fallback - skanowanie plików bezpośrednio
            scan_models()
    except Exception as e:
        logger.error(f"Błąd podczas wyświetlania modeli: {e}")

def scan_models():
    """Skanuje katalogi i wyświetla listę modeli - wersja fallback."""
    try:
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
                        metadata_info = f", Dokładność: {metadata['accuracy']:.2f}%"
                    elif isinstance(metadata, dict) and 'version' in metadata:
                        metadata_info = f", Wersja: {metadata['version']}"
                except json.JSONDecodeError:
                    metadata_info = " (Błąd odczytu metadanych)"

            print(f"- {model_name:<20} | {size_kb:6.1f} KB | {modified_time.strftime('%Y-%m-%d %H:%M:%S')}{metadata_info}")

        # Sprawdź również katalog ai_models
        ai_models_dir = 'ai_models'
        if os.path.exists(ai_models_dir):
            py_models = [f for f in os.listdir(ai_models_dir) if f.endswith('.py') and not f.startswith('__')]
            if py_models:
                print(f"\nZnaleziono {len(py_models)} potencjalnych klas modeli w katalogu {ai_models_dir}:")
                for py_file in py_models:
                    print(f"- {py_file}")

    except Exception as e:
        logger.error(f"Błąd podczas skanowania modeli: {e}")

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
        # Upewnij się, że katalog python_libs jest w ścieżce Pythona
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Upewnij się, że katalog logów istnieje
        os.makedirs("logs", exist_ok=True)
        
        # Spróbuj zaimportować ModelTester
        try:
            from python_libs.model_tester import ModelTester
            print("✅ Zaimportowano ModelTester z python_libs")
        except ImportError as e:
            print(f"❌ Błąd importu ModelTester: {e}")
            print("Tworzenie alternatywnej implementacji ModelTester...")
            
            # Implementacja klasy ModelTester (uproszczona wersja)
            class ModelTester:
                def __init__(self, models_path='ai_models', log_path='logs/model_tests.log'):
                    self.models_path = models_path
                    self.log_path = log_path
                    self.loaded_models = []
                    print(f"ModelTester zainicjalizowany (wersja zastępcza)")
                    
                def load_models(self):
                    print("Próba ładowania modeli...")
                    # Tutaj umieścilibyśmy kod do ładowania modeli
                    return []
                    
                def run_tests(self):
                    print("Wykonywanie testów...")
                    return {}

        # Inicjalizacja testera modeli
        tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')

        # Ładowanie modeli
        loaded_models = tester.load_models()

        print(f"\nZaładowano {len(loaded_models)} modeli\n")

        # Przeprowadź testy
        results = tester.run_tests()

        print("\nWyniki testów:\n")

        # Formatowanie wyników w postaci tabeli
        headers = ["Model", "Status", "Accuracy", "Predict", "Fit", "Czas (s)"]
        rows = []

        for model_name, result in results.items():
            status = "✅ Sukces" if result.get("success", False) else "❌ Błąd"
            accuracy = result.get("accuracy", "N/A")
            if isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.2f}%"

            predict = "✓" if result.get("predict_successful", False) else "✗"
            fit = "✓" if result.get("fit_successful", False) else "✗"

            predict_time = result.get("predict_time", 0)
            fit_time = result.get("fit_time", 0)
            total_time = predict_time + fit_time

            if isinstance(total_time, (int, float)):
                time_str = f"{total_time:.3f}"
            else:
                time_str = "N/A"

            rows.append([
                model_name,
                status,
                accuracy,
                predict,
                fit,
                time_str
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

        print(f"\nPodsumowanie: {sum(1 for r in results.values() if r.get('success', False))}/{len(results)} modeli przeszło testy")

    except ImportError as e:
        print(f"Nie można zaimportować testera modeli: {e}")
        print("Zalecane rozwiązanie: Uruchom skrypt bezpośrednio w katalogu projektu lub upewnij się, że python_libs jest dostępne w PYTHONPATH.")

def show_model_details(model_name):
    """Wyświetla szczegółowe informacje o modelu."""
    try:
        from ai_models.model_utils import load_model

        # Ładowanie modelu i metadanych
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
    except ImportError as e:
        # Fallback - użyj pickle bezpośrednio
        try:
            import pickle
            import json

            model_path = os.path.join("models", f"{model_name}_model.pkl")
            metadata_path = os.path.join("models", f"{model_name}_metadata.json")

            if not os.path.exists(model_path):
                print(f"❌ Model {model_name} nie istnieje")
                return

            # Ładowanie modelu
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            print(f"\n📋 Szczegóły modelu {model_name}:")
            print(f"Typ: {type(model).__name__}")
            print(f"Moduł: {type(model).__module__}")

            # Ładowanie metadanych
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    print("\nMetadane:")
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")
                except:
                    print("\nBłąd podczas ładowania metadanych.")
        except Exception as e2:
            print(f"❌ Błąd podczas wyświetlania szczegółów modelu: {e2}")

def main():
    """Funkcja główna."""
    # Upewnij się, że katalogi istnieją
    ensure_model_dirs()

    parser = argparse.ArgumentParser(description="Narzędzie do zarządzania modelami AI")
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

    # Polecenie details
    details_parser = subparsers.add_parser('details', help='Wyświetl szczegóły modelu')
    details_parser.add_argument('model', help='Nazwa modelu')

    # Dodaj argumenty dla wszystkich poleceń
    parser.add_argument("--list", action="store_true", help="Wyświetl listę dostępnych modeli")
    parser.add_argument("--test", action="store_true", help="Uruchom testy modeli")
    parser.add_argument("--model", type=str, help="Określony model do przetestowania")
    parser.add_argument("--clean", action="store_true", help="Usuń uszkodzone modele")
    parser.add_argument("--backup", action="store_true", help="Utwórz kopię zapasową modeli")
    parser.add_argument("--details", type=str, help="Wyświetl szczegóły modelu")
    parser.add_argument("--force", action="store_true", help="Wykonaj akcje bez potwierdzenia")

    args = parser.parse_args()

    # Obsługa poleceń w trybie subparsera
    if args.command == 'list':
        list_models()
    elif args.command == 'clean':
        clean_models(args.force)
    elif args.command == 'backup':
        backup_models()
    elif args.command == 'test':
        test_models()
    elif args.command == 'details':
        show_model_details(args.model)

    # Obsługa argumentów bezpośrednich (dla kompatybilności wstecznej)
    elif args.list:
        list_models()
    elif args.test:
        test_model(args.model)
    elif args.clean:
        clean_models(args.force)
    elif args.backup:
        backup_models()
    elif args.details:
        show_model_details(args.details)
    else:
        list_models()  # Domyślnie wyświetl listę modeli

def test_model(model_name=None):
    """Testuje określony model AI."""
    try:
        from python_libs.model_tester import ModelTester
        from ai_models.model_utils import save_model, load_model
        import os

        tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')

        if model_name:
            logger.info(f"Testowanie modelu: {model_name}")
            
            # Sprawdź czy model ma już zapisany plik .pkl
            model_path = os.path.join("models", f"{model_name}_model.pkl")
            if os.path.exists(model_path) and not force_retrain:
                logger.info(f"Znaleziono zapisany model {model_name}. Wczytywanie z pliku...")
                model, metadata, success = load_model(model_name)
                if success:
                    logger.info(f"Model {model_name} załadowany pomyślnie. Metadane: {metadata}")
                else:
                    logger.warning(f"Nie udało się załadować modelu {model_name}. Przeprowadzam nowy trening.")
                    result = tester.test_model_by_name(model_name)
                    
                    # Zapisz model po treningu
                    if result["success"] and "instance" in result:
                        save_model(result["instance"], model_name, {
                            "accuracy": result.get("accuracy", 0),
                            "trained_at": datetime.now().isoformat(),
                            "test_result": "success"
                        })
                        logger.info(f"Model {model_name} zapisany po treningu")
            else:
                # Trenuj model jeśli nie istnieje
                result = tester.test_model_by_name(model_name)
                if result["success"]:
                    logger.info(f"Test zakończony pomyślnie. Dokładność: {result.get('accuracy', 'N/A')}%")
                    
                    # Zapisz model po treningu
                    if "instance" in result:
                        save_model(result["instance"], model_name, {
                            "accuracy": result.get("accuracy", 0),
                            "trained_at": datetime.now().isoformat(),
                            "test_result": "success"
                        })
                        logger.info(f"Model {model_name} zapisany po treningu")
                else:
                    logger.error(f"Test nie powiódł się: {result.get('error', 'Nieznany błąd')}")
        else:
            logger.info("Testowanie wszystkich modeli...")
            results = tester.run_tests()
            logger.info(f"Testy zakończone. Wyniki: {len(results)} modeli przetestowano.")

            # Wyświetlanie wyników i zapis modeli
            for model_name, result in results.items():
                if isinstance(result, dict):  # Upewnij się, że result to słownik
                    status = "✅ Sukces" if result.get("success", False) else "❌ Błąd"
                    logger.info(f"{model_name}: {status} - Dokładność: {result.get('accuracy', 'N/A')}%")
                    
                    # Zapisz model po pomyślnych testach
                    if result.get("success", False) and "instance" in result:
                        save_model(result["instance"], model_name, {
                            "accuracy": result.get("accuracy", 0),
                            "trained_at": datetime.now().isoformat(),
                            "test_result": "success"
                        })
                        logger.info(f"Model {model_name} zapisany po pomyślnych testach")

    except Exception as e:
        logger.error(f"Błąd podczas testowania modeli: {e}")

if __name__ == "__main__":
    logger.info("Narzędzie zarządzania modelami AI")
    main()
import os
import shutil
import logging
import pickle
import datetime
import argparse
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def fix_corrupted_models():
    """Usuwa lub regeneruje uszkodzone modele."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        logging.error("Katalog modeli nie istnieje!")
        return
    
    # Tworzenie katalogu na uszkodzone modele
    corrupted_dir = os.path.join(models_dir, 'corrupted')
    os.makedirs(corrupted_dir, exist_ok=True)
    
    # Skanowanie plików modeli
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    fixed_count = 0
    moved_count = 0
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            # Sprawdź czy plik jest pusty
            if os.path.getsize(model_path) == 0:
                logging.warning(f"Model {model_file} jest pusty. Przenoszenie do {corrupted_dir}")
                shutil.move(model_path, os.path.join(corrupted_dir, model_file))
                moved_count += 1
                continue
                
            # Sprawdź czy model można załadować
            with open(model_path, 'rb') as f:
                try:
                    model = pickle.load(f)
                    # Model załadowany poprawnie
                    continue
                except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                    logging.error(f"Model {model_file} jest uszkodzony: {str(e)}")
                    
                    # Sprawdź, czy istnieje kopia zapasowa
                    backup_file = model_file.replace('.pkl', '_backup.pkl')
                    backup_path = os.path.join(models_dir, backup_file)
                    
                    if os.path.exists(backup_path):
                        # Przywróć z kopii zapasowej
                        shutil.copy(backup_path, model_path)
                        logging.info(f"Przywrócono model {model_file} z kopii zapasowej")
                        fixed_count += 1
                    else:
                        # Przenieś uszkodzony model do folderu corrupted
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        corrupt_name = f"{model_file}.corrupted.{timestamp}"
                        shutil.move(model_path, os.path.join(corrupted_dir, corrupt_name))
                        logging.info(f"Przeniesiono uszkodzony model {model_file} do {corrupted_dir}/{corrupt_name}")
                        moved_count += 1
                        
                        # Sprawdź czy istnieje alternatywny model w saved_models
                        alt_model_name = model_file.replace('model.pkl', '')
                        if os.path.exists('saved_models'):
                            alt_models = [f for f in os.listdir('saved_models') if alt_model_name in f and f.endswith('.pkl')]
                            if alt_models:
                                # Użyj najnowszego alternatywnego modelu
                                alt_models.sort(key=lambda x: os.path.getmtime(os.path.join('saved_models', x)), reverse=True)
                                src_path = os.path.join('saved_models', alt_models[0])
                                shutil.copy(src_path, model_path)
                                logging.info(f"Zastąpiono uszkodzony model {model_file} alternatywnym: {alt_models[0]}")
                                fixed_count += 1
        except Exception as e:
            logging.error(f"Błąd podczas sprawdzania modelu {model_file}: {str(e)}")
    
    logging.info(f"Naprawa zakończona: {fixed_count} modeli naprawionych, {moved_count} modeli przeniesionych")
    return fixed_count, moved_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zarządzanie modelami AI")
    parser.add_argument('--fix', action='store_true', help='Napraw uszkodzone modele')
    
    # Dodaj pozostałe opcje argumentów jeśli istnieją w oryginalnym pliku
    
    args = parser.parse_args()
    
    if args.fix:
        fix_corrupted_models()
