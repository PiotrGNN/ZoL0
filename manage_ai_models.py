
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
