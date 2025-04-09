
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
