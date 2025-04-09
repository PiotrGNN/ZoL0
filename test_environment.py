
"""
test_environment.py
------------------
Skrypt do testowania środowiska i dostępności wymaganych bibliotek.
"""

import sys
import importlib
import os
from typing import List, Dict

def check_required_libs() -> Dict[str, bool]:
    """
    Sprawdza dostępność wymaganych bibliotek.
    
    Returns:
        Dict[str, bool]: Słownik nazw bibliotek i statusów dostępności
    """
    required_libs = [
        "numpy", "pandas", "sklearn", "xgboost", 
        "matplotlib", "flask", "dotenv", "tensorflow"
    ]
    results = {}
    
    for lib in required_libs:
        try:
            importlib.import_module(lib)
            results[lib] = True
            print(f"✅ Biblioteka {lib} jest dostępna")
        except ImportError:
            results[lib] = False
            print(f"❌ Biblioteka {lib} NIE jest dostępna")
    
    return results

def check_ai_models() -> Dict[str, bool]:
    """
    Sprawdza dostępność modeli AI.
    
    Returns:
        Dict[str, bool]: Słownik nazw modeli i statusów dostępności
    """
    ai_models = [
        "SentimentAnalyzer", "AnomalyDetector", "ModelRecognizer"
    ]
    results = {}
    
    try:
        import ai_models
        
        for model_name in ai_models:
            try:
                # Sprawdź czy model jest dostępny w available_models
                if hasattr(ai_models, 'get_available_models'):
                    available = ai_models.get_available_models()
                    model_key = model_name.lower().replace('analyzer', '_analyzer').replace('detector', '_detector').replace('recognizer', '_recognizer')
                    
                    if model_key in available:
                        results[model_name] = True
                        print(f"✅ Model {model_name} jest dostępny")
                    else:
                        # Spróbuj bezpośredni import
                        try:
                            module_name = model_name.lower()
                            if 'analyzer' in module_name:
                                from ai_models.sentiment_ai import SentimentAnalyzer
                                results[model_name] = True
                                print(f"✅ Model {model_name} jest dostępny (bezpośredni import)")
                            elif 'detector' in module_name:
                                from ai_models.anomaly_detection import AnomalyDetector
                                results[model_name] = True
                                print(f"✅ Model {model_name} jest dostępny (bezpośredni import)")
                            elif 'recognizer' in module_name:
                                from ai_models.model_recognition import ModelRecognizer
                                results[model_name] = True
                                print(f"✅ Model {model_name} jest dostępny (bezpośredni import)")
                            else:
                                results[model_name] = False
                                print(f"❌ Model {model_name} NIE jest dostępny")
                        except ImportError:
                            results[model_name] = False
                            print(f"❌ Model {model_name} NIE jest dostępny")
                else:
                    # Jeśli get_available_models() nie istnieje
                    results[model_name] = False
                    print(f"❓ Nie można sprawdzić dostępności modelu {model_name}")
            except Exception as e:
                results[model_name] = False
                print(f"❌ Błąd podczas sprawdzania modelu {model_name}: {e}")
    except ImportError:
        print("❌ Nie można zaimportować pakietu ai_models")
        for model_name in ai_models:
            results[model_name] = False
    
    return results

def check_model_methods():
    """
    Sprawdza dostępność metod fit() i predict() w modelach.
    """
    print("\n📊 Sprawdzanie metod w modelach AI:")
    
    try:
        # SentimentAnalyzer
        try:
            from ai_models.sentiment_ai import SentimentAnalyzer
            model = SentimentAnalyzer()
            has_predict = hasattr(model, 'predict')
            has_fit = hasattr(model, 'fit')
            print(f"SentimentAnalyzer: predict={has_predict}, fit={has_fit}")
            
            # Test predict
            if has_predict:
                result = model.predict("Markets look promising today")
                print(f"  - Test predict(): {result['analysis'] if isinstance(result, dict) and 'analysis' in result else 'Błąd'}")
        except Exception as e:
            print(f"❌ Błąd podczas testowania SentimentAnalyzer: {e}")
        
        # AnomalyDetector
        try:
            from ai_models.anomaly_detection import AnomalyDetector
            model = AnomalyDetector()
            has_predict = hasattr(model, 'predict')
            has_fit = hasattr(model, 'fit')
            print(f"AnomalyDetector: predict={has_predict}, fit={has_fit}")
            
            # Test detect
            test_data = [{"value": 1.0}, {"value": 2.0}, {"value": 100.0}]
            result = model.detect(test_data)
            print(f"  - Test detect(): {len(result)} anomalii wykrytych")
        except Exception as e:
            print(f"❌ Błąd podczas testowania AnomalyDetector: {e}")
        
        # ModelRecognizer
        try:
            from ai_models.model_recognition import ModelRecognizer
            model = ModelRecognizer()
            has_predict = hasattr(model, 'predict')
            has_fit = hasattr(model, 'fit')
            print(f"ModelRecognizer: predict={has_predict}, fit={has_fit}")
        except Exception as e:
            print(f"❌ Błąd podczas testowania ModelRecognizer: {e}")
    
    except Exception as e:
        print(f"❌ Ogólny błąd podczas testowania metod: {e}")

def main():
    """
    Główna funkcja testująca środowisko.
    """
    print("🔍 Sprawdzanie środowiska Python...")
    print(f"Python: {sys.version}")
    print(f"Ścieżka wykonywalna: {sys.executable}")
    print(f"Katalog bieżący: {os.getcwd()}")
    
    print("\n📚 Sprawdzanie wymaganych bibliotek...")
    lib_results = check_required_libs()
    
    print("\n🤖 Sprawdzanie dostępności modeli AI...")
    model_results = check_ai_models()
    
    # Sprawdzenie metod w modelach
    check_model_methods()
    
    # Podsumowanie
    print("\n📋 Podsumowanie:")
    libs_ok = all(lib_results.values())
    models_ok = all(model_results.values())
    
    if libs_ok:
        print("✅ Wszystkie wymagane biblioteki są dostępne")
    else:
        print("⚠️ Brakuje niektórych bibliotek!")
        for lib, available in lib_results.items():
            if not available:
                print(f"  - {lib}")
    
    if models_ok:
        print("✅ Wszystkie modele AI są dostępne")
    else:
        print("⚠️ Brakuje niektórych modeli AI!")
        for model, available in model_results.items():
            if not available:
                print(f"  - {model}")
    
    # Wypisanie statystyk, jeśli istnieją
    if 'stats' in locals():
        print("\n📊 Statystyki:")
        if 'py_files_scanned' in stats:
            print(f"- Przeskanowano plików .py: {stats['py_files_scanned']}")
        else:
            print("- Brak danych o liczbie przeskanowanych plików .py")
    
    if libs_ok and models_ok:
        print("\n🎉 Środowisko jest gotowe do pracy!")
        returnn 0
    else:
        print("\n⚠️ Środowisko wymaga konfiguracji!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
