"""
model_loader.py
--------------
Moduł do automatycznego ładowania i zarządzania modelami AI.

Funkcjonalności:
- Automatyczne wykrywanie i ładowanie modeli z folderu ai_models/
- Testowanie poprawności modeli przed załadowaniem
- Pełna obsługa błędów i logowanie
- Dostęp do modeli poprzez wygodne API
"""

import os
import logging
import traceback
import sys
import importlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Dodanie ścieżki do python_libs, jeśli jest potrzebna
python_libs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python_libs')
if python_libs_path not in sys.path:
    sys.path.append(python_libs_path)
    print(f"Dodano katalog python_libs do ścieżki Pythona.")

try:
    from python_libs.model_tester import ModelTester
except ImportError as e:
    print(f"Nie można zaimportować modułu model_loader: {e}")
    # Tworzymy prostą wersję zastępczą ModelTester
    class ModelTester:
        def __init__(self, models_path='ai_models', log_path='logs/model_loader.log'):
            self.models_path = models_path
            self.log_path = log_path
            self.loaded_models = []
            print(f"Utworzono zastępczą klasę ModelTester (brak właściwego modułu)")
            
        def run_tests(self):
            print("Uruchomienie zastępczej metody run_tests")
            # Tu powinniśmy zaimplementować podstawową funkcjonalność ładowania modeli
            # aby program działał nawet bez właściwego ModelTester
            try:
                # Próba bezpośredniego importu znanych modeli
                from ai_models.sentiment_ai import SentimentAnalyzer
                from ai_models.anomaly_detection import AnomalyDetector
                from ai_models.model_recognition import ModelRecognizer
                
                self.loaded_models = [
                    {"name": "sentimentanalyzer", "instance": SentimentAnalyzer(), 
                     "module": "sentiment_ai", "class": "SentimentAnalyzer"},
                    {"name": "anomalydetector", "instance": AnomalyDetector(), 
                     "module": "anomaly_detection", "class": "AnomalyDetector"},
                    {"name": "modelrecognizer", "instance": ModelRecognizer(), 
                     "module": "model_recognition", "class": "ModelRecognizer"}
                ]
            except Exception as e:
                print(f"Nie można załadować podstawowych modeli: {e}")
            
            return {"status": "zastępcza implementacja"}
            
        def get_loaded_models(self):
            return self.loaded_models

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/model_loader.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Klasa do ładowania i zarządzania modelami AI.
    
    Automatycznie wykrywa i ładuje modele z folderu ai_models/,
    zapewniając dostęp do nim poprzez wygodne API.
    """
    
    def __init__(self, models_path: str = 'ai_models'):
        """
        Inicjalizuje loader modeli.
        
        Args:
            models_path: Ścieżka do folderu z modelami
        """
        self.models_path = models_path
        self.models: Dict[str, Any] = {}
        self.model_tester = ModelTester(models_path=models_path, log_path='logs/model_loader.log')
        self.training_data = None
        
        logger.info(f"Inicjalizacja ModelLoader dla ścieżki: {models_path}")
    
    def load_models(self) -> Dict[str, Any]:
        """
        Ładuje wszystkie dostępne modele.
        
        Returns:
            Dict[str, Any]: Słownik załadowanych modeli
        """
        logger.info("Rozpoczęcie ładowania modeli...")
        
        try:
            # Uruchomienie testera modeli
            self.model_tester.run_tests()
            
            # Pobranie załadowanych modeli
            loaded_models = self.model_tester.get_loaded_models()
            
            # Przechowanie modeli w słowniku
            self.models = {}  # Reset słownika modeli
            for model_info in loaded_models:
                model_name = model_info['name']
                self.models[model_name] = model_info['instance']
                logger.info(f"Załadowano model: {model_name}")
            
            logger.info(f"Załadowano {len(self.models)} modeli")
            return self.models
            
        except Exception as e:
            logger.error(f"Błąd podczas ładowania modeli: {e}")
            traceback.print_exc()
            
            # Jeśli wystąpił błąd, spróbuj załadować podstawowe modele bezpośrednio
            try:
                self._load_fallback_models()
            except Exception as fallback_error:
                logger.error(f"Również nie udało się załadować modeli zapasowych: {fallback_error}")
                
            return self.models
    
    def _load_fallback_models(self):
        """
        Awaryjne ładowanie podstawowych modeli w przypadku problemów z testerem.
        """
        logger.info("Próba awaryjnego ładowania podstawowych modeli...")
        self.models = {}  # Reset słownika modeli
        
        try:
            # Bezpośredni import znanych modeli
            from ai_models.sentiment_ai import SentimentAnalyzer
            self.models["sentimentanalyzer"] = SentimentAnalyzer()
            logger.info("Załadowano model: sentimentanalyzer")
        except Exception as e:
            logger.error(f"Nie udało się załadować modelu SentimentAnalyzer: {e}")
            
        try:
            from ai_models.anomaly_detection import AnomalyDetector
            self.models["anomalydetector"] = AnomalyDetector()
            logger.info("Załadowano model: anomalydetector")
        except Exception as e:
            logger.error(f"Nie udało się załadować modelu AnomalyDetector: {e}")
            
        try:
            from ai_models.model_recognition import ModelRecognizer
            self.models["modelrecognizer"] = ModelRecognizer()
            logger.info("Załadowano model: modelrecognizer")
        except Exception as e:
            logger.error(f"Nie udało się załadować modelu ModelRecognizer: {e}")
            
        logger.info(f"Awaryjnie załadowano {len(self.models)} modeli")
            
    def predict(self, data=None):
        """
        Zbiera predykcje ze wszystkich załadowanych modeli.
        
        Args:
            data: Dane do predykcji (opcjonalne)
            
        Returns:
            Dict: Słownik z wynikami predykcji
        """
        predictions = {}
        
        # Jeśli modele nie są jeszcze załadowane, załaduj je
        if not self.models:
            self.load_models()
            
        # Zbierz predykcje z każdego modelu, który ma metodę predict
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    result = model.predict(data)
                    predictions[name] = result
                    logger.info(f"Wykonano predykcję dla modelu {name}")
                else:
                    predictions[name] = {"error": "Model nie ma metody predict"}
            except Exception as e:
                logger.error(f"Błąd podczas predykcji modelu {name}: {e}")
                predictions[name] = {"error": str(e)}
                
        return {
            "predictions": predictions,
            "models_count": len(self.models),
            "timestamp": datetime.now().isoformat()
        }
    
    def fit(self, data, model_names=None):
        """
        Trenuje wybrane lub wszystkie załadowane modele.
        
        Args:
            data: Dane treningowe
            model_names: Lista nazw modeli do wytrenowania (None dla wszystkich)
            
        Returns:
            Dict: Wyniki treningu
        """
        results = {}
        self.training_data = data
        
        # Jeśli modele nie są jeszcze załadowane, załaduj je
        if not self.models:
            self.load_models()
            
        # Ustal listę modeli do treningu
        models_to_train = model_names if model_names else self.models.keys()
        
        # Trenuj każdy wybrany model, który ma metodę fit
        for name in models_to_train:
            if name in self.models:
                model = self.models[name]
                try:
                    if hasattr(model, 'fit'):
                        success = model.fit(data)
                        results[name] = {"success": success}
                        logger.info(f"Trening modelu {name} {'zakończony sukcesem' if success else 'nieudany'}")
                    else:
                        results[name] = {"error": "Model nie ma metody fit"}
                except Exception as e:
                    logger.error(f"Błąd podczas treningu modelu {name}: {e}")
                    results[name] = {"error": str(e)}
            else:
                results[name] = {"error": "Model nie istnieje"}
                
        return {
            "training_results": results,
            "models_trained": len(results),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_model(self, name: str) -> Optional[Any]:
        """
        Pobiera model o podanej nazwie.
        
        Args:
            name: Nazwa modelu
            
        Returns:
            Optional[Any]: Model lub None, jeśli nie znaleziono
        """
        return self.models.get(name)
    
    def get_all_models(self) -> Dict[str, Any]:
        """
        Pobiera wszystkie załadowane modele.
        
        Returns:
            Dict[str, Any]: Słownik załadowanych modeli
        """
        return self.models
    
    def get_models_summary(self) -> List[Dict[str, str]]:
        """
        Zwraca podsumowanie załadowanych modeli.
        
        Returns:
            List[Dict[str, str]]: Lista informacji o modelach
        """
        return [
            {
                'name': name,
                'type': model.__class__.__name__,
                'status': 'Active',
                'has_predict': hasattr(model, 'predict'),
                'has_fit': hasattr(model, 'fit')
            }
            for name, model in self.models.items()
        ]


# Instancja globalna dla łatwego dostępu
model_loader = ModelLoader()

# Automatyczne ładowanie modeli przy imporcie
if __name__ != "__main__":
    model_loader.load_models()
