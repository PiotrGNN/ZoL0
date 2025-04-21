"""
Główny moduł autonomicznego handlu - integruje wszystkie komponenty autonomicznego systemu AI.

Ten moduł odpowiada za:
1. Koordynację pracy Meta-Agenta, autonomicznego zarządzania ryzykiem i systemu wyjaśnialności
2. Inicjalizację i rejestrację istniejących modeli AI w Meta-Agencie
3. Obsługę pełnego cyklu decyzyjnego: od analizy danych do wykonania zlecenia
4. Komunikację zwrotną do Dashboardu
"""

import os
import json
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import random
import base64
from io import BytesIO
import uuid
import time
import matplotlib.pyplot as plt
import importlib
import pickle

# Import modułów autonomicznego systemu
from ai_models.autonomous.meta_agent import MetaAgent
from ai_models.autonomous.risk_manager import AutonomousRiskManager
from ai_models.autonomous.explainable_ai import AIExplainer
from ai_models.autonomous.vector_memory import VectorMemory

# Import modułów do ładowania istniejących modeli AI
import importlib.util
import sys

logger = logging.getLogger(__name__)

class ModelWrapper:
    """
    Wrapper dla modeli AI, zapewniający standardowy interfejs predict().
    """
    
    def __init__(self, model_instance, model_type: str = "unknown", name: str = None):
        """
        Inicjalizacja wrappera modelu.
        
        Args:
            model_instance: Instancja modelu AI
            model_type: Typ modelu (ml, rule_based, etc.)
            name: Nazwa modelu
        """
        self.model = model_instance
        self.model_type = model_type
        self.name = name or getattr(model_instance, "__class__.__name__", "UnknownModel")
        
    def predict(self, market_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Standardowa metoda predykcji, dostosowująca różne interfejsy modeli.
        
        Args:
            market_data: Dane rynkowe
            context: Dodatkowy kontekst (np. stan portfela)
            
        Returns:
            Słownik z decyzją, pewnością i ewentualnie wyjaśnieniem
        """
        try:
            # Sprawdź dostępne metody predykcji
            if hasattr(self.model, "predict_with_explanation"):
                result = self.model.predict_with_explanation(market_data, context)
                
                if isinstance(result, tuple) and len(result) >= 3:
                    # Format: (decyzja, pewność, wyjaśnienie)
                    return {
                        "decision": result[0],
                        "confidence": result[1],
                        "explanation": result[2]
                    }
                    
                elif isinstance(result, dict):
                    # Już zwrócono słownik
                    return result
                    
            elif hasattr(self.model, "predict"):
                result = self.model.predict(market_data, context)
                
                if isinstance(result, tuple) and len(result) >= 2:
                    # Format: (decyzja, pewność)
                    return {
                        "decision": result[0],
                        "confidence": result[1],
                        "explanation": None
                    }
                elif isinstance(result, dict):
                    # Już zwrócono słownik
                    return result
                else:
                    # Tylko decyzja
                    return {
                        "decision": result,
                        "confidence": 0.5,  # Domyślna pewność
                        "explanation": None
                    }
            else:
                # Fallback - spróbuj wywołać model bezpośrednio
                result = self.model(market_data, context)
                return {
                    "decision": result,
                    "confidence": 0.5,
                    "explanation": None
                }
                
        except Exception as e:
            logger.error(f"Błąd podczas predykcji modelu {self.name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "decision": "ERROR",
                "confidence": 0.0,
                "explanation": f"Błąd: {str(e)}"
            }


class AutonomousTrading:
    """
    Główny system autonomicznego handlu, integrujący wszystkie komponenty AI.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicjalizacja systemu autonomicznego handlu.
        
        Args:
            config: Konfiguracja systemu
        """
        self.config = config or self._load_default_config()
        
        # Inicjalizacja komponentów
        self.meta_agent = MetaAgent(self.config.get('meta_agent', {}))
        self.risk_manager = AutonomousRiskManager(self.config.get('risk_management', {}))
        self.explainer = AIExplainer(self.config.get('explainer', {}))
        
        # Śledzenie stanu
        self.is_initialized = False
        self.models = {}  # Zarejestrowane modele
        self.last_decision = None
        self.last_explanation = None
        self.performance_metrics = {}
        
        # Ścieżki do modeli
        self.model_paths = self.config.get('model_paths', {
            'saved_models': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_models'),
            'ai_models': os.path.join(os.path.dirname(os.path.dirname(__file__)))
        })
        
        logger.info("Zainicjalizowano system autonomicznego handlu")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Ładuje domyślną konfigurację systemu.
        
        Returns:
            Słownik z domyślną konfiguracją
        """
        return {
            'meta_agent': {
                'decision_threshold': 0.65,
                'memory_size': 10000,
                'embedding_dim': 64
            },
            'risk_management': {
                'base_position_size': 0.05,
                'max_position_size': 0.2,
                'min_position_size': 0.01,
                'base_stop_loss_pct': 0.02,
                'base_take_profit_pct': 0.04,  # Dodany brakujący klucz
                'max_capital_at_risk': 0.1
            },
            'explainer': {
                'explanation_detail_level': 'medium',
                'include_charts': True,
                'max_factors_to_show': 5
            },
            'execution': {
                'auto_execute': False,  # Czy automatycznie wykonywać zlecenia
                'confirmation_threshold': 0.8  # Próg pewności dla automatycznego wykonania
            },
            'models': {
                'RandomForest': {'enabled': True, 'weight': 1.0},
                'ModelRecognizer': {'enabled': True, 'weight': 1.0},
                'SentimentAnalyzer': {'enabled': True, 'weight': 0.8},
                'AnomalyDetector': {'enabled': True, 'weight': 0.7}
            }
        }
    
    def initialize(self) -> bool:
        """
        Inicjalizuje system: ładuje modele i rejestruje je w Meta-Agencie.
        
        Returns:
            True jeśli inicjalizacja zakończyła się sukcesem, False w przeciwnym wypadku
        """
        try:
            # 1. Ładowanie i rejestracja modeli
            success = self._load_models()
            if not success:
                logger.error("Nie udało się załadować wymaganych modeli")
                return False
                
            # 2. Utworzenie katalogów na dane
            os.makedirs(os.path.dirname(os.path.dirname(__file__)), exist_ok=True)
            
            # 3. Wczytanie zapisanego stanu Meta-Agenta jeśli istnieje
            latest_state = self._find_latest_state_file('meta_agent_state')
            if latest_state:
                self.meta_agent.load_state(latest_state)
                logger.info(f"Wczytano wcześniejszy stan Meta-Agenta z {latest_state}")
                
            # 4. Wczytanie zapisanego stanu menedżera ryzyka jeśli istnieje
            latest_risk_state = self._find_latest_state_file('risk_manager_state')
            if latest_risk_state:
                self.risk_manager.load_state(latest_risk_state)
                logger.info(f"Wczytano wcześniejszy stan menedżera ryzyka z {latest_risk_state}")
                
            self.is_initialized = True
            logger.info("System autonomicznego handlu został zainicjalizowany")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji systemu autonomicznego handlu: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _find_latest_state_file(self, prefix: str) -> Optional[str]:
        """
        Znajduje najnowszy plik stanu.
        
        Args:
            prefix: Prefiks pliku stanu
            
        Returns:
            Ścieżka do najnowszego pliku stanu lub None jeśli nie znaleziono
        """
        search_paths = [
            os.path.join(os.path.dirname(__file__), 'memory_storage'),
            os.path.join(os.path.dirname(__file__), 'risk_data')
        ]
        
        latest_file = None
        latest_timestamp = 0
        
        for path in search_paths:
            if os.path.exists(path):
                for filename in os.listdir(path):
                    if filename.startswith(prefix):
                        try:
                            filepath = os.path.join(path, filename)
                            file_time = os.path.getmtime(filepath)
                            if file_time > latest_timestamp:
                                latest_timestamp = file_time
                                latest_file = filepath
                        except:
                            pass
                            
        return latest_file
    
    def _load_models(self) -> bool:
        """
        Ładuje istniejące modele AI i rejestruje je w Meta-Agencie.
        
        Returns:
            True jeśli załadowano przynajmniej jeden model, False w przeciwnym wypadku
        """
        # Wczytywanie modeli z konfiguracji
        enabled_models = [name for name, config in self.config.get('models', {}).items() 
                          if config.get('enabled', True)]
        
        if not enabled_models:
            logger.warning("Brak włączonych modeli w konfiguracji")
            return False
            
        # Licznik pomyślnie załadowanych modeli
        loaded_count = 0
        
        # 1. Załaduj modele pickle z katalogu saved_models
        if os.path.exists(self.model_paths['saved_models']):
            for filename in os.listdir(self.model_paths['saved_models']):
                if filename.endswith('.pkl'):
                    try:
                        model_name = os.path.splitext(filename)[0]
                        if any(enabled in model_name for enabled in enabled_models):
                            model_path = os.path.join(self.model_paths['saved_models'], filename)
                            
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                                
                            # Rejestracja w systemie
                            wrapper = ModelWrapper(model, 'pickle', model_name)
                            self._register_model(model_name, wrapper)
                            loaded_count += 1
                            
                            logger.info(f"Załadowano model {model_name} z pliku {filename}")
                    except Exception as e:
                        logger.error(f"Błąd podczas ładowania modelu {filename}: {str(e)}")
        
        # 2. Załaduj modele z modułów Python
        for model_name in enabled_models:
            try:
                # Sprawdź różne lokalizacje modelu
                for module_name in [f'ai_models.{model_name.lower()}', f'ai_models.{model_name}']:
                    try:
                        module = importlib.import_module(module_name)
                        
                        # Szukaj klasy modelu w module
                        for attr_name in dir(module):
                            if attr_name.lower() == model_name.lower() or attr_name == model_name:
                                model_class = getattr(module, attr_name)
                                if callable(model_class):
                                    model = model_class()
                                    
                                    # Rejestracja w systemie
                                    wrapper = ModelWrapper(model, 'python', model_name)
                                    self._register_model(model_name, wrapper)
                                    loaded_count += 1
                                    
                                    logger.info(f"Załadowano model {model_name} z modułu {module_name}")
                                    break
                        
                        break  # Przerwij pętlę jeśli udało się zaimportować moduł
                    except ImportError:
                        continue
            except Exception as e:
                logger.error(f"Błąd podczas ładowania modelu {model_name}: {str(e)}")
        
        logger.info(f"Załadowano {loaded_count} modeli AI")
        return loaded_count > 0
    
    def _register_model(self, model_name: str, model_wrapper: ModelWrapper) -> None:
        """
        Rejestruje model w systemie i Meta-Agencie.
        
        Args:
            model_name: Nazwa modelu
            model_wrapper: Wrapper modelu
        """
        self.models[model_name] = model_wrapper
        
        # Pobierz wagę z konfiguracji
        weight = 1.0
        if 'models' in self.config and model_name in self.config['models']:
            weight = self.config['models'][model_name].get('weight', 1.0)
            
        # Rejestracja w Meta-Agencie
        self.meta_agent.register_model(model_name, model_wrapper, weight)
    
    def process_market_data(self, market_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Przetwarza dane rynkowe i zwraca decyzję handlową wraz z wyjaśnieniem.
        
        Args:
            market_data: Dane rynkowe (ceny, wolumeny, wskaźniki itp.)
            context: Dodatkowy kontekst (stan portfela, bieżące pozycje itp.)
            
        Returns:
            Słownik z decyzją, parametrami, wyjaśnieniem i szczegółami
        """
        if not self.is_initialized:
            self.initialize()
            
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'System autonomicznego handlu nie został zainicjalizowany'
            }
            
        # 1. Aktualizacja warunków rynkowych w menedżerze ryzyka
        risk_parameters = self.risk_manager.update_market_conditions(market_data)
        
        # 2. Uzyskaj decyzję z Meta-Agenta
        try:
            meta_decision = self.meta_agent.get_decision(market_data, context)
            self.last_decision = meta_decision
        except Exception as e:
            logger.error(f"Błąd podczas uzyskiwania decyzji z Meta-Agenta: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Błąd podczas analizy danych: {str(e)}'
            }
        
        # 3. Oblicz parametry pozycji na podstawie decyzji i ryzyka
        portfolio_value = context.get('portfolio_value', 10000) if context else 10000
        signal_strength = meta_decision.get('confidence', 0.5)
        position_config = self.risk_manager.calculate_position_size(portfolio_value, signal_strength)
        
        # 4. Wyekstrahuj kluczowe czynniki wpływające na decyzję
        model_outputs = {
            name: model.model.predict(market_data, context) if hasattr(model.model, 'predict') else {'decision': 'UNKNOWN'}
            for name, model in self.models.items()
        }
        
        factors = self.explainer.extract_key_factors(market_data, model_outputs)
        
        # 5. Wygeneruj wyjaśnienie decyzji
        explanation_data = {
            'decision': meta_decision.get('decision'),
            'confidence': meta_decision.get('confidence', 0.0),
            'factors': factors,
            'explanation': meta_decision.get('explanation', ''),
            'details': meta_decision.get('details', {})
        }
        
        explanation = self.explainer.explain_decision(explanation_data)
        self.last_explanation = explanation
        
        # 6. Sprawdź, czy decyzja powinna być wykonana automatycznie
        auto_execute = self.config.get('execution', {}).get('auto_execute', False)
        confirmation_threshold = self.config.get('execution', {}).get('confirmation_threshold', 0.8)
        needs_confirmation = meta_decision.get('confidence', 0) < confirmation_threshold
        
        # 7. Przygotuj pełną odpowiedź
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'decision': {
                'type': meta_decision.get('decision'),
                'confidence': meta_decision.get('confidence', 0.0),
                'needs_confirmation': needs_confirmation or not auto_execute
            },
            'position': position_config,
            'risk': risk_parameters,
            'explanation': {
                'text': explanation.get('text_explanation', ''),
                'html': explanation.get('html_explanation', ''),
                'visuals': explanation.get('visual_components', {})
            },
            'details': meta_decision.get('details', {}),
            'models_used': list(self.models.keys())
        }
        
        logger.info(f"Wygenerowano decyzję: {meta_decision.get('decision')} (pewność: {meta_decision.get('confidence', 0.0):.2f})")
        return response
    
    def update_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Aktualizuje system o wynik transakcji.
        
        Args:
            trade_result: Wynik transakcji (zawiera profit_pct, max_drawdown, itp.)
        """
        # Aktualizuj menedżera ryzyka
        self.risk_manager.update_trade_result(trade_result)
        
        # Zapisz wynik w pamięci wektorowej Meta-Agenta
        if self.last_decision:
            # Znajdź ostatni wpis w pamięci związany z tą decyzją i zaktualizuj go
            memory_context = {
                'market_data': self.last_decision.get('context', {}).get('market_data', {}),
                'model_decisions': self.last_decision.get('details', {}).get('model_decisions', {}),
                'final_decision': self.last_decision.get('decision')
            }
            
            self.meta_agent.memory.store(memory_context, trade_result)
            
        # Aktualizuj metryki wydajności modeli
        decision_type = self.last_decision.get('decision') if self.last_decision else None
        profit_pct = trade_result.get('profit_pct', 0)
        
        # Oblicz pozytywny/negatywny wynik dla różnych typów decyzji
        performance = 0.5  # Neutralny wynik domyślnie
        
        if decision_type in ['BUY', 'LONG'] and profit_pct > 0:
            # Dobra decyzja kupna
            performance = 0.5 + min(0.5, profit_pct / 10)
        elif decision_type in ['BUY', 'LONG'] and profit_pct < 0:
            # Zła decyzja kupna
            performance = 0.5 - min(0.5, abs(profit_pct) / 10)
        elif decision_type in ['SELL', 'SHORT'] and profit_pct > 0:
            # Dobra decyzja sprzedaży
            performance = 0.5 + min(0.5, profit_pct / 10)
        elif decision_type in ['SELL', 'SHORT'] and profit_pct < 0:
            # Zła decyzja sprzedaży
            performance = 0.5 - min(0.5, abs(profit_pct) / 10)
            
        # Aktualizuj wydajność modeli, które przyczyniły się do tej decyzji
        if self.last_decision and 'details' in self.last_decision:
            model_decisions = self.last_decision['details'].get('model_decisions', {})
            model_confidences = self.last_decision['details'].get('model_confidences', {})
            
            for model_name, model_decision in model_decisions.items():
                # Normalizuj decyzję modelu do formatu tekstowego
                if isinstance(model_decision, (int, float)):
                    model_decision_type = 'BUY' if model_decision > 0 else ('SELL' if model_decision < 0 else 'HOLD')
                else:
                    model_decision_type = str(model_decision)
                    
                # Jeśli model podjął decyzję zgodną z finalną, nagradzamy go
                if model_decision_type == decision_type:
                    model_performance = performance
                else:
                    model_performance = 1.0 - performance  # Odwróć wynik dla przeciwnych decyzji
                    
                # Aktualizuj wydajność modelu w Meta-Agencie
                self.meta_agent.update_model_performance(model_name, model_performance)
                
                # Zapisz metrykę do lokalnego śledzenia
                if model_name not in self.performance_metrics:
                    self.performance_metrics[model_name] = []
                    
                self.performance_metrics[model_name].append({
                    'timestamp': datetime.now().isoformat(),
                    'performance': model_performance,
                    'trade_result': profit_pct
                })
                
        # Okresowe zapisywanie stanu
        self._periodic_save_state()
        
        logger.info(f"Zaktualizowano system o wynik transakcji: {profit_pct:.2f}%")
        
    def _periodic_save_state(self) -> None:
        """Okresowo zapisuj stan systemu."""
        # Co 10 transakcji zapisuj stan całego systemu
        total_trades = sum(len(metrics) for metrics in self.performance_metrics.values())
        
        if total_trades % 10 == 0:
            self.save_state()
            
    def save_state(self) -> None:
        """Zapisuje stan całego systemu autonomicznego handlu."""
        # Zapisz stan Meta-Agenta
        self.meta_agent.save_state()
        
        # Zapisz stan menedżera ryzyka
        self.risk_manager.save_state()
        
        # Zapisz metryki wydajności modeli
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(
            os.path.dirname(__file__),
            f"model_performance_{timestamp}.json"
        )
        
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, default=str)
            
        logger.info("Zapisano stan systemu autonomicznego handlu")
            
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generuje raport wydajności modeli AI.
        
        Returns:
            Słownik z raportem wydajności
        """
        if not self.performance_metrics:
            return {
                'status': 'no_data',
                'message': 'Brak danych o wydajności modeli'
            }
            
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'overall': {
                'best_model': None,
                'worst_model': None,
                'avg_performance': 0.0
            }
        }
        
        # Analizuj wydajność każdego modelu
        best_avg = 0.0
        worst_avg = 1.0
        total_performance = 0.0
        models_count = 0
        
        for model_name, metrics in self.performance_metrics.items():
            if not metrics:
                continue
                
            # Oblicz statystyki
            performances = [m['performance'] for m in metrics]
            avg_performance = sum(performances) / len(performances)
            recent_performance = sum(performances[-5:]) / min(5, len(performances)) if performances else 0
            profit_correlation = self._calculate_correlation(
                [m['performance'] for m in metrics],
                [m['trade_result'] for m in metrics]
            )
            
            report['models'][model_name] = {
                'avg_performance': avg_performance,
                'recent_performance': recent_performance,
                'samples_count': len(metrics),
                'profit_correlation': profit_correlation
            }
            
            # Aktualizuj globalną statystykę
            if avg_performance > best_avg:
                best_avg = avg_performance
                report['overall']['best_model'] = model_name
                
            if avg_performance < worst_avg:
                worst_avg = avg_performance
                report['overall']['worst_model'] = model_name
                
            total_performance += avg_performance
            models_count += 1
            
        # Ustaw średnią wydajność
        if models_count > 0:
            report['overall']['avg_performance'] = total_performance / models_count
            
        return report
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Oblicza korelację między dwoma listami liczb.
        
        Args:
            x: Pierwsza lista wartości
            y: Druga lista wartości
            
        Returns:
            Współczynnik korelacji
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0.0
            
    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Generuje wizualizacje wydajności systemu.
        
        Returns:
            Słownik z wizualizacjami
        """
        visuals = {}
        
        # 1. Wizualizacja wag modeli
        try:
            model_weights_fig = self.meta_agent.visualize_model_weights()
            if model_weights_fig:
                buffer = BytesIO()
                model_weights_fig.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                visuals['model_weights'] = {
                    'type': 'image',
                    'format': 'base64',
                    'data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                    'alt': 'Wagi modeli w Meta-Agencie'
                }
                plt.close(model_weights_fig)
        except Exception as e:
            logger.error(f"Błąd podczas generowania wizualizacji wag modeli: {str(e)}")
            
        # 2. Wizualizacja historii decyzji
        try:
            decision_history_fig = self.meta_agent.visualize_decision_history()
            if decision_history_fig:
                buffer = BytesIO()
                decision_history_fig.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                visuals['decision_history'] = {
                    'type': 'image',
                    'format': 'base64',
                    'data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                    'alt': 'Historia decyzji Meta-Agenta'
                }
                plt.close(decision_history_fig)
        except Exception as e:
            logger.error(f"Błąd podczas generowania wizualizacji historii decyzji: {str(e)}")
            
        # 3. Wizualizacja parametrów ryzyka
        try:
            risk_params_fig = self.risk_manager.visualize_risk_parameters()
            if risk_params_fig:
                buffer = BytesIO()
                risk_params_fig.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                visuals['risk_parameters'] = {
                    'type': 'image',
                    'format': 'base64',
                    'data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                    'alt': 'Parametry ryzyka w czasie'
                }
                plt.close(risk_params_fig)
        except Exception as e:
            logger.error(f"Błąd podczas generowania wizualizacji parametrów ryzyka: {str(e)}")
            
        # 4. Wizualizacja wyników handlu
        try:
            trade_perf_fig = self.risk_manager.visualize_trade_performance()
            if trade_perf_fig:
                buffer = BytesIO()
                trade_perf_fig.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                visuals['trade_performance'] = {
                    'type': 'image',
                    'format': 'base64',
                    'data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                    'alt': 'Wyniki handlu'
                }
                plt.close(trade_perf_fig)
        except Exception as e:
            logger.error(f"Błąd podczas generowania wizualizacji wyników handlu: {str(e)}")
            
        return visuals