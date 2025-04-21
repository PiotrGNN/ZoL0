"""
Moduł do zarządzania modelami AI i ich parametrami.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Klasa przechowująca metryki modelu."""
    accuracy: float
    loss: float
    validation_accuracy: float
    validation_loss: float
    training_duration: int
    epochs_completed: int
    dataset_version: str
    metrics: Dict[str, float]

class ModelManager:
    def __init__(self, db_path: str = 'users.db'):
        """Inicjalizuje menedżera modeli."""
        self.db_path = db_path
        self.default_save_path = 'saved_models'
        os.makedirs(self.default_save_path, exist_ok=True)
        
    def train_model(self, model_id: str, training_data: pd.DataFrame, 
                   parameters: Dict[str, Any]) -> ModelMetrics:
        """
        Trenuje model AI z zadanymi parametrami.
        
        Args:
            model_id: Identyfikator modelu
            training_data: Dane treningowe
            parameters: Parametry modelu
            
        Returns:
            ModelMetrics: Metryki wydajności modelu
        """
        try:
            start_time = datetime.now()
            
            # Przygotuj dane
            X = training_data.drop(columns=['target'])
            y = training_data['target']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            
            # Załaduj model
            model = self._load_model(model_id)
            if model is None:
                raise ValueError(f"Nie znaleziono modelu o ID {model_id}")
            
            # Ustaw parametry
            self._set_model_parameters(model, parameters)
            
            # Trenuj model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=parameters.get('epochs', 10),
                batch_size=parameters.get('batch_size', 32),
                verbose=1
            )
            
            # Zbierz metryki
            metrics = ModelMetrics(
                accuracy=history.history['accuracy'][-1],
                loss=history.history['loss'][-1],
                validation_accuracy=history.history['val_accuracy'][-1],
                validation_loss=history.history['val_loss'][-1],
                training_duration=int((datetime.now() - start_time).total_seconds()),
                epochs_completed=len(history.history['loss']),
                dataset_version=parameters.get('dataset_version', 'unknown'),
                metrics={
                    'max_accuracy': max(history.history['accuracy']),
                    'min_loss': min(history.history['loss']),
                    'max_val_accuracy': max(history.history['val_accuracy']),
                    'min_val_loss': min(history.history['val_loss'])
                }
            )
            
            # Zapisz model i metryki
            self._save_model(model, model_id, parameters)
            self._save_training_history(model_id, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu {model_id}: {e}")
            raise
    
    def _load_model(self, model_id: str):
        """Ładuje model z dysku."""
        try:
            model_path = os.path.join(self.default_save_path, f"{model_id}.pkl")
            if os.path.exists(model_path):
                return joblib.load(model_path)
            return None
        except Exception as e:
            logger.error(f"Błąd podczas ładowania modelu {model_id}: {e}")
            return None
    
    def _save_model(self, model, model_id: str, parameters: Dict[str, Any]):
        """Zapisuje model na dysku."""
        try:
            model_path = os.path.join(self.default_save_path, f"{model_id}.pkl")
            joblib.dump(model, model_path)
            
            # Zapisz parametry
            params_path = os.path.join(self.default_save_path, f"{model_id}_params.json")
            with open(params_path, 'w') as f:
                json.dump(parameters, f)
                
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania modelu {model_id}: {e}")
    
    def _save_training_history(self, model_id: str, metrics: ModelMetrics):
        """Zapisuje historię treningu w bazie danych."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT INTO ai_model_history (
                    model_id, training_duration, epochs_completed, accuracy,
                    loss, validation_accuracy, validation_loss, hyperparameters,
                    metrics, dataset_version, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                metrics.training_duration,
                metrics.epochs_completed,
                metrics.accuracy,
                metrics.loss,
                metrics.validation_accuracy,
                metrics.validation_loss,
                '{}',  # hyperparameters jako JSON
                json.dumps(metrics.metrics),
                metrics.dataset_version,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania historii treningu: {e}")
        finally:
            conn.close()
    
    def get_model_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Pobiera historię treningu modelu."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT * FROM ai_model_history 
                WHERE model_id = ? 
                ORDER BY timestamp DESC
            """, (model_id,))
            
            columns = [description[0] for description in c.description]
            history = []
            
            for row in c.fetchall():
                entry = dict(zip(columns, row))
                entry['metrics'] = json.loads(entry['metrics'])
                history.append(entry)
                
            return history
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania historii modelu: {e}")
            return []
        finally:
            conn.close()
    
    def _set_model_parameters(self, model, parameters: Dict[str, Any]):
        """Ustawia parametry modelu."""
        try:
            for param_name, param_value in parameters.items():
                if hasattr(model, param_name):
                    setattr(model, param_name, param_value)
        except Exception as e:
            logger.error(f"Błąd podczas ustawiania parametrów modelu: {e}")
    
    def get_model_parameters(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Pobiera obecne parametry modelu."""
        try:
            params_path = os.path.join(self.default_save_path, f"{model_id}_params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Błąd podczas pobierania parametrów modelu: {e}")
            return None
    
    def update_model_parameters(self, model_id: str, parameters: Dict[str, Any]) -> bool:
        """Aktualizuje parametry modelu."""
        try:
            model = self._load_model(model_id)
            if model is None:
                return False
                
            self._set_model_parameters(model, parameters)
            self._save_model(model, model_id, parameters)
            
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji parametrów modelu: {e}")
            return False
    
    def generate_training_visualizations(self, model_id: str) -> Dict[str, Any]:
        """Generuje wizualizacje procesu uczenia."""
        try:
            history = self.get_model_history(model_id)
            if not history:
                return {}
            
            # Wykres dokładności
            fig_accuracy = go.Figure()
            timestamps = [entry['timestamp'] for entry in history]
            accuracies = [entry['accuracy'] for entry in history]
            val_accuracies = [entry['validation_accuracy'] for entry in history]
            
            fig_accuracy.add_trace(go.Scatter(
                x=timestamps,
                y=accuracies,
                name='Dokładność treningu',
                mode='lines+markers'
            ))
            
            fig_accuracy.add_trace(go.Scatter(
                x=timestamps,
                y=val_accuracies,
                name='Dokładność walidacji',
                mode='lines+markers'
            ))
            
            fig_accuracy.update_layout(
                title='Historia dokładności modelu',
                xaxis_title='Data',
                yaxis_title='Dokładność',
                template='plotly_dark'
            )
            
            # Wykres straty
            fig_loss = go.Figure()
            losses = [entry['loss'] for entry in history]
            val_losses = [entry['validation_loss'] for entry in history]
            
            fig_loss.add_trace(go.Scatter(
                x=timestamps,
                y=losses,
                name='Strata treningu',
                mode='lines+markers'
            ))
            
            fig_loss.add_trace(go.Scatter(
                x=timestamps,
                y=val_losses,
                name='Strata walidacji',
                mode='lines+markers'
            ))
            
            fig_loss.update_layout(
                title='Historia straty modelu',
                xaxis_title='Data',
                yaxis_title='Strata',
                template='plotly_dark'
            )
            
            # Wykres czasu treningu
            fig_duration = go.Figure()
            durations = [entry['training_duration'] for entry in history]
            
            fig_duration.add_trace(go.Bar(
                x=timestamps,
                y=durations,
                name='Czas treningu'
            ))
            
            fig_duration.update_layout(
                title='Czas treningu modelu',
                xaxis_title='Data',
                yaxis_title='Czas (sekundy)',
                template='plotly_dark'
            )
            
            return {
                'accuracy_plot': fig_accuracy,
                'loss_plot': fig_loss,
                'duration_plot': fig_duration
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania wizualizacji: {e}")
            return {}
    
    def analyze_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Analizuje wydajność modelu w czasie."""
        try:
            history = self.get_model_history(model_id)
            if not history:
                return {}
            
            # Oblicz trendy
            accuracies = [entry['accuracy'] for entry in history]
            val_accuracies = [entry['validation_accuracy'] for entry in history]
            
            # Trend dokładności
            accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            val_accuracy_trend = np.polyfit(range(len(val_accuracies)), val_accuracies, 1)[0]
            
            # Oblicz stabilność
            accuracy_stability = np.std(accuracies)
            val_accuracy_stability = np.std(val_accuracies)
            
            # Znajdź najlepszy model
            best_model = max(history, key=lambda x: x['validation_accuracy'])
            
            return {
                'accuracy_trend': accuracy_trend,
                'val_accuracy_trend': val_accuracy_trend,
                'accuracy_stability': accuracy_stability,
                'val_accuracy_stability': val_accuracy_stability,
                'best_model': {
                    'timestamp': best_model['timestamp'],
                    'accuracy': best_model['accuracy'],
                    'validation_accuracy': best_model['validation_accuracy']
                },
                'latest_metrics': {
                    'accuracy': accuracies[-1],
                    'validation_accuracy': val_accuracies[-1],
                    'improvement': accuracies[-1] - accuracies[0]
                }
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas analizy wydajności modelu: {e}")
            return {}