"""
model_training.py
---------------
Zoptymalizowany moduł do treningu modeli AI z:
- Automatycznym doborem hiperparametrów
- Walidacją krzyżową
- Early stopping
- Checkpointami
- Równoległym treningiem
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.model_selection import KFold, TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicjalizacja trenera modeli z konfiguracją.
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego JSON
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.early_stopping = {}
        self.best_params = {}
        
        # Utwórz katalog na checkpointy
        os.makedirs('saved_models', exist_ok=True)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Ładuje konfigurację z pliku JSON lub używa domyślnych wartości."""
        default_config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 10,
            'num_folds': 5,
            'validation_split': 0.2,
            'optimizer': 'adam',
            'loss_function': 'mse',
            'parallel_training': True,
            'max_workers': 4,
            'checkpointing': True,
            'checkpoint_frequency': 10
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logger.error(f"Błąd podczas ładowania konfiguracji: {e}")
                
        return default_config

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Przygotowuje dane do treningu."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Podział na zbiór treningowy i walidacyjny
        train_size = int((1 - self.config['validation_split']) * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config['batch_size']
        )
        
        return train_loader, valid_loader

    def optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optymalizuje hiperparametry za pomocą Optuna."""
        def objective(trial):
            # Przestrzeń parametrów do optymalizacji
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_uniform('dropout', 0.1, 0.5)
            }
            
            # Walidacja krzyżowa
            kf = TimeSeriesSplit(n_splits=self.config['num_folds'])
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Trenuj model z aktualnym zestawem parametrów
                model = self.create_model(model_name, params)
                train_loader, valid_loader = self.prepare_data(X_train, y_train)
                
                try:
                    self.train_model(
                        model,
                        train_loader,
                        valid_loader,
                        epochs=20,  # Mniej epok dla optymalizacji
                        early_stopping_patience=5
                    )
                    
                    # Ewaluacja
                    score = self.evaluate_model(model, X_val, y_val)
                    scores.append(score['accuracy'])
                except Exception as e:
                    logger.error(f"Błąd podczas optymalizacji parametrów: {e}")
                    return float('-inf')
                
            return np.mean(scores)
        
        # Uruchom optymalizację
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        return study.best_params

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int = None,
        early_stopping_patience: int = None
    ) -> Dict[str, List[float]]:
        """
        Trenuje model z wykorzystaniem early stopping i checkpointów.
        
        Returns:
            Dict zawierający historię treningu
        """
        epochs = epochs or self.config['epochs']
        early_stopping_patience = early_stopping_patience or self.config['early_stopping_patience']
        
        model = model.to(self.device)
        optimizer = self._get_optimizer(model)
        criterion = self._get_loss_function()
        
        # Historia treningu
        history = {
            'train_loss': [],
            'valid_loss': [],
            'train_acc': [],
            'valid_acc': []
        }
        
        best_valid_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Tryb treningu
            model.train()
            train_losses = []
            train_preds = []
            train_true = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_preds.extend(outputs.detach().cpu().numpy())
                train_true.extend(batch_y.cpu().numpy())
            
            # Tryb ewaluacji
            model.eval()
            valid_losses = []
            valid_preds = []
            valid_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in valid_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    valid_losses.append(loss.item())
                    valid_preds.extend(outputs.cpu().numpy())
                    valid_true.extend(batch_y.cpu().numpy())
            
            # Oblicz średnie straty i dokładności
            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)
            train_acc = accuracy_score(np.round(train_true), np.round(train_preds))
            valid_acc = accuracy_score(np.round(valid_true), np.round(valid_preds))
            
            # Zapisz historię
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_acc'].append(train_acc)
            history['valid_acc'].append(valid_acc)
            
            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                
            # Checkpointing
            if self.config['checkpointing'] and epoch % self.config['checkpoint_frequency'] == 0:
                self._save_checkpoint(model, optimizer, epoch, valid_loss)
            
            # Logowanie postępu
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}"
            )
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping na epoce {epoch+1}")
                break
        
        # Przywróć najlepszy model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return history

    def train_multiple_models(
        self,
        models: Dict[str, nn.Module],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Trenuje wiele modeli równolegle."""
        results = {}
        
        if self.config['parallel_training']:
            with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
                future_to_model = {
                    executor.submit(
                        self._train_single_model,
                        model_name,
                        model,
                        X,
                        y
                    ): model_name for model_name, model in models.items()
                }
                
                for future in future_to_model:
                    model_name = future_to_model[future]
                    try:
                        results[model_name] = future.result()
                    except Exception as e:
                        logger.error(f"Błąd podczas treningu modelu {model_name}: {e}")
        else:
            for model_name, model in models.items():
                try:
                    results[model_name] = self._train_single_model(model_name, model, X, y)
                except Exception as e:
                    logger.error(f"Błąd podczas treningu modelu {model_name}: {e}")
        
        return results

    def _train_single_model(
        self,
        model_name: str,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Helper do treningu pojedynczego modelu."""
        # Optymalizacja hiperparametrów
        best_params = self.optimize_hyperparameters(model_name, X, y)
        
        # Aktualizacja modelu z optymalnymi parametrami
        model = self.create_model(model_name, best_params)
        
        # Przygotowanie danych
        train_loader, valid_loader = self.prepare_data(X, y)
        
        # Trening modelu
        history = self.train_model(model, train_loader, valid_loader)
        
        # Ewaluacja
        metrics = self.evaluate_model(model, X, y)
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'best_params': best_params
        }

    def evaluate_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Ewaluacja modelu ze szczegółowymi metrykami."""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
        
        # Oblicz metryki
        accuracy = accuracy_score(np.round(y), np.round(predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            np.round(y),
            np.round(predictions),
            average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Tworzy optymalizator na podstawie konfiguracji."""
        optimizer_name = self.config['optimizer'].lower()
        lr = self.config['learning_rate']
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Nieznany optymalizator: {optimizer_name}")

    def _get_loss_function(self) -> nn.Module:
        """Tworzy funkcję straty na podstawie konfiguracji."""
        loss_name = self.config['loss_function'].lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError(f"Nieznana funkcja straty: {loss_name}")

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        valid_loss: float
    ) -> None:
        """Zapisuje checkpoint modelu."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': valid_loss
        }
        
        checkpoint_path = os.path.join(
            'saved_models',
            f'checkpoint_epoch_{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Zapisano checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[nn.Module, int, float]:
        """Ładuje model z checkpointu."""
        checkpoint = torch.load(checkpoint_path)
        
        model = self.models.get(checkpoint['model_name'])
        if model is None:
            raise ValueError(f"Model {checkpoint['model_name']} nie został zainicjalizowany")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = self._get_optimizer(model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, checkpoint['epoch'], checkpoint['valid_loss']

    def create_model(self, model_name: str, params: Dict[str, Any]) -> nn.Module:
        """Tworzy model o zadanej architekturze z podanymi parametrami."""
        # Przykładowa implementacja - można rozszerzyć o więcej architektur
        if model_name == 'lstm':
            return LSTMModel(
                input_size=params['input_size'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        elif model_name == 'gru':
            return GRUModel(
                input_size=params['input_size'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        else:
            raise ValueError(f"Nieznana architektura modelu: {model_name}")

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])