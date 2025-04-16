
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging

class ReinforcementLearner:
    """
    Implementacja prostego uczenia ze wzmocnieniem dla handlu.
    """
    
    def __init__(self, state_size=10, action_size=3, learning_rate=0.001):
        """
        Inicjalizuje model uczenia ze wzmocnieniem.
        
        Args:
            state_size (int): Rozmiar stanu (liczba cech wejściowych)
            action_size (int): Liczba możliwych akcji
            learning_rate (float): Współczynnik uczenia
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.logger = logging.getLogger(__name__)
        
    def _build_model(self):
        """
        Buduje model sieci neuronowej.
        
        Returns:
            model: Zbudowany model sieci neuronowej
        """
        input_dim = self.state_size
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        # Compile the model with appropriate optimizer and loss function
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Trenuje model na podstawie danych.
        
        Args:
            X_train: Dane wejściowe do treningu
            y_train: Docelowe wartości
            epochs: Liczba epok treningu
            batch_size: Rozmiar batcha
            validation_split: Proporcja danych walidacyjnych
            
        Returns:
            history: Historia treningu modelu
        """
        try:
            # Add early stopping to prevent overfitting
            from tensorflow.keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = self.model.fit(
                X_train, y_train, 
                epochs=epochs,
                batch_size=batch_size,
                verbose=1, 
                validation_split=validation_split,
                callbacks=[early_stopping]
            )
            
            self.logger.info(f"Model trenowany przez {len(history.history['loss'])} epok")
            return history
        except Exception as e:
            self.logger.error(f"Błąd podczas treningu modelu: {e}")
            return None
    
    def predict(self, state):
        """
        Przewiduje akcję na podstawie stanu.
        
        Args:
            state: Stan środowiska
            
        Returns:
            predictions: Przewidywane wartości Q dla wszystkich akcji
        """
        try:
            # Ensure state is properly shaped for prediction
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
                
            return self.model.predict(state)
        except Exception as e:
            self.logger.error(f"Błąd podczas predykcji: {e}")
            return np.zeros(self.action_size)
    
    def save_model(self, filepath):
        """
        Zapisuje model do pliku.
        
        Args:
            filepath: Ścieżka do pliku
        """
        try:
            self.model.save(filepath)
            self.logger.info(f"Model zapisany do {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania modelu: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Ładuje model z pliku.
        
        Args:
            filepath: Ścieżka do pliku
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.logger.info(f"Model załadowany z {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas ładowania modelu: {e}")
            return False
    
    def get_best_action(self, state):
        """
        Zwraca najlepszą akcję dla danego stanu.
        
        Args:
            state: Stan środowiska
            
        Returns:
            action: Najlepsza akcja
        """
        q_values = self.predict(state)
        return np.argmax(q_values[0])
