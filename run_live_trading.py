
"""
run_live_trading.py
------------------
Skrypt do uruchamiania godzinnego tradingu z wykorzystaniem AI w trybie symulacji.
"""

import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Importy z własnych modułów
from python_libs.market_simulator import MarketSimulator
from python_libs.model_trainer import ModelTrainer
from python_libs.feature_processor import FeatureProcessor

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/live_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("live_trading")

def setup_trading_environment(config):
    """
    Przygotowuje środowisko do symulacji tradingu.
    
    Parameters:
        config (dict): Konfiguracja symulacji
        
    Returns:
        tuple: (symulator, procesor_cech, trener_modeli)
    """
    # Inicjalizacja symulatora rynku
    simulator = MarketSimulator(
        data_source=config.get("data_source", "synthetic"),
        initial_capital=config.get("initial_capital", 10000),
        leverage=config.get("leverage", 1.0),
        spread=config.get("spread", 0.0005),
        commission=config.get("commission", 0.001),
        slippage=config.get("slippage", 0.0002),
        volatility=config.get("volatility", 0.01),
        data_file=config.get("data_file")
    )
    
    # Inicjalizacja procesora cech
    feature_processor = FeatureProcessor(scaler_type=config.get("scaler_type", "standard"))
    
    # Inicjalizacja trenera modeli
    model_trainer = ModelTrainer(model_dir=config.get("model_dir", "saved_models"))
    
    logger.info("Środowisko tradingowe zostało zainicjalizowane")
    
    return simulator, feature_processor, model_trainer

def create_model(model_type="randomforest", **model_params):
    """
    Tworzy model uczenia maszynowego.
    
    Parameters:
        model_type (str): Typ modelu - "randomforest", "gradientboosting", "xgboost"
        **model_params: Parametry modelu
        
    Returns:
        Model uczenia maszynowego
    """
    if model_type == "randomforest":
        model = RandomForestClassifier(**model_params)
    elif model_type == "gradientboosting":
        model = GradientBoostingClassifier(**model_params)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(**model_params)
    else:
        logger.warning(f"Nieznany typ modelu: {model_type}. Używam RandomForest.")
        model = RandomForestClassifier(**model_params)
    
    logger.info(f"Utworzono model: {model_type}")
    return model

def train_ai_model(feature_processor, model_trainer, simulator, model_type="randomforest"):
    """
    Trenuje model AI na danych z symulatora.
    
    Parameters:
        feature_processor: Procesor cech
        model_trainer: Trener modeli
        simulator: Symulator rynku
        model_type: Typ modelu
        
    Returns:
        Wytrenowany model
    """
    # Pobierz dane z symulatora
    price_data = simulator.price_data.copy()
    
    # Tworzenie cech
    logger.info("Tworzenie cech dla modelu AI...")
    features_df = feature_processor.create_features(price_data, indicators=True, window_features=True)
    
    # Przygotowanie danych treningowych
    X_train, X_test, y_train, y_test, feature_cols = feature_processor.prepare_data(
        features_df, target_col='direction', train_ratio=0.8
    )
    
    # Utwórz model
    if model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42
        )
    elif model_type == "gradientboosting":
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42
        )
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42
        )
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Trenuj model
    logger.info(f"Trening modelu AI ({model_type})...")
    
    if model_type == "xgboost":
        # Dla XGBoost, używamy early stopping
        eval_set = [(X_test, y_test)]
        model = model_trainer.train_model(
            model, X_train, y_train,
            model_name=f"{model_type.upper()}_DirectionPredictor",
            eval_set=eval_set, early_stopping_rounds=10,
            verbose=False
        )
    else:
        model = model_trainer.train_model(
            model, X_train, y_train,
            model_name=f"{model_type.upper()}_DirectionPredictor"
        )
    
    # Ocena modelu
    metrics = model_trainer.evaluate_model(model, X_test, y_test)
    logger.info(f"Wyniki modelu: {metrics}")
    
    # Zapisz model
    model_trainer.save_model()
    
    # Generuj raport
    model_trainer.generate_report(model, X_test, y_test)
    
    return model, feature_cols

def run_trading_simulation(simulator, feature_processor, model, feature_cols, duration=60, save_results=True):
    """
    Uruchamia symulację tradingu z wykorzystaniem modelu AI.
    
    Parameters:
        simulator: Symulator rynku
        feature_processor: Procesor cech
        model: Wytrenowany model AI
        feature_cols: Lista kolumn z cechami
        duration: Czas trwania symulacji w minutach
        save_results: Czy zapisać wyniki symulacji
        
    Returns:
        dict: Podsumowanie symulacji
    """
    # Resetuj symulator
    simulator.reset()
    simulator.max_steps = duration
    
    # Główna pętla symulacji
    logger.info(f"Rozpoczynam symulację tradingu ({duration} minut)...")
    
    state = simulator.get_state()
    step = 0
    
    while state is not None and step < duration:
        # Przygotuj dane do predykcji
        current_data = pd.DataFrame([state])
        features_df = feature_processor.create_features(current_data)
        
        # Niektóre cechy mogą być niedostępne w pierwszych krokach
        if len(features_df) == 0 or features_df.isnull().values.any():
            # Przejdź do następnego kroku bez podejmowania akcji
            state, reward, done, info = simulator.step("hold")
            step += 1
            continue
        
        # Wybierz tylko cechy, których używaliśmy do treningu
        available_features = [col for col in feature_cols if col in features_df.columns]
        
        # Jeśli nie ma wystarczająco cech, przejdź dalej
        if len(available_features) < len(feature_cols) * 0.8:
            logger.warning(f"Brak wystarczającej liczby cech: {len(available_features)}/{len(feature_cols)}")
            state, reward, done, info = simulator.step("hold")
            step += 1
            continue
        
        X_pred = features_df[available_features]
        
        # Skalowanie cech
        for col in available_features:
            if col in feature_processor.scalers:
                X_pred[col] = feature_processor.scalers[col].transform(X_pred[[col]])
        
        # Wykonaj predykcję
        try:
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(X_pred)[0]
                prediction = model.predict(X_pred)[0]
                confidence = prediction_proba[prediction] if len(prediction_proba) > 1 else prediction_proba[0]
            else:
                prediction = model.predict(X_pred)[0]
                confidence = 0.5  # Domyślna wartość dla modeli bez predict_proba
            
            if prediction == 1 and confidence > 0.6:
                action = "buy"
                # Wielkość pozycji proporcjonalna do pewności przewidywania
                position_size = min(0.2, confidence * 0.3) * simulator.current_capital / state['close']
            elif prediction == 0 and confidence > 0.6:
                action = "sell"
                position_size = min(0.2, confidence * 0.3) * simulator.current_capital / state['close']
            else:
                # Jeśli mamy otwartą pozycję, sprawdź warunki do zamknięcia
                if simulator.position != 0:
                    # Sprawdź, czy osiągnęliśmy próg zysku lub straty
                    unrealized_pnl = state['unrealized_pnl']
                    entry_value = simulator.position_size * simulator.entry_price
                    
                    # Zamknij pozycję, gdy zysk > 3% lub strata > 1.5%
                    if (unrealized_pnl > entry_value * 0.03) or (unrealized_pnl < -entry_value * 0.015):
                        action = "close"
                    else:
                        action = "hold"
                else:
                    action = "hold"
                position_size = None
                
            logger.info(f"Krok {step+1}/{duration}: Akcja={action}, Pewność={confidence:.2f}, Cena={state['close']:.2f}")
            
            # Wykonaj akcję
            state, reward, done, info = simulator.step(action, position_size)
            
        except Exception as e:
            logger.error(f"Błąd podczas podejmowania decyzji: {e}")
            state, reward, done, info = simulator.step("hold")
        
        step += 1
        
        if done:
            logger.info(f"Symulacja zakończona po {step} krokach. Powód: {info.get('reason', 'nieznany')}")
            break
    
    # Zbierz wyniki
    summary = simulator.get_summary()
    
    # Zapisz wyniki
    if save_results:
        simulator.save_results()
        simulator.plot_results(save_path="reports/trading_simulation_result.png")
    
    logger.info(f"Symulacja zakończona. Wynik: {summary['profit']:.2f} ({summary['profit_percentage']:.2f}%)")
    
    return summary

def main():
    """Główna funkcja programu"""
    parser = argparse.ArgumentParser(description="Symulacja godzinnego tradingu z wykorzystaniem AI")
    parser.add_argument("--data-file", type=str, help="Plik CSV z danymi historycznymi")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["randomforest", "gradientboosting", "xgboost"], help="Typ modelu AI")
    parser.add_argument("--duration", type=int, default=60, help="Czas trwania symulacji w minutach")
    parser.add_argument("--capital", type=float, default=10000, help="Początkowy kapitał")
    parser.add_argument("--leverage", type=float, default=1.0, help="Dźwignia finansowa")
    
    args = parser.parse_args()
    
    # Utworzenie katalogów
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Konfiguracja
    config = {
        "data_source": "file" if args.data_file else "synthetic",
        "data_file": args.data_file,
        "initial_capital": args.capital,
        "leverage": args.leverage,
        "model_dir": "saved_models"
    }
    
    # Inicjalizacja środowiska
    simulator, feature_processor, model_trainer = setup_trading_environment(config)
    
    # Trenowanie modelu AI
    model, feature_cols = train_ai_model(feature_processor, model_trainer, simulator, model_type=args.model_type)
    
    # Uruchomienie symulacji
    run_trading_simulation(simulator, feature_processor, model, feature_cols, duration=args.duration, save_results=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Błąd podczas wykonywania programu: {e}", exc_info=True)
