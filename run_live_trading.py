"""
run_live_trading.py
------------------
Skrypt do uruchamiania godzinnego tradingu z wykorzystaniem AI w trybie symulacji lub z danymi w czasie rzeczywistym.
"""

import argparse
import logging
import os
import time
import random  # Dodany brakujący import
from datetime import datetime
import sqlite3
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Importy z własnych modułów
from python_libs.market_simulator import MarketSimulator
from python_libs.model_trainer import ModelTrainer
from python_libs.feature_processor import FeatureProcessor
from python_libs.realtime_market_connector import RealTimeMarketConnector

# Importy zaawansowanych modeli AI
try:
    import ai_models
    from ai_models.sentiment_ai import SentimentAnalyzer
    from ai_models.model_recognition import ModelRecognizer
    from ai_models.anomaly_detection import AnomalyDetector
    AI_MODELS_AVAILABLE = True
    logger = logging.getLogger("live_trading")
    logger.info("Modele AI zostały zaimportowane pomyślnie")
except ImportError as e:
    AI_MODELS_AVAILABLE = False
    logger = logging.getLogger("live_trading")
    logger.warning(f"Nie można zaimportować modeli AI: {e}")

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
    # Inicjalizacja symulatora rynku lub konektora danych w czasie rzeczywistym
    if config.get("realtime_trading", False):
        logger.info("Inicjalizacja konektora danych w czasie rzeczywistym...")
        
        # Pobierz klucze API z konfiguracji lub zmiennych środowiskowych
        api_key = config.get("api_key")
        api_secret = config.get("api_secret")
        
        simulator = RealTimeMarketConnector(
            symbol=config.get("symbol", "BTCUSDT"),
            timeframe=config.get("timeframe", "15m"),
            initial_capital=config.get("initial_capital", 10000),
            use_testnet=config.get("use_testnet", True),
            api_key=api_key,
            api_secret=api_secret,
            leverage=config.get("leverage", 1.0),
            commission=config.get("commission", 0.001),
            buffer_size=config.get("buffer_size", 100)
        )
        
        # Sprawdź czy konektor działa w trybie rzeczywistym czy symulacji
        if simulator.is_live_trading():
            logger.info("Uruchomiono w trybie handlu na żywo z ByBit API")
        else:
            logger.warning("Brak dostępu do ByBit API, używam trybu symulacyjnego")
    else:
        logger.info("Inicjalizacja symulatora rynku w trybie symulacyjnym...")
        
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
        # Dla XGBoost, używamy early stopping - poprawione dla nowszych wersji XGBoost
        eval_set = [(X_test, y_test)]
        model = model_trainer.train_model(
            model, X_train, y_train,
            model_name=f"{model_type.upper()}_DirectionPredictor",
            eval_set=eval_set, 
            verbose=False
        )
        # early_stopping_rounds jest teraz właściwością modelu
        if hasattr(model, 'set_params'):
            model.set_params(early_stopping_rounds=10)
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

def run_trading_simulation(simulator, feature_processor, model, feature_cols, duration=60, save_results=True, ask_iteration_interval=10):
    """
    Uruchamia symulację tradingu z wykorzystaniem modelu AI.
    
    Parameters:
        simulator: Symulator rynku lub konektor danych w czasie rzeczywistym
        feature_processor: Procesor cech
        model: Wytrenowany model AI
        feature_cols: Lista kolumn z cechami
        duration: Czas trwania symulacji w minutach
        save_results: Czy zapisać wyniki symulacji
        ask_iteration_interval: Co ile iteracji pytać użytkownika o kontynuację (0 lub None wyłącza pytanie)
        
    Returns:
        dict: Podsumowanie symulacji
    """
    # Resetuj symulator
    simulator.reset()
    
    # Ustaw maksymalną liczbę kroków dla symulatorów, które to obsługują
    if hasattr(simulator, 'max_steps'):
        simulator.max_steps = duration
    
    # Główna pętla symulacji
    logger.info(f"Rozpoczynam symulację tradingu ({duration} minut)...")
    
    state = simulator.get_state()
    step = 0
    
    # Statystyki podejmowania decyzji
    decision_stats = {
        "buy": 0,
        "sell": 0,
        "close": 0,
        "hold": 0,
        "predictions": []
    }
    
    # Jeszcze bardziej obniżam próg pewności dla modeli ML dla większej aktywności
    CONFIDENCE_THRESHOLD = 0.20  # Obniżony z 0.30 do 0.20
    
    # Bufor dla ostatnich N świeczek (potrzebne dla modeli rozpoznających wzorce)
    price_buffer = []
    
    # Bufor do gromadzenia cech w czasie rzeczywistym
    feature_buffer = []

    # Inicjalizujemy domyślne wartości
    last_direction = "neutral"
    consecutive_same_signals = 0  # Licznik kolejnych takich samych sygnałów
    
    # Liczniki do monitorowania uczenia
    learning_counter = 0
    features_built = 0
    model_updates = 0
    
    # Sprawdzenie czy mamy do czynienia z trybem rzeczywistym
    is_realtime = isinstance(simulator, RealTimeMarketConnector) and simulator.is_live_trading()
    if is_realtime:
        logger.info("Uruchamiam trading w trybie rzeczywistym z danymi z ByBit")
    
    while state is not None and step < duration:
        # Zapisz aktualną świeczkę do bufora
        current_data = pd.DataFrame([state])
        price_buffer.append(state)
        
        # Zachowujemy tylko ostatnie 50 świeczek w buforze
        if len(price_buffer) > 50:
            price_buffer = price_buffer[-50:]
        
        # Pobierz sygnały z zaawansowanych modeli AI, jeśli mamy wystarczająco danych w buforze
        ai_signals = {"direction": "neutral", "strength": 0.0}
        if len(price_buffer) >= 20:
            price_buffer_df = pd.DataFrame(price_buffer)
            ai_signals = get_advanced_ai_signals(price_buffer_df)
        
        # Przygotuj dane do predykcji dla modelu ML
        features_df = feature_processor.create_features(current_data)
        
        # Dodaj do bufora cech
        if not features_df.empty:
            feature_buffer.append(features_df)
            features_built += 1
            
        # Inicjalizacja domyślnych wartości dla predykcji
        ml_prediction = None
        ml_confidence = 0.0
        buy_signal = False
        sell_signal = False
        
        # Sprawdź, czy mamy wystarczająco cech do użycia modelu ML
        sufficient_features = False
        available_features = []
        
        if len(features_df) > 0 and not features_df.isnull().values.all():
            # Wybierz tylko cechy, których używaliśmy do treningu
            available_features = [col for col in feature_cols if col in features_df.columns and not features_df[col].isnull().any()]
            
            # Już tylko 20% cech wystarczy do próby predykcji
            if len(available_features) >= len(feature_cols) * 0.2:
                sufficient_features = True
                
                X_pred = features_df[available_features]
                
                # Skalowanie cech
                for col in available_features:
                    if col in feature_processor.scalers:
                        X_pred[col] = feature_processor.scalers[col].transform(X_pred[[col]])
                
                # Wykonaj predykcję z modelu ML
                try:
                    if hasattr(model, 'predict_proba'):
                        prediction_proba = model.predict_proba(X_pred)[0]
                        ml_prediction = model.predict(X_pred)[0]
                        ml_confidence = prediction_proba[ml_prediction] if len(prediction_proba) > 1 else prediction_proba[0]
                    else:
                        ml_prediction = model.predict(X_pred)[0]
                        ml_confidence = 0.5  # Domyślna wartość dla modeli bez predict_proba
                    
                    # Sygnał z modelu ML
                    if ml_prediction == 1 and ml_confidence > CONFIDENCE_THRESHOLD:
                        buy_signal = True
                    elif ml_prediction == 0 and ml_confidence > CONFIDENCE_THRESHOLD:
                        sell_signal = True
                except Exception as e:
                    logger.error(f"Błąd podczas predykcji ML: {e}")
                    ml_prediction = None
        
        # Zapisz statystyki predykcji
        prediction_record = {
            "step": step,
            "ml_prediction": int(ml_prediction) if ml_prediction is not None else -1,
            "ml_confidence": float(ml_confidence),
            "price": float(state['close']),
            "ai_direction": ai_signals["direction"],
            "ai_strength": ai_signals["strength"]
        }
        decision_stats["predictions"].append(prediction_record)
        
        # Sprawdzamy sygnały AI nawet jeśli nie mamy wystarczających cech ML
        # Jedyne co potrzebujemy to kierunek z AI większy niż neutralny
        if not sufficient_features:
            logger.info(f"Brak wystarczającej liczby cech ({len(available_features)}/{len(feature_cols)}), polegamy tylko na AI")
            
            # Próbuj inkrementalnie uczyć model co 5 kroków na zgromadzonych danych
            if len(feature_buffer) >= 5 and step % 5 == 0:
                try:
                    logger.info(f"Próbujemy zaktualizować model z bufora cech (rozmiar: {len(feature_buffer)})")
                    # Połącz wszystkie cechy z bufora
                    combined_features = pd.concat(feature_buffer)
                    
                    # Generuj etykiety na podstawie zmian cen
                    combined_features['direction'] = (combined_features['close'].shift(-1) > combined_features['close']).astype(int)
                    combined_features = combined_features.dropna()
                    
                    # Sprawdź dostępne cechy wspólne z oryginalnym zestawem cech
                    common_features = [col for col in feature_cols if col in combined_features.columns]
                    
                    if len(common_features) >= len(feature_cols) * 0.4:  # Przynajmniej 40% oryginalnych cech
                        learning_counter += 1
                        X_learn = combined_features[common_features]
                        y_learn = combined_features['direction']
                        
                        # Skalowanie cech
                        for col in common_features:
                            if col in feature_processor.scalers:
                                X_learn[col] = feature_processor.scalers[col].transform(X_learn[[col]])
                        
                        # Zaktualizuj model
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_learn, y_learn, classes=np.array([0, 1]))
                            model_updates += 1
                            logger.info(f"Model zaktualizowany inkrementalnie - {len(common_features)}/{len(feature_cols)} cech")
                        elif hasattr(model, 'fit') and hasattr(model, 'get_params') and 'warm_start' in model.get_params():
                            model.set_params(warm_start=True)
                            model.fit(X_learn, y_learn)
                            model_updates += 1
                            logger.info(f"Model zaktualizowany przez warm_start - {len(common_features)}/{len(feature_cols)} cech")
                        
                        # Wyczyść bufor
                        feature_buffer = feature_buffer[-10:]  # Zachowaj ostatnie 10 rekordów
                except Exception as e:
                    logger.error(f"Błąd podczas aktualizacji modelu z bufora: {e}")
            
            # Wykrywamy powtarzające się sygnały AI aby wzmocnić pewność
            if ai_signals["direction"] == last_direction and ai_signals["direction"] != "neutral":
                consecutive_same_signals += 1
            else:
                consecutive_same_signals = 0
                
            last_direction = ai_signals["direction"]
            
            # Wzmacniamy siłę sygnału AI jeśli powtarza się, ale z limitem - max 5 kolejnych wzmocnień
            boost_factor = min(consecutive_same_signals * 0.05, 0.25)  # Ograniczenie wzmocnienia do 0.25
            ai_strength_boosted = min(ai_signals["strength"] + boost_factor, 0.7)  # Ograniczenie max pewności do 0.7
            
            # Co 5 kroków, wprowadźmy czynnik losowy, który może zmienić kierunek
            # To zapobiegnie zbyt długim seriom tych samych decyzji
            if step % 5 == 0:
                # 30% szans na zmianę kierunku jeśli mamy już dużo tych samych sygnałów
                if consecutive_same_signals > 3 and random.random() < 0.3:
                    logger.info("Wprowadzam czynnik losowy, aby uniknąć zbyt długich serii tych samych decyzji")
                    if ai_signals["direction"] == "buy":
                        ai_signals["direction"] = "sell"
                    else:
                        ai_signals["direction"] = "buy"
                    ai_strength_boosted = 0.5  # Reset pewności
                    consecutive_same_signals = 0
            
            # Ustawiamy sygnały na podstawie AI
            if ai_signals["direction"] == "buy" and ai_strength_boosted > 0.25:
                buy_signal = True
                sell_signal = False
                ml_confidence = ai_strength_boosted  # Używamy siły AI jako pewności
            elif ai_signals["direction"] == "sell" and ai_strength_boosted > 0.25:
                sell_signal = True
                buy_signal = False
                
        # Jeśli mamy zarówno cechy ML i AI, łączymy sygnały
        else:
            # Wzmacniamy sygnał ML za pomocą AI
            if ai_signals["direction"] == "buy" and ai_signals["strength"] > 0.05:
                # Wzmacniamy istniejący sygnał lub tworzymy nowy
                if buy_signal:
                    ml_confidence += 0.15  # Wzmacniamy pewność
                else:
                    # Tworzymy nowy sygnał nawet bez sygnału z modelu ML
                    buy_signal = True
                    ml_confidence = 0.40
                    sell_signal = False  # Priorytet dla sygnału kupna
            
            elif ai_signals["direction"] == "sell" and ai_signals["strength"] > 0.05:
                # Wzmacniamy istniejący sygnał lub tworzymy nowy
                if sell_signal:
                    ml_confidence += 0.15  # Wzmacniamy pewność
                elif not buy_signal:  # Nie nadpisujemy sygnału kupna
                    # Tworzymy nowy sygnał
                    sell_signal = True
                    ml_confidence = 0.40
        
        # Logika podejmowania decyzji handlowych
        action = "hold"
        position_size = None
        
        try:
            if buy_signal:
                action = "buy"
                # Wielkość pozycji proporcjonalna do pewności przewidywania
                position_size = min(0.4, ml_confidence * 0.6) * simulator.current_capital / state['close']
                decision_stats["buy"] += 1
                logger.info(f"Krok {step+1}/{duration}: Akcja={action}, Pewność={ml_confidence:.2f}, AI={ai_signals.get('direction', 'N/A')}, Cena={state['close']:.2f}")
            elif sell_signal:
                action = "sell"
                position_size = min(0.4, ml_confidence * 0.6) * simulator.current_capital / state['close']
                decision_stats["sell"] += 1
                logger.info(f"Krok {step+1}/{duration}: Akcja={action}, Pewność={ml_confidence:.2f}, AI={ai_signals.get('direction', 'N/A')}, Cena={state['close']:.2f}")
            else:
                # Jeśli mamy otwartą pozycję, sprawdź warunki do zamknięcia
                if simulator.position != 0:
                    # Sprawdź, czy osiągnęliśmy próg zysku lub straty
                    unrealized_pnl = state.get('unrealized_pnl', 0)
                    entry_value = abs(simulator.position_size * simulator.entry_price) if hasattr(simulator, 'position_size') and hasattr(simulator, 'entry_price') else 0
                    
                    # Nawet mniejsze progi zamknięcia pozycji
                    if entry_value > 0 and ((unrealized_pnl > entry_value * 0.003) or (unrealized_pnl < -entry_value * 0.003)):
                        action = "close"
                        decision_stats["close"] += 1
                        logger.info(f"Krok {step+1}/{duration}: Akcja={action}, Zysk/Strata={unrealized_pnl:.2f}, Cena={state['close']:.2f}")
                    else:
                        action = "hold"
                        decision_stats["hold"] += 1
                        logger.info(f"Krok {step+1}/{duration}: Akcja={action}, Utrzymuję pozycję, Cena={state['close']:.2f}")
                else:
                    action = "hold"
                    decision_stats["hold"] += 1
                    logging_message = f"Krok {step+1}/{duration}: Akcja={action}"
                    if sufficient_features:
                        logging_message += " (brak wystarczających sygnałów)"
                    else:
                        logging_message += " (brak wystarczających cech i słabe sygnały AI)"
                    logger.info(logging_message)
            
            # Wykonaj akcję
            state, reward, done, info = simulator.step(action, position_size)
            
            # Aktualizacja modelu w trakcie symulacji po każdej transakcji
            if action in ["buy", "sell"]:
                last_price_change = state['close'] - state['open']
                model, updated_features = update_model_during_simulation(model, feature_processor, state, feature_cols, last_price_change)
                if updated_features:
                    model_updates += 1
                    logger.info(f"Model zaktualizowany w kroku {step+1} po akcji {action}")
            
        except Exception as e:
            logger.error(f"Błąd podczas podejmowania decyzji: {e}")
            state, reward, done, info = simulator.step("hold")
            decision_stats["hold"] += 1
        
        # Pytanie użytkownika o kontynuację iteracji, jeśli włączono tę opcję
        should_ask = ask_iteration_interval and ask_iteration_interval > 0 and (
            step % ask_iteration_interval == 0 or  # Co określoną liczbę kroków
            action in ["buy", "sell", "close"]     # Lub po każdej transakcji
        )
        
        if should_ask:
            current_pnl = info.get('unrealized_pnl', 0)
            print(f"\n--- Iteracja {step+1}/{duration} ---")
            print(f"Akcja: {action.upper()}")
            print(f"Cena: {state['close']:.2f}")
            print(f"Kapitał: {simulator.current_capital:.2f}")
            print(f"Niezrealizowany zysk/strata: {current_pnl:.2f}")
            print(f"Pozycja: {simulator.position}")
            
            try:
                answer = input("Czy kontynuować iterację? (t/n): ").lower()
                if answer.startswith('n'):
                    logger.info("Przerwanie symulacji na żądanie użytkownika")
                    print("Zatrzymuję symulację...")
                    break
            except KeyboardInterrupt:
                logger.info("Przerwanie symulacji przez KeyboardInterrupt")
                print("\nZatrzymuję symulację...")
                break
        
        step += 1
        
        if done:
            logger.info(f"Symulacja zakończona po {step} krokach. Powód: {info.get('reason', 'nieznany')}")
            break
            
        # W trybie rzeczywistym, dodajemy opóźnienie między krokami
        if is_realtime:
            # W zależności od timeframe, czekamy odpowiednią ilość sekund
            wait_time = 5  # Domyślnie 5 sekund między decyzjami
            logger.debug(f"Oczekiwanie {wait_time} sekund przed następną decyzją...")
            time.sleep(wait_time)
    
    # Zbierz wyniki
    summary = simulator.get_summary()
    
    # Wyświetl statystyki decyzji
    logger.info(f"Statystyki decyzji: Kupno={decision_stats['buy']}, Sprzedaż={decision_stats['sell']}, Zamknięcie={decision_stats['close']}, Wstrzymanie={decision_stats['hold']}")
    
    # Statystyki uczenia w czasie rzeczywistym
    logger.info(f"Statystyki uczenia: Zgromadzone cechy={features_built}, Aktualizacje modelu={model_updates}, Cykle uczenia={learning_counter}")
    
    # Zapisz wyniki
    if save_results:
        try:
            simulator.save_results()
            simulator.plot_results(save_path="reports/trading_simulation_result.png")
            
            # Zapisz decyzje i predykcje do pliku CSV
            if decision_stats["predictions"]:
                pd.DataFrame(decision_stats["predictions"]).to_csv("reports/ai_predictions.csv", index=False)
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania wyników: {e}")
    
    logger.info(f"Symulacja zakończona. Wynik: {summary['profit']:.2f} ({summary['profit_percentage']:.2f}%)")
    
    return summary

def load_historical_data(source_type="db", source_path=None, symbol="BTCUSDT", timeframe="1h", limit=1000):
    """
    Wczytuje rzeczywiste dane historyczne z bazy danych lub pliku CSV.
    
    Parameters:
        source_type (str): Typ źródła danych - "db" (baza danych) lub "csv" (plik CSV)
        source_path (str): Ścieżka do źródła danych
        symbol (str): Symbol instrumentu (np. "BTCUSDT")
        timeframe (str): Przedział czasowy (np. "1h", "15m")
        limit (int): Maksymalna liczba rekordów do pobrania
        
    Returns:
        DataFrame: DataFrame z danymi historycznymi
    """
    try:
        if source_type == "db" and source_path and os.path.exists(source_path):
            logger.info(f"Wczytywanie danych historycznych z bazy: {source_path}")
            
            # Połączenie z bazą danych SQLite
            conn = sqlite3.connect(source_path)
            
            # Określamy zapytanie SQL
            query = f"""
                SELECT * FROM price_data 
                WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            
            # Próba wczytania danych z bazy
            try:
                df = pd.read_sql_query(query, conn)
                conn.close()
                if not df.empty:
                    logger.info(f"Wczytano {len(df)} rekordów z bazy danych")
                    # Upewnij się, że mamy wymagane kolumny
                    if all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"]):
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp")
                        return df
                    else:
                        logger.warning("Brakuje wymaganych kolumn w danych z bazy")
                else:
                    logger.warning(f"Brak danych dla {symbol} w bazie")
            except Exception as e:
                logger.error(f"Błąd podczas wczytywania danych z bazy: {e}")
                conn.close()
        
        elif source_type == "csv" and source_path and os.path.exists(source_path):
            logger.info(f"Wczytywanie danych historycznych z pliku: {source_path}")
            
            # Wczytanie danych z pliku CSV
            df = pd.read_csv(source_path)
            
            # Upewnij się, że mamy wymagane kolumny
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if all(col in df.columns for col in required_cols):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                logger.info(f"Wczytano {len(df)} rekordów z pliku CSV")
                return df
            else:
                logger.warning("Brakuje wymaganych kolumn w pliku CSV")
        
        elif source_type == "directory" and source_path and os.path.isdir(source_path):
            logger.info(f"Szukanie plików CSV w katalogu: {source_path}")
            
            # Znajdź wszystkie pliki CSV w katalogu
            csv_files = glob.glob(os.path.join(source_path, "*.csv"))
            
            # Szukaj pliku z odpowiednim symbolem
            matching_files = [f for f in csv_files if symbol.lower() in os.path.basename(f).lower()]
            
            if matching_files:
                file_path = matching_files[0]
                logger.info(f"Znaleziono pasujący plik: {file_path}")
                
                # Wczytanie danych z pliku CSV
                df = pd.read_csv(file_path)
                
                # Upewnij się, że mamy wymagane kolumny
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                
                # Sprawdź typowe nazwy kolumn w danych finansowych, które mogą być używane
                column_mapping = {
                    'date': 'timestamp',
                    'time': 'timestamp',
                    'Date': 'timestamp',
                    'Time': 'timestamp',
                    'Timestamp': 'timestamp',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                }
                
                # Mapuj nazwy kolumn, jeśli to konieczne
                df = df.rename(columns=column_mapping)
                
                if all(col in df.columns for col in required_cols):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")
                    logger.info(f"Wczytano {len(df)} rekordów z pliku CSV")
                    return df
                else:
                    logger.warning("Brakuje wymaganych kolumn w pliku CSV")
            else:
                logger.warning(f"Nie znaleziono plików CSV dla symbolu {symbol} w katalogu {source_path}")
                
        # Gdy nie udało się wczytać danych, informujemy o tym i generujemy dane syntetyczne
        logger.warning("Nie można wczytać danych historycznych. Używam danych syntetycznych.")
        return None
            
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania danych historycznych: {e}")
        return None

def get_advanced_ai_signals(data, window_size=30):
    """
    Pobiera sygnały z zaawansowanych modeli AI na podstawie danych.
    
    Args:
        data: DataFrame z danymi cenowymi
        window_size: Rozmiar okna do analizy
        
    Returns:
        Dict: Słownik z sygnałami i ich siłą
    """
    try:
        # Inicjalizacja modeli AI
        sentiment = SentimentAnalyzer()
        logging.info("SentimentAnalyzer zainicjalizowany")
        
        recognizer = ModelRecognizer()
        logging.info("ModelRecognizer zainicjalizowany")
        
        # Wykryj wzorce rynkowe
        window_data = data.tail(window_size)
        pattern_results = recognizer.identify_model_type(window_data)
        
        # Bezpieczne pobieranie wartości confidence - zabezpieczenie przed sytuacją gdy jest to słownik
        confidence_value = 0.1  # Wartość domyślna
        if isinstance(pattern_results, dict):
            if "confidence" in pattern_results:
                if isinstance(pattern_results["confidence"], (float, int)):
                    confidence_value = pattern_results["confidence"]
                elif isinstance(pattern_results["confidence"], dict):
                    # Jeśli confidence jest słownikiem, spróbuj znaleźć największą wartość
                    if pattern_results["confidence"]:
                        confidence_value = max(pattern_results["confidence"].values())
            
            # Bezpieczne sprawdzanie typu wzorca
            pattern_type = pattern_results.get("type", "")
            if isinstance(pattern_type, dict):
                # Jeśli typ jest słownikiem, połącz wartości
                pattern_type = " ".join(str(v) for v in pattern_type.values())
            elif isinstance(pattern_type, list):
                pattern_type = " ".join(str(v) for v in pattern_type)
            else:
                pattern_type = str(pattern_type)
        else:
            pattern_type = ""
        
        # Określenie sygnału na podstawie typu wzorca i pewności
        pattern_signal = "buy" if confidence_value > 0.5 and "Bull" in pattern_type else "sell"
        pattern_strength = confidence_value
        
        # Wykryj anomalie
        anomaly_detector = AnomalyDetector()
        logging.info("Zainicjalizowano domyślny model detekcji anomalii (Isolation Forest)")
        
        if not hasattr(anomaly_detector, '_model') or anomaly_detector._model is None:
            logging.info("Model IsolationForest nie był wytrenowany. Trenuję na bieżących danych...")
            anomaly_detector.fit(window_data)
            logging.info(f"Model wytrenowany na {len(window_data)} próbkach")
        
        # Wynik detekcji anomalii
        anomaly_score = anomaly_detector.detect(window_data)
        if isinstance(anomaly_score, (float, int)):
            anomaly_signal = "sell" if anomaly_score > 0.5 else "buy"
        else:
            # Jeśli anomaly_score nie jest liczbą, użyj wartości domyślnej
            anomaly_signal = "sell"
            anomaly_score = 0.5
        
        # Analiza sentymentu (symulowana)
        sentiment_signal = "buy" if random.random() > 0.5 else "sell"
        sentiment_strength = random.uniform(0.4, 0.8)
        
        # Łączenie sygnałów z różnych źródeł
        # Dajemy największą wagę wzorcom cenowym, następnie anomaliom i najmniejszą sentymentom
        combined_signal = "sell"  # Domyślnie sell jako zachowawcze podejście
        
        # Jeśli mamy silny sygnał wzorcowy, użyj go
        if pattern_strength > 0.7:
            combined_signal = pattern_signal
        
        # Wzmocnienie sygnału, jeśli anomalia i wzorzec są zgodne
        if pattern_signal == anomaly_signal:
            combined_strength = max(pattern_strength, 0.4)  # Minimum 0.4 jeśli sygnały się zgadzają
        else:
            combined_strength = min(pattern_strength, 0.3)  # Maximum 0.3 jeśli sygnały się nie zgadzają
        
        # Wzmocnienie lub osłabienie sygnału w zależności od sentymentu
        if sentiment_signal == combined_signal:
            combined_strength = min(combined_strength + 0.1, 0.9)  # Wzmocnienie max do 0.9
        else:
            combined_strength = max(combined_strength - 0.1, 0.1)  # Osłabienie min do 0.1
        
        logging.info(f"Sygnały AI: Kierunek={combined_signal}, Siła={combined_strength:.2f}")
        return {"direction": combined_signal, "strength": combined_strength}
    
    except Exception as e:
        logging.error(f"Błąd w get_advanced_ai_signals: {e}")
        return {"direction": "sell", "strength": 0.1}  # Bezpieczna wartość domyślna

def update_model_during_simulation(model, feature_processor, new_data, feature_cols, last_price_change):
    """
    Aktualizuje model ML w trakcie symulacji, przeprowadzając inkrementalne uczenie.
    
    Parameters:
        model: Model ML do aktualizacji
        feature_processor: Procesor cech
        new_data: Nowe dane cenowe
        feature_cols: Lista dostępnych kolumn z cechami
        last_price_change: Ostatnia zmiana ceny (do utworzenia etykiety 'direction')
        
    Returns:
        model: Zaktualizowany model
        updated_features: Nowe cechy dla aktualizacji
    """
    try:
        # Tworzymy pojedynczą ramkę danych do przetworzenia
        if isinstance(new_data, dict):
            single_df = pd.DataFrame([new_data])
        else:
            single_df = new_data
            
        # Sprawdzamy czy dane wejściowe są poprawne
        if single_df.empty:
            logging.warning("update_model_during_simulation: Otrzymano pustą ramkę danych wejściowych")
            return model, None
            
        # Wyświetlamy informacje o danych wejściowych dla debugowania
        logging.debug(f"update_model_during_simulation: Kolumny w danych wejściowych: {single_df.columns.tolist()}")
        logging.debug(f"update_model_during_simulation: Kształt danych wejściowych: {single_df.shape}")
            
        # Dla etykiety kierunku - używamy przekazanego parametru
        direction = 1 if last_price_change >= 0 else 0
        single_df.loc[:, 'direction'] = direction
        
        # Generujemy cechy dla nowych danych - używamy dedykowanej obsługi dla pojedynczego wiersza
        try:
            # Upewnijmy się, że dane zawierają wymagane kolumny dla feature_processor
            required_cols = ['open', 'high', 'low', 'close', 'volume'] 
            if not all(col in single_df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in single_df.columns]
                logging.warning(f"Brak wymaganych kolumn w danych: {missing_cols}")
                
                # Próbujemy uzupełnić brakujące kolumny jeśli to możliwe
                for col in missing_cols:
                    if col == 'volume' and 'close' in single_df.columns:
                        # Używamy zastępczych wartości dla volume jeśli brakuje
                        single_df.loc[:, 'volume'] = single_df['close'] * random.uniform(100, 1000)
                        logging.info(f"Dodano sztuczne dane wolumenu: {single_df['volume'].values[0]:.2f}")
                    elif 'close' in single_df.columns:
                        # Dla innych kolumn używamy wartości close jako przybliżenia
                        single_df.loc[:, col] = single_df['close']
                        logging.info(f"Dodano sztuczne dane dla kolumny {col}: {single_df[col].values[0]:.2f}")
            
            # Używamy ulepszonej funkcji create_features, która obsługuje pojedyncze wiersze
            features_df = feature_processor.create_features(single_df, indicators=True, window_features=False)
            
            # Sprawdź czy wygenerowano jakieś cechy
            if features_df.empty:
                logging.warning("Nie udało się wygenerować cech dla pojedynczego wiersza danych")
                return model, None
                
            # Informacja o wygenerowanych cechach
            logging.debug(f"Wygenerowano cechy dla pojedynczego wiersza: {features_df.columns.tolist()}")
                
        except Exception as e:
            logging.error(f"Błąd podczas generowania cech: {e}")
            return model, None
        
        # Wybieramy tylko cechy, które nie zawierają wartości NaN
        valid_features = [col for col in features_df.columns 
                         if col in feature_cols 
                         and not features_df[col].isnull().any()
                         and col != 'direction']
        
        # Jeśli brak wystarczającej liczby cech, zwróć model bez zmian
        if len(valid_features) < 3:  # Minimum 3 cechy
            logging.info(f"Niewystarczająca liczba cech do aktualizacji modelu: {len(valid_features)}/{len(feature_cols)}")
            return model, None
        else:
            logging.info(f"Znaleziono {len(valid_features)} cech dla aktualizacji modelu")
            
        # Przygotowujemy dane do uczenia
        X = features_df[valid_features].copy()  # Używamy .copy() aby uniknąć ostrzeżeń o widoku
        y = features_df['direction'].copy()
        
        # Jeśli nie mamy żadnych danych, zwróć model bez zmian
        if X.empty or len(X) == 0:
            logging.warning("Puste dane treningowe X - brak możliwości aktualizacji modelu")
            return model, None
        
        # Sprawdź i konwertuj typy kolumn do float64, aby uniknąć ostrzeżeń o niezgodności typów
        for col in valid_features:
            # Konwertujemy wszystkie kolumny na float64 dla spójności
            X[col] = X[col].astype('float64')
        
        # Skalowanie cech - tylko jeśli mamy rzeczywiste dane do skalowania
        for col in valid_features:
            if col in feature_processor.scalers and len(X[col]) > 0:
                try:
                    # Przekształć dane tylko jeśli kolumna zawiera wartości
                    if not X[col].isnull().all() and X[col].shape[0] > 0:
                        # Używamy .loc[] zamiast bezpośredniego przypisania
                        transformed_values = feature_processor.scalers[col].transform(X[[col]])
                        # Konwertujemy wynik na 1D array, a następnie przypisujemy z zachowaniem typu float64
                        X.loc[:, col] = transformed_values.flatten().astype('float64')
                except Exception as e:
                    logging.warning(f"Błąd skalowania cechy {col}: {e}")
                    # W przypadku błędu skalowania, używamy surowych danych
                    pass
        
        # Inkrementalne uczenie modelu
        if hasattr(model, 'partial_fit'):
            # Dla modeli wspierających partial_fit (np. SGDClassifier)
            if len(valid_features) > 0 and len(X) > 0:
                try:
                    model.partial_fit(X, y, classes=np.array([0, 1]))
                    logging.info(f"Model zaktualizowany inkrementalnie z {len(valid_features)}/{len(feature_cols)} cech")
                    return model, {"features": valid_features, "X": X, "y": y}
                except Exception as e:
                    logging.warning(f"Nie można zaktualizować modelu przez partial_fit: {e}")
        elif hasattr(model, 'fit'):
            # Dla standardowych modeli scikit-learn, które wspierają tylko pełne uczenie
            # Możemy użyć warm_start=True dla niektórych modeli
            if hasattr(model, 'set_params') and 'warm_start' in model.get_params() and len(X) > 0:
                try:
                    model.set_params(warm_start=True)
                    model.fit(X, y)
                    logging.info(f"Model zaktualizowany przez warm_start z {len(valid_features)}/{len(feature_cols)} cech")
                    return model, {"features": valid_features, "X": X, "y": y}
                except Exception as e:
                    logging.warning(f"Nie można zaktualizować modelu przez warm_start: {e}")
    except Exception as e:
        logging.error(f"Błąd podczas aktualizacji modelu: {e}")
    
    return model, None

def main():
    """Główna funkcja programu"""
    parser = argparse.ArgumentParser(description="Symulacja godzinnego tradingu z wykorzystaniem AI")
    parser.add_argument("--data-file", type=str, help="Plik CSV z danymi historycznymi")
    parser.add_argument("--data-dir", type=str, help="Katalog z plikami CSV danych historycznych")
    parser.add_argument("--db-file", type=str, help="Plik bazy danych SQLite z danymi historycznymi")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol instrumentu (np. BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Przedział czasowy (np. 1h, 15m)")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["randomforest", "gradientboosting", "xgboost"], help="Typ modelu AI")
    parser.add_argument("--duration", type=int, default=60, help="Czas trwania symulacji w minutach")
    parser.add_argument("--capital", type=float, default=10000, help="Początkowy kapitał")
    parser.add_argument("--leverage", type=float, default=1.0, help="Dźwignia finansowa")
    
    # Nowe argumenty dla trybu danych w czasie rzeczywistym
    parser.add_argument("--realtime", action="store_true", help="Uruchom w trybie danych w czasie rzeczywistym z ByBit")
    parser.add_argument("--testnet", action="store_true", help="Użyj testnet API ByBit (tylko dla trybu realtime)")
    parser.add_argument("--api-key", type=str, help="Klucz API ByBit (tylko dla trybu realtime)")
    parser.add_argument("--api-secret", type=str, help="Sekretny klucz API ByBit (tylko dla trybu realtime)")
    
    args = parser.parse_args()
    
    # Utworzenie katalogów
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Określenie źródła danych i wczytanie danych historycznych
    historical_data = None
    data_source = "synthetic"
    
    if args.realtime:
        # Tryb danych w czasie rzeczywistym
        data_source = "realtime"
        logger.info("Wybrano tryb danych w czasie rzeczywistym z ByBit")
    elif args.db_file and os.path.exists(args.db_file):
        # Próba wczytania danych z bazy danych SQLite
        historical_data = load_historical_data(
            source_type="db", 
            source_path=args.db_file,
            symbol=args.symbol,
            timeframe=args.timeframe
        )
        if historical_data is not None:
            data_source = "historical"
            logger.info(f"Używam danych historycznych z bazy: {args.db_file}")
    
    elif args.data_file and os.path.exists(args.data_file):
        # Próba wczytania danych z pliku CSV
        historical_data = load_historical_data(
            source_type="csv", 
            source_path=args.data_file
        )
        if historical_data is not None:
            data_source = "historical"
            logger.info(f"Używam danych historycznych z pliku: {args.data_file}")
    
    elif args.data_dir and os.path.isdir(args.data_dir):
        # Próba wczytania danych z katalogu z plikami CSV
        historical_data = load_historical_data(
            source_type="directory", 
            source_path=args.data_dir,
            symbol=args.symbol
        )
        if historical_data is not None:
            data_source = "historical"
            logger.info(f"Używam danych historycznych z katalogu: {args.data_dir}")
    
    else:
        # Używaj standardowych ścieżek w projekcie jako ostatniej próby
        standard_paths = [
            "/workspaces/ZoL0/data/historical_data.db",
            "/workspaces/ZoL0/data/data",
            "/workspaces/ZoL0/data/backtesting"
        ]
        
        for path in standard_paths:
            if os.path.exists(path):
                if path.endswith('.db'):
                    historical_data = load_historical_data(
                        source_type="db", 
                        source_path=path,
                        symbol=args.symbol,
                        timeframe=args.timeframe
                    )
                elif os.path.isdir(path):
                    historical_data = load_historical_data(
                        source_type="directory", 
                        source_path=path,
                        symbol=args.symbol
                    )
                
                if historical_data is not None:
                    data_source = "historical"
                    logger.info(f"Używam danych historycznych z: {path}")
                    break
    
    # Jeśli tryb realtime, ignorujemy dane historyczne
    if args.realtime:
        historical_data = None
        
    if not args.realtime and historical_data is None:
        logger.warning("Nie znaleziono danych historycznych. Używam danych syntetycznych.")
    
    # Konfiguracja
    config = {
        "data_source": data_source,
        "historical_data": historical_data,
        "initial_capital": args.capital,
        "leverage": args.leverage,
        "model_dir": "saved_models"
    }
    
    # Dodatkowe konfiguracje dla trybu realtime
    if args.realtime:
        config["realtime_trading"] = True
        config["symbol"] = args.symbol
        config["timeframe"] = args.timeframe
        config["use_testnet"] = args.testnet
        config["api_key"] = args.api_key
        config["api_secret"] = args.api_secret
    
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
