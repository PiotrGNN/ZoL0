
"""
model_factory.py
----------------
Fabryka modeli AI - tworzy instancje modeli z domyślnymi parametrami.
"""

import logging
from typing import Dict, Any, Optional, Union, Type

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Fabryka modeli AI - tworzy instancje modeli z domyślnymi parametrami.
    """
    
    @staticmethod
    def create_model_trainer():
        """
        Tworzy instancję ModelTrainer z domyślnymi parametrami.
        """
        try:
            from .model_training import ModelTrainer
            from sklearn.ensemble import RandomForestRegressor
            
            # Tworzenie instancji prostego modelu
            model = RandomForestRegressor(n_estimators=10)
            
            # Tworzenie instancji ModelTrainer z tym modelem
            trainer = ModelTrainer(
                model=model,
                model_name="DefaultRandomForest",
                saved_model_dir="saved_models",
                online_learning=False
            )
            
            logger.info("Utworzono instancję ModelTrainer z domyślnymi parametrami")
            return trainer
            
        except ImportError as e:
            logger.error(f"Nie można utworzyć ModelTrainer: {e}")
            return None
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia ModelTrainer: {e}")
            return None
    
    @staticmethod
    def create_real_exchange_env():
        """
        Tworzy instancję RealExchangeEnv z domyślnymi parametrami.
        """
        try:
            from .real_exchange_env import RealExchangeEnv
            
            # Utworzenie instancji z domyślnymi parametrami
            env = RealExchangeEnv(
                api_url="https://api-testnet.bybit.com/v5/market/kline",
                api_key="testnet_key",
                initial_capital=1000,
                leverage=1.0
            )
            
            logger.info("Utworzono instancję RealExchangeEnv z domyślnymi parametrami")
            return env
            
        except ImportError as e:
            logger.error(f"Nie można utworzyć RealExchangeEnv: {e}")
            return None
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia RealExchangeEnv: {e}")
            return None

# Słownik instancji modeli
_model_instances: Dict[str, Any] = {}

def get_model_instance(model_name: str) -> Optional[Any]:
    """
    Pobiera lub tworzy instancję modelu.
    
    Args:
        model_name: Nazwa modelu
        
    Returns:
        Optional[Any]: Instancja modelu lub None
    """
    if model_name in _model_instances:
        return _model_instances[model_name]
    
    if model_name == "ModelTrainer":
        instance = ModelFactory.create_model_trainer()
    elif model_name == "RealExchangeEnv":
        instance = ModelFactory.create_real_exchange_env()
    else:
        logger.warning(f"Nieznany model: {model_name}")
        return None
    
    if instance:
        _model_instances[model_name] = instance
        
    return instance
