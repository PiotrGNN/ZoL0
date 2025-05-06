"""
Advanced feature engineering with pattern recognition and GPU acceleration.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from .scalar import FeatureScaler
from .model_recognition import PatternType, ModelRecognizer

@dataclass
class FeatureConfig:
    """Configuration for feature generation"""
    # Price features
    use_returns: bool = True
    return_periods: List[int] = (1, 5, 10, 20)
    use_log_returns: bool = True
    
    # Technical indicators
    use_ma: bool = True
    ma_periods: List[int] = (5, 10, 20, 50, 100)
    use_volatility: bool = True
    volatility_periods: List[int] = (5, 10, 20)
    use_momentum: bool = True
    momentum_periods: List[int] = (14, 28)
    
    # Pattern features
    use_patterns: bool = True
    min_pattern_confidence: float = 0.7
    pattern_window: int = 100
    
    # Volume features
    use_volume: bool = True
    volume_ma_periods: List[int] = (5, 10, 20)
    use_vwap: bool = True
    
    # Feature selection
    n_components: Optional[int] = None
    importance_threshold: float = 0.01
    use_gpu: bool = True

class FeatureEngineer:
    """Advanced feature engineering with GPU acceleration"""
    
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        device: Optional[str] = None
    ):
        self.config = config or FeatureConfig()
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pattern_recognizer = ModelRecognizer(device=str(self.device))
        self.scaler = FeatureScaler(device=str(self.device))
        self.pca = None
        self.feature_importance = None
        
        # Cache for pattern features
        self._pattern_cache = {}
    
    def _calculate_returns(
        self,
        prices: pd.Series,
        periods: List[int],
        log_returns: bool = True
    ) -> pd.DataFrame:
        """Calculate price returns"""
        returns = pd.DataFrame(index=prices.index)
        
        for period in periods:
            if log_returns:
                ret = np.log(prices / prices.shift(period))
                returns[f'log_return_{period}'] = ret
            else:
                ret = prices.pct_change(period)
                returns[f'return_{period}'] = ret
        
        return returns
    
    def _calculate_technical_indicators(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate technical indicators"""
        indicators = pd.DataFrame(index=data.index)
        
        if self.config.use_ma:
            for period in self.config.ma_periods:
                # SMA
                sma = SMAIndicator(data['close'], period).sma_indicator()
                indicators[f'sma_{period}'] = sma
                
                # EMA
                ema = EMAIndicator(data['close'], period).ema_indicator()
                indicators[f'ema_{period}'] = ema
                
                # MA Crossovers
                if period < max(self.config.ma_periods):
                    next_period = next(p for p in self.config.ma_periods if p > period)
                    indicators[f'ma_cross_{period}_{next_period}'] = (
                        (sma > SMAIndicator(data['close'], next_period).sma_indicator())
                        .astype(int)
                    )
        
        if self.config.use_momentum:
            for period in self.config.momentum_periods:
                # RSI
                rsi = RSIIndicator(data['close'], period).rsi()
                indicators[f'rsi_{period}'] = rsi
                
                # Stochastic
                stoch = StochasticOscillator(
                    data['high'],
                    data['low'],
                    data['close'],
                    period
                )
                indicators[f'stoch_k_{period}'] = stoch.stoch()
                indicators[f'stoch_d_{period}'] = stoch.stoch_signal()
        
        if self.config.use_volatility:
            for period in self.config.volatility_periods:
                # Bollinger Bands
                bb = BollingerBands(data['close'], period)
                indicators[f'bb_upper_{period}'] = bb.bollinger_hband()
                indicators[f'bb_lower_{period}'] = bb.bollinger_lband()
                indicators[f'bb_width_{period}'] = (
                    (bb.bollinger_hband() - bb.bollinger_lband()) /
                    bb.bollinger_mavg()
                )
        
        if self.config.use_volume:
            # OBV
            obv = OnBalanceVolumeIndicator(
                data['close'],
                data['volume']
            ).on_balance_volume()
            indicators['obv'] = obv
            
            # Volume MA
            for period in self.config.volume_ma_periods:
                vol_ma = data['volume'].rolling(period).mean()
                indicators[f'volume_ma_{period}'] = vol_ma
                indicators[f'volume_ma_ratio_{period}'] = (
                    data['volume'] / vol_ma
                )
        
        if self.config.use_vwap:
            # VWAP
            vwap = VolumeWeightedAveragePrice(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                volume=data['volume']
            ).volume_weighted_average_price()
            indicators['vwap'] = vwap
            indicators['vwap_ratio'] = data['close'] / vwap
        
        return indicators
    
    def _extract_pattern_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract pattern-based features"""
        if not self.config.use_patterns:
            return pd.DataFrame(index=data.index)
        
        # Check cache
        cache_key = hash(str(data.iloc[-self.config.pattern_window:].values))
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
        
        # Initialize pattern features
        pattern_features = pd.DataFrame(
            0,
            index=data.index,
            columns=[p.value for p in PatternType]
        )
        
        # Detect patterns
        for i in range(len(data) - self.config.pattern_window + 1):
            window = data.iloc[i:i + self.config.pattern_window]
            patterns = self.pattern_recognizer.identify_patterns(
                window,
                min_confidence=self.config.min_pattern_confidence
            )
            
            for pattern in patterns['patterns']:
                pattern_features.iloc[i + self.config.pattern_window - 1][
                    pattern['type']
                ] = pattern['confidence']
        
        # Cache results
        self._pattern_cache[cache_key] = pattern_features
        
        return pattern_features
    
    def _select_features(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Select most important features"""
        # Convert to numpy for processing
        feature_array = features.values
        
        if self.config.n_components:
            # Apply PCA
            if self.pca is None:
                self.pca = PCA(n_components=self.config.n_components)
                feature_array = self.pca.fit_transform(feature_array)
            else:
                feature_array = self.pca.transform(feature_array)
            
            # Create DataFrame with PCA components
            selected = pd.DataFrame(
                feature_array,
                index=features.index,
                columns=[f'PC_{i+1}' for i in range(self.config.n_components)]
            )
        
        elif target is not None:
            # Calculate feature importance using mutual information
            if self.feature_importance is None:
                self.feature_importance = mutual_info_regression(
                    feature_array,
                    target.values
                )
                
                # Select features above threshold
                important_features = [
                    col for col, imp in zip(features.columns, self.feature_importance)
                    if imp > self.config.importance_threshold
                ]
                
                selected = features[important_features]
            else:
                selected = features[
                    [col for col, imp in zip(features.columns, self.feature_importance)
                     if imp > self.config.importance_threshold]
                ]
        else:
            selected = features
        
        return selected
    
    def generate_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate features from market data"""
        try:
            if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                raise ValueError("Data must contain OHLCV columns")
            
            feature_sets = {}
            
            # Price-based features
            if self.config.use_returns:
                returns = self._calculate_returns(
                    data['close'],
                    self.config.return_periods,
                    self.config.use_log_returns
                )
                feature_sets['returns'] = returns
            
            # Technical indicators
            indicators = self._calculate_technical_indicators(data)
            feature_sets['technical'] = indicators
            
            # Pattern features
            if self.config.use_patterns:
                patterns = self._extract_pattern_features(data)
                feature_sets['patterns'] = patterns
            
            # Combine all features
            features = pd.concat(feature_sets.values(), axis=1)
            
            # Handle missing values
            features = features.fillna(method='ffill').fillna(0)
            
            # Select features
            features = self._select_features(features, target)
            
            # Scale features
            if fit:
                self.scaler.fit(features)
            scaled_features = self.scaler.transform(features)
            
            # Move to GPU if needed
            if self.config.use_gpu:
                scaled_features = torch.from_numpy(
                    scaled_features.values
                ).float().to(self.device)
            
            return scaled_features, {
                'n_features': features.shape[1],
                'feature_sets': {
                    name: set(df.columns)
                    for name, df in feature_sets.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            raise
    
    def save(self, path: Union[str, Path]):
        """Save feature engineering state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(path / 'config.json', 'w') as f:
            import json
            json.dump(vars(self.config), f, indent=4)
        
        # Save scaler
        self.scaler.save(path / 'scaler')
        
        # Save PCA if used
        if self.pca is not None:
            import joblib
            joblib.dump(self.pca, path / 'pca.joblib')
        
        # Save feature importance if calculated
        if self.feature_importance is not None:
            np.save(path / 'feature_importance.npy', self.feature_importance)
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None
    ) -> 'FeatureEngineer':
        """Load feature engineering state"""
        path = Path(path)
        
        # Load configuration
        with open(path / 'config.json', 'r') as f:
            import json
            config_dict = json.load(f)
            config = FeatureConfig(**config_dict)
        
        # Create instance
        instance = cls(config=config, device=device)
        
        # Load scaler
        instance.scaler = FeatureScaler.load(path / 'scaler', device=device)
        
        # Load PCA if exists
        pca_path = path / 'pca.joblib'
        if pca_path.exists():
            import joblib
            instance.pca = joblib.load(pca_path)
        
        # Load feature importance if exists
        importance_path = path / 'feature_importance.npy'
        if importance_path.exists():
            instance.feature_importance = np.load(importance_path)
        
        return instance

def run_tests():
    """Run unit tests"""
    try:
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 102,
            'low': np.random.randn(1000).cumsum() + 98,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        # Test with default config
        engineer = FeatureEngineer()
        features, info = engineer.generate_features(data)
        
        # Test with custom config
        config = FeatureConfig(
            use_patterns=True,
            n_components=10,
            use_gpu=True
        )
        engineer = FeatureEngineer(config=config)
        features, info = engineer.generate_features(data)
        
        # Test persistence
        tmp_path = Path('test_features')
        engineer.save(tmp_path)
        loaded = FeatureEngineer.load(tmp_path)
        features2, _ = loaded.generate_features(data, fit=False)
        
        # Cleanup
        import shutil
        shutil.rmtree(tmp_path)
        
        logging.info("All feature engineering tests passed!")
        
    except Exception as e:
        logging.error(f"Feature engineering test failed: {e}")
        raise

if __name__ == "__main__":
    run_tests()
