"""
Pattern recognition module with GPU acceleration and advanced pattern detection.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

class PatternType(str, Enum):
    """Types of market patterns"""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INV_HEAD_SHOULDERS = "inverse_head_shoulders"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCending = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT_BULLISH = "pennant_bullish"
    PENNANT_BEARISH = "pennant_bearish"

@dataclass
class PatternConfig:
    """Configuration for pattern recognition"""
    min_pattern_size: int = 10
    max_pattern_size: int = 100
    peak_prominence: float = 0.1
    peak_width: int = 5
    slope_threshold: float = 0.01
    breakout_threshold: float = 0.02
    pattern_completion: float = 0.8
    use_volume: bool = True
    smoothing_window: int = 5
    use_gpu: bool = True

class ConvPatternDetector(nn.Module):
    """Convolutional neural network for pattern detection"""
    
    def __init__(
        self,
        input_channels: int = 5,  # OHLCV
        pattern_types: int = len(PatternType),
        base_filters: int = 32
    ):
        super().__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv1d(input_channels, base_filters, 5, padding=2)
        self.conv2 = nn.Conv1d(base_filters, base_filters * 2, 5, padding=2)
        self.conv3 = nn.Conv1d(base_filters * 2, base_filters * 4, 5, padding=2)
        
        # Pattern detection
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(base_filters * 4, base_filters * 2)
        self.fc2 = nn.Linear(base_filters * 2, pattern_types)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling
        x = self.gap(x).squeeze(-1)
        
        # Pattern classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.sigmoid(x)

class ModelRecognizer:
    """Advanced pattern recognition with GPU acceleration"""
    
    def __init__(
        self,
        config: Optional[PatternConfig] = None,
        device: Optional[str] = None
    ):
        self.config = config or PatternConfig()
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.detector = ConvPatternDetector().to(self.device)
        self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pretrained pattern detection weights"""
        try:
            weights_path = Path(__file__).parent / "weights" / "pattern_detector.pth"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self.device)
                self.detector.load_state_dict(state_dict)
                self.detector.eval()
        except Exception as e:
            self.logger.warning(f"Could not load pretrained weights: {e}")
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        window_size: Optional[int] = None
    ) -> torch.Tensor:
        """Prepare data for pattern detection"""
        if window_size is None:
            window_size = self.config.max_pattern_size
        
        # Extract OHLCV
        ohlcv = data[['open', 'high', 'low', 'close', 'volume']].values
        
        # Scale data
        scaled = self.scaler.fit_transform(ohlcv)
        
        # Convert to tensor
        tensor = torch.from_numpy(scaled).float()
        
        # Add batch and channel dimensions
        tensor = tensor.transpose(0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _detect_peaks_and_troughs(
        self,
        prices: np.ndarray,
        smoothed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect peaks and troughs in price data"""
        if smoothed:
            prices = pd.Series(prices).rolling(
                self.config.smoothing_window,
                center=True
            ).mean().values
        
        peaks, _ = find_peaks(
            prices,
            prominence=self.config.peak_prominence,
            width=self.config.peak_width
        )
        troughs, _ = find_peaks(
            -prices,
            prominence=self.config.peak_prominence,
            width=self.config.peak_width
        )
        
        return peaks, troughs
    
    def _analyze_slopes(
        self,
        prices: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Analyze slopes between points"""
        slopes = np.zeros(len(indices) - 1)
        for i in range(len(indices) - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            slopes[i] = (prices[idx2] - prices[idx1]) / (idx2 - idx1)
        return slopes
    
    def _check_volume_confirmation(
        self,
        pattern: Dict,
        data: pd.DataFrame
    ) -> float:
        """Check if volume confirms the pattern"""
        if not self.config.use_volume:
            return 1.0
        
        start_idx = pattern['start_idx']
        end_idx = pattern['end_idx']
        pattern_volume = data.iloc[start_idx:end_idx + 1]['volume']
        
        # Calculate volume trend
        volume_ma = pattern_volume.rolling(self.config.smoothing_window).mean()
        volume_trend = (volume_ma.iloc[-1] / volume_ma.iloc[0]) - 1
        
        # Check if volume confirms price action
        if pattern['type'] in [
            PatternType.DOUBLE_TOP,
            PatternType.HEAD_SHOULDERS,
            PatternType.TRIANGLE_DESCENDING,
            PatternType.WEDGE_FALLING,
            PatternType.FLAG_BEARISH,
            PatternType.PENNANT_BEARISH
        ]:
            # Bearish patterns should have increasing volume
            return min(1.0, max(0.0, volume_trend + 1))
        else:
            # Bullish patterns should have increasing volume
            return min(1.0, max(0.0, volume_trend + 1))
    
    def identify_patterns(
        self,
        data: pd.DataFrame,
        min_confidence: float = 0.7
    ) -> Dict[str, List[Dict]]:
        """Identify patterns in market data"""
        try:
            patterns = []
            
            # Prepare data for CNN
            tensor = self._prepare_data(data)
            
            # Get pattern probabilities from CNN
            with torch.no_grad():
                probs = self.detector(tensor).cpu().numpy()[0]
            
            # Process each pattern type
            close_prices = data['close'].values
            peaks, troughs = self._detect_peaks_and_troughs(close_prices)
            
            for pattern_type, prob in zip(PatternType, probs):
                if prob < min_confidence:
                    continue
                
                # Find pattern-specific formations
                if pattern_type in [PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM]:
                    points = peaks if pattern_type == PatternType.DOUBLE_TOP else troughs
                    for i in range(len(points) - 1):
                        if abs(close_prices[points[i]] - close_prices[points[i + 1]]) < self.config.peak_prominence:
                            pattern = {
                                'type': pattern_type,
                                'confidence': prob,
                                'start_idx': points[i],
                                'end_idx': points[i + 1],
                                'points': [points[i], points[i + 1]]
                            }
                            patterns.append(pattern)
                
                elif pattern_type in [PatternType.HEAD_SHOULDERS, PatternType.INV_HEAD_SHOULDERS]:
                    points = peaks if pattern_type == PatternType.HEAD_SHOULDERS else troughs
                    for i in range(len(points) - 2):
                        if (close_prices[points[i + 1]] > close_prices[points[i]] and
                            close_prices[points[i + 1]] > close_prices[points[i + 2]]):
                            pattern = {
                                'type': pattern_type,
                                'confidence': prob,
                                'start_idx': points[i],
                                'end_idx': points[i + 2],
                                'points': [points[i], points[i + 1], points[i + 2]]
                            }
                            patterns.append(pattern)
                
                elif pattern_type in [
                    PatternType.TRIANGLE_ASCENDING,
                    PatternType.TRIANGLE_DESCENDING,
                    PatternType.TRIANGLE_SYMMETRICAL
                ]:
                    for i in range(len(peaks) - 1):
                        for j in range(len(troughs) - 1):
                            upper_slope = self._analyze_slopes(close_prices, peaks[i:i + 2])
                            lower_slope = self._analyze_slopes(close_prices, troughs[j:j + 2])
                            
                            if (
                                (pattern_type == PatternType.TRIANGLE_ASCENDING and
                                 upper_slope < -self.config.slope_threshold and
                                 lower_slope > self.config.slope_threshold) or
                                (pattern_type == PatternType.TRIANGLE_DESCENDING and
                                 upper_slope < -self.config.slope_threshold and
                                 lower_slope < -self.config.slope_threshold) or
                                (pattern_type == PatternType.TRIANGLE_SYMMETRICAL and
                                 abs(upper_slope + lower_slope) < self.config.slope_threshold)
                            ):
                                pattern = {
                                    'type': pattern_type,
                                    'confidence': prob,
                                    'start_idx': min(peaks[i], troughs[j]),
                                    'end_idx': max(peaks[i + 1], troughs[j + 1]),
                                    'points': [peaks[i], peaks[i + 1], troughs[j], troughs[j + 1]]
                                }
                                patterns.append(pattern)
                
                # Check volume confirmation for all patterns
                for pattern in patterns:
                    volume_conf = self._check_volume_confirmation(pattern, data)
                    pattern['confidence'] *= volume_conf
            
            # Filter and sort patterns
            patterns = [
                p for p in patterns
                if p['confidence'] >= min_confidence and
                p['end_idx'] - p['start_idx'] >= self.config.min_pattern_size and
                p['end_idx'] - p['start_idx'] <= self.config.max_pattern_size
            ]
            
            patterns.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {'patterns': patterns}
            
        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")
            raise
    
    def save(self, path: Union[str, Path]):
        """Save pattern recognition state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save detector weights
        torch.save(
            self.detector.state_dict(),
            path / 'pattern_detector.pth'
        )
        
        # Save configuration
        import json
        with open(path / 'config.json', 'w') as f:
            json.dump(vars(self.config), f, indent=4)
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None
    ) -> 'ModelRecognizer':
        """Load pattern recognition state"""
        path = Path(path)
        
        # Load configuration
        with open(path / 'config.json', 'r') as f:
            import json
            config_dict = json.load(f)
            config = PatternConfig(**config_dict)
        
        # Create instance
        instance = cls(config=config, device=device)
        
        # Load detector weights
        weights_path = path / 'pattern_detector.pth'
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=instance.device)
            instance.detector.load_state_dict(state_dict)
            instance.detector.eval()
        
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
        
        # Test pattern recognition
        recognizer = ModelRecognizer()
        patterns = recognizer.identify_patterns(data)
        
        # Test persistence
        tmp_path = Path('test_patterns')
        recognizer.save(tmp_path)
        loaded = ModelRecognizer.load(tmp_path)
        patterns2 = loaded.identify_patterns(data)
        
        # Cleanup
        import shutil
        shutil.rmtree(tmp_path)
        
        logging.info("All pattern recognition tests passed!")
        
    except Exception as e:
        logging.error(f"Pattern recognition test failed: {e}")
        raise

if __name__ == "__main__":
    run_tests()
