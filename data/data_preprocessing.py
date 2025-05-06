"""Data preprocessing module for feature engineering and data cleaning."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

def prepare_data_for_model(data: Union[Dict, List, np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Convert various data formats to numpy array for ML models."""
    if data is None:
        raise ValueError("Input data cannot be None")
    
    if isinstance(data, np.ndarray):
        return data
    
    if isinstance(data, pd.DataFrame):
        return data.values
    
    if isinstance(data, list):
        if not data:
            raise ValueError("Empty list provided")
        return np.array(data)
    
    if isinstance(data, dict):
        if not data:
            raise ValueError("Empty dictionary provided")
            
        if all(k in data for k in ['open', 'high', 'low', 'close', 'volume']):
            df = pd.DataFrame({
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            })
            return df.values
            
        if 'close' in data:
            return np.array(data['close']).reshape(-1, 1)
            
        first_key = next(iter(data))
        if isinstance(data[first_key], (list, np.ndarray)):
            return np.array(data[first_key]).reshape(-1, 1)
            
    raise ValueError(f"Unsupported data format: {type(data)}")

def preprocess_data(data: pd.DataFrame, 
                   use_ta: bool = True,
                   handle_missing: bool = True,
                   remove_outliers: bool = True,
                   normalize: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocess data for model training/prediction."""
    if data is None or data.empty:
        raise ValueError("Input data cannot be None or empty")
        
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        raise ValueError("Data must contain OHLCV columns")
        
    metadata = {}
    df = data.copy()
    
    try:
        # Handle missing values first
        if handle_missing:
            df = clean_data(df)
            
        # Add technical indicators if requested
        if use_ta:
            try:
                df = add_technical_features(df)
            except ValueError as e:
                logger.warning(f"Could not add technical features: {str(e)}")
                
        # Remove outliers if requested
        if remove_outliers:
            df_no_outliers, outlier_params = remove_outliers_zscore(df)
            # Only apply outlier removal if it doesn't remove too much data
            if len(df_no_outliers) >= len(df) * 0.7:
                df = df_no_outliers
                metadata['outlier_params'] = outlier_params
            else:
                logger.warning("Skipping outlier removal as it would remove too much data")
            
        # Normalize features if requested
        if normalize:
            df, scalers = normalize_features(df)
            metadata['scalers'] = scalers
            
        metadata['feature_names'] = list(df.columns)
        metadata['preprocessing_steps'] = {
            'use_ta': use_ta,
            'handle_missing': handle_missing,
            'remove_outliers': remove_outliers,
            'normalize': normalize
        }
        metadata['rows_processed'] = len(df)
        
        return df, metadata
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by handling missing values and enforcing data constraints."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be None or empty")
        
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Missing required columns. Expected: {required_cols}")
        
    df_clean = df.copy()
    
    # Handle completely missing columns
    all_null_cols = df_clean.columns[df_clean.isnull().all()].tolist()
    if all_null_cols:
        logger.warning(f"Columns with all missing values: {all_null_cols}")
        # For price columns with all NaN, use other price columns as reference
        price_cols = ['open', 'high', 'low', 'close']
        for col in all_null_cols:
            if col in price_cols:
                valid_prices = df_clean[list(set(price_cols) - {col})].mean(axis=1)
                df_clean[col] = valid_prices
    
    # Forward fill price data
    price_cols = ['open', 'high', 'low', 'close']
    df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill')
    df_clean[price_cols] = df_clean[price_cols].fillna(method='bfill')
    
    # Replace any remaining NaN with column medians
    df_clean = df_clean.fillna(df_clean.median())
    
    # Handle zero or negative prices
    min_valid_price = df_clean[price_cols].values[df_clean[price_cols] > 0].min()
    if pd.isna(min_valid_price):
        min_valid_price = 1.0  # Fallback if no valid prices
        
    for col in price_cols:
        mask = df_clean[col] <= 0
        if mask.any():
            df_clean.loc[mask, col] = min_valid_price
            
    # Ensure price relationships (high >= open/close >= low)
    for i in range(len(df_clean)):
        high = max(df_clean.loc[i, ['open', 'high', 'low', 'close']])
        low = min(df_clean.loc[i, ['open', 'high', 'low', 'close']])
        
        # If high == low, add a small spread
        if high == low:
            spread = high * 0.001  # 0.1% spread
            high += spread
            low -= spread
            
        df_clean.loc[i, 'high'] = high
        df_clean.loc[i, 'low'] = low
        df_clean.loc[i, 'open'] = np.clip(df_clean.loc[i, 'open'], low, high)
        df_clean.loc[i, 'close'] = np.clip(df_clean.loc[i, 'close'], low, high)
    
    # Ensure non-negative volume with minimum value
    df_clean['volume'] = df_clean['volume'].clip(lower=1)  # Minimum volume of 1
    
    return df_clean

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical analysis features."""
    if len(df) < 20:
        raise ValueError("Insufficient data for technical analysis (min 20 periods)")
        
    if (df['close'] <= 0).any():
        raise ValueError("Invalid price data: prices must be positive")
    
    df_tech = df.copy()
    
    # Price-based features
    df_tech['returns'] = df_tech['close'].pct_change()
    df_tech['log_returns'] = np.log(df_tech['close'] / df_tech['close'].shift(1))
    df_tech['volatility'] = df_tech['returns'].rolling(window=20).std()
    
    # Volume features
    df_tech['volume_ma'] = df_tech['volume'].rolling(window=20).mean()
    df_tech['volume_std'] = df_tech['volume'].rolling(window=20).std()
    
    # Price movement features
    df_tech['price_ma'] = df_tech['close'].rolling(window=20).mean()
    df_tech['upper_shadow'] = df_tech['high'] - df_tech[['open', 'close']].max(axis=1)
    df_tech['lower_shadow'] = df_tech[['open', 'close']].min(axis=1) - df_tech['low']
    
    # Remove rows with NaN from rolling calculations
    df_tech = df_tech.dropna()
    
    return df_tech

def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict]:
    """Remove outliers using z-score method with adaptive thresholding."""
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
        
    df_clean = df.copy()
    params = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Process price columns separately to maintain relationships
    price_cols = ['open', 'high', 'low', 'close']
    available_price_cols = [col for col in price_cols if col in df.columns]
    
    if available_price_cols:
        # Store parameters for each price column
        for col in available_price_cols:
            mean = df[col].mean()
            std = df[col].std()
            params[col] = {'mean': mean, 'std': std, 'threshold': threshold}
            
        # Use price range for detecting outliers in price columns as a group
        price_range = df[available_price_cols].max(axis=1) - df[available_price_cols].min(axis=1)
        range_mean = price_range.mean()
        range_std = price_range.std()
        
        if range_std > 0:
            z_scores = abs((price_range - range_mean) / range_std)
            mask = z_scores < threshold
            df_clean = df_clean.loc[mask]  # Use .loc for proper index alignment
            params['price_range'] = {'mean': range_mean, 'std': range_std, 'threshold': threshold}
    
    # Process other columns individually
    other_cols = [col for col in numeric_cols if col not in available_price_cols]
    for col in other_cols:
        mean = df[col].mean()
        std = df[col].std()
        
        if std == 0:
            logger.warning(f"Column {col} has zero standard deviation, skipping outlier detection")
            continue
            
        z_scores = abs((df[col] - mean) / std)
        mask = z_scores < threshold
        df_clean = df_clean.loc[mask]  # Use .loc for proper index alignment
        params[col] = {'mean': mean, 'std': std, 'threshold': threshold}
    
    return df_clean, params

def normalize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Normalize features while preserving relationships."""
    df_norm = df.copy()
    scalers = {}
    
    # Check for non-numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) != len(df.columns):
        raise TypeError("All columns must be numeric for normalization")
    
    # Preserve price relationships by scaling them together
    price_cols = ['open', 'high', 'low', 'close']
    price_cols_present = [col for col in price_cols if col in df.columns]
    
    if price_cols_present:
        # Use MinMaxScaler for prices to preserve relationships
        price_scaler = MinMaxScaler(feature_range=(0, 1))
        price_array = df[price_cols_present].values
        
        # Scale prices while preserving relationships
        price_min = price_array.min()
        price_max = price_array.max()
        price_range = price_max - price_min
        
        if price_range > 0:
            normalized_prices = (price_array - price_min) / price_range
            df_norm[price_cols_present] = normalized_prices
        else:
            df_norm[price_cols_present] = 0.5  # Default to middle if no range
            
        for col in price_cols_present:
            scalers[col] = price_scaler
    
    # Use StandardScaler for other columns
    other_cols = [col for col in numeric_cols if col not in price_cols_present]
    for col in other_cols:
        if df[col].std() == 0:
            df_norm[col] = 0
            logger.warning(f"Column {col} has constant value, setting to zero")
            continue
            
        scaler = StandardScaler()
        df_norm[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    
    return df_norm, scalers
