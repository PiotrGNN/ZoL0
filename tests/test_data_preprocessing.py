"""Tests for data preprocessing functionality."""

import unittest
import numpy as np
import pandas as pd
from data.data_preprocessing import (
    preprocess_data,
    clean_data,
    add_technical_features,
    remove_outliers_zscore,
    normalize_features,
    prepare_data_for_model
)

class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing module."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        self.data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.randint(1000, 2000, 100)
        })
        
        # Add some test cases
        self.data.loc[10, 'close'] = np.nan  # Missing value
        self.data.loc[20, 'close'] = self.data['close'].max() * 2  # Outlier
        self.data.loc[30, 'volume'] = 0  # Zero volume
        self.data.loc[40:45, ['open', 'high', 'low', 'close']] = np.nan  # Multiple missing
        
        # Add more test cases
        self.data.loc[50:55, 'close'] = self.data['close'].mean() + self.data['close'].std() * 4  # Multiple outliers
        self.data.loc[60:65, 'volume'] = 0  # Multiple zero volumes
        self.data.loc[70:75, 'close'] *= 1.5  # Price surge
        self.data.loc[80:85, 'close'] *= 0.7  # Price drop
        
        # Create empty and invalid dataframes for edge cases
        self.empty_df = pd.DataFrame()
        self.invalid_df = pd.DataFrame({'invalid_col': np.random.rand(100)})
        
    def test_clean_data(self):
        """Test data cleaning functionality."""
        df_clean = clean_data(self.data)
        
        # Check no missing values
        self.assertFalse(df_clean.isnull().any().any())
        
        # Check price relationships maintained
        self.assertTrue((df_clean['high'] >= df_clean['low']).all())
        self.assertTrue((df_clean['volume'] >= 0).all())
        
    def test_technical_features(self):
        """Test technical feature generation."""
        df_clean = clean_data(self.data)
        df_tech = add_technical_features(df_clean)
        
        # Check required features exist
        required_features = ['returns', 'log_returns', 'volatility', 
                           'volume_ma', 'volume_std']
        for feature in required_features:
            self.assertIn(feature, df_tech.columns)
            
        # Check feature properties
        self.assertTrue(np.isfinite(df_tech['returns']).all())
        self.assertTrue(np.isfinite(df_tech['log_returns']).all())
        self.assertTrue((df_tech['volatility'] >= 0).all())
        
    def test_outlier_removal(self):
        """Test outlier detection and removal."""
        df_clean = clean_data(self.data)
        df_no_outliers, params = remove_outliers_zscore(df_clean)
        
        # Check outlier parameters captured
        self.assertIn('close', params)
        self.assertIn('mean', params['close'])
        self.assertIn('std', params['close'])
        
        # Verify severe outliers removed
        close_zscore = abs((df_no_outliers['close'] - params['close']['mean']) 
                          / params['close']['std'])
        self.assertTrue((close_zscore < params['close']['threshold']).all())
        
    def test_normalize_features(self):
        """Test feature normalization."""
        df_clean = clean_data(self.data)
        df_norm, scalers = normalize_features(df_clean)
        
        # Check scalers created for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertIn(col, scalers)
            
        # Verify normalization properties for price columns (MinMaxScaler)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_norm.columns:
                self.assertTrue((df_norm[col] >= 0).all())
                self.assertTrue((df_norm[col] <= 1).all())
                
        # Verify other columns (StandardScaler)
        other_cols = [col for col in numeric_cols if col not in price_cols]
        for col in other_cols:
            if df_clean[col].std() > 0:  # Skip constant columns
                self.assertAlmostEqual(df_norm[col].mean(), 0, places=1)
                self.assertAlmostEqual(df_norm[col].std(), 1, places=1)
            
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        df_processed, metadata = preprocess_data(
            self.data,
            use_ta=True,
            handle_missing=True,
            remove_outliers=True,
            normalize=True
        )
        
        # Check metadata contents
        self.assertIn('outlier_params', metadata)
        self.assertIn('scalers', metadata)
        self.assertIn('feature_names', metadata)
        
        # Verify data quality
        self.assertFalse(df_processed.isnull().any().any())
        self.assertTrue(all(col in df_processed.columns 
                          for col in ['returns', 'volatility']))
        
        # Check normalization (price columns use MinMaxScaler)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_processed.columns:
                self.assertTrue((df_processed[col] >= 0).all())
                self.assertTrue((df_processed[col] <= 1).all())
                
        # Other columns use StandardScaler
        other_cols = [col for col in df_processed.columns if col not in price_cols]
        for col in other_cols:
            if df_processed[col].std() > 0:  # Skip constant columns
                self.assertAlmostEqual(df_processed[col].mean(), 0, places=1)
                self.assertAlmostEqual(df_processed[col].std(), 1, places=1)
            
    def test_prepare_data_for_model(self):
        """Test data preparation for model input."""
        # Test with DataFrame
        array_data = prepare_data_for_model(self.data)
        self.assertIsInstance(array_data, np.ndarray)
        self.assertEqual(array_data.shape[0], len(self.data))
        
        # Test with dict
        dict_data = {
            'open': self.data['open'].values,
            'high': self.data['high'].values,
            'low': self.data['low'].values,
            'close': self.data['close'].values,
            'volume': self.data['volume'].values
        }
        array_data = prepare_data_for_model(dict_data)
        self.assertIsInstance(array_data, np.ndarray)
        
        # Test with numpy array
        array_data = prepare_data_for_model(self.data.values)
        self.assertIsInstance(array_data, np.ndarray)
        
        # Test error handling
        with self.assertRaises(ValueError):
            prepare_data_for_model(None)
        with self.assertRaises(ValueError):
            prepare_data_for_model([])
            
    def test_clean_data_edge_cases(self):
        """Test data cleaning with edge cases."""
        # Test empty dataframe
        with self.assertRaises(ValueError):
            clean_data(self.empty_df)
            
        # Test invalid columns
        with self.assertRaises(KeyError):
            clean_data(self.invalid_df)
            
        # Test all missing values in a column
        df_all_missing = self.data.copy()
        df_all_missing['close'] = np.nan
        df_clean = clean_data(df_all_missing)
        self.assertFalse(df_clean['close'].isnull().any())
        
    def test_technical_features_edge_cases(self):
        """Test technical feature generation with edge cases."""
        # Test with very short data
        short_df = self.data.iloc[:5].copy()
        with self.assertRaises(ValueError):
            add_technical_features(short_df)
            
        # Test with zero prices
        zero_df = self.data.copy()
        zero_df.loc[:5, 'close'] = 0
        with self.assertRaises(ValueError):
            add_technical_features(zero_df)
            
        # Test with negative prices
        neg_df = self.data.copy()
        neg_df.loc[:5, 'close'] = -1
        with self.assertRaises(ValueError):
            add_technical_features(neg_df)
            
    def test_outlier_removal_edge_cases(self):
        """Test outlier removal with edge cases."""
        # Test with different thresholds in ascending order
        thresholds = [3.0, 2.0, 1.5]  # Stricter thresholds should remove more outliers
        next_rows = float('inf')
        
        for threshold in thresholds:
            df_no_outliers, _ = remove_outliers_zscore(self.data, threshold)
            self.assertLessEqual(len(df_no_outliers), next_rows,
                             "Stricter threshold should remove more outliers")
            next_rows = len(df_no_outliers)
            
        # Test with all outliers
        outlier_df = self.data.copy()
        outlier_df['close'] = outlier_df['close'] * 1000
        df_no_outliers, params = remove_outliers_zscore(outlier_df)
        self.assertGreater(len(df_no_outliers), 0, 
                          "Should keep some data even with all outliers")
            
    def test_normalize_features_edge_cases(self):
        """Test feature normalization with edge cases."""
        # Test with constant values
        const_df = self.data.copy()
        const_df['const'] = 5.0
        df_norm, scalers = normalize_features(const_df)
        self.assertTrue(np.allclose(df_norm['const'], 0))
        
        # Test with large values
        large_df = self.data.copy()
        large_df['large'] = large_df['close'] * 1e6
        df_norm, scalers = normalize_features(large_df)
        self.assertAlmostEqual(df_norm['large'].mean(), 0, places=1)
        self.assertAlmostEqual(df_norm['large'].std(), 1, places=1)
        
    def test_full_pipeline_edge_cases(self):
        """Test preprocessing pipeline with edge cases."""
        # Test without technical indicators
        df_no_ta, metadata = preprocess_data(
            self.data,
            use_ta=False,
            handle_missing=True,
            remove_outliers=True,
            normalize=True
        )
        self.assertNotIn('returns', df_no_ta.columns)
        
        # Test without outlier removal
        df_with_outliers, metadata = preprocess_data(
            self.data,
            use_ta=True,
            handle_missing=True,
            remove_outliers=False,
            normalize=True
        )
        self.assertNotIn('outlier_params', metadata)
        
        # Test without normalization
        df_no_norm, metadata = preprocess_data(
            self.data,
            use_ta=True,
            handle_missing=True,
            remove_outliers=True,
            normalize=False
        )
        self.assertNotIn('scalers', metadata)
        
        # Verify data consistency
        self.assertEqual(len(df_no_ta.columns), 5)  # Only OHLCV
        self.assertGreater(len(df_no_ta), len(df_with_outliers))  # Technical indicators reduce rows due to lookback period
        
    def test_advanced_cleaning_scenarios(self):
        """Test advanced data cleaning scenarios."""
        # Test extreme price movements
        df_extreme = self.data.copy()
        df_extreme.loc[90:95, 'close'] = df_extreme['close'].mean() * 10  # Huge spike
        df_clean = clean_data(df_extreme)
        self.assertTrue((df_clean['high'] >= df_clean['close']).all())
        self.assertTrue((df_clean['close'] >= df_clean['low']).all())
        
        # Test zero prices
        df_zeros = self.data.copy()
        df_zeros.loc[50:55, ['open', 'high', 'low', 'close']] = 0
        df_clean = clean_data(df_zeros)
        self.assertTrue((df_clean['close'] > 0).all())
        
        # Test negative prices
        df_negative = self.data.copy()
        df_negative.loc[60:65, 'low'] = -1
        df_clean = clean_data(df_negative)
        self.assertTrue((df_clean['low'] >= 0).all())
        
        # Test inverted price relationships
        df_inverted = self.data.copy()
        df_inverted.loc[70:75, 'high'] = df_inverted.loc[70:75, 'low'] * 0.5
        df_clean = clean_data(df_inverted)
        self.assertTrue((df_clean['high'] >= df_clean['low']).all())
        
    def test_extended_technical_features(self):
        """Test extended technical feature calculations."""
        df_clean = clean_data(self.data)
        df_tech = add_technical_features(df_clean)
        
        # Test return calculations
        self.assertTrue(np.isfinite(df_tech['returns']).all())
        self.assertTrue(np.isfinite(df_tech['log_returns']).all())
        
        # Test moving averages
        self.assertTrue((df_tech['volume_ma'] > 0).all())
        self.assertTrue((df_tech['volume_std'] >= 0).all())
        
        # Test candlestick features
        self.assertTrue((df_tech['upper_shadow'] >= 0).all())
        self.assertTrue((df_tech['lower_shadow'] >= 0).all())
        
    def test_sequential_preprocessing(self):
        """Test preprocessing steps in sequence."""
        # Test cleaning -> technical features
        df_clean = clean_data(self.data)
        df_tech = add_technical_features(df_clean)
        self.assertGreater(len(df_tech.columns), len(df_clean.columns))
        
        # Test cleaning -> outliers -> normalization
        df_clean = clean_data(self.data)
        df_no_outliers, _ = remove_outliers_zscore(df_clean)
        df_norm, _ = normalize_features(df_no_outliers)
        
        # Verify MinMaxScaler preserves price relationships
        self.assertTrue((df_norm['high'] >= df_norm['open']).all())
        self.assertTrue((df_norm['high'] >= df_norm['close']).all())
        self.assertTrue((df_norm['open'] >= df_norm['low']).all())
        self.assertTrue((df_norm['close'] >= df_norm['low']).all())
        
    def test_data_consistency(self):
        """Test data consistency across preprocessing steps."""
        # Initial cleaning
        df_clean = clean_data(self.data)
        row_count = len(df_clean)
        
        # Technical features (should drop some rows due to lookback)
        df_tech = add_technical_features(df_clean)
        self.assertLess(len(df_tech), row_count)
        self.assertTrue(df_tech.index.is_monotonic_increasing)
        
        # Outlier removal (should keep most data)
        df_no_outliers, params = remove_outliers_zscore(df_tech)
        self.assertGreater(len(df_no_outliers), len(df_tech) * 0.7)
        
        # Normalization (should preserve row count)
        df_norm, _ = normalize_features(df_no_outliers)
        self.assertEqual(len(df_norm), len(df_no_outliers))
        
    def test_error_propagation(self):
        """Test error handling and propagation."""
        # Test invalid threshold
        with self.assertRaises(ValueError):
            remove_outliers_zscore(self.data, threshold=-1)
            
        # Test insufficient data
        with self.assertRaises(ValueError):
            add_technical_features(self.data.iloc[:10])
            
        # Test missing required columns
        invalid_df = pd.DataFrame({
            'price': np.random.rand(100),
            'qty': np.random.rand(100)
        })
        with self.assertRaises(KeyError):
            clean_data(invalid_df)
            
        # Test non-numeric data
        str_df = self.data.copy()
        str_df['close'] = 'invalid'
        with self.assertRaises(TypeError):
            normalize_features(str_df)

    def test_metadata_generation(self):
        """Test metadata generation in preprocessing."""
        df_processed, metadata = preprocess_data(
            self.data,
            use_ta=True,
            handle_missing=True,
            remove_outliers=True,
            normalize=True
        )
        
        # Check metadata completeness
        required_keys = ['feature_names', 'preprocessing_steps', 'rows_processed']
        self.assertTrue(all(key in metadata for key in required_keys))
        
        # Verify preprocessing flags
        steps = metadata['preprocessing_steps']
        self.assertTrue(steps['use_ta'])
        self.assertTrue(steps['handle_missing'])
        self.assertTrue(steps['remove_outliers'])
        self.assertTrue(steps['normalize'])
        
        # Check data consistency
        self.assertEqual(len(df_processed), metadata['rows_processed'])
        self.assertEqual(len(df_processed.columns), len(metadata['feature_names']))

    def test_compute_log_returns(self):
        """Test log returns computation with various scenarios."""
        # Test normal case
        df = pd.DataFrame({
            'close': [100, 110, 105, 115, 120]
        })
        log_returns = preprocess_data.compute_log_returns(df)
        self.assertEqual(len(log_returns), len(df) - 1)  # First row will be NaN
        
        # Test with missing values
        df_missing = pd.DataFrame({
            'close': [100, None, 105, 115, 120]
        })
        with self.assertRaises(Exception):
            preprocess_data.compute_log_returns(df_missing)
        
        # Test with different column name
        df_custom = pd.DataFrame({
            'price': [100, 110, 105, 115, 120]
        })
        with self.assertRaises(KeyError):
            preprocess_data.compute_log_returns(df_custom)

    def test_detect_outliers(self):
        """Test outlier detection functionality."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 150, 101, 99, 98, 200]  # 150 and 200 are outliers
        })
        result = preprocess_data.detect_outliers(df, 'close', threshold=2.0)
        outliers = result[result['is_outlier']]['close'].tolist()
        self.assertEqual(len(outliers), 2)  # Should detect 2 outliers
        self.assertIn(150, outliers)
        self.assertIn(200, outliers)

    def test_winsorize_series(self):
        """Test winsorization of extreme values."""
        series = pd.Series([1, 2, 3, 100, 4, 5, 200, 6])  # 100 and 200 are extreme values
        limits = (0.1, 0.1)  # 10% on each side
        result = preprocess_data.winsorize_series(series, limits)
        
        # Check if extreme values were clipped
        self.assertLess(result.max(), 200)
        self.assertGreater(result.min(), 1)
        self.assertEqual(len(result), len(series))

    def test_preprocess_pipeline_edge_cases(self):
        """Test the complete preprocessing pipeline with edge cases."""
        # Test with all missing values in a column
        df_all_missing = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5),
            'close': [None] * 5
        })
        with self.assertRaises(Exception):
            preprocess_data.preprocess_pipeline(df_all_missing)

        # Test with negative values
        df_negative = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5),
            'close': [-100, -50, 100, 150, 200]
        })
        result = preprocess_data.preprocess_pipeline(df_negative)
        self.assertIn('log_return', result.columns)

        # Test with zero values (should handle log calculation properly)
        df_zero = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5),
            'close': [0, 100, 150, 200, 250]
        })
        with self.assertRaises(Exception):
            preprocess_data.preprocess_pipeline(df_zero)

if __name__ == '__main__':
    unittest.main()