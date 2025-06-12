import pandas as pd
import numpy as np

class FeatureEngineer:
    """Simplified feature engineering for tests."""
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ['open', 'high', 'low', 'close']:
            if col in df:
                df[f'{col}_ma'] = df[col].rolling(window=3, min_periods=1).mean()
        return df.fillna(method='ffill').fillna(0)
