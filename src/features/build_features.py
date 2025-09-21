"""
Feature engineering for electricity demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build features for electricity demand forecasting."""
    
    def __init__(self):
        """Initialize feature builder."""
        self.scalers = {}
        self.encoders = {}
        
    def add_time_features(self, df: pd.DataFrame, 
                         timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        
        if timestamp_col not in df.columns:
            logger.error(f"Timestamp column '{timestamp_col}' not found")
            return df
            
        # Ensure datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['year'] = df[timestamp_col].dt.year
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofyear'] = df[timestamp_col].dt.dayofyear
        df['quarter'] = df[timestamp_col].dt.quarter
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Peak hours (7-9 AM, 5-8 PM)
        df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        
        return df
        
    def add_lag_features(self, df: pd.DataFrame, 
                        target_col: str,
                        lags: List[int] = [1, 2, 3, 24, 48, 168]) -> pd.DataFrame:
        """
        Add lag features for time series.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found")
            return df
            
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        return df
        
    def add_rolling_features(self, df: pd.DataFrame,
                           target_col: str,
                           windows: List[int] = [3, 6, 12, 24, 48]) -> pd.DataFrame:
        """
        Add rolling window features.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found")
            return df
            
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).std()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).max()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).min()
                
        return df
        
    def add_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather interaction features.
        
        Args:
            df: Input DataFrame with weather features
            
        Returns:
            DataFrame with weather interaction features
        """
        df = df.copy()
        
        # Temperature interactions with time
        if 'temp' in df.columns and 'hour' in df.columns:
            df['temp_hour_interaction'] = df['temp'] * df['hour']
            
        # Cooling/heating degree interactions with weekend
        if 'cdd' in df.columns and 'is_weekend' in df.columns:
            df['cdd_weekend_interaction'] = df['cdd'] * df['is_weekend']
        if 'hdd' in df.columns and 'is_weekend' in df.columns:
            df['hdd_weekend_interaction'] = df['hdd'] * df['is_weekend']
            
        return df
        
    def build_features(self, df: pd.DataFrame,
                      target_col: str = 'demand',
                      timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Build all features for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            timestamp_col: Timestamp column name
            
        Returns:
            DataFrame with all features
        """
        logger.info("Building features...")
        
        # Add time features
        df = self.add_time_features(df, timestamp_col)
        
        # Add lag features
        df = self.add_lag_features(df, target_col)
        
        # Add rolling features
        df = self.add_rolling_features(df, target_col)
        
        # Add weather interactions if weather data exists
        if any(col in df.columns for col in ['temp', 'cdd', 'hdd']):
            df = self.add_weather_interactions(df)
            
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df