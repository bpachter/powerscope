"""
Rolling window utilities for time series analysis.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class RollingWindow:
    """Utilities for rolling window operations on time series data."""
    
    def __init__(self, window_size: int, min_periods: Optional[int] = None):
        """
        Initialize rolling window utility.
        
        Args:
            window_size: Size of the rolling window
            min_periods: Minimum number of observations required to have a value
        """
        self.window_size = window_size
        self.min_periods = min_periods or 1
        
    def rolling_mean(self, series: pd.Series) -> pd.Series:
        """
        Calculate rolling mean.
        
        Args:
            series: Input time series
            
        Returns:
            Rolling mean series
        """
        return series.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).mean()
        
    def rolling_std(self, series: pd.Series) -> pd.Series:
        """
        Calculate rolling standard deviation.
        
        Args:
            series: Input time series
            
        Returns:
            Rolling standard deviation series
        """
        return series.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).std()
        
    def rolling_min(self, series: pd.Series) -> pd.Series:
        """
        Calculate rolling minimum.
        
        Args:
            series: Input time series
            
        Returns:
            Rolling minimum series
        """
        return series.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).min()
        
    def rolling_max(self, series: pd.Series) -> pd.Series:
        """
        Calculate rolling maximum.
        
        Args:
            series: Input time series
            
        Returns:
            Rolling maximum series
        """
        return series.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).max()
        
    def rolling_quantile(self, series: pd.Series, quantile: float) -> pd.Series:
        """
        Calculate rolling quantile.
        
        Args:
            series: Input time series
            quantile: Quantile to calculate (0-1)
            
        Returns:
            Rolling quantile series
        """
        return series.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).quantile(quantile)
        
    def rolling_sum(self, series: pd.Series) -> pd.Series:
        """
        Calculate rolling sum.
        
        Args:
            series: Input time series
            
        Returns:
            Rolling sum series
        """
        return series.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).sum()
        
    def rolling_correlation(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Calculate rolling correlation between two series.
        
        Args:
            series1: First time series
            series2: Second time series
            
        Returns:
            Rolling correlation series
        """
        return series1.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).corr(series2)


class ExpandingWindow:
    """Utilities for expanding window operations."""
    
    def __init__(self, min_periods: int = 1):
        """
        Initialize expanding window utility.
        
        Args:
            min_periods: Minimum number of observations required to have a value
        """
        self.min_periods = min_periods
        
    def expanding_mean(self, series: pd.Series) -> pd.Series:
        """Calculate expanding mean."""
        return series.expanding(min_periods=self.min_periods).mean()
        
    def expanding_std(self, series: pd.Series) -> pd.Series:
        """Calculate expanding standard deviation."""
        return series.expanding(min_periods=self.min_periods).std()
        
    def expanding_min(self, series: pd.Series) -> pd.Series:
        """Calculate expanding minimum."""
        return series.expanding(min_periods=self.min_periods).min()
        
    def expanding_max(self, series: pd.Series) -> pd.Series:
        """Calculate expanding maximum."""
        return series.expanding(min_periods=self.min_periods).max()


class SeasonalRolling:
    """Seasonal rolling window operations."""
    
    def __init__(self, season_length: int, window_size: int):
        """
        Initialize seasonal rolling utility.
        
        Args:
            season_length: Length of seasonal period (e.g., 24 for hourly data)
            window_size: Number of seasonal periods to include
        """
        self.season_length = season_length
        self.window_size = window_size
        
    def seasonal_mean(self, series: pd.Series) -> pd.Series:
        """
        Calculate seasonal rolling mean.
        
        Args:
            series: Input time series
            
        Returns:
            Seasonal rolling mean
        """
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            # Get seasonal indices
            seasonal_indices = []
            for j in range(self.window_size):
                idx = i - j * self.season_length
                if idx >= 0:
                    seasonal_indices.append(idx)
                    
            if seasonal_indices:
                result.iloc[i] = series.iloc[seasonal_indices].mean()
                
        return result
        
    def seasonal_std(self, series: pd.Series) -> pd.Series:
        """
        Calculate seasonal rolling standard deviation.
        
        Args:
            series: Input time series
            
        Returns:
            Seasonal rolling standard deviation
        """
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            seasonal_indices = []
            for j in range(self.window_size):
                idx = i - j * self.season_length
                if idx >= 0:
                    seasonal_indices.append(idx)
                    
            if len(seasonal_indices) > 1:
                result.iloc[i] = series.iloc[seasonal_indices].std()
                
        return result


def create_rolling_features(df: pd.DataFrame,
                          target_col: str,
                          windows: List[int] = [3, 6, 12, 24, 48, 168],
                          functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create multiple rolling window features.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        windows: List of window sizes
        functions: List of functions to apply
        
    Returns:
        DataFrame with rolling features added
    """
    result_df = df.copy()
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return result_df
        
    for window in windows:
        rolling_util = RollingWindow(window)
        
        for func in functions:
            if hasattr(rolling_util, f'rolling_{func}'):
                method = getattr(rolling_util, f'rolling_{func}')
                feature_name = f'{target_col}_rolling_{func}_{window}'
                result_df[feature_name] = method(df[target_col])
                
    return result_df


def create_lag_features(df: pd.DataFrame,
                       target_col: str,
                       lags: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]) -> pd.DataFrame:
    """
    Create lag features.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features added
    """
    result_df = df.copy()
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return result_df
        
    for lag in lags:
        feature_name = f'{target_col}_lag_{lag}'
        result_df[feature_name] = df[target_col].shift(lag)
        
    return result_df


def create_diff_features(df: pd.DataFrame,
                        target_col: str,
                        periods: List[int] = [1, 24, 168]) -> pd.DataFrame:
    """
    Create differencing features.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        periods: List of differencing periods
        
    Returns:
        DataFrame with difference features added
    """
    result_df = df.copy()
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return result_df
        
    for period in periods:
        feature_name = f'{target_col}_diff_{period}'
        result_df[feature_name] = df[target_col].diff(periods=period)
        
    return result_df


def create_pct_change_features(df: pd.DataFrame,
                              target_col: str,
                              periods: List[int] = [1, 24, 168]) -> pd.DataFrame:
    """
    Create percentage change features.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        periods: List of periods for percentage change
        
    Returns:
        DataFrame with percentage change features added
    """
    result_df = df.copy()
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return result_df
        
    for period in periods:
        feature_name = f'{target_col}_pct_change_{period}'
        result_df[feature_name] = df[target_col].pct_change(periods=period)
        
    return result_df


class RollingStatistics:
    """Advanced rolling statistics for time series."""
    
    @staticmethod
    def rolling_zscore(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
        """
        Calculate rolling z-score.
        
        Args:
            series: Input series
            window: Window size
            min_periods: Minimum periods
            
        Returns:
            Rolling z-score
        """
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        
        return (series - rolling_mean) / rolling_std
        
    @staticmethod
    def rolling_rank(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
        """
        Calculate rolling rank (percentile within window).
        
        Args:
            series: Input series
            window: Window size
            min_periods: Minimum periods
            
        Returns:
            Rolling rank (0-1)
        """
        def rank_within_window(x):
            if len(x) < min_periods:
                return np.nan
            return (x.iloc[-1] <= x).mean()
            
        return series.rolling(window=window, min_periods=min_periods).apply(rank_within_window)
        
    @staticmethod
    def rolling_skewness(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
        """
        Calculate rolling skewness.
        
        Args:
            series: Input series
            window: Window size
            min_periods: Minimum periods
            
        Returns:
            Rolling skewness
        """
        return series.rolling(window=window, min_periods=min_periods).skew()
        
    @staticmethod
    def rolling_kurtosis(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
        """
        Calculate rolling kurtosis.
        
        Args:
            series: Input series
            window: Window size
            min_periods: Minimum periods
            
        Returns:
            Rolling kurtosis
        """
        return series.rolling(window=window, min_periods=min_periods).kurt()


def create_comprehensive_rolling_features(df: pd.DataFrame,
                                        target_col: str,
                                        short_windows: List[int] = [3, 6, 12],
                                        long_windows: List[int] = [24, 48, 168]) -> pd.DataFrame:
    """
    Create comprehensive set of rolling features.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        short_windows: Short-term windows
        long_windows: Long-term windows
        
    Returns:
        DataFrame with comprehensive rolling features
    """
    result_df = df.copy()
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return result_df
        
    # Basic rolling features
    all_windows = short_windows + long_windows
    result_df = create_rolling_features(result_df, target_col, all_windows)
    
    # Lag features
    lags = [1, 2, 3] + [w for w in long_windows if w <= 168]
    result_df = create_lag_features(result_df, target_col, lags)
    
    # Difference features
    result_df = create_diff_features(result_df, target_col, [1, 24, 168])
    
    # Percentage change features
    result_df = create_pct_change_features(result_df, target_col, [1, 24, 168])
    
    # Advanced rolling statistics for key windows
    for window in [24, 168]:  # Daily and weekly
        col_name = f'{target_col}_zscore_{window}'
        result_df[col_name] = RollingStatistics.rolling_zscore(df[target_col], window)
        
        col_name = f'{target_col}_rank_{window}'
        result_df[col_name] = RollingStatistics.rolling_rank(df[target_col], window)
        
    return result_df