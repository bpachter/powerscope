"""
Evaluation metrics for electricity demand forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value
    """
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return np.inf
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray,
                                           y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE value
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    
    if not mask.any():
        return 0.0
        
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def mean_absolute_scaled_error(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_train: np.ndarray,
                              seasonality: int = 1) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonality: Seasonal period
        
    Returns:
        MASE value
    """
    # Calculate naive forecast MAE
    if len(y_train) <= seasonality:
        naive_mae = np.mean(np.abs(np.diff(y_train)))
    else:
        naive_forecast = y_train[:-seasonality]
        naive_mae = np.mean(np.abs(y_train[seasonality:] - naive_forecast))
    
    if naive_mae == 0:
        return np.inf
        
    mae = mean_absolute_error(y_true, y_pred)
    return mae / naive_mae


def quantile_loss(y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 quantile: float) -> float:
    """
    Calculate quantile loss (pinball loss).
    
    Args:
        y_true: True values
        y_pred: Predicted quantile values
        quantile: Quantile level (0-1)
        
    Returns:
        Quantile loss
    """
    error = y_true - y_pred
    loss = np.where(error >= 0, 
                   quantile * error,
                   (quantile - 1) * error)
    return np.mean(loss)


def coverage_probability(y_true: np.ndarray,
                        y_lower: np.ndarray,
                        y_upper: np.ndarray) -> float:
    """
    Calculate coverage probability for prediction intervals.
    
    Args:
        y_true: True values
        y_lower: Lower bound predictions
        y_upper: Upper bound predictions
        
    Returns:
        Coverage probability (0-1)
    """
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(covered)


def interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate average prediction interval width.
    
    Args:
        y_lower: Lower bound predictions
        y_upper: Upper bound predictions
        
    Returns:
        Average interval width
    """
    return np.mean(y_upper - y_lower)


def normalized_interval_width(y_true: np.ndarray,
                             y_lower: np.ndarray,
                             y_upper: np.ndarray) -> float:
    """
    Calculate normalized prediction interval width.
    
    Args:
        y_true: True values
        y_lower: Lower bound predictions
        y_upper: Upper bound predictions
        
    Returns:
        Normalized interval width
    """
    width = interval_width(y_lower, y_upper)
    range_true = np.max(y_true) - np.min(y_true)
    
    if range_true == 0:
        return np.inf
        
    return width / range_true


def interval_score(y_true: np.ndarray,
                  y_lower: np.ndarray,
                  y_upper: np.ndarray,
                  alpha: float = 0.1) -> float:
    """
    Calculate interval score for prediction intervals.
    
    Args:
        y_true: True values
        y_lower: Lower bound predictions
        y_upper: Upper bound predictions
        alpha: Miscoverage level (e.g., 0.1 for 90% intervals)
        
    Returns:
        Interval score
    """
    width = y_upper - y_lower
    lower_penalty = (2 / alpha) * (y_lower - y_true) * (y_true < y_lower)
    upper_penalty = (2 / alpha) * (y_true - y_upper) * (y_true > y_upper)
    
    return np.mean(width + lower_penalty + upper_penalty)


class QuantileMetrics:
    """Metrics for quantile regression evaluation."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize quantile metrics.
        
        Args:
            quantiles: List of quantiles to evaluate
        """
        self.quantiles = quantiles
        
    def evaluate_quantiles(self, 
                          y_true: np.ndarray,
                          quantile_predictions: Dict[float, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate quantile predictions.
        
        Args:
            y_true: True values
            quantile_predictions: Dictionary mapping quantiles to predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Individual quantile losses
        for quantile in self.quantiles:
            if quantile in quantile_predictions:
                y_pred = quantile_predictions[quantile]
                loss = quantile_loss(y_true, y_pred, quantile)
                metrics[f'quantile_loss_{quantile}'] = loss
                
                # Additional metrics for median (0.5 quantile)
                if quantile == 0.5:
                    metrics['mae'] = mean_absolute_error(y_true, y_pred)
                    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
                    metrics['smape'] = symmetric_mean_absolute_percentage_error(y_true, y_pred)
                    metrics['r2'] = r2_score(y_true, y_pred)
        
        # Interval metrics if we have upper and lower bounds
        if len(self.quantiles) >= 2:
            lower_q = min(self.quantiles)
            upper_q = max(self.quantiles)
            
            if lower_q in quantile_predictions and upper_q in quantile_predictions:
                y_lower = quantile_predictions[lower_q]
                y_upper = quantile_predictions[upper_q]
                
                metrics['coverage_probability'] = coverage_probability(y_true, y_lower, y_upper)
                metrics['interval_width'] = interval_width(y_lower, y_upper)
                metrics['normalized_interval_width'] = normalized_interval_width(y_true, y_lower, y_upper)
                
                alpha = upper_q - lower_q
                metrics['interval_score'] = interval_score(y_true, y_lower, y_upper, 1 - alpha)
                
        return metrics


class ForecastMetrics:
    """Comprehensive metrics for forecasting evaluation."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all standard forecasting metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data (for MASE calculation)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'smape': symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Add MASE if training data provided
        if y_train is not None:
            metrics['mase'] = mean_absolute_scaled_error(y_true, y_pred, y_train)
            
        return metrics
        
    @staticmethod
    def evaluate_by_time_of_day(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               timestamps: pd.DatetimeIndex) -> Dict[int, Dict[str, float]]:
        """
        Evaluate metrics by time of day.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Datetime index
            
        Returns:
            Dictionary mapping hours to metrics
        """
        results = {}
        
        for hour in range(24):
            hour_mask = timestamps.hour == hour
            
            if hour_mask.sum() > 0:
                y_true_hour = y_true[hour_mask]
                y_pred_hour = y_pred[hour_mask]
                
                results[hour] = ForecastMetrics.calculate_all_metrics(y_true_hour, y_pred_hour)
                
        return results
        
    @staticmethod
    def evaluate_by_day_of_week(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               timestamps: pd.DatetimeIndex) -> Dict[int, Dict[str, float]]:
        """
        Evaluate metrics by day of week.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Datetime index
            
        Returns:
            Dictionary mapping day of week to metrics
        """
        results = {}
        
        for day in range(7):
            day_mask = timestamps.dayofweek == day
            
            if day_mask.sum() > 0:
                y_true_day = y_true[day_mask]
                y_pred_day = y_pred[day_mask]
                
                results[day] = ForecastMetrics.calculate_all_metrics(y_true_day, y_pred_day)
                
        return results


def create_metrics_report(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         quantile_predictions: Optional[Dict[float, np.ndarray]] = None,
                         timestamps: Optional[pd.DatetimeIndex] = None,
                         y_train: Optional[np.ndarray] = None) -> Dict[str, any]:
    """
    Create comprehensive metrics report.
    
    Args:
        y_true: True values
        y_pred: Point predictions
        quantile_predictions: Quantile predictions
        timestamps: Datetime index
        y_train: Training data
        
    Returns:
        Comprehensive metrics report
    """
    report = {}
    
    # Overall metrics
    report['overall'] = ForecastMetrics.calculate_all_metrics(y_true, y_pred, y_train)
    
    # Quantile metrics
    if quantile_predictions:
        quantile_metrics = QuantileMetrics()
        report['quantile'] = quantile_metrics.evaluate_quantiles(y_true, quantile_predictions)
    
    # Time-based metrics
    if timestamps is not None:
        report['by_hour'] = ForecastMetrics.evaluate_by_time_of_day(y_true, y_pred, timestamps)
        report['by_day_of_week'] = ForecastMetrics.evaluate_by_day_of_week(y_true, y_pred, timestamps)
    
    return report