"""
Conformal prediction methods for uncertainty quantification.
"""

import numpy as np
from typing import Tuple, Optional


class ConformalPredictor:
    """Conformal prediction for uncertainty quantification."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.quantile = None
        
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Calibrate conformal predictor on validation data.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        scores = np.abs(y_true - y_pred)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(scores, q_level)
        
    def predict(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate conformal prediction intervals.
        
        Args:
            y_pred: Point predictions
            
        Returns:
            Lower and upper bounds of prediction intervals
        """
        if self.quantile is None:
            raise ValueError("Must calibrate predictor first")
            
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        
        return lower, upper