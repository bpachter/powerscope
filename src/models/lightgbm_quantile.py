"""
LightGBM quantile regression model for electricity demand forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import logging

logger = logging.getLogger(__name__)


class LightGBMQuantileRegressor:
    """LightGBM model for quantile regression."""
    
    def __init__(self, 
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 **lgb_params):
        """
        Initialize LightGBM quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict
            **lgb_params: Additional LightGBM parameters
        """
        self.quantiles = quantiles
        self.models = {}
        
        # Default LightGBM parameters
        self.lgb_params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.lgb_params.update(lgb_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            num_boost_round: int = 100) -> None:
        """
        Fit quantile regression models.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation target
            num_boost_round: Number of boosting rounds
        """
        logger.info(f"Training LightGBM models for quantiles: {self.quantiles}")
        
        for quantile in self.quantiles:
            logger.info(f"Training quantile {quantile}")
            
            # Set quantile-specific parameters
            params = self.lgb_params.copy()
            params['alpha'] = quantile
            
            # Create datasets
            train_data = lgb.Dataset(X, label=y)
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(val_data)
                valid_names.append('valid')
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
            )
            
            self.models[quantile] = model
            
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Predict quantiles.
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary mapping quantiles to predictions
        """
        predictions = {}
        
        for quantile in self.quantiles:
            if quantile not in self.models:
                raise ValueError(f"Model for quantile {quantile} not trained")
                
            pred = self.models[quantile].predict(X)
            predictions[quantile] = pred
            
        return predictions
        
    def predict_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantile intervals as DataFrame.
        
        Args:
            X: Features for prediction
            
        Returns:
            DataFrame with quantile predictions
        """
        predictions = self.predict(X)
        
        result_df = pd.DataFrame(index=X.index)
        for quantile, pred in predictions.items():
            result_df[f'quantile_{quantile}'] = pred
            
        return result_df
        
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for each quantile model.
        
        Args:
            importance_type: Type of importance ('gain', 'split', etc.)
            
        Returns:
            Dictionary mapping quantiles to feature importance DataFrames
        """
        importance_dfs = {}
        
        for quantile, model in self.models.items():
            importance = model.feature_importance(importance_type=importance_type)
            feature_names = model.feature_name()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            importance_dfs[quantile] = importance_df
            
        return importance_dfs
        
    def save_models(self, filepath_template: str) -> None:
        """
        Save trained models.
        
        Args:
            filepath_template: Template for model filepaths (should contain {quantile})
        """
        for quantile, model in self.models.items():
            filepath = filepath_template.format(quantile=quantile)
            model.save_model(filepath)
            logger.info(f"Saved model for quantile {quantile} to {filepath}")
            
    def load_models(self, filepath_template: str) -> None:
        """
        Load trained models.
        
        Args:
            filepath_template: Template for model filepaths (should contain {quantile})
        """
        for quantile in self.quantiles:
            filepath = filepath_template.format(quantile=quantile)
            model = lgb.Booster(model_file=filepath)
            self.models[quantile] = model
            logger.info(f"Loaded model for quantile {quantile} from {filepath}")
            
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv_folds: int = 5) -> Dict[float, float]:
        """
        Perform cross-validation for all quantiles.
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary mapping quantiles to CV scores
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error
        
        cv_scores = {q: [] for q in self.quantiles}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"CV Fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit models for this fold
            temp_models = {}
            for quantile in self.quantiles:
                params = self.lgb_params.copy()
                params['alpha'] = quantile
                
                train_data = lgb.Dataset(X_train, label=y_train)
                model = lgb.train(params, train_data, num_boost_round=100, verbose_eval=False)
                temp_models[quantile] = model
                
                # Predict and score
                pred = model.predict(X_val)
                score = mean_absolute_error(y_val, pred)
                cv_scores[quantile].append(score)
                
        # Average scores
        avg_scores = {q: np.mean(scores) for q, scores in cv_scores.items()}
        
        return avg_scores