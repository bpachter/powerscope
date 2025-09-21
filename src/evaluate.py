"""
Evaluation script for PowerScope electricity demand forecasting models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional

from src.utils.io import load_config, load_pickle, save_json, load_csv
from src.data.iso_local_csv import ISOLocalCSVLoader
from src.data.weather_meteostat import WeatherMeteostatLoader
from src.features.build_features import FeatureBuilder
from src.models.lightgbm_quantile import LightGBMQuantileRegressor
from src.models.torch_models import TorchQuantileTrainer
from src.utils.metrics import create_metrics_report, QuantileMetrics
from src.calibration.conformal import ConformalPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load test data for evaluation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Test DataFrame
    """
    logger.info("Loading test data...")
    
    # For now, use the same data loading logic as training
    # In practice, this would load a separate test set
    iso_loader = ISOLocalCSVLoader(config['data']['iso']['source_path'])
    demand_df = iso_loader.load_demand_data()
    
    if demand_df.empty:
        logger.warning("No ISO demand data loaded, creating sample data")
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='h')
        demand_df = pd.DataFrame({
            'timestamp': dates,
            'demand': np.random.normal(1000, 200, len(dates)) + 
                     100 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
        })
    
    # Take last 20% as test data
    test_size = int(len(demand_df) * 0.2)
    test_df = demand_df.tail(test_size).copy()
    
    return test_df


def load_trained_model(model_type: str, config: Dict[str, Any]):
    """
    Load trained model.
    
    Args:
        model_type: Type of model ('lightgbm' or 'torch')
        config: Configuration dictionary
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading {model_type} model...")
    
    if model_type == 'lightgbm':
        model_path = "artifacts/models/lightgbm_model.pkl"
        if Path(model_path).exists():
            return load_pickle(model_path)
        else:
            # Try to load individual quantile models
            quantiles = config['forecast']['quantiles']
            model = LightGBMQuantileRegressor(quantiles=quantiles)
            try:
                model.load_models("artifacts/models/lightgbm_quantile_{quantile}.txt")
                return model
            except Exception as e:
                logger.error(f"Could not load LightGBM model: {e}")
                return None
                
    elif model_type == 'torch':
        model_path = "artifacts/models/torch_trainer.pkl"
        if Path(model_path).exists():
            return load_pickle(model_path)
        else:
            logger.error(f"Could not find PyTorch model at {model_path}")
            return None
    
    return None


def evaluate_model_performance(model, X_test: pd.DataFrame, y_test: pd.Series,
                             model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_type: Type of model
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating {model_type} model performance...")
    
    if model_type == 'lightgbm':
        predictions = model.predict(X_test)
        quantile_preds = predictions
        
        # Get median prediction
        median_pred = predictions.get(0.5)
        if median_pred is None:
            median_pred = list(predictions.values())[len(predictions)//2]
            
    elif model_type == 'torch':
        X_test_np = X_test.values.astype(np.float32)
        pred_array = model.predict(X_test_np)
        
        quantiles = config['forecast']['quantiles']
        quantile_preds = {q: pred_array[:, i] for i, q in enumerate(quantiles)}
        
        # Get median prediction
        median_pred = quantile_preds.get(0.5)
        if median_pred is None:
            median_pred = list(quantile_preds.values())[len(quantile_preds)//2]
    
    # Create comprehensive metrics report
    report = create_metrics_report(
        y_true=y_test.values,
        y_pred=median_pred,
        quantile_predictions=quantile_preds,
        timestamps=X_test.index if hasattr(X_test.index, 'hour') else None
    )
    
    # Add model-specific information
    report['model_type'] = model_type
    report['test_size'] = len(y_test)
    report['predictions'] = {
        'median': median_pred.tolist() if hasattr(median_pred, 'tolist') else median_pred,
        'quantiles': {str(k): v.tolist() if hasattr(v, 'tolist') else v 
                     for k, v in quantile_preds.items()}
    }
    
    return report


def evaluate_conformal_prediction(model, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate conformal prediction intervals.
    
    Args:
        model: Trained model
        X_train: Training features (for calibration)
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_type: Type of model
        config: Configuration dictionary
        
    Returns:
        Conformal prediction results
    """
    logger.info("Evaluating conformal prediction...")
    
    # Get point predictions for calibration
    if model_type == 'lightgbm':
        train_preds = model.predict(X_train)
        median_train_pred = train_preds.get(0.5, list(train_preds.values())[0])
        
        test_preds = model.predict(X_test)
        median_test_pred = test_preds.get(0.5, list(test_preds.values())[0])
        
    elif model_type == 'torch':
        X_train_np = X_train.values.astype(np.float32)
        train_pred_array = model.predict(X_train_np)
        median_idx = len(config['forecast']['quantiles']) // 2
        median_train_pred = train_pred_array[:, median_idx]
        
        X_test_np = X_test.values.astype(np.float32)
        test_pred_array = model.predict(X_test_np)
        median_test_pred = test_pred_array[:, median_idx]
    
    # Calibrate conformal predictor
    conformal = ConformalPredictor(alpha=0.1)  # 90% coverage
    conformal.calibrate(y_train.values, median_train_pred)
    
    # Generate conformal intervals
    lower, upper = conformal.predict(median_test_pred)
    
    # Calculate coverage and width
    coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
    avg_width = np.mean(upper - lower)
    normalized_width = avg_width / (np.max(y_test.values) - np.min(y_test.values))
    
    conformal_results = {
        'coverage_probability': coverage,
        'average_width': avg_width,
        'normalized_width': normalized_width,
        'expected_coverage': 0.9,
        'coverage_gap': abs(coverage - 0.9)
    }
    
    logger.info(f"Conformal prediction coverage: {coverage:.3f} (expected: 0.9)")
    logger.info(f"Average interval width: {avg_width:.2f}")
    
    return conformal_results


def create_evaluation_plots(y_test: pd.Series, predictions: Dict[str, Any],
                          model_type: str, save_path: str = "artifacts/reports") -> None:
    """
    Create evaluation plots.
    
    Args:
        y_test: Test targets
        predictions: Model predictions
        model_type: Type of model
        save_path: Path to save plots
    """
    logger.info("Creating evaluation plots...")
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_type.upper()} Model Evaluation', fontsize=16)
    
    # Plot 1: Actual vs Predicted
    median_pred = predictions['median']
    axes[0, 0].scatter(y_test.values, median_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Demand')
    axes[0, 0].set_ylabel('Predicted Demand')
    axes[0, 0].set_title('Actual vs Predicted')
    
    # Plot 2: Time series plot (last 168 hours)
    n_show = min(168, len(y_test))
    x_time = range(n_show)
    axes[0, 1].plot(x_time, y_test.values[-n_show:], label='Actual', linewidth=2)
    axes[0, 1].plot(x_time, median_pred[-n_show:], label='Predicted', linewidth=2)
    
    # Add quantile bands if available
    if 'quantiles' in predictions and len(predictions['quantiles']) >= 2:
        quantiles = sorted(predictions['quantiles'].keys())
        lower_q = quantiles[0]
        upper_q = quantiles[-1]
        
        lower_pred = predictions['quantiles'][lower_q]
        upper_pred = predictions['quantiles'][upper_q]
        
        axes[0, 1].fill_between(x_time, lower_pred[-n_show:], upper_pred[-n_show:], 
                               alpha=0.3, label=f'{lower_q}-{upper_q} Interval')
    
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Demand')
    axes[0, 1].set_title('Time Series Forecast (Last 168 Hours)')
    axes[0, 1].legend()
    
    # Plot 3: Residuals
    residuals = y_test.values - median_pred
    axes[1, 0].scatter(median_pred, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Demand')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot')
    
    # Plot 4: Residual histogram
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    
    plt.tight_layout()
    plot_path = save_path / f'{model_type}_evaluation_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Evaluation plots saved to {plot_path}")
    plt.close()


def create_feature_importance_plot(model, model_type: str, save_path: str = "artifacts/reports") -> None:
    """
    Create feature importance plot for LightGBM model.
    
    Args:
        model: Trained model
        model_type: Type of model
        save_path: Path to save plots
    """
    if model_type != 'lightgbm':
        return
        
    logger.info("Creating feature importance plot...")
    
    try:
        importance_dfs = model.get_feature_importance()
        
        # Plot importance for median quantile
        median_importance = importance_dfs.get(0.5)
        if median_importance is not None:
            plt.figure(figsize=(10, 8))
            top_features = median_importance.head(20)
            
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importance (Median Quantile)')
            plt.gca().invert_yaxis()
            
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / 'feature_importance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {plot_path}")
            plt.close()
            
    except Exception as e:
        logger.warning(f"Could not create feature importance plot: {e}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate PowerScope forecasting models')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['lightgbm', 'torch', 'both'], 
                       default='both', help='Model type to evaluate')
    parser.add_argument('--conformal', action='store_true',
                       help='Evaluate conformal prediction intervals')
    parser.add_argument('--plots', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load test data
    test_df = load_test_data(config)
    
    # Engineer features
    feature_builder = FeatureBuilder()
    test_df_features = feature_builder.build_features(test_df, 'demand', 'timestamp')
    
    # Prepare test data
    test_df_features = test_df_features.dropna(subset=['demand'])
    feature_cols = [col for col in test_df_features.columns if col not in ['demand', 'timestamp']]
    test_df_features[feature_cols] = test_df_features[feature_cols].fillna(method='ffill').fillna(0)
    
    X_test = test_df_features[feature_cols]
    y_test = test_df_features['demand']
    
    # For conformal evaluation, we need training data too
    X_train, y_train = None, None
    if args.conformal:
        # Load training data (simplified - in practice load actual training set)
        train_size = int(len(test_df_features) * 0.5)  # Use first half as "training"
        X_train = X_test.head(train_size)
        y_train = y_test.head(train_size)
        X_test = X_test.tail(len(X_test) - train_size)
        y_test = y_test.tail(len(y_test) - train_size)
    
    results = {}
    
    # Evaluate models
    if args.model in ['lightgbm', 'both']:
        lightgbm_model = load_trained_model('lightgbm', config)
        if lightgbm_model:
            results['lightgbm'] = evaluate_model_performance(
                lightgbm_model, X_test, y_test, 'lightgbm', config
            )
            
            if args.conformal and X_train is not None:
                results['lightgbm']['conformal'] = evaluate_conformal_prediction(
                    lightgbm_model, X_train, y_train, X_test, y_test, 'lightgbm', config
                )
            
            if args.plots:
                create_evaluation_plots(y_test, results['lightgbm']['predictions'], 'lightgbm')
                create_feature_importance_plot(lightgbm_model, 'lightgbm')
    
    if args.model in ['torch', 'both']:
        torch_model = load_trained_model('torch', config)
        if torch_model:
            results['torch'] = evaluate_model_performance(
                torch_model, X_test, y_test, 'torch', config
            )
            
            if args.conformal and X_train is not None:
                results['torch']['conformal'] = evaluate_conformal_prediction(
                    torch_model, X_train, y_train, X_test, y_test, 'torch', config
                )
            
            if args.plots:
                create_evaluation_plots(y_test, results['torch']['predictions'], 'torch')
    
    # Save results
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove predictions from results before saving (too large)
    results_summary = {}
    for model_name, model_results in results.items():
        results_summary[model_name] = {k: v for k, v in model_results.items() if k != 'predictions'}
    
    save_json(results_summary, "artifacts/reports/evaluation_results.json")
    logger.info("Evaluation results saved to artifacts/reports/evaluation_results.json")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} Model:")
        print("-" * 20)
        
        if 'overall' in model_results:
            for metric, value in model_results['overall'].items():
                print(f"  {metric:20s}: {value:8.4f}")
        
        if 'quantile' in model_results:
            print("  Quantile Metrics:")
            for metric, value in model_results['quantile'].items():
                print(f"    {metric:18s}: {value:8.4f}")
        
        if 'conformal' in model_results:
            print("  Conformal Prediction:")
            for metric, value in model_results['conformal'].items():
                print(f"    {metric:18s}: {value:8.4f}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()