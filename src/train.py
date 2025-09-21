"""
Training script for PowerScope electricity demand forecasting models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

from src.utils.io import load_config, save_pickle, load_csv
from src.data.iso_local_csv import ISOLocalCSVLoader
from src.data.weather_meteostat import WeatherMeteostatLoader
from src.features.build_features import FeatureBuilder
from src.models.lightgbm_quantile import LightGBMQuantileRegressor
from src.models.torch_models import TorchQuantileTrainer, DeepQuantileRegressor
from src.utils.metrics import QuantileMetrics, create_metrics_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and prepare training data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Prepared DataFrame
    """
    logger.info("Loading data...")
    
    # Load ISO data
    iso_loader = ISOLocalCSVLoader(config['data']['iso']['source_path'])
    demand_df = iso_loader.load_demand_data()
    
    if demand_df.empty:
        logger.warning("No ISO demand data loaded, creating sample data")
        # Create sample data for demonstration
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='h')
        demand_df = pd.DataFrame({
            'timestamp': dates,
            'demand': np.random.normal(1000, 200, len(dates)) + 
                     100 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Daily pattern
        })
    
    # Load weather data if available
    try:
        weather_loader = WeatherMeteostatLoader()
        if 'locations' in config['data']['weather'] and config['data']['weather']['locations']:
            start_date = demand_df['timestamp'].min().strftime('%Y-%m-%d')
            end_date = demand_df['timestamp'].max().strftime('%Y-%m-%d')
            
            weather_df = weather_loader.load_weather_data(
                stations=config['data']['weather']['locations'],
                start_date=start_date,
                end_date=end_date
            )
            
            if not weather_df.empty:
                weather_df = weather_loader.get_temperature_features(weather_df)
                demand_df = weather_loader.merge_with_demand(demand_df, weather_df)
                logger.info("Weather data merged successfully")
    except Exception as e:
        logger.warning(f"Could not load weather data: {e}")
    
    return demand_df


def engineer_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Engineer features for training.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")
    
    feature_builder = FeatureBuilder()
    
    # Build all features
    df_features = feature_builder.build_features(
        df, 
        target_col='demand',
        timestamp_col='timestamp'
    )
    
    return df_features


def prepare_training_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare data for training.
    
    Args:
        df: Feature DataFrame
        config: Configuration dictionary
        
    Returns:
        Training and validation splits
    """
    logger.info("Preparing training data...")
    
    # Remove rows with NaN in target
    df = df.dropna(subset=['demand'])
    
    # Define feature columns (exclude target and timestamp)
    feature_cols = [col for col in df.columns if col not in ['demand', 'timestamp']]
    
    # Handle missing values in features
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
    
    X = df[feature_cols]
    y = df['demand']
    
    # Train-validation split
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        config: Dict[str, Any]) -> LightGBMQuantileRegressor:
    """
    Train LightGBM quantile model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration dictionary
        
    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBM quantile model...")
    
    quantiles = config['forecast']['quantiles']
    lgb_params = config['models']['lightgbm']
    
    model = LightGBMQuantileRegressor(
        quantiles=quantiles,
        **lgb_params
    )
    
    model.fit(X_train, y_train, X_val, y_val, num_boost_round=200)
    
    return model


def train_torch_model(X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     config: Dict[str, Any]) -> TorchQuantileTrainer:
    """
    Train PyTorch quantile model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration dictionary
        
    Returns:
        Trained PyTorch model
    """
    logger.info("Training PyTorch quantile model...")
    
    quantiles = config['forecast']['quantiles']
    torch_params = config['models']['torch']
    
    # Create model
    input_size = X_train.shape[1]
    model = DeepQuantileRegressor(
        input_size=input_size,
        hidden_sizes=[torch_params['hidden_size'], torch_params['hidden_size'] // 2, torch_params['hidden_size'] // 4],
        dropout=torch_params['dropout'],
        num_quantiles=len(quantiles)
    )
    
    # Create trainer
    trainer = TorchQuantileTrainer(
        model=model,
        quantiles=quantiles,
        learning_rate=0.001
    )
    
    # Convert to numpy
    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    y_val_np = y_val.values.astype(np.float32)
    
    # Train
    history = trainer.fit(
        X_train_np, y_train_np,
        X_val_np, y_val_np,
        epochs=100,
        batch_size=32
    )
    
    logger.info(f"Training completed. Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return trainer


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series, 
                  model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained model.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets
        model_type: Type of model ('lightgbm' or 'torch')
        config: Configuration dictionary
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating {model_type} model...")
    
    if model_type == 'lightgbm':
        predictions = model.predict(X_val)
        # Convert to point prediction for median
        y_pred = predictions[0.5] if 0.5 in predictions else list(predictions.values())[0]
    else:  # torch
        X_val_np = X_val.values.astype(np.float32)
        pred_array = model.predict(X_val_np)
        
        quantiles = config['forecast']['quantiles']
        predictions = {q: pred_array[:, i] for i, q in enumerate(quantiles)}
        y_pred = predictions[0.5] if 0.5 in predictions else list(predictions.values())[0]
    
    # Create metrics report
    report = create_metrics_report(
        y_true=y_val.values,
        y_pred=y_pred,
        quantile_predictions=predictions
    )
    
    logger.info(f"{model_type.upper()} Model Performance:")
    for metric, value in report['overall'].items():
        logger.info(f"  {metric}: {value:.4f}")
        
    if 'quantile' in report:
        logger.info("Quantile Performance:")
        for metric, value in report['quantile'].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    return report


def save_models(lightgbm_model: Optional[LightGBMQuantileRegressor],
               torch_model: Optional[TorchQuantileTrainer],
               config: Dict[str, Any]) -> None:
    """
    Save trained models.
    
    Args:
        lightgbm_model: Trained LightGBM model
        torch_model: Trained PyTorch model
        config: Configuration dictionary
    """
    logger.info("Saving models...")
    
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if lightgbm_model:
        # Save LightGBM models
        lightgbm_model.save_models("artifacts/models/lightgbm_quantile_{quantile}.txt")
        
        # Save model object
        save_pickle(lightgbm_model, "artifacts/models/lightgbm_model.pkl")
        logger.info("LightGBM model saved")
    
    if torch_model:
        # Save PyTorch model
        import torch
        torch.save(torch_model.model.state_dict(), "artifacts/models/torch_model.pth")
        
        # Save trainer object
        save_pickle(torch_model, "artifacts/models/torch_trainer.pkl")
        logger.info("PyTorch model saved")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PowerScope forecasting models')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['lightgbm', 'torch', 'both'], 
                       default='both', help='Model type to train')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save trained models')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load and prepare data
    df = load_and_prepare_data(config)
    
    # Engineer features
    df_features = engineer_features(df, config)
    
    # Prepare training data
    X_train, y_train, X_val, y_val = prepare_training_data(df_features, config)
    
    lightgbm_model = None
    torch_model = None
    
    # Train models
    if args.model in ['lightgbm', 'both']:
        try:
            lightgbm_model = train_lightgbm_model(X_train, y_train, X_val, y_val, config)
            evaluate_model(lightgbm_model, X_val, y_val, 'lightgbm', config)
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
    
    if args.model in ['torch', 'both']:
        try:
            torch_model = train_torch_model(X_train, y_train, X_val, y_val, config)
            evaluate_model(torch_model, X_val, y_val, 'torch', config)
        except Exception as e:
            logger.error(f"Error training PyTorch model: {e}")
    
    # Save models
    if not args.no_save:
        save_models(lightgbm_model, torch_model, config)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()